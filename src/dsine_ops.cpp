#include "brovisionml/dsine_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#if defined(BROVISIONML_WITH_CUDA)
#include "dsine_ops_cuda.h"
#endif

namespace brovisionml::dsine {

namespace {

using brotensor::Tensor;

const std::string kWho = "dsine ops: ";

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

constexpr int   kPs     = 5;          // NRN propagation window (5x5)
constexpr int   kPad    = 2;          // (ps-1)/2
constexpr int   kPP     = kPs * kPs;  // 25 neighbors
constexpr float kRayEps = 1e-2f;

// axis_angle_to_matrix (PyTorch3D port): ax = axis*angle (3 comps) -> row-major
// 3x3 R. Identical to the historical host port; double throughout.
void axis_angle_to_matrix(double ax, double ay, double az, double R[9]) {
    const double angle = std::sqrt(ax * ax + ay * ay + az * az);
    const double half = angle * 0.5;
    double s;  // sin(half)/angle
    if (angle < 1e-6)
        s = 0.5 - (angle * angle) / 48.0;   // Taylor of sin(half)/angle
    else
        s = std::sin(half) / angle;
    const double r = std::cos(half);
    const double i = ax * s, j = ay * s, k = az * s;
    const double two_s = 2.0 / (r * r + i * i + j * j + k * k);
    R[0] = 1.0 - two_s * (j * j + k * k);
    R[1] = two_s * (i * j - k * r);
    R[2] = two_s * (i * k + j * r);
    R[3] = two_s * (i * j + k * r);
    R[4] = 1.0 - two_s * (i * i + k * k);
    R[5] = two_s * (j * k - i * r);
    R[6] = two_s * (i * k - j * r);
    R[7] = two_s * (j * k + i * r);
    R[8] = 1.0 - two_s * (i * i + j * j);
}

// ── CPU twins ─────────────────────────────────────────────────────────────────

void ray_relu_cpu(Tensor& normal, const Tensor& ray, int H, int W) {
    const int HW = H * W;
    float* n = normal.host_f32_mut();
    const float* r = ray.host_f32();
    for (int p = 0; p < HW; ++p) {
        const double nx = n[p], ny = n[HW + p], nz = n[2 * HW + p];
        const double rx = r[p], ry = r[HW + p], rz = r[2 * HW + p];
        const double nn = std::sqrt(nx * nx + ny * ny + nz * nz);
        const double rr = std::sqrt(rx * rx + ry * ry + rz * rz);
        const double denom = std::max(nn * rr, 1e-8);   // torch cosine_sim eps
        const double cos = (nx * rx + ny * ry + nz * rz) / denom;
        const double relu_cm = std::max(cos - kRayEps, 0.0) + kRayEps;
        const double dcoef = relu_cm - cos;             // ray*(relu_cm - cos)
        const double mx = nx + rx * dcoef;
        const double my = ny + ry * dcoef;
        const double mz = nz + rz * dcoef;
        const double inv = 1.0 / std::max(std::sqrt(mx * mx + my * my + mz * mz), 1e-12);
        n[p]          = static_cast<float>(mx * inv);
        n[HW + p]     = static_cast<float>(my * inv);
        n[2 * HW + p] = static_cast<float>(mz * inv);
    }
}

void angmf_propagate_cpu(const Tensor& pred_norm, const Tensor& prob,
                         const Tensor& xy, const Tensor& angle, const Tensor& ray,
                         double fu, double cu, double fv, double cv,
                         int H, int W, Tensor& out) {
    const int HW = H * W;
    out = Tensor::mat(1, 3 * HW);
    const float* pn  = pred_norm.host_f32();
    const float* pr  = prob.host_f32();
    const float* xyp = xy.host_f32();
    const float* ap  = angle.host_f32();
    const float* rp  = ray.host_f32();
    float* op = out.host_f32_mut();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int p = y * W + x;
            // The OUTPUT pixel's ray drives the per-neighbor RayReLU.
            const double rx = rp[p], ry = rp[HW + p], rz = rp[2 * HW + p];
            double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;

            for (int n = 0; n < kPP; ++n) {
                const int ky = n / kPs, kx = n % kPs;
                int sy = std::min(std::max(y - kPad + ky, 0), H - 1);  // replicate
                int sx = std::min(std::max(x - kPad + kx, 0), W - 1);
                const int sp = sy * W + sx;

                // Unfolded neighbor normal + its pixel-center coordinate.
                const double nx = pn[sp], ny = pn[HW + sp], nz = pn[2 * HW + sp];
                const double pix_x = sx + 0.5, pix_y = sy + 0.5;

                // Per-neighbor head outputs at THIS pixel (channel n). The raw
                // xy pair is L2-normalized here (F.normalize over the 2 axis).
                double xs = xyp[static_cast<std::size_t>(n) * HW + p];
                double ys = xyp[static_cast<std::size_t>(kPP + n) * HW + p];
                const double xyinv = 1.0 / std::max(std::sqrt(xs * xs + ys * ys), 1e-12);
                xs *= xyinv; ys *= xyinv;
                const double theta = ap[static_cast<std::size_t>(n) * HW + p];
                const double w = pr[static_cast<std::size_t>(n) * HW + p];

                const double du_over_fu = xs / fu;
                const double dv_over_fv = ys / fv;
                const double term_u = (pix_x + xs - cu) / fu;
                const double term_v = (pix_y + ys - cv) / fv;

                const double num = -(du_over_fu * nx + dv_over_fv * ny);
                double dd = term_u * nx + term_v * ny + nz;
                if (std::abs(dd) < 1e-8) {
                    const double sgn = (dd > 0.0) ? 1.0 : (dd < 0.0 ? -1.0 : 0.0);
                    dd = 1e-8 * sgn;
                }
                const double dz = num / dd;

                double axx = du_over_fu + dz * term_u;
                double axy = dv_over_fv + dz * term_v;
                double axz = dz;
                const double an = std::sqrt(axx * axx + axy * axy + axz * axz);
                const double ainv = 1.0 / std::max(an, 1e-12);
                axx *= ainv; axy *= ainv; axz *= ainv;
                if (!std::isfinite(axx) || !std::isfinite(axy) || !std::isfinite(axz)) {
                    axx = axy = axz = 0.0;   // invalid axis -> identity rotation
                }

                double R[9];
                axis_angle_to_matrix(axx * theta, axy * theta, axz * theta, R);

                double rnx = R[0] * nx + R[1] * ny + R[2] * nz;
                double rny = R[3] * nx + R[4] * ny + R[5] * nz;
                double rnz = R[6] * nx + R[7] * ny + R[8] * nz;
                double rinv = 1.0 /
                    std::max(std::sqrt(rnx * rnx + rny * rny + rnz * rnz), 1e-12);
                rnx *= rinv; rny *= rinv; rnz *= rinv;

                // RayReLU against the output pixel's ray.
                const double nn = std::sqrt(rnx * rnx + rny * rny + rnz * rnz);
                const double rr = std::sqrt(rx * rx + ry * ry + rz * rz);
                const double denom = std::max(nn * rr, 1e-8);
                const double cos = (rnx * rx + rny * ry + rnz * rz) / denom;
                const double relu_cm = std::max(cos - kRayEps, 0.0) + kRayEps;
                const double dcoef = relu_cm - cos;
                double mx = rnx + rx * dcoef, my = rny + ry * dcoef, mz = rnz + rz * dcoef;
                const double minv = 1.0 /
                    std::max(std::sqrt(mx * mx + my * my + mz * mz), 1e-12);
                mx *= minv; my *= minv; mz *= minv;

                acc0 += w * mx; acc1 += w * my; acc2 += w * mz;
            }

            const double inv = 1.0 /
                std::max(std::sqrt(acc0 * acc0 + acc1 * acc1 + acc2 * acc2), 1e-12);
            op[p]          = static_cast<float>(acc0 * inv);
            op[HW + p]     = static_cast<float>(acc1 * inv);
            op[2 * HW + p] = static_cast<float>(acc2 * inv);
        }
    }
}

}  // namespace

// ── Public dispatch ─────────────────────────────────────────────────────────

void ray_relu(brotensor::Tensor& normal, const brotensor::Tensor& ray,
              int H, int W) {
    if (normal.device != ray.device)
        fail("ray_relu: normal and ray must share a device");
    if (normal.device == brotensor::Device::CPU) {
        ray_relu_cpu(normal, ray, H, W);
        return;
    }
#if defined(BROVISIONML_WITH_CUDA)
    detail::ray_relu_cuda(normal, ray, H, W);
#else
    fail("ray_relu: tensor on a non-CPU device, but brovisionml was built "
         "without CUDA");
#endif
}

void angmf_propagate(const brotensor::Tensor& pred_norm,
                     const brotensor::Tensor& prob, const brotensor::Tensor& xy,
                     const brotensor::Tensor& angle, const brotensor::Tensor& ray,
                     double fu, double cu, double fv, double cv,
                     int H, int W, brotensor::Tensor& out) {
    const brotensor::Device dev = pred_norm.device;
    if (prob.device != dev || xy.device != dev || angle.device != dev ||
        ray.device != dev)
        fail("angmf_propagate: all inputs must share a device");
    if (dev == brotensor::Device::CPU) {
        angmf_propagate_cpu(pred_norm, prob, xy, angle, ray, fu, cu, fv, cv,
                            H, W, out);
        return;
    }
#if defined(BROVISIONML_WITH_CUDA)
    detail::angmf_propagate_cuda(pred_norm, prob, xy, angle, ray, fu, cu, fv, cv,
                                 H, W, out);
#else
    fail("angmf_propagate: tensors on a non-CPU device, but brovisionml was "
         "built without CUDA");
#endif
}

}  // namespace brovisionml::dsine
