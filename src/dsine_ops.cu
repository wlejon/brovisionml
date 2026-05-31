// CUDA kernels for the DSINE NRN device ops — brovisionml's first CUDA code.
// The host wrappers (declared in dsine_ops_cuda.h) cast the brotensor tensors'
// raw device pointers and launch one thread per /8-grid pixel on the default
// stream (the stream brotensor's own ops use, so no cross-stream sync needed).
// The per-pixel geometry runs in double to match the CPU twin in dsine_ops.cpp.

#include "dsine_ops_cuda.h"

#include "brotensor/tensor.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brovisionml::dsine::detail {

namespace {

constexpr int kPs   = 5;          // NRN propagation window (5x5)
constexpr int kPad  = 2;          // (ps-1)/2
constexpr int kPP   = kPs * kPs;  // 25 neighbors
constexpr double kRayEps = 1e-2;

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("dsine cuda: ") + what + ": " +
                                 cudaGetErrorString(e));
}

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// axis_angle_to_matrix (PyTorch3D port): ax = axis*angle -> row-major 3x3 R.
__device__ void axis_angle_to_matrix(double ax, double ay, double az, double R[9]) {
    const double angle = sqrt(ax * ax + ay * ay + az * az);
    const double half = angle * 0.5;
    double s;
    if (angle < 1e-6)
        s = 0.5 - (angle * angle) / 48.0;
    else
        s = sin(half) / angle;
    const double r = cos(half);
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

__global__ void ray_relu_kernel(float* n, const float* r, int HW) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= HW) return;
    const double nx = n[p], ny = n[HW + p], nz = n[2 * HW + p];
    const double rx = r[p], ry = r[HW + p], rz = r[2 * HW + p];
    const double nn = sqrt(nx * nx + ny * ny + nz * nz);
    const double rr = sqrt(rx * rx + ry * ry + rz * rz);
    const double denom = fmax(nn * rr, 1e-8);
    const double c = (nx * rx + ny * ry + nz * rz) / denom;
    const double relu_cm = fmax(c - kRayEps, 0.0) + kRayEps;
    const double dcoef = relu_cm - c;
    const double mx = nx + rx * dcoef, my = ny + ry * dcoef, mz = nz + rz * dcoef;
    const double inv = 1.0 / fmax(sqrt(mx * mx + my * my + mz * mz), 1e-12);
    n[p]          = static_cast<float>(mx * inv);
    n[HW + p]     = static_cast<float>(my * inv);
    n[2 * HW + p] = static_cast<float>(mz * inv);
}

__global__ void angmf_kernel(const float* pn, const float* pr, const float* xyp,
                             const float* ap, const float* rp,
                             double fu, double cu, double fv, double cv,
                             int H, int W, float* op) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const long HW = static_cast<long>(H) * W;
    if (p >= HW) return;
    const int y = p / W, x = p % W;

    const double rx = rp[p], ry = rp[HW + p], rz = rp[2 * HW + p];
    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;

    for (int nn = 0; nn < kPP; ++nn) {
        const int ky = nn / kPs, kx = nn % kPs;
        const int sy = clampi(y - kPad + ky, 0, H - 1);   // replicate pad
        const int sx = clampi(x - kPad + kx, 0, W - 1);
        const long sp = static_cast<long>(sy) * W + sx;

        const double nx = pn[sp], ny = pn[HW + sp], nz = pn[2 * HW + sp];
        const double pix_x = sx + 0.5, pix_y = sy + 0.5;

        double xs = xyp[static_cast<long>(nn) * HW + p];
        double ys = xyp[static_cast<long>(kPP + nn) * HW + p];
        const double xyinv = 1.0 / fmax(sqrt(xs * xs + ys * ys), 1e-12);
        xs *= xyinv; ys *= xyinv;   // F.normalize over the 2 (x,y) axis
        const double theta = ap[static_cast<long>(nn) * HW + p];
        const double w     = pr[static_cast<long>(nn) * HW + p];

        const double du_over_fu = xs / fu;
        const double dv_over_fv = ys / fv;
        const double term_u = (pix_x + xs - cu) / fu;
        const double term_v = (pix_y + ys - cv) / fv;

        const double num = -(du_over_fu * nx + dv_over_fv * ny);
        double dd = term_u * nx + term_v * ny + nz;
        if (fabs(dd) < 1e-8) {
            const double sgn = (dd > 0.0) ? 1.0 : (dd < 0.0 ? -1.0 : 0.0);
            dd = 1e-8 * sgn;
        }
        const double dz = num / dd;

        double axx = du_over_fu + dz * term_u;
        double axy = dv_over_fv + dz * term_v;
        double axz = dz;
        const double an = sqrt(axx * axx + axy * axy + axz * axz);
        const double ainv = 1.0 / fmax(an, 1e-12);
        axx *= ainv; axy *= ainv; axz *= ainv;
        if (!isfinite(axx) || !isfinite(axy) || !isfinite(axz)) {
            axx = axy = axz = 0.0;   // invalid axis -> identity rotation
        }

        double R[9];
        axis_angle_to_matrix(axx * theta, axy * theta, axz * theta, R);

        double rnx = R[0] * nx + R[1] * ny + R[2] * nz;
        double rny = R[3] * nx + R[4] * ny + R[5] * nz;
        double rnz = R[6] * nx + R[7] * ny + R[8] * nz;
        double rinv = 1.0 / fmax(sqrt(rnx * rnx + rny * rny + rnz * rnz), 1e-12);
        rnx *= rinv; rny *= rinv; rnz *= rinv;

        // RayReLU against the output pixel's ray.
        const double rn = sqrt(rnx * rnx + rny * rny + rnz * rnz);
        const double rr = sqrt(rx * rx + ry * ry + rz * rz);
        const double denom = fmax(rn * rr, 1e-8);
        const double c = (rnx * rx + rny * ry + rnz * rz) / denom;
        const double relu_cm = fmax(c - kRayEps, 0.0) + kRayEps;
        const double dcoef = relu_cm - c;
        double mx = rnx + rx * dcoef, my = rny + ry * dcoef, mz = rnz + rz * dcoef;
        const double minv = 1.0 / fmax(sqrt(mx * mx + my * my + mz * mz), 1e-12);
        mx *= minv; my *= minv; mz *= minv;

        acc0 += w * mx; acc1 += w * my; acc2 += w * mz;
    }

    const double inv = 1.0 /
        fmax(sqrt(acc0 * acc0 + acc1 * acc1 + acc2 * acc2), 1e-12);
    op[p]          = static_cast<float>(acc0 * inv);
    op[HW + p]     = static_cast<float>(acc1 * inv);
    op[2 * HW + p] = static_cast<float>(acc2 * inv);
}

}  // namespace

void ray_relu_cuda(brotensor::Tensor& normal, const brotensor::Tensor& ray,
                   int H, int W) {
    const int HW = H * W;
    if (HW == 0) return;
    const int block = 256;
    const int grid = (HW + block - 1) / block;
    ray_relu_kernel<<<grid, block>>>(static_cast<float*>(normal.data),
                                     static_cast<const float*>(ray.data), HW);
    cuda_check(cudaGetLastError(), "ray_relu launch");
}

void angmf_propagate_cuda(const brotensor::Tensor& pred_norm,
                          const brotensor::Tensor& prob,
                          const brotensor::Tensor& xy,
                          const brotensor::Tensor& angle,
                          const brotensor::Tensor& ray,
                          double fu, double cu, double fv, double cv,
                          int H, int W, brotensor::Tensor& out) {
    const int HW = H * W;
    out = brotensor::Tensor::zeros_on(brotensor::Device::CUDA, 1, 3 * HW);
    if (HW == 0) return;
    const int block = 256;
    const int grid = (HW + block - 1) / block;
    angmf_kernel<<<grid, block>>>(
        static_cast<const float*>(pred_norm.data),
        static_cast<const float*>(prob.data),
        static_cast<const float*>(xy.data),
        static_cast<const float*>(angle.data),
        static_cast<const float*>(ray.data),
        fu, cu, fv, cv, H, W,
        static_cast<float*>(out.data));
    cuda_check(cudaGetLastError(), "angmf_propagate launch");
}

}  // namespace brovisionml::dsine::detail
