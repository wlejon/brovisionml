// Metal kernels for the DSINE NRN device ops — the Metal twin of dsine_ops.cu.
//
// These are brovisionml's only hand-written GPU kernels (the rest of the repo
// composes brotensor ops). They are authored against brotensor's public Metal
// custom-kernel surface (<brotensor/metal_interop.h>): the shared MTLDevice,
// the Tensor -> MTLBuffer resolver, and the MSL-source pipeline compiler. No
// private MTLDevice is created here.
//
// Layout matches the CPU/CUDA twins: every map is a (1, C*H*W) row-major
// brotensor tensor indexed planar as channel*HW + pixel; FP32; one thread per
// pixel; threadgroup 256; no threadgroup memory. Unlike the CPU/CUDA paths the
// per-pixel geometry runs in `float`, not `double` — Metal Shading Language has
// no double type. `precise::` trig/sqrt keep the extra error small; the
// float-vs-double divergence is covered by a tolerance-based parity test
// (tests/test_dsine_ops_parity.cpp), not bit-exact equality.

#include "dsine_ops_metal.h"

#include "brotensor/tensor.h"

#import <brotensor/metal_interop.h>

#include <stdexcept>
#include <string>

namespace brovisionml::dsine::detail {

using brotensor::Tensor;
using brotensor::metal_impl::buffer_for;
using brotensor::metal_impl::buffer_offset_for;
using brotensor::metal_impl::compile_pipeline;
using brotensor::metal_impl::new_command_buffer;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(std::string("dsine metal: ") + msg);
}

// Parameter block for angmf_kernel — must match the MSL struct below.
struct AngmfParams {
    float    fu, cu, fv, cv;
    uint32_t H, W, HW;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant int   kPs     = 5;      // NRN propagation window (5x5)
constant int   kPad    = 2;      // (ps-1)/2
constant int   kPP     = 25;     // kPs*kPs neighbors
constant float kRayEps = 1e-2f;

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

struct AngmfParams {
    float fu, cu, fv, cv;
    uint  H, W, HW;
};

// axis_angle_to_matrix (PyTorch3D port): ax = axis*angle -> row-major 3x3 R.
// Float twin of the double host/CUDA version; same Taylor guard on the small
// angle branch.
static void axis_angle_to_matrix(float ax, float ay, float az, thread float* R) {
    const float angle = precise::sqrt(ax * ax + ay * ay + az * az);
    const float halfang = angle * 0.5f;
    float s;  // sin(halfang)/angle
    if (angle < 1e-6f)
        s = 0.5f - (angle * angle) / 48.0f;   // Taylor of sin(halfang)/angle
    else
        s = precise::sin(halfang) / angle;
    const float r = precise::cos(halfang);
    const float i = ax * s, j = ay * s, k = az * s;
    const float two_s = 2.0f / (r * r + i * i + j * j + k * k);
    R[0] = 1.0f - two_s * (j * j + k * k);
    R[1] = two_s * (i * j - k * r);
    R[2] = two_s * (i * k + j * r);
    R[3] = two_s * (i * j + k * r);
    R[4] = 1.0f - two_s * (i * i + k * k);
    R[5] = two_s * (j * k - i * r);
    R[6] = two_s * (i * k - j * r);
    R[7] = two_s * (j * k + i * r);
    R[8] = 1.0f - two_s * (i * i + j * j);
}

kernel void k_ray_relu(device float*       n  [[buffer(0)]],
                       device const float* r  [[buffer(1)]],
                       constant uint&      HW [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= HW) return;
    const uint p = gid;
    const float nx = n[p], ny = n[HW + p], nz = n[2 * HW + p];
    const float rx = r[p], ry = r[HW + p], rz = r[2 * HW + p];
    const float nn = precise::sqrt(nx * nx + ny * ny + nz * nz);
    const float rr = precise::sqrt(rx * rx + ry * ry + rz * rz);
    const float denom = fmax(nn * rr, 1e-8f);   // torch cosine_sim eps
    const float c = (nx * rx + ny * ry + nz * rz) / denom;
    const float relu_cm = fmax(c - kRayEps, 0.0f) + kRayEps;
    const float dcoef = relu_cm - c;
    const float mx = nx + rx * dcoef, my = ny + ry * dcoef, mz = nz + rz * dcoef;
    const float inv = 1.0f / fmax(precise::sqrt(mx * mx + my * my + mz * mz), 1e-12f);
    n[p]          = mx * inv;
    n[HW + p]     = my * inv;
    n[2 * HW + p] = mz * inv;
}

kernel void k_angmf(device const float* pn  [[buffer(0)]],
                    device const float* pr  [[buffer(1)]],
                    device const float* xyp [[buffer(2)]],
                    device const float* ap  [[buffer(3)]],
                    device const float* rp  [[buffer(4)]],
                    device float*       op  [[buffer(5)]],
                    constant AngmfParams& q [[buffer(6)]],
                    uint gid [[thread_position_in_grid]]) {
    const uint HW = q.HW;
    if (gid >= HW) return;
    const int W = int(q.W), H = int(q.H);
    const uint p = gid;
    const int y = int(p) / W, x = int(p) % W;
    const float fu = q.fu, cu = q.cu, fv = q.fv, cv = q.cv;

    const float rx = rp[p], ry = rp[HW + p], rz = rp[2 * HW + p];
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;

    for (int nn = 0; nn < kPP; ++nn) {
        const int ky = nn / kPs, kx = nn % kPs;
        const int sy = clampi(y - kPad + ky, 0, H - 1);   // replicate pad
        const int sx = clampi(x - kPad + kx, 0, W - 1);
        const uint sp = uint(sy) * uint(W) + uint(sx);

        const float nx = pn[sp], ny = pn[HW + sp], nz = pn[2 * HW + sp];
        const float pix_x = float(sx) + 0.5f, pix_y = float(sy) + 0.5f;

        float xs = xyp[uint(nn) * HW + p];
        float ys = xyp[uint(kPP + nn) * HW + p];
        const float xyinv = 1.0f / fmax(precise::sqrt(xs * xs + ys * ys), 1e-12f);
        xs *= xyinv; ys *= xyinv;   // F.normalize over the 2 (x,y) axis
        const float theta = ap[uint(nn) * HW + p];
        const float w     = pr[uint(nn) * HW + p];

        const float du_over_fu = xs / fu;
        const float dv_over_fv = ys / fv;
        const float term_u = (pix_x + xs - cu) / fu;
        const float term_v = (pix_y + ys - cv) / fv;

        const float num = -(du_over_fu * nx + dv_over_fv * ny);
        float dd = term_u * nx + term_v * ny + nz;
        if (fabs(dd) < 1e-8f) {
            const float sgn = (dd > 0.0f) ? 1.0f : (dd < 0.0f ? -1.0f : 0.0f);
            dd = 1e-8f * sgn;
        }
        const float dz = num / dd;

        float axx = du_over_fu + dz * term_u;
        float axy = dv_over_fv + dz * term_v;
        float axz = dz;
        const float an = precise::sqrt(axx * axx + axy * axy + axz * axz);
        const float ainv = 1.0f / fmax(an, 1e-12f);
        axx *= ainv; axy *= ainv; axz *= ainv;
        if (!isfinite(axx) || !isfinite(axy) || !isfinite(axz)) {
            axx = axy = axz = 0.0f;   // invalid axis -> identity rotation
        }

        float R[9];
        axis_angle_to_matrix(axx * theta, axy * theta, axz * theta, R);

        float rnx = R[0] * nx + R[1] * ny + R[2] * nz;
        float rny = R[3] * nx + R[4] * ny + R[5] * nz;
        float rnz = R[6] * nx + R[7] * ny + R[8] * nz;
        const float rinv = 1.0f /
            fmax(precise::sqrt(rnx * rnx + rny * rny + rnz * rnz), 1e-12f);
        rnx *= rinv; rny *= rinv; rnz *= rinv;

        // RayReLU against the output pixel's ray.
        const float rn = precise::sqrt(rnx * rnx + rny * rny + rnz * rnz);
        const float rr = precise::sqrt(rx * rx + ry * ry + rz * rz);
        const float denom = fmax(rn * rr, 1e-8f);
        const float c = (rnx * rx + rny * ry + rnz * rz) / denom;
        const float relu_cm = fmax(c - kRayEps, 0.0f) + kRayEps;
        const float dcoef = relu_cm - c;
        float mx = rnx + rx * dcoef, my = rny + ry * dcoef, mz = rnz + rz * dcoef;
        const float minv = 1.0f /
            fmax(precise::sqrt(mx * mx + my * my + mz * mz), 1e-12f);
        mx *= minv; my *= minv; mz *= minv;

        acc0 += w * mx; acc1 += w * my; acc2 += w * mz;
    }

    const float inv = 1.0f /
        fmax(precise::sqrt(acc0 * acc0 + acc1 * acc1 + acc2 * acc2), 1e-12f);
    op[p]          = acc0 * inv;
    op[HW + p]     = acc1 * inv;
    op[2 * HW + p] = acc2 * inv;
}
)msl";

id<MTLComputePipelineState> pso_ray_relu() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_ray_relu"); });
    return p;
}

id<MTLComputePipelineState> pso_angmf() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_angmf"); });
    return p;
}

NSUInteger threadgroup_size(id<MTLComputePipelineState> ps) {
    NSUInteger tpt = [ps maxTotalThreadsPerThreadgroup];
    return tpt > 256 ? 256 : tpt;   // matches the CUDA block of 256
}

}  // namespace

void ray_relu_metal(brotensor::Tensor& normal, const brotensor::Tensor& ray,
                    int H, int W) {
    if (normal.dtype != brotensor::Dtype::FP32 || ray.dtype != brotensor::Dtype::FP32)
        fail("ray_relu: normal and ray must be FP32");
    const uint32_t HW = static_cast<uint32_t>(H) * static_cast<uint32_t>(W);
    if (HW == 0) return;

    @autoreleasepool {
        id<MTLComputePipelineState> ps = pso_ray_relu();
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_for(normal) offset:buffer_offset_for(normal) atIndex:0];
        [enc setBuffer:buffer_for(ray)    offset:buffer_offset_for(ray)    atIndex:1];
        [enc setBytes:&HW length:sizeof(HW) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(HW, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_size(ps), 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void angmf_propagate_metal(const brotensor::Tensor& pred_norm,
                           const brotensor::Tensor& prob,
                           const brotensor::Tensor& xy,
                           const brotensor::Tensor& angle,
                           const brotensor::Tensor& ray,
                           double fu, double cu, double fv, double cv,
                           int H, int W, brotensor::Tensor& out) {
    const uint32_t HW = static_cast<uint32_t>(H) * static_cast<uint32_t>(W);
    out = brotensor::Tensor::zeros_on(brotensor::Device::Metal, 1, 3 * static_cast<int>(HW));
    if (HW == 0) return;

    AngmfParams q{};
    q.fu = static_cast<float>(fu);
    q.cu = static_cast<float>(cu);
    q.fv = static_cast<float>(fv);
    q.cv = static_cast<float>(cv);
    q.H  = static_cast<uint32_t>(H);
    q.W  = static_cast<uint32_t>(W);
    q.HW = HW;

    @autoreleasepool {
        id<MTLComputePipelineState> ps = pso_angmf();
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_for(pred_norm) offset:buffer_offset_for(pred_norm) atIndex:0];
        [enc setBuffer:buffer_for(prob)      offset:buffer_offset_for(prob)      atIndex:1];
        [enc setBuffer:buffer_for(xy)        offset:buffer_offset_for(xy)        atIndex:2];
        [enc setBuffer:buffer_for(angle)     offset:buffer_offset_for(angle)     atIndex:3];
        [enc setBuffer:buffer_for(ray)       offset:buffer_offset_for(ray)       atIndex:4];
        [enc setBuffer:buffer_for(out)       offset:buffer_offset_for(out)       atIndex:5];
        [enc setBytes:&q length:sizeof(AngmfParams) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(HW, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_size(ps), 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

}  // namespace brovisionml::dsine::detail
