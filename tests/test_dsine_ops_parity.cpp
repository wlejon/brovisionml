// CPU-vs-GPU parity for the two DSINE NRN device ops (ray_relu, angmf_propagate).
//
// These are the only hand-written GPU kernels in brovisionml. The CPU path runs
// the per-pixel geometry in double; the CUDA path mirrors it in double; the
// Metal path mirrors it in FLOAT (MSL has no double type). So this test asserts
// numeric agreement within a tolerance sized for float-vs-double, not bit
// exactness — the Metal port is the reason the tolerance is not ~0.
//
// Runs whichever GPU backend the binary was built with (CUDA or Metal); on a
// CPU-only build preferred_gpu() returns CPU and the test skips cleanly.

#include "brovisionml/dsine_ops.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_device.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Tensor;

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

// Max-abs difference between two host FP32 tensors of equal length.
double max_abs_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.host_f32();
    const float* pb = b.host_f32();
    const int n = a.rows * a.cols;
    double m = 0.0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::fabs(static_cast<double>(pa[i]) - static_cast<double>(pb[i])));
    return m;
}

// A (1, C*HW) NCHW map filled from `gen`.
Tensor make_map(int C, int HW, std::mt19937& rng,
                std::uniform_real_distribution<float>& dist) {
    Tensor t = Tensor::mat(1, C * HW);
    float* p = t.host_f32_mut();
    for (int i = 0; i < C * HW; ++i) p[i] = dist(rng);
    return t;
}

// A 3-channel unit-normal-ish map biased to face the camera (-z), so the
// RayReLU cos-eps kink is comfortably clear of its branch on both backends.
Tensor make_normals(int HW, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    Tensor t = Tensor::mat(1, 3 * HW);
    float* p = t.host_f32_mut();
    for (int i = 0; i < HW; ++i) {
        float x = d(rng), y = d(rng), z = -1.0f + d(rng) * 0.3f;
        float inv = 1.0f / std::sqrt(x * x + y * y + z * z);
        p[i] = x * inv; p[HW + i] = y * inv; p[2 * HW + i] = z * inv;
    }
    return t;
}

}  // namespace

int main() {
    brotensor::init();
    const brotensor::Device gpu = brovisionml_test::preferred_gpu();
    if (gpu == brotensor::Device::CPU) {
        std::printf("test_dsine_ops_parity: no GPU backend registered — skipping.\n");
        return 0;
    }
    std::printf("test_dsine_ops_parity: backend = %s\n",
                brovisionml_test::device_name(gpu));

    const int H = 12, W = 10, HW = H * W;
    std::mt19937 rng(1234);

    // ── ray_relu: in-place, so keep a pristine copy for each backend ──────────
    {
        Tensor normals = make_normals(HW, rng);
        std::uniform_real_distribution<float> rd(-0.6f, 0.6f);
        Tensor ray = Tensor::mat(1, 3 * HW);
        {
            float* p = ray.host_f32_mut();
            for (int i = 0; i < HW; ++i) {
                float x = rd(rng), y = rd(rng), z = -1.0f + rd(rng) * 0.2f;
                float inv = 1.0f / std::sqrt(x * x + y * y + z * z);
                p[i] = x * inv; p[HW + i] = y * inv; p[2 * HW + i] = z * inv;
            }
        }

        Tensor cpu_n = normals.clone();
        brovisionml::dsine::ray_relu(cpu_n, ray, H, W);

        Tensor gpu_n = normals.to(gpu);
        Tensor gpu_ray = ray.to(gpu);
        brovisionml::dsine::ray_relu(gpu_n, gpu_ray, H, W);
        brotensor::sync_all();
        Tensor gpu_n_host = gpu_n.to(brotensor::Device::CPU);

        double m = max_abs_diff(cpu_n, gpu_n_host);
        std::printf("  ray_relu max-abs-diff = %.3e\n", m);
        check(m < 1e-3, "ray_relu CPU vs GPU parity");
    }

    // ── angmf_propagate: allocates its own output on the input device ─────────
    {
        std::uniform_real_distribution<float> d01(0.0f, 1.0f);
        std::uniform_real_distribution<float> dxy(-1.0f, 1.0f);
        std::uniform_real_distribution<float> dang(0.0f, 3.14159265f);

        Tensor pred_norm = make_normals(HW, rng);
        Tensor prob  = make_map(25, HW, rng, d01);
        Tensor xy    = make_map(50, HW, rng, dxy);
        Tensor angle = make_map(25, HW, rng, dang);
        Tensor ray   = make_normals(HW, rng);
        const double fu = 120.0, cu = W * 0.5, fv = 120.0, cv = H * 0.5;

        Tensor cpu_out;
        brovisionml::dsine::angmf_propagate(pred_norm, prob, xy, angle, ray,
                                            fu, cu, fv, cv, H, W, cpu_out);

        Tensor g_pn = pred_norm.to(gpu), g_pr = prob.to(gpu), g_xy = xy.to(gpu),
               g_an = angle.to(gpu), g_ray = ray.to(gpu), g_out;
        brovisionml::dsine::angmf_propagate(g_pn, g_pr, g_xy, g_an, g_ray,
                                            fu, cu, fv, cv, H, W, g_out);
        brotensor::sync_all();
        Tensor g_out_host = g_out.to(brotensor::Device::CPU);

        check(g_out_host.rows == cpu_out.rows && g_out_host.cols == cpu_out.cols,
              "angmf_propagate output shape matches");
        double m = max_abs_diff(cpu_out, g_out_host);
        std::printf("  angmf_propagate max-abs-diff = %.3e\n", m);
        // Looser than ray_relu: 25-neighbor accumulation of the float-vs-double
        // rotation geometry compounds the per-step round-off.
        check(m < 3e-3, "angmf_propagate CPU vs GPU parity");
    }

    if (failures == 0) { std::printf("test_dsine_ops_parity: OK\n"); return 0; }
    std::printf("test_dsine_ops_parity: %d failure(s)\n", failures);
    return 1;
}
