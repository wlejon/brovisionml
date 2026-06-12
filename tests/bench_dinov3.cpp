// DINOv3 ViT-H encode benchmark at the TripoSplat production shape: a 1024x1024
// image -> 64x64 patch grid -> K = 5 + 4096 = 4101 tokens through the 32-block
// ViT-H+ stack on the preferred GPU. Gated on the real checkpoint
// (weights/triposplat/clip_vision/dino_v3_vit_h.safetensors); exits 0 with a
// note when absent. Not registered as a ctest — perf, not correctness.
//
// Set BROVISIONML_DINO_PROFILE=1 for the per-op breakdown (syncs serialise the
// stream, so profiled totals run above the clean timing this prints).

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/dinov3.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_device.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

using brovisionml::dinov3::Backbone;
using brovisionml::dinov3::BackboneOutput;
using brovisionml::dinov3::Config;

namespace {
bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}
}  // namespace

int main() {
    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
    const std::string ckpt = base + "/triposplat/clip_vision/dino_v3_vit_h.safetensors";
    if (!file_exists(ckpt)) {
        std::printf("bench_dinov3: no checkpoint at '%s' — skipping.\n", ckpt.c_str());
        return 0;
    }

    brotensor::init();
    const brotensor::Device gpu = brovisionml_test::preferred_gpu();
    if (gpu == brotensor::Device::CPU) {
        std::printf("bench_dinov3: no GPU available — skipping.\n");
        return 0;
    }

    const int H = 1024, W = 1024;
    Backbone bb(Config::vit_h());
    std::printf("loading %s ...\n", ckpt.c_str());
    bb.load_file(ckpt);
    bb.to(gpu);

    brotensor::Tensor px = brotensor::Tensor::mat(1, 3 * H * W);
    float* p = px.host_f32_mut();
    for (int i = 0; i < px.cols; ++i)
        p[i] = std::sin(static_cast<float>(i) * 0.0021f);  // bounded, non-trivial
    brotensor::Tensor px_gpu = px.to(gpu);

    // Warmup (allocations, JIT'd module loads).
    BackboneOutput out = bb.encode(px_gpu, H, W);
    brotensor::sync_all();
    std::printf("warmup done: K=%d, D=%d\n",
                out.last_hidden_state.rows, out.last_hidden_state.cols);

    const int iters = 3;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        out = bb.encode(px_gpu, H, W);
        brotensor::sync_all();
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count() / iters;
    std::printf("encode %dx%d (%s): %.1f ms/iter (%d iters)\n",
                H, W, brovisionml_test::device_name(gpu), ms, iters);
    return 0;
}
