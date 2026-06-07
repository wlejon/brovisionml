// Depth-Anything-V2 end-to-end test.
//
// Two layers:
//   * Always-on structural checks: the v2_small/base/large presets keep the
//     backbone and DPT head dimensionally consistent (the head's reassemble
//     width equals the backbone hidden size; stage counts line up).
//   * Weights-gated real-checkpoint run: when an HF Depth-Anything-V2 checkpoint
//     is present under weights/, load it and estimate depth on a procedurally
//     rendered scene, asserting the map's shape (== input), finiteness, that it
//     actually varies, and CPU/CUDA parity. Download with
//       scripts/download-weights.sh depth-anything-v2-small   (~100 MB)
//     When absent the test prints why and exits 0 (clean skip).
//
// Exact agreement with the HF Python output is not asserted here (that needs a
// reference activation dump); this guards shape/finiteness/variation/parity and
// the full HF weight-naming path end to end.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/depth_anything.h"

#include "brotensor/runtime.h"

#include "test_device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

using brovisionml::depth::DepthAnythingConfig;
using brovisionml::depth::DepthEstimator;
using brovisionml::depth::DepthMap;

// A simple two-region scene: a bright rectangle over a gradient background. Just
// enough structure that a working model produces a spatially varying map.
std::vector<uint8_t> make_scene(int W, int H) {
    std::vector<uint8_t> img(static_cast<std::size_t>(W) * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const std::size_t o = (static_cast<std::size_t>(y) * W + x) * 3;
            const bool box = (x > W / 4 && x < 3 * W / 4 && y > H / 3 && y < 2 * H / 3);
            const uint8_t bg = static_cast<uint8_t>(40 + 180 * y / H);
            img[o + 0] = box ? 250 : bg;
            img[o + 1] = box ? 250 : static_cast<uint8_t>(bg / 2);
            img[o + 2] = box ? 250 : static_cast<uint8_t>(255 - bg);
        }
    return img;
}

void structural_checks() {
    auto consistent = [&](const DepthAnythingConfig& c, const char* name) {
        check(c.head.reassemble_hidden_size == c.backbone.embed_dim,
              name);
        check(c.backbone.out_stages.size() == c.head.neck_hidden_sizes.size(),
              name);
        check(c.backbone.patch_size == c.head.patch_size, name);
    };
    consistent(DepthAnythingConfig::v2_small(), "v2_small consistent");
    consistent(DepthAnythingConfig::v2_base(),  "v2_base consistent");
    consistent(DepthAnythingConfig::v2_large(), "v2_large consistent");
}

}  // namespace

int main() {
    structural_checks();

    // ── Locate a real checkpoint (clean skip if absent) ──
    std::string base = BROVISIONML_WEIGHTS_DIR;
    if (const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR")) base = env;
    std::string dir = base + "/Depth-Anything-V2-Small";
    std::string ckpt = dir + "/model.safetensors";

    if (!file_exists(ckpt)) {
        std::printf("test_depth_anything: no checkpoint at %s — skipping the "
                    "weights-gated run (structural checks passed).\n", ckpt.c_str());
        return failures == 0 ? 0 : 1;
    }

    const int W = 320, H = 240;
    std::vector<uint8_t> img = make_scene(W, H);

    try {
        DepthEstimator est(DepthAnythingConfig::v2_small());
        est.load(dir);

        DepthMap dm = est.estimate(img.data(), W, H, 3);
        check(dm.width == W && dm.height == H, "depth map matches input size");
        check(dm.depth.size() == static_cast<std::size_t>(W) * H, "depth map element count");

        bool finite = true;
        float lo = dm.depth[0], hi = dm.depth[0];
        for (float v : dm.depth) {
            if (!std::isfinite(v)) finite = false;
            lo = std::min(lo, v); hi = std::max(hi, v);
        }
        check(finite, "depth map all-finite");
        check(hi - lo > 1e-3f, "depth map varies spatially");
        std::printf("CPU depth range [%.4f, %.4f]\n", lo, hi);

        // ── CPU/GPU parity ──
        brotensor::init();
        const brotensor::Device gpu = brovisionml_test::preferred_gpu();
        if (gpu != brotensor::Device::CPU) {
            const char* dev = brovisionml_test::device_name(gpu);
            DepthEstimator gest(DepthAnythingConfig::v2_small());
            gest.load(dir);
            gest.to(gpu);
            DepthMap gd = gest.estimate(img.data(), W, H, 3);
            float worst = 0.0f, scale = std::max(1e-3f, hi - lo);
            for (std::size_t i = 0; i < dm.depth.size(); ++i)
                worst = std::max(worst, std::fabs(dm.depth[i] - gd.depth[i]));
            std::printf("CPU/%s worst depth diff %.5f (range scale %.4f)\n", dev, worst, scale);
            // FP16 GPU vs FP32 CPU over a deep ViT + DPT; relative tolerance.
            check(worst < 0.05f * scale + 1e-2f, "CPU/GPU depth parity within tolerance");
        } else {
            std::printf("(GPU not available — parity check skipped)\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (failures == 0) { std::printf("test_depth_anything: OK\n"); return 0; }
    std::printf("test_depth_anything: %d failure(s)\n", failures);
    return 1;
}
