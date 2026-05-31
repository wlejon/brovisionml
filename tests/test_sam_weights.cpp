// SAM real-checkpoint end-to-end test. Unlike test_sam.cpp (which synthesizes a
// tiny checkpoint), this loads an ACTUAL facebook/sam-vit-* checkpoint from the
// weights/ directory and runs the whole pipeline on it: load -> set_image ->
// segment (point + box) -> mask postprocess. It is the test that would have
// caught the tied shared_image_embedding key, which a self-built checkpoint
// cannot.
//
// Gated on the weights being present: download them with
//   scripts/download-weights.sh sam-vit-base      (~375 MB, the default)
// When no checkpoint is found the test prints why and exits 0 (a clean skip),
// so CI without weights stays green. The weights directory is baked in at
// configure time via -DBROVISIONML_WEIGHTS_DIR; the env var of the same name
// overrides it at run time.
//
// The input is a synthetic deterministic image — we are validating that real
// weights LOAD and the pipeline produces finite, correctly-shaped, backend-
// consistent output, not segmentation quality (which needs a real photo).

#define _CRT_SECURE_NO_WARNINGS  // std::getenv, matching tools/sam_segment.cpp

#include "brovisionml/sam.h"

#include "brotensor/runtime.h"

#include <algorithm>
#include <array>
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

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

// A deterministic non-square RGB image with some structure (a bright block) so
// a foreground click has something to latch onto. Quality is not asserted.
std::vector<uint8_t> make_image(int W, int H) {
    std::vector<uint8_t> img(static_cast<std::size_t>(W) * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const std::size_t o = (static_cast<std::size_t>(y) * W + x) * 3;
            const bool block = (x > W / 3 && x < 2 * W / 3 &&
                                y > H / 3 && y < 2 * H / 3);
            img[o + 0] = static_cast<uint8_t>(block ? 240 : ((x * 5) & 0xff));
            img[o + 1] = static_cast<uint8_t>(block ? 240 : ((y * 7) & 0xff));
            img[o + 2] = static_cast<uint8_t>(block ? 240 : (((x + y) * 3) & 0xff));
        }
    return img;
}

struct Variant {
    const char* subdir;          // weights/<subdir>/model.safetensors
    brovisionml::sam::SamConfig (*cfg)();
    const char* label;
};

// Run the full pipeline for one loaded model and assert shape/finiteness. When
// CUDA is available, also run on the GPU and assert CPU/CUDA parity. `cpu` is
// already loaded on the host.
void exercise(brovisionml::sam::Sam& cpu, const std::string& path,
              brovisionml::sam::SamConfig (*make_cfg)(), const char* label) {
    using namespace brovisionml::sam;

    const int W = 96, H = 72;
    const std::vector<uint8_t> img = make_image(W, H);
    const std::vector<std::array<float, 2>> pt = {{48.0f, 36.0f}};
    const std::vector<int> pt_labels = {1};

    cpu.set_image(img.data(), W, H, 3);

    // Point prompt, multimask -> 3 masks at original resolution.
    Segmentation seg = cpu.segment(pt, pt_labels, {}, /*multimask=*/true);
    check(seg.num == 3, "multimask returns 3 masks");
    check(seg.height == H && seg.width == W, "masks at original resolution");
    check(seg.logits.size() == static_cast<std::size_t>(seg.num) * H * W,
          "logits buffer size");
    check(static_cast<int>(seg.iou.size()) == seg.num, "one iou per mask");
    check(all_finite(seg.logits) && all_finite(seg.iou), "outputs finite");
    check(seg.best() >= 0 && seg.best() < seg.num, "best() in range");
    // best() must actually pick the max-iou mask.
    {
        int argmax = 0;
        for (int i = 1; i < seg.num; ++i)
            if (seg.iou[i] > seg.iou[argmax]) argmax = i;
        check(seg.best() == argmax, "best() selects max iou");
    }

    // Box prompt, single mask.
    {
        Segmentation b = cpu.segment({}, {}, {{10.f, 10.f, 80.f, 60.f}},
                                     /*multimask=*/false);
        check(b.num == 1 && b.height == H && b.width == W, "box single-mask shape");
        check(all_finite(b.logits), "box mask finite");
    }

    std::printf("  %s: loaded + ran on CPU (best #%d iou=%.3f)\n",
                label, seg.best(), seg.iou[seg.best()]);

    // CPU/CUDA parity on the real weights.
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        Sam gpu(make_cfg());
        gpu.load_file(path);
        gpu.to(brotensor::Device::CUDA);
        check(gpu.device() == brotensor::Device::CUDA, "migrated to CUDA");
        gpu.set_image(img.data(), W, H, 3);
        Segmentation g = gpu.segment(pt, pt_labels, {}, /*multimask=*/true);
        check(g.num == seg.num && g.height == H && g.width == W,
              "GPU segmentation shape matches CPU");
        float worst = 0.0f;
        for (std::size_t i = 0; i < seg.logits.size() && i < g.logits.size(); ++i)
            worst = std::max(worst, std::fabs(seg.logits[i] - g.logits[i]));
        if (worst > 1e-2f) {
            std::fprintf(stderr, "FAIL: %s CPU/CUDA logit diff %g > 1e-2\n",
                         label, worst);
            ++failures;
        }
        std::printf("  %s: CUDA parity max abs diff %g\n", label, worst);
    }
}

}  // namespace

int main() {
    using namespace brovisionml::sam;

    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;

    const Variant variants[] = {
        {"sam-vit-base",  &SamConfig::vit_b, "sam-vit-base"},
        {"sam-vit-large", &SamConfig::vit_l, "sam-vit-large"},
        {"sam-vit-huge",  &SamConfig::vit_h, "sam-vit-huge"},
    };

    brotensor::init();

    int ran = 0;
    for (const Variant& v : variants) {
        const std::string path = base + "/" + v.subdir + "/model.safetensors";
        if (!file_exists(path)) continue;
        std::printf("sam (real weights): %s\n", v.subdir);
        Sam sam(v.cfg());
        sam.load_file(path);
        exercise(sam, path, v.cfg, v.label);
        ++ran;
    }

    if (ran == 0) {
        std::printf("sam (real weights): no checkpoint under '%s' "
                    "(run scripts/download-weights.sh sam-vit-base); skipping\n",
                    base.empty() ? "<unset>" : base.c_str());
        return 0;
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed across %d model(s)\n",
                     failures, ran);
        return 1;
    }
    std::printf("sam (real weights): all checks passed across %d model(s)\n", ran);
    return 0;
}
