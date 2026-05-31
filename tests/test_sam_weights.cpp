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
// The input is a procedurally-rendered high-contrast disk with an exactly-known
// ground-truth mask, so we can assert correctness, not just that the pipeline
// runs: a single click at the disk's center (and a box around it) must recover
// the disk with high IoU. This is what distinguishes "loaded correctly" from
// "loaded plausibly-but-wrong" — a transposed weight, a wrong activation, or an
// off-by-one positional encoding stays finite and backend-consistent but tanks
// the IoU. We still also assert shape, finiteness, and CPU/CUDA parity.

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

// Minimum IoU a correctly-loaded model must reach recovering the disk. The real
// pipeline measures ~0.996; 0.90 is a generous regression tripwire, not a tight
// fit — a transposed weight or wrong activation drops this to near 0.
constexpr float kMinIoU = 0.90f;

// A hard-edged filled disk (bright) over a gradient background — an unambiguous
// "object" with an exactly-known ground-truth mask. Fills `gt` (W*H, 1 inside
// the disk) alongside the returned interleaved-RGB buffer.
std::vector<uint8_t> make_disk_image(int W, int H, int cx, int cy, int r,
                                     std::vector<uint8_t>& gt) {
    std::vector<uint8_t> img(static_cast<std::size_t>(W) * H * 3);
    gt.assign(static_cast<std::size_t>(W) * H, 0);
    const long long r2 = static_cast<long long>(r) * r;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const std::size_t px = static_cast<std::size_t>(y) * W + x;
            const long long dx = x - cx, dy = y - cy;
            const bool inside = dx * dx + dy * dy <= r2;
            const std::size_t o = px * 3;
            if (inside) {
                img[o + 0] = img[o + 1] = img[o + 2] = 245;
                gt[px] = 1;
            } else {
                img[o + 0] = static_cast<uint8_t>((x * 5) & 0xff);
                img[o + 1] = static_cast<uint8_t>((y * 7) & 0xff);
                img[o + 2] = static_cast<uint8_t>(((x + y) * 3) & 0xff);
            }
        }
    return img;
}

// IoU of a binarized mask (logit > 0) against the ground-truth mask.
float mask_iou(const float* logits, const std::vector<uint8_t>& gt) {
    long long inter = 0, uni = 0;
    for (std::size_t i = 0; i < gt.size(); ++i) {
        const bool m = logits[i] > 0.0f;
        const bool g = gt[i] != 0;
        if (m && g) ++inter;
        if (m || g) ++uni;
    }
    return uni ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

struct Variant {
    const char* subdir;          // weights/<subdir>/model.safetensors
    brovisionml::sam::SamConfig (*cfg)();
    const char* label;
};

// Run the full pipeline for one loaded model and assert shape/finiteness/IoU.
// When CUDA is available, also run on the GPU and assert CPU/CUDA parity. `cpu`
// is already loaded on the host.
void exercise(brovisionml::sam::Sam& cpu, const std::string& path,
              brovisionml::sam::SamConfig (*make_cfg)(), const char* label) {
    using namespace brovisionml::sam;

    const int W = 320, H = 256, cx = 160, cy = 128, r = 80;
    std::vector<uint8_t> gt;
    const std::vector<uint8_t> img = make_disk_image(W, H, cx, cy, r, gt);
    const std::vector<std::array<float, 2>> pt = {
        {static_cast<float>(cx), static_cast<float>(cy)}};
    const std::vector<int> pt_labels = {1};

    cpu.set_image(img.data(), W, H, 3);

    // Point prompt at the disk center, multimask -> 3 masks at original res.
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

    // Correctness: the click must recover the disk.
    const int b = seg.best();
    const float iou_best =
        mask_iou(seg.logits.data() + static_cast<std::size_t>(b) * H * W, gt);
    std::printf("  %s: point disk IoU best=%.3f (head iou=%.3f)\n",
                label, iou_best, seg.iou[b]);
    check(iou_best >= kMinIoU, "point: best mask recovers the disk (IoU)");

    // Box prompt tight around the disk -> single mask should also recover it.
    {
        const std::array<float, 4> box = {
            static_cast<float>(cx - r - 4), static_cast<float>(cy - r - 4),
            static_cast<float>(cx + r + 4), static_cast<float>(cy + r + 4)};
        Segmentation bs = cpu.segment({}, {}, {box}, /*multimask=*/false);
        check(bs.num == 1 && bs.height == H && bs.width == W,
              "box single-mask shape");
        check(all_finite(bs.logits), "box mask finite");
        const float biou = mask_iou(bs.logits.data(), gt);
        std::printf("  %s: box disk IoU=%.3f\n", label, biou);
        check(biou >= kMinIoU, "box: mask recovers the disk (IoU)");
    }

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
