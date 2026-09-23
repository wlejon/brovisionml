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
#include "brovisionml/sam_preprocess.h"

#include "brotensor/runtime.h"

#include "test_device.h"

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

// CPU-vs-GPU bounds, stage by stage (sam-vit-base, the disk image below).
//
// The production GPU path is mixed precision: the encoder's GEMMs and the mask
// decoder's attention q/k/v/out projections run FP16 (the decoder's since
// 59d79d6, 2026-06-11). The old single 1.5e-2 bound on the final logits dates
// from when only the encoder was FP16 and was never met afterwards. Measured:
// rounding every FP16-path operand of the decoder on the CPU (weights,
// activations, projections) moves the low-res logits (|max| ~14.5) by 0.169
// from FP32 — the same as the GPU FP16 decoder (0.17) — so ~0.17 is the
// inherent cost of the FP16 decoder, not a fault. Hence:
//   * kernel faults: the decoder kept FP32 on the GPU must match the CPU
//     tightly (observed 5.6e-4);
//   * the FP16 production path: 0.25 on low-res and final logits (~1.5x the
//     measured FP16 cost). The FP16-score attention fault 422e4d7 fixed in
//     brotensor sat at 0.70 low-res / 0.42 final, well outside it.
constexpr float kMaxEmbedDiff    = 1.0e-2f;   // FP16 encoder; observed 1.7e-3
constexpr float kMaxFp32DecDiff  = 5.0e-3f;   // FP32 decoder on GPU; observed 5.6e-4
constexpr float kMaxLogitDiff    = 0.25f;     // FP16 decoder, low-res and final

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

    // Encoder stage on its own: the dense image embedding must be run-to-run
    // deterministic on the CPU and track the CPU on the GPU, so a parity
    // failure below names the encoder or the decoder rather than "SAM".
    const brotensor::Device gpu = brovisionml_test::preferred_gpu();
    {
        ImageEncoder enc(make_cfg().encoder);
        enc.load_file(path);
        PreprocessedImage pp = preprocess(img.data(), W, H, 3, make_cfg().encoder.img_size);
        brotensor::Tensor e1 = enc.encode(pp.pixels);
        brotensor::Tensor e2 = enc.encode(pp.pixels);
        float rr = 0.0f, scale = 0.0f;
        for (int i = 0; i < e1.size(); ++i) {
            rr = std::max(rr, std::fabs(e1.host_f32()[i] - e2.host_f32()[i]));
            scale = std::max(scale, std::fabs(e1.host_f32()[i]));
        }
        std::printf("  %s: CPU embedding run-to-run max abs diff %g (max |e| %g)\n",
                    label, rr, scale);
        check(rr == 0.0f, "CPU image embedding is run-to-run deterministic");
        if (gpu != brotensor::Device::CPU) {
            enc.to(gpu);
            brotensor::Tensor eg = enc.encode(pp.pixels.to(gpu)).to(brotensor::Device::CPU);
            float gd = 0.0f;
            for (int i = 0; i < e1.size() && i < eg.size(); ++i)
                gd = std::max(gd, std::fabs(e1.host_f32()[i] - eg.host_f32()[i]));
            std::printf("  %s: %s embedding vs CPU max abs diff %g\n", label,
                        brovisionml_test::device_name(gpu), gd);
            check(gd <= kMaxEmbedDiff, "GPU image embedding tracks the CPU");

            // Decoder stage on the SAME (CPU) embedding: low-res mask logits.
            PromptEncoder pe(make_cfg().prompt);
            MaskDecoder md(make_cfg().decoder);
            pe.load_file(path);
            md.load_file(path);
            PromptInput in;
            in.labels = pt_labels;
            float mx = 0, my = 0;
            apply_coords(pp.transform, pt[0][0], pt[0][1], mx, my);
            in.points.push_back({mx, my});
            PromptEmbeddings pc = pe.encode(in);
            DecodedMasks dc = md.decode(e1, pe.dense_pe(), pc.sparse, pc.dense, true);
            pe.to(gpu);
            md.to(gpu);
            PromptEmbeddings pg = pe.encode(in);
            DecodedMasks dg = md.decode(e1.to(gpu), pe.dense_pe(), pg.sparse, pg.dense, true);
            brotensor::Tensor mc = dc.masks, mg = dg.masks.to(brotensor::Device::CPU);
            float dd = 0.0f, ms = 0.0f;
            for (int i = 0; i < mc.size() && i < mg.size(); ++i) {
                dd = std::max(dd, std::fabs(mc.host_f32()[i] - mg.host_f32()[i]));
                ms = std::max(ms, std::fabs(mc.host_f32()[i]));
            }
            std::printf("  %s: %s low-res mask logits vs CPU (same embedding) max abs diff %g"
                        " (max |logit| %g)\n", label, brovisionml_test::device_name(gpu), dd, ms);
            check(dd <= kMaxLogitDiff, "GPU FP16 decoder low-res logits within the FP16 bound");

            // The same decoder kept FP32 on the GPU (migrated while the CPU is the
            // default device, so compute_dtype() says FP32): separates FP16
            // rounding from a kernel fault.
            MaskDecoder md32(make_cfg().decoder);
            md32.load_file(path);
            {
                brotensor::DeviceScope cpu_default(brotensor::Device::CPU);
                md32.to(gpu);
            }
            DecodedMasks d32 = md32.decode(e1.to(gpu), pe.dense_pe(), pg.sparse, pg.dense, true);
            brotensor::Tensor m32 = d32.masks.to(brotensor::Device::CPU);
            float d32d = 0.0f;
            for (int i = 0; i < mc.size() && i < m32.size(); ++i)
                d32d = std::max(d32d, std::fabs(mc.host_f32()[i] - m32.host_f32()[i]));
            std::printf("  %s: %s FP32 decoder low-res mask logits vs CPU max abs diff %g\n",
                        label, brovisionml_test::device_name(gpu), d32d);
            check(d32d <= kMaxFp32DecDiff, "GPU FP32 decoder matches the CPU");
        }
    }

    // CPU/GPU parity on the real weights.
    if (gpu != brotensor::Device::CPU) {
        const char* dev = brovisionml_test::device_name(gpu);
        Sam gpu_sam(make_cfg());
        gpu_sam.load_file(path);
        gpu_sam.to(gpu);
        check(gpu_sam.device() == gpu, "migrated to GPU");
        gpu_sam.set_image(img.data(), W, H, 3);
        Segmentation g = gpu_sam.segment(pt, pt_labels, {}, /*multimask=*/true);
        check(g.num == seg.num && g.height == H && g.width == W,
              "GPU segmentation shape matches CPU");
        float worst = 0.0f;
        for (std::size_t i = 0; i < seg.logits.size() && i < g.logits.size(); ++i)
            worst = std::max(worst, std::fabs(seg.logits[i] - g.logits[i]));
        if (worst > kMaxLogitDiff) {
            std::fprintf(stderr, "FAIL: %s CPU/%s logit diff %g > %g\n",
                         label, dev, worst, kMaxLogitDiff);
            ++failures;
        }
        std::printf("  %s: %s parity max abs diff %g\n", label, dev, worst);
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
