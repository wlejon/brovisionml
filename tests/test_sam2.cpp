// SAM 2.1 parity against the original facebookresearch/sam2 package.
//
// tests/ref/gen_sam2_fixture.py writes tests/fixtures/sam2_base_plus.bin (a
// safetensors file, gitignored): a synthetic four-frame scene, the reference
// preprocessing and encoder features of frame 0, image-predictor outputs for
// a click / a box / two clicks, and a tracking run (click on frame 0,
// propagate, correction click on frame 2, propagate again). This test loads
// the HF checkpoint of the same weights and checks:
//
//   * exact FP32 on the GPU (migrated under a CPU DeviceScope) against the
//     fixture: preprocessing, features, low-res logits, IoU, object scores
//     and the tracker's per-frame masks;
//   * the default mixed-precision GPU path against the FP32 one (mask IoU);
//   * ms per frame for encode and track, both paths.
//
// Skips (exit 0) without a GPU, the checkpoint or the fixture. Weights:
// BROVISIONML_SAM2_DIR (default <repo>/weights/sam2.1-hiera-base-plus).
#define _CRT_SECURE_NO_WARNINGS  // std::getenv

#include "brovisionml/sam2.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include "test_device.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace st = brotensor::safetensors;
using brotensor::Tensor;
using brovisionml::Sam2;
using brovisionml::Sam2Prompt;
using brovisionml::Sam2Tracker;

namespace {

int failures = 0;
void check(bool cond, const std::string& msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg.c_str()); ++failures; }
}

std::vector<float> fx(const st::File& f, const std::string& name) {
    const st::TensorView& v = f.get(name);
    const float* p = reinterpret_cast<const float*>(v.data);
    std::size_t n = 1;
    for (auto d : v.shape) n *= static_cast<std::size_t>(d);
    return std::vector<float>(p, p + n);
}

std::vector<float> host(const Tensor& t) {
    Tensor h = t.to(brotensor::Device::CPU);
    if (h.dtype != brotensor::Dtype::FP32) { Tensor c; brotensor::cast(h, c, brotensor::Dtype::FP32); h = c; }
    const float* p = h.host_f32();
    return std::vector<float>(p, p + h.size());
}

struct Diff { double max_abs = 0, max_ref = 0, mean_abs = 0; };
Diff diff(const std::vector<float>& a, const std::vector<float>& ref) {
    Diff d;
    if (a.size() != ref.size()) { d.max_abs = INFINITY; return d; }
    double sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - ref[i]);
        d.max_abs = std::max(d.max_abs, e);
        d.max_ref = std::max(d.max_ref, std::fabs(static_cast<double>(ref[i])));
        sum += e;
    }
    d.mean_abs = sum / static_cast<double>(a.size());
    return d;
}

double mask_iou(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t inter = 0, uni = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const bool x = a[i] > 0, y = b[i] > 0;
        inter += (x && y);
        uni += (x || y);
    }
    return uni ? static_cast<double>(inter) / static_cast<double>(uni) : 1.0;
}

// Feature parity: relative max error and relative mean error.
void check_close(const char* what, const std::vector<float>& a, const std::vector<float>& ref,
                 double rel_max, double rel_mean) {
    const Diff d = diff(a, ref);
    const double rm = d.max_abs / std::max(d.max_ref, 1e-12);
    const double rn = d.mean_abs / std::max(d.max_ref, 1e-12);
    std::printf("  %-22s max|d| %.3g (rel %.2e)  mean|d| %.3g (rel %.2e)\n", what, d.max_abs, rm,
                d.mean_abs, rn);
    check(rm <= rel_max && rn <= rel_mean, std::string(what) + " outside tolerance");
}

// Bilinear 256 -> (h, w) of a fixture low-res plane, through the same op the
// model uses (on the CPU), so the reference and ours meet at input resolution.
std::vector<float> upscale_ref(const std::vector<float>& low, int w, int h) {
    Tensor t = Tensor::from_host(low.data(), 1, static_cast<int>(low.size()));
    Tensor up;
    brotensor::interp2d_forward(t, 1, 1, 256, 256, h, w, 1, up);
    return host(up);
}

std::vector<uint8_t> frame_u8(const st::File& f, int i) {
    const std::vector<float> v = fx(f, "frame" + std::to_string(i));
    std::vector<uint8_t> out(v.size());
    for (std::size_t k = 0; k < v.size(); ++k) out[k] = static_cast<uint8_t>(v[k]);
    return out;
}

Sam2Prompt click(float x, float y, int label) {
    Sam2Prompt p;
    p.points = {{x, y}};
    p.labels = {label};
    return p;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct TrackRun {
    std::vector<std::vector<float>> logits;   // per frame after the last propagate
    std::vector<float> scores;
    double encode_ms = 0, track_ms = 0;
};

constexpr int kW = 640, kH = 480, kFrames = 4;

// Click on frame 0, propagate, correct frame 2, propagate. Compares each
// stage with the fixture when `fixture` is set.
TrackRun run_tracker(const Sam2& m, const st::File& fix, brotensor::Device gpu, bool compare) {
    TrackRun r;
    Sam2Tracker trk(m);
    for (int i = 0; i < kFrames; ++i) {
        const std::vector<uint8_t> px = frame_u8(fix, i);
        brotensor::sync(gpu);
        const auto t0 = std::chrono::steady_clock::now();
        trk.add_frame(px.data(), kW, kH, 3);
        brotensor::sync(gpu);
        if (i > 0) r.encode_ms += ms_since(t0) / (kFrames - 1);   // frame 0 warms up
    }
    auto cmp = [&](const char* stage, int i, const brovisionml::Sam2FrameResult& res,
                   const std::string& key) {
        if (!compare) return;
        const std::vector<float> ref = upscale_ref(fx(fix, key + "_low_res"), kW, kH);
        const float ref_score = fx(fix, key + "_score")[0];
        const Diff d = diff(res.logits, ref);
        const double iou = mask_iou(res.logits, ref);
        std::printf("  %s frame %d: mask IoU %.5f  max|dlogit| %.3g  score %.4f (ref %.4f)\n",
                    stage, i, iou, d.max_abs, res.object_score, ref_score);
        check(iou >= 0.995, std::string(stage) + " frame " + std::to_string(i) + " mask IoU");
        check(std::fabs(res.object_score - ref_score) <= 0.02f * std::max(1.0f, std::fabs(ref_score)),
              std::string(stage) + " frame " + std::to_string(i) + " object score");
    };
    cmp("prompt", 0, trk.prompt(0, click(300, 236, 1)), "trk0_f0");
    {
        const auto t0 = std::chrono::steady_clock::now();
        const auto all = trk.propagate();
        const double ms = ms_since(t0);
        (void)ms;
        for (int i = 0; i < kFrames; ++i) cmp("propagate", i, all[static_cast<std::size_t>(i)], "trk1_f" + std::to_string(i));
    }
    cmp("correct", 2, trk.prompt(2, click(362, 286, 0)), "trk2_f2");
    const auto all = trk.propagate();
    for (int i = 0; i < kFrames; ++i) {
        cmp("re-propagate", i, all[static_cast<std::size_t>(i)], "trk3_f" + std::to_string(i));
        r.logits.push_back(all[static_cast<std::size_t>(i)].logits);
        r.scores.push_back(all[static_cast<std::size_t>(i)].object_score);
    }
    // Per-frame tracking cost: re-track frame 3 from frames 0..2 a few times.
    for (int rep = 0; rep < 4; ++rep) {
        brotensor::sync(gpu);
        const auto t0 = std::chrono::steady_clock::now();
        trk.track(3);
        brotensor::sync(gpu);
        if (rep > 0) r.track_ms += ms_since(t0) / 3;
    }
    return r;
}

struct ImageRun { std::vector<std::vector<float>> logits; std::vector<std::vector<float>> iou; };

ImageRun run_image(const Sam2& m0, Sam2& m, const st::File& fix, bool compare) {
    (void)m0;
    ImageRun r;
    const std::vector<uint8_t> px = frame_u8(fix, 0);
    m.set_image(px.data(), kW, kH, 3);
    struct P { const char* name; Sam2Prompt p; bool multi; };
    std::vector<P> ps;
    ps.push_back({"point", click(300, 236, 1), true});
    {
        Sam2Prompt b;
        b.has_box = true;
        b.box = {226, 182, 382, 292};
        ps.push_back({"box", b, false});
    }
    {
        Sam2Prompt t;
        t.points = {{300, 236}, {120, 300}};
        t.labels = {1, 0};
        ps.push_back({"two", t, false});
    }
    for (const P& p : ps) {
        const brovisionml::Sam2Segmentation s = m.segment(p.p, p.multi);
        r.logits.push_back(s.logits);
        r.iou.push_back(s.iou);
        if (!compare) continue;
        const std::string k = std::string("img_") + p.name;
        const std::vector<float> ref_logits = fx(fix, k + "_logits");
        const std::vector<float> ref_iou = fx(fix, k + "_iou");
        check(s.num == static_cast<int>(ref_iou.size()), k + " mask count");
        const Diff dl = diff(host(s.low_res), fx(fix, k + "_low_res"));
        std::printf("  %-6s low-res max|d| %.3g (of %.3g), mean %.3g;", p.name, dl.max_abs, dl.max_ref, dl.mean_abs);
        const std::size_t plane = static_cast<std::size_t>(kW) * kH;
        for (int i = 0; i < s.num; ++i) {
            const std::vector<float> a(s.logits.begin() + static_cast<long long>(i * plane),
                                       s.logits.begin() + static_cast<long long>((i + 1) * plane));
            const std::vector<float> b(ref_logits.begin() + static_cast<long long>(i * plane),
                                       ref_logits.begin() + static_cast<long long>((i + 1) * plane));
            const double iou = mask_iou(a, b);
            std::printf(" mask%d IoU %.5f iou %.5f (ref %.5f)", i, iou, s.iou[static_cast<std::size_t>(i)],
                        ref_iou[static_cast<std::size_t>(i)]);
            check(iou >= 0.995, k + " mask " + std::to_string(i) + " IoU vs reference");
            check(std::fabs(s.iou[static_cast<std::size_t>(i)] - ref_iou[static_cast<std::size_t>(i)]) < 2e-3f,
                  k + " predicted IoU");
        }
        std::printf("\n");
        check(dl.max_abs <= 0.05 * std::max(dl.max_ref, 1.0), k + " low-res logits");
    }
    return r;
}

}  // namespace

int run();

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }
}

int run() {
    brotensor::init();
    const brotensor::Device gpu = brovisionml_test::preferred_gpu();
    std::string dir = BROVISIONML_SAM2_DIR;
    if (const char* e = std::getenv("BROVISIONML_SAM2_DIR")) dir = e;
    const std::string fixture = std::string(BROVISIONML_FIXTURES_DIR) + "/sam2_base_plus.bin";

    // Config presets agree with what config.json says (no weights needed for
    // the presets themselves; the comparison needs the file).
    {
        const auto bp = brovisionml::sam2::Sam2Config::base_plus();
        check(bp.hiera.total_blocks() == 24 && bp.grid() == 64, "base_plus preset shape");
        check(brovisionml::sam2::Sam2Config::large().hiera.total_blocks() == 48, "large preset depth");
    }
    if (gpu == brotensor::Device::CPU) { std::printf("SKIP: no GPU backend\n"); return failures ? 1 : 0; }
    if (!std::filesystem::exists(dir + "/model.safetensors")) {
        std::printf("SKIP: no SAM 2.1 checkpoint at %s (set BROVISIONML_SAM2_DIR)\n", dir.c_str());
        return failures ? 1 : 0;
    }
    if (!std::filesystem::exists(fixture)) {
        std::printf("SKIP: no fixture %s (run tests/ref/gen_sam2_fixture.py)\n", fixture.c_str());
        return failures ? 1 : 0;
    }
    {
        const auto cfg = brovisionml::sam2::Sam2Config::from_json_file(dir + "/config.json");
        const auto bp = brovisionml::sam2::Sam2Config::base_plus();
        check(cfg.hiera.blocks_per_stage == bp.hiera.blocks_per_stage &&
              cfg.hiera.embed_dim_per_stage == bp.hiera.embed_dim_per_stage &&
              cfg.hiera.global_attention_blocks == bp.hiera.global_attention_blocks,
              "config.json matches the base_plus preset");
    }
    const st::File fix = st::File::open(fixture);
    std::printf("SAM 2.1 base-plus on %s\n", brovisionml_test::device_name(gpu));

    // ── Exact FP32 on the GPU vs the reference ──
    Sam2 m32;
    m32.load(dir);
    {
        brotensor::DeviceScope scope(brotensor::Device::CPU);
        m32.to(gpu);
    }
    std::printf("FP32:\n");
    {
        const std::vector<uint8_t> px = frame_u8(fix, 0);
        const Tensor pre = m32.preprocess(px.data(), kW, kH, 3);
        check_close("preprocess", host(pre), fx(fix, "pixels"), 1e-5, 1e-6);
        check_close("preprocess (device)", host(m32.preprocess_device(px.data(), kW, kH, 3)),
                    fx(fix, "pixels"), 1e-5, 1e-6);
        // Encode the reference's own pixels so the features isolate the encoder.
        const std::vector<float> ref_px = fx(fix, "pixels");
        Tensor rp = Tensor::from_host(ref_px.data(), 1, static_cast<int>(ref_px.size())).to(gpu);
        const brovisionml::sam2::ImageFeatures feats = m32.encoder().encode(rp);
        check_close("top", host(feats.top), fx(fix, "top"), 2e-3, 1e-4);
        check_close("top_pos", host(m32.encoder().top_pos()), fx(fix, "top_pos"), 1e-5, 1e-6);
        check_close("feat_s0", host(feats.feat_s0), fx(fix, "feat_s0"), 2e-3, 1e-4);
        check_close("feat_s1", host(feats.feat_s1), fx(fix, "feat_s1"), 2e-3, 1e-4);
    }
    const ImageRun img32 = run_image(m32, m32, fix, true);
    const TrackRun trk32 = run_tracker(m32, fix, gpu, true);

    // ── Default (mixed precision on an FP16 GPU) vs FP32 ──
    Sam2 m16;
    m16.load(dir);
    m16.to(gpu);
    std::printf("default precision (%s):\n", m16.encoder().mixed_precision() ? "mixed FP16" : "FP32");
    const ImageRun img16 = run_image(m16, m16, fix, false);
    for (std::size_t p = 0; p < img16.logits.size(); ++p) {
        const double iou = mask_iou(img16.logits[p], img32.logits[p]);
        std::printf("  image prompt %zu: mask IoU vs FP32 %.5f\n", p, iou);
        check(iou >= 0.98, "mixed-precision image mask IoU vs FP32");
    }
    const TrackRun trk16 = run_tracker(m16, fix, gpu, false);
    for (int i = 0; i < kFrames; ++i) {
        const double iou = mask_iou(trk16.logits[static_cast<std::size_t>(i)], trk32.logits[static_cast<std::size_t>(i)]);
        std::printf("  track frame %d: mask IoU vs FP32 %.5f  score %.3f (FP32 %.3f)\n", i, iou,
                    trk16.scores[static_cast<std::size_t>(i)], trk32.scores[static_cast<std::size_t>(i)]);
        check(iou >= 0.98, "mixed-precision track mask IoU vs FP32");
    }

    // Another size through the same code: hiera-tiny (head_dim 96, its own
    // global blocks) should find the same bird.
    {
        std::string tiny = BROVISIONML_SAM2_TINY_DIR;
        if (const char* e = std::getenv("BROVISIONML_SAM2_TINY_DIR")) tiny = e;
        if (std::filesystem::exists(tiny + "/model.safetensors")) {
            Sam2 mt;
            mt.load(tiny);
            mt.to(gpu);
            const ImageRun it = run_image(mt, mt, fix, false);
            const TrackRun tt = run_tracker(mt, fix, gpu, false);
            const double iou_img = mask_iou(it.logits[1], img32.logits[1]);   // box prompt
            double iou_trk = 1.0;
            for (int i = 0; i < kFrames; ++i)
                iou_trk = std::min(iou_trk, mask_iou(tt.logits[static_cast<std::size_t>(i)],
                                                     trk32.logits[static_cast<std::size_t>(i)]));
            std::printf("hiera-tiny: box mask IoU vs base-plus %.4f, worst track frame %.4f; "
                        "encode %.1f ms, track %.1f ms\n", iou_img, iou_trk, tt.encode_ms, tt.track_ms);
            check(iou_img >= 0.9 && iou_trk >= 0.85, "hiera-tiny agrees with base-plus");
        } else {
            std::printf("(no hiera-tiny at %s; size check skipped)\n", tiny.c_str());
        }
    }

    std::printf("timing (ms/frame, 640x480 frames):\n");
    std::printf("  FP32    encode %.1f  track %.1f\n", trk32.encode_ms, trk32.track_ms);
    std::printf("  default encode %.1f  track %.1f\n", trk16.encode_ms, trk16.track_ms);

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("PASS\n");
    return 0;
}
