// Depth-Anything-V2 numeric parity against the HF Python reference.
//
// Unlike test_depth_anything.cpp (which only proves shape / finiteness /
// variation / CPU-CUDA self-consistency), this asserts that brovisionml's
// output matches the actual HuggingFace `DepthAnythingForDepthEstimation`
// output, pixel for pixel, within a numeric tolerance — the real correctness
// proof the other test is explicit about *not* providing.
//
// The reference is a "golden dump": a throwaway PyTorch script (kept OUTSIDE
// this repo — Python is never part of the brovisionml stack) runs HF once and
// writes golden_*.bin holding the exact input bytes, the post-preprocess pixel
// tensor, the selected backbone feature maps, and the predicted depth. This
// test reads that dump, runs brovisionml on the identical input, and compares.
// The goldens live under weights/.../golden (gitignored, like the checkpoints);
// when absent the test prints why and exits 0 (clean skip).
//
// The square-518 case is the tight oracle: at a 518x518 input the DPT resize is
// identity and DINOv2's position embedding is used at its native grid (no
// interpolation), so brovisionml and HF should agree to FP32 round-off. It
// bisects the pipeline — preprocess, backbone, full estimate — so a divergence
// localizes to one stage. The wide case additionally exercises the resize +
// position-embedding-interpolation path end to end.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/depth_anything.h"
#include "brovisionml/dinov2.h"
#include "brovisionml/dpt_preprocess.h"

#include "brotensor/runtime.h"

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

// ── Golden-dump reader (format BVMLDPG2; see gen_golden.py) ──
struct Golden {
    int W = 0, H = 0, model_w = 0, model_h = 0;
    int n_stages = 0, stage_seq = 0, stage_dim = 0;
    std::vector<uint8_t> input;            // H*W*3 HWC
    std::vector<float>   pixels;           // 3*model_h*model_w NCHW
    std::vector<float>   depth;            // H*W
    std::vector<std::vector<float>> stages; // each stage_seq*stage_dim
};

template <class T>
bool read_vec(std::ifstream& f, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(f);
}

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 8) != "BVMLDPG2") return false;
    int version = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&g.W), 4);
    f.read(reinterpret_cast<char*>(&g.H), 4);
    f.read(reinterpret_cast<char*>(&g.model_w), 4);
    f.read(reinterpret_cast<char*>(&g.model_h), 4);
    f.read(reinterpret_cast<char*>(&g.n_stages), 4);
    f.read(reinterpret_cast<char*>(&g.stage_seq), 4);
    f.read(reinterpret_cast<char*>(&g.stage_dim), 4);
    if (!f) return false;

    const std::size_t in_n = static_cast<std::size_t>(g.W) * g.H * 3;
    const std::size_t px_n = static_cast<std::size_t>(g.model_w) * g.model_h * 3;
    const std::size_t dp_n = static_cast<std::size_t>(g.W) * g.H;
    if (!read_vec(f, g.input, in_n)) return false;
    if (!read_vec(f, g.pixels, px_n)) return false;
    if (!read_vec(f, g.depth, dp_n)) return false;
    g.stages.resize(g.n_stages);
    for (int s = 0; s < g.n_stages; ++s) {
        const std::size_t sn = static_cast<std::size_t>(g.stage_seq) * g.stage_dim;
        if (!read_vec(f, g.stages[s], sn)) return false;
    }
    return true;
}

// max-abs and mean-abs difference between two equal-length float ranges.
struct Diff { double max_abs = 0.0, mean_abs = 0.0; };
Diff diff(const float* a, const float* b, std::size_t n) {
    Diff d;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        d.max_abs = std::max(d.max_abs, e);
        sum += e;
    }
    d.mean_abs = n ? sum / static_cast<double>(n) : 0.0;
    return d;
}

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

using namespace brovisionml;

}  // namespace

int main() {
    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
    const std::string dir = base + "/Depth-Anything-V2-Small";
    const std::string gdir = dir + "/golden";

    if (!file_exists(dir + "/model.safetensors") ||
        !file_exists(gdir + "/golden_square518.bin")) {
        std::printf("test_depth_parity: no checkpoint+golden under '%s' — skipping "
                    "(generate goldens with the out-of-repo gen_golden.py).\n",
                    gdir.c_str());
        return 0;
    }

    brotensor::init();

    // ─────────────────────────────────────────────────────────────────────────
    // square-518: the tight oracle. Resize is identity, no pos-embed interp.
    // ─────────────────────────────────────────────────────────────────────────
    {
        Golden g;
        check(load_golden(gdir + "/golden_square518.bin", g), "load square518 golden");
        std::printf("square518: in %dx%d model %dx%d, %d stages [%dx%d]\n",
                    g.W, g.H, g.model_w, g.model_h, g.n_stages, g.stage_seq, g.stage_dim);

        auto cfg = depth::DepthAnythingConfig::v2_small();

        // (a) Preprocess parity — isolates resize+rescale+normalize.
        dpt::PreprocessedImage pp = dpt::preprocess(
            g.input.data(), g.W, g.H, 3, cfg.input_size, cfg.multiple,
            cfg.keep_aspect_ratio);
        check(pp.transform.resized_w == g.model_w && pp.transform.resized_h == g.model_h,
              "preprocess model dims match HF");
        Diff dpx = diff(pp.pixels.host_f32(), g.pixels.data(), g.pixels.size());
        std::printf("  preprocess pixels: max=%.3e mean=%.3e\n", dpx.max_abs, dpx.mean_abs);

        // (b) Backbone parity — isolates the DINOv2 forward against HF feature_maps.
        dinov2::Backbone bb(cfg.backbone);
        bb.load(dir);
        dinov2::BackboneOutput bo = bb.encode(pp.pixels, g.model_h, g.model_w);
        check(static_cast<int>(bo.feature_maps.size()) == g.n_stages,
              "backbone stage count matches HF");
        double stage_max = 0.0;
        for (int s = 0; s < g.n_stages && s < (int)bo.feature_maps.size(); ++s) {
            const auto& fm = bo.feature_maps[s];
            Diff ds = diff(fm.host_f32(), g.stages[s].data(),
                           std::min<std::size_t>(g.stages[s].size(),
                               static_cast<std::size_t>(g.stage_seq) * g.stage_dim));
            std::printf("  backbone stage %d: max=%.3e mean=%.3e\n", s, ds.max_abs, ds.mean_abs);
            stage_max = std::max(stage_max, ds.max_abs);
        }

        // (c) Full estimate parity — the end-to-end depth map.
        depth::DepthEstimator est(cfg);
        est.load(dir);
        depth::DepthMap dm = est.estimate(g.input.data(), g.W, g.H, 3);
        check(dm.depth.size() == g.depth.size(), "depth size matches golden");
        Diff dd = diff(dm.depth.data(), g.depth.data(), g.depth.size());
        float lo = *std::min_element(g.depth.begin(), g.depth.end());
        float hi = *std::max_element(g.depth.begin(), g.depth.end());
        double rel = dd.max_abs / std::max(1e-6, (double)(hi - lo));
        std::printf("  depth: max=%.3e mean=%.3e (range %.3f..%.3f, rel-max=%.3e)\n",
                    dd.max_abs, dd.mean_abs, lo, hi, rel);

        // Tripwires set from the measured FP32 round-off floor (preprocess ~7e-7,
        // backbone stages ~5e-4 max, depth rel ~1.4e-5) with generous headroom: a
        // real transcription bug moves these by orders of magnitude, not factors.
        check(dpx.max_abs   < 1e-5, "square518 preprocess parity (exact)");
        check(stage_max     < 5e-3, "square518 backbone parity");
        check(rel           < 1e-3, "square518 depth parity (HF, relative)");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // wide: exercises resize + pos-embed interpolation end to end. Looser — the
    // known DINOv2 bicubic a=-0.75 vs Catmull-Rom a=-0.5 gap lives here.
    // ─────────────────────────────────────────────────────────────────────────
    if (file_exists(gdir + "/golden_wide.bin")) {
        Golden g;
        check(load_golden(gdir + "/golden_wide.bin", g), "load wide golden");
        auto wcfg = depth::DepthAnythingConfig::v2_small();
        // Isolate the image resize (broimage bicubic vs HF PIL BICUBIC) from the
        // in-model pos-embed interpolation: compare preprocessed pixels too.
        dpt::PreprocessedImage wpp = dpt::preprocess(
            g.input.data(), g.W, g.H, 3, wcfg.input_size, wcfg.multiple,
            wcfg.keep_aspect_ratio);
        Diff wpx = diff(wpp.pixels.host_f32(), g.pixels.data(), g.pixels.size());
        std::printf("wide preprocess pixels: max=%.3e mean=%.3e\n",
                    wpx.max_abs, wpx.mean_abs);
        depth::DepthEstimator est(wcfg);
        est.load(dir);
        depth::DepthMap dm = est.estimate(g.input.data(), g.W, g.H, 3);
        check(dm.depth.size() == g.depth.size(), "wide depth size matches golden");
        Diff dd = diff(dm.depth.data(), g.depth.data(), g.depth.size());
        float lo = *std::min_element(g.depth.begin(), g.depth.end());
        float hi = *std::max_element(g.depth.begin(), g.depth.end());
        double rel = dd.max_abs / std::max(1e-6, (double)(hi - lo));
        std::printf("wide %dx%d -> model %dx%d depth: max=%.3e mean=%.3e rel-max=%.3e\n",
                    g.W, g.H, g.model_w, g.model_h, dd.max_abs, dd.mean_abs, rel);
        // The DINOv2 position-embedding interpolation now matches torch (interp2d
        // mode 3, a=-0.75), so the mean depth error is ~0.1%. The residual max
        // (~0.7%) is the preprocess image resize: broimage's bicubic differs from
        // PIL only at image borders / hard edges (PIL renormalizes boundary
        // weights), visible above as a localized ~0.1 pixel spike against a ~3e-3
        // mean. It is a preprocessing nuance, not a model gap — the model path is
        // exact (see square518). This guards against regression; it tightens once
        // broimage's resize matches PIL's border handling.
        check(wpx.mean_abs < 1e-2, "wide preprocess parity (interior matches PIL)");
        check(rel < 1e-2, "wide depth parity (within broimage resize border gap)");
    }

    if (failures == 0) { std::printf("test_depth_parity: OK\n"); return 0; }
    std::printf("test_depth_parity: %d failure(s)\n", failures);
    return 1;
}
