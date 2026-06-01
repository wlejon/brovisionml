// SegFormer SegformerDetector: real-weights END-TO-END parity test.
//
// Weights-gated (skips cleanly when weights/segformer-b0-ade/model.safetensors
// or golden_segformer.bin is absent). Two gates:
//
//   Gate 1 (neural): feed the EXACT normalized input tensor from the golden
//   (resize convention out of the loop) and compare the decode-head logits
//   (150,128,128) against the golden — tight transformer parity, like the
//   dinov2/depth models. The MiT encoder is a moderately deep transformer with
//   spatial-reduction cross-attention + a depthwise MixFFN, so a large logit
//   diff points at a real bug (reshape/transpose, concat order, the SR conv, BN
//   eps) rather than FP accumulation.
//
//   Gate 2 (end-to-end): detect() from the raw resized RGB bytes -> class map,
//   compared to the golden argmax map. Boundary pixels can flip under
//   interpolation rounding, so the gate is a high agreement fraction, not exact.
//   colorize() is checked against the embedded ADE20K palette.
//
// A CUDA device, when present, is exercised on both gates: CUDA logits must
// track CPU tightly, and the class maps must agree to within rounding.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/segformer.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

using brovisionml::segformer::Logits;
using brovisionml::segformer::SegformerConfig;
using brovisionml::segformer::SegformerDetector;
using brovisionml::segformer::SegMap;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

namespace {

template <class T>
bool read_one(std::ifstream& f, T& x) {
    f.read(reinterpret_cast<char*>(&x), sizeof(T));
    return static_cast<bool>(f);
}
template <class T>
bool read_vec(std::ifstream& f, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(f);
}
bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

struct Golden {
    int W = 0, H = 0;
    std::vector<uint8_t> resized;      // H*W*3 HWC RGB (pre-normalize)
    std::vector<float> input;          // 3*H*W NCHW normalized
    int LC = 0, LH = 0, LW = 0;
    std::vector<float> logits;         // LC*LH*LW
    int CW = 0, CH = 0;
    std::vector<uint8_t> classmap;     // CH*CW
    std::vector<uint8_t> palette;      // 150*3
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 8) != "BVMLSEG1") return false;
    int version = 0;
    if (!read_one(f, version)) return false;
    if (!read_one(f, g.W) || !read_one(f, g.H)) return false;
    if (!read_vec(f, g.resized, static_cast<std::size_t>(g.H) * g.W * 3)) return false;
    if (!read_vec(f, g.input, static_cast<std::size_t>(3) * g.H * g.W)) return false;
    if (!read_one(f, g.LC) || !read_one(f, g.LH) || !read_one(f, g.LW)) return false;
    if (!read_vec(f, g.logits, static_cast<std::size_t>(g.LC) * g.LH * g.LW)) return false;
    if (!read_one(f, g.CW) || !read_one(f, g.CH)) return false;
    if (!read_vec(f, g.classmap, static_cast<std::size_t>(g.CH) * g.CW)) return false;
    int np = 0;
    if (!read_one(f, np)) return false;
    if (!read_vec(f, g.palette, static_cast<std::size_t>(np) * 3)) return false;
    return true;
}

void diff(const std::vector<float>& a, const std::vector<float>& b,
          double& max_abs, double& mean_abs) {
    double m = 0.0, s = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        m = std::max(m, d);
        s += d;
    }
    max_abs = m;
    mean_abs = n ? s / (double)n : 0.0;
}

double agreement(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0;
    std::size_t hit = 0;
    for (std::size_t i = 0; i < n; ++i) if (a[i] == b[i]) ++hit;
    return (double)hit / (double)n;
}

void run_case(const std::string& dir, const std::string& path) {
    Golden g;
    if (!load_golden(path, g)) {
        std::printf("  FAIL  could not load/parse golden %s\n", path.c_str());
        ++g_failures;
        return;
    }
    std::printf("  case %s  (in %dx%d  logits %dx%dx%d  classmap %dx%d)\n",
                path.c_str(), g.W, g.H, g.LC, g.LH, g.LW, g.CW, g.CH);

    SegformerConfig cfg;
    cfg.model_size = g.W;   // square
    SegformerDetector det(cfg);
    det.load(dir);

    // ── Gate 1 (CPU): exact normalized input -> decode-head logits ──
    brotensor::Tensor px = brotensor::Tensor::mat(1, static_cast<int>(g.input.size()));
    std::copy(g.input.begin(), g.input.end(), px.host_f32_mut());
    Logits lg = det.infer_logits_from_tensor(px);
    CHECK(lg.channels == g.LC && lg.height == g.LH && lg.width == g.LW);
    double mx = 0, mn = 0;
    if (lg.data.size() == g.logits.size()) {
        diff(lg.data, g.logits, mx, mn);
        std::printf("    CPU Gate1 logits vs golden: max-abs=%.3e  mean-abs=%.3e\n", mx, mn);
        // mean-abs is the tight gate: transformer parity should be near-exact.
        // max-abs is a worst-single-logit tripwire; logits span ~[-15,15].
        CHECK(mn < 5e-3);
        CHECK(mx < 5e-2);
    } else {
        CHECK(false);
    }

    // ── Gate 2 (CPU): end-to-end class map ──
    SegMap sm = det.detect(g.resized.data(), g.W, g.H, 3);
    CHECK(sm.width == g.CW && sm.height == g.CH);
    double agree = agreement(sm.classes, g.classmap);
    std::printf("    CPU Gate2 class agreement: %.4f\n", agree);
    CHECK(agree >= 0.98);

    // colorize matches the palette mapping for the produced classes.
    std::vector<uint8_t> col = SegformerDetector::colorize(sm);
    bool palette_ok = (col.size() == static_cast<std::size_t>(g.CW) * g.CH * 3);
    if (palette_ok && g.palette.size() == 150 * 3) {
        const std::size_t hw = static_cast<std::size_t>(g.CW) * g.CH;
        for (std::size_t p = 0; p < hw && palette_ok; ++p) {
            const uint8_t c = sm.classes[p];
            if (c < 150) {
                if (col[p*3+0] != g.palette[c*3+0] ||
                    col[p*3+1] != g.palette[c*3+1] ||
                    col[p*3+2] != g.palette[c*3+2]) palette_ok = false;
            }
        }
    }
    CHECK(palette_ok);

    // ── CUDA ──
    brotensor::init();
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        det.to(brotensor::Device::CUDA);
        CHECK(det.device() == brotensor::Device::CUDA);

        brotensor::Tensor pxd = px.to(brotensor::Device::CUDA);
        Logits glg = det.infer_logits_from_tensor(pxd);
        double gmx = 0, gmn = 0, cmx = 0, cmn = 0;
        if (glg.data.size() == g.logits.size()) {
            diff(glg.data, g.logits, gmx, gmn);
            diff(glg.data, lg.data, cmx, cmn);
            std::printf("    CUDA Gate1 vs golden: mean-abs=%.3e  vs CPU: max-abs=%.3e mean-abs=%.3e\n",
                        gmn, cmx, cmn);
            CHECK(gmn < 5e-3);     // CUDA tracks the golden like the CPU path
            CHECK(cmx < 5e-2);     // CPU vs CUDA worst-logit tripwire
        } else {
            CHECK(false);
        }

        SegMap gsm = det.detect(g.resized.data(), g.W, g.H, 3);
        double gagree = agreement(gsm.classes, g.classmap);
        double cagree = agreement(gsm.classes, sm.classes);
        std::printf("    CUDA Gate2 vs golden: %.4f  vs CPU: %.4f\n", gagree, cagree);
        CHECK(gagree >= 0.98);
        CHECK(cagree >= 0.999);    // CPU/CUDA class maps near-identical
    } else {
        std::printf("    (no CUDA device available — on-device check skipped)\n");
    }
}

}  // namespace

int main() {
    std::printf("test_segformer:\n");

    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
    const std::string dir = base + "/segformer-b0-ade";
    const std::string ckpt = dir + "/model.safetensors";
    const std::string golden = base + "/segformer-b0-ade/golden_segformer.bin";

    if (!file_exists(ckpt) || !file_exists(golden)) {
        std::printf("  no checkpoint/golden under '%s' — skipping (weights-gated).\n",
                    dir.c_str());
        return 0;
    }

    try {
        run_case(dir, golden);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  error: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) {
        std::printf("  OK  segformer parity checks passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
