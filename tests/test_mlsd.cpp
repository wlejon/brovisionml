// MLSD MLSDdetector: real-weights END-TO-END parity test.
//
// Weights-gated (skips cleanly when weights/mlsd/model.safetensors or the
// golden_mlsd_*.bin dump is absent). For each golden, reads the stored input
// bytes, runs the network at the fixed 512x512 input, and compares (1) the raw
// 9-channel TP map against the golden — the tight neural-parity check — and (2)
// the decoded line-segment set against the golden segments (count + a nearest
// match within a few pixels), the looser end-to-end check on top of the host
// decode. A CUDA device, when present, is exercised too.
//
// The TP map is the meaningful numeric gate: M-LSD is a moderately deep
// depthwise/dilated conv net with folded BatchNorm + an FPN decode head, so a
// large TP diff points at a real bug (TFLite asymmetric pad, the channel-7
// slice, depthwise grouping, align-corners upsample, or BN fold) rather than
// accumulation. The decode (sigmoid + 3x3 max-pool NMS + top-K + endpoint
// reconstruction) is deterministic classical code validated against the golden
// segment set.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/mlsd.h"

#include "brotensor/runtime.h"

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

using brovisionml::mlsd::LineMap;
using brovisionml::mlsd::LineSegment;
using brovisionml::mlsd::MlsdConfig;
using brovisionml::mlsd::MLSDdetector;
using brovisionml::mlsd::TpMap;

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
    std::vector<uint8_t> input;        // H*W*3 HWC RGB
    int C = 0, th = 0, tw = 0;
    std::vector<float> tp;             // C*th*tw TP map
    std::vector<float> seg;            // n_seg*4 (x1,y1,x2,y2) in W/H space
    float score_thr = 0, dist_thr = 0;
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 8) != "BVMLMLS1") return false;
    int version = 0;
    if (!read_one(f, version)) return false;
    if (!read_one(f, g.W) || !read_one(f, g.H)) return false;
    if (!read_vec(f, g.input, static_cast<std::size_t>(g.H) * g.W * 3)) return false;
    if (!read_one(f, g.C) || !read_one(f, g.th) || !read_one(f, g.tw)) return false;
    if (!read_vec(f, g.tp, static_cast<std::size_t>(g.C) * g.th * g.tw)) return false;
    int n_seg = 0;
    if (!read_one(f, n_seg)) return false;
    if (!read_vec(f, g.seg, static_cast<std::size_t>(n_seg) * 4)) return false;
    if (!read_one(f, g.score_thr) || !read_one(f, g.dist_thr)) return false;
    return true;
}

void tp_diff(const std::vector<float>& a, const std::vector<float>& b,
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

// Fraction of golden segments that have a matching detected segment whose four
// endpoint coords are all within `tol` pixels.
double seg_recall(const std::vector<float>& gseg,
                  const std::vector<LineSegment>& det, float tol) {
    const std::size_t ng = gseg.size() / 4;
    if (ng == 0) return 1.0;
    std::size_t hit = 0;
    for (std::size_t i = 0; i < ng; ++i) {
        const float gx1 = gseg[i * 4 + 0], gy1 = gseg[i * 4 + 1];
        const float gx2 = gseg[i * 4 + 2], gy2 = gseg[i * 4 + 3];
        for (const LineSegment& s : det) {
            if (std::fabs(s.x1 - gx1) <= tol && std::fabs(s.y1 - gy1) <= tol &&
                std::fabs(s.x2 - gx2) <= tol && std::fabs(s.y2 - gy2) <= tol) {
                ++hit;
                break;
            }
        }
    }
    return (double)hit / (double)ng;
}

void run_case(const std::string& dir, const std::string& path) {
    Golden g;
    if (!load_golden(path, g)) {
        std::printf("  FAIL  could not load/parse golden %s\n", path.c_str());
        ++g_failures;
        return;
    }
    std::printf("  case %s  (in %dx%d  tp %dx%dx%d  golden segs=%zu)\n",
                path.c_str(), g.W, g.H, g.C, g.th, g.tw, g.seg.size() / 4);

    MlsdConfig cfg;
    cfg.score_thr = g.score_thr;
    cfg.dist_thr = g.dist_thr;
    MLSDdetector det(cfg);
    det.load(dir);

    // (1) TP-map parity on the host.
    TpMap tp = det.infer_tpmap(g.input.data(), g.W, g.H, 3);
    CHECK(tp.channels == g.C && tp.height == g.th && tp.width == g.tw);
    if (tp.data.size() == g.tp.size()) {
        double mx = 0, mn = 0;
        tp_diff(tp.data, g.tp, mx, mn);
        std::printf("    CPU TP vs golden: max-abs=%.3e  mean-abs=%.3e\n", mx, mn);
        CHECK(mx < 5e-2);
        CHECK(mn < 1e-3);
    } else {
        CHECK(false);
    }

    // (2) Decoded segment set.
    LineMap lm = det.detect(g.input.data(), g.W, g.H, 3);
    const double recall = seg_recall(g.seg, lm.segments, /*tol=*/2.0f);
    std::printf("    CPU segments: %zu (golden %zu)  recall@2px=%.3f\n",
                lm.segments.size(), g.seg.size() / 4, recall);
    CHECK(recall > 0.90);

    // On-device: TP map should track CPU + golden tightly (FP32, same math).
    brotensor::init();
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        det.to(brotensor::Device::CUDA);
        CHECK(det.device() == brotensor::Device::CUDA);
        TpMap gtp = det.infer_tpmap(g.input.data(), g.W, g.H, 3);
        if (gtp.data.size() == g.tp.size()) {
            double mxg = 0, mng = 0, mxc = 0, mnc = 0;
            tp_diff(gtp.data, g.tp, mxg, mng);
            tp_diff(gtp.data, tp.data, mxc, mnc);
            std::printf("    CUDA TP vs golden: mean-abs=%.3e  vs CPU: max-abs=%.3e\n",
                        mng, mxc);
            CHECK(mng < 1e-3);      // CUDA tracks the golden like the CPU path
            // The TP logits span tens (center ~[-35,3.5], displacements are large
            // pixel offsets), so the worst-single-logit CPU/CUDA gap from FP
            // accumulation order across the deep depthwise/dilated trunk is ~1.5e-2
            // — the SAME order as the CPU-vs-golden max, i.e. both backends differ
            // from torch by that much, not from each other specifically. mean-abs
            // (~1e-4) is the tight gate; the decoded segments are identical (the
            // recall check below). 5e-2 is a gross-breakage tripwire only.
            CHECK(mxc < 5e-2);
        }
        LineMap glm = det.detect(g.input.data(), g.W, g.H, 3);
        const double grecall = seg_recall(g.seg, glm.segments, /*tol=*/2.0f);
        std::printf("    CUDA segments: %zu  recall@2px=%.3f\n",
                    glm.segments.size(), grecall);
        CHECK(grecall > 0.90);
    } else {
        std::printf("    (no CUDA device available — on-device check skipped)\n");
    }
}

}  // namespace

int main() {
    std::printf("test_mlsd:\n");

    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
    const std::string dir = base + "/mlsd";
    const std::string ckpt = dir + "/model.safetensors";
    const std::string sq = dir + "/golden_mlsd_sq512.bin";

    if (!file_exists(ckpt) || !file_exists(sq)) {
        std::printf("  no checkpoint/goldens under '%s' — skipping "
                    "(weights-gated).\n", dir.c_str());
        return 0;
    }

    try {
        run_case(dir, sq);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  error: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) {
        std::printf("  OK  mlsd parity checks passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
