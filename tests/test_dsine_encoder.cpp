// DSINE EfficientNet-B5 encoder: real-weights golden parity test.
//
// Weights-gated (skips cleanly when weights/dsine/model.safetensors or the
// golden_dsine_*.bin dumps are absent). For each golden, reads the stored input
// bytes, runs dsine::preprocess, runs the encoder, and compares the three
// encoder taps (stage2 64ch /8, stage4 176ch /16, conv_head 2048ch /32) against
// the golden's taps. The golden tap order is [s8, s16, s32]; we match by shape.
//
// A deep conv+BN stack is not bit-exact vs PyTorch, so the bound is ~2e-3
// max-abs. A wildly larger diff points at a real bug (TF-same padding, BN eps,
// depthwise layout, SE, or a residual rule) rather than accumulation.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/dsine_encoder.h"
#include "brovisionml/dsine_preprocess.h"

#include "brotensor/tensor.h"

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

using brovisionml::dsine::EncoderB5;
using brovisionml::dsine::EncoderTaps;
using brovisionml::dsine::preprocess;
using brovisionml::dsine::PreprocessedImage;

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

struct Tap {
    int C = 0, H = 0, W = 0;
    std::vector<float> data;
};

// Reads the BVMLDSN1 golden up to and including the three encoder taps; the
// decoder/normal tail is ignored.
struct Golden {
    int W = 0, H = 0, padW = 0, padH = 0, l = 0, r = 0, t = 0, b = 0;
    float intrins4[4] = {0, 0, 0, 0};
    std::vector<uint8_t> input;     // H*W*3 HWC RGB
    std::vector<Tap>     taps;      // 3 encoder taps
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 8) != "BVMLDSN1") return false;
    int version = 0;
    if (!read_one(f, version)) return false;
    if (!read_one(f, g.W) || !read_one(f, g.H)) return false;
    if (!read_one(f, g.padW) || !read_one(f, g.padH)) return false;
    if (!read_one(f, g.l) || !read_one(f, g.r) ||
        !read_one(f, g.t) || !read_one(f, g.b)) return false;
    f.read(reinterpret_cast<char*>(g.intrins4), 4 * sizeof(float));
    if (!f) return false;

    const std::size_t in_n = static_cast<std::size_t>(g.H) * g.W * 3;
    const std::size_t px_n = static_cast<std::size_t>(g.padH) * g.padW * 3;
    if (!read_vec(f, g.input, in_n)) return false;
    std::vector<float> pixels;     // skip the preprocessed pixels
    if (!read_vec(f, pixels, px_n)) return false;

    int n_taps = 0;
    if (!read_one(f, n_taps)) return false;
    if (n_taps != 3) return false;
    g.taps.resize(static_cast<std::size_t>(n_taps));
    for (int i = 0; i < n_taps; ++i) {
        Tap& tp = g.taps[static_cast<std::size_t>(i)];
        if (!read_one(f, tp.C) || !read_one(f, tp.H) || !read_one(f, tp.W))
            return false;
        const std::size_t n =
            static_cast<std::size_t>(tp.C) * tp.H * tp.W;
        if (!read_vec(f, tp.data, n)) return false;
    }
    return true;
}

void diff_stats(const float* a, const float* b, std::size_t n,
                double& max_abs, double& mean_abs) {
    double m = 0.0, s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d =
            std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        m = std::max(m, d);
        s += d;
    }
    max_abs = m;
    mean_abs = n ? s / static_cast<double>(n) : 0.0;
}

void compare_tap(const char* name, const brotensor::Tensor& got,
                 int gC, int gH, int gW, const Tap& want) {
    std::printf("    %s: got (%d,%d,%d)  golden (%d,%d,%d)\n",
                name, gC, gH, gW, want.C, want.H, want.W);
    CHECK(gC == want.C && gH == want.H && gW == want.W);
    const std::size_t n = static_cast<std::size_t>(want.C) * want.H * want.W;
    if (static_cast<std::size_t>(got.cols) != n) {
        std::printf("    FAIL  %s element count %d != golden %zu\n",
                    name, got.cols, n);
        ++g_failures;
        return;
    }
    double mx = 0.0, mn = 0.0;
    diff_stats(got.host_f32(), want.data.data(), n, mx, mn);
    std::printf("    %s: max-abs=%.3e  mean-abs=%.3e\n", name, mx, mn);
    CHECK(mx < 2e-3);
}

void run_case(const std::string& dir, const std::string& path) {
    Golden g;
    if (!load_golden(path, g)) {
        std::printf("  FAIL  could not load/parse golden %s\n", path.c_str());
        ++g_failures;
        return;
    }
    std::printf("  case %s  (%dx%d -> pad %dx%d)\n",
                path.c_str(), g.W, g.H, g.padW, g.padH);

    PreprocessedImage pp = preprocess(g.input.data(), g.W, g.H, 3);
    CHECK(pp.transform.pad_w == g.padW && pp.transform.pad_h == g.padH);

    EncoderB5 enc;
    enc.load(dir);
    EncoderTaps t = enc.forward(pp.pixels, pp.transform.pad_h, pp.transform.pad_w);

    compare_tap("s8 ", t.s8,  64,   t.h8,  t.w8,  g.taps[0]);
    compare_tap("s16", t.s16, 176,  t.h16, t.w16, g.taps[1]);
    compare_tap("s32", t.s32, 2048, t.h32, t.w32, g.taps[2]);
}

}  // namespace

int main() {
    std::printf("test_dsine_encoder:\n");

    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
    const std::string dir = base + "/dsine";
    const std::string ckpt = dir + "/model.safetensors";
    const std::string sq = dir + "/golden_dsine_sq256.bin";
    const std::string pad = dir + "/golden_dsine_pad500x348.bin";

    if (!file_exists(ckpt) || !file_exists(sq) || !file_exists(pad)) {
        std::printf("  no checkpoint/goldens under '%s' — skipping "
                    "(weights-gated).\n", dir.c_str());
        return 0;
    }

    run_case(dir, sq);
    run_case(dir, pad);

    if (g_failures == 0) {
        std::printf("  OK  all dsine_encoder parity checks passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
