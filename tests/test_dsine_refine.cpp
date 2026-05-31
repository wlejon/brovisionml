// DSINE v02 iterative refinement (NRN): real-weights END-TO-END golden parity.
//
// Weights-gated (skips cleanly when weights/dsine/model.safetensors or the
// golden_dsine_*.bin dumps are absent). For each golden, reads the stored input
// bytes, runs dsine::preprocess -> encoder -> decoder -> refiner, and compares
// the final full-resolution surface normals against the golden `final` record
// (3, H, W) at the ORIGINAL (unpadded) resolution — the LAST record in the dump.
//
// The refinement rides on top of the encoder+decoder stack and runs 5 iterations
// of rotation/normalize, so the bound is looser than the decoder's: max-abs
// ~1e-2, mean-abs ~1e-3. A wildly larger diff points at a real bug
// (axis_angle_to_matrix, get_unfold ordering, the convex-upsample softmax/
// reshape, RayReLU, or the intrinsics scaling) rather than accumulation.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/dsine_decoder.h"
#include "brovisionml/dsine_encoder.h"
#include "brovisionml/dsine_preprocess.h"
#include "brovisionml/dsine_refine.h"

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

using brovisionml::dsine::Decoder;
using brovisionml::dsine::DecoderOutput;
using brovisionml::dsine::EncoderB5;
using brovisionml::dsine::EncoderTaps;
using brovisionml::dsine::Intrinsics;
using brovisionml::dsine::preprocess;
using brovisionml::dsine::PreprocessedImage;
using brovisionml::dsine::Refiner;

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

// Reads the full BVMLDSN1 golden, seeking past the encoder taps and dec_normal
// to reach the final full-res normal map.
struct Golden {
    int W = 0, H = 0, padW = 0, padH = 0, l = 0, r = 0, t = 0, b = 0;
    float intrins4[4] = {0, 0, 0, 0};   // fx, fy, cx, cy (pre-"+0.5")
    std::vector<uint8_t> input;          // H*W*3 HWC RGB
    std::vector<float> final;            // 3*H*W
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
    for (int i = 0; i < n_taps; ++i) {     // seek past the encoder taps
        Tap tp;
        if (!read_one(f, tp.C) || !read_one(f, tp.H) || !read_one(f, tp.W))
            return false;
        const std::size_t n = static_cast<std::size_t>(tp.C) * tp.H * tp.W;
        if (!read_vec(f, tp.data, n)) return false;
    }

    // seek past dec_normal.
    int dn_C = 0, dn_H = 0, dn_W = 0;
    if (!read_one(f, dn_C) || !read_one(f, dn_H) || !read_one(f, dn_W))
        return false;
    std::vector<float> dec_normal;
    const std::size_t dn_n =
        static_cast<std::size_t>(dn_C) * dn_H * dn_W;
    if (!read_vec(f, dec_normal, dn_n)) return false;

    // final[3*H*W].
    const std::size_t fin_n = static_cast<std::size_t>(3) * g.H * g.W;
    if (!read_vec(f, g.final, fin_n)) return false;
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

    // Use the golden's stored intrinsics (pre-"+0.5"); build_uv adds the +0.5.
    Intrinsics in;
    in.fx = g.intrins4[0];
    in.fy = g.intrins4[1];
    in.cx = g.intrins4[2];
    in.cy = g.intrins4[3];

    Decoder dec;
    dec.load(dir);
    DecoderOutput dout =
        dec.forward(t, in, pp.transform.pad_h, pp.transform.pad_w);

    Refiner ref;
    ref.load(dir);
    brotensor::Tensor final = ref.forward(dout, in, pp.transform.pad_h,
                                          pp.transform.pad_w, pp.transform);

    std::printf("    final: got (3,%d,%d)  golden (3,%d,%d)\n",
                g.H, g.W, g.H, g.W);
    const std::size_t n = static_cast<std::size_t>(3) * g.H * g.W;
    if (static_cast<std::size_t>(final.cols) != n) {
        std::printf("    FAIL  final element count %d != golden %zu\n",
                    final.cols, n);
        ++g_failures;
        return;
    }
    double mx = 0.0, mn = 0.0;
    diff_stats(final.host_f32(), g.final.data(), n, mx, mn);
    std::printf("    final: max-abs=%.3e  mean-abs=%.3e\n", mx, mn);
    CHECK(mx < 1e-2);
    CHECK(mn < 1e-3);
}

}  // namespace

int main() {
    std::printf("test_dsine_refine:\n");

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

    run_case(dir, sq);    // square: no padding — isolates the refinement math
    run_case(dir, pad);

    if (g_failures == 0) {
        std::printf("  OK  all dsine_refine parity checks passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
