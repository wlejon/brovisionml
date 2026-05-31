// CPU test for the DSINE surface-normal image preprocessor.
//
// Two parts:
//   1. Unconditional CPU unit tests (always run): get_padding for a multiple-of-
//      32 dim (no pad) and a non-multiple (exact split); intrins_from_fov formula
//      for a known size; preprocess pad dims / tensor element count; channel
//      handling (gray/RGBA) without crashing or producing NaNs; a known padded
//      pixel equals (0 - mean)/std.
//   2. Weights-gated golden parity (skips cleanly when absent): reads the
//      golden_dsine_*.bin dumps of the reference preprocess, runs preprocess on
//      the stored input bytes, and asserts the pixel tensor (~1e-5 max-abs) and
//      the intrinsics (~1e-4) match.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/dsine_preprocess.h"

#include "broimage/presets.h"

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

using brovisionml::dsine::get_padding;
using brovisionml::dsine::intrins_from_fov;
using brovisionml::dsine::preprocess;
using brovisionml::dsine::Intrinsics;
using brovisionml::dsine::PreprocessedImage;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static bool approx(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

// ── 1. get_padding ───────────────────────────────────────────────────────────
static void test_get_padding() {
    int l = -1, r = -1, t = -1, b = -1;

    // Multiple of 32 -> no padding.
    get_padding(256, 256, l, r, t, b);
    CHECK(l == 0 && r == 0 && t == 0 && b == 0);

    // Non-multiple W=500 -> new_W=512, gap 12 -> l=6, r=6.
    // Non-multiple H=348 -> new_H=352, gap 4  -> t=2, b=2.
    get_padding(348, 500, l, r, t, b);
    CHECK(l == 6 && r == 6);
    CHECK(t == 2 && b == 2);

    // Odd gap splits with the smaller half on the top/left.
    // W=481 -> new_W=512, gap 31 -> l=15, r=16.
    get_padding(481, 481, l, r, t, b);
    CHECK(l == 15 && r == 16);
    CHECK(t == 15 && b == 16);
}

// ── 2. intrins_from_fov ──────────────────────────────────────────────────────
static void test_intrins_from_fov() {
    // 60° FOV on a 256x256 image: f = (256/2) / tan(30°) = 128 / 0.57735... .
    Intrinsics k = intrins_from_fov(256, 256, 60.0f);
    const double half = (60.0 / 2.0) * 3.14159265358979323846 / 180.0;
    const double f = (256.0 / 2.0) / std::tan(half);
    CHECK(approx(k.fx, f, 1e-3));
    CHECK(approx(k.fy, f, 1e-3));
    CHECK(approx(k.cx, 256.0 / 2.0 - 0.5, 1e-5));
    CHECK(approx(k.cy, 256.0 / 2.0 - 0.5, 1e-5));

    // Non-square: focal uses max(H,W); centers are per-axis.
    Intrinsics k2 = intrins_from_fov(348, 500, 60.0f);
    const double f2 = (500.0 / 2.0) / std::tan(half);
    CHECK(approx(k2.fx, f2, 1e-3));
    CHECK(approx(k2.cx, 500.0 / 2.0 - 0.5, 1e-5));
    CHECK(approx(k2.cy, 348.0 / 2.0 - 0.5, 1e-5));
}

// ── 3. preprocess shape + pad dims + padded-pixel value ──────────────────────
static void test_preprocess_shape_and_pad() {
    const int W = 500, H = 348;
    std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3, 200);  // content
    PreprocessedImage pp = preprocess(img.data(), W, H, 3);

    CHECK(pp.transform.orig_w == W && pp.transform.orig_h == H);
    CHECK(pp.transform.pad_w == 512 && pp.transform.pad_h == 352);
    CHECK(pp.transform.pad_l == 6 && pp.transform.pad_r == 6);
    CHECK(pp.transform.pad_t == 2 && pp.transform.pad_b == 2);
    CHECK(pp.pixels.rows == 1 && pp.pixels.cols == 3 * 512 * 352);

    // Intrinsics carry the pad offset (cx += l, cy += t).
    CHECK(approx(pp.intrins.cx, 500.0 / 2.0 - 0.5 + 6.0, 1e-4));
    CHECK(approx(pp.intrins.cy, 348.0 / 2.0 - 0.5 + 2.0, 1e-4));

    // A top-left corner pixel is pure zero pad -> (0 - mean)/std per channel.
    const float* p = pp.pixels.host_f32();
    const int plane = 512 * 352;
    for (int c = 0; c < 3; ++c) {
        const float expected =
            (0.0f - broimage::IMAGENET_MEAN[c]) / broimage::IMAGENET_STD[c];
        // (row 0, col 0) of channel c.
        CHECK(approx(p[static_cast<size_t>(c) * plane + 0], expected, 1e-5));
    }

    // An interior content pixel is the normalized 200/255 value.
    const int cy = H / 2 + 2;  // + pad_t
    const int cx = W / 2 + 6;  // + pad_l
    const int idx = cy * 512 + cx;
    for (int c = 0; c < 3; ++c) {
        const float expected =
            (200.0f / 255.0f - broimage::IMAGENET_MEAN[c]) / broimage::IMAGENET_STD[c];
        CHECK(approx(p[static_cast<size_t>(c) * plane + idx], expected, 1e-4));
    }
}

// ── 4. channel handling + finiteness ─────────────────────────────────────────
static void test_channels_and_finite() {
    const int W = 100, H = 70;
    for (int ch : {1, 3, 4}) {
        std::vector<uint8_t> img(static_cast<size_t>(W) * H * ch);
        for (size_t i = 0; i < img.size(); ++i)
            img[i] = static_cast<uint8_t>((i * 37) & 0xFF);
        PreprocessedImage pp = preprocess(img.data(), W, H, ch);
        CHECK(pp.pixels.rows == 1);
        const float* p = pp.pixels.host_f32();
        bool finite = true;
        for (int i = 0; i < pp.pixels.cols; ++i)
            if (!std::isfinite(p[i])) { finite = false; break; }
        CHECK(finite);
    }
}

// ── 5. golden parity (weights-gated) ─────────────────────────────────────────
namespace {

template <class T>
bool read_vec(std::ifstream& f, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(f);
}

template <class T>
bool read_one(std::ifstream& f, T& x) {
    f.read(reinterpret_cast<char*>(&x), sizeof(T));
    return static_cast<bool>(f);
}

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

// Reads the BVMLDSN1 golden through `pixels` (the rest of the file is ignored
// for this preprocessing-only commit).
struct Golden {
    int W = 0, H = 0, padW = 0, padH = 0, l = 0, r = 0, t = 0, b = 0;
    float intrins4[4] = {0, 0, 0, 0};   // fx, fy, cx, cy
    std::vector<uint8_t> input;          // H*W*3 HWC RGB
    std::vector<float>   pixels;         // 3*padH*padW NCHW
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
    if (!read_vec(f, g.pixels, px_n)) return false;
    return true;
}

double max_abs(const float* a, const float* b, std::size_t n) {
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return m;
}

void run_golden_case(const std::string& path) {
    Golden g;
    if (!load_golden(path, g)) {
        std::printf("  FAIL  could not load/parse golden %s\n", path.c_str());
        ++g_failures;
        return;
    }
    PreprocessedImage pp = preprocess(g.input.data(), g.W, g.H, 3);

    CHECK(pp.transform.pad_w == g.padW && pp.transform.pad_h == g.padH);
    CHECK(pp.transform.pad_l == g.l && pp.transform.pad_r == g.r);
    CHECK(pp.transform.pad_t == g.t && pp.transform.pad_b == g.b);
    CHECK(static_cast<std::size_t>(pp.pixels.cols) == g.pixels.size());

    if (static_cast<std::size_t>(pp.pixels.cols) == g.pixels.size()) {
        double dpx = max_abs(pp.pixels.host_f32(), g.pixels.data(), g.pixels.size());
        std::printf("  %s: pixels max-abs=%.3e\n", path.c_str(), dpx);
        CHECK(dpx < 1e-5);
    }

    const float mine[4] = {pp.intrins.fx, pp.intrins.fy, pp.intrins.cx, pp.intrins.cy};
    double dk = max_abs(mine, g.intrins4, 4);
    std::printf("  %s: intrins max-abs=%.3e (mine fx=%.4f fy=%.4f cx=%.4f cy=%.4f | "
                "golden fx=%.4f fy=%.4f cx=%.4f cy=%.4f)\n",
                path.c_str(), dk, mine[0], mine[1], mine[2], mine[3],
                g.intrins4[0], g.intrins4[1], g.intrins4[2], g.intrins4[3]);
    CHECK(dk < 1e-4);
}

}  // namespace

static void test_golden_parity() {
    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
    const std::string dir = base + "/dsine";
    const std::string sq = dir + "/golden_dsine_sq256.bin";
    const std::string pad = dir + "/golden_dsine_pad500x348.bin";

    if (!file_exists(sq) || !file_exists(pad)) {
        std::printf("  golden parity: no golden_dsine_*.bin under '%s' — skipping "
                    "(generate with the out-of-repo dump script).\n", dir.c_str());
        return;
    }
    run_golden_case(sq);
    run_golden_case(pad);
}

int main() {
    std::printf("test_dsine_preprocess (CPU):\n");
    test_get_padding();
    test_intrins_from_fov();
    test_preprocess_shape_and_pad();
    test_channels_and_finite();
    test_golden_parity();
    if (g_failures == 0) {
        std::printf("  OK  all dsine_preprocess tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
