// CPU test for the DPT / Depth-Anything image preprocessor.
//
// Coverage:
//   1. resize_output_size: square inputs map to the target; output dims are
//      multiples of 14; keep_aspect_ratio vs. independent-axis scaling.
//   2. Aspect ratio preserved on a non-square image (known hand-computed dims).
//   3. preprocess output shape (1, 3*resized_h*resized_w) and ImageNet
//      normalization of a flat mid-gray image.
//   4. Channel handling (1 / 3 / 4) and that no NaNs/Infs appear.

#include "brovisionml/dpt_preprocess.h"

#include "broimage/presets.h"

#include <cmath>
#include <cstdio>
#include <vector>

using brovisionml::dpt::preprocess;
using brovisionml::dpt::resize_output_size;
using brovisionml::dpt::PreprocessedImage;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static bool approx(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(a) + std::fabs(b));
}

// ── 1. resize_output_size ───────────────────────────────────────────────────
static void test_resize_output_size() {
    int ow = 0, oh = 0;

    // Square -> target square (37*14 == 518), no interpolation later.
    resize_output_size(518, 518, 518, 14, /*keep_aspect=*/true, ow, oh);
    CHECK(ow == 518 && oh == 518);

    resize_output_size(100, 100, 518, 14, true, ow, oh);
    CHECK(ow == 518 && oh == 518);

    // Non-square, keep aspect: shorter side -> 518, longer overshoots, both
    // multiples of 14. 640x480 -> 686x518 (49*14 x 37*14).
    resize_output_size(640, 480, 518, 14, true, ow, oh);
    CHECK(ow == 686 && oh == 518);
    CHECK(ow % 14 == 0 && oh % 14 == 0);

    // Same image without keep_aspect: each axis hits the target independently.
    resize_output_size(640, 480, 518, 14, /*keep_aspect=*/false, ow, oh);
    CHECK(ow == 518 && oh == 518);

    // Tiny image floors at one `multiple`.
    resize_output_size(3, 3, 518, 14, true, ow, oh);
    CHECK(ow >= 14 && oh >= 14 && ow % 14 == 0 && oh % 14 == 0);
}

// ── 2. aspect ratio preserved (portrait) ────────────────────────────────────
static void test_aspect_portrait() {
    int ow = 0, oh = 0;
    resize_output_size(480, 640, 518, 14, true, ow, oh);
    // Portrait: width is the shorter side -> 518; height overshoots.
    CHECK(ow == 518 && oh == 686);
    CHECK(ow % 14 == 0 && oh % 14 == 0);
}

// ── 3. preprocess shape + ImageNet normalization of mid-gray ────────────────
static void test_preprocess_gray_normalization() {
    const int W = 256, H = 192;
    std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3, 128);  // mid gray
    PreprocessedImage pp = preprocess(img.data(), W, H, 3);

    int ew = 0, eh = 0;
    resize_output_size(W, H, 518, 14, true, ew, eh);
    CHECK(pp.transform.resized_w == ew && pp.transform.resized_h == eh);
    CHECK(pp.transform.orig_w == W && pp.transform.orig_h == H);
    CHECK(pp.pixels.rows == 1 && pp.pixels.cols == 3 * ew * eh);

    // A constant 128 image stays constant after resize, so each channel plane
    // should hold a single normalized value (128/255 - mean)/std.
    const float* p = pp.pixels.host_f32();
    const int plane = ew * eh;
    for (int c = 0; c < 3; ++c) {
        const float expected =
            (128.0f / 255.0f - broimage::IMAGENET_MEAN[c]) / broimage::IMAGENET_STD[c];
        // Sample a few interior pixels (borders can differ slightly under the
        // bicubic edge clamp, but a constant image is exact everywhere).
        for (int s : {0, plane / 2, plane - 1}) {
            CHECK(approx(p[static_cast<size_t>(c) * plane + s], expected, 1e-3f));
        }
    }
}

// ── 4. channel handling + finiteness ────────────────────────────────────────
static void test_channels_and_finite() {
    const int W = 64, H = 48;
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

int main() {
    std::printf("test_dpt_preprocess (CPU):\n");
    test_resize_output_size();
    test_aspect_portrait();
    test_preprocess_gray_normalization();
    test_channels_and_finite();
    if (g_failures == 0) {
        std::printf("  OK  all dpt_preprocess tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
