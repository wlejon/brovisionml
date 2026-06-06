// CPU test for the DINOv3 ViT-H image preprocessor.
//
// Coverage:
//   1. Square output shape (1, 3*res*res) and transform fields at the default
//      and at a lowered resolution (the speed knob).
//   2. ImageNet normalization of a flat mid-gray image (constant per channel).
//   3. The resolution knob is honoured and validated: a non-multiple of
//      patch_size and a non-positive resolution both throw.
//   4. Channel handling (1 / 3 / 4) and that no NaNs/Infs appear.

#include "brovisionml/dinov3_preprocess.h"

#include "broimage/presets.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using brovisionml::dinov3::preprocess;
using brovisionml::dinov3::PreprocessedImage;
using brovisionml::dinov3::kDefaultResolution;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static bool approx(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(a) + std::fabs(b));
}

// ── 1. square shape + transform, default and lowered resolution ──────────────
static void test_shape_and_knob() {
    const int W = 256, H = 192;
    std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3, 128);

    PreprocessedImage def = preprocess(img.data(), W, H, 3);
    CHECK(def.transform.resolution == kDefaultResolution);
    CHECK(def.pixels.rows == 1 &&
          def.pixels.cols == 3 * kDefaultResolution * kDefaultResolution);
    CHECK(def.transform.orig_w == W && def.transform.orig_h == H);
    CHECK(approx(def.transform.scale_x, float(kDefaultResolution) / W));
    CHECK(approx(def.transform.scale_y, float(kDefaultResolution) / H));

    // Lowered resolution (the knob): fewer tokens, same square contract.
    PreprocessedImage lo = preprocess(img.data(), W, H, 3, /*resolution=*/512);
    CHECK(lo.transform.resolution == 512);
    CHECK(lo.pixels.rows == 1 && lo.pixels.cols == 3 * 512 * 512);
}

// ── 2. ImageNet normalization of mid-gray ────────────────────────────────────
static void test_gray_normalization() {
    const int W = 200, H = 200, R = 256;
    std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3, 128);  // mid gray
    PreprocessedImage pp = preprocess(img.data(), W, H, 3, R);

    // A constant image stays constant after resize: each channel plane holds the
    // single normalized value (128/255 - mean)/std.
    const float* p = pp.pixels.host_f32();
    const int plane = R * R;
    for (int c = 0; c < 3; ++c) {
        const float expected =
            (128.0f / 255.0f - broimage::IMAGENET_MEAN[c]) / broimage::IMAGENET_STD[c];
        for (int s : {0, plane / 2, plane - 1})
            CHECK(approx(p[static_cast<size_t>(c) * plane + s], expected));
    }
}

// ── 3. resolution validation ─────────────────────────────────────────────────
static void test_resolution_validation() {
    const int W = 64, H = 64;
    std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3, 100);

    bool threw = false;
    try { preprocess(img.data(), W, H, 3, /*resolution=*/1000); }  // 1000 % 16 != 0
    catch (const std::exception&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { preprocess(img.data(), W, H, 3, /*resolution=*/0); }
    catch (const std::exception&) { threw = true; }
    CHECK(threw);

    // A custom patch_size that divides the resolution is accepted.
    threw = false;
    try { preprocess(img.data(), W, H, 3, /*resolution=*/224, /*patch_size=*/14); }
    catch (const std::exception&) { threw = true; }
    CHECK(!threw);
}

// ── 4. channel handling + finiteness ─────────────────────────────────────────
static void test_channels_and_finite() {
    const int W = 48, H = 64, R = 128;
    for (int ch : {1, 3, 4}) {
        std::vector<uint8_t> img(static_cast<size_t>(W) * H * ch);
        for (size_t i = 0; i < img.size(); ++i)
            img[i] = static_cast<uint8_t>((i * 37) & 0xFF);
        PreprocessedImage pp = preprocess(img.data(), W, H, ch, R);
        CHECK(pp.pixels.rows == 1 && pp.pixels.cols == 3 * R * R);
        const float* p = pp.pixels.host_f32();
        bool finite = true;
        for (int i = 0; i < pp.pixels.cols; ++i)
            if (!std::isfinite(p[i])) { finite = false; break; }
        CHECK(finite);
    }
}

int main() {
    std::printf("test_dinov3_preprocess (CPU):\n");
    test_shape_and_knob();
    test_gray_normalization();
    test_resolution_validation();
    test_channels_and_finite();
    if (g_failures == 0) {
        std::printf("  OK  all dinov3_preprocess tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
