// SAM preprocessing: longest-side resize, per-channel normalize with the SAM
// presets, and zero-padding to a square — no checkpoint required.
#include "brovisionml/sam_preprocess.h"
#include "broimage/presets.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

int failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

void check_near(float a, float b, float tol, const char* msg) {
    if (std::fabs(a - b) > tol) {
        std::fprintf(stderr, "FAIL: %s (got %f, want %f)\n", msg, a, b);
        ++failures;
    }
}

float expected_norm(uint8_t v, int c) {
    return (static_cast<float>(v) / 255.0f - broimage::SAM_MEAN[c]) /
           broimage::SAM_STD[c];
}

}  // namespace

int main() {
    using namespace brovisionml::sam;

    // ── get_preprocess_shape: longest side maps to input_size, aspect kept ──
    {
        int rw = 0, rh = 0;
        preprocess_shape(100, 50, 1024, rw, rh);
        check(rw == 1024 && rh == 512, "wide 100x50 -> 1024x512");
        preprocess_shape(50, 100, 1024, rw, rh);
        check(rw == 512 && rh == 1024, "tall 50x100 -> 512x1024");
        preprocess_shape(64, 64, 1024, rw, rh);
        check(rw == 1024 && rh == 1024, "square 64x64 -> 1024x1024");
    }

    // ── Content normalize + zero pad on a small square (constant color) ──
    {
        const int w = 4, h = 2, S = 8;
        const uint8_t R = 200, G = 30, B = 100;
        std::vector<uint8_t> img(static_cast<size_t>(w) * h * 3);
        for (int i = 0; i < w * h; ++i) {
            img[i * 3 + 0] = R; img[i * 3 + 1] = G; img[i * 3 + 2] = B;
        }

        PreprocessedImage p = preprocess(img.data(), w, h, 3, S);

        // 4x2 image, longest side 4 -> scale 2 -> content 8x4 in an 8x8 square.
        check(p.transform.resized_w == 8 && p.transform.resized_h == 4,
              "4x2 content rect is 8x4");
        check(p.pixels.rows == 1 && p.pixels.cols == 3 * S * S,
              "pixels shape is (1, 3*S*S)");
        check_near(p.transform.scale, 2.0f, 1e-6f, "scale == 2");

        const float* px = p.pixels.host_f32();
        const int plane = S * S;
        const uint8_t chan[3] = {R, G, B};

        // Content region: every pixel equals the per-channel normalized color
        // (bilinear of a constant image is the constant).
        bool content_ok = true, pad_ok = true;
        for (int c = 0; c < 3; ++c) {
            const float want = expected_norm(chan[c], c);
            for (int y = 0; y < S; ++y) {
                for (int x = 0; x < S; ++x) {
                    const float v = px[c * plane + y * S + x];
                    const bool in_content = (y < 4 && x < 8);
                    if (in_content) {
                        if (std::fabs(v - want) > 1e-4f) content_ok = false;
                    } else {
                        if (v != 0.0f) pad_ok = false;
                    }
                }
            }
        }
        check(content_ok, "content pixels carry the normalized color");
        check(pad_ok, "padding region is exactly 0.0");
    }

    // ── Alpha is dropped: RGBA matches the RGB result ──
    {
        const int w = 3, h = 3, S = 6;
        std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
        std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4);
        for (int i = 0; i < w * h; ++i) {
            uint8_t r = static_cast<uint8_t>(i * 7);
            uint8_t g = static_cast<uint8_t>(i * 11);
            uint8_t b = static_cast<uint8_t>(i * 13);
            rgb[i * 3 + 0] = r; rgb[i * 3 + 1] = g; rgb[i * 3 + 2] = b;
            rgba[i * 4 + 0] = r; rgba[i * 4 + 1] = g; rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = static_cast<uint8_t>(i * 17);  // arbitrary alpha
        }
        PreprocessedImage a = preprocess(rgb.data(),  w, h, 3, S);
        PreprocessedImage c = preprocess(rgba.data(), w, h, 4, S);
        const float* pa = a.pixels.host_f32();
        const float* pc = c.pixels.host_f32();
        bool same = true;
        for (int i = 0; i < 3 * S * S; ++i)
            if (std::fabs(pa[i] - pc[i]) > 1e-6f) same = false;
        check(same, "RGBA (alpha ignored) matches RGB");
    }

    // ── apply_coords maps original -> model-input space ──
    {
        ImageTransform t;
        t.orig_w = 100; t.orig_h = 50; t.input_size = 1024; t.scale = 1024.0f / 100.0f;
        float ox = 0.f, oy = 0.f;
        apply_coords(t, 50.0f, 25.0f, ox, oy);
        check_near(ox, 512.0f, 1e-3f, "apply_coords x: 50 -> 512");
        check_near(oy, 256.0f, 1e-3f, "apply_coords y: 25 -> 256");
    }

    // ── Default 1024 input produces the stock SAM tensor shape ──
    {
        std::vector<uint8_t> img(static_cast<size_t>(8) * 8 * 3, 128);
        PreprocessedImage p = preprocess(img.data(), 8, 8, 3);  // default 1024
        check(p.transform.input_size == 1024, "default input_size is 1024");
        check(p.pixels.rows == 1 && p.pixels.cols == 3 * 1024 * 1024,
              "default pixels shape is (1, 3*1024*1024)");
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("sam_preprocess: all checks passed\n");
    return 0;
}
