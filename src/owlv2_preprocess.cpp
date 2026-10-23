// OWLv2 preprocessing: pad to a square with 0.5 grey, anti-aliased resize to
// the model input, CLIP mean/std, planar NCHW.
//
// The resize is HF Owlv2ImageProcessor's (itself the big_vision recipe, after
// skimage's resize): when downscaling by `factor` = S / size, a Gaussian blur
// of sigma = (factor - 1) / 2 over 2 * ceil(3 * sigma) + 1 taps (torchvision
// gaussian_blur: reflect border, normalised taps), then bilinear sampling with
// align_corners=False (torch's interpolate: source clamped at 0, no
// antialias). Upscaling skips the blur.
//
// Only the blurred values the bilinear taps read are computed: for each output
// row, the (at most two) source rows it samples are blurred vertically across
// the width once, then horizontally only at the sampled columns. A 6000-pixel
// photo therefore costs ~2 x 960 blurred rows, not 6000^2 pixels. Bands of
// output rows run on up to 8 threads, each with its own row cache.

#include "brovisionml/owlv2.h"

#include "brotensor/detail/cpu/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace brovisionml::owlv2 {

namespace {

constexpr float kMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float kStd[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

// torch's bilinear source taps for one axis (align_corners=False).
struct Taps {
    std::vector<int> i0, i1;
    std::vector<float> l1;
};

Taps bilinear_taps(int in, int out) {
    Taps t;
    t.i0.resize(out); t.i1.resize(out); t.l1.resize(out);
    const float scale = static_cast<float>(in) / static_cast<float>(out);
    for (int o = 0; o < out; ++o) {
        float src = scale * (static_cast<float>(o) + 0.5f) - 0.5f;
        if (src < 0.f) src = 0.f;
        const int i0 = std::min(static_cast<int>(src), in - 1);
        t.i0[o] = i0;
        t.i1[o] = i0 < in - 1 ? i0 + 1 : i0;
        t.l1[o] = src - static_cast<float>(i0);
    }
    return t;
}

int reflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i;
    return i;
}

}  // namespace

Pixels preprocess(const uint8_t* pixels, int width, int height, int channels, int size) {
    if (!pixels || width <= 0 || height <= 0 || (channels != 3 && channels != 4) || size <= 0)
        throw std::runtime_error("owlv2::preprocess: bad image (needs 8-bit RGB or RGBA rows)");
    const int S = std::max(width, height);

    // The padded square in [0, 1], one plane per channel, read through this.
    auto at = [&](int c, int y, int x) -> float {
        if (x >= width || y >= height) return 0.5f;
        return static_cast<float>(pixels[(static_cast<std::size_t>(y) * width + x) * channels + c]) *
               (1.0f / 255.0f);
    };

    // Gaussian taps (empty when not downscaling).
    const double factor = static_cast<double>(S) / size;
    const double sigma = std::max((factor - 1.0) / 2.0, 0.0);
    std::vector<float> g;
    int radius = 0;
    if (sigma > 0.0) {
        const int k = 2 * static_cast<int>(std::ceil(3.0 * sigma)) + 1;
        radius = k / 2;
        g.resize(k);
        double sum = 0.0;
        for (int i = 0; i < k; ++i) {
            const double x = (i - radius) / sigma;
            g[i] = static_cast<float>(std::exp(-0.5 * x * x));
            sum += g[i];
        }
        for (float& v : g) v = static_cast<float>(v / sum);
    }

    const Taps ty = bilinear_taps(S, size), tx = bilinear_taps(S, size);

    Pixels out;
    out.width = width;
    out.height = height;
    out.padded = S;
    out.nchw = brotensor::Tensor::mat(1, 3 * size * size);
    float* dst = out.nchw.host_f32_mut();
    const std::size_t plane = static_cast<std::size_t>(size) * size;

    // Horizontal blur of a vertically blurred row at one column.
    auto hval = [&](const std::vector<float>& v, int c, int x) -> float {
        const float* row = v.data() + static_cast<std::size_t>(c) * S;
        if (g.empty()) return row[x];
        float acc = 0.f;
        for (int k = -radius; k <= radius; ++k) acc += g[k + radius] * row[reflect(x + k, S)];
        return acc;
    };

    // Output rows [oy0, oy1): each band keeps its own row cache, and every
    // output value is computed the same way in any band, so the split does
    // not change the result.
    auto band = [&](int oy0, int oy1) {
        // Vertically blurred source rows, cached by row index (consecutive
        // output rows share their taps). Each is 3 planes of S floats.
        struct Row { int y = -1; std::vector<float> v; };
        Row cache[2];
        int next_slot = 0;
        auto vrow = [&](int y) -> const std::vector<float>& {
            for (Row& r : cache) if (r.y == y) return r.v;
            Row& r = cache[next_slot];
            next_slot ^= 1;
            r.y = y;
            r.v.assign(static_cast<std::size_t>(3) * S, 0.f);
            for (int c = 0; c < 3; ++c) {
                float* row = r.v.data() + static_cast<std::size_t>(c) * S;
                if (g.empty()) {
                    for (int x = 0; x < S; ++x) row[x] = at(c, y, x);
                    continue;
                }
                for (int k = -radius; k <= radius; ++k) {
                    const int yy = reflect(y + k, S);
                    const float w = g[k + radius];
                    for (int x = 0; x < S; ++x) row[x] += w * at(c, yy, x);
                }
            }
            return r.v;
        };

        std::vector<float> top(static_cast<std::size_t>(3) * size), bot(static_cast<std::size_t>(3) * size);
        for (int oy = oy0; oy < oy1; ++oy) {
            // Horizontal resample of the two source rows, then blend them.
            for (int pass = 0; pass < 2; ++pass) {
                const std::vector<float>& v = vrow(pass == 0 ? ty.i0[oy] : ty.i1[oy]);
                std::vector<float>& h = pass == 0 ? top : bot;
                for (int c = 0; c < 3; ++c) {
                    for (int ox = 0; ox < size; ++ox) {
                        const float a = hval(v, c, tx.i0[ox]), b = hval(v, c, tx.i1[ox]);
                        h[static_cast<std::size_t>(c) * size + ox] = a + (b - a) * tx.l1[ox];
                    }
                }
            }
            const float l = ty.l1[oy];
            for (int c = 0; c < 3; ++c) {
                float* d = dst + c * plane + static_cast<std::size_t>(oy) * size;
                const float* a = top.data() + static_cast<std::size_t>(c) * size;
                const float* b = bot.data() + static_cast<std::size_t>(c) * size;
                for (int ox = 0; ox < size; ++ox)
                    d[ox] = ((a[ox] + (b[ox] - a[ox]) * l) - kMean[c]) / kStd[c];
            }
        }
    };

    // Bands of output rows via brotensor's parallel_for; a small input stays on one.
    if (static_cast<std::size_t>(S) * S < (std::size_t(1) << 20)) {
        band(0, size);
    } else {
        constexpr int kBand = 16;
        const std::size_t n_bands = static_cast<std::size_t>((size + kBand - 1) / kBand);
        brotensor::detail::cpu::parallel_for(n_bands, [&](std::size_t bi) {
            const int oy0 = static_cast<int>(bi) * kBand;
            const int oy1 = std::min(size, oy0 + kBand);
            band(oy0, oy1);
        });
    }
    return out;
}

}  // namespace brovisionml::owlv2
