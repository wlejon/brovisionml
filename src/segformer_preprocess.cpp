#include "brovisionml/segformer_preprocess.h"

#include "broimage/geometric.h"
#include "broimage/preproc.h"
#include "broimage/preproc_jit.h"
#include "broimage/normalize.h"
#include "broimage/presets.h"

#include <stdexcept>
#include <vector>

namespace brovisionml::segformer {

PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int model_size) {
    if (!rgb || w <= 0 || h <= 0)
        throw std::runtime_error("segformer::preprocess: zero-size or null image");
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("segformer::preprocess: channels must be 1, 3, or 4");
    if (model_size <= 0)
        throw std::runtime_error("segformer::preprocess: model_size must be positive");

    // 1. Compact to packed RGB (drop alpha; broadcast gray).
    std::vector<uint8_t> rgb3(static_cast<size_t>(w) * h * 3);
    for (int i = 0; i < w * h; ++i) {
        const uint8_t* src = rgb + static_cast<size_t>(i) * channels;
        uint8_t r, g, b;
        if (channels == 1) { r = g = b = src[0]; }
        else               { r = src[0]; g = src[1]; b = src[2]; }
        rgb3[static_cast<size_t>(i) * 3 + 0] = r;
        rgb3[static_cast<size_t>(i) * 3 + 1] = g;
        rgb3[static_cast<size_t>(i) * 3 + 2] = b;
    }

    // 2+3. Resize to model_size×model_size (bilinear) + rescale [0,255]->[0,1]
    //      into planar NCHW + per-channel ImageNet normalize via broimage JIT.
    const int S = model_size;
    const int plane = S * S;
    PreprocessedImage out;
    out.pixels = brotensor::Tensor::mat(1, 3 * plane);
    float* dst = out.pixels.host_f32_mut();
    if (w == S && h == S) {
        broimage::u8_nhwc_to_f32_nchw(rgb3.data(), /*N=*/1, S, S, /*C=*/3,
                                      /*scale=*/1.0f / 255.0f, /*bias=*/0.0f, dst);
        broimage::image_normalize_nchw_f32(dst,
                                           broimage::IMAGENET_MEAN, broimage::IMAGENET_STD,
                                           /*N=*/1, /*C=*/3, S, S, dst);
    } else {
        broimage::fused_u8_nhwc_to_f32_nchw_normalized(
            rgb3.data(), w, h, dst, S, S,
            broimage::IMAGENET_MEAN, broimage::IMAGENET_STD);
    }

    out.transform.orig_w = w;
    out.transform.orig_h = h;
    out.transform.model_size = S;
    return out;
}

}  // namespace brovisionml::segformer
