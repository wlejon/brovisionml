#include "brovisionml/mlsd_preprocess.h"

#include "broimage/geometric.h"
#include "broimage/preproc.h"

#include <stdexcept>
#include <vector>

namespace brovisionml::mlsd {

PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int model_size) {
    if (!rgb || w <= 0 || h <= 0)
        throw std::runtime_error("mlsd::preprocess: zero-size or null image");
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("mlsd::preprocess: channels must be 1, 3, or 4");
    if (model_size <= 0)
        throw std::runtime_error("mlsd::preprocess: model_size must be positive");

    // 1. Compact to packed RGB.
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

    // 2. Resize to model_size×model_size (bilinear), or keep if already square.
    const int S = model_size;
    const uint8_t* src = rgb3.data();
    std::vector<uint8_t> resized;
    if (w != S || h != S) {
        resized.resize(static_cast<size_t>(S) * S * 3);
        broimage::resize_hwc_u8(rgb3.data(), w, h, 3, resized.data(), S, S,
                                broimage::Filter::Bilinear);
        src = resized.data();
    }

    // 3. Pack RGB into the first 3 NCHW planes, normalized by (x/127.5 - 1)...
    const int plane = S * S;
    PreprocessedImage out;
    out.pixels = brotensor::Tensor::mat(1, 4 * plane);
    float* dst = out.pixels.host_f32_mut();
    broimage::u8_nhwc_to_f32_nchw(src, /*N=*/1, S, S, /*C=*/3,
                                  /*scale=*/1.0f / 127.5f, /*bias=*/-1.0f, dst);

    // 4. ...and fill the 4th plane with the normalized "ones" channel: the
    //    reference appends a ones plane BEFORE normalize, so it becomes the
    //    constant (1/127.5 - 1).
    const float ones_norm = 1.0f / 127.5f - 1.0f;
    float* plane3 = dst + static_cast<size_t>(3) * plane;
    for (int i = 0; i < plane; ++i) plane3[i] = ones_norm;

    out.transform.orig_w = w;
    out.transform.orig_h = h;
    out.transform.model_size = S;
    out.transform.tp_w = S / 2;
    out.transform.tp_h = S / 2;
    return out;
}

}  // namespace brovisionml::mlsd
