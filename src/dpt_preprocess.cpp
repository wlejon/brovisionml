#include "brovisionml/dpt_preprocess.h"

#include "broimage/geometric.h"
#include "broimage/preproc.h"
#include "broimage/normalize.h"
#include "broimage/presets.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace brovisionml::dpt {

namespace {

// HF DPT `constrain_to_multiple_of(val, multiple, min_val=multiple)`:
// round to the nearest multiple (round-half-to-even, matching Python round),
// then floor at one `multiple` so a tiny image never collapses to zero.
int constrain_to_multiple(double val, int multiple) {
    int x = static_cast<int>(std::nearbyint(val / multiple)) * multiple;
    if (x < multiple) x = multiple;
    return x;
}

}  // namespace

void resize_output_size(int w, int h, int target, int multiple,
                        bool keep_aspect_ratio, int& out_w, int& out_h) {
    double scale_h = static_cast<double>(target) / static_cast<double>(h);
    double scale_w = static_cast<double>(target) / static_cast<double>(w);

    if (keep_aspect_ratio) {
        // Share the scale closest to 1 across both axes (HF DPT). This keeps the
        // aspect ratio; the shorter side lands on `target`, the longer overshoots.
        if (std::fabs(1.0 - scale_w) < std::fabs(1.0 - scale_h))
            scale_h = scale_w;
        else
            scale_w = scale_h;
    }

    out_h = constrain_to_multiple(scale_h * h, multiple);
    out_w = constrain_to_multiple(scale_w * w, multiple);
}

PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int target, int multiple, bool keep_aspect_ratio) {
    if (!rgb || w <= 0 || h <= 0)
        throw std::runtime_error("dpt::preprocess: zero-size or null image");
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("dpt::preprocess: channels must be 1, 3, or 4");
    if (target <= 0 || multiple <= 0)
        throw std::runtime_error("dpt::preprocess: target and multiple must be positive");

    // 1. Compact to packed RGB (drop alpha; broadcast gray) so resize/normalize
    //    operate on exactly the 3 channels the model wants.
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

    int new_w = 0, new_h = 0;
    resize_output_size(w, h, target, multiple, keep_aspect_ratio, new_w, new_h);

    // 2. Bicubic resize (Catmull-Rom == PIL BICUBIC). Aspect ratio preserved;
    //    no padding.
    std::vector<uint8_t> resized(static_cast<size_t>(new_w) * new_h * 3);
    broimage::resize_hwc_u8(rgb3.data(), w, h, 3,
                            resized.data(), new_w, new_h,
                            broimage::Filter::Bicubic);

    // 3. Scale [0,255]->[0,1] into planar NCHW, then per-channel ImageNet
    //    normalize, in place.
    const int plane = new_w * new_h;
    PreprocessedImage out;
    out.pixels = brotensor::Tensor::mat(1, 3 * plane);
    float* dst = out.pixels.host_f32_mut();
    broimage::u8_nhwc_to_f32_nchw(resized.data(), /*N=*/1, new_h, new_w, /*C=*/3,
                                  /*scale=*/1.0f / 255.0f, /*bias=*/0.0f, dst);
    broimage::image_normalize_nchw_f32(dst,
                                       broimage::IMAGENET_MEAN, broimage::IMAGENET_STD,
                                       /*N=*/1, /*C=*/3, new_h, new_w, dst);

    out.transform.orig_w    = w;
    out.transform.orig_h    = h;
    out.transform.resized_w = new_w;
    out.transform.resized_h = new_h;
    return out;
}

}  // namespace brovisionml::dpt
