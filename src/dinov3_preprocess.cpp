#include "brovisionml/dinov3_preprocess.h"

#include "broimage/geometric.h"
#include "broimage/preproc.h"
#include "broimage/normalize.h"
#include "broimage/presets.h"

#include <stdexcept>
#include <vector>

namespace brovisionml::dinov3 {

PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int resolution, int patch_size) {
    if (!rgb || w <= 0 || h <= 0)
        throw std::runtime_error("dinov3::preprocess: zero-size or null image");
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("dinov3::preprocess: channels must be 1, 3, or 4");
    if (patch_size <= 0)
        throw std::runtime_error("dinov3::preprocess: patch_size must be positive");
    if (resolution <= 0 || resolution % patch_size != 0)
        throw std::runtime_error(
            "dinov3::preprocess: resolution must be a positive multiple of patch_size");

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

    // 2. Square bicubic resize (Catmull-Rom == PIL BICUBIC). Straight stretch to
    //    (resolution × resolution); no aspect preservation, no padding.
    std::vector<uint8_t> resized(static_cast<size_t>(resolution) * resolution * 3);
    broimage::resize_hwc_u8(rgb3.data(), w, h, 3,
                            resized.data(), resolution, resolution,
                            broimage::Filter::Bicubic);

    // 3. Scale [0,255]->[0,1] into planar NCHW, then per-channel ImageNet
    //    normalize, in place.
    const int plane = resolution * resolution;
    PreprocessedImage out;
    out.pixels = brotensor::Tensor::mat(1, 3 * plane);
    float* dst = out.pixels.host_f32_mut();
    broimage::u8_nhwc_to_f32_nchw(resized.data(), /*N=*/1, resolution, resolution, /*C=*/3,
                                  /*scale=*/1.0f / 255.0f, /*bias=*/0.0f, dst);
    broimage::image_normalize_nchw_f32(dst,
                                       broimage::IMAGENET_MEAN, broimage::IMAGENET_STD,
                                       /*N=*/1, /*C=*/3, resolution, resolution, dst);

    out.transform.orig_w     = w;
    out.transform.orig_h     = h;
    out.transform.resolution = resolution;
    out.transform.scale_x    = static_cast<float>(resolution) / static_cast<float>(w);
    out.transform.scale_y    = static_cast<float>(resolution) / static_cast<float>(h);
    return out;
}

}  // namespace brovisionml::dinov3
