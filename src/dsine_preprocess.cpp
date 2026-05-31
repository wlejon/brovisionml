#include "brovisionml/dsine_preprocess.h"

#include "broimage/geometric.h"
#include "broimage/preproc.h"
#include "broimage/normalize.h"
#include "broimage/presets.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace brovisionml::dsine {

void get_padding(int orig_h, int orig_w, int& l, int& r, int& t, int& b) {
    // Width: grow to the next multiple of 32, smaller half on the left.
    if (orig_w % 32 == 0) {
        l = 0;
        r = 0;
    } else {
        int new_w = 32 * ((orig_w / 32) + 1);
        l = (new_w - orig_w) / 2;
        r = (new_w - orig_w) - l;
    }
    // Height: same, smaller half on the top.
    if (orig_h % 32 == 0) {
        t = 0;
        b = 0;
    } else {
        int new_h = 32 * ((orig_h / 32) + 1);
        t = (new_h - orig_h) / 2;
        b = (new_h - orig_h) - t;
    }
}

Intrinsics intrins_from_fov(int orig_h, int orig_w, float fov_deg) {
    // f = (max(H,W)/2) / tan(fov/2); principal point at the image center - 0.5.
    constexpr double kPi = 3.14159265358979323846;
    const double half_fov_rad = (static_cast<double>(fov_deg) / 2.0) * kPi / 180.0;
    const double f = (std::max(orig_h, orig_w) / 2.0) / std::tan(half_fov_rad);
    Intrinsics k;
    k.fx = static_cast<float>(f);
    k.fy = static_cast<float>(f);
    k.cx = static_cast<float>(orig_w / 2.0 - 0.5);
    k.cy = static_cast<float>(orig_h / 2.0 - 0.5);
    return k;
}

PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             float fov_deg) {
    if (!rgb || w <= 0 || h <= 0)
        throw std::runtime_error("dsine::preprocess: zero-size or null image");
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("dsine::preprocess: channels must be 1, 3, or 4");

    // 1. Compact to packed RGB (drop alpha; broadcast gray) so pad/normalize
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

    int l = 0, rgt = 0, t = 0, btm = 0;
    get_padding(h, w, l, rgt, t, btm);
    const int pad_w = w + l + rgt;
    const int pad_h = h + t + btm;

    // 2. Zero-pad to a multiple of 32 (content placed at offset (l, t)). The pad
    //    is applied on the uint8 image so padded pixels are 0 before normalize.
    std::vector<uint8_t> padded(static_cast<size_t>(pad_w) * pad_h * 3);
    broimage::pad_hwc_u8(rgb3.data(), w, h, 3,
                         padded.data(), pad_w, pad_h,
                         /*off_x=*/l, /*off_y=*/t,
                         /*pad_r=*/0, /*pad_g=*/0, /*pad_b=*/0, /*pad_a=*/0);

    // 3. Scale [0,255]->[0,1] into planar NCHW, then per-channel ImageNet
    //    normalize, in place. Padded (0) pixels become (0 - mean)/std.
    const int plane = pad_w * pad_h;
    PreprocessedImage out;
    out.pixels = brotensor::Tensor::mat(1, 3 * plane);
    float* dst = out.pixels.host_f32_mut();
    broimage::u8_nhwc_to_f32_nchw(padded.data(), /*N=*/1, pad_h, pad_w, /*C=*/3,
                                  /*scale=*/1.0f / 255.0f, /*bias=*/0.0f, dst);
    broimage::image_normalize_nchw_f32(dst,
                                       broimage::IMAGENET_MEAN, broimage::IMAGENET_STD,
                                       /*N=*/1, /*C=*/3, pad_h, pad_w, dst);

    // 4. Synthesized intrinsics on the original dims, then shifted by the pad
    //    offset so the principal point stays aligned to image content.
    Intrinsics k = intrins_from_fov(h, w, fov_deg);
    k.cx += static_cast<float>(l);
    k.cy += static_cast<float>(t);

    out.transform.orig_w = w;
    out.transform.orig_h = h;
    out.transform.pad_w  = pad_w;
    out.transform.pad_h  = pad_h;
    out.transform.pad_l  = l;
    out.transform.pad_r  = rgt;
    out.transform.pad_t  = t;
    out.transform.pad_b  = btm;
    out.intrins = k;
    return out;
}

}  // namespace brovisionml::dsine
