#include "brovisionml/lineart_preprocess.h"

#include "broimage/geometric.h"
#include "broimage/preproc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace brovisionml::lineart {

void lineart_proc_dims(int orig_w, int orig_h, int detect_resolution,
                       int& proc_w, int& proc_h) {
    if (detect_resolution <= 0) {
        proc_w = orig_w;
        proc_h = orig_h;
        return;
    }
    const int longer = std::max(orig_w, orig_h);
    const double s = static_cast<double>(detect_resolution) / longer;
    proc_w = std::max(1, static_cast<int>(std::lround(orig_w * s)));
    proc_h = std::max(1, static_cast<int>(std::lround(orig_h * s)));
}

PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int detect_resolution) {
    if (!rgb || w <= 0 || h <= 0)
        throw std::runtime_error("lineart::preprocess: zero-size or null image");
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("lineart::preprocess: channels must be 1, 3, or 4");

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

    int proc_w = 0, proc_h = 0;
    lineart_proc_dims(w, h, detect_resolution, proc_w, proc_h);

    // 2. Resize to the working resolution (bilinear), or keep native.
    const uint8_t* src = rgb3.data();
    std::vector<uint8_t> resized;
    if (proc_w != w || proc_h != h) {
        resized.resize(static_cast<size_t>(proc_w) * proc_h * 3);
        broimage::resize_hwc_u8(rgb3.data(), w, h, 3,
                                resized.data(), proc_w, proc_h,
                                broimage::Filter::Bilinear);
        src = resized.data();
    }

    // 3. Pack to planar NCHW FP32, scaling pixels to [0,1] (value/255 — the
    //    generator was trained on /255 input; no mean/std normalization).
    const int plane = proc_w * proc_h;
    PreprocessedImage out;
    out.pixels = brotensor::Tensor::mat(1, 3 * plane);
    float* dst = out.pixels.host_f32_mut();
    broimage::u8_nhwc_to_f32_nchw(src, /*N=*/1, proc_h, proc_w, /*C=*/3,
                                  /*scale=*/1.0f / 255.0f, /*bias=*/0.0f, dst);

    out.transform.orig_w = w;
    out.transform.orig_h = h;
    out.transform.proc_w = proc_w;
    out.transform.proc_h = proc_h;
    return out;
}

}  // namespace brovisionml::lineart
