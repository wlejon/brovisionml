#include "brovisionml/sam_preprocess.h"

#include "broimage/geometric.h"
#include "broimage/preproc.h"
#include "broimage/normalize.h"
#include "broimage/presets.h"

#include <stdexcept>
#include <vector>

namespace brovisionml::sam {

void preprocess_shape(int w, int h, int input_size,
                      int& resized_w, int& resized_h) {
    // segment_anything ResizeLongestSide.get_preprocess_shape:
    //   scale = long_side / max(h, w); newh = h*scale; neww = w*scale;
    //   round to nearest int.
    const long long longest = (w > h) ? w : h;
    const double scale = static_cast<double>(input_size) / static_cast<double>(longest);
    resized_w = static_cast<int>(static_cast<double>(w) * scale + 0.5);
    resized_h = static_cast<int>(static_cast<double>(h) * scale + 0.5);
    // The longest side maps to exactly input_size; guard rounding from
    // overshooting (e.g. 0.5 ties) so the content always fits the square.
    if (resized_w > input_size) resized_w = input_size;
    if (resized_h > input_size) resized_h = input_size;
    if (resized_w < 1) resized_w = 1;
    if (resized_h < 1) resized_h = 1;
}

void apply_coords(const ImageTransform& t, float x, float y,
                  float& out_x, float& out_y) {
    out_x = x * t.scale;
    out_y = y * t.scale;
}

PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int input_size) {
    if (!rgb || w <= 0 || h <= 0)
        throw std::runtime_error("sam::preprocess: zero-size or null image");
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("sam::preprocess: channels must be 1, 3, or 4");
    if (input_size <= 0)
        throw std::runtime_error("sam::preprocess: input_size must be positive");

    // 1. Compact to packed RGB (drop alpha; broadcast gray) so the resize and
    //    normalize work on exactly the 3 channels the model wants.
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
    preprocess_shape(w, h, input_size, new_w, new_h);

    // 2. Resize longest side. Bilinear to match upstream's PIL BILINEAR; the
    //    content keeps its aspect ratio and lands at the top-left of the square.
    std::vector<uint8_t> resized(static_cast<size_t>(new_w) * new_h * 3);
    broimage::resize_hwc_u8(rgb3.data(), w, h, 3,
                            resized.data(), new_w, new_h,
                            broimage::Filter::Bilinear);

    // 3. Scale [0,255]->[0,1] into planar NCHW, then per-channel normalize. Run
    //    on the (new_h x new_w) content only; the pad is added in step 4 as
    //    exact zeros (matching Sam.preprocess, which pads after normalize).
    const int content = new_w * new_h;
    std::vector<float> norm(static_cast<size_t>(content) * 3);
    broimage::u8_nhwc_to_f32_nchw(resized.data(), /*N=*/1, new_h, new_w, /*C=*/3,
                                  /*scale=*/1.0f / 255.0f, /*bias=*/0.0f,
                                  norm.data());
    broimage::image_normalize_nchw_f32(norm.data(),
                                       broimage::SAM_MEAN, broimage::SAM_STD,
                                       /*N=*/1, /*C=*/3, new_h, new_w,
                                       norm.data());

    // 4. Place each normalized channel plane into the top-left of a zeroed
    //    input_size x input_size square. mat() zero-fills, so the bottom/right
    //    padding is already 0.0.
    PreprocessedImage out;
    out.pixels = brotensor::Tensor::mat(1, 3 * input_size * input_size);
    float* dst = out.pixels.host_f32_mut();
    const int plane = input_size * input_size;
    for (int c = 0; c < 3; ++c) {
        const float* src_plane = norm.data() + static_cast<size_t>(c) * content;
        float* dst_plane = dst + static_cast<size_t>(c) * plane;
        for (int y = 0; y < new_h; ++y) {
            const float* src_row = src_plane + static_cast<size_t>(y) * new_w;
            float* dst_row = dst_plane + static_cast<size_t>(y) * input_size;
            for (int x = 0; x < new_w; ++x) dst_row[x] = src_row[x];
        }
    }

    out.transform.orig_w     = w;
    out.transform.orig_h     = h;
    out.transform.resized_w  = new_w;
    out.transform.resized_h  = new_h;
    out.transform.input_size = input_size;
    out.transform.scale      = static_cast<float>(input_size) /
                               static_cast<float>((w > h) ? w : h);
    return out;
}

}  // namespace brovisionml::sam
