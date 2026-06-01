#include "brovisionml/openpose_preprocess.h"

#include "broimage/geometric.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace brovisionml::openpose {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("openpose::preprocess: " + msg);
}

// Compact an interleaved HWC image (channels 1/3/4) to packed RGB (3ch). For a
// 4-channel source the reference HWC3 alpha-composites over white.
std::vector<uint8_t> to_rgb3(const uint8_t* rgb, int w, int h, int channels) {
    std::vector<uint8_t> out(static_cast<std::size_t>(w) * h * 3);
    for (int i = 0; i < w * h; ++i) {
        const uint8_t* s = rgb + static_cast<std::size_t>(i) * channels;
        uint8_t r, g, b;
        if (channels == 1) {
            r = g = b = s[0];
        } else if (channels == 3) {
            r = s[0]; g = s[1]; b = s[2];
        } else {  // 4: composite over white by alpha
            const float a = s[3] / 255.0f;
            r = static_cast<uint8_t>(s[0] * a + 255.0f * (1.0f - a) + 0.5f);
            g = static_cast<uint8_t>(s[1] * a + 255.0f * (1.0f - a) + 0.5f);
            b = static_cast<uint8_t>(s[2] * a + 255.0f * (1.0f - a) + 0.5f);
        }
        out[static_cast<std::size_t>(i) * 3 + 0] = r;
        out[static_cast<std::size_t>(i) * 3 + 1] = g;
        out[static_cast<std::size_t>(i) * 3 + 2] = b;
    }
    return out;
}

}  // namespace

std::vector<uint8_t> resize_to_detect(const uint8_t* rgb, int w, int h,
                                      int channels, int detect_resolution,
                                      int& out_w, int& out_h) {
    if (!rgb || w <= 0 || h <= 0) fail("zero-size or null image");
    if (channels != 1 && channels != 3 && channels != 4)
        fail("channels must be 1, 3, or 4");
    if (detect_resolution <= 0) fail("detect_resolution must be positive");

    std::vector<uint8_t> rgb3 = to_rgb3(rgb, w, h, channels);

    // resize_image: scale so the shorter side == resolution, rounded to a
    // multiple of 64. Lanczos on upscale (k>1), Area on downscale.
    const double k = static_cast<double>(detect_resolution) /
                     static_cast<double>(std::min(h, w));
    int H = static_cast<int>(std::lround(h * k / 64.0)) * 64;
    int W = static_cast<int>(std::lround(w * k / 64.0)) * 64;
    if (H < 64) H = 64;
    if (W < 64) W = 64;

    out_w = W;
    out_h = H;
    std::vector<uint8_t> resized(static_cast<std::size_t>(W) * H * 3);
    if (W == w && H == h) {
        resized = std::move(rgb3);
    } else {
        const broimage::Filter filt =
            (k > 1.0) ? broimage::Filter::Lanczos3 : broimage::Filter::Area;
        broimage::resize_hwc_u8(rgb3.data(), w, h, 3, resized.data(), W, H, filt);
    }
    return resized;
}

PreprocessedImage preprocess_from_detect(const uint8_t* rgb_detect,
                                         int detect_w, int detect_h) {
    if (!rgb_detect || detect_w <= 0 || detect_h <= 0)
        fail("zero-size detect image");

    const int H = detect_h, W = detect_w;

    // Promote the detect-res RGB to float (cv2 works in float for the resize),
    // flipping RGB->BGR so the channel order fed to conv1_1 is B,G,R.
    std::vector<float> bgr(static_cast<std::size_t>(W) * H * 3);
    for (int i = 0; i < W * H; ++i) {
        bgr[static_cast<std::size_t>(i) * 3 + 0] =
            static_cast<float>(rgb_detect[static_cast<std::size_t>(i) * 3 + 2]);  // B
        bgr[static_cast<std::size_t>(i) * 3 + 1] =
            static_cast<float>(rgb_detect[static_cast<std::size_t>(i) * 3 + 1]);  // G
        bgr[static_cast<std::size_t>(i) * 3 + 2] =
            static_cast<float>(rgb_detect[static_cast<std::size_t>(i) * 3 + 0]);  // R
    }

    // scale = 0.5 * 368 / H; imageToTest dims = round(dim * scale).
    const float scale = 0.5f * 368.0f / static_cast<float>(H);
    const int tH = static_cast<int>(std::lround(H * static_cast<double>(scale)));
    const int tW = static_cast<int>(std::lround(W * static_cast<double>(scale)));

    // smart_resize_k: Area on downscale (k<1), Lanczos otherwise. For scale ~0.5
    // this is always a downscale.
    std::vector<float> test(static_cast<std::size_t>(tW) * tH * 3);
    {
        const double sk = static_cast<double>(tH + tW) /
                          static_cast<double>(H + W);
        const broimage::Filter filt =
            (sk < 1.0) ? broimage::Filter::Area : broimage::Filter::Lanczos3;
        broimage::resize_hwc_f32(bgr.data(), W, H, 3, test.data(), tW, tH, filt);
    }

    // padRightDownCorner(stride=8, padValue=128): pad bottom+right to a multiple
    // of 8 with the constant 128 (pixel space).
    const int stride = 8;
    const int pad_down = (tH % stride == 0) ? 0 : stride - (tH % stride);
    const int pad_right = (tW % stride == 0) ? 0 : stride - (tW % stride);
    const int Hp = tH + pad_down;
    const int Wp = tW + pad_right;

    // Normalize im = padded/256 - 0.5, transpose HWC(BGR)->CHW, batch 1.
    const std::size_t plane = static_cast<std::size_t>(Hp) * Wp;
    PreprocessedImage out;
    out.pixels = brotensor::Tensor::mat(1, static_cast<int>(3 * plane));
    float* dst = out.pixels.host_f32_mut();
    for (int c = 0; c < 3; ++c) {
        float* ch = dst + static_cast<std::size_t>(c) * plane;
        for (int y = 0; y < Hp; ++y) {
            for (int x = 0; x < Wp; ++x) {
                float px;
                if (y < tH && x < tW) {
                    px = test[(static_cast<std::size_t>(y) * tW + x) * 3 + c];
                } else {
                    px = 128.0f;  // pad value
                }
                ch[static_cast<std::size_t>(y) * Wp + x] = px / 256.0f - 0.5f;
            }
        }
    }

    out.transform.detect_w = W;
    out.transform.detect_h = H;
    out.transform.test_w = tW;
    out.transform.test_h = tH;
    out.transform.pad_w = Wp;
    out.transform.pad_h = Hp;
    out.transform.pad_down = pad_down;
    out.transform.pad_right = pad_right;
    out.transform.scale = scale;
    return out;
}

PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int detect_resolution) {
    int dw = 0, dh = 0;
    std::vector<uint8_t> det =
        resize_to_detect(rgb, w, h, channels, detect_resolution, dw, dh);
    return preprocess_from_detect(det.data(), dw, dh);
}

}  // namespace brovisionml::openpose
