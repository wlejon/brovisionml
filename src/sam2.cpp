#include "brovisionml/sam2.h"

#include "sam2_ops.h"

#include "broimage/geometric.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace brovisionml {

namespace bt = brotensor;
using bt::Tensor;

int Sam2Segmentation::best() const {
    int b = 0;
    for (int i = 1; i < static_cast<int>(iou.size()); ++i)
        if (iou[static_cast<std::size_t>(i)] > iou[static_cast<std::size_t>(b)]) b = i;
    return b;
}

bool sam2_mask_box(const float* logits, int width, int height, std::array<int, 4>& box) {
    int x0 = width, y0 = height, x1 = -1, y1 = -1;
    for (int y = 0; y < height; ++y) {
        const float* row = logits + static_cast<std::size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            if (row[x] > 0.0f) {
                x0 = std::min(x0, x); x1 = std::max(x1, x);
                y0 = std::min(y0, y); y1 = std::max(y1, y);
            }
        }
    }
    if (x1 < 0) { box = {0, 0, 0, 0}; return false; }
    box = {x0, y0, x1 + 1, y1 + 1};
    return true;
}

Sam2::Sam2(sam2::Sam2Config cfg)
    : cfg_(cfg), encoder_(cfg), heads_(cfg), memory_(cfg) {}

void Sam2::load(const std::string& dir) {
    const std::string cfg_path = dir + "/config.json";
    if (std::filesystem::exists(cfg_path)) {
        cfg_ = sam2::Sam2Config::from_json_file(cfg_path);
        encoder_ = sam2::ImageEncoder(cfg_);
        heads_ = sam2::SamHeads(cfg_);
        memory_ = sam2::MemoryModule(cfg_);
    }
    const bt::safetensors::File f = bt::safetensors::File::open(dir + "/model.safetensors");
    encoder_.load(f);
    heads_.load(f);
    memory_.load(f);
    features_ = {};
}

void Sam2::to(bt::Device dev) {
    encoder_.to(dev);
    heads_.to(dev);
    memory_.to(dev);
    norm_w_ = Tensor();
    norm_b_ = Tensor();
    if (!features_.empty()) {
        features_.top = features_.top.to(dev);
        features_.feat_s0 = features_.feat_s0.to(dev);
        features_.feat_s1 = features_.feat_s1.to(dev);
    }
}

Tensor Sam2::preprocess(const uint8_t* pixels, int w, int h, int channels) const {
    if (!pixels || w <= 0 || h <= 0) throw std::runtime_error("Sam2: empty image");
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("Sam2: channels must be 1, 3 or 4");
    const int S = cfg_.image_size;
    // ToTensor (/255) -> Resize((S, S)) on the float image -> ImageNet
    // normalize, as SAM2Transforms does. Area for a reduction (the reference
    // resizes antialiased), bilinear otherwise.
    std::vector<float> src(static_cast<std::size_t>(w) * h * 3);
    for (int i = 0; i < w * h; ++i) {
        const uint8_t* p = pixels + static_cast<std::size_t>(i) * channels;
        for (int c = 0; c < 3; ++c)
            src[static_cast<std::size_t>(i) * 3 + c] = (channels == 1 ? p[0] : p[c]) / 255.0f;
    }
    std::vector<float> dst(static_cast<std::size_t>(S) * S * 3);
    const bool reduce = w > S || h > S;
    broimage::resize_hwc_f32(src.data(), w, h, 3, dst.data(), S, S,
                             reduce ? broimage::Filter::Area : broimage::Filter::Bilinear);
    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
    Tensor out = Tensor::mat(1, 3 * S * S);
    float* d = out.host_f32_mut();
    const std::size_t plane = static_cast<std::size_t>(S) * S;
    for (std::size_t i = 0; i < plane; ++i)
        for (int c = 0; c < 3; ++c)
            d[static_cast<std::size_t>(c) * plane + i] = (dst[i * 3 + c] - kMean[c]) / kStd[c];
    return out;
}

Tensor Sam2::preprocess_device(const uint8_t* pixels, int w, int h, int channels) const {
    const int S = cfg_.image_size;
    const bt::Device dev = encoder_.device();
    // A reduction keeps the host path (area filter, standing in for the
    // reference's antialiased resize); an enlargement is plain bilinear,
    // which the device does: planar /255 on the host, upload the small image,
    // resize, then normalize per channel as a depthwise 1x1 convolution.
    if (w > S || h > S || dev == bt::Device::CPU) return preprocess(pixels, w, h, channels).to(dev);
    if (!pixels || w <= 0 || h <= 0) throw std::runtime_error("Sam2: empty image");
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("Sam2: channels must be 1, 3 or 4");
    const std::size_t plane = static_cast<std::size_t>(w) * h;
    Tensor src = Tensor::mat(1, static_cast<int>(3 * plane));
    float* d = src.host_f32_mut();
    for (std::size_t i = 0; i < plane; ++i) {
        const uint8_t* p = pixels + i * channels;
        for (int c = 0; c < 3; ++c) d[c * plane + i] = (channels == 1 ? p[0] : p[c]) / 255.0f;
    }
    Tensor up;
    bt::interp2d_forward(src.to(dev), 1, 3, h, w, S, S, /*bilinear*/ 1, up);
    if (!norm_w_.data) {
        static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
        static constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
        Tensor nw = Tensor::mat(3, 1), nb = Tensor::mat(3, 1);
        for (int c = 0; c < 3; ++c) {
            nw.host_f32_mut()[c] = 1.0f / kStd[c];
            nb.host_f32_mut()[c] = -kMean[c] / kStd[c];
        }
        norm_w_ = nw.to(dev);
        norm_b_ = nb.to(dev);
    }
    Tensor out;
    bt::conv2d_forward(up, norm_w_, &norm_b_, 1, 3, S, S, 3, 1, 1, 1, 1, 0, 0, 1, 1, /*groups=*/3, out);
    return out;
}

sam2::ImageFeatures Sam2::encode(const uint8_t* pixels, int w, int h, int channels) const {
    return encoder_.encode(preprocess_device(pixels, w, h, channels));
}

void Sam2::set_image(const uint8_t* pixels, int w, int h, int channels) {
    features_ = encode(pixels, w, h, channels);
    image_w_ = w;
    image_h_ = h;
}

void Sam2::map_prompt(const Sam2Prompt& prompt, int w, int h,
                      std::vector<std::array<float, 2>>& points, std::vector<int>& labels) const {
    if (prompt.labels.size() != prompt.points.size())
        throw std::runtime_error("Sam2: points and labels differ in length");
    const float sx = static_cast<float>(cfg_.image_size) / static_cast<float>(w);
    const float sy = static_cast<float>(cfg_.image_size) / static_cast<float>(h);
    points.clear();
    labels.clear();
    if (prompt.has_box) {
        points.push_back({prompt.box[0] * sx, prompt.box[1] * sy});
        points.push_back({prompt.box[2] * sx, prompt.box[3] * sy});
        labels.push_back(2);
        labels.push_back(3);
    }
    for (std::size_t i = 0; i < prompt.points.size(); ++i) {
        const int l = prompt.labels[i];
        if (l != 0 && l != 1) throw std::runtime_error("Sam2: click labels must be 0 or 1");
        points.push_back({prompt.points[i][0] * sx, prompt.points[i][1] * sy});
        labels.push_back(l);
    }
}

std::vector<float> Sam2::upscale(const Tensor& low_res, int num, int w, int h) const {
    const int ms = cfg_.mask_size();
    Tensor v = Tensor::view(low_res.device, low_res.data, num, ms * ms);
    Tensor up;
    bt::interp2d_forward(v, num, 1, ms, ms, h, w, /*bilinear*/ 1, up);
    Tensor hostv = up.to(bt::Device::CPU);
    const float* p = hostv.host_f32();
    return std::vector<float>(p, p + static_cast<std::size_t>(num) * w * h);
}

Sam2Segmentation Sam2::segment(const Sam2Prompt& prompt, bool multimask,
                               const Tensor* mask_input) const {
    if (features_.empty()) throw std::runtime_error("Sam2: segment() before set_image()");
    sam2::HeadsPrompt hp;
    map_prompt(prompt, image_w_, image_h_, hp.points, hp.labels);
    hp.mask_input = mask_input;
    if (hp.points.empty() && !mask_input) throw std::runtime_error("Sam2: empty prompt");
    // The image predictor conditions the decoder on "no memory".
    const Tensor image = memory_.no_memory(features_.top);
    sam2::HeadsOutput o = heads_.decode(image, features_.feat_s0, features_.feat_s1, hp, multimask);
    Sam2Segmentation out;
    out.num = o.num;
    out.width = image_w_;
    out.height = image_h_;
    out.iou = o.iou;
    out.object_score = o.object_score;
    out.logits = upscale(o.low_res, o.num, image_w_, image_h_);
    out.low_res = std::move(o.low_res);
    return out;
}

}  // namespace brovisionml
