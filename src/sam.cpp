#include "brovisionml/sam.h"

#include "brotensor/ops.h"

#include <stdexcept>
#include <utility>

namespace brovisionml::sam {

namespace {
using brotensor::Tensor;
[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("sam::Sam: " + msg);
}
}  // namespace

// ─── Config presets ─────────────────────────────────────────────────────────

SamConfig SamConfig::vit_h() {
    SamConfig c;
    c.encoder = EncoderConfig::vit_h();
    return c;  // prompt / decoder use their shared defaults
}
SamConfig SamConfig::vit_l() {
    SamConfig c;
    c.encoder = EncoderConfig::vit_l();
    return c;
}
SamConfig SamConfig::vit_b() {
    SamConfig c;
    c.encoder = EncoderConfig::vit_b();
    return c;
}

int Segmentation::best() const {
    int b = 0;
    for (int i = 1; i < num; ++i)
        if (iou[static_cast<std::size_t>(i)] > iou[static_cast<std::size_t>(b)]) b = i;
    return b;
}

// ─── Construction / loading / migration ──────────────────────────────────────

Sam::Sam(SamConfig cfg)
    : cfg_(cfg), enc_(cfg.encoder), pe_(cfg.prompt), dec_(cfg.decoder) {}

void Sam::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void Sam::load_file(const std::string& path) {
    // One HF SamModel checkpoint carries all three tensor namespaces.
    enc_.load_file(path);
    pe_.load_file(path);
    dec_.load_file(path);
}

void Sam::to(brotensor::Device dev) {
    if (dev == device_) return;
    enc_.to(dev);
    pe_.to(dev);
    dec_.to(dev);
    if (has_image_ && image_embedding_.data)
        image_embedding_ = image_embedding_.to(dev);
    device_ = dev;
}

// ─── Encode image once ───────────────────────────────────────────────────────

void Sam::set_image(const uint8_t* pixels, int w, int h, int channels) {
    PreprocessedImage pp = preprocess(pixels, w, h, channels, cfg_.encoder.img_size);
    transform_ = pp.transform;
    Tensor px = (device_ == brotensor::Device::CPU) ? pp.pixels
                                                     : pp.pixels.to(device_);
    image_embedding_ = enc_.encode(px);   // (1, D*grid*grid) on device_
    has_image_ = true;
}

// ─── Segment ─────────────────────────────────────────────────────────────────

Segmentation Sam::segment(const std::vector<std::array<float, 2>>& points,
                          const std::vector<int>& labels,
                          const std::vector<std::array<float, 4>>& boxes,
                          bool multimask_output) const {
    if (!has_image_) fail("segment() called before set_image()");
    if (labels.size() != points.size())
        fail("points and labels must have equal length");

    // Map prompts from original-image coords into the model's input space.
    PromptInput in;
    in.labels = labels;
    in.points.reserve(points.size());
    for (const auto& p : points) {
        float mx, my;
        apply_coords(transform_, p[0], p[1], mx, my);
        in.points.push_back({mx, my});
    }
    in.boxes.reserve(boxes.size());
    for (const auto& b : boxes) {
        float x1, y1, x2, y2;
        apply_coords(transform_, b[0], b[1], x1, y1);
        apply_coords(transform_, b[2], b[3], x2, y2);
        in.boxes.push_back({x1, y1, x2, y2});
    }

    PromptEmbeddings emb = pe_.encode(in);
    DecodedMasks dm = dec_.decode(image_embedding_, pe_.dense_pe(),
                                  emb.sparse, emb.dense, multimask_output);

    // ── Postprocess masks back to the original image resolution (host). SAM:
    //    upscale ms->input_size, crop off the letterbox padding, resize to the
    //    original size — both resamples bilinear / align_corners=False. ──
    const int num = dm.num_out;
    const int ms  = dm.mask_size;
    const int S   = cfg_.encoder.img_size;
    const int rh  = transform_.resized_h, rw = transform_.resized_w;
    const int oh  = transform_.orig_h,    ow = transform_.orig_w;

    Tensor masks_cpu = dm.masks.to(brotensor::Device::CPU);   // (num, ms*ms)
    Tensor in0 = Tensor::view(brotensor::Device::CPU, masks_cpu.data,
                              1, num * ms * ms);

    Tensor up1;
    brotensor::interp2d_forward(in0, 1, num, ms, ms, S, S, /*bilinear=*/1, up1);

    // Crop each mask plane's top-left content region (the rest is padding).
    Tensor cropped = Tensor::mat(1, num * rh * rw);
    {
        const float* u = up1.host_f32();
        float* c = cropped.host_f32_mut();
        for (int ch = 0; ch < num; ++ch) {
            const float* up = u + static_cast<std::size_t>(ch) * S * S;
            float* cp = c + static_cast<std::size_t>(ch) * rh * rw;
            for (int y = 0; y < rh; ++y)
                for (int x = 0; x < rw; ++x)
                    cp[static_cast<std::size_t>(y) * rw + x] =
                        up[static_cast<std::size_t>(y) * S + x];
        }
    }

    Tensor up2;
    brotensor::interp2d_forward(cropped, 1, num, rh, rw, oh, ow, /*bilinear=*/1, up2);

    Segmentation seg;
    seg.num    = num;
    seg.height = oh;
    seg.width  = ow;
    const float* logit = up2.host_f32();
    seg.logits.assign(logit,
                      logit + static_cast<std::size_t>(num) * oh * ow);

    Tensor iou_cpu = dm.iou.to(brotensor::Device::CPU);   // (num,1)
    const float* iv = iou_cpu.host_f32();
    seg.iou.assign(iv, iv + num);

    return seg;
}

}  // namespace brovisionml::sam
