#include "brovisionml/sam.h"

#include "brotensor/ops.h"

#include "profile.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

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
    detail::profile_mark(device_, nullptr);
    PreprocessedImage pp = preprocess(pixels, w, h, channels, cfg_.encoder.img_size);
    transform_ = pp.transform;
    detail::profile_mark(device_, "preprocess");
    Tensor px = (device_ == brotensor::Device::CPU) ? pp.pixels
                                                     : pp.pixels.to(device_);
    image_embedding_ = enc_.encode(px);   // (1, D*grid*grid) on device_
    detail::profile_mark(device_, "encode");
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

    const int num = dm.num_out;
    Segmentation seg;
    seg.num    = num;
    seg.height = transform_.orig_h;
    seg.width  = transform_.orig_w;
    seg.logits = upscale_masks(dm.masks, num, dm.mask_size);

    Tensor iou_cpu = dm.iou.to(brotensor::Device::CPU);   // (num,1)
    const float* iv = iou_cpu.host_f32();
    seg.iou.assign(iv, iv + num);

    return seg;
}

// ─── Batched single-point segment ─────────────────────────────────────────────

std::vector<Segmentation> Sam::segment_points(
    const std::vector<std::array<float, 2>>& points,
    bool multimask_output) const {
    if (!has_image_) fail("segment_points() called before set_image()");
    std::vector<Segmentation> out;
    const int B = static_cast<int>(points.size());
    if (B == 0) return out;

    // Encode each one-foreground-point prompt; collect the per-prompt sparse
    // tokens (uniform count — point + its padding token) and the shared no-mask
    // dense embedding (identical for every prompt, so we keep just the first).
    detail::profile_mark(device_, nullptr);
    std::vector<Tensor> sparse_hosts;
    sparse_hosts.reserve(static_cast<std::size_t>(B));
    Tensor dense;
    int S = -1;
    for (const auto& p : points) {
        PromptInput in;
        float mx, my;
        apply_coords(transform_, p[0], p[1], mx, my);
        in.points.push_back({mx, my});
        in.labels.push_back(1);
        PromptEmbeddings emb = pe_.encode(in);
        const int s = (emb.sparse.data && emb.sparse.size() > 0) ? emb.sparse.rows : 0;
        if (S < 0) S = s;
        else if (s != S) fail("segment_points: non-uniform sparse token count");
        sparse_hosts.push_back(emb.sparse.to(brotensor::Device::CPU));
        if (!dense.data) dense = emb.dense;   // shared no-mask dense embedding
    }
    detail::profile_mark(device_, "prompt encode");

    const int D = cfg_.prompt.hidden_size;
    Tensor sparse_batched;
    if (S > 0) {
        Tensor sb = Tensor::mat(B * S, D);
        float* dst = sb.host_f32_mut();
        for (int b = 0; b < B; ++b) {
            const float* sp = sparse_hosts[static_cast<std::size_t>(b)].host_f32();
            std::copy(sp, sp + static_cast<std::size_t>(S) * D,
                      dst + static_cast<std::size_t>(b) * S * D);
        }
        sparse_batched = sb.to(device_);
    }

    DecodedMasks dm = dec_.decode_batched(image_embedding_, pe_.dense_pe(),
                                          sparse_batched, B, S, dense,
                                          multimask_output);
    detail::profile_mark(device_, "decode batched");

    const int num = dm.num_out;
    std::vector<float> up = upscale_masks(dm.masks, B * num, dm.mask_size);
    detail::profile_mark(device_, "upscale masks");

    Tensor iou_cpu = dm.iou.to(brotensor::Device::CPU);   // (B*num,1)
    const float* iv = iou_cpu.host_f32();
    const int oh = transform_.orig_h, ow = transform_.orig_w;
    const std::size_t plane = static_cast<std::size_t>(oh) * ow;

    out.reserve(static_cast<std::size_t>(B));
    for (int b = 0; b < B; ++b) {
        Segmentation seg;
        seg.num    = num;
        seg.height = oh;
        seg.width  = ow;
        const float* base = up.data() + static_cast<std::size_t>(b) * num * plane;
        seg.logits.assign(base, base + static_cast<std::size_t>(num) * plane);
        seg.iou.assign(iv + static_cast<std::size_t>(b) * num,
                       iv + static_cast<std::size_t>(b + 1) * num);
        out.push_back(std::move(seg));
    }
    return out;
}

// ─── Mask upscale (shared postprocess) ────────────────────────────────────────
//
// SAM's mask postprocess: bilinear-upscale each (mask_size, mask_size) logit
// plane to the model's square input size, crop off the letterbox padding (the
// resized content occupies the top-left resized_h x resized_w), then resize to
// the original image size. Both resamples are bilinear / align_corners=False.

std::vector<float> Sam::upscale_masks(const brotensor::Tensor& masks,
                                      int num, int mask_size) const {
    const int ms = mask_size;
    const int S  = cfg_.encoder.img_size;
    const int rh = transform_.resized_h, rw = transform_.resized_w;
    const int oh = transform_.orig_h,    ow = transform_.orig_w;

    // All three steps run on the masks' device (upscale -> de-letterbox crop
    // -> resize to original); only the final full-resolution logits are
    // downloaded. Under the automatic mask generator this is per point-batch
    // hot path. (num, ms*ms) row-major is already NCHW (1, num, ms, ms).
    Tensor in0 = Tensor::view(masks.device, masks.data, 1, num * ms * ms);

    Tensor up1;
    brotensor::interp2d_forward(in0, 1, num, ms, ms, S, S, /*bilinear=*/1, up1);

    // Crop each plane's top-left content region (the rest is padding).
    Tensor cropped;
    brotensor::slice2d_forward(up1, 1, num, S, S, /*h0=*/0, /*w0=*/0,
                               rh, rw, cropped);

    Tensor up2;
    brotensor::interp2d_forward(cropped, 1, num, rh, rw, oh, ow, /*bilinear=*/1, up2);

    Tensor host = (up2.device == brotensor::Device::CPU)
                      ? up2 : up2.to(brotensor::Device::CPU);
    const float* logit = host.host_f32();
    return std::vector<float>(
        logit, logit + static_cast<std::size_t>(num) * oh * ow);
}

}  // namespace brovisionml::sam
