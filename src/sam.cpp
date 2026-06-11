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
    bool multimask_output, float min_iou) const {
    if (!has_image_) fail("segment_points() called before set_image()");
    std::vector<Segmentation> out;
    const int B = static_cast<int>(points.size());
    if (B == 0) return out;

    // One batched prompt encode for the whole grid: every prompt is a single
    // foreground click, S=2 sparse tokens (point + padding), shared no-mask
    // dense embedding, one upload each.
    detail::profile_mark(device_, nullptr);
    std::vector<std::array<float, 2>> mapped(static_cast<std::size_t>(B));
    for (int b = 0; b < B; ++b)
        apply_coords(transform_, points[static_cast<std::size_t>(b)][0],
                     points[static_cast<std::size_t>(b)][1],
                     mapped[static_cast<std::size_t>(b)][0],
                     mapped[static_cast<std::size_t>(b)][1]);
    PromptEmbeddings emb = pe_.encode_point_batch(mapped);
    const int S = 2;
    detail::profile_mark(device_, "prompt encode");

    DecodedMasks dm = dec_.decode_batched(image_embedding_, pe_.dense_pe(),
                                          emb.sparse, B, S, emb.dense,
                                          multimask_output);
    detail::profile_mark(device_, "decode batched");

    const int num = dm.num_out;
    Tensor iou_cpu = dm.iou.to(brotensor::Device::CPU);   // (B*num, 1) — tiny
    const float* iv = iou_cpu.host_f32();

    // Predicted-IoU pre-filter: select surviving mask rows BEFORE the
    // full-resolution upscale, so rejected masks are never upscaled or
    // downloaded. keep stays sorted, so survivors group by prompt below.
    std::vector<int32_t> keep;
    keep.reserve(static_cast<std::size_t>(B) * num);
    for (int i = 0; i < B * num; ++i)
        if (min_iou < 0.0f || iv[i] > min_iou)
            keep.push_back(i);
    const int n_keep = static_cast<int>(keep.size());

    std::vector<float> up;
    if (n_keep == B * num) {
        up = upscale_masks(dm.masks, B * num, dm.mask_size);
    } else if (n_keep > 0) {
        Tensor idx_h = Tensor::zeros_on(brotensor::Device::CPU, n_keep, 1,
                                        brotensor::Dtype::INT32);
        int32_t* ip = static_cast<int32_t*>(idx_h.data);
        std::copy(keep.begin(), keep.end(), ip);
        Tensor idx = (device_ == brotensor::Device::CPU) ? idx_h
                                                         : idx_h.to(device_);
        Tensor kept_masks;
        brotensor::gather_rows(dm.masks, idx, kept_masks);
        up = upscale_masks(kept_masks, n_keep, dm.mask_size);
    }
    detail::profile_mark(device_, "upscale masks");

    const int oh = transform_.orig_h, ow = transform_.orig_w;
    const std::size_t plane = static_cast<std::size_t>(oh) * ow;

    out.reserve(static_cast<std::size_t>(B));
    std::size_t ki = 0;   // cursor into keep / up, grouped by prompt
    for (int b = 0; b < B; ++b) {
        Segmentation seg;
        seg.height = oh;
        seg.width  = ow;
        while (ki < keep.size() && keep[ki] < (b + 1) * num) {
            const float* base = up.data() + ki * plane;
            seg.logits.insert(seg.logits.end(), base, base + plane);
            seg.iou.push_back(iv[keep[ki]]);
            ++seg.num;
            ++ki;
        }
        out.push_back(std::move(seg));
    }
    return out;
}

// ─── Batched binary segment (AMG hot path) ────────────────────────────────────

std::vector<BinarySegmentation> Sam::segment_points_binary(
    const std::vector<std::array<float, 2>>& points,
    bool multimask_output, float min_iou,
    float stability_offset, float min_stability) const {
    if (!has_image_) fail("segment_points_binary() called before set_image()");
    std::vector<BinarySegmentation> out;
    const int B = static_cast<int>(points.size());
    if (B == 0) return out;

    detail::profile_mark(device_, nullptr);
    std::vector<std::array<float, 2>> mapped(static_cast<std::size_t>(B));
    for (int b = 0; b < B; ++b)
        apply_coords(transform_, points[static_cast<std::size_t>(b)][0],
                     points[static_cast<std::size_t>(b)][1],
                     mapped[static_cast<std::size_t>(b)][0],
                     mapped[static_cast<std::size_t>(b)][1]);
    PromptEmbeddings emb = pe_.encode_point_batch(mapped);
    const int S = 2;
    detail::profile_mark(device_, "prompt encode");

    DecodedMasks dm = dec_.decode_batched(image_embedding_, pe_.dense_pe(),
                                          emb.sparse, B, S, emb.dense,
                                          multimask_output);
    detail::profile_mark(device_, "decode batched");

    const int num = dm.num_out;
    Tensor iou_cpu = dm.iou.to(brotensor::Device::CPU);   // (B*num, 1) — tiny
    const float* iv = iou_cpu.host_f32();

    // Predicted-IoU pre-filter (same as segment_points): survivors only ever
    // reach the upscale. keep stays sorted, so survivors group by prompt.
    std::vector<int32_t> keep;
    keep.reserve(static_cast<std::size_t>(B) * num);
    for (int i = 0; i < B * num; ++i)
        if (min_iou < 0.0f || iv[i] > min_iou)
            keep.push_back(i);
    const int n1 = static_cast<int>(keep.size());

    const int oh = transform_.orig_h, ow = transform_.orig_w;
    const std::size_t plane = static_cast<std::size_t>(oh) * ow;

    auto upload_idx = [&](const std::vector<int32_t>& rows) {
        Tensor idx_h = Tensor::zeros_on(brotensor::Device::CPU,
                                        static_cast<int>(rows.size()), 1,
                                        brotensor::Dtype::INT32);
        std::copy(rows.begin(), rows.end(), static_cast<int32_t*>(idx_h.data));
        return (device_ == brotensor::Device::CPU) ? idx_h : idx_h.to(device_);
    };

    Tensor up;   // (1, n1*plane) full-resolution logits on device_
    if (n1 == B * num) {
        up = upscale_masks_dev(dm.masks, B * num, dm.mask_size);
    } else if (n1 > 0) {
        Tensor idx = upload_idx(keep);
        Tensor kept_masks;
        brotensor::gather_rows(dm.masks, idx, kept_masks);
        up = upscale_masks_dev(kept_masks, n1, dm.mask_size);
    }
    detail::profile_mark(device_, "upscale masks");

    // Stability score on device: per upscaled plane, count(logit > -offset)
    // (union) and count(logit > +offset) (intersection); only 2 INT32 per mask
    // cross the bus. Same comparisons on the same FP32 values as the old host
    // scan, so the kept set is bit-identical.
    std::vector<float>   stab(static_cast<std::size_t>(n1), 0.0f);
    std::vector<int32_t> keep2;   // local row indices into `up`
    keep2.reserve(static_cast<std::size_t>(n1));
    if (n1 > 0) {
        Tensor up_rows = Tensor::view(up.device, up.data, n1,
                                      static_cast<int>(plane));
        Tensor counts_dev;
        brotensor::rows_count_above(up_rows, -stability_offset,
                                    stability_offset, counts_dev);
        Tensor counts_h = counts_dev.to(brotensor::Device::CPU);  // (n1,2)
        const int32_t* cv = static_cast<const int32_t*>(counts_h.data);
        for (int i = 0; i < n1; ++i) {
            const int32_t uni = cv[2 * i], inter = cv[2 * i + 1];
            const float s = uni > 0
                                ? static_cast<float>(inter) /
                                      static_cast<float>(uni)
                                : 0.0f;
            stab[static_cast<std::size_t>(i)] = s;
            if (min_stability <= 0.0f || s >= min_stability)
                keep2.push_back(i);
        }
    }

    // Binarize the stability survivors at logit 0 on device and download 0/1
    // bytes — a quarter of the FP32 logit volume.
    const int n2 = static_cast<int>(keep2.size());
    Tensor bin_h;
    if (n2 > 0) {
        Tensor up_rows = Tensor::view(up.device, up.data, n1,
                                      static_cast<int>(plane));
        Tensor src = up_rows;
        if (n2 != n1) {
            Tensor idx = upload_idx(keep2);
            Tensor gathered;
            brotensor::gather_rows(up_rows, idx, gathered);
            src = std::move(gathered);
        }
        Tensor bin_dev;
        brotensor::threshold_u8(src, 0.0f, bin_dev);
        bin_h = (bin_dev.device == brotensor::Device::CPU)
                    ? bin_dev : bin_dev.to(brotensor::Device::CPU);
    }
    detail::profile_mark(device_, "stability + masks download");

    const uint8_t* bp =
        n2 > 0 ? static_cast<const uint8_t*>(bin_h.data) : nullptr;
    out.reserve(static_cast<std::size_t>(B));
    std::size_t ki = 0;   // cursor into keep2 / bin rows, grouped by prompt
    for (int b = 0; b < B; ++b) {
        BinarySegmentation seg;
        seg.height = oh;
        seg.width  = ow;
        while (ki < keep2.size() &&
               keep[static_cast<std::size_t>(keep2[ki])] < (b + 1) * num) {
            const uint8_t* base = bp + ki * plane;
            seg.masks.insert(seg.masks.end(), base, base + plane);
            seg.iou.push_back(iv[keep[static_cast<std::size_t>(keep2[ki])]]);
            seg.stability.push_back(stab[static_cast<std::size_t>(keep2[ki])]);
            ++seg.num;
            ++ki;
        }
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

brotensor::Tensor Sam::upscale_masks_dev(const brotensor::Tensor& masks,
                                         int num, int mask_size) const {
    const int ms = mask_size;
    const int S  = cfg_.encoder.img_size;
    const int rh = transform_.resized_h, rw = transform_.resized_w;
    const int oh = transform_.orig_h,    ow = transform_.orig_w;

    // All three steps run on the masks' device (upscale -> de-letterbox crop
    // -> resize to original). Under the automatic mask generator this is per
    // point-batch hot path. (num, ms*ms) row-major is already NCHW (1, num,
    // ms, ms).
    Tensor in0 = Tensor::view(masks.device, masks.data, 1, num * ms * ms);

    Tensor up1;
    brotensor::interp2d_forward(in0, 1, num, ms, ms, S, S, /*bilinear=*/1, up1);

    // Crop each plane's top-left content region (the rest is padding).
    Tensor cropped;
    brotensor::slice2d_forward(up1, 1, num, S, S, /*h0=*/0, /*w0=*/0,
                               rh, rw, cropped);

    Tensor up2;
    brotensor::interp2d_forward(cropped, 1, num, rh, rw, oh, ow, /*bilinear=*/1, up2);
    return up2;
}

std::vector<float> Sam::upscale_masks(const brotensor::Tensor& masks,
                                      int num, int mask_size) const {
    const int oh = transform_.orig_h, ow = transform_.orig_w;
    Tensor up2 = upscale_masks_dev(masks, num, mask_size);
    Tensor host = (up2.device == brotensor::Device::CPU)
                      ? up2 : up2.to(brotensor::Device::CPU);
    const float* logit = host.host_f32();
    return std::vector<float>(
        logit, logit + static_cast<std::size_t>(num) * oh * ow);
}

}  // namespace brovisionml::sam
