#include "brovisionml/depth_anything.h"

#include "brovisionml/dpt_preprocess.h"

#include "brotensor/ops.h"

#include "profile.h"

#include <utility>

namespace brovisionml::depth {

// ─── Config presets ─────────────────────────────────────────────────────────

DepthAnythingConfig DepthAnythingConfig::v2_small() {
    DepthAnythingConfig c;
    c.backbone = dinov2::Config::vit_s();
    c.head     = dpt::HeadConfig::vit_s();
    return c;
}

DepthAnythingConfig DepthAnythingConfig::v2_base() {
    DepthAnythingConfig c;
    c.backbone = dinov2::Config::vit_b();
    c.head     = dpt::HeadConfig::vit_b();
    return c;
}

DepthAnythingConfig DepthAnythingConfig::v2_large() {
    DepthAnythingConfig c;
    c.backbone = dinov2::Config::vit_l();
    c.head     = dpt::HeadConfig::vit_l();
    return c;
}

// ─── Construction / loading / migration ──────────────────────────────────────

DepthEstimator::DepthEstimator(DepthAnythingConfig cfg)
    : cfg_(std::move(cfg)), backbone_(cfg_.backbone), head_(cfg_.head) {}

DepthEstimator::~DepthEstimator() = default;
DepthEstimator::DepthEstimator(DepthEstimator&&) noexcept = default;
DepthEstimator& DepthEstimator::operator=(DepthEstimator&&) noexcept = default;

void DepthEstimator::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void DepthEstimator::load_file(const std::string& path) {
    // One HF DepthAnythingForDepthEstimation checkpoint carries both the
    // `backbone.` and the `neck.`/`head.` tensor namespaces.
    backbone_.load_file(path);
    head_.load_file(path);
}

void DepthEstimator::to(brotensor::Device dev) {
    if (dev == device_) return;
    backbone_.to(dev);
    head_.to(dev);
    device_ = dev;
}

// ─── Estimate ─────────────────────────────────────────────────────────────────

DepthMap DepthEstimator::estimate(const uint8_t* rgb, int w, int h,
                                  int channels) const {
    detail::profile_mark(device_, nullptr);
    // 1. Preprocess (host) -> upload.
    dpt::PreprocessedImage pp = dpt::preprocess(
        rgb, w, h, channels, cfg_.input_size, cfg_.multiple, cfg_.keep_aspect_ratio);
    const int rh = pp.transform.resized_h, rw = pp.transform.resized_w;
    brotensor::Tensor px = (device_ == brotensor::Device::CPU)
                               ? pp.pixels : pp.pixels.to(device_);
    detail::profile_mark(device_, "preprocess");

    // 2. Backbone -> stage feature maps; 3. DPT head -> depth at model res.
    dinov2::BackboneOutput bo = backbone_.encode(px, rh, rw);
    detail::profile_mark(device_, "backbone");
    brotensor::Tensor depth_model =
        head_.forward(bo.feature_maps, bo.patch_h, bo.patch_w);  // (1, rh*rw)
    detail::profile_mark(device_, "dpt head");

    // 4. Resize back to the original image size (bilinear, align_corners=False —
    //    HF post_process_depth_estimation).
    brotensor::Tensor depth_full;
    brotensor::interp2d_forward(depth_model, /*N=*/1, /*C=*/1, rh, rw, h, w,
                                /*bilinear=*/1, depth_full);
    detail::profile_mark(device_, "resize+download");

    brotensor::Tensor host = depth_full.to(brotensor::Device::CPU);
    const float* d = host.host_f32();
    DepthMap out;
    out.width  = w;
    out.height = h;
    out.depth.assign(d, d + static_cast<std::size_t>(w) * h);
    return out;
}

}  // namespace brovisionml::depth
