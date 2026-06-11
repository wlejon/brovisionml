#include "brovisionml/dsine.h"

#include "brotensor/tensor.h"

#include "broimage/geometric.h"

#include "profile.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace brovisionml::dsine {

namespace {

// Resized dims with the longer side capped to `cap`, aspect preserved.
void capped_dims(int w, int h, int cap, int& rw, int& rh) {
    const double s = static_cast<double>(cap) / std::max(w, h);
    rw = std::max(1, static_cast<int>(std::lround(w * s)));
    rh = std::max(1, static_cast<int>(std::lround(h * s)));
}

// Re-normalize each pixel's (nx,ny,nz) to unit length. Planar (3,H,W); upscaling
// the normal map interpolates components independently, so the result drifts off
// the unit sphere — this puts it back.
void renormalize_unit(std::vector<float>& n, int w, int h) {
    const std::size_t plane = static_cast<std::size_t>(w) * h;
    for (std::size_t i = 0; i < plane; ++i) {
        const float x = n[i], y = n[plane + i], z = n[2 * plane + i];
        const float len = std::sqrt(x * x + y * y + z * z);
        if (len > 1e-12f) {
            const float inv = 1.0f / len;
            n[i] = x * inv; n[plane + i] = y * inv; n[2 * plane + i] = z * inv;
        }
    }
}

}  // namespace

// ─── Construction / loading ──────────────────────────────────────────────────

NormalEstimator::NormalEstimator(DsineConfig cfg) : cfg_(std::move(cfg)) {}

NormalEstimator::~NormalEstimator() = default;
NormalEstimator::NormalEstimator(NormalEstimator&&) noexcept = default;
NormalEstimator& NormalEstimator::operator=(NormalEstimator&&) noexcept = default;

void NormalEstimator::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void NormalEstimator::load_file(const std::string& path) {
    // One DSINE_v02 checkpoint carries the encoder (`encoder.original_model.*`),
    // the decoder (`decoder.*`), and the refinement (`gru.*`, `*_head.*`)
    // tensor namespaces.
    encoder_.load_file(path);
    decoder_.load_file(path);
    refiner_.load_file(path);
    if (device_ != brotensor::Device::CPU) {
        encoder_.to(device_);
        decoder_.to(device_);
        refiner_.to(device_);
    }
}

void NormalEstimator::to(brotensor::Device dev) {
    if (dev == device_) return;
    encoder_.to(dev);
    decoder_.to(dev);
    refiner_.to(dev);
    device_ = dev;
}

// ─── Estimate ─────────────────────────────────────────────────────────────────

NormalMap NormalEstimator::run(const PreprocessedImage& pp) const {
    const int padH = pp.transform.pad_h;
    const int padW = pp.transform.pad_w;

    // Upload the host-preprocessed pixels to the active device, then run
    // encoder -> decoder -> refiner on-device (CPU FP32 or CUDA FP32).
    detail::profile_mark(device_, nullptr);
    brotensor::Tensor px = (device_ == brotensor::Device::CPU)
                               ? pp.pixels
                               : pp.pixels.to(device_);
    EncoderTaps taps = encoder_.forward(px, padH, padW);
    detail::profile_mark(device_, "encoder");
    DecoderOutput dout = decoder_.forward(taps, pp.intrins, padH, padW);
    detail::profile_mark(device_, "decoder");
    brotensor::Tensor normals =
        refiner_.forward(dout, pp.intrins, padH, padW, pp.transform);
    detail::profile_mark(device_, "refine");

    // Pull the final normal map back to the host for the caller.
    brotensor::Tensor host = (normals.device == brotensor::Device::CPU)
                                 ? normals
                                 : normals.to(brotensor::Device::CPU);

    const int W = pp.transform.orig_w;
    const int H = pp.transform.orig_h;
    NormalMap out;
    out.width  = W;
    out.height = H;
    const float* n = host.host_f32();
    out.normals.assign(n, n + static_cast<std::size_t>(3) * H * W);
    return out;
}

// Run at a capped working resolution: downscale the input, run, then upscale the
// normal map back to (w,h) and re-normalize. `scale_intrins`, if set, receives
// the resize factor (rw/w, rh/h) so an explicit-intrinsics caller can scale its
// intrinsics to the downscaled image.
NormalMap NormalEstimator::run_capped(const uint8_t* rgb, int w, int h,
                                      int channels,
                                      const std::function<void(PreprocessedImage&,
                                                               float, float)>&
                                          set_intrins) const {
    int rw = 0, rh = 0;
    capped_dims(w, h, cfg_.max_resolution, rw, rh);
    std::vector<uint8_t> small(static_cast<std::size_t>(rw) * rh * channels);
    broimage::resize_hwc_u8(rgb, w, h, channels, small.data(), rw, rh,
                            broimage::Filter::Area);

    PreprocessedImage pp = preprocess(small.data(), rw, rh, channels, cfg_.fov_deg);
    if (set_intrins)
        set_intrins(pp, static_cast<float>(rw) / w, static_cast<float>(rh) / h);
    NormalMap small_map = run(pp);   // rw × rh

    NormalMap out;
    out.width  = w;
    out.height = h;
    out.normals.resize(static_cast<std::size_t>(3) * w * h);
    broimage::resize_chw_f32(small_map.normals.data(), rw, rh, /*channels=*/3,
                             out.normals.data(), w, h, broimage::Filter::Bilinear);
    renormalize_unit(out.normals, w, h);
    return out;
}

NormalMap NormalEstimator::estimate(const uint8_t* rgb, int w, int h,
                                    int channels) const {
    if (cfg_.max_resolution > 0 && std::max(w, h) > cfg_.max_resolution)
        return run_capped(rgb, w, h, channels, /*set_intrins=*/{});
    PreprocessedImage pp = preprocess(rgb, w, h, channels, cfg_.fov_deg);
    return run(pp);
}

NormalMap NormalEstimator::estimate(const uint8_t* rgb, int w, int h,
                                    int channels, float fx, float fy,
                                    float cx, float cy) const {
    // The principal point is given on the ORIGINAL unpadded image (pre-"+0.5");
    // shift it by the pad offset so it stays aligned to content after padding —
    // exactly what `preprocess` does internally for the FOV-synthesized intrinsics.
    // When capped, the intrinsics also scale by the downscale factor.
    auto set_intrins = [&](PreprocessedImage& pp, float sx, float sy) {
        pp.intrins.fx = fx * sx;
        pp.intrins.fy = fy * sy;
        pp.intrins.cx = cx * sx + static_cast<float>(pp.transform.pad_l);
        pp.intrins.cy = cy * sy + static_cast<float>(pp.transform.pad_t);
    };
    if (cfg_.max_resolution > 0 && std::max(w, h) > cfg_.max_resolution)
        return run_capped(rgb, w, h, channels, set_intrins);

    PreprocessedImage pp = preprocess(rgb, w, h, channels, cfg_.fov_deg);
    set_intrins(pp, 1.0f, 1.0f);
    return run(pp);
}

}  // namespace brovisionml::dsine
