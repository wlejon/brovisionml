#include "brovisionml/dsine.h"

#include "brotensor/tensor.h"

#include <utility>

namespace brovisionml::dsine {

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
}

// ─── Estimate ─────────────────────────────────────────────────────────────────

NormalMap NormalEstimator::run(const PreprocessedImage& pp) const {
    const int padH = pp.transform.pad_h;
    const int padW = pp.transform.pad_w;

    // encoder -> decoder -> refiner. All host-side FP32 for parity.
    EncoderTaps taps = encoder_.forward(pp.pixels, padH, padW);
    DecoderOutput dout = decoder_.forward(taps, pp.intrins, padH, padW);
    brotensor::Tensor normals =
        refiner_.forward(dout, pp.intrins, padH, padW, pp.transform);

    const int W = pp.transform.orig_w;
    const int H = pp.transform.orig_h;
    NormalMap out;
    out.width  = W;
    out.height = H;
    const float* n = normals.host_f32();
    out.normals.assign(n, n + static_cast<std::size_t>(3) * H * W);
    return out;
}

NormalMap NormalEstimator::estimate(const uint8_t* rgb, int w, int h,
                                    int channels) const {
    PreprocessedImage pp = preprocess(rgb, w, h, channels, cfg_.fov_deg);
    return run(pp);
}

NormalMap NormalEstimator::estimate(const uint8_t* rgb, int w, int h,
                                    int channels, float fx, float fy,
                                    float cx, float cy) const {
    // Preprocess for the pixel padding + transform, then override the synthesized
    // intrinsics with the caller's. The principal point is given on the ORIGINAL
    // unpadded image (pre-"+0.5"); shift it by the pad offset so it stays aligned
    // to content after padding — exactly what `preprocess` does internally for the
    // FOV-synthesized intrinsics.
    PreprocessedImage pp = preprocess(rgb, w, h, channels, cfg_.fov_deg);
    pp.intrins.fx = fx;
    pp.intrins.fy = fy;
    pp.intrins.cx = cx + static_cast<float>(pp.transform.pad_l);
    pp.intrins.cy = cy + static_cast<float>(pp.transform.pad_t);
    return run(pp);
}

}  // namespace brovisionml::dsine
