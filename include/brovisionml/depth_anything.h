#pragma once

// Depth-Anything-V2 — monocular relative-depth estimation, end to end. Ties the
// DPT preprocessing, the DINOv2 backbone, and the DPT depth head into a single
// estimator that loads one HF `DepthAnythingForDepthEstimation` safetensors
// checkpoint and maps an image to a dense depth map at the original resolution.
//
//   estimate(): preprocess (aspect-preserving resize to a multiple of 14,
//   ImageNet normalize) -> DINOv2 encode -> DPT head -> bilinear resize back to
//   the input size.
//
// The output is *relative* inverse-depth in Depth-Anything's convention: larger
// values are nearer. It is not metric; callers normalize for visualization.

#include "brovisionml/dinov2.h"
#include "brovisionml/dpt_head.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brovisionml::depth {

struct DepthAnythingConfig {
    dinov2::Config  backbone;
    dpt::HeadConfig head;
    int  input_size       = 518;   // DPT resize target (square box)
    int  multiple         = 14;    // patch size; output dims are multiples of it
    bool keep_aspect_ratio = true;

    static DepthAnythingConfig v2_small();
    static DepthAnythingConfig v2_base();
    static DepthAnythingConfig v2_large();
};

struct DepthMap {
    int width  = 0;             // == original image width
    int height = 0;             // == original image height
    std::vector<float> depth;   // row-major height*width, relative (nearer = larger)

    float at(int x, int y) const {
        return depth[static_cast<std::size_t>(y) * width + x];
    }
};

class DepthEstimator {
public:
    explicit DepthEstimator(DepthAnythingConfig cfg);
    ~DepthEstimator();
    DepthEstimator(DepthEstimator&&) noexcept;
    DepthEstimator& operator=(DepthEstimator&&) noexcept;

    // Load both sub-modules from one checkpoint directory / file.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }
    const DepthAnythingConfig& config() const { return cfg_; }

    // Estimate depth for an 8-bit interleaved HWC image (channels 1/3/4).
    DepthMap estimate(const uint8_t* rgb, int w, int h, int channels) const;

private:
    DepthAnythingConfig cfg_;
    brotensor::Device   device_ = brotensor::Device::CPU;
    dinov2::Backbone    backbone_;
    dpt::DepthHead      head_;
};

}  // namespace brovisionml::depth
