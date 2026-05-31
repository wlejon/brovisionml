#pragma once

// DPT depth head — the neck + head that turn a DINOv2 backbone's stage feature
// maps into a dense depth map, as in HF `DepthAnythingForDepthEstimation`
// (the `neck.` and `head.` namespaces). The pipeline:
//
//   reassemble : per stage, drop the cls token, reshape tokens to a (gh, gw)
//                feature map, 1x1-project to the stage channel width, then
//                resample by a per-stage factor [4, 2, 1, 0.5] (ConvTranspose2d
//                up / Identity / strided Conv2d down).
//   neck convs : a bias-free 3x3 conv maps each reassembled map to the shared
//                fusion width.
//   fusion     : a RefineNet feature-fusion stage — deepest-first, each layer
//                adds a pre-activation residual unit on the skip connection,
//                runs a second residual unit, bilinearly upsamples (×2, or to the
//                next stage's size) with align_corners=True, and 1x1-projects.
//   head       : conv -> align_corners bilinear upsample to (gh, gw)*patch_size
//                -> conv -> ReLU -> 1x1 conv -> ReLU, yielding a single-channel
//                relative-depth map at the model input resolution.
//
// The align_corners=True upsamples are why brotensor gained
// interp2d_align_corners_forward. Loads directly from an HF
// DepthAnythingForDepthEstimation safetensors checkpoint.

#include "brotensor/tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace brovisionml::dpt {

struct HeadConfig {
    int reassemble_hidden_size = 384;          // == backbone hidden_size
    std::vector<int> neck_hidden_sizes = {48, 96, 192, 384};
    std::vector<double> reassemble_factors = {4.0, 2.0, 1.0, 0.5};
    int fusion_hidden_size = 64;
    int head_hidden_size   = 32;
    int patch_size         = 14;

    static HeadConfig vit_s();  // Depth-Anything-V2-Small
    static HeadConfig vit_b();  // Depth-Anything-V2-Base
    static HeadConfig vit_l();  // Depth-Anything-V2-Large
};

class DepthHead {
public:
    explicit DepthHead(HeadConfig cfg);
    ~DepthHead();
    DepthHead(DepthHead&&) noexcept;
    DepthHead& operator=(DepthHead&&) noexcept;

    void load(const std::string& dir);
    void load_file(const std::string& path);

    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }
    const HeadConfig& config() const { return cfg_; }

    // feature_maps: one (1 + patch_h*patch_w, reassemble_hidden_size) map per
    // stage (backbone out order, cls in row 0), on this head's device.
    // Returns a (1, out_h*out_w) FP32 NCHW single-channel relative-depth map at
    // out_h = patch_h*patch_size, out_w = patch_w*patch_size.
    brotensor::Tensor forward(const std::vector<brotensor::Tensor>& feature_maps,
                              int patch_h, int patch_w) const;

private:
    HeadConfig cfg_;
    brotensor::Device device_ = brotensor::Device::CPU;
    struct Weights;
    std::unique_ptr<Weights> w_;
};

}  // namespace brovisionml::dpt
