#pragma once

// HED — soft-edge estimation, end to end. The ControlNet "softedge" conditioning
// annotator: a Holistically-nested Edge Detection network that maps an image to
// a dense single-channel edge probability map. This wraps the self-contained
// "ControlNetHED" reimplementation lllyasviel ships (Apache-2 licensed): a
// VGG-style 5-block convolutional trunk where each block emits a 1-channel side
// output; the side maps are bilinear-resized to the working resolution,
// averaged, and passed through a sigmoid.
//
//   detect(): preprocess (optional resize to a working resolution, pack to RGB
//   FP32 in [0,255]) -> 5 conv blocks (block_k ends in a 1x1 projection to one
//   channel; blocks 2-5 max-pool 2x2 first) -> resize each side map to the
//   working resolution -> mean -> sigmoid -> resize the edge map back to the
//   original input size.
//
// The learned per-channel `norm` bias (the VGG mean, in the [0,255] scale) is
// folded into the first conv's bias at load time, so the forward pass is a pure
// composition of brotensor ops (conv2d / relu / max_pool2d / interp2d / sigmoid)
// with no HED-specific kernel.
//
// Output: a dense edge map at the ORIGINAL input resolution, FP32 in [0,1]
// (higher = stronger edge). Visualize directly as grayscale (edge*255).
//
// Device: like DepthEstimator/DSINE, call to(Device) after load() to run on a
// device. detect() preprocesses on the host, uploads the pixels to the active
// device, runs the trunk on-device, and copies the edge map back to the host.
// Default device is CPU.

#include "brovisionml/hed_preprocess.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brovisionml::hed {

struct HedConfig {
    // Longer-side target for the working resolution; 0 = run at native size.
    // ControlNet's softedge preprocessor runs HED at a fixed resolution (often
    // 512) and resizes the edge map back afterwards — set this to match.
    int detect_resolution = 0;
};

// A dense edge probability map at the original image resolution. `edge` holds
// height*width FP32 in row-major (H,W) order, each value in [0,1].
struct EdgeMap {
    int width  = 0;           // == original image width
    int height = 0;           // == original image height
    std::vector<float> edge;  // height*width, row-major, [0,1]

    float at(int x, int y) const {
        return edge[static_cast<std::size_t>(y) * width + x];
    }
};

class SoftEdgeDetector {
public:
    explicit SoftEdgeDetector(HedConfig cfg = {});
    ~SoftEdgeDetector();
    SoftEdgeDetector(SoftEdgeDetector&&) noexcept;
    SoftEdgeDetector& operator=(SoftEdgeDetector&&) noexcept;

    // Load the network from a checkpoint directory / file (one HED
    // `model.safetensors` carrying `norm`, `block{1..5}.convs.*`, and
    // `block{1..5}.projection.*`).
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Migrate the network to `dev` (no-op if already there). detect() then runs
    // on `dev` and returns a host-resident EdgeMap.
    void to(brotensor::Device dev);
    brotensor::Device device() const;

    const HedConfig& config() const { return cfg_; }

    // Detect soft edges for an 8-bit interleaved HWC image (channels 1/3/4).
    EdgeMap detect(const uint8_t* rgb, int w, int h, int channels) const;

private:
    EdgeMap run(const PreprocessedImage& pp) const;

    HedConfig cfg_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brovisionml::hed
