#pragma once

// Lineart — line-drawing extraction, end to end. The ControlNet "lineart"
// conditioning annotator: lllyasviel's "Informative Drawings" generator (Chan
// et al., CVPR'22), a small image-to-image CNN that maps a photo to a dense
// single-channel line map. Architecture is Generator(3, 1, 3):
//
//   model0: ReflectionPad(3) -> 7x7 conv (3->64) -> InstanceNorm -> ReLU
//   model1: 2x [ 3x3 stride-2 conv -> InstanceNorm -> ReLU ]   (64->128->256)
//   model2: 3x residual block [ refl-pad1, 3x3 conv, IN, ReLU,
//                               refl-pad1, 3x3 conv, IN ; + skip ]  (256)
//   model3: 2x [ 3x3 stride-2 conv-transpose -> InstanceNorm -> ReLU ] (256->128->64)
//   model4: ReflectionPad(3) -> 7x7 conv (64->1) -> Sigmoid
//
// Every norm is InstanceNorm2d with affine=False — no learnable scale/shift, so
// the checkpoint holds only the conv weights/biases. Instance norm is realized
// as a group norm with num_groups == channels (gamma=1, beta=0). The forward is
// a pure composition of brotensor ops (pad2d reflect / conv2d / group_norm /
// relu / conv_transpose2d / sigmoid) with no lineart-specific kernel.
//
//   detect(): preprocess (optional resize to a working resolution, pack to RGB
//   FP32 in [0,1]) -> generator -> resize the line map back to the original
//   input size.
//
// Output: a dense line map at the ORIGINAL input resolution, FP32 in [0,1]. The
// raw generator output is "bright background, dark lines"; `invert` (the default,
// matching ControlNet's annotator) flips it to "dark background, bright lines"
// so it visualizes and conditions like the other edge maps.
//
// Device: like DepthEstimator/DSINE/HED, call to(Device) after load() to run on
// a device. detect() preprocesses on the host, uploads the pixels, runs the
// generator on-device, and copies the line map back to the host. Default CPU.

#include "brovisionml/lineart_preprocess.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brovisionml::lineart {

struct LineartConfig {
    // Longer-side target for the working resolution; 0 = run at native size.
    // ControlNet's lineart preprocessor runs the detector at a fixed resolution
    // (often 512) and resizes the line map back afterwards — set this to match.
    int detect_resolution = 0;

    // Invert the raw generator output (1 - line). The model emits a bright
    // field with dark lines; inverting yields bright lines on a dark field, the
    // convention ControlNet feeds downstream. Defaults to true.
    bool invert = true;
};

// A dense line map at the original image resolution. `line` holds height*width
// FP32 in row-major (H,W) order, each value in [0,1].
struct LineMap {
    int width  = 0;           // == original image width
    int height = 0;           // == original image height
    std::vector<float> line;  // height*width, row-major, [0,1]

    float at(int x, int y) const {
        return line[static_cast<std::size_t>(y) * width + x];
    }
};

class LineartDetector {
public:
    explicit LineartDetector(LineartConfig cfg = {});
    ~LineartDetector();
    LineartDetector(LineartDetector&&) noexcept;
    LineartDetector& operator=(LineartDetector&&) noexcept;

    // Load the generator from a checkpoint directory / file (one lineart
    // `model.safetensors` carrying `model0.*`, `model1.*`, `model2.*`,
    // `model3.*`, `model4.*` conv / conv-transpose weights + biases).
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Migrate the network to `dev` (no-op if already there). detect() then runs
    // on `dev` and returns a host-resident LineMap.
    void to(brotensor::Device dev);
    brotensor::Device device() const;

    const LineartConfig& config() const { return cfg_; }

    // Detect line drawing for an 8-bit interleaved HWC image (channels 1/3/4).
    LineMap detect(const uint8_t* rgb, int w, int h, int channels) const;

private:
    LineMap run(const PreprocessedImage& pp) const;

    LineartConfig cfg_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brovisionml::lineart
