#pragma once

// SAM 2 image encoder: the Hiera trunk plus the FPN neck (HF Sam2VisionModel),
// with the mask decoder's two high-resolution projections (conv_s0 / conv_s1)
// folded in the way SAM 2 runs them — once per image rather than once per click.
//
//   patch embed (conv 7x7 s4) + windowed abs pos embed (bicubic background +
//   tiled window table)
//     -> 4 stages of multiscale blocks: LayerNorm -> (windowed | global)
//        multi-head attention, the first block of stages 1..3 max-pooling its
//        queries 2x (and its residual through a linear + the same pool)
//        -> residual -> LayerNorm -> GELU MLP -> residual
//     -> FPN: 1x1 laterals, nearest top-down only into the stride-16 level
//
// Output for a 1024 input: the stride-16 map that conditions the decoder and
// the memory (256 x 64 x 64, as tokens), and the stride-4 / stride-8 maps the
// decoder's upscaler adds back in (32 x 256 x 256 and 64 x 128 x 128, NCHW).
// One implementation serves tiny / small / base-plus / large; the Sam2Config
// read from config.json carries the differences.

#include "brovisionml/sam2_config.h"

#include "brotensor/tensor.h"

#include <memory>
#include <string>

namespace brotensor::safetensors { class File; }

namespace brovisionml::sam2 {

struct ImageFeatures {
    brotensor::Tensor top;     // (grid*grid, hidden) FP32 tokens, row = y*grid + x
    brotensor::Tensor feat_s0; // (1, hidden/8 * (4g)^2) FP32 NCHW, conv_s0 applied
    brotensor::Tensor feat_s1; // (1, hidden/4 * (2g)^2) FP32 NCHW, conv_s1 applied
    bool empty() const { return top.data == nullptr; }
};

struct ImageEncoderWeights;

class ImageEncoder {
public:
    explicit ImageEncoder(const Sam2Config& cfg);
    ~ImageEncoder();
    ImageEncoder(ImageEncoder&&) noexcept;
    ImageEncoder& operator=(ImageEncoder&&) noexcept;
    ImageEncoder(const ImageEncoder&) = delete;
    ImageEncoder& operator=(const ImageEncoder&) = delete;

    // Reads vision_encoder.* and mask_decoder.conv_s{0,1}.* from a HF SAM 2
    // checkpoint. Throws naming the first missing / mis-shaped tensor.
    void load(const brotensor::safetensors::File& f);

    // Migrate onto `dev`. On a GPU whose compute dtype is FP16 the GEMM
    // weights go FP16 (mixed precision); under a CPU DeviceScope they stay FP32.
    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }
    bool mixed_precision() const { return fp16_; }

    // `pixels`: (1, 3*S*S) FP32 NCHW, normalized, on device().
    ImageFeatures encode(const brotensor::Tensor& pixels) const;

    // Sine position encoding of the stride-16 map (grid*grid, hidden) FP32 on
    // device() — what the memory attention adds to the current frame.
    const brotensor::Tensor& top_pos() const;

private:
    Sam2Config cfg_;
    std::unique_ptr<ImageEncoderWeights> w_;
    brotensor::Device device_ = brotensor::Device::CPU;
    bool fp16_ = false;
};

// The DETR-style normalized sine position encoding SAM 2 uses for its FPN
// levels and its memories: (H*W, 2*npf) host FP32 tokens, the first npf
// channels encoding y, the next npf encoding x.
brotensor::Tensor sine_position_tokens(int H, int W, int npf);

}  // namespace brovisionml::sam2
