#pragma once

// SAM (Segment Anything) prompt encoder — turns user prompts (points, boxes,
// and an optional coarse mask) into the embeddings the mask decoder consumes.
// Port of segment_anything's PromptEncoder, matched against the HuggingFace
// `SamModel` checkpoint layout (the `prompt_encoder.*` tensors).
//
// It produces two things, exactly as SAM's decoder expects:
//
//   * sparse embeddings — one (hidden_size) token per point and per box corner,
//     each the sum of a random-Fourier positional encoding of the coordinate
//     and a learned per-type embedding (foreground / background point, box
//     top-left / bottom-right corner, or the "not a point" padding token).
//   * a dense embedding — a (hidden_size, grid, grid) map. With no mask prompt
//     it is the learned `no_mask_embed` broadcast over the grid; with a mask
//     prompt it is that mask run through a small strided-conv downscaler.
//
// It also exposes the image-grid positional encoding (`dense_pe()`), the
// `get_dense_pe()` of upstream SAM — the decoder adds it to the image keys.
//
// Coordinates are in MODEL-INPUT space: the 1024-square the image encoder sees,
// i.e. original-image coords already run through sam::apply_coords. The +0.5
// pixel-center shift SAM applies is done here, internally.
//
// The positional encoding (a matmul by a fixed Gaussian matrix followed by
// sin/cos) and the sparse-token assembly are tiny and inherently host-side, so
// they run on the host in FP32; the assembled sparse / dense / dense-pe tensors
// are then migrated to device() so they meet the image embedding on the same
// device the mask decoder runs on. A mask-prompt's conv downscaler runs through
// brotensor ops on device(), like the rest of the model.

#include "brotensor/tensor.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace brovisionml::sam {

// Prompt-encoder hyperparameters. Defaults mirror HuggingFace's
// SamPromptEncoderConfig for the stock facebook/sam-* checkpoints (all three
// ViT sizes share the same 256-d prompt encoder).
struct PromptEncoderConfig {
    int hidden_size         = 256;   // embedding / token dim
    int image_embedding_size = 64;   // image-feature grid side (== encoder grid)
    int input_image_size    = 1024;  // model input square (coords are in here)
    int mask_input_channels = 16;    // mask downscaler hidden width
    int num_point_embeddings = 4;    // {bg point, fg point, box TL, box BR}

    int num_pos_feats() const { return hidden_size / 2; }  // 128; sin+cos -> 256
    int grid()         const { return image_embedding_size; }
};

// A single decode's prompts, in model-input (1024-square) coordinates.
//   points/labels : N click points; label 1 = foreground, 0 = background,
//                   -1 = explicit "not a point" padding token.
//   boxes         : M boxes as {x1, y1, x2, y2}; each contributes two corner
//                   tokens.
//   mask / mask_h / mask_w : optional coarse mask logits, a single-channel
//                   (mask_h x mask_w) buffer (stock SAM uses 256x256, i.e.
//                   4x the feature grid). Empty `mask` selects the no-mask path.
struct PromptInput {
    std::vector<std::array<float, 2>> points;
    std::vector<int>                  labels;
    std::vector<std::array<float, 4>> boxes;
    std::vector<float>                mask;
    int                               mask_h = 0;
    int                               mask_w = 0;
};

struct PromptEmbeddings {
    // (num_sparse_tokens, hidden_size) — points (with any pad token) then box
    // corners. num_sparse_tokens is 0 only when there are no prompts at all.
    brotensor::Tensor sparse;
    // (1, hidden_size*grid*grid) FP32 NCHW dense embedding.
    brotensor::Tensor dense;
};

struct PromptEncoderWeights;  // defined in the .cpp; held opaque

class PromptEncoder {
public:
    explicit PromptEncoder(PromptEncoderConfig cfg);
    ~PromptEncoder();

    PromptEncoder(PromptEncoder&&) noexcept;
    PromptEncoder& operator=(PromptEncoder&&) noexcept;
    PromptEncoder(const PromptEncoder&)            = delete;
    PromptEncoder& operator=(const PromptEncoder&) = delete;

    // Load a HuggingFace SAM checkpoint. `load` expects `dir/model.safetensors`;
    // `load_file` reads one explicit path. Both read the `prompt_encoder.*`
    // tensors and throw std::runtime_error (prefix "sam::PromptEncoder: ")
    // naming the first missing / mismatched tensor.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Migrate the device-resident outputs onto `dev`. encode()/dense_pe() then
    // return tensors on `dev`. No-op if already there. Throws before load().
    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }

    // The image-grid positional encoding, (1, hidden_size*grid*grid) NCHW on
    // device() — upstream SAM's get_dense_pe(). The mask decoder adds it to the
    // image keys. Computed once at load() and migrated by to().
    brotensor::Tensor dense_pe() const;

    // Encode a prompt set into sparse + dense embeddings on device().
    PromptEmbeddings encode(const PromptInput& prompt) const;

    const PromptEncoderConfig& config() const { return cfg_; }

private:
    PromptEncoderConfig                   cfg_;
    std::unique_ptr<PromptEncoderWeights> w_;
    brotensor::Device                     device_ = brotensor::Device::CPU;
};

}  // namespace brovisionml::sam
