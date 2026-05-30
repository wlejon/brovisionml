#pragma once

// SAM (Segment Anything) ViT image encoder — the heavy front half of SAM.
//
// Takes a preprocessed (1, 3*img_size*img_size) FP32 NCHW pixel tensor (see
// sam_preprocess.h) and produces the dense image embedding the prompt encoder
// and mask decoder consume: a (1, out_chans*grid*grid) FP32 NCHW feature map
// (256x64x64 for stock SAM). This is the "encode image once" half of SAM's
// encode-once / decode-many-prompts split.
//
// The architecture is a ViTDet-style hierarchical ViT (segment_anything's
// ImageEncoderViT, matched against the HuggingFace `SamModel` checkpoint
// layout):
//
//   patch_embed (conv 16x16 stride 16) -> + abs pos embed
//     -> depth x transformer block
//          (LayerNorm -> multi-head attention with a DECOMPOSED 2D relative-
//           position bias -> residual -> LayerNorm -> MLP -> residual)
//     -> neck (conv 1x1 -> LayerNorm2d -> conv 3x3 -> LayerNorm2d)
//
// Most blocks attend over local 14x14 windows; a handful (global_attn_indexes)
// attend globally over the full 64x64 grid. Every op is composed from brotensor
// primitives — the decomposed-rel-pos attention is `brotensor`'s
// self_attention_decomposed_rel_pos_forward, the exact SAM/ViTDet kernel.
//
// CPU/FP32 today. The windowed-attention path gathers per-window tokens with
// host loops, so encode() requires host-resident FP32 tensors; the brotensor
// ops it composes are device-neutral, so a GPU path only awaits a
// window-partition gather op in brotensor (the "missing kernel -> add it to
// brotensor" rule). Weights are loaded widened to host FP32 regardless of the
// checkpoint's on-disk dtype (F32 / F16 / BF16).

#include "brotensor/tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace brovisionml::sam {

// ViT image-encoder hyperparameters. Defaults are ViT-H (facebook/sam-vit-huge);
// vit_l() / vit_b() return the other two stock variants. These mirror
// HuggingFace's SamVisionConfig fields.
struct EncoderConfig {
    int   img_size       = 1024;
    int   patch_size     = 16;
    int   in_chans       = 3;
    int   embed_dim      = 1280;            // ViT-H hidden size
    int   depth          = 32;
    int   num_heads      = 16;
    float mlp_ratio      = 4.0f;
    int   out_chans      = 256;             // neck output == image-embed channels
    int   window_size    = 14;
    float layer_norm_eps = 1e-6f;
    // Blocks that attend globally (over the full grid) instead of over windows.
    std::vector<int> global_attn_indexes = {7, 15, 23, 31};

    static EncoderConfig vit_h();
    static EncoderConfig vit_l();
    static EncoderConfig vit_b();

    int grid()     const { return img_size / patch_size; }  // tokens per side (64)
    int seq_len()  const { return grid() * grid(); }         // 4096
    int head_dim() const { return embed_dim / num_heads; }
    int mlp_dim()  const { return static_cast<int>(embed_dim * mlp_ratio); }
};

struct EncoderWeights;  // defined in the .cpp; held opaque

class ImageEncoder {
public:
    explicit ImageEncoder(EncoderConfig cfg);
    ~ImageEncoder();

    ImageEncoder(ImageEncoder&&) noexcept;
    ImageEncoder& operator=(ImageEncoder&&) noexcept;
    ImageEncoder(const ImageEncoder&)            = delete;
    ImageEncoder& operator=(const ImageEncoder&) = delete;

    // Load a HuggingFace SAM checkpoint. `load` expects `dir/model.safetensors`
    // (the single-file layout of the stock facebook/sam-* checkpoints);
    // `load_file` reads one explicit .safetensors path. Both read the
    // `vision_encoder.*` tensors and throw std::runtime_error (prefix
    // "sam::ImageEncoder: ") naming the first missing/mismatched tensor.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Run the encoder. `pixels` must be a host FP32 (1, 3*img_size*img_size)
    // NCHW tensor — the output of sam::preprocess. Returns the dense image
    // embedding as a host FP32 (1, out_chans*grid*grid) NCHW tensor.
    // Throws if pixels has the wrong shape/dtype/device or weights aren't loaded.
    brotensor::Tensor encode(const brotensor::Tensor& pixels) const;

    const EncoderConfig& config() const { return cfg_; }

private:
    EncoderConfig                   cfg_;
    std::unique_ptr<EncoderWeights> w_;
};

}  // namespace brovisionml::sam
