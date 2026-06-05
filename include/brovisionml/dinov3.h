#pragma once

// DINOv3 ViT backbone — the image encoder behind TripoSplat (single-image -> 3D
// Gaussian splat) and a natural base for other DINOv3-headed vision tasks. Like
// dinov2.h this is a *feature-extractor* backbone: patch embed + a pre-norm
// transformer stack returning the final hidden states (with the backbone's final
// LayerNorm applied). It matches HF `DINOv3ViTModel` (transformers v5.x,
// `dinov3_vit`) in eval mode.
//
// DINOv3 differs from DINOv2 in three ways that matter for a byte-exact port:
//   * Sequence is [CLS, register_tokens, patch_tokens] (DINOv2 has no registers).
//   * No learned/interpolated absolute position embedding — position is injected
//     as 2D *axial* RoPE applied to the q/k of the patch tokens only (the prefix
//     CLS+register tokens are never rotated). Frequencies are computed, not
//     stored. HF uses the rotate-half (GPT-NeoX) pairing; we convert it to the
//     interleaved pairing brotensor::rope_apply expects by permuting the q/k
//     projection rows within each head at load time (see dinov3.cpp), so the
//     forward is a plain rope_apply with a host-built cos/sin table.
//   * The FFN is a gated SwiGLU (gate_proj/up_proj/down_proj + SiLU), not the
//     fc1/GELU/fc2 of DINOv2.
//
// Per-block LayerScale is folded into the preceding output projection (LS1 -> Wo,
// LS2 -> down_proj) at load time, exactly as the dinov2 loader does. Weights load
// directly from a `dino_v3_vit_h.safetensors` checkpoint (the VAST-AI/TripoSplat
// bundle ships raw weights with no config.json, so the architecture lives in the
// Config preset below). Attention biases follow the HF config: query/value/proj
// biased, key unbiased.

#include "brotensor/tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace brovisionml::dinov3 {

struct Config {
    int   embed_dim          = 1280;  // hidden_size
    int   depth              = 32;    // num_hidden_layers
    int   num_heads          = 20;    // head_dim = embed_dim / num_heads = 64
    int   patch_size         = 16;
    int   num_register_tokens = 4;
    int   intermediate_size  = 5120;  // SwiGLU inner width
    int   in_chans           = 3;
    int   img_size           = 224;   // informational; RoPE coords are dynamic
    float rope_theta         = 100.0f;
    float layer_norm_eps     = 1e-5f;

    int head_dim()          const { return embed_dim / num_heads; }
    int num_prefix_tokens() const { return 1 + num_register_tokens; }

    static Config vit_h();  // DINOv3 ViT-H+/16 (1280 / 32 / 20, SwiGLU, 4 regs)
};

struct BackboneOutput {
    // (num_prefix_tokens + patch_h*patch_w, embed_dim) on the backbone's device.
    // Rows [0, num_prefix_tokens) are CLS + register tokens; the rest are patch
    // tokens in row-major (h-major) order. The backbone's final LayerNorm has
    // been applied (HF apply_layernorm=True).
    brotensor::Tensor last_hidden_state;
    int patch_h = 0;             // H / patch_size
    int patch_w = 0;             // W / patch_size
    int num_prefix_tokens = 0;   // 1 (cls) + num_register_tokens
};

class Backbone {
public:
    explicit Backbone(Config cfg);
    ~Backbone();
    Backbone(Backbone&&) noexcept;
    Backbone& operator=(Backbone&&) noexcept;

    // Load from a checkpoint directory (reads `<dir>/dino_v3_vit_h.safetensors`)
    // or a specific file. Weights land host-resident FP32; call to() to migrate.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Migrate weights to a compute device (CPU/CUDA/Metal). No-op if already there.
    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }

    const Config& config() const { return cfg_; }

    // Run the backbone on a preprocessed pixel tensor (1, in_chans*H*W) NCHW, on
    // the backbone's device, where H and W are multiples of patch_size.
    BackboneOutput encode(const brotensor::Tensor& pixels, int H, int W) const;

private:
    Config cfg_;
    brotensor::Device device_ = brotensor::Device::CPU;
    struct Weights;
    std::unique_ptr<Weights> w_;
};

}  // namespace brovisionml::dinov3
