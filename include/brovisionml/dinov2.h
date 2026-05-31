#pragma once

// DINOv2 ViT backbone — the image encoder behind Depth-Anything (and a natural
// base for other DINOv2-headed vision tasks: normals, semantic segmentation).
// This is a *feature-extractor* backbone: it runs the patch embed + the pre-norm
// transformer stack and returns the hidden states at a chosen set of stages (the
// 4 the DPT neck consumes), each with the backbone's final LayerNorm applied —
// matching HF `Dinov2Backbone` with apply_layernorm=True, reshape_hidden_states
// =False (the cls token is kept; the DPT reassemble drops it).
//
// Unlike SAM's ViT (windowed + decomposed relative-position attention), DINOv2 is
// plain global multi-head self-attention with a learned cls token, an absolute
// position embedding (bicubically interpolated to the input's patch grid), and
// per-block LayerScale. LayerScale is folded into the preceding output
// projection at load time, so the forward is a standard pre-norm ViT.
//
// Weights load directly from an HF `DepthAnythingForDepthEstimation`
// safetensors checkpoint (the `backbone.` namespace); see dpt_head.h for the
// neck/head that consume the feature maps.

#include "brotensor/tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace brovisionml::dinov2 {

struct Config {
    int   embed_dim     = 384;   // hidden_size
    int   depth         = 12;    // num_hidden_layers
    int   num_heads     = 6;
    int   patch_size    = 14;
    int   img_size      = 518;   // native size the stored pos-embed grid is for
    int   in_chans      = 3;
    float mlp_ratio     = 4.0f;
    float layer_norm_eps = 1e-6f;
    // 1-based stage indices whose hidden states feed the DPT neck. Stage k is the
    // output of transformer block (k-1); HF `out_features` = ["stage3", ...].
    std::vector<int> out_stages = {3, 6, 9, 12};

    int head_dim()    const { return embed_dim / num_heads; }
    int mlp_dim()     const { return static_cast<int>(embed_dim * mlp_ratio); }
    int native_grid() const { return img_size / patch_size; }  // 37 at 518/14

    static Config vit_s();  // Depth-Anything-V2-Small   (384 / 12 / 6)
    static Config vit_b();  // Depth-Anything-V2-Base    (768 / 12 / 12)
    static Config vit_l();  // Depth-Anything-V2-Large   (1024 / 24 / 16)
};

struct BackboneOutput {
    // One (1 + patch_h*patch_w, embed_dim) feature map per out_stage, in
    // out_stages order, on the backbone's device. Row 0 is the cls token; the
    // backbone's final LayerNorm has been applied.
    std::vector<brotensor::Tensor> feature_maps;
    int patch_h = 0;   // resized_h / patch_size
    int patch_w = 0;   // resized_w / patch_size
};

class Backbone {
public:
    explicit Backbone(Config cfg);
    ~Backbone();
    Backbone(Backbone&&) noexcept;
    Backbone& operator=(Backbone&&) noexcept;

    // Load from a checkpoint directory (reads `<dir>/model.safetensors`) or a
    // file. Weights land host-resident FP32; call to() to migrate to a device.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Migrate weights to a compute device (CPU/CUDA). No-op if already there.
    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }

    const Config& config() const { return cfg_; }

    // Run the backbone on a preprocessed pixel tensor (1, in_chans*H*W) NCHW, on
    // the backbone's device, where H and W are multiples of patch_size. Returns
    // the selected stage feature maps.
    BackboneOutput encode(const brotensor::Tensor& pixels, int H, int W) const;

private:
    Config cfg_;
    brotensor::Device device_ = brotensor::Device::CPU;
    struct Weights;
    std::unique_ptr<Weights> w_;
};

}  // namespace brovisionml::dinov2
