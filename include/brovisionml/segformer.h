#pragma once

// SegFormer — semantic segmentation, end to end. The ControlNet "seg"
// conditioning annotator: an ADE20K 150-class semantic map, colorized with the
// canonical ADE20K palette. The model is HF
// `SegformerForSemanticSegmentation` — a hierarchical Mix-Transformer (MiT)
// encoder feeding an all-MLP decode head — loaded directly from an nvidia/
// segformer-* safetensors checkpoint.
//
// Encoder (MiT): four hierarchical stages. Each stage starts with an
// OverlapPatchEmbed (a strided/padded Conv2d that downsamples + a token
// LayerNorm), then runs `depths[i]` transformer blocks. Each block is a
// pre-norm Efficient-Self-Attention (with a spatial-reduction Conv2d that
// shrinks K/V when `sr_ratios[i] > 1`) followed by a MixFFN (Linear → 3×3
// depthwise Conv2d → GELU → Linear). A per-stage final LayerNorm closes each
// stage; all four stage feature maps feed the head.
//
// Decode head (all-MLP): each stage map is projected (Linear) to a shared
// `decoder_hidden_size`, bilinearly upsampled to the stage-0 grid, the four are
// concatenated in REVERSED stage order, fused by a 1×1 Conv2d + BatchNorm +
// ReLU, and a 1×1 Conv2d classifier emits `num_labels` logits. The logits are
// bilinearly upsampled to the input resolution and argmax'd per pixel.
//
// No SegFormer-specific kernel: the forward is a pure composition of brotensor
// ops — `conv2d_forward` (strided/padded patch embeds, the spatial-reduction
// conv, the 3×3 depthwise MixFFN conv, the 1×1 fuse/classifier convs),
// `layernorm_forward_inference_batched`, `linear_forward_batched`,
// `gelu_exact_forward`, `batch_norm_inference`, `interp2d_forward` (bilinear,
// align_corners=False), and the cross-attention primitive `cross_attention_
// forward`. The Efficient-Self-Attention has unequal query / key lengths when
// sr_ratios[i] > 1 (Q over the full HW grid, K/V over the reduced grid), so it
// is expressed with `cross_attention_forward` (Ctx == X when sr_ratio == 1, a
// reduced context tensor otherwise) rather than the equal-length `mha_forward`.
//
// Config-driven: all dims (hidden_sizes, num_attention_heads, depths,
// sr_ratios, mlp_ratio, patch_sizes, strides, decoder_hidden_size, num_labels,
// layer_norm_eps) are read from the checkpoint's config.json, so MiT-B0 through
// B5 load from their respective checkpoints. Validated against
// nvidia/segformer-b0-finetuned-ade-512-512 (the 150-class ADE20K head).
//
// Device: like the other detectors, call to(Device) after load() to run on a
// device; detect() preprocesses on the host, uploads the pixels, runs the net
// on-device, copies the logits back, and argmaxes on the host.

#include "brotensor/tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brovisionml::segformer {

struct SegformerConfig {
    int model_size = 512;   // fixed square model input (the b0-ade checkpoint)
};

// A per-pixel semantic class map at the original image size; ids in [0, 149].
struct SegMap {
    int width = 0, height = 0;
    std::vector<uint8_t> classes;   // height*width row-major class ids

    uint8_t at(int y, int x) const {
        return classes[static_cast<std::size_t>(y) * width + x];
    }
};

// The raw decode-head logits at the head grid (128×128 for a 512 input),
// before the final upsample — exposed for testing/debugging (the neural gate).
struct Logits {
    int channels = 0, height = 0, width = 0;   // (num_labels, 128, 128)
    std::vector<float> data;                    // channels*height*width, NCHW

    float at(int c, int y, int x) const {
        return data[(static_cast<std::size_t>(c) * height + y) * width + x];
    }
};

class SegformerDetector {
public:
    explicit SegformerDetector(SegformerConfig cfg = {});
    ~SegformerDetector();
    SegformerDetector(SegformerDetector&&) noexcept;
    SegformerDetector& operator=(SegformerDetector&&) noexcept;

    // Load from a checkpoint directory (reads <dir>/config.json and
    // <dir>/model.safetensors) or directly from a safetensors file (config.json
    // is read from the file's directory). Weights land host-resident FP32; call
    // to() to migrate to a device.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    void to(brotensor::Device dev);
    brotensor::Device device() const;

    const SegformerConfig& config() const { return cfg_; }

    // Run the encoder + decode head on a pre-normalized NCHW pixel tensor
    // (1, 3*model_size*model_size) and return the raw decode-head logits at the
    // head grid (the clean neural gate, resize-convention out of the loop).
    Logits infer_logits_from_tensor(const brotensor::Tensor& pixels) const;

    // Run the full preprocess → encoder → head → upsample → argmax pipeline on
    // an 8-bit interleaved HWC image (channels 1/3/4), returning the decode-head
    // logits at the head grid (no upsample/argmax).
    Logits infer_logits(const uint8_t* rgb, int w, int h, int channels) const;

    // Full pipeline → per-pixel class ids upsampled back to the original size.
    SegMap detect(const uint8_t* rgb, int w, int h, int channels) const;

    // Colorize a class map to HxWx3 interleaved RGB via the ADE20K palette.
    static std::vector<uint8_t> colorize(const SegMap& m);

private:
    // Encoder + decode head on a pre-normalized pixel tensor; returns the
    // logits NCHW (num_labels, grid_h, grid_w) ON THE COMPUTE DEVICE so
    // detect() can upsample + argmax there without a logits download.
    brotensor::Tensor infer_logits_dev_(const brotensor::Tensor& pixels,
                                        int* grid_h, int* grid_w) const;

    SegformerConfig cfg_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brovisionml::segformer
