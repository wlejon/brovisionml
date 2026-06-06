#pragma once

// BiRefNet — dichotomous image segmentation / background removal (ZhengPeng7/
// BiRefNet, the v1 Swin-L release used by VAST-AI/TripoSplat as its `rmbg`
// front-end). Single RGB image in → a 1-channel alpha matte in [0,1].
//
// Architecture (faithful to the reference triposplat_ref/model.py):
//   * Swin-Large backbone (embed_dim 192, depths [2,2,18,2], heads
//     [6,12,24,48], window 12), run twice — full and half resolution — and the
//     two feature pyramids concatenated (mul_scl_ipt='cat'), giving doubled
//     channel counts (3072,1536,768,384) high→low resolution.
//   * An ASPP-deformable squeeze block over the top (lowest-res) level.
//   * A 4-level decoder with per-level ASPP-deformable attention, lateral
//     skips, multi-scale input modulation (image patches re-injected at each
//     level), and gradient-decoder-triggering (gdt) gating, ending in a 1×1
//     head → sigmoid alpha.
//
// No BiRefNet-specific kernel: the forward is a pure composition of brotensor
// ops — conv2d / deform_conv2d (the modulated DCNv2 the ASPP-deformable blocks
// need), batch_norm_inference, the Swin window attention via
// self_attention_bias_forward (qkv/proj bias + a precomputed relative-position
// + shifted-window bias), layernorm / linear / gelu, interp2d_align_corners
// (BiRefNet upsamples with align_corners=True), and gather_rows (Swin's
// pad+roll+window-partition+patch-merge baked into precomputed permutations).
//
// Device: load() reads the checkpoint to host FP32; call to(Device) to migrate.
// The heavy ops run on-device; the window permutations are INT32 index gathers
// that ride along on the same device. Validated against the reference oracle
// (out-of-repo goldens, gen_birefnet_golden.py): the Swin backbone stage
// features and the end-to-end alpha matte.

#include "brotensor/tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace brovisionml::birefnet {

using brotensor::Tensor;

// Predicted matte at the original image resolution; alpha in [0,1].
struct Matte {
    int width = 0, height = 0;
    std::vector<float> alpha;   // row-major, width*height
};

class BiRefNet {
public:
    BiRefNet();
    ~BiRefNet();
    BiRefNet(BiRefNet&&) noexcept;
    BiRefNet& operator=(BiRefNet&&) noexcept;

    // Load the Swin-L BiRefNet safetensors (any of F32/F16/BF16 on disk; widened
    // to host FP32). Throws on a missing/mismatched key.
    void load(const std::string& path);

    // Migrate all weights to a device (CPU/CUDA/Metal). Call after load().
    void to(brotensor::Device dev);

    brotensor::Device device() const;

    // Run the full network. `imgNCHW` is a (1, 3*H*W) NCHW FP32 tensor already
    // normalised by the BiRefNet recipe (ImageNet mean/std) and sized to the
    // model input (square, a multiple of 32; the reference uses 1024). Returns
    // the pre-sigmoid logits as a (1, H*W) tensor — caller applies sigmoid.
    Tensor forwardLogits(const Tensor& imgNCHW, int H, int W) const;

    // Convenience: take raw RGB (row-major, 0..255 or 0..1) at any size, do the
    // BiRefNet preprocessing (resize to `modelSize`², ImageNet-normalise), run
    // the net, sigmoid, and resize the matte back to (origW, origH).
    Matte removeBackground(const float* rgb, int origW, int origH,
                           bool rgbIs255 = true, int modelSize = 1024) const;

    // ── debug / golden hooks ──
    // The 4 raw Swin-L backbone stage features for an NCHW input (no mul_scl
    // dual-resolution concat) — used by the backbone golden test.
    std::vector<Tensor> debugBackbone(const Tensor& xNCHW, int H, int W) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brovisionml::birefnet
