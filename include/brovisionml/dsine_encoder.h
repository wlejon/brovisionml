#pragma once

// EfficientNet-B5 encoder for DSINE (Discontinuity-aware Surface-normal
// Estimation). This is the *backbone* the DSINE decoder/refinement consume: it
// runs the geffnet `tf_efficientnet_b5_ap` (GenEfficientNet) feature stack on a
// preprocessed pixel tensor and returns the three feature taps the DSINE decoder
// reads. It does NOT implement the decoder or the iterative refinement.
//
// The checkpoint is geffnet-layout (keys under `encoder.original_model.*`),
// FP32. The top-level forward is:
//   conv_stem -> bn1 -> act1(SiLU) -> blocks(7 stages) -> conv_head -> bn2
//   -> act2(SiLU)
// and the DSINE encoder taps the output of stage 2, stage 4, and conv_head:
//   tap s8  = stage-2 output (64ch,  /8)
//   tap s16 = stage-4 output (176ch, /16)
//   tap s32 = conv_head output (2048ch, /32) — the RAW 1x1 conv output, taken
//             before geffnet's head BN (bn2) and act2(SiLU); those belong to
//             the classifier head, not to this feature tap.
// (These are features[6], features[8], features[11] in the reference's feature
// list. Producing s32 requires running ALL stages + conv_head.)
//
// Stage table (block counts are read from the checkpoint; the kernel/stride/
// residual rules below are the connectivity that shapes alone do not encode):
//   stage0: DepthwiseSeparable, k3, stride 1   (48 -> 24)
//   stage1: InvertedResidual,   k3, stride 2   (24 -> 40)
//   stage2: InvertedResidual,   k5, stride 2   (40 -> 64)   <- s8
//   stage3: InvertedResidual,   k3, stride 2   (64 -> 128)
//   stage4: InvertedResidual,   k5, stride 1   (128 -> 176) <- s16
//   stage5: InvertedResidual,   k5, stride 2   (176 -> 304)
//   stage6: InvertedResidual,   k3, stride 1   (304 -> 512)
// Only block 0 of each stage applies the stage stride and changes channels;
// blocks 1+ run stride 1 and out->out.
//
// Block structures (geffnet key names; all convs except the depthwise are 1x1):
//   DepthwiseSeparable (stage0):
//     conv_dw (depthwise, TF-same pad) -> bn1 -> SiLU -> SE
//       -> conv_pw (project) -> bn2  [+ input if stride==1 && in==out]
//   InvertedResidual (stages1-6):
//     conv_pw (expand) -> bn1 -> SiLU -> conv_dw (depthwise, TF-same pad)
//       -> bn2 -> SiLU -> SE -> conv_pwl (project) -> bn3
//       [+ input if stride==1 && in==out]
//   Squeeze-Excite (SE) over the post-dw feature (C = expanded ch for IR,
//   dw ch for DS): global-avg-pool(H,W) -> conv_reduce(1x1) -> SiLU
//     -> conv_expand(1x1) -> sigmoid -> channel-wise multiply into the SE input.
//
// conv_head: 1x1 (512 -> 2048). Its raw output is tap s32. (geffnet follows it
//   with bn2 + act2(SiLU) for the classifier; the DSINE feature tap does not.)
//
// TF-"SAME" dynamic padding (geffnet `tf_efficientnet_*`): the stem and every
// depthwise conv pad asymmetrically so out = ceil(i/s). For input size i,
// stride s, kernel k (dilation 1):
//   out = ceil(i/s); pad_total = max((out-1)*s + k - i, 0);
//   pad_before = pad_total/2; pad_after = pad_total - pad_before.
// Computed independently per axis; the input is pre-padded with the 2D zero pad
// op (asymmetric) and the conv then runs with padding 0. The 1x1 convs need no
// padding.
//
// Activation is SiLU/Swish throughout. BatchNorm runs in inference mode
// (affine weight/bias + running_mean/running_var, eps 1e-3 — geffnet's
// tf_efficientnet default); num_batches_tracked is unused.
//
// Weights load directly from a `model.safetensors` (the `encoder.original_model.`
// namespace). They land host-resident FP32; the forward runs on the host (CPU
// FP32) for parity.

#include "brotensor/tensor.h"

#include <memory>
#include <string>

namespace brovisionml::dsine {

// The three encoder taps the DSINE decoder consumes, as flat NCHW (1, C*H*W)
// FP32 tensors. Channels/strides for the standard B5 config are noted; the
// spatial dims follow the (padded) input.
struct EncoderTaps {
    brotensor::Tensor s8;    // stage-2 output:   (1, 64*H/8*W/8)
    brotensor::Tensor s16;   // stage-4 output:   (1, 176*H/16*W/16)
    brotensor::Tensor s32;   // conv_head output: (1, 2048*H/32*W/32)
    int h8 = 0,  w8 = 0;
    int h16 = 0, w16 = 0;
    int h32 = 0, w32 = 0;
};

class EncoderB5 {
public:
    EncoderB5();
    ~EncoderB5();
    EncoderB5(EncoderB5&&) noexcept;
    EncoderB5& operator=(EncoderB5&&) noexcept;

    // Load from a checkpoint directory (reads `<dir>/model.safetensors`) or a
    // file. Block counts and all per-block channel dims are read from the
    // checkpoint. Weights land host-resident FP32.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Run the encoder on a preprocessed pixel tensor (1, 3*H*W) NCHW FP32, where
    // H and W are the (already zero-padded, multiple-of-32) input dims from
    // dsine::preprocess. Runs on the host; returns the three taps.
    EncoderTaps forward(const brotensor::Tensor& pixels, int H, int W) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brovisionml::dsine
