#pragma once

// DSINE v02 decoder — the DPT-style decoder that turns the three EfficientNet-B5
// encoder taps into the *initial* surface-normal map (plus the feature/hidden
// maps the iterative refinement consumes). This header covers ONLY the decoder
// and its ray/uv geometric conditioning; the NRN iterative refinement (ConvGRU +
// the prob/xy/angle heads) is a separate module.
//
// Structure (port of `Decoder` in models/dsine/v02.py, B=5, NF=2048, down=8):
//   conv2  = Conv2d(2048+2, 2048, 1x1, no pad)
//   up1    = UpSampleGN(skip_input=2048+176+2=2226, out=1024)
//   up2    = UpSampleGN(skip_input=1024+64+2=1090,  out=512)
//   normal_head  = get_prediction_head(512+2=514, 128, 3)
//   feature_head = get_prediction_head(514, 128, 64)
//   hidden_head  = get_prediction_head(514, 128, 64)
//
// Forward (uv tensors are 2ch each, built from the camera intrinsics — see
// build_uv below; uv_32 @ /32, uv_16 @ /16, uv_8 @ /8):
//   x_d0   = conv2( cat([s32, uv_32]) )                  # 2048ch @ /32
//   x_d1   = up1( x_d0, cat([s16, uv_16]) )              # 1024ch @ /16
//   x_feat = up2( x_d1, cat([s8,  uv_8 ]) )              #  512ch @ /8
//   x_feat = cat([x_feat, uv_8])                         #  514ch
//   normal = L2normalize_over_channels( normal_head(x_feat) )   # 3ch  (golden)
//   f      = feature_head(x_feat)                        # 64ch
//   h      = hidden_head(x_feat)                         # 64ch
//
// UpSampleGN(skip_input,out).forward(x, concat_with):
//   bilinear-upsample x to concat_with's (H,W) with align_corners=FALSE
//   (brotensor interp2d_forward, mode=1 — half-pixel mapping), cat([up_x,
//   concat_with], dim=1), then `_net`:
//     [ Conv2d_WS(skip_input,out,3,pad1), GroupNorm(8,out), LeakyReLU(0.01),
//       Conv2d_WS(out,out,3,pad1),        GroupNorm(8,out), LeakyReLU(0.01) ]
//   GroupNorm eps is torch's default 1e-5; LeakyReLU negative_slope is torch's
//   default 0.01 (nn.LeakyReLU()).
//
// Conv2d_WS — weight-standardized conv. The 3x3 weight is standardized PER OUTPUT
// channel: subtract the mean over (in,kH,kW), divide by (std + 1e-5) where std is
// torch's UNBIASED (n-1) standard deviation, with the +1e-5 added to std (not in
// quadrature). Because this depends only on the weight, it is folded ONCE at LOAD
// time: we standardize and store the standardized weight, then run a plain
// conv2d (3x3, pad1, +bias) in forward.
//
// get_prediction_head(in,hid,out) = plain convs (NOT weight-standardized):
//   [ Conv2d(in,hid,3,pad1), ReLU, Conv2d(hid,hid,1), ReLU, Conv2d(hid,out,1) ].
//
// ── Ray / uv conditioning ───────────────────────────────────────────────────
// DSINE is geometry-conditioned: every decoder stage is fed a 2ch uv map derived
// from the camera intrinsics (`DSINE_v02.get_ray(..., return_uv=True)`). For a
// feature grid (Hf,Wf) given the full PADDED dims (origH,origW) and intrinsics
// (fx,fy,cx,cy):
//   fu = fx*(Wf/origW);  cu = cx*(Wf/origW)
//   fv = fy*(Hf/origH);  cv = cy*(Hf/origH)
//   uv[0,y,x] = ((x+0.5) - cu) / fu      # pixel-CENTER grid (get_pixel_coords)
//   uv[1,y,x] = ((y+0.5) - cv) / fv
// (NOT L2-normalized; that is the return_uv=True path.) The 3ch ray used by the
// refinement is L2normalize([ray_x, ray_y, 1]); build_ray8 exposes it for the
// next commit but the decoder itself does not use it.
//
// CRITICAL +0.5: the model forward adds `cx += 0.5; cy += 0.5` to the principal
// point BEFORE building any ray. The `dsine::Intrinsics` produced by preprocess
// are in the pre-"+0.5" state, so build_uv / build_ray8 add 0.5 to cx and cy
// internally. Pass the raw preprocess intrinsics.
//
// uv_32 uses (Hf,Wf)=(origH/32, origW/32); uv_16 uses /16; uv_8 uses /8. origH,
// origW are the PADDED (multiple-of-32) dims from preprocess.
//
// Weights load directly from a `model.safetensors` (`decoder.*` namespace), FP32,
// host-resident. After load, `to(Device)` migrates the weights so the forward
// runs on whatever device the encoder taps live on — every stage is a brotensor
// op (conv2d / group_norm / leaky_relu / relu / interp2d, channel-concat, and
// the NCHW channel-L2-normalize), so the decoder is device-agnostic (CPU FP32 or
// CUDA FP32). The uv conditioning maps are built host-side and uploaded to the
// active device inside forward().

#include "brovisionml/dsine_encoder.h"
#include "brovisionml/dsine_preprocess.h"

#include "brotensor/tensor.h"

#include <memory>
#include <string>

namespace brovisionml::dsine {

// A 2ch uv (or 3ch ray) geometric-conditioning map, as a flat NCHW (1, C*Hf*Wf)
// host FP32 tensor.
struct UvMap {
    brotensor::Tensor data;   // (1, C*Hf*Wf), C = 2 (uv) or 3 (ray)
    int c  = 0;
    int hf = 0;
    int wf = 0;
};

// The decoder outputs: the initial normal map (the golden) plus the feature and
// hidden maps the refinement consumes. All at /8, flat NCHW (1, C*h8*w8).
struct DecoderOutput {
    brotensor::Tensor normal;    // (1, 3*h8*w8), L2-normalized over channels
    brotensor::Tensor feature;   // (1, 64*h8*w8)
    brotensor::Tensor hidden;    // (1, 64*h8*w8)
    int h8 = 0;
    int w8 = 0;
};

// Build the 2ch uv conditioning map for a feature grid (Hf,Wf) from the camera
// intrinsics (pre-"+0.5" preprocess state — the +0.5 is applied internally) and
// the full PADDED dims (origH,origW). return_uv=True path of get_ray.
UvMap build_uv(const Intrinsics& intrins, int Hf, int Wf, int origH, int origW);

// Build the 3ch ray (L2-normalized [ray_x, ray_y, 1]) for a feature grid — the
// return_uv=False path of get_ray. The refinement consumes ray_8; the decoder
// does not. Exposed so the next commit can build it without re-deriving get_ray.
UvMap build_ray8(const Intrinsics& intrins, int Hf, int Wf, int origH, int origW);

class Decoder {
public:
    Decoder();
    ~Decoder();
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;

    // Load from a checkpoint directory (reads `<dir>/model.safetensors`) or a
    // file. Reads the `decoder.*` namespace; weight standardization for the two
    // Conv2d_WS convs in each UpSampleGN is folded here. Weights land FP32.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Migrate the loaded weights to `dev` (no-op if already there). The forward
    // then runs on `dev`; the encoder taps passed to forward() must live on the
    // same device. Mirrors DepthEstimator/SAM's runtime device-selection pattern.
    void to(brotensor::Device dev);
    brotensor::Device device() const;

    // Run the decoder on the three encoder taps + the camera intrinsics. origH,
    // origW are the PADDED dims (taps.h*/w* derive from them). The uv maps are
    // built internally from `intrins` (pre-"+0.5"; +0.5 applied inside) and
    // uploaded to the active device. Returns the normal/feature/hidden maps at
    // /8, resident on the active device.
    DecoderOutput forward(const EncoderTaps& taps, const Intrinsics& intrins,
                          int origH, int origW) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brovisionml::dsine
