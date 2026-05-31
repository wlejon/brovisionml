#pragma once

// DSINE surface-normal image preprocessing — the input contract for the DSINE
// (Discontinuity-aware Surface-normal estimation) model. Port of DSINE's
// inference front-end, which is deliberately *unlike* the DPT/Depth-Anything
// one: there is NO resize and NO letterbox. The image is consumed at its native
// resolution; it is only zero-padded so each spatial dim becomes a multiple of
// 32 (the encoder's total downsampling stride), then ImageNet-normalized.
//
//   1. Pad to a multiple of 32 with ZEROS (`get_padding`, ported exactly):
//      for each axis, if it is already a multiple of 32 no padding is added;
//      otherwise it grows to the next multiple of 32 and the extra rows/cols are
//      split as evenly as possible with the smaller half on the top/left
//      (l/t = floor(gap/2), r/b = gap - l/t). The pad is applied to the uint8
//      image with fill value 0.
//   2. Rescale [0,255] -> [0,1].
//   3. Normalize with ImageNet mean/std (`broimage::IMAGENET_MEAN/STD`).
//
// Because the zero pad is applied BEFORE the normalize, the padded pixels become
// `(0 - mean) / std` per channel — exactly what the reference produces (it pads
// the already-[0,1] image with 0.0, then normalizes). The pad is along the
// border only; the interior is untouched content.
//
// DSINE is geometry-conditioned: the forward pass takes camera intrinsics. With
// no calibration available the reference synthesizes them from an assumed
// field-of-view (`intrins_from_fov`, default 60°): a centered principal point
// and a focal length set so the longer image side subtends `fov_deg`. These are
// computed on the ORIGINAL (unpadded) dims, then the principal point is shifted
// by the pad offset (cx += l, cy += t) so it stays aligned to image content
// after padding. NOTE: this is the pre-padding-offset *and* pre-"+0.5" state —
// the model's forward() adds 0.5 to cx/cy internally (top-left pixel at
// (0.5,0.5)); that adjustment belongs to the model, not the preprocessor, and
// is intentionally NOT applied here.
//
// The model input is a flat (1, 3*pad_h*pad_w) FP32 tensor in NCHW order. Pure
// host-side; the caller uploads `pixels` to the compute device before invoking
// the model.

#include "brotensor/tensor.h"

#include <cstdint>

namespace brovisionml::dsine {

// Maps between the original image and the (multiple-of-32) zero-padded model
// input. `pad_l/r/t/b` are the zero-pad widths on each side; the original image
// content lives at offset (pad_l, pad_t) inside the (pad_w x pad_h) plane.
struct DsineTransform {
    int orig_w = 0;
    int orig_h = 0;
    int pad_w  = 0;   // multiple of 32, what the model sees
    int pad_h  = 0;
    int pad_l  = 0;   // zero-pad on the left
    int pad_r  = 0;   // ...right
    int pad_t  = 0;   // ...top
    int pad_b  = 0;   // ...bottom
};

// Pinhole camera intrinsics passed into the DSINE forward (pre-"+0.5" state;
// see the header comment). fx/fy are focal lengths, cx/cy the principal point.
struct Intrinsics {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
};

struct PreprocessedImage {
    // (1, 3*pad_h*pad_w) FP32, host-resident, NCHW, ImageNet-normalized.
    brotensor::Tensor pixels;
    DsineTransform     transform;
    Intrinsics         intrins;
};

// DSINE `get_padding(orig_H, orig_W)`: the zero-pad widths that grow each axis
// to the next multiple of 32 (none when it is already a multiple). The gap is
// split with the smaller half on the top/left.
void get_padding(int orig_h, int orig_w, int& l, int& r, int& t, int& b);

// DSINE `intrins_from_fov(new_fov, H, W)`: focal length such that the longer
// side subtends `fov_deg`, principal point at the image center minus half a
// pixel. Computed on the ORIGINAL unpadded dims; the pad offset (cx += l,
// cy += t) is applied separately inside `preprocess`. NO "+0.5" is added here.
Intrinsics intrins_from_fov(int orig_h, int orig_w, float fov_deg = 60.0f);

// Preprocess an 8-bit interleaved HWC image at native resolution.
//   rgb      : pixel buffer, `w*h*channels` bytes.
//   channels : 1 (gray, replicated to RGB), 3 (RGB), or 4 (RGBA, alpha dropped).
//   fov_deg  : assumed field-of-view for the synthesized intrinsics (60°).
// Throws std::runtime_error (prefix "dsine::preprocess: ") on a zero-size image
// or an unsupported channel count.
PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             float fov_deg = 60.0f);

}  // namespace brovisionml::dsine
