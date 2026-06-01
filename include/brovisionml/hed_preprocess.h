#pragma once

// HED soft-edge image preprocessing — the input contract for the HED
// (Holistically-nested Edge Detection) ControlNet soft-edge annotator. The
// front-end is deliberately minimal: HED is fully convolutional, so there is no
// fixed input size and no ImageNet normalization here. The learned per-channel
// `norm` bias is part of the *model* (a checkpoint parameter the forward pass
// subtracts), NOT the preprocessor — so unlike DSINE/DPT this step does not
// touch the channel statistics.
//
//   1. Compact to packed RGB (drop alpha; broadcast gray).
//   2. Optionally resize so the LONGER side == `detect_resolution` (preserve
//      aspect, bilinear). `detect_resolution == 0` keeps the native size. This
//      mirrors the ControlNet annotator, which runs HED at a fixed working
//      resolution and resizes the edge map back to the original afterwards.
//   3. Pack to a flat (1, 3*proc_h*proc_w) FP32 tensor in NCHW order, with pixel
//      values left in the [0,255] range (the scale the `norm` bias expects).
//
// Pure host-side; the caller uploads `pixels` to the compute device before
// invoking the model. The model's edge map comes out at the processed
// resolution; map it back to the original with `transform`.

#include "brotensor/tensor.h"

#include <cstdint>

namespace brovisionml::hed {

// Maps between the original image and the (possibly resized) model input. When
// `detect_resolution == 0` the proc dims equal the original dims.
struct HedTransform {
    int orig_w = 0;
    int orig_h = 0;
    int proc_w = 0;   // what the model runs at
    int proc_h = 0;
};

struct PreprocessedImage {
    // (1, 3*proc_h*proc_w) FP32, host-resident, NCHW, RGB in the [0,255] range.
    brotensor::Tensor pixels;
    HedTransform      transform;
};

// Compute the processed dims for an original (w,h) at a given detect resolution.
// `detect_resolution == 0` returns (w,h) unchanged; otherwise the longer side is
// scaled to `detect_resolution` and the shorter side scaled proportionally
// (rounded to the nearest pixel, floored to at least 1).
void hed_proc_dims(int orig_w, int orig_h, int detect_resolution,
                   int& proc_w, int& proc_h);

// Preprocess an 8-bit interleaved HWC image.
//   rgb              : pixel buffer, `w*h*channels` bytes.
//   channels         : 1 (gray, replicated to RGB), 3 (RGB), or 4 (RGBA, alpha
//                      dropped).
//   detect_resolution: longer-side target for the working resolution; 0 = native.
// Throws std::runtime_error (prefix "hed::preprocess: ") on a zero-size image or
// an unsupported channel count.
PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int detect_resolution = 0);

}  // namespace brovisionml::hed
