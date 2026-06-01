#pragma once

// Lineart image preprocessing — the input contract for the ControlNet "lineart"
// annotator (lllyasviel's "Informative Drawings" generator). Like HED the
// front-end is deliberately minimal: the generator is fully convolutional, so
// there is no fixed input size and no ImageNet normalization — just a plain
// scale to [0,1]. There is no learned input bias here (unlike HED), so this step
// only divides by 255.
//
//   1. Compact to packed RGB (drop alpha; broadcast gray).
//   2. Optionally resize so the LONGER side == `detect_resolution` (preserve
//      aspect, bilinear). `detect_resolution == 0` keeps the native size. This
//      mirrors the ControlNet annotator, which runs the detector at a fixed
//      working resolution and resizes the line map back to the original after.
//   3. Pack to a flat (1, 3*proc_h*proc_w) FP32 tensor in NCHW order, with pixel
//      values scaled to [0,1] (value/255 — the scale the generator was trained on).
//
// Pure host-side; the caller uploads `pixels` to the compute device before
// invoking the model. The model's line map comes out at the processed
// resolution; map it back to the original with `transform`.

#include "brotensor/tensor.h"

#include <cstdint>

namespace brovisionml::lineart {

// Maps between the original image and the (possibly resized) model input. When
// `detect_resolution == 0` the proc dims equal the original dims.
struct LineartTransform {
    int orig_w = 0;
    int orig_h = 0;
    int proc_w = 0;   // what the model runs at
    int proc_h = 0;
};

struct PreprocessedImage {
    // (1, 3*proc_h*proc_w) FP32, host-resident, NCHW, RGB in the [0,1] range.
    brotensor::Tensor pixels;
    LineartTransform  transform;
};

// Compute the processed dims for an original (w,h) at a given detect resolution.
// `detect_resolution == 0` returns (w,h) unchanged; otherwise the longer side is
// scaled to `detect_resolution` and the shorter side scaled proportionally
// (rounded to the nearest pixel, floored to at least 1).
void lineart_proc_dims(int orig_w, int orig_h, int detect_resolution,
                       int& proc_w, int& proc_h);

// Preprocess an 8-bit interleaved HWC image.
//   rgb              : pixel buffer, `w*h*channels` bytes.
//   channels         : 1 (gray, replicated to RGB), 3 (RGB), or 4 (RGBA, alpha
//                      dropped).
//   detect_resolution: longer-side target for the working resolution; 0 = native.
// Throws std::runtime_error (prefix "lineart::preprocess: ") on a zero-size image
// or an unsupported channel count.
PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int detect_resolution = 0);

}  // namespace brovisionml::lineart
