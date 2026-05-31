#pragma once

// DPT / Depth-Anything image preprocessing — the input contract for a DINOv2 +
// DPT depth (or segmentation) model. Port of HF `DPTImageProcessor` as
// configured for Depth-Anything-V2:
//
//   1. Resize, preserving aspect ratio, so that each side becomes a multiple of
//      `multiple` (14, the DINOv2 patch size). The target is a `target` x
//      `target` box (518 default); with keep_aspect_ratio the scale closest to
//      1 is shared across both axes — HF's `get_resize_output_image_size` — so
//      the resized image is generally NOT square and NOT padded.
//   2. Rescale [0,255] -> [0,1].
//   3. Normalize with ImageNet mean/std (`broimage::IMAGENET_MEAN/STD`).
//
// There is NO padding (the HF processor sets do_pad=False) and NO letterbox: the
// whole resized image is content. The resize is bicubic — broimage's Bicubic is
// the Catmull-Rom (a=-0.5) kernel, matching PIL's BICUBIC (resample=3) that the
// HF processor uses on the uint8 image.
//
// The model input is a flat (1, 3*resized_h*resized_w) FP32 tensor in NCHW
// order. Because the spatial dims vary per image, the DINOv2 backbone
// interpolates its position embedding to the (resized_h/14, resized_w/14) patch
// grid (skipped when the grid already matches the native 37x37).
//
// Pure host-side; the caller uploads `pixels` to the compute device before
// invoking the model.

#include "brotensor/tensor.h"

#include <cstdint>

namespace brovisionml::dpt {

// Maps between the original image and the (multiple-of-14) model input. There is
// no padding, so the whole (resized_w x resized_h) plane is content; a predicted
// map is resized straight back to (orig_w x orig_h).
struct DptTransform {
    int orig_w    = 0;
    int orig_h    = 0;
    int resized_w = 0;   // multiple of `multiple`, what the model sees
    int resized_h = 0;
};

struct PreprocessedImage {
    // (1, 3*resized_h*resized_w) FP32, host-resident, NCHW, ImageNet-normalized.
    brotensor::Tensor pixels;
    DptTransform       transform;
};

// HF `DPTImageProcessor.get_resize_output_image_size`: the resized dims for a
// (w, h) image given a square `target`, rounding each axis to a multiple of
// `multiple` (round-half-to-even, as Python's round). With keep_aspect_ratio the
// per-axis scale closest to 1 is shared; without it each axis hits `target`
// independently (then rounded to a multiple). Output dims are clamped to be at
// least `multiple`.
void resize_output_size(int w, int h, int target, int multiple,
                        bool keep_aspect_ratio, int& out_w, int& out_h);

// Preprocess an 8-bit interleaved HWC image.
//   rgb      : pixel buffer, `w*h*channels` bytes.
//   channels : 1 (gray, replicated to RGB), 3 (RGB), or 4 (RGBA, alpha dropped).
//   target   : square resize target; Depth-Anything-V2 uses 518.
//   multiple : output dims are constrained to a multiple of this (14).
// Throws std::runtime_error (prefix "dpt::preprocess: ") on a zero-size image or
// an unsupported channel count.
PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int target = 518, int multiple = 14,
                             bool keep_aspect_ratio = true);

}  // namespace brovisionml::dpt
