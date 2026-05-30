#pragma once

// SAM (Segment Anything) image preprocessing — the input contract for the ViT
// image encoder. Port of segment_anything's `ResizeLongestSide` +
// `Sam.preprocess`:
//
//   1. Resize the longest side to `input_size` (default 1024), preserving
//      aspect ratio (bilinear, center-pixel / align_corners=False — what the
//      upstream PIL resize does).
//   2. Normalize: scale [0,255] -> [0,1] then `(x - mean) / std` per channel,
//      using broimage's SAM presets (ImageNet stats; upstream's pixel_mean/std
//      are these same stats expressed in [0,255]).
//   3. Pad bottom/right with zeros to a square `input_size x input_size`. The
//      pad is applied *after* normalize, so padded pixels are exactly 0.0 in
//      normalized space (F.pad default), not the channel mean.
//
// The model input is a flat (1, 3*input_size*input_size) FP32 tensor in NCHW
// order — the same batch-of-1 flattening brolm's vision encoders use, so the
// conv2d patch-embed reads it with explicit (N=1, C=3, H, W) dims.
//
// Pure host-side; no GPU kernels. The caller uploads `pixels` to the compute
// device before invoking the encoder.

#include "brotensor/tensor.h"

#include <cstdint>

namespace brovisionml::sam {

// Maps coordinates between the original image and the model's square input, so
// callers can transform prompt points/boxes *in* and predicted masks *out*.
// The resized content sits at the top-left; everything below/right of
// (resized_w, resized_h) is zero padding.
struct ImageTransform {
    int   orig_w     = 0;
    int   orig_h     = 0;
    int   resized_w  = 0;   // content width inside the square (<= input_size)
    int   resized_h  = 0;   // content height inside the square (<= input_size)
    int   input_size = 0;   // square side (1024 for stock SAM)
    float scale      = 0.f; // resized / orig == input_size / max(orig_w, orig_h)
};

struct PreprocessedImage {
    // (1, 3*input_size*input_size) FP32, host-resident, NCHW, normalized, with
    // the bottom/right zero padding already applied.
    brotensor::Tensor pixels;
    ImageTransform     transform;
};

// Preprocess an 8-bit interleaved HWC image.
//   rgb      : pixel buffer, `w*h*channels` bytes.
//   channels : 1 (gray, replicated to RGB), 3 (RGB), or 4 (RGBA, alpha dropped).
//   input_size: square model input side; stock SAM is 1024.
// Throws std::runtime_error (prefix "sam::preprocess: ") on a zero-size image
// or an unsupported channel count.
PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int input_size = 1024);

// segment_anything `ResizeLongestSide.get_preprocess_shape`: the resized
// (content) dimensions for an (w, h) image at the given longest-side target.
void preprocess_shape(int w, int h, int input_size,
                      int& resized_w, int& resized_h);

// Map a point in original-image pixel coords into model-input coords (the
// space the prompt encoder expects). Inverse of mapping a mask back out.
void apply_coords(const ImageTransform& t, float x, float y,
                  float& out_x, float& out_y);

}  // namespace brovisionml::sam
