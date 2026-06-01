#pragma once

// SegFormer image preprocessing — the input contract for the ControlNet "seg"
// semantic-segmentation annotator (HF SegformerImageProcessor). The b0-ade
// checkpoint runs at a fixed 512×512 input:
//
//   1. Compact to packed RGB (drop alpha; broadcast gray).
//   2. Resize to model_size×model_size (bilinear). The processor's size is the
//      square {"height":512,"width":512}, so this is an exact-square resize, not
//      a shorter-edge rescale.
//   3. Rescale [0,255]->[0,1] into planar NCHW, then per-channel ImageNet
//      normalize (broimage IMAGENET_MEAN / IMAGENET_STD).
//   4. Pack to a flat (1, 3*model_size*model_size) FP32 tensor in NCHW order.
//
// Pure host-side; the caller uploads `pixels` to the compute device before
// invoking the model. `transform` carries the original (W,H) so the model's
// class map can be upsampled back.

#include "brotensor/tensor.h"

#include <cstdint>

namespace brovisionml::segformer {

struct SegformerTransform {
    int orig_w = 0;
    int orig_h = 0;
    int model_size = 512;   // fixed square model input
};

struct PreprocessedImage {
    // (1, 3*model_size*model_size) FP32, host-resident, NCHW, ImageNet-normalized.
    brotensor::Tensor pixels;
    SegformerTransform transform;
};

// Preprocess an 8-bit interleaved HWC image (channels 1/3/4) to the fixed
// model_size×model_size 3-channel ImageNet-normalized NCHW tensor. Throws
// std::runtime_error (prefix "segformer::preprocess: ") on a zero-size image or
// an unsupported channel count.
PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int model_size = 512);

}  // namespace brovisionml::segformer
