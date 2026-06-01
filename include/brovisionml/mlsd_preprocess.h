#pragma once

// MLSD image preprocessing — the input contract for the ControlNet "mlsd"
// straight-line annotator (M-LSD, MobileNetV2 backbone). The M-LSD checkpoint is
// trained at a fixed 512×512 input, so this front-end always produces a 512×512
// tensor:
//
//   1. Compact to packed RGB (drop alpha; broadcast gray).
//   2. Resize to model_size×model_size (512, bilinear). The reference uses cv2
//      INTER_AREA; for a same-size source this is an identity, and for downscales
//      bilinear is the ballpark-equivalent the annotator role tolerates.
//   3. Append a constant 4th channel of ones, then normalize the whole 4-channel
//      tensor by (x/127.5 - 1). The ones plane is normalized too, so channel 3 is
//      the constant 1/127.5 - 1 ≈ -0.992157 — not 1.
//   4. Pack to a flat (1, 4*512*512) FP32 tensor in NCHW order.
//
// Pure host-side; the caller uploads `pixels` to the compute device before
// invoking the model. The model emits a TP map at model_size/2 (256); use
// `transform` to scale decoded segments back to the original image.

#include "brotensor/tensor.h"

#include <cstdint>

namespace brovisionml::mlsd {

// Maps between the original image and the fixed model input + TP-map grid.
struct MlsdTransform {
    int orig_w = 0;
    int orig_h = 0;
    int model_size = 512;   // fixed square model input
    int tp_w = 256;         // TP-map width  (model_size / 2)
    int tp_h = 256;         // TP-map height
};

struct PreprocessedImage {
    // (1, 4*model_size*model_size) FP32, host-resident, NCHW, (x/127.5 - 1).
    brotensor::Tensor pixels;
    MlsdTransform     transform;
};

// Preprocess an 8-bit interleaved HWC image (channels 1/3/4) to the fixed
// model_size×model_size 4-channel normalized NCHW tensor. Throws
// std::runtime_error (prefix "mlsd::preprocess: ") on a zero-size image or an
// unsupported channel count.
PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int model_size = 512);

}  // namespace brovisionml::mlsd
