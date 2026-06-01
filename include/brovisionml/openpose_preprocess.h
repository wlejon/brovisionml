#pragma once

// OpenPose body-pose preprocessing — the input contract for the ControlNet
// "openpose" body annotator (the CMU body_25/COCO-18 PAF network as packaged by
// pytorch-openpose / controlnet_aux). Body-only: hand and face sub-networks are
// out of scope (the canonical ControlNet openpose control image is body-only,
// `include_hand=False, include_face=False`).
//
// The reference pipeline (controlnet_aux open_pose body.py / __init__.py):
//   1. HWC3 the input, then resize_image(detect_resolution): scale so the
//      shorter side == detect_resolution, rounded to a multiple of 64
//      (Lanczos on upscale, Area on downscale). This is `oriImg` at (H,W).
//   2. RGB -> BGR (the net was trained on BGR; channel order fed to conv1_1 is
//      B,G,R). The flip happens here in preprocess.
//   3. scale = 0.5 * 368 / H. imageToTest = smart_resize_k(oriImg, fx=fy=scale).
//   4. padRightDownCorner(stride=8, padValue=128): pad bottom+right to a
//      multiple of 8 with constant 128 (in pixel/255 space).
//   5. Normalize im = padded/256 - 0.5; transpose HWC->CHW (BGR order); batch 1.
//      This is the (1,3,Hp,Wp) network input.
//
// Pure host-side; the caller uploads `pixels` to the compute device. The
// network emits PAF + heatmap at (Hp/8, Wp/8); the postproc uses the transform
// bookkeeping here to upsample, crop the pad, and resize back to (H,W).

#include "brotensor/tensor.h"

#include <cstdint>
#include <vector>

namespace brovisionml::openpose {

// Maps between the detect-resolution image and the padded network input.
struct OpenposeTransform {
    int detect_w = 0;     // oriImg width  (W, after resize_image)
    int detect_h = 0;     // oriImg height (H)
    int test_w = 0;       // imageToTest width  (round(W*scale))
    int test_h = 0;       // imageToTest height (round(H*scale))
    int pad_w = 0;        // padded width  (Wp, multiple of 8)
    int pad_h = 0;        // padded height (Hp)
    int pad_down = 0;     // rows padded at bottom
    int pad_right = 0;    // cols padded at right
    float scale = 0.0f;   // 0.5 * 368 / H
};

struct PreprocessedImage {
    // (1, 3*pad_h*pad_w) FP32, host-resident, NCHW, BGR order, padded/256 - 0.5.
    brotensor::Tensor  pixels;
    OpenposeTransform  transform;
};

// Resize an 8-bit interleaved HWC image (channels 1/3/4) to detect resolution
// per resize_image, returning the detect-res RGB bytes (HWC, 3ch). `out_w` /
// `out_h` receive the detect-res dims. Throws std::runtime_error (prefix
// "openpose::preprocess: ") on a zero-size image or unsupported channel count.
std::vector<uint8_t> resize_to_detect(const uint8_t* rgb, int w, int h,
                                      int channels, int detect_resolution,
                                      int& out_w, int& out_h);

// Build the padded NCHW network-input tensor from detect-res RGB bytes (HWC,
// 3ch, W=detect_w H=detect_h). Applies BGR flip + smart resize + pad + normalize.
PreprocessedImage preprocess_from_detect(const uint8_t* rgb_detect,
                                         int detect_w, int detect_h);

// Convenience: resize to detect resolution then build the network input.
PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int detect_resolution = 512);

}  // namespace brovisionml::openpose
