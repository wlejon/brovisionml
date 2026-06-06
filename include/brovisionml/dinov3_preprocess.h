#pragma once

// DINOv3 ViT-H image preprocessing — the input contract for dinov3::Backbone.
// Unlike the DPT front-end (aspect-preserving resize to a multiple of the patch
// size) this resizes to a fixed SQUARE so the patch grid is a clean
// (resolution/patch_size)². That single `resolution` is the speed/quality dial
// for the whole encoder: DINOv3 ViT-H is a fixed cost that scales ~linearly
// (projections) and ~quadratically (attention) with the patch-token count, and
// it runs once per image regardless of any downstream sampler steps. Dropping it
// — 1024 → 768 is ~0.56× tokens, 1024 → 512 ~0.25× — is the cheapest lever on
// the encoder's wall-clock, independent of the FP16 path, at the usual ballpark
// cost in feature fidelity.
//
//   1. Square resize to (resolution, resolution), bicubic (Catmull-Rom == PIL
//      BICUBIC). NO aspect-ratio preservation and NO padding — the object-centric
//      inputs DINOv3 feeds are squared directly, matching the encoder's fixed
//      grid. `resolution` must be a multiple of `patch_size` (16).
//   2. Rescale [0,255] -> [0,1].
//   3. Normalize with ImageNet mean/std (`broimage::IMAGENET_MEAN/STD`), as
//      DINOv2/DINOv3 do.
//
// The model input is a flat (1, 3*resolution*resolution) FP32 tensor in NCHW
// order. Pure host-side; the caller uploads `pixels` to the compute device
// before invoking the backbone.

#include "brotensor/tensor.h"

#include <cstdint>

namespace brovisionml::dinov3 {

// Default square resolution the encoder runs at. The patch grid is
// resolution/patch_size on a side (64×64 at 1024/16). Lower it for speed.
inline constexpr int kDefaultResolution = 1024;

// Maps between the original image and the square model input. The resize is a
// straight stretch to (resolution × resolution); there is no padding, so the
// whole plane is content and the (sx, sy) scales recover original coordinates.
struct Dinov3Transform {
    int   orig_w     = 0;
    int   orig_h     = 0;
    int   resolution = 0;   // square side the model sees (multiple of patch_size)
    float scale_x    = 0.f; // resolution / orig_w
    float scale_y    = 0.f; // resolution / orig_h
};

struct PreprocessedImage {
    // (1, 3*resolution*resolution) FP32, host-resident, NCHW, ImageNet-normalized.
    brotensor::Tensor pixels;
    Dinov3Transform    transform;
};

// Preprocess an 8-bit interleaved HWC image for the DINOv3 backbone.
//   rgb        : pixel buffer, `w*h*channels` bytes.
//   channels   : 1 (gray, replicated to RGB), 3 (RGB), or 4 (RGBA, alpha dropped).
//   resolution : square side fed to the encoder (the speed/quality knob; default
//                kDefaultResolution). Must be a positive multiple of patch_size.
//   patch_size : the backbone's patch size (16); resolution must be divisible by it.
// Throws std::runtime_error (prefix "dinov3::preprocess: ") on a zero-size image,
// an unsupported channel count, or a resolution that is not a positive multiple
// of patch_size.
PreprocessedImage preprocess(const uint8_t* rgb, int w, int h, int channels,
                             int resolution = kDefaultResolution,
                             int patch_size = 16);

}  // namespace brovisionml::dinov3
