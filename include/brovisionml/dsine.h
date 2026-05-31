#pragma once

// DSINE — surface-normal estimation, end to end. Ties the four DSINE stages
// (preprocess, EfficientNet-B5 encoder, DPT-style decoder, AngMF/ConvGRU
// refinement) into a single estimator that loads one DSINE_v02 `model.safetensors`
// checkpoint and maps an image to a dense per-pixel surface-normal map at the
// original resolution.
//
//   estimate(): preprocess (zero-pad to a multiple of 32, ImageNet normalize, and
//   synthesize camera intrinsics from an assumed field-of-view) -> EncoderB5
//   feature taps -> Decoder (initial normal + feature/hidden) -> Refiner (5
//   iterations of rotation-based angular mean-field smoothing + convex upsample)
//   -> crop back to the input size.
//
// DSINE is geometry-conditioned: every decoder/refine stage is fed a uv/ray map
// built from the camera intrinsics. With no calibration available the model
// assumes a field-of-view (default 60°): a centered principal point and a focal
// length set so the longer image side subtends `fov_deg`. An overload accepts
// explicit pinhole intrinsics (fx,fy,cx,cy) when they are known.
//
// The output is a unit surface normal per pixel in CAMERA space — the same
// convention DSINE outputs. The three components are planar NCHW (3,H,W):
// channel 0 = nx, 1 = ny, 2 = nz. Each pixel's (nx,ny,nz) is L2-normalized.
// Visualize by mapping to RGB with (n+1)*0.5.
//
// CPU only: the four DSINE stages run host-side FP32 for parity with the
// reference. There is no CUDA path here (unlike DepthEstimator).

#include "brovisionml/dsine_decoder.h"
#include "brovisionml/dsine_encoder.h"
#include "brovisionml/dsine_preprocess.h"
#include "brovisionml/dsine_refine.h"

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace brovisionml::dsine {

// DSINE ships a single official checkpoint (DSINE_v02 / EfficientNet-B5), so
// there are no small/base/large variants — just the assumed field-of-view used
// to synthesize intrinsics when none are supplied.
struct DsineConfig {
    float fov_deg = 60.0f;   // assumed FOV for the synthesized intrinsics
};

// A dense surface-normal map at the original image resolution. `normals` holds
// 3*height*width FP32 in planar NCHW (3,H,W): channel 0 = nx, 1 = ny, 2 = nz.
// Each pixel's (nx,ny,nz) is an L2-normalized UNIT normal in CAMERA space (the
// DSINE convention). Visualize as (n+1)*0.5.
struct NormalMap {
    int width  = 0;             // == original image width
    int height = 0;             // == original image height
    std::vector<float> normals; // 3*height*width, planar NCHW (3,H,W), unit normals

    // The c-th normal component (0=nx,1=ny,2=nz) at pixel (x,y).
    float at(int c, int x, int y) const {
        const std::size_t plane = static_cast<std::size_t>(height) * width;
        return normals[static_cast<std::size_t>(c) * plane +
                       static_cast<std::size_t>(y) * width + x];
    }
};

class NormalEstimator {
public:
    explicit NormalEstimator(DsineConfig cfg = {});
    ~NormalEstimator();
    NormalEstimator(NormalEstimator&&) noexcept;
    NormalEstimator& operator=(NormalEstimator&&) noexcept;

    // Load encoder + decoder + refiner from one checkpoint directory / file. The
    // single DSINE_v02 `model.safetensors` carries the encoder, decoder, and
    // refinement (gru/*_head) tensor namespaces.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    const DsineConfig& config() const { return cfg_; }

    // Estimate surface normals for an 8-bit interleaved HWC image (channels
    // 1/3/4). Intrinsics are synthesized from `cfg_.fov_deg`.
    NormalMap estimate(const uint8_t* rgb, int w, int h, int channels) const;

    // Estimate with explicit pinhole intrinsics (fx,fy focal lengths; cx,cy
    // principal point, in the reference's pre-"+0.5" convention on the ORIGINAL
    // unpadded image). The principal point is shifted by the pad offset
    // internally to stay aligned to content.
    NormalMap estimate(const uint8_t* rgb, int w, int h, int channels,
                       float fx, float fy, float cx, float cy) const;

private:
    NormalMap run(const PreprocessedImage& pp) const;

    DsineConfig cfg_;
    EncoderB5   encoder_;
    Decoder     decoder_;
    Refiner     refiner_;
};

}  // namespace brovisionml::dsine
