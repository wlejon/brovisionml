#pragma once

// MLSD — straight line-segment detection, end to end. The ControlNet "mlsd"
// conditioning annotator: M-LSD (Mobile Line Segment Detection, NAVER, 2021),
// the `MobileV2_MLSD_Large` network lllyasviel ships. A truncated MobileNetV2
// backbone (4-channel input = RGB + a constant plane; five FPN taps) feeds an
// FPN-style decode head; the final 16-channel conv output is sliced to its last
// 9 channels as a "TP map" at half resolution (256×256 for the 512×512 input).
//
// TP-map decode (`pred_lines`): channel 0 is a center heatmap (sigmoid → 3×3
// max-pool non-max-suppression → top-K), channels 1:5 are per-pixel
// start/end displacement vectors. A line segment is reconstructed at each kept
// center as (x+dx_s, y+dy_s)→(x+dx_e, y+dy_e), kept when its score and length
// clear the thresholds, then scaled from the 256 grid back to the original image.
//
// The forward is a pure composition of brotensor ops with no MLSD-specific
// kernel: grouped/dilated `conv2d` (depthwise inverted-residuals + the dilated
// BlockTypeC), `batch_norm` folded into each preceding conv at load, ReLU6 via
// `clamp`, align-corners bilinear `interp2d` (the FPN upsamples), and
// `concat_nchw_channels` (the skip merges). The center NMS reuses `max_pool2d`.
//
// Device: like the other detectors, call to(Device) after load() to run on a
// device; detect() preprocesses on the host, uploads the pixels, runs the net
// on-device, copies the TP map back, and decodes segments on the host.

#include "brovisionml/mlsd_preprocess.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brovisionml::mlsd {

struct MlsdConfig {
    float score_thr = 0.1f;   // minimum center score (post-sigmoid) to keep
    float dist_thr  = 0.1f;   // minimum segment length (in the 256 TP grid)
    int   topk      = 200;    // top-K center candidates from the heatmap
    int   model_size = 512;   // fixed square model input
};

// A detected line segment in ORIGINAL image coordinates (pixels).
struct LineSegment {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float score = 0;          // center-heatmap score in [0,1]
};

struct LineMap {
    int width  = 0;           // original image width
    int height = 0;           // original image height
    std::vector<LineSegment> segments;
};

// The raw network output (the 9-channel TP map) — exposed for testing/debugging
// and for callers that want to run their own decode.
struct TpMap {
    int channels = 0, height = 0, width = 0;   // (9, 256, 256) for a 512 input
    std::vector<float> data;                    // channels*height*width, NCHW

    float at(int c, int y, int x) const {
        return data[(static_cast<std::size_t>(c) * height + y) * width + x];
    }
};

class MLSDdetector {
public:
    explicit MLSDdetector(MlsdConfig cfg = {});
    ~MLSDdetector();
    MLSDdetector(MLSDdetector&&) noexcept;
    MLSDdetector& operator=(MLSDdetector&&) noexcept;

    // Load the network from a checkpoint directory / file (one MLSD
    // `model.safetensors` carrying `backbone.features.*` + `block15..23.*`).
    void load(const std::string& dir);
    void load_file(const std::string& path);

    void to(brotensor::Device dev);
    brotensor::Device device() const;

    const MlsdConfig& config() const { return cfg_; }

    // Run the network and return the raw TP map (no decode).
    TpMap infer_tpmap(const uint8_t* rgb, int w, int h, int channels) const;

    // Detect line segments for an 8-bit interleaved HWC image (channels 1/3/4).
    LineMap detect(const uint8_t* rgb, int w, int h, int channels) const;

private:
    TpMap run(const PreprocessedImage& pp) const;

    MlsdConfig cfg_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brovisionml::mlsd
