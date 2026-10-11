#pragma once

// SAM 2 prompt encoder + mask decoder — the "SAM heads" shared by the image
// predictor and the tracker.
//
// Prompts follow the reference implementation's convention: a box is two
// corner points labelled 2 (top-left) and 3 (bottom-right) placed before the
// clicks, and the prompt encoder always appends one padding point (label -1).
// Labels: 1 = foreground click, 0 = background click, 2/3 = box corners.
//
// The decoder is SAM's two-way transformer with SAM 2's additions: an object-
// score token in front of the IoU token, the stride-4 / stride-8 encoder maps
// added into the upscaler, sigmoid IoU predictions, and the single-mask
// stability fallback (when one mask is asked for and mask 0 is unstable, the
// best of the three multimask proposals stands in).

#include "brovisionml/sam2_config.h"

#include "brotensor/tensor.h"

#include <array>
#include <memory>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brovisionml::sam2 {

struct HeadsPrompt {
    // Model-input coordinates (0..image_size), corners first if a box. Empty
    // means no sparse prompt tokens at all; the tracker's prompt-free frames
    // pass one (0, 0) point labelled -1, as the reference does.
    std::vector<std::array<float, 2>> points;
    std::vector<int> labels;
    // Optional dense prompt: (1, mask_size^2) low-res mask logits on the
    // heads' device (a previous prediction), or nullptr.
    const brotensor::Tensor* mask_input = nullptr;
};

struct HeadsOutput {
    int num = 0;                   // masks returned (3 multimask, else 1)
    brotensor::Tensor low_res;     // (num, mask_size^2) FP32 logits on device
    brotensor::Tensor sam_tokens;  // (num, hidden) FP32 output tokens on device
    std::vector<float> iou;        // num, in [0, 1]
    float object_score = 0.0f;     // logit; > 0 means the object is present
};

struct SamHeadsWeights;

class SamHeads {
public:
    explicit SamHeads(const Sam2Config& cfg);
    ~SamHeads();
    SamHeads(SamHeads&&) noexcept;
    SamHeads& operator=(SamHeads&&) noexcept;
    SamHeads(const SamHeads&) = delete;
    SamHeads& operator=(const SamHeads&) = delete;

    void load(const brotensor::safetensors::File& f);   // prompt_encoder.*, mask_decoder.*
    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }

    // `image`: (grid*grid, hidden) tokens — the stride-16 map with either the
    // no-memory embedding or the memory attention already applied. feat_s0 /
    // feat_s1 from ImageEncoder::encode. All on device().
    HeadsOutput decode(const brotensor::Tensor& image, const brotensor::Tensor& feat_s0,
                       const brotensor::Tensor& feat_s1, const HeadsPrompt& prompt,
                       bool multimask) const;

private:
    Sam2Config cfg_;
    std::unique_ptr<SamHeadsWeights> w_;
    brotensor::Device device_ = brotensor::Device::CPU;
};

}  // namespace brovisionml::sam2
