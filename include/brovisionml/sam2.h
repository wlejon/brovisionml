#pragma once

// SAM 2.1 (Meta, Apache-2.0): promptable segmentation of one image, and the
// same model tracking one object through a sequence of frames.
//
//   Sam2         image predictor — set_image() once, segment() per prompt.
//   Sam2Tracker  sequence tracker over caller-supplied frames — add_frame()
//                per frame (encodes it), prompt() an object on any frame,
//                track() / propagate() it through the rest, prompt() again on
//                a tracked frame to correct it.
//
// Checkpoints are the Hugging Face transformers layout (Sam2VideoModel /
// Sam2Model): a directory with config.json and model.safetensors; the four
// published sizes (tiny / small / base-plus / large) load through one code
// path. Coordinates are input-image pixels throughout; an image is resized to
// the model's square input (1024) without keeping its aspect ratio, as the
// reference predictor does.
//
// GPU by default, like the other brovisionml models: to(dev) on a GPU whose
// compute dtype is FP16 runs the encoder and the memory attention in mixed
// precision; migrating under a CPU DeviceScope keeps everything exact FP32.

#include "brovisionml/sam2_config.h"
#include "brovisionml/sam2_decoder.h"
#include "brovisionml/sam2_image_encoder.h"
#include "brovisionml/sam2_memory.h"

#include "brotensor/tensor.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace brovisionml {

struct Sam2Prompt {
    std::vector<std::array<float, 2>> points;   // input-image pixels
    std::vector<int> labels;                    // 1 = foreground, 0 = background
    bool has_box = false;
    std::array<float, 4> box{};                 // x0, y0, x1, y1 in pixels
};

// Masks at the input image's resolution.
struct Sam2Segmentation {
    int num = 0, width = 0, height = 0;
    std::vector<float> logits;     // num * height * width, threshold at 0
    std::vector<float> iou;        // num, predicted IoU in [0, 1]
    float object_score = 0.0f;     // logit, > 0 = object present
    // (num, 256*256) low-res logits on the model's device — feed one row back
    // as Sam2::segment's mask_input to refine.
    brotensor::Tensor low_res;
    int best() const;
};

// Bounding box of (logits > 0) in a height x width plane: x0, y0 inclusive,
// x1, y1 exclusive. Returns false (box zeroed) when the mask is empty.
bool sam2_mask_box(const float* logits, int width, int height, std::array<int, 4>& box);

class Sam2 {
public:
    explicit Sam2(sam2::Sam2Config cfg = sam2::Sam2Config::base_plus());
    Sam2(Sam2&&) noexcept = default;
    Sam2& operator=(Sam2&&) noexcept = default;
    Sam2(const Sam2&) = delete;
    Sam2& operator=(const Sam2&) = delete;

    // `dir` holds config.json (optional: the constructor's config stands
    // without one) and model.safetensors.
    void load(const std::string& dir);
    void to(brotensor::Device dev);
    brotensor::Device device() const { return encoder_.device(); }
    const sam2::Sam2Config& config() const { return cfg_; }

    // RGB / RGBA / gray, row-major 8-bit.
    void set_image(const uint8_t* pixels, int w, int h, int channels);
    bool has_image() const { return !features_.empty(); }

    // Prompt the current image. points/labels and/or a box; `mask_input` is an
    // optional (1, 256*256) logits row on device() (a previous low_res). With
    // `multimask` three proposals come back, else one (with SAM 2's stability
    // fallback to the best proposal when the single mask is unstable).
    Sam2Segmentation segment(const Sam2Prompt& prompt, bool multimask = true,
                             const brotensor::Tensor* mask_input = nullptr) const;

    // ── Building blocks (the tracker, tests) ──
    // (1, 3*S*S) FP32 normalized NCHW on the host.
    brotensor::Tensor preprocess(const uint8_t* pixels, int w, int h, int channels) const;
    // The same on device(): resized there when the image is enlarged, on the
    // host when it is reduced.
    brotensor::Tensor preprocess_device(const uint8_t* pixels, int w, int h, int channels) const;
    sam2::ImageFeatures encode(const uint8_t* pixels, int w, int h, int channels) const;
    // Model-input points + labels for a prompt on a w x h image: box corners
    // (labels 2, 3) first, then the clicks.
    void map_prompt(const Sam2Prompt& prompt, int w, int h,
                    std::vector<std::array<float, 2>>& points, std::vector<int>& labels) const;
    // (num, 256^2) device logits -> host (num, h*w) logits, bilinear.
    std::vector<float> upscale(const brotensor::Tensor& low_res, int num, int w, int h) const;
    const sam2::ImageFeatures& features() const { return features_; }

    const sam2::ImageEncoder& encoder() const { return encoder_; }
    const sam2::SamHeads& heads() const { return heads_; }
    const sam2::MemoryModule& memory() const { return memory_; }

private:
    sam2::Sam2Config cfg_;
    sam2::ImageEncoder encoder_;
    sam2::SamHeads heads_;
    sam2::MemoryModule memory_;
    sam2::ImageFeatures features_;
    int image_w_ = 0, image_h_ = 0;
    mutable brotensor::Tensor norm_w_, norm_b_;   // device normalize, made on first use
};

// One frame's tracking result, at that frame's input resolution.
struct Sam2FrameResult {
    int frame = -1, width = 0, height = 0;
    std::vector<float> logits;     // height * width
    float iou = 0.0f;
    float object_score = 0.0f;
    bool present = false;          // object_score > 0
    bool conditioning = false;     // a prompted frame
    std::array<int, 4> box{};      // of logits > 0; see sam2_mask_box
    bool has_box = false;
};

// Tracks one object through frames the caller adds. The model is borrowed and
// must outlive the tracker. Frames keep their encoded features (about 16 MB
// each at 1024) so re-prompting and re-propagating never re-encode.
class Sam2Tracker {
public:
    explicit Sam2Tracker(const Sam2& model);

    // Encode a frame; returns its index (frames are numbered in add order,
    // which is the sequence order the memory's temporal encodings see).
    int add_frame(const uint8_t* pixels, int w, int h, int channels);
    int num_frames() const { return static_cast<int>(frames_.size()); }

    // Prompt the object on `frame`. On a frame that was never tracked this is
    // an independent prediction (a fresh start); on a tracked frame it is a
    // correction conditioned on the memory and the frame's previous mask. The
    // frame becomes a conditioning frame either way; every call replaces that
    // frame's earlier clicks.
    Sam2FrameResult prompt(int frame, const Sam2Prompt& prompt);

    // Track the object into `frame` from its memories (the frames before it,
    // or after it when `reverse`). A conditioning frame returns its result.
    Sam2FrameResult track(int frame, bool reverse = false);

    // From the earliest prompted frame forward to the last frame, then back
    // to frame 0 when the earliest prompt is past it. Returns every frame.
    std::vector<Sam2FrameResult> propagate();

    // The latest result for `frame` (empty `logits` if none yet).
    Sam2FrameResult result(int frame) const;

    // Forget prompts and tracking; keep the encoded frames.
    void reset();

private:
    struct Frame {
        int w = 0, h = 0;
        sam2::ImageFeatures feats;
        bool has_output = false, cond = false, tracked = false, reverse = false;
        brotensor::Tensor low_res;    // (1, 256^2) device logits, -1024 where absent
        brotensor::Tensor obj_ptr;    // (1, hidden)
        brotensor::Tensor memory;     // (grid^2, mem_dim); empty = not encoded yet
        float iou = 0.0f, object_score = 0.0f;
    };
    struct Step {
        brotensor::Tensor low_res, obj_ptr;
        float iou = 0.0f, object_score = 0.0f;
    };

    Step run_frame(int frame, bool is_init, const std::vector<std::array<float, 2>>& points,
                   const std::vector<int>& labels, const brotensor::Tensor* mask_input,
                   bool reverse) const;
    void preflight();
    void check(int frame) const;
    Sam2FrameResult make_result(int frame) const;

    const Sam2* model_;
    std::vector<Frame> frames_;
};

}  // namespace brovisionml
