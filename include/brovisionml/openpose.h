#pragma once

// OpenPose body-pose estimation, end to end. The ControlNet "openpose"
// conditioning annotator: the CMU multi-person 2D body-pose network (the
// COCO-18 keypoint / 19-limb PAF model) as packaged by pytorch-openpose and
// controlnet_aux. Pixels in, a set of people (each up to 18 keypoints) out;
// the CLI tool rasterizes them to the canonical openpose control image.
//
// BODY-ONLY SCOPE. The reference OpenposeDetector defaults to
// `include_hand=False, include_face=False` for the ControlNet control image, so
// only the body network is implemented here. The hand and face sub-networks
// (handpose_model / a separate face net) are intentionally not ported.
//
// Architecture (bodypose_model): a VGG-style trunk `model0` (10 convs + 3
// maxpools, 3->128ch at /8 spatial) followed by 6 two-branch refinement stages.
// Branch L1 emits a 38-channel Part-Affinity-Field map, branch L2 a 19-channel
// confidence/heatmap (18 parts + background). Stage 1 takes the trunk feature;
// stages 2..6 each take cat([prev_L1, prev_L2, trunk]) = 185ch and apply five
// 7x7 convs then two 1x1 convs. Every conv has bias + in-place ReLU except the
// final Mconv7 / conv5_5 of each branch. The forward is a pure composition of
// brotensor ops (conv2d_forward + relu_forward + max_pool2d_forward +
// concat_nchw_channels) — no OpenPose-specific kernel.
//
// Decode (host, ported from body.py): the heatmap/PAF are upsampled x8
// (Lanczos), the pad cropped, and resized to detect resolution; per-part peak
// detection (Gaussian blur sigma=3 + local-max NMS + thre1) yields candidate
// keypoints; PAF line-integral scoring + greedy bipartite matching connects
// limbs; a subset-merge assembles people; rows with < 4 parts or mean score
// < 0.4 are pruned. Keypoints are normalized to [0,1] over the detect-res image.
//
// Device: like the other detectors, call to(Device) after load() to run on a
// device; detect() preprocesses on the host, uploads pixels, runs the net
// on-device, copies the raw maps back, and decodes on the host.

#include "brovisionml/openpose_preprocess.h"

#include "brotensor/tensor.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brovisionml::openpose {

struct OpenposeConfig {
    int   detect_resolution = 512;   // shorter side, rounded to a multiple of 64
    float thre1 = 0.1f;              // heatmap peak threshold
    float thre2 = 0.05f;             // PAF mid-point threshold
};

// One body keypoint; x,y are normalized [0,1] over the detect-res image.
struct Keypoint {
    float x = -1, y = -1, score = 0;
    bool  present = false;
};

// One detected person: the 18 COCO-ordered body keypoints (some may be absent).
struct BodyPose {
    std::array<Keypoint, 18> keypoints;
    float total_score = 0;
    int   total_parts = 0;
};

struct PoseResult {
    int width = 0, height = 0;       // detect-res canvas dims
    std::vector<BodyPose> bodies;
};

// The raw stage-6 network outputs at network resolution (Hp/8, Wp/8) — exposed
// for testing/debugging (the tight conv-parity gate). PAF is out6_1 (38ch),
// heatmap is out6_2 (19ch); both NCHW (no batch).
struct PafHeatmap {
    int paf_c = 38, hm_c = 19, height = 0, width = 0;
    std::vector<float> paf;       // paf_c*height*width
    std::vector<float> heatmap;   // hm_c*height*width

    float paf_at(int c, int y, int x) const {
        return paf[(static_cast<std::size_t>(c) * height + y) * width + x];
    }
    float hm_at(int c, int y, int x) const {
        return heatmap[(static_cast<std::size_t>(c) * height + y) * width + x];
    }
};

class OpenposeDetector {
public:
    explicit OpenposeDetector(OpenposeConfig cfg = {});
    ~OpenposeDetector();
    OpenposeDetector(OpenposeDetector&&) noexcept;
    OpenposeDetector& operator=(OpenposeDetector&&) noexcept;

    // Load the body network from a checkpoint directory / file (one
    // `model.safetensors` carrying `model0.* model1_1.* ... model6_2.*`).
    void load(const std::string& dir);
    void load_file(const std::string& path);

    void to(brotensor::Device dev);
    brotensor::Device device() const;

    const OpenposeConfig& config() const { return cfg_; }

    // Run the network and return the raw stage-6 PAF + heatmap (no postproc).
    PafHeatmap infer_maps(const uint8_t* rgb, int w, int h, int channels) const;

    // Full pipeline: preprocess -> net -> host decode -> people.
    PoseResult detect(const uint8_t* rgb, int w, int h, int channels) const;

    // Rasterize a PoseResult to a HxWx3 RGB canvas (draw_bodypose).
    static std::vector<uint8_t> draw(const PoseResult& r);

private:
    PafHeatmap run(const PreprocessedImage& pp) const;

    OpenposeConfig cfg_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brovisionml::openpose
