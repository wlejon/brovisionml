#pragma once

// SAM (Segment Anything) automatic mask generator — the "segment everything"
// pipeline. Port of segment_anything's `SamAutomaticMaskGenerator`. Where the
// `Sam` orchestrator answers a single prompt, this drives the model with a
// dense regular grid of point prompts and post-processes the union of their
// predictions into a clean, deduplicated set of object masks for the whole
// image — no user clicks required.
//
// It is pure host-side orchestration layered on top of `Sam` (set_image /
// segment); it adds no new tensor ops. The pipeline, faithful to upstream:
//
//   1. Lay down a `points_per_side`^2 regular grid over the image (and, for
//      `crop_n_layers > 0`, over a pyramid of overlapping image crops, with a
//      coarser grid per layer).
//   2. For every grid point, run a single-foreground-point segment() to get the
//      3 multimask proposals + their predicted IoU.
//   3. Drop proposals below `pred_iou_thresh`, then below `stability_score_thresh`
//      (a measure of how little the binarized mask changes when the logit
//      threshold is nudged by ±`stability_score_offset`).
//   4. Threshold to binary, take the bounding box, and discard masks that hug a
//      crop edge that is not also the image edge.
//   5. Greedy box-NMS within each crop (`box_nms_thresh`), then across crops
//      (`crop_nms_thresh`, preferring masks from the smaller/finer crops).
//   6. Optionally remove connected components and holes smaller than
//      `min_mask_region_area` and re-run NMS to drop the duplicates that fix-up
//      creates.
//
// Coordinates in and out are ORIGINAL-image pixels; crops are taken from the
// raw pixel buffer passed to generate(). The model decodes one grid point at a
// time (the segment() prompt path treats multiple points as one multi-point
// prompt, not a batch of independent prompts), so this is intentionally
// correctness-first; batching independent point prompts through the decoder is
// a future optimization.

#include "brovisionml/sam.h"

#include <array>
#include <cstdint>
#include <vector>

namespace brovisionml::sam {

// Automatic-mask-generator hyperparameters. Defaults mirror upstream
// SamAutomaticMaskGenerator (the values used to produce the SA-1B masks).
struct AmgConfig {
    int   points_per_side               = 32;          // grid is this squared
    float pred_iou_thresh               = 0.88f;       // drop masks below this predicted IoU
    float stability_score_thresh        = 0.95f;       // drop masks below this stability
    float stability_score_offset        = 1.0f;        // logit nudge for the stability measure
    float box_nms_thresh                = 0.7f;        // within-crop box-NMS IoU threshold
    int   crop_n_layers                 = 0;           // 0 = whole image only; N adds 2^i crops per layer
    float crop_nms_thresh               = 0.7f;        // across-crop box-NMS IoU threshold
    float crop_overlap_ratio            = 512.0f / 1500.0f;  // crop overlap as a fraction of the short side
    int   crop_n_points_downscale_factor = 1;          // grid side divides by this^layer per crop layer
    int   min_mask_region_area          = 0;           // 0 = off; else fill holes / drop islands below this area
};

// One generated object mask, in ORIGINAL-image pixel space.
struct GeneratedMask {
    std::vector<uint8_t> mask;    // height*width, row-major; 1 = foreground, 0 = background
    int height = 0;
    int width  = 0;

    std::array<int, 4>   bbox{};  // {x, y, w, h} tight bounding box of the mask
    long long            area = 0;  // foreground pixel count

    float predicted_iou   = 0.f;  // the model's IoU score for this mask
    float stability_score = 0.f;  // ±offset stability of the binarized mask

    std::array<float, 2> point{};     // the grid point that produced this mask
    std::array<int, 4>   crop_box{};  // {x, y, w, h} of the crop it came from
};

// Drives a loaded `Sam` to produce a full set of masks for an image. The model
// must already be load()ed (and to()-migrated to the desired device); the
// generator does not own it. generate() calls Sam::set_image repeatedly, so it
// leaves an image cached on the model when it returns.
class AutomaticMaskGenerator {
public:
    explicit AutomaticMaskGenerator(Sam& model, AmgConfig cfg = {});

    // Generate masks for an 8-bit interleaved HWC image (same pixel contract as
    // Sam::set_image: `channels` is 1 / 3 / 4). Returns masks sorted by
    // descending area. Throws std::runtime_error (prefix
    // "sam::AutomaticMaskGenerator: ") on an invalid image or config.
    std::vector<GeneratedMask> generate(const uint8_t* pixels, int w, int h,
                                        int channels);

    const AmgConfig& config() const { return cfg_; }

private:
    Sam&      model_;
    AmgConfig cfg_;
};

}  // namespace brovisionml::sam
