#pragma once

// SAM (Segment Anything) — the assembled, runnable model. Ties the three
// pieces together into the encode-image-once / decode-many-prompts workflow:
//
//   set_image()  -> sam::preprocess + ImageEncoder         (the slow half)
//   segment()    -> PromptEncoder + MaskDecoder + upscale   (cheap per click)
//
// A single HuggingFace `SamModel` checkpoint (`model.safetensors`, carrying
// vision_encoder.* / prompt_encoder.* / mask_decoder.* together) loads all
// three sub-modules. Prompts are given in ORIGINAL-image pixel coordinates;
// Sam maps them into the model's 1024-square input space, and maps the
// predicted masks back out to the original image resolution.

#include "brovisionml/sam_image_encoder.h"
#include "brovisionml/sam_prompt_encoder.h"
#include "brovisionml/sam_mask_decoder.h"
#include "brovisionml/sam_preprocess.h"

#include "brotensor/tensor.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace brovisionml::sam {

// The three sub-module configs. The prompt encoder and mask decoder are shared
// across SAM sizes; only the ViT encoder differs, so the presets just swap it.
struct SamConfig {
    EncoderConfig       encoder;
    PromptEncoderConfig prompt;
    MaskDecoderConfig   decoder;

    static SamConfig vit_h();
    static SamConfig vit_l();
    static SamConfig vit_b();
};

// Segmentation result: `num` masks at the ORIGINAL image resolution, as FP32
// logits (threshold at 0 for a binary mask), each `height*width` row-major,
// plus the model's predicted IoU quality score per mask.
struct Segmentation {
    std::vector<float> logits;   // num * height * width
    std::vector<float> iou;      // num
    int num    = 0;
    int height = 0;
    int width  = 0;

    // Index of the highest-IoU mask (0 if empty).
    int best() const;
};

class Sam {
public:
    explicit Sam(SamConfig cfg);

    Sam(Sam&&) noexcept            = default;
    Sam& operator=(Sam&&) noexcept = default;
    Sam(const Sam&)                = delete;
    Sam& operator=(const Sam&)     = delete;

    // Load all three sub-modules from one HF SAM checkpoint. `load` expects
    // `dir/model.safetensors`; `load_file` reads one explicit path. Throws
    // std::runtime_error naming the first missing/mismatched tensor.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Migrate the whole model onto `dev`. set_image()/segment() then run there.
    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }

    // Encode an image once. `pixels` is an 8-bit interleaved HWC buffer with
    // `channels` 1 (gray), 3 (RGB), or 4 (RGBA). Caches the dense image
    // embedding and the preprocessing transform for subsequent segment() calls.
    void set_image(const uint8_t* pixels, int w, int h, int channels);
    bool has_image() const { return has_image_; }

    // Segment the current image. points/labels (label 1 = foreground click,
    // 0 = background click) and boxes ({x1,y1,x2,y2}) are in ORIGINAL-image
    // pixel coords. multimask_output=true returns the three multi-mask
    // proposals (ranked-by-IoU selectable via Segmentation::best()); false
    // returns the single best mask. Throws if no image has been set.
    Segmentation segment(const std::vector<std::array<float, 2>>& points,
                         const std::vector<int>& labels,
                         const std::vector<std::array<float, 4>>& boxes,
                         bool multimask_output = true) const;

    const SamConfig& config() const { return cfg_; }

private:
    SamConfig     cfg_;
    ImageEncoder  enc_;
    PromptEncoder pe_;
    MaskDecoder   dec_;

    brotensor::Device device_ = brotensor::Device::CPU;

    // Cached by set_image().
    brotensor::Tensor image_embedding_;   // (1, D*grid*grid) on device_
    ImageTransform    transform_{};
    bool              has_image_ = false;
};

}  // namespace brovisionml::sam
