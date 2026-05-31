#pragma once

// SAM (Segment Anything) mask decoder — the "decode-many-prompts" half. Given
// the image encoder's dense embedding and the prompt encoder's sparse + dense
// prompt embeddings, it predicts a small set of mask logits and per-mask IoU
// quality scores. Port of segment_anything's MaskDecoder + TwoWayTransformer,
// matched against the HuggingFace `SamModel` `mask_decoder.*` layout.
//
// Pipeline (single image, single prompt set):
//
//   tokens = [iou_token, mask_tokens(0..K-1), sparse_prompt_tokens]
//   keys   = flatten(image_embedding + dense_prompt_embedding)        (grid^2, D)
//   two-way transformer (num_hidden_layers blocks + a final t->image attn):
//     per block: self-attn on tokens; cross token->image; MLP;
//                cross image->token (updates keys). Query positional embedding
//                (the tokens themselves) is re-added before every attention,
//                and the image positional encoding before every image attention
//                — both added to the attention *inputs*, pre-projection, which
//                is why this can't reuse a bundled QKV-projecting attention op.
//   upscale keys 4x (two stride-2 transposed convs + a channels-first LN),
//   a per-mask-token hypernetwork MLP produces a (D/8) filter, and
//   mask_logits = filters @ upscaled  (one (4*grid x 4*grid) map per token).
//   iou_scores  = iou_prediction_head(iou_token_out).
//
// The core scaled-dot-product attention runs through brotensor's
// flash_attention_varlen_forward over pre-projected Q/K/V (device-neutral:
// FP32 on CPU, FP16/FP32 on GPU); projections are brotensor linear layers and
// the upscaler is conv_transpose2d, so the whole decode is device-neutral and
// runs wherever the image embedding lives.

#include "brotensor/tensor.h"

#include <memory>
#include <string>

namespace brovisionml::sam {

// Mask-decoder hyperparameters. Defaults mirror HuggingFace's
// SamMaskDecoderConfig for the stock facebook/sam-* checkpoints (shared across
// the three ViT sizes).
struct MaskDecoderConfig {
    int hidden_size             = 256;   // token / image-embed channel dim (D)
    int image_embedding_size    = 64;    // image-feature grid side
    int num_multimask_outputs   = 3;     // -> num_mask_tokens = this + 1
    int num_hidden_layers       = 2;     // two-way transformer blocks
    int num_attention_heads     = 8;
    int attention_downsample_rate = 2;   // cross/final attn internal_dim = D/rate
    int mlp_dim                 = 2048;  // two-way block MLP hidden width
    int iou_head_depth          = 3;
    int iou_head_hidden_dim     = 256;
    float layer_norm_eps        = 1e-6f;

    int num_mask_tokens() const { return num_multimask_outputs + 1; }
    int grid()            const { return image_embedding_size; }      // 64
    int mask_size()       const { return image_embedding_size * 4; }  // 256
};

// Predicted masks for one prompt set.
struct DecodedMasks {
    // (num_out, mask_size*mask_size) FP32 mask LOGITS on device(); row-major
    // per mask (a 4*grid square). Threshold at 0 for a binary mask.
    brotensor::Tensor masks;
    // (num_out, 1) FP32 predicted IoU quality score per mask, on device().
    brotensor::Tensor iou;
    int num_out   = 0;   // num_mask_tokens-1 if multimask else 1
    int mask_size = 0;   // 4*grid (256 for stock SAM)
};

struct MaskDecoderWeights;  // defined in the .cpp; held opaque

class MaskDecoder {
public:
    explicit MaskDecoder(MaskDecoderConfig cfg);
    ~MaskDecoder();

    MaskDecoder(MaskDecoder&&) noexcept;
    MaskDecoder& operator=(MaskDecoder&&) noexcept;
    MaskDecoder(const MaskDecoder&)            = delete;
    MaskDecoder& operator=(const MaskDecoder&) = delete;

    // Load a HuggingFace SAM checkpoint. `load` expects `dir/model.safetensors`;
    // `load_file` reads one explicit path. Both read the `mask_decoder.*`
    // tensors and throw std::runtime_error (prefix "sam::MaskDecoder: ") naming
    // the first missing / mismatched tensor.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Migrate weights onto `dev`. decode() then runs on `dev` and expects its
    // input tensors resident there. No-op if already on `dev`. Throws before
    // load().
    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }

    // Decode masks. All four inputs must be FP32 and resident on device():
    //   image_embedding : (1, hidden_size*grid*grid) NCHW, from ImageEncoder.
    //   image_pe        : (1, hidden_size*grid*grid) NCHW, PromptEncoder::dense_pe().
    //   sparse          : (num_sparse_tokens, hidden_size), or an empty tensor.
    //   dense           : (1, hidden_size*grid*grid) NCHW, the dense prompt embed.
    // multimask_output selects the three multi-mask outputs (true) or the single
    // best mask (false), matching SAM's mask-token slicing. Throws on a
    // shape/device mismatch or before load().
    DecodedMasks decode(const brotensor::Tensor& image_embedding,
                        const brotensor::Tensor& image_pe,
                        const brotensor::Tensor& sparse,
                        const brotensor::Tensor& dense,
                        bool multimask_output) const;

    // Batched decode of `batch` independent prompt sets that share one image.
    // The whole two-way transformer runs once over a block-diagonal pack of the
    // prompts (packed variable-length attention), so the per-prompt decode
    // overhead — tiny matmuls, kernel launches, the host token assembly — is
    // paid once instead of `batch` times. This is what makes the automatic mask
    // generator's dense point grid affordable.
    //
    // Constraints (the cheap, common case — point grids with no mask prompt):
    //   * image_embedding / image_pe / dense are the SHARED (1, hidden*grid*grid)
    //     maps, broadcast across every prompt (a single image, the no-mask dense
    //     embedding). Per-prompt dense (mask prompts) is not supported here.
    //   * sparse is (batch * n_sparse_per_prompt, hidden_size): the prompts'
    //     sparse tokens packed in order, EVERY prompt contributing the same
    //     `n_sparse_per_prompt` rows (a uniform prompt shape — e.g. one point
    //     plus its padding token). Pass n_sparse_per_prompt explicitly.
    //
    // Returns masks (batch*num_out, mask_size*mask_size) and iou (batch*num_out,
    // 1) with prompt b occupying rows [b*num_out, (b+1)*num_out); num_out is the
    // per-prompt count (num_mask_tokens-1 if multimask else 1). Equivalent,
    // within floating-point tolerance, to calling decode() once per prompt.
    DecodedMasks decode_batched(const brotensor::Tensor& image_embedding,
                                const brotensor::Tensor& image_pe,
                                const brotensor::Tensor& sparse,
                                int batch, int n_sparse_per_prompt,
                                const brotensor::Tensor& dense,
                                bool multimask_output) const;

    const MaskDecoderConfig& config() const { return cfg_; }

private:
    MaskDecoderConfig                   cfg_;
    std::unique_ptr<MaskDecoderWeights> w_;
    brotensor::Device                   device_ = brotensor::Device::CPU;
};

}  // namespace brovisionml::sam
