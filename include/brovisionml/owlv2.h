#pragma once

// OWLv2 — open-vocabulary object detection (google/owlv2-base-patch16-ensemble,
// HF transformers `Owlv2ForObjectDetection`). Text queries in ("a bird",
// "a squirrel"), boxes + per-query scores out: every one of the 3600 patch
// tokens of a 960 x 960 input predicts one box, and its score for a query is
// the sigmoid of a cosine logit against that query's CLIP text embedding.
//
// Architecture:
//   * Vision: CLIP ViT-B/16 at 960 px (60 x 60 patches + a class token, 12
//     pre-norm blocks of width 768, QuickGELU MLPs). The detector reads every
//     token: post_layernorm, times the class token, layer_norm -> the
//     (3600, 768) image embeddings.
//   * Text: the CLIP text tower (12 causal blocks of width 512, 16 positions),
//     pooled at EOS, text_projection, L2-normalised. Queries are embedded once
//     and reused for any number of images.
//   * Class head: dense0 (768 -> 512), L2-normalised, dotted with the query
//     embeddings, then a learned per-box shift and scale (ELU + 1).
//   * Box head: a 768-768-768-4 GELU MLP plus a bias from the box's place on
//     the patch grid, sigmoid -> (cx, cy, w, h) relative to the padded square.
//   * Objectness head: the same MLP shape with one output (query-free).
//
// Preprocessing (owlv2_preprocess.cpp): pad to a square at the bottom/right
// with 0.5 grey (in [0, 1], before normalisation — the original big_vision
// recipe), Gaussian anti-alias then bilinear resize to 960, CLIP mean/std.
// Boxes map back to the image by the padded square's side.
//
// The tokenizer is CLIP's byte-level BPE (vocab.json + merges.txt), local to
// this model: OWLv2's queries are short ASCII-English phrases, and
// brovisionml carries no brolm dependency.
//
// Device: load() reads the checkpoint to host FP32; to(Device) migrates. On a
// GPU backend whose compute dtype is FP16 the vision blocks run mixed
// precision (FP16 GEMMs and flash attention, FP32 residual stream and
// LayerNorms; the DINOv2 pattern); the text tower and the heads stay FP32.
// The CPU path is all FP32 and is the parity reference
// (tests/ref/gen_owlv2_fixture.py, tests/test_owlv2_parity.cpp).

#include "brotensor/tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brovisionml::owlv2 {

struct Config {
    // Vision tower.
    int   image_size     = 960;
    int   patch_size     = 16;
    int   vision_width   = 768;
    int   vision_layers  = 12;
    int   vision_heads   = 12;
    int   vision_mlp     = 3072;
    // Text tower.
    int   vocab_size     = 49408;
    int   max_positions  = 16;
    int   text_width     = 512;
    int   text_layers    = 12;
    int   text_heads     = 8;
    int   text_mlp       = 2048;
    int   embed_dim      = 512;    // text_projection output / class-head width
    float layer_norm_eps = 1e-5f;

    int grid()        const { return image_size / patch_size; }   // 60
    int num_patches() const { return grid() * grid(); }           // 3600

    static Config base16();   // google/owlv2-base-patch16(-ensemble)
};

// CLIP byte-level BPE, lowercased, ASCII-focused pre-tokenisation (letters,
// single digits, punctuation runs, the 's 't 're 've 'm 'll 'd splits): what
// HF's CLIPTokenizer does on English text.
class Tokenizer {
public:
    static constexpr int32_t kBos = 49406;   // <|startoftext|>
    static constexpr int32_t kEos = 49407;   // <|endoftext|>
    static constexpr int32_t kBang = 0;      // "!", an added token in OWLv2's release

    // Throws std::runtime_error on an unreadable or malformed file.
    static Tokenizer load(const std::string& vocab_json, const std::string& merges_txt);

    // BOS, the text's tokens, EOS — truncated so the whole fits `max_len`
    // (EOS kept). No padding: the text tower is causal and pools at EOS.
    std::vector<int32_t> encode(std::string_view text, int max_len = 16) const;

private:
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> ranks_;   // "a\x01b" -> rank
    std::string byte_to_unicode_[256];
};

// A preprocessed image: the (1, 3*S*S) NCHW pixel tensor on the host and the
// side of the padded square the boxes are relative to.
struct Pixels {
    brotensor::Tensor nchw;
    int width = 0, height = 0;   // the original image
    int padded = 0;              // max(width, height)
};

// Pad + anti-aliased resize + CLIP normalisation of 8-bit RGB / RGBA rows
// (channels 3 or 4; alpha ignored) to a `size` x `size` model input.
Pixels preprocess(const uint8_t* pixels, int width, int height, int channels,
                  int size = 960);

// Raw head outputs for one image, on the host. P = 3600 patches, Q queries.
struct Predictions {
    int num_patches = 0, num_queries = 0;
    std::vector<float> logits;       // (P, Q) class logits (pre-sigmoid)
    std::vector<float> boxes;        // (P, 4) cx, cy, w, h in [0,1] of the padded square
    std::vector<float> objectness;   // (P)    objectness logits (pre-sigmoid)
};

struct Detection {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // pixels of the original image
    float score = 0;                         // sigmoid of the best query's logit
    int   label = -1;                        // index of that query
    float objectness = 0;                    // sigmoid of the objectness logit
};

struct DetectOptions {
    float threshold      = 0.1f;   // keep score >= threshold
    float nms_iou        = 0.3f;   // greedy NMS across labels; <= 0 or >= 1 disables
    int   max_detections = 100;
};

// The intermediate stages the parity test compares (host FP32, row-major).
struct VisionTrace {
    std::vector<float> embeddings;    // (1+P, 768) class + patches + positions
    std::vector<float> block0;        // (1+P, 768) after the first block
    std::vector<float> last_hidden;   // (1+P, 768) after the last block
    std::vector<float> image_embeds;  // (P, 768)   what the heads read
};

class Detector {
public:
    explicit Detector(Config cfg = Config::base16());
    ~Detector();
    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;

    // A directory holding model.safetensors, vocab.json and merges.txt (the
    // HF release as-is). Weights land on the host as FP32.
    void load(const std::string& dir);

    void to(brotensor::Device dev);
    brotensor::Device device() const;
    const Config& config() const { return cfg_; }
    const Tokenizer& tokenizer() const;

    // Projected, L2-normalised text embeddings: (n, embed_dim) row-major.
    std::vector<float> embed_text(const std::vector<std::string>& queries) const;
    std::vector<float> embed_tokens(const std::vector<std::vector<int32_t>>& ids) const;

    // Run the vision tower and the heads on a preprocessed image against
    // `num_queries` embeddings from embed_text (host, row-major). `trace`,
    // when given, receives the vision stages.
    Predictions predict(const Pixels& px, const float* query_embeds, int num_queries,
                        VisionTrace* trace = nullptr) const;

    // Post-process: best query per box, sigmoid, threshold, NMS, boxes in the
    // original image's pixels; sorted by score, highest first.
    static std::vector<Detection> decode(const Predictions& pred, const Pixels& px,
                                         const DetectOptions& opts);

    // preprocess + predict + decode.
    std::vector<Detection> detect(const uint8_t* pixels, int width, int height, int channels,
                                  const float* query_embeds, int num_queries,
                                  const DetectOptions& opts = {}) const;

private:
    Config cfg_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brovisionml::owlv2
