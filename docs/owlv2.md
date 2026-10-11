# Open-vocabulary detection (OWLv2)

Text queries in, boxes out. The model is Google's **OWLv2**
(`google/owlv2-base-patch16-ensemble`, the transformers
`Owlv2ForObjectDetection` checkpoint): a CLIP ViT-B/16 image tower at 960 px
whose every patch token predicts one box, scored against CLIP text
embeddings of the queries ("a bird", "a squirrel"). Nothing in the label set
is fixed: a query is any short English phrase, embedded once and reused for
any number of images.

Like everything here it is a composition of `brotensor` ops; its CLIP towers,
tokenizer and preprocessing are local to brovisionml (`src/owlv2*.cpp`)
rather than borrowed from brolm, which brovisionml does not depend on.

| Component | brotensor ops |
|---|---|
| Patch embed | 16 x 16 stride-16 `conv2d` (no bias), `nchw_to_sequence`, `concat_rows` with the class token, the stored 3601-row position embedding |
| Vision tower (12 x 768) | pre-norm blocks: `layernorm`, q/k/v/out `linear`, attention (`flash_attention_varlen_forward` FP16 on a GPU, `flash_attention_gqa_forward` FP32 on the CPU), QuickGELU MLP fused into the fc1 GEMM |
| Image embeddings | `layernorm` (post), class-token `broadcast_mul` over the patches, `layernorm` |
| Text tower (12 x 512) | the same block, causal (`flash_attention_gqa_forward`), pooled at EOS, `text_projection`, L2-normalised |
| Class head | `linear` (768 -> 512), `l2_norm_forward`, a `linear` against the normalised queries; learned per-box shift and scale (ELU + 1) |
| Box / objectness heads | 768-768-768-{4,1} MLPs with exact GELU fused into the GEMMs; box bias from the patch grid, sigmoid |

## API

```cpp
#include "brovisionml/owlv2.h"
namespace ow = brovisionml::owlv2;

ow::Detector det;                       // Config::base16()
det.load("brovisionml-data/owlv2");     // model.safetensors, vocab.json, merges.txt
det.to(brotensor::Device::VULKAN);

const std::vector<std::string> queries = {"a bird", "a squirrel", "a deer"};
const std::vector<float> text = det.embed_text(queries);    // (3, 512), once

ow::DetectOptions o;                    // threshold 0.1, nms_iou 0.3, max 100
auto found = det.detect(rgb, w, h, /*channels=*/3, text.data(), 3, o);
// found[i]: x0 y0 x1 y1 (image pixels), score, label (query index), objectness
```

`detect()` is `preprocess()` + `predict()` + `decode()`, each public:
`preprocess` pads the image to a square at the bottom/right with 0.5 grey,
anti-aliases (Gaussian, sigma = (factor - 1) / 2) and bilinearly resizes it to
960, CLIP-normalised; `predict` returns the raw per-patch logits (P x Q), boxes
(cx, cy, w, h of the padded square) and objectness logits; `decode` takes the
best query per box, sigmoid, threshold, greedy NMS across queries, and maps the
boxes to the original image. `predict(..., &trace)` also returns the
intermediate vision stages for parity work.

The pad colour is the original big_vision recipe's grey; transformers' current
`Owlv2ImageProcessor` pads with 0 (black). The resize matches the processor's
exactly (checked by the fixture generator).

The tokenizer is CLIP's byte-level BPE on ASCII text, plus one quirk of this
release: its pad token "!" is a registered added token, so every "!" in a
query is id 0, as transformers tokenises it.

## Precision and speed

The CPU path is FP32 throughout. On a GPU whose compute dtype is FP16 the
vision blocks run mixed precision: q/k/v/o, fc1 (+QuickGELU) and fc2 GEMMs and
the attention in FP16, the residual stream and LayerNorms FP32; the patch
embed is FP16. The text tower and the heads stay FP32. Holding any one group
of projections (or the attention) in FP32 removes only a few percent of the
FP16 error (10-30% of it) at 1.3-3.7x the time, so none is.

On a Radeon 8060S (Vulkan): `predict` at 960 is ~75 ms (vision blocks 70,
heads 4.5, patch embed 1.5; `BROVISIONML_PROFILE=1`), all-FP32 on the same GPU
~360 ms. `preprocess` of a 1280 x 800 RGB image is ~22 ms on one core.

## Weights & tests

The HF release loads as-is (no conversion): `brovisionml-data/owlv2`
(see its `ATTRIBUTION.md`), or `hf download
google/owlv2-base-patch16-ensemble`.

`tests/test_owlv2.cpp` checks preprocessing and decode with no weights.
`tests/test_owlv2_parity.cpp` compares every stage with
`tests/ref/gen_owlv2_fixture.py`'s transformers dump (a synthetic 1280 x 800
image, five queries) and skips without the weights or the fixture:

| Stage | CPU max abs | Vulkan FP16 max abs (p99) |
|---|---|---|
| pixels (preprocess) | 9.5e-7 | — |
| text embeddings | 2.1e-7 | 1.2e-7 |
| token embeddings | 7.6e-6 | 2.5e-3 (4.0e-4) |
| block 0 | 9.8e-6 | 2.6e-3 (5.2e-4) |
| last block (values up to 270) | 4.0e-3 | 1.8 (5.1e-2) |
| image embeddings | 1.0e-3 | 0.43 (1.3e-2) |
| class logits | 3.4e-3 | 1.4 (0.57) |
| boxes (cxcywh) | 6.8e-4 | 7.6e-2 (8.8e-3) |
| objectness logits | 2.6e-3 | 1.0 (0.42) |
| scores (sigmoid) | 8.0e-7 | 1.1e-4 |
| top detections, box px | 0.011 (top 20) | 0.081 (top 5) |
| top detections, score | 4.3e-7 | 4.4e-5 |
