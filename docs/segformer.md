# Semantic segmentation (SegFormer)

Per-pixel semantic segmentation — the ControlNet **seg** conditioning
annotator (an ADE20K 150-class semantic map, colorized with the canonical
ADE20K palette). The model is HF `SegformerForSemanticSegmentation`
(`nvidia/segformer-b0-finetuned-ade-512-512`): a hierarchical
Mix-Transformer (MiT) encoder feeding an all-MLP decode head, loaded
directly from the clean HF safetensors. The b0 encoder has four stages with
`hidden_sizes=[32,64,160,256]`, `num_attention_heads=[1,2,5,8]`,
`depths=[2,2,2,2]`, `sr_ratios=[8,4,2,1]`, `patch_sizes=[7,3,3,3]`,
`strides=[4,2,2,2]`; at the fixed 512×512 input the stage grids are
128/64/32/16. It ships **no** SegFormer-specific kernel; the forward is a
pure composition of ops `brotensor` already exposes:

| Stage | brotensor ops |
|---|---|
| OverlapPatchEmbed (per stage) | strided/padded `conv2d` (+ bias), `nchw_to_sequence`, `layernorm` |
| Efficient-Self-Attention | `linear` (biased q/k/v), strided `conv2d` + `layernorm` for the K/V spatial reduction (`sr_ratio>1`), scaled-dot-product softmax, `linear` (biased output) |
| MixFFN | `linear` → 3×3 **depthwise** `conv2d` (groups=hidden) → `gelu` (exact/erf) → `linear`, residual |
| Per-stage close | final `layernorm`, `sequence_to_nchw` |
| Decode head (all-MLP) | per-stage `linear` projection to `decoder_hidden_size`, bilinear `interp2d` (align_corners=False) to the stage-0 grid, `concat_nchw_channels` in reversed stage order, 1×1 `conv2d` fuse (bias-free) + `batch_norm` (inference, eps 1e-5) + `relu`, 1×1 `conv2d` classifier |
| Decode | bilinear upsample of logits to the input size + per-pixel argmax → class ids (on-device when the model is on GPU); ADE20K-palette colorize |

**SR cross-attention.** The Efficient-Self-Attention reduces K/V spatially
when `sr_ratio>1` (stages 0–2), so the query length (the full H·W grid)
differs from the key/value length (the reduced grid). That unequal-length
attention is expressed as a manual per-head scaled-dot-product: the biased
q/k/v projections run through `linear_forward_batched` (the K/V context is
the `sr`-conv-reduced + LayerNorm'd sequence; when `sr_ratio==1` the context
is the input itself, i.e. plain self-attention), the small softmax(QKᵀ)·V is
composed directly, and the biased output projection closes the block. Heads
are contiguous channel chunks, matching HF.
(`brotensor::cross_attention_forward` also handles unequal Q/K lengths, but
it folds no q/k/v/o biases, which SegFormer needs — hence the explicit
biased path.)

## API

The `SegformerDetector` orchestrator (`brovisionml/segformer.h`) loads one
`model.safetensors` + `config.json` and maps pixels to a per-pixel class
map:

```cpp
brovisionml::segformer::SegformerDetector det;       // default SegformerConfig
det.load("/path/to/segformer-b0-ade");                // dir with model.safetensors + config.json
det.to(brotensor::Device::CUDA);                      // optional GPU migration
auto seg = det.detect(rgb, w, h, /*channels=*/3);     // seg.classes: H*W ids in [0,149]
auto cond = brovisionml::segformer::SegformerDetector::colorize(seg);  // HxWx3 RGB
// det.infer_logits(...) exposes the raw decode-head logits @128×128 (the neural gate).
```

The `segformer_seg` CLI tool writes the ADE20K-palette-colorized class map
(the "seg" control image):

```bash
segformer_seg /path/to/segformer-b0-ade photo.jpg --out seg.png
# flags: --cuda
```

**Config-driven.** All dims (hidden sizes, heads, depths, sr ratios, patch
sizes, strides, decoder hidden size, layer-norm eps; `num_labels` from the
classifier weight) are read from the checkpoint's `config.json`, so the
larger MiT-B1…B5 variants load from their respective checkpoints. Only B0
(the 150-class ADE20K head) is parity-validated here.

## GPU path

The GPU path runs FP32 (no FP16 path), but stays device-resident through the
decode: attention, the logits upsample, and the per-pixel argmax all run
on-device, and only the byte-sized class ids download. See
[performance.md](performance.md).

## Parity

Parity is validated against an out-of-repo golden dump of the HF model
(never committed): `tests/test_segformer.cpp` runs two gates. **Gate 1**
(the tight neural gate) feeds the processor's exact normalized input tensor
and compares the decode-head logits @128×128 — pure transformer parity,
**mean-abs 8.0e-4**, worst single logit 7.5e-3, for both CPU and CUDA.
**Gate 2** (end-to-end) runs the full preprocess → encoder → head → upsample
→ argmax and compares the per-pixel class map: **99.99 %** pixel agreement
on both backends (the handful of disagreements are sub-pixel-boundary class
flips under interpolation rounding). CPU and CUDA agree on the logits to
~3.8e-6 mean-abs and produce **identical** class maps.
