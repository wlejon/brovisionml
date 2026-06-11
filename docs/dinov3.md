# DINOv3 backbone

A standalone DINOv3 ViT-H+/16 image encoder — TripoSplat's vision backbone,
usable as a general dense-feature extractor. Unlike the orchestrator models,
this is a *substrate*: `image → token embeddings`, no task head, no CLI
tool. It is byte-faithful to HF `DINOv3ViTModel`.

What distinguishes DINOv3 from the DINOv2 backbone in `dinov2.h`:

- **Register tokens** — the sequence is `[CLS, 4 registers, patch tokens]`.
- **2D axial RoPE** applied only to the patch tokens (CLS/registers carry no
  rotation); frequencies are computed, not stored. HF's rotate-half RoPE is
  converted to brotensor's interleaved pairing at load time by permuting
  q/k projection rows.
- **Gated SwiGLU FFN** (`gate_proj` / `up_proj` / `down_proj` + SiLU)
  instead of fc1/GELU/fc2.
- **LayerScale folded at load** into the output projections (LS1 → o_proj,
  LS2 → down_proj), like DINOv2.

The forward composes brotensor ops only: `conv2d` (patch embed),
`rope_apply`, `flash_attention_varlen_forward`, batched layernorm/linear,
`swiglu_forward`.

## API

```cpp
brovisionml::dinov3::Backbone bb(brovisionml::dinov3::Config::vit_h());
bb.load("/path/to/triposplat/clip_vision");   // reads dino_v3_vit_h.safetensors
bb.to(brotensor::Device::CUDA);

auto pre = brovisionml::dinov3::preprocess(rgb, w, h, /*channels=*/3,
                                           /*resolution=*/1024);
// upload pre.pixels to the backbone's device, then:
auto out = bb.encode(pixels, /*H=*/1024, /*W=*/1024);
// out.last_hidden_state: (num_prefix + patch_h*patch_w, embed_dim) FP32
// out.num_prefix_tokens = 5 (1 CLS + 4 registers); out.patch_h / patch_w
```

`Config::vit_h()` is the ViT-H+/16 preset (embed 1280, depth 32, 20 heads,
4 registers, SwiGLU width 5120, rope_theta 100). There is no `config.json`
in the checkpoint; the architecture lives in the preset.

## Preprocessing: resolution is the speed knob

`dinov3::preprocess` square-resizes to `resolution × resolution` (bicubic
Catmull-Rom, no aspect preservation, no padding), rescales to [0,1], and
ImageNet-normalizes. `resolution` (default 1024, any positive multiple of
the 16-px patch size) is the speed/quality dial: cost scales ~linearly with
token count in the projections and ~quadratically in attention, so 1024→512
is roughly a 4× token reduction. The returned `Dinov3Transform` carries the
scale factors to map token-grid coordinates back to the original image.

## GPU path

On CUDA the backbone runs mixed-precision FP16: the q/k/v, gate, up, and
patch-embed GEMMs narrow to FP16; the o-projection and down-projection
(which carry the folded LayerScale), the residual stream, and the norms stay
FP32 — DINOv3's massive late-layer activations (~1e5 by layer 22) would
overflow FP16's 65504 ceiling if the residual stream narrowed. See
[performance.md](performance.md).

## Weights & tests

Fetch with `scripts/download-triposplat.sh dinov3`
(`VAST-AI/TripoSplat`, `clip_vision/dino_v3_vit_h.safetensors`, ~1.7 GB).
Tensor names match the HF state dict with the encoder prefix stripped.

`tests/test_dinov3.cpp` runs structural checks against the tiny
`dinov3_test.safetensors` fixture unconditionally, and — when the real
checkpoint plus an out-of-repo HF golden are present — validates parity:
FP32 max-abs < 2e-2 against the HF reference, with a documented wider
envelope for the GPU mixed-precision path. `test_dinov3_preprocess.cpp`
covers the front-end (shapes, normalization, resolution-knob edge cases,
1/3/4-channel inputs).
