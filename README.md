# brovisionml

Vision-ML model inference. brovisionml is the **expression layer** for neural
vision models — it composes the op surface in `brotensor` (2D convolution and
transposed convolution, interpolation/upsampling, padding, attention with
additive bias, group/layer norm, the activation family) and the host-side
preprocessing in `broimage` (decode, resize/letterbox, normalize) into runnable
image-understanding models.

It is to vision what [`brosoundml`](https://github.com/wlejon/brosoundml) is to
audio, [`brodiffusion`](https://github.com/wlejon/brodiffusion) is to image
*generation*, and [`brolm`](https://github.com/wlejon/brolm) is to text: a
sibling library that turns a tensor op surface into a model.

Unlike the vision *encoders* that live in `brolm` (CLIP vision, the Qwen-VL
tower), the models here are standalone image→X tasks with no language model in
the graph — there is **no tokenizer dependency**.

Models planned / implemented:

- **SAM (Segment Anything)** — promptable segmentation, **runnable end to end**:
  ViT image encoder + prompt encoder (points / boxes / mask) + lightweight mask
  decoder, tied together by a `Sam` orchestrator (encode-image-once /
  decode-many-prompts) and a `sam_segment` CLI driver.
- Depth estimation, detection, and matting are natural follow-ons that reuse
  the same ViT-encoder + task-head shape.

## Dependencies

brovisionml is a standalone sibling repo. It links three siblings and ships no
GPU kernels of its own — GPU work happens inside `brotensor`.

| Library | Role |
|---|---|
| [`bromath`](https://github.com/wlejon/bromath) | header-only math (Vec/Quat/Mat, easing) |
| [`brotensor`](https://github.com/wlejon/brotensor) | the unified `Tensor` + device-neutral op surface (conv2d / conv_transpose2d / interp2d / attention / norms), plus the safetensors weight loader |
| [`broimage`](https://github.com/wlejon/broimage) | host-side decode + geometric resampling + normalize presets (CLIP / ImageNet / SAM) |

No `brolm` dependency: these models take pixels in and emit masks / maps /
boxes, not text.

## Data and weights

brovisionml ships **code only** — no trained weights are checked into this
repo. Loaders take file paths; the caller resolves them. HF `safetensors`
checkpoints are read directly through `brotensor::safetensors` — no offline
conversion step.

## Build

```bash
# CPU-only
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release

# CPU + CUDA (forwards the choice to brotensor's CUDA backend)
cmake -B build -DBROTENSOR_WITH_CUDA=ON
cmake --build build --config Release
```

CPU-by-default (FP32 scalar backend); a GPU backend (FP16) is enabled by
forwarding `BROTENSOR_WITH_CUDA=ON` or `BROTENSOR_WITH_METAL=ON` to brotensor.

## SAM

The first target. SAM splits cleanly into three pieces, and every piece maps
onto ops `brotensor` already exposes:

| SAM component | brotensor ops |
|---|---|
| ViT image encoder | `conv2d` (patch embed), `self_attention_bias_forward` (decomposed relative-position bias, the same additive-bias attention T5 uses), `layernorm`, `gelu` |
| Prompt encoder | positional encoding of point/box coords + learned embeddings (`embedding_lookup`, elementwise) |
| Mask decoder | two-way `cross_attention_forward`, `conv_transpose2d` (4× upscale), `interp2d`, MLP heads (`linear`) + IoU head |

Preprocessing (resize longest-side → normalize with `broimage::SAM_MEAN/STD` →
pad to 1024×1024) is `broimage`'s job; the model takes the prepared pixel
tensor.

Encode-image-once / decode-many-prompts is the natural split: a slow image
encode followed by cheap per-click mask decodes. The `Sam` orchestrator
(`brovisionml/sam.h`) wires all three pieces together — load one HF
`model.safetensors`, `set_image()` once, then `segment()` with points / boxes
given in original-image pixel coordinates and get masks back at the original
resolution:

```cpp
brovisionml::sam::Sam sam(brovisionml::sam::SamConfig::vit_h());
sam.load("/path/to/sam-vit-huge");          // dir holding model.safetensors
sam.set_image(rgb, w, h, /*channels=*/3);    // preprocess + ViT encode (once)
auto seg = sam.segment({{x, y}}, {1}, {});    // a foreground click
// seg.logits[seg.best()*h*w ...] — threshold at 0 for a binary mask
```

The `sam_segment` CLI tool is the same flow from the shell:

```bash
sam_segment /path/to/sam-vit-huge photo.jpg --point 320,240 --out mask.png
```

## License

[MIT](LICENSE)
