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
  decode-many-prompts) and a `sam_segment` CLI driver. The **automatic mask
  generator** ("segment everything") rides on top of the same orchestrator —
  see below.
- **Depth-Anything-V2** — monocular relative-depth estimation, **runnable end
  to end**: a DINOv2 ViT backbone + a DPT (reassemble / RefineNet-fusion / head)
  decoder, tied together by a `DepthEstimator` orchestrator and a
  `depth_estimate` CLI driver. See below.
- Detection and matting are natural follow-ons; the DINOv2 backbone and DPT head
  here are the reusable substrate for further DPT-style tasks (e.g. surface
  normals, semantic segmentation).

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

### Automatic mask generator ("segment everything")

`AutomaticMaskGenerator` (`brovisionml/sam_amg.h`) is the C++ port of
`segment_anything`'s `SamAutomaticMaskGenerator`: it samples a regular grid of
foreground points, decodes a multi-mask proposal at each, and filters the pile
down to a clean set — no prompts required. The pipeline matches upstream:

regular point grid → per-point multi-mask decode → predicted-IoU filter →
stability-score filter → binarize + box + crop-edge drop → box-NMS (within and
across crops) → optional connected-components small-region / hole cleanup →
masks sorted by area.

It composes a `Sam` you already loaded — `set_image()` is called for you per
crop — so it runs on the same backend (CPU or CUDA) the model lives on:

```cpp
brovisionml::sam::Sam sam(brovisionml::sam::SamConfig::vit_h());
sam.load("/path/to/sam-vit-huge");

brovisionml::sam::AmgConfig cfg;        // upstream defaults
cfg.points_per_side = 32;               // 32x32 grid (the SAM default)
brovisionml::sam::AutomaticMaskGenerator gen(sam, cfg);

auto masks = gen.generate(rgb, w, h, /*channels=*/3);
// each: .mask (h*w, 1=fg), .bbox {x,y,w,h}, .area,
//       .predicted_iou, .stability_score, .point, .crop_box
```

The grid points are decoded in batches (`AmgConfig::points_per_batch`, default
64) through one batched mask-decoder pass — independent prompts packed into a
single block-diagonal attention call — so the per-click decode overhead is
amortized instead of paid one grid point at a time. The slow ViT encode still
happens once per crop.

Knobs worth knowing (all on `AmgConfig`, defaults mirror upstream):

| Field | Meaning |
|---|---|
| `points_per_side` | grid density (N×N points); `32` is the SAM default |
| `points_per_batch` | grid points decoded per batched pass (`64`) |
| `pred_iou_thresh` | drop masks below this predicted IoU (`0.88`) |
| `stability_score_thresh` | drop masks below this binarization stability (`0.95`) |
| `box_nms_thresh` / `crop_nms_thresh` | NMS IoU within / across crops (`0.7`) |
| `crop_n_layers` | crop-pyramid layers for higher recall on big images (`0`) |
| `min_mask_region_area` | remove islands / holes smaller than this (`0` = off) |

The `sam_amg` CLI tool is the same from the shell — it writes a colored overlay
PNG with every generated mask:

```bash
sam_amg /path/to/sam-vit-huge photo.jpg --points-per-side 32 --out everything.png
# other flags: --pred-iou-thresh --stability-thresh --crop-n-layers
#              --min-region-area --points-per-batch --variant --cuda
```

## Depth-Anything-V2

Monocular relative-depth estimation. The model is a DINOv2 ViT backbone feeding
a DPT decoder — the same `image → dense map` shape SAM's encoder + head has, and
every piece maps onto ops `brotensor` already exposes:

| Component | brotensor ops |
|---|---|
| DINOv2 backbone | `conv2d` (patch embed), `mha_forward` (plain global self-attention), `layernorm`, `gelu`; absolute position embedding bicubically interpolated to the patch grid; per-block LayerScale folded into the projections at load |
| DPT reassemble | `conv2d` (1×1 project), `conv_transpose2d` (up) / `interp2d` / strided `conv2d` (down) |
| DPT RefineNet fusion | pre-activation residual `conv2d` units + `interp2d_align_corners_forward` (the align_corners=True bilinear upsample DPT needs) + 1×1 `conv2d` |
| Depth head | `conv2d` + align-corners upsample + ReLU |

Preprocessing (aspect-preserving resize to a multiple of 14 → ImageNet
normalize, no padding) is `broimage`'s job via `dpt::preprocess`. The
`DepthEstimator` orchestrator (`brovisionml/depth_anything.h`) loads one HF
`DepthAnythingForDepthEstimation` `model.safetensors` and maps pixels to a depth
map at the original resolution:

```cpp
brovisionml::depth::DepthEstimator est(
    brovisionml::depth::DepthAnythingConfig::v2_small());
est.load("/path/to/Depth-Anything-V2-Small");   // dir holding model.safetensors
auto dm = est.estimate(rgb, w, h, /*channels=*/3);
// dm.depth is row-major h*w relative inverse-depth (nearer = larger); not metric
```

The `depth_estimate` CLI tool is the same flow from the shell — it writes a
min-max-normalized grayscale depth PNG (brighter = nearer):

```bash
depth_estimate /path/to/Depth-Anything-V2-Small photo.jpg --out depth.png
# flags: --variant small|base|large  --invert  --cuda
```

Fetch a checkpoint with `scripts/download-weights.sh depth-anything-v2-small`
(`-base` / `-large` for the larger ViT-B / ViT-L variants). The output is
*relative* depth in Depth-Anything's convention (larger = nearer), not metric.

Parity is measured against the actual HF `DepthAnythingForDepthEstimation`
output, not just asserted: `tests/test_depth_parity.cpp` compares brovisionml to
a golden dump of the Python model (generated out-of-repo; the goldens download
like the weights). At a **square 518²** input — where the DPT resize is identity
and DINOv2's position embedding is used at its native grid (no interpolation) —
brovisionml agrees with HF to FP32 round-off: depth **rel-max ~1.4e-5**,
preprocess ~7e-7, backbone stages ~5e-4. The model path is exact. CPU (FP32) and
CUDA (FP16) additionally agree to a few times 1e-5 on the real Small checkpoint.

For **non-square** inputs the agreement is ~0.1% mean / ~0.7% max. DINOv2's
position-embedding interpolation is faithful — brotensor's bicubic mode 3 uses
the torch coefficient a=-0.75 (`torch.interpolate(mode="bicubic")`), matched
exactly; mode 2 (a=-0.5, Catmull-Rom / PIL) is reserved for the PIL-matching
image resize. The DPT fusion/head upsamples are bilinear (no cubic coefficient).
The residual max is a preprocessing nuance: `broimage`'s bicubic image resize
matches PIL in the interior but differs at image borders / hard edges (PIL
renormalizes boundary weights), a few localized pixels. Square inputs avoid the
resize entirely and are exact.

## License

[MIT](LICENSE)
