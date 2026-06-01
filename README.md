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

Models implemented:

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
- **DSINE** — per-pixel surface-normal estimation, **runnable end to end** on
  CPU or CUDA: an EfficientNet-B5 encoder + DPT-style decoder + iterative
  ConvGRU/AngMF refinement, tied together by a `NormalEstimator` orchestrator and
  a `normal_estimate` CLI driver. See below.
- **HED** — soft-edge estimation (the ControlNet "softedge" annotator),
  **runnable end to end** on CPU or CUDA: a VGG-style 5-block conv trunk with
  per-block side outputs, fused into one edge map, tied together by a
  `SoftEdgeDetector` orchestrator and a `hed_edges` CLI driver. See below.
- **Lineart** — line-drawing extraction (the ControlNet "lineart" annotator),
  **runnable end to end** on CPU or CUDA: lllyasviel's "Informative Drawings"
  image-to-image generator (residual conv encoder/decoder with InstanceNorm),
  tied together by a `LineartDetector` orchestrator and a `lineart` CLI driver.
  See below.
- **MLSD** — straight line-segment detection (the ControlNet "mlsd" annotator),
  **runnable end to end** on CPU or CUDA: M-LSD's `MobileV2_MLSD_Large` (a
  truncated MobileNetV2 backbone + FPN decode head) plus a host-side TP-map
  decode, tied together by an `MLSDdetector` orchestrator and an `mlsd_lines` CLI
  driver. See below.
- **OpenPose** — multi-person 2D body-pose estimation (the ControlNet "openpose"
  annotator), **runnable end to end** on CPU or CUDA: a VGG trunk + six two-branch
  Part-Affinity-Field / heatmap refinement stages, with a host-side peak-NMS +
  bipartite limb-matching decode, tied together by an `OpenposeDetector`
  orchestrator and an `openpose_pose` CLI driver. See below.
- **SegFormer** — per-pixel semantic segmentation (the ControlNet "seg"
  annotator), **runnable end to end** on CPU or CUDA: a hierarchical
  Mix-Transformer (MiT) encoder + all-MLP decode head producing an ADE20K class
  map, tied together by a `SegformerDetector` orchestrator and a `segformer_seg`
  CLI driver. See below.

The DINOv2 backbone and DPT head are a reusable substrate shared by the
DPT-style tasks here (Depth-Anything, DSINE).

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

SAM splits cleanly into three pieces, and every piece maps onto ops `brotensor`
already exposes:

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

## Surface normals (DSINE)

Per-pixel surface-normal estimation. The model is **DSINE_v02** (Discontinuity-aware
Surface-normal Estimation): an EfficientNet-B5 backbone feeding a DPT-style
decoder, then an iterative ConvGRU / AngMF ("angular mean-field") refinement that
rotation-smooths the normals in SO(3) and convex-upsamples back to full
resolution. Like SAM and Depth-Anything it is `image → dense map` and runs
on-device (CPU FP32 or CUDA FP32) — `est.to(Device::CUDA)`. The encoder and
decoder compose ops `brotensor` already exposes; the refinement adds the only
two pieces with no brotensor primitive — DSINE's surface-normal *domain* math —
as brovisionml's own ops, each with a CPU FP32 path and a CUDA kernel:

| Component | implementation |
|---|---|
| EfficientNet-B5 encoder | brotensor `conv2d` (stem / 1×1 / depthwise, TF-"SAME" dynamic pad), batchnorm fold, SiLU; squeeze-excite via `adaptive_avg_pool2d` + 1×1 conv + `broadcast_mul` |
| DPT-style decoder | weight-standardized `conv2d` (folded at load), bilinear `interp2d`, GroupNorm, LeakyReLU, `concat_nchw_channels`, `l2_normalize_nchw`; 2-channel uv geometry conditioning per stage |
| AngMF refinement | brotensor `conv2d` (5×5 ConvGRU gates + heads), GRU elementwise, `convex_upsample`, `slice2d`; plus brovisionml's **RayReLU** and the fused **AngMF propagate** (per-pixel per-neighbor axis-angle rotation + neighborhood unfold) ops — CPU FP32 + CUDA |

Unlike DPT/Depth-Anything there is **no resize and no letterbox**: the image is
consumed at native resolution and only zero-padded to a multiple of 32, then
ImageNet-normalized (`dsine::preprocess`). DSINE is geometry-conditioned — it
takes camera intrinsics; with no calibration available they are synthesized from
an assumed field-of-view (60° by default). The `NormalEstimator` orchestrator
(`brovisionml/dsine.h`) loads one DSINE_v02 `model.safetensors` and maps pixels
to a normal map at the original resolution:

```cpp
brovisionml::dsine::NormalEstimator est;          // default DsineConfig (fov 60°)
est.load("/path/to/dsine");                        // dir holding model.safetensors
auto nm = est.estimate(rgb, w, h, /*channels=*/3);
// nm.normals is 3*h*w planar NCHW (3,H,W): channel 0=nx, 1=ny, 2=nz; each pixel
// is a UNIT normal in CAMERA space. nm.at(c,x,y) indexes a component.
// Or pass explicit pinhole intrinsics: est.estimate(rgb,w,h,3, fx,fy,cx,cy);
```

The `normal_estimate` CLI tool is the same flow from the shell — it writes the
standard normal-map PNG (RGB = `(n+1)*0.5` of nx/ny/nz):

```bash
normal_estimate /path/to/dsine photo.jpg --out normal.png
# flags: --fov DEG   (assumed field-of-view for the synthesized intrinsics)
```

The output convention is **camera-space unit normals**; visualize each component
mapped from [-1,1] to [0,1] as `(n+1)/2` (the usual blue-ish normal map). It runs
host-side FP32 (CPU only — there is no CUDA path here).

### Weights

DSINE ships only a pickled `dsine.pt` (`dylanebert/DSINE/dsine.pt`); there is no
clean HF safetensors release. The checkpoint this loader reads is produced by a
one-off, **out-of-repo** conversion of that `.pt` to `model.safetensors` — no
Python is committed here and the repo carries no Python dependency, consistent
with the project's C++/shell-only rule. (`scripts/` could host a convert helper,
but none is implemented here, and there is deliberately **no**
`download-weights.sh` entry that pretends a direct safetensors download exists.)
Once converted, the loader reads the safetensors directly via
`brotensor::safetensors`.

Parity is validated against an out-of-repo golden dump of the reference DSINE
model (generated like the Depth-Anything goldens; never committed):
`tests/test_dsine.cpp` runs the full `NormalEstimator` end to end and
`tests/test_dsine_{encoder,decoder,refine}.cpp` check each stage. The end-to-end
agreement is **exact to FP32 round-off — max-abs ~2–5e-6** on the unit normals
for both square and zero-padded inputs; the staged refinement test bounds at
max-abs ~1e-2 only to leave headroom over the 5-iteration rotation accumulation.

## Soft edges (HED)

Soft-edge estimation — the ControlNet **softedge** conditioning annotator. The
model is the self-contained "ControlNetHED" reimplementation lllyasviel ships
(Apache-2): a VGG-style five-block convolutional trunk where each block ends in a
1×1 projection to a single-channel side output. Like SAM / Depth-Anything / DSINE
it is `image → dense map` and runs on-device (CPU FP32 or CUDA FP32) —
`det.to(Device::CUDA)`. It ships **no** HED-specific kernel; the forward pass is a
pure composition of ops `brotensor` already exposes:

| Component | brotensor ops |
|---|---|
| 5-block conv trunk | `conv2d` (3×3 blocks, ReLU between), `max_pool2d` (2×2 before blocks 2–5), 1×1 `conv2d` per-block side projection |
| Side-map fusion | bilinear `interp2d` (each side → working resolution), mean, `sigmoid` |

The learned per-channel `norm` bias (the VGG mean, in the [0,255] scale the model
consumes) is **folded into the first conv's bias at load time** — `b'[o] = b[o] -
Σ_c norm[c]·Σ_k W[o,c,k]` — so the forward never needs a broadcast subtract.
Preprocessing is minimal (optional resize to a working resolution, pack to RGB
FP32 in [0,255]) and lives in `hed::preprocess`; there is no normalize step here
because the `norm` bias is part of the model. The `SoftEdgeDetector` orchestrator
(`brovisionml/hed.h`) loads one HED `model.safetensors` and maps pixels to an
edge probability map at the original resolution:

```cpp
brovisionml::hed::SoftEdgeDetector det;            // default HedConfig (native res)
det.load("/path/to/hed");                           // dir holding model.safetensors
auto em = det.detect(rgb, w, h, /*channels=*/3);
// em.edge is row-major h*w FP32 in [0,1] (higher = stronger edge). em.at(x,y).
```

The `hed_edges` CLI tool is the same flow from the shell — it writes a grayscale
edge PNG (`edge*255`):

```bash
hed_edges /path/to/hed photo.jpg --out edges.png
# flags: --resolution N (longer-side working resolution; 0 = native)  --cuda
```

### Weights

HED ships only a pickled `ControlNetHED.pth` (`lllyasviel/Annotators`); there is
no clean HF safetensors release. The checkpoint this loader reads is produced by
a one-off, **out-of-repo** conversion of that `.pth` to `model.safetensors` — no
Python is committed here and the repo carries no Python dependency, consistent
with the project's C++/shell-only rule. Once converted, the loader reads the
safetensors directly via `brotensor::safetensors`.

Parity is validated against an out-of-repo golden dump of the reference network
(generated like the Depth-Anything / DSINE goldens; never committed):
`tests/test_hed.cpp` runs the full `SoftEdgeDetector` end to end and compares the
edge map against the golden. Whole-map agreement is **mean-abs ~2–4e-4** on the
[0,1] edge map for both square and non-square inputs, and CPU↔CUDA agree to
~4e-7. The worst single pixel can reach ~0.1: the edge map is the sigmoid of the
*mean* of five side maps bilinear-upsampled to the working resolution (the
coarsest is a 16× upsample of a 32×32 map). brotensor's bilinear matches torch's
`F.interpolate` to ~1e-7, so the resize is exact — the residual is inherent
`conv2d` FP-accumulation difference at coarse-map edges, where the 16× upsample
through the steep sigmoid turns a sub-pixel logit shift into a narrow band of
flipped fine pixels (~1% of pixels). The map is visually identical; this is the
ballpark-exact agreement the annotator's ControlNet-conditioning role needs.

## Lineart

Line-drawing extraction — the ControlNet **lineart** conditioning annotator. The
model is lllyasviel's detector, the "Informative Drawings" generator (Chan et
al., CVPR'22), built as `Generator(3, 1, 3)`: an image-to-image CNN that turns a
photo into a dense single-channel line map. Like SAM / Depth-Anything / DSINE /
HED it is `image → dense map` and runs on-device (CPU FP32 or CUDA FP32) —
`det.to(Device::CUDA)`. It ships **no** lineart-specific kernel; the forward pass
is a pure composition of ops `brotensor` already exposes:

| Stage | brotensor ops |
|---|---|
| `model0` head | reflect `pad2d` (3) → 7×7 `conv2d` → InstanceNorm → `relu` |
| `model1` downsample ×2 | 3×3 stride-2 `conv2d` (zero-pad 1) → InstanceNorm → `relu` (64→128→256) |
| `model2` 3× residual block | reflect `pad2d` (1) → 3×3 `conv2d` → IN → `relu` → reflect `pad2d` (1) → 3×3 `conv2d` → IN, + skip (`add_inplace`) |
| `model3` upsample ×2 | 3×3 stride-2 `conv_transpose2d` (pad 1, output-pad 1) → InstanceNorm → `relu` (256→128→64) |
| `model4` tail | reflect `pad2d` (3) → 7×7 `conv2d` (→1ch) → `sigmoid` |

Every norm is `InstanceNorm2d` with `affine=False` — no learnable scale/shift, so
the checkpoint holds only the conv weights/biases. Instance norm is realized as a
`group_norm` with `num_groups == channels` and a constant gamma=1 / beta=0.
Preprocessing is minimal (optional resize to a working resolution, pack to RGB
FP32 scaled to [0,1]) and lives in `lineart::preprocess`. The `LineartDetector`
orchestrator (`brovisionml/lineart.h`) loads one lineart `model.safetensors` and
maps pixels to a line map at the original resolution:

```cpp
brovisionml::lineart::LineartDetector det;          // default config (native res, invert on)
det.load("/path/to/lineart");                        // dir holding model.safetensors
auto lm = det.detect(rgb, w, h, /*channels=*/3);
// lm.line is row-major h*w FP32 in [0,1] (brighter = stronger line). lm.at(x,y).
```

The raw generator output is a bright field with dark lines; `LineartConfig::invert`
(on by default) flips it to bright lines on a dark field — the convention
ControlNet feeds downstream. The `lineart` CLI tool is the same flow from the
shell — it writes a grayscale line PNG (`line*255`):

```bash
lineart /path/to/lineart photo.jpg --out lineart.png
# flags: --resolution N (longer-side working resolution; 0 = native)
#        --no-invert (write the raw bright-field/dark-line output)  --cuda
```

### Weights

Lineart ships only a pickled `sk_model.pth` (`lllyasviel/Annotators`); there is no
clean HF safetensors release. The checkpoint this loader reads is produced by a
one-off, **out-of-repo** conversion of that `.pth` to `model.safetensors` (the
`InstanceNorm2d` layers are `affine=False`, so it is just the conv / conv-transpose
weights) — no Python is committed here and the repo carries no Python dependency,
consistent with the project's C++/shell-only rule. Once converted, the loader
reads the safetensors directly via `brotensor::safetensors`.

Parity is validated against an out-of-repo golden dump of the reference network
(generated like the Depth-Anything / DSINE / HED goldens; never committed):
`tests/test_lineart.cpp` runs the full `LineartDetector` end to end and compares
the line map against the golden. Agreement is **mean-abs ~6–9e-7** on the [0,1]
line map with **max-abs ~8e-5** and zero outlier pixels, for both square and
non-square inputs; CPU↔CUDA agree to ~8e-5 (and the CUDA path tracks the torch
golden to ~1e-8). Goldens use input dimensions divisible by 4 so the two
stride-2 downsamples and two conv-transpose upsamples round-trip exactly (the
model output matches the input size, so the line map needs no resize-back). This
parity-exact agreement is well inside what the annotator's ControlNet-conditioning
role needs.

## Straight lines (MLSD)

Straight line-segment detection — the ControlNet **mlsd** conditioning annotator.
The model is M-LSD (Mobile Line Segment Detection, NAVER 2021), the
`MobileV2_MLSD_Large` network lllyasviel ships: a truncated MobileNetV2 backbone
(four-channel input = RGB + a constant plane; five FPN taps) feeding an FPN-style
decode head, whose final 16-channel conv output is sliced to its last 9 channels
as a "TP map" at half resolution (256×256 for the 512×512 input). Like the other
detectors it runs on-device (CPU FP32 or CUDA FP32) — `det.to(Device::CUDA)`. It
ships **no** MLSD-specific kernel; the forward is a pure composition of ops
`brotensor` already exposes:

| Stage | brotensor ops |
|---|---|
| MobileNetV2 backbone | grouped (depthwise) `conv2d`, 1×1 `conv2d`, `pad2d` (TFLite right/bottom stride-2 pad), ReLU6 via `clamp(0,6)`; BatchNorm **folded into each preceding conv at load** |
| FPN decode head | 1×1 / 3×3 / dilated (pad 5, dil 5) `conv2d`, align-corners bilinear `interp2d` (the ×2 upsamples), `concat_nchw_channels` (the skip merges), residual `add_inplace` |
| TP-map decode (host) | channel 0 = center heatmap (`sigmoid` → 3×3 `max_pool2d` NMS → top-K), channels 1:5 = per-pixel start/end displacements → line segments, kept by score + length thresholds, scaled 256→512→original |

Folding BatchNorm into the conv (`W'=W·s`, `b'=(b−μ)·s+β`, `s=γ/√(var+ε)`) keeps
the forward a clean conv→activation sequence with no separate norm op. Input is
normalized `(x/127.5−1)`; the appended ones plane is normalized too, so the 4th
channel is the constant `1/127.5−1`. The `MLSDdetector` orchestrator
(`brovisionml/mlsd.h`) loads one MLSD `model.safetensors` and maps pixels to a
list of line segments in the original image's coordinates:

```cpp
brovisionml::mlsd::MLSDdetector det;               // default MlsdConfig (thr 0.1)
det.load("/path/to/mlsd");                          // dir holding model.safetensors
auto lm = det.detect(rgb, w, h, /*channels=*/3);
// lm.segments: {x1,y1,x2,y2,score}, original-image pixels.
// det.infer_tpmap(...) exposes the raw 9-channel TP map for custom decodes.
```

The `mlsd_lines` CLI tool is the same flow from the shell — it rasterizes the
detected segments as white lines on black (the conditioning image):

```bash
mlsd_lines /path/to/mlsd photo.jpg --out mlsd.png
# flags: --score-thr F  --dist-thr F  --cuda
```

### Weights

MLSD ships only a pickled `mlsd_large_512_fp32.pth` (`lllyasviel/Annotators`);
there is no clean HF safetensors release. The checkpoint this loader reads is
produced by a one-off, **out-of-repo** conversion of that `.pth` to
`model.safetensors` — no Python is committed here and the repo carries no Python
dependency, consistent with the project's C++/shell-only rule. Once converted,
the loader reads the safetensors directly via `brotensor::safetensors`.

Parity is validated against an out-of-repo golden dump of the reference network
(generated like the other goldens; never committed): `tests/test_mlsd.cpp`
compares the raw 9-channel TP map (the tight neural gate, **mean-abs ~1e-4**) and
the decoded segment set. CPU and CUDA both reconstruct the **identical** segment
set (recall 1.0 vs the reference's). The worst single TP logit differs by ~1.5e-2
between the backends — the same order as each one's gap from torch, an expected
FP-accumulation difference across the deep depthwise/dilated trunk on logits that
span tens; it is far below what the sigmoid + NMS + thresholds amplify into a
segment change, which is why the decoded lines are identical. This is the
ballpark-exact agreement the annotator's ControlNet-conditioning role needs.

## Pose (OpenPose)

Body-pose estimation — the ControlNet **openpose** conditioning annotator. The
model is the CMU multi-person 2D body-pose network (the COCO-18 keypoint,
19-limb Part-Affinity-Field model) as packaged by pytorch-openpose and
lllyasviel's `body_pose_model.pth`: a VGG-style trunk `model0` (ten 3×3 convs +
three 2×2 max-pools, 3→128 channels at ⅛ spatial) followed by six two-branch
refinement stages. Each stage's branch **L1** emits a 38-channel PAF map and
branch **L2** a 19-channel confidence/heatmap (18 parts + background); stage 1
takes the trunk feature, stages 2–6 each take `cat([prev_L1, prev_L2, trunk])`
(185ch) and apply five 7×7 convs then two 1×1 convs. Like the other detectors it
runs on-device (CPU FP32 or CUDA FP32) — `det.to(Device::CUDA)`. It ships **no**
OpenPose-specific kernel; the forward is a pure composition of ops `brotensor`
already exposes:

| Stage | brotensor ops |
|---|---|
| `model0` VGG trunk | 3×3 `conv2d` (+ bias) + `relu`, 2×2 `max_pool2d` (3 downsamples) |
| Stages 1–6 (L1 PAF + L2 heatmap) | `concat_nchw_channels` (the `[L1,L2,trunk]` re-concat each stage), 7×7 / 3×3 / 1×1 `conv2d` (+ bias) + `relu` (every conv but the final `Mconv7`/`conv5_5`) |
| Decode (host) | ×8 upsample + crop pad + resize to detect-res; per-part Gaussian-blur (σ=3, scipy-`reflect`) + local-max NMS peak detection; PAF line-integral scoring + greedy bipartite limb matching; subset-merge people assembly; prune (<4 parts or mean score <0.4) |

Every conv carries a bias and an in-place ReLU except the final 1×1 of each
branch (`Mconv7_stage{i}_L{1,2}` / `conv5_5_CPM_L{1,2}`). Input is the reference
front-end: `resize_image` (shorter side → detect resolution, rounded to a
multiple of 64), RGB→**BGR** (the net was trained on BGR), a single-scale
`smart_resize` to `0.5·368/H`, right/bottom pad to a multiple of 8 with constant
128, normalize `(x/256 − 0.5)`. The `OpenposeDetector` orchestrator
(`brovisionml/openpose.h`) loads one `model.safetensors` and maps pixels to a set
of people, each up to 18 keypoints (normalized to the detect-res image):

```cpp
brovisionml::openpose::OpenposeDetector det;        // default OpenposeConfig
det.load("/path/to/openpose");                       // dir holding model.safetensors
auto pose = det.detect(rgb, w, h, /*channels=*/3);
// pose.bodies: per person std::array<Keypoint,18> (x,y normalized, score, present)
// det.infer_maps(...) exposes the raw stage-6 PAF (38ch) + heatmap (19ch).
auto canvas = brovisionml::openpose::OpenposeDetector::draw(pose);  // HxWx3 RGB
```

The `openpose_pose` CLI tool is the same flow from the shell — it rasterizes the
detected people as the canonical colored limb-sticks-and-joints control image:

```bash
openpose_pose /path/to/openpose photo.jpg --out openpose.png
# flags: --resolution N  --cuda
```

**Body-only scope.** The canonical ControlNet openpose control image is
body-only (`include_hand=False, include_face=False` in the reference), so only
the body network is implemented here — the hand and face sub-networks are
intentionally not ported.

### Weights

OpenPose ships only a pickled `body_pose_model.pth` (`lllyasviel/Annotators`);
there is no clean HF safetensors release. The checkpoint this loader reads is
produced by a one-off, **out-of-repo** conversion of that `.pth` to
`model.safetensors` (the reference `transfer` key-remap → natural
`model0.conv1_1.weight …` keys) — no Python is committed here. Once converted,
the loader reads the safetensors directly via `brotensor::safetensors`.

Parity is validated against an out-of-repo golden dump of the reference network
(never committed): `tests/test_openpose.cpp` runs two gates. **Gate 1** (the
tight neural gate) compares the raw stage-6 PAF + heatmap at network resolution —
pure conv parity, **mean-abs ~4e-5 (PAF) / ~1.8e-4 (heatmap)**, worst single
logit ~9e-3 / ~1.7e-2 across the deep VGG+6-stage trunk, for both CPU and CUDA.
**Gate 2** (end-to-end) decodes people and matches them against the golden:
**4/4 people recovered with keypoint recall@6px = 1.0** on both backends. The
host decode uses Lanczos3 (broimage has no cv2 Lanczos4) and a hand-rolled
scipy-`reflect` Gaussian; the raw-map gate isolates the network from those
classical-decode approximations, which here leave keypoint locations within a
few pixels — the ballpark-exact agreement the annotator's ControlNet-conditioning
role needs. CPU and CUDA agree on the raw maps to ~6e-9 mean-abs and produce the
**identical** people decode.

## Semantic segmentation (SegFormer)

Per-pixel semantic segmentation — the ControlNet **seg** conditioning annotator
(an ADE20K 150-class semantic map, colorized with the canonical ADE20K palette).
The model is HF `SegformerForSemanticSegmentation`
(`nvidia/segformer-b0-finetuned-ade-512-512`): a hierarchical Mix-Transformer
(MiT) encoder feeding an all-MLP decode head, loaded directly from the clean HF
safetensors. The b0 encoder has four stages with `hidden_sizes=[32,64,160,256]`,
`num_attention_heads=[1,2,5,8]`, `depths=[2,2,2,2]`, `sr_ratios=[8,4,2,1]`,
`patch_sizes=[7,3,3,3]`, `strides=[4,2,2,2]`; at the fixed 512×512 input the
stage grids are 128/64/32/16. Like the other detectors it runs on-device (CPU
FP32 or CUDA FP32) — `det.to(Device::CUDA)`. It ships **no** SegFormer-specific
kernel; the forward is a pure composition of ops `brotensor` already exposes:

| Stage | brotensor ops |
|---|---|
| OverlapPatchEmbed (per stage) | strided/padded `conv2d` (+ bias), `nchw_to_sequence`, `layernorm` |
| Efficient-Self-Attention | `linear` (biased q/k/v), strided `conv2d` + `layernorm` for the K/V spatial reduction (`sr_ratio>1`), scaled-dot-product softmax, `linear` (biased output) |
| MixFFN | `linear` → 3×3 **depthwise** `conv2d` (groups=hidden) → `gelu` (exact/erf) → `linear`, residual |
| Per-stage close | final `layernorm`, `sequence_to_nchw` |
| Decode head (all-MLP) | per-stage `linear` projection to `decoder_hidden_size`, bilinear `interp2d` (align_corners=False) to the stage-0 grid, `concat_nchw_channels` in reversed stage order, 1×1 `conv2d` fuse (bias-free) + `batch_norm` (inference, eps 1e-5) + `relu`, 1×1 `conv2d` classifier |
| Decode (host) | bilinear upsample of logits to the input size + per-pixel argmax → class ids; ADE20K-palette colorize |

**SR cross-attention.** The Efficient-Self-Attention reduces K/V spatially when
`sr_ratio>1` (stages 0–2), so the query length (the full H·W grid) differs from
the key/value length (the reduced grid). That unequal-length attention is
expressed as a manual per-head scaled-dot-product: the biased q/k/v projections
run through `linear_forward_batched` (the K/V context is the `sr`-conv-reduced +
LayerNorm'd sequence; when `sr_ratio==1` the context is the input itself, i.e.
plain self-attention), the small softmax(QKᵀ)·V is composed directly, and the
biased output projection closes the block. Heads are contiguous channel chunks,
matching HF. (`brotensor::cross_attention_forward` also handles unequal Q/K
lengths, but it folds no q/k/v/o biases, which SegFormer needs — hence the
explicit biased path.)

The `SegformerDetector` orchestrator (`brovisionml/segformer.h`) loads one
`model.safetensors` + `config.json` and maps pixels to a per-pixel class map:

```cpp
brovisionml::segformer::SegformerDetector det;       // default SegformerConfig
det.load("/path/to/segformer-b0-ade");                // dir with model.safetensors + config.json
auto seg = det.detect(rgb, w, h, /*channels=*/3);     // seg.classes: H*W ids in [0,149]
auto cond = brovisionml::segformer::SegformerDetector::colorize(seg);  // HxWx3 RGB
// det.infer_logits(...) exposes the raw decode-head logits @128×128 (the neural gate).
```

The `segformer_seg` CLI tool is the same flow from the shell — it writes the
ADE20K-palette-colorized class map (the "seg" control image):

```bash
segformer_seg /path/to/segformer-b0-ade photo.jpg --out seg.png
# flags: --cuda
```

**Config-driven.** All dims (hidden sizes, heads, depths, sr ratios, patch
sizes, strides, decoder hidden size, layer-norm eps; `num_labels` from the
classifier weight) are read from the checkpoint's `config.json`, so the larger
MiT-B1…B5 variants load from their respective checkpoints. Only B0 (the 150-class
ADE20K head) is parity-validated here.

Parity is validated against an out-of-repo golden dump of the HF model (never
committed): `tests/test_segformer.cpp` runs two gates. **Gate 1** (the tight
neural gate) feeds the processor's exact normalized input tensor and compares the
decode-head logits @128×128 — pure transformer parity, **mean-abs 8.0e-4**, worst
single logit 7.5e-3, for both CPU and CUDA. **Gate 2** (end-to-end) runs the full
preprocess → encoder → head → upsample → argmax and compares the per-pixel class
map: **99.99 %** pixel agreement on both backends (the handful of disagreements
are sub-pixel-boundary class flips under interpolation rounding). CPU and CUDA
agree on the logits to ~3.8e-6 mean-abs and produce **identical** class maps.

## License

[MIT](LICENSE)
