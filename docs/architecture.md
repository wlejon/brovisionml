# Architecture

brovisionml is an **expression layer**: every model in it is a composition of
the device-neutral op surface in [`brotensor`](https://github.com/wlejon/brotensor)
(2D convolution and transposed convolution, attention, interpolation, norms,
the activation family) plus the host-side image preprocessing in
[`broimage`](https://github.com/wlejon/broimage). brovisionml itself ships
**almost no GPU kernels** — when a model needs an op, the op is added to
brotensor where every sibling library can use it, and brovisionml composes it.

## Sibling dependencies

Three sibling repos, resolved at `../<name>` with a `third_party/<name>`
fallback. Override with `-DBROMATH_DIR=...`, `-DBROTENSOR_DIR=...`,
`-DBROIMAGE_DIR=...`.

| Library | Role |
|---|---|
| [`bromath`](https://github.com/wlejon/bromath) | header-only math (Vec/Quat/Mat, easing) |
| [`brotensor`](https://github.com/wlejon/brotensor) | the unified `Tensor` + device-neutral op surface (conv2d / conv_transpose2d / interp2d / attention / norms / GAN primitives), plus the `brotensor::safetensors` weight loader |
| [`broimage`](https://github.com/wlejon/broimage) | host-side decode + geometric resampling + normalize presets (CLIP / ImageNet / SAM) + tile blending |

There is no `brolm` dependency — see [Ecosystem](#ecosystem) below.

## The one kernel exception

DSINE's surface-normal *domain* math — **RayReLU** and the fused **AngMF
propagate** (per-pixel per-neighbor axis-angle rotation + neighborhood unfold)
— has no brotensor primitive and is too domain-specific to belong there.
`src/dsine_ops.cu` provides the CUDA path for those two ops, compiled into the
library when `BROTENSOR_WITH_CUDA=ON` and dispatched behind
`BROVISIONML_WITH_CUDA` (whole-program compilation, no separable device code).
Each op also has a CPU FP32 path.

The rule: cross-model mechanics go to brotensor; only genuine single-model
domain math may live here, and today that set is exactly those two DSINE ops.

## Device model

Every model loads on the CPU in FP32, then optionally migrates:

```cpp
det.load("/path/to/checkpoint");
det.to(brotensor::Device::CUDA);   // or Device::METAL
```

`to()` moves the weights (and, where the model has an FP16 path, narrows them
— see [performance.md](performance.md)), and subsequent forwards run
on-device. Host↔device traffic is confined to the input upload and the final
result download; intermediate maps stay resident.

## Shared substrates

- **`dinov2.h` + `dpt_head.h`** — the DINOv2 ViT backbone and DPT
  (reassemble / RefineNet-fusion) decoder, shared by the DPT-style dense
  tasks (Depth-Anything; DSINE's decoder is DPT-style).
- **`dinov3.h`** — the DINOv3 ViT-H backbone (RoPE, register tokens, SwiGLU),
  a standalone feature extractor; see [dinov3.md](dinov3.md).
- **`tile_runner.h`** — the overlapping-tile planner/driver for the fully
  convolutional local annotators (HED, lineart). Tile *blending* lives in
  `broimage::tiling`; planning and the run loop live here. Global /
  relative-scale tasks (depth, semantic segmentation) must not be tiled —
  they need whole-image context.

## Source layout

```
include/brovisionml/   public headers: one orchestrator per model (sam.h,
                       depth_anything.h, dsine.h, hed.h, lineart.h, mlsd.h,
                       openpose.h, segformer.h, stylegan3.h, birefnet.h),
                       their stage/preprocess headers, the dinov2.h / dinov3.h
                       / dpt_head.h substrate, tile_runner.h, version.h
src/                   one .cpp per public header; dsine_ops.cu is the only
                       GPU kernel source
tests/                 one test file per header; CPU-only tests are
                       unconditional, real-checkpoint / golden-parity tests
                       skip cleanly when weights/ is empty
tools/                 CLI drivers (one per runnable model) + the bench
                       driver; built standalone only, not run by ctest
scripts/               weight download helpers + the StyleGAN3 pickle
                       converter
```

When adding a model family: header in `include/brovisionml/`, impl in `src/`,
test in `tests/`, wire all three into `CMakeLists.txt` (top-level
`target_sources` and `tests/CMakeLists.txt`), and a CLI driver in `tools/` if
the model is end-user runnable.

## Ecosystem

brovisionml is the vision-inference member of the **bro** stack — the sibling
of [`brolm`](https://github.com/wlejon/brolm) (text),
[`brosoundml`](https://github.com/wlejon/brosoundml) (audio), and
[`brodiffusion`](https://github.com/wlejon/brodiffusion) (image generation).
Each turns the brotensor op surface into runnable models for its modality.

The boundary with `brolm` matters: the vision *encoders* that live in brolm
(CLIP vision, the Qwen-VL tower) are components of text/multimodal language
models. The models here are standalone image→X tasks with no language model
and no tokenizer in the graph — which is why brovisionml exists as its own
library and carries no brolm dependency. It is also not wired into the bro
app: its own build, its own tests, its own CLI tools.
