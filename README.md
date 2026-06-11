# brovisionml

Vision-model inference in pure C++20. brovisionml runs standalone
image-understanding models — promptable segmentation, depth, surface
normals, pose, edges, lines, semantic segmentation, background matting, and
GAN image generation — on CPU or GPU (CUDA / Metal), loading HuggingFace
safetensors checkpoints directly with no conversion or Python runtime.

Each model is a small orchestrator class (`load()` → optional
`to(Device::CUDA)` → one call per image) plus a matching CLI tool. There is
no tokenizer and no language model anywhere in the graph: pixels in, masks /
maps / boxes / images out.

## Models

| Model | Task | API / CLI | Docs |
|---|---|---|---|
| **SAM** | promptable segmentation + automatic "segment everything" | `sam::Sam`, `sam::AutomaticMaskGenerator` / `sam_segment`, `sam_amg` | [docs/sam.md](docs/sam.md) |
| **Depth-Anything-V2** | monocular relative depth | `depth::DepthEstimator` / `depth_estimate` | [docs/depth-anything.md](docs/depth-anything.md) |
| **DSINE** | per-pixel surface normals | `dsine::NormalEstimator` / `normal_estimate` | [docs/dsine.md](docs/dsine.md) |
| **HED** | soft edges (ControlNet "softedge") | `hed::SoftEdgeDetector` / `hed_edges` | [docs/hed.md](docs/hed.md) |
| **Lineart** | line drawing (ControlNet "lineart") | `lineart::LineartDetector` / `lineart` | [docs/lineart.md](docs/lineart.md) |
| **MLSD** | straight line segments (ControlNet "mlsd") | `mlsd::MLSDdetector` / `mlsd_lines` | [docs/mlsd.md](docs/mlsd.md) |
| **OpenPose** | multi-person body pose (ControlNet "openpose") | `openpose::OpenposeDetector` / `openpose_pose` | [docs/openpose.md](docs/openpose.md) |
| **SegFormer** | ADE20K semantic segmentation (ControlNet "seg") | `segformer::SegformerDetector` / `segformer_seg` | [docs/segformer.md](docs/segformer.md) |
| **BiRefNet** | background removal / matting | `birefnet::BiRefNet` | [docs/birefnet.md](docs/birefnet.md) |
| **DINOv3** | ViT-H dense-feature backbone | `dinov3::Backbone` | [docs/dinov3.md](docs/dinov3.md) |
| **StyleGAN3** | image generation (config-R/T) + image→W+ inversion | `stylegan3::Generator` / `stylegan3_generate` | [docs/stylegan3.md](docs/stylegan3.md) |

Every model is runnable end to end on CPU and CUDA; each doc covers the
architecture→op mapping, the API, the GPU/precision path, where the weights
come from, and the measured parity against the reference implementation.

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

CPU is the FP32 default; `BROTENSOR_WITH_CUDA=ON` / `BROTENSOR_WITH_METAL=ON`
enable the GPU backend. Other options: `BROVISIONML_TESTS` /
`BROVISIONML_TOOLS` (default ON standalone), `BROVISIONML_INSTALL` (default
OFF — consume via `add_subdirectory`).

brovisionml links three sibling libraries, resolved at `../<name>` with a
`third_party/<name>` fallback (override with `-DBROMATH_DIR` /
`-DBROTENSOR_DIR` / `-DBROIMAGE_DIR`):
[`bromath`](https://github.com/wlejon/bromath) (header-only math),
[`brotensor`](https://github.com/wlejon/brotensor) (tensors + compute
kernels + the safetensors loader), and
[`broimage`](https://github.com/wlejon/broimage) (image decode, resampling,
normalize presets). See [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
scripts/download-weights.sh sam-vit-huge       # fetch a checkpoint
sam_segment weights/sam-vit-huge photo.jpg --point 320,240 --out mask.png --cuda
```

The same flow in C++:

```cpp
brovisionml::sam::Sam sam(brovisionml::sam::SamConfig::vit_h());
sam.load("weights/sam-vit-huge");            // dir holding model.safetensors
sam.to(brotensor::Device::CUDA);              // optional; CPU works too
sam.set_image(rgb, w, h, /*channels=*/3);     // slow ViT encode, once
auto seg = sam.segment({{x, y}}, {1}, {});     // cheap per-click decode
// seg.logits[seg.best()*h*w ...] — threshold at 0 for a binary mask
```

## Weights

brovisionml ships **code only** — no trained weights. Checkpoints with clean
HF safetensors releases (SAM, Depth-Anything, SegFormer, DINOv3, BiRefNet)
load as-is; `scripts/download-weights.sh`, `download-triposplat.sh`, and
`download-stylegan3.sh` fetch them. The annotators that upstream ships only
as PyTorch pickles (DSINE, HED, lineart, MLSD, OpenPose) read a one-off,
out-of-repo conversion to safetensors. Details: [docs/weights.md](docs/weights.md).

Tests that need real checkpoints look under `weights/` and skip cleanly when
absent — a fresh clone builds and passes ctest with no downloads.

## GPU & performance

Models load on CPU and migrate with `.to(Device::CUDA)`. On CUDA most
forwards run FP16 where it's safe — full-FP16 WMMA trunks for the conv
annotators, mixed-precision (FP16 GEMMs, FP32 residual streams) for the ViT
backbones — engaged automatically by the backend's compute dtype.
`tools/bench` times every family; `BROVISIONML_PROFILE=1` prints per-stage
timings. The per-model precision table, bench usage, profiler, and the
overlapping-tile path for large images are in
[docs/performance.md](docs/performance.md).

Almost all GPU work happens inside brotensor ops; brovisionml ships exactly
one CUDA source of its own (DSINE's two surface-normal domain ops — see
[docs/architecture.md](docs/architecture.md)).

## Ecosystem

brovisionml is the vision member of the **bro** stack of sibling inference
libraries — [`brolm`](https://github.com/wlejon/brolm) (text),
[`brosoundml`](https://github.com/wlejon/brosoundml) (audio),
[`brodiffusion`](https://github.com/wlejon/brodiffusion) (image generation)
— but stands alone: its own build, tests, and tools, with no dependency on
any of them. How it relates to the rest of the stack (and why the
vision-language encoders live in brolm instead) is covered in
[docs/architecture.md](docs/architecture.md#ecosystem).

## License

[MIT](LICENSE)
