# CLAUDE.md — brovisionml

Project-specific guidance for Claude Code. Read this before editing.

## What this repo is

`brovisionml` is the vision-model inference library for the **bro** stack — the
vision counterpart to `brolm` (text), `brosoundml` (audio), and `brodiffusion`
(image generation). Pure C++20. It owns standalone image→X models: promptable
segmentation (SAM, the first target), and — as natural follow-ons — depth
estimation, detection, and matting.

These are *vision tasks*, not multimodal language models. There is no tokenizer
and no text in the graph, which is exactly why they do **not** belong in
`brolm`: the vision encoders that live in brolm (CLIP vision, the Qwen-VL
tower) are components of text/multimodal models; the models here take pixels in
and emit masks / maps / boxes.

CPU-by-default (FP32 scalar backend); a GPU backend (FP16) is enabled by
forwarding `BROTENSOR_WITH_CUDA=ON` or `BROTENSOR_WITH_METAL=ON` to brotensor.
brovisionml itself ships **no GPU kernels** — it composes brotensor ops.

## Sibling dependencies

Three sibling repos, resolved at `../<name>` with a `third_party/<name>`
fallback (the `bro/docs/multi-repo-workflow.md` pattern):

- **bromath** — header-only scalar / math helpers.
- **brotensor** — tensors + compute kernels (CPU + optional GPU backend), and
  the `brotensor::safetensors` weight loader (mmap reader + `upload_compute*`
  helpers that handle BF16/F16/F32 → compute-dtype). Load HF checkpoints
  through it directly; do not add a conversion step.
- **broimage** — host-side image decode + geometric resampling + normalize
  presets (`CLIP_MEAN/STD`, `IMAGENET_*`, `SAM_*`). The preprocessing
  front-end every model here feeds: decode → resize/letterbox → normalize →
  NHWC→NCHW. Pixel preprocessing belongs in broimage, not here.

No `brolm` dependency — that's the whole point of the split.

Override paths with `-DBROMATH_DIR=...`, `-DBROTENSOR_DIR=...`,
`-DBROIMAGE_DIR=...`.

## Build & test

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

CMake options: `BROVISIONML_TESTS` (default ON standalone), `BROVISIONML_TOOLS`
(default ON standalone), `BROVISIONML_INSTALL` (default OFF — brovisionml is
consumed via `add_subdirectory`, not `find_package`; no package config is
generated, by design).

Tests live in `tests/`, one file per public header. CPU-only tests are
unconditional; tests gated on real HF checkpoints look for files under
`weights/` and skip cleanly when absent.

## Source layout

```
include/brovisionml/
  version.h
src/                       — one .cpp per public header
tests/                     — one test file per header; test_smoke.cpp proves
                             the brotensor + broimage links
tools/                     — ad-hoc CLI driver binaries (e.g. a SAM segment
                             tool), built standalone only, not run by ctest
```

When adding a new model family: header in `include/brovisionml/`, impl in
`src/`, test in `tests/`, wire all three into `CMakeLists.txt` (top-level
`target_sources` and `tests/CMakeLists.txt`).

## Conventions

- **C++20, pure C++.** No Python anywhere in this stack — not for scripts, not
  for validation, not for weight conversion. Shell or C++ only.
- **No CUDA/Metal in this repo.** GPU paths must go through brotensor ops; if a
  kernel is missing, add it to brotensor.
- **Load HF safetensors directly** through `brotensor::safetensors` — no
  offline conversion step. Models read sharded safetensors + `config.json`
  from a directory.
- **Not wired into bro.** brovisionml stands on its own — its own build, its
  own test suite, its own CLI tools. JS bindings in the bro engine come later,
  if at all; do not add a bro dependency here.

## Git / workflow

- Commit directly to `main` unless explicitly asked to branch.
- One capability per commit. Describe commits by what they add or fix — never
  use "phase 1/2/N" language in commit messages, comments, or PR descriptions.
- For large mechanical changes, chunk into ~2k-LOC pieces and commit each.
