# Weights

brovisionml ships **code only** — no trained weights are checked into the
repo. Loaders take file paths; the caller resolves them. Checkpoints are read
directly through `brotensor::safetensors` (an mmap reader whose upload
helpers widen BF16/F16/F32 to the compute dtype), so models that publish
clean HF safetensors load with **no conversion step**.

## Direct downloads

`scripts/download-weights.sh` (and `download-weights.ps1` on Windows) fetch
the checkpoints that exist as clean HF safetensors, into `weights/<name>/`:

| Subcommand | Source |
|---|---|
| `sam-vit-base` / `sam-vit-large` / `sam-vit-huge` | `facebook/sam-vit-*` |
| `depth-anything-v2-small` / `-base` / `-large` | `depth-anything/Depth-Anything-V2-*-hf` |

Two model-specific scripts cover the rest of the downloadable set:

- `scripts/download-stylegan3.sh` — downloads an NVlabs StyleGAN3 pickle and
  runs the converter (see below). Variants: `stylegan3-{r,t}-ffhqu-256`,
  `-afhqv2-512`, `-ffhq-1024`, `-ffhqu-1024`, `-metfaces-1024`,
  `-metfacesu-1024`.
- `scripts/download-triposplat.sh` — fetches the TripoSplat backbones from
  `VAST-AI/TripoSplat`: `dinov3` (`clip_vision/dino_v3_vit_h.safetensors`,
  ~1.7 GB) and `birefnet` (`background_removal/birefnet.safetensors`,
  ~444 MB).

SegFormer's checkpoint (`nvidia/segformer-b0-finetuned-ade-512-512`) is clean
HF safetensors + `config.json`; fetch it from the hub directly.

## Pickled checkpoints (out-of-repo conversion)

Five annotator models ship upstream only as PyTorch pickles with no clean
safetensors release:

| Model | Upstream file |
|---|---|
| DSINE | `dylanebert/DSINE/dsine.pt` |
| HED | `lllyasviel/Annotators/ControlNetHED.pth` |
| Lineart | `lllyasviel/Annotators/sk_model.pth` |
| MLSD | `lllyasviel/Annotators/mlsd_large_512_fp32.pth` |
| OpenPose | `lllyasviel/Annotators/body_pose_model.pth` |

The checkpoints these loaders read are produced by one-off, **out-of-repo**
conversions of those pickles to `model.safetensors`. The repo carries no
Python dependency for them and `download-weights.sh` deliberately has no
entry that pretends a direct safetensors download exists. Once converted, the
loaders read the safetensors directly.

## StyleGAN3: the in-repo converter

StyleGAN3 is the one model with a committed conversion script, because the
NVlabs pickles need the NVlabs repo on `PYTHONPATH` to unpickle at all:

```bash
python scripts/convert-stylegan3.py stylegan3-r-afhqv2-512x512.pkl \
    weights/stylegan3-r-afhqv2-512/model.safetensors \
    --repo /path/to/NVlabs/stylegan3
```

It flattens the EMA generator's `state_dict` into FP32 safetensors with the
tensor names the loader reads, dropping the per-layer `up_filter` /
`down_filter` buffers (brovisionml designs those filters itself — see
[stylegan3.md](stylegan3.md)). `download-stylegan3.sh` wraps download +
conversion in one step.

## Golden parity dumps

Models are validated against golden dumps of the reference (HF / upstream
PyTorch) implementations. The goldens are generated **out-of-repo** — like
the conversions, no Python is committed for them except
`scripts/dump-stylegan3-golden.py` (which shares the NVlabs-unpickle
machinery with the converter) — and are never committed; they live next to
the weights (e.g. `weights/<model>/golden/`). Tests that need real
checkpoints or goldens look under `weights/` and **skip cleanly when absent**,
so a fresh clone builds and passes ctest with no downloads.

The three small `*_test.safetensors` files at the repo root
(`dinov2_test.safetensors`, `dinov3_test.safetensors`,
`dpt_head_test.safetensors`) are tiny random-weight fixtures for structural
unit tests — they validate loading, shape contracts, and op composition
without real weights.
