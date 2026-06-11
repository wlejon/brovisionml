# Performance

The GPU path is the primary path. Models load on CPU in FP32 and migrate with
`.to(Device::CUDA)` (or `Device::METAL`); where a model has an FP16 path it
engages automatically when the brotensor backend's compute dtype is FP16 —
there is no per-model flag to set (StyleGAN3's `force_fp32` debug switch is
the one exception).

## Precision by model

| Model | GPU precision | Notes |
|---|---|---|
| SAM image encoder | mixed FP16 | q/k/v/o + patch-embed GEMMs FP16; residual stream / norms FP32 |
| SAM mask decoder | mixed FP16 | attention projections FP16; MLPs / upscale FP32; decode + AMG filtering device-resident |
| DINOv2 (Depth-Anything) | mixed FP16 | q/k/v + fc1 GEMMs FP16; o-proj / fc2 (carrying folded LayerScale) and residual stream FP32 |
| DINOv3 | mixed FP16 | q/k/v + gate/up GEMMs FP16; o-proj / down-proj and residual stream FP32 (DINOv3's huge activations overflow FP16) |
| DSINE | FP16 encoder + decoder | whole EfficientNet trunk and decoder FP16 via the WMMA conv path; ConvGRU/AngMF refiner FP32 |
| HED | full FP16 | every conv is a WMMA-covered shape; only the final edge map widens for download |
| Lineart | full FP16 | whole generator (convs, conv-transpose, instance norms) FP16 |
| OpenPose | full FP16 | whole VGG + 6-stage net FP16 |
| MLSD | FP32 | no FP16 path |
| SegFormer | FP32 | attention, logits upsample, and argmax run on-device; class ids download as bytes |
| StyleGAN3 | FP16 top resolutions | top `num_fp16_res` (default 4) synthesis bands FP16; mapping and the inversion backward stay FP32; `Config::force_fp32` disables |
| BiRefNet | FP32 | no FP16 path |

The mixed-precision pattern is consistent: tensor-core GEMMs and
WMMA-covered convs get FP16 weights at `to()` time; residual streams,
norms, and any projection that carries a folded LayerScale stay FP32.

## Benchmarking

`tools/bench.cpp` times each family's forward stages over warmed-up reps,
on a real image or a deterministic synthetic one (reproducible without
assets):

```
bench <family> <checkpoint-dir-or-file> [options]
  family: sam | amg | depth | dsine | hed | lineart | mlsd | openpose | segformer
  --image PATH  bench on a real image (default: synthetic)
  --size WxH    synthetic image size (default 1024x768)
  --variant V   sam: b|l|h; depth: small|base|large
  --warmup N    untimed warmup reps (default 2)
  --reps N      timed reps (default 5)
  --cpu         force the CPU backend (default: CUDA when available)
```

`sam` reports encode and decode separately; `openpose` reports the neural
`infer_maps` and the full detect separately.

## Profiling

Set `BROVISIONML_PROFILE=1` to get per-stage timings on stderr. Stage marks
are placed through the model forwards (e.g. DSINE's encoder / decoder /
refine); each mark syncs the device first, so async GPU work is attributed
to the stage that issued it:

```
[brovisionml-prof] encoder                          41.32 ms
[brovisionml-prof] decoder                          63.10 ms
```

When the variable is unset (or `0`) a mark costs a single branch.

## Tiling

HED and lineart are fully convolutional — an output pixel depends only on a
local neighbourhood — so they support an overlapping-tile path that bounds
working resolution (and GPU memory) on large images without the detail loss
of a whole-image downscale:

```cpp
brovisionml::tiling::TileConfig tile;
tile.tile = 1024;      // max tile size in pixels (0 = tiling disabled)
tile.overlap = 128;    // pixels adjacent tiles share, feather-blended
det.config().tile = tile;   // HedConfig::tile / LineartConfig::tile
```

Planning and the run loop live in `brovisionml/tile_runner.h`; the feathered
blend lives in `broimage::tiling`. Global / relative-scale maps — depth,
surface normals, semantic segmentation — must **not** be tiled; they need
whole-image context, and cap their working resolution instead.
