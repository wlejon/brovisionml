# Lineart

Line-drawing extraction — the ControlNet **lineart** conditioning annotator.
The model is lllyasviel's detector, the "Informative Drawings" generator
(Chan et al., CVPR'22), built as `Generator(3, 1, 3)`: an image-to-image CNN
that turns a photo into a dense single-channel line map. It ships **no**
lineart-specific kernel; the forward pass is a pure composition of ops
`brotensor` already exposes:

| Stage | brotensor ops |
|---|---|
| `model0` head | reflect `pad2d` (3) → 7×7 `conv2d` → InstanceNorm → `relu` |
| `model1` downsample ×2 | 3×3 stride-2 `conv2d` (zero-pad 1) → InstanceNorm → `relu` (64→128→256) |
| `model2` 3× residual block | reflect `pad2d` (1) → 3×3 `conv2d` → IN → `relu` → reflect `pad2d` (1) → 3×3 `conv2d` → IN, + skip (`add_inplace`) |
| `model3` upsample ×2 | 3×3 stride-2 `conv_transpose2d` (pad 1, output-pad 1) → InstanceNorm → `relu` (256→128→64) |
| `model4` tail | reflect `pad2d` (3) → 7×7 `conv2d` (→1ch) → `sigmoid` |

Every norm is `InstanceNorm2d` with `affine=False` — no learnable
scale/shift, so the checkpoint holds only the conv weights/biases. Instance
norm is realized as a `group_norm` with `num_groups == channels` and a
constant gamma=1 / beta=0. Preprocessing is minimal (optional resize to a
working resolution, pack to RGB FP32 scaled to [0,1]) and lives in
`lineart::preprocess`.

## API

The `LineartDetector` orchestrator (`brovisionml/lineart.h`) loads one
lineart `model.safetensors` and maps pixels to a line map at the original
resolution:

```cpp
brovisionml::lineart::LineartDetector det;          // default config (native res, invert on)
det.load("/path/to/lineart");                        // dir holding model.safetensors
det.to(brotensor::Device::CUDA);                     // optional GPU migration
auto lm = det.detect(rgb, w, h, /*channels=*/3);
// lm.line is row-major h*w FP32 in [0,1] (brighter = stronger line). lm.at(x,y).
```

The raw generator output is a bright field with dark lines;
`LineartConfig::invert` (on by default) flips it to bright lines on a dark
field — the convention ControlNet feeds downstream.

Like HED, lineart is fully convolutional, so it supports the same
overlapping-tile path — `LineartConfig::tile` bounds the per-pass working
resolution on large images; inversion is pointwise so it commutes with the
blend. See the tiling section of [performance.md](performance.md).

The `lineart` CLI tool writes a grayscale line PNG (`line*255`):

```bash
lineart /path/to/lineart photo.jpg --out lineart.png
# flags: --resolution N (longer-side working resolution; 0 = native)
#        --no-invert (write the raw bright-field/dark-line output)  --cuda
```

## GPU path

On CUDA the whole generator runs **FP16** through the WMMA conv path — the
3×3, 7×7, and conv-transpose layers are all WMMA-covered shapes, and the
instance-norm gamma/beta are cast to FP16 to match. See
[performance.md](performance.md).

## Weights

Lineart ships only a pickled `sk_model.pth` (`lllyasviel/Annotators`); there
is no clean HF safetensors release. The checkpoint this loader reads is
produced by a one-off, **out-of-repo** conversion of that `.pth` to
`model.safetensors` (the `InstanceNorm2d` layers are `affine=False`, so it
is just the conv / conv-transpose weights) — see [weights.md](weights.md).

## Parity

Parity is validated against an out-of-repo golden dump of the reference
network (never committed): `tests/test_lineart.cpp` runs the full
`LineartDetector` end to end and compares the line map against the golden.
Agreement is **mean-abs ~1e-6** on the [0,1] line map with max-abs ~1e-4 and
zero outlier pixels on the FP32 path, for both square and non-square inputs.
Goldens use input dimensions divisible by 4 so the two stride-2 downsamples
and two conv-transpose upsamples round-trip exactly (the model output
matches the input size, so the line map needs no resize-back). The CUDA FP16
path holds the same test gates (mean-abs < 5e-4, < 1% outliers). This is
well inside what the annotator's ControlNet-conditioning role needs.
