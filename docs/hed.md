# Soft edges (HED)

Soft-edge estimation — the ControlNet **softedge** conditioning annotator.
The model is the self-contained "ControlNetHED" reimplementation lllyasviel
ships (Apache-2): a VGG-style five-block convolutional trunk where each
block ends in a 1×1 projection to a single-channel side output. It ships
**no** HED-specific kernel; the forward pass is a pure composition of ops
`brotensor` already exposes:

| Component | brotensor ops |
|---|---|
| 5-block conv trunk | `conv2d` (3×3 blocks, ReLU between), `max_pool2d` (2×2 before blocks 2–5), 1×1 `conv2d` per-block side projection |
| Side-map fusion | bilinear `interp2d` (each side → working resolution), mean, `sigmoid` |

The learned per-channel `norm` bias (the VGG mean, in the [0,255] scale the
model consumes) is **folded into the first conv's bias at load time** —
`b'[o] = b[o] − Σ_c norm[c]·Σ_k W[o,c,k]` — so the forward never needs a
broadcast subtract. Preprocessing is minimal (optional resize to a working
resolution, pack to RGB FP32 in [0,255]) and lives in `hed::preprocess`;
there is no normalize step because the `norm` bias is part of the model.

## API

The `SoftEdgeDetector` orchestrator (`brovisionml/hed.h`) loads one HED
`model.safetensors` and maps pixels to an edge probability map at the
original resolution:

```cpp
brovisionml::hed::SoftEdgeDetector det;            // default HedConfig (native res)
det.load("/path/to/hed");                           // dir holding model.safetensors
det.to(brotensor::Device::CUDA);                    // optional GPU migration
auto em = det.detect(rgb, w, h, /*channels=*/3);
// em.edge is row-major h*w FP32 in [0,1] (higher = stronger edge). em.at(x,y).
```

HED is fully convolutional, so an edge at a pixel depends only on a local
neighbourhood — `HedConfig::tile` (a `brovisionml::tiling::TileConfig`)
splits a large source into overlapping tiles, runs each at its native size,
and feathers the per-tile maps back into one full-resolution edge map. See
the tiling section of [performance.md](performance.md).

The `hed_edges` CLI tool writes a grayscale edge PNG (`edge*255`):

```bash
hed_edges /path/to/hed photo.jpg --out edges.png
# flags: --resolution N (longer-side working resolution; 0 = native)  --cuda
```

## GPU path

On CUDA the entire trunk runs **FP16**: every conv in the network is a
WMMA-covered shape (3×3 stride-1 pad-1 / 1×1), so the whole forward runs
FP16 and only the final edge map is widened for download. See
[performance.md](performance.md).

## Weights

HED ships only a pickled `ControlNetHED.pth` (`lllyasviel/Annotators`);
there is no clean HF safetensors release. The checkpoint this loader reads
is produced by a one-off, **out-of-repo** conversion of that `.pth` to
`model.safetensors` — see [weights.md](weights.md).

## Parity

Parity is validated against an out-of-repo golden dump of the reference
network (never committed): `tests/test_hed.cpp` runs the full
`SoftEdgeDetector` end to end and compares the edge map against the golden.
Whole-map agreement is **mean-abs ~2–4e-4** on the [0,1] edge map for both
square and non-square inputs. The worst single pixel can reach ~0.1: the
edge map is the sigmoid of the *mean* of five side maps bilinear-upsampled
to the working resolution (the coarsest is a 16× upsample of a 32×32 map).
brotensor's bilinear matches torch's `F.interpolate` to ~1e-7, so the resize
is exact — the residual is inherent `conv2d` FP-accumulation difference at
coarse-map edges, where the 16× upsample through the steep sigmoid turns a
sub-pixel logit shift into a narrow band of flipped fine pixels (~1% of
pixels). The map is visually identical; this is the ballpark-exact agreement
the annotator's ControlNet-conditioning role needs. The CUDA FP16 path holds
the same test gates.
