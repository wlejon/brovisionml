# Depth-Anything-V2

Monocular relative-depth estimation. The model is a DINOv2 ViT backbone
feeding a DPT decoder — the same `image → dense map` shape SAM's encoder +
head has — and every piece maps onto ops `brotensor` already exposes:

| Component | brotensor ops |
|---|---|
| DINOv2 backbone | `conv2d` (patch embed), `mha_forward` (plain global self-attention), `layernorm`, `gelu`; absolute position embedding bicubically interpolated to the patch grid; per-block LayerScale folded into the projections at load |
| DPT reassemble | `conv2d` (1×1 project), `conv_transpose2d` (up) / `interp2d` / strided `conv2d` (down) |
| DPT RefineNet fusion | pre-activation residual `conv2d` units + `interp2d_align_corners_forward` (the align_corners=True bilinear upsample DPT needs) + 1×1 `conv2d` |
| Depth head | `conv2d` + align-corners upsample + ReLU |

The DINOv2 backbone and DPT head live in `dinov2.h` / `dpt_head.h` as a
reusable substrate (DSINE's decoder is DPT-style too).

## API

Preprocessing (aspect-preserving resize to a multiple of 14 → ImageNet
normalize, no padding) is `broimage`'s job via `dpt::preprocess`. The
`DepthEstimator` orchestrator (`brovisionml/depth_anything.h`) loads one HF
`DepthAnythingForDepthEstimation` `model.safetensors` and maps pixels to a
depth map at the original resolution:

```cpp
brovisionml::depth::DepthEstimator est(
    brovisionml::depth::DepthAnythingConfig::v2_small());
est.load("/path/to/Depth-Anything-V2-Small");   // dir holding model.safetensors
est.to(brotensor::Device::CUDA);                 // optional GPU migration
auto dm = est.estimate(rgb, w, h, /*channels=*/3);
// dm.depth is row-major h*w relative inverse-depth (nearer = larger); not metric
```

The `depth_estimate` CLI tool writes a min-max-normalized grayscale depth
PNG (brighter = nearer):

```bash
depth_estimate /path/to/Depth-Anything-V2-Small photo.jpg --out depth.png
# flags: --variant small|base|large  --invert  --cuda
```

Fetch a checkpoint with `scripts/download-weights.sh depth-anything-v2-small`
(`-base` / `-large` for the larger ViT-B / ViT-L variants). The output is
*relative* depth in Depth-Anything's convention (larger = nearer), not
metric.

## GPU path

On CUDA the DINOv2 backbone runs mixed-precision FP16: the q/k/v and fc1
GEMMs narrow to FP16, while the o-projection / fc2 (which carry the folded
LayerScale), the residual stream, and the norms stay FP32. The DPT head runs
FP32. See [performance.md](performance.md).

## Parity

Parity is measured against the actual HF `DepthAnythingForDepthEstimation`
output, not just asserted: `tests/test_depth_parity.cpp` compares
brovisionml to a golden dump of the Python model (generated out-of-repo; the
goldens download like the weights). At a **square 518²** input — where the
DPT resize is identity and DINOv2's position embedding is used at its native
grid (no interpolation) — brovisionml agrees with HF to FP32 round-off:
depth **rel-max ~1.4e-5**, preprocess ~7e-7, backbone stages ~5e-4. The
model path is exact.

For **non-square** inputs the agreement is ~0.1% mean / ~0.7% max. DINOv2's
position-embedding interpolation is faithful — brotensor's bicubic mode 3
uses the torch coefficient a=-0.75 (`torch.interpolate(mode="bicubic")`),
matched exactly; mode 2 (a=-0.5, Catmull-Rom / PIL) is reserved for the
PIL-matching image resize. The DPT fusion/head upsamples are bilinear (no
cubic coefficient). The residual max is a preprocessing nuance: `broimage`'s
bicubic image resize matches PIL in the interior but differs at image
borders / hard edges (PIL renormalizes boundary weights), a few localized
pixels. Square inputs avoid the resize entirely and are exact.
