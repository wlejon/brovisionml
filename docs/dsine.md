# Surface normals (DSINE)

Per-pixel surface-normal estimation. The model is **DSINE_v02**
(Discontinuity-aware Surface-normal Estimation): an EfficientNet-B5 backbone
feeding a DPT-style decoder, then an iterative ConvGRU / AngMF ("angular
mean-field") refinement that rotation-smooths the normals in SO(3) and
convex-upsamples back to full resolution. The encoder and decoder compose
ops `brotensor` already exposes; the refinement adds the only two pieces in
the whole library with no brotensor primitive — DSINE's surface-normal
*domain* math — as brovisionml's own ops, each with a CPU FP32 path and a
CUDA kernel (`src/dsine_ops.cu`):

| Component | implementation |
|---|---|
| EfficientNet-B5 encoder | brotensor `conv2d` (stem / 1×1 / depthwise, TF-"SAME" dynamic pad), batchnorm fold, SiLU; squeeze-excite via `adaptive_avg_pool2d` + 1×1 conv + `broadcast_mul` |
| DPT-style decoder | weight-standardized `conv2d` (folded at load), bilinear `interp2d`, GroupNorm, LeakyReLU, `concat_nchw_channels`, `l2_normalize_nchw`; 2-channel uv geometry conditioning per stage |
| AngMF refinement | brotensor `conv2d` (5×5 ConvGRU gates + heads), GRU elementwise, `convex_upsample`, `slice2d`; plus brovisionml's **RayReLU** and the fused **AngMF propagate** (per-pixel per-neighbor axis-angle rotation + neighborhood unfold) ops |

## API

Unlike DPT/Depth-Anything there is **no resize and no letterbox**: the image
is consumed at native resolution and only zero-padded to a multiple of 32,
then ImageNet-normalized (`dsine::preprocess`). DSINE is
geometry-conditioned — it takes camera intrinsics; with no calibration
available they are synthesized from an assumed field-of-view (60° by
default). The `NormalEstimator` orchestrator (`brovisionml/dsine.h`) loads
one DSINE_v02 `model.safetensors` and maps pixels to a normal map at the
original resolution:

```cpp
brovisionml::dsine::NormalEstimator est;          // default DsineConfig (fov 60°)
est.load("/path/to/dsine");                        // dir holding model.safetensors
est.to(brotensor::Device::CUDA);                   // optional GPU migration
auto nm = est.estimate(rgb, w, h, /*channels=*/3);
// nm.normals is 3*h*w planar NCHW (3,H,W): channel 0=nx, 1=ny, 2=nz; each pixel
// is a UNIT normal in CAMERA space. nm.at(c,x,y) indexes a component.
// Or pass explicit pinhole intrinsics: est.estimate(rgb,w,h,3, fx,fy,cx,cy);
```

The output convention is **camera-space unit normals**; visualize each
component mapped from [-1,1] to [0,1] as `(n+1)/2` (the usual blue-ish
normal map).

The `normal_estimate` CLI tool writes that standard normal-map PNG. It runs
CPU only; use the C++ API for the CUDA path:

```bash
normal_estimate /path/to/dsine photo.jpg --out normal.png
# flags: --fov DEG   (assumed field-of-view for the synthesized intrinsics)
```

## GPU path

`est.to(Device::CUDA)` moves the whole pipeline — the brotensor-composed
encoder/decoder plus the RayReLU and AngMF-propagate CUDA kernels — onto the
GPU. On CUDA the encoder and decoder run **FP16** through the WMMA conv
path; the ConvGRU/AngMF refiner stays FP32. With `BROVISIONML_PROFILE=1`
the encoder / decoder / refine stages report individually. See
[performance.md](performance.md).

## Weights

DSINE ships only a pickled `dsine.pt` (`dylanebert/DSINE/dsine.pt`); there
is no clean HF safetensors release. The checkpoint this loader reads is
produced by a one-off, **out-of-repo** conversion of that `.pt` to
`model.safetensors` — see [weights.md](weights.md). Once converted, the
loader reads the safetensors directly via `brotensor::safetensors`.

## Parity

Parity is validated against an out-of-repo golden dump of the reference
DSINE model (generated like the Depth-Anything goldens; never committed):
`tests/test_dsine.cpp` runs the full `NormalEstimator` end to end and
`tests/test_dsine_{encoder,decoder,refine}.cpp` check each stage. The
end-to-end agreement is **exact to FP32 round-off — max-abs ~2–5e-6** on the
unit normals for both square and zero-padded inputs; the staged refinement
test bounds at max-abs ~1e-2 only to leave headroom over the 5-iteration
rotation accumulation.
