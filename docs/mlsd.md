# Straight lines (MLSD)

Straight line-segment detection — the ControlNet **mlsd** conditioning
annotator. The model is M-LSD (Mobile Line Segment Detection, NAVER 2021),
the `MobileV2_MLSD_Large` network lllyasviel ships: a truncated MobileNetV2
backbone (four-channel input = RGB + a constant plane; five FPN taps)
feeding an FPN-style decode head, whose final 16-channel conv output is
sliced to its last 9 channels as a "TP map" at half resolution (256×256 for
the 512×512 input). It ships **no** MLSD-specific kernel; the forward is a
pure composition of ops `brotensor` already exposes:

| Stage | brotensor ops |
|---|---|
| MobileNetV2 backbone | grouped (depthwise) `conv2d`, 1×1 `conv2d`, `pad2d` (TFLite right/bottom stride-2 pad), ReLU6 via `clamp(0,6)`; BatchNorm **folded into each preceding conv at load** |
| FPN decode head | 1×1 / 3×3 / dilated (pad 5, dil 5) `conv2d`, align-corners bilinear `interp2d` (the ×2 upsamples), `concat_nchw_channels` (the skip merges), residual `add_inplace` |
| TP-map decode (host) | channel 0 = center heatmap (`sigmoid` → 3×3 `max_pool2d` NMS → top-K), channels 1:5 = per-pixel start/end displacements → line segments, kept by score + length thresholds, scaled 256→512→original |

Folding BatchNorm into the conv (`W'=W·s`, `b'=(b−μ)·s+β`, `s=γ/√(var+ε)`)
keeps the forward a clean conv→activation sequence with no separate norm op.
Input is normalized `(x/127.5−1)`; the appended ones plane is normalized
too, so the 4th channel is the constant `1/127.5−1`.

## API

The `MLSDdetector` orchestrator (`brovisionml/mlsd.h`) loads one MLSD
`model.safetensors` and maps pixels to a list of line segments in the
original image's coordinates:

```cpp
brovisionml::mlsd::MLSDdetector det;               // default MlsdConfig (thr 0.1)
det.load("/path/to/mlsd");                          // dir holding model.safetensors
det.to(brotensor::Device::CUDA);                    // optional GPU migration
auto lm = det.detect(rgb, w, h, /*channels=*/3);
// lm.segments: {x1,y1,x2,y2,score}, original-image pixels.
// det.infer_tpmap(...) exposes the raw 9-channel TP map for custom decodes.
```

The `mlsd_lines` CLI tool rasterizes the detected segments as white lines on
black (the conditioning image):

```bash
mlsd_lines /path/to/mlsd photo.jpg --out mlsd.png
# flags: --score-thr F  --dist-thr F  --cuda
```

The GPU path runs FP32 (no FP16 path; the network is small and the TP-map
decode is host-side anyway).

## Weights

MLSD ships only a pickled `mlsd_large_512_fp32.pth`
(`lllyasviel/Annotators`); there is no clean HF safetensors release. The
checkpoint this loader reads is produced by a one-off, **out-of-repo**
conversion of that `.pth` to `model.safetensors` — see
[weights.md](weights.md).

## Parity

Parity is validated against an out-of-repo golden dump of the reference
network (never committed): `tests/test_mlsd.cpp` compares the raw 9-channel
TP map (the tight neural gate, **mean-abs ~1e-4**) and the decoded segment
set. CPU and CUDA both reconstruct the **identical** segment set (recall 1.0
vs the reference's). The worst single TP logit differs by ~1.5e-2 between
the backends — the same order as each one's gap from torch, an expected
FP-accumulation difference across the deep depthwise/dilated trunk on logits
that span tens; it is far below what the sigmoid + NMS + thresholds amplify
into a segment change, which is why the decoded lines are identical. This is
the ballpark-exact agreement the annotator's ControlNet-conditioning role
needs.
