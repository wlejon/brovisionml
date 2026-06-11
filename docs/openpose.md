# Pose (OpenPose)

Body-pose estimation — the ControlNet **openpose** conditioning annotator.
The model is the CMU multi-person 2D body-pose network (the COCO-18
keypoint, 19-limb Part-Affinity-Field model) as packaged by pytorch-openpose
and lllyasviel's `body_pose_model.pth`: a VGG-style trunk `model0` (ten 3×3
convs + three 2×2 max-pools, 3→128 channels at ⅛ spatial) followed by six
two-branch refinement stages. Each stage's branch **L1** emits a 38-channel
PAF map and branch **L2** a 19-channel confidence/heatmap (18 parts +
background); stage 1 takes the trunk feature, stages 2–6 each take
`cat([prev_L1, prev_L2, trunk])` (185ch) and apply five 7×7 convs then two
1×1 convs. It ships **no** OpenPose-specific kernel; the forward is a pure
composition of ops `brotensor` already exposes:

| Stage | brotensor ops |
|---|---|
| `model0` VGG trunk | 3×3 `conv2d` (+ bias) + `relu`, 2×2 `max_pool2d` (3 downsamples) |
| Stages 1–6 (L1 PAF + L2 heatmap) | `concat_nchw_channels` (the `[L1,L2,trunk]` re-concat each stage), 7×7 / 3×3 / 1×1 `conv2d` (+ bias) + `relu` (every conv but the final `Mconv7`/`conv5_5`) |
| Decode (host) | ×8 upsample + crop pad + resize to detect-res; per-part Gaussian-blur (σ=3, scipy-`reflect`) + local-max NMS peak detection; PAF line-integral scoring + greedy bipartite limb matching; subset-merge people assembly; prune (<4 parts or mean score <0.4) |

Input is the reference front-end: `resize_image` (shorter side → detect
resolution, rounded to a multiple of 64), RGB→**BGR** (the net was trained
on BGR), a single-scale `smart_resize` to `0.5·368/H`, right/bottom pad to a
multiple of 8 with constant 128, normalize `(x/256 − 0.5)`.

## API

The `OpenposeDetector` orchestrator (`brovisionml/openpose.h`) loads one
`model.safetensors` and maps pixels to a set of people, each up to 18
keypoints (normalized to the detect-res image):

```cpp
brovisionml::openpose::OpenposeDetector det;        // default OpenposeConfig
det.load("/path/to/openpose");                       // dir holding model.safetensors
det.to(brotensor::Device::CUDA);                     // optional GPU migration
auto pose = det.detect(rgb, w, h, /*channels=*/3);
// pose.bodies: per person std::array<Keypoint,18> (x,y normalized, score, present)
// det.infer_maps(...) exposes the raw stage-6 PAF (38ch) + heatmap (19ch).
auto canvas = brovisionml::openpose::OpenposeDetector::draw(pose);  // HxWx3 RGB
```

The `openpose_pose` CLI tool rasterizes the detected people as the canonical
colored limb-sticks-and-joints control image:

```bash
openpose_pose /path/to/openpose photo.jpg --out openpose.png
# flags: --resolution N  --cuda
```

**Body-only scope.** The canonical ControlNet openpose control image is
body-only (`include_hand=False, include_face=False` in the reference), so
only the body network is implemented here — the hand and face sub-networks
are intentionally not ported.

## GPU path

On CUDA the whole network runs **FP16** through the WMMA conv path (the 3×3,
7×7, and 1×1 convs are all WMMA-covered shapes). The peak-NMS + bipartite
matching decode is host-side. See [performance.md](performance.md).

## Weights

OpenPose ships only a pickled `body_pose_model.pth`
(`lllyasviel/Annotators`); there is no clean HF safetensors release. The
checkpoint this loader reads is produced by a one-off, **out-of-repo**
conversion of that `.pth` to `model.safetensors` (the reference `transfer`
key-remap → natural `model0.conv1_1.weight …` keys) — see
[weights.md](weights.md).

## Parity

Parity is validated against an out-of-repo golden dump of the reference
network (never committed): `tests/test_openpose.cpp` runs two gates.
**Gate 1** (the tight neural gate) compares the raw stage-6 PAF + heatmap at
network resolution — pure conv parity, **mean-abs ~4e-5 (PAF) / ~1.8e-4
(heatmap)** on the FP32 path, for both CPU and CUDA. **Gate 2** (end-to-end)
decodes people and matches them against the golden: **4/4 people recovered
with keypoint recall@6px = 1.0** on both backends. The host decode uses
Lanczos3 (broimage has no cv2 Lanczos4) and a hand-rolled scipy-`reflect`
Gaussian; the raw-map gate isolates the network from those classical-decode
approximations, which leave keypoint locations within a few pixels — the
ballpark-exact agreement the annotator's ControlNet-conditioning role needs.
