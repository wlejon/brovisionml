# Background removal (BiRefNet)

Foreground/background matting — dichotomous image segmentation. The model is
ZhengPeng7's **BiRefNet** (the v1 Swin-L release), as used by TripoSplat's
`rmbg` front-end: RGB in, single-channel alpha matte in [0,1] out. It ships
**no** BiRefNet-specific kernel; the forward is a pure composition of ops
`brotensor` already exposes:

| Component | brotensor ops |
|---|---|
| Swin-L backbone | 4×4 stride-4 `conv2d` patch embed; windowed attention via `self_attention_bias_forward` (relative-position + shifted-window bias) with the pad / cyclic-roll / window-partition / patch-merge mechanics expressed as precomputed INT32 row permutations driven through `gather_rows`; `layernorm`, `linear`, `gelu` |
| Dual-resolution concat | the backbone runs twice (full and half resolution); the half-res stage features are upsampled (`interp2d_align_corners_forward`) and channel-concatenated stage-wise (`mul_scl_ipt='cat'`, doubling [192,384,768,1536] → [384,…,3072]) |
| ASPP-deformable decoder | modulated DCNv2 branches via `deform_conv2d` (k=1/3/7 + 1×1 + global-pool branch), `batch_norm_inference`, `conv2d`, `concat_nchw_channels` |
| 4-level decode + GDT | per-level decode blocks with lateral 1×1 skips, image-patch re-injection, gradient-decoder-triggering gates (conv→bn→relu → per-pixel 1×1 attention → `sigmoid` → `broadcast_mul`), final 1×1 conv → `sigmoid` |

Swin's window mechanics never leave the device: the permutation index
tensors are host-computed per H/W/shift but ride the same device as the
activations.

## API

`BiRefNet` (`brovisionml/birefnet.h`) is a self-contained class — no
separate config; the Swin-L architecture is fixed:

```cpp
brovisionml::birefnet::BiRefNet net;
net.load("/path/to/birefnet.safetensors");   // F32/F16/BF16 widened on load
net.to(brotensor::Device::CUDA);

auto matte = net.removeBackground(rgb_fp32, origW, origH,
                                  /*rgbIs255=*/true, /*modelSize=*/1024);
// matte.alpha: row-major origW*origH floats in [0,1]
```

`removeBackground()` handles the full round trip: resize to `modelSize`²
(a multiple of 32; default 1024), ImageNet-normalize, forward, sigmoid,
resize back to the original size. `forwardLogits(imgNCHW, H, W)` exposes the
raw pre-sigmoid logits for callers that preprocess themselves, and
`debugBackbone()` exposes the four raw Swin stage features.

There is no CLI tool; the GPU path runs FP32 (no FP16 path yet).

## Weights & tests

Fetch with `scripts/download-triposplat.sh birefnet`
(`VAST-AI/TripoSplat`, `background_removal/birefnet.safetensors`, ~444 MB).
No `config.json`; the architecture is hard-coded.

`tests/test_birefnet_backbone.cpp` validates against out-of-repo golden
dumps of the PyTorch reference (skipping cleanly when absent): the raw
Swin-L backbone stage features at mean-abs < 2e-3, and the full
input→alpha pipeline at mean-abs < 3e-3, on both CPU and GPU when
available.
