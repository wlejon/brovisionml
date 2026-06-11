# SAM (Segment Anything)

Promptable segmentation: a ViT image encoder + prompt encoder (points /
boxes / mask) + lightweight mask decoder, tied together by a `Sam`
orchestrator and the `sam_segment` CLI driver. The automatic mask generator
("segment everything") rides on the same orchestrator — see below.

SAM splits cleanly into three pieces, and every piece maps onto ops
`brotensor` already exposes:

| SAM component | brotensor ops |
|---|---|
| ViT image encoder | `conv2d` (patch embed), `self_attention_bias_forward` (decomposed relative-position bias, the same additive-bias attention T5 uses), `layernorm`, `gelu` |
| Prompt encoder | positional encoding of point/box coords + learned embeddings (`embedding_lookup`, elementwise) |
| Mask decoder | two-way `cross_attention_forward`, `conv_transpose2d` (4× upscale), `interp2d`, MLP heads (`linear`) + IoU head |

Preprocessing (resize longest-side → normalize with `broimage::SAM_MEAN/STD`
→ pad to 1024×1024) is `broimage`'s job; the model takes the prepared pixel
tensor.

## API

Encode-image-once / decode-many-prompts is the natural split: a slow image
encode followed by cheap per-click mask decodes. The `Sam` orchestrator
(`brovisionml/sam.h`) wires the three pieces together — load one HF
`model.safetensors`, `set_image()` once, then `segment()` with points /
boxes given in original-image pixel coordinates and get masks back at the
original resolution:

```cpp
brovisionml::sam::Sam sam(brovisionml::sam::SamConfig::vit_h());
sam.load("/path/to/sam-vit-huge");          // dir holding model.safetensors
sam.to(brotensor::Device::CUDA);             // optional GPU migration
sam.set_image(rgb, w, h, /*channels=*/3);    // preprocess + ViT encode (once)
auto seg = sam.segment({{x, y}}, {1}, {});    // a foreground click
// seg.logits[seg.best()*h*w ...] — threshold at 0 for a binary mask
```

`SamConfig::vit_b()` / `vit_l()` / `vit_h()` select the variant; fetch
checkpoints with `scripts/download-weights.sh sam-vit-{base,large,huge}`.

The `sam_segment` CLI tool is the same flow from the shell:

```bash
sam_segment /path/to/sam-vit-huge photo.jpg --point 320,240 --out mask.png
# flags: --point X,Y  --bg-point X,Y  --box X1,Y1,X2,Y2  (each repeatable)
#        --variant vit_h|vit_l|vit_b  --single  --cuda
```

## GPU path

On CUDA the image encoder runs mixed-precision FP16 (the q/k/v/o and
patch-embed GEMMs; residual stream and norms stay FP32) and the mask
decoder's attention projections run FP16. Mask upscale, crop, and resize
happen on-device, and batched point prompts decode device-resident with the
predicted-IoU filter applied *before* the expensive upscale — only surviving
masks are ever downloaded (as packed binary masks, not FP32 logits). See
[performance.md](performance.md).

## Automatic mask generator ("segment everything")

`AutomaticMaskGenerator` (`brovisionml/sam_amg.h`) is the C++ port of
`segment_anything`'s `SamAutomaticMaskGenerator`: it samples a regular grid
of foreground points, decodes a multi-mask proposal at each, and filters the
pile down to a clean set — no prompts required. The pipeline matches
upstream:

regular point grid → per-point multi-mask decode → predicted-IoU filter →
stability-score filter → binarize + box + crop-edge drop → box-NMS (within
and across crops) → optional connected-components small-region / hole
cleanup → masks sorted by area.

It composes a `Sam` you already loaded — `set_image()` is called for you per
crop — so it runs on the model's backend, where the IoU / stability filters
and binarization also run device-side:

```cpp
brovisionml::sam::Sam sam(brovisionml::sam::SamConfig::vit_h());
sam.load("/path/to/sam-vit-huge");

brovisionml::sam::AmgConfig cfg;        // upstream defaults
cfg.points_per_side = 32;               // 32x32 grid (the SAM default)
brovisionml::sam::AutomaticMaskGenerator gen(sam, cfg);

auto masks = gen.generate(rgb, w, h, /*channels=*/3);
// each: .mask (h*w, 1=fg), .bbox {x,y,w,h}, .area,
//       .predicted_iou, .stability_score, .point, .crop_box
```

The grid points are decoded in batches (`AmgConfig::points_per_batch`,
default 64) through one batched mask-decoder pass — independent prompts
packed into a single block-diagonal attention call — so the per-click decode
overhead is amortized. The slow ViT encode still happens once per crop.

Knobs worth knowing (all on `AmgConfig`, defaults mirror upstream):

| Field | Meaning |
|---|---|
| `points_per_side` | grid density (N×N points); `32` is the SAM default |
| `points_per_batch` | grid points decoded per batched pass (`64`) |
| `pred_iou_thresh` | drop masks below this predicted IoU (`0.88`) |
| `stability_score_thresh` | drop masks below this binarization stability (`0.95`) |
| `box_nms_thresh` / `crop_nms_thresh` | NMS IoU within / across crops (`0.7`) |
| `crop_n_layers` | crop-pyramid layers for higher recall on big images (`0`) |
| `min_mask_region_area` | remove islands / holes smaller than this (`0` = off) |

The `sam_amg` CLI tool writes a colored overlay PNG with every generated
mask:

```bash
sam_amg /path/to/sam-vit-huge photo.jpg --points-per-side 32 --out everything.png
# other flags: --pred-iou-thresh --stability-thresh --crop-n-layers
#              --min-region-area --points-per-batch --variant --cuda
```
