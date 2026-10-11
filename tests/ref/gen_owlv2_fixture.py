#!/usr/bin/env python3
"""Regenerate the OWLv2 parity fixture.

Validates brovisionml's OWLv2 detector (src/owlv2*.cpp) against the genuine
upstream model: transformers' Owlv2ForObjectDetection loaded from
google/owlv2-base-patch16-ensemble, FP32 on the CPU. It dumps, for one
deterministic synthetic image (gradients, rings and two dark ellipses; no
photograph) and a handful of text queries:

    * the RGB image (1280 x 800, so the square pad and the anti-aliased
      downscale to 960 both run) and the normalised pixel tensor the OWLv2
      recipe makes of it: pad to a square at the bottom/right with 0.5 grey
      (in [0, 1], before normalisation; the original big_vision recipe),
      Gaussian anti-alias with sigma = (factor - 1) / 2 and a kernel of
      2 * ceil(3 * sigma) + 1 taps (reflect border), bilinear resize to 960
      (align_corners=False, no antialias), CLIP mean/std. The resize half is
      checked against transformers' Owlv2ImageProcessor on a square image,
    * vision stages: the token embeddings (class + patches + positions, before
      pre_layernorm), the hidden state after block 0 and after the last block,
      and the detector's image embeddings (post_layernorm, times the class
      token, layer_norm: the (3600, 768) feature the heads read),
    * per query: the CLIP token ids (BOS ... EOS, no padding) and the
      projected, L2-normalised text embedding,
    * the heads: class logits (3600, Q), boxes (3600, 4, cxcywh relative to
      the padded square, after the box bias and the sigmoid) and objectness
      logits (3600),
    * the detections: max over queries, sigmoid, top 20 after greedy NMS at
      IoU 0.3 (torchvision.ops.nms), boxes as [x0, y0, x1, y1] pixels of the
      original image (the padded square's coordinates, clipped to the image).

The fixture is gitignored (*.bin): regenerate locally to enable the numeric
checks in tests/test_owlv2_parity.cpp (it skips when it is absent).

Requirements: torch, torchvision, transformers, numpy (CPU). The model:
    hf download google/owlv2-base-patch16-ensemble \
        --local-dir ~/models/owlv2-base-patch16-ensemble

Usage: python3 -I tests/ref/gen_owlv2_fixture.py [--model DIR] [--out FILE]

Fixture layout (little-endian, brolm's CLIP fixtures' format): the magic
"OWL2FIX1", int32 record count, then per record: int32 name length, the name
bytes, int32 dtype (0 f32, 1 i32, 2 utf-8 bytes), int32 ndim, int32 dims[ndim],
the data.
"""
import argparse
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "tests", "fixtures", "owlv2.bin")

QUERIES = [
    "a bird",
    "a squirrel",
    "a photo of a deer",
    "an empty bird feeder",
    "A Red-tailed Hawk!",
]
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
SIZE = 960
NMS_IOU = 0.3
TOP = 20


def synthetic_image(w, h):
    """RGB uint8 (h, w, 3): smooth gradients, rings and two dark ellipses."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x /= (w - 1)
    y /= (h - 1)
    r = 0.2 + 0.6 * x
    g = 0.25 + 0.5 * y * (1.0 - 0.3 * x)
    d = np.sqrt((x - 0.7) ** 2 + (y - 0.3) ** 2)
    b = 0.5 + 0.35 * np.sin(d * 30.0)
    rgb = np.stack([r, g, b], axis=-1)
    e1 = ((x - 0.35) / 0.12) ** 2 + ((y - 0.55) / 0.10) ** 2 < 1.0
    e2 = ((x - 0.75) / 0.05) ** 2 + ((y - 0.75) / 0.08) ** 2 < 1.0
    rgb[e1] = rgb[e1] * 0.3 + np.array([0.35, 0.25, 0.15], np.float32) * 0.7
    rgb[e2] *= 0.2
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def resize_960(img):
    """(1, 3, S, S) float in [0, 1] -> (1, 3, 960, 960): the OWLv2 resize."""
    import torchvision.transforms.v2.functional as tvF
    s = img.shape[-1]
    factor = s / SIZE
    sigma = max((factor - 1.0) / 2.0, 0.0)
    if sigma > 0:
        k = 2 * math.ceil(3 * sigma) + 1
        img = tvF.gaussian_blur(img, [k, k], [sigma, sigma])
    return F.interpolate(img, size=(SIZE, SIZE), mode="bilinear", align_corners=False,
                         antialias=False)


def preprocess(rgb):
    h, w, _ = rgb.shape
    s = max(h, w)
    img = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    padded = torch.full((1, 3, s, s), 0.5)
    padded[:, :, :h, :w] = img
    out = resize_960(padded)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (out - mean) / std


def check_resize_against_processor(model_dir):
    """The hand resize must be transformers' on an image that needs no pad."""
    from transformers import Owlv2ImageProcessor
    proc = Owlv2ImageProcessor.from_pretrained(model_dir)
    rgb = synthetic_image(1100, 1100)
    via = proc(images=rgb, return_tensors="pt")["pixel_values"]
    mine = preprocess(rgb)
    err = (via - mine).abs().max().item()
    assert err < 1e-4, "hand preprocessing != Owlv2ImageProcessor: %g" % err


class Writer:
    def __init__(self):
        self.recs = []

    def add(self, name, arr, dtype=None):
        if isinstance(arr, str):
            data = arr.encode("utf-8")
            self.recs.append((name, 2, [len(data)], data))
            return
        a = np.asarray(arr)
        if dtype == "i32" or a.dtype.kind in "iub":
            a = a.astype("<i4")
            code = 1
        else:
            a = a.astype("<f4")
            code = 0
        self.recs.append((name, code, list(a.shape) or [1], a.tobytes()))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"OWL2FIX1")
            f.write(struct.pack("<i", len(self.recs)))
            for name, code, dims, data in self.recs:
                nb = name.encode("utf-8")
                f.write(struct.pack("<i", len(nb)))
                f.write(nb)
                f.write(struct.pack("<ii", code, len(dims)))
                f.write(struct.pack("<%di" % len(dims), *dims))
                f.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/owlv2-base-patch16-ensemble"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import torchvision
    from transformers import CLIPTokenizer, Owlv2ForObjectDetection

    torch.set_grad_enabled(False)
    check_resize_against_processor(args.model)
    model = Owlv2ForObjectDetection.from_pretrained(args.model, attn_implementation="eager").eval()
    tok = CLIPTokenizer.from_pretrained(args.model)

    w = Writer()
    W, H = 1280, 800
    rgb = synthetic_image(W, H)
    pixels = preprocess(rgb)
    w.add("image/rgb", rgb, "i32")
    w.add("image/pixels", pixels[0].numpy())

    # Vision tower, walked by hand so the intermediate stages are exposed.
    vm = model.owlv2.vision_model
    emb = vm.embeddings(pixels)
    w.add("vision/embeddings", emb[0].numpy())
    x = vm.pre_layernorm(emb)
    layers = vm.encoder.layers
    for i, layer in enumerate(layers):
        x = layer(x, None)
        if isinstance(x, tuple):
            x = x[0]
        if i == 0:
            w.add("vision/block0", x[0].numpy())
    w.add("vision/last_hidden", x[0].numpy())
    feature_map, _ = model.image_embedder(pixel_values=pixels)
    image_feats = feature_map.reshape(1, -1, feature_map.shape[-1])
    # The hand walk must be the model's own.
    post = vm.post_layernorm(x)
    walk = model.layer_norm(post[:, 1:, :] * post[:, :1, :])
    assert torch.allclose(walk, image_feats, atol=1e-5), "hand vision walk != image_embedder"
    w.add("vision/image_embeds", image_feats[0].numpy())

    # Text queries: CLIP ids with BOS/EOS, no padding (the tower is causal,
    # so the pooled EOS row does not see the padding anyway).
    ids_list = [tok(q)["input_ids"] for q in QUERIES]
    w.add("text/queries", "\n".join(QUERIES))
    for i, ids in enumerate(ids_list):
        w.add("text/ids/%d" % i, np.array(ids), "i32")
    maxlen = 16
    padded_ids = torch.zeros((len(QUERIES), maxlen), dtype=torch.long)
    mask = torch.zeros((len(QUERIES), maxlen), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded_ids[i, :len(ids)] = torch.tensor(ids)
        mask[i, :len(ids)] = 1
    out = model(input_ids=padded_ids, pixel_values=pixels, attention_mask=mask)
    w.add("text/embeds", out.text_embeds[0].numpy())

    w.add("heads/logits", out.logits[0].numpy())
    w.add("heads/boxes", out.pred_boxes[0].numpy())
    w.add("heads/objectness", out.objectness_logits[0].numpy())

    # Detections: best query per box, sigmoid, NMS, top TOP, original pixels.
    logits = out.logits[0]
    best, label = logits.max(dim=-1)
    scores = torch.sigmoid(best)
    s = max(W, H)
    cx, cy, bw, bh = out.pred_boxes[0].unbind(-1)
    xyxy = torch.stack([(cx - bw / 2) * s, (cy - bh / 2) * s,
                        (cx + bw / 2) * s, (cy + bh / 2) * s], dim=-1)
    xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, W)
    xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, H)
    keep = torchvision.ops.nms(xyxy, scores, NMS_IOU)[:TOP]
    w.add("detect/boxes", xyxy[keep].numpy())
    w.add("detect/scores", scores[keep].numpy())
    w.add("detect/labels", label[keep].numpy(), "i32")
    w.add("detect/params", np.array([NMS_IOU, float(TOP)], np.float32))

    w.save(args.out)
    print("wrote", args.out)
    for i in range(min(5, len(keep))):
        print("  %-24s %.4f  %s" % (QUERIES[int(label[keep[i]])], float(scores[keep[i]]),
                                    [round(v, 1) for v in xyxy[keep[i]].tolist()]))


if __name__ == "__main__":
    main()
