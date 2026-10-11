"""Reference fixture for brovisionml's SAM 2.1 port (tests/test_sam2.cpp).

Runs facebookresearch/sam2 (the original package, not transformers) with the
original .pt checkpoint, in exact FP32 on CUDA (TF32 off), on a synthetic
scene: a "bird" (two ellipses) on a branch over a textured gradient, moving
across four 640x480 frames. Writes tests/fixtures/sam2_base_plus.bin, a
safetensors-format file (gitignored), holding:

  frame{i}            (480, 640, 3) the uint8 RGB frames, as float32
  pixels              (3, 1024, 1024) the reference preprocessing of frame 0
  top, top_pos        (4096, 256) raw stride-16 tokens and their sine encoding
  feat_s0, feat_s1    (32, 256, 256) / (64, 128, 128) conv_s0 / conv_s1 maps
  img_{point,box,two}_{low_res,iou,logits}
                      image-predictor outputs for three prompts
  trk{stage}_f{i}_{low_res,score}
                      tracker low-res logits / object score per frame after:
                      stage 0 = the frame-0 click, 1 = forward propagation,
                      2 = a correction click on frame 2, 3 = propagating again

The tracker's frames are preprocessed the image-predictor way (float resize),
as brovisionml does, by replacing the video predictor's loaded frames.

Usage:
  python gen_sam2_fixture.py --sam2-src D:/models/src-sam2/sam2-main \
      --ckpt D:/models/sam2.1-pt/sam2.1_hiera_base_plus.pt
"""

import argparse
import json
import os
import struct
import sys
import tempfile

import numpy as np

PROMPTS = {
    "point": dict(point_coords=[[300.0, 236.0]], point_labels=[1], box=None, multimask=True),
    "box": dict(point_coords=None, point_labels=None, box=[226.0, 182.0, 382.0, 292.0],
                multimask=False),
    "two": dict(point_coords=[[300.0, 236.0], [120.0, 300.0]], point_labels=[1, 0], box=None,
                multimask=False),
}
TRACK_CLICK = ([300.0, 236.0], 1)          # frame 0
CORRECTION = (2, [362.0, 286.0], 0)        # frame, point, label (the branch under the bird)
W, H, NF = 640, 480, 4


def make_frame(i):
    rng = np.random.default_rng(1234)   # same texture every frame: a still camera
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.empty((H, W, 3), np.float32)
    img[..., 0] = 70 + 60 * xx / W
    img[..., 1] = 110 + 50 * yy / H
    img[..., 2] = 150 - 40 * xx / W
    img += rng.normal(0, 6, (H, W, 1)).astype(np.float32)
    # Branch: a dark diagonal bar.
    d = np.abs((yy - 300) - 0.25 * (xx - 320))
    img[d < 9] = [92, 64, 40]
    # Bird: body ellipse + head circle, drifting right and up.
    cx, cy = 300 + 14 * i, 236 - 6 * i
    body = ((xx - cx) / 70) ** 2 + ((yy - cy) / 42) ** 2 < 1
    head = ((xx - (cx + 58)) ** 2 + (yy - (cy - 30)) ** 2) < 26 ** 2
    img[body] = [214, 120, 38]
    img[head] = [60, 52, 48]
    eye = ((xx - (cx + 66)) ** 2 + (yy - (cy - 34)) ** 2) < 4 ** 2
    img[eye] = [240, 240, 230]
    return np.clip(img, 0, 255).astype(np.uint8)


def write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam2-src", default="D:/models/src-sam2/sam2-main")
    ap.add_argument("--ckpt", default="D:/models/sam2.1-pt/sam2.1_hiera_base_plus.pt")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "fixtures",
                                                  "sam2_base_plus.bin"))
    args = ap.parse_args()
    sys.path.insert(0, args.sam2_src)

    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.utils.transforms import SAM2Transforms

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    frames = [make_frame(i) for i in range(NF)]
    for i, f in enumerate(frames):
        out[f"frame{i}"] = f.astype(np.float32)

    def np32(t):
        return t.detach().float().cpu().numpy()

    with torch.inference_mode():
        # ── Image predictor ──
        model = build_sam2(args.config, args.ckpt, device=dev)
        pred = SAM2ImagePredictor(model)
        px = pred._transforms(frames[0])[None].to(dev)
        out["pixels"] = np32(px[0])
        backbone = model.forward_image(px)
        _, feats, pos, _ = model._prepare_backbone_features(backbone)
        out["top"] = np32(feats[-1][:, 0, :])
        out["top_pos"] = np32(pos[-1][:, 0, :])
        out["feat_s0"] = np32(feats[0][:, 0, :].permute(1, 0).reshape(32, 256, 256))
        out["feat_s1"] = np32(feats[1][:, 0, :].permute(1, 0).reshape(64, 128, 128))

        pred.set_image(frames[0])
        for name, p in PROMPTS.items():
            masks, iou, low = pred.predict(
                point_coords=None if p["point_coords"] is None else np.array(p["point_coords"]),
                point_labels=None if p["point_labels"] is None else np.array(p["point_labels"]),
                box=None if p["box"] is None else np.array(p["box"]),
                multimask_output=p["multimask"], return_logits=True)
            out[f"img_{name}_logits"] = masks.astype(np.float32)
            out[f"img_{name}_iou"] = iou.astype(np.float32)
            out[f"img_{name}_low_res"] = low.astype(np.float32)
        del pred, model

        # ── Tracker ──
        vp = build_sam2_video_predictor(
            args.config, args.ckpt, device=dev,
            hydra_overrides_extra=["++model.add_all_frames_to_correct_as_cond=true"])
        vp.fill_hole_area = 0   # brovisionml does no hole filling
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(frames):
                Image.fromarray(f).save(os.path.join(tmp, f"{i:05d}.jpg"))
            state = vp.init_state(video_path=tmp)
        tf = SAM2Transforms(resolution=vp.image_size, mask_threshold=0.0)
        imgs = torch.stack([tf(f) for f in frames]).to(dev)
        state["images"] = imgs
        state["cached_features"] = {}
        state["video_height"], state["video_width"] = H, W

        def record(stage):
            od = state["output_dict_per_obj"][0]
            td = state["temp_output_dict_per_obj"][0]
            for i in range(NF):
                o = None
                for d in (td["cond_frame_outputs"], td["non_cond_frame_outputs"],
                          od["cond_frame_outputs"], od["non_cond_frame_outputs"]):
                    if i in d:
                        o = d[i]
                        break
                if o is None:
                    continue
                out[f"trk{stage}_f{i}_low_res"] = np32(o["pred_masks"][0, 0])
                out[f"trk{stage}_f{i}_score"] = np32(o["object_score_logits"].reshape(1))

        vp.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                                 points=np.array([TRACK_CLICK[0]], np.float32),
                                 labels=np.array([TRACK_CLICK[1]], np.int32))
        record(0)
        for _ in vp.propagate_in_video(state):
            pass
        record(1)
        fi, pt, lb = CORRECTION
        vp.add_new_points_or_box(state, frame_idx=fi, obj_id=1,
                                 points=np.array([pt], np.float32),
                                 labels=np.array([lb], np.int32))
        out["trk2_frame"] = np.array([fi], np.float32)
        o = state["temp_output_dict_per_obj"][0]["cond_frame_outputs"][fi]
        out[f"trk2_f{fi}_low_res"] = np32(o["pred_masks"][0, 0])
        out[f"trk2_f{fi}_score"] = np32(o["object_score_logits"].reshape(1))
        for _ in vp.propagate_in_video(state):
            pass
        record(3)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_safetensors(args.out, out)
    print("wrote", os.path.abspath(args.out), len(out), "tensors")
    for k in sorted(out):
        if k.endswith("_iou") or k.endswith("_score"):
            print(" ", k, out[k])


if __name__ == "__main__":
    main()
