#!/usr/bin/env python3
"""Dump a StyleGAN3 reference golden for brovisionml's numeric-parity test.

Runs the NVlabs reference generator (the same EMA network
scripts/convert-stylegan3.py flattens) for a fixed seed and writes a small
binary the C++ test (tests/test_stylegan3_parity.cpp) replays: the exact latent
`z`, the mapped `w+`, and the raw FP32 synthesis image. The C++ side loads the
*converted* checkpoint, feeds the same `z`/`w+`, and compares — closing the loop
from the released pickle to brovisionml's own forward pass.

Like every other brovisionml golden, the `.bin` is generated out-of-repo and is
never committed (weights/ is .gitignored); this script *is* committed so the
parity is reproducible, exactly as scripts/convert-stylegan3.py is.

Golden format `BVMLSG31` (little-endian):
    magic(8)               "BVMLSG31"
    version(i32)           1
    z_dim,num_ws,w_dim(i32)
    channels,height,width(i32)
    truncation_psi(f32)
    truncation_cutoff(i32) -1 == all rows (the reference's None)
    z[z_dim]               f32
    ws[num_ws*w_dim]       f32   (mapping output, NCHW-irrelevant; row-major)
    img[channels*H*W]      f32   (synthesis output, NCHW, raw ~[-1,1])

Usage:
    python scripts/dump-stylegan3-golden.py STYLEGAN3.pkl OUT/golden_stylegan3.bin \
        [--repo /path/to/NVlabs/stylegan3] [--seed 42] [--trunc 0.7]

Requires: torch, numpy, and the NVlabs stylegan3 repo (to unpickle + run).
"""

import argparse
import os
import pickle
import struct
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", help="input StyleGAN3 .pkl")
    ap.add_argument("out", help="output golden .bin path")
    ap.add_argument("--repo", default=os.environ.get("STYLEGAN3_REPO", ""),
                    help="path to the NVlabs/stylegan3 repo (needed to unpickle); "
                         "or set STYLEGAN3_REPO")
    ap.add_argument("--key", default="G_ema", help="network key (default: G_ema)")
    ap.add_argument("--seed", type=int, default=42, help="latent seed (default: 42)")
    ap.add_argument("--trunc", type=float, default=0.7,
                    help="truncation psi (default: 0.7); 1.0 disables truncation")
    ap.add_argument("--cutoff", type=int, default=-1,
                    help="truncation cutoff rows; -1 == all (reference None)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="reference device (default: cpu). The released NVlabs "
                         "ops JIT-compile fragile CUDA plugins on first use; the "
                         "CPU path uses the pure-PyTorch reference and needs no "
                         "build toolchain. The golden is FP32 reference truth "
                         "either way — brovisionml inference still runs on CUDA.")
    args = ap.parse_args()

    import numpy as np
    import torch

    if args.repo:
        sys.path.insert(0, args.repo)

    try:
        with open(args.pkl, "rb") as f:
            data = pickle.load(f)
    except ModuleNotFoundError as e:
        sys.stderr.write(
            f"error: unpickling needs the NVlabs stylegan3 repo on PYTHONPATH "
            f"({e}). Pass --repo /path/to/stylegan3.\n")
        return 1

    G = data[args.key] if isinstance(data, dict) else data
    device = torch.device(args.device)
    G = G.eval().requires_grad_(False).to(device)

    z_dim = G.z_dim
    # Match NVlabs gen_images.py: a per-seed RandomState draw, fp32.
    z_np = np.random.RandomState(args.seed).randn(1, z_dim).astype(np.float32)
    z = torch.from_numpy(z_np).to(device)
    c = torch.zeros([1, G.c_dim], device=device)

    cutoff = None if args.cutoff < 0 else args.cutoff
    ws = G.mapping(z, c, truncation_psi=args.trunc, truncation_cutoff=cutoff)
    # force_fp32 so the golden matches brovisionml's all-FP32 synthesis path.
    img = G.synthesis(ws, noise_mode="const", force_fp32=True)

    ws_np = ws.detach().float().cpu().numpy().reshape(-1)        # (num_ws*w_dim,)
    img_np = img.detach().float().cpu().numpy().reshape(-1)      # (C*H*W,)

    num_ws = int(G.num_ws)
    w_dim = int(G.w_dim)
    C = int(G.img_channels)
    H = W = int(G.img_resolution)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"BVMLSG31")
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<3i", z_dim, num_ws, w_dim))
        f.write(struct.pack("<3i", C, H, W))
        f.write(struct.pack("<f", float(args.trunc)))
        f.write(struct.pack("<i", int(args.cutoff)))
        f.write(z_np.reshape(-1).astype("<f4").tobytes())
        f.write(ws_np.astype("<f4").tobytes())
        f.write(img_np.astype("<f4").tobytes())

    sys.stderr.write(
        f"wrote golden to {args.out}: z[{z_dim}] ws[{num_ws}x{w_dim}] "
        f"img[{C}x{H}x{W}] (seed {args.seed}, psi {args.trunc}, device {device.type})\n")
    sys.stderr.write(
        f"  img range [{img_np.min():.4f}, {img_np.max():.4f}]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
