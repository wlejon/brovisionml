#!/usr/bin/env python3
"""Convert an NVlabs StyleGAN3 generator pickle to a brovisionml safetensors file.

brovisionml otherwise loads checkpoints directly (no offline step), but the
released StyleGAN3 weights ship as Python *pickles* whose unpickling requires the
NVlabs `stylegan3` repo on PYTHONPATH (the modules use torch_utils.persistence).
This one-time converter flattens the EMA generator's state_dict into a flat
FP32 safetensors file whose tensor names are exactly the ones the C++ loader
reads:

    mapping.fc0.weight / .bias ... mapping.w_avg
    synthesis.input.{weight,affine.weight,affine.bias,freqs,phases,transform}
    synthesis.L{idx}_{size}_{channels}.{weight,bias,affine.weight,affine.bias,
                                        magnitude_ema}

The per-layer up_filter / down_filter buffers are dropped — brovisionml designs
those itself (Kaiser/firwin + radial jinc), matching the reference.

Usage:
    python scripts/convert-stylegan3.py STYLEGAN3.pkl OUT/model.safetensors \
        [--repo /path/to/NVlabs/stylegan3]

Requires: torch, numpy, safetensors, and (to unpickle) the NVlabs stylegan3 repo.
"""

import argparse
import os
import pickle
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", help="input StyleGAN3 .pkl (e.g. stylegan3-r-afhqv2-512x512.pkl)")
    ap.add_argument("out", help="output .safetensors path")
    ap.add_argument("--repo", default=os.environ.get("STYLEGAN3_REPO", ""),
                    help="path to the NVlabs/stylegan3 repo (needed to unpickle); "
                         "or set STYLEGAN3_REPO")
    ap.add_argument("--key", default="G_ema",
                    help="network key inside the pickle (default: G_ema)")
    args = ap.parse_args()

    import torch  # noqa: F401  (needed for unpickling tensors)
    from safetensors.torch import save_file

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
    sd = G.state_dict()

    out = {}
    for name, t in sd.items():
        if name.endswith(".up_filter") or name.endswith(".down_filter"):
            continue  # designed in C++, not loaded
        out[name] = t.detach().to(torch.float32).contiguous().cpu()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out)
    sys.stderr.write(f"wrote {len(out)} tensors to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
