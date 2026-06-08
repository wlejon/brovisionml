#!/usr/bin/env bash
# Fetch + convert an NVlabs StyleGAN3 checkpoint for brovisionml testing.
#
# Unlike the SAM / Depth-Anything models (download-weights.sh — plain HF
# safetensors, no offline step), StyleGAN3 ships as Python *pickles* that must be
# flattened to safetensors with the NVlabs repo on PYTHONPATH (see
# scripts/convert-stylegan3.py). This script does the whole chain:
#
#   1. download the released pickle from NVIDIA NGC into .cache/
#   2. clone NVlabs/stylegan3 into .cache/stylegan3-repo (needed to unpickle)
#   3. run convert-stylegan3.py -> weights/<subdir>/model.safetensors
#
# so test_stylegan3_generate / test_stylegan3_parity light up. Pass --no-convert
# to stop after the download (e.g. on a box without torch).
#
# Both config families load: config-R (rotation-equivariant, 1x1 convs, radial
# filters) and config-T (translation-equivariant, 3x3 convs, separable filters).
# The C++ loader reads each checkpoint's own channel schedule, so the only thing
# that has to match is the resolution preset (256/512/1024) and the variant.
#
# Usage:
#   scripts/download-stylegan3.sh [model] [--no-convert] [--force]
#
#   model   one of (default: stylegan3-r-ffhqu-256, the smallest, ~212 MB):
#             stylegan3-{r,t}-ffhqu-256
#             stylegan3-{r,t}-afhqv2-512
#             stylegan3-{r,t}-ffhq-1024
#             stylegan3-{r,t}-ffhqu-1024
#             stylegan3-{r,t}-metfaces-1024
#             stylegan3-{r,t}-metfacesu-1024
#   --no-convert   download the pickle only; skip the safetensors conversion
#   --force        re-download / re-convert even if outputs already exist
#
# Conversion needs: python, torch, safetensors (pip install torch safetensors).
# The pickle and the cloned repo live under .cache/ (git-ignored); only the
# converted weights/<subdir>/model.safetensors is what the tests read.

set -euo pipefail

MODEL="stylegan3-r-ffhqu-256"
CONVERT=1
FORCE=0
# The released zoo: {r,t} x {ffhqu-256, afhqv2-512, ffhq-1024, ffhqu-1024,
# metfaces-1024, metfacesu-1024}. Both variants exist for every dataset.
VALID='^stylegan3-[rt]-(ffhqu-256|afhqv2-512|ffhq-1024|ffhqu-1024|metfaces-1024|metfacesu-1024)$'
while [ $# -gt 0 ]; do
    case "$1" in
        --no-convert) CONVERT=0; shift ;;
        --force)      FORCE=1; shift ;;
        -h|--help)    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "error: unknown flag '$1' (try --help)" >&2; exit 2 ;;
        *)  if [[ "$1" =~ $VALID ]]; then MODEL="$1"; shift
            else echo "error: unknown model '$1' (try --help)" >&2; exit 2; fi ;;
    esac
done

# subdir (on-disk, matches what the tests probe) -> released pickle basename:
# the pickle just spells the resolution out as RESxRES (e.g. ...-256 -> ...-256x256).
RES="${MODEL##*-}"
PKL="${MODEL%-*}-${RES}x${RES}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE="$REPO_ROOT/.cache"
PKL_PATH="$CACHE/$PKL.pkl"
NV_REPO="$CACHE/stylegan3-repo"
OUT_DIR="$REPO_ROOT/weights/$MODEL"
OUT="$OUT_DIR/model.safetensors"

# NVIDIA NGC redirects (302) to a signed S3 URL; -L follows it.
NGC="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/$PKL.pkl"

mkdir -p "$CACHE"

echo "Model:   $MODEL"
echo "Pickle:  $PKL.pkl"
echo "Output:  $OUT"
echo

# --- 1. download the pickle -------------------------------------------------
if [ "$FORCE" -eq 0 ] && [ -s "$PKL_PATH" ]; then
    echo "==> $PKL.pkl  (cached, skipping)"
else
    echo "==> downloading $PKL.pkl from NGC"
    curl -fL --retry 3 --retry-delay 2 -o "$PKL_PATH.part" "$NGC"
    mv "$PKL_PATH.part" "$PKL_PATH"
fi
echo "    $(wc -c < "$PKL_PATH" | tr -d ' ') bytes"
echo

if [ "$CONVERT" -eq 0 ]; then
    echo "Done (download only). Convert with:"
    echo "  python scripts/convert-stylegan3.py '$PKL_PATH' '$OUT' --repo '$NV_REPO'"
    exit 0
fi

# --- 2. clone the NVlabs repo (needed to unpickle) --------------------------
if [ ! -f "$NV_REPO/torch_utils/persistence.py" ]; then
    echo "==> cloning NVlabs/stylegan3 (shallow) for unpickling"
    rm -rf "$NV_REPO"
    git clone --depth 1 https://github.com/NVlabs/stylegan3.git "$NV_REPO"
    echo
fi

# --- 3. convert to safetensors ----------------------------------------------
if [ "$FORCE" -eq 0 ] && [ -s "$OUT" ]; then
    echo "==> model.safetensors  (cached, skipping conversion)"
else
    echo "==> converting pickle -> safetensors"
    python "$SCRIPT_DIR/convert-stylegan3.py" "$PKL_PATH" "$OUT" --repo "$NV_REPO"
fi

echo
echo "Done. $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
echo "Run the gated tests with weights present, e.g.:"
echo "  ctest -C Release -R stylegan3 --output-on-failure"
