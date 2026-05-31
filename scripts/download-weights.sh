#!/usr/bin/env bash
# Download model weights for brovisionml end-to-end testing.
#
# brovisionml loads HuggingFace `safetensors` checkpoints directly (no offline
# conversion step — see CLAUDE.md), so this script just fetches the right files
# from the HF `resolve` endpoint with `curl` and lays them out under
# `weights/<model>/`. It works with any vintage of huggingface_hub (or none) —
# no Python, no HF CLI dependency.
#
# Usage:
#   scripts/download-weights.sh [model] [--repo R] [--out-dir D] [--force]
#
#   model            sam-vit-base (default) | sam-vit-large | sam-vit-huge
#                    | depth-anything-v2-small | depth-anything-v2-base
#                    | depth-anything-v2-large
#   --repo R         override the HuggingFace repo id
#   --out-dir D      override the output directory
#   --force          re-download even if the file already exists
#
# Models (all promptable-segmentation SAM checkpoints in HF `SamModel` format,
# i.e. the vision_encoder.* / prompt_encoder.* / mask_decoder.* tensor
# namespaces the loaders in src/sam_*.cpp expect — load with the matching
# SamConfig preset: vit_b / vit_l / vit_h):
#
#   sam-vit-base     facebook/sam-vit-base   (~375 MB) — smallest; the default
#                    target for a quick end-to-end smoke run.
#   sam-vit-large    facebook/sam-vit-large  (~1.25 GB)
#   sam-vit-huge     facebook/sam-vit-huge   (~2.56 GB) — the canonical "SAM",
#                    matches SamConfig::vit_h().
#
# Each repo ships a single un-sharded `model.safetensors` carrying all three
# sub-modules; `config.json` (architecture, informational — the loaders take
# their dims from the SamConfig preset) and `preprocessor_config.json` (the
# pixel mean/std, which broimage's SAM preset already encodes) come along for
# reference.
#
# Auth: these repos are public and need no token. For rate-limited or gated
# repos, export HF_TOKEN=hf_... and it is sent as a bearer token.
#
# Output: <repo>/weights/<model>/
#   config.json
#   preprocessor_config.json
#   model.safetensors

set -euo pipefail

# --- arg parsing ------------------------------------------------------------
MODEL="sam-vit-base"
REPO=""
OUT_DIR=""
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        sam-vit-base|sam-vit-large|sam-vit-huge) MODEL="$1"; shift ;;
        depth-anything-v2-small|depth-anything-v2-base|depth-anything-v2-large) MODEL="$1"; shift ;;
        --repo)    REPO="${2:?--repo needs a value}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help)
            sed -n '2,41p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- per-model file lists ---------------------------------------------------
# Every SAM checkpoint is the same three files; only the source repo differs.
# SUBDIR is the on-disk weights/<SUBDIR> name (kept HF-cased for the depth
# models so it matches what the loaders/tests look for).
SUBDIR="$MODEL"
case "$MODEL" in
    sam-vit-base)  [ -n "$REPO" ] || REPO="facebook/sam-vit-base"  ;;
    sam-vit-large) [ -n "$REPO" ] || REPO="facebook/sam-vit-large" ;;
    sam-vit-huge)  [ -n "$REPO" ] || REPO="facebook/sam-vit-huge"  ;;
    depth-anything-v2-small)
        [ -n "$REPO" ] || REPO="depth-anything/Depth-Anything-V2-Small-hf"
        SUBDIR="Depth-Anything-V2-Small" ;;
    depth-anything-v2-base)
        [ -n "$REPO" ] || REPO="depth-anything/Depth-Anything-V2-Base-hf"
        SUBDIR="Depth-Anything-V2-Base" ;;
    depth-anything-v2-large)
        [ -n "$REPO" ] || REPO="depth-anything/Depth-Anything-V2-Large-hf"
        SUBDIR="Depth-Anything-V2-Large" ;;
esac
[ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/$SUBDIR"
FILES=(
    "config.json"
    "preprocessor_config.json"
    "model.safetensors"
)

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "Model:   $MODEL"
echo "Repo:    $REPO"
echo "Target:  $OUT_DIR"
[ -n "${HF_TOKEN:-}" ] && echo "Auth:    HF_TOKEN (bearer)"
echo

# --- download helper --------------------------------------------------------
# fetch <relative-path> <dest-file> [repo] -> 0 on success, 1 on a 404,
# 2 on any other error. `repo` defaults to the model's $REPO.
fetch() {
    local rel="$1" dest="$2" repo="${3:-$REPO}"
    local url="https://huggingface.co/$repo/resolve/main/$rel"
    local auth=()
    [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")

    mkdir -p "$(dirname "$dest")"
    local code
    code="$(curl -sL --retry 3 --retry-delay 2 \
                 "${auth[@]}" \
                 -o "$dest.part" -w '%{http_code}' "$url")" || {
        echo "    curl failed for $url" >&2
        rm -f "$dest.part"
        return 2
    }
    if [ "$code" = "200" ]; then
        mv "$dest.part" "$dest"
        return 0
    fi
    rm -f "$dest.part"
    if [ "$code" = "404" ]; then return 1; fi
    echo "    HTTP $code for $url" >&2
    return 2
}

# A FILES entry may be a bare path (fetched from $REPO) or `repo|path` to pull
# that one file from a different repo — kept for parity with the sibling repos'
# multi-source models, even though every SAM file lives in one repo today.
for entry in "${FILES[@]}"; do
    f="$entry"
    ent_repo="$REPO"
    case "$entry" in
        *"|"*) ent_repo="${entry%%|*}"; f="${entry#*|}" ;;
    esac
    dest="$OUT_DIR/$f"
    if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then
        echo "==> $f  (cached, skipping)"
        continue
    fi
    echo "==> $f${ent_repo:+  [$ent_repo]}"
    if ! fetch "$f" "$dest" "$ent_repo"; then
        echo "error: download failed for $f" >&2
        exit 1
    fi
done

echo
echo "Done. Files in $OUT_DIR :"
find "$OUT_DIR" -type f | sort | while read -r p; do
    sz="$(wc -c < "$p" | tr -d ' ')"
    printf '  %12s  %s\n' "$sz" "${p#$OUT_DIR/}"
done
