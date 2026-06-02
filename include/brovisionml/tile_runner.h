#pragma once

// Tiled inference for the dense, locally-computed annotators (HED soft edges,
// lineart). A large image is split into overlapping tiles, each run through the
// model at its own bounded size, and the per-tile maps are feathered and glued
// back into one full-resolution map. This keeps the working resolution — and so
// the GPU memory — of every pass bounded no matter how large the source is,
// while preserving native detail (unlike a whole-image downscale).
//
// Only correct for operators whose output at a pixel depends on a LOCAL
// neighbourhood (edges, lines, normals). Global / relative-scale maps (depth,
// semantic segmentation) must NOT be tiled — they need whole-image context and
// produce seams; cap their working resolution instead.
//
// The blend mechanics (feather window, weighted accumulate, normalize) live in
// broimage::tiling; this header is the brovisionml-side planning + driver that
// the per-model detect() paths share.

#include <cstdint>
#include <functional>
#include <vector>

namespace brovisionml::tiling {

// How to tile. `tile` is the working extent of a tile in pixels (each tile is
// at most tile×tile); `overlap` is how many pixels adjacent tiles share so the
// feather has room to blend. `tile == 0` disables tiling entirely.
struct TileConfig {
    int tile    = 0;   // 0 = tiling disabled
    int overlap = 0;   // shared pixels between neighbours (clamped to < tile)
};

// One planned tile: the crop rect in the original image plus the feather
// overlap to apply on each edge. Border edges carry overlap 0 (no neighbour to
// blend with, so those pixels keep full weight).
struct TileRect {
    int x = 0, y = 0, w = 0, h = 0;
    int ov_l = 0, ov_r = 0, ov_t = 0, ov_b = 0;
};

// Plan the tiles covering a (W,H) image. Returns an EMPTY vector when tiling is
// disabled (cfg.tile <= 0) or the image already fits within one tile — the
// caller should then run the whole image in a single pass. Otherwise the tiles
// step by (tile - overlap) and the trailing tile in each axis is clamped to the
// image edge (so it may be narrower/shorter than `tile`).
std::vector<TileRect> plan_tiles(int W, int H, const TileConfig& cfg);

// Drive a tiled single-channel pass. For each planned tile, `run_tile(rect)`
// must return a row-major (rect.h × rect.w) float map. The maps are feathered
// by their per-edge overlap and accumulated into a full W×H map, which is
// returned (length W*H). `tiles` must be non-empty (use plan_tiles first).
std::vector<float> blend_1ch(
    int W, int H, const std::vector<TileRect>& tiles,
    const std::function<std::vector<float>(const TileRect&)>& run_tile);

}  // namespace brovisionml::tiling
