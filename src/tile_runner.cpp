#include "brovisionml/tile_runner.h"

#include "broimage/tiling.h"

#include <algorithm>
#include <cstddef>

namespace brovisionml::tiling {

namespace {

// Tile start offsets covering [0, extent). Steps by `step`; the trailing tile
// is snapped back to `extent - tile` so every tile is the FULL `tile` size (it
// just overlaps its neighbour by more than `overlap` at the edge). Keeping tiles
// uniform avoids feeding a thin ragged strip — which would collapse to a
// degenerate size through a model's downsampling stack — and still guarantees
// gap-free coverage. Caller handles `extent <= tile` (single pass) before here.
std::vector<int> axis_starts(int extent, int tile, int step) {
    std::vector<int> starts;
    if (extent <= tile) { starts.push_back(0); return starts; }  // single band
    for (int p = 0; ; p += step) {
        if (p + tile >= extent) {        // last tile: snap to the edge, full size
            starts.push_back(extent - tile);
            break;
        }
        starts.push_back(p);
    }
    return starts;
}

}  // namespace

std::vector<TileRect> plan_tiles(int W, int H, const TileConfig& cfg) {
    std::vector<TileRect> tiles;
    if (cfg.tile <= 0 || W <= 0 || H <= 0) return tiles;
    if (W <= cfg.tile && H <= cfg.tile) return tiles;   // single whole-image pass

    const int tile = cfg.tile;
    const int overlap = std::clamp(cfg.overlap, 0, tile - 1);
    const int step = std::max(1, tile - overlap);

    const std::vector<int> xs = axis_starts(W, tile, step);
    const std::vector<int> ys = axis_starts(H, tile, step);

    for (int y : ys) {
        const int h = std::min(tile, H - y);
        for (int x : xs) {
            const int w = std::min(tile, W - x);
            TileRect t;
            t.x = x; t.y = y; t.w = w; t.h = h;
            // Feather only toward a neighbour; border edges keep full weight.
            t.ov_l = (x > 0)       ? overlap : 0;
            t.ov_t = (y > 0)       ? overlap : 0;
            t.ov_r = (x + w < W)   ? overlap : 0;
            t.ov_b = (y + h < H)   ? overlap : 0;
            tiles.push_back(t);
        }
    }
    return tiles;
}

std::vector<float> blend_1ch(
    int W, int H, const std::vector<TileRect>& tiles,
    const std::function<std::vector<float>(const TileRect&)>& run_tile) {
    std::vector<float> acc(static_cast<std::size_t>(W) * H, 0.0f);
    std::vector<float> wacc(static_cast<std::size_t>(W) * H, 0.0f);
    std::vector<float> win;

    for (const TileRect& t : tiles) {
        std::vector<float> map = run_tile(t);   // (t.h × t.w) row-major
        win.assign(static_cast<std::size_t>(t.w) * t.h, 0.0f);
        broimage::feather_window_f32(win.data(), t.w, t.h,
                                     t.ov_l, t.ov_r, t.ov_t, t.ov_b);
        broimage::accumulate_tile_f32(acc.data(), wacc.data(), W, H, /*channels=*/1,
                                      map.data(), t.w, t.h, t.x, t.y, win.data());
    }
    broimage::normalize_accumulator_f32(acc.data(), wacc.data(),
                                        W * H, /*channels=*/1);
    return acc;
}

}  // namespace brovisionml::tiling
