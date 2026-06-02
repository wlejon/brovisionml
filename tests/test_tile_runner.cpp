#include "brovisionml/tile_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brovisionml;

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

static bool nearf(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    // ----- plan_tiles: disabled / single-pass cases -------------------------
    {
        tiling::TileConfig off;                       // tile == 0
        CHECK(tiling::plan_tiles(4000, 3000, off).empty());

        tiling::TileConfig cfg; cfg.tile = 512; cfg.overlap = 64;
        // Image fits in one tile -> empty (caller runs whole-image).
        CHECK(tiling::plan_tiles(400, 300, cfg).empty());
        CHECK(tiling::plan_tiles(512, 512, cfg).empty());
    }

    // ----- plan_tiles: coverage + border overlap flags ----------------------
    {
        tiling::TileConfig cfg; cfg.tile = 256; cfg.overlap = 64;
        const int W = 900, H = 500;
        auto tiles = tiling::plan_tiles(W, H, cfg);
        CHECK(!tiles.empty());

        // Every tile is within bounds and at most tile×tile.
        for (const auto& t : tiles) {
            CHECK(t.x >= 0 && t.y >= 0);
            CHECK(t.x + t.w <= W && t.y + t.h <= H);
            CHECK(t.w >= 1 && t.w <= cfg.tile && t.h >= 1 && t.h <= cfg.tile);
            // Border edges carry no feather; interior edges feather by overlap.
            CHECK(t.ov_l == (t.x > 0 ? cfg.overlap : 0));
            CHECK(t.ov_t == (t.y > 0 ? cfg.overlap : 0));
            CHECK(t.ov_r == (t.x + t.w < W ? cfg.overlap : 0));
            CHECK(t.ov_b == (t.y + t.h < H ? cfg.overlap : 0));
        }

        // Union of tiles covers every pixel (no gaps).
        std::vector<uint8_t> covered(static_cast<std::size_t>(W) * H, 0);
        for (const auto& t : tiles)
            for (int y = t.y; y < t.y + t.h; ++y)
                for (int x = t.x; x < t.x + t.w; ++x)
                    covered[static_cast<std::size_t>(y) * W + x] = 1;
        bool all = true;
        for (auto c : covered) if (!c) all = false;
        CHECK(all);
    }

    // ----- one axis below tile size: tile the long axis, single band on short -
    {
        tiling::TileConfig cfg; cfg.tile = 256; cfg.overlap = 64;
        const int W = 900, H = 200;   // H < tile
        auto tiles = tiling::plan_tiles(W, H, cfg);
        CHECK(!tiles.empty());
        std::vector<uint8_t> covered(static_cast<std::size_t>(W) * H, 0);
        for (const auto& t : tiles) {
            CHECK(t.x >= 0 && t.y == 0);        // no negative starts; single row
            CHECK(t.h == H && t.ov_t == 0 && t.ov_b == 0);
            CHECK(t.x + t.w <= W);
            for (int y = 0; y < t.h; ++y)
                for (int x = t.x; x < t.x + t.w; ++x)
                    covered[static_cast<std::size_t>(y) * W + x] = 1;
        }
        bool all = true;
        for (auto c : covered) if (!c) all = false;
        CHECK(all);
    }

    // ----- overlap clamped below tile size ----------------------------------
    {
        tiling::TileConfig cfg; cfg.tile = 100; cfg.overlap = 250;  // absurd overlap
        auto tiles = tiling::plan_tiles(640, 100, cfg);
        // step = max(1, tile - clamp(overlap,0,tile-1)) = max(1, 100-99) = 1; must
        // still terminate and cover.
        CHECK(!tiles.empty());
        CHECK(tiles.back().x + tiles.back().w == 640);
    }

    // ----- blend_1ch: a tiled identity pass reconstructs a known field ------
    {
        tiling::TileConfig cfg; cfg.tile = 64; cfg.overlap = 16;
        const int W = 200, H = 150;
        auto tiles = tiling::plan_tiles(W, H, cfg);
        CHECK(!tiles.empty());

        // Ground-truth full field; each tile returns the matching sub-rect.
        auto field = [W](int x, int y) {
            return std::sin(0.05f * x) * std::cos(0.04f * y) + 0.001f * (x + y);
        };
        auto out = tiling::blend_1ch(W, H, tiles, [&](const tiling::TileRect& t) {
            std::vector<float> m(static_cast<std::size_t>(t.w) * t.h);
            for (int j = 0; j < t.h; ++j)
                for (int i = 0; i < t.w; ++i)
                    m[static_cast<std::size_t>(j) * t.w + i] = field(t.x + i, t.y + j);
            return m;
        });
        CHECK(out.size() == static_cast<std::size_t>(W) * H);
        bool exact = true;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (!nearf(out[static_cast<std::size_t>(y) * W + x], field(x, y)))
                    exact = false;
        CHECK(exact);
    }

    return g_failed == 0 ? 0 : 1;
}
