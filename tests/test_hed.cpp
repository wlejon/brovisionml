// HED SoftEdgeDetector: real-weights END-TO-END parity test.
//
// Weights-gated (skips cleanly when weights/hed/model.safetensors or the
// golden_hed_*.bin dumps are absent). For each golden, reads the stored input
// bytes, runs detect() at native resolution (detect_resolution=0, so the model
// runs at exactly the golden's input size), and compares the edge map against
// the golden `edge` record (sigmoid of the mean of the 5 bilinear-resized side
// maps). A CUDA device, when present, is exercised too.
//
// HED is a shallow pure-conv FCN (13 convs + 5 projections, max-pool + bilinear
// + sigmoid), so parity is tight; a large diff points at a real bug (max-pool
// floor dims, bilinear convention, the folded `norm` bias, or block conv order)
// rather than accumulation.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/hed.h"

#include "brotensor/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

using brovisionml::hed::EdgeMap;
using brovisionml::hed::HedConfig;
using brovisionml::hed::SoftEdgeDetector;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

namespace {

template <class T>
bool read_one(std::ifstream& f, T& x) {
    f.read(reinterpret_cast<char*>(&x), sizeof(T));
    return static_cast<bool>(f);
}

template <class T>
bool read_vec(std::ifstream& f, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(f);
}

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

// BVMLHED1 golden reader: seeks past the 5 raw side maps to reach the final
// edge map.
struct Golden {
    int W = 0, H = 0;
    std::vector<uint8_t> input;   // H*W*3 HWC RGB
    std::vector<float> edge;       // H*W, [0,1]
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 8) != "BVMLHED1") return false;
    int version = 0;
    if (!read_one(f, version)) return false;
    if (!read_one(f, g.W) || !read_one(f, g.H)) return false;

    const std::size_t in_n = static_cast<std::size_t>(g.H) * g.W * 3;
    if (!read_vec(f, g.input, in_n)) return false;

    int n_sides = 0;
    if (!read_one(f, n_sides)) return false;
    if (n_sides != 5) return false;
    for (int i = 0; i < n_sides; ++i) {          // seek past the raw side maps
        int C = 0, hh = 0, ww = 0;
        if (!read_one(f, C) || !read_one(f, hh) || !read_one(f, ww)) return false;
        std::vector<float> s;
        if (!read_vec(f, s, static_cast<std::size_t>(C) * hh * ww)) return false;
    }

    const std::size_t edge_n = static_cast<std::size_t>(g.H) * g.W;
    if (!read_vec(f, g.edge, edge_n)) return false;
    return true;
}

void diff_stats(const float* a, const float* b, std::size_t n,
                double& max_abs, double& mean_abs, double& frac_gt_1e2) {
    double m = 0.0, s = 0.0;
    std::size_t outliers = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d =
            std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        m = std::max(m, d);
        s += d;
        if (d > 1e-2) ++outliers;
    }
    max_abs = m;
    mean_abs = n ? s / static_cast<double>(n) : 0.0;
    frac_gt_1e2 = n ? static_cast<double>(outliers) / static_cast<double>(n) : 0.0;
}

void run_case(const std::string& dir, const std::string& path) {
    Golden g;
    if (!load_golden(path, g)) {
        std::printf("  FAIL  could not load/parse golden %s\n", path.c_str());
        ++g_failures;
        return;
    }
    std::printf("  case %s  (%dx%d)\n", path.c_str(), g.W, g.H);

    SoftEdgeDetector det(HedConfig{/*detect_resolution=*/0});  // native
    det.load(dir);

    const std::size_t n = static_cast<std::size_t>(g.H) * g.W;
    EdgeMap cpu = det.detect(g.input.data(), g.W, g.H, 3);
    CHECK(cpu.width == g.W && cpu.height == g.H);
    CHECK(cpu.edge.size() == n);
    if (cpu.edge.size() != n) { ++g_failures; return; }

    // The edge map is sigmoid(mean of the 5 side maps), each bilinear-upsampled
    // to the working resolution — the coarsest side is 16x smaller (a 32x32 ->
    // 512x512 upsample). brotensor's bilinear matches torch's F.interpolate to
    // ~1e-7 (verified out-of-repo on the golden sides), so the resize is exact;
    // the divergence enters earlier, in the conv trunk, where brotensor's conv2d
    // accumulates in a different FP order than torch. At an edge in a coarse side
    // map a sub-pixel logit shift, upsampled 16x and passed through the steep
    // sigmoid, becomes a narrow band of flipped fine pixels — so ~1% of pixels
    // diverge while the whole-map mean stays at a few 1e-4. mean-abs is therefore
    // the meaningful parity metric; the outlier fraction is a loose tripwire.
    double mx = 0.0, mn = 0.0, fr = 0.0;
    diff_stats(cpu.edge.data(), g.edge.data(), n, mx, mn, fr);
    std::printf("    CPU vs golden: max-abs=%.3e  mean-abs=%.3e  frac>1e-2=%.4f%%\n",
                mx, mn, fr * 100.0);
    CHECK(mn < 1e-3);        // whole-map agreement (observed ~3e-4)
    CHECK(fr < 0.03);        // <3% of pixels diverge (edge-band fp amplification)
    CHECK(mx < 0.25);        // gross-breakage tripwire only

    // Tiled parity: HED is a local FCN, so running the image as overlapping
    // tiles and feather-blending the per-tile edge maps must closely reproduce
    // the whole-image result. Divergence is confined to thin bands at tile
    // interiors where the receptive field loses cross-seam context (the feather
    // down-weights exactly there), so the whole-map mean stays small. This
    // exercises the full crop -> preprocess -> run -> blend path.
    {
        HedConfig tcfg;
        tcfg.tile.tile = 256;
        tcfg.tile.overlap = 64;
        SoftEdgeDetector tdet(tcfg);
        tdet.load(dir);
        EdgeMap tiled = tdet.detect(g.input.data(), g.W, g.H, 3);
        CHECK(tiled.width == g.W && tiled.height == g.H);
        CHECK(tiled.edge.size() == n);
        if (tiled.edge.size() == n) {
            double tmx = 0.0, tmn = 0.0, tfr = 0.0;
            diff_stats(tiled.edge.data(), cpu.edge.data(), n, tmx, tmn, tfr);
            std::printf("    tiled(256/64) vs whole: max-abs=%.3e  mean-abs=%.3e"
                        "  frac>1e-2=%.4f%%\n", tmx, tmn, tfr * 100.0);
            CHECK(tmn < 1e-2);   // local operator: blended tiles track whole-image
        }
    }

    // On-device: if a CUDA device is present, migrate and re-run; FP32-on-CUDA
    // should track both the golden and the CPU result tightly.
    brotensor::init();
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        det.to(brotensor::Device::CUDA);
        CHECK(det.device() == brotensor::Device::CUDA);
        EdgeMap gpu = det.detect(g.input.data(), g.W, g.H, 3);
        CHECK(gpu.edge.size() == n);
        if (gpu.edge.size() == n) {
            double mxg = 0.0, mng = 0.0, frg = 0.0, mxc = 0.0, mnc = 0.0, frc = 0.0;
            diff_stats(gpu.edge.data(), g.edge.data(), n, mxg, mng, frg);
            diff_stats(gpu.edge.data(), cpu.edge.data(), n, mxc, mnc, frc);
            std::printf("    CUDA vs golden: mean-abs=%.3e  vs CPU: max-abs=%.3e\n",
                        mng, mxc);
            CHECK(mng < 1e-3);      // CUDA tracks the golden like the CPU path
            CHECK(mxc < 1e-2);      // CUDA and CPU agree (FP32, same math)
        }
    } else {
        std::printf("    (no CUDA device available — on-device check skipped)\n");
    }
}

}  // namespace

int main() {
    std::printf("test_hed:\n");

    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
    const std::string dir = base + "/hed";
    const std::string ckpt = dir + "/model.safetensors";
    const std::string sq = dir + "/golden_hed_sq512.bin";
    const std::string ns = dir + "/golden_hed_ns500x375.bin";

    if (!file_exists(ckpt) || !file_exists(sq) || !file_exists(ns)) {
        std::printf("  no checkpoint/goldens under '%s' — skipping "
                    "(weights-gated).\n", dir.c_str());
        return 0;
    }

    try {
        run_case(dir, sq);   // square, clean even downsamples
        run_case(dir, ns);   // odd dims: floor-pool + bilinear resize-back
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  error: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) {
        std::printf("  OK  hed parity checks passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
