// DSINE NormalEstimator orchestrator: real-weights END-TO-END test.
//
// Weights-gated (skips cleanly when weights/dsine/model.safetensors or
// weights/dsine/golden_dsine_sq256.bin is absent). Loads the real DSINE_v02
// checkpoint through the single-class `NormalEstimator`, reads the golden's
// stored input bytes, runs estimate(), and:
//   (a) asserts every output normal is finite and unit-length to ~1e-4, and
//   (b) compares against the golden `final` record (the LAST record in the
//       dump) to confirm the orchestrator reproduces the staged
//       preprocess->encoder->decoder->refiner pipeline (max-abs ~1e-2).
//
// The orchestrator just wires the four stages, so (b) should match the
// staged-pipeline parity that test_dsine_refine already validates. This test
// runs the full EfficientNet-B5 encoder (~50s); it is intentionally end to end.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/dsine.h"

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

using brovisionml::dsine::DsineConfig;
using brovisionml::dsine::NormalEstimator;
using brovisionml::dsine::NormalMap;

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

struct Tap {
    int C = 0, H = 0, W = 0;
    std::vector<float> data;
};

// BVMLDSN1 golden reader (mirrors tests/test_dsine_refine.cpp): seeks past the
// preprocessed pixels, the three encoder taps, and dec_normal to reach the final
// full-res normal map.
struct Golden {
    int W = 0, H = 0, padW = 0, padH = 0, l = 0, r = 0, t = 0, b = 0;
    float intrins4[4] = {0, 0, 0, 0};   // fx, fy, cx, cy (pre-"+0.5")
    std::vector<uint8_t> input;          // H*W*3 HWC RGB
    std::vector<float> final;            // 3*H*W
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 8) != "BVMLDSN1") return false;
    int version = 0;
    if (!read_one(f, version)) return false;
    if (!read_one(f, g.W) || !read_one(f, g.H)) return false;
    if (!read_one(f, g.padW) || !read_one(f, g.padH)) return false;
    if (!read_one(f, g.l) || !read_one(f, g.r) ||
        !read_one(f, g.t) || !read_one(f, g.b)) return false;
    f.read(reinterpret_cast<char*>(g.intrins4), 4 * sizeof(float));
    if (!f) return false;

    const std::size_t in_n = static_cast<std::size_t>(g.H) * g.W * 3;
    const std::size_t px_n = static_cast<std::size_t>(g.padH) * g.padW * 3;
    if (!read_vec(f, g.input, in_n)) return false;
    std::vector<float> pixels;     // skip the preprocessed pixels
    if (!read_vec(f, pixels, px_n)) return false;

    int n_taps = 0;
    if (!read_one(f, n_taps)) return false;
    if (n_taps != 3) return false;
    for (int i = 0; i < n_taps; ++i) {     // seek past the encoder taps
        Tap tp;
        if (!read_one(f, tp.C) || !read_one(f, tp.H) || !read_one(f, tp.W))
            return false;
        const std::size_t n = static_cast<std::size_t>(tp.C) * tp.H * tp.W;
        if (!read_vec(f, tp.data, n)) return false;
    }

    // seek past dec_normal.
    int dn_C = 0, dn_H = 0, dn_W = 0;
    if (!read_one(f, dn_C) || !read_one(f, dn_H) || !read_one(f, dn_W))
        return false;
    std::vector<float> dec_normal;
    const std::size_t dn_n = static_cast<std::size_t>(dn_C) * dn_H * dn_W;
    if (!read_vec(f, dec_normal, dn_n)) return false;

    // final[3*H*W].
    const std::size_t fin_n = static_cast<std::size_t>(3) * g.H * g.W;
    if (!read_vec(f, g.final, fin_n)) return false;
    return true;
}

void diff_stats(const float* a, const float* b, std::size_t n,
                double& max_abs, double& mean_abs) {
    double m = 0.0, s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d =
            std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        m = std::max(m, d);
        s += d;
    }
    max_abs = m;
    mean_abs = n ? s / static_cast<double>(n) : 0.0;
}

}  // namespace

int main() {
    std::printf("test_dsine:\n");

    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
    const std::string dir = base + "/dsine";
    const std::string ckpt = dir + "/model.safetensors";
    const std::string sq = dir + "/golden_dsine_sq256.bin";

    if (!file_exists(ckpt) || !file_exists(sq)) {
        std::printf("  no checkpoint/golden under '%s' — skipping "
                    "(weights-gated).\n", dir.c_str());
        return 0;
    }

    Golden g;
    if (!load_golden(sq, g)) {
        std::printf("  FAIL  could not load/parse golden %s\n", sq.c_str());
        return 1;
    }
    std::printf("  case %s  (%dx%d -> pad %dx%d)\n",
                sq.c_str(), g.W, g.H, g.padW, g.padH);

    try {
        // The golden's intrinsics were synthesized from FOV-60 on the unpadded
        // dims; the default DsineConfig (fov_deg=60) reproduces them, so the
        // orchestrator's FOV path matches the staged pipeline the golden records.
        NormalEstimator est(DsineConfig{});
        est.load(dir);

        NormalMap nm = est.estimate(g.input.data(), g.W, g.H, 3);
        CHECK(nm.width == g.W && nm.height == g.H);
        const std::size_t n = static_cast<std::size_t>(3) * g.H * g.W;
        CHECK(nm.normals.size() == n);
        if (nm.normals.size() != n) {
            std::printf("  %d failure(s)\n", g_failures);
            return 1;
        }

        // (a) every normal finite and unit-length to ~1e-4.
        const std::size_t plane = static_cast<std::size_t>(g.H) * g.W;
        bool finite = true;
        double worst_unit = 0.0;
        for (std::size_t p = 0; p < plane; ++p) {
            const float nx = nm.normals[0 * plane + p];
            const float ny = nm.normals[1 * plane + p];
            const float nz = nm.normals[2 * plane + p];
            if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz))
                finite = false;
            const double len = std::sqrt(static_cast<double>(nx) * nx +
                                         static_cast<double>(ny) * ny +
                                         static_cast<double>(nz) * nz);
            worst_unit = std::max(worst_unit, std::fabs(len - 1.0));
        }
        std::printf("    unit-norm: worst |len-1| = %.3e\n", worst_unit);
        CHECK(finite);
        CHECK(worst_unit < 1e-4);

        // (b) orchestrator reproduces the staged pipeline (golden `final`).
        double mx = 0.0, mn = 0.0;
        diff_stats(nm.normals.data(), g.final.data(), n, mx, mn);
        std::printf("    vs golden final: max-abs=%.3e  mean-abs=%.3e\n", mx, mn);
        CHECK(mx < 1e-2);
        CHECK(mn < 1e-3);

        // (c) on-device run: if a CUDA device is present, migrate the estimator
        // and re-run end to end on the GPU (encoder/decoder via brotensor ops,
        // the refinement via brovisionml's RayReLU + AngMF-propagate kernels).
        // FP32-on-CUDA, so it should track both the golden and the CPU result to
        // the ballpark bar.
        brotensor::init();
        if (brotensor::is_available(brotensor::Device::CUDA)) {
            est.to(brotensor::Device::CUDA);
            CHECK(est.device() == brotensor::Device::CUDA);
            NormalMap gm = est.estimate(g.input.data(), g.W, g.H, 3);
            CHECK(gm.normals.size() == n);
            if (gm.normals.size() == n) {
                double mxg = 0.0, mng = 0.0, mxc = 0.0, mnc = 0.0;
                diff_stats(gm.normals.data(), g.final.data(), n, mxg, mng);
                diff_stats(gm.normals.data(), nm.normals.data(), n, mxc, mnc);
                std::printf("    CUDA vs golden: max-abs=%.3e  vs CPU: max-abs=%.3e\n",
                            mxg, mxc);
                CHECK(mxg < 1e-2);
                CHECK(mxc < 1e-2);
            }
        } else {
            std::printf("    (no CUDA device available — on-device check skipped)\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  error: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) {
        std::printf("  OK  dsine orchestrator parity checks passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
