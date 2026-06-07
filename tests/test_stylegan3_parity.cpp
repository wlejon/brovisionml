// StyleGAN3-R numeric-parity test against the NVlabs reference.
//
// Structural part: always runs — the r256 preset reproduces the released
// config-R schedule (num_ws, and the tapering per-layer channel counts the
// reference assigns: 1024×7, 724, 512, 362, 256, 181, 128, 128, 3).
//
// Golden parity part: gated on the converted checkpoint
// (weights/stylegan3-r-ffhqu-256/model.safetensors, via
// scripts/convert-stylegan3.py) and a golden dump
// (weights/stylegan3-r-ffhqu-256/golden/golden_stylegan3.bin, produced
// out-of-repo by scripts/dump-stylegan3-golden.py — never committed). Skips
// cleanly when either is absent. Replays the golden's exact latent z and mapped
// w+, and compares:
//   Gate 1 (mapping):   g.map(z, psi) vs the golden w+.
//   Gate 2 (synthesis): g.synthesize(golden_w+) vs the golden raw FP32 image.
// Each gate runs on CPU, and on CUDA when available.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/stylegan3.h"

#include "brotensor/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}
bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

using brovisionml::stylegan3::Config;
using brovisionml::stylegan3::Generator;
using brotensor::Tensor;

// Golden format BVMLSG31 — see scripts/dump-stylegan3-golden.py.
struct Golden {
    int   z_dim = 0, num_ws = 0, w_dim = 0, channels = 0, height = 0, width = 0;
    float trunc = 1.0f;
    int   cutoff = -1;
    std::vector<float> z;     // (z_dim,)
    std::vector<float> ws;    // (num_ws*w_dim,)
    std::vector<float> img;   // (channels*H*W,) NCHW, raw ~[-1,1]
};

bool read_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "BVMLSG31", 8) != 0) return false;
    int version = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&g.z_dim), 4);
    f.read(reinterpret_cast<char*>(&g.num_ws), 4);
    f.read(reinterpret_cast<char*>(&g.w_dim), 4);
    f.read(reinterpret_cast<char*>(&g.channels), 4);
    f.read(reinterpret_cast<char*>(&g.height), 4);
    f.read(reinterpret_cast<char*>(&g.width), 4);
    f.read(reinterpret_cast<char*>(&g.trunc), 4);
    f.read(reinterpret_cast<char*>(&g.cutoff), 4);
    g.z.resize(static_cast<std::size_t>(g.z_dim));
    g.ws.resize(static_cast<std::size_t>(g.num_ws) * g.w_dim);
    g.img.resize(static_cast<std::size_t>(g.channels) * g.height * g.width);
    f.read(reinterpret_cast<char*>(g.z.data()),
           static_cast<std::streamsize>(g.z.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(g.ws.data()),
           static_cast<std::streamsize>(g.ws.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(g.img.data()),
           static_cast<std::streamsize>(g.img.size() * sizeof(float)));
    return static_cast<bool>(f);
}

// mean / max abs difference between two equal-length spans.
struct Diff { double mean = 0.0, max = 0.0; };
Diff diff(const float* a, const float* b, std::size_t n) {
    Diff d;
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        d.mean += e;
        d.max = std::max(d.max, e);
    }
    if (n) d.mean /= static_cast<double>(n);
    return d;
}

// Run both gates on `dev`; `tag` labels the backend in the printout.
void run_gates(const std::string& dir, const Config& cfg, const Golden& g,
               brotensor::Device dev, const char* tag,
               double ws_tol_mean, double img_tol_mean, double img_tol_max) {
    Generator gen(cfg);
    gen.load(dir);
    gen.to(dev);

    // ── Gate 1: mapping (z -> w+) ──
    Tensor z = Tensor::mat(1, g.z_dim);
    for (int i = 0; i < g.z_dim; ++i) z[i] = g.z[i];
    if (dev != brotensor::Device::CPU) z = z.to(dev);

    Tensor ws_got = gen.map(z, g.trunc, g.cutoff).to(brotensor::Device::CPU);
    Diff dws = diff(ws_got.host_f32(), g.ws.data(), g.ws.size());
    std::printf("  [%s] mapping  w+: mean=%.3e max=%.3e (tol mean<%.1e)\n",
                tag, dws.mean, dws.max, ws_tol_mean);
    check(dws.mean < ws_tol_mean, "mapping w+ parity");

    // ── Gate 2: synthesis (golden w+ -> image) ──
    Tensor ws = Tensor::mat(g.num_ws, g.w_dim);
    for (std::size_t i = 0; i < g.ws.size(); ++i) ws[static_cast<int>(i)] = g.ws[i];
    if (dev != brotensor::Device::CPU) ws = ws.to(dev);

    Tensor img = gen.synthesize(ws).to(brotensor::Device::CPU);
    Diff dimg = diff(img.host_f32(), g.img.data(), g.img.size());
    std::printf("  [%s] synth  img: mean=%.3e max=%.3e (tol mean<%.1e max<%.1e)\n",
                tag, dimg.mean, dimg.max, img_tol_mean, img_tol_max);
    check(dimg.mean < img_tol_mean, "synthesis image mean parity");
    check(dimg.max < img_tol_max, "synthesis image max parity");
}

}  // namespace

int main() {
    // ── Always-on structural check: the corrected config-R channel schedule ──
    {
        Generator g(Config::r256());
        check(g.num_ws() == 16, "r256 num_ws == 16");
        // The tapering channel counts are encoded in the synthesis layer names.
        const auto& names = g.synthesis().layer_names();
        const char* expect[] = {
            "L0_36_1024", "L1_36_1024", "L2_36_1024", "L3_52_1024", "L4_52_1024",
            "L5_84_1024", "L6_84_1024", "L7_148_724", "L8_148_512", "L9_148_362",
            "L10_276_256", "L11_276_181", "L12_276_128", "L13_256_128", "L14_256_3",
        };
        check(names.size() == 15, "r256 has 15 synthesis layers");
        if (names.size() == 15) {
            for (int i = 0; i < 15; ++i)
                check(names[i] == expect[i], expect[i]);
        }
    }

    // ── Locate the converted checkpoint + golden (clean skip if absent) ──
    std::string base = BROVISIONML_WEIGHTS_DIR;
    if (const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR")) base = env;
    const std::string dir   = base + "/stylegan3-r-ffhqu-256";
    const std::string ckpt  = dir + "/model.safetensors";
    const std::string gpath = dir + "/golden/golden_stylegan3.bin";

    Golden g;
    const bool have_ckpt = file_exists(ckpt);
    const bool have_gold = read_golden(gpath, g);
    if (!have_ckpt || !have_gold) {
        std::printf("test_stylegan3_parity: structural OK; parity SKIPPED "
                    "(ckpt=%d golden=%d under %s)\n",
                    have_ckpt ? 1 : 0, have_gold ? 1 : 0, base.c_str());
        return failures == 0 ? 0 : 1;
    }

    check(g.z_dim == 512 && g.num_ws == 16 && g.w_dim == 512, "golden dims (z/ws)");
    check(g.channels == 3 && g.height == 256 && g.width == 256, "golden dims (img)");

    try {
        const Config cfg = Config::r256();
        // FP32 on both backends. Mapping is a couple of matmuls + a lerp, so it's
        // tight; synthesis threads many upfirdn/filtered-lrelu stages, so it
        // accumulates more — but still far under the image's ~[-1,1] range.
        // Observed FP32 errors are mapping ~2e-7 / synthesis ~2e-4 max on both
        // backends; these bounds keep >100x headroom (cross-GPU FP32 variance,
        // and the future FP16 synthesis fast path) while still failing hard on a
        // schedule/weight regression, which lands errors of order 0.1-1.0.
        std::printf("test_stylegan3_parity: ffhqu-256 (seed-replayed golden)\n");
        run_gates(dir, cfg, g, brotensor::Device::CPU, "cpu",
                  /*ws_tol_mean=*/5e-4, /*img_tol_mean=*/5e-3, /*img_tol_max=*/5e-2);

        brotensor::init();
        if (brotensor::is_available(brotensor::Device::CUDA)) {
            run_gates(dir, cfg, g, brotensor::Device::CUDA, "cuda",
                      /*ws_tol_mean=*/5e-4, /*img_tol_mean=*/5e-3, /*img_tol_max=*/5e-2);
        } else {
            std::printf("  (CUDA not available — GPU parity skipped)\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (failures == 0) { std::printf("test_stylegan3_parity: OK\n"); return 0; }
    std::printf("test_stylegan3_parity: %d failure(s)\n", failures);
    return 1;
}
