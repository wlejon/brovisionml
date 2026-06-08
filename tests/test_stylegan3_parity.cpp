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

#include "test_device.h"

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

// Verify a generator's synthesis channel schedule (the tapering channel counts
// are encoded in the layer names) against the reference for that config.
static void check_schedule(const char* tag, const Config& cfg,
                           const std::vector<std::string>& expect) {
    Generator g(cfg);
    check(g.num_ws() == 16, "num_ws == 16");
    const auto& names = g.synthesis().layer_names();
    check(names.size() == expect.size(), "synthesis layer count");
    if (names.size() == expect.size())
        for (std::size_t i = 0; i < expect.size(); ++i)
            check(names[i] == expect[i], expect[i].c_str());
    (void)tag;
}

int main() {
    // ── Always-on structural check: the config-R and config-T channel schedules.
    // config-R caps channels at 1024, config-T at 512 (16384/512 base/max), and
    // config-T's 3x3 conv leaves the schedule otherwise identically shaped. ──
    check_schedule("r256", Config::r256(), {
        "L0_36_1024", "L1_36_1024", "L2_36_1024", "L3_52_1024", "L4_52_1024",
        "L5_84_1024", "L6_84_1024", "L7_148_724", "L8_148_512", "L9_148_362",
        "L10_276_256", "L11_276_181", "L12_276_128", "L13_256_128", "L14_256_3",
    });
    check_schedule("t256", Config::t256(), {
        "L0_36_512", "L1_36_512", "L2_36_512", "L3_52_512", "L4_52_512",
        "L5_84_512", "L6_84_512", "L7_148_362", "L8_148_256", "L9_148_181",
        "L10_276_128", "L11_276_91", "L12_276_64", "L13_256_64", "L14_256_3",
    });
    // 512² uses the larger channel_base (65536 for config-R): more channels stay
    // clamped at channel_max before the taper, and the size schedule ramps faster.
    check_schedule("r512", Config::r512(), {
        "L0_36_1024", "L1_36_1024", "L2_52_1024", "L3_52_1024", "L4_84_1024",
        "L5_84_1024", "L6_148_1024", "L7_148_967", "L8_276_645", "L9_276_431",
        "L10_532_287", "L11_532_192", "L12_532_128", "L13_512_128", "L14_512_3",
    });

    std::string base = BROVISIONML_WEIGHTS_DIR;
    if (const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR")) base = env;

    brotensor::init();
    const brotensor::Device gpu = brovisionml_test::preferred_gpu();
    int ran = 0;

    // Run both gates for one (subdir, cfg) checkpoint on CPU + GPU; clean-skip if
    // the converted checkpoint / golden aren't present. FP32 on both backends:
    // mapping is a couple of matmuls + a lerp (tight); synthesis threads many
    // upfirdn/filtered-lrelu stages (accumulates more) — but still far under the
    // image's ~[-1,1] range. Tolerances keep >100x headroom over observed FP32
    // error (cross-GPU variance + the FP16 fast path) yet fail hard on a
    // schedule/weight regression, which lands errors of order 0.1-1.0.
    auto run_parity = [&](const char* subdir, const Config& cfg) {
        const std::string dir   = base + "/" + subdir;
        const std::string ckpt  = dir + "/model.safetensors";
        const std::string gpath = dir + "/golden/golden_stylegan3.bin";
        Golden g;
        if (!file_exists(ckpt) || !read_golden(gpath, g)) {
            std::printf("test_stylegan3_parity: %s parity SKIPPED (no ckpt/golden)\n",
                        subdir);
            return;
        }
        check(g.z_dim == 512 && g.num_ws == 16 && g.w_dim == 512, "golden dims (z/ws)");
        check(g.channels == 3, "golden dims (img channels)");
        std::printf("test_stylegan3_parity: %s (seed-replayed golden)\n", subdir);
        // The CPU synthesis forward is a single-threaded FP32 full-generator pass;
        // at 256² that's seconds, but at 512²+ (with the larger channel_base) it's
        // many minutes. Run the CPU gate only at <=256; at higher resolutions the
        // GPU gate vs. the same NVlabs CPU golden is the numeric check.
        if (g.height <= 256)
            run_gates(dir, cfg, g, brotensor::Device::CPU, "cpu",
                      /*ws_tol_mean=*/5e-4, /*img_tol_mean=*/5e-3, /*img_tol_max=*/5e-2);
        else
            std::printf("  (%d² — CPU gate skipped (too slow); GPU only)\n", g.height);
        if (gpu != brotensor::Device::CPU)
            run_gates(dir, cfg, g, gpu, brovisionml_test::device_name(gpu),
                      /*ws_tol_mean=*/5e-4, /*img_tol_mean=*/5e-3, /*img_tol_max=*/5e-2);
        else if (g.height > 256)
            std::printf("  (no GPU — %d² parity not verified this run)\n", g.height);
        else
            std::printf("  (no GPU available — GPU parity skipped)\n");
        ++ran;
    };

    try {
        run_parity("stylegan3-r-ffhqu-256", Config::r256());
        run_parity("stylegan3-t-ffhqu-256", Config::t256());
        run_parity("stylegan3-r-afhqv2-512", Config::r512());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (ran == 0)
        std::printf("test_stylegan3_parity: structural OK; parity SKIPPED "
                    "(no checkpoints under %s)\n", base.c_str());

    if (failures == 0) { std::printf("test_stylegan3_parity: OK\n"); return 0; }
    std::printf("test_stylegan3_parity: %d failure(s)\n", failures);
    return 1;
}
