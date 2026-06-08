// StyleGAN3-R Generator end-to-end test.
//
//   * Always-on: the Generator constructs from each preset and reports the
//     expected W+ row count.
//   * Weights-gated: when a converted StyleGAN3-R checkpoint is present under
//     weights/, run the full z -> W+ -> RGB pipeline and assert the image shape
//     (res x res x 3), finiteness, spatial variation, and CPU/CUDA parity.
//     Convert a checkpoint with scripts/convert-stylegan3.py; when absent the
//     test prints why and exits 0 (clean skip).
//
// This is the gate that exercises the whole synthesis stack on real weights;
// exact numeric parity against a reference dump can be layered on top via a
// golden file when one is generated out-of-repo.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/stylegan3.h"

#include "brotensor/runtime.h"

#include "test_device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
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
using brovisionml::stylegan3::Image;
using brotensor::Tensor;

Tensor seeded_z(int z_dim, unsigned long long seed) {
    Tensor z = Tensor::mat(1, z_dim);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < z_dim; ++i) z[i] = nd(rng);
    return z;
}

}  // namespace

// End-to-end generate for one checkpoint: CPU render varies + (when a GPU is
// present) CPU/GPU image parity. Both backends are FP32 today.
static void run_checkpoint(const std::string& dir, const Config& cfg, int res,
                           brotensor::Device gpu) {
    // The CPU forward is single-threaded FP32; fine at 256², far too slow at 512²+.
    // At higher resolutions run GPU-only (generate + sanity); CPU/GPU parity is
    // established at 256².
    const bool run_cpu = res <= 256;
    Tensor z = seeded_z(cfg.z_dim, /*seed=*/42);

    Image cpu;
    if (run_cpu) {
        Generator g(cfg);
        g.load(dir);
        cpu = g.generate(z, /*truncation_psi=*/0.7f);
        check(cpu.width == res && cpu.height == res && cpu.channels == 3,
              "image shape res x res x 3");
        int lo = 255, hi = 0;
        for (unsigned char v : cpu.rgb) { lo = std::min(lo, (int)v); hi = std::max(hi, (int)v); }
        check(hi > lo, "image varies (not constant)");
        std::printf("  CPU image range [%d, %d]\n", lo, hi);
    }

    if (gpu != brotensor::Device::CPU) {
        const char* dev = brovisionml_test::device_name(gpu);
        Generator gg(cfg);
        gg.load(dir);
        gg.to(gpu);
        Image gpu_img = gg.generate(z.to(gpu), 0.7f);
        check(gpu_img.width == res && gpu_img.height == res && gpu_img.channels == 3,
              "image shape res x res x 3");
        if (run_cpu) {
            long long worst = 0, nbig = 0;
            for (std::size_t i = 0; i < cpu.rgb.size(); ++i) {
                long long d = std::llabs((long long)cpu.rgb[i] - (long long)gpu_img.rgb[i]);
                worst = std::max(worst, d);
                if (d > 4) ++nbig;
            }
            const double frac = (double)nbig / (double)cpu.rgb.size();
            std::printf("  CPU/%s worst uint8 diff %lld; %.4f%% of pixels differ by >4\n",
                        dev, worst, frac * 100.0);
            check(frac < 0.01, "CPU/GPU image parity (>99% within 4 levels)");
        } else {
            int lo = 255, hi = 0;
            for (unsigned char v : gpu_img.rgb) { lo = std::min(lo, (int)v); hi = std::max(hi, (int)v); }
            check(hi > lo, "image varies (not constant)");
            std::printf("  %s image range [%d, %d] (CPU skipped at %d²)\n", dev, lo, hi, res);
        }
    } else if (!run_cpu) {
        std::printf("  (no GPU and %d² — nothing run)\n", res);
    } else {
        std::printf("  (no GPU available — parity check skipped)\n");
    }
}

int main() {
    // ── Always-on structural checks (config-R and config-T both 16 W+ rows) ──
    check(Generator(Config::r256()).num_ws() == 16, "r256 num_ws");
    check(Generator(Config::r1024()).num_ws() == 16, "r1024 num_ws");
    check(Generator(Config::t256()).num_ws() == 16, "t256 num_ws");

    // ── Run every converted checkpoint present (clean skip if none) ──
    std::string base = BROVISIONML_WEIGHTS_DIR;
    if (const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR")) base = env;
    struct Cand { const char* dir; int res; Config (*cfg)(); };
    const Cand cands[] = {
        {"stylegan3-r-ffhqu-256",   256,  &Config::r256},
        {"stylegan3-t-ffhqu-256",   256,  &Config::t256},
        {"stylegan3-r-afhqv2-512",  512,  &Config::r512},
        {"stylegan3-t-afhqv2-512",  512,  &Config::t512},
        {"stylegan3-r-ffhq-1024",   1024, &Config::r1024},
        {"stylegan3-t-ffhq-1024",   1024, &Config::t1024},
    };

    brotensor::init();
    const brotensor::Device gpu = brovisionml_test::preferred_gpu();

    int ran = 0;
    try {
        for (const Cand& c : cands) {
            std::string d = base + "/" + c.dir;
            if (!file_exists(d + "/model.safetensors")) continue;
            std::printf("test_stylegan3_generate: %s\n", c.dir);
            run_checkpoint(d, c.cfg(), c.res, gpu);
            ++ran;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (ran == 0)
        std::printf("test_stylegan3_generate: no converted checkpoint under %s — "
                    "skipping the weights-gated run (structural checks passed).\n",
                    base.c_str());

    if (failures == 0) { std::printf("test_stylegan3_generate: OK\n"); return 0; }
    std::printf("test_stylegan3_generate: %d failure(s)\n", failures);
    return 1;
}
