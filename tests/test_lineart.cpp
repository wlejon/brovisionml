// Lineart LineartDetector: real-weights END-TO-END parity test.
//
// Weights-gated (skips cleanly when weights/lineart/model.safetensors or the
// golden_lineart_*.bin dumps are absent). For each golden, reads the stored
// input bytes, runs detect() at native resolution (detect_resolution=0) with
// invert=false so the detector returns the RAW generator output, and compares
// it against the golden `line` record (model(image)[0][0], the [0,1] line map).
// A CUDA device, when present, is exercised too.
//
// The generator is a moderately deep conv/conv-transpose stack with InstanceNorm
// (realized as group_norm, num_groups==channels) and a final sigmoid. The golden
// dims are divisible by 4 so the two downsamples + two upsamples round-trip
// (model output dims == input dims) and the line map needs no resize-back. Parity
// is tight: a large diff points at a real bug (reflect-pad convention, the
// transposed-conv weight layout, instance-norm groups, or block order) rather
// than accumulation.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/lineart.h"

#include "brotensor/runtime.h"

#include "test_device.h"

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

using brovisionml::lineart::LineMap;
using brovisionml::lineart::LineartConfig;
using brovisionml::lineart::LineartDetector;

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

// BVMLLIN1 golden reader.
struct Golden {
    int W = 0, H = 0;
    int OW = 0, OH = 0;
    std::vector<uint8_t> input;   // H*W*3 HWC RGB
    std::vector<float> line;       // OH*OW, [0,1]
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 8) != "BVMLLIN1") return false;
    int version = 0;
    if (!read_one(f, version)) return false;
    if (!read_one(f, g.W) || !read_one(f, g.H)) return false;

    const std::size_t in_n = static_cast<std::size_t>(g.H) * g.W * 3;
    if (!read_vec(f, g.input, in_n)) return false;

    if (!read_one(f, g.OW) || !read_one(f, g.OH)) return false;
    const std::size_t line_n = static_cast<std::size_t>(g.OH) * g.OW;
    if (!read_vec(f, g.line, line_n)) return false;
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
    std::printf("  case %s  (in %dx%d  out %dx%d)\n",
                path.c_str(), g.W, g.H, g.OW, g.OH);
    // Goldens use /4 dims so the generator round-trips to the input size.
    CHECK(g.OW == g.W && g.OH == g.H);

    LineartConfig cfg;
    cfg.detect_resolution = 0;   // native
    cfg.invert = false;          // compare the raw generator output
    LineartDetector det(cfg);
    det.load(dir);

    const std::size_t n = static_cast<std::size_t>(g.H) * g.W;
    LineMap cpu = det.detect(g.input.data(), g.W, g.H, 3);
    CHECK(cpu.width == g.W && cpu.height == g.H);
    CHECK(cpu.line.size() == n);
    if (cpu.line.size() != n) { ++g_failures; return; }

    double mx = 0.0, mn = 0.0, fr = 0.0;
    diff_stats(cpu.line.data(), g.line.data(), n, mx, mn, fr);
    std::printf("    CPU vs golden: max-abs=%.3e  mean-abs=%.3e  frac>1e-2=%.4f%%\n",
                mx, mn, fr * 100.0);
    CHECK(mn < 5e-4);        // whole-map agreement
    CHECK(fr < 0.01);        // <1% of pixels diverge past 1e-2
    CHECK(mx < 0.10);        // gross-breakage tripwire

    // On-device: FP32-on-GPU should track both the golden and the CPU result.
    brotensor::init();
    const brotensor::Device gpu_dev = brovisionml_test::preferred_gpu();
    if (gpu_dev != brotensor::Device::CPU) {
        const char* dev = brovisionml_test::device_name(gpu_dev);
        det.to(gpu_dev);
        CHECK(det.device() == gpu_dev);
        LineMap gpu = det.detect(g.input.data(), g.W, g.H, 3);
        CHECK(gpu.line.size() == n);
        if (gpu.line.size() == n) {
            double mxg = 0.0, mng = 0.0, frg = 0.0, mxc = 0.0, mnc = 0.0, frc = 0.0;
            diff_stats(gpu.line.data(), g.line.data(), n, mxg, mng, frg);
            diff_stats(gpu.line.data(), cpu.line.data(), n, mxc, mnc, frc);
            std::printf("    %s vs golden: mean-abs=%.3e  vs CPU: max-abs=%.3e\n",
                        dev, mng, mxc);
            CHECK(mng < 5e-4);      // GPU tracks the golden like the CPU path
            CHECK(mxc < 1e-2);      // GPU and CPU agree (FP32, same math)
        }
    } else {
        std::printf("    (no GPU device available — on-device check skipped)\n");
    }
}

}  // namespace

int main() {
    std::printf("test_lineart:\n");

    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
    const std::string dir = base + "/lineart";
    const std::string ckpt = dir + "/model.safetensors";
    const std::string sq = dir + "/golden_lineart_sq512.bin";
    const std::string rect = dir + "/golden_lineart_rect512x384.bin";

    if (!file_exists(ckpt) || !file_exists(sq) || !file_exists(rect)) {
        std::printf("  no checkpoint/goldens under '%s' — skipping "
                    "(weights-gated).\n", dir.c_str());
        return 0;
    }

    try {
        run_case(dir, sq);     // square
        run_case(dir, rect);   // non-square (both /4)
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  error: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) {
        std::printf("  OK  lineart parity checks passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
