// bench — forward-pass benchmark driver for every brovisionml model family.
//
// Loads a checkpoint, runs N warmup + M timed repetitions of the model's
// public entry point on one image, and reports per-rep / best / mean wall
// times. The input is either a real image (--image) or a deterministic
// synthetic one (--size), so runs are reproducible without assets.
//
//   bench <family> <checkpoint-dir-or-file> [options]
//     family        sam | amg | depth | dsine | hed | lineart | mlsd |
//                   openpose | segformer
//     --image PATH  bench on a real image instead of the synthetic one
//     --size WxH    synthetic image size (default 1024x768)
//     --variant V   sam: b (default) | l | h;  depth: small (default) | base | large
//     --warmup N    untimed warmup reps (default 2)
//     --reps N      timed reps (default 5)
//     --cpu         force the CPU backend (default: CUDA when available)
//
// SAM reports encode (set_image) and decode (segment, one center click) as
// separate stages; every other family is one end-to-end call.
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite.

#define _CRT_SECURE_NO_WARNINGS  // std::sscanf for the tiny WxH arg parsing

#include "brovisionml/depth_anything.h"
#include "brovisionml/dsine.h"
#include "brovisionml/hed.h"
#include "brovisionml/lineart.h"
#include "brovisionml/mlsd.h"
#include "brovisionml/openpose.h"
#include "brovisionml/sam.h"
#include "brovisionml/sam_amg.h"
#include "brovisionml/segformer.h"

#include "brotensor/runtime.h"
#include "broimage/decode.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace {

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <family> <checkpoint-dir-or-file> [options]\n"
        "  family        sam | amg | depth | dsine | hed | lineart | mlsd |\n"
        "                openpose | segformer\n"
        "  --image PATH  bench on a real image (default: synthetic)\n"
        "  --size WxH    synthetic image size (default 1024x768)\n"
        "  --variant V   sam: b|l|h; depth: small|base|large\n"
        "  --warmup N    untimed warmup reps (default 2)\n"
        "  --reps N      timed reps (default 5)\n"
        "  --cpu         force the CPU backend\n", prog);
    std::exit(2);
}

// Deterministic synthetic RGB: smooth low-frequency color field with a few
// hard edges so edge/line detectors have real structure to chew on.
std::vector<uint8_t> synthetic_rgb(int w, int h) {
    std::vector<uint8_t> px(static_cast<std::size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float fx = static_cast<float>(x) / w;
            const float fy = static_cast<float>(y) / h;
            float r = 0.5f + 0.5f * std::sin(6.28318f * (2.0f * fx + fy));
            float g = 0.5f + 0.5f * std::sin(6.28318f * (fx + 3.0f * fy));
            float b = fx * fy;
            // A grid of hard rectangles for edge/line structure.
            if (((x / 96) + (y / 96)) % 2 == 0) { r = 1.0f - r; b = 1.0f - b; }
            uint8_t* p = px.data() + (static_cast<std::size_t>(y) * w + x) * 3;
            p[0] = static_cast<uint8_t>(r * 255.0f + 0.5f);
            p[1] = static_cast<uint8_t>(g * 255.0f + 0.5f);
            p[2] = static_cast<uint8_t>(b * 255.0f + 0.5f);
        }
    }
    return px;
}

struct Stage {
    const char* name;
    std::function<void()> run;
};

void bench_stages(brotensor::Device dev, int warmup, int reps,
                  const std::vector<Stage>& stages) {
    using clock = std::chrono::steady_clock;
    for (int i = 0; i < warmup; ++i)
        for (const Stage& s : stages) s.run();
    brotensor::sync(dev);

    std::vector<std::vector<double>> ms(stages.size());
    for (int r = 0; r < reps; ++r) {
        for (std::size_t si = 0; si < stages.size(); ++si) {
            brotensor::sync(dev);
            const auto t0 = clock::now();
            stages[si].run();
            brotensor::sync(dev);
            const auto t1 = clock::now();
            ms[si].push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }
    for (std::size_t si = 0; si < stages.size(); ++si) {
        double best = ms[si][0], sum = 0.0;
        std::printf("%-12s", stages[si].name);
        for (double v : ms[si]) {
            std::printf(" %8.2f", v);
            best = std::min(best, v);
            sum += v;
        }
        std::printf("   best %8.2f ms   mean %8.2f ms\n",
                    best, sum / static_cast<double>(ms[si].size()));
    }
}

bool ends_with_safetensors(const std::string& s) {
    return s.size() >= 12 && s.compare(s.size() - 12, 12, ".safetensors") == 0;
}

template <typename Model>
void load_any(Model& m, const std::string& ckpt) {
    if (ends_with_safetensors(ckpt)) m.load_file(ckpt); else m.load(ckpt);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);

    const std::string family = argv[1];
    const std::string ckpt   = argv[2];
    std::string image_path, variant;
    int w = 1024, h = 768, warmup = 2, reps = 5;
    bool force_cpu = false;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--image")        image_path = next();
        else if (a == "--size")    { if (std::sscanf(next(), "%dx%d", &w, &h) != 2) usage(argv[0]); }
        else if (a == "--variant") variant = next();
        else if (a == "--warmup")  warmup = std::atoi(next());
        else if (a == "--reps")    reps = std::atoi(next());
        else if (a == "--cpu")     force_cpu = true;
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); }
    }
    if (reps < 1) usage(argv[0]);

    std::vector<uint8_t> px;
    int channels = 3;
    if (!image_path.empty()) {
        broimage::Image im;
        std::string err;
        if (!broimage::decode_file(image_path, im, &err)) {
            std::fprintf(stderr, "failed to decode '%s': %s\n",
                         image_path.c_str(), err.c_str());
            return 1;
        }
        px = std::move(im.pixels);
        w = im.width; h = im.height; channels = im.channels;
    } else {
        px = synthetic_rgb(w, h);
    }

    brotensor::init();
    brotensor::Device dev = brotensor::Device::CPU;
    if (!force_cpu && brotensor::is_available(brotensor::Device::CUDA))
        dev = brotensor::Device::CUDA;
    std::printf("%s | %dx%d | %s | warmup %d, reps %d\n", family.c_str(), w, h,
                dev == brotensor::Device::CPU ? "CPU" : "CUDA", warmup, reps);

    try {
        if (family == "sam" || family == "amg") {
            using brovisionml::sam::SamConfig;
            SamConfig cfg = variant == "h" ? SamConfig::vit_h()
                          : variant == "l" ? SamConfig::vit_l()
                                           : SamConfig::vit_b();
            brovisionml::sam::Sam model(cfg);
            load_any(model, ckpt);
            model.to(dev);
            if (family == "sam") {
                const std::vector<std::array<float, 2>> pt{{w * 0.5f, h * 0.5f}};
                const std::vector<int> lbl{1};
                bench_stages(dev, warmup, reps, {
                    {"encode", [&] { model.set_image(px.data(), w, h, channels); }},
                    {"decode", [&] { (void)model.segment(pt, lbl, {}); }},
                });
            } else {
                brovisionml::sam::AutomaticMaskGenerator amg(model);
                bench_stages(dev, warmup, reps, {
                    {"generate", [&] { (void)amg.generate(px.data(), w, h, channels); }},
                });
            }
        } else if (family == "depth") {
            using brovisionml::depth::DepthAnythingConfig;
            DepthAnythingConfig cfg =
                  variant == "base"  ? DepthAnythingConfig::v2_base()
                : variant == "large" ? DepthAnythingConfig::v2_large()
                                     : DepthAnythingConfig::v2_small();
            brovisionml::depth::DepthEstimator est(cfg);
            load_any(est, ckpt);
            est.to(dev);
            bench_stages(dev, warmup, reps, {
                {"estimate", [&] { (void)est.estimate(px.data(), w, h, channels); }},
            });
        } else if (family == "dsine") {
            brovisionml::dsine::NormalEstimator est;
            load_any(est, ckpt);
            est.to(dev);
            bench_stages(dev, warmup, reps, {
                {"estimate", [&] { (void)est.estimate(px.data(), w, h, channels); }},
            });
        } else if (family == "hed") {
            brovisionml::hed::SoftEdgeDetector det;
            load_any(det, ckpt);
            det.to(dev);
            bench_stages(dev, warmup, reps, {
                {"detect", [&] { (void)det.detect(px.data(), w, h, channels); }},
            });
        } else if (family == "lineart") {
            brovisionml::lineart::LineartDetector det;
            load_any(det, ckpt);
            det.to(dev);
            bench_stages(dev, warmup, reps, {
                {"detect", [&] { (void)det.detect(px.data(), w, h, channels); }},
            });
        } else if (family == "mlsd") {
            brovisionml::mlsd::MLSDdetector det;
            load_any(det, ckpt);
            det.to(dev);
            bench_stages(dev, warmup, reps, {
                {"detect", [&] { (void)det.detect(px.data(), w, h, channels); }},
            });
        } else if (family == "openpose") {
            brovisionml::openpose::OpenposeDetector det;
            load_any(det, ckpt);
            det.to(dev);
            bench_stages(dev, warmup, reps, {
                {"infer",  [&] { (void)det.infer_maps(px.data(), w, h, channels); }},
                {"detect", [&] { (void)det.detect(px.data(), w, h, channels); }},
            });
        } else if (family == "segformer") {
            brovisionml::segformer::SegformerDetector det;
            load_any(det, ckpt);
            det.to(dev);
            bench_stages(dev, warmup, reps, {
                {"detect", [&] { (void)det.detect(px.data(), w, h, channels); }},
            });
        } else {
            std::fprintf(stderr, "unknown family: %s\n", family.c_str());
            usage(argv[0]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
