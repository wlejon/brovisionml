// depth_estimate — minimal Depth-Anything-V2 command-line driver.
//
// Loads an HF Depth-Anything-V2 checkpoint, estimates a relative depth map for
// an image, and writes it as a grayscale PNG (min-max normalized; brighter =
// nearer, matching Depth-Anything's convention).
//
//   depth_estimate <checkpoint-dir-or-file> <image> [options]
//     --variant V   small (default) | base | large
//     --out PATH    output PNG path (default: depth.png)
//     --invert      write darker = nearer instead
//     --cuda        run on the CUDA backend if available
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite.

#include "brovisionml/depth_anything.h"

#include "brotensor/runtime.h"
#include "broimage/decode.h"
#include "broimage/encode.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using brovisionml::depth::DepthAnythingConfig;
using brovisionml::depth::DepthEstimator;
using brovisionml::depth::DepthMap;

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <checkpoint-dir-or-file> <image> [options]\n"
        "  --variant V   small (default) | base | large\n"
        "  --out PATH    output PNG (default: depth.png)\n"
        "  --invert      darker = nearer (default: brighter = nearer)\n"
        "  --cuda        use the CUDA backend if available\n", prog);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);

    const std::string ckpt = argv[1];
    const std::string image_path = argv[2];
    std::string variant = "small";
    std::string out_path = "depth.png";
    bool invert = false;
    bool use_cuda = false;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--variant")    variant = next();
        else if (a == "--out")   out_path = next();
        else if (a == "--invert") invert = true;
        else if (a == "--cuda")  use_cuda = true;
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); }
    }

    broimage::Image im;
    std::string err;
    if (!broimage::decode_file(image_path, im, &err)) {
        std::fprintf(stderr, "failed to decode '%s': %s\n", image_path.c_str(), err.c_str());
        return 1;
    }

    DepthAnythingConfig cfg = variant == "base"  ? DepthAnythingConfig::v2_base()
                            : variant == "large" ? DepthAnythingConfig::v2_large()
                                                 : DepthAnythingConfig::v2_small();
    try {
        DepthEstimator est(cfg);
        const bool is_file = ckpt.size() >= 12 &&
            ckpt.compare(ckpt.size() - 12, 12, ".safetensors") == 0;
        if (is_file) est.load_file(ckpt); else est.load(ckpt);

        if (use_cuda) {
            brotensor::init();
            if (brotensor::is_available(brotensor::Device::CUDA)) {
                est.to(brotensor::Device::CUDA);
                std::printf("running on CUDA\n");
            } else {
                std::fprintf(stderr, "CUDA requested but unavailable; using CPU\n");
            }
        }

        std::printf("image: %dx%d, estimating depth...\n", im.width, im.height);
        DepthMap dm = est.estimate(im.pixels.data(), im.width, im.height, im.channels);

        // Min-max normalize to [0,255] for visualization.
        float lo = dm.depth.empty() ? 0.0f : dm.depth[0];
        float hi = lo;
        for (float v : dm.depth) { lo = std::min(lo, v); hi = std::max(hi, v); }
        const float span = (hi > lo) ? (hi - lo) : 1.0f;

        std::vector<uint8_t> gray(dm.depth.size());
        for (std::size_t i = 0; i < dm.depth.size(); ++i) {
            float t = (dm.depth[i] - lo) / span;          // 0..1, nearer -> larger
            if (invert) t = 1.0f - t;
            gray[i] = static_cast<uint8_t>(std::clamp(t, 0.0f, 1.0f) * 255.0f + 0.5f);
        }
        if (!broimage::encode_png_file(out_path, gray.data(), dm.width, dm.height, 1)) {
            std::fprintf(stderr, "failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        std::printf("wrote %s (%dx%d, depth range [%.4f, %.4f])\n",
                    out_path.c_str(), dm.width, dm.height, lo, hi);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
