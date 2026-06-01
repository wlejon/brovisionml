// hed_edges — minimal HED soft-edge command-line driver.
//
// Loads a ControlNetHED checkpoint, detects soft edges for an image, and writes
// the edge probability map as a single-channel grayscale PNG (edge*255).
//
//   hed_edges <checkpoint-dir-or-file> <image> [options]
//     --out PATH        output PNG path (default: edges.png)
//     --resolution N    longer-side working resolution (default: 0 = native)
//     --cuda            run on the CUDA device if available
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite.

#include "brovisionml/hed.h"

#include "brotensor/runtime.h"

#include "broimage/decode.h"
#include "broimage/encode.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using brovisionml::hed::EdgeMap;
using brovisionml::hed::HedConfig;
using brovisionml::hed::SoftEdgeDetector;

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <checkpoint-dir-or-file> <image> [options]\n"
        "  --out PATH        output PNG (default: edges.png)\n"
        "  --resolution N    longer-side working resolution (default: 0 = native)\n"
        "  --cuda            run on the CUDA device if available\n", prog);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);

    const std::string ckpt = argv[1];
    const std::string image_path = argv[2];
    std::string out_path = "edges.png";
    int resolution = 0;
    bool want_cuda = false;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--out")             out_path = next();
        else if (a == "--resolution") resolution = std::atoi(next());
        else if (a == "--cuda")       want_cuda = true;
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); }
    }

    broimage::Image im;
    std::string err;
    if (!broimage::decode_file(image_path, im, &err)) {
        std::fprintf(stderr, "failed to decode '%s': %s\n", image_path.c_str(), err.c_str());
        return 1;
    }

    try {
        HedConfig cfg;
        cfg.detect_resolution = resolution;
        SoftEdgeDetector det(cfg);
        const bool is_file = ckpt.size() >= 12 &&
            ckpt.compare(ckpt.size() - 12, 12, ".safetensors") == 0;
        if (is_file) det.load_file(ckpt); else det.load(ckpt);

        if (want_cuda) {
            brotensor::init();
            if (brotensor::is_available(brotensor::Device::CUDA)) {
                det.to(brotensor::Device::CUDA);
                std::printf("running on CUDA\n");
            } else {
                std::fprintf(stderr, "CUDA requested but unavailable; using CPU\n");
            }
        }

        std::printf("image: %dx%d, detecting soft edges...\n", im.width, im.height);
        EdgeMap em = det.detect(im.pixels.data(), im.width, im.height, im.channels);

        // Edge map is [0,1], row-major (H,W); write as grayscale (edge*255).
        const std::size_t n = static_cast<std::size_t>(em.height) * em.width;
        std::vector<uint8_t> gray(n);
        for (std::size_t p = 0; p < n; ++p)
            gray[p] = static_cast<uint8_t>(std::clamp(em.edge[p], 0.0f, 1.0f) * 255.0f + 0.5f);

        if (!broimage::encode_png_file(out_path, gray.data(), em.width, em.height, 1)) {
            std::fprintf(stderr, "failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        std::printf("wrote %s (%dx%d, soft-edge map)\n",
                    out_path.c_str(), em.width, em.height);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
