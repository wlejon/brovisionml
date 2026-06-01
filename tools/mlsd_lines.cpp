// mlsd_lines — minimal ControlNet MLSD straight-line command-line driver.
//
// Loads an M-LSD checkpoint, detects straight line segments for an image, and
// writes them as white lines on black (the ControlNet "mlsd" conditioning image).
//
//   mlsd_lines <checkpoint-dir-or-file> <image> [options]
//     --out PATH        output PNG path (default: mlsd.png)
//     --score-thr F     center-score threshold (default: 0.1)
//     --dist-thr F      minimum segment length on the 256 grid (default: 0.1)
//     --cuda            run on the CUDA device if available
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite.

#include "brovisionml/mlsd.h"

#include "brotensor/runtime.h"

#include "broimage/decode.h"
#include "broimage/encode.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using brovisionml::mlsd::LineMap;
using brovisionml::mlsd::LineSegment;
using brovisionml::mlsd::MlsdConfig;
using brovisionml::mlsd::MLSDdetector;

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <checkpoint-dir-or-file> <image> [options]\n"
        "  --out PATH      output PNG (default: mlsd.png)\n"
        "  --score-thr F   center-score threshold (default: 0.1)\n"
        "  --dist-thr F    minimum segment length on the 256 grid (default: 0.1)\n"
        "  --cuda          run on the CUDA device if available\n", prog);
    std::exit(2);
}

// Bresenham line into a single-channel image (value 255).
void draw_line(std::vector<uint8_t>& img, int W, int H,
               int x0, int y0, int x1, int y1) {
    const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H)
            img[static_cast<std::size_t>(y0) * W + x0] = 255;
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);

    const std::string ckpt = argv[1];
    const std::string image_path = argv[2];
    std::string out_path = "mlsd.png";
    MlsdConfig cfg;
    bool want_cuda = false;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--out")             out_path = next();
        else if (a == "--score-thr")  cfg.score_thr = static_cast<float>(std::atof(next()));
        else if (a == "--dist-thr")   cfg.dist_thr = static_cast<float>(std::atof(next()));
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
        MLSDdetector det(cfg);
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

        std::printf("image: %dx%d, detecting straight lines...\n", im.width, im.height);
        LineMap lm = det.detect(im.pixels.data(), im.width, im.height, im.channels);

        std::vector<uint8_t> out(static_cast<std::size_t>(lm.width) * lm.height, 0);
        for (const LineSegment& s : lm.segments)
            draw_line(out, lm.width, lm.height,
                      (int)std::lround(s.x1), (int)std::lround(s.y1),
                      (int)std::lround(s.x2), (int)std::lround(s.y2));

        if (!broimage::encode_png_file(out_path, out.data(), lm.width, lm.height, 1)) {
            std::fprintf(stderr, "failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        std::printf("wrote %s (%dx%d, %zu line segments)\n",
                    out_path.c_str(), lm.width, lm.height, lm.segments.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
