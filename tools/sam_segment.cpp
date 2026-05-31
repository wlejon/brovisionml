// sam_segment — minimal SAM command-line driver.
//
// Loads a HuggingFace SAM checkpoint, encodes an image once, segments it with
// a point and/or box prompt (in original-image pixel coordinates), and writes
// the best mask as a grayscale PNG.
//
//   sam_segment <checkpoint-dir-or-file> <image> [options]
//     --point X,Y          a foreground click (repeatable)
//     --bg-point X,Y       a background click (repeatable)
//     --box X1,Y1,X2,Y2    a box prompt (repeatable)
//     --variant V          vit_h (default) | vit_l | vit_b
//     --single             return the single best mask (default: multimask)
//     --out PATH           output PNG path (default: mask.png)
//     --cuda               run on the CUDA backend if available
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite.

#define _CRT_SECURE_NO_WARNINGS  // std::sscanf for the tiny X,Y arg parsing

#include "brovisionml/sam.h"

#include "brotensor/runtime.h"
#include "broimage/decode.h"
#include "broimage/encode.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using brovisionml::sam::Sam;
using brovisionml::sam::SamConfig;

bool parse2(const char* s, float& a, float& b) {
    return std::sscanf(s, "%f,%f", &a, &b) == 2;
}
bool parse4(const char* s, float& a, float& b, float& c, float& d) {
    return std::sscanf(s, "%f,%f,%f,%f", &a, &b, &c, &d) == 4;
}

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <checkpoint-dir-or-file> <image> [options]\n"
        "  --point X,Y        foreground click (repeatable)\n"
        "  --bg-point X,Y     background click (repeatable)\n"
        "  --box X1,Y1,X2,Y2  box prompt (repeatable)\n"
        "  --variant V        vit_h (default) | vit_l | vit_b\n"
        "  --single           single best mask (default: multimask)\n"
        "  --out PATH         output PNG (default: mask.png)\n"
        "  --cuda             use the CUDA backend if available\n", prog);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);

    const std::string ckpt = argv[1];
    const std::string image_path = argv[2];
    std::string variant = "vit_h";
    std::string out_path = "mask.png";
    bool multimask = true;
    bool use_cuda = false;

    std::vector<std::array<float, 2>> points;
    std::vector<int> labels;
    std::vector<std::array<float, 4>> boxes;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--point" || a == "--bg-point") {
            float x, y;
            if (!parse2(next(), x, y)) usage(argv[0]);
            points.push_back({x, y});
            labels.push_back(a == "--point" ? 1 : 0);
        } else if (a == "--box") {
            float x1, y1, x2, y2;
            if (!parse4(next(), x1, y1, x2, y2)) usage(argv[0]);
            boxes.push_back({x1, y1, x2, y2});
        } else if (a == "--variant") {
            variant = next();
        } else if (a == "--single") {
            multimask = false;
        } else if (a == "--out") {
            out_path = next();
        } else if (a == "--cuda") {
            use_cuda = true;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage(argv[0]);
        }
    }

    if (points.empty() && boxes.empty()) {
        std::fprintf(stderr, "error: provide at least one --point / --box prompt\n");
        return 2;
    }

    // Decode the input image (broimage emits RGBA).
    broimage::Image im;
    std::string err;
    if (!broimage::decode_file(image_path, im, &err)) {
        std::fprintf(stderr, "failed to decode '%s': %s\n", image_path.c_str(),
                     err.c_str());
        return 1;
    }

    SamConfig cfg = variant == "vit_l" ? SamConfig::vit_l()
                  : variant == "vit_b" ? SamConfig::vit_b()
                                       : SamConfig::vit_h();
    try {
        Sam sam(cfg);
        const bool is_file = ckpt.size() >= 12 &&
            ckpt.compare(ckpt.size() - 12, 12, ".safetensors") == 0;
        if (is_file) sam.load_file(ckpt); else sam.load(ckpt);

        if (use_cuda) {
            brotensor::init();
            if (brotensor::is_available(brotensor::Device::CUDA)) {
                sam.to(brotensor::Device::CUDA);
                std::printf("running on CUDA\n");
            } else {
                std::fprintf(stderr, "CUDA requested but unavailable; using CPU\n");
            }
        }

        std::printf("image: %dx%d, encoding...\n", im.width, im.height);
        sam.set_image(im.pixels.data(), im.width, im.height, im.channels);

        brovisionml::sam::Segmentation seg =
            sam.segment(points, labels, boxes, multimask);

        const int b = seg.best();
        std::printf("predicted %d mask(s); best #%d iou=%.3f\n",
                    seg.num, b, seg.iou.empty() ? 0.0f : seg.iou[b]);

        // Write the best mask as a grayscale PNG (255 where logit > 0).
        const int w = seg.width, h = seg.height;
        std::vector<uint8_t> gray(static_cast<std::size_t>(w) * h);
        const float* m = seg.logits.data() + static_cast<std::size_t>(b) * w * h;
        long long on = 0;
        for (std::size_t i = 0; i < gray.size(); ++i) {
            const bool fg = m[i] > 0.0f;
            gray[i] = fg ? 255 : 0;
            on += fg ? 1 : 0;
        }
        if (!broimage::encode_png_file(out_path, gray.data(), w, h, 1)) {
            std::fprintf(stderr, "failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        std::printf("wrote %s (%lld / %d px foreground)\n",
                    out_path.c_str(), on, w * h);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
