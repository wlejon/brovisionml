// lineart — minimal ControlNet lineart command-line driver.
//
// Loads a lineart generator checkpoint, extracts a line drawing for an image,
// and writes the line map as a single-channel grayscale PNG (line*255).
//
//   lineart <checkpoint-dir-or-file> <image> [options]
//     --out PATH        output PNG path (default: lineart.png)
//     --resolution N    longer-side working resolution (default: 0 = native)
//     --no-invert       write the raw generator output (bright field, dark
//                       lines) instead of the inverted ControlNet convention
//     --device D        cpu|cuda|hip|rocm|metal|vulkan|gpu (default: cpu)
//     --cuda            same as --device gpu: the best available GPU
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite.

#include "brovisionml/lineart.h"

#include "brotensor/runtime.h"

#include "tool_device.h"

#include "broimage/decode.h"
#include "broimage/encode.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using brovisionml::lineart::LineMap;
using brovisionml::lineart::LineartConfig;
using brovisionml::lineart::LineartDetector;

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <checkpoint-dir-or-file> <image> [options]\n"
        "  --out PATH        output PNG (default: lineart.png)\n"
        "  --resolution N    longer-side working resolution (default: 0 = native)\n"
        "  --no-invert       write the raw output (bright field, dark lines)\n"
        "  --device D        cpu|cuda|hip|rocm|metal|vulkan|gpu (default: cpu)\n"
        "  --cuda            same as --device gpu (best available GPU)\n", prog);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);

    const std::string ckpt = argv[1];
    const std::string image_path = argv[2];
    std::string out_path = "lineart.png";
    int resolution = 0;
    bool invert = true;
    brovisionml_tools::DeviceRequest device;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--out")             out_path = next();
        else if (a == "--resolution") resolution = std::atoi(next());
        else if (a == "--no-invert")  invert = false;
        else if (a == "--cuda")       device.parse("gpu");
        else if (a == "--device")     { if (!device.parse(next())) usage(argv[0]); }
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); }
    }

    broimage::Image im;
    std::string err;
    if (!broimage::decode_file(image_path, im, &err)) {
        std::fprintf(stderr, "failed to decode '%s': %s\n", image_path.c_str(), err.c_str());
        return 1;
    }

    try {
        LineartConfig cfg;
        cfg.detect_resolution = resolution;
        cfg.invert = invert;
        LineartDetector det(cfg);
        const bool is_file = ckpt.size() >= 12 &&
            ckpt.compare(ckpt.size() - 12, 12, ".safetensors") == 0;
        if (is_file) det.load_file(ckpt); else det.load(ckpt);

        const brotensor::Device dev = device.resolve();
        if (dev.is_gpu()) det.to(dev);

        std::printf("image: %dx%d, extracting line drawing...\n", im.width, im.height);
        LineMap lm = det.detect(im.pixels.data(), im.width, im.height, im.channels);

        // Line map is [0,1], row-major (H,W); write as grayscale (line*255).
        const std::size_t n = static_cast<std::size_t>(lm.height) * lm.width;
        std::vector<uint8_t> gray(n);
        for (std::size_t p = 0; p < n; ++p)
            gray[p] = static_cast<uint8_t>(std::clamp(lm.line[p], 0.0f, 1.0f) * 255.0f + 0.5f);

        if (!broimage::encode_png_file(out_path, gray.data(), lm.width, lm.height, 1)) {
            std::fprintf(stderr, "failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        std::printf("wrote %s (%dx%d, line map)\n",
                    out_path.c_str(), lm.width, lm.height);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
