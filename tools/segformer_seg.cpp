// segformer_seg — minimal ControlNet SegFormer semantic-segmentation driver.
//
// Loads a SegFormer (MiT + all-MLP head) checkpoint, runs semantic segmentation
// on an image, and writes the ADE20K-palette-colorized class map (the
// ControlNet "seg" conditioning image).
//
//   segformer_seg <checkpoint-dir-or-file> <image> [options]
//     --out PATH        output PNG path (default: seg.png)
//     --device D        cpu|cuda|metal|vulkan|gpu (default: cpu)
//     --cuda            same as --device gpu: the best available GPU
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite.

#include "brovisionml/segformer.h"

#include "brotensor/runtime.h"

#include "tool_device.h"

#include "broimage/decode.h"
#include "broimage/encode.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using brovisionml::segformer::SegformerConfig;
using brovisionml::segformer::SegformerDetector;
using brovisionml::segformer::SegMap;

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <checkpoint-dir-or-file> <image> [options]\n"
        "  --out PATH      output PNG (default: seg.png)\n"
        "  --device D      cpu|cuda|metal|vulkan|gpu (default: cpu)\n"
        "  --cuda          same as --device gpu (best available GPU)\n", prog);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);

    const std::string ckpt = argv[1];
    const std::string image_path = argv[2];
    std::string out_path = "seg.png";
    brovisionml_tools::DeviceRequest device;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--out")       out_path = next();
        else if (a == "--cuda") device.parse("gpu");
        else if (a == "--device"){ if (!device.parse(next())) usage(argv[0]); }
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); }
    }

    broimage::Image im;
    std::string err;
    if (!broimage::decode_file(image_path, im, &err)) {
        std::fprintf(stderr, "failed to decode '%s': %s\n", image_path.c_str(), err.c_str());
        return 1;
    }

    try {
        SegformerDetector det;
        const bool is_file = ckpt.size() >= 12 &&
            ckpt.compare(ckpt.size() - 12, 12, ".safetensors") == 0;
        if (is_file) det.load_file(ckpt); else det.load(ckpt);

        const brotensor::Device dev = device.resolve();
        if (dev.is_gpu()) det.to(dev);

        std::printf("image: %dx%d, segmenting...\n", im.width, im.height);
        SegMap sm = det.detect(im.pixels.data(), im.width, im.height, im.channels);
        std::vector<uint8_t> rgb = SegformerDetector::colorize(sm);

        if (!broimage::encode_png_file(out_path, rgb.data(), sm.width, sm.height, 3)) {
            std::fprintf(stderr, "failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        std::printf("wrote %s (%dx%d)\n", out_path.c_str(), sm.width, sm.height);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
