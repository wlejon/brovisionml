// openpose_pose — minimal ControlNet OpenPose body-pose command-line driver.
//
// Loads the body_pose checkpoint, detects body keypoints for an image, and
// writes the canonical openpose control image (colored limb sticks + joints on
// black). Body-only: hand and face are not detected or drawn.
//
//   openpose_pose <checkpoint-dir-or-file> <image> [options]
//     --out PATH        output PNG path (default: openpose.png)
//     --resolution N    detect resolution (shorter side, default: 512)
//     --cuda            run on the CUDA device if available
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite.

#include "brovisionml/openpose.h"

#include "brotensor/runtime.h"

#include "broimage/decode.h"
#include "broimage/encode.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using brovisionml::openpose::OpenposeConfig;
using brovisionml::openpose::OpenposeDetector;
using brovisionml::openpose::PoseResult;

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <checkpoint-dir-or-file> <image> [options]\n"
        "  --out PATH      output PNG (default: openpose.png)\n"
        "  --resolution N  detect resolution (shorter side, default: 512)\n"
        "  --cuda          run on the CUDA device if available\n", prog);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);

    const std::string ckpt = argv[1];
    const std::string image_path = argv[2];
    std::string out_path = "openpose.png";
    OpenposeConfig cfg;
    bool want_cuda = false;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--out")             out_path = next();
        else if (a == "--resolution") cfg.detect_resolution = std::atoi(next());
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
        OpenposeDetector det(cfg);
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

        std::printf("image: %dx%d, detecting body pose...\n", im.width, im.height);
        PoseResult pose = det.detect(im.pixels.data(), im.width, im.height, im.channels);
        std::printf("detected %zu people at %dx%d\n",
                    pose.bodies.size(), pose.width, pose.height);

        std::vector<uint8_t> canvas = OpenposeDetector::draw(pose);
        if (!broimage::encode_png_file(out_path, canvas.data(),
                                       pose.width, pose.height, 3)) {
            std::fprintf(stderr, "failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        std::printf("wrote %s (%dx%d, %zu people)\n",
                    out_path.c_str(), pose.width, pose.height, pose.bodies.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
