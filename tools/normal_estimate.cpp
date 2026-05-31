// normal_estimate — minimal DSINE surface-normal command-line driver.
//
// Loads a DSINE_v02 checkpoint, estimates a per-pixel surface-normal map for an
// image, and writes it as an RGB PNG (the standard normal-map encoding:
// R=nx, G=ny, B=nz, each mapped from [-1,1] to [0,255] via (n+1)*0.5).
//
//   normal_estimate <checkpoint-dir-or-file> <image> [options]
//     --out PATH    output PNG path (default: normal.png)
//     --fov DEG     assumed field-of-view for the synthesized intrinsics (60)
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite. CPU only.

#include "brovisionml/dsine.h"

#include "broimage/decode.h"
#include "broimage/encode.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using brovisionml::dsine::DsineConfig;
using brovisionml::dsine::NormalEstimator;
using brovisionml::dsine::NormalMap;

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <checkpoint-dir-or-file> <image> [options]\n"
        "  --out PATH    output PNG (default: normal.png)\n"
        "  --fov DEG     assumed field-of-view for intrinsics (default: 60)\n", prog);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);

    const std::string ckpt = argv[1];
    const std::string image_path = argv[2];
    std::string out_path = "normal.png";
    float fov_deg = 60.0f;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--out")      out_path = next();
        else if (a == "--fov") fov_deg = static_cast<float>(std::atof(next()));
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); }
    }

    broimage::Image im;
    std::string err;
    if (!broimage::decode_file(image_path, im, &err)) {
        std::fprintf(stderr, "failed to decode '%s': %s\n", image_path.c_str(), err.c_str());
        return 1;
    }

    try {
        DsineConfig cfg;
        cfg.fov_deg = fov_deg;
        NormalEstimator est(cfg);
        const bool is_file = ckpt.size() >= 12 &&
            ckpt.compare(ckpt.size() - 12, 12, ".safetensors") == 0;
        if (is_file) est.load_file(ckpt); else est.load(ckpt);

        std::printf("image: %dx%d, estimating surface normals...\n", im.width, im.height);
        NormalMap nm = est.estimate(im.pixels.data(), im.width, im.height, im.channels);

        // Encode (nx,ny,nz) -> RGB with (n+1)*0.5, clamped to [0,1]. The map is
        // planar NCHW (3,H,W); the PNG is interleaved HWC RGB.
        const std::size_t plane = static_cast<std::size_t>(nm.height) * nm.width;
        std::vector<uint8_t> rgb(plane * 3);
        for (std::size_t p = 0; p < plane; ++p) {
            for (int c = 0; c < 3; ++c) {
                float v = (nm.normals[static_cast<std::size_t>(c) * plane + p] + 1.0f) * 0.5f;
                rgb[p * 3 + c] =
                    static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
        if (!broimage::encode_png_file(out_path, rgb.data(), nm.width, nm.height, 3)) {
            std::fprintf(stderr, "failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        std::printf("wrote %s (%dx%d, camera-space unit normals)\n",
                    out_path.c_str(), nm.width, nm.height);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
