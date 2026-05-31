// sam_amg — SAM "segment everything" command-line driver.
//
// Loads a HuggingFace SAM checkpoint and runs the automatic mask generator over
// a whole image (no prompts), then writes a colored overlay PNG with every
// generated mask blended over the input.
//
//   sam_amg <checkpoint-dir-or-file> <image> [options]
//     --points-per-side N     grid density per side (default 32)
//     --points-per-batch N    grid points per batched decode pass (default 64)
//     --pred-iou-thresh F     predicted-IoU cutoff (default 0.88)
//     --stability-thresh F    stability-score cutoff (default 0.95)
//     --crop-n-layers N       crop pyramid layers (default 0)
//     --min-region-area N     remove regions/holes smaller than N px (default 0)
//     --variant V             vit_h (default) | vit_l | vit_b
//     --out PATH              output overlay PNG (default: amg.png)
//     --cuda                  run on the CUDA backend if available
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite.

#define _CRT_SECURE_NO_WARNINGS  // std::sscanf for the tiny arg parsing

#include "brovisionml/sam_amg.h"

#include "brotensor/runtime.h"
#include "broimage/decode.h"
#include "broimage/encode.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using brovisionml::sam::AmgConfig;
using brovisionml::sam::AutomaticMaskGenerator;
using brovisionml::sam::GeneratedMask;
using brovisionml::sam::Sam;
using brovisionml::sam::SamConfig;

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <checkpoint-dir-or-file> <image> [options]\n"
        "  --points-per-side N   grid density per side (default 32)\n"
        "  --points-per-batch N  grid points per batched decode (default 64)\n"
        "  --pred-iou-thresh F   predicted-IoU cutoff (default 0.88)\n"
        "  --stability-thresh F  stability-score cutoff (default 0.95)\n"
        "  --crop-n-layers N     crop pyramid layers (default 0)\n"
        "  --min-region-area N   drop regions/holes < N px (default 0)\n"
        "  --variant V           vit_h (default) | vit_l | vit_b\n"
        "  --out PATH            output overlay PNG (default: amg.png)\n"
        "  --cuda                use the CUDA backend if available\n", prog);
    std::exit(2);
}

// A small fixed palette; masks cycle through it by index.
const uint8_t kPalette[][3] = {
    {230, 25, 75}, {60, 180, 75}, {255, 225, 25}, {0, 130, 200},
    {245, 130, 48}, {145, 30, 180}, {70, 240, 240}, {240, 50, 230},
    {210, 245, 60}, {250, 190, 190}, {0, 128, 128}, {230, 190, 255},
    {170, 110, 40}, {255, 250, 200}, {128, 0, 0}, {170, 255, 195},
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);

    const std::string ckpt = argv[1];
    const std::string image_path = argv[2];
    std::string variant = "vit_h";
    std::string out_path = "amg.png";
    bool use_cuda = false;
    AmgConfig amg;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--points-per-side")     amg.points_per_side = std::atoi(next());
        else if (a == "--points-per-batch") amg.points_per_batch = std::atoi(next());
        else if (a == "--pred-iou-thresh") amg.pred_iou_thresh = static_cast<float>(std::atof(next()));
        else if (a == "--stability-thresh") amg.stability_score_thresh = static_cast<float>(std::atof(next()));
        else if (a == "--crop-n-layers")  amg.crop_n_layers = std::atoi(next());
        else if (a == "--min-region-area") amg.min_mask_region_area = std::atoi(next());
        else if (a == "--variant")        variant = next();
        else if (a == "--out")            out_path = next();
        else if (a == "--cuda")           use_cuda = true;
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); }
    }

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

        std::printf("image: %dx%d, generating masks (%d^2 grid)...\n",
                    im.width, im.height, amg.points_per_side);
        AutomaticMaskGenerator gen(sam, amg);
        std::vector<GeneratedMask> masks =
            gen.generate(im.pixels.data(), im.width, im.height, im.channels);
        std::printf("generated %zu mask(s)\n", masks.size());

        // Build an RGB overlay: grayscale base, each mask blended in its color.
        const int W = im.width, H = im.height;
        std::vector<uint8_t> rgb(static_cast<std::size_t>(W) * H * 3);
        for (int i = 0; i < W * H; ++i) {
            const uint8_t* px = im.pixels.data() +
                                static_cast<std::size_t>(i) * im.channels;
            // luma of the (RGBA) source as a dim base
            const uint8_t base = static_cast<uint8_t>(
                (px[0] * 30 + px[1] * 59 + px[2] * 11) / 100 / 2);
            rgb[static_cast<std::size_t>(i) * 3 + 0] = base;
            rgb[static_cast<std::size_t>(i) * 3 + 1] = base;
            rgb[static_cast<std::size_t>(i) * 3 + 2] = base;
        }
        // generate() already sorts by descending area, so larger masks are laid
        // down first and smaller ones paint on top.
        const int npal = static_cast<int>(sizeof(kPalette) / sizeof(kPalette[0]));
        for (std::size_t mi = 0; mi < masks.size(); ++mi) {
            const uint8_t* col = kPalette[mi % static_cast<std::size_t>(npal)];
            const GeneratedMask& m = masks[mi];
            for (int i = 0; i < W * H; ++i) {
                if (!m.mask[static_cast<std::size_t>(i)]) continue;
                uint8_t* o = rgb.data() + static_cast<std::size_t>(i) * 3;
                for (int c = 0; c < 3; ++c)
                    o[c] = static_cast<uint8_t>((o[c] + col[c]) / 2);
            }
        }

        if (!broimage::encode_png_file(out_path, rgb.data(), W, H, 3)) {
            std::fprintf(stderr, "failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        std::printf("wrote %s\n", out_path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
