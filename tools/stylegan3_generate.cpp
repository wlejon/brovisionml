// stylegan3_generate — minimal StyleGAN3-R command-line driver.
//
// Loads a converted StyleGAN3-R checkpoint (scripts/convert-stylegan3.py),
// samples a latent z from a seed, and writes the generated RGB image as a PNG.
//
//   stylegan3_generate <checkpoint-dir-or-file> [options]
//     --res R        256 (default) | 512 | 1024   (must match the checkpoint)
//     --seed N       PRNG seed for z (default: 0)
//     --trunc PSI    truncation psi (default: 1.0 = none)
//     --out PATH     output PNG path (default: stylegan3.png)
//     --cuda         run on the CUDA backend if available
//
// Built standalone only (BROVISIONML_TOOLS); not part of the test suite.

#include "brovisionml/stylegan3.h"

#include "brotensor/runtime.h"
#include "broimage/encode.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

namespace {

using brovisionml::stylegan3::Config;
using brovisionml::stylegan3::Generator;
using brovisionml::stylegan3::Image;
using brotensor::Tensor;

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <checkpoint-dir-or-file> [options]\n"
        "  --res R       256 (default) | 512 | 1024  (must match checkpoint)\n"
        "  --seed N      PRNG seed for z (default: 0)\n"
        "  --trunc PSI   truncation psi (default: 1.0)\n"
        "  --out PATH    output PNG (default: stylegan3.png)\n"
        "  --cuda        use the CUDA backend if available\n", prog);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) usage(argv[0]);

    const std::string ckpt = argv[1];
    int res = 256;
    unsigned long long seed = 0;
    float trunc = 1.0f;
    std::string out_path = "stylegan3.png";
    bool use_cuda = false;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (a == "--res")        res = std::atoi(next());
        else if (a == "--seed")  seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--trunc") trunc = static_cast<float>(std::atof(next()));
        else if (a == "--out")   out_path = next();
        else if (a == "--cuda")  use_cuda = true;
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); }
    }

    Config cfg = (res == 512)  ? Config::r512()
               : (res == 1024) ? Config::r1024()
                               : Config::r256();

    try {
        brotensor::init();
        Generator g(cfg);

        const bool is_file = ckpt.size() >= 12 &&
            ckpt.compare(ckpt.size() - 12, 12, ".safetensors") == 0;
        if (is_file) g.load_file(ckpt); else g.load(ckpt);

        if (use_cuda) {
            if (brotensor::is_available(brotensor::Device::CUDA))
                g.to(brotensor::Device::CUDA);
            else
                std::fprintf(stderr, "(CUDA not available — running on CPU)\n");
        }

        // Sample z ~ N(0,1) from the seed (host), then migrate with the model.
        Tensor z = Tensor::mat(1, cfg.z_dim);
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int i = 0; i < cfg.z_dim; ++i) z[i] = nd(rng);
        if (g.device() != brotensor::Device::CPU) z = z.to(g.device());

        Image img = g.generate(z, trunc);
        if (!broimage::encode_png_file(out_path, img.rgb.data(), img.width,
                                       img.height, img.channels)) {
            std::fprintf(stderr, "failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        std::printf("wrote %dx%d image to %s (seed %llu, psi %.3f)\n",
                    img.width, img.height, out_path.c_str(), seed,
                    static_cast<double>(trunc));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
