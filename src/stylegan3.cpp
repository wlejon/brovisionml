#include "brovisionml/stylegan3.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include <cmath>
#include <utility>

namespace brovisionml::stylegan3 {

namespace st = brotensor::safetensors;
using brotensor::Tensor;

Generator::Generator(Config cfg)
    : cfg_(std::move(cfg)), mapping_(cfg_), synthesis_(cfg_) {}

void Generator::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void Generator::load_file(const std::string& path) {
    // One converted checkpoint carries both the `mapping.` and `synthesis.`
    // tensor namespaces (see scripts/convert-stylegan3.py).
    st::File f = st::File::open(path);
    mapping_.load(&f, "mapping");
    synthesis_.load(&f, "synthesis");
}

void Generator::to(brotensor::Device dev) {
    if (dev == device_) return;
    mapping_.to(dev);
    synthesis_.to(dev);
    device_ = dev;
}

Tensor Generator::map(const Tensor& z, float truncation_psi,
                      int truncation_cutoff) const {
    return mapping_.forward(z, truncation_psi, truncation_cutoff);
}

Tensor Generator::synthesize(const Tensor& ws) const {
    return synthesis_.forward(ws);
}

Image Generator::generate(const Tensor& z, float truncation_psi,
                          int truncation_cutoff) const {
    return render(map(z, truncation_psi, truncation_cutoff));
}

Image Generator::to_image(const Tensor& img) const {
    const int C = cfg_.img_channels;
    const int H = cfg_.img_resolution;
    const int W = cfg_.img_resolution;

    Tensor host = img.to(brotensor::Device::CPU);
    const float* d = host.host_f32();

    Image out;
    out.width = W;
    out.height = H;
    out.channels = C;
    out.rgb.resize(static_cast<std::size_t>(W) * H * C);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                // (x * 127.5 + 128).clamp(0, 255), matching the reference's
                // uint8 image post-processing.
                float v = d[(static_cast<std::size_t>(c) * H + y) * W + x];
                v = v * 127.5f + 128.0f;
                v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
                out.rgb[(static_cast<std::size_t>(y) * W + x) * C + c] =
                    static_cast<unsigned char>(v + 0.5f);
            }
        }
    }
    return out;
}

}  // namespace brovisionml::stylegan3
