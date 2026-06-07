#include "brovisionml/stylegan3.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

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

Generator::InvertResult Generator::invert(const Image& target,
                                          const InvertOptions& opt) const {
    const int C = cfg_.img_channels;
    const int H = cfg_.img_resolution;
    const int W = cfg_.img_resolution;
    if (target.width != W || target.height != H || target.channels != C)
        throw std::runtime_error(
            "stylegan3::Generator::invert: target image size must match the model resolution");

    // Target in synthesis (raw) space, NCHW — the exact inverse of to_image's
    // uint8 post-process: raw = (px - 128) / 127.5.
    const int HW = H * W;
    std::vector<float> traw(static_cast<std::size_t>(C) * HW);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < C; ++c) {
                const unsigned char px =
                    target.rgb[(static_cast<std::size_t>(y) * W + x) * C + c];
                traw[(static_cast<std::size_t>(c) * H + y) * W + x] =
                    (static_cast<float>(px) - 128.0f) / 127.5f;
            }
    Tensor target_dev = Tensor::from_host_on(device_, traw.data(), 1, C * HW);

    // Init W+ = w_avg broadcast over the rows (+ optional gaussian jitter).
    const int n_ws = cfg_.num_ws();
    const int w_dim = cfg_.w_dim;
    Tensor wavg_host = mapping_.w_avg().to(brotensor::Device::CPU);
    const float* wa = wavg_host.host_f32();

    uint64_t rng = opt.seed ? opt.seed : 0x9E3779B97F4A7C15ull;
    auto next_unit = [&]() -> double {           // splitmix64 → uniform (0,1)
        rng += 0x9E3779B97F4A7C15ull;
        uint64_t z = rng;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z = z ^ (z >> 31);
        return (z >> 11) * (1.0 / 9007199254740992.0);
    };
    std::vector<float> w0(static_cast<std::size_t>(n_ws) * w_dim);
    for (int r = 0; r < n_ws; ++r)
        for (int j = 0; j < w_dim; ++j) {
            float v = wa[j];
            if (opt.init_noise > 0.0f) {
                const double u1 = next_unit(), u2 = next_unit();
                const double gn = std::sqrt(-2.0 * std::log(u1 + 1e-12)) *
                                  std::cos(2.0 * 3.14159265358979323846 * u2);
                v += opt.init_noise * static_cast<float>(gn);
            }
            w0[static_cast<std::size_t>(r) * w_dim + j] = v;
        }
    Tensor ws = Tensor::from_host_on(device_, w0.data(), n_ws, w_dim);

    // Optional L2-toward-w_avg regularizer: precompute the negated broadcast.
    Tensor neg_wavg_bc;
    if (opt.reg_w > 0.0f) {
        std::vector<float> nb(static_cast<std::size_t>(n_ws) * w_dim);
        for (int r = 0; r < n_ws; ++r)
            for (int j = 0; j < w_dim; ++j)
                nb[static_cast<std::size_t>(r) * w_dim + j] = -wa[j];
        neg_wavg_bc = Tensor::from_host_on(device_, nb.data(), n_ws, w_dim);
    }

    // Adam state.
    Tensor m = Tensor::zeros_on(device_, n_ws, w_dim);
    Tensor v = Tensor::zeros_on(device_, n_ws, w_dim);
    const float inv_n = 1.0f / static_cast<float>(C * HW);

    InvertResult res;
    SynthesisNetwork::Cache cache;
    const double kPi = 3.14159265358979323846;
    for (int step = 1; step <= opt.num_steps; ++step) {
        // NVlabs-style LR schedule: a short warm-up then a cosine ramp-down, with
        // `opt.lr` as the peak. Smooths the late-stage refinement.
        float lr = opt.lr;
        if (opt.num_steps > 1) {
            const double t = static_cast<double>(step - 1) / (opt.num_steps - 1);
            double r = std::min(1.0, (1.0 - t) / 0.25);     // ramp down over last 25%
            r = 0.5 - 0.5 * std::cos(r * kPi);
            r *= std::min(1.0, t / 0.05);                   // warm up over first 5%
            lr = static_cast<float>(opt.lr * r);
        }
        Tensor img = synthesis_.forward_cached(ws, cache);   // (1, C*HW) raw

        // diff = img - target.
        Tensor diff = img.clone();
        Tensor negt = target_dev.clone();
        brotensor::scale_inplace(negt, -1.0f);
        brotensor::add_inplace(diff, negt);

        const bool want_loss = opt.on_step || (step == opt.num_steps);
        if (want_loss) {
            const std::vector<float> dh = diff.to_host_vector();
            double sse = 0;
            for (float d : dh) sse += static_cast<double>(d) * d;
            res.loss = static_cast<float>(sse * inv_n);
            if (opt.on_step) opt.on_step(step, res.loss);
        }

        // dL/dimg = (2/N) · diff, then backprop to the W+ rows.
        brotensor::scale_inplace(diff, 2.0f * inv_n);
        Tensor dws;
        synthesis_.backward(ws, cache, diff, dws);

        // grad += 2·reg_w·(ws - w_avg).
        if (opt.reg_w > 0.0f) {
            Tensor reg = ws.clone();
            brotensor::add_inplace(reg, neg_wavg_bc);
            brotensor::scale_inplace(reg, 2.0f * opt.reg_w);
            brotensor::add_inplace(dws, reg);
        }

        brotensor::adam_step(ws, dws, m, v, lr, 0.9f, 0.999f, 1e-8f, step);
    }

    res.w = ws;
    return res;
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
