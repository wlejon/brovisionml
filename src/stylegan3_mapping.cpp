#include "brovisionml/stylegan3.h"

#include "weights_util.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace brovisionml::stylegan3 {

namespace st = brotensor::safetensors;
using brotensor::Tensor;

// ─── Config presets ──────────────────────────────────────────────────────────

// channel_base is resolution-dependent in the released zoo, not constant per
// config family: the 256² FFHQ-U checkpoints are the "cheap" config trained with
// --cbase=16384, while 512/1024 use the default --cbase=32768 (NVlabs
// docs/configs.md). Config-R then internally doubles channel_base, config-T does
// not — so the *pickle* init_kwargs read: R 256→32768, 512/1024→65536; T
// 256→16384, 512/1024→32768. channel_max stays constant per variant (R 1024,
// T 512). Getting this wrong yields the right layer sizes but wrong channel
// counts, so the by-name weight load fails loudly (missing 'L*_*_<count>').
static int base_for(int resolution, int base256) {
    return resolution <= 256 ? base256 : base256 * 2;
}

static Config config_r(int resolution) {
    Config c;
    c.img_resolution    = resolution;
    c.channel_base      = base_for(resolution, 32768);
    c.channel_max       = 1024;
    c.conv_kernel       = 1;
    c.use_radial_filters = true;
    return c;
}

// config-T: translation-equivariant — 3x3 conv, separable (non-radial) filters,
// and half config-R's channel budget (channel_max 512, channel_base 16384/32768).
static Config config_t(int resolution) {
    Config c;
    c.img_resolution    = resolution;
    c.channel_base      = base_for(resolution, 16384);
    c.channel_max       = 512;
    c.conv_kernel       = 3;
    c.use_radial_filters = false;
    return c;
}

Config Config::r256()  { return config_r(256); }
Config Config::r512()  { return config_r(512); }
Config Config::r1024() { return config_r(1024); }

Config Config::t256()  { return config_t(256); }
Config Config::t512()  { return config_t(512); }
Config Config::t1024() { return config_t(1024); }

// ─── FullyConnectedLayer ─────────────────────────────────────────────────────

void FullyConnectedLayer::load(const void* file, const std::string& who,
                               const std::string& prefix, int in, int out,
                               float lr_multiplier, bool with_bias, Act activation) {
    const st::File& f = *reinterpret_cast<const st::File*>(file);
    in_features  = in;
    out_features = out;
    has_bias     = with_bias;
    act          = activation;

    // weight (out, in), gain = lr_multiplier / sqrt(in) baked in.
    weight = brovisionml::detail::load_whole(f, who, prefix + ".weight", out, in);
    const float weight_gain = lr_multiplier / std::sqrt(static_cast<float>(in));
    if (weight_gain != 1.0f) {
        float* w = weight.host_f32_mut();
        const std::size_t n = static_cast<std::size_t>(out) * in;
        for (std::size_t i = 0; i < n; ++i) w[i] *= weight_gain;
    }

    // bias (out,), gain = lr_multiplier baked in. Absent -> zero bias.
    if (with_bias) {
        bias = brovisionml::detail::load_whole(f, who, prefix + ".bias", out, 1);
        const float bias_gain = lr_multiplier;
        if (bias_gain != 1.0f) {
            float* b = bias.host_f32_mut();
            for (int i = 0; i < out; ++i) b[i] *= bias_gain;
        }
    } else {
        bias = Tensor::vec(out);  // zeros
    }
}

void FullyConnectedLayer::to(brotensor::Device dev) {
    weight = weight.to(dev);
    bias   = bias.to(dev);
}

void FullyConnectedLayer::forward(const Tensor& X, Tensor& Y) const {
    // linear_forward_batched computes Y = X W^T + b over X's rows.
    if (act == Act::Linear) {
        brotensor::linear_forward_batched(weight, bias, X, Y);
        return;
    }
    // LRelu: y = leaky_relu(X W^T + b) * sqrt(2)  (the reference's bias_act
    // default gain for 'lrelu', alpha 0.2, no clamp). Bias is folded into the
    // GEMM; bias_act then applies the activation + gain with a null bias.
    Tensor t;
    brotensor::linear_forward_batched(weight, bias, X, t);
    brotensor::bias_act_forward(t, /*b=*/nullptr, /*N=*/X.rows, /*C=*/out_features,
                                /*HW=*/1, /*act=*/1, /*alpha=*/0.2f,
                                /*gain=*/std::sqrt(2.0f), /*clamp=*/-1.0f, Y);
}

// ─── MappingNetwork ──────────────────────────────────────────────────────────

MappingNetwork::MappingNetwork(const Config& cfg) : cfg_(cfg) {
    if (cfg_.c_dim != 0)
        throw std::runtime_error(
            "stylegan3::MappingNetwork: conditional (c_dim>0) generation is not "
            "supported yet — only unconditional StyleGAN3-R models");
    fc_.resize(static_cast<std::size_t>(cfg_.map_num_layers));
}

void MappingNetwork::load(const void* file, const std::string& prefix) {
    const char* who = "stylegan3::MappingNetwork: ";
    for (int idx = 0; idx < cfg_.map_num_layers; ++idx) {
        const int in  = (idx == 0) ? cfg_.z_dim : cfg_.w_dim;
        const int out = cfg_.w_dim;
        fc_[static_cast<std::size_t>(idx)].load(
            file, who, prefix + ".fc" + std::to_string(idx), in, out,
            cfg_.lr_multiplier, /*with_bias=*/true, FullyConnectedLayer::Act::LRelu);
    }
    const st::File& f = *reinterpret_cast<const st::File*>(file);
    w_avg_ = brovisionml::detail::load_whole(f, who, prefix + ".w_avg", 1, cfg_.w_dim);
}

void MappingNetwork::to(brotensor::Device dev) {
    for (auto& fc : fc_) fc.to(dev);
    w_avg_  = w_avg_.to(dev);
    device_ = dev;
}

Tensor MappingNetwork::forward(const Tensor& z, float truncation_psi,
                               int truncation_cutoff) const {
    if (z.rows != 1 || z.cols != cfg_.z_dim)
        throw std::runtime_error(
            "stylegan3::MappingNetwork::forward: z must be (1, z_dim); batched "
            "mapping is not supported yet");

    // Migrate z onto the model's device if needed (mapping runs in FP32).
    Tensor z_local;
    const Tensor* zp = &z;
    if (z.device != device_) { z_local = z.to(device_); zp = &z_local; }

    // normalize_2nd_moment: x = z * rsqrt(mean(z^2) + 1e-8).
    Tensor x;
    brotensor::pixel_norm_forward(*zp, 1e-8f, x);

    // Mapping FCs (leaky-ReLU).
    for (const auto& fc : fc_) {
        Tensor y;
        fc.forward(x, y);
        x = std::move(y);
    }
    // x is now (1, w_dim).

    const int n_ws   = cfg_.num_ws();
    const int cutoff = (truncation_cutoff < 0)
                           ? n_ws
                           : std::min(truncation_cutoff, n_ws);
    const bool do_trunc = (truncation_psi != 1.0f);

    // w_trunc = lerp(w_avg, x, psi) = psi*x + (1-psi)*w_avg.
    Tensor w_trunc;
    if (do_trunc) {
        w_trunc = x.clone();
        brotensor::scale_inplace(w_trunc, truncation_psi);
        Tensor wavg = w_avg_.clone();
        brotensor::scale_inplace(wavg, 1.0f - truncation_psi);
        brotensor::add_inplace(w_trunc, wavg);
    }

    // Broadcast to (num_ws, w_dim): rows [0,cutoff) truncated, the rest raw.
    Tensor ws = Tensor::zeros_on(device_, n_ws, cfg_.w_dim);
    for (int r = 0; r < n_ws; ++r) {
        const Tensor& src = (do_trunc && r < cutoff) ? w_trunc : x;
        brotensor::copy_d2d(src, 0, ws, r * cfg_.w_dim, cfg_.w_dim);
    }
    return ws;
}

}  // namespace brovisionml::stylegan3
