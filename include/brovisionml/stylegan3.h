#pragma once

// StyleGAN3 (NVlabs "Alias-Free GAN", config-R "rotation equivariant") image
// generation — the first *generative* model family in brovisionml, and the
// first with a trainable / differentiable surface. It composes the StyleGAN3
// op primitives that live in brotensor (modulated_conv2d, upfirdn2d, bias_act,
// filtered_lrelu, the sin/cos Fourier elementwise, pixel_norm) into a runnable
// z -> w+ -> RGB generator.
//
// Mirrors NVlabs/stylegan3 `training/networks_stylegan3.py` so a converted
// checkpoint (see scripts/convert-stylegan3.py — pickle -> safetensors)
// reproduces the reference numerics. The synthesis path's per-layer low-pass
// filters are *designed* here (Kaiser/firwin + the radial jinc variant), not
// read from the checkpoint, exactly as the reference designs them at __init__.
//
// Config-R is the default and only preset family exposed: conv_kernel = 1,
// channel_base = 65536, channel_max = 1024, radially-symmetric down filters.
// (The config-T translation-equivariant variant is a parameter change away —
// conv_kernel = 3, channel_base = 32768, channel_max = 512, separable filters —
// but the released-weight target here is the -r models.)
//
// Pieces are split across translation units the way the reference splits
// classes: stylegan3_mapping.cpp (FullyConnectedLayer + MappingNetwork),
// stylegan3_filter.cpp (design_lowpass_filter), stylegan3_synthesis.cpp
// (SynthesisInput + SynthesisLayer + SynthesisNetwork), stylegan3.cpp (the
// Generator facade + checkpoint loading).

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brovisionml::stylegan3 {

// ─── Config ─────────────────────────────────────────────────────────────────
//
// One struct carries the whole generator's hyperparameters — the mapping and
// synthesis sub-networks read the fields they need. `r256/r512/r1024` are the
// released config-R presets; `num_ws()` is derived, not stored.
struct Config {
    // Latent dimensionalities.
    int z_dim = 512;   // input latent Z
    int c_dim = 0;     // conditioning label C (0 = unconditional)
    int w_dim = 512;   // intermediate latent W

    // Output.
    int img_resolution = 1024;
    int img_channels   = 3;

    // Mapping network.
    int   map_num_layers = 2;
    float lr_multiplier  = 0.01f;   // mapping FC learning-rate multiplier (baked at load)
    float w_avg_beta     = 0.998f;  // EMA decay (training only; inference reads the w_avg buffer)

    // Synthesis network.
    int   channel_base       = 65536;   // config-R: 32768 * 2
    int   channel_max        = 1024;    // config-R: 512 * 2
    int   num_layers         = 14;      // synthesis layers excluding Fourier input and ToRGB
    int   num_critical       = 2;       // critically-sampled layers at the end
    float first_cutoff       = 2.0f;    // f_{c,0}
    float first_stopband     = 4.2870938502f;  // 2^2.1
    float last_stopband_rel  = 1.2311444134f;  // 2^0.3
    int   margin_size        = 10;      // extra pixels outside the image on each side
    float output_scale       = 0.25f;
    int   num_fp16_res       = 4;       // top-N resolutions that run in FP16 on GPU

    // config-R vs config-T knobs.
    int   conv_kernel        = 1;       // R: 1, T: 3
    bool  use_radial_filters = true;    // R: true, T: false
    float conv_clamp         = 256.0f;  // post-activation clamp magnitude
    int   filter_size        = 6;       // base low-pass filter tap count
    int   lrelu_upsampling   = 2;       // internal up factor around the leaky ReLU

    // Number of intermediate latents W consumed (mapping output rows, and the
    // W+ row count the synthesis path indexes). Reference: num_layers + 2.
    int num_ws() const { return num_layers + 2; }

    static Config r256();
    static Config r512();
    static Config r1024();
};

namespace detail {

// design_lowpass_filter — the synthesis path's per-layer low-pass filters,
// designed (not loaded) exactly as NVlabs/stylegan3's SynthesisLayer designs
// them at construction. Returns a 2D (numtaps, numtaps) FP32 host tensor whose
// taps sum to 1 (unit DC gain):
//   * numtaps == 1            -> the 1x1 identity filter [[1]].
//   * radial == false         -> the separable Kaiser/firwin low-pass, returned
//                                as the outer product f1d (x) f1d (brotensor's
//                                upfirdn2d/filtered_lrelu take a 2D kernel).
//   * radial == true          -> the radially-symmetric jinc filter windowed by
//                                an outer Kaiser window (config-R down filters).
// cutoff/width/fs are in the layer's temporary sampling-rate units.
brotensor::Tensor design_lowpass_filter(int numtaps, double cutoff,
                                        double width, double fs, bool radial);

}  // namespace detail

// ─── FullyConnectedLayer ─────────────────────────────────────────────────────
//
// The reference's FullyConnectedLayer: y = act((W * weight_gain) x + b *
// bias_gain). The two gains (weight_gain = lr_multiplier / sqrt(in), bias_gain
// = lr_multiplier) are *baked into the stored tensors at load*, so forward() is
// a plain linear (+ optional leaky-ReLU). Used by the mapping FCs (lrelu) and
// every affine layer in synthesis (linear). Always FP32.
struct FullyConnectedLayer {
    enum class Act { Linear, LRelu };

    brotensor::Tensor weight;     // (out, in), gain baked in
    brotensor::Tensor bias;       // (out, 1), gain baked in; empty if no bias
    bool has_bias = true;
    Act  act      = Act::Linear;
    int  in_features  = 0;
    int  out_features = 0;

    // Load `prefix.weight` (out,in) and, if present, `prefix.bias` (out,) from a
    // safetensors file, baking weight_gain / bias_gain. `f` is a
    // brotensor::safetensors::File (forward-declared to keep it out of the
    // public header surface — see the .cpp).
    void load(const void* file, const std::string& who, const std::string& prefix,
              int in, int out, float lr_multiplier, bool with_bias, Act activation);

    void to(brotensor::Device dev);

    // Y (B, out) = act(W X^T + b), X (B, in). All FP32.
    void forward(const brotensor::Tensor& X, brotensor::Tensor& Y) const;
};

// ─── MappingNetwork ──────────────────────────────────────────────────────────
//
// z -> w+. Normalizes z to unit second moment (pixel_norm), runs map_num_layers
// leaky-ReLU FCs, broadcasts the result to num_ws rows, and applies truncation
// toward the stored w_avg. Always FP32 (the reference runs mapping in fp32 on
// every device).
class MappingNetwork {
public:
    explicit MappingNetwork(const Config& cfg);

    // Load fc0..fc{N-1} and the w_avg buffer from a checkpoint (prefix e.g.
    // "mapping"). `file` is a brotensor::safetensors::File*.
    void load(const void* file, const std::string& prefix);
    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }

    // z: (1, z_dim) FP32 -> ws: (num_ws, w_dim) FP32 on the model's device.
    // truncation_psi = 1 disables truncation; truncation_cutoff < 0 means all
    // num_ws rows are truncated (the reference default).
    brotensor::Tensor forward(const brotensor::Tensor& z,
                              float truncation_psi = 1.0f,
                              int truncation_cutoff = -1) const;

    int num_ws() const { return cfg_.num_ws(); }

private:
    Config cfg_;
    brotensor::Device device_ = brotensor::Device::CPU;
    std::vector<FullyConnectedLayer> fc_;
    brotensor::Tensor w_avg_;   // (1, w_dim) FP32
};

// ─── SynthesisInput ──────────────────────────────────────────────────────────
//
// The Fourier-feature input layer: maps w[0] to the first feature map via a
// learned affine that produces a per-sample rotation+translation, applied to a
// set of fixed random frequencies/phases, sampled on a grid as sin() features
// and projected by a learned channel mixing matrix. Mirrors
// networks_stylegan3.py SynthesisInput, run in FP32 (single sample, N=1).
//
// The small per-channel/grid math (transform, frequencies, amplitudes, the
// affine_grid, the sin features) is done on the host — the input map is always
// low resolution (~36x36) regardless of the output resolution — and only the
// final channel-mixing matmul (C_in x HW) runs on the device, producing an
// NCHW feature map directly.
class SynthesisInput {
public:
    SynthesisInput() = default;
    SynthesisInput(int w_dim, int channels, int size,
                   double sampling_rate, double bandwidth);

    void load(const void* file, const std::string& prefix);
    void to(brotensor::Device dev);

    // w: (1, w_dim) -> (1, channels*H*W) FP32 NCHW.
    brotensor::Tensor forward(const brotensor::Tensor& w) const;

    int channels() const { return channels_; }
    int out_h() const { return size_; }
    int out_w() const { return size_; }

    // Parameters (public for the lab's adapter/inspection surface). freqs,
    // phases, transform stay on the host (used in the host-side math); weight
    // and the affine layer migrate with to().
    FullyConnectedLayer affine;      // w_dim -> 4 (rotation cos/sin + translate)
    brotensor::Tensor   weight;      // (channels, channels), 1/sqrt(channels) baked
    brotensor::Tensor   freqs;       // (channels, 2) host
    brotensor::Tensor   phases;      // (channels, 1) host
    brotensor::Tensor   transform;   // (3, 3) host base transform

private:
    int    w_dim_         = 0;
    int    channels_      = 0;
    int    size_          = 0;
    double sampling_rate_ = 0.0;
    double bandwidth_     = 0.0;
    brotensor::Device device_ = brotensor::Device::CPU;
};

}  // namespace brovisionml::stylegan3
