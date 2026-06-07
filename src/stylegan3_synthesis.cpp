#include "brovisionml/stylegan3.h"

#include "weights_util.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace brovisionml::stylegan3 {

namespace st = brotensor::safetensors;
using brotensor::Tensor;

namespace {
constexpr double kPi = 3.14159265358979323846;

// Python-style floor division (rounds toward -inf), for the symmetric padding
// split that mirrors the reference's `(pad_total + up_factor) // 2`.
long floordiv(long a, long b) {
    long q = a / b, r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) --q;
    return q;
}
}  // namespace

// ─── SynthesisInput ──────────────────────────────────────────────────────────

SynthesisInput::SynthesisInput(int w_dim, int channels, int size,
                               double sampling_rate, double bandwidth)
    : w_dim_(w_dim), channels_(channels), size_(size),
      sampling_rate_(sampling_rate), bandwidth_(bandwidth) {}

void SynthesisInput::load(const void* file, const std::string& prefix) {
    const char* who = "stylegan3::SynthesisInput: ";
    const st::File& f = *reinterpret_cast<const st::File*>(file);

    // affine: w_dim -> 4, linear, default lr_multiplier (gains: 1/sqrt(w_dim), 1).
    affine.load(file, who, prefix + ".affine", w_dim_, 4, /*lr=*/1.0f,
                /*with_bias=*/true, FullyConnectedLayer::Act::Linear);

    // Channel-mixing weight (channels, channels), 1/sqrt(channels) baked in.
    weight = brovisionml::detail::load_whole(f, who, prefix + ".weight",
                                             channels_, channels_);
    const float wscale = 1.0f / std::sqrt(static_cast<float>(channels_));
    {
        float* w = weight.host_f32_mut();
        const std::size_t n = static_cast<std::size_t>(channels_) * channels_;
        for (std::size_t i = 0; i < n; ++i) w[i] *= wscale;
    }

    // Fixed buffers — kept on the host (used by the host-side feature math).
    freqs     = brovisionml::detail::load_whole(f, who, prefix + ".freqs", channels_, 2);
    phases    = brovisionml::detail::load_whole(f, who, prefix + ".phases", channels_, 1);
    transform = brovisionml::detail::load_whole(f, who, prefix + ".transform", 3, 3);
}

void SynthesisInput::to(brotensor::Device dev) {
    affine.to(dev);
    weight  = weight.to(dev);
    device_ = dev;
    // freqs / phases / transform stay on the host.
}

Tensor SynthesisInput::forward(const Tensor& w) const {
    if (w.rows != 1 || w.cols != w_dim_)
        throw std::runtime_error("stylegan3::SynthesisInput::forward: w must be (1, w_dim)");

    // 1. Affine -> t = [cos, sin, tx, ty]; normalize the rotation part.
    Tensor t_dev;
    affine.forward(w, t_dev);                       // (1, 4) on device
    std::vector<float> t = t_dev.to_host_vector();  // 4 floats
    {
        const double n = std::sqrt(static_cast<double>(t[0]) * t[0] +
                                   static_cast<double>(t[1]) * t[1]);
        for (float& v : t) v = static_cast<float>(v / n);
    }

    // 2. transforms = m_r @ m_t @ base_transform  (3x3, host).
    double m_r[3][3] = {{t[0], -t[1], 0}, {t[1], t[0], 0}, {0, 0, 1}};
    double m_t[3][3] = {{1, 0, -t[2]}, {0, 1, -t[3]}, {0, 0, 1}};
    const float* base = transform.host_f32();       // (3,3) row-major
    double rt[3][3] = {};                            // m_r @ m_t
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) rt[i][j] += m_r[i][k] * m_t[k][j];
    double T[3][3] = {};                             // (m_r @ m_t) @ base
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                T[i][j] += rt[i][k] * static_cast<double>(base[k * 3 + j]);

    const double A00 = T[0][0], A01 = T[0][1], A10 = T[1][0], A11 = T[1][1];
    const double tcol0 = T[0][2], tcol1 = T[1][2];

    // 3. Per-channel transformed freqs', phases', and amplitudes.
    const int C = channels_;
    const float* fr = freqs.host_f32();   // (C,2)
    const float* ph = phases.host_f32();  // (C,1)
    std::vector<double> fpx(C), fpy(C), phc(C), amp(C);
    const double half_minus_bw = sampling_rate_ / 2.0 - bandwidth_;
    for (int c = 0; c < C; ++c) {
        const double f0 = fr[c * 2 + 0], f1 = fr[c * 2 + 1];
        // freqs' = freqs0 @ A
        const double gx = f0 * A00 + f1 * A10;
        const double gy = f0 * A01 + f1 * A11;
        fpx[c] = gx;
        fpy[c] = gy;
        // phases' = phases0 + freqs0 . tcol
        phc[c] = static_cast<double>(ph[c]) + f0 * tcol0 + f1 * tcol1;
        // amplitudes from the transformed-freq magnitude
        const double mag = std::sqrt(gx * gx + gy * gy);
        double a = 1.0 - (mag - bandwidth_) / half_minus_bw;
        a = a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);
        amp[c] = a;
    }

    // 4. affine_grid (align_corners=False) with theta = diag(0.5*size/sr).
    const int H = size_, W = size_;
    const double s = 0.5 * static_cast<double>(size_) / sampling_rate_;
    std::vector<double> gxs(W), gys(H);
    for (int x = 0; x < W; ++x) gxs[x] = s * ((2.0 * x + 1.0) / W - 1.0);
    for (int y = 0; y < H; ++y) gys[y] = s * ((2.0 * y + 1.0) / H - 1.0);

    // 5. Fourier features, transposed to (C, HW) so the channel mix produces
    //    an NCHW map directly: Xs_T[c, y*W+x] = sin(2pi*(gx*fx+gy*fy+ph))*amp.
    const int HW = H * W;
    Tensor Xs_T = Tensor::mat(C, HW);   // host FP32
    float* xs = Xs_T.host_f32_mut();
    for (int c = 0; c < C; ++c) {
        const double fx = fpx[c], fy = fpy[c], pc = phc[c], ac = amp[c];
        float* row = xs + static_cast<std::size_t>(c) * HW;
        for (int y = 0; y < H; ++y) {
            const double gy = gys[y];
            for (int x = 0; x < W; ++x) {
                const double arg = gxs[x] * fx + gy * fy + pc;
                row[y * W + x] =
                    static_cast<float>(std::sin(arg * (2.0 * kPi)) * ac);
            }
        }
    }

    // 6. Channel mix: Y(C_out, HW) = weight(C_out, C_in) @ Xs_T(C_in, HW).
    Tensor xs_dev = (device_ == brotensor::Device::CPU) ? Xs_T : Xs_T.to(device_);
    Tensor Y;
    brotensor::matmul(weight, xs_dev, Y);            // (C, HW)

    // Relabel the contiguous (C, HW) buffer as (1, C*H*W) NCHW for downstream.
    Y.rows = 1;
    Y.cols = C * HW;
    return Y;
}

// ─── SynthesisLayer ──────────────────────────────────────────────────────────

SynthesisLayer::SynthesisLayer(const SynthesisLayerParams& p) {
    w_dim_        = p.w_dim;
    in_channels_  = p.in_channels;
    out_channels_ = p.out_channels;
    in_size_      = p.in_size;
    out_size_     = p.out_size;
    conv_kernel_  = p.conv_kernel;
    is_torgb_     = p.is_torgb;
    conv_clamp_   = p.conv_clamp;
    use_fp16_     = p.use_fp16;

    // Temporary sampling rate around the leaky-ReLU (no internal up for ToRGB).
    const double tmp_sr = std::max(p.in_sampling_rate, p.out_sampling_rate) *
                          (p.is_torgb ? 1.0 : static_cast<double>(p.lrelu_upsampling));

    // Upsampling filter.
    up_factor_ = static_cast<int>(std::lround(tmp_sr / p.in_sampling_rate));
    up_taps_   = (up_factor_ > 1 && !p.is_torgb) ? p.filter_size * up_factor_ : 1;
    up_filter  = detail::design_lowpass_filter(up_taps_, p.in_cutoff,
                                               p.in_half_width * 2.0, tmp_sr,
                                               /*radial=*/false);

    // Downsampling filter (radial for non-critical config-R layers).
    down_factor_ = static_cast<int>(std::lround(tmp_sr / p.out_sampling_rate));
    down_taps_   = (down_factor_ > 1 && !p.is_torgb) ? p.filter_size * down_factor_ : 1;
    const bool down_radial = p.use_radial_filters && !p.is_critically_sampled;
    down_filter  = detail::design_lowpass_filter(down_taps_, p.out_cutoff,
                                                 p.out_half_width * 2.0, tmp_sr,
                                                 down_radial);

    // Padding so the up -> conv -> down chain lands exactly on out_size.
    long pad_total = static_cast<long>(out_size_ - 1) * down_factor_ + 1;
    pad_total -= static_cast<long>(in_size_ + conv_kernel_ - 1) * up_factor_;
    pad_total += up_taps_ + down_taps_ - 2;
    const long lo = floordiv(pad_total + up_factor_, 2);
    pad_lo_ = static_cast<int>(lo);
    pad_hi_ = static_cast<int>(pad_total - lo);
}

void SynthesisLayer::load(const void* file, const std::string& prefix) {
    const char* who = "stylegan3::SynthesisLayer: ";
    const st::File& f = *reinterpret_cast<const st::File*>(file);

    affine.load(file, who, prefix + ".affine", w_dim_, in_channels_, /*lr=*/1.0f,
                /*with_bias=*/true, FullyConnectedLayer::Act::Linear);

    weight = brovisionml::detail::load_whole(
        f, who, prefix + ".weight", out_channels_,
        in_channels_ * conv_kernel_ * conv_kernel_);
    bias = brovisionml::detail::load_whole(f, who, prefix + ".bias", out_channels_, 1);

    // magnitude_ema (scalar) -> input_gain = 1/sqrt(magnitude_ema).
    Tensor mema = brovisionml::detail::load_whole(f, who, prefix + ".magnitude_ema", 1, 1);
    input_gain_ = 1.0f / std::sqrt(mema.host_f32()[0]);
}

void SynthesisLayer::to(brotensor::Device dev) {
    affine.to(dev);
    weight      = weight.to(dev);
    bias        = bias.to(dev);
    up_filter   = up_filter.to(dev);
    down_filter = down_filter.to(dev);
    device_     = dev;

    // Pre-cast the frozen params for the FP16 forward fast path (GPU only) so
    // forward() casts only the per-call activation + styles, never the weights.
    if (fp16_active()) {
        const auto h = brotensor::Dtype::FP16;
        brotensor::cast(weight, weight_h_, h);
        brotensor::cast(bias, bias_h_, h);
        brotensor::cast(up_filter, up_filter_h_, h);
        brotensor::cast(down_filter, down_filter_h_, h);
    } else {
        weight_h_ = bias_h_ = up_filter_h_ = down_filter_h_ = {};
    }
}

Tensor SynthesisLayer::forward(const Tensor& w, const Tensor& x) const {
    const bool fp16 = fp16_active();

    // Per-channel styles from w (the affine + ToRGB gain stay FP32; styles are
    // tiny and the demod is sensitive — only the conv/activation tensors go FP16).
    Tensor styles;
    affine.forward(w, styles);                 // (1, in_channels)
    if (is_torgb_) {
        const float wg = 1.0f / std::sqrt(static_cast<float>(in_channels_) *
                                          conv_kernel_ * conv_kernel_);
        brotensor::scale_inplace(styles, wg);
    }

    // FP16 fast path: route the modulated 1×1 conv to brotensor's WMMA
    // implicit-GEMM by feeding half activations/styles + the pre-cast half
    // weights. x arrives FP16 from a prior FP16 layer, or FP32 at the band edge
    // (cast once here). The frozen weight/bias/filters are the cached half copies.
    Tensor x_h, styles_h;
    const Tensor* xp = &x;
    const Tensor* sp = &styles;
    const Tensor* wp = &weight;
    const Tensor* bp = &bias;
    const Tensor* fup = &up_filter;
    const Tensor* fdp = &down_filter;
    if (fp16) {
        if (x.dtype != brotensor::Dtype::FP16) {
            brotensor::cast(x, x_h, brotensor::Dtype::FP16);
            xp = &x_h;
        }
        brotensor::cast(styles, styles_h, brotensor::Dtype::FP16);
        sp = &styles_h;
        wp = &weight_h_;  bp = &bias_h_;  fup = &up_filter_h_;  fdp = &down_filter_h_;
    }

    // Modulated convolution (demodulate off for ToRGB).
    const int pad = conv_kernel_ - 1;
    Tensor dcoef, yconv;
    brotensor::modulated_conv2d_forward(
        *xp, *wp, *sp, /*N=*/1, in_channels_, in_size_, in_size_,
        out_channels_, conv_kernel_, conv_kernel_, pad, pad,
        /*demodulate=*/!is_torgb_, /*eps=*/1e-8f, dcoef, yconv);

    // magnitude_ema input gain (a uniform scalar => post-scale the conv output).
    if (input_gain_ != 1.0f) brotensor::scale_inplace(yconv, input_gain_);

    // Filtered leaky-ReLU: bias -> upsample -> lrelu -> downsample.
    const int Hconv = in_size_ + (conv_kernel_ - 1);
    const float gain  = is_torgb_ ? 1.0f : std::sqrt(2.0f);
    const float slope = is_torgb_ ? 1.0f : 0.2f;
    Tensor up_buf, act_buf, out;
    brotensor::filtered_lrelu_forward(
        yconv, *fup, *fdp, bp, /*N=*/1, out_channels_,
        Hconv, Hconv, up_factor_, down_factor_, pad_lo_, pad_hi_, pad_lo_, pad_hi_,
        gain, slope, static_cast<float>(conv_clamp_), up_buf, act_buf, out);
    return out;  // (1, out_channels * out_size * out_size); FP16 when fp16_active
}

// ─── SynthesisNetwork ────────────────────────────────────────────────────────

SynthesisNetwork::SynthesisNetwork(const Config& cfg) : cfg_(cfg) {
    const int L = cfg_.num_layers;           // hidden synthesis layers
    const int N = L + 1;                      // including ToRGB
    const double last_cutoff   = cfg_.img_resolution / 2.0;
    const double last_stopband = last_cutoff * cfg_.last_stopband_rel;

    std::vector<double> cutoffs(N), stopbands(N), sampling_rates(N), half_widths(N);
    std::vector<int>    sizes(N), channels(N);
    for (int i = 0; i < N; ++i) {
        const double e = std::min(static_cast<double>(i) / (L - cfg_.num_critical), 1.0);
        cutoffs[i]   = cfg_.first_cutoff   * std::pow(last_cutoff   / cfg_.first_cutoff,   e);
        stopbands[i] = cfg_.first_stopband * std::pow(last_stopband / cfg_.first_stopband, e);
        sampling_rates[i] = std::exp2(std::ceil(std::log2(
            std::min(stopbands[i] * 2.0, static_cast<double>(cfg_.img_resolution)))));
        half_widths[i] = std::max(stopbands[i], sampling_rates[i] / 2.0) - cutoffs[i];
        sizes[i]    = static_cast<int>(sampling_rates[i]) + cfg_.margin_size * 2;
        channels[i] = static_cast<int>(std::nearbyint(  // np.rint: half-to-even
            std::min((cfg_.channel_base / 2.0) / cutoffs[i],
                     static_cast<double>(cfg_.channel_max))));
    }
    sizes[N - 1] = cfg_.img_resolution;          // sizes[-2:] = img_resolution
    if (N >= 2) sizes[N - 2] = cfg_.img_resolution;
    channels[N - 1] = cfg_.img_channels;         // channels[-1] = img_channels

    // Fourier-feature input (index 0).
    input_ = SynthesisInput(cfg_.w_dim, channels[0], sizes[0],
                            sampling_rates[0], cutoffs[0]);

    layers_.clear();
    layer_names_.clear();
    for (int idx = 0; idx < N; ++idx) {
        const int prev = std::max(idx - 1, 0);
        SynthesisLayerParams p;
        p.w_dim                 = cfg_.w_dim;
        p.is_torgb              = (idx == L);
        p.is_critically_sampled = (idx >= L - cfg_.num_critical);
        p.use_fp16              = !cfg_.force_fp32 &&
                                  (sampling_rates[idx] * std::pow(2.0, cfg_.num_fp16_res)
                                   > cfg_.img_resolution);
        p.in_channels           = channels[prev];
        p.out_channels          = channels[idx];
        p.in_size               = sizes[prev];
        p.out_size              = sizes[idx];
        p.in_sampling_rate      = sampling_rates[prev];
        p.out_sampling_rate     = sampling_rates[idx];
        p.in_cutoff             = cutoffs[prev];
        p.out_cutoff            = cutoffs[idx];
        p.in_half_width         = half_widths[prev];
        p.out_half_width        = half_widths[idx];
        // ToRGB always uses a 1x1 conv regardless of config (matters for -T).
        p.conv_kernel           = p.is_torgb ? 1 : cfg_.conv_kernel;
        p.filter_size           = cfg_.filter_size;
        p.lrelu_upsampling      = cfg_.lrelu_upsampling;
        p.use_radial_filters    = cfg_.use_radial_filters;
        p.conv_clamp            = cfg_.conv_clamp;
        layers_.emplace_back(p);
        layer_names_.push_back("L" + std::to_string(idx) + "_" +
                               std::to_string(sizes[idx]) + "_" +
                               std::to_string(channels[idx]));
    }
}

void SynthesisNetwork::load(const void* file, const std::string& prefix) {
    input_.load(file, prefix + ".input");
    for (std::size_t i = 0; i < layers_.size(); ++i)
        layers_[i].load(file, prefix + "." + layer_names_[i]);
}

void SynthesisNetwork::to(brotensor::Device dev) {
    input_.to(dev);
    for (auto& l : layers_) l.to(dev);
    device_ = dev;
}

Tensor SynthesisNetwork::forward(const Tensor& ws) const {
    if (ws.rows != cfg_.num_ws() || ws.cols != cfg_.w_dim)
        throw std::runtime_error("stylegan3::SynthesisNetwork::forward: ws must be (num_ws, w_dim)");

    // Non-owning (1, w_dim) view onto row k of ws.
    auto row = [&](int k) {
        return Tensor::view(
            ws.device,
            static_cast<char*>(ws.data) +
                static_cast<std::size_t>(k) * cfg_.w_dim * sizeof(float),
            1, cfg_.w_dim, brotensor::Dtype::FP32);
    };

    Tensor x = input_.forward(row(0));
    for (std::size_t i = 0; i < layers_.size(); ++i)
        x = layers_[i].forward(row(static_cast<int>(i) + 1), x);

    // The FP16 fast path leaves the final (ToRGB) output in FP16 — bring it back
    // to FP32 before the output scale so callers (to_image, parity) see FP32.
    if (x.dtype != brotensor::Dtype::FP32) {
        Tensor xf;
        brotensor::cast(x, xf, brotensor::Dtype::FP32);
        x = std::move(xf);
    }
    if (cfg_.output_scale != 1.0f) brotensor::scale_inplace(x, cfg_.output_scale);
    return x;  // (1, img_channels * res * res)
}

// ═══ Backward (inversion support) ════════════════════════════════════════════
//
// Manual backprop through the synthesis path w.r.t. the W+ rows. The op-level
// backwards live in brotensor (modulated_conv2d / filtered_lrelu / bias_act /
// upfirdn2d, all CPU+CUDA); the only host-side gradient is SynthesisInput's
// Fourier/affine trig, differentiated here by hand to mirror its forward.
// Weights and biases are frozen — only latent gradients are produced.

// ─── SynthesisInput backward ──────────────────────────────────────────────────

Tensor SynthesisInput::forward_cached(const Tensor& w, Cache& cache) const {
    // Identical math to forward(), additionally capturing t_raw (pre-rotation-
    // norm affine output) and the device-side sin-feature map for the backward.
    if (w.rows != 1 || w.cols != w_dim_)
        throw std::runtime_error("stylegan3::SynthesisInput::forward_cached: w must be (1, w_dim)");

    Tensor t_dev;
    affine.forward(w, t_dev);
    cache.t_raw = t_dev.to_host_vector();           // 4 floats, pre-normalization
    std::vector<float> t = cache.t_raw;
    {
        const double n = std::sqrt(static_cast<double>(t[0]) * t[0] +
                                   static_cast<double>(t[1]) * t[1]);
        for (float& v : t) v = static_cast<float>(v / n);
    }

    double m_r[3][3] = {{t[0], -t[1], 0}, {t[1], t[0], 0}, {0, 0, 1}};
    double m_t[3][3] = {{1, 0, -t[2]}, {0, 1, -t[3]}, {0, 0, 1}};
    const float* base = transform.host_f32();
    double rt[3][3] = {}, T[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) rt[i][j] += m_r[i][k] * m_t[k][j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                T[i][j] += rt[i][k] * static_cast<double>(base[k * 3 + j]);

    const double A00 = T[0][0], A01 = T[0][1], A10 = T[1][0], A11 = T[1][1];
    const double tcol0 = T[0][2], tcol1 = T[1][2];

    const int C = channels_;
    const float* fr = freqs.host_f32();
    const float* ph = phases.host_f32();
    std::vector<double> fpx(C), fpy(C), phc(C), amp(C);
    const double half_minus_bw = sampling_rate_ / 2.0 - bandwidth_;
    for (int c = 0; c < C; ++c) {
        const double f0 = fr[c * 2 + 0], f1 = fr[c * 2 + 1];
        const double gx = f0 * A00 + f1 * A10;
        const double gy = f0 * A01 + f1 * A11;
        fpx[c] = gx;
        fpy[c] = gy;
        phc[c] = static_cast<double>(ph[c]) + f0 * tcol0 + f1 * tcol1;
        const double mag = std::sqrt(gx * gx + gy * gy);
        double a = 1.0 - (mag - bandwidth_) / half_minus_bw;
        a = a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);
        amp[c] = a;
    }

    const int H = size_, W = size_;
    const double sc = 0.5 * static_cast<double>(size_) / sampling_rate_;
    std::vector<double> gxs(W), gys(H);
    for (int x = 0; x < W; ++x) gxs[x] = sc * ((2.0 * x + 1.0) / W - 1.0);
    for (int y = 0; y < H; ++y) gys[y] = sc * ((2.0 * y + 1.0) / H - 1.0);

    const int HW = H * W;
    Tensor Xs_T = Tensor::mat(C, HW);
    float* xs = Xs_T.host_f32_mut();
    for (int c = 0; c < C; ++c) {
        const double fx = fpx[c], fy = fpy[c], pc = phc[c], ac = amp[c];
        float* rowp = xs + static_cast<std::size_t>(c) * HW;
        for (int y = 0; y < H; ++y) {
            const double gyv = gys[y];
            for (int x = 0; x < W; ++x) {
                const double arg = gxs[x] * fx + gyv * fy + pc;
                rowp[y * W + x] =
                    static_cast<float>(std::sin(arg * (2.0 * kPi)) * ac);
            }
        }
    }

    cache.Xs_T = (device_ == brotensor::Device::CPU) ? Xs_T : Xs_T.to(device_);
    Tensor Y;
    brotensor::matmul(weight, cache.Xs_T, Y);        // (C, HW)
    Y.rows = 1;
    Y.cols = C * HW;
    return Y;
}

void SynthesisInput::backward(const Tensor& w, const Cache& cache,
                              const Tensor& dY, Tensor& dw0) const {
    const int C = channels_;
    const int H = size_, W = size_, HW = H * W;

    // 1. Channel mix: Y = weight @ Xs_T  ⇒  dXs_T = weightᵀ @ dY  (matmul bwd's dB).
    Tensor dY_mat = Tensor::view(dY.device, dY.data, C, HW, brotensor::Dtype::FP32);
    Tensor dA_dummy = Tensor::zeros_on(device_, C, C);
    Tensor dXs_T = Tensor::zeros_on(device_, C, HW);
    brotensor::matmul_backward(weight, cache.Xs_T, dY_mat, dA_dummy, dXs_T);
    const std::vector<float> g = dXs_T.to_host_vector();   // (C*HW) host

    // Rebuild the transform exactly as the forward (from the cached pre-norm t).
    std::vector<float> t = cache.t_raw;
    const double nrm = std::sqrt(static_cast<double>(t[0]) * t[0] +
                                 static_cast<double>(t[1]) * t[1]);
    for (float& v : t) v = static_cast<float>(v / nrm);
    double m_r[3][3] = {{t[0], -t[1], 0}, {t[1], t[0], 0}, {0, 0, 1}};
    double m_t[3][3] = {{1, 0, -t[2]}, {0, 1, -t[3]}, {0, 0, 1}};
    const float* base = transform.host_f32();
    double rt[3][3] = {}, T[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) rt[i][j] += m_r[i][k] * m_t[k][j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                T[i][j] += rt[i][k] * static_cast<double>(base[k * 3 + j]);
    const double A00 = T[0][0], A01 = T[0][1], A10 = T[1][0], A11 = T[1][1];
    const double tcol0 = T[0][2], tcol1 = T[1][2];

    const double sc = 0.5 * static_cast<double>(size_) / sampling_rate_;
    std::vector<double> gxs(W), gys(H);
    for (int x = 0; x < W; ++x) gxs[x] = sc * ((2.0 * x + 1.0) / W - 1.0);
    for (int y = 0; y < H; ++y) gys[y] = sc * ((2.0 * y + 1.0) / H - 1.0);

    const float* fr = freqs.host_f32();
    const float* ph = phases.host_f32();
    const double half_minus_bw = sampling_rate_ / 2.0 - bandwidth_;
    const double TWO_PI = 2.0 * kPi;

    // Accumulate gradients into the transform's used entries.
    double dA00 = 0, dA01 = 0, dA10 = 0, dA11 = 0, dtcol0 = 0, dtcol1 = 0;
    for (int c = 0; c < C; ++c) {
        const double f0 = fr[c * 2 + 0], f1 = fr[c * 2 + 1];
        const double fpx = f0 * A00 + f1 * A10;
        const double fpy = f0 * A01 + f1 * A11;
        const double pc  = static_cast<double>(ph[c]) + f0 * tcol0 + f1 * tcol1;
        const double mag = std::sqrt(fpx * fpx + fpy * fpy);
        double a = 1.0 - (mag - bandwidth_) / half_minus_bw;
        const bool clamped = (a < 0.0) || (a > 1.0);
        a = a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);

        const float* grow = g.data() + static_cast<std::size_t>(c) * HW;
        double dfpx = 0, dfpy = 0, dpc = 0, dAmp = 0;
        for (int y = 0; y < H; ++y) {
            const double gyv = gys[y];
            for (int x = 0; x < W; ++x) {
                const double arg = gxs[x] * fpx + gyv * fpy + pc;
                const double ang = arg * TWO_PI;
                const double gg = grow[y * W + x];     // dL/dXs[c,p]
                dAmp += gg * std::sin(ang);            // Xs = a*sin(ang)
                const double h = gg * a * std::cos(ang) * TWO_PI;  // dL/d(arg)
                dfpx += h * gxs[x];
                dfpy += h * gyv;
                dpc  += h;
            }
        }
        // amplitude pulls on the transformed-freq magnitude (dead when clamped).
        if (!clamped && mag > 0.0) {
            const double da_dmag = -1.0 / half_minus_bw;
            dfpx += dAmp * da_dmag * (fpx / mag);
            dfpy += dAmp * da_dmag * (fpy / mag);
        }
        // fpx = f0·A00 + f1·A10 ; fpy = f0·A01 + f1·A11 ; pc = … + f0·tcol0 + f1·tcol1
        dA00 += dfpx * f0; dA10 += dfpx * f1;
        dA01 += dfpy * f0; dA11 += dfpy * f1;
        dtcol0 += dpc * f0; dtcol1 += dpc * f1;
    }

    // Through T = (m_r @ m_t) @ base  (base constant).
    double dT[3][3] = {};
    dT[0][0] = dA00; dT[0][1] = dA01; dT[0][2] = dtcol0;
    dT[1][0] = dA10; dT[1][1] = dA11; dT[1][2] = dtcol1;
    double drt[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                drt[i][k] += dT[i][j] * static_cast<double>(base[k * 3 + j]);
    double dm_r[3][3] = {}, dm_t[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int p = 0; p < 3; ++p)
            for (int k = 0; k < 3; ++k) dm_r[i][p] += drt[i][k] * m_t[p][k];
    for (int p = 0; p < 3; ++p)
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < 3; ++i) dm_t[p][k] += drt[i][k] * m_r[i][p];

    // m_r/m_t entries back to the normalized affine 4-vector.
    double dt_n[4];
    dt_n[0] = dm_r[0][0] + dm_r[1][1];
    dt_n[1] = -dm_r[0][1] + dm_r[1][0];
    dt_n[2] = -dm_t[0][2];
    dt_n[3] = -dm_t[1][2];

    // Through the rotation normalization t_n[i] = t_raw[i] / nrm.
    const double tr0 = cache.t_raw[0], tr1 = cache.t_raw[1];
    double dL_dn = 0;
    for (int i = 0; i < 4; ++i) dL_dn += dt_n[i] * (-cache.t_raw[i] / (nrm * nrm));
    double dt_r[4];
    for (int i = 0; i < 4; ++i) dt_r[i] = dt_n[i] / nrm;
    dt_r[0] += dL_dn * (tr0 / nrm);
    dt_r[1] += dL_dn * (tr1 / nrm);

    // Affine: t_raw = w·affineᵀ + b  ⇒  dw0 = dt_r @ affine.weight.
    Tensor dtr = Tensor::mat(1, 4);
    for (int i = 0; i < 4; ++i) dtr.host_f32_mut()[i] = static_cast<float>(dt_r[i]);
    Tensor dtr_dev = (device_ == brotensor::Device::CPU) ? dtr : dtr.to(device_);
    brotensor::matmul(dtr_dev, affine.weight, dw0);   // (1,4)·(4,w_dim) = (1,w_dim)
}

// ─── SynthesisLayer backward ──────────────────────────────────────────────────

Tensor SynthesisLayer::forward_cached(const Tensor& w, const Tensor& x,
                                      Cache& cache) const {
    Tensor styles;
    affine.forward(w, styles);
    if (is_torgb_) {
        const float wg = 1.0f / std::sqrt(static_cast<float>(in_channels_) *
                                          conv_kernel_ * conv_kernel_);
        brotensor::scale_inplace(styles, wg);
    }
    cache.styles = styles;                       // post-(ToRGB) scale

    const int pad = conv_kernel_ - 1;
    Tensor dcoef, yconv;
    brotensor::modulated_conv2d_forward(
        x, weight, styles, /*N=*/1, in_channels_, in_size_, in_size_,
        out_channels_, conv_kernel_, conv_kernel_, pad, pad,
        /*demodulate=*/!is_torgb_, /*eps=*/1e-8f, dcoef, yconv);
    cache.dcoef = dcoef;

    if (input_gain_ != 1.0f) brotensor::scale_inplace(yconv, input_gain_);
    cache.yconv_scaled = yconv;                  // input to the filtered lrelu

    const int Hconv = in_size_ + (conv_kernel_ - 1);
    const float gain  = is_torgb_ ? 1.0f : std::sqrt(2.0f);
    const float slope = is_torgb_ ? 1.0f : 0.2f;
    Tensor up_buf, act_buf, out;
    brotensor::filtered_lrelu_forward(
        yconv, up_filter, down_filter, &bias, /*N=*/1, out_channels_,
        Hconv, Hconv, up_factor_, down_factor_, pad_lo_, pad_hi_, pad_lo_, pad_hi_,
        gain, slope, static_cast<float>(conv_clamp_), up_buf, act_buf, out);
    cache.up_buf = up_buf;
    return out;
}

void SynthesisLayer::backward(const Tensor& w, const Tensor& x, const Cache& cache,
                              const Tensor& dY, Tensor& dx, Tensor& dw_row) const {
    const int Hconv = in_size_ + (conv_kernel_ - 1);
    const float gain  = is_torgb_ ? 1.0f : std::sqrt(2.0f);
    const float slope = is_torgb_ ? 1.0f : 0.2f;

    // 1. Through the filtered leaky-ReLU (bias frozen → no dB).
    Tensor d_yconv;
    brotensor::filtered_lrelu_backward(
        dY, cache.yconv_scaled, up_filter, down_filter, &bias, /*N=*/1,
        out_channels_, Hconv, Hconv, up_factor_, down_factor_,
        pad_lo_, pad_hi_, pad_lo_, pad_hi_, gain, slope,
        static_cast<float>(conv_clamp_), cache.up_buf, d_yconv, /*dB=*/nullptr);

    // 2. Undo the magnitude_ema input gain.
    if (input_gain_ != 1.0f) brotensor::scale_inplace(d_yconv, input_gain_);

    // 3. Through the modulated conv → dx (to prev layer) and ds (style grad).
    //    Weights are frozen: dW goes to a scratch buffer the op accumulates into.
    const int pad = conv_kernel_ - 1;
    Tensor ds;
    Tensor dW_scratch = Tensor::zeros_on(weight.device, weight.rows, weight.cols);
    brotensor::modulated_conv2d_backward(
        x, weight, cache.styles, cache.dcoef, d_yconv, /*N=*/1, in_channels_,
        in_size_, in_size_, out_channels_, conv_kernel_, conv_kernel_, pad, pad,
        /*demodulate=*/!is_torgb_, /*eps=*/1e-8f, dx, dW_scratch, ds);

    // 4. Undo the ToRGB style gain.
    if (is_torgb_) {
        const float wg = 1.0f / std::sqrt(static_cast<float>(in_channels_) *
                                          conv_kernel_ * conv_kernel_);
        brotensor::scale_inplace(ds, wg);
    }

    // 5. Affine backward: dw_row = ds @ affine.weight  (affine is linear here).
    brotensor::matmul(ds, affine.weight, dw_row);
}

// ─── SynthesisNetwork backward ────────────────────────────────────────────────

Tensor SynthesisNetwork::forward_cached(const Tensor& ws, Cache& cache) const {
    if (ws.rows != cfg_.num_ws() || ws.cols != cfg_.w_dim)
        throw std::runtime_error("stylegan3::SynthesisNetwork::forward_cached: ws must be (num_ws, w_dim)");

    auto row = [&](int k) {
        return Tensor::view(
            ws.device,
            static_cast<char*>(ws.data) +
                static_cast<std::size_t>(k) * cfg_.w_dim * sizeof(float),
            1, cfg_.w_dim, brotensor::Dtype::FP32);
    };

    cache.layers.resize(layers_.size());
    cache.layer_in.resize(layers_.size());

    Tensor x = input_.forward_cached(row(0), cache.input);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        cache.layer_in[i] = x;                   // input fed to layer i
        x = layers_[i].forward_cached(row(static_cast<int>(i) + 1), x, cache.layers[i]);
    }
    if (cfg_.output_scale != 1.0f) brotensor::scale_inplace(x, cfg_.output_scale);
    return x;
}

void SynthesisNetwork::backward(const Tensor& ws, const Cache& cache,
                                const Tensor& dimg, Tensor& dws) const {
    auto row = [&](int k) {
        return Tensor::view(
            ws.device,
            static_cast<char*>(ws.data) +
                static_cast<std::size_t>(k) * cfg_.w_dim * sizeof(float),
            1, cfg_.w_dim, brotensor::Dtype::FP32);
    };

    dws = Tensor::zeros_on(ws.device, cfg_.num_ws(), cfg_.w_dim);

    Tensor dx = dimg.clone();
    if (cfg_.output_scale != 1.0f) brotensor::scale_inplace(dx, cfg_.output_scale);

    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        Tensor dx_prev, dw_row;
        layers_[i].backward(row(i + 1), cache.layer_in[i], cache.layers[i],
                            dx, dx_prev, dw_row);
        brotensor::copy_d2d(dw_row, 0, dws, (i + 1) * cfg_.w_dim, cfg_.w_dim);
        dx = std::move(dx_prev);
    }

    Tensor dw0;
    input_.backward(row(0), cache.input, dx, dw0);
    brotensor::copy_d2d(dw0, 0, dws, 0, cfg_.w_dim);
}

}  // namespace brovisionml::stylegan3
