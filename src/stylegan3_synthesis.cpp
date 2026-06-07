#include "brovisionml/stylegan3.h"

#include "weights_util.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace brovisionml::stylegan3 {

namespace st = brotensor::safetensors;
using brotensor::Tensor;

namespace {
constexpr double kPi = 3.14159265358979323846;
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

}  // namespace brovisionml::stylegan3
