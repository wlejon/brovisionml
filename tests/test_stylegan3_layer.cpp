// StyleGAN3 SynthesisLayer test.
//
// Two parts, both checkpoint-free:
//   * Sampling-schedule derivation — the bug-prone __init__ math. For a known
//     (in/out sampling rate, size, kernel) we assert up/down factors, filter
//     tap counts, and the symmetric padding split match hand-computed values,
//     for both a normal layer and a ToRGB layer.
//   * Forward shape + finiteness — with hand-set affine/weight/bias the layer
//     runs the affine -> modulated_conv2d -> filtered_lrelu chain end to end on
//     CPU and must emit exactly (1, out_channels*out_size*out_size), finite.
//     (The padding formula is algebraically constructed so the up->conv->down
//     chain always lands on out_size; this exercises that wiring.)
// Exact checkpoint parity is covered later by the end-to-end weights test.

#include "brovisionml/stylegan3.h"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

using brotensor::Tensor;
using brovisionml::stylegan3::FullyConnectedLayer;
using brovisionml::stylegan3::SynthesisLayer;
using brovisionml::stylegan3::SynthesisLayerParams;

void derivation_checks() {
    // Normal layer: in_sr=16, out_sr=32, in_size=36, out_size=52, k=1.
    // tmp = 32*2 = 64; up = 64/16 = 4 (taps 24); down = 64/32 = 2 (taps 12).
    // pad_total = 51*2+1 - 36*4 + 24+12-2 = 103-144+34 = -7;
    // pad_lo = floor((-7+4)/2) = -2; pad_hi = -7-(-2) = -5.
    {
        SynthesisLayerParams p;
        p.in_sampling_rate = 16; p.out_sampling_rate = 32;
        p.in_size = 36; p.out_size = 52; p.conv_kernel = 1;
        p.in_channels = 8; p.out_channels = 8;
        p.in_cutoff = 4; p.out_cutoff = 8; p.in_half_width = 4; p.out_half_width = 8;
        SynthesisLayer L(p);
        check(L.up_factor() == 4, "up_factor 4");
        check(L.down_factor() == 2, "down_factor 2");
        check(L.up_taps() == 24, "up_taps 24");
        check(L.down_taps() == 12, "down_taps 12");
        check(L.pad_lo() == -2, "pad_lo -2");
        check(L.pad_hi() == -5, "pad_hi -5");
    }
    // ToRGB: in_sr=out_sr=64, in_size=out_size=52, k=1.
    // tmp = 64*1 = 64; up=down=1, taps 1; pad_total = 0 => pad_lo=pad_hi=0.
    {
        SynthesisLayerParams p;
        p.is_torgb = true; p.is_critically_sampled = true;
        p.in_sampling_rate = 64; p.out_sampling_rate = 64;
        p.in_size = 52; p.out_size = 52; p.conv_kernel = 1;
        p.in_channels = 8; p.out_channels = 3;
        p.in_cutoff = 32; p.out_cutoff = 32; p.in_half_width = 4; p.out_half_width = 4;
        SynthesisLayer L(p);
        check(L.up_factor() == 1 && L.down_factor() == 1, "ToRGB up/down 1");
        check(L.up_taps() == 1 && L.down_taps() == 1, "ToRGB taps 1");
        check(L.pad_lo() == 0 && L.pad_hi() == 0, "ToRGB no padding");
    }
}

void forward_shape_check() {
    SynthesisLayerParams p;
    p.w_dim = 8; p.in_channels = 4; p.out_channels = 4;
    p.in_size = 16; p.out_size = 16; p.conv_kernel = 1;
    p.in_sampling_rate = 16; p.out_sampling_rate = 16;   // tmp=32, up=down=2, taps 12
    p.in_cutoff = 4; p.out_cutoff = 4; p.in_half_width = 4; p.out_half_width = 4;
    SynthesisLayer L(p);

    // affine: styles = 1 for all in_channels (weight 0, bias 1).
    L.affine.in_features = p.w_dim;
    L.affine.out_features = p.in_channels;
    L.affine.has_bias = true;
    L.affine.act = FullyConnectedLayer::Act::Linear;
    L.affine.weight = Tensor::mat(p.in_channels, p.w_dim);  // zeros
    L.affine.bias = Tensor::vec(p.in_channels);
    for (int i = 0; i < p.in_channels; ++i) L.affine.bias[i] = 1.0f;

    // conv weight (out, in*k*k) and zero bias.
    L.weight = Tensor::mat(p.out_channels, p.in_channels);
    for (int i = 0; i < L.weight.size(); ++i) L.weight[i] = 0.1f;
    L.bias = Tensor::vec(p.out_channels);

    Tensor w = Tensor::mat(1, p.w_dim);
    Tensor x = Tensor::mat(1, p.in_channels * p.in_size * p.in_size);
    for (int i = 0; i < x.size(); ++i)
        x[i] = std::sin(0.01f * static_cast<float>(i));

    Tensor y = L.forward(w, x);
    check(y.rows == 1 && y.cols == p.out_channels * p.out_size * p.out_size,
          "layer output shape (1, C_out*out*out)");
    bool finite = true;
    for (int i = 0; i < y.size(); ++i)
        if (!std::isfinite(y[i])) finite = false;
    check(finite, "layer output finite");
}

}  // namespace

int main() {
    derivation_checks();
    forward_shape_check();
    if (failures == 0) { std::printf("test_stylegan3_layer: OK\n"); return 0; }
    std::printf("test_stylegan3_layer: %d failure(s)\n", failures);
    return 1;
}
