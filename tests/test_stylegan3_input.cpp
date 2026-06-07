// StyleGAN3 SynthesisInput test.
//
// The Fourier-feature input is the trickiest layer (affine -> rotation/
// translation transform -> transformed freqs/phases -> affine_grid -> sin
// features -> channel mix -> NCHW). With no checkpoint available we drive it
// with hand-set parameters that reduce the transform to identity, so the whole
// pipeline collapses to a closed form we can check exactly:
//
//   identity affine (weight 0, bias [1,0,0,0]) and identity base transform =>
//   freqs' = freqs, phases' = phases, amplitudes = 1 (cutoffs chosen so the
//   clamp saturates), and an identity channel-mix => the output equals the raw
//   sin features sampled on the align_corners=False grid.
//
// This validates the transform algebra, the grid construction, the sin-feature
// math, and the (C,HW)->NCHW relabeling together. Exact checkpoint parity is
// covered later by the end-to-end weights test.

#include "brovisionml/stylegan3.h"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}
bool approx(float a, float b, float tol = 1e-3f) { return std::fabs(a - b) <= tol; }

using brotensor::Tensor;
using brovisionml::stylegan3::FullyConnectedLayer;
using brovisionml::stylegan3::SynthesisInput;

}  // namespace

int main() {
    const int w_dim = 4, C = 2, size = 2;
    const double sr = 4.0, bw = 1.0;  // half_minus_bw = 1; amplitudes saturate to 1
    SynthesisInput in(w_dim, C, size, sr, bw);

    // Identity affine: weight 0, bias [1,0,0,0] -> t = [1,0,0,0] for any w.
    in.affine.in_features = w_dim;
    in.affine.out_features = 4;
    in.affine.has_bias = true;
    in.affine.act = FullyConnectedLayer::Act::Linear;
    in.affine.weight = Tensor::mat(4, w_dim);  // zeros
    in.affine.bias = Tensor::vec(4);
    in.affine.bias[0] = 1.0f;

    // Identity channel mix (already represents weight/sqrt(C)).
    in.weight = Tensor::mat(C, C);
    in.weight(0, 0) = 1.0f; in.weight(1, 1) = 1.0f;

    // freqs = [[0.5,0],[0,0.5]], phases = 0, base transform = I.
    in.freqs = Tensor::mat(C, 2);
    in.freqs(0, 0) = 0.5f; in.freqs(1, 1) = 0.5f;
    in.phases = Tensor::mat(C, 1);  // zeros
    in.transform = Tensor::mat(3, 3);
    in.transform(0, 0) = 1.0f; in.transform(1, 1) = 1.0f; in.transform(2, 2) = 1.0f;

    Tensor w = Tensor::mat(1, w_dim);  // any value
    Tensor y = in.forward(w);
    check(y.rows == 1 && y.cols == C * size * size, "input output shape (1, C*H*W)");

    // Expected: grid scale s = 0.5*size/sr = 0.25; gx in {-0.125,+0.125}.
    // channel 0 (fx=0.5): val = sin(2pi * gx*0.5), independent of y.
    // channel 1 (fy=0.5): val = sin(2pi * gy*0.5), independent of x.
    const double pi = 3.14159265358979323846;
    const float p = static_cast<float>(std::sin(2.0 * pi * 0.0625));      // +0.38268
    const float m = -p;
    const float* d = y.host_f32();
    // NCHW flat: [ch0: (y0x0,y0x1,y1x0,y1x1), ch1: (...)]
    check(approx(d[0], m) && approx(d[1], p) && approx(d[2], m) && approx(d[3], p),
          "channel 0 = sin over x");
    check(approx(d[4], m) && approx(d[5], m) && approx(d[6], p) && approx(d[7], p),
          "channel 1 = sin over y");

    if (failures == 0) { std::printf("test_stylegan3_input: OK\n"); return 0; }
    std::printf("test_stylegan3_input: %d failure(s)\n", failures);
    return 1;
}
