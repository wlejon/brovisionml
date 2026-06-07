// StyleGAN3 low-pass filter design test.
//
// design_lowpass_filter is the numerically delicate part of the synthesis path
// (Kaiser/firwin separable filters and the radial jinc filter), designed in C++
// to match NVlabs/stylegan3. Exact bit-parity against scipy is covered later by
// the weights-gated end-to-end test; here we assert the structural invariants
// that catch gross design bugs:
//   * numtaps == 1 -> the 1x1 identity [[1]].
//   * every filter sums to ~1 (unit DC gain) and is square (numtaps,numtaps).
//   * separable filters are an outer product: f(i,j) == f1d(i)*f1d(j), so they
//     are symmetric and rank-1 (f(i,j)*f(k,l) == f(i,l)*f(k,j)).
//   * radial filters are 4-fold symmetric (f(i,j)==f(j,i)==f(n-1-i,j)).
//   * the center tap is the largest (a low-pass peak at DC).

#include "brovisionml/stylegan3.h"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

using brotensor::Tensor;
using brovisionml::stylegan3::detail::design_lowpass_filter;

float sum_of(const Tensor& f) {
    float s = 0.0f;
    for (int i = 0; i < f.size(); ++i) s += f[i];
    return s;
}

bool center_is_max(const Tensor& f) {
    const int n = f.rows;
    float peak = f(n / 2, n / 2);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (f(i, j) > peak + 1e-6f) return false;
    return true;
}

}  // namespace

int main() {
    // Identity.
    {
        Tensor f = design_lowpass_filter(1, 2.0, 2.0, 16.0, false);
        check(f.rows == 1 && f.cols == 1 && std::fabs(f(0, 0) - 1.0f) < 1e-6f,
              "numtaps==1 identity");
    }

    // Separable Kaiser/firwin (an up filter: filter_size*up_factor = 6*2 = 12).
    {
        const int n = 12;
        Tensor f = design_lowpass_filter(n, /*cutoff=*/2.0, /*width=*/4.0,
                                         /*fs=*/16.0, /*radial=*/false);
        check(f.rows == n && f.cols == n, "separable shape");
        check(std::fabs(sum_of(f) - 1.0f) < 1e-4f, "separable unit DC gain");
        bool symm = true, rank1 = true;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (std::fabs(f(i, j) - f(j, i)) > 1e-6f) symm = false;
                // rank-1: f(i,j)*f(k,l) == f(i,l)*f(k,j)
                if (std::fabs(f(i, j) * f(0, 0) - f(i, 0) * f(0, j)) > 1e-6f)
                    rank1 = false;
            }
        check(symm, "separable symmetric");
        check(rank1, "separable rank-1 (outer product)");
        check(center_is_max(f), "separable peak at center");
    }

    // Radial jinc (a down filter: filter_size*down_factor = 6*2 = 12).
    {
        const int n = 12;
        Tensor f = design_lowpass_filter(n, /*cutoff=*/2.0, /*width=*/4.0,
                                         /*fs=*/16.0, /*radial=*/true);
        check(f.rows == n && f.cols == n, "radial shape");
        check(std::fabs(sum_of(f) - 1.0f) < 1e-4f, "radial unit DC gain");
        bool sym4 = true;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (std::fabs(f(i, j) - f(j, i)) > 1e-6f) sym4 = false;
                if (std::fabs(f(i, j) - f(n - 1 - i, j)) > 1e-6f) sym4 = false;
                if (std::fabs(f(i, j) - f(i, n - 1 - j)) > 1e-6f) sym4 = false;
            }
        check(sym4, "radial 4-fold symmetric");
        check(center_is_max(f), "radial peak at center");
    }

    if (failures == 0) { std::printf("test_stylegan3_filter: OK\n"); return 0; }
    std::printf("test_stylegan3_filter: %d failure(s)\n", failures);
    return 1;
}
