#include "brovisionml/stylegan3.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace brovisionml::stylegan3::detail {

using brotensor::Tensor;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Modified Bessel I0 and Bessel J1. These match scipy.special.i0 / j1 (hence
// numpy.kaiser's window) to ~1e-12, which is what the reference filter design
// relies on.
//
// The C++17 special-math functions (P0226) are the natural fit, but libc++
// (Apple clang / macOS) does not implement them — only libstdc++ and MSVC do.
// Use them where available and fall back to portable equivalents otherwise:
// I0 from its defining series (exact to full double precision over the small
// Kaiser-beta arguments used here) and J1 from the C math library, which both
// macOS and glibc provide (it is the same fdlibm/Cephes routine scipy.j1 is
// derived from).
#if defined(__cpp_lib_math_special_functions) && \
    __cpp_lib_math_special_functions >= 201603L
inline double i0(double x)  { return std::cyl_bessel_i(0.0, x); }
inline double j1(double x)  { return std::cyl_bessel_j(1.0, x); }
#else
inline double i0(double x) {
    // I0(x) = sum_{k>=0} (x^2/4)^k / (k!)^2
    const double y = 0.25 * x * x;
    double term = 1.0, sum = 1.0;
    for (int k = 1; k < 256; ++k) {
        term *= y / (static_cast<double>(k) * static_cast<double>(k));
        sum += term;
        if (term <= sum * 1e-18) break;
    }
    return sum;
}
inline double j1(double x)  { return ::j1(x); }
#endif

// numpy sinc: sin(pi x)/(pi x), with sinc(0) = 1.
inline double sinc(double x) {
    if (x == 0.0) return 1.0;
    const double px = kPi * x;
    return std::sin(px) / px;
}

// scipy.signal.kaiser_atten(numtaps, width): stop-band attenuation in dB for a
// Kaiser window of `numtaps` taps and normalized transition `width`.
inline double kaiser_atten(int numtaps, double width) {
    return 2.285 * (numtaps - 1) * kPi * width + 7.95;
}

// scipy.signal.kaiser_beta(a): Kaiser beta from a target attenuation `a` (dB).
inline double kaiser_beta(double a) {
    if (a > 50.0) return 0.1102 * (a - 8.7);
    if (a > 21.0) return 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
    return 0.0;
}

// numpy.kaiser(M, beta) window.
std::vector<double> kaiser_window(int M, double beta) {
    std::vector<double> w(static_cast<std::size_t>(M));
    if (M == 1) { w[0] = 1.0; return w; }
    const double denom = i0(beta);
    const double half = (M - 1) / 2.0;
    for (int n = 0; n < M; ++n) {
        const double t = (n - half) / half;       // in [-1, 1]
        const double arg = 1.0 - t * t;
        w[static_cast<std::size_t>(n)] =
            i0(beta * std::sqrt(arg > 0.0 ? arg : 0.0)) / denom;
    }
    return w;
}

// scipy.signal.firwin(numtaps, cutoff, width, fs) for a single-cutoff lowpass
// (pass_zero=True, scale=True): Kaiser-windowed ideal lowpass, normalized to
// unit gain at DC.
std::vector<double> firwin(int numtaps, double cutoff, double width, double fs) {
    const double nyq   = fs / 2.0;
    const double fc    = cutoff / nyq;            // normalized cutoff in (0,1)
    const double wnorm = width / nyq;             // normalized transition width
    const double beta  = kaiser_beta(kaiser_atten(numtaps, wnorm));
    const std::vector<double> win = kaiser_window(numtaps, beta);

    const double alpha = 0.5 * (numtaps - 1);
    std::vector<double> h(static_cast<std::size_t>(numtaps));
    double sum = 0.0;
    for (int n = 0; n < numtaps; ++n) {
        const double m = n - alpha;
        const double v = fc * sinc(fc * m) * win[static_cast<std::size_t>(n)];
        h[static_cast<std::size_t>(n)] = v;
        sum += v;
    }
    for (double& v : h) v /= sum;                 // scale: unit DC gain
    return h;
}

Tensor outer_to_tensor(const std::vector<double>& f) {
    const int n = static_cast<int>(f.size());
    Tensor out = Tensor::mat(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            out(i, j) = static_cast<float>(f[static_cast<std::size_t>(i)] *
                                           f[static_cast<std::size_t>(j)]);
    return out;
}

}  // namespace

Tensor design_lowpass_filter(int numtaps, double cutoff, double width,
                             double fs, bool radial) {
    if (numtaps < 1)
        throw std::runtime_error("stylegan3::design_lowpass_filter: numtaps < 1");

    // Identity filter.
    if (numtaps == 1) {
        Tensor out = Tensor::mat(1, 1);
        out(0, 0) = 1.0f;
        return out;
    }

    // Separable Kaiser/firwin lowpass -> 2D outer product.
    if (!radial)
        return outer_to_tensor(firwin(numtaps, cutoff, width, fs));

    // Radially-symmetric jinc filter, windowed by an outer Kaiser window.
    std::vector<double> x(static_cast<std::size_t>(numtaps));
    for (int n = 0; n < numtaps; ++n)
        x[static_cast<std::size_t>(n)] = (n - (numtaps - 1) / 2.0) / fs;

    const double beta = kaiser_beta(kaiser_atten(numtaps, width / (fs / 2.0)));
    const std::vector<double> w = kaiser_window(numtaps, beta);

    Tensor out = Tensor::mat(numtaps, numtaps);
    std::vector<double> f(static_cast<std::size_t>(numtaps) * numtaps);
    double sum = 0.0;
    for (int i = 0; i < numtaps; ++i) {
        for (int j = 0; j < numtaps; ++j) {
            const double xi = x[static_cast<std::size_t>(i)];
            const double xj = x[static_cast<std::size_t>(j)];
            const double r  = std::hypot(xj, xi);
            // jinc: j1(2*pi*cutoff*r)/(pi*r); limit at r->0 is `cutoff`. For the
            // even tap counts config-R uses, r is never exactly 0, but guard it.
            double v = (r == 0.0) ? cutoff
                                  : j1(2.0 * kPi * cutoff * r) / (kPi * r);
            v *= w[static_cast<std::size_t>(i)] * w[static_cast<std::size_t>(j)];
            f[static_cast<std::size_t>(i) * numtaps + j] = v;
            sum += v;
        }
    }
    for (int i = 0; i < numtaps; ++i)
        for (int j = 0; j < numtaps; ++j)
            out(i, j) = static_cast<float>(
                f[static_cast<std::size_t>(i) * numtaps + j] / sum);
    return out;
}

}  // namespace brovisionml::stylegan3::detail
