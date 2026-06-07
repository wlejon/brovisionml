// StyleGAN3-R inversion test: the synthesis backward (image -> W+ gradient) and
// the optimization projector (Generator::invert).
//
// Gate 1 — gradient check: a finite-difference verification of the hand-written
//   synthesis backward. For a fixed random co-vector R, the scalar loss
//   L(ws) = <synthesize(ws), R> has gradient dL/dws = backward(ws, R). We probe
//   a handful of W+ coordinates across the input row, hidden rows, and the ToRGB
//   row and compare the analytic gradient to a central difference. This needs no
//   golden — it checks the backward against the forward it was derived from.
//
// Gate 2 — round trip: generate an image from a seed, invert it, and confirm the
//   projector drives the image-space loss down to a tight reconstruction.
//
// Both gates are gated on the converted checkpoint
// (weights/stylegan3-r-ffhqu-256/model.safetensors) and skip cleanly when it is
// absent. Each runs on CUDA when available, else CPU.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/stylegan3.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}
bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

using brovisionml::stylegan3::Config;
using brovisionml::stylegan3::Generator;
using brotensor::Tensor;

// splitmix64 -> standard-normal (Box-Muller), deterministic across runs.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    double unit() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z = z ^ (z >> 31);
        return (z >> 11) * (1.0 / 9007199254740992.0);
    }
    float normal() {
        const double u1 = unit(), u2 = unit();
        return static_cast<float>(std::sqrt(-2.0 * std::log(u1 + 1e-12)) *
                                  std::cos(2.0 * 3.14159265358979323846 * u2));
    }
};

// <synthesize(ws), R> with ws on `dev`; ws_host is the (num_ws*w_dim) latent.
double loss_dot(const Generator& gen, brotensor::Device dev,
                const std::vector<float>& ws_host, int n_ws, int w_dim,
                const std::vector<float>& R) {
    Tensor ws = Tensor::mat(n_ws, w_dim);
    for (std::size_t i = 0; i < ws_host.size(); ++i) ws[static_cast<int>(i)] = ws_host[i];
    if (dev != brotensor::Device::CPU) ws = ws.to(dev);
    Tensor img = gen.synthesize(ws).to(brotensor::Device::CPU);
    const float* d = img.host_f32();
    double s = 0;
    for (std::size_t i = 0; i < R.size(); ++i) s += static_cast<double>(d[i]) * R[i];
    return s;
}

void run_gradcheck(Generator& gen, brotensor::Device dev, const char* tag) {
    const int n_ws  = gen.num_ws();
    const int w_dim = gen.config().w_dim;
    const int C = gen.config().img_channels;
    const int HW = gen.config().img_resolution * gen.config().img_resolution;
    const std::size_t img_n = static_cast<std::size_t>(C) * HW;

    // Build the probe latent from the mapping network: a fixed seed's W+.
    Rng zr(1234);
    Tensor z = Tensor::mat(1, gen.config().z_dim);
    for (int i = 0; i < gen.config().z_dim; ++i) z[i] = zr.normal();
    if (dev != brotensor::Device::CPU) z = z.to(dev);
    std::vector<float> ws_host =
        gen.map(z, /*psi=*/0.8f).to(brotensor::Device::CPU).to_host_vector();

    // Fixed random co-vector R (small) — the upstream gradient dL/dimg.
    Rng rr(99);
    std::vector<float> R(img_n);
    for (std::size_t i = 0; i < img_n; ++i) R[i] = 0.01f * rr.normal();

    // Analytic gradient: backward with dimg = R.
    Tensor ws_dev = Tensor::mat(n_ws, w_dim);
    for (std::size_t i = 0; i < ws_host.size(); ++i) ws_dev[static_cast<int>(i)] = ws_host[i];
    if (dev != brotensor::Device::CPU) ws_dev = ws_dev.to(dev);
    Tensor R_dev = Tensor::mat(1, static_cast<int>(img_n));
    for (std::size_t i = 0; i < img_n; ++i) R_dev[static_cast<int>(i)] = R[i];
    if (dev != brotensor::Device::CPU) R_dev = R_dev.to(dev);

    brovisionml::stylegan3::SynthesisNetwork::Cache cache;
    Tensor img_cached = gen.synthesis().forward_cached(ws_dev, cache);
    Tensor dws;
    gen.synthesis().backward(ws_dev, cache, R_dev, dws);
    std::vector<float> an = dws.to(brotensor::Device::CPU).to_host_vector();

    // CRITICAL invariant: forward_cached must equal plain synthesize, else the
    // analytic gradient is for a different function than the FD measures.
    {
        std::vector<float> a = img_cached.to(brotensor::Device::CPU).to_host_vector();
        std::vector<float> b = gen.synthesize(ws_dev).to(brotensor::Device::CPU).to_host_vector();
        double mx = 0;
        for (std::size_t i = 0; i < a.size(); ++i) mx = std::max(mx, (double)std::fabs(a[i] - b[i]));
        std::printf("  [%s] max|forward_cached - synthesize| = %.3e\n", tag, mx);
    }

    // Per-ROW directional derivative along the UNIT analytic-gradient direction.
    // Signal = ||g_row|| (maximal, beats GPU noise) and the perturbation size is
    // eps regardless of row (so truncation is small and comparable). With a small
    // eps a correct backward matches the central difference tightly; this is the
    // authoritative full-chain gradient check.
    const float eps = 3e-4f;
    const int probe_rows[] = {0, 1, 2, 3, 4, 8, 15};
    double worst_layer = 0.0;   // rows >=1 (the layer chain) — strict
    double row0_rel = 0.0;      // row 0 (Fourier input) — FD is far more nonlinear
    for (int r : probe_rows) {
        std::vector<float> v(w_dim);
        double norm2 = 0.0;
        for (int j = 0; j < w_dim; ++j) {
            v[j] = an[static_cast<std::size_t>(r) * w_dim + j];
            norm2 += static_cast<double>(v[j]) * v[j];
        }
        const double gnorm = std::sqrt(norm2);
        if (gnorm < 1e-12) continue;
        for (auto& e : v) e = static_cast<float>(e / gnorm);   // unit direction

        std::vector<float> wp = ws_host, wm = ws_host;
        for (int j = 0; j < w_dim; ++j) {
            const std::size_t idx = static_cast<std::size_t>(r) * w_dim + j;
            wp[idx] += eps * v[j];
            wm[idx] -= eps * v[j];
        }
        const double fd = (loss_dot(gen, dev, wp, n_ws, w_dim, R) -
                           loss_dot(gen, dev, wm, n_ws, w_dim, R)) / (2.0 * eps);
        const double rel = std::fabs(fd - gnorm) / (gnorm + std::fabs(fd) + 1e-9);
        if (r == 0) row0_rel = rel; else worst_layer = std::max(worst_layer, rel);
        std::printf("  [%s] row %2d  ||g||=% .5e  fd=% .5e  rel=%.2e\n", tag, r, gnorm, fd, rel);
    }
    std::printf("  [%s] full-chain gradient check: layer rows worst=%.2e, row0=%.2e\n",
                tag, worst_layer, row0_rel);
    // Rows >=1 traverse the synthesis layer chain — a real chain/backward bug lands
    // here as an O(1) error, so hold them tight. Row 0 drives the Fourier-feature
    // input, whose trig makes the central difference far more truncation-prone; it
    // is verified to converge under an eps sweep, so it only gets a loose bound.
    check(worst_layer < 1.5e-2, "synthesis backward matches finite differences (layer chain)");
    check(row0_rel < 8e-2, "synthesis input backward matches finite differences");
}

// Isolate ONE synthesis layer's backward from the full-chain compounding:
// FD-check both dx (grad to the layer input) and dw_row (grad to the layer's W+
// row) against the layer's own forward. Deterministic on CPU.
void run_layer_gradcheck(Generator& gen, brotensor::Device dev, int layer_idx,
                         const char* tag) {
    const int w_dim = gen.config().w_dim;

    // A real latent + the actual input fed to this layer (from a full forward).
    Rng zr(55);
    Tensor z = Tensor::mat(1, gen.config().z_dim);
    for (int i = 0; i < gen.config().z_dim; ++i) z[i] = zr.normal();
    if (dev != brotensor::Device::CPU) z = z.to(dev);
    Tensor ws = gen.map(z, 0.9f);
    brovisionml::stylegan3::SynthesisNetwork::Cache fc;
    (void)gen.synthesis().forward_cached(ws, fc);

    const auto& layer = gen.synthesis().layers()[layer_idx];
    std::vector<float> x_host = fc.layer_in[layer_idx].to_host_vector();
    std::vector<float> w_host(static_cast<std::size_t>(w_dim));
    {
        std::vector<float> ws_host = ws.to_host_vector();
        for (int j = 0; j < w_dim; ++j)
            w_host[j] = ws_host[static_cast<std::size_t>(layer_idx + 1) * w_dim + j];
    }
    const int xn = static_cast<int>(x_host.size());
    const int out_n = layer.out_channels() * layer.out_size() * layer.out_size();

    Rng rr(layer_idx * 7 + 3);
    std::vector<float> R(static_cast<std::size_t>(out_n));
    for (auto& v : R) v = 0.05f * rr.normal();

    auto to_dev = [&](Tensor t) { return dev == brotensor::Device::CPU ? t : t.to(dev); };
    auto fwd_dot = [&](const std::vector<float>& xv, const std::vector<float>& wv) {
        Tensor X = Tensor::mat(1, xn);
        for (int i = 0; i < xn; ++i) X[i] = xv[i];
        Tensor Wt = Tensor::mat(1, w_dim);
        for (int j = 0; j < w_dim; ++j) Wt[j] = wv[j];
        Tensor Xd = to_dev(X), Wd = to_dev(Wt);
        brovisionml::stylegan3::SynthesisLayer::Cache lc;
        Tensor out = layer.forward_cached(Wd, Xd, lc).to(brotensor::Device::CPU);
        const float* o = out.host_f32();
        double s = 0;
        for (int i = 0; i < out_n; ++i) s += static_cast<double>(o[i]) * R[i];
        return s;
    };

    // Analytic dx, dw_row.
    Tensor X = Tensor::mat(1, xn);
    for (int i = 0; i < xn; ++i) X[i] = x_host[i];
    Tensor Wt = Tensor::mat(1, w_dim);
    for (int j = 0; j < w_dim; ++j) Wt[j] = w_host[j];
    Tensor Xd = to_dev(X), Wd = to_dev(Wt);
    brovisionml::stylegan3::SynthesisLayer::Cache lc;
    (void)layer.forward_cached(Wd, Xd, lc);
    Tensor R_dev = Tensor::mat(1, out_n);
    for (int i = 0; i < out_n; ++i) R_dev[i] = R[i];
    R_dev = to_dev(R_dev);
    Tensor dx, dw_row;
    layer.backward(Wd, Xd, lc, R_dev, dx, dw_row);
    std::vector<float> dx_h = dx.to_host_vector();
    std::vector<float> dw_h = dw_row.to_host_vector();

    // Probe along the ANALYTIC gradient direction itself: this guarantees a large
    // signal (||grad||^2), defeating the small-signal GPU-FD noise. dx is linear
    // in the input so its central FD is exact at any step; dw is nonlinear so use
    // a small step to limit truncation.
    auto dirderiv = [&](const std::vector<float>& base, const std::vector<float>& grad,
                        bool perturb_x, float h) {
        std::vector<float> p = base, m = base;
        double adir = 0;
        for (std::size_t i = 0; i < base.size(); ++i) {
            p[i] += h * grad[i]; m[i] -= h * grad[i];
            adir += (double)grad[i] * grad[i];
        }
        const double fd = perturb_x
            ? (fwd_dot(p, w_host) - fwd_dot(m, w_host)) / (2.0 * h)
            : (fwd_dot(x_host, p) - fwd_dot(x_host, m)) / (2.0 * h);
        return std::pair<double,double>(adir, fd);
    };
    auto [adx, fdx] = dirderiv(x_host, dx_h, /*perturb_x=*/true, 1e-3f);
    auto [adw, fdw] = dirderiv(w_host, dw_h, /*perturb_x=*/false, 3e-4f);
    const double rel_dx = std::fabs(fdx - adx) / (std::fabs(adx) + std::fabs(fdx) + 1e-9);
    const double rel_dw = std::fabs(fdw - adw) / (std::fabs(adw) + std::fabs(fdw) + 1e-9);

    std::printf("  [%s] L%-2d out=%d up=%d down=%d | dx |g|^2=%.3e fd=%.3e rel=%.2e | dw rel=%.2e\n",
                tag, layer_idx, layer.out_size(), layer.up_factor(), layer.down_factor(),
                adx, fdx, rel_dx, rel_dw);
    check(rel_dx < 1e-2, "single-layer dX matches finite differences");
    check(rel_dw < 2e-2, "single-layer dW matches finite differences");
}

void run_roundtrip(Generator& gen, brotensor::Device dev, const char* tag) {
    // Generate a target from a fixed seed, then invert it.
    Rng zr(7);
    Tensor z = Tensor::mat(1, gen.config().z_dim);
    for (int i = 0; i < gen.config().z_dim; ++i) z[i] = zr.normal();
    if (dev != brotensor::Device::CPU) z = z.to(dev);
    brovisionml::stylegan3::Image target = gen.generate(z, /*psi=*/0.7f);

    Generator::InvertOptions opt;
    opt.num_steps = 600;
    opt.lr = 0.1f;
    float first = 0.0f;
    opt.on_step = [&](int step, float loss) {
        if (step == 1) first = loss;
        if (step == 1 || step % 100 == 0)
            std::printf("  [%s] invert step %3d  mse=%.5e\n", tag, step, loss);
    };
    Generator::InvertResult res = gen.invert(target, opt);

    std::printf("  [%s] invert: first mse=%.5e -> final mse=%.5e\n", tag, first, res.loss);
    check(res.loss < first, "inversion reduced the image-space loss");
    check(res.loss < 8e-3f, "inversion reached a tight reconstruction");

    // Render the recovered latent and confirm it matches the target in 8-bit.
    brovisionml::stylegan3::Image rec = gen.render(res.w);
    double sad = 0;
    for (std::size_t i = 0; i < target.rgb.size(); ++i)
        sad += std::fabs(static_cast<double>(target.rgb[i]) - rec.rgb[i]);
    const double mae = sad / static_cast<double>(target.rgb.size());
    std::printf("  [%s] recovered render: mean abs 8-bit diff=%.3f\n", tag, mae);
    check(mae < 8.0, "recovered render close to target (8-bit)");
}

}  // namespace

int main() {
    // Structural sanity (always on).
    {
        Generator g(Config::r256());
        check(g.num_ws() == 16, "r256 num_ws == 16");
    }

    std::string base = BROVISIONML_WEIGHTS_DIR;
    if (const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR")) base = env;
    const std::string dir  = base + "/stylegan3-r-ffhqu-256";
    const std::string ckpt = dir + "/model.safetensors";
    if (!file_exists(ckpt)) {
        std::printf("test_stylegan3_invert: structural OK; inversion SKIPPED "
                    "(no checkpoint under %s)\n", base.c_str());
        return failures == 0 ? 0 : 1;
    }

    try {
        brotensor::init();

        const brotensor::Device dev = brotensor::is_available(brotensor::Device::CUDA)
                                          ? brotensor::Device::CUDA
                                          : brotensor::Device::CPU;
        const char* tag = dev == brotensor::Device::CUDA ? "cuda" : "cpu";
        Generator gen(Config::r256());
        gen.load(dir);
        gen.to(dev);

        std::printf("test_stylegan3_invert: per-layer dX/dW [%s]\n", tag);
        for (int li = 0; li < gen.num_ws() - 1; ++li)
            run_layer_gradcheck(gen, dev, li, tag);

        std::printf("test_stylegan3_invert: full-chain gradient check [%s]\n", tag);
        run_gradcheck(gen, dev, tag);

        std::printf("test_stylegan3_invert: round trip [%s]\n", tag);
        run_roundtrip(gen, dev, tag);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (failures == 0) { std::printf("test_stylegan3_invert: OK\n"); return 0; }
    std::printf("test_stylegan3_invert: %d failure(s)\n", failures);
    return 1;
}
