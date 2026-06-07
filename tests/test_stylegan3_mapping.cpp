// StyleGAN3 mapping-network test.
//
// Two layers:
//   * Always-on checks: config-R presets carry the documented hyperparameters
//     (channel_base/max, conv_kernel, radial filters, num_ws = num_layers+2),
//     and FullyConnectedLayer.forward reproduces hand-computed linear and
//     leaky-ReLU outputs (the latter with the reference's sqrt(2) gain / 0.2
//     slope).
//   * Weights-gated run: when a converted StyleGAN3-R checkpoint is present
//     under weights/, load the mapping network and assert the W+ output shape
//     (num_ws, w_dim), finiteness, and the truncation invariant — with psi=0
//     every truncated row collapses onto w_avg. Convert a checkpoint with
//       scripts/convert-stylegan3.py
//     When absent the test prints why and exits 0 (clean skip).

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/stylegan3.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}
bool approx(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) <= tol; }

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

namespace st = brotensor::safetensors;
using brovisionml::stylegan3::Config;
using brovisionml::stylegan3::FullyConnectedLayer;
using brovisionml::stylegan3::MappingNetwork;
using brotensor::Tensor;

void config_checks() {
    Config c = Config::r256();
    check(c.img_resolution == 256, "r256 resolution");
    // Config-R shares channel_base=32768 with config-T; it differs by raising
    // channel_max to 1024 (the released pickles' init_kwargs confirm 32768/1024).
    check(c.channel_base == 32768, "config-R channel_base 32768");
    check(c.channel_max == 1024, "config-R channel_max raised to 1024");
    check(c.conv_kernel == 1, "config-R 1x1 conv");
    check(c.use_radial_filters, "config-R radial filters");
    check(c.num_ws() == c.num_layers + 2, "num_ws = num_layers + 2");
    check(c.num_ws() == 16, "default num_ws == 16");
    check(Config::r1024().img_resolution == 1024, "r1024 resolution");
}

void fc_numeric_checks() {
    // Linear: W = [[1,2],[3,4]], b = [10,20], x = [1,1] -> [13, 27].
    {
        FullyConnectedLayer fc;
        fc.in_features = 2;
        fc.out_features = 2;
        fc.has_bias = true;
        fc.act = FullyConnectedLayer::Act::Linear;
        fc.weight = Tensor::mat(2, 2);
        fc.weight(0, 0) = 1; fc.weight(0, 1) = 2;
        fc.weight(1, 0) = 3; fc.weight(1, 1) = 4;
        fc.bias = Tensor::vec(2);
        fc.bias[0] = 10; fc.bias[1] = 20;

        Tensor x = Tensor::mat(1, 2);
        x[0] = 1; x[1] = 1;
        Tensor y;
        fc.forward(x, y);
        check(y.rows == 1 && y.cols == 2, "FC linear output shape");
        check(approx(y[0], 13.0f) && approx(y[1], 27.0f), "FC linear values");
    }
    // LRelu: W = [[-1,0],[0,1]], b = 0, x = [1,1] -> Wx = [-1, 1];
    // leaky_relu(x)*sqrt(2): [-0.2*sqrt2, 1*sqrt2].
    {
        FullyConnectedLayer fc;
        fc.in_features = 2;
        fc.out_features = 2;
        fc.has_bias = true;
        fc.act = FullyConnectedLayer::Act::LRelu;
        fc.weight = Tensor::mat(2, 2);
        fc.weight(0, 0) = -1; fc.weight(0, 1) = 0;
        fc.weight(1, 0) = 0;  fc.weight(1, 1) = 1;
        fc.bias = Tensor::vec(2);  // zeros

        Tensor x = Tensor::mat(1, 2);
        x[0] = 1; x[1] = 1;
        Tensor y;
        fc.forward(x, y);
        const float s2 = std::sqrt(2.0f);
        check(approx(y[0], -0.2f * s2) && approx(y[1], 1.0f * s2), "FC lrelu values");
    }
}

}  // namespace

int main() {
    config_checks();
    fc_numeric_checks();

    // ── Weights-gated mapping run (clean skip if absent) ──
    std::string base = BROVISIONML_WEIGHTS_DIR;
    if (const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR")) base = env;
    // Any converted config-R checkpoint; 256 is the smallest common target.
    std::string ckpt = base + "/stylegan3-r-afhqv2-256/model.safetensors";

    if (!file_exists(ckpt)) {
        std::printf("test_stylegan3_mapping: no checkpoint at %s — skipping the "
                    "weights-gated run (always-on checks done).\n", ckpt.c_str());
        return failures == 0 ? 0 : 1;
    }

    try {
        Config cfg = Config::r256();
        MappingNetwork map(cfg);
        st::File f = st::File::open(ckpt);
        map.load(&f, "mapping");

        Tensor z = Tensor::mat(1, cfg.z_dim);
        for (int i = 0; i < cfg.z_dim; ++i)
            z[i] = std::sin(0.1f * static_cast<float>(i));  // deterministic z

        Tensor ws = map.forward(z, /*truncation_psi=*/1.0f);
        Tensor ws_host = ws.to(brotensor::Device::CPU);
        check(ws_host.rows == cfg.num_ws() && ws_host.cols == cfg.w_dim,
              "W+ shape (num_ws, w_dim)");
        bool finite = true;
        for (int i = 0; i < ws_host.size(); ++i)
            if (!std::isfinite(ws_host[i])) finite = false;
        check(finite, "W+ all finite");

        // Truncation invariant: psi=0 collapses every truncated row onto w_avg,
        // so all rows are identical.
        Tensor ws0 = map.forward(z, /*truncation_psi=*/0.0f).to(brotensor::Device::CPU);
        bool rows_equal = true;
        for (int r = 1; r < ws0.rows; ++r)
            for (int c = 0; c < ws0.cols; ++c)
                if (!approx(ws0(r, c), ws0(0, c), 1e-5f)) rows_equal = false;
        check(rows_equal, "psi=0 collapses all W+ rows onto w_avg");
        std::printf("test_stylegan3_mapping: weights-gated run OK\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (failures == 0) { std::printf("test_stylegan3_mapping: OK\n"); return 0; }
    std::printf("test_stylegan3_mapping: %d failure(s)\n", failures);
    return 1;
}
