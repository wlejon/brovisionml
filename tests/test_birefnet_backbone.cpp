// BiRefNet Swin-L backbone: numeric parity against the PyTorch reference
// (triposplat_ref/model.py), gated on the real checkpoint + an out-of-repo
// golden dump (gen_birefnet_golden.py, format BRFB). When either is absent the
// test prints why and exits 0 (clean skip).
//
// The golden is the raw _SwinLarge backbone (no mul_scl dual-resolution concat)
// on a deterministic 256x256 input: 4 stage feature maps. This isolates the
// Swin machinery — patch embed, windowed attention with relative-position +
// shifted-window bias, patch merging, per-stage LayerNorm — before the decoder.

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/birefnet.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

using brotensor::Tensor;

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary); return f.good();
}

// Golden reader: 4s magic, int32 num_tensors, then per tensor
//   int32 ndim, int32[ndim] shape, float32[prod] data.
struct GTensor { std::vector<int> shape; std::vector<float> data; };
bool load_golden(const std::string& path, const char* magic,
                 std::vector<GTensor>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char m[4]; f.read(m, 4);
    if (std::string(m, 4) != magic) return false;
    int n = 0; f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n <= 0) return false;
    out.resize(n);
    for (int i = 0; i < n; ++i) {
        int nd = 0; f.read(reinterpret_cast<char*>(&nd), 4);
        if (!f || nd <= 0 || nd > 8) return false;
        out[i].shape.resize(nd);
        f.read(reinterpret_cast<char*>(out[i].shape.data()), nd * 4);
        std::size_t cnt = 1;
        for (int d : out[i].shape) cnt *= static_cast<std::size_t>(d);
        out[i].data.resize(cnt);
        f.read(reinterpret_cast<char*>(out[i].data.data()),
               static_cast<std::streamsize>(cnt * sizeof(float)));
        if (!f) return false;
    }
    return true;
}

struct Diff { double max_abs = 0, mean_abs = 0; };
Diff diff(const float* a, const float* b, std::size_t n) {
    Diff d; double s = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::fabs(double(a[i]) - double(b[i]));
        d.max_abs = std::max(d.max_abs, e); s += e;
    }
    d.mean_abs = n ? s / double(n) : 0; return d;
}

}  // namespace

int main() {
    brotensor::init();

    const char* wenv = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (wenv && *wenv) ? wenv : BROVISIONML_WEIGHTS_DIR;
    const std::string ckpt = base + "/triposplat/background_removal/birefnet.safetensors";

    const char* genv = std::getenv("BIREFNET_GOLDEN_DIR");
    const std::string gdir = (genv && *genv) ? genv : "D:/projects/_splat_assets";
    const std::string gpath = gdir + "/birefnet_backbone_golden.bin";

    if (!file_exists(ckpt) || !file_exists(gpath)) {
        std::printf("test_birefnet_backbone: checkpoint or golden absent "
                    "(ckpt '%s', golden '%s') — skipping.\n",
                    ckpt.c_str(), gpath.c_str());
        return 0;
    }

    std::vector<GTensor> g;
    check(load_golden(gpath, "BRFB", g), "load backbone golden");
    check(g.size() == 5, "golden has input + 4 stage features");
    if (failures) { std::printf("test_birefnet_backbone: bad golden\n"); return 1; }

    const int H = g[0].shape[2], W = g[0].shape[3];
    std::printf("backbone golden: input %dx%d, stages %d\n", H, W, int(g.size()) - 1);

    brovisionml::birefnet::BiRefNet net;
    net.load(ckpt);

    Tensor xin = Tensor::mat(1, 3 * H * W);
    std::copy(g[0].data.begin(), g[0].data.end(), xin.host_f32_mut());

    auto run_and_check = [&](brotensor::Device dev, const char* tag) {
        net.to(dev);
        Tensor xd = (dev == brotensor::Device::CPU) ? xin : xin.to(dev);
        std::vector<Tensor> outs = net.debugBackbone(xd, H, W);
        check(outs.size() == 4, "4 stage features");
        double worst = 0;
        for (int s = 0; s < 4 && s + 1 < int(g.size()); ++s) {
            Tensor o = (outs[s].device == brotensor::Device::CPU)
                           ? outs[s] : outs[s].to(brotensor::Device::CPU);
            const std::size_t n = g[s + 1].data.size();
            check(std::size_t(o.rows) * o.cols == n, "stage feature size matches golden");
            Diff d = diff(o.host_f32(), g[s + 1].data.data(), n);
            std::printf("  [%s] stage %d (%d,%d): max=%.3e mean=%.3e\n",
                        tag, s, o.rows, o.cols, d.max_abs, d.mean_abs);
            worst = std::max(worst, d.mean_abs);
        }
        // FP32 host vs FP32 reference: tight mean. (Per-element max can spike on
        // a few large activations; mean is the honest transcription signal.)
        check(worst < 2e-3, "backbone stage-feature mean error within tolerance");
    };

    run_and_check(brotensor::Device::CPU, "cpu");

    if (brotensor::is_available(brotensor::Device::CUDA)) {
        run_and_check(brotensor::Device::CUDA, "cuda");
        net.to(brotensor::Device::CPU);
    } else {
        std::printf("  (CUDA not available — GPU backbone check skipped)\n");
    }

    // ── full pipeline (decoder) golden: input -> sigmoid(forwardLogits) ──
    const std::string fpath = gdir + "/birefnet_full_golden.bin";
    if (!file_exists(fpath)) {
        std::printf("  (no full golden at '%s' — decoder parity skipped)\n", fpath.c_str());
    } else {
        std::vector<GTensor> gf;
        check(load_golden(fpath, "BRFF", gf), "load full golden");
        check(gf.size() == 2, "full golden = input + alpha");
        if (!failures) {
            const int fH = gf[0].shape[2], fW = gf[0].shape[3];
            std::printf("full golden: input %dx%d -> alpha\n", fH, fW);
            net.to(brotensor::Device::CPU);
            Tensor fin = Tensor::mat(1, 3 * fH * fW);
            std::copy(gf[0].data.begin(), gf[0].data.end(), fin.host_f32_mut());
            Tensor logits = net.forwardLogits(fin, fH, fW);
            Tensor lh = (logits.device == brotensor::Device::CPU)
                            ? logits : logits.to(brotensor::Device::CPU);
            const std::size_t n = gf[1].data.size();
            check(std::size_t(lh.rows) * lh.cols == n, "logits size matches alpha");
            std::vector<float> a(n);
            const float* lp = lh.host_f32();
            for (std::size_t i = 0; i < n; ++i) a[i] = 1.0f / (1.0f + std::exp(-lp[i]));
            Diff d = diff(a.data(), gf[1].data.data(), n);
            std::printf("  [cpu] alpha: max=%.3e mean=%.3e\n", d.max_abs, d.mean_abs);
            check(d.mean_abs < 3e-3, "full-pipeline alpha mean error within tolerance");

            if (brotensor::is_available(brotensor::Device::CUDA)) {
                net.to(brotensor::Device::CUDA);
                Tensor gl = net.forwardLogits(fin.to(brotensor::Device::CUDA), fH, fW);
                Tensor glh = gl.to(brotensor::Device::CPU);
                std::vector<float> ga(n);
                const float* glp = glh.host_f32();
                for (std::size_t i = 0; i < n; ++i) ga[i] = 1.0f / (1.0f + std::exp(-glp[i]));
                Diff dg = diff(ga.data(), gf[1].data.data(), n);
                std::printf("  [cuda] alpha: max=%.3e mean=%.3e\n", dg.max_abs, dg.mean_abs);
                check(dg.mean_abs < 3e-3, "full-pipeline alpha (CUDA) within tolerance");
                net.to(brotensor::Device::CPU);
            }
        }
    }

    if (failures == 0) { std::printf("test_birefnet_backbone: OK\n"); return 0; }
    std::printf("test_birefnet_backbone: %d failure(s)\n", failures);
    return 1;
}
