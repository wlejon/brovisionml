// DPT depth head: structural / end-to-end CPU test with no real checkpoint.
// We synthesize a tiny HF-named `neck.`/`head.` checkpoint, feed four synthetic
// stage feature maps (cls + 2x2 patch grid), run forward(), and assert the
// depth map shape (1, (gh*patch)*(gw*patch)), finiteness, and CPU/CUDA parity.
// This exercises reassemble (1x1 proj + ConvTranspose up / Identity / strided
// Conv down), the bias-free neck convs, the RefineNet fusion (pre-act residual
// units + align_corners upsample + projection), and the depth head.

#include "brovisionml/dpt_head.h"

#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"
#include "brotensor/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

namespace st = brotensor::safetensors;
using brovisionml::dpt::DepthHead;
using brovisionml::dpt::HeadConfig;

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

struct CheckpointBuilder {
    std::deque<std::vector<float>> store;
    std::vector<st::WriteEntry>    entries;
    void add(const std::string& name, std::vector<int64_t> shape) {
        int64_t n = 1; for (int64_t d : shape) n *= d;
        std::vector<float> buf(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < buf.size(); ++i)
            buf[i] = (static_cast<float>((i + name.size()) % 7) - 3.0f) * 0.05f;
        store.push_back(std::move(buf));
        const std::vector<float>& b = store.back();
        entries.push_back(st::WriteEntry{name, st::Dtype::F32, std::move(shape),
                                         b.data(), b.size() * sizeof(float)});
    }
};

void build(CheckpointBuilder& cb, const HeadConfig& cfg) {
    const int D = cfg.reassemble_hidden_size, F = cfg.fusion_hidden_size;
    const int n = static_cast<int>(cfg.neck_hidden_sizes.size());
    for (int i = 0; i < n; ++i) {
        const int c = cfg.neck_hidden_sizes[i];
        const std::string lp = "neck.reassemble_stage.layers." + std::to_string(i) + ".";
        cb.add(lp + "projection.weight", {c, D, 1, 1});
        cb.add(lp + "projection.bias",   {c});
        const double factor = cfg.reassemble_factors[i];
        if (factor > 1.0) {
            const int k = static_cast<int>(std::lround(factor));
            cb.add(lp + "resize.weight", {c, c, k, k});   // ConvTranspose2d
            cb.add(lp + "resize.bias",   {c});
        } else if (factor < 1.0) {
            cb.add(lp + "resize.weight", {c, c, 3, 3});   // strided Conv2d
            cb.add(lp + "resize.bias",   {c});
        }  // factor == 1 -> Identity, no params
    }
    for (int i = 0; i < n; ++i)
        cb.add("neck.convs." + std::to_string(i) + ".weight",
               {F, cfg.neck_hidden_sizes[i], 3, 3});       // bias-free
    for (int j = 0; j < n; ++j) {
        const std::string lp = "neck.fusion_stage.layers." + std::to_string(j) + ".";
        cb.add(lp + "projection.weight", {F, F, 1, 1});
        cb.add(lp + "projection.bias",   {F});
        for (const char* ru : {"residual_layer1.", "residual_layer2."}) {
            cb.add(lp + ru + "convolution1.weight", {F, F, 3, 3});
            cb.add(lp + ru + "convolution1.bias",   {F});
            cb.add(lp + ru + "convolution2.weight", {F, F, 3, 3});
            cb.add(lp + ru + "convolution2.bias",   {F});
        }
    }
    const int Fh = F / 2, H = cfg.head_hidden_size;
    cb.add("head.conv1.weight", {Fh, F, 3, 3});  cb.add("head.conv1.bias", {Fh});
    cb.add("head.conv2.weight", {H, Fh, 3, 3});  cb.add("head.conv2.bias", {H});
    cb.add("head.conv3.weight", {1, H, 1, 1});   cb.add("head.conv3.bias", {1});
}

std::vector<brotensor::Tensor> make_feature_maps(int D, int gh, int gw) {
    std::vector<brotensor::Tensor> maps;
    for (int s = 0; s < 4; ++s) {
        brotensor::Tensor t = brotensor::Tensor::mat(1 + gh * gw, D);
        float* p = t.host_f32_mut();
        for (int i = 0; i < t.rows * t.cols; ++i)
            p[i] = std::sin(0.05f * (i + 1) + 0.7f * s);
        maps.push_back(std::move(t));
    }
    return maps;
}

}  // namespace

int main() {
    // ── Config presets ──
    {
        HeadConfig s = HeadConfig::vit_s();
        check(s.fusion_hidden_size == 64 && s.neck_hidden_sizes ==
              std::vector<int>({48, 96, 192, 384}), "vit_s head config");
        HeadConfig b = HeadConfig::vit_b();
        check(b.fusion_hidden_size == 128 && b.reassemble_hidden_size == 768,
              "vit_b head config");
        HeadConfig l = HeadConfig::vit_l();
        check(l.fusion_hidden_size == 256 && l.neck_hidden_sizes ==
              std::vector<int>({256, 512, 1024, 1024}), "vit_l head config");
    }

    // ── Tiny head: D=16, neck {4,6,8,16}, fusion 8, head_hidden 4, patch 14 ──
    HeadConfig cfg;
    cfg.reassemble_hidden_size = 16;
    cfg.neck_hidden_sizes = {4, 6, 8, 16};
    cfg.fusion_hidden_size = 8;
    cfg.head_hidden_size = 4;
    cfg.patch_size = 14;

    CheckpointBuilder cb;
    build(cb, cfg);
    const std::string path = "dpt_head_test.safetensors";
    st::write_file(path, cb.entries);

    DepthHead head(cfg);
    head.load_file(path);

    const int gh = 2, gw = 2;
    auto maps = make_feature_maps(cfg.reassemble_hidden_size, gh, gw);

    brotensor::Tensor depth = head.forward(maps, gh, gw);
    const int oh = gh * cfg.patch_size, ow = gw * cfg.patch_size;
    check(depth.rows == 1 && depth.cols == oh * ow,
          "depth map is (1, (gh*patch)*(gw*patch))");
    {
        bool finite = true, nonneg = true;
        const float* d = depth.host_f32();
        for (int i = 0; i < depth.cols; ++i) {
            if (!std::isfinite(d[i])) finite = false;
            if (d[i] < 0.0f) nonneg = false;
        }
        check(finite, "depth map all-finite");
        check(nonneg, "depth map non-negative (final ReLU)");
    }

    // ── non-square grid (gh=3, gw=2) shape ──
    {
        auto m2 = make_feature_maps(cfg.reassemble_hidden_size, 3, 2);
        brotensor::Tensor d2 = head.forward(m2, 3, 2);
        check(d2.cols == (3 * 14) * (2 * 14), "non-square depth map shape");
    }

    // ── CUDA parity ──
    brotensor::init();
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        head.to(brotensor::Device::CUDA);
        check(head.device() == brotensor::Device::CUDA, "to(CUDA) moved weights");
        std::vector<brotensor::Tensor> gmaps;
        for (auto& m : maps) gmaps.push_back(m.to(brotensor::Device::CUDA));
        brotensor::Tensor gd = head.forward(gmaps, gh, gw);
        brotensor::Tensor back = gd.to(brotensor::Device::CPU);
        float worst = 0.0f;
        const float* a = depth.host_f32();
        const float* b = back.host_f32();
        for (int i = 0; i < depth.cols; ++i)
            worst = std::max(worst, std::fabs(a[i] - b[i]));
        if (worst > 1e-3f) {
            std::fprintf(stderr, "FAIL: CPU/CUDA dpt_head diff %g > 1e-3\n", worst);
            ++failures;
        } else {
            std::printf("  CUDA parity OK (worst diff %g)\n", worst);
        }
    } else {
        std::printf("  (CUDA not available — parity check skipped)\n");
    }

    if (failures == 0) { std::printf("test_dpt_head: OK\n"); return 0; }
    std::printf("test_dpt_head: %d failure(s)\n", failures);
    return 1;
}
