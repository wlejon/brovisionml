// DINOv2 ViT backbone: structural / end-to-end CPU test with no real checkpoint.
// We synthesize a tiny HF-named `backbone.` checkpoint (a 4-block, 16-dim ViT),
// write it with brotensor's safetensors writer, load it through the real loader,
// and run encode(). This exercises the full path — HF weight naming, the
// LayerScale fold, cls token + position-embedding add, the interpolated-pos-embed
// branch (non-native grid), plain MHSA, and stage selection — asserting the
// per-stage feature-map shapes, finiteness, and CPU/CUDA parity.

#include "brovisionml/dinov2.h"

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
using brovisionml::dinov2::Backbone;
using brovisionml::dinov2::BackboneOutput;
using brovisionml::dinov2::Config;

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

struct CheckpointBuilder {
    std::deque<std::vector<float>> store;
    std::vector<st::WriteEntry>    entries;

    void add(const std::string& name, std::vector<int64_t> shape) {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        std::vector<float> buf(static_cast<std::size_t>(n));

        const bool is_norm = name.find("norm") != std::string::npos;
        const bool is_scale = name.find("layer_scale") != std::string::npos;
        const bool is_w = name.size() >= 7 &&
                          name.compare(name.size() - 7, 7, ".weight") == 0;
        for (std::size_t i = 0; i < buf.size(); ++i) {
            if (is_norm) buf[i] = is_w ? 1.0f : 0.0f;          // gamma=1, beta=0
            else if (is_scale) buf[i] = 0.4f + 0.01f * (i % 5); // non-trivial LS
            else buf[i] = (static_cast<float>(i % 7) - 3.0f) * 0.03f;
        }
        store.push_back(std::move(buf));
        const std::vector<float>& b = store.back();
        entries.push_back(st::WriteEntry{name, st::Dtype::F32, std::move(shape),
                                         b.data(), b.size() * sizeof(float)});
    }
};

void build(CheckpointBuilder& cb, const Config& cfg) {
    const int D = cfg.embed_dim, p = cfg.patch_size, md = cfg.mlp_dim();
    const int ng = cfg.native_grid();
    const std::string pre = "backbone.";
    cb.add(pre + "embeddings.patch_embeddings.projection.weight", {D, cfg.in_chans, p, p});
    cb.add(pre + "embeddings.patch_embeddings.projection.bias",   {D});
    cb.add(pre + "embeddings.cls_token",            {1, 1, D});
    cb.add(pre + "embeddings.position_embeddings",  {1, 1 + ng * ng, D});
    for (int i = 0; i < cfg.depth; ++i) {
        const std::string lp = pre + "encoder.layer." + std::to_string(i) + ".";
        cb.add(lp + "norm1.weight", {D}); cb.add(lp + "norm1.bias", {D});
        cb.add(lp + "attention.attention.query.weight", {D, D});
        cb.add(lp + "attention.attention.query.bias",   {D});
        cb.add(lp + "attention.attention.key.weight",   {D, D});
        cb.add(lp + "attention.attention.key.bias",     {D});
        cb.add(lp + "attention.attention.value.weight", {D, D});
        cb.add(lp + "attention.attention.value.bias",   {D});
        cb.add(lp + "attention.output.dense.weight",    {D, D});
        cb.add(lp + "attention.output.dense.bias",      {D});
        cb.add(lp + "layer_scale1.lambda1",             {D});
        cb.add(lp + "norm2.weight", {D}); cb.add(lp + "norm2.bias", {D});
        cb.add(lp + "mlp.fc1.weight", {md, D}); cb.add(lp + "mlp.fc1.bias", {md});
        cb.add(lp + "mlp.fc2.weight", {D, md}); cb.add(lp + "mlp.fc2.bias", {D});
        cb.add(lp + "layer_scale2.lambda1",             {D});
    }
    cb.add(pre + "layernorm.weight", {D}); cb.add(pre + "layernorm.bias", {D});
}

brotensor::Tensor make_pixels(int C, int H, int W) {
    brotensor::Tensor px = brotensor::Tensor::mat(1, C * H * W);
    float* p = px.host_f32_mut();
    for (int i = 0; i < px.cols; ++i) p[i] = std::sin(static_cast<float>(i) * 0.002f);
    return px;
}

}  // namespace

int main() {
    // ── Config presets carry the documented ViT-S/B/L hyperparameters ──
    {
        Config s = Config::vit_s();
        check(s.embed_dim == 384 && s.depth == 12 && s.num_heads == 6, "vit_s dims");
        Config b = Config::vit_b();
        check(b.embed_dim == 768 && b.depth == 12 && b.num_heads == 12, "vit_b dims");
        Config l = Config::vit_l();
        check(l.embed_dim == 1024 && l.depth == 24 && l.num_heads == 16, "vit_l dims");
        check(l.out_stages == std::vector<int>({5, 12, 18, 24}), "vit_l out_stages");
        check(s.native_grid() == 37 && s.head_dim() == 64, "vit_s derived dims");
    }

    // ── Tiny backbone: 28px image, patch 14 -> native 2x2 grid, 16-dim/2-head,
    //    4 blocks, stages {2,4} collected. ──
    Config cfg;
    cfg.embed_dim = 16; cfg.depth = 4; cfg.num_heads = 2;
    cfg.patch_size = 14; cfg.img_size = 28; cfg.in_chans = 3;
    cfg.mlp_ratio = 2.0f; cfg.out_stages = {2, 4};

    CheckpointBuilder cb;
    build(cb, cfg);
    const std::string path = "dinov2_test.safetensors";
    st::write_file(path, cb.entries);

    Backbone bb(cfg);
    bb.load_file(path);

    // Native grid (28x28 -> 2x2): no pos-embed interpolation.
    {
        brotensor::Tensor px = make_pixels(3, 28, 28);
        BackboneOutput o = bb.encode(px, 28, 28);
        check(o.patch_h == 2 && o.patch_w == 2, "native patch grid 2x2");
        check(o.feature_maps.size() == 2, "two stages collected");
        bool ok = true, finite = true;
        for (const auto& fm : o.feature_maps) {
            if (fm.rows != 1 + 2 * 2 || fm.cols != cfg.embed_dim) ok = false;
            const float* d = fm.host_f32();
            for (int i = 0; i < fm.cols * fm.rows; ++i)
                if (!std::isfinite(d[i])) finite = false;
        }
        check(ok, "feature maps are (1+gh*gw, D)");
        check(finite, "feature maps all-finite");
    }

    // Non-native grid (42x28 -> 3x2): exercises pos-embed interpolation.
    BackboneOutput o_cpu;
    brotensor::Tensor px = make_pixels(3, 42, 28);
    {
        o_cpu = bb.encode(px, 42, 28);
        check(o_cpu.patch_h == 3 && o_cpu.patch_w == 2, "interp patch grid 3x2");
        check(o_cpu.feature_maps[0].rows == 1 + 3 * 2, "interp feature map rows");
    }

    // ── encode() rejects a non-multiple-of-patch input ──
    {
        bool threw = false;
        try { bb.encode(make_pixels(3, 30, 28), 30, 28); }
        catch (const std::exception&) { threw = true; }
        check(threw, "encode rejects non-multiple-of-patch H");
    }

    // ── CUDA parity on the 42x28 (interpolated) case ──
    brotensor::init();
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        bb.to(brotensor::Device::CUDA);
        check(bb.device() == brotensor::Device::CUDA, "to(CUDA) moved weights");
        brotensor::Tensor px_gpu = px.to(brotensor::Device::CUDA);
        BackboneOutput o_gpu = bb.encode(px_gpu, 42, 28);
        float worst = 0.0f;
        for (size_t s = 0; s < o_cpu.feature_maps.size(); ++s) {
            brotensor::Tensor back = o_gpu.feature_maps[s].to(brotensor::Device::CPU);
            const float* a = o_cpu.feature_maps[s].host_f32();
            const float* b = back.host_f32();
            const int n = o_cpu.feature_maps[s].rows * o_cpu.feature_maps[s].cols;
            for (int i = 0; i < n; ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
        }
        if (worst > 1e-3f) {
            std::fprintf(stderr, "FAIL: CPU/CUDA dinov2 diff %g > 1e-3\n", worst);
            ++failures;
        } else {
            std::printf("  CUDA parity OK (worst diff %g)\n", worst);
        }
    } else {
        std::printf("  (CUDA not available — parity check skipped)\n");
    }

    if (failures == 0) { std::printf("test_dinov2: OK\n"); return 0; }
    std::printf("test_dinov2: %d failure(s)\n", failures);
    return 1;
}
