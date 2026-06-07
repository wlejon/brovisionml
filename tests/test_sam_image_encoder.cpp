// SAM ViT image encoder: structural / end-to-end CPU test with no real
// checkpoint. We synthesize a tiny HF-named SAM checkpoint (a few-channel,
// 2-block encoder), write it with brotensor's safetensors writer, then load it
// through the real loader and run encode(). This exercises the full path —
// HF weight naming, the combined-qkv split, windowed + global attention, and
// the neck — and asserts the image-embedding shape and finiteness.
#include "brovisionml/sam_image_encoder.h"

#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"
#include "brotensor/runtime.h"

#include "test_device.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace st = brotensor::safetensors;

namespace {

int failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

// Stable backing store for the float blobs the WriteEntry views point into
// (std::deque keeps element addresses stable across push_back).
struct CheckpointBuilder {
    std::deque<std::vector<float>> store;
    std::vector<st::WriteEntry>    entries;

    void add(const std::string& name, std::vector<int64_t> shape) {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        std::vector<float> buf(static_cast<std::size_t>(n));

        const bool is_ln = name.find("layer_norm") != std::string::npos;
        const bool is_w  = name.size() >= 7 &&
                           name.compare(name.size() - 7, 7, ".weight") == 0;
        for (std::size_t i = 0; i < buf.size(); ++i) {
            if (is_ln) buf[i] = is_w ? 1.0f : 0.0f;             // gamma=1, beta=0
            else       buf[i] = (static_cast<float>(i % 7) - 3.0f) * 0.02f;
        }
        store.push_back(std::move(buf));
        const std::vector<float>& b = store.back();
        entries.push_back(st::WriteEntry{name, st::Dtype::F32, std::move(shape),
                                         b.data(), b.size() * sizeof(float)});
    }
};

}  // namespace

int main() {
    using namespace brovisionml::sam;

    // ── Config presets carry the documented ViT-H/L/B hyperparameters ──
    {
        EncoderConfig h = EncoderConfig::vit_h();
        check(h.embed_dim == 1280 && h.depth == 32 && h.num_heads == 16,
              "vit_h dims");
        check(h.global_attn_indexes == std::vector<int>({7, 15, 23, 31}),
              "vit_h global indexes");
        check(h.grid() == 64 && h.seq_len() == 4096 && h.head_dim() == 80,
              "vit_h derived dims");

        EncoderConfig l = EncoderConfig::vit_l();
        check(l.embed_dim == 1024 && l.depth == 24 && l.num_heads == 16,
              "vit_l dims");
        EncoderConfig b = EncoderConfig::vit_b();
        check(b.embed_dim == 768 && b.depth == 12 && b.num_heads == 12,
              "vit_b dims");
    }

    // ── Tiny encoder: 64px image, patch 16 -> 4x4 grid, window 2 (4 windows),
    //    8-dim/2-head, 2 blocks with block 1 global. Exercises both attention
    //    paths and the neck. ──
    EncoderConfig cfg;
    cfg.img_size    = 64;
    cfg.patch_size  = 16;       // grid = 4, seq_len = 16
    cfg.in_chans    = 3;
    cfg.embed_dim   = 8;
    cfg.depth       = 2;
    cfg.num_heads   = 2;        // head_dim = 4
    cfg.mlp_ratio   = 2.0f;     // mlp_dim = 16
    cfg.out_chans   = 4;
    cfg.window_size = 2;
    cfg.global_attn_indexes = {1};

    const int D  = cfg.embed_dim;     // 8
    const int g  = cfg.grid();        // 4
    const int hd = cfg.head_dim();    // 4
    const int md = cfg.mlp_dim();     // 16
    const int oc = cfg.out_chans;     // 4
    const int p  = cfg.patch_size;    // 16

    CheckpointBuilder cb;
    const std::string pre = "vision_encoder.";
    cb.add(pre + "patch_embed.projection.weight", {D, cfg.in_chans, p, p});
    cb.add(pre + "patch_embed.projection.bias",   {D});
    cb.add(pre + "pos_embed",                      {1, g, g, D});

    for (int i = 0; i < cfg.depth; ++i) {
        const std::string lp = pre + "layers." + std::to_string(i) + ".";
        const bool global = (i == 1);
        const int  rg = global ? g : cfg.window_size;  // rel-pos grid side
        cb.add(lp + "layer_norm1.weight", {D});
        cb.add(lp + "layer_norm1.bias",   {D});
        cb.add(lp + "attn.qkv.weight",    {3 * D, D});
        cb.add(lp + "attn.qkv.bias",      {3 * D});
        cb.add(lp + "attn.proj.weight",   {D, D});
        cb.add(lp + "attn.proj.bias",     {D});
        cb.add(lp + "attn.rel_pos_h",     {2 * rg - 1, hd});
        cb.add(lp + "attn.rel_pos_w",     {2 * rg - 1, hd});
        cb.add(lp + "layer_norm2.weight", {D});
        cb.add(lp + "layer_norm2.bias",   {D});
        cb.add(lp + "mlp.lin1.weight",    {md, D});
        cb.add(lp + "mlp.lin1.bias",      {md});
        cb.add(lp + "mlp.lin2.weight",    {D, md});
        cb.add(lp + "mlp.lin2.bias",      {D});
    }

    cb.add(pre + "neck.conv1.weight",       {oc, D, 1, 1});
    cb.add(pre + "neck.layer_norm1.weight", {oc});
    cb.add(pre + "neck.layer_norm1.bias",   {oc});
    cb.add(pre + "neck.conv2.weight",       {oc, oc, 3, 3});
    cb.add(pre + "neck.layer_norm2.weight", {oc});
    cb.add(pre + "neck.layer_norm2.bias",   {oc});

    const std::string path = "sam_encoder_test.safetensors";
    st::write_file(path, cb.entries);

    ImageEncoder enc(cfg);
    enc.load_file(path);

    check(enc.config().grid() == 4 && enc.config().head_dim() == 4 &&
          enc.config().mlp_dim() == 16,
          "tiny config derived dims");

    // A finite, non-trivial pixel tensor (1, 3*64*64) in normalized range.
    brotensor::Tensor pixels = brotensor::Tensor::mat(1, cfg.in_chans * cfg.img_size * cfg.img_size);
    {
        float* px = pixels.host_f32_mut();
        for (int i = 0; i < pixels.cols; ++i)
            px[i] = std::sin(static_cast<float>(i) * 0.001f);
    }

    brotensor::Tensor emb = enc.encode(pixels);

    check(emb.rows == 1 && emb.cols == oc * g * g,
          "image embedding is (1, out_chans*grid*grid)");
    check(emb.device == brotensor::Device::CPU && emb.dtype == brotensor::Dtype::FP32,
          "image embedding is host FP32");

    bool finite = true;
    {
        const float* e = emb.host_f32();
        for (int i = 0; i < emb.cols; ++i)
            if (!std::isfinite(e[i])) finite = false;
    }
    check(finite, "image embedding is all-finite");

    // ── encode() rejects a wrong-shaped input ──
    {
        bool threw = false;
        try {
            brotensor::Tensor bad = brotensor::Tensor::mat(1, 10);
            enc.encode(bad);
        } catch (const std::exception&) { threw = true; }
        check(threw, "encode rejects a mis-shaped pixel tensor");
    }

    // ── CUDA parity: the same encoder on the GPU must match the CPU result.
    //    Exercises the device-neutral path end to end, including the new
    //    windowed decomposed-rel-pos attention op on CUDA. Skipped cleanly when
    //    no GPU is present. ──
    brotensor::init();
    const brotensor::Device gpu = brovisionml_test::preferred_gpu();
    if (gpu != brotensor::Device::CPU) {
        const char* dev = brovisionml_test::device_name(gpu);
        enc.to(gpu);
        check(enc.device() == gpu, "to(gpu) moved weights");

        brotensor::Tensor pixels_gpu = pixels.to(gpu);
        brotensor::Tensor emb_gpu = enc.encode(pixels_gpu);
        check(emb_gpu.device == gpu,
              "GPU embedding stays on the GPU");

        brotensor::Tensor emb_back = emb_gpu.to(brotensor::Device::CPU);
        check(emb_back.rows == emb.rows && emb_back.cols == emb.cols,
              "GPU embedding has the CPU shape");
        float worst = 0.0f;
        const float* a = emb.host_f32();
        const float* b = emb_back.host_f32();
        for (int i = 0; i < emb.cols; ++i)
            worst = std::max(worst, std::fabs(a[i] - b[i]));
        if (worst > 1e-3f) {
            std::fprintf(stderr, "FAIL: CPU/%s embedding diff %g > 1e-3\n", dev, worst);
            ++failures;
        }
        std::printf("sam_image_encoder: %s parity max abs diff %g\n", dev, worst);
    } else {
        std::printf("sam_image_encoder: no GPU device, GPU parity skipped\n");
    }

    std::remove(path.c_str());

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("sam_image_encoder: all checks passed\n");
    return 0;
}
