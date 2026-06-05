// DINOv3 ViT-H backbone: structural CPU test (always) + numeric parity against
// the HF Python reference (gated on the real checkpoint + golden dump).
//
// Structural part: synthesize a tiny DINOv3-named checkpoint (a 2-block, 16-dim,
// 2-register ViT), write it with brotensor's safetensors writer, load it through
// the real loader, and run encode(). This exercises the full path — HF DINOv3
// weight naming, the q/k RoPE row-permutation, the LayerScale fold, the
// [cls,register,patch] assembly, axial-RoPE attention, and the SwiGLU FFN —
// asserting token-sequence shape, finiteness, and CPU/CUDA parity.
//
// Parity part: compares brovisionml's last_hidden_state to a golden dump of the
// actual HuggingFace `DINOv3ViTModel` (run in FP32 on the same pixel tensor,
// generated out-of-repo — Python is never part of the brovisionml stack). The
// golden lives under weights/triposplat/clip_vision/golden (gitignored, like the
// checkpoint); when absent the test prints why and exits 0 (clean skip).

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/dinov3.h"

#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"
#include "brotensor/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

namespace st = brotensor::safetensors;
using brovisionml::dinov3::Backbone;
using brovisionml::dinov3::BackboneOutput;
using brovisionml::dinov3::Config;

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

// ── Tiny synthetic checkpoint (real DINOv3 tensor names) ──
struct CheckpointBuilder {
    std::deque<std::vector<float>> store;
    std::vector<st::WriteEntry>    entries;

    void add(const std::string& name, std::vector<int64_t> shape) {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        std::vector<float> buf(static_cast<std::size_t>(n));
        const bool is_norm  = name.find("norm") != std::string::npos;
        const bool is_scale = name.find("layer_scale") != std::string::npos;
        const bool is_w = name.size() >= 7 &&
                          name.compare(name.size() - 7, 7, ".weight") == 0;
        for (std::size_t i = 0; i < buf.size(); ++i) {
            if (is_norm) buf[i] = is_w ? 1.0f : 0.0f;
            else if (is_scale) buf[i] = 0.4f + 0.01f * (i % 5);
            else buf[i] = (static_cast<float>(i % 7) - 3.0f) * 0.03f;
        }
        store.push_back(std::move(buf));
        const std::vector<float>& b = store.back();
        entries.push_back(st::WriteEntry{name, st::Dtype::F32, std::move(shape),
                                         b.data(), b.size() * sizeof(float)});
    }
};

void build(CheckpointBuilder& cb, const Config& cfg) {
    const int D = cfg.embed_dim, p = cfg.patch_size, md = cfg.intermediate_size;
    const int R = cfg.num_register_tokens;
    cb.add("embeddings.patch_embeddings.weight", {D, cfg.in_chans, p, p});
    cb.add("embeddings.patch_embeddings.bias",   {D});
    cb.add("embeddings.cls_token",      {1, 1, D});
    cb.add("embeddings.register_tokens", {1, R, D});
    for (int i = 0; i < cfg.depth; ++i) {
        const std::string lp = "layer." + std::to_string(i) + ".";
        cb.add(lp + "norm1.weight", {D}); cb.add(lp + "norm1.bias", {D});
        cb.add(lp + "attention.q_proj.weight", {D, D});
        cb.add(lp + "attention.q_proj.bias",   {D});
        cb.add(lp + "attention.k_proj.weight", {D, D});  // no k bias
        cb.add(lp + "attention.v_proj.weight", {D, D});
        cb.add(lp + "attention.v_proj.bias",   {D});
        cb.add(lp + "attention.o_proj.weight", {D, D});
        cb.add(lp + "attention.o_proj.bias",   {D});
        cb.add(lp + "layer_scale1.lambda1",    {D});
        cb.add(lp + "norm2.weight", {D}); cb.add(lp + "norm2.bias", {D});
        cb.add(lp + "mlp.gate_proj.weight", {md, D}); cb.add(lp + "mlp.gate_proj.bias", {md});
        cb.add(lp + "mlp.up_proj.weight",   {md, D}); cb.add(lp + "mlp.up_proj.bias",   {md});
        cb.add(lp + "mlp.down_proj.weight", {D, md}); cb.add(lp + "mlp.down_proj.bias", {D});
        cb.add(lp + "layer_scale2.lambda1", {D});
    }
    cb.add("norm.weight", {D}); cb.add("norm.bias", {D});
}

brotensor::Tensor make_pixels(int C, int H, int W) {
    brotensor::Tensor px = brotensor::Tensor::mat(1, C * H * W);
    float* p = px.host_f32_mut();
    for (int i = 0; i < px.cols; ++i) p[i] = std::sin(static_cast<float>(i) * 0.002f);
    return px;
}

// ── Golden-dump reader (format BVMLD3G1; see gen_dinov3_golden.py) ──
struct Golden {
    int H = 0, W = 0, D = 0, K = 0;
    std::vector<float> pixels;  // 3*H*W NCHW
    std::vector<float> hidden;  // K*D
};

template <class T>
bool read_vec(std::ifstream& f, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(f);
}

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 8) != "BVMLD3G1") return false;
    int version = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&g.H), 4);
    f.read(reinterpret_cast<char*>(&g.W), 4);
    f.read(reinterpret_cast<char*>(&g.D), 4);
    f.read(reinterpret_cast<char*>(&g.K), 4);
    if (!f) return false;
    if (!read_vec(f, g.pixels, static_cast<std::size_t>(3) * g.H * g.W)) return false;
    if (!read_vec(f, g.hidden, static_cast<std::size_t>(g.K) * g.D)) return false;
    return true;
}

struct Diff { double max_abs = 0.0, mean_abs = 0.0; };
Diff diff(const float* a, const float* b, std::size_t n) {
    Diff d; double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        d.max_abs = std::max(d.max_abs, e); sum += e;
    }
    d.mean_abs = n ? sum / static_cast<double>(n) : 0.0;
    return d;
}

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary); return f.good();
}

}  // namespace

int main() {
    // ── Config preset carries the ViT-H+/16 hyperparameters ──
    {
        Config h = Config::vit_h();
        check(h.embed_dim == 1280 && h.depth == 32 && h.num_heads == 20, "vit_h dims");
        check(h.head_dim() == 64 && h.num_prefix_tokens() == 5, "vit_h derived dims");
        check(h.patch_size == 16 && h.num_register_tokens == 4, "vit_h patch/registers");
    }

    // ── Tiny backbone: 32px image, patch 16 -> 2x2 grid, 16-dim/2-head, 2 regs,
    //    2 blocks. K = 1 + 2 + 4 = 7 tokens. ──
    Config cfg;
    cfg.embed_dim = 16; cfg.depth = 2; cfg.num_heads = 2;
    cfg.patch_size = 16; cfg.num_register_tokens = 2; cfg.intermediate_size = 32;
    cfg.in_chans = 3;

    CheckpointBuilder cb;
    build(cb, cfg);
    const std::string path = "dinov3_test.safetensors";
    st::write_file(path, cb.entries);

    Backbone bb(cfg);
    bb.load_file(path);

    BackboneOutput o_cpu;
    brotensor::Tensor px = make_pixels(3, 32, 32);
    {
        o_cpu = bb.encode(px, 32, 32);
        check(o_cpu.patch_h == 2 && o_cpu.patch_w == 2, "patch grid 2x2");
        check(o_cpu.num_prefix_tokens == 3, "prefix tokens = 1 + 2 registers");
        check(o_cpu.last_hidden_state.rows == 7 &&
              o_cpu.last_hidden_state.cols == cfg.embed_dim,
              "hidden state (1+R+gh*gw, D)");
        const float* d = o_cpu.last_hidden_state.host_f32();
        bool finite = true;
        for (int i = 0; i < o_cpu.last_hidden_state.rows * o_cpu.last_hidden_state.cols; ++i)
            if (!std::isfinite(d[i])) finite = false;
        check(finite, "hidden state all-finite");
    }

    // ── encode() rejects a non-multiple-of-patch input ──
    {
        bool threw = false;
        try { bb.encode(make_pixels(3, 40, 32), 40, 32); }
        catch (const std::exception&) { threw = true; }
        check(threw, "encode rejects non-multiple-of-patch H");
    }

    brotensor::init();

    // ── CUDA parity on the tiny model ──
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        bb.to(brotensor::Device::CUDA);
        check(bb.device() == brotensor::Device::CUDA, "to(CUDA) moved weights");
        brotensor::Tensor px_gpu = px.to(brotensor::Device::CUDA);
        BackboneOutput o_gpu = bb.encode(px_gpu, 32, 32);
        brotensor::Tensor back = o_gpu.last_hidden_state.to(brotensor::Device::CPU);
        Diff dg = diff(o_cpu.last_hidden_state.host_f32(), back.host_f32(),
                       static_cast<std::size_t>(o_cpu.last_hidden_state.rows) *
                           o_cpu.last_hidden_state.cols);
        if (dg.max_abs > 1e-2) {
            std::fprintf(stderr, "FAIL: CPU/CUDA dinov3 diff %g > 1e-2\n", dg.max_abs);
            ++failures;
        } else {
            std::printf("  CUDA parity OK (worst diff %g)\n", dg.max_abs);
        }
        bb.to(brotensor::Device::CPU);
    } else {
        std::printf("  (CUDA not available — parity check skipped)\n");
    }

    // ── Numeric parity against the real HF DINOv3 (gated on checkpoint+golden) ──
    {
        const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
        const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
        const std::string cdir = base + "/triposplat/clip_vision";
        const std::string gpath = cdir + "/golden/golden_dinov3.bin";
        if (!file_exists(cdir + "/dino_v3_vit_h.safetensors") || !file_exists(gpath)) {
            std::printf("test_dinov3: no checkpoint+golden under '%s/golden' — skipping "
                        "HF parity (generate with the out-of-repo gen_dinov3_golden.py).\n",
                        cdir.c_str());
        } else {
            Golden g;
            check(load_golden(gpath, g), "load dinov3 golden");
            std::printf("HF parity: %dx%d, D=%d, K=%d\n", g.H, g.W, g.D, g.K);

            Backbone real(Config::vit_h());
            real.load_file(cdir + "/dino_v3_vit_h.safetensors");

            brotensor::Tensor px_real = brotensor::Tensor::mat(1, 3 * g.H * g.W);
            std::copy(g.pixels.begin(), g.pixels.end(), px_real.host_f32_mut());
            BackboneOutput ro = real.encode(px_real, g.H, g.W);
            check(ro.last_hidden_state.rows == g.K && ro.last_hidden_state.cols == g.D,
                  "real hidden state shape matches golden");
            Diff dh = diff(ro.last_hidden_state.host_f32(), g.hidden.data(),
                           std::min<std::size_t>(g.hidden.size(),
                               static_cast<std::size_t>(ro.last_hidden_state.rows) *
                                   ro.last_hidden_state.cols));
            // FP32 round-off across 32 layers of widened-BF16 weights: a real
            // transcription bug moves this by orders of magnitude, not factors.
            std::printf("  last_hidden_state: max=%.3e mean=%.3e\n", dh.max_abs, dh.mean_abs);
            check(dh.max_abs < 2e-2, "DINOv3 HF parity (last_hidden_state)");
        }
    }

    if (failures == 0) { std::printf("test_dinov3: OK\n"); return 0; }
    std::printf("test_dinov3: %d failure(s)\n", failures);
    return 1;
}
