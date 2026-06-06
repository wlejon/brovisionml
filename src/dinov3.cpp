#include "brovisionml/dinov3.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include "weights_util.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brovisionml::dinov3 {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "dinov3::Backbone: ";

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

// Fold a per-output-channel scale `s` (rows,1) into a linear layer's weight
// (rows,cols) and bias (rows,1), in place — absorbs DINOv3's LayerScale into the
// preceding output projection so the forward is a plain pre-norm ViT (same trick
// as the dinov2 loader).
void fold_scale(Tensor& w, Tensor& b, const Tensor& s) {
    const int rows = w.rows, cols = w.cols;
    float* wp = w.host_f32_mut();
    float* bp = b.host_f32_mut();
    const float* sp = s.host_f32();
    for (int r = 0; r < rows; ++r) {
        const float sc = sp[r];
        float* row = wp + static_cast<std::size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) row[c] *= sc;
        bp[r] *= sc;
    }
}

// Reorder the rows of a q/k projection within each head from HF's rotate-half
// (GPT-NeoX) RoPE pairing to the interleaved pairing brotensor::rope_apply uses.
// HF rotates the pair (dim j, dim j+head_dim/2) of each head together; rope_apply
// rotates the adjacent pair (dim 2j, dim 2j+1). Mapping output dim j -> 2j and
// j+half -> 2j+1 (per head) makes a plain interleaved rope_apply reproduce
// rotate-half exactly. Attention is invariant to this permutation as long as q
// and k share it (and v / o_proj are left untouched), so only the q/k projection
// rows move. `b` may be null (DINOv3's key projection is unbiased).
void permute_qk_rows(Tensor& w, Tensor* b, int num_heads, int head_dim) {
    const int cols = w.cols;
    const int half = head_dim / 2;
    Tensor wn = Tensor::mat(w.rows, cols);
    float* dst = wn.host_f32_mut();
    const float* src = w.host_f32();
    float* bdst = nullptr;
    const float* bsrc = nullptr;
    Tensor bn;
    if (b) {
        bn = Tensor::mat(b->rows, 1);
        bdst = bn.host_f32_mut();
        bsrc = b->host_f32();
    }
    for (int h = 0; h < num_heads; ++h) {
        const int base = h * head_dim;
        for (int j = 0; j < half; ++j) {
            const float* s0 = src + static_cast<std::size_t>(base + j) * cols;
            const float* s1 = src + static_cast<std::size_t>(base + j + half) * cols;
            float* d0 = dst + static_cast<std::size_t>(base + 2 * j) * cols;
            float* d1 = dst + static_cast<std::size_t>(base + 2 * j + 1) * cols;
            for (int c = 0; c < cols; ++c) { d0[c] = s0[c]; d1[c] = s1[c]; }
            if (b) {
                bdst[base + 2 * j]     = bsrc[base + j];
                bdst[base + 2 * j + 1] = bsrc[base + j + half];
            }
        }
    }
    w = std::move(wn);
    if (b) *b = std::move(bn);
}

}  // namespace

// ─── Config presets ──────────────────────────────────────────────────────────

Config Config::vit_h() { return Config{}; }  // defaults are ViT-H+/16

// ─── Weight tables (host FP32) ────────────────────────────────────────────────

namespace {
struct BlockWeights {
    Tensor n1_w, n1_b;                   // (D,1) LayerNorm
    Tensor q_w, q_b, k_w, k_b, v_w, v_b; // (D,D) / (D,1)  — q/k rows permuted for RoPE
    Tensor o_w, o_b;                     // (D,D) / (D,1)  — LayerScale1 folded in
    Tensor n2_w, n2_b;                   // (D,1) LayerNorm
    Tensor gate_w, gate_b;               // (md,D) / (md,1)
    Tensor up_w, up_b;                   // (md,D) / (md,1)
    Tensor down_w, down_b;               // (D,md) / (D,1) — LayerScale2 folded in
};
}  // namespace

struct Backbone::Weights {
    bool   loaded = false;
    Tensor patch_w, patch_b;             // (D, in*p*p) / (D,1)
    Tensor cls_token;                    // (1, D)
    Tensor register_tokens;              // (R, D)
    Tensor final_ln_w, final_ln_b;       // (D,1)
    std::vector<BlockWeights> blocks;
};

// ─── Construction ─────────────────────────────────────────────────────────────

Backbone::Backbone(Config cfg)
    : cfg_(std::move(cfg)), w_(std::make_unique<Weights>()) {
    if (cfg_.embed_dim % cfg_.num_heads != 0)
        fail("embed_dim must be divisible by num_heads");
    if (cfg_.head_dim() % 4 != 0)
        fail("head_dim must be divisible by 4 (axial RoPE pairs)");
}

Backbone::~Backbone() = default;
Backbone::Backbone(Backbone&&) noexcept = default;
Backbone& Backbone::operator=(Backbone&&) noexcept = default;

void Backbone::load(const std::string& dir) {
    load_file(dir + "/dino_v3_vit_h.safetensors");
}

void Backbone::load_file(const std::string& path) {
    st::File f = st::File::open(path);

    const int D  = cfg_.embed_dim;
    const int p  = cfg_.patch_size;
    const int md = cfg_.intermediate_size;
    const int R  = cfg_.num_register_tokens;
    const int nh = cfg_.num_heads;
    const int hd = cfg_.head_dim();

    Weights w;

    // The VAST-AI/TripoSplat bundle stores the DINOv3 weights with no namespace
    // prefix (embeddings.* / layer.N.* / norm.*), i.e. an HF DINOv3ViTModel state
    // dict with the encoder's leading "model." stripped.
    w.patch_w = load_whole(f, kWho, "embeddings.patch_embeddings.weight",
                           D, cfg_.in_chans * p * p);
    w.patch_b = load_whole(f, kWho, "embeddings.patch_embeddings.bias", D, 1);
    w.cls_token = load_whole(f, kWho, "embeddings.cls_token", 1, D);
    if (R > 0)
        w.register_tokens = load_whole(f, kWho, "embeddings.register_tokens", R, D);

    w.blocks.resize(cfg_.depth);
    for (int i = 0; i < cfg_.depth; ++i) {
        BlockWeights& b = w.blocks[i];
        const std::string lp = "layer." + std::to_string(i) + ".";

        b.n1_w = load_whole(f, kWho, lp + "norm1.weight", D, 1);
        b.n1_b = load_whole(f, kWho, lp + "norm1.bias",   D, 1);

        // query/value/output are biased; key is not (HF key_bias=False) — use a
        // zero bias so the projection path is uniform. q/k rows are permuted to
        // turn rotate-half RoPE into the interleaved form rope_apply applies.
        b.q_w = load_whole(f, kWho, lp + "attention.q_proj.weight", D, D);
        b.q_b = load_whole(f, kWho, lp + "attention.q_proj.bias",   D, 1);
        b.k_w = load_whole(f, kWho, lp + "attention.k_proj.weight", D, D);
        b.k_b = Tensor::mat(D, 1);  // zero (key projection is unbiased)
        b.v_w = load_whole(f, kWho, lp + "attention.v_proj.weight", D, D);
        b.v_b = load_whole(f, kWho, lp + "attention.v_proj.bias",   D, 1);
        permute_qk_rows(b.q_w, &b.q_b, nh, hd);
        permute_qk_rows(b.k_w, nullptr, nh, hd);

        b.o_w = load_whole(f, kWho, lp + "attention.o_proj.weight", D, D);
        b.o_b = load_whole(f, kWho, lp + "attention.o_proj.bias",   D, 1);
        Tensor ls1 = load_whole(f, kWho, lp + "layer_scale1.lambda1", D, 1);
        fold_scale(b.o_w, b.o_b, ls1);

        b.n2_w = load_whole(f, kWho, lp + "norm2.weight", D, 1);
        b.n2_b = load_whole(f, kWho, lp + "norm2.bias",   D, 1);

        b.gate_w = load_whole(f, kWho, lp + "mlp.gate_proj.weight", md, D);
        b.gate_b = load_whole(f, kWho, lp + "mlp.gate_proj.bias",   md, 1);
        b.up_w   = load_whole(f, kWho, lp + "mlp.up_proj.weight",   md, D);
        b.up_b   = load_whole(f, kWho, lp + "mlp.up_proj.bias",     md, 1);
        b.down_w = load_whole(f, kWho, lp + "mlp.down_proj.weight", D, md);
        b.down_b = load_whole(f, kWho, lp + "mlp.down_proj.bias",   D, 1);
        Tensor ls2 = load_whole(f, kWho, lp + "layer_scale2.lambda1", D, 1);
        fold_scale(b.down_w, b.down_b, ls2);
    }

    w.final_ln_w = load_whole(f, kWho, "norm.weight", D, 1);
    w.final_ln_b = load_whole(f, kWho, "norm.bias",   D, 1);

    w.loaded = true;
    *w_ = std::move(w);
}

// ─── Migration ────────────────────────────────────────────────────────────────

void Backbone::to(brotensor::Device dev) {
    if (!w_->loaded) fail("to() called before load()");
    if (dev == device_) return;
    // Mixed precision on a GPU backend: the projection weights run FP16
    // tensor-core GEMMs (the speed win), but the residual stream and every
    // LayerNorm stay FP32. DINOv3 ViT-H has "massive activations" whose magnitude
    // runs into the thousands — well past FP16's 65504 ceiling — so an FP16
    // residual would store them as inf and flood the output with NaNs. The
    // LayerNorm params and token embeddings therefore live on the FP32 side (they
    // touch the FP32 residual directly); only the matmul weights go to half.
    // CPU stays all-FP32 (its exact parity path). cast is registered both
    // directions on CPU and CUDA, so to(CPU) after a GPU stint widens back.
    const bool fp16 = dev != brotensor::Device::CPU &&
                      brotensor::compute_dtype() == brotensor::Dtype::FP16;
    auto to_dt = [&](Tensor& t, brotensor::Dtype want) {
        if (!t.data) return;
        t = t.to(dev);
        if (t.dtype != want) { Tensor c; brotensor::cast(t, c, want); t = std::move(c); }
    };
    const brotensor::Dtype proj = fp16 ? brotensor::Dtype::FP16 : brotensor::Dtype::FP32;
    auto mv32 = [&](Tensor& t) { to_dt(t, brotensor::Dtype::FP32); };  // FP32 residual side
    auto mv16 = [&](Tensor& t) { to_dt(t, proj); };                   // FP16 GEMM weights
    // The four projections that read from the LayerNorm'd (bounded) stream stay
    // FP16 — q/k/v/gate/up — as does the patch embed. The two that WRITE the
    // residual — o_proj and down_proj — carry the LayerScale-amplified massive
    // activations (down_proj's output hits ~1e5 by layer 22, well over FP16's
    // 65504 ceiling, and would store as inf), so they run FP32.
    mv16(w_->patch_w); mv16(w_->patch_b);
    mv32(w_->cls_token); mv32(w_->register_tokens);
    mv32(w_->final_ln_w); mv32(w_->final_ln_b);
    for (BlockWeights& b : w_->blocks) {
        mv32(b.n1_w); mv32(b.n1_b);
        mv16(b.q_w); mv16(b.q_b); mv16(b.k_w); mv16(b.k_b); mv16(b.v_w); mv16(b.v_b);
        mv32(b.o_w); mv32(b.o_b);
        mv32(b.n2_w); mv32(b.n2_b);
        mv16(b.gate_w); mv16(b.gate_b); mv16(b.up_w); mv16(b.up_b);
        mv32(b.down_w); mv32(b.down_b);
    }
    device_ = dev;
    fp16_ = fp16;
}

// ─── RoPE tables ──────────────────────────────────────────────────────────────

namespace {

// Build the (K, head_dim/2) cos/sin tables for one forward: identity (cos=1,
// sin=0) for the num_prefix prefix tokens, and DINOv3's 2D axial angles for the
// gh*gw patch tokens (row-major / h-major). The half-vector of a patch is
// [coord_h * inv_freq_0..q-1, coord_w * inv_freq_0..q-1] with q = head_dim/4 and
// inv_freq_m = theta^(-4m/head_dim); coord_a = 2*(idx+0.5)/grid - 1 in [-1,1].
// These feed rope_apply against the head-permuted q/k (see permute_qk_rows).
void build_rope_tables(int K, int num_prefix, int gh, int gw, int head_dim,
                       float theta, Tensor& cos_out, Tensor& sin_out) {
    const int half = head_dim / 2;   // pairs per head
    const int q    = head_dim / 4;   // axial frequencies per coordinate
    std::vector<float> inv_freq(q);
    for (int m = 0; m < q; ++m)
        inv_freq[m] = std::pow(theta, -static_cast<float>(4 * m) / head_dim);

    cos_out = Tensor::mat(K, half);
    sin_out = Tensor::mat(K, half);
    float* cp = cos_out.host_f32_mut();
    float* sp = sin_out.host_f32_mut();

    const float two_pi = 6.28318530717958647692f;
    for (int r = 0; r < K; ++r) {
        float* cr = cp + static_cast<std::size_t>(r) * half;
        float* sr = sp + static_cast<std::size_t>(r) * half;
        if (r < num_prefix) {
            for (int j = 0; j < half; ++j) { cr[j] = 1.0f; sr[j] = 0.0f; }
            continue;
        }
        const int t  = r - num_prefix;       // patch index
        const int ph = t / gw, pw = t % gw;
        const float coord_h = 2.0f * (ph + 0.5f) / gh - 1.0f;
        const float coord_w = 2.0f * (pw + 0.5f) / gw - 1.0f;
        for (int m = 0; m < q; ++m) {
            const float ah = two_pi * coord_h * inv_freq[m];
            const float aw = two_pi * coord_w * inv_freq[m];
            cr[m]     = std::cos(ah); sr[m]     = std::sin(ah);
            cr[q + m] = std::cos(aw); sr[q + m] = std::sin(aw);
        }
    }
}

// Dtype-aware projection GEMM: FP16 weights/activations on the GPU path, FP32 on
// CPU. The other dispatched ops the forward uses (conv2d, nchw_to_sequence,
// concat, rope_apply, flash-attn, swiglu) pick their kernel from the tensor
// dtype, so they need no fork; LayerNorm always runs FP32 (the residual is FP32).
void linear_b(bool fp16, const Tensor& W, const Tensor& bias,
              const Tensor& X, Tensor& Y) {
    if (fp16) brotensor::linear_forward_batched_fp16(W, &bias, X, Y);
    else      brotensor::linear_forward_batched(W, bias, X, Y);
}

// Half-precision copy of a (range-bounded) FP32 activation, for feeding an FP16
// GEMM; widen back the same way after. No-op pass-through when fp16 is off.
Tensor to16(bool fp16, const Tensor& t) {
    if (!fp16) return t;
    Tensor c; brotensor::cast(t, c, brotensor::Dtype::FP16); return c;
}
Tensor to32(bool fp16, const Tensor& t) {
    if (!fp16) return t;
    Tensor c; brotensor::cast(t, c, brotensor::Dtype::FP32); return c;
}

}  // namespace

// ─── Forward ──────────────────────────────────────────────────────────────────

BackboneOutput Backbone::encode(const brotensor::Tensor& pixels,
                                int H, int W) const {
    if (!w_->loaded) fail("encode() called before load()");
    if (pixels.dtype != brotensor::Dtype::FP32)
        fail("encode() requires an FP32 pixel tensor");
    if (pixels.device != device_)
        fail("encode() pixel tensor must be on the backbone's device");
    const int p = cfg_.patch_size;
    if (H % p != 0 || W % p != 0)
        fail("H and W must be multiples of patch_size");
    if (pixels.rows != 1 || pixels.cols != cfg_.in_chans * H * W)
        fail("pixels must be (1, in_chans*H*W)");

    const int D   = cfg_.embed_dim;
    const int nh  = cfg_.num_heads;
    const int hd  = cfg_.head_dim();
    const int gh  = H / p, gw = W / p;
    const int np  = gh * gw;
    const int pre = cfg_.num_prefix_tokens();
    const int K   = pre + np;
    const float eps = cfg_.layer_norm_eps;

    // 0. The patch-embed conv runs in the projection dtype (FP16 on GPU); its
    //    input image is range-bounded, so half precision is safe here. Everything
    //    downstream of the embedding — the residual stream — is FP32.
    Tensor px16;
    if (fp16_) brotensor::cast(pixels, px16, brotensor::Dtype::FP16);
    const Tensor& px = fp16_ ? px16 : pixels;

    // 1. Patch embed -> token-major (np, D), then widen to the FP32 residual.
    Tensor feat;
    brotensor::conv2d_forward(px, w_->patch_w, &w_->patch_b,
                              /*N=*/1, cfg_.in_chans, H, W,
                              /*C_out=*/D, p, p, /*stride=*/p, p,
                              /*pad=*/0, 0, /*dil=*/1, 1, feat);
    Tensor patch_seq;
    brotensor::nchw_to_sequence(feat, 1, D, gh, gw, patch_seq);
    Tensor patch_tokens = to32(fp16_, patch_seq);  // FP32 token embeddings

    // 2. Assemble [cls, register..., patches] -> (K, D). concat_rows yields a flat
    //    (K*D,1) buffer whose row-major bytes already are the (K, D) matrix.
    std::vector<const Tensor*> parts = {&w_->cls_token};
    if (cfg_.num_register_tokens > 0) parts.push_back(&w_->register_tokens);
    parts.push_back(&patch_tokens);
    Tensor x_owned;
    brotensor::concat_rows(parts, x_owned);
    Tensor x = Tensor::view(device_, x_owned.data, K, D, x_owned.dtype);

    // 3. RoPE cos/sin tables for the patch tokens, on the compute device.
    Tensor cos_tbl, sin_tbl;
    build_rope_tables(K, pre, gh, gw, hd, cfg_.rope_theta, cos_tbl, sin_tbl);
    if (device_ != brotensor::Device::CPU) {
        cos_tbl = cos_tbl.to(device_);
        sin_tbl = sin_tbl.to(device_);
    }

    // cu_seqlens for the single-sequence varlen attention: [0, K]. Built host-side
    // (Tensor::zeros would land on the default device, which is the GPU after
    // brotensor::init()) then migrated to the compute device.
    Tensor cu = Tensor::zeros_on(brotensor::Device::CPU, 2, 1, brotensor::Dtype::INT32);
    static_cast<int32_t*>(cu.data)[0] = 0;
    static_cast<int32_t*>(cu.data)[1] = K;
    if (device_ != brotensor::Device::CPU) cu = cu.to(device_);
    const int32_t* cu_p = static_cast<const int32_t*>(cu.data);

    // 4. Pre-norm transformer blocks.
    for (int i = 0; i < cfg_.depth; ++i) {
        const BlockWeights& b = w_->blocks[i];

        // Attention: x = x + LS1·o_proj(attn(rope(q),rope(k),v)).  LS1 in o_w/o_b.
        // LN(x) is FP32 and range-bounded; drop to FP16 only to feed q/k/v. The
        // attention output is a convex mix of (bounded) V, but o_proj carries LS1
        // so its result re-enters the residual at full FP32 width.
        Tensor h1;
        brotensor::layernorm_forward_inference_batched(x, b.n1_w, b.n1_b, h1, eps);
        Tensor h1c = to16(fp16_, h1);
        Tensor q, k, v;
        linear_b(fp16_, b.q_w, b.q_b, h1c, q);
        linear_b(fp16_, b.k_w, b.k_b, h1c, k);
        linear_b(fp16_, b.v_w, b.v_b, h1c, v);
        Tensor qr, kr;
        brotensor::rope_apply(q, cos_tbl, sin_tbl, hd, nh, qr);
        brotensor::rope_apply(k, cos_tbl, sin_tbl, hd, nh, kr);
        Tensor attn;
        brotensor::flash_attention_varlen_forward(qr, kr, v, cu_p, cu_p,
                                                  /*batch_size=*/1,
                                                  /*max_seqlen_q=*/K,
                                                  /*max_seqlen_k=*/K,
                                                  nh, hd, /*causal=*/false, attn);
        Tensor attn32 = to32(fp16_, attn);
        Tensor o;
        linear_b(/*fp16=*/false, b.o_w, b.o_b, attn32, o);    // FP32 o_proj
        brotensor::add_inplace(x, o);                         // FP32 residual

        // MLP: x = x + LS2·down(silu(gate(LN2(x))) * up(LN2(x))).  LS2 in down.
        // gate/up read bounded LN(x) → FP16; down_proj writes the residual and
        // carries LS2 + the massive activations (~1e5), so it runs FP32.
        Tensor h2;
        brotensor::layernorm_forward_inference_batched(x, b.n2_w, b.n2_b, h2, eps);
        Tensor h2c = to16(fp16_, h2);
        Tensor gate, up;
        linear_b(fp16_, b.gate_w, b.gate_b, h2c, gate);
        linear_b(fp16_, b.up_w, b.up_b, h2c, up);
        Tensor swin;
        brotensor::concat_batched_rows({&gate, &up}, swin);  // per row [gate|up]
        Tensor act;
        brotensor::swiglu_forward(swin, act);
        Tensor act32 = to32(fp16_, act);
        Tensor down;
        linear_b(/*fp16=*/false, b.down_w, b.down_b, act32, down);  // FP32 down_proj
        brotensor::add_inplace(x, down);                      // FP32 residual
    }

    // 5. Final LayerNorm (HF apply_layernorm=True). Output is FP32: the residual
    //    and the final norm are FP32, and the normalized result is range-safe.
    BackboneOutput out;
    brotensor::layernorm_forward_inference_batched(
        x, w_->final_ln_w, w_->final_ln_b, out.last_hidden_state, eps);
    out.patch_h = gh;
    out.patch_w = gw;
    out.num_prefix_tokens = pre;
    return out;
}

}  // namespace brovisionml::dinov3
