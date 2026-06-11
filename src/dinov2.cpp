#include "brovisionml/dinov2.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include "weights_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brovisionml::dinov2 {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "dinov2::Backbone: ";

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

// Fold a per-output-channel scale `s` (rows,1) into a linear layer's weight
// (rows,cols) and bias (rows,1), in place — used to absorb DINOv2's LayerScale
// into the preceding output projection so the forward is a plain pre-norm ViT.
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

}  // namespace

// ─── Config presets ─────────────────────────────────────────────────────────

Config Config::vit_s() { return Config{}; }  // defaults are ViT-S/14

Config Config::vit_b() {
    Config c;
    c.embed_dim = 768;
    c.depth     = 12;
    c.num_heads = 12;
    c.out_stages = {3, 6, 9, 12};
    return c;
}

Config Config::vit_l() {
    Config c;
    c.embed_dim = 1024;
    c.depth     = 24;
    c.num_heads = 16;
    c.out_stages = {5, 12, 18, 24};   // Depth-Anything-V2-Large intermediates
    return c;
}

// ─── Weight tables (host FP32) ────────────────────────────────────────────────

namespace {
struct BlockWeights {
    Tensor ln1_w, ln1_b;                 // (D,1)
    Tensor q_w, q_b, k_w, k_b, v_w, v_b; // (D,D) / (D,1)
    Tensor o_w, o_b;                     // (D,D) / (D,1)  — LayerScale1 folded in
    Tensor ln2_w, ln2_b;                 // (D,1)
    Tensor fc1_w, fc1_b;                 // (md,D) / (md,1)
    Tensor fc2_w, fc2_b;                 // (D,md) / (D,1) — LayerScale2 folded in
};
}  // namespace

struct Backbone::Weights {
    bool   loaded = false;
    Tensor patch_w, patch_b;             // (D, in*p*p) / (D,1)
    Tensor cls_token;                    // (1, D)
    Tensor cls_pos;                      // (1, D)         position embed (cls)
    Tensor patch_pos;                    // (native^2, D)  position embed (patches)
    Tensor final_ln_w, final_ln_b;       // (D,1)
    std::vector<BlockWeights> blocks;
};

// ─── Construction ─────────────────────────────────────────────────────────────

Backbone::Backbone(Config cfg)
    : cfg_(std::move(cfg)), w_(std::make_unique<Weights>()) {
    if (cfg_.embed_dim % cfg_.num_heads != 0)
        fail("embed_dim must be divisible by num_heads");
}

Backbone::~Backbone() = default;
Backbone::Backbone(Backbone&&) noexcept = default;
Backbone& Backbone::operator=(Backbone&&) noexcept = default;

void Backbone::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void Backbone::load_file(const std::string& path) {
    st::File f = st::File::open(path);

    const int D  = cfg_.embed_dim;
    const int p  = cfg_.patch_size;
    const int md = cfg_.mlp_dim();
    const int ng = cfg_.native_grid();
    const std::string pre = "backbone.";

    Weights w;

    // Patch embed conv (D, in, p, p) -> OIHW (D, in*p*p); bias (D,1).
    w.patch_w = load_whole(f, kWho, pre + "embeddings.patch_embeddings.projection.weight",
                           D, cfg_.in_chans * p * p);
    w.patch_b = load_whole(f, kWho, pre + "embeddings.patch_embeddings.projection.bias",
                           D, 1);

    // cls token (1,1,D) -> (1,D).
    w.cls_token = load_whole(f, kWho, pre + "embeddings.cls_token", 1, D);

    // Position embedding (1, 1+native^2, D): split row 0 (cls) from the patch
    // grid so the patch part can be bicubically interpolated to a non-native
    // input grid (DINOv2 interpolate_pos_encoding).
    {
        Tensor pos = load_whole(f, kWho, pre + "embeddings.position_embeddings",
                                1 + ng * ng, D);
        const float* src = pos.host_f32();
        w.cls_pos   = Tensor::mat(1, D);
        w.patch_pos = Tensor::mat(ng * ng, D);
        std::copy(src, src + D, w.cls_pos.host_f32_mut());
        std::copy(src + D, src + static_cast<std::size_t>(1 + ng * ng) * D,
                  w.patch_pos.host_f32_mut());
    }

    w.blocks.resize(cfg_.depth);
    for (int i = 0; i < cfg_.depth; ++i) {
        BlockWeights& b = w.blocks[i];
        const std::string lp = pre + "encoder.layer." + std::to_string(i) + ".";

        b.ln1_w = load_whole(f, kWho, lp + "norm1.weight", D, 1);
        b.ln1_b = load_whole(f, kWho, lp + "norm1.bias",   D, 1);

        b.q_w = load_whole(f, kWho, lp + "attention.attention.query.weight", D, D);
        b.q_b = load_whole(f, kWho, lp + "attention.attention.query.bias",   D, 1);
        b.k_w = load_whole(f, kWho, lp + "attention.attention.key.weight",   D, D);
        b.k_b = load_whole(f, kWho, lp + "attention.attention.key.bias",     D, 1);
        b.v_w = load_whole(f, kWho, lp + "attention.attention.value.weight", D, D);
        b.v_b = load_whole(f, kWho, lp + "attention.attention.value.bias",   D, 1);

        b.o_w = load_whole(f, kWho, lp + "attention.output.dense.weight", D, D);
        b.o_b = load_whole(f, kWho, lp + "attention.output.dense.bias",   D, 1);
        Tensor ls1 = load_whole(f, kWho, lp + "layer_scale1.lambda1", D, 1);
        fold_scale(b.o_w, b.o_b, ls1);   // absorb LayerScale1 into Wo / bo

        b.ln2_w = load_whole(f, kWho, lp + "norm2.weight", D, 1);
        b.ln2_b = load_whole(f, kWho, lp + "norm2.bias",   D, 1);

        b.fc1_w = load_whole(f, kWho, lp + "mlp.fc1.weight", md, D);
        b.fc1_b = load_whole(f, kWho, lp + "mlp.fc1.bias",   md, 1);
        b.fc2_w = load_whole(f, kWho, lp + "mlp.fc2.weight", D, md);
        b.fc2_b = load_whole(f, kWho, lp + "mlp.fc2.bias",   D, 1);
        Tensor ls2 = load_whole(f, kWho, lp + "layer_scale2.lambda1", D, 1);
        fold_scale(b.fc2_w, b.fc2_b, ls2);   // absorb LayerScale2 into fc2

        (void)i;
    }

    w.final_ln_w = load_whole(f, kWho, pre + "layernorm.weight", D, 1);
    w.final_ln_b = load_whole(f, kWho, pre + "layernorm.bias",   D, 1);

    w.loaded = true;
    *w_ = std::move(w);
}

// ─── Migration ────────────────────────────────────────────────────────────────

void Backbone::to(brotensor::Device dev) {
    if (!w_->loaded) fail("to() called before load()");
    if (dev == device_) return;
    // Mixed precision on a GPU backend (the DINOv3 pattern, taken one step
    // further): every block projection — q/k/v, o_proj, fc1, fc2 — runs FP16;
    // the residual stream itself, the LayerNorms and the embeddings stay FP32.
    // Unlike DINOv3 ViT-H there are no massive activations here, and the
    // folded LayerScale (gamma << 1) shrinks the residual-writing outputs, so
    // o/fc2 are safe in half (the real-checkpoint depth parity test guards
    // this). CPU stays all-FP32 (the exact parity path); to(CPU) widens back
    // via cast.
    const bool fp16 = dev != brotensor::Device::CPU &&
                      brotensor::compute_dtype() == brotensor::Dtype::FP16;
    auto to_dt = [&](Tensor& t, brotensor::Dtype want) {
        if (!t.data) return;
        t = t.to(dev);
        if (t.dtype != want) { Tensor c; brotensor::cast(t, c, want); t = std::move(c); }
    };
    const brotensor::Dtype half = fp16 ? brotensor::Dtype::FP16
                                       : brotensor::Dtype::FP32;
    auto mv   = [&](Tensor& t) { to_dt(t, brotensor::Dtype::FP32); };
    auto mv16 = [&](Tensor& t) { to_dt(t, half); };
    mv16(w_->patch_w); mv16(w_->patch_b);
    mv(w_->cls_token); mv(w_->cls_pos); mv(w_->patch_pos);
    mv(w_->final_ln_w); mv(w_->final_ln_b);
    for (BlockWeights& b : w_->blocks) {
        mv(b.ln1_w); mv(b.ln1_b);
        mv16(b.q_w); mv16(b.q_b); mv16(b.k_w); mv16(b.k_b); mv16(b.v_w); mv16(b.v_b);
        mv16(b.o_w); mv16(b.o_b);
        mv(b.ln2_w); mv(b.ln2_b);
        mv16(b.fc1_w); mv16(b.fc1_b); mv16(b.fc2_w); mv16(b.fc2_b);
    }
    device_ = dev;
    fp16_ = fp16;
}

// ─── Forward ──────────────────────────────────────────────────────────────────

namespace {

// Absolute position embedding for a (gh, gw) patch grid: cls position row +
// (bicubically interpolated, if the grid differs from native) patch positions.
// align_corners=False bicubic with a=-0.75 (interp2d mode 3), matching DINOv2
// interpolate_pos_encoding — torch.nn.functional.interpolate(mode="bicubic").
// The native-grid branch is bit-exact (HF returns the stored embedding as-is
// when the grid matches); the interpolated branch is the non-518 input path.
Tensor position_embedding(const Tensor& cls_pos, const Tensor& patch_pos_native,
                          int D, int ng, int gh, int gw) {
    Tensor patch_pos;
    if (gh == ng && gw == ng) {
        patch_pos = patch_pos_native;   // (ng*ng, D), shared storage — read-only
    } else {
        Tensor nchw;
        brotensor::sequence_to_nchw(patch_pos_native, 1, D, ng, ng, nchw);
        Tensor resized;
        brotensor::interp2d_forward(nchw, 1, D, ng, ng, gh, gw,
                                    /*bicubic a=-0.75 (torch)=*/3, resized);
        brotensor::nchw_to_sequence(resized, 1, D, gh, gw, patch_pos);
    }
    Tensor pos;
    brotensor::concat_rows({&cls_pos, &patch_pos}, pos);  // (1+gh*gw, D)
    return pos;
}

}  // namespace

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

    const int D  = cfg_.embed_dim;
    const int gh = H / p, gw = W / p;
    const float eps = cfg_.layer_norm_eps;

    // Mixed precision (fp16_, GPU only): the residual stream and LayerNorms
    // stay FP32; LayerNorm'd inputs drop to FP16 to feed the q/k/v + fc1
    // GEMMs and the flash attention, then widen as they re-enter the
    // residual. No-op pass-throughs when fp16_ is off.
    auto to16 = [&](const Tensor& t) {
        if (!fp16_) return t;
        Tensor c; brotensor::cast(t, c, brotensor::Dtype::FP16); return c;
    };
    auto to32 = [&](const Tensor& t) {
        if (!fp16_) return t;
        Tensor c; brotensor::cast(t, c, brotensor::Dtype::FP32); return c;
    };
    auto linear16 = [&](const Tensor& Wt, const Tensor& bias, const Tensor& X,
                        Tensor& Y) {
        if (fp16_) brotensor::linear_forward_batched_fp16(Wt, &bias, X, Y);
        else       brotensor::linear_forward_batched(Wt, bias, X, Y);
    };

    // 1. Patch embed -> (1, D*gh*gw), then token-major (gh*gw, D).
    Tensor px = to16(pixels);
    Tensor feat;
    brotensor::conv2d_forward(px, w_->patch_w, &w_->patch_b,
                              /*N=*/1, cfg_.in_chans, H, W,
                              /*C_out=*/D, p, p, /*stride=*/p, p,
                              /*pad=*/0, 0, /*dil=*/1, 1, feat);
    Tensor patch_seq;
    brotensor::nchw_to_sequence(feat, 1, D, gh, gw, patch_seq);
    Tensor patch_tokens = to32(patch_seq);

    // 2. Prepend cls token, add the (interpolated) position embedding. concat_rows
    //    yields a flat (K*D,1) buffer whose row-major bytes already ARE the
    //    (K, D) token matrix, so we view it with the proper 2-D shape (the owning
    //    tensors stay alive for the whole forward).
    const int K = 1 + gh * gw;
    Tensor x_owned;
    brotensor::concat_rows({&w_->cls_token, &patch_tokens}, x_owned);
    Tensor x = Tensor::view(device_, x_owned.data, K, D);

    Tensor pos_owned = position_embedding(w_->cls_pos, w_->patch_pos, D,
                                          cfg_.native_grid(), gh, gw);
    Tensor pos = Tensor::view(device_, pos_owned.data, K, D);
    brotensor::add_inplace(x, pos);

    // 3. Pre-norm transformer blocks; collect the requested stages.
    auto wants_stage = [&](int stage) {
        return std::find(cfg_.out_stages.begin(), cfg_.out_stages.end(), stage)
               != cfg_.out_stages.end();
    };

    BackboneOutput out;
    out.patch_h = gh;
    out.patch_w = gw;
    out.feature_maps.reserve(cfg_.out_stages.size());

    // Throwaway attention caches (forward-only, FP32 path).
    Tensor Qh, Kh, Vh, Attnh, Yc;

    // cu_seqlens for the single-sequence varlen flash attention (FP16 path):
    // [0, K], built host-side then migrated alongside the activations.
    const int hd = D / cfg_.num_heads;
    Tensor cu;
    if (fp16_) {
        cu = Tensor::zeros_on(brotensor::Device::CPU, 2, 1,
                              brotensor::Dtype::INT32);
        static_cast<int32_t*>(cu.data)[1] = K;
        cu = cu.to(device_);
    }

    for (int i = 0; i < cfg_.depth; ++i) {
        const BlockWeights& b = w_->blocks[i];

        // Attention: x = x + LS1·attn(LN1(x)).  LS1 is folded into Wo/bo.
        Tensor h;
        brotensor::layernorm_forward_inference_batched(x, b.ln1_w, b.ln1_b, h, eps);
        Tensor attn;
        if (fp16_) {
            Tensor hc = to16(h);
            Tensor q, k, v;
            linear16(b.q_w, b.q_b, hc, q);
            linear16(b.k_w, b.k_b, hc, k);
            linear16(b.v_w, b.v_b, hc, v);
            Tensor ao;
            brotensor::flash_attention_varlen_forward(
                q, k, v, static_cast<const int32_t*>(cu.data),
                static_cast<const int32_t*>(cu.data),
                /*batch_size=*/1, /*max_seqlen_q=*/K, /*max_seqlen_k=*/K,
                cfg_.num_heads, hd, /*causal=*/false, ao);
            Tensor o16;
            linear16(b.o_w, b.o_b, ao, o16);
            attn = to32(o16);
        } else {
            brotensor::mha_forward(h, b.q_w, b.k_w, b.v_w, b.o_w,
                                   &b.q_b, &b.k_b, &b.v_b, &b.o_b,
                                   /*d_mask=*/nullptr, cfg_.num_heads,
                                   Qh, Kh, Vh, Attnh, Yc, attn);
        }
        brotensor::add_inplace(x, attn);

        // MLP: x = x + LS2·fc2(gelu(fc1(LN2(x)))).  LS2 is folded into fc2.
        Tensor h2;
        brotensor::layernorm_forward_inference_batched(x, b.ln2_w, b.ln2_b, h2, eps);
        Tensor h2c = to16(h2);
        Tensor m1;
        linear16(b.fc1_w, b.fc1_b, h2c, m1);
        Tensor act;
        brotensor::gelu_exact_forward(m1, act);
        Tensor m2;
        linear16(b.fc2_w, b.fc2_b, act, m2);
        Tensor m2f = to32(m2);
        brotensor::add_inplace(x, m2f);

        if (wants_stage(i + 1)) {
            Tensor fm;
            brotensor::layernorm_forward_inference_batched(
                x, w_->final_ln_w, w_->final_ln_b, fm, eps);
            out.feature_maps.push_back(std::move(fm));
        }
    }

    if (static_cast<int>(out.feature_maps.size()) !=
        static_cast<int>(cfg_.out_stages.size()))
        fail("an out_stage exceeds the block count");

    return out;
}

}  // namespace brovisionml::dinov2
