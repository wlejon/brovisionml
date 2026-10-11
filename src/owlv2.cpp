#include "brovisionml/owlv2.h"

#include "brotensor/ops.h"
#include "brotensor/ops/fused.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include "profile.h"
#include "weights_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace brovisionml::owlv2 {

namespace {

using brotensor::Dtype;
using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "owlv2::Detector: ";

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

// One pre-norm CLIP encoder block (vision or text).
struct Block {
    Tensor ln1_w, ln1_b, ln2_w, ln2_b;   // (D,1)
    Tensor q_w, q_b, k_w, k_b, v_w, v_b; // (D,D) / (D,1)
    Tensor o_w, o_b;                     // (D,D) / (D,1)
    Tensor fc1_w, fc1_b;                 // (F,D) / (F,1)
    Tensor fc2_w, fc2_b;                 // (D,F) / (D,1)
};

Block load_block(const st::File& f, const std::string& p, int D, int F) {
    Block b;
    b.ln1_w = load_whole(f, kWho, p + "layer_norm1.weight", D, 1);
    b.ln1_b = load_whole(f, kWho, p + "layer_norm1.bias", D, 1);
    b.q_w = load_whole(f, kWho, p + "self_attn.q_proj.weight", D, D);
    b.q_b = load_whole(f, kWho, p + "self_attn.q_proj.bias", D, 1);
    b.k_w = load_whole(f, kWho, p + "self_attn.k_proj.weight", D, D);
    b.k_b = load_whole(f, kWho, p + "self_attn.k_proj.bias", D, 1);
    b.v_w = load_whole(f, kWho, p + "self_attn.v_proj.weight", D, D);
    b.v_b = load_whole(f, kWho, p + "self_attn.v_proj.bias", D, 1);
    b.o_w = load_whole(f, kWho, p + "self_attn.out_proj.weight", D, D);
    b.o_b = load_whole(f, kWho, p + "self_attn.out_proj.bias", D, 1);
    b.ln2_w = load_whole(f, kWho, p + "layer_norm2.weight", D, 1);
    b.ln2_b = load_whole(f, kWho, p + "layer_norm2.bias", D, 1);
    b.fc1_w = load_whole(f, kWho, p + "mlp.fc1.weight", F, D);
    b.fc1_b = load_whole(f, kWho, p + "mlp.fc1.bias", F, 1);
    b.fc2_w = load_whole(f, kWho, p + "mlp.fc2.weight", D, F);
    b.fc2_b = load_whole(f, kWho, p + "mlp.fc2.bias", D, 1);
    return b;
}

// A 3-layer box/objectness MLP: dense0, GELU, dense1, GELU, dense2.
struct Mlp {
    Tensor w0, b0, w1, b1, w2, b2;
};

Mlp load_mlp(const st::File& f, const std::string& p, int D, int out) {
    Mlp m;
    m.w0 = load_whole(f, kWho, p + "dense0.weight", D, D);
    m.b0 = load_whole(f, kWho, p + "dense0.bias", D, 1);
    m.w1 = load_whole(f, kWho, p + "dense1.weight", D, D);
    m.b1 = load_whole(f, kWho, p + "dense1.bias", D, 1);
    m.w2 = load_whole(f, kWho, p + "dense2.weight", out, D);
    m.b2 = load_whole(f, kWho, p + "dense2.bias", out, 1);
    return m;
}

// Y = act(X W^T + b), FP32 (or Y += ... with accumulate).
void linear(const Tensor& W, const Tensor* b, const Tensor& X, Tensor& Y,
            int act = brotensor::kLinearActNone, bool accumulate = false) {
    brotensor::linear_forward_batched_ex(W, b, X, act,
                                         accumulate ? brotensor::kLinearEpiAccumulate
                                                    : brotensor::kLinearEpiStore,
                                         nullptr, Y);
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

Config Config::base16() { return Config{}; }

struct Detector::Impl {
    bool loaded = false;
    brotensor::Device device = brotensor::Device::CPU;
    bool fp16 = false;   // vision block weights are FP16 (GPU mixed precision)
    Tokenizer tok;

    // Vision tower.
    Tensor patch_w;                      // (D, 3*p*p), no bias
    Tensor class_emb;                    // (1, D)
    Tensor pos;                          // (1+P, D)
    Tensor pre_ln_w, pre_ln_b, post_ln_w, post_ln_b;
    std::vector<Block> vblocks;

    // Detection heads.
    Tensor det_ln_w, det_ln_b;           // layer_norm over class-token-merged patches
    Tensor cls_w, cls_b;                 // class_head.dense0 (E, D)
    Tensor ss_w, ss_b;                   // logit_shift ; logit_scale stacked (2, D)
    Mlp box, obj;
    std::vector<float> box_bias;         // (P, 4) host

    // Text tower. The token table stays on the host: a query is <= 16 rows.
    Tensor tok_emb;                      // (vocab, T) host
    Tensor text_pos;                     // (16, T) host
    std::vector<Block> tblocks;
    Tensor final_ln_w, final_ln_b;
    Tensor text_proj;                    // (E, T), no bias
};

Detector::Detector(Config cfg) : cfg_(cfg), impl_(std::make_unique<Impl>()) {}
Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

brotensor::Device Detector::device() const { return impl_->device; }
const Tokenizer& Detector::tokenizer() const {
    if (!impl_->loaded) fail("tokenizer() before load()");
    return impl_->tok;
}

void Detector::load(const std::string& dir) {
    namespace fs = std::filesystem;
    const fs::path d(dir);
    for (const char* name : {"model.safetensors", "vocab.json", "merges.txt"})
        if (!fs::exists(d / name)) fail("missing " + (d / name).string());

    Impl w;
    w.tok = Tokenizer::load((d / "vocab.json").string(), (d / "merges.txt").string());
    st::File f = st::File::open((d / "model.safetensors").string());

    const int D = cfg_.vision_width, p = cfg_.patch_size, P = cfg_.num_patches();
    const int T = cfg_.text_width, E = cfg_.embed_dim;
    const std::string vp = "owlv2.vision_model.", tp = "owlv2.text_model.";

    w.patch_w = load_whole(f, kWho, vp + "embeddings.patch_embedding.weight", D, 3 * p * p);
    w.class_emb = load_whole(f, kWho, vp + "embeddings.class_embedding", 1, D);
    w.pos = load_whole(f, kWho, vp + "embeddings.position_embedding.weight", 1 + P, D);
    w.pre_ln_w = load_whole(f, kWho, vp + "pre_layernorm.weight", D, 1);
    w.pre_ln_b = load_whole(f, kWho, vp + "pre_layernorm.bias", D, 1);
    w.post_ln_w = load_whole(f, kWho, vp + "post_layernorm.weight", D, 1);
    w.post_ln_b = load_whole(f, kWho, vp + "post_layernorm.bias", D, 1);
    for (int i = 0; i < cfg_.vision_layers; ++i)
        w.vblocks.push_back(load_block(f, vp + "encoder.layers." + std::to_string(i) + ".", D,
                                       cfg_.vision_mlp));

    w.det_ln_w = load_whole(f, kWho, "layer_norm.weight", D, 1);
    w.det_ln_b = load_whole(f, kWho, "layer_norm.bias", D, 1);
    w.cls_w = load_whole(f, kWho, "class_head.dense0.weight", E, D);
    w.cls_b = load_whole(f, kWho, "class_head.dense0.bias", E, 1);
    {
        Tensor sw = load_whole(f, kWho, "class_head.logit_shift.weight", 1, D);
        Tensor sb = load_whole(f, kWho, "class_head.logit_shift.bias", 1, 1);
        Tensor cw = load_whole(f, kWho, "class_head.logit_scale.weight", 1, D);
        Tensor cb = load_whole(f, kWho, "class_head.logit_scale.bias", 1, 1);
        w.ss_w = Tensor::mat(2, D);
        w.ss_b = Tensor::mat(2, 1);
        std::copy(sw.host_f32(), sw.host_f32() + D, w.ss_w.host_f32_mut());
        std::copy(cw.host_f32(), cw.host_f32() + D, w.ss_w.host_f32_mut() + D);
        w.ss_b.host_f32_mut()[0] = sb.host_f32()[0];
        w.ss_b.host_f32_mut()[1] = cb.host_f32()[0];
    }
    w.box = load_mlp(f, "box_head.", D, 4);
    w.obj = load_mlp(f, "objectness_head.", D, 1);

    w.tok_emb = load_whole(f, kWho, tp + "embeddings.token_embedding.weight", cfg_.vocab_size, T);
    w.text_pos = load_whole(f, kWho, tp + "embeddings.position_embedding.weight", cfg_.max_positions, T);
    for (int i = 0; i < cfg_.text_layers; ++i)
        w.tblocks.push_back(load_block(f, tp + "encoder.layers." + std::to_string(i) + ".", T,
                                       cfg_.text_mlp));
    w.final_ln_w = load_whole(f, kWho, tp + "final_layer_norm.weight", T, 1);
    w.final_ln_b = load_whole(f, kWho, tp + "final_layer_norm.bias", T, 1);
    w.text_proj = load_whole(f, kWho, "owlv2.text_projection.weight", E, T);

    // The box bias: each patch's box centre starts at its grid corner, its
    // size at one patch (HF compute_box_bias: inverse sigmoid, eps 1e-4).
    const int g = cfg_.grid();
    auto inv_sigmoid = [](double c) { return std::log(c + 1e-4) - std::log1p(-c + 1e-4); };
    w.box_bias.resize(static_cast<std::size_t>(P) * 4);
    for (int r = 0; r < g; ++r) {
        for (int c = 0; c < g; ++c) {
            float* b = w.box_bias.data() + (static_cast<std::size_t>(r) * g + c) * 4;
            b[0] = static_cast<float>(inv_sigmoid(std::min(1.0, (c + 1.0) / g)));
            b[1] = static_cast<float>(inv_sigmoid(std::min(1.0, (r + 1.0) / g)));
            b[2] = b[3] = static_cast<float>(inv_sigmoid(1.0 / g));
        }
    }

    w.loaded = true;
    *impl_ = std::move(w);
}

void Detector::to(brotensor::Device dev) {
    Impl& w = *impl_;
    if (!w.loaded) fail("to() before load()");
    if (dev == w.device) return;
    // Mixed precision on a GPU with an FP16 compute dtype: the vision block
    // projections and the patch embed run FP16; the residual stream, the
    // LayerNorms, the embeddings, the text tower and the heads stay FP32.
    const bool fp16 = dev != brotensor::Device::CPU &&
                      brotensor::compute_dtype() == Dtype::FP16;
    auto to_dt = [&](Tensor& t, Dtype want) {
        if (!t.data) return;
        t = t.to(dev);
        if (t.dtype != want) { Tensor c; brotensor::cast(t, c, want); t = std::move(c); }
    };
    auto mv = [&](Tensor& t) { to_dt(t, Dtype::FP32); };
    auto mv16 = [&](Tensor& t) { to_dt(t, fp16 ? Dtype::FP16 : Dtype::FP32); };
    auto block = [&](Block& b, bool half) {
        auto m = [&](Tensor& t) { if (half) mv16(t); else mv(t); };
        mv(b.ln1_w); mv(b.ln1_b); mv(b.ln2_w); mv(b.ln2_b);
        m(b.q_w); m(b.q_b); m(b.k_w); m(b.k_b); m(b.v_w); m(b.v_b);
        m(b.o_w); m(b.o_b); m(b.fc1_w); m(b.fc1_b); m(b.fc2_w); m(b.fc2_b);
    };
    mv16(w.patch_w);
    mv(w.class_emb); mv(w.pos);
    mv(w.pre_ln_w); mv(w.pre_ln_b); mv(w.post_ln_w); mv(w.post_ln_b);
    for (Block& b : w.vblocks) block(b, true);
    mv(w.det_ln_w); mv(w.det_ln_b); mv(w.cls_w); mv(w.cls_b); mv(w.ss_w); mv(w.ss_b);
    for (Mlp* m : {&w.box, &w.obj}) {
        mv(m->w0); mv(m->b0); mv(m->w1); mv(m->b1); mv(m->w2); mv(m->b2);
    }
    for (Block& b : w.tblocks) block(b, false);
    mv(w.final_ln_w); mv(w.final_ln_b); mv(w.text_proj);
    w.device = dev;
    w.fp16 = fp16;
}

// ─── Text tower ─────────────────────────────────────────────────────────────

std::vector<float> Detector::embed_text(const std::vector<std::string>& queries) const {
    if (!impl_->loaded) fail("embed_text() before load()");
    std::vector<std::vector<int32_t>> ids;
    ids.reserve(queries.size());
    for (const std::string& q : queries) ids.push_back(impl_->tok.encode(q, cfg_.max_positions));
    return embed_tokens(ids);
}

std::vector<float> Detector::embed_tokens(const std::vector<std::vector<int32_t>>& ids) const {
    const Impl& w = *impl_;
    if (!w.loaded) fail("embed_tokens() before load()");
    const int T = cfg_.text_width, E = cfg_.embed_dim;
    const float eps = cfg_.layer_norm_eps;
    std::vector<float> out;
    out.reserve(ids.size() * E);
    for (const std::vector<int32_t>& seq : ids) {
        const int L = static_cast<int>(seq.size());
        if (L < 1 || L > cfg_.max_positions) fail("a query must be 1..16 tokens");
        // Token rows + positions, gathered on the host.
        std::vector<float> x0(static_cast<std::size_t>(L) * T);
        const float* te = w.tok_emb.host_f32();
        const float* pe = w.text_pos.host_f32();
        for (int i = 0; i < L; ++i) {
            if (seq[i] < 0 || seq[i] >= cfg_.vocab_size) fail("token id out of range");
            for (int c = 0; c < T; ++c)
                x0[static_cast<std::size_t>(i) * T + c] =
                    te[static_cast<std::size_t>(seq[i]) * T + c] + pe[static_cast<std::size_t>(i) * T + c];
        }
        Tensor x = Tensor::from_host_on(w.device, x0.data(), L, T);
        Tensor h, q, k, v, ao, m;
        for (const Block& b : w.tblocks) {
            brotensor::layernorm_forward_inference_batched(x, b.ln1_w, b.ln1_b, h, eps);
            linear(b.q_w, &b.q_b, h, q);
            linear(b.k_w, &b.k_b, h, k);
            linear(b.v_w, &b.v_b, h, v);
            brotensor::flash_attention_gqa_forward(q, k, v, nullptr, cfg_.text_heads,
                                                   cfg_.text_heads, /*causal=*/true, ao);
            linear(b.o_w, &b.o_b, ao, x, brotensor::kLinearActNone, /*accumulate=*/true);
            brotensor::layernorm_forward_inference_batched(x, b.ln2_w, b.ln2_b, h, eps);
            linear(b.fc1_w, &b.fc1_b, h, m, brotensor::kLinearActQuickGelu);
            linear(b.fc2_w, &b.fc2_b, m, x, brotensor::kLinearActNone, /*accumulate=*/true);
        }
        // Pool at EOS: the last row (encode() never pads).
        Tensor eos = Tensor::empty_on(w.device, 1, T);
        brotensor::copy_d2d(x, (L - 1) * T, eos, 0, T);
        brotensor::layernorm_forward_inference_batched(eos, w.final_ln_w, w.final_ln_b, h, eps);
        Tensor proj;
        linear(w.text_proj, nullptr, h, proj);
        std::vector<float> e = proj.to_host_vector();
        double n = 0.0;
        for (float val : e) n += static_cast<double>(val) * val;
        const float inv = static_cast<float>(1.0 / std::sqrt(n));
        for (float val : e) out.push_back(val * inv);
    }
    return out;
}

// ─── Vision tower + heads ─────────────────────────────────────────────────────

Predictions Detector::predict(const Pixels& px, const float* query_embeds, int num_queries,
                              VisionTrace* trace) const {
    const Impl& w = *impl_;
    if (!w.loaded) fail("predict() before load()");
    const int S = cfg_.image_size, p = cfg_.patch_size, g = cfg_.grid();
    const int D = cfg_.vision_width, P = cfg_.num_patches(), K = 1 + P, E = cfg_.embed_dim;
    const int H = cfg_.vision_heads;
    const float eps = cfg_.layer_norm_eps;
    if (px.nchw.rows != 1 || px.nchw.cols != 3 * S * S) fail("pixels must be (1, 3*960*960)");
    if (num_queries < 0 || (num_queries > 0 && !query_embeds)) fail("bad query embeddings");
    const brotensor::Device dev = w.device;
    const bool fp16 = w.fp16;

    auto cast_to = [](const Tensor& t, Dtype dt) {
        if (t.dtype == dt) return t;
        Tensor c; brotensor::cast(t, c, dt); return c;
    };
    const Dtype half = fp16 ? Dtype::FP16 : Dtype::FP32;

    detail::profile_mark(dev, nullptr);
    // 1. Patch embed -> (P, D) tokens; class token + positions.
    Tensor pix = cast_to(px.nchw.to(dev), half);
    Tensor feat, seq;
    brotensor::conv2d_forward(pix, w.patch_w, nullptr, 1, 3, S, S, D, p, p, p, p, 0, 0, 1, 1, feat);
    brotensor::nchw_to_sequence(feat, 1, D, g, g, seq);
    Tensor patches = cast_to(seq, Dtype::FP32);
    Tensor x_owned;
    brotensor::concat_rows({&w.class_emb, &patches}, x_owned);
    Tensor x = Tensor::view(dev, x_owned.data, K, D);
    brotensor::add_inplace(x, w.pos);
    if (trace) trace->embeddings = x.to_host_vector();
    detail::profile_mark(dev, "owlv2 patch embed");

    Tensor xs;
    brotensor::layernorm_forward_inference_batched(x, w.pre_ln_w, w.pre_ln_b, xs, eps);
    x = xs;

    // 2. Pre-norm blocks. FP16 GEMMs + varlen flash attention on a GPU (FP32
    //    residual); FP32 everywhere on the CPU.
    Tensor cu;
    if (fp16) {
        cu = Tensor::zeros_on(brotensor::Device::CPU, 2, 1, Dtype::INT32);
        static_cast<int32_t*>(cu.data)[1] = K;
        cu = cu.to(dev);
    }
    Tensor h, q, k, v, ao, o, m, m2;
    for (int i = 0; i < cfg_.vision_layers; ++i) {
        const Block& b = w.vblocks[i];
        brotensor::layernorm_forward_inference_batched(x, b.ln1_w, b.ln1_b, h, eps);
        if (fp16) {
            // The FP16 error is spread over every projection and the
            // attention alike (keeping any one of them FP32 removes a few
            // percent of it at up to 3x the time), so all of them run half.
            const Tensor h16 = cast_to(h, Dtype::FP16);
            brotensor::linear_forward_batched_fp16(b.q_w, &b.q_b, h16, q);
            brotensor::linear_forward_batched_fp16(b.k_w, &b.k_b, h16, k);
            brotensor::linear_forward_batched_fp16(b.v_w, &b.v_b, h16, v);
            brotensor::flash_attention_varlen_forward(
                q, k, v, static_cast<const int32_t*>(cu.data), static_cast<const int32_t*>(cu.data),
                1, K, K, H, D / H, /*causal=*/false, ao);
            brotensor::linear_forward_batched_fp16(b.o_w, &b.o_b, ao, o);
            brotensor::add_inplace(x, o);
            brotensor::layernorm_forward_inference_batched(x, b.ln2_w, b.ln2_b, h, eps);
            brotensor::linear_forward_batched_fp16_act(b.fc1_w, &b.fc1_b, cast_to(h, Dtype::FP16),
                                                       brotensor::kLinearActQuickGelu, m);
            brotensor::linear_forward_batched_fp16(b.fc2_w, &b.fc2_b, m, m2);
            brotensor::add_inplace(x, m2);
        } else {
            linear(b.q_w, &b.q_b, h, q);
            linear(b.k_w, &b.k_b, h, k);
            linear(b.v_w, &b.v_b, h, v);
            brotensor::flash_attention_gqa_forward(q, k, v, nullptr, H, H, /*causal=*/false, ao);
            linear(b.o_w, &b.o_b, ao, x, brotensor::kLinearActNone, /*accumulate=*/true);
            brotensor::layernorm_forward_inference_batched(x, b.ln2_w, b.ln2_b, h, eps);
            linear(b.fc1_w, &b.fc1_b, h, m, brotensor::kLinearActQuickGelu);
            linear(b.fc2_w, &b.fc2_b, m, x, brotensor::kLinearActNone, /*accumulate=*/true);
        }
        if (i == 0 && trace) trace->block0 = x.to_host_vector();
    }
    if (trace) trace->last_hidden = x.to_host_vector();
    detail::profile_mark(dev, "owlv2 vision blocks");

    // 3. Image embeddings: post_layernorm, patches times the class token,
    //    layer_norm -> (P, D).
    Tensor y;
    brotensor::layernorm_forward_inference_batched(x, w.post_ln_w, w.post_ln_b, y, eps);
    Tensor cls = Tensor::empty_on(dev, 1, D), pt = Tensor::empty_on(dev, P, D), merged, feats;
    brotensor::copy_d2d(y, 0, cls, 0, D);
    brotensor::copy_d2d(y, D, pt, 0, P * D);
    brotensor::broadcast_mul(pt, cls, merged);
    brotensor::layernorm_forward_inference_batched(merged, w.det_ln_w, w.det_ln_b, feats, eps);
    if (trace) trace->image_embeds = feats.to_host_vector();

    // 4. Heads (FP32). Class: normalised dense0 . normalised queries; the
    //    per-box shift and scale are applied on the host below.
    Predictions out;
    out.num_patches = P;
    out.num_queries = num_queries;
    std::vector<float> logits;
    if (num_queries > 0) {
        Tensor ce, cen, lg;
        linear(w.cls_w, &w.cls_b, feats, ce);
        brotensor::l2_norm_forward(ce, E, 1, 1e-12f, cen);
        // HF normalises the query embeddings again (+1e-6 on the norm).
        std::vector<float> qn(query_embeds, query_embeds + static_cast<std::size_t>(num_queries) * E);
        for (int qi = 0; qi < num_queries; ++qi) {
            float* r = qn.data() + static_cast<std::size_t>(qi) * E;
            double n = 0.0;
            for (int c = 0; c < E; ++c) n += static_cast<double>(r[c]) * r[c];
            const float inv = static_cast<float>(1.0 / (std::sqrt(n) + 1e-6));
            for (int c = 0; c < E; ++c) r[c] *= inv;
        }
        Tensor qt = Tensor::from_host_on(dev, qn.data(), num_queries, E);
        linear(qt, nullptr, cen, lg);
        logits = lg.to_host_vector();
    }
    Tensor ss, b0, b1, braw, o0, o1, oraw;
    linear(w.ss_w, &w.ss_b, feats, ss);
    linear(w.box.w0, &w.box.b0, feats, b0, brotensor::kLinearActGeluExact);
    linear(w.box.w1, &w.box.b1, b0, b1, brotensor::kLinearActGeluExact);
    linear(w.box.w2, &w.box.b2, b1, braw);
    linear(w.obj.w0, &w.obj.b0, feats, o0, brotensor::kLinearActGeluExact);
    linear(w.obj.w1, &w.obj.b1, o0, o1, brotensor::kLinearActGeluExact);
    linear(w.obj.w2, &w.obj.b2, o1, oraw);
    const std::vector<float> shift_scale = ss.to_host_vector();
    out.boxes = braw.to_host_vector();
    out.objectness = oraw.to_host_vector();
    detail::profile_mark(dev, "owlv2 heads");

    // logit = (cos + shift) * (elu(scale) + 1); box = sigmoid(raw + bias).
    out.logits.resize(static_cast<std::size_t>(P) * num_queries);
    for (int i = 0; i < P; ++i) {
        const float shift = shift_scale[2 * i];
        const float sc = shift_scale[2 * i + 1];
        const float scale = (sc > 0.f ? sc : std::expm1(sc)) + 1.0f;
        for (int qi = 0; qi < num_queries; ++qi) {
            const std::size_t at = static_cast<std::size_t>(i) * num_queries + qi;
            out.logits[at] = (logits[at] + shift) * scale;
        }
    }
    for (std::size_t i = 0; i < out.boxes.size(); ++i)
        out.boxes[i] = sigmoid(out.boxes[i] + w.box_bias[i]);
    return out;
}

std::vector<Detection> Detector::decode(const Predictions& pred, const Pixels& px,
                                        const DetectOptions& opts) {
    const int P = pred.num_patches, Q = pred.num_queries;
    const float S = static_cast<float>(px.padded);
    const float W = static_cast<float>(px.width), H = static_cast<float>(px.height);
    std::vector<Detection> cand;
    if (Q <= 0) return cand;
    for (int i = 0; i < P; ++i) {
        const float* l = pred.logits.data() + static_cast<std::size_t>(i) * Q;
        const int best = static_cast<int>(std::max_element(l, l + Q) - l);
        const float score = sigmoid(l[best]);
        if (score < opts.threshold) continue;
        const float* b = pred.boxes.data() + static_cast<std::size_t>(i) * 4;
        Detection d;
        d.x0 = std::clamp((b[0] - 0.5f * b[2]) * S, 0.f, W);
        d.y0 = std::clamp((b[1] - 0.5f * b[3]) * S, 0.f, H);
        d.x1 = std::clamp((b[0] + 0.5f * b[2]) * S, 0.f, W);
        d.y1 = std::clamp((b[1] + 0.5f * b[3]) * S, 0.f, H);
        d.score = score;
        d.label = best;
        d.objectness = sigmoid(pred.objectness[i]);
        cand.push_back(d);
    }
    std::stable_sort(cand.begin(), cand.end(),
                     [](const Detection& a, const Detection& b) { return a.score > b.score; });
    const bool nms = opts.nms_iou > 0.f && opts.nms_iou < 1.f;
    std::vector<Detection> keep;
    for (const Detection& d : cand) {
        if (static_cast<int>(keep.size()) >= opts.max_detections) break;
        bool drop = false;
        if (nms) {
            const float ad = (d.x1 - d.x0) * (d.y1 - d.y0);
            for (const Detection& k : keep) {
                const float iw = std::min(d.x1, k.x1) - std::max(d.x0, k.x0);
                const float ih = std::min(d.y1, k.y1) - std::max(d.y0, k.y0);
                if (iw <= 0.f || ih <= 0.f) continue;
                const float inter = iw * ih;
                const float uni = ad + (k.x1 - k.x0) * (k.y1 - k.y0) - inter;
                if (uni > 0.f && inter / uni > opts.nms_iou) { drop = true; break; }
            }
        }
        if (!drop) keep.push_back(d);
    }
    return keep;
}

std::vector<Detection> Detector::detect(const uint8_t* pixels, int width, int height, int channels,
                                        const float* query_embeds, int num_queries,
                                        const DetectOptions& opts) const {
    const Pixels px = preprocess(pixels, width, height, channels, cfg_.image_size);
    return decode(predict(px, query_embeds, num_queries), px, opts);
}

}  // namespace brovisionml::owlv2
