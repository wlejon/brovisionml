#include "brovisionml/sam2_memory.h"

#include "brovisionml/sam2_image_encoder.h"   // sine_position_tokens
#include "sam2_ops.h"
#include "profile.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brovisionml::sam2 {

namespace {

namespace bt = brotensor;
using bt::Tensor;
using namespace detail;

constexpr const char* kWho = "sam2::MemoryModule: ";
constexpr float kAttnEps = 1e-5f;   // nn.LayerNorm default in the memory attention
constexpr float kConvEps = 1e-6f;   // channels-first LayerNorm in the memory encoder

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(kWho + msg); }

struct Lin { Tensor w, b; };
struct LN { Tensor w, b; };
struct MemAttnLayer {
    Lin sq, sk, sv, so;      // self-attention
    Lin cq, ck, cv, co;      // cross-attention into the memory (k / v from mem_dim)
    LN ln1, ln2, ln3;
    Lin ff1, ff2;
};
struct FuserBlock {
    Tensor dw_w, dw_b;       // (C, k*k) depthwise
    LN ln;
    Lin pw1, pw2;            // pw2 carries the layer scale folded in
};

}  // namespace

struct MemoryWeights {
    bool loaded = false;
    Tensor no_mem;                 // (1, hidden)
    Tensor tpos;                   // (num_maskmem, mem_dim)
    Tensor no_obj_ptr;             // (1, hidden)
    Tensor occlusion;              // (1, mem_dim)
    Lin ptr_tpos;                  // hidden -> mem_dim
    Lin ptr_in, ptr_mid, ptr_out;  // object_pointer_proj
    // Memory encoder.
    std::vector<Tensor> md_w, md_b;
    std::vector<LN> md_ln;
    Lin md_final, feat_proj, out_proj;
    std::vector<FuserBlock> fuser;
    Tensor mem_sine;               // (g*g, mem_dim) sine encoding of a memory
    // Memory attention.
    std::vector<MemAttnLayer> layers;
    LN ln_final;
    Tensor rope_cos, rope_sin;     // (g*g, head_dim/2) on device
    Tensor rope_cos_host, rope_sin_host;
    // The cross-attention key tables for the last memory layout used.
    Tensor rope_cos_k, rope_sin_k;
    int rope_k_frames = 0, rope_k_ptr_rows = -1;
};

Tensor sine_pe_1d(const std::vector<float>& pos, int dim) {
    const int half = dim / 2;
    Tensor out = Tensor::mat(static_cast<int>(pos.size()), dim);
    float* d = out.host_f32_mut();
    for (std::size_t r = 0; r < pos.size(); ++r) {
        for (int i = 0; i < half; ++i) {
            const float dt = std::pow(10000.0f, 2.0f * static_cast<float>(i / 2) / static_cast<float>(half));
            const float v = pos[r] / dt;
            d[r * dim + i] = std::sin(v);
            d[r * dim + half + i] = std::cos(v);
        }
    }
    return out;
}

MemoryModule::MemoryModule(const Sam2Config& cfg) : cfg_(cfg), w_(std::make_unique<MemoryWeights>()) {}
MemoryModule::~MemoryModule() = default;
MemoryModule::MemoryModule(MemoryModule&&) noexcept = default;
MemoryModule& MemoryModule::operator=(MemoryModule&&) noexcept = default;

void MemoryModule::load(const st::File& f) {
    const int D = cfg_.hidden_size;
    const int M = cfg_.mem_dim;
    const int g = cfg_.grid();
    MemoryWeights w;
    auto lin = [&](const std::string& p, int out, int in) {
        return Lin{load_w(f, kWho, p + ".weight", out, in), load_w(f, kWho, p + ".bias", out, 1)};
    };
    auto ln = [&](const std::string& p, int n) {
        return LN{load_w(f, kWho, p + ".weight", n, 1), load_w(f, kWho, p + ".bias", n, 1)};
    };

    w.no_mem     = load_w(f, kWho, "no_memory_embedding", 1, D);
    w.tpos       = load_w(f, kWho, "memory_temporal_positional_encoding", cfg_.num_maskmem, M);
    w.no_obj_ptr = load_w(f, kWho, "no_object_pointer", 1, D);
    if (cfg_.occlusion_spatial_embedding)
        w.occlusion = load_w(f, kWho, "occlusion_spatial_embedding_parameter", 1, M);
    w.ptr_tpos = lin("temporal_positional_encoding_projection_layer", M, D);
    w.ptr_in  = lin("object_pointer_proj.proj_in", D, D);
    w.ptr_mid = lin("object_pointer_proj.layers.0", D, D);
    w.ptr_out = lin("object_pointer_proj.proj_out", D, D);

    // Memory encoder: mask downsampler 1 -> s^2 -> s^4 ... -> D channels.
    const std::string me = "memory_encoder.";
    {
        const int s = cfg_.mask_down_stride, k = cfg_.mask_down_kernel;
        int layers = 0;
        for (int t = 1; t < cfg_.mask_down_total_stride; t *= s) ++layers;
        int cin = 1;
        for (int i = 0; i < layers; ++i) {
            const int cout = cin * s * s;
            const std::string p = me + "mask_downsampler.layers." + std::to_string(i) + ".";
            w.md_w.push_back(load_w(f, kWho, p + "conv.weight", cout, cin * k * k));
            w.md_b.push_back(load_w(f, kWho, p + "conv.bias", cout, 1));
            w.md_ln.push_back(ln(p + "layer_norm", cout));
            cin = cout;
        }
        if (cin != D) fail("mask downsampler does not end at the hidden size");
        w.md_final = lin(me + "mask_downsampler.final_conv", D, D);
    }
    w.feat_proj = lin(me + "feature_projection", D, D);
    for (int i = 0; i < cfg_.fuser_layers; ++i) {
        const std::string p = me + "memory_fuser.layers." + std::to_string(i) + ".";
        const int kk = cfg_.fuser_kernel;
        FuserBlock b;
        b.dw_w = load_w(f, kWho, p + "depthwise_conv.weight", D, kk * kk);
        b.dw_b = load_w(f, kWho, p + "depthwise_conv.bias", D, 1);
        b.ln = ln(p + "layer_norm", D);
        b.pw1 = lin(p + "pointwise_conv1", cfg_.fuser_intermediate, D);
        b.pw2 = lin(p + "pointwise_conv2", D, cfg_.fuser_intermediate);
        // gamma * (W x + b) == (diag(gamma) W) x + gamma b
        Tensor gamma = load_w(f, kWho, p + "scale", D, 1);
        float* wv = b.pw2.w.host_f32_mut();
        float* bv = b.pw2.b.host_f32_mut();
        const float* gv = gamma.host_f32();
        for (int r = 0; r < D; ++r) {
            for (int c = 0; c < cfg_.fuser_intermediate; ++c)
                wv[static_cast<std::size_t>(r) * cfg_.fuser_intermediate + c] *= gv[r];
            bv[r] *= gv[r];
        }
        w.fuser.push_back(std::move(b));
    }
    w.out_proj = lin(me + "projection", M, D);
    w.mem_sine = sine_position_tokens(g, g, M / 2);

    // Memory attention.
    const int kv = M;
    for (int i = 0; i < cfg_.mem_attn_layers; ++i) {
        const std::string p = "memory_attention.layers." + std::to_string(i) + ".";
        MemAttnLayer L;
        L.sq = lin(p + "self_attn.q_proj", D, D);
        L.sk = lin(p + "self_attn.k_proj", D, D);
        L.sv = lin(p + "self_attn.v_proj", D, D);
        L.so = lin(p + "self_attn.o_proj", D, D);
        L.cq = lin(p + "cross_attn_image.q_proj", D, D);
        L.ck = lin(p + "cross_attn_image.k_proj", D, kv);
        L.cv = lin(p + "cross_attn_image.v_proj", D, kv);
        L.co = lin(p + "cross_attn_image.o_proj", D, D);
        L.ln1 = ln(p + "layer_norm1", D);
        L.ln2 = ln(p + "layer_norm2", D);
        L.ln3 = ln(p + "layer_norm3", D);
        L.ff1 = lin(p + "linear1", cfg_.mem_attn_ff, D);
        L.ff2 = lin(p + "linear2", D, cfg_.mem_attn_ff);
        w.layers.push_back(std::move(L));
    }
    w.ln_final = ln("memory_attention.layer_norm", D);

    // Axial RoPE (compute_axial_cis): pairs [0, hd/4) rotate by x, the next
    // hd/4 by y, frequency theta^(-4i/hd).
    {
        const int hd = D / cfg_.mem_attn_heads;
        const int q4 = hd / 4;
        const int ex = cfg_.rope_feat_sizes[0];
        const int L = g * g;
        w.rope_cos = Tensor::mat(L, hd / 2);
        w.rope_sin = Tensor::mat(L, hd / 2);
        float* c = w.rope_cos.host_f32_mut();
        float* s = w.rope_sin.host_f32_mut();
        for (int idx = 0; idx < L; ++idx) {
            const float tx = static_cast<float>(idx % ex), ty = static_cast<float>(idx / ex);
            for (int i = 0; i < q4; ++i) {
                const float fr = 1.0f / std::pow(cfg_.rope_theta,
                                                 static_cast<float>(4 * i) / static_cast<float>(hd));
                const float ax = tx * fr, ay = ty * fr;
                c[static_cast<std::size_t>(idx) * (hd / 2) + i] = std::cos(ax);
                s[static_cast<std::size_t>(idx) * (hd / 2) + i] = std::sin(ax);
                c[static_cast<std::size_t>(idx) * (hd / 2) + q4 + i] = std::cos(ay);
                s[static_cast<std::size_t>(idx) * (hd / 2) + q4 + i] = std::sin(ay);
            }
        }
        w.rope_cos_host = w.rope_cos;
        w.rope_sin_host = w.rope_sin;
    }
    w.loaded = true;
    *w_ = std::move(w);
    device_ = bt::Device::CPU;
    fp16_ = false;
}

void MemoryModule::to(bt::Device dev) {
    if (!w_->loaded) fail("to() called before load()");
    if (dev == device_) return;
    const bool fp16 = dev != bt::Device::CPU && bt::compute_dtype() == bt::Dtype::FP16;
    const bt::Dtype f32 = bt::Dtype::FP32;
    const bt::Dtype half = fp16 ? bt::Dtype::FP16 : f32;
    MemoryWeights& w = *w_;
    auto mv = [&](Tensor& t, bt::Dtype dt) { move(t, dev, dt); };
    auto ml = [&](Lin& l, bt::Dtype dt) { mv(l.w, dt); mv(l.b, dt); };
    auto mn = [&](LN& l) { mv(l.w, f32); mv(l.b, f32); };
    mv(w.no_mem, f32); mv(w.tpos, f32); mv(w.no_obj_ptr, f32); mv(w.occlusion, f32);
    ml(w.ptr_tpos, f32); ml(w.ptr_in, f32); ml(w.ptr_mid, f32); ml(w.ptr_out, f32);
    for (Tensor& t : w.md_w) mv(t, f32);
    for (Tensor& t : w.md_b) mv(t, f32);
    for (LN& l : w.md_ln) mn(l);
    ml(w.md_final, f32); ml(w.feat_proj, f32); ml(w.out_proj, f32);
    for (FuserBlock& b : w.fuser) {
        mv(b.dw_w, f32); mv(b.dw_b, f32); mn(b.ln); ml(b.pw1, f32); ml(b.pw2, f32);
    }
    mv(w.mem_sine, f32);
    // The memory attention is the per-frame hot path: its GEMMs and attention
    // go FP16 on an FP16 GPU, the residual stream and LayerNorms stay FP32.
    for (MemAttnLayer& L : w.layers) {
        ml(L.sq, half); ml(L.sk, half); ml(L.sv, half); ml(L.so, half);
        ml(L.cq, half); ml(L.ck, half); ml(L.cv, half); ml(L.co, half);
        mn(L.ln1); mn(L.ln2); mn(L.ln3);
        ml(L.ff1, half); ml(L.ff2, half);
    }
    mn(w.ln_final);
    mv(w.rope_cos, f32); mv(w.rope_sin, f32);
    w.rope_cos_k = Tensor(); w.rope_sin_k = Tensor();
    w.rope_k_frames = 0; w.rope_k_ptr_rows = -1;
    device_ = dev;
    fp16_ = fp16;
}

Tensor MemoryModule::no_memory(const Tensor& top) const {
    Tensor t = top.clone();
    bt::add_row_bias_inplace(t, w_->no_mem);
    return t;
}

Tensor MemoryModule::encode(const Tensor& top, const Tensor& low_res, bool from_points,
                            float object_score) const {
    if (!w_->loaded) fail("encode() called before load()");
    const MemoryWeights& w = *w_;
    const int D = cfg_.hidden_size;
    const int g = cfg_.grid();
    const int ms = cfg_.mask_size();
    const int S = cfg_.image_size;
    if (top.device != device_ || top.rows != g * g || top.cols != D) fail("top must be (grid^2, hidden)");
    if (low_res.device != device_ || low_res.size() != ms * ms) fail("low_res must be (1, mask_size^2)");

    // High-res mask: bilinear 4x of the low-res logits, then the binarized
    // (prompted frame) or sigmoid mask, scaled into the encoder's range.
    Tensor m;
    {
        Tensor lr = Tensor::view(low_res.device, low_res.data, 1, ms * ms);
        Tensor hr;
        bt::interp2d_forward(lr, 1, 1, ms, ms, S, S, /*bilinear*/ 1, hr);
        if (from_points) {
            // (x > 0) exactly: x * FLT_MAX is >= 1 for every normal positive and
            // <= 0 otherwise; clamp to [0, 1] and round to {0, 1}.
            bt::scale_inplace(hr, FLT_MAX);
            bt::clamp(hr, 0.0f, 1.0f);
            bt::round_forward(hr, m);
        } else {
            bt::sigmoid_forward(hr, m);
        }
        bt::scale_inplace(m, cfg_.sigmoid_scale_for_mem_enc);
        bt::add_scalar_inplace(m, cfg_.sigmoid_bias_for_mem_enc);
    }

    // Mask downsampler: strided conv -> LayerNorm2d -> GELU, then a 1x1.
    int side = S, cin = 1;
    for (std::size_t i = 0; i < w.md_w.size(); ++i) {
        const int cout = w.md_w[i].rows;
        const int k = cfg_.mask_down_kernel, s = cfg_.mask_down_stride, p = cfg_.mask_down_padding;
        const int out_side = (side + 2 * p - k) / s + 1;
        Tensor c;
        bt::conv2d_forward(m, w.md_w[i], &w.md_b[i], 1, cin, side, side, cout, k, k, s, s, p, p,
                           1, 1, c);
        Tensor n = layer_norm_2d(c, w.md_ln[i].w, w.md_ln[i].b, 1, cout, out_side, out_side, kConvEps);
        bt::gelu_exact_forward(n, m);
        side = out_side;
        cin = cout;
    }
    if (side != g) fail("mask downsampler does not land on the feature grid");
    Tensor x;
    {
        Tensor mt = to_tokens(m, 1, D, g, g);
        Tensor md;
        bt::linear_forward_batched(w.md_final.w, w.md_final.b, mt, md);
        bt::linear_forward_batched(w.feat_proj.w, w.feat_proj.b, top, x);
        bt::add_inplace(x, md);
    }

    // Fuser: ConvNeXt blocks (depthwise 7x7, LayerNorm, GELU MLP, scale).
    const int kk = cfg_.fuser_kernel;
    for (const FuserBlock& b : w.fuser) {
        Tensor nchw = to_nchw(x, 1, D, g, g);
        Tensor dw;
        bt::conv2d_forward(nchw, b.dw_w, &b.dw_b, 1, D, g, g, D, kk, kk, 1, 1, kk / 2, kk / 2, 1, 1,
                           /*groups=*/D, dw);
        Tensor t = layer_norm(to_tokens(dw, 1, D, g, g), b.ln.w, b.ln.b, kConvEps);
        Tensor h1;
        bt::linear_forward_batched(b.pw1.w, b.pw1.b, t, h1);
        Tensor a;
        bt::gelu_exact_forward(h1, a);
        Tensor h2;
        bt::linear_forward_batched(b.pw2.w, b.pw2.b, a, h2);
        bt::add_inplace(x, h2);
    }
    Tensor feat;
    bt::linear_forward_batched(w.out_proj.w, w.out_proj.b, x, feat);
    if (cfg_.occlusion_spatial_embedding && !(object_score > 0.0f))
        bt::add_row_bias_inplace(feat, w.occlusion);

    // The reference keeps its memory bank in BF16.
    Tensor b16;
    bt::cast(feat, b16, bt::Dtype::BF16);
    Tensor out;
    bt::cast(b16, out, bt::Dtype::FP32);
    brovisionml::detail::profile_mark(device_, "memory-encode");
    return out;
}

Tensor MemoryModule::object_pointer(const Tensor& sam_token, bool appearing) const {
    const MemoryWeights& w = *w_;
    if (!appearing) return w.no_obj_ptr.clone();
    Tensor h1, h2, out;
    bt::linear_forward_batched(w.ptr_in.w, w.ptr_in.b, sam_token, h1);
    bt::relu_forward_batched(h1, h1);
    bt::linear_forward_batched(w.ptr_mid.w, w.ptr_mid.b, h1, h2);
    bt::relu_forward_batched(h2, h2);
    bt::linear_forward_batched(w.ptr_out.w, w.ptr_out.b, h2, out);
    return out;
}

namespace {

Tensor rope(const Tensor& x, const Tensor& c, const Tensor& s, int hd, int heads) {
    Tensor y;
    bt::rope_apply(x, c, s, hd, heads, y);
    return y;
}

}  // namespace

Tensor MemoryModule::attend(const Tensor& top, const Tensor& top_pos,
                            const std::vector<Memory>& memories,
                            const std::vector<Pointer>& pointers) const {
    if (!w_->loaded) fail("attend() called before load()");
    if (memories.empty()) fail("attend() needs at least one memory");
    MemoryWeights& w = *w_;
    const int D = cfg_.hidden_size;
    const int M = cfg_.mem_dim;
    const int L = cfg_.grid() * cfg_.grid();
    const int heads = cfg_.mem_attn_heads;
    const int hd = D / heads;
    const int split = D / M;   // tokens per object pointer
    const bool fp16 = fp16_;
    const int nm = static_cast<int>(memories.size());
    const int np = static_cast<int>(pointers.size());
    const int n_rope = nm * L;
    const int n_mem = n_rope + np * split;

    // ── Memory tokens + positions ──
    Tensor mem = Tensor::zeros_on(device_, n_mem, M);
    Tensor mem_pos = Tensor::zeros_on(device_, n_mem, M);
    for (int i = 0; i < nm; ++i) {
        const Memory& e = memories[static_cast<std::size_t>(i)];
        if (!e.features || e.features->rows != L || e.features->cols != M)
            fail("memory features must be (grid^2, mem_dim)");
        if (e.tpos < 0 || e.tpos >= cfg_.num_maskmem) fail("memory tpos out of range");
        bt::copy_d2d(*e.features, 0, mem, i * L * M, L * M);
        Tensor pos = w.mem_sine.clone();
        bt::add_row_bias_inplace(pos, view_rows(w.tpos, static_cast<long long>(e.tpos) * M, 1, M));
        bt::copy_d2d(pos, 0, mem_pos, i * L * M, L * M);
    }
    if (np > 0) {
        std::vector<float> pv;
        for (int j = 0; j < np; ++j) {
            const Pointer& p = pointers[static_cast<std::size_t>(j)];
            if (!p.ptr || p.ptr->size() != D) fail("object pointer must be (1, hidden)");
            bt::copy_d2d(*p.ptr, 0, mem, n_rope * M + j * D, D);
            pv.push_back(p.pos);
        }
        Tensor pe = sine_pe_1d(pv, D).to(device_);
        Tensor pp;
        if (cfg_.temporal_pos_for_obj_ptrs) {
            bt::linear_forward_batched(w.ptr_tpos.w, w.ptr_tpos.b, pe, pp);   // (np, M)
            for (int j = 0; j < np; ++j)
                for (int r = 0; r < split; ++r)
                    bt::copy_d2d(pp, j * M, mem_pos, n_rope * M + (j * split + r) * M, M);
        }
    }

    // Keys / values for every layer's cross-attention.
    Tensor kin = mem.clone();
    bt::add_inplace(kin, mem_pos);
    Tensor kin16 = to16(kin, fp16);
    Tensor vin16 = to16(mem, fp16);
    // Key RoPE tables: the frame table repeated per memory frame, then the
    // identity rotation (cos 1, sin 0) over the object-pointer tokens.
    if (w.rope_k_frames != nm || w.rope_k_ptr_rows != n_mem - n_rope) {
        const int half = hd / 2, pr = n_mem - n_rope;
        Tensor ch = Tensor::mat(n_mem, half), sh = Tensor::mat(n_mem, half);
        const float* c0 = w.rope_cos_host.host_f32();
        const float* s0 = w.rope_sin_host.host_f32();
        float* cd = ch.host_f32_mut();
        float* sd = sh.host_f32_mut();
        for (int i = 0; i < nm; ++i) {
            std::copy(c0, c0 + static_cast<std::size_t>(L) * half, cd + static_cast<std::size_t>(i) * L * half);
            std::copy(s0, s0 + static_cast<std::size_t>(L) * half, sd + static_cast<std::size_t>(i) * L * half);
        }
        std::fill(cd + static_cast<std::size_t>(n_rope) * half,
                  cd + static_cast<std::size_t>(n_rope + pr) * half, 1.0f);
        w.rope_cos_k = ch.to(device_);
        w.rope_sin_k = sh.to(device_);
        w.rope_k_frames = nm;
        w.rope_k_ptr_rows = pr;
    }

    // ── Layers ──
    Tensor out = top.clone();
    bt::axpby_inplace(out, top_pos, 1.0f, 0.1f);
    for (const MemAttnLayer& Lw : w.layers) {
        {
            Tensor t = to16(layer_norm(out, Lw.ln1.w, Lw.ln1.b, kAttnEps), fp16);
            Tensor q, k, v, o;
            linear(Lw.sq.w, Lw.sq.b, t, q, fp16);
            linear(Lw.sk.w, Lw.sk.b, t, k, fp16);
            linear(Lw.sv.w, Lw.sv.b, t, v, fp16);
            Tensor att = attention(rope(q, w.rope_cos, w.rope_sin, hd, heads),
                                   rope(k, w.rope_cos, w.rope_sin, hd, heads), v, 1, L, L, heads);
            linear(Lw.so.w, Lw.so.b, att, o, fp16);
            bt::add_inplace(out, to32(o));
        }
        {
            Tensor t = to16(layer_norm(out, Lw.ln2.w, Lw.ln2.b, kAttnEps), fp16);
            Tensor q, k, v, o;
            linear(Lw.cq.w, Lw.cq.b, t, q, fp16);
            linear(Lw.ck.w, Lw.ck.b, kin16, k, fp16);
            linear(Lw.cv.w, Lw.cv.b, vin16, v, fp16);
            // RoPE on the spatial memory keys only, not the object pointers.
            Tensor kr = rope(k, w.rope_cos_k, w.rope_sin_k, hd, heads);
            Tensor att = attention(rope(q, w.rope_cos, w.rope_sin, hd, heads), kr, v, 1, L, n_mem, heads);
            linear(Lw.co.w, Lw.co.b, att, o, fp16);
            bt::add_inplace(out, to32(o));
        }
        {
            Tensor t = to16(layer_norm(out, Lw.ln3.w, Lw.ln3.b, kAttnEps), fp16);
            Tensor h, o;
            linear(Lw.ff1.w, Lw.ff1.b, t, h, fp16);
            bt::relu_forward_batched(h, h);
            linear(Lw.ff2.w, Lw.ff2.b, h, o, fp16);
            bt::add_inplace(out, to32(o));
        }
    }
    Tensor res = layer_norm(out, w.ln_final.w, w.ln_final.b, kAttnEps);
    brovisionml::detail::profile_mark(device_, "memory-attention");
    return res;
}

}  // namespace brovisionml::sam2
