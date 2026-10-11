#include "brovisionml/sam2_decoder.h"

#include "sam2_ops.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brovisionml::sam2 {

namespace {

namespace bt = brotensor;
using bt::Tensor;
using namespace detail;

constexpr const char* kWho = "sam2::SamHeads: ";
constexpr float kTwoWayEps = 1e-5f;    // nn.LayerNorm default in the two-way transformer
constexpr float kUpscaleEps = 1e-6f;   // channels-first LayerNorm in the upscaler

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(kWho + msg); }

struct AttnW { Tensor q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b; };
struct LN { Tensor w, b; };
struct FFW {
    Tensor in_w, in_b;
    std::vector<Tensor> lw, lb;
    Tensor out_w, out_b;
};
struct LayerW {
    AttnW self_attn; LN ln1;
    AttnW t2i;       LN ln2;
    Tensor mlp_in_w, mlp_in_b, mlp_out_w, mlp_out_b; LN ln3;
    AttnW i2t;       LN ln4;
};

}  // namespace

struct SamHeadsWeights {
    bool loaded = false;
    // Prompt encoder (host: the sparse tokens are assembled on the host).
    Tensor point_pe;          // (2, hidden/2) prompt_encoder.shared_embedding
    Tensor point_embed;       // (4, hidden)
    Tensor not_a_point;       // (1, hidden)
    Tensor no_mask;           // (hidden, 1) on device — row bias
    Tensor me_c1_w, me_c1_b, me_c2_w, me_c2_b, me_c3_w, me_c3_b;
    LN me_ln1, me_ln2;
    Tensor image_pe;          // (g*g, hidden) dense positional encoding
    // Mask decoder.
    Tensor out_tokens;        // (2 + K, hidden) host: obj, iou, masks
    std::vector<LayerW> layers;
    AttnW final_attn; LN ln_final;
    Tensor up1_w, up1_b, up2_w, up2_b;
    LN up_ln;
    std::vector<FFW> hyper;
    FFW iou_head, obj_head;
};

namespace {

Tensor ff_apply(const FFW& ff, const Tensor& x) {
    Tensor h;
    bt::linear_forward_batched(ff.in_w, ff.in_b, x, h);
    bt::relu_forward_batched(h, h);
    for (std::size_t j = 0; j < ff.lw.size(); ++j) {
        Tensor t;
        bt::linear_forward_batched(ff.lw[j], ff.lb[j], h, t);
        bt::relu_forward_batched(t, t);
        h = std::move(t);
    }
    Tensor o;
    bt::linear_forward_batched(ff.out_w, ff.out_b, h, o);
    return o;
}

Tensor run_attn(const AttnW& a, const Tensor& qin, const Tensor& kin, const Tensor& vin,
                int heads) {
    Tensor q, k, v, o;
    bt::linear_forward_batched(a.q_w, a.q_b, qin, q);
    bt::linear_forward_batched(a.k_w, a.k_b, kin, k);
    bt::linear_forward_batched(a.v_w, a.v_b, vin, v);
    Tensor att = attention(q, k, v, 1, q.rows, k.rows, heads);
    bt::linear_forward_batched(a.o_w, a.o_b, att, o);
    return o;
}

Tensor plus(const Tensor& a, const Tensor& b) {
    Tensor t = a.clone();
    bt::add_inplace(t, b);
    return t;
}

void ln_inplace(Tensor& x, const LN& l, float eps) { x = layer_norm(x, l.w, l.b, eps); }

// Random-Fourier positional encoding of a point already normalized to [0,1]^2.
void fourier(const float* pe, int half, float u, float v, float* out) {
    const float cx = 2.0f * u - 1.0f, cy = 2.0f * v - 1.0f;
    const float two_pi = 6.2831853071795864769f;
    for (int j = 0; j < half; ++j) {
        const float p = two_pi * (cx * pe[j] + cy * pe[half + j]);
        out[j] = std::sin(p);
        out[half + j] = std::cos(p);
    }
}

}  // namespace

SamHeads::SamHeads(const Sam2Config& cfg) : cfg_(cfg), w_(std::make_unique<SamHeadsWeights>()) {}
SamHeads::~SamHeads() = default;
SamHeads::SamHeads(SamHeads&&) noexcept = default;
SamHeads& SamHeads::operator=(SamHeads&&) noexcept = default;

void SamHeads::load(const st::File& f) {
    const int D = cfg_.hidden_size;
    const int K = cfg_.num_mask_tokens();
    const int ci = D / cfg_.attention_downsample;
    const int mic = cfg_.mask_input_channels;
    SamHeadsWeights w;

    const std::string pe = "prompt_encoder.";
    w.point_pe    = load_w(f, kWho, pe + "shared_embedding.positional_embedding", 2, D / 2);
    w.point_embed = load_w(f, kWho, pe + "point_embed.weight", 4, D);
    w.not_a_point = load_w(f, kWho, pe + "not_a_point_embed.weight", 1, D);
    w.no_mask     = load_w(f, kWho, pe + "no_mask_embed.weight", D, 1);
    w.me_c1_w = load_w(f, kWho, pe + "mask_embed.conv1.weight", mic / 4, 1 * 2 * 2);
    w.me_c1_b = load_w(f, kWho, pe + "mask_embed.conv1.bias", mic / 4, 1);
    w.me_c2_w = load_w(f, kWho, pe + "mask_embed.conv2.weight", mic, (mic / 4) * 2 * 2);
    w.me_c2_b = load_w(f, kWho, pe + "mask_embed.conv2.bias", mic, 1);
    w.me_c3_w = load_w(f, kWho, pe + "mask_embed.conv3.weight", D, mic);
    w.me_c3_b = load_w(f, kWho, pe + "mask_embed.conv3.bias", D, 1);
    w.me_ln1 = {load_w(f, kWho, pe + "mask_embed.layer_norm1.weight", mic / 4, 1),
                load_w(f, kWho, pe + "mask_embed.layer_norm1.bias", mic / 4, 1)};
    w.me_ln2 = {load_w(f, kWho, pe + "mask_embed.layer_norm2.weight", mic, 1),
                load_w(f, kWho, pe + "mask_embed.layer_norm2.bias", mic, 1)};

    // Dense positional encoding of the decoder grid (cell centres), from the
    // model-level shared_image_embedding.
    {
        Tensor ipe = load_w(f, kWho, "shared_image_embedding.positional_embedding", 2, D / 2);
        const int g = cfg_.grid();
        w.image_pe = Tensor::mat(g * g, D);
        for (int y = 0; y < g; ++y)
            for (int x = 0; x < g; ++x)
                fourier(ipe.host_f32(), D / 2, (x + 0.5f) / g, (y + 0.5f) / g,
                        w.image_pe.host_f32_mut() + (static_cast<std::size_t>(y) * g + x) * D);
    }

    const std::string md = "mask_decoder.";
    {
        Tensor obj = load_w(f, kWho, md + "obj_score_token.weight", 1, D);
        Tensor iou = load_w(f, kWho, md + "iou_token.weight", 1, D);
        Tensor mt  = load_w(f, kWho, md + "mask_tokens.weight", K, D);
        w.out_tokens = Tensor::mat(2 + K, D);
        float* d = w.out_tokens.host_f32_mut();
        std::copy(obj.host_f32(), obj.host_f32() + D, d);
        std::copy(iou.host_f32(), iou.host_f32() + D, d + D);
        std::copy(mt.host_f32(), mt.host_f32() + static_cast<std::size_t>(K) * D, d + 2 * D);
    }
    auto attn = [&](const std::string& p, int internal) {
        AttnW a;
        a.q_w = load_w(f, kWho, p + "q_proj.weight", internal, D);
        a.q_b = load_w(f, kWho, p + "q_proj.bias", internal, 1);
        a.k_w = load_w(f, kWho, p + "k_proj.weight", internal, D);
        a.k_b = load_w(f, kWho, p + "k_proj.bias", internal, 1);
        a.v_w = load_w(f, kWho, p + "v_proj.weight", internal, D);
        a.v_b = load_w(f, kWho, p + "v_proj.bias", internal, 1);
        a.o_w = load_w(f, kWho, p + "o_proj.weight", D, internal);
        a.o_b = load_w(f, kWho, p + "o_proj.bias", D, 1);
        return a;
    };
    auto ln = [&](const std::string& p, int n) {
        return LN{load_w(f, kWho, p + ".weight", n, 1), load_w(f, kWho, p + ".bias", n, 1)};
    };
    auto ff = [&](const std::string& p, int in, int hidden, int out, int depth) {
        FFW r;
        r.in_w = load_w(f, kWho, p + "proj_in.weight", hidden, in);
        r.in_b = load_w(f, kWho, p + "proj_in.bias", hidden, 1);
        for (int j = 0; j < depth - 2; ++j) {
            r.lw.push_back(load_w(f, kWho, p + "layers." + std::to_string(j) + ".weight", hidden, hidden));
            r.lb.push_back(load_w(f, kWho, p + "layers." + std::to_string(j) + ".bias", hidden, 1));
        }
        r.out_w = load_w(f, kWho, p + "proj_out.weight", out, hidden);
        r.out_b = load_w(f, kWho, p + "proj_out.bias", out, 1);
        return r;
    };
    for (int i = 0; i < cfg_.decoder_layers; ++i) {
        const std::string p = md + "transformer.layers." + std::to_string(i) + ".";
        LayerW L;
        L.self_attn = attn(p + "self_attn.", D);
        L.ln1 = ln(p + "layer_norm1", D);
        L.t2i = attn(p + "cross_attn_token_to_image.", ci);
        L.ln2 = ln(p + "layer_norm2", D);
        L.mlp_in_w  = load_w(f, kWho, p + "mlp.proj_in.weight", cfg_.decoder_mlp_dim, D);
        L.mlp_in_b  = load_w(f, kWho, p + "mlp.proj_in.bias", cfg_.decoder_mlp_dim, 1);
        L.mlp_out_w = load_w(f, kWho, p + "mlp.proj_out.weight", D, cfg_.decoder_mlp_dim);
        L.mlp_out_b = load_w(f, kWho, p + "mlp.proj_out.bias", D, 1);
        L.ln3 = ln(p + "layer_norm3", D);
        L.i2t = attn(p + "cross_attn_image_to_token.", ci);
        L.ln4 = ln(p + "layer_norm4", D);
        w.layers.push_back(std::move(L));
    }
    w.final_attn = attn(md + "transformer.final_attn_token_to_image.", ci);
    w.ln_final = ln(md + "transformer.layer_norm_final_attn", D);
    w.up1_w = load_w(f, kWho, md + "upscale_conv1.weight", D, (D / 4) * 2 * 2);
    w.up1_b = load_w(f, kWho, md + "upscale_conv1.bias", D / 4, 1);
    w.up2_w = load_w(f, kWho, md + "upscale_conv2.weight", D / 4, (D / 8) * 2 * 2);
    w.up2_b = load_w(f, kWho, md + "upscale_conv2.bias", D / 8, 1);
    w.up_ln = ln(md + "upscale_layer_norm", D / 4);
    for (int i = 0; i < K; ++i)
        w.hyper.push_back(ff(md + "output_hypernetworks_mlps." + std::to_string(i) + ".",
                             D, D, D / 8, 3));
    w.iou_head = ff(md + "iou_prediction_head.", D, cfg_.iou_head_hidden, K, cfg_.iou_head_depth);
    w.obj_head = ff(md + "pred_obj_score_head.", D, D, 1, 3);
    w.loaded = true;
    *w_ = std::move(w);
    device_ = bt::Device::CPU;
}

void SamHeads::to(bt::Device dev) {
    if (!w_->loaded) fail("to() called before load()");
    if (dev == device_) return;
    // The heads are cheap next to the encoder and run exact FP32 everywhere.
    // point_pe / point_embed / not_a_point / out_tokens stay on the host: they
    // feed the host-side token assembly.
    const bt::Dtype f32 = bt::Dtype::FP32;
    SamHeadsWeights& w = *w_;
    auto mv = [&](Tensor& t) { move(t, dev, f32); };
    auto mva = [&](AttnW& a) { mv(a.q_w); mv(a.q_b); mv(a.k_w); mv(a.k_b); mv(a.v_w); mv(a.v_b); mv(a.o_w); mv(a.o_b); };
    auto mvl = [&](LN& l) { mv(l.w); mv(l.b); };
    auto mvf = [&](FFW& r) {
        mv(r.in_w); mv(r.in_b);
        for (Tensor& t : r.lw) mv(t);
        for (Tensor& t : r.lb) mv(t);
        mv(r.out_w); mv(r.out_b);
    };
    mv(w.no_mask);
    mv(w.me_c1_w); mv(w.me_c1_b); mv(w.me_c2_w); mv(w.me_c2_b); mv(w.me_c3_w); mv(w.me_c3_b);
    mvl(w.me_ln1); mvl(w.me_ln2);
    mv(w.image_pe);
    for (LayerW& L : w.layers) {
        mva(L.self_attn); mvl(L.ln1); mva(L.t2i); mvl(L.ln2);
        mv(L.mlp_in_w); mv(L.mlp_in_b); mv(L.mlp_out_w); mv(L.mlp_out_b); mvl(L.ln3);
        mva(L.i2t); mvl(L.ln4);
    }
    mva(w.final_attn); mvl(w.ln_final);
    mv(w.up1_w); mv(w.up1_b); mv(w.up2_w); mv(w.up2_b); mvl(w.up_ln);
    for (FFW& r : w.hyper) mvf(r);
    mvf(w.iou_head); mvf(w.obj_head);
    device_ = dev;
}

HeadsOutput SamHeads::decode(const Tensor& image, const Tensor& feat_s0, const Tensor& feat_s1,
                             const HeadsPrompt& prompt, bool multimask) const {
    if (!w_->loaded) fail("decode() called before load()");
    const SamHeadsWeights& w = *w_;
    const int D = cfg_.hidden_size;
    const int g = cfg_.grid();
    const int HW = g * g;
    const int K = cfg_.num_mask_tokens();
    const int heads = cfg_.decoder_heads;
    const int ms = cfg_.mask_size();
    const float S = static_cast<float>(cfg_.image_size);
    if (image.device != device_ || image.rows != HW || image.cols != D)
        fail("image must be (grid*grid, hidden) on device()");
    if (prompt.labels.size() != prompt.points.size()) fail("points and labels differ in length");

    // ── Tokens: [obj, iou, mask x K, points..., pad] assembled on the host. ──
    // No points at all means no sparse tokens (a mask-only prompt); otherwise
    // the reference prompt encoder appends one padding point.
    const int n_pts = prompt.points.empty() ? 0 : static_cast<int>(prompt.points.size()) + 1;
    const int T = 2 + K + n_pts;
    Tensor tokens;
    {
        Tensor th = Tensor::mat(T, D);
        float* d = th.host_f32_mut();
        std::copy(w.out_tokens.host_f32(), w.out_tokens.host_f32() + (2 + K) * D, d);
        const float* pe = w.point_pe.host_f32();
        const float* pemb = w.point_embed.host_f32();
        const float* nap = w.not_a_point.host_f32();
        for (int i = 0; i < n_pts; ++i) {
            float* row = d + static_cast<std::size_t>(2 + K + i) * D;
            const bool pad = i == n_pts - 1;
            const int label = pad ? -1 : prompt.labels[static_cast<std::size_t>(i)];
            if (label == -1) {
                std::copy(nap, nap + D, row);
                continue;
            }
            if (label < 0 || label > 3) fail("point label must be 0..3 (or -1)");
            const auto& p = prompt.points[static_cast<std::size_t>(i)];
            fourier(pe, D / 2, (p[0] + 0.5f) / S, (p[1] + 0.5f) / S, row);
            for (int c = 0; c < D; ++c) row[c] += pemb[label * D + c];
        }
        tokens = th.to(device_);
    }

    // ── keys = image + dense prompt ──
    Tensor keys = image.clone();
    if (prompt.mask_input) {
        const Tensor& m = *prompt.mask_input;
        if (m.device != device_ || m.size() != ms * ms) fail("mask_input must be (1, mask_size^2) on device()");
        const int c1 = cfg_.mask_input_channels / 4, c2 = cfg_.mask_input_channels;
        const float eps = cfg_.prompt_ln_eps;
        Tensor mi = Tensor::view(m.device, m.data, 1, ms * ms);
        Tensor a;
        bt::conv2d_forward(mi, w.me_c1_w, &w.me_c1_b, 1, 1, ms, ms, c1, 2, 2, 2, 2, 0, 0, 1, 1, a);
        a = layer_norm_2d(a, w.me_ln1.w, w.me_ln1.b, 1, c1, ms / 2, ms / 2, eps);
        Tensor ga; bt::gelu_exact_forward(a, ga);
        Tensor b;
        bt::conv2d_forward(ga, w.me_c2_w, &w.me_c2_b, 1, c1, ms / 2, ms / 2, c2, 2, 2, 2, 2, 0, 0, 1, 1, b);
        b = layer_norm_2d(b, w.me_ln2.w, w.me_ln2.b, 1, c2, g, g, eps);
        Tensor gb; bt::gelu_exact_forward(b, gb);
        Tensor dense;
        bt::conv2d_forward(gb, w.me_c3_w, &w.me_c3_b, 1, c2, g, g, D, 1, 1, 1, 1, 0, 0, 1, 1, dense);
        bt::add_inplace(keys, to_tokens(dense, 1, D, g, g));
    } else {
        bt::add_row_bias_inplace(keys, w.no_mask);
    }

    // ── Two-way transformer ──
    const Tensor& qpe = tokens;
    const Tensor& kpe = w.image_pe;
    Tensor queries = tokens;
    for (int i = 0; i < cfg_.decoder_layers; ++i) {
        const LayerW& L = w.layers[static_cast<std::size_t>(i)];
        if (i == 0) {
            queries = run_attn(L.self_attn, queries, queries, queries, heads);
        } else {
            Tensor q = plus(queries, qpe);
            bt::add_inplace(queries, run_attn(L.self_attn, q, q, queries, heads));
        }
        ln_inplace(queries, L.ln1, kTwoWayEps);
        {
            Tensor q = plus(queries, qpe), k = plus(keys, kpe);
            bt::add_inplace(queries, run_attn(L.t2i, q, k, keys, heads));
            ln_inplace(queries, L.ln2, kTwoWayEps);
        }
        {
            Tensor m1; bt::linear_forward_batched(L.mlp_in_w, L.mlp_in_b, queries, m1);
            bt::relu_forward_batched(m1, m1);
            Tensor m2; bt::linear_forward_batched(L.mlp_out_w, L.mlp_out_b, m1, m2);
            bt::add_inplace(queries, m2);
            ln_inplace(queries, L.ln3, kTwoWayEps);
        }
        {
            Tensor q = plus(queries, qpe), k = plus(keys, kpe);
            bt::add_inplace(keys, run_attn(L.i2t, k, q, queries, heads));
            ln_inplace(keys, L.ln4, kTwoWayEps);
        }
    }
    {
        Tensor q = plus(queries, qpe), k = plus(keys, kpe);
        bt::add_inplace(queries, run_attn(w.final_attn, q, k, keys, heads));
        ln_inplace(queries, w.ln_final, kTwoWayEps);
    }

    // ── Upscale 4x with the high-resolution maps added back in ──
    const int c1 = D / 4, c2 = D / 8;
    Tensor up;
    {
        Tensor src = to_nchw(keys, 1, D, g, g);
        Tensor u1;
        bt::conv_transpose2d_forward(src, w.up1_w, &w.up1_b, 1, D, g, g, c1, 2, 2, 2, 2, 0, 0, 0, 0,
                                     1, 1, 1, u1);
        bt::add_inplace(u1, feat_s1);
        u1 = layer_norm_2d(u1, w.up_ln.w, w.up_ln.b, 1, c1, 2 * g, 2 * g, kUpscaleEps);
        Tensor a1; bt::gelu_exact_forward(u1, a1);
        Tensor u2;
        bt::conv_transpose2d_forward(a1, w.up2_w, &w.up2_b, 1, c1, 2 * g, 2 * g, c2, 2, 2, 2, 2, 0, 0,
                                     0, 0, 1, 1, 1, u2);
        bt::add_inplace(u2, feat_s0);
        bt::gelu_exact_forward(u2, up);   // (1, c2*ms*ms)
    }

    // ── Hypernetwork filters -> K masks ──
    Tensor masks;
    {
        Tensor fh = Tensor::mat(K, c2);
        for (int i = 0; i < K; ++i) {
            Tensor tok = view_rows(queries, static_cast<long long>(2 + i) * D, 1, D);
            Tensor hi = host(ff_apply(w.hyper[static_cast<std::size_t>(i)], tok));
            std::copy(hi.host_f32(), hi.host_f32() + c2, fh.host_f32_mut() + static_cast<std::size_t>(i) * c2);
        }
        Tensor filt = fh.to(device_);
        Tensor upm = view_rows(up, 0, c2, ms * ms);
        bt::matmul(filt, upm, masks);   // (K, ms*ms)
    }

    // ── IoU + object score ──
    Tensor iou_h, obj_h;
    {
        Tensor it = view_rows(queries, D, 1, D);
        Tensor iv = ff_apply(w.iou_head, it);
        bt::sigmoid_forward(iv, iv);
        iou_h = host(iv);
        Tensor ot = view_rows(queries, 0, 1, D);
        obj_h = host(ff_apply(w.obj_head, ot));
    }
    const float* iou = iou_h.host_f32();

    HeadsOutput out;
    out.object_score = obj_h.host_f32()[0];
    const long long plane = static_cast<long long>(ms) * ms;
    if (multimask) {
        out.num = K - 1;
        out.low_res = view_rows(masks, plane, K - 1, ms * ms).clone();
        out.iou.assign(iou + 1, iou + K);
        out.sam_tokens = view_rows(queries, static_cast<long long>(3) * D, K - 1, D).clone();
        return out;
    }
    int pick = 0;
    if (cfg_.dynamic_multimask_via_stability) {
        Tensor counts;
        bt::rows_count_above(view_rows(masks, 0, 1, ms * ms), -cfg_.stability_delta,
                             cfg_.stability_delta, counts);
        Tensor ch = counts.to(bt::Device::CPU);
        const int32_t* cv = static_cast<const int32_t*>(ch.data);
        const float stab = cv[0] > 0 ? static_cast<float>(cv[1]) / static_cast<float>(cv[0]) : 1.0f;
        if (stab < cfg_.stability_thresh) {
            pick = 1;
            for (int i = 2; i < K; ++i) if (iou[i] > iou[pick]) pick = i;
        }
    }
    out.num = 1;
    out.low_res = view_rows(masks, pick * plane, 1, ms * ms).clone();
    out.iou = {iou[pick]};
    out.sam_tokens = view_rows(queries, static_cast<long long>(2) * D, 1, D).clone();
    return out;
}

}  // namespace brovisionml::sam2
