#include "brovisionml/sam_mask_decoder.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include "sam_weights_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brovisionml::sam {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using detail::load_whole;

constexpr const char* kWho = "sam::MaskDecoder: ";

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

}  // namespace

// ─── Weight tables (host FP32; migrated to device by to(), except the two
//     token-embedding tables which feed host-side token assembly) ────────────

struct AttnW {  // q/k/v project D->internal; out projects internal->D
    Tensor q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b;
};
struct LN { Tensor w, b; };  // (D,1) gamma/beta

struct LayerW {
    AttnW self_attn;  LN ln1;
    AttnW cross_t2i;  LN ln2;
    Tensor mlp1_w, mlp1_b, mlp2_w, mlp2_b;  LN ln3;
    AttnW cross_i2t;  LN ln4;
};

// SamFeedForward: proj_in -> (relu -> hidden linear)* -> relu -> proj_out.
struct FFW {
    Tensor in_w, in_b;
    std::vector<Tensor> lw, lb;
    Tensor out_w, out_b;
};

struct MaskDecoderWeights {
    bool   loaded = false;
    Tensor iou_token;     // (1,D)  host
    Tensor mask_tokens;   // (K,D)  host
    std::vector<LayerW> layers;
    AttnW  final_attn;  LN ln_final;
    Tensor uc1_w, uc1_b;  LN up_ln;  Tensor uc2_w, uc2_b;  // 4x upscaler
    std::vector<FFW> hyper;  // num_mask_tokens hypernetwork MLPs
    FFW    iou_head;
};

// ─── Construction / loading ─────────────────────────────────────────────────

MaskDecoder::MaskDecoder(MaskDecoderConfig cfg)
    : cfg_(std::move(cfg)), w_(std::make_unique<MaskDecoderWeights>()) {
    if (cfg_.hidden_size % cfg_.num_attention_heads != 0)
        fail("hidden_size must be divisible by num_attention_heads");
    if (cfg_.hidden_size % cfg_.attention_downsample_rate != 0)
        fail("hidden_size must be divisible by attention_downsample_rate");
    if (cfg_.hidden_size % 8 != 0)
        fail("hidden_size must be divisible by 8 (upscaler / hypernetwork dim)");
}

MaskDecoder::~MaskDecoder() = default;
MaskDecoder::MaskDecoder(MaskDecoder&&) noexcept = default;
MaskDecoder& MaskDecoder::operator=(MaskDecoder&&) noexcept = default;

void MaskDecoder::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void MaskDecoder::load_file(const std::string& path) {
    st::File f = st::File::open(path);

    const int D    = cfg_.hidden_size;
    const int K    = cfg_.num_mask_tokens();
    const int rate = cfg_.attention_downsample_rate;
    const int cint = D / rate;          // cross/final internal dim
    const int md   = cfg_.mlp_dim;
    const int cup1 = D / 4, cup2 = D / 8;
    const std::string pre = "mask_decoder.";

    auto load_attn = [&](const std::string& p, int internal) {
        AttnW a;
        a.q_w   = load_whole(f, kWho, p + "q_proj.weight",   internal, D);
        a.q_b   = load_whole(f, kWho, p + "q_proj.bias",     internal, 1);
        a.k_w   = load_whole(f, kWho, p + "k_proj.weight",   internal, D);
        a.k_b   = load_whole(f, kWho, p + "k_proj.bias",     internal, 1);
        a.v_w   = load_whole(f, kWho, p + "v_proj.weight",   internal, D);
        a.v_b   = load_whole(f, kWho, p + "v_proj.bias",     internal, 1);
        a.out_w = load_whole(f, kWho, p + "out_proj.weight", D, internal);
        a.out_b = load_whole(f, kWho, p + "out_proj.bias",   D, 1);
        return a;
    };
    auto load_ln = [&](const std::string& p) {
        LN l;
        l.w = load_whole(f, kWho, p + ".weight", D, 1);
        l.b = load_whole(f, kWho, p + ".bias",   D, 1);
        return l;
    };
    // SamFeedForward with `hidden_layers` intermediate (hidden,hidden) linears.
    auto load_ff = [&](const std::string& p, int in, int hidden, int out,
                       int hidden_layers) {
        FFW ff;
        ff.in_w  = load_whole(f, kWho, p + "proj_in.weight",  hidden, in);
        ff.in_b  = load_whole(f, kWho, p + "proj_in.bias",    hidden, 1);
        for (int j = 0; j < hidden_layers; ++j) {
            ff.lw.push_back(load_whole(
                f, kWho, p + "layers." + std::to_string(j) + ".weight", hidden, hidden));
            ff.lb.push_back(load_whole(
                f, kWho, p + "layers." + std::to_string(j) + ".bias", hidden, 1));
        }
        ff.out_w = load_whole(f, kWho, p + "proj_out.weight", out, hidden);
        ff.out_b = load_whole(f, kWho, p + "proj_out.bias",   out, 1);
        return ff;
    };

    MaskDecoderWeights w;
    w.iou_token   = load_whole(f, kWho, pre + "iou_token.weight",   1, D);
    w.mask_tokens = load_whole(f, kWho, pre + "mask_tokens.weight", K, D);

    w.layers.resize(cfg_.num_hidden_layers);
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        LayerW& L = w.layers[i];
        const std::string lp = pre + "transformer.layers." + std::to_string(i) + ".";
        L.self_attn = load_attn(lp + "self_attn.", D);     // self: downsample 1
        L.ln1 = load_ln(lp + "layer_norm1");
        L.cross_t2i = load_attn(lp + "cross_attn_token_to_image.", cint);
        L.ln2 = load_ln(lp + "layer_norm2");
        L.mlp1_w = load_whole(f, kWho, lp + "mlp.lin1.weight", md, D);
        L.mlp1_b = load_whole(f, kWho, lp + "mlp.lin1.bias",   md, 1);
        L.mlp2_w = load_whole(f, kWho, lp + "mlp.lin2.weight", D, md);
        L.mlp2_b = load_whole(f, kWho, lp + "mlp.lin2.bias",   D, 1);
        L.ln3 = load_ln(lp + "layer_norm3");
        L.cross_i2t = load_attn(lp + "cross_attn_image_to_token.", cint);
        L.ln4 = load_ln(lp + "layer_norm4");
    }
    w.final_attn = load_attn(pre + "transformer.final_attn_token_to_image.", cint);
    w.ln_final   = load_ln(pre + "transformer.layer_norm_final_attn");

    w.uc1_w  = load_whole(f, kWho, pre + "upscale_conv1.weight", D, cup1 * 2 * 2);
    w.uc1_b  = load_whole(f, kWho, pre + "upscale_conv1.bias",   cup1, 1);
    w.up_ln.w = load_whole(f, kWho, pre + "upscale_layer_norm.weight", cup1, 1);
    w.up_ln.b = load_whole(f, kWho, pre + "upscale_layer_norm.bias",   cup1, 1);
    w.uc2_w  = load_whole(f, kWho, pre + "upscale_conv2.weight", cup1, cup2 * 2 * 2);
    w.uc2_b  = load_whole(f, kWho, pre + "upscale_conv2.bias",   cup2, 1);

    // Hypernetwork MLPs: SamFeedForward(D, D, D/8, num_layers=3) -> 1 hidden.
    w.hyper.resize(K);
    for (int i = 0; i < K; ++i)
        w.hyper[i] = load_ff(
            pre + "output_hypernetworks_mlps." + std::to_string(i) + ".",
            D, D, cup2, /*hidden_layers=*/1);

    w.iou_head = load_ff(pre + "iou_prediction_head.",
                         D, cfg_.iou_head_hidden_dim, K,
                         /*hidden_layers=*/cfg_.iou_head_depth - 2);

    w.loaded = true;
    *w_ = std::move(w);
}

namespace {

// Attention projections migrate at `dt` — FP16 on a mixed-precision GPU
// backend. They are the only decoder weights whose GEMMs scale with B*HW
// rows under decode_batched; everything else (token MLPs, hypernetworks,
// IoU head, upscaler convs — conv_transpose2d is FP32-only) stays FP32.
void mv_attn(AttnW& a, brotensor::Device dev, brotensor::Dtype dt) {
    auto mv = [&](Tensor& t) {
        if (!t.data) return;
        t = t.to(dev);
        if (t.dtype != dt) { Tensor c; brotensor::cast(t, c, dt); t = std::move(c); }
    };
    mv(a.q_w); mv(a.q_b); mv(a.k_w); mv(a.k_b);
    mv(a.v_w); mv(a.v_b); mv(a.out_w); mv(a.out_b);
}
void mv_ln(LN& l, brotensor::Device dev) {
    if (l.w.data) l.w = l.w.to(dev);
    if (l.b.data) l.b = l.b.to(dev);
}
void mv_ff(FFW& ff, brotensor::Device dev) {
    auto mv = [dev](Tensor& t) { if (t.data) t = t.to(dev); };
    mv(ff.in_w); mv(ff.in_b);
    for (auto& t : ff.lw) mv(t);
    for (auto& t : ff.lb) mv(t);
    mv(ff.out_w); mv(ff.out_b);
}

}  // namespace

void MaskDecoder::to(brotensor::Device dev) {
    if (!w_->loaded) fail("to() called before load()");
    if (dev == device_) return;
    const bool fp16 = dev != brotensor::Device::CPU &&
                      brotensor::compute_dtype() == brotensor::Dtype::FP16;
    const brotensor::Dtype attn_dt = fp16 ? brotensor::Dtype::FP16
                                          : brotensor::Dtype::FP32;
    auto mv = [dev](Tensor& t) { if (t.data) t = t.to(dev); };
    // iou_token / mask_tokens stay host: they feed the host token assembly in
    // decode(), which uploads the finished token block once.
    for (LayerW& L : w_->layers) {
        mv_attn(L.self_attn, dev, attn_dt); mv_ln(L.ln1, dev);
        mv_attn(L.cross_t2i, dev, attn_dt); mv_ln(L.ln2, dev);
        mv(L.mlp1_w); mv(L.mlp1_b); mv(L.mlp2_w); mv(L.mlp2_b); mv_ln(L.ln3, dev);
        mv_attn(L.cross_i2t, dev, attn_dt); mv_ln(L.ln4, dev);
    }
    mv_attn(w_->final_attn, dev, attn_dt); mv_ln(w_->ln_final, dev);
    mv(w_->uc1_w); mv(w_->uc1_b); mv_ln(w_->up_ln, dev); mv(w_->uc2_w); mv(w_->uc2_b);
    for (FFW& ff : w_->hyper) mv_ff(ff, dev);
    mv_ff(w_->iou_head, dev);
    device_ = dev;
    fp16_ = fp16;
}

// ─── Forward helpers ─────────────────────────────────────────────────────────

namespace {

// A device-resident INT32 cu_seqlens prefix-sum [0, seqlen, 2*seqlen, ...,
// batch*seqlen] (length batch+1) for a packed batch of equal-length sequences.
// Built on the host then migrated, so the same code path yields a host pointer
// on CPU and a device pointer on GPU. batch==1 reduces to [0, seqlen].
Tensor make_cu(brotensor::Device dev, int batch, int seqlen) {
    std::vector<int32_t> h(static_cast<std::size_t>(batch) + 1);
    for (int i = 0; i <= batch; ++i) h[static_cast<std::size_t>(i)] = i * seqlen;
    Tensor cpu = Tensor::view(brotensor::Device::CPU, h.data(), batch + 1, 1,
                              brotensor::Dtype::INT32);
    return cpu.to(dev);  // owning copy on dev (clone() when dev == CPU)
}

// out = SamAttention(q_in, k_in, v_in): project to internal dim, multi-head
// scaled-dot-product over pre-projected Q/K/V, output projection back to D.
// `batch` packs that many independent sequences block-diagonally: Q holds
// batch*seqlen_q rows, K/V batch*seqlen_k rows; sequences do not cross-attend.
// batch==1 (the default, with seqlen_* derived from the row counts) is the
// single-prompt path.
// out = SamAttention(q_in, k_in, v_in) with FP32 inputs/output. When fp16 is
// set (mixed-precision GPU path, weights pre-cast by to()), the inputs drop
// to half for the projections + flash attention and the result widens back —
// these are the only decoder GEMMs whose row counts scale with B*HW.
void run_attn(const AttnW& a, const Tensor& q_in, const Tensor& k_in,
              const Tensor& v_in, int num_heads, bool fp16, Tensor& out,
              int batch = 1, int seqlen_q = -1, int seqlen_k = -1) {
    auto to16 = [&](const Tensor& t) {
        if (!fp16) return t;
        Tensor c; brotensor::cast(t, c, brotensor::Dtype::FP16); return c;
    };
    auto linear = [&](const Tensor& W, const Tensor& bias, const Tensor& X,
                      Tensor& Y) {
        if (fp16) brotensor::linear_forward_batched_fp16(W, &bias, X, Y);
        else      brotensor::linear_forward_batched(W, bias, X, Y);
    };
    Tensor qc = to16(q_in), kc = to16(k_in), vc = to16(v_in);
    Tensor q, k, v;
    linear(a.q_w, a.q_b, qc, q);  // (Lq, internal)
    linear(a.k_w, a.k_b, kc, k);  // (Lk, internal)
    linear(a.v_w, a.v_b, vc, v);  // (Lk, internal)
    const int Lq = q.rows, Lk = k.rows, internal = q.cols;
    const int head_dim = internal / num_heads;
    const int sq = seqlen_q < 0 ? Lq : seqlen_q;
    const int sk = seqlen_k < 0 ? Lk : seqlen_k;
    Tensor cu_q = make_cu(q.device, batch, sq);
    Tensor cu_k = make_cu(k.device, batch, sk);
    Tensor o;
    brotensor::flash_attention_varlen_forward(
        q, k, v,
        reinterpret_cast<const int32_t*>(cu_q.data),
        reinterpret_cast<const int32_t*>(cu_k.data),
        batch, /*max_seqlen_q=*/sq, /*max_seqlen_k=*/sk,
        num_heads, head_dim, /*causal=*/false, o);              // (Lq, internal)
    Tensor proj;
    linear(a.out_w, a.out_b, o, proj);                          // (Lq, D)
    if (fp16) brotensor::cast(proj, out, brotensor::Dtype::FP32);
    else      out = std::move(proj);
}

// Replicate a (R,C) tensor B times along its rows -> (B*R, C), entirely on
// the tensor's device. Used to broadcast the shared image keys / positional
// encoding across a prompt batch so each prompt gets its own evolving copy.
Tensor tile_rows(const Tensor& t, int B) {
    const int R = t.rows, C = t.cols;
    Tensor out = Tensor::empty_on(t.device, B * R, C, t.dtype);
    const int plane = R * C;
    for (int b = 0; b < B; ++b)
        brotensor::copy_d2d(t, 0, out, b * plane, plane);
    return out;
}

Tensor add_pe(const Tensor& base, const Tensor& pe) {
    Tensor t = base.clone();
    brotensor::add_inplace(t, pe);
    return t;
}

void ln_apply(Tensor& X, const LN& l, float eps) {
    Tensor y;
    brotensor::layernorm_forward_inference_batched(X, l.w, l.b, y, eps);
    X = std::move(y);
}

// SamFeedForward forward over (B,in) -> (B,out): relu after proj_in and each
// hidden linear; proj_out raw.
Tensor ff_apply(const FFW& ff, const Tensor& x) {
    Tensor h;
    brotensor::linear_forward_batched(ff.in_w, ff.in_b, x, h);
    brotensor::relu_forward_batched(h, h);
    for (std::size_t j = 0; j < ff.lw.size(); ++j) {
        Tensor t;
        brotensor::linear_forward_batched(ff.lw[j], ff.lb[j], h, t);
        brotensor::relu_forward_batched(t, t);
        h = std::move(t);
    }
    Tensor o;
    brotensor::linear_forward_batched(ff.out_w, ff.out_b, h, o);
    return o;
}

// Non-owning (rows, cols) FP32 view starting at element offset `off` into t.
Tensor row_view(const Tensor& t, int off, int rows, int cols) {
    return Tensor::view(t.device,
                        reinterpret_cast<float*>(t.data) + off, rows, cols);
}

}  // namespace

// ─── Decode ──────────────────────────────────────────────────────────────────

DecodedMasks MaskDecoder::decode(const brotensor::Tensor& image_embedding,
                                 const brotensor::Tensor& image_pe,
                                 const brotensor::Tensor& sparse,
                                 const brotensor::Tensor& dense,
                                 bool multimask_output) const {
    if (!w_->loaded) fail("decode() called before load()");
    const int D  = cfg_.hidden_size;
    const int g  = cfg_.grid();
    const int HW = g * g;
    const int K  = cfg_.num_mask_tokens();
    const int nh = cfg_.num_attention_heads;
    const float eps = cfg_.layer_norm_eps;

    auto want_map = [&](const Tensor& t, const char* name) {
        if (t.dtype != brotensor::Dtype::FP32) fail(std::string(name) + " must be FP32");
        if (t.device != device_) fail(std::string(name) + " must be on device()");
        if (t.rows != 1 || t.cols != D * HW)
            fail(std::string(name) + " must be (1, hidden*grid*grid)");
    };
    want_map(image_embedding, "image_embedding");
    want_map(image_pe, "image_pe");
    want_map(dense, "dense");
    const int n_sparse = (sparse.data && sparse.size() > 0) ? sparse.rows : 0;
    if (n_sparse > 0) {
        if (sparse.dtype != brotensor::Dtype::FP32) fail("sparse must be FP32");
        if (sparse.device != device_) fail("sparse must be on device()");
        if (sparse.cols != D) fail("sparse must have hidden_size columns");
    }

    // ── Assemble query tokens [iou, masks(0..K-1), sparse] on host, upload. ──
    const int n_tok = 1 + K + n_sparse;
    Tensor tokens;
    {
        Tensor th = Tensor::mat(n_tok, D);
        float* dst = th.host_f32_mut();
        const float* iou = w_->iou_token.host_f32();          // (1,D)
        const float* mt  = w_->mask_tokens.host_f32();         // (K,D)
        for (int c = 0; c < D; ++c) dst[c] = iou[c];
        for (int i = 0; i < K; ++i)
            for (int c = 0; c < D; ++c)
                dst[static_cast<std::size_t>(1 + i) * D + c] = mt[i * D + c];
        if (n_sparse > 0) {
            Tensor sp_host = sparse.to(brotensor::Device::CPU);
            const float* sp = sp_host.host_f32();
            for (int i = 0; i < n_sparse; ++i)
                for (int c = 0; c < D; ++c)
                    dst[static_cast<std::size_t>(1 + K + i) * D + c] = sp[i * D + c];
        }
        tokens = th.to(device_);
    }

    // ── keys = flatten(image_embedding + dense); image PE flattened. ──
    Tensor src = add_pe(image_embedding, dense);   // (1, D*HW) NCHW
    Tensor keys;       brotensor::nchw_to_sequence(src, 1, D, g, g, keys);     // (HW,D)
    Tensor image_pe_s; brotensor::nchw_to_sequence(image_pe, 1, D, g, g, image_pe_s);

    Tensor queries = tokens;                  // (n_tok, D)
    const Tensor point_embeddings = tokens;   // query PE re-added each attention

    // ── Two-way transformer ──
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const LayerW& L = w_->layers[i];
        Tensor attn;

        // (a) self-attention on the tokens. The first layer skips the query PE
        //     AND has no residual — SAM's skip_first_layer_pe replaces queries
        //     with the attention output rather than adding it.
        if (i == 0) {
            run_attn(L.self_attn, queries, queries, queries, nh, fp16_, attn);
            queries = std::move(attn);
        } else {
            Tensor q = add_pe(queries, point_embeddings);
            run_attn(L.self_attn, q, q, queries, nh, fp16_, attn);
            brotensor::add_inplace(queries, attn);
        }
        ln_apply(queries, L.ln1, eps);

        // (b) cross-attention tokens -> image
        {
            Tensor q = add_pe(queries, point_embeddings);
            Tensor k = add_pe(keys, image_pe_s);
            run_attn(L.cross_t2i, q, k, keys, nh, fp16_, attn);
            brotensor::add_inplace(queries, attn);
            ln_apply(queries, L.ln2, eps);
        }

        // (c) token MLP (ReLU activation — SAM's mask-decoder hidden_act)
        {
            Tensor m1; brotensor::linear_forward_batched(L.mlp1_w, L.mlp1_b, queries, m1);
            Tensor act; brotensor::relu_forward_batched(m1, act);
            Tensor m2; brotensor::linear_forward_batched(L.mlp2_w, L.mlp2_b, act, m2);
            brotensor::add_inplace(queries, m2);
            ln_apply(queries, L.ln3, eps);
        }

        // (d) cross-attention image -> tokens (updates keys)
        {
            Tensor q = add_pe(queries, point_embeddings);
            Tensor k = add_pe(keys, image_pe_s);
            // query = image (k), key = tokens (q), value = tokens (queries)
            run_attn(L.cross_i2t, k, q, queries, nh, fp16_, attn);   // (HW, D)
            brotensor::add_inplace(keys, attn);
            ln_apply(keys, L.ln4, eps);
        }
    }

    // ── Final token -> image attention ──
    {
        Tensor attn;
        Tensor q = add_pe(queries, point_embeddings);
        Tensor k = add_pe(keys, image_pe_s);
        run_attn(w_->final_attn, q, k, keys, nh, fp16_, attn);
        brotensor::add_inplace(queries, attn);
        ln_apply(queries, w_->ln_final, eps);
    }

    // ── Upscale keys 4x: NCHW -> convT(2) -> LN -> gelu -> convT(2) -> gelu ──
    const int cup1 = D / 4, cup2 = D / 8, ms = cfg_.mask_size();
    Tensor src_nchw; brotensor::sequence_to_nchw(keys, 1, D, g, g, src_nchw);
    Tensor u1;
    brotensor::conv_transpose2d_forward(src_nchw, w_->uc1_w, &w_->uc1_b,
                                        1, D, g, g, cup1, 2, 2, 2, 2, 0, 0,
                                        0, 0, 1, 1, 1, u1);     // (1, cup1*2g*2g)
    {
        Tensor seq; brotensor::nchw_to_sequence(u1, 1, cup1, 2 * g, 2 * g, seq);
        Tensor n;   brotensor::layernorm_forward_inference_batched(seq, w_->up_ln.w,
                                                                   w_->up_ln.b, n, eps);
        brotensor::sequence_to_nchw(n, 1, cup1, 2 * g, 2 * g, u1);
    }
    Tensor a1; brotensor::gelu_exact_forward(u1, a1);
    Tensor u2;
    brotensor::conv_transpose2d_forward(a1, w_->uc2_w, &w_->uc2_b,
                                        1, cup1, 2 * g, 2 * g, cup2, 2, 2, 2, 2,
                                        0, 0, 0, 0, 1, 1, 1, u2);  // (1, cup2*4g*4g)
    Tensor upscaled; brotensor::gelu_exact_forward(u2, upscaled);

    // ── Hypernetwork filters per mask token, then masks = filters @ upscaled.
    //    Each token's MLP yields a (cup2) filter; stack the K rows into a
    //    (K, cup2) matrix (rows gathered host-side — K is tiny). ──
    Tensor mask_tokens_out = row_view(queries, D, K, D);   // queries rows 1..K
    Tensor filters;
    {
        Tensor fh = Tensor::mat(K, cup2);
        for (int i = 0; i < K; ++i) {
            Tensor tok = row_view(mask_tokens_out, i * D, 1, D);
            Tensor hi  = ff_apply(w_->hyper[i], tok).to(brotensor::Device::CPU);
            const float* hp = hi.host_f32();
            float* dst = fh.host_f32_mut() + static_cast<std::size_t>(i) * cup2;
            for (int c = 0; c < cup2; ++c) dst[c] = hp[c];
        }
        filters = fh.to(device_);
    }

    Tensor upscaled_mat = row_view(upscaled, 0, cup2, ms * ms);  // (cup2, ms*ms)
    Tensor masks_all;
    brotensor::matmul(filters, upscaled_mat, masks_all);         // (K, ms*ms)

    // ── IoU scores from the iou token ──
    Tensor iou_tok = row_view(queries, 0, 1, D);
    Tensor iou_all = ff_apply(w_->iou_head, iou_tok);            // (1, K)

    // ── Select multimask slice (skip the first "ambiguity" mask) or best. ──
    const int start   = multimask_output ? 1 : 0;
    const int num_out = multimask_output ? K - 1 : 1;

    DecodedMasks out;
    out.num_out   = num_out;
    out.mask_size = ms;
    out.masks = row_view(masks_all, start * ms * ms, num_out, ms * ms).clone();
    {
        Tensor iou_host = iou_all.to(brotensor::Device::CPU);
        const float* ip = iou_host.host_f32();
        Tensor io = Tensor::mat(num_out, 1);
        for (int j = 0; j < num_out; ++j) io[j] = ip[start + j];
        out.iou = io.to(device_);
    }
    return out;
}

// ─── Batched decode ────────────────────────────────────────────────────────────
//
// The same two-way transformer as decode(), run once over B prompts packed
// block-diagonally. Tokens, keys, and the image PE all carry a batch dimension;
// the packed-varlen attention keeps prompts from cross-attending. The control
// flow mirrors decode() exactly (so B==1 is numerically the single-prompt path),
// differing only in the per-prompt tiling of the shared image maps and the
// per-prompt gather of the hypernetwork filters / IoU tokens at the end.

DecodedMasks MaskDecoder::decode_batched(const brotensor::Tensor& image_embedding,
                                         const brotensor::Tensor& image_pe,
                                         const brotensor::Tensor& sparse,
                                         int batch, int n_sparse_per_prompt,
                                         const brotensor::Tensor& dense,
                                         bool multimask_output) const {
    if (!w_->loaded) fail("decode_batched() called before load()");
    const int B  = batch;
    const int S  = n_sparse_per_prompt;
    const int D  = cfg_.hidden_size;
    const int g  = cfg_.grid();
    const int HW = g * g;
    const int K  = cfg_.num_mask_tokens();
    const int nh = cfg_.num_attention_heads;
    const float eps = cfg_.layer_norm_eps;
    if (B < 1) fail("batch must be >= 1");
    if (S < 0) fail("n_sparse_per_prompt must be >= 0");

    auto want_map = [&](const Tensor& t, const char* name) {
        if (t.dtype != brotensor::Dtype::FP32) fail(std::string(name) + " must be FP32");
        if (t.device != device_) fail(std::string(name) + " must be on device()");
        if (t.rows != 1 || t.cols != D * HW)
            fail(std::string(name) + " must be (1, hidden*grid*grid)");
    };
    want_map(image_embedding, "image_embedding");
    want_map(image_pe, "image_pe");
    want_map(dense, "dense");
    if (S > 0) {
        if (sparse.dtype != brotensor::Dtype::FP32) fail("sparse must be FP32");
        if (sparse.device != device_) fail("sparse must be on device()");
        if (sparse.cols != D) fail("sparse must have hidden_size columns");
        if (sparse.rows != B * S) fail("sparse must have batch*n_sparse_per_prompt rows");
    }

    // ── Assemble query tokens, per prompt [iou, masks(0..K-1), sparse(S)]. ──
    const int n_tok = 1 + K + S;
    const int T     = B * n_tok;
    Tensor tokens;
    {
        Tensor th = Tensor::mat(T, D);
        float* dst = th.host_f32_mut();
        const float* iou = w_->iou_token.host_f32();   // (1,D)
        const float* mt  = w_->mask_tokens.host_f32();  // (K,D)
        Tensor sp_host = (S > 0) ? sparse.to(brotensor::Device::CPU) : Tensor{};
        const float* sp = (S > 0) ? sp_host.host_f32() : nullptr;
        for (int b = 0; b < B; ++b) {
            float* base = dst + static_cast<std::size_t>(b) * n_tok * D;
            for (int c = 0; c < D; ++c) base[c] = iou[c];
            for (int i = 0; i < K; ++i)
                for (int c = 0; c < D; ++c)
                    base[static_cast<std::size_t>(1 + i) * D + c] = mt[i * D + c];
            for (int i = 0; i < S; ++i)
                for (int c = 0; c < D; ++c)
                    base[static_cast<std::size_t>(1 + K + i) * D + c] =
                        sp[static_cast<std::size_t>(b * S + i) * D + c];
        }
        tokens = th.to(device_);
    }

    // ── keys = flatten(image_embedding + dense), tiled B times; image PE too. ──
    Tensor src = add_pe(image_embedding, dense);   // (1, D*HW) NCHW
    Tensor keys1;       brotensor::nchw_to_sequence(src, 1, D, g, g, keys1);     // (HW,D)
    Tensor image_pe1;   brotensor::nchw_to_sequence(image_pe, 1, D, g, g, image_pe1);
    Tensor keys       = tile_rows(keys1, B);       // (B*HW, D)
    Tensor image_pe_s = tile_rows(image_pe1, B);   // (B*HW, D)

    Tensor queries = tokens;                  // (T, D)
    const Tensor point_embeddings = tokens;   // query PE re-added each attention

    // ── Two-way transformer (block-diagonal over B prompts) ──
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const LayerW& L = w_->layers[i];
        Tensor attn;

        // (a) self-attention on the tokens (skip_first_layer_pe on layer 0).
        if (i == 0) {
            run_attn(L.self_attn, queries, queries, queries, nh, fp16_, attn, B, n_tok, n_tok);
            queries = std::move(attn);
        } else {
            Tensor q = add_pe(queries, point_embeddings);
            run_attn(L.self_attn, q, q, queries, nh, fp16_, attn, B, n_tok, n_tok);
            brotensor::add_inplace(queries, attn);
        }
        ln_apply(queries, L.ln1, eps);

        // (b) cross-attention tokens -> image
        {
            Tensor q = add_pe(queries, point_embeddings);
            Tensor k = add_pe(keys, image_pe_s);
            run_attn(L.cross_t2i, q, k, keys, nh, fp16_, attn, B, n_tok, HW);
            brotensor::add_inplace(queries, attn);
            ln_apply(queries, L.ln2, eps);
        }

        // (c) token MLP (ReLU)
        {
            Tensor m1; brotensor::linear_forward_batched(L.mlp1_w, L.mlp1_b, queries, m1);
            Tensor act; brotensor::relu_forward_batched(m1, act);
            Tensor m2; brotensor::linear_forward_batched(L.mlp2_w, L.mlp2_b, act, m2);
            brotensor::add_inplace(queries, m2);
            ln_apply(queries, L.ln3, eps);
        }

        // (d) cross-attention image -> tokens (updates keys)
        {
            Tensor q = add_pe(queries, point_embeddings);
            Tensor k = add_pe(keys, image_pe_s);
            run_attn(L.cross_i2t, k, q, queries, nh, fp16_, attn, B, HW, n_tok);   // (B*HW, D)
            brotensor::add_inplace(keys, attn);
            ln_apply(keys, L.ln4, eps);
        }
    }

    // ── Final token -> image attention ──
    {
        Tensor attn;
        Tensor q = add_pe(queries, point_embeddings);
        Tensor k = add_pe(keys, image_pe_s);
        run_attn(w_->final_attn, q, k, keys, nh, fp16_, attn, B, n_tok, HW);
        brotensor::add_inplace(queries, attn);
        ln_apply(queries, w_->ln_final, eps);
    }

    // ── Upscale keys 4x (batched NCHW) ──
    const int cup1 = D / 4, cup2 = D / 8, ms = cfg_.mask_size();
    Tensor src_nchw; brotensor::sequence_to_nchw(keys, B, D, g, g, src_nchw);  // (B, D*HW)
    Tensor u1;
    brotensor::conv_transpose2d_forward(src_nchw, w_->uc1_w, &w_->uc1_b,
                                        B, D, g, g, cup1, 2, 2, 2, 2, 0, 0,
                                        0, 0, 1, 1, 1, u1);     // (B, cup1*2g*2g)
    {
        Tensor seq; brotensor::nchw_to_sequence(u1, B, cup1, 2 * g, 2 * g, seq);
        Tensor n;   brotensor::layernorm_forward_inference_batched(seq, w_->up_ln.w,
                                                                   w_->up_ln.b, n, eps);
        brotensor::sequence_to_nchw(n, B, cup1, 2 * g, 2 * g, u1);
    }
    Tensor a1; brotensor::gelu_exact_forward(u1, a1);
    Tensor u2;
    brotensor::conv_transpose2d_forward(a1, w_->uc2_w, &w_->uc2_b,
                                        B, cup1, 2 * g, 2 * g, cup2, 2, 2, 2, 2,
                                        0, 0, 0, 0, 1, 1, 1, u2);  // (B, cup2*4g*4g)
    Tensor upscaled; brotensor::gelu_exact_forward(u2, upscaled);  // (B, cup2*ms*ms)

    // ── Hypernetwork filters per (prompt, mask token), device-resident. For
    //    token i, gather_rows pulls the B token rows (one per prompt) out of
    //    queries, hyper[i] maps the (B, D) block to (B, cup2), and a strided
    //    copy interleaves it into F so prompt b's filter matrix is the
    //    contiguous (K, cup2) block F[b*K .. b*K+K). ──
    Tensor F = Tensor::empty_on(device_, B * K, cup2);
    {
        Tensor idx_h = Tensor::zeros_on(brotensor::Device::CPU, B, 1,
                                        brotensor::Dtype::INT32);
        int32_t* ip = static_cast<int32_t*>(idx_h.data);
        for (int i = 0; i < K; ++i) {
            for (int b = 0; b < B; ++b) ip[b] = b * n_tok + 1 + i;
            Tensor idx = idx_h.to(device_);
            Tensor mt_i;
            brotensor::gather_rows(queries, idx, mt_i);            // (B, D)
            Tensor hi = ff_apply(w_->hyper[i], mt_i);              // (B, cup2)
            brotensor::copy_d2d_strided(hi, 0, cup2,
                                        F, i * cup2, K * cup2, cup2, B);
        }
    }

    // ── IoU scores: gather the B iou tokens, run the head once over (B, D);
    //    the (B, K) result is tiny, so the slice stays on the host. ──
    Tensor iou_all_host;
    {
        Tensor idx_h = Tensor::zeros_on(brotensor::Device::CPU, B, 1,
                                        brotensor::Dtype::INT32);
        int32_t* ip = static_cast<int32_t*>(idx_h.data);
        for (int b = 0; b < B; ++b) ip[b] = b * n_tok;
        Tensor idx = idx_h.to(device_);
        Tensor iou_tok;
        brotensor::gather_rows(queries, idx, iou_tok);             // (B, D)
        iou_all_host = ff_apply(w_->iou_head, iou_tok)
                           .to(brotensor::Device::CPU);            // (B, K)
    }

    // ── Per prompt: masks = filters @ upscaled_b on-device, then copy out the
    //    multimask slice (skip the first "ambiguity" mask) or the single best
    //    mask. The full-batch mask tensor never visits the host. ──
    const int start   = multimask_output ? 1 : 0;
    const int num_out = multimask_output ? K - 1 : 1;

    Tensor masks_dev = Tensor::empty_on(device_, B * num_out, ms * ms);
    for (int b = 0; b < B; ++b) {
        Tensor fb = row_view(F, b * K * cup2, K, cup2);
        Tensor upscaled_b = row_view(upscaled, b * cup2 * ms * ms, cup2, ms * ms);
        Tensor mb_all;
        brotensor::matmul(fb, upscaled_b, mb_all);                 // (K, ms*ms)
        brotensor::copy_d2d(mb_all, start * ms * ms,
                            masks_dev, b * num_out * ms * ms,
                            num_out * ms * ms);
    }

    Tensor iou_h = Tensor::mat(B * num_out, 1);
    float* idst = iou_h.host_f32_mut();
    const float* ibp = iou_all_host.host_f32();
    for (int b = 0; b < B; ++b)
        for (int j = 0; j < num_out; ++j)
            idst[static_cast<std::size_t>(b * num_out + j)] = ibp[b * K + start + j];

    DecodedMasks out;
    out.num_out   = num_out;
    out.mask_size = ms;
    out.masks = std::move(masks_dev);  // (B*num_out, ms*ms)
    out.iou   = iou_h.to(device_);     // (B*num_out, 1)
    return out;
}

}  // namespace brovisionml::sam
