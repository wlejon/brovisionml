#include "brovisionml/sam_mask_decoder.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include "sam_weights_util.h"

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

void mv_attn(AttnW& a, brotensor::Device dev) {
    auto mv = [dev](Tensor& t) { if (t.data) t = t.to(dev); };
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
    auto mv = [dev](Tensor& t) { if (t.data) t = t.to(dev); };
    // iou_token / mask_tokens stay host: they feed the host token assembly in
    // decode(), which uploads the finished token block once.
    for (LayerW& L : w_->layers) {
        mv_attn(L.self_attn, dev); mv_ln(L.ln1, dev);
        mv_attn(L.cross_t2i, dev); mv_ln(L.ln2, dev);
        mv(L.mlp1_w); mv(L.mlp1_b); mv(L.mlp2_w); mv(L.mlp2_b); mv_ln(L.ln3, dev);
        mv_attn(L.cross_i2t, dev); mv_ln(L.ln4, dev);
    }
    mv_attn(w_->final_attn, dev); mv_ln(w_->ln_final, dev);
    mv(w_->uc1_w); mv(w_->uc1_b); mv_ln(w_->up_ln, dev); mv(w_->uc2_w); mv(w_->uc2_b);
    for (FFW& ff : w_->hyper) mv_ff(ff, dev);
    mv_ff(w_->iou_head, dev);
    device_ = dev;
}

// ─── Forward helpers ─────────────────────────────────────────────────────────

namespace {

// A device-resident INT32 cu_seqlens prefix-sum [0, total] (length 2, the
// batch_size==1 case). Built on the host then migrated, so the same code path
// yields a host pointer on CPU and a device pointer on GPU.
Tensor make_cu(brotensor::Device dev, int total) {
    int32_t h[2] = {0, total};
    Tensor cpu = Tensor::view(brotensor::Device::CPU, h, 2, 1, brotensor::Dtype::INT32);
    return cpu.to(dev);  // owning copy on dev (clone() when dev == CPU)
}

// out = SamAttention(q_in, k_in, v_in): project to internal dim, multi-head
// scaled-dot-product over pre-projected Q/K/V, output projection back to D.
void run_attn(const AttnW& a, const Tensor& q_in, const Tensor& k_in,
              const Tensor& v_in, int num_heads, Tensor& out) {
    Tensor q, k, v;
    brotensor::linear_forward_batched(a.q_w, a.q_b, q_in, q);  // (Lq, internal)
    brotensor::linear_forward_batched(a.k_w, a.k_b, k_in, k);  // (Lk, internal)
    brotensor::linear_forward_batched(a.v_w, a.v_b, v_in, v);  // (Lk, internal)
    const int Lq = q.rows, Lk = k.rows, internal = q.cols;
    const int head_dim = internal / num_heads;
    Tensor cu_q = make_cu(q.device, Lq);
    Tensor cu_k = make_cu(k.device, Lk);
    Tensor o;
    brotensor::flash_attention_varlen_forward(
        q, k, v,
        reinterpret_cast<const int32_t*>(cu_q.data),
        reinterpret_cast<const int32_t*>(cu_k.data),
        /*batch_size=*/1, /*max_seqlen_q=*/Lq, /*max_seqlen_k=*/Lk,
        num_heads, head_dim, /*causal=*/false, o);              // (Lq, internal)
    brotensor::linear_forward_batched(a.out_w, a.out_b, o, out);  // (Lq, D)
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

        // (a) self-attention on the tokens (skip query PE on the first layer)
        if (i == 0) {
            run_attn(L.self_attn, queries, queries, queries, nh, attn);
        } else {
            Tensor q = add_pe(queries, point_embeddings);
            run_attn(L.self_attn, q, q, queries, nh, attn);
        }
        brotensor::add_inplace(queries, attn);
        ln_apply(queries, L.ln1, eps);

        // (b) cross-attention tokens -> image
        {
            Tensor q = add_pe(queries, point_embeddings);
            Tensor k = add_pe(keys, image_pe_s);
            run_attn(L.cross_t2i, q, k, keys, nh, attn);
            brotensor::add_inplace(queries, attn);
            ln_apply(queries, L.ln2, eps);
        }

        // (c) token MLP
        {
            Tensor m1; brotensor::linear_forward_batched(L.mlp1_w, L.mlp1_b, queries, m1);
            Tensor act; brotensor::gelu_exact_forward(m1, act);
            Tensor m2; brotensor::linear_forward_batched(L.mlp2_w, L.mlp2_b, act, m2);
            brotensor::add_inplace(queries, m2);
            ln_apply(queries, L.ln3, eps);
        }

        // (d) cross-attention image -> tokens (updates keys)
        {
            Tensor q = add_pe(queries, point_embeddings);
            Tensor k = add_pe(keys, image_pe_s);
            // query = image (k), key = tokens (q), value = tokens (queries)
            run_attn(L.cross_i2t, k, q, queries, nh, attn);   // (HW, D)
            brotensor::add_inplace(keys, attn);
            ln_apply(keys, L.ln4, eps);
        }
    }

    // ── Final token -> image attention ──
    {
        Tensor attn;
        Tensor q = add_pe(queries, point_embeddings);
        Tensor k = add_pe(keys, image_pe_s);
        run_attn(w_->final_attn, q, k, keys, nh, attn);
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

}  // namespace brovisionml::sam
