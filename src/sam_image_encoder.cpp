#include "brovisionml/sam_image_encoder.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace brovisionml::sam {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("sam::ImageEncoder: " + msg);
}

// Widen a flat element range [elem_off, elem_off + rows*cols) of a safetensors
// view (F32 / F16 / BF16) into a fresh host FP32 (rows, cols) tensor. This is
// the single load primitive: it keeps every weight host-resident FP32 (the
// encoder's compute dtype) regardless of the checkpoint's on-disk precision,
// and the element-offset lets a combined qkv block be sliced into Q/K/V without
// a second copy.
Tensor load_f32(const st::TensorView& v, std::size_t elem_off,
                int rows, int cols, const std::string& label) {
    const std::size_t need = static_cast<std::size_t>(rows) * cols;
    const std::size_t have = static_cast<std::size_t>(v.numel());
    if (elem_off + need > have)
        fail(label + ": range [" + std::to_string(elem_off) + ", " +
             std::to_string(elem_off + need) + ") exceeds tensor numel " +
             std::to_string(have) + " (key '" + v.name + "')");

    Tensor out = Tensor::mat(rows, cols);  // host FP32, zero-filled
    float* dst = out.host_f32_mut();
    switch (v.dtype) {
        case st::Dtype::F32: {
            const float* src = reinterpret_cast<const float*>(v.data) + elem_off;
            for (std::size_t i = 0; i < need; ++i) dst[i] = src[i];
            break;
        }
        case st::Dtype::F16: {
            const uint16_t* src =
                reinterpret_cast<const uint16_t*>(v.data) + elem_off;
            for (std::size_t i = 0; i < need; ++i)
                dst[i] = brotensor::fp16_bits_to_fp32(src[i]);
            break;
        }
        case st::Dtype::BF16: {
            const uint16_t* src =
                reinterpret_cast<const uint16_t*>(v.data) + elem_off;
            for (std::size_t i = 0; i < need; ++i)
                dst[i] = brotensor::bf16_bits_to_fp32(src[i]);
            break;
        }
        default:
            fail(label + ": unsupported dtype " +
                 std::string(st::dtype_name(v.dtype)) + " (key '" + v.name + "')");
    }
    return out;
}

// Fetch a named view, asserting its total element count.
const st::TensorView& need_view(const st::File& f, const std::string& name,
                                int64_t expect_numel) {
    const st::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    if (v->numel() != expect_numel)
        fail("tensor '" + name + "' has " + std::to_string(v->numel()) +
             " elements, expected " + std::to_string(expect_numel));
    return *v;
}

// Whole-tensor widen-to-FP32 with an element-count check.
Tensor load_whole(const st::File& f, const std::string& name,
                  int rows, int cols) {
    const st::TensorView& v =
        need_view(f, name, static_cast<int64_t>(rows) * cols);
    return load_f32(v, 0, rows, cols, name);
}

}  // namespace

// ─── Per-block + whole-encoder weight tables (host FP32) ───────────────────

struct BlockWeights {
    Tensor ln1_w, ln1_b;                       // (D,1)
    Tensor q_w, q_b, k_w, k_b, v_w, v_b;       // (D,D) / (D,1)
    Tensor proj_w, proj_b;                     // (D,D) / (D,1)
    Tensor rel_pos_h, rel_pos_w;               // (2*grid-1, head_dim)
    Tensor ln2_w, ln2_b;                       // (D,1)
    Tensor mlp1_w, mlp1_b;                      // (mlp_dim,D) / (mlp_dim,1)
    Tensor mlp2_w, mlp2_b;                      // (D,mlp_dim) / (D,1)
    bool   global = false;
};

struct EncoderWeights {
    bool   loaded = false;
    Tensor patch_w, patch_b;                   // (D, in*p*p) / (D,1)
    Tensor pos_embed;                          // (grid*grid, D)
    std::vector<BlockWeights> blocks;
    Tensor neck_c1_w;                          // (out, D)         1x1 conv
    Tensor neck_ln1_w, neck_ln1_b;             // (out,1)
    Tensor neck_c2_w;                          // (out, out*3*3)   3x3 conv
    Tensor neck_ln2_w, neck_ln2_b;             // (out,1)
};

// ─── Config presets ────────────────────────────────────────────────────────

EncoderConfig EncoderConfig::vit_h() { return EncoderConfig{}; }  // defaults

EncoderConfig EncoderConfig::vit_l() {
    EncoderConfig c;
    c.embed_dim = 1024;
    c.depth     = 24;
    c.num_heads = 16;
    c.global_attn_indexes = {5, 11, 17, 23};
    return c;
}

EncoderConfig EncoderConfig::vit_b() {
    EncoderConfig c;
    c.embed_dim = 768;
    c.depth     = 12;
    c.num_heads = 12;
    c.global_attn_indexes = {2, 5, 8, 11};
    return c;
}

// ─── Construction / loading ─────────────────────────────────────────────────

ImageEncoder::ImageEncoder(EncoderConfig cfg)
    : cfg_(std::move(cfg)), w_(std::make_unique<EncoderWeights>()) {
    if (cfg_.embed_dim % cfg_.num_heads != 0)
        fail("embed_dim must be divisible by num_heads");
    if (cfg_.img_size % cfg_.patch_size != 0)
        fail("img_size must be divisible by patch_size");
}

ImageEncoder::~ImageEncoder() = default;
ImageEncoder::ImageEncoder(ImageEncoder&&) noexcept = default;
ImageEncoder& ImageEncoder::operator=(ImageEncoder&&) noexcept = default;

void ImageEncoder::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void ImageEncoder::load_file(const std::string& path) {
    st::File f = st::File::open(path);

    const int D   = cfg_.embed_dim;
    const int g   = cfg_.grid();
    const int hd  = cfg_.head_dim();
    const int md  = cfg_.mlp_dim();
    const int out = cfg_.out_chans;
    const int p   = cfg_.patch_size;
    const std::string pre = "vision_encoder.";

    EncoderWeights w;

    // Patch embed: PyTorch conv weight (D, in, p, p) flattens to brotensor's
    // OIHW (C_out, C_in*kH*kW); bias is a (D,1) column.
    w.patch_w = load_whole(f, pre + "patch_embed.projection.weight",
                           D, cfg_.in_chans * p * p);
    w.patch_b = load_whole(f, pre + "patch_embed.projection.bias", D, 1);

    // Absolute position embedding: stored (1, g, g, D); its row-major (h,w,c)
    // layout is exactly the (g*g, D) token order nchw_to_sequence emits.
    w.pos_embed = load_whole(f, pre + "pos_embed", g * g, D);

    auto is_global = [&](int i) {
        for (int gi : cfg_.global_attn_indexes) if (gi == i) return true;
        return false;
    };

    w.blocks.resize(cfg_.depth);
    for (int i = 0; i < cfg_.depth; ++i) {
        BlockWeights& b = w.blocks[i];
        const std::string lp = pre + "layers." + std::to_string(i) + ".";
        b.global = is_global(i);

        b.ln1_w = load_whole(f, lp + "layer_norm1.weight", D, 1);
        b.ln1_b = load_whole(f, lp + "layer_norm1.bias",   D, 1);

        // Combined qkv: weight (3D, D), bias (3D,). Rows split q | k | v, each
        // (D, D) — matching SAM's reshape(.., 3, num_heads, head_dim).
        const st::TensorView& qkv_w =
            need_view(f, lp + "attn.qkv.weight", static_cast<int64_t>(3) * D * D);
        const st::TensorView& qkv_b =
            need_view(f, lp + "attn.qkv.bias", static_cast<int64_t>(3) * D);
        const std::size_t Dsq = static_cast<std::size_t>(D) * D;
        b.q_w = load_f32(qkv_w, 0 * Dsq, D, D, lp + "attn.qkv.weight[q]");
        b.k_w = load_f32(qkv_w, 1 * Dsq, D, D, lp + "attn.qkv.weight[k]");
        b.v_w = load_f32(qkv_w, 2 * Dsq, D, D, lp + "attn.qkv.weight[v]");
        b.q_b = load_f32(qkv_b, 0 * D, D, 1, lp + "attn.qkv.bias[q]");
        b.k_b = load_f32(qkv_b, 1 * D, D, 1, lp + "attn.qkv.bias[k]");
        b.v_b = load_f32(qkv_b, 2 * D, D, 1, lp + "attn.qkv.bias[v]");

        b.proj_w = load_whole(f, lp + "attn.proj.weight", D, D);
        b.proj_b = load_whole(f, lp + "attn.proj.bias",   D, 1);

        // rel_pos sized for this block's attention grid: window for local
        // blocks, full grid for global blocks (no interpolation needed).
        const int rg = b.global ? g : cfg_.window_size;
        b.rel_pos_h = load_whole(f, lp + "attn.rel_pos_h", 2 * rg - 1, hd);
        b.rel_pos_w = load_whole(f, lp + "attn.rel_pos_w", 2 * rg - 1, hd);

        b.ln2_w = load_whole(f, lp + "layer_norm2.weight", D, 1);
        b.ln2_b = load_whole(f, lp + "layer_norm2.bias",   D, 1);

        b.mlp1_w = load_whole(f, lp + "mlp.lin1.weight", md, D);
        b.mlp1_b = load_whole(f, lp + "mlp.lin1.bias",   md, 1);
        b.mlp2_w = load_whole(f, lp + "mlp.lin2.weight", D, md);
        b.mlp2_b = load_whole(f, lp + "mlp.lin2.bias",   D, 1);
    }

    // Neck: 1x1 then 3x3 conv, each bias-free, each followed by a channels-first
    // LayerNorm2d. 1x1 weight (out, D, 1, 1) -> (out, D); 3x3 (out, out, 3, 3).
    w.neck_c1_w  = load_whole(f, pre + "neck.conv1.weight", out, D);
    w.neck_ln1_w = load_whole(f, pre + "neck.layer_norm1.weight", out, 1);
    w.neck_ln1_b = load_whole(f, pre + "neck.layer_norm1.bias",   out, 1);
    w.neck_c2_w  = load_whole(f, pre + "neck.conv2.weight", out, out * 3 * 3);
    w.neck_ln2_w = load_whole(f, pre + "neck.layer_norm2.weight", out, 1);
    w.neck_ln2_b = load_whole(f, pre + "neck.layer_norm2.bias",   out, 1);

    w.loaded = true;
    *w_ = std::move(w);
}

// ─── Forward ─────────────────────────────────────────────────────────────

namespace {

// LayerNorm2d (channels-first): normalize across the C channels at each of the
// H*W pixels, then per-channel affine. Expressed device-neutrally by viewing
// the NCHW map as (H*W, C) tokens — one per pixel — and reusing batched
// LayerNorm, which normalizes each row over its C entries. X is (1, C*H*W).
void layer_norm_2d(Tensor& X, const Tensor& gamma, const Tensor& beta,
                   int C, int H, int W, float eps) {
    Tensor seq;                                       // (H*W, C)
    brotensor::nchw_to_sequence(X, 1, C, H, W, seq);
    Tensor normed;
    brotensor::layernorm_forward_inference_batched(seq, gamma, beta, normed, eps);
    brotensor::sequence_to_nchw(normed, 1, C, H, W, X);
}

// Windowed multi-head attention with decomposed rel-pos. Mirrors SAM's
// window_partition: zero-pad the bottom/right of the grid to a multiple of the
// window, run the attention kernel independently per window (grid == window),
// then crop back. The per-window gather/scatter is host-side, so X must be
// host FP32.
Tensor windowed_attention(const Tensor& X, const BlockWeights& b,
                          int g, int win, int num_heads, float scale) {
    const int D   = X.cols;
    const int pad = (win - g % win) % win;
    const int gp  = g + pad;          // padded grid side
    const int nw  = gp / win;         // windows per side

    // Scatter content into a zero-padded (gp*gp, D) grid (top-left aligned).
    Tensor padded = Tensor::mat(gp * gp, D);
    {
        const float* src = X.host_f32();
        float* dst = padded.host_f32_mut();
        for (int h = 0; h < g; ++h)
            for (int wv = 0; wv < g; ++wv) {
                const float* s = src + static_cast<std::size_t>(h * g + wv) * D;
                float* d = dst + static_cast<std::size_t>(h * gp + wv) * D;
                for (int c = 0; c < D; ++c) d[c] = s[c];
            }
    }

    Tensor out_padded = Tensor::mat(gp * gp, D);
    Tensor win_in     = Tensor::mat(win * win, D);
    Tensor win_out;
    for (int wy = 0; wy < nw; ++wy) {
        for (int wx = 0; wx < nw; ++wx) {
            // Gather this window's tokens in row-major (r, c) order.
            float* wi = win_in.host_f32_mut();
            const float* pad_p = padded.host_f32();
            for (int r = 0; r < win; ++r)
                for (int c = 0; c < win; ++c) {
                    const int srow = (wy * win + r) * gp + (wx * win + c);
                    const float* s = pad_p + static_cast<std::size_t>(srow) * D;
                    float* d = wi + static_cast<std::size_t>(r * win + c) * D;
                    for (int e = 0; e < D; ++e) d[e] = s[e];
                }

            brotensor::self_attention_decomposed_rel_pos_forward(
                win_in, b.q_w, &b.q_b, b.k_w, &b.k_b, b.v_w, &b.v_b,
                b.proj_w, &b.proj_b, b.rel_pos_h, b.rel_pos_w,
                num_heads, win, win, scale, win_out);

            const float* wo = win_out.host_f32();
            float* op = out_padded.host_f32_mut();
            for (int r = 0; r < win; ++r)
                for (int c = 0; c < win; ++c) {
                    const int drow = (wy * win + r) * gp + (wx * win + c);
                    const float* s = wo + static_cast<std::size_t>(r * win + c) * D;
                    float* d = op + static_cast<std::size_t>(drow) * D;
                    for (int e = 0; e < D; ++e) d[e] = s[e];
                }
        }
    }

    // Crop the padding away, back to (g*g, D).
    Tensor out = Tensor::mat(g * g, D);
    {
        const float* src = out_padded.host_f32();
        float* dst = out.host_f32_mut();
        for (int h = 0; h < g; ++h)
            for (int wv = 0; wv < g; ++wv) {
                const float* s = src + static_cast<std::size_t>(h * gp + wv) * D;
                float* d = dst + static_cast<std::size_t>(h * g + wv) * D;
                for (int c = 0; c < D; ++c) d[c] = s[c];
            }
    }
    return out;
}

}  // namespace

brotensor::Tensor ImageEncoder::encode(const brotensor::Tensor& pixels) const {
    if (!w_->loaded) fail("encode() called before load()");
    if (pixels.device != brotensor::Device::CPU || pixels.dtype != brotensor::Dtype::FP32)
        fail("encode() requires a host FP32 pixel tensor");
    const int S = cfg_.img_size;
    if (pixels.rows != 1 || pixels.cols != cfg_.in_chans * S * S)
        fail("pixels must be (1, in_chans*img_size*img_size)");

    const int D  = cfg_.embed_dim;
    const int g  = cfg_.grid();
    const int p  = cfg_.patch_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim()));
    const float eps   = cfg_.layer_norm_eps;

    // 1. Patch embed: conv 3->D, kernel=stride=patch, no padding -> (1, D*g*g).
    Tensor feat;
    brotensor::conv2d_forward(pixels, w_->patch_w, &w_->patch_b,
                              /*N=*/1, cfg_.in_chans, S, S,
                              /*C_out=*/D, p, p,
                              /*stride=*/p, p, /*pad=*/0, 0, /*dil=*/1, 1, feat);

    // 2. NCHW -> token-major (g*g, D), then add absolute position embedding.
    Tensor x;
    brotensor::nchw_to_sequence(feat, 1, D, g, g, x);
    brotensor::add_inplace(x, w_->pos_embed);

    // 3. Transformer blocks.
    for (const BlockWeights& b : w_->blocks) {
        // Attention sub-block: x = x + attn(LN1(x)).
        Tensor h;
        brotensor::layernorm_forward_inference_batched(x, b.ln1_w, b.ln1_b, h, eps);
        Tensor attn;
        if (b.global) {
            brotensor::self_attention_decomposed_rel_pos_forward(
                h, b.q_w, &b.q_b, b.k_w, &b.k_b, b.v_w, &b.v_b,
                b.proj_w, &b.proj_b, b.rel_pos_h, b.rel_pos_w,
                cfg_.num_heads, g, g, scale, attn);
        } else {
            attn = windowed_attention(h, b, g, cfg_.window_size,
                                      cfg_.num_heads, scale);
        }
        brotensor::add_inplace(x, attn);

        // MLP sub-block: x = x + lin2(gelu(lin1(LN2(x)))).
        Tensor h2;
        brotensor::layernorm_forward_inference_batched(x, b.ln2_w, b.ln2_b, h2, eps);
        Tensor m1;
        brotensor::linear_forward_batched(b.mlp1_w, b.mlp1_b, h2, m1);
        Tensor act;
        brotensor::gelu_exact_forward(m1, act);
        Tensor m2;
        brotensor::linear_forward_batched(b.mlp2_w, b.mlp2_b, act, m2);
        brotensor::add_inplace(x, m2);
    }

    // 4. Neck: back to NCHW, conv1 (1x1) -> LN2d -> conv2 (3x3) -> LN2d.
    const int out = cfg_.out_chans;
    Tensor nchw;
    brotensor::sequence_to_nchw(x, 1, D, g, g, nchw);

    Tensor c1;
    brotensor::conv2d_forward(nchw, w_->neck_c1_w, /*bias=*/nullptr,
                              1, D, g, g, out, 1, 1, 1, 1, 0, 0, 1, 1, c1);
    layer_norm_2d(c1, w_->neck_ln1_w, w_->neck_ln1_b, out, g, g, eps);

    Tensor c2;
    brotensor::conv2d_forward(c1, w_->neck_c2_w, /*bias=*/nullptr,
                              1, out, g, g, out, 3, 3, 1, 1, 1, 1, 1, 1, c2);
    layer_norm_2d(c2, w_->neck_ln2_w, w_->neck_ln2_b, out, g, g, eps);

    return c2;  // (1, out*g*g) NCHW image embedding
}

}  // namespace brovisionml::sam
