#include "brovisionml/sam2_image_encoder.h"

#include "sam2_ops.h"
#include "profile.h"

#include "brotensor/ops.h"
#include "brotensor/ops/fused.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brovisionml::sam2 {

namespace {

namespace bt = brotensor;
using bt::Tensor;
using namespace detail;

constexpr const char* kWho = "sam2::ImageEncoder: ";

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(kWho + msg); }

}  // namespace

// ─── Weights ────────────────────────────────────────────────────────────────

struct HieraBlock {
    int dim = 0, dim_out = 0, heads = 0, window = 0, qstride = 0;
    bool stage_end = false;
    Tensor ln1_w, ln1_b;
    Tensor q_w, q_b, k_w, k_b, v_w, v_b;   // (dim_out, dim) / (dim_out, 1)
    Tensor proj_w, proj_b;                 // (dim_out, dim_out)
    Tensor ln2_w, ln2_b;
    Tensor mlp1_w, mlp1_b, mlp2_w, mlp2_b;
    Tensor dim_w, dim_b;                   // residual projection when dim != dim_out
};

struct ImageEncoderWeights {
    bool loaded = false;
    Tensor patch_w, patch_b;   // (C0, 3*k*k)
    Tensor pos_tokens;         // (H0*W0, C0): interpolated background + tiled window table
    std::vector<HieraBlock> blocks;
    std::vector<Tensor> lat_w, lat_b;   // lateral 1x1 per stage, indexed by stage
    Tensor s0_w, s0_b, s1_w, s1_b;
    Tensor top_pos;            // (g*g, hidden)
};

// ─── Sine position encoding ─────────────────────────────────────────────────

brotensor::Tensor sine_position_tokens(int H, int W, int npf) {
    // Sam2SinePositionEmbedding(normalize=True, scale=2*pi, temperature=1e4):
    // the cumulative-sum embeds are 1..H / 1..W, normalized by (last + 1e-6).
    const double two_pi = 6.283185307179586;
    Tensor out = Tensor::mat(H * W, 2 * npf);
    float* d = out.host_f32_mut();
    std::vector<double> dim_t(static_cast<std::size_t>(npf));
    for (int i = 0; i < npf; ++i)
        dim_t[static_cast<std::size_t>(i)] =
            std::pow(10000.0, 2.0 * static_cast<double>(i / 2) / static_cast<double>(npf));
    for (int y = 0; y < H; ++y) {
        const float ye = static_cast<float>(y + 1) / (static_cast<float>(H) + 1e-6f) *
                         static_cast<float>(two_pi);
        for (int x = 0; x < W; ++x) {
            const float xe = static_cast<float>(x + 1) / (static_cast<float>(W) + 1e-6f) *
                             static_cast<float>(two_pi);
            float* row = d + (static_cast<std::size_t>(y) * W + x) * 2 * npf;
            for (int i = 0; i < npf; ++i) {
                const float dt = static_cast<float>(dim_t[static_cast<std::size_t>(i)]);
                const float py = ye / dt, px = xe / dt;
                row[i]       = (i % 2 == 0) ? std::sin(py) : std::cos(py);
                row[npf + i] = (i % 2 == 0) ? std::sin(px) : std::cos(px);
            }
        }
    }
    return out;
}

// ─── Construction / loading ─────────────────────────────────────────────────

ImageEncoder::ImageEncoder(const Sam2Config& cfg)
    : cfg_(cfg), w_(std::make_unique<ImageEncoderWeights>()) {}
ImageEncoder::~ImageEncoder() = default;
ImageEncoder::ImageEncoder(ImageEncoder&&) noexcept = default;
ImageEncoder& ImageEncoder::operator=(ImageEncoder&&) noexcept = default;

void ImageEncoder::load(const st::File& f) {
    const HieraConfig& h = cfg_.hiera;
    const int S = cfg_.image_size;
    const int k = h.patch_kernel;
    const int C0 = h.embed_dim_per_stage[0];
    const int H0 = (S + 2 * h.patch_padding - k) / h.patch_stride + 1;
    const std::string bb = "vision_encoder.backbone.";
    ImageEncoderWeights w;

    w.patch_w = load_w(f, kWho, bb + "patch_embed.projection.weight", C0, 3 * k * k);
    w.patch_b = load_w(f, kWho, bb + "patch_embed.projection.bias", C0, 1);

    // Windowed absolute position embedding: bicubic-resize the background
    // table to the patch grid, then add the window table tiled across it.
    {
        const int bh = h.pos_embed_bg[0], bw = h.pos_embed_bg[1];
        const int w0 = h.window_size_per_stage[0];
        if (H0 % w0 != 0) fail("patch grid not a multiple of the stage-0 window");
        Tensor bg = load_w(f, kWho, bb + "pos_embed", 1, C0 * bh * bw);
        Tensor win = load_w(f, kWho, bb + "pos_embed_window", 1, C0 * w0 * w0);
        Tensor up;
        bt::interp2d_forward(bg, 1, C0, bh, bw, H0, H0, /*bicubic a=-0.75*/ 3, up);
        float* u = up.host_f32_mut();
        const float* wv = win.host_f32();
        for (int c = 0; c < C0; ++c)
            for (int y = 0; y < H0; ++y)
                for (int x = 0; x < H0; ++x)
                    u[(static_cast<std::size_t>(c) * H0 + y) * H0 + x] +=
                        wv[(static_cast<std::size_t>(c) * w0 + y % w0) * w0 + x % w0];
        bt::nchw_to_sequence(up, 1, C0, H0, H0, w.pos_tokens);
    }

    // Blocks.
    int total = 0;
    for (int s = 0; s < h.num_stages(); ++s) {
        for (int b = 0; b < h.blocks_per_stage[static_cast<std::size_t>(s)]; ++b, ++total) {
            HieraBlock B;
            const bool first = s > 0 && b == 0;
            B.dim     = first ? h.embed_dim_per_stage[static_cast<std::size_t>(s - 1)]
                              : h.embed_dim_per_stage[static_cast<std::size_t>(s)];
            B.dim_out = h.embed_dim_per_stage[static_cast<std::size_t>(s)];
            B.heads   = h.heads_per_stage[static_cast<std::size_t>(s)];
            B.window  = first ? h.window_size_per_stage[static_cast<std::size_t>(s - 1)]
                              : h.window_size_per_stage[static_cast<std::size_t>(s)];
            if (std::find(h.global_attention_blocks.begin(), h.global_attention_blocks.end(),
                          total) != h.global_attention_blocks.end())
                B.window = 0;
            B.qstride = (s > 0 && s <= h.num_query_pool_stages && b == 0) ? h.query_stride : 0;
            B.stage_end = (b == h.blocks_per_stage[static_cast<std::size_t>(s)] - 1);

            const std::string p = bb + "blocks." + std::to_string(total) + ".";
            const int D = B.dim, Do = B.dim_out, md = static_cast<int>(Do * h.mlp_ratio);
            B.ln1_w = load_w(f, kWho, p + "layer_norm1.weight", D, 1);
            B.ln1_b = load_w(f, kWho, p + "layer_norm1.bias", D, 1);
            B.q_w = load_rows(f, kWho, p + "attn.qkv.weight", 0 * Do, Do, D);
            B.k_w = load_rows(f, kWho, p + "attn.qkv.weight", 1 * Do, Do, D);
            B.v_w = load_rows(f, kWho, p + "attn.qkv.weight", 2 * Do, Do, D);
            B.q_b = load_rows(f, kWho, p + "attn.qkv.bias", 0 * Do, Do, 1);
            B.k_b = load_rows(f, kWho, p + "attn.qkv.bias", 1 * Do, Do, 1);
            B.v_b = load_rows(f, kWho, p + "attn.qkv.bias", 2 * Do, Do, 1);
            B.proj_w = load_w(f, kWho, p + "attn.proj.weight", Do, Do);
            B.proj_b = load_w(f, kWho, p + "attn.proj.bias", Do, 1);
            B.ln2_w = load_w(f, kWho, p + "layer_norm2.weight", Do, 1);
            B.ln2_b = load_w(f, kWho, p + "layer_norm2.bias", Do, 1);
            B.mlp1_w = load_w(f, kWho, p + "mlp.proj_in.weight", md, Do);
            B.mlp1_b = load_w(f, kWho, p + "mlp.proj_in.bias", md, 1);
            B.mlp2_w = load_w(f, kWho, p + "mlp.proj_out.weight", Do, md);
            B.mlp2_b = load_w(f, kWho, p + "mlp.proj_out.bias", Do, 1);
            if (D != Do) {
                B.dim_w = load_w(f, kWho, p + "proj.weight", Do, D);
                B.dim_b = load_w(f, kWho, p + "proj.bias", Do, 1);
            }
            w.blocks.push_back(std::move(B));
        }
    }

    // FPN laterals: neck.convs[j] reads stage (n - j).
    const int n = h.num_stages() - 1;
    const int F = cfg_.fpn_hidden_size;
    w.lat_w.resize(static_cast<std::size_t>(n + 1));
    w.lat_b.resize(static_cast<std::size_t>(n + 1));
    for (int i = 0; i <= n; ++i) {
        const std::string p = "vision_encoder.neck.convs." + std::to_string(n - i) + ".";
        const int C = h.embed_dim_per_stage[static_cast<std::size_t>(i)];
        w.lat_w[static_cast<std::size_t>(i)] = load_w(f, kWho, p + "weight", F, C);
        w.lat_b[static_cast<std::size_t>(i)] = load_w(f, kWho, p + "bias", F, 1);
    }
    const int hid = cfg_.hidden_size;
    w.s0_w = load_w(f, kWho, "mask_decoder.conv_s0.weight", hid / 8, hid);
    w.s0_b = load_w(f, kWho, "mask_decoder.conv_s0.bias", hid / 8, 1);
    w.s1_w = load_w(f, kWho, "mask_decoder.conv_s1.weight", hid / 4, hid);
    w.s1_b = load_w(f, kWho, "mask_decoder.conv_s1.bias", hid / 4, 1);

    w.top_pos = sine_position_tokens(cfg_.grid(), cfg_.grid(), F / 2);
    w.loaded = true;
    *w_ = std::move(w);
    device_ = bt::Device::CPU;
    fp16_ = false;
}

void ImageEncoder::to(bt::Device dev) {
    if (!w_->loaded) fail("to() called before load()");
    if (dev == device_) return;
    const bool fp16 = dev != bt::Device::CPU && bt::compute_dtype() == bt::Dtype::FP16;
    const bt::Dtype f32 = bt::Dtype::FP32;
    const bt::Dtype half = fp16 ? bt::Dtype::FP16 : f32;
    ImageEncoderWeights& w = *w_;
    // FP16 GEMMs on the hot path; layout ops, LayerNorms, the residual stream,
    // position tables and the FPN stay FP32.
    move(w.patch_w, dev, half); move(w.patch_b, dev, half);
    move(w.pos_tokens, dev, f32);
    for (HieraBlock& b : w.blocks) {
        move(b.ln1_w, dev, f32); move(b.ln1_b, dev, f32);
        move(b.q_w, dev, half); move(b.q_b, dev, half);
        move(b.k_w, dev, half); move(b.k_b, dev, half);
        move(b.v_w, dev, half); move(b.v_b, dev, half);
        move(b.proj_w, dev, half); move(b.proj_b, dev, half);
        move(b.ln2_w, dev, f32); move(b.ln2_b, dev, f32);
        move(b.mlp1_w, dev, half); move(b.mlp1_b, dev, half);
        move(b.mlp2_w, dev, half); move(b.mlp2_b, dev, half);
        move(b.dim_w, dev, half); move(b.dim_b, dev, half);
    }
    for (Tensor& t : w.lat_w) move(t, dev, f32);
    for (Tensor& t : w.lat_b) move(t, dev, f32);
    move(w.s0_w, dev, f32); move(w.s0_b, dev, f32);
    move(w.s1_w, dev, f32); move(w.s1_b, dev, f32);
    move(w.top_pos, dev, f32);
    device_ = dev;
    fp16_ = fp16;
}

const brotensor::Tensor& ImageEncoder::top_pos() const { return w_->top_pos; }

// ─── Forward ────────────────────────────────────────────────────────────────

namespace {

// 2x2 max pool of a token map (H*W, C) -> (H/2*W/2, C), N maps packed.
Tensor pool_tokens(const Tensor& t, int N, int C, int H, int W) {
    Tensor n = to_nchw(t, N, C, H, W);
    Tensor y, idx;
    bt::max_pool2d_forward(n, N, C, H, W, 2, 2, 2, 2, 0, 0, y, idx);
    return to_tokens(y, N, C, H / 2, W / 2);
}

// (H*W, C) tokens -> (nw*ws*ws, C) windows, zero-padding H/W up to ws.
Tensor partition(const Tensor& t, int C, int H, int W, int ws, int& Hp, int& Wp) {
    Hp = (H + ws - 1) / ws * ws;
    Wp = (W + ws - 1) / ws * ws;
    Tensor n = to_nchw(t, 1, C, H, W);
    if (Hp != H || Wp != W) {
        Tensor p;
        bt::pad2d_forward(n, 1, C, H, W, 0, Hp - H, 0, Wp - W, /*zero*/ 0, p);
        n = std::move(p);
    }
    Tensor win;
    bt::window_partition_forward(n, 1, C, Hp, Wp, ws, win);
    const int nw = (Hp / ws) * (Wp / ws);
    return to_tokens(win, nw, C, ws, ws);
}

// Inverse of partition at window side `wq`, cropping the pad back off.
Tensor unpartition(const Tensor& t, int C, int wq, int H, int W) {
    const int Hp = (H + wq - 1) / wq * wq;
    const int Wp = (W + wq - 1) / wq * wq;
    const int nw = (Hp / wq) * (Wp / wq);
    Tensor n = to_nchw(t, nw, C, wq, wq);
    Tensor full;
    bt::window_reverse_forward(n, 1, C, Hp, Wp, wq, full);
    if (Hp != H || Wp != W) {
        Tensor c;
        bt::slice2d_forward(full, 1, C, Hp, Wp, 0, 0, H, W, c);
        full = std::move(c);
    }
    return to_tokens(full, 1, C, H, W);
}

}  // namespace

ImageFeatures ImageEncoder::encode(const brotensor::Tensor& pixels) const {
    if (!w_->loaded) fail("encode() called before load()");
    const int S = cfg_.image_size;
    if (pixels.dtype != bt::Dtype::FP32 || pixels.device != device_ ||
        pixels.rows != 1 || pixels.cols != 3 * S * S)
        fail("pixels must be FP32 (1, 3*S*S) on the encoder's device");
    const HieraConfig& h = cfg_.hiera;
    const ImageEncoderWeights& w = *w_;
    const float eps = h.layer_norm_eps;
    const bool fp16 = fp16_;

    // 1. Patch embed + position table.
    const int C0 = h.embed_dim_per_stage[0];
    const int k = h.patch_kernel;
    int H = (S + 2 * h.patch_padding - k) / h.patch_stride + 1;
    int W = H;
    Tensor x;
    {
        Tensor px = to16(pixels, fp16);
        Tensor feat;
        bt::conv2d_forward(px, w.patch_w, &w.patch_b, 1, 3, S, S, C0, k, k,
                           h.patch_stride, h.patch_stride, h.patch_padding, h.patch_padding,
                           1, 1, feat);
        x = to32(to_tokens(feat, 1, C0, H, W));
        bt::add_inplace(x, w.pos_tokens);
    }
    brovisionml::detail::profile_mark(device_, nullptr);

    // 2. Multiscale blocks; keep each stage's output for the FPN.
    std::vector<Tensor> stage_out;
    std::vector<int> stage_hw;
    for (const HieraBlock& b : w.blocks) {
        const int Do = b.dim_out;
        Tensor hn = layer_norm(x, b.ln1_w, b.ln1_b, eps);

        const int Ho = b.qstride ? H / b.qstride : H;
        const int Wo = b.qstride ? W / b.qstride : W;
        Tensor residual;
        if (b.dim != Do) {
            residual = linear32(b.dim_w, b.dim_b, hn, fp16);
            if (b.qstride) residual = pool_tokens(residual, 1, Do, H, W);
        } else {
            residual = std::move(x);   // x is reassigned from residual below
        }

        // Window partition (window 0 = one global "window" spanning the map).
        int nw = 1, wh = H, ww = W;
        Tensor hw;
        if (b.window > 0) {
            int Hp = 0, Wp = 0;
            hw = partition(hn, b.dim, H, W, b.window, Hp, Wp);
            nw = (Hp / b.window) * (Wp / b.window);
            wh = ww = b.window;
        } else {
            hw = std::move(hn);
        }

        Tensor h16 = to16(hw, fp16);
        Tensor q, kk, v;
        linear(b.q_w, b.q_b, h16, q, fp16);
        linear(b.k_w, b.k_b, h16, kk, fp16);
        linear(b.v_w, b.v_b, h16, v, fp16);
        int qh = wh, qw = ww;
        if (b.qstride) {
            q = pool_tokens(q, nw, Do, wh, ww);
            qh /= b.qstride;
            qw /= b.qstride;
        }
        Tensor o = attention(q, kk, v, nw, qh * qw, wh * ww, b.heads);
        Tensor proj;
        linear(b.proj_w, b.proj_b, o, proj, fp16);
        Tensor a = to32(proj);
        if (b.window > 0) a = unpartition(a, Do, qh, Ho, Wo);

        x = std::move(residual);
        Tensor h2;
        bt::fused_residual_layernorm(x, a, b.ln2_w, b.ln2_b, eps, h2);   // x += a; h2 = LN2(x)
        Tensor m1;
        linear(b.mlp1_w, b.mlp1_b, to16(h2, fp16), m1, fp16);
        Tensor act;
        bt::gelu_exact_forward(m1, act);
        Tensor m2;
        linear(b.mlp2_w, b.mlp2_b, act, m2, fp16);
        bt::add_inplace(x, to32(m2));
        H = Ho;
        W = Wo;
        if (b.stage_end) {
            stage_out.push_back(x);
            stage_hw.push_back(H);
            static const char* kStage[] = {"hiera-stage0", "hiera-stage1", "hiera-stage2",
                                           "hiera-stage3"};
            brovisionml::detail::profile_mark(device_, kStage[std::min<std::size_t>(stage_out.size() - 1, 3)]);
        }
    }

    // 3. FPN neck, deepest level first. Only the levels in fpn_top_down_levels
    //    receive the nearest-upsampled coarser map.
    const int n = h.num_stages() - 1;
    const int F = cfg_.fpn_hidden_size;
    std::vector<Tensor> level(static_cast<std::size_t>(n + 1));
    Tensor prev;
    int prev_hw = 0;
    for (int i = n; i >= 0; --i) {
        const int side = stage_hw[static_cast<std::size_t>(i)];
        Tensor lat;
        bt::linear_forward_batched(w.lat_w[static_cast<std::size_t>(i)],
                                   w.lat_b[static_cast<std::size_t>(i)],
                                   stage_out[static_cast<std::size_t>(i)], lat);
        const bool td = std::find(cfg_.fpn_top_down_levels.begin(),
                                  cfg_.fpn_top_down_levels.end(), i) !=
                        cfg_.fpn_top_down_levels.end();
        if (td && i != n) {
            Tensor up;
            bt::upsample_nearest_2x(to_nchw(prev, 1, F, prev_hw, prev_hw), 1, F, prev_hw,
                                    prev_hw, up);
            bt::add_inplace(lat, to_tokens(up, 1, F, side, side));
        }
        prev = std::move(lat);
        prev_hw = side;
        level[static_cast<std::size_t>(i)] = prev;
    }

    ImageFeatures out;
    const int hid = cfg_.hidden_size;
    if (stage_hw[2] != cfg_.grid()) fail("stride-16 level does not match the decoder grid");
    out.top = std::move(level[2]);
    {
        Tensor s0;
        bt::linear_forward_batched(w.s0_w, w.s0_b, level[0], s0);
        out.feat_s0 = to_nchw(s0, 1, hid / 8, stage_hw[0], stage_hw[0]);
        Tensor s1;
        bt::linear_forward_batched(w.s1_w, w.s1_b, level[1], s1);
        out.feat_s1 = to_nchw(s1, 1, hid / 4, stage_hw[1], stage_hw[1]);
    }
    brovisionml::detail::profile_mark(device_, "fpn");
    return out;
}

}  // namespace brovisionml::sam2
