// BiRefNet — Swin-L backbone + ASPP-deformable decoder, composed from brotensor
// ops. See include/brovisionml/birefnet.h for the architecture overview.
//
// This translation unit holds the whole net. The Swin window machinery (pad +
// cyclic roll + window partition + reverse, and patch merging) is expressed as
// precomputed INT32 row permutations driven through gather_rows, so it runs on
// whatever device the activations live on. The per-window attention is the
// extended self_attention_bias_forward (qkv/proj bias + a precomputed
// relative-position + shifted-window additive bias).

#include "brovisionml/birefnet.h"

#include "weights_util.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brovisionml::birefnet {

namespace bt = brotensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_f32;
using brovisionml::detail::need_view;
using bt::Tensor;
using bt::Device;
using bt::Dtype;

namespace {

constexpr int WIN = 12;
constexpr float LN_EPS = 1e-5f;
const int DEPTHS[4] = {2, 2, 18, 2};
const int HEADS[4]  = {6, 12, 24, 48};
const int DIMS[4]   = {192, 384, 768, 1536};

// ── Swin weight containers ──────────────────────────────────────────────────

struct Attn {
    Tensor Wq, Wk, Wv, bq, bk, bv, Wo, bo;
    Tensor rpb;          // (heads*N, N) FP32 relative-position bias (no shift)
    int heads = 0, N = 0;
};
struct Block {
    Tensor ln1_w, ln1_b, ln2_w, ln2_b;
    Attn   attn;
    Tensor fc1_w, fc1_b, fc2_w, fc2_b;
    int    shift = 0;    // 0 or WIN/2
};
struct Layer {
    std::vector<Block> blocks;
    bool   down = false;
    Tensor dn_norm_w, dn_norm_b, dn_red;  // patch-merge: LN(4C) + reduction(2C,4C)
    Tensor norm_w, norm_b;                // per-stage output LN
    int    dim = 0, heads = 0;
};
struct Swin {
    Tensor pe_w, pe_b, pe_nw, pe_nb;      // patch embed conv + token LN
    std::vector<Layer> layers;
};

// ── helpers ─────────────────────────────────────────────────────────────────

// Reinterpret a contiguous (R, C) tensor as (R/g, C*g) (or any equal-size
// shape) — pure metadata, shares storage. Used to fold gathered quadruples
// into wider rows (patch merge) without a copy.
Tensor reshape_view(const Tensor& t, int rows, int cols) {
    return Tensor::view(t.device, t.data, rows, cols, t.dtype);
}

// A non-owning (rows, cols) view starting `row_off` rows into `t`.
Tensor row_view(const Tensor& t, int row_off, int rows) {
    char* p = static_cast<char*>(t.data) +
              static_cast<size_t>(row_off) * t.cols * bt::dtype_size_bytes(t.dtype);
    return Tensor::view(t.device, p, rows, t.cols, t.dtype);
}

Tensor int32_index(const std::vector<int32_t>& idx, Device dev) {
    Tensor h = Tensor::zeros_on(Device::CPU, static_cast<int>(idx.size()), 1, Dtype::INT32);
    auto* p = static_cast<int32_t*>(h.host_raw_mut());
    for (size_t i = 0; i < idx.size(); ++i) p[i] = idx[i];
    return dev == Device::CPU ? h : h.to(dev);
}

// Read an I64 safetensors view into a host int vector (relative_position_index).
std::vector<int64_t> read_i64(const st::TensorView& v) {
    std::vector<int64_t> out(v.numel());
    const auto* src = reinterpret_cast<const int64_t*>(v.data);
    for (int64_t i = 0; i < v.numel(); ++i) out[i] = src[i];
    return out;
}

}  // namespace

// ─── Impl ───────────────────────────────────────────────────────────────────

struct BiRefNet::Impl {
    Swin bb;
    Device dev = Device::CPU;
    bool loaded = false;

    // ── loading ──
    void load(const std::string& path);
    void to(Device d);

    // ── Swin backbone ──
    // Run the raw Swin-L backbone on an NCHW FP32 input; returns the 4 stage
    // features as NCHW tensors (on `dev`).
    std::vector<Tensor> backbone(const Tensor& xNCHW, int H, int W) const;

    // One Swin block (token-major in/out, residual applied).
    Tensor runBlock(const Block& blk, const Tensor& x, int H, int W,
                    int heads) const;
};

// ── load ─────────────────────────────────────────────────────────────────────

void BiRefNet::Impl::load(const std::string& path) {
    st::File f = st::File::open(path);
    const std::string who = "birefnet::load: ";

    auto W2 = [&](const std::string& name, int rows, int cols) {
        const auto& v = need_view(f, who, name, static_cast<int64_t>(rows) * cols);
        return load_f32(v, 0, rows, cols, who, name);
    };
    auto V1 = [&](const std::string& name, int n) {
        const auto& v = need_view(f, who, name, n);
        return load_f32(v, 0, n, 1, who, name);
    };

    // patch embed: conv (192, 3,4,4) flattened to (192, 48); token LN(192).
    bb.pe_w  = W2("bb.patch_embed.proj.weight", 192, 3 * 4 * 4);
    bb.pe_b  = V1("bb.patch_embed.proj.bias", 192);
    bb.pe_nw = V1("bb.patch_embed.norm.weight", 192);
    bb.pe_nb = V1("bb.patch_embed.norm.bias", 192);

    bb.layers.resize(4);
    for (int s = 0; s < 4; ++s) {
        Layer& ly = bb.layers[s];
        const int C = DIMS[s];
        const int H = HEADS[s];
        const int Nwin = WIN * WIN;
        ly.dim = C; ly.heads = H;
        ly.blocks.resize(DEPTHS[s]);
        for (int b = 0; b < DEPTHS[s]; ++b) {
            Block& blk = ly.blocks[b];
            const std::string bp =
                "bb.layers." + std::to_string(s) + ".blocks." + std::to_string(b) + ".";
            blk.shift = (b % 2 == 0) ? 0 : WIN / 2;
            blk.ln1_w = V1(bp + "norm1.weight", C);
            blk.ln1_b = V1(bp + "norm1.bias", C);
            blk.ln2_w = V1(bp + "norm2.weight", C);
            blk.ln2_b = V1(bp + "norm2.bias", C);
            // qkv (3C, C) -> Wq/Wk/Wv (C,C); qkv bias (3C) -> bq/bk/bv (C).
            const auto& qkvw = need_view(f, who, bp + "attn.qkv.weight",
                                         static_cast<int64_t>(3) * C * C);
            blk.attn.Wq = load_f32(qkvw, 0,                  C, C, who, bp + "qkv.q");
            blk.attn.Wk = load_f32(qkvw, static_cast<size_t>(C) * C, C, C, who, bp + "qkv.k");
            blk.attn.Wv = load_f32(qkvw, static_cast<size_t>(2) * C * C, C, C, who, bp + "qkv.v");
            const auto& qkvb = need_view(f, who, bp + "attn.qkv.bias", 3 * C);
            blk.attn.bq = load_f32(qkvb, 0,     C, 1, who, bp + "qkvb.q");
            blk.attn.bk = load_f32(qkvb, C,     C, 1, who, bp + "qkvb.k");
            blk.attn.bv = load_f32(qkvb, 2 * C, C, 1, who, bp + "qkvb.v");
            blk.attn.Wo = W2(bp + "attn.proj.weight", C, C);
            blk.attn.bo = V1(bp + "attn.proj.bias", C);
            blk.attn.heads = H; blk.attn.N = Nwin;

            // Relative-position bias: table ((2w-1)^2, H), index (N, N) i64.
            const int tab_rows = (2 * WIN - 1) * (2 * WIN - 1);
            Tensor table = W2(bp + "attn.relative_position_bias_table", tab_rows, H);
            const auto& iv = need_view(f, who, bp + "attn.relative_position_index",
                                       static_cast<int64_t>(Nwin) * Nwin);
            std::vector<int64_t> idx = read_i64(iv);
            // rpb (H*N, N): rpb[h*N+i, j] = table[idx[i*N+j], h].
            Tensor rpb = Tensor::mat(H * Nwin, Nwin);
            float* rp = rpb.host_f32_mut();
            const float* tp = table.host_f32();
            for (int h = 0; h < H; ++h)
                for (int i = 0; i < Nwin; ++i)
                    for (int j = 0; j < Nwin; ++j)
                        rp[(static_cast<size_t>(h) * Nwin + i) * Nwin + j] =
                            tp[static_cast<size_t>(idx[i * Nwin + j]) * H + h];
            blk.attn.rpb = std::move(rpb);

            blk.fc1_w = W2(bp + "mlp.fc1.weight", 4 * C, C);
            blk.fc1_b = V1(bp + "mlp.fc1.bias", 4 * C);
            blk.fc2_w = W2(bp + "mlp.fc2.weight", C, 4 * C);
            blk.fc2_b = V1(bp + "mlp.fc2.bias", C);
        }
        ly.down = (s < 3);
        if (ly.down) {
            const std::string dp = "bb.layers." + std::to_string(s) + ".downsample.";
            ly.dn_norm_w = V1(dp + "norm.weight", 4 * C);
            ly.dn_norm_b = V1(dp + "norm.bias", 4 * C);
            ly.dn_red    = W2(dp + "reduction.weight", 2 * C, 4 * C);
        }
        ly.norm_w = V1("bb.norm" + std::to_string(s) + ".weight", C);
        ly.norm_b = V1("bb.norm" + std::to_string(s) + ".bias", C);
    }
    loaded = true;
}

void BiRefNet::Impl::to(Device d) {
    auto mv = [d](Tensor& t) { if (t.data) t = t.to(d); };
    mv(bb.pe_w); mv(bb.pe_b); mv(bb.pe_nw); mv(bb.pe_nb);
    for (auto& ly : bb.layers) {
        for (auto& blk : ly.blocks) {
            mv(blk.ln1_w); mv(blk.ln1_b); mv(blk.ln2_w); mv(blk.ln2_b);
            mv(blk.attn.Wq); mv(blk.attn.Wk); mv(blk.attn.Wv);
            mv(blk.attn.bq); mv(blk.attn.bk); mv(blk.attn.bv);
            mv(blk.attn.Wo); mv(blk.attn.bo); mv(blk.attn.rpb);
            mv(blk.fc1_w); mv(blk.fc1_b); mv(blk.fc2_w); mv(blk.fc2_b);
        }
        mv(ly.dn_norm_w); mv(ly.dn_norm_b); mv(ly.dn_red);
        mv(ly.norm_w); mv(ly.norm_b);
    }
    dev = d;
}

// ── Swin window permutation precompute (host, input-independent per H,W) ─────

namespace {

struct WinPlan {
    int Hp, Wp, nWh, nWw, nW, N;
    std::vector<int32_t> gather;   // nW*N -> source token in [0,H*W] (H*W = pad row)
    std::vector<int32_t> scatter;  // H*W  -> window-row in [0,nW*N)
    std::vector<int32_t> region;   // nW*N -> shift region label (0..8), shift>0 only
};

WinPlan make_plan(int H, int W, int shift) {
    WinPlan p;
    p.N = WIN * WIN;
    p.Hp = ((H + WIN - 1) / WIN) * WIN;
    p.Wp = ((W + WIN - 1) / WIN) * WIN;
    p.nWh = p.Hp / WIN; p.nWw = p.Wp / WIN; p.nW = p.nWh * p.nWw;
    p.gather.assign(static_cast<size_t>(p.nW) * p.N, H * W);    // default = pad row
    p.scatter.assign(static_cast<size_t>(H) * W, 0);
    p.region.assign(static_cast<size_t>(p.nW) * p.N, 0);

    auto h_region = [&](int a) {
        if (a < p.Hp - WIN) return 0;
        if (a < p.Hp - shift) return 1;
        return 2;
    };
    auto w_region = [&](int b) {
        if (b < p.Wp - WIN) return 0;
        if (b < p.Wp - shift) return 1;
        return 2;
    };
    for (int wh = 0; wh < p.nWh; ++wh)
        for (int ww = 0; ww < p.nWw; ++ww) {
            const int wi = wh * p.nWw + ww;
            for (int lh = 0; lh < WIN; ++lh)
                for (int lw = 0; lw < WIN; ++lw) {
                    const int a = wh * WIN + lh;       // rolled-grid row
                    const int b = ww * WIN + lw;       // rolled-grid col
                    const size_t wr = static_cast<size_t>(wi) * p.N + lh * WIN + lw;
                    if (shift > 0)
                        p.region[wr] = h_region(a) * 3 + w_region(b);
                    // rolled[a,b] = padded[(a+shift)%Hp, (b+shift)%Wp]
                    const int hh = (a + shift) % p.Hp;
                    const int wpos = (b + shift) % p.Wp;
                    if (hh < H && wpos < W) {
                        const int src = hh * W + wpos;
                        p.gather[wr] = src;
                        p.scatter[src] = static_cast<int32_t>(wr);
                    }
                }
        }
    return p;
}

// 2x2 patch-merge gather plan: out (H2*W2*4) rows pulling the 4 quadrants of
// each 2x2 cell (cat order x0[0,0] x1[1,0] x2[0,1] x3[1,1]); odd H/W pad to a
// zero row (index H*W).
struct MergePlan { int H2, W2; std::vector<int32_t> gather; };
MergePlan make_merge_plan(int H, int W) {
    MergePlan p;
    const int Hp = H + (H & 1), Wp = W + (W & 1);
    p.H2 = Hp / 2; p.W2 = Wp / 2;
    p.gather.assign(static_cast<size_t>(p.H2) * p.W2 * 4, H * W);
    const int dh[4] = {0, 1, 0, 1}, dw[4] = {0, 0, 1, 1};
    for (int h2 = 0; h2 < p.H2; ++h2)
        for (int w2 = 0; w2 < p.W2; ++w2)
            for (int q = 0; q < 4; ++q) {
                const int hh = 2 * h2 + dh[q], ww = 2 * w2 + dw[q];
                if (hh < H && ww < W)
                    p.gather[(static_cast<size_t>(h2) * p.W2 + w2) * 4 + q] = hh * W + ww;
            }
    return p;
}

// Append a zero row to (L,C) -> (L+1,C) view; `store` owns the buffer.
Tensor with_pad_row(const Tensor& x, int L, int C, Device dev, Tensor& store) {
    Tensor zrow = Tensor::zeros_on(dev, 1, C, x.dtype);
    bt::concat_rows({&x, &zrow}, store);          // flat (L*C + C, 1)
    return reshape_view(store, L + 1, C);
}

}  // namespace

// One Swin block, in-place residual update of token tensor x (L=H*W, C).
Tensor BiRefNet::Impl::runBlock(const Block& blk, const Tensor& xin,
                                int H, int W, int heads) const {
    const int L = H * W, C = xin.cols, N = WIN * WIN;
    const float scale = 1.0f / std::sqrt(static_cast<float>(C / heads));

    Tensor xn; bt::layernorm_forward_inference_batched(xin, blk.ln1_w, blk.ln1_b, xn, LN_EPS);

    WinPlan plan = make_plan(H, W, blk.shift);
    Tensor xpad_store;
    Tensor xpad = with_pad_row(xn, L, C, dev, xpad_store);
    Tensor windows; bt::gather_rows(xpad, int32_index(plan.gather, dev), windows);

    Tensor attn_out = Tensor::zeros_on(dev, plan.nW * N, C, xn.dtype);
    Tensor rpb_host = (blk.attn.rpb.device == Device::CPU)
                          ? blk.attn.rpb : blk.attn.rpb.to(Device::CPU);
    for (int w = 0; w < plan.nW; ++w) {
        Tensor win = row_view(windows, w * N, N);
        Tensor out = row_view(attn_out, w * N, N);
        const Tensor* abias = &blk.attn.rpb;
        Tensor wb;
        if (blk.shift > 0) {
            const int32_t* reg = &plan.region[static_cast<size_t>(w) * N];
            bool mixed = false;
            for (int i = 1; i < N && !mixed; ++i) if (reg[i] != reg[0]) mixed = true;
            if (mixed) {
                wb = Tensor::mat(heads * N, N);
                float* wp = wb.host_f32_mut();
                const float* rph = rpb_host.host_f32();
                for (int h = 0; h < heads; ++h)
                    for (int i = 0; i < N; ++i)
                        for (int j = 0; j < N; ++j)
                            wp[(static_cast<size_t>(h) * N + i) * N + j] =
                                rph[(static_cast<size_t>(h) * N + i) * N + j] +
                                ((reg[i] != reg[j]) ? -100.0f : 0.0f);
                if (dev != Device::CPU) wb = wb.to(dev);
                abias = &wb;
            }
        }
        bt::self_attention_bias_forward(win, blk.attn.Wq, blk.attn.Wk, blk.attn.Wv,
                                        blk.attn.Wo, &blk.attn.bq, &blk.attn.bk,
                                        &blk.attn.bv, &blk.attn.bo,
                                        nullptr, abias, heads, scale, out);
    }
    Tensor x_attn; bt::gather_rows(attn_out, int32_index(plan.scatter, dev), x_attn);

    Tensor x = xin.to(dev);                 // deep copy for the residual
    bt::add_inplace(x, x_attn);             // x = xin + attn

    Tensor xn2; bt::layernorm_forward_inference_batched(x, blk.ln2_w, blk.ln2_b, xn2, LN_EPS);
    Tensor f1; bt::linear_forward_batched(blk.fc1_w, blk.fc1_b, xn2, f1);
    Tensor act; bt::gelu_exact_forward(f1, act);
    Tensor f2; bt::linear_forward_batched(blk.fc2_w, blk.fc2_b, act, f2);
    bt::add_inplace(x, f2);
    return x;
}

std::vector<Tensor> BiRefNet::Impl::backbone(const Tensor& xNCHW, int H, int W) const {
    // Patch embed: conv 4x4 s4 -> (1, 192*Wh*Ww) NCHW -> tokens -> LN(192).
    Tensor pe; bt::conv2d_forward(xNCHW, bb.pe_w, &bb.pe_b, 1, 3, H, W,
                                  192, 4, 4, 4, 4, 0, 0, 1, 1, pe);
    int Wh = H / 4, Ww = W / 4;
    Tensor tok; bt::nchw_to_sequence(pe, 1, 192, Wh, Ww, tok);   // (Wh*Ww, 192)
    Tensor x; bt::layernorm_forward_inference_batched(tok, bb.pe_nw, bb.pe_nb, x, LN_EPS);

    std::vector<Tensor> outs;
    int curH = Wh, curW = Ww;
    for (int s = 0; s < 4; ++s) {
        const Layer& ly = bb.layers[s];
        // Run the stage; capture the pre-downsample feature by re-running blocks
        // here so we hold both the feature and the downsample. runLayer returns
        // the downsampled tokens; to get the feature we re-derive it.
        Tensor feat = x.to(dev);
        for (const Block& blk : ly.blocks)
            feat = runBlock(blk, feat, curH, curW, ly.heads);
        Tensor xdown = feat;
        if (ly.down) {
            const int L = curH * curW, C = ly.dim;
            MergePlan mp = make_merge_plan(curH, curW);
            Tensor xpad_store;
            Tensor xpad = with_pad_row(feat, L, C, dev, xpad_store);
            Tensor quads; bt::gather_rows(xpad, int32_index(mp.gather, dev), quads);
            Tensor merged = reshape_view(quads, mp.H2 * mp.W2, 4 * C);
            Tensor mn; bt::layernorm_forward_inference_batched(merged, ly.dn_norm_w, ly.dn_norm_b, mn, LN_EPS);
            Tensor zb = Tensor::zeros_on(dev, 2 * C, 1, mn.dtype);  // reduction has no bias
            bt::linear_forward_batched(ly.dn_red, zb, mn, xdown);
        }
        // Stage output feature: LN_s(feat) -> NCHW (C, curH, curW).
        Tensor fn; bt::layernorm_forward_inference_batched(feat, ly.norm_w, ly.norm_b, fn, LN_EPS);
        Tensor nchw; bt::sequence_to_nchw(fn, 1, ly.dim, curH, curW, nchw);
        outs.push_back(nchw);
        x = xdown;
        if (ly.down) { curH = (curH + 1) / 2; curW = (curW + 1) / 2; }
    }
    return outs;
}

// ─── public BiRefNet ─────────────────────────────────────────────────────────

BiRefNet::BiRefNet() : impl_(std::make_unique<Impl>()) {}
BiRefNet::~BiRefNet() = default;
BiRefNet::BiRefNet(BiRefNet&&) noexcept = default;
BiRefNet& BiRefNet::operator=(BiRefNet&&) noexcept = default;

void BiRefNet::load(const std::string& path) { impl_->load(path); }
void BiRefNet::to(brotensor::Device dev) { impl_->to(dev); }
brotensor::Device BiRefNet::device() const { return impl_->dev; }

std::vector<Tensor> BiRefNet::debugBackbone(const Tensor& xNCHW, int H, int W) const {
    if (!impl_->loaded) throw std::runtime_error("birefnet: debugBackbone before load()");
    return impl_->backbone(xNCHW.to(impl_->dev), H, W);
}

Tensor BiRefNet::forwardLogits(const Tensor&, int, int) const {
    throw std::runtime_error("birefnet: forwardLogits (decoder) not yet wired");
}
Matte BiRefNet::removeBackground(const float*, int, int, bool, int) const {
    throw std::runtime_error("birefnet: removeBackground not yet wired");
}

}  // namespace brovisionml::birefnet
