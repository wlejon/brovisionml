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

// ── ASPP-deformable decoder weight containers ───────────────────────────────

struct BN { Tensor w, b, m, v; };         // batch-norm (gamma/beta/mean/var)
struct DeformConv {                       // _DeformableConv2d (modulated DCNv2)
    Tensor off_w, off_b, mod_w, mod_b, reg_w;   // reg_w has no bias
    int k = 0, pad = 0;
};
struct ASPPMod { DeformConv dc; BN bn; }; // _ASPPModuleDeformable (+ relu)
struct ASPP {                             // _ASPPDeformable (in=out=64, inter=256)
    ASPPMod aspp1;                        // k=1
    std::vector<ASPPMod> deforms;         // k=1,3,7
    Tensor gap_conv_w;                    // global pool: conv 64->256 (no bias)
    BN gap_bn;
    Tensor conv1_w;                       // 1280->64 (no bias)
    BN bn1;
};
struct DecBlk {                           // _BasicDecBlk
    Tensor cin_w, cin_b; BN bin;          // conv_in (->64) + bn_in + relu
    ASPP att;
    Tensor cout_w, cout_b; BN bout;       // conv_out (64->out) + bn_out
    int in_ch = 0, out_ch = 0;
};
struct SimpleConvs { Tensor c1_w, c1_b, co_w, co_b; };  // ipt blocks
struct GdtBranch {                        // gdt_convs_N (conv->16 + bn + relu) + attn 16->1
    Tensor c_w, c_b; BN bn; Tensor attn_w, attn_b;
};
struct LatBlk { Tensor w, b; };           // lateral 1x1 conv

struct Decoder {
    DecBlk squeeze;                       // squeeze_module.0  (5760->3072)
    SimpleConvs ipt[5];                   // [0]=ipt_blk5 .. [4]=ipt_blk1
    DecBlk block[4];                      // [0]=decoder_block4 .. [3]=decoder_block1
    LatBlk lat[3];                        // [0]=lateral_block4,3,2
    GdtBranch gdt[3];                     // [0]=gdt_4,3,2
    Tensor convout1_w, convout1_b;        // 240->1
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
    Decoder dec;
    Device dev = Device::CPU;
    bool loaded = false;

    // ── loading ──
    void load(const std::string& path);
    void to(Device d);

    // ── decoder / top-level ──
    // Dual-resolution backbone + mul_scl concat (returns x1,x2,x3 and the
    // squeeze input x4).
    void forwardEnc(const Tensor& img, int H, int W,
                    Tensor& x1, Tensor& x2, Tensor& x3, Tensor& x4sq) const;
    Tensor forwardLogits(const Tensor& img, int H, int W) const;

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

    // ── decoder ──
    auto conv_w = [&](const std::string& n, int oc, int ic, int k) {
        return W2(n, oc, ic * k * k);
    };
    auto load_bn = [&](const std::string& p, int C) {
        BN bn; bn.w = V1(p + ".weight", C); bn.b = V1(p + ".bias", C);
        bn.m = V1(p + ".running_mean", C); bn.v = V1(p + ".running_var", C);
        return bn;
    };
    auto load_deform = [&](const std::string& p, int ic, int oc, int k) {
        DeformConv dc; dc.k = k; dc.pad = k / 2;
        dc.off_w = conv_w(p + ".offset_conv.weight", 2 * k * k, ic, k);
        dc.off_b = V1(p + ".offset_conv.bias", 2 * k * k);
        dc.mod_w = conv_w(p + ".modulator_conv.weight", k * k, ic, k);
        dc.mod_b = V1(p + ".modulator_conv.bias", k * k);
        dc.reg_w = conv_w(p + ".regular_conv.weight", oc, ic, k);
        return dc;
    };
    auto load_asppmod = [&](const std::string& p, int ic, int k) {
        ASPPMod m; m.dc = load_deform(p + ".atrous_conv", ic, 256, k);
        m.bn = load_bn(p + ".bn", 256); return m;
    };
    auto load_aspp = [&](const std::string& p) {   // in=out=64, inter=256
        ASPP a;
        a.aspp1 = load_asppmod(p + ".aspp1", 64, 1);
        const int ks[3] = {1, 3, 7};
        for (int i = 0; i < 3; ++i)
            a.deforms.push_back(load_asppmod(p + ".aspp_deforms." + std::to_string(i), 64, ks[i]));
        a.gap_conv_w = conv_w(p + ".global_avg_pool.1.weight", 256, 64, 1);
        a.gap_bn = load_bn(p + ".global_avg_pool.2", 256);
        a.conv1_w = conv_w(p + ".conv1.weight", 64, 256 * 5, 1);
        a.bn1 = load_bn(p + ".bn1", 64);
        return a;
    };
    auto load_decblk = [&](const std::string& p, int ic, int oc) {
        DecBlk d; d.in_ch = ic; d.out_ch = oc;
        d.cin_w = conv_w(p + ".conv_in.weight", 64, ic, 3);
        d.cin_b = V1(p + ".conv_in.bias", 64);
        d.bin = load_bn(p + ".bn_in", 64);
        d.att = load_aspp(p + ".dec_att");
        d.cout_w = conv_w(p + ".conv_out.weight", oc, 64, 3);
        d.cout_b = V1(p + ".conv_out.bias", oc);
        d.bout = load_bn(p + ".bn_out", oc);
        return d;
    };
    auto load_simple = [&](const std::string& p, int ic, int oc) {
        SimpleConvs s;
        s.c1_w = conv_w(p + ".conv1.weight", 64, ic, 3); s.c1_b = V1(p + ".conv1.bias", 64);
        s.co_w = conv_w(p + ".conv_out.weight", oc, 64, 3); s.co_b = V1(p + ".conv_out.bias", oc);
        return s;
    };
    auto load_gdt = [&](const std::string& p, int ic) {
        GdtBranch g;
        g.c_w = conv_w(p + ".0.weight", 16, ic, 3); g.c_b = V1(p + ".0.bias", 16);
        g.bn = load_bn(p + ".1", 16);
        return g;
    };

    dec.squeeze = load_decblk("squeeze_module.0", 5760, 3072);
    const int IPT_IN[5]  = {3072, 768, 192, 48, 3};   // ipt_blk5..1
    const int IPT_OUT[5] = {384, 384, 192, 96, 48};
    for (int i = 0; i < 5; ++i)
        dec.ipt[i] = load_simple("decoder.ipt_blk" + std::to_string(5 - i), IPT_IN[i], IPT_OUT[i]);
    const int BIN[4]  = {3456, 1920, 960, 480};       // decoder_block4..1 in
    const int BOUT[4] = {1536, 768, 384, 192};        // out
    for (int i = 0; i < 4; ++i)
        dec.block[i] = load_decblk("decoder.decoder_block" + std::to_string(4 - i), BIN[i], BOUT[i]);
    const int LAT[3] = {1536, 768, 384};              // lateral_block4,3,2
    for (int i = 0; i < 3; ++i) {
        const std::string lp = "decoder.lateral_block" + std::to_string(4 - i) + ".conv";
        dec.lat[i].w = conv_w(lp + ".weight", LAT[i], LAT[i], 1);
        dec.lat[i].b = V1(lp + ".bias", LAT[i]);
    }
    const int GDT[3] = {1536, 768, 384};              // gdt_4,3,2 in
    for (int i = 0; i < 3; ++i) {
        dec.gdt[i] = load_gdt("decoder.gdt_convs_" + std::to_string(4 - i), GDT[i]);
        dec.gdt[i].attn_w = conv_w("decoder.gdt_convs_attn_" + std::to_string(4 - i) + ".0.weight", 1, 16, 1);
        dec.gdt[i].attn_b = V1("decoder.gdt_convs_attn_" + std::to_string(4 - i) + ".0.bias", 1);
    }
    dec.convout1_w = conv_w("decoder.conv_out1.0.weight", 1, 240, 1);
    dec.convout1_b = V1("decoder.conv_out1.0.bias", 1);

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

    auto mvbn = [&](BN& b) { mv(b.w); mv(b.b); mv(b.m); mv(b.v); };
    auto mvdc = [&](DeformConv& dc) {
        mv(dc.off_w); mv(dc.off_b); mv(dc.mod_w); mv(dc.mod_b); mv(dc.reg_w);
    };
    auto mvaspp = [&](ASPP& a) {
        mvdc(a.aspp1.dc); mvbn(a.aspp1.bn);
        for (auto& m : a.deforms) { mvdc(m.dc); mvbn(m.bn); }
        mv(a.gap_conv_w); mvbn(a.gap_bn); mv(a.conv1_w); mvbn(a.bn1);
    };
    auto mvdecblk = [&](DecBlk& db) {
        mv(db.cin_w); mv(db.cin_b); mvbn(db.bin);
        mvaspp(db.att);
        mv(db.cout_w); mv(db.cout_b); mvbn(db.bout);
    };
    mvdecblk(dec.squeeze);
    for (auto& s : dec.ipt) { mv(s.c1_w); mv(s.c1_b); mv(s.co_w); mv(s.co_b); }
    for (auto& b : dec.block) mvdecblk(b);
    for (auto& l : dec.lat) { mv(l.w); mv(l.b); }
    for (auto& g : dec.gdt) { mv(g.c_w); mv(g.c_b); mvbn(g.bn); mv(g.attn_w); mv(g.attn_b); }
    mv(dec.convout1_w); mv(dec.convout1_b);
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

// ── decoder forward helpers ─────────────────────────────────────────────────

namespace {

Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor* b,
              int Cin, int Cout, int H, int W, int k, int pad) {
    Tensor y;
    bt::conv2d_forward(x, w, b, 1, Cin, H, W, Cout, k, k, 1, 1, pad, pad, 1, 1, y);
    return y;
}
Tensor bnorm(const Tensor& x, const BN& bn, int C, int H, int W) {
    Tensor y; bt::batch_norm_inference(x, bn.w, bn.b, bn.m, bn.v, 1, C, H, W, 1e-5f, y);
    return y;
}
Tensor relu_(const Tensor& x) { Tensor y; bt::relu_forward(x, y); return y; }
Tensor interpAC(const Tensor& x, int C, int Hin, int Win, int Hout, int Wout) {
    if (Hin == Hout && Win == Wout) return x;
    Tensor y; bt::interp2d_align_corners_forward(x, 1, C, Hin, Win, Hout, Wout, 1, y);
    return y;
}

// Modulated deformable conv (offset + 2*sigmoid modulator) -> (256, H, W).
Tensor deformConv(const Tensor& x, const DeformConv& dc, int Cin, int H, int W) {
    Tensor off = conv2d(x, dc.off_w, &dc.off_b, Cin, 2 * dc.k * dc.k, H, W, dc.k, dc.pad);
    Tensor modr = conv2d(x, dc.mod_w, &dc.mod_b, Cin, dc.k * dc.k, H, W, dc.k, dc.pad);
    Tensor mod; bt::sigmoid_forward(modr, mod); bt::scale_inplace(mod, 2.0f);
    Tensor y;
    bt::deform_conv2d_forward(x, off, &mod, dc.reg_w, /*bias=*/nullptr,
                              1, Cin, H, W, 256, dc.k, dc.k, 1, 1,
                              dc.pad, dc.pad, 1, 1, /*groups=*/1, /*deform_groups=*/1, y);
    return y;
}
Tensor asppMod(const Tensor& x, const ASPPMod& m, int Cin, int H, int W) {
    return relu_(bnorm(deformConv(x, m.dc, Cin, H, W), m.bn, 256, H, W));
}
// ASPP-deformable: in=out=64.
Tensor aspp(const Tensor& x, const ASPP& a, int H, int W) {
    Tensor x1 = asppMod(x, a.aspp1, 64, H, W);
    Tensor d0 = asppMod(x, a.deforms[0], 64, H, W);
    Tensor d1 = asppMod(x, a.deforms[1], 64, H, W);
    Tensor d2 = asppMod(x, a.deforms[2], 64, H, W);
    Tensor gp; bt::adaptive_avg_pool2d_forward(x, 1, 64, H, W, 1, 1, gp);  // (64,1,1)
    gp = relu_(bnorm(conv2d(gp, a.gap_conv_w, nullptr, 64, 256, 1, 1, 1, 0), a.gap_bn, 256, 1, 1));
    gp = interpAC(gp, 256, 1, 1, H, W);
    Tensor cat;
    bt::concat_nchw_channels({&x1, &d0, &d1, &d2, &gp}, 1, H, W,
                             {256, 256, 256, 256, 256}, cat);  // (1280,H,W)
    return relu_(bnorm(conv2d(cat, a.conv1_w, nullptr, 1280, 64, H, W, 1, 0), a.bn1, 64, H, W));
}
// _BasicDecBlk: conv_in+bn+relu -> aspp -> conv_out+bn (no final relu).
Tensor basicDecBlk(const Tensor& x, const DecBlk& d, int H, int W) {
    Tensor y = relu_(bnorm(conv2d(x, d.cin_w, &d.cin_b, d.in_ch, 64, H, W, 3, 1), d.bin, 64, H, W));
    y = aspp(y, d.att, H, W);
    return bnorm(conv2d(y, d.cout_w, &d.cout_b, 64, d.out_ch, H, W, 3, 1), d.bout, d.out_ch, H, W);
}
Tensor simpleConvs(const Tensor& x, const SimpleConvs& s, int in, int out, int H, int W) {
    Tensor y = conv2d(x, s.c1_w, &s.c1_b, in, 64, H, W, 3, 1);
    return conv2d(y, s.co_w, &s.co_b, 64, out, H, W, 3, 1);
}

// einops 'b c (hg h) (wg w) -> b (c hg wg) h w' on the input image.
Tensor image2patches(const Tensor& img, int Hin, int Win, int refH, int refW, Device dev) {
    const int hg = Hin / refH, wg = Win / refW;
    const int h = Hin / hg, w = Win / wg;          // == refH, refW
    const int Cout = 3 * hg * wg;
    std::vector<int32_t> idx(static_cast<size_t>(Cout) * h * w);
    size_t k = 0;
    for (int c = 0; c < 3; ++c)
        for (int ph = 0; ph < hg; ++ph)
            for (int pw = 0; pw < wg; ++pw)
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        idx[k++] = (c * Hin + (ph * h + y)) * Win + (pw * w + x);
    Tensor img_col = reshape_view(img, 3 * Hin * Win, 1);
    Tensor g; bt::gather_rows(img_col, int32_index(idx, dev), g);   // (M,1) owning
    return g;   // flat NCHW (1, Cout*h*w); downstream ops carry explicit dims
}

// p = p * sigmoid(attn(relu(bn(conv(p))))), attn broadcast across channels.
Tensor gdtGate(const Tensor& p, const GdtBranch& g, int C, int H, int W) {
    Tensor t = relu_(bnorm(conv2d(p, g.c_w, &g.c_b, C, 16, H, W, 3, 1), g.bn, 16, H, W));
    Tensor a = conv2d(t, g.attn_w, &g.attn_b, 16, 1, H, W, 1, 0);  // (1, H*W)
    Tensor s; bt::sigmoid_forward(a, s);
    Tensor pflat = reshape_view(p, C, H * W);                      // (C, H*W) view
    Tensor gated; bt::broadcast_mul(pflat, s, gated);              // (C,H*W) owning
    return gated;   // flat NCHW (1, C*H*W); downstream ops carry explicit dims
}

}  // namespace

void BiRefNet::Impl::forwardEnc(const Tensor& img, int H, int W,
                                Tensor& x1, Tensor& x2, Tensor& x3, Tensor& x4sq) const {
    std::vector<Tensor> f = backbone(img, H, W);                       // 192,384,768,1536
    Tensor imgHalf = interpAC(img, 3, H, W, H / 2, W / 2);
    std::vector<Tensor> fh = backbone(imgHalf, H / 2, W / 2);
    const int CH[4] = {192, 384, 768, 1536};
    Tensor xs[4];
    for (int i = 0; i < 4; ++i) {
        const int fH = (H / 4) >> i, fW = (W / 4) >> i;     // full-res stage grid
        const int hH = (H / 8) >> i, hW = (W / 8) >> i;     // half-res stage grid
        Tensor up = interpAC(fh[i], CH[i], hH, hW, fH, fW);
        bt::concat_nchw_channels({&f[i], &up}, 1, fH, fW, {CH[i], CH[i]}, xs[i]);
    }
    x1 = xs[0]; x2 = xs[1]; x3 = xs[2];                     // 384,768,1536
    const int x4H = H / 32, x4W = W / 32;
    Tensor u1 = interpAC(x1, 384,  H / 4,  W / 4,  x4H, x4W);
    Tensor u2 = interpAC(x2, 768,  H / 8,  W / 8,  x4H, x4W);
    Tensor u3 = interpAC(x3, 1536, H / 16, W / 16, x4H, x4W);
    Tensor x4cat;
    bt::concat_nchw_channels({&u1, &u2, &u3, &xs[3]}, 1, x4H, x4W,
                             {384, 768, 1536, 3072}, x4cat);   // 5760
    x4sq = basicDecBlk(x4cat, dec.squeeze, x4H, x4W);          // 3072
}

Tensor BiRefNet::Impl::forwardLogits(const Tensor& img, int H, int W) const {
    Tensor x1, x2, x3, x4;
    forwardEnc(img, H, W, x1, x2, x3, x4);   // 384@H/4, 768@H/8, 1536@H/16, 3072@H/32
    const int x4H = H / 32, x4W = W / 32, x3H = H / 16, x3W = W / 16;
    const int x2H = H / 8, x2W = W / 8, x1H = H / 4, x1W = W / 4;

    auto inject = [&](const Tensor& p, int pc, int pH, int pW, int ipt_i,
                      int ipt_out) {
        Tensor ip = image2patches(img, H, W, pH, pW, dev);
        ip = simpleConvs(ip, dec.ipt[ipt_i], 3 * (H / pH) * (W / pW), ipt_out, pH, pW);
        Tensor out; bt::concat_nchw_channels({&p, &ip}, 1, pH, pW, {pc, ipt_out}, out);
        return out;
    };

    Tensor x4i = inject(x4, 3072, x4H, x4W, 0, 384);          // 3456
    Tensor p4 = basicDecBlk(x4i, dec.block[0], x4H, x4W);     // 1536
    p4 = gdtGate(p4, dec.gdt[0], 1536, x4H, x4W);
    Tensor _p4 = interpAC(p4, 1536, x4H, x4W, x3H, x3W);
    Tensor lat4 = conv2d(x3, dec.lat[0].w, &dec.lat[0].b, 1536, 1536, x3H, x3W, 1, 0);
    bt::add_inplace(_p4, lat4);
    Tensor _p3i = inject(_p4, 1536, x3H, x3W, 1, 384);        // 1920
    Tensor p3 = basicDecBlk(_p3i, dec.block[1], x3H, x3W);    // 768
    p3 = gdtGate(p3, dec.gdt[1], 768, x3H, x3W);
    Tensor _p3 = interpAC(p3, 768, x3H, x3W, x2H, x2W);
    Tensor lat3 = conv2d(x2, dec.lat[1].w, &dec.lat[1].b, 768, 768, x2H, x2W, 1, 0);
    bt::add_inplace(_p3, lat3);
    Tensor _p2i = inject(_p3, 768, x2H, x2W, 2, 192);         // 960
    Tensor p2 = basicDecBlk(_p2i, dec.block[2], x2H, x2W);    // 384
    p2 = gdtGate(p2, dec.gdt[2], 384, x2H, x2W);
    Tensor _p2 = interpAC(p2, 384, x2H, x2W, x1H, x1W);
    Tensor lat2 = conv2d(x1, dec.lat[2].w, &dec.lat[2].b, 384, 384, x1H, x1W, 1, 0);
    bt::add_inplace(_p2, lat2);
    Tensor _p1i = inject(_p2, 384, x1H, x1W, 3, 96);          // 480
    Tensor _p1 = basicDecBlk(_p1i, dec.block[3], x1H, x1W);   // 192
    _p1 = interpAC(_p1, 192, x1H, x1W, H, W);                 // full res
    Tensor _p1f = inject(_p1, 192, H, W, 4, 48);             // 240
    return conv2d(_p1f, dec.convout1_w, &dec.convout1_b, 240, 1, H, W, 1, 0);  // (1, H*W)
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

Tensor BiRefNet::forwardLogits(const Tensor& imgNCHW, int H, int W) const {
    if (!impl_->loaded) throw std::runtime_error("birefnet: forwardLogits before load()");
    return impl_->forwardLogits(imgNCHW.to(impl_->dev), H, W);
}

Matte BiRefNet::removeBackground(const float* rgb, int origW, int origH,
                                 bool rgbIs255, int modelSize) const {
    if (!impl_->loaded) throw std::runtime_error("birefnet: removeBackground before load()");
    // Interleaved RGB (origH*origW*3) -> NCHW (1, 3*origH*origW), [0,1].
    const float inv = rgbIs255 ? 1.0f / 255.0f : 1.0f;
    Tensor src = Tensor::mat(1, 3 * origH * origW);
    float* sp = src.host_f32_mut();
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < origH; ++y)
            for (int x = 0; x < origW; ++x)
                sp[(static_cast<size_t>(c) * origH + y) * origW + x] =
                    rgb[(static_cast<size_t>(y) * origW + x) * 3 + c] * inv;
    // Resize to modelSize² (align_corners) + ImageNet normalize.
    Tensor rs;
    bt::interp2d_align_corners_forward(src, 1, 3, origH, origW, modelSize, modelSize, 1, rs);
    const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
    float* rp = rs.host_f32_mut();
    const int sp2 = modelSize * modelSize;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < sp2; ++i)
            rp[c * sp2 + i] = (rp[c * sp2 + i] - mean[c]) / sd[c];

    Tensor logits = impl_->forwardLogits(rs.to(impl_->dev), modelSize, modelSize);
    Tensor lh = (logits.device == Device::CPU) ? logits : logits.to(Device::CPU);
    Tensor alpha; bt::sigmoid_forward(lh, alpha);
    Tensor back;
    bt::interp2d_align_corners_forward(alpha, 1, 1, modelSize, modelSize, origH, origW, 1, back);

    Matte mt; mt.width = origW; mt.height = origH;
    const float* bp = back.host_f32();
    mt.alpha.assign(bp, bp + static_cast<size_t>(origW) * origH);
    return mt;
}

}  // namespace brovisionml::birefnet
