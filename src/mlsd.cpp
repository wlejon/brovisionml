#include "brovisionml/mlsd.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include "weights_util.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brovisionml::mlsd {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "mlsd::MLSDdetector: ";
constexpr float kBnEps = 1e-5f;   // torch BatchNorm2d default

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

// A conv with BatchNorm already folded into (weight, bias). Layout matches
// brotensor: weight (c_out, (c_in/groups)*k*k); bias (c_out,1).
struct Conv {
    Tensor weight, bias;
    int c_in = 0, c_out = 0, k = 1, groups = 1;
};

// Load a conv and fold the following BatchNorm into it (if `bn_prefix`). For a
// conv→BN, with s = gamma/sqrt(var+eps): W'[o]=W[o]*s[o], b'[o]=(b[o]-mean[o])*
// s[o]+beta[o] (b=0 when the conv is bias-less). groups_arg==0 means depthwise
// (groups = c_out, one input channel per group).
Conv load_conv(const st::File& f, const std::string& conv_prefix,
               const std::string& bn_prefix, int groups_arg) {
    const st::TensorView* wv = f.find(conv_prefix + ".weight");
    if (!wv) fail("missing tensor '" + conv_prefix + ".weight'");
    if (wv->shape.size() != 4) fail("conv '" + conv_prefix + "' weight not 4D");
    const int c_out = static_cast<int>(wv->shape[0]);
    const int cin_g = static_cast<int>(wv->shape[1]);
    const int k     = static_cast<int>(wv->shape[2]);

    Conv c;
    c.c_out = c_out;
    c.k = k;
    if (groups_arg == 0) { c.groups = c_out; c.c_in = c_out; }  // depthwise
    else                 { c.groups = groups_arg; c.c_in = cin_g * groups_arg; }

    c.weight = load_whole(f, kWho, conv_prefix + ".weight", c_out, cin_g * k * k);
    if (f.find(conv_prefix + ".bias"))
        c.bias = load_whole(f, kWho, conv_prefix + ".bias", c_out, 1);
    else
        c.bias = Tensor::mat(c_out, 1);   // zero-filled

    if (!bn_prefix.empty()) {
        Tensor gamma = load_whole(f, kWho, bn_prefix + ".weight", c_out, 1);
        Tensor beta  = load_whole(f, kWho, bn_prefix + ".bias", c_out, 1);
        Tensor mean  = load_whole(f, kWho, bn_prefix + ".running_mean", c_out, 1);
        Tensor var   = load_whole(f, kWho, bn_prefix + ".running_var", c_out, 1);
        const float* g = gamma.host_f32();
        const float* bt = beta.host_f32();
        const float* mu = mean.host_f32();
        const float* vr = var.host_f32();
        float* w = c.weight.host_f32_mut();
        float* b = c.bias.host_f32_mut();
        const int per = cin_g * k * k;
        for (int o = 0; o < c_out; ++o) {
            const float s = g[o] / std::sqrt(vr[o] + kBnEps);
            float* wr = w + static_cast<std::size_t>(o) * per;
            for (int i = 0; i < per; ++i) wr[i] *= s;
            b[o] = (b[o] - mu[o]) * s + bt[o];
        }
    }
    return c;
}

// One truncated-MobileNetV2 inverted-residual block.
struct IRBlock {
    bool has_expand = false;
    int stride = 1;
    bool use_res = false;
    int hidden = 0;       // depthwise channel count (== dw.c_out)
    Conv pw, dw, pwl;     // pw used only when has_expand
};

struct TypeA { Conv conv1, conv2; bool upscale = false; };  // conv1=b-path, conv2=a-path
struct TypeB { Conv conv1, conv2; };
struct TypeC { Conv conv1, conv2, conv3; };

// ── forward helpers ──────────────────────────────────────────────────────────

Tensor convf(const Conv& c, const Tensor& x, int H, int W,
             int stride, int pad, int dil, int& Ho, int& Wo) {
    Tensor y;
    brotensor::conv2d_forward(x, c.weight, &c.bias, /*N=*/1, c.c_in, H, W,
                              c.c_out, c.k, c.k, stride, stride, pad, pad,
                              dil, dil, c.groups, y);
    Ho = (H + 2 * pad - dil * (c.k - 1) - 1) / stride + 1;
    Wo = (W + 2 * pad - dil * (c.k - 1) - 1) / stride + 1;
    return y;
}

void relu6(Tensor& x) { brotensor::clamp(x, 0.0f, 6.0f); }
void relu(Tensor& x) {
    Tensor y;
    brotensor::relu_forward(x, y);
    x = std::move(y);
}

// TFLite "same" stride-2 padding: pad right + bottom by 1 (zeros), then the
// caller convolves with pad 0. Updates H,W to H+1,W+1.
Tensor tfl_pad(const Tensor& x, int C, int& H, int& W) {
    Tensor y;
    brotensor::pad2d_forward(x, /*N=*/1, C, H, W,
                             /*pt=*/0, /*pb=*/1, /*pl=*/0, /*pr=*/1,
                             /*mode=*/0, y);
    H += 1; W += 1;
    return y;
}

Tensor cat_ch(const Tensor& a, int ca, const Tensor& b, int cb, int H, int W) {
    std::vector<const Tensor*> parts{&a, &b};
    std::vector<int> cs{ca, cb};
    Tensor out;
    brotensor::concat_nchw_channels(parts, /*N=*/1, H, W, cs, out);
    return out;
}

}  // namespace

// ── Impl ────────────────────────────────────────────────────────────────────

struct MLSDdetector::Impl {
    bool loaded = false;
    brotensor::Device device = brotensor::Device::CPU;

    Conv stem;                       // features.0: 4->32, 3x3 s2 (TFLite pad)
    std::vector<IRBlock> blocks;     // features.1..13
    TypeA b15, b17, b19, b21;
    TypeB b16, b18, b20, b22;
    TypeC b23;

    // Backbone tap block indices (fpn_selected {1,3,6,10,13} -> blocks idx).
    static constexpr int kTap[5] = {0, 2, 5, 9, 12};

    Tensor ir(const IRBlock& B, const Tensor& x_in, int& H, int& W) const {
        Tensor cur = x_in;
        int h = H, w = W, ho, wo;
        if (B.has_expand) {                       // pw 1x1 + ReLU6
            cur = convf(B.pw, cur, h, w, 1, 0, 1, ho, wo); h = ho; w = wo;
            relu6(cur);
        }
        if (B.stride == 2) {                      // dw 3x3 s2 (TFLite pad)
            Tensor p = tfl_pad(cur, B.hidden, h, w);
            cur = convf(B.dw, p, h, w, 2, 0, 1, ho, wo); h = ho; w = wo;
        } else {                                  // dw 3x3 s1 pad1
            cur = convf(B.dw, cur, h, w, 1, 1, 1, ho, wo); h = ho; w = wo;
        }
        relu6(cur);
        cur = convf(B.pwl, cur, h, w, 1, 0, 1, ho, wo); h = ho; w = wo;  // pw-linear
        if (B.use_res) brotensor::add_inplace(cur, x_in);
        H = h; W = w;
        return cur;
    }

    Tensor typeA(const TypeA& blk, const Tensor& a, int /*ca*/, int aH, int aW,
                 const Tensor& b, int /*cb*/, int bH, int bW) const {
        int ho, wo;
        Tensor bo = convf(blk.conv1, b, bH, bW, 1, 0, 1, ho, wo); relu(bo);
        Tensor ao = convf(blk.conv2, a, aH, aW, 1, 0, 1, ho, wo); relu(ao);
        if (blk.upscale) {
            Tensor up;
            brotensor::interp2d_align_corners_forward(
                bo, /*N=*/1, blk.conv1.c_out, bH, bW, aH, aW, /*mode=*/1, up);
            bo = std::move(up);
        }
        return cat_ch(ao, blk.conv2.c_out, bo, blk.conv1.c_out, aH, aW);
    }

    Tensor typeB(const TypeB& blk, const Tensor& x, int H, int W) const {
        int ho, wo;
        Tensor t = convf(blk.conv1, x, H, W, 1, 1, 1, ho, wo); relu(t);
        brotensor::add_inplace(t, x);
        Tensor o = convf(blk.conv2, t, H, W, 1, 1, 1, ho, wo); relu(o);
        return o;
    }

    Tensor typeC(const TypeC& blk, const Tensor& x, int H, int W) const {
        int ho, wo;
        Tensor t = convf(blk.conv1, x, H, W, 1, /*pad=*/5, /*dil=*/5, ho, wo);
        relu(t);
        t = convf(blk.conv2, t, H, W, 1, 1, 1, ho, wo); relu(t);
        t = convf(blk.conv3, t, H, W, 1, 0, 1, ho, wo);   // 1x1, no BN/ReLU
        return t;
    }
};

constexpr int MLSDdetector::Impl::kTap[5];

MLSDdetector::MLSDdetector(MlsdConfig cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>()) {}
MLSDdetector::~MLSDdetector() = default;
MLSDdetector::MLSDdetector(MLSDdetector&&) noexcept = default;
MLSDdetector& MLSDdetector::operator=(MLSDdetector&&) noexcept = default;

void MLSDdetector::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void MLSDdetector::load_file(const std::string& path) {
    st::File f = st::File::open(path);
    Impl m;

    // Stem: ConvBNReLU(4, 32, stride 2).
    m.stem = load_conv(f, "backbone.features.0.0", "backbone.features.0.1", 1);

    // Truncated MobileNetV2 inverted-residual settings: {t, c, n, s}.
    struct Setting { int t, c, n, s; };
    const Setting settings[5] = {
        {1, 16, 1, 1}, {6, 24, 2, 2}, {6, 32, 3, 2}, {6, 64, 4, 2}, {6, 96, 3, 1},
    };
    int input_channel = 32, fi = 1;
    for (const Setting& s : settings) {
        for (int i = 0; i < s.n; ++i) {
            const int stride = (i == 0) ? s.s : 1;
            const int inp = input_channel, oup = s.c, t = s.t;
            const std::string p = "backbone.features." + std::to_string(fi);
            IRBlock B;
            B.has_expand = (t != 1);
            B.stride = stride;
            if (B.has_expand) {
                B.pw  = load_conv(f, p + ".conv.0.0", p + ".conv.0.1", 1);
                B.dw  = load_conv(f, p + ".conv.1.0", p + ".conv.1.1", 0);  // depthwise
                B.pwl = load_conv(f, p + ".conv.2",   p + ".conv.3",   1);
            } else {
                B.dw  = load_conv(f, p + ".conv.0.0", p + ".conv.0.1", 0);  // depthwise
                B.pwl = load_conv(f, p + ".conv.1",   p + ".conv.2",   1);
            }
            B.hidden = B.dw.c_out;
            B.use_res = (stride == 1 && inp == oup);
            m.blocks.push_back(std::move(B));
            input_channel = oup;
            ++fi;
        }
    }

    auto A = [&](const std::string& n, bool up) {
        TypeA a;
        a.conv1 = load_conv(f, n + ".conv1.0", n + ".conv1.1", 1);
        a.conv2 = load_conv(f, n + ".conv2.0", n + ".conv2.1", 1);
        a.upscale = up;
        return a;
    };
    auto Bk = [&](const std::string& n) {
        TypeB b;
        b.conv1 = load_conv(f, n + ".conv1.0", n + ".conv1.1", 1);
        b.conv2 = load_conv(f, n + ".conv2.0", n + ".conv2.1", 1);
        return b;
    };
    m.b15 = A("block15", false);
    m.b16 = Bk("block16");
    m.b17 = A("block17", true);
    m.b18 = Bk("block18");
    m.b19 = A("block19", true);
    m.b20 = Bk("block20");
    m.b21 = A("block21", true);
    m.b22 = Bk("block22");
    m.b23.conv1 = load_conv(f, "block23.conv1.0", "block23.conv1.1", 1);
    m.b23.conv2 = load_conv(f, "block23.conv2.0", "block23.conv2.1", 1);
    m.b23.conv3 = load_conv(f, "block23.conv3", "", 1);   // 1x1, no BN

    m.loaded = true;
    *impl_ = std::move(m);
}

void MLSDdetector::to(brotensor::Device dev) {
    Impl& m = *impl_;
    if (!m.loaded) fail("to() called before load()");
    if (dev == m.device) return;
    auto mv      = [dev](Tensor& t) { if (t.data) t = t.to(dev); };
    auto mv_conv = [&](Conv& c) { mv(c.weight); mv(c.bias); };
    mv_conv(m.stem);
    for (IRBlock& B : m.blocks) { mv_conv(B.pw); mv_conv(B.dw); mv_conv(B.pwl); }
    for (TypeA* a : {&m.b15, &m.b17, &m.b19, &m.b21}) {
        mv_conv(a->conv1); mv_conv(a->conv2);
    }
    for (TypeB* b : {&m.b16, &m.b18, &m.b20, &m.b22}) {
        mv_conv(b->conv1); mv_conv(b->conv2);
    }
    mv_conv(m.b23.conv1); mv_conv(m.b23.conv2); mv_conv(m.b23.conv3);
    m.device = dev;
}

brotensor::Device MLSDdetector::device() const { return impl_->device; }

// ── Run ─────────────────────────────────────────────────────────────────────

TpMap MLSDdetector::run(const PreprocessedImage& pp) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("infer_tpmap() called before load()");

    const int S = pp.transform.model_size;
    Tensor x = (m.device == brotensor::Device::CPU) ? pp.pixels
                                                     : pp.pixels.to(m.device);

    // Stem: TFLite pad + 3x3 s2 conv + ReLU6 -> 32ch @ S/2.
    int H = S, W = S, ho, wo;
    {
        Tensor p = tfl_pad(x, 4, H, W);
        x = convf(m.stem, p, H, W, 2, 0, 1, ho, wo); H = ho; W = wo;
        relu6(x);
    }

    // Backbone: run all blocks, capturing the five FPN taps (c1..c5).
    Tensor taps[5];
    int tap_i = 0;
    for (std::size_t i = 0; i < m.blocks.size(); ++i) {
        x = m.ir(m.blocks[i], x, H, W);
        if (tap_i < 5 && static_cast<int>(i) == Impl::kTap[tap_i])
            taps[tap_i++] = x;
    }
    const Tensor& c1 = taps[0];   // 16 @ S/2
    const Tensor& c2 = taps[1];   // 24 @ S/4
    const Tensor& c3 = taps[2];   // 32 @ S/8
    const Tensor& c4 = taps[3];   // 64 @ S/16
    const Tensor& c5 = taps[4];   // 96 @ S/16
    const int s2 = S / 2, s4 = S / 4, s8 = S / 8, s16 = S / 16;

    // FPN decode head.
    Tensor y = m.typeA(m.b15, c4, 64, s16, s16, c5, 96, s16, s16);   // 128 @ s16
    y = m.typeB(m.b16, y, s16, s16);                                  // 64  @ s16
    y = m.typeA(m.b17, c3, 32, s8, s8, y, 64, s16, s16);             // 128 @ s8
    y = m.typeB(m.b18, y, s8, s8);                                    // 64  @ s8
    y = m.typeA(m.b19, c2, 24, s4, s4, y, 64, s8, s8);              // 128 @ s4
    y = m.typeB(m.b20, y, s4, s4);                                    // 64  @ s4
    y = m.typeA(m.b21, c1, 16, s2, s2, y, 64, s4, s4);             // 128 @ s2
    y = m.typeB(m.b22, y, s2, s2);                                    // 64  @ s2
    y = m.typeC(m.b23, y, s2, s2);                                    // 16  @ s2

    // Pull the 16-channel output to the host and slice channels 7:16 -> 9.
    Tensor host = (y.device == brotensor::Device::CPU)
                      ? y : y.to(brotensor::Device::CPU);
    const float* d = host.host_f32();
    const int plane = s2 * s2;
    TpMap tp;
    tp.channels = 9;
    tp.height = s2;
    tp.width = s2;
    tp.data.resize(static_cast<std::size_t>(9) * plane);
    for (int c = 0; c < 9; ++c) {
        const float* sp = d + static_cast<std::size_t>(7 + c) * plane;
        std::copy(sp, sp + plane, tp.data.begin() + static_cast<std::size_t>(c) * plane);
    }
    return tp;
}

TpMap MLSDdetector::infer_tpmap(const uint8_t* rgb, int w, int h,
                                int channels) const {
    PreprocessedImage pp = preprocess(rgb, w, h, channels, cfg_.model_size);
    return run(pp);
}

// ── Decode ────────────────────────────────────────────────────────────────

LineMap MLSDdetector::detect(const uint8_t* rgb, int w, int h,
                             int channels) const {
    const TpMap tp = infer_tpmap(rgb, w, h, channels);
    const int H = tp.height, W = tp.width, plane = H * W;
    const float* center = tp.data.data();                 // channel 0
    const float* d0 = tp.data.data() + 1 * plane;         // disp x_start
    const float* d1 = tp.data.data() + 2 * plane;         // disp y_start
    const float* d2 = tp.data.data() + 3 * plane;         // disp x_end
    const float* d3 = tp.data.data() + 4 * plane;         // disp y_end

    // Center heatmap: sigmoid, then 3x3 max-pool NMS (keep local maxima).
    std::vector<float> heat(plane);
    for (int i = 0; i < plane; ++i)
        heat[i] = 1.0f / (1.0f + std::exp(-center[i]));
    std::vector<float> masked(plane, 0.0f);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float mx = -1e30f;
            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= H) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = x + dx;
                    if (xx < 0 || xx >= W) continue;
                    mx = std::max(mx, heat[yy * W + xx]);
                }
            }
            const int idx = y * W + x;
            if (heat[idx] == mx) masked[idx] = heat[idx];   // local-max keep
        }
    }

    // Top-K centers by score (value desc, index asc for stable ties).
    std::vector<int> order(plane);
    for (int i = 0; i < plane; ++i) order[i] = i;
    const int K = std::min(cfg_.topk, plane);
    std::partial_sort(order.begin(), order.begin() + K, order.end(),
                      [&](int a, int b) {
                          if (masked[a] != masked[b]) return masked[a] > masked[b];
                          return a < b;
                      });

    // Map 256-grid endpoints to the original image: x2 (256->512) then by the
    // original/model_size ratio (the reference's w_ratio / h_ratio).
    const float rx = static_cast<float>(w) / static_cast<float>(cfg_.model_size);
    const float ry = static_cast<float>(h) / static_cast<float>(cfg_.model_size);

    LineMap out;
    out.width = w;
    out.height = h;
    for (int n = 0; n < K; ++n) {
        const int idx = order[n];
        const float score = masked[idx];
        if (score <= cfg_.score_thr) continue;
        const int cy = idx / W, cx = idx % W;
        const float dxs = d0[idx], dys = d1[idx], dxe = d2[idx], dye = d3[idx];
        const float ddx = dxs - dxe, ddy = dys - dye;
        const float dist = std::sqrt(ddx * ddx + ddy * ddy);
        if (dist <= cfg_.dist_thr) continue;
        LineSegment seg;
        // 256-grid endpoints -> 2x (256->512) -> original-image scale.
        seg.x1 = (cx + dxs) * 2.0f * rx;
        seg.y1 = (cy + dys) * 2.0f * ry;
        seg.x2 = (cx + dxe) * 2.0f * rx;
        seg.y2 = (cy + dye) * 2.0f * ry;
        seg.score = score;
        out.segments.push_back(seg);
    }
    return out;
}

}  // namespace brovisionml::mlsd
