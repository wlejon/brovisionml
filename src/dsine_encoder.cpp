#include "brovisionml/dsine_encoder.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include "weights_util.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brovisionml::dsine {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "dsine::EncoderB5: ";

// geffnet tf_efficientnet BatchNorm eps (PyTorch _BN_EPS_TF_DEFAULT).
constexpr float kBnEps = 1e-3f;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

// Per-stage connectivity that shapes alone do not encode: kernel size and the
// stride applied by block 0 (blocks 1+ are stride 1). Block counts are read
// from the checkpoint, not from here.
struct StageSpec {
    int kernel;
    int stride0;
};
constexpr StageSpec kStages[7] = {
    {3, 1},   // stage0: DepthwiseSeparable
    {3, 2},   // stage1
    {5, 2},   // stage2  <- s8
    {3, 2},   // stage3
    {5, 1},   // stage4  <- s16
    {5, 2},   // stage5
    {3, 1},   // stage6
};

// A BatchNorm's four host FP32 (C,1) tensors.
struct BN {
    Tensor weight, bias, mean, var;
    int C = 0;
};

// One EfficientNet block (DepthwiseSeparable for stage0, InvertedResidual for
// stages 1-6). Channel dims are read from the checkpoint per block. For
// DepthwiseSeparable there is no expand conv (conv_pw is the *projection* and
// bn3 is unused); `is_ir` distinguishes the two forwards.
struct Block {
    bool is_ir = false;       // InvertedResidual vs DepthwiseSeparable
    int  in_ch = 0;
    int  out_ch = 0;
    int  exp_ch = 0;          // expanded (== dw) channels
    int  kernel = 3;
    int  stride = 1;
    bool residual = false;    // stride==1 && in==out

    // InvertedResidual: conv_pw (exp,in) expand. DepthwiseSeparable: unused.
    Tensor conv_pw_expand;    // (exp, in)  flattened 1x1
    BN     bn1;               // post-expand (IR) / post-dw (DS) BN
    // Depthwise (exp,1,k,k) -> brotensor layout (exp, k*k).
    Tensor conv_dw;
    BN     bn2;               // post-dw (IR) / projection BN (DS)
    // SE.
    Tensor se_reduce_w, se_reduce_b;   // (rd, C) / (rd,1)
    Tensor se_expand_w, se_expand_b;   // (C, rd) / (C,1)
    int    se_rd = 0;
    // Projection: conv_pwl (out,exp) for IR; conv_pw (out,dw) for DS.
    Tensor conv_project;
    BN     bn3;               // post-project BN (IR only)
};

// ─── TF-"SAME" padding ────────────────────────────────────────────────────────

// out = ceil(i/s); pad_total = max((out-1)*s + k - i, 0). Returns before/after
// split (smaller half before).
void tf_same_pad(int i, int s, int k, int& before, int& after) {
    const int out = (i + s - 1) / s;
    const int total = std::max((out - 1) * s + k - i, 0);
    before = total / 2;
    after  = total - before;
}

// ─── Host helpers ─────────────────────────────────────────────────────────────

// Number of elements (rows) of a 1-D / leading dim of a named tensor's shape.
int dim0(const st::File& f, const std::string& name) {
    const st::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    if (v->shape.empty()) fail("tensor '" + name + "' has no shape");
    return static_cast<int>(v->shape[0]);
}
int dim1(const st::File& f, const std::string& name) {
    const st::TensorView* v = f.find(name);
    if (!v) fail("missing tensor '" + name + "'");
    if (v->shape.size() < 2) fail("tensor '" + name + "' has <2 dims");
    return static_cast<int>(v->shape[1]);
}

// Load the four BN tensors for `prefix` ("...bn1"), reading C from the weight.
BN load_bn(const st::File& f, const std::string& prefix) {
    const int C = dim0(f, prefix + ".weight");
    BN bn;
    bn.C = C;
    bn.weight = load_whole(f, kWho, prefix + ".weight", C, 1);
    bn.bias   = load_whole(f, kWho, prefix + ".bias", C, 1);
    bn.mean   = load_whole(f, kWho, prefix + ".running_mean", C, 1);
    bn.var    = load_whole(f, kWho, prefix + ".running_var", C, 1);
    return bn;
}

}  // namespace

// ─── Impl ─────────────────────────────────────────────────────────────────────

struct EncoderB5::Impl {
    bool loaded = false;

    // Stem.
    Tensor conv_stem;   // (48, 3*3*3) OIHW
    int    stem_out = 0;
    BN     bn1;

    // 7 stages of blocks.
    std::vector<std::vector<Block>> stages;

    // Head: raw conv_head only (no bn2 — see load_file).
    Tensor conv_head;   // (2048, 512) 1x1
    int    head_out = 0;
};

EncoderB5::EncoderB5() : impl_(std::make_unique<Impl>()) {}
EncoderB5::~EncoderB5() = default;
EncoderB5::EncoderB5(EncoderB5&&) noexcept = default;
EncoderB5& EncoderB5::operator=(EncoderB5&&) noexcept = default;

void EncoderB5::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void EncoderB5::load_file(const std::string& path) {
    st::File f = st::File::open(path);
    const std::string pre = "encoder.original_model.";

    Impl m;

    // Stem: conv_stem (out,3,3,3) -> OIHW (out, 3*3*3), bn1.
    {
        const int out = dim0(f, pre + "conv_stem.weight");
        m.stem_out = out;
        m.conv_stem = load_whole(f, kWho, pre + "conv_stem.weight",
                                 out, 3 * 3 * 3);
        m.bn1 = load_bn(f, pre + "bn1");
    }

    // Stages. Block count per stage is discovered by probing for the bn1.weight
    // key (every block has it); per-block channels are read from the shapes.
    m.stages.resize(7);
    int prev_out = m.stem_out;
    for (int s = 0; s < 7; ++s) {
        const StageSpec spec = kStages[s];
        const bool is_ir = (s != 0);
        for (int bi = 0;; ++bi) {
            const std::string bp = pre + "blocks." + std::to_string(s) + "." +
                                   std::to_string(bi) + ".";
            if (!f.find(bp + "bn1.weight")) break;   // end of stage

            Block blk;
            blk.is_ir  = is_ir;
            blk.kernel = spec.kernel;
            blk.stride = (bi == 0) ? spec.stride0 : 1;

            if (is_ir) {
                // conv_pw (exp,in) expand.
                const int exp = dim0(f, bp + "conv_pw.weight");
                const int in  = dim1(f, bp + "conv_pw.weight");
                const int out = dim0(f, bp + "conv_pwl.weight");
                blk.in_ch = in; blk.exp_ch = exp; blk.out_ch = out;
                blk.conv_pw_expand =
                    load_whole(f, kWho, bp + "conv_pw.weight", exp, in);
                blk.bn1 = load_bn(f, bp + "bn1");
                blk.conv_dw = load_whole(f, kWho, bp + "conv_dw.weight",
                                         exp, spec.kernel * spec.kernel);
                blk.bn2 = load_bn(f, bp + "bn2");
                // SE on the expanded feature.
                blk.se_rd = dim0(f, bp + "se.conv_reduce.weight");
                blk.se_reduce_w = load_whole(f, kWho, bp + "se.conv_reduce.weight",
                                             blk.se_rd, exp);
                blk.se_reduce_b = load_whole(f, kWho, bp + "se.conv_reduce.bias",
                                             blk.se_rd, 1);
                blk.se_expand_w = load_whole(f, kWho, bp + "se.conv_expand.weight",
                                             exp, blk.se_rd);
                blk.se_expand_b = load_whole(f, kWho, bp + "se.conv_expand.bias",
                                             exp, 1);
                blk.conv_project =
                    load_whole(f, kWho, bp + "conv_pwl.weight", out, exp);
                blk.bn3 = load_bn(f, bp + "bn3");
            } else {
                // DepthwiseSeparable: conv_dw (dw,1,k,k) -> bn1; conv_pw is the
                // projection (out,dw) -> bn2. dw channels == input channels.
                const int dw  = dim0(f, bp + "conv_dw.weight");
                const int out = dim0(f, bp + "conv_pw.weight");
                blk.in_ch = dw; blk.exp_ch = dw; blk.out_ch = out;
                blk.conv_dw = load_whole(f, kWho, bp + "conv_dw.weight",
                                         dw, spec.kernel * spec.kernel);
                blk.bn1 = load_bn(f, bp + "bn1");   // post-dw BN
                blk.se_rd = dim0(f, bp + "se.conv_reduce.weight");
                blk.se_reduce_w = load_whole(f, kWho, bp + "se.conv_reduce.weight",
                                             blk.se_rd, dw);
                blk.se_reduce_b = load_whole(f, kWho, bp + "se.conv_reduce.bias",
                                             blk.se_rd, 1);
                blk.se_expand_w = load_whole(f, kWho, bp + "se.conv_expand.weight",
                                             dw, blk.se_rd);
                blk.se_expand_b = load_whole(f, kWho, bp + "se.conv_expand.bias",
                                             dw, 1);
                blk.conv_project =
                    load_whole(f, kWho, bp + "conv_pw.weight", out, dw);
                blk.bn2 = load_bn(f, bp + "bn2");   // projection BN
            }

            blk.residual = (blk.stride == 1 && blk.in_ch == blk.out_ch);
            m.stages[s].push_back(std::move(blk));
        }
        if (m.stages[s].empty()) fail("stage " + std::to_string(s) + " has no blocks");
        prev_out = m.stages[s].back().out_ch;
    }

    // Head: conv_head (2048,512,1,1). Only the raw 1x1 conv is part of the
    // s32 tap; geffnet's head BN (bn2) and act2 are the classifier's, not the
    // feature tap's, so they are intentionally not loaded.
    {
        const int out = dim0(f, pre + "conv_head.weight");
        const int in  = dim1(f, pre + "conv_head.weight");
        if (in != prev_out)
            fail("conv_head in-channels (" + std::to_string(in) +
                 ") != last stage out (" + std::to_string(prev_out) + ")");
        m.head_out = out;
        m.conv_head = load_whole(f, kWho, pre + "conv_head.weight", out, in);
    }

    m.loaded = true;
    *impl_ = std::move(m);
}

// ─── Forward ──────────────────────────────────────────────────────────────────

namespace {

void apply_bn(Tensor& x, const BN& bn, int C, int H, int W) {
    Tensor y;
    brotensor::batch_norm_inference(x, bn.weight, bn.bias, bn.mean, bn.var,
                                    /*N=*/1, C, H, W, kBnEps, y);
    x = std::move(y);
}

void apply_silu(Tensor& x) {
    Tensor y;
    brotensor::silu_forward(x, y);
    x = std::move(y);
}

// Depthwise / standard 1x1 conv with optional TF-same pre-pad. `pad_same`
// asymmetrically zero-pads so out = ceil(i/s); 1x1 convs pass pad_same=false.
Tensor conv(const Tensor& x, const Tensor& Wt, int C_in, int H, int W,
            int C_out, int k, int stride, bool depthwise, bool pad_same,
            int& H_out, int& W_out) {
    const Tensor* in = &x;
    Tensor padded;
    int ph = 0, pw = 0;   // symmetric pad passed to conv2d (0 once pre-padded)
    int H_eff = H, W_eff = W;
    if (pad_same && k > 1) {
        int pt, pb, pl, pr;
        tf_same_pad(H, stride, k, pt, pb);
        tf_same_pad(W, stride, k, pl, pr);
        brotensor::pad2d_forward(x, /*N=*/1, C_in, H, W, pt, pb, pl, pr,
                                 /*mode=*/0, padded);
        in = &padded;
        H_eff = H + pt + pb;
        W_eff = W + pl + pr;
    }
    const int groups = depthwise ? C_in : 1;
    Tensor y;
    brotensor::conv2d_forward(*in, Wt, /*bias=*/nullptr, /*N=*/1, C_in,
                              H_eff, W_eff, C_out, k, k, stride, stride,
                              ph, pw, /*dil=*/1, 1, groups, y);
    H_out = (H_eff - k) / stride + 1;
    W_out = (W_eff - k) / stride + 1;
    return y;
}

// Squeeze-Excite on an NCHW feature (1, C*H*W): global avg pool -> reduce(1x1)
// -> SiLU -> expand(1x1) -> sigmoid -> channel-wise multiply in place.
void apply_se(Tensor& x, const Block& b, int C, int H, int W) {
    const int HW = H * W;
    const float* xp = x.host_f32();

    // Global average pool -> (C,1).
    std::vector<float> pooled(static_cast<std::size_t>(C));
    for (int c = 0; c < C; ++c) {
        const float* base = xp + static_cast<std::size_t>(c) * HW;
        double acc = 0.0;
        for (int i = 0; i < HW; ++i) acc += base[i];
        pooled[c] = static_cast<float>(acc / HW);
    }

    // reduce: (rd,C) @ (C,1) + bias -> SiLU.
    const int rd = b.se_rd;
    const float* rw = b.se_reduce_w.host_f32();
    const float* rb = b.se_reduce_b.host_f32();
    std::vector<float> red(static_cast<std::size_t>(rd));
    for (int o = 0; o < rd; ++o) {
        const float* row = rw + static_cast<std::size_t>(o) * C;
        double acc = rb[o];
        for (int c = 0; c < C; ++c) acc += static_cast<double>(row[c]) * pooled[c];
        const float v = static_cast<float>(acc);
        red[o] = v / (1.0f + std::exp(-v));   // SiLU
    }

    // expand: (C,rd) @ (rd,1) + bias -> sigmoid -> gate.
    const float* ew = b.se_expand_w.host_f32();
    const float* eb = b.se_expand_b.host_f32();
    std::vector<float> gate(static_cast<std::size_t>(C));
    for (int c = 0; c < C; ++c) {
        const float* row = ew + static_cast<std::size_t>(c) * rd;
        double acc = eb[c];
        for (int o = 0; o < rd; ++o) acc += static_cast<double>(row[o]) * red[o];
        const float v = static_cast<float>(acc);
        gate[c] = 1.0f / (1.0f + std::exp(-v));   // sigmoid
    }

    // Channel-wise multiply.
    float* xm = x.host_f32_mut();
    for (int c = 0; c < C; ++c) {
        const float g = gate[c];
        float* base = xm + static_cast<std::size_t>(c) * HW;
        for (int i = 0; i < HW; ++i) base[i] *= g;
    }
}

// Run one block; returns its output and updates H/W.
Tensor run_block(const Block& b, const Tensor& x_in, int& H, int& W) {
    const int H_in = H, W_in = W;
    Tensor x = x_in;   // shallow-copy header; data shared until first op writes

    if (b.is_ir) {
        // Expand 1x1 -> bn1 -> SiLU.
        int h, w;
        x = conv(x, b.conv_pw_expand, b.in_ch, H_in, W_in, b.exp_ch,
                 /*k=*/1, /*stride=*/1, /*depthwise=*/false, /*pad_same=*/false,
                 h, w);
        apply_bn(x, b.bn1, b.exp_ch, h, w);
        apply_silu(x);
        // Depthwise k,stride (TF-same) -> bn2 -> SiLU.
        int hd, wd;
        x = conv(x, b.conv_dw, b.exp_ch, h, w, b.exp_ch, b.kernel, b.stride,
                 /*depthwise=*/true, /*pad_same=*/true, hd, wd);
        apply_bn(x, b.bn2, b.exp_ch, hd, wd);
        apply_silu(x);
        // SE.
        apply_se(x, b, b.exp_ch, hd, wd);
        // Project 1x1 -> bn3.
        int hp, wp;
        x = conv(x, b.conv_project, b.exp_ch, hd, wd, b.out_ch,
                 /*k=*/1, /*stride=*/1, /*depthwise=*/false, /*pad_same=*/false,
                 hp, wp);
        apply_bn(x, b.bn3, b.out_ch, hp, wp);
        H = hp; W = wp;
    } else {
        // DepthwiseSeparable: depthwise (TF-same) -> bn1 -> SiLU -> SE
        //   -> project 1x1 -> bn2.
        int hd, wd;
        x = conv(x, b.conv_dw, b.exp_ch, H_in, W_in, b.exp_ch, b.kernel,
                 b.stride, /*depthwise=*/true, /*pad_same=*/true, hd, wd);
        apply_bn(x, b.bn1, b.exp_ch, hd, wd);
        apply_silu(x);
        apply_se(x, b, b.exp_ch, hd, wd);
        int hp, wp;
        x = conv(x, b.conv_project, b.exp_ch, hd, wd, b.out_ch,
                 /*k=*/1, /*stride=*/1, /*depthwise=*/false, /*pad_same=*/false,
                 hp, wp);
        apply_bn(x, b.bn2, b.out_ch, hp, wp);
        H = hp; W = wp;
    }

    if (b.residual) brotensor::add_inplace(x, x_in);
    return x;
}

}  // namespace

EncoderTaps EncoderB5::forward(const brotensor::Tensor& pixels,
                               int H, int W) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("forward() called before load()");
    if (pixels.dtype != brotensor::Dtype::FP32)
        fail("forward() requires an FP32 pixel tensor");
    if (pixels.device != brotensor::Device::CPU)
        fail("forward() runs on the host; pixel tensor must be CPU-resident");
    if (pixels.rows != 1 || pixels.cols != 3 * H * W)
        fail("pixels must be (1, 3*H*W)");

    // Stem: conv_stem (s2,k3,TF-same) -> bn1 -> SiLU.
    int h, w;
    Tensor x = conv(pixels, m.conv_stem, /*C_in=*/3, H, W, m.stem_out,
                    /*k=*/3, /*stride=*/2, /*depthwise=*/false,
                    /*pad_same=*/true, h, w);
    apply_bn(x, m.bn1, m.stem_out, h, w);
    apply_silu(x);

    EncoderTaps taps;

    // Stages.
    for (int s = 0; s < 7; ++s) {
        for (const Block& b : m.stages[s]) {
            x = run_block(b, x, h, w);
        }
        if (s == 2) { taps.s8  = x; taps.h8  = h; taps.w8  = w; }
        if (s == 4) { taps.s16 = x; taps.h16 = h; taps.w16 = w; }
    }

    // Head tap: the DSINE encoder taps conv_head's RAW 1x1 output — geffnet's
    // head BN (bn2) and act2(SiLU) are NOT part of the tap (this is
    // `features[11]` in the reference, captured right after the conv).
    int hh, ww;
    x = conv(x, m.conv_head, m.stages[6].back().out_ch, h, w, m.head_out,
             /*k=*/1, /*stride=*/1, /*depthwise=*/false, /*pad_same=*/false,
             hh, ww);
    taps.s32 = x; taps.h32 = hh; taps.w32 = ww;

    return taps;
}

}  // namespace brovisionml::dsine
