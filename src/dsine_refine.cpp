#include "brovisionml/dsine_refine.h"

#include "brovisionml/dsine_ops.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include "weights_util.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace brovisionml::dsine {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "dsine::Refiner: ";

constexpr int   kK    = 8;   // downsample_ratio
constexpr int   kIter = 5;   // num_iter_test
constexpr float kPi   = 3.14159265358979323846f;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

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

// A plain conv: weight (C_out, C_in*k*k) OIHW + bias (C_out,1).
struct Conv {
    Tensor weight, bias;
    int c_in = 0, c_out = 0, k = 1;
};

Conv load_conv(const st::File& f, const std::string& prefix) {
    Conv c;
    c.c_out = dim0(f, prefix + ".weight");
    c.c_in  = dim1(f, prefix + ".weight");
    const st::TensorView* v = f.find(prefix + ".weight");
    const int64_t per = static_cast<int64_t>(c.c_out) * c.c_in;
    const int64_t kk = v->numel() / per;
    int k = 1;
    while (k * k < kk) ++k;
    if (static_cast<int64_t>(k) * k != kk)
        fail("conv '" + prefix + "' has non-square kernel");
    c.k = k;
    c.weight = load_whole(f, kWho, prefix + ".weight", c.c_out, c.c_in * k * k);
    c.bias   = load_whole(f, kWho, prefix + ".bias", c.c_out, 1);
    return c;
}

// get_prediction_head: 3 plain convs (3x3 pad1, 1x1, 1x1) with ReLU between.
struct PredHead {
    Conv c0, c2, c4;
};

PredHead load_head(const st::File& f, const std::string& prefix) {
    PredHead h;
    h.c0 = load_conv(f, prefix + ".0");
    h.c2 = load_conv(f, prefix + ".2");
    h.c4 = load_conv(f, prefix + ".4");
    return h;
}

struct ConvGRU {
    Conv convz, convr, convq;   // 5x5 pad2, 130 -> 64
};

ConvGRU load_gru(const st::File& f) {
    ConvGRU g;
    g.convz = load_conv(f, "gru.convz");
    g.convr = load_conv(f, "gru.convr");
    g.convq = load_conv(f, "gru.convq");
    return g;
}

// ── device-agnostic helpers (every step is a brotensor op) ───────────────────

Tensor conv2d(const Conv& c, const Tensor& x, int H, int W, int pad) {
    Tensor y;
    brotensor::conv2d_forward(x, c.weight, &c.bias, /*N=*/1, c.c_in, H, W,
                              c.c_out, c.k, c.k, /*sh=*/1, /*sw=*/1,
                              pad, pad, /*dh=*/1, /*dw=*/1, /*groups=*/1, y);
    return y;
}

void apply_relu(Tensor& x) {
    Tensor y;
    brotensor::relu_forward(x, y);
    x = std::move(y);
}
void apply_sigmoid(Tensor& x) {
    Tensor y;
    brotensor::sigmoid_forward(x, y);
    x = std::move(y);
}
void apply_tanh(Tensor& x) {
    Tensor y;
    brotensor::tanh_forward(x, y);
    x = std::move(y);
}

// get_prediction_head forward: conv3x3(pad1) -> ReLU -> conv1x1 -> ReLU -> conv1x1.
Tensor run_head(const PredHead& hd, const Tensor& x, int H, int W) {
    Tensor h = conv2d(hd.c0, x, H, W, /*pad=*/1);
    apply_relu(h);
    h = conv2d(hd.c2, h, H, W, /*pad=*/0);
    apply_relu(h);
    h = conv2d(hd.c4, h, H, W, /*pad=*/0);
    return h;
}

// Channel-concat two NCHW (1, C*H*W) tensors with the same (H,W). Device-agnostic
// (out adopts the inputs' device); both inputs must share a device.
Tensor cat_ch(const Tensor& a, int ca, const Tensor& b, int cb, int H, int W) {
    Tensor out;
    const std::vector<const Tensor*> parts{&a, &b};
    const std::vector<int> cs{ca, cb};
    brotensor::concat_nchw_channels(parts, /*N=*/1, H, W, cs, out);
    return out;
}

// ConvGRU (submodules.ConvGRU, ks=5 pad=2), all brotensor ops:
//   hx = cat([h, x]); z = sigmoid(convz(hx)); r = sigmoid(convr(hx));
//   q  = tanh(convq(cat([r*h, x])));   h_new = (1-z)*h + z*q
Tensor run_gru(const ConvGRU& g, const Tensor& h, const Tensor& x,
               int hidden, int input, int H, int W) {
    Tensor hx = cat_ch(h, hidden, x, input, H, W);
    Tensor z = conv2d(g.convz, hx, H, W, /*pad=*/2); apply_sigmoid(z);
    Tensor r = conv2d(g.convr, hx, H, W, /*pad=*/2); apply_sigmoid(r);
    brotensor::mul_inplace(r, h);                       // r = r * h
    Tensor rhx = cat_ch(r, hidden, x, input, H, W);
    Tensor q = conv2d(g.convq, rhx, H, W, /*pad=*/2); apply_tanh(q);
    // h_new = (1-z)*h + z*q
    Tensor omz = z.clone();
    brotensor::scale_inplace(omz, -1.0f);
    brotensor::add_scalar_inplace(omz, 1.0f);           // omz = 1 - z
    brotensor::mul_inplace(omz, h);                     // (1-z) * h
    brotensor::mul_inplace(z, q);                       // z * q
    brotensor::add_inplace(omz, z);                     // h_new
    return omz;
}

}  // namespace

// ── Impl ────────────────────────────────────────────────────────────────────

struct Refiner::Impl {
    bool loaded = false;
    brotensor::Device device = brotensor::Device::CPU;
    ConvGRU  gru;
    PredHead prob_head;     // out 25
    PredHead xy_head;       // out 50
    PredHead angle_head;    // out 25
    PredHead up_prob_head;  // out 576
    int hidden_dim = 64;
    int input_dim  = 66;    // feature_dim + 2
};

Refiner::Refiner() : impl_(std::make_unique<Impl>()) {}
Refiner::~Refiner() = default;
Refiner::Refiner(Refiner&&) noexcept = default;
Refiner& Refiner::operator=(Refiner&&) noexcept = default;

void Refiner::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void Refiner::load_file(const std::string& path) {
    st::File f = st::File::open(path);
    Impl m;
    m.gru          = load_gru(f);
    m.prob_head    = load_head(f, "prob_head");
    m.xy_head      = load_head(f, "xy_head");
    m.angle_head   = load_head(f, "angle_head");
    m.up_prob_head = load_head(f, "up_prob_head");
    m.loaded = true;
    *impl_ = std::move(m);
}

void Refiner::to(brotensor::Device dev) {
    Impl& m = *impl_;
    if (!m.loaded) fail("to() called before load()");
    if (dev == m.device) return;
    auto mv      = [dev](Tensor& t) { if (t.data) t = t.to(dev); };
    auto mv_conv = [&](Conv& c) { mv(c.weight); mv(c.bias); };
    auto mv_head = [&](PredHead& h) { mv_conv(h.c0); mv_conv(h.c2); mv_conv(h.c4); };
    mv_conv(m.gru.convz); mv_conv(m.gru.convr); mv_conv(m.gru.convq);
    mv_head(m.prob_head); mv_head(m.xy_head);
    mv_head(m.angle_head); mv_head(m.up_prob_head);
    m.device = dev;
}

brotensor::Device Refiner::device() const { return impl_->device; }

brotensor::Tensor Refiner::forward(const DecoderOutput& dec,
                                   const Intrinsics& intrins, int origH,
                                   int origW, const DsineTransform& tf) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("forward() called before load()");

    const brotensor::Device dev = m.device;
    const int H = dec.h8, W = dec.w8;

    // ray_8 / uv_8 (build_ray8/build_uv apply the +0.5 internally) — built
    // host-side then uploaded so the on-device ops below can consume them.
    UvMap ray8 = build_ray8(intrins, H, W, origH, origW);
    UvMap uv8  = build_uv(intrins, H, W, origH, origW);
    if (dev != brotensor::Device::CPU) {
        ray8.data = ray8.data.to(dev);
        uv8.data  = uv8.data.to(dev);
    }

    // Scaled intrinsics on the /8 grid, from the SAME +0.5 intrinsics build_ray8
    // uses (consistency requirement) — passed to the AngMF propagate op.
    const double sw = static_cast<double>(W) / origW;
    const double sh = static_cast<double>(H) / origH;
    const double fu = intrins.fx * sw;
    const double cu = (intrins.cx + 0.5) * sw;
    const double fv = intrins.fy * sh;
    const double cv = (intrins.cy + 0.5) * sh;

    // ── Step 4 init ──────────────────────────────────────────────────────────
    // pred_norm = RayReLU(decoder.normal, ray_8)   (deep-copy then clamp in place)
    Tensor pred_norm = dec.normal;
    ray_relu(pred_norm, ray8.data, H, W);

    // feat_map = cat([decoder.feature, uv_8]) -> 66ch ; h = decoder.hidden.
    Tensor feat_map = cat_ch(dec.feature, 64, uv8.data, 2, H, W);
    Tensor h = dec.hidden;

    // up0 = upsample(h, pred_norm, uv_8) — pred_list[0], not used for the result.

    Tensor up_pred_norm;   // (1, 3*(8H)*(8W)), refreshed each iter

    // ── 5 refinement iterations ──────────────────────────────────────────────
    for (int it = 0; it < kIter; ++it) {
        Tensor h_new = run_gru(m.gru, h, feat_map, m.hidden_dim, m.input_dim, H, W);

        // head input: cat([h_new, uv_8]) -> 66ch
        Tensor hin = cat_ch(h_new, m.hidden_dim, uv8.data, 2, H, W);

        // Per-neighbor head outputs. prob/angle are activated here (brotensor);
        // the xy pair is normalized inside angmf_propagate (no brotensor op
        // covers the strided 2-vector normalize).
        Tensor prob = run_head(m.prob_head, hin, H, W);   // (25)
        apply_sigmoid(prob);
        Tensor xy = run_head(m.xy_head, hin, H, W);        // (50) RAW
        Tensor angle = run_head(m.angle_head, hin, H, W);  // (25)
        apply_sigmoid(angle);
        brotensor::scale_inplace(angle, kPi);

        // Fused AngMF rotation-based mean-field update -> new coarse normal.
        Tensor new_pred;
        angmf_propagate(pred_norm, prob, xy, angle, ray8.data,
                        fu, cu, fv, cv, H, W, new_pred);
        pred_norm = std::move(new_pred);

        // up_pred_norm = normalize(convex_upsample(pred_norm, up_prob_head(...)))
        Tensor up_mask = run_head(m.up_prob_head, hin, H, W);   // (576)
        Tensor up;
        brotensor::convex_upsample_forward(pred_norm, up_mask, /*N=*/1, /*C=*/3,
                                           H, W, /*scale=*/kK, up);
        brotensor::l2_normalize_nchw_forward(up, /*N=*/1, /*C=*/3, kK * H, kK * W,
                                             1e-12f, up);   // X/Y may alias
        up_pred_norm = std::move(up);

        h = std::move(h_new);
    }

    // ── crop the full-res result to the original (unpadded) image ─────────────
    const int fullH = kK * H, fullW = kK * W;
    Tensor out;
    brotensor::slice2d_forward(up_pred_norm, /*N=*/1, /*C=*/3, fullH, fullW,
                               /*h0=*/tf.pad_t, /*w0=*/tf.pad_l,
                               /*H_out=*/tf.orig_h, /*W_out=*/tf.orig_w, out);
    return out;
}

}  // namespace brovisionml::dsine
