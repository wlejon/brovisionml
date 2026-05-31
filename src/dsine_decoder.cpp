#include "brovisionml/dsine_decoder.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include "weights_util.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brovisionml::dsine {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "dsine::Decoder: ";

// GroupNorm eps (torch nn.GroupNorm default).
constexpr float kGnEps = 1e-5f;
// GroupNorm groups (UpSampleGN uses GroupNorm(8, out)).
constexpr int kGnGroups = 8;
// LeakyReLU negative slope (torch nn.LeakyReLU() default).
constexpr float kLeakySlope = 0.01f;
// Conv2d_WS weight-standardization epsilon (added to std, not in quadrature).
constexpr float kWsEps = 1e-5f;

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
    // Infer kernel from total numel / (c_out*c_in).
    const int64_t per = static_cast<int64_t>(c.c_out) * c.c_in;
    const int64_t kk = v->numel() / per;
    int k = 1;
    while (k * k < kk) ++k;
    if (static_cast<int64_t>(k) * k != kk)
        fail("conv '" + prefix + "' has non-square kernel (numel " +
             std::to_string(v->numel()) + ")");
    c.k = k;
    c.weight = load_whole(f, kWho, prefix + ".weight", c.c_out, c.c_in * k * k);
    c.bias   = load_whole(f, kWho, prefix + ".bias", c.c_out, 1);
    return c;
}

// Weight-standardize a (C_out, C_in*k*k) conv weight in place, per OUTPUT
// channel: subtract mean over (in,kH,kW), divide by (unbiased_std + 1e-5).
void standardize_weight(Tensor& w, int c_out, int per_out) {
    float* p = w.host_f32_mut();
    for (int o = 0; o < c_out; ++o) {
        float* row = p + static_cast<std::size_t>(o) * per_out;
        double sum = 0.0;
        for (int i = 0; i < per_out; ++i) sum += row[i];
        const double mean = sum / per_out;
        double var = 0.0;
        for (int i = 0; i < per_out; ++i) {
            const double d = row[i] - mean;
            var += d * d;
        }
        // Unbiased (n-1) variance, matching torch.Tensor.std() default.
        var /= (per_out - 1);
        const double std = std::sqrt(var) + kWsEps;
        for (int i = 0; i < per_out; ++i)
            row[i] = static_cast<float>((row[i] - mean) / std);
    }
}

// Load a Conv2d_WS (folds weight standardization at load).
Conv load_conv_ws(const st::File& f, const std::string& prefix) {
    Conv c = load_conv(f, prefix);
    standardize_weight(c.weight, c.c_out, c.c_in * c.k * c.k);
    return c;
}

// GroupNorm affine params.
struct GN {
    Tensor weight, bias;
    int C = 0;
};

GN load_gn(const st::File& f, const std::string& prefix) {
    GN g;
    g.C = dim0(f, prefix + ".weight");
    g.weight = load_whole(f, kWho, prefix + ".weight", g.C, 1);
    g.bias   = load_whole(f, kWho, prefix + ".bias", g.C, 1);
    return g;
}

// One UpSampleGN: two WS convs + two GroupNorms.
struct UpSampleGN {
    Conv conv0;   // _net.0  Conv2d_WS(skip_input, out, 3, pad1)
    GN   gn1;     // _net.1
    Conv conv3;   // _net.3  Conv2d_WS(out, out, 3, pad1)
    GN   gn4;     // _net.4
    int out = 0;
};

UpSampleGN load_up(const st::File& f, const std::string& prefix) {
    UpSampleGN u;
    u.conv0 = load_conv_ws(f, prefix + "._net.0");
    u.gn1   = load_gn(f, prefix + "._net.1");
    u.conv3 = load_conv_ws(f, prefix + "._net.3");
    u.gn4   = load_gn(f, prefix + "._net.4");
    u.out   = u.conv0.c_out;
    return u;
}

// get_prediction_head: 3 plain convs (3x3, 1x1, 1x1) with ReLU between.
struct PredHead {
    Conv c0;   // .0  Conv2d(in, hid, 3, pad1)
    Conv c2;   // .2  Conv2d(hid, hid, 1)
    Conv c4;   // .4  Conv2d(hid, out, 1)
};

PredHead load_head(const st::File& f, const std::string& prefix) {
    PredHead h;
    h.c0 = load_conv(f, prefix + ".0");
    h.c2 = load_conv(f, prefix + ".2");
    h.c4 = load_conv(f, prefix + ".4");
    return h;
}

// ── Forward helpers ─────────────────────────────────────────────────────────

// Channel-concat two NCHW (1, C*H*W) tensors with the same (H,W).
Tensor cat_ch(const Tensor& a, int ca, const Tensor& b, int cb, int H, int W) {
    const int HW = H * W;
    Tensor out = Tensor::mat(1, (ca + cb) * HW);
    float* o = out.host_f32_mut();
    const float* ap = a.host_f32();
    const float* bp = b.host_f32();
    std::memcpy(o, ap, static_cast<std::size_t>(ca) * HW * sizeof(float));
    std::memcpy(o + static_cast<std::size_t>(ca) * HW, bp,
                static_cast<std::size_t>(cb) * HW * sizeof(float));
    return out;
}

// Plain conv2d (pad p, stride 1) with bias.
Tensor conv2d(const Conv& c, const Tensor& x, int H, int W, int pad) {
    Tensor y;
    brotensor::conv2d_forward(x, c.weight, &c.bias, /*N=*/1, c.c_in, H, W,
                              c.c_out, c.k, c.k, /*sh=*/1, /*sw=*/1,
                              pad, pad, /*dh=*/1, /*dw=*/1, /*groups=*/1, y);
    return y;
}

void apply_gn(Tensor& x, const GN& g, int H, int W) {
    Tensor y;
    brotensor::group_norm_forward(x, g.weight, g.bias, /*N=*/1, g.C, H, W,
                                  kGnGroups, kGnEps, y);
    x = std::move(y);
}

void apply_leaky(Tensor& x) {
    Tensor y;
    brotensor::leaky_relu_forward(x, kLeakySlope, y);
    x = std::move(y);
}

void apply_relu(Tensor& x) {
    Tensor y;
    brotensor::relu_forward(x, y);
    x = std::move(y);
}

// UpSampleGN.forward(x, concat_with): bilinear-upsample x (align_corners=False)
// to (Ht,Wt), cat with concat_with, run _net. x is (1, Cx*Hx*Wx); concat_with is
// (1, Cc*Ht*Wt). Returns (1, out*Ht*Wt).
Tensor run_up(const UpSampleGN& u, const Tensor& x, int Cx, int Hx, int Wx,
              const Tensor& concat_with, int Cc, int Ht, int Wt) {
    // Upsample x to (Ht,Wt), align_corners=False bilinear (mode=1).
    Tensor up_x;
    brotensor::interp2d_forward(x, /*N=*/1, Cx, Hx, Wx, Ht, Wt, /*mode=*/1, up_x);
    // cat([up_x, concat_with]).
    Tensor f = cat_ch(up_x, Cx, concat_with, Cc, Ht, Wt);
    // _net.
    Tensor h = conv2d(u.conv0, f, Ht, Wt, /*pad=*/1);
    apply_gn(h, u.gn1, Ht, Wt);
    apply_leaky(h);
    h = conv2d(u.conv3, h, Ht, Wt, /*pad=*/1);
    apply_gn(h, u.gn4, Ht, Wt);
    apply_leaky(h);
    return h;
}

// get_prediction_head forward: conv3x3(pad1) -> ReLU -> conv1x1 -> ReLU ->
// conv1x1.
Tensor run_head(const PredHead& hd, const Tensor& x, int H, int W) {
    Tensor h = conv2d(hd.c0, x, H, W, /*pad=*/1);
    apply_relu(h);
    h = conv2d(hd.c2, h, H, W, /*pad=*/0);
    apply_relu(h);
    h = conv2d(hd.c4, h, H, W, /*pad=*/0);
    return h;
}

// L2-normalize a (1, C*H*W) tensor over the channel axis (per pixel), in place.
void l2norm_channels(Tensor& x, int C, int H, int W) {
    const int HW = H * W;
    float* p = x.host_f32_mut();
    for (int i = 0; i < HW; ++i) {
        double ss = 0.0;
        for (int c = 0; c < C; ++c) {
            const double v = p[static_cast<std::size_t>(c) * HW + i];
            ss += v * v;
        }
        // torch F.normalize default eps = 1e-12.
        const double inv = 1.0 / std::max(std::sqrt(ss), 1e-12);
        for (int c = 0; c < C; ++c)
            p[static_cast<std::size_t>(c) * HW + i] *= static_cast<float>(inv);
    }
}

}  // namespace

// ── build_uv / build_ray8 ───────────────────────────────────────────────────

namespace {

// Shared get_ray core. Fills ray_x[y,x] and ray_y[y,x] over the (Hf,Wf) grid
// using the pixel-center coords (x+0.5, y+0.5) and the scaled intrinsics. The
// caller applies +0.5 to cx/cy before passing them in.
void get_ray_core(const Intrinsics& in, int Hf, int Wf, int origH, int origW,
                  std::vector<float>& ray_x, std::vector<float>& ray_y) {
    const double sw = static_cast<double>(Wf) / origW;
    const double sh = static_cast<double>(Hf) / origH;
    const double fu = in.fx * sw;
    const double cu = (in.cx + 0.5) * sw;   // +0.5 on principal point
    const double fv = in.fy * sh;
    const double cv = (in.cy + 0.5) * sh;
    ray_x.resize(static_cast<std::size_t>(Hf) * Wf);
    ray_y.resize(static_cast<std::size_t>(Hf) * Wf);
    for (int y = 0; y < Hf; ++y) {
        for (int x = 0; x < Wf; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * Wf + x;
            ray_x[idx] = static_cast<float>(((x + 0.5) - cu) / fu);
            ray_y[idx] = static_cast<float>(((y + 0.5) - cv) / fv);
        }
    }
}

}  // namespace

UvMap build_uv(const Intrinsics& intrins, int Hf, int Wf, int origH, int origW) {
    std::vector<float> rx, ry;
    get_ray_core(intrins, Hf, Wf, origH, origW, rx, ry);
    const int HW = Hf * Wf;
    UvMap m;
    m.c = 2; m.hf = Hf; m.wf = Wf;
    m.data = Tensor::mat(1, 2 * HW);
    float* p = m.data.host_f32_mut();
    std::memcpy(p, rx.data(), static_cast<std::size_t>(HW) * sizeof(float));
    std::memcpy(p + HW, ry.data(), static_cast<std::size_t>(HW) * sizeof(float));
    return m;
}

UvMap build_ray8(const Intrinsics& intrins, int Hf, int Wf, int origH, int origW) {
    std::vector<float> rx, ry;
    get_ray_core(intrins, Hf, Wf, origH, origW, rx, ry);
    const int HW = Hf * Wf;
    UvMap m;
    m.c = 3; m.hf = Hf; m.wf = Wf;
    m.data = Tensor::mat(1, 3 * HW);
    float* p = m.data.host_f32_mut();
    // L2-normalize [ray_x, ray_y, 1] per pixel (F.normalize over channels).
    for (int i = 0; i < HW; ++i) {
        const double x = rx[i], y = ry[i], z = 1.0;
        const double inv = 1.0 / std::max(std::sqrt(x * x + y * y + z * z), 1e-12);
        p[i]          = static_cast<float>(x * inv);
        p[HW + i]     = static_cast<float>(y * inv);
        p[2 * HW + i] = static_cast<float>(z * inv);
    }
    return m;
}

// ── Impl ────────────────────────────────────────────────────────────────────

struct Decoder::Impl {
    bool loaded = false;
    Conv       conv2;        // decoder.conv2  Conv2d(2050, 2048, 1x1)
    UpSampleGN up1;          // decoder.up1
    UpSampleGN up2;          // decoder.up2
    PredHead   normal_head;  // decoder.normal_head (out 3)
    PredHead   feature_head; // decoder.feature_head (out 64)
    PredHead   hidden_head;  // decoder.hidden_head (out 64)
};

Decoder::Decoder() : impl_(std::make_unique<Impl>()) {}
Decoder::~Decoder() = default;
Decoder::Decoder(Decoder&&) noexcept = default;
Decoder& Decoder::operator=(Decoder&&) noexcept = default;

void Decoder::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void Decoder::load_file(const std::string& path) {
    st::File f = st::File::open(path);
    Impl m;
    m.conv2        = load_conv(f, "decoder.conv2");
    m.up1          = load_up(f, "decoder.up1");
    m.up2          = load_up(f, "decoder.up2");
    m.normal_head  = load_head(f, "decoder.normal_head");
    m.feature_head = load_head(f, "decoder.feature_head");
    m.hidden_head  = load_head(f, "decoder.hidden_head");
    m.loaded = true;
    *impl_ = std::move(m);
}

DecoderOutput Decoder::forward(const EncoderTaps& taps, const Intrinsics& intrins,
                               int origH, int origW) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("forward() called before load()");

    const int h32 = taps.h32, w32 = taps.w32;
    const int h16 = taps.h16, w16 = taps.w16;
    const int h8  = taps.h8,  w8  = taps.w8;

    // uv maps from the intrinsics (+0.5 applied inside build_uv).
    const UvMap uv32 = build_uv(intrins, origH / 32, origW / 32, origH, origW);
    const UvMap uv16 = build_uv(intrins, origH / 16, origW / 16, origH, origW);
    const UvMap uv8  = build_uv(intrins, origH / 8,  origW / 8,  origH, origW);

    // The uv grid dims must match the encoder taps (both derive from origH/W).
    if (uv32.hf != h32 || uv32.wf != w32 || uv16.hf != h16 || uv16.wf != w16 ||
        uv8.hf != h8 || uv8.wf != w8)
        fail("uv grid dims do not match encoder taps");

    // x_d0 = conv2(cat([s32, uv_32]))           # 2048ch @ /32
    Tensor cat32 = cat_ch(taps.s32, 2048, uv32.data, 2, h32, w32);
    Tensor x_d0  = conv2d(m.conv2, cat32, h32, w32, /*pad=*/0);   // 2048ch

    // x_d1 = up1(x_d0, cat([s16, uv_16]))       # 1024ch @ /16
    Tensor cat16 = cat_ch(taps.s16, 176, uv16.data, 2, h16, w16);  // 178ch
    Tensor x_d1  = run_up(m.up1, x_d0, /*Cx=*/2048, h32, w32,
                          cat16, /*Cc=*/178, h16, w16);            // 1024ch

    // x_feat = up2(x_d1, cat([s8, uv_8]))       # 512ch @ /8
    Tensor cat8  = cat_ch(taps.s8, 64, uv8.data, 2, h8, w8);       // 66ch
    Tensor x_feat = run_up(m.up2, x_d1, /*Cx=*/1024, h16, w16,
                           cat8, /*Cc=*/66, h8, w8);               // 512ch

    // x_feat = cat([x_feat, uv_8])              # 514ch
    Tensor x_feat2 = cat_ch(x_feat, 512, uv8.data, 2, h8, w8);     // 514ch

    DecoderOutput out;
    out.h8 = h8; out.w8 = w8;

    // normal = L2normalize( normal_head(x_feat) )   # 3ch  (the golden)
    out.normal = run_head(m.normal_head, x_feat2, h8, w8);
    l2norm_channels(out.normal, 3, h8, w8);

    // f = feature_head(x_feat); h = hidden_head(x_feat)
    out.feature = run_head(m.feature_head, x_feat2, h8, w8);
    out.hidden  = run_head(m.hidden_head, x_feat2, h8, w8);

    return out;
}

}  // namespace brovisionml::dsine
