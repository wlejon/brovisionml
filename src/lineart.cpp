#include "brovisionml/lineart.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include "broimage/geometric.h"

#include "weights_util.h"

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brovisionml::lineart {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "lineart::LineartDetector: ";

constexpr int kResBlocks = 3;
constexpr float kEps = 1e-5f;   // torch InstanceNorm2d default

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

// A conv (or transposed conv). For a standard conv the checkpoint weight is
// (C_out, C_in, kH, kW); for a transposed conv it is (C_in, C_out, kH, kW). In
// both cases brotensor wants the flat row-major buffer as
// (rows, cols) = (dim0, numel/dim0), which already matches its expected layout
// (OIHW for conv, IO HW input-major for conv-transpose).
struct Conv {
    Tensor weight, bias;
    int c_in = 0, c_out = 0, k = 1;
    bool transpose = false;
};

Conv load_conv(const st::File& f, const std::string& prefix, bool transpose) {
    Conv c;
    c.transpose = transpose;
    const int d0 = dim0(f, prefix + ".weight");
    const int d1 = dim1(f, prefix + ".weight");
    if (transpose) { c.c_in = d0; c.c_out = d1; }
    else           { c.c_out = d0; c.c_in = d1; }

    const st::TensorView* v = f.find(prefix + ".weight");
    const int64_t per = static_cast<int64_t>(c.c_out) * c.c_in;
    const int64_t kk = v->numel() / per;
    int k = 1;
    while (static_cast<int64_t>(k) * k < kk) ++k;
    if (static_cast<int64_t>(k) * k != kk)
        fail("conv '" + prefix + "' has non-square kernel");
    c.k = k;

    const int rows = transpose ? c.c_in : c.c_out;
    const int cols = (transpose ? c.c_out : c.c_in) * k * k;
    c.weight = load_whole(f, kWho, prefix + ".weight", rows, cols);
    c.bias   = load_whole(f, kWho, prefix + ".bias", c.c_out, 1);  // len == C_out
    return c;
}

// ── forward helpers ──────────────────────────────────────────────────────────

// ReflectionPad2d(p): mirror-pad H and W by `p` on every side (pad2d mode 1).
Tensor reflect_pad(const Tensor& x, int C, int& H, int& W, int p) {
    Tensor y;
    brotensor::pad2d_forward(x, /*N=*/1, C, H, W, p, p, p, p, /*mode=*/1, y);
    H += 2 * p;
    W += 2 * p;
    return y;
}

// Plain conv (padding already applied by reflect_pad, so pad=0 here). Updates
// (H,W) to the convolution output dims for the given stride.
Tensor conv2d(const Conv& c, const Tensor& x, int& H, int& W, int stride) {
    Tensor y;
    brotensor::conv2d_forward(x, c.weight, &c.bias, /*N=*/1, c.c_in, H, W,
                              c.c_out, c.k, c.k, stride, stride,
                              /*pad_h=*/0, /*pad_w=*/0, /*dh=*/1, /*dw=*/1,
                              /*groups=*/1, y);
    H = (H - c.k) / stride + 1;
    W = (W - c.k) / stride + 1;
    return y;
}

// 3x3 stride-2 down-conv with symmetric zero pad 1 (torch padding=1). Updates
// (H,W). Used by the two model1 downsamples.
Tensor conv2d_down(const Conv& c, const Tensor& x, int& H, int& W) {
    Tensor y;
    brotensor::conv2d_forward(x, c.weight, &c.bias, /*N=*/1, c.c_in, H, W,
                              c.c_out, c.k, c.k, /*sh=*/2, /*sw=*/2,
                              /*pad_h=*/1, /*pad_w=*/1, /*dh=*/1, /*dw=*/1,
                              /*groups=*/1, y);
    H = (H + 2 - c.k) / 2 + 1;
    W = (W + 2 - c.k) / 2 + 1;
    return y;
}

// 3x3 stride-2 conv-transpose with pad 1, output_padding 1 (the model3 2x
// upsamplers). Doubles (H,W). Updates them.
Tensor conv_transpose_up(const Conv& c, const Tensor& x, int& H, int& W) {
    Tensor y;
    brotensor::conv_transpose2d_forward(
        x, c.weight, &c.bias, /*N=*/1, c.c_in, H, W,
        c.c_out, c.k, c.k, /*sh=*/2, /*sw=*/2, /*pad_h=*/1, /*pad_w=*/1,
        /*op_h=*/1, /*op_w=*/1, /*dh=*/1, /*dw=*/1, /*groups=*/1, y);
    H = (H - 1) * 2 - 2 + (c.k - 1) + 1 + 1;
    W = (W - 1) * 2 - 2 + (c.k - 1) + 1 + 1;
    return y;
}

void apply_relu(Tensor& x) {
    Tensor y;
    brotensor::relu_forward(x, y);
    x = std::move(y);
}

}  // namespace

// ── Impl ────────────────────────────────────────────────────────────────────

struct LineartDetector::Impl {
    bool loaded = false;
    brotensor::Device device = brotensor::Device::CPU;

    Conv m0;                 // 3->64, 7x7   (reflect pad 3)
    Conv d0, d1;             // 64->128, 128->256, 3x3 stride-2 (zero pad 1)
    struct Res { Conv a, b; } res[kResBlocks];   // 256->256, 3x3 (reflect pad 1)
    Conv u0, u1;             // 256->128, 128->64, 3x3 conv-transpose stride-2
    Conv m4;                 // 64->1, 7x7   (reflect pad 3)

    // InstanceNorm2d(affine=False) == group norm with num_groups == channels and
    // gamma=1, beta=0. Constant per channel-count; built once, migrated in to().
    std::map<int, std::pair<Tensor, Tensor>> norm_const;  // C -> (gamma, beta)

    const std::pair<Tensor, Tensor>& norm(int C) const {
        auto it = norm_const.find(C);
        if (it == norm_const.end())
            throw std::runtime_error(kWho + "no instance-norm constants for C=" +
                                     std::to_string(C));
        return it->second;
    }
};

LineartDetector::LineartDetector(LineartConfig cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>()) {}
LineartDetector::~LineartDetector() = default;
LineartDetector::LineartDetector(LineartDetector&&) noexcept = default;
LineartDetector& LineartDetector::operator=(LineartDetector&&) noexcept = default;

void LineartDetector::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void LineartDetector::load_file(const std::string& path) {
    st::File f = st::File::open(path);
    Impl m;

    m.m0 = load_conv(f, "model0.1", /*transpose=*/false);
    m.d0 = load_conv(f, "model1.0", false);
    m.d1 = load_conv(f, "model1.3", false);
    for (int i = 0; i < kResBlocks; ++i) {
        const std::string p = "model2." + std::to_string(i) + ".conv_block.";
        m.res[i].a = load_conv(f, p + "1", false);
        m.res[i].b = load_conv(f, p + "5", false);
    }
    m.u0 = load_conv(f, "model3.0", /*transpose=*/true);
    m.u1 = load_conv(f, "model3.3", /*transpose=*/true);
    m.m4 = load_conv(f, "model4.1", false);

    // Build the instance-norm constants for every channel count the trunk uses.
    for (int C : {64, 128, 256}) {
        Tensor gamma = Tensor::mat(C, 1);
        Tensor beta  = Tensor::mat(C, 1);   // zero-filled
        float* g = gamma.host_f32_mut();
        for (int i = 0; i < C; ++i) g[i] = 1.0f;
        m.norm_const.emplace(C, std::make_pair(std::move(gamma), std::move(beta)));
    }

    m.loaded = true;
    *impl_ = std::move(m);
}

void LineartDetector::to(brotensor::Device dev) {
    Impl& m = *impl_;
    if (!m.loaded) fail("to() called before load()");
    if (dev == m.device) return;
    auto mv      = [dev](Tensor& t) { if (t.data) t = t.to(dev); };
    auto mv_conv = [&](Conv& c) { mv(c.weight); mv(c.bias); };
    mv_conv(m.m0);
    mv_conv(m.d0); mv_conv(m.d1);
    for (auto& r : m.res) { mv_conv(r.a); mv_conv(r.b); }
    mv_conv(m.u0); mv_conv(m.u1);
    mv_conv(m.m4);
    for (auto& kv : m.norm_const) { mv(kv.second.first); mv(kv.second.second); }
    m.device = dev;
}

brotensor::Device LineartDetector::device() const { return impl_->device; }

// ── Run ─────────────────────────────────────────────────────────────────────

LineMap LineartDetector::run(const PreprocessedImage& pp) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("detect() called before load()");

    int H = pp.transform.proc_h;
    int W = pp.transform.proc_w;

    Tensor x = (m.device == brotensor::Device::CPU) ? pp.pixels
                                                     : pp.pixels.to(m.device);

    // InstanceNorm applied in place to `x` for the given channel count.
    auto instance_norm = [&](Tensor& t, int C, int h, int w) {
        const auto& gb = m.norm(C);
        Tensor y;
        brotensor::group_norm_forward(t, gb.first, gb.second, /*N=*/1, C, h, w,
                                      /*num_groups=*/C, kEps, y);
        t = std::move(y);
    };

    // model0: reflect-pad 3 -> 7x7 conv -> IN -> ReLU.
    x = reflect_pad(x, 3, H, W, 3);
    x = conv2d(m.m0, x, H, W, /*stride=*/1);
    instance_norm(x, 64, H, W);
    apply_relu(x);

    // model1: two 3x3 stride-2 downsamples (-> 128 -> 256), each IN + ReLU.
    x = conv2d_down(m.d0, x, H, W);
    instance_norm(x, 128, H, W);
    apply_relu(x);
    x = conv2d_down(m.d1, x, H, W);
    instance_norm(x, 256, H, W);
    apply_relu(x);

    // model2: three residual blocks at 256ch, H/4 x W/4 (reflect-pad 1 convs).
    for (int i = 0; i < kResBlocks; ++i) {
        const Impl::Res& r = m.res[i];
        int hh = H, ww = W;
        Tensor hpad = reflect_pad(x, 256, hh, ww, 1);
        Tensor h = conv2d(r.a, hpad, hh, ww, /*stride=*/1);  // back to H,W
        instance_norm(h, 256, hh, ww);
        apply_relu(h);
        Tensor h2pad = reflect_pad(h, 256, hh, ww, 1);
        Tensor h2 = conv2d(r.b, h2pad, hh, ww, /*stride=*/1);
        instance_norm(h2, 256, hh, ww);
        brotensor::add_inplace(x, h2);                       // residual skip
    }

    // model3: two 3x3 stride-2 conv-transpose upsamples (-> 128 -> 64), IN+ReLU.
    x = conv_transpose_up(m.u0, x, H, W);
    instance_norm(x, 128, H, W);
    apply_relu(x);
    x = conv_transpose_up(m.u1, x, H, W);
    instance_norm(x, 64, H, W);
    apply_relu(x);

    // model4: reflect-pad 3 -> 7x7 conv to 1 channel -> sigmoid.
    x = reflect_pad(x, 64, H, W, 3);
    x = conv2d(m.m4, x, H, W, /*stride=*/1);
    Tensor line;
    brotensor::sigmoid_forward(x, line);

    // Resize the line map back to the original input resolution (identity when
    // the model ran at native size and the dims are divisible by 4).
    const int procH = pp.transform.proc_h, procW = pp.transform.proc_w;
    const int origH = pp.transform.orig_h, origW = pp.transform.orig_w;
    if (H != origH || W != origW) {
        Tensor back;
        brotensor::interp2d_forward(line, /*N=*/1, /*C=*/1, H, W,
                                    origH, origW, /*mode=*/1, back);
        line = std::move(back);
        H = origH; W = origW;
    }
    (void)procH; (void)procW;

    // Pull to host.
    Tensor host = (line.device == brotensor::Device::CPU)
                      ? line
                      : line.to(brotensor::Device::CPU);
    LineMap out;
    out.width  = W;
    out.height = H;
    const float* p = host.host_f32();
    const std::size_t n = static_cast<std::size_t>(H) * W;
    out.line.resize(n);
    if (cfg_.invert)
        for (std::size_t i = 0; i < n; ++i) out.line[i] = 1.0f - p[i];
    else
        out.line.assign(p, p + n);
    return out;
}

LineMap LineartDetector::detect(const uint8_t* rgb, int w, int h,
                                int channels) const {
    std::vector<tiling::TileRect> tiles = tiling::plan_tiles(w, h, cfg_.tile);
    if (tiles.empty()) {
        PreprocessedImage pp = preprocess(rgb, w, h, channels, cfg_.detect_resolution);
        return run(pp);
    }

    // Tiled path: crop each tile, run it at native resolution, feather-blend the
    // per-tile line maps into one full-size map. (run() applies cfg_.invert per
    // tile; 1 - x is pointwise so it commutes with the weighted blend.)
    LineMap out;
    out.width  = w;
    out.height = h;
    out.line = tiling::blend_1ch(w, h, tiles, [&](const tiling::TileRect& t) {
        std::vector<uint8_t> crop(static_cast<std::size_t>(t.w) * t.h * channels);
        broimage::crop_hwc_u8(rgb, w, h, channels, crop.data(),
                              t.x, t.y, t.w, t.h);
        PreprocessedImage pp = preprocess(crop.data(), t.w, t.h, channels,
                                          /*detect_resolution=*/0);
        return run(pp).line;
    });
    return out;
}

}  // namespace brovisionml::lineart
