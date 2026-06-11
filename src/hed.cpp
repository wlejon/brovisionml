#include "brovisionml/hed.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include "broimage/geometric.h"

#include "weights_util.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brovisionml::hed {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "hed::SoftEdgeDetector: ";

constexpr int kBlocks = 5;
// Conv counts per block (block3-5 are triple-conv); from the VGG-style trunk.
constexpr int kConvCount[kBlocks] = {2, 2, 3, 3, 3};

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

// One VGG block: N 3x3 convs (ReLU between) + a 1x1 projection to 1 channel.
struct Block {
    std::vector<Conv> convs;
    Conv projection;
    int in_ch = 0;   // == convs[0].c_in (channels seen by the optional max-pool)
};

Block load_block(const st::File& f, const std::string& prefix, int n_convs) {
    Block b;
    for (int i = 0; i < n_convs; ++i)
        b.convs.push_back(load_conv(f, prefix + ".convs." + std::to_string(i)));
    b.projection = load_conv(f, prefix + ".projection");
    b.in_ch = b.convs.front().c_in;
    return b;
}

// max_pool2d output dim for kernel 2, stride 2, no pad: floor((n-2)/2)+1.
int pool_dim(int n) { return (n - 2) / 2 + 1; }

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

// Run one block: optionally max-pool 2x2 first, then the conv stack (ReLU
// between every conv), then the 1x1 projection. Updates the running trunk `x`
// and its spatial dims (H,W) in place; returns the 1-channel side map at (H,W).
Tensor run_block(const Block& b, Tensor& x, int& H, int& W, bool down) {
    if (down) {
        Tensor pooled, idx;
        const int Ho = pool_dim(H), Wo = pool_dim(W);
        brotensor::max_pool2d_forward(x, /*N=*/1, b.in_ch, H, W,
                                      /*kH=*/2, /*kW=*/2, /*sh=*/2, /*sw=*/2,
                                      /*pad_h=*/0, /*pad_w=*/0, pooled, idx);
        x = std::move(pooled);
        H = Ho; W = Wo;
    }
    for (const Conv& c : b.convs) {
        x = conv2d(c, x, H, W, /*pad=*/1);
        apply_relu(x);
    }
    return conv2d(b.projection, x, H, W, /*pad=*/0);
}

}  // namespace

// ── Impl ────────────────────────────────────────────────────────────────────

struct SoftEdgeDetector::Impl {
    bool loaded = false;
    bool fp16 = false;
    brotensor::Device device = brotensor::Device::CPU;
    Block blocks[kBlocks];
};

SoftEdgeDetector::SoftEdgeDetector(HedConfig cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>()) {}
SoftEdgeDetector::~SoftEdgeDetector() = default;
SoftEdgeDetector::SoftEdgeDetector(SoftEdgeDetector&&) noexcept = default;
SoftEdgeDetector& SoftEdgeDetector::operator=(SoftEdgeDetector&&) noexcept = default;

void SoftEdgeDetector::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void SoftEdgeDetector::load_file(const std::string& path) {
    st::File f = st::File::open(path);
    Impl m;
    for (int i = 0; i < kBlocks; ++i)
        m.blocks[i] = load_block(f, "block" + std::to_string(i + 1),
                                 kConvCount[i]);

    // Fold the learned per-channel `norm` bias (subtracted from the [0,255]
    // input before block1) into block1.convs.0's bias. For input (x - norm),
    // conv0 output o becomes W*x + (b[o] - sum_{c,k} W[o,c,k]*norm[c]); precompute
    // the corrected bias so the forward pass never has to subtract norm.
    Tensor norm = load_whole(f, kWho, "norm", 3, 1);   // (1,3,1,1) -> 3 elems
    const float* nrm = norm.host_f32();
    Conv& c0 = m.blocks[0].convs.front();
    if (c0.c_in != 3) fail("block1.convs.0 expects 3 input channels");
    const int per_ch = c0.k * c0.k;             // 9 for a 3x3 kernel
    const int per_out = c0.c_in * per_ch;       // 27
    const float* w = c0.weight.host_f32();
    float* bias = c0.bias.host_f32_mut();
    for (int o = 0; o < c0.c_out; ++o) {
        double delta = 0.0;
        for (int c = 0; c < c0.c_in; ++c) {
            double wsum = 0.0;
            const float* wk = w + static_cast<std::size_t>(o) * per_out + c * per_ch;
            for (int kk = 0; kk < per_ch; ++kk) wsum += wk[kk];
            delta += static_cast<double>(nrm[c]) * wsum;
        }
        bias[o] = static_cast<float>(bias[o] - delta);
    }

    m.loaded = true;
    *impl_ = std::move(m);
}

void SoftEdgeDetector::to(brotensor::Device dev) {
    Impl& m = *impl_;
    if (!m.loaded) fail("to() called before load()");
    if (dev == m.device) return;
    // Mixed precision on a GPU backend: every conv in the trunk is a WMMA-
    // covered shape (3x3 s1 p1 / 1x1 s1 p0), so the whole forward — convs,
    // pools, side-map fusion, sigmoid — runs FP16 and only the final edge map
    // is widened for download. CPU stays all-FP32; to(CPU) widens back.
    const bool fp16 = dev != brotensor::Device::CPU &&
                      brotensor::compute_dtype() == brotensor::Dtype::FP16;
    const brotensor::Dtype want = fp16 ? brotensor::Dtype::FP16
                                       : brotensor::Dtype::FP32;
    auto mv = [&](Tensor& t) {
        if (!t.data) return;
        t = t.to(dev);
        if (t.dtype != want) { Tensor c; brotensor::cast(t, c, want); t = std::move(c); }
    };
    auto mv_conv = [&](Conv& c) { mv(c.weight); mv(c.bias); };
    for (Block& b : m.blocks) {
        for (Conv& c : b.convs) mv_conv(c);
        mv_conv(b.projection);
    }
    m.device = dev;
    m.fp16 = fp16;
}

brotensor::Device SoftEdgeDetector::device() const { return impl_->device; }

// ── Run ─────────────────────────────────────────────────────────────────────

EdgeMap SoftEdgeDetector::run(const PreprocessedImage& pp) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("detect() called before load()");

    const int procH = pp.transform.proc_h;
    const int procW = pp.transform.proc_w;

    // Upload the host-preprocessed pixels to the active device; under mixed
    // precision the whole trunk runs FP16, so narrow the input once here.
    Tensor x = (m.device == brotensor::Device::CPU) ? pp.pixels
                                                     : pp.pixels.to(m.device);
    if (m.fp16) {
        Tensor h;
        brotensor::cast(x, h, brotensor::Dtype::FP16);
        x = std::move(h);
    }

    // Run the 5-block trunk; block1 keeps the input resolution, blocks 2-5
    // max-pool 2x2 first. Each block emits a 1-channel side map at its own scale.
    int H = procH, W = procW;
    std::vector<Tensor> sides;
    std::vector<int> sideH, sideW;
    sides.reserve(kBlocks);
    for (int i = 0; i < kBlocks; ++i) {
        Tensor side = run_block(m.blocks[i], x, H, W, /*down=*/i != 0);
        sides.push_back(std::move(side));
        sideH.push_back(H);
        sideW.push_back(W);
    }

    // Resize each side map to the working resolution (bilinear) and accumulate
    // the mean, then sigmoid -> edge map at (procH, procW).
    Tensor acc;
    for (int i = 0; i < kBlocks; ++i) {
        Tensor up;
        brotensor::interp2d_forward(sides[i], /*N=*/1, /*C=*/1,
                                    sideH[i], sideW[i], procH, procW,
                                    /*mode=*/1, up);
        if (i == 0) acc = std::move(up);
        else        brotensor::add_inplace(acc, up);
    }
    brotensor::scale_inplace(acc, 1.0f / static_cast<float>(kBlocks));
    Tensor edge;
    brotensor::sigmoid_forward(acc, edge);

    // Resize the edge map back to the original input resolution (identity when
    // the model ran at native size).
    const int origH = pp.transform.orig_h, origW = pp.transform.orig_w;
    if (origH != procH || origW != procW) {
        Tensor back;
        brotensor::interp2d_forward(edge, /*N=*/1, /*C=*/1, procH, procW,
                                    origH, origW, /*mode=*/1, back);
        edge = std::move(back);
    }

    // Widen the final edge map back to FP32 and pull to the host.
    if (edge.dtype != brotensor::Dtype::FP32) {
        Tensor f;
        brotensor::cast(edge, f, brotensor::Dtype::FP32);
        edge = std::move(f);
    }
    Tensor host = (edge.device == brotensor::Device::CPU)
                      ? edge
                      : edge.to(brotensor::Device::CPU);
    EdgeMap out;
    out.width  = origW;
    out.height = origH;
    const float* e = host.host_f32();
    out.edge.assign(e, e + static_cast<std::size_t>(origH) * origW);
    return out;
}

EdgeMap SoftEdgeDetector::detect(const uint8_t* rgb, int w, int h,
                                 int channels) const {
    std::vector<tiling::TileRect> tiles = tiling::plan_tiles(w, h, cfg_.tile);
    if (tiles.empty()) {
        PreprocessedImage pp = preprocess(rgb, w, h, channels, cfg_.detect_resolution);
        return run(pp);
    }

    // Tiled path: crop each tile, run it at native resolution, feather-blend the
    // per-tile edge maps into one full-size map. Each tile pass is bounded by the
    // tile size, so a huge image never allocates a full-frame activation.
    EdgeMap out;
    out.width  = w;
    out.height = h;
    out.edge = tiling::blend_1ch(w, h, tiles, [&](const tiling::TileRect& t) {
        std::vector<uint8_t> crop(static_cast<std::size_t>(t.w) * t.h * channels);
        broimage::crop_hwc_u8(rgb, w, h, channels, crop.data(),
                              t.x, t.y, t.w, t.h);
        PreprocessedImage pp = preprocess(crop.data(), t.w, t.h, channels,
                                          /*detect_resolution=*/0);
        return run(pp).edge;
    });
    return out;
}

}  // namespace brovisionml::hed
