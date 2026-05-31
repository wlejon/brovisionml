#include "brovisionml/dpt_head.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include "weights_util.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brovisionml::dpt {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "dpt::DepthHead: ";

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

enum class ResizeKind { Identity, Up, Down };

// A reassembled feature plane with its spatial dims tracked alongside.
struct Plane {
    Tensor t;
    int C = 0, H = 0, W = 0;
};

// y = ReLU(x) elementwise (fresh tensor).
Tensor relu(const Tensor& x) {
    Tensor y;
    brotensor::relu_forward(x, y);
    return y;
}

// 1x1 conv (with bias): Cin -> Cout, spatial preserved.
Tensor conv1x1(const Tensor& x, const Tensor& w, const Tensor& b,
               int Cin, int Cout, int H, int W) {
    Tensor y;
    brotensor::conv2d_forward(x, w, &b, 1, Cin, H, W, Cout, 1, 1, 1, 1, 0, 0, 1, 1, y);
    return y;
}

// 3x3 conv, pad 1, stride 1 (optional bias): Cin -> Cout, spatial preserved.
Tensor conv3x3(const Tensor& x, const Tensor& w, const Tensor* b,
               int Cin, int Cout, int H, int W) {
    Tensor y;
    brotensor::conv2d_forward(x, w, b, 1, Cin, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1, y);
    return y;
}

}  // namespace

// ─── Config presets ─────────────────────────────────────────────────────────

HeadConfig HeadConfig::vit_s() { return HeadConfig{}; }  // defaults are Small

HeadConfig HeadConfig::vit_b() {
    HeadConfig c;
    c.reassemble_hidden_size = 768;
    c.neck_hidden_sizes = {96, 192, 384, 768};
    c.fusion_hidden_size = 128;
    return c;
}

HeadConfig HeadConfig::vit_l() {
    HeadConfig c;
    c.reassemble_hidden_size = 1024;
    c.neck_hidden_sizes = {256, 512, 1024, 1024};
    c.fusion_hidden_size = 256;
    return c;
}

// ─── Weight tables (host FP32) ────────────────────────────────────────────────

namespace {

struct ReassembleLayer {
    Tensor proj_w, proj_b;   // (c, D) / (c,1)   1x1
    ResizeKind kind = ResizeKind::Identity;
    Tensor resize_w, resize_b;  // Up: convT (c, c*k*k); Down: conv (c, c*9)
    int k = 0, stride = 0;      // Up: k==stride==factor; Down: k=3, stride=1/factor
    int channels = 0;
};

struct PreAct {
    Tensor c1_w, c1_b, c2_w, c2_b;   // (f, f*9) / (f,1)
};

struct FusionLayer {
    Tensor proj_w, proj_b;   // (f, f) / (f,1)  1x1
    PreAct rl1, rl2;
    bool has_rl1 = true;     // false for the deepest (first) layer
};

}  // namespace

struct DepthHead::Weights {
    bool loaded = false;
    std::vector<ReassembleLayer> reassemble;   // one per stage
    std::vector<Tensor>          neck_convs;    // (f, c_i*9), bias-free
    std::vector<FusionLayer>     fusion;        // one per stage
    Tensor h1_w, h1_b;   // conv1 (f/2, f*9) / (f/2,1)
    Tensor h2_w, h2_b;   // conv2 (head_hidden, (f/2)*9) / (head_hidden,1)
    Tensor h3_w, h3_b;   // conv3 (1, head_hidden) / (1,1)  1x1
};

// ─── Construction / loading ─────────────────────────────────────────────────

DepthHead::DepthHead(HeadConfig cfg)
    : cfg_(std::move(cfg)), w_(std::make_unique<Weights>()) {
    if (cfg_.neck_hidden_sizes.size() != cfg_.reassemble_factors.size())
        fail("neck_hidden_sizes and reassemble_factors must have equal length");
}

DepthHead::~DepthHead() = default;
DepthHead::DepthHead(DepthHead&&) noexcept = default;
DepthHead& DepthHead::operator=(DepthHead&&) noexcept = default;

void DepthHead::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void DepthHead::load_file(const std::string& path) {
    st::File f = st::File::open(path);

    const int D  = cfg_.reassemble_hidden_size;
    const int F  = cfg_.fusion_hidden_size;
    const int nstages = static_cast<int>(cfg_.neck_hidden_sizes.size());
    Weights w;

    // Reassemble stage.
    w.reassemble.resize(nstages);
    for (int i = 0; i < nstages; ++i) {
        ReassembleLayer& r = w.reassemble[i];
        const int c = cfg_.neck_hidden_sizes[i];
        r.channels = c;
        const std::string lp =
            "neck.reassemble_stage.layers." + std::to_string(i) + ".";
        r.proj_w = load_whole(f, kWho, lp + "projection.weight", c, D);
        r.proj_b = load_whole(f, kWho, lp + "projection.bias",   c, 1);

        const double factor = cfg_.reassemble_factors[i];
        if (factor > 1.0) {
            r.kind = ResizeKind::Up;
            r.k = r.stride = static_cast<int>(std::lround(factor));
            // ConvTranspose2d(c, c, k, stride): weight (c, c*k*k).
            r.resize_w = load_whole(f, kWho, lp + "resize.weight", c, c * r.k * r.k);
            r.resize_b = load_whole(f, kWho, lp + "resize.bias",   c, 1);
        } else if (factor < 1.0) {
            r.kind = ResizeKind::Down;
            r.k = 3;
            r.stride = static_cast<int>(std::lround(1.0 / factor));
            // Conv2d(c, c, 3, stride, pad 1): weight (c, c*9).
            r.resize_w = load_whole(f, kWho, lp + "resize.weight", c, c * 9);
            r.resize_b = load_whole(f, kWho, lp + "resize.bias",   c, 1);
        } else {
            r.kind = ResizeKind::Identity;   // no resize params
        }
    }

    // Neck convs (3x3, bias-free): c_i -> F.
    w.neck_convs.resize(nstages);
    for (int i = 0; i < nstages; ++i) {
        const int c = cfg_.neck_hidden_sizes[i];
        w.neck_convs[i] = load_whole(f, kWho, "neck.convs." + std::to_string(i) + ".weight",
                                     F, c * 9);
    }

    // Fusion stage.
    auto load_preact = [&](const std::string& lp) {
        PreAct p;
        p.c1_w = load_whole(f, kWho, lp + "convolution1.weight", F, F * 9);
        p.c1_b = load_whole(f, kWho, lp + "convolution1.bias",   F, 1);
        p.c2_w = load_whole(f, kWho, lp + "convolution2.weight", F, F * 9);
        p.c2_b = load_whole(f, kWho, lp + "convolution2.bias",   F, 1);
        return p;
    };
    w.fusion.resize(nstages);
    for (int j = 0; j < nstages; ++j) {
        FusionLayer& fl = w.fusion[j];
        const std::string lp =
            "neck.fusion_stage.layers." + std::to_string(j) + ".";
        fl.proj_w = load_whole(f, kWho, lp + "projection.weight", F, F);
        fl.proj_b = load_whole(f, kWho, lp + "projection.bias",   F, 1);
        // The deepest (first-applied) layer runs with no skip, so its
        // residual_layer1 is never used; the others apply it to the skip.
        fl.has_rl1 = (j != 0);
        if (fl.has_rl1) fl.rl1 = load_preact(lp + "residual_layer1.");
        fl.rl2 = load_preact(lp + "residual_layer2.");
    }

    // Head.
    const int Fh = F / 2;
    const int H  = cfg_.head_hidden_size;
    w.h1_w = load_whole(f, kWho, "head.conv1.weight", Fh, F * 9);
    w.h1_b = load_whole(f, kWho, "head.conv1.bias",   Fh, 1);
    w.h2_w = load_whole(f, kWho, "head.conv2.weight", H, Fh * 9);
    w.h2_b = load_whole(f, kWho, "head.conv2.bias",   H, 1);
    w.h3_w = load_whole(f, kWho, "head.conv3.weight", 1, H);
    w.h3_b = load_whole(f, kWho, "head.conv3.bias",   1, 1);

    w.loaded = true;
    *w_ = std::move(w);
}

// ─── Migration ────────────────────────────────────────────────────────────────

void DepthHead::to(brotensor::Device dev) {
    if (!w_->loaded) fail("to() called before load()");
    if (dev == device_) return;
    auto mv = [dev](Tensor& t) { if (t.data) t = t.to(dev); };
    for (ReassembleLayer& r : w_->reassemble) {
        mv(r.proj_w); mv(r.proj_b); mv(r.resize_w); mv(r.resize_b);
    }
    for (Tensor& t : w_->neck_convs) mv(t);
    auto mv_preact = [&](PreAct& p) { mv(p.c1_w); mv(p.c1_b); mv(p.c2_w); mv(p.c2_b); };
    for (FusionLayer& fl : w_->fusion) {
        mv(fl.proj_w); mv(fl.proj_b);
        if (fl.has_rl1) mv_preact(fl.rl1);
        mv_preact(fl.rl2);
    }
    mv(w_->h1_w); mv(w_->h1_b); mv(w_->h2_w); mv(w_->h2_b); mv(w_->h3_w); mv(w_->h3_b);
    device_ = dev;
}

// ─── Forward ──────────────────────────────────────────────────────────────────

namespace {

// Pre-activation residual unit: x + conv2(relu(conv1(relu(x)))).
Tensor preact_residual(const PreAct& p, const Tensor& x, int F, int H, int W) {
    Tensor a  = relu(x);
    Tensor c1 = conv3x3(a, p.c1_w, &p.c1_b, F, F, H, W);
    Tensor a2 = relu(c1);
    Tensor c2 = conv3x3(a2, p.c2_w, &p.c2_b, F, F, H, W);
    brotensor::add_inplace(c2, x);
    return c2;
}

}  // namespace

brotensor::Tensor DepthHead::forward(const std::vector<brotensor::Tensor>& feature_maps,
                                     int gh, int gw) const {
    if (!w_->loaded) fail("forward() called before load()");
    const int nstages = static_cast<int>(cfg_.neck_hidden_sizes.size());
    if (static_cast<int>(feature_maps.size()) != nstages)
        fail("expected " + std::to_string(nstages) + " feature maps");

    const int D = cfg_.reassemble_hidden_size;
    const int F = cfg_.fusion_hidden_size;

    // 1. Reassemble + neck conv -> one Plane per stage at the shared width F.
    std::vector<Plane> feats(nstages);
    for (int i = 0; i < nstages; ++i) {
        const Tensor& fm = feature_maps[i];
        if (fm.device != device_) fail("feature map must be on the head's device");
        if (fm.rows != 1 + gh * gw || fm.cols != D)
            fail("feature map shape mismatch");
        const ReassembleLayer& r = w_->reassemble[i];
        const int c = r.channels;

        // Drop the cls token: view rows [1..] of the (1+gh*gw, D) map, then make
        // it NCHW. The view shares the feature map's storage (no copy).
        Tensor patches = Tensor::view(
            device_, static_cast<char*>(fm.data) + sizeof(float) * D,
            gh * gw, D);
        Tensor nchw;
        brotensor::sequence_to_nchw(patches, 1, D, gh, gw, nchw);

        Tensor proj = conv1x1(nchw, r.proj_w, r.proj_b, D, c, gh, gw);

        Tensor resized;
        int H = gh, W = gw;
        if (r.kind == ResizeKind::Identity) {
            resized = std::move(proj);
        } else if (r.kind == ResizeKind::Up) {
            brotensor::conv_transpose2d_forward(
                proj, r.resize_w, &r.resize_b, 1, c, gh, gw, c, r.k, r.k,
                r.stride, r.stride, /*pad=*/0, 0, /*outpad=*/0, 0, /*dil=*/1, 1,
                /*groups=*/1, resized);
            H = gh * r.stride;
            W = gw * r.stride;
        } else {  // Down: conv 3x3 stride s pad 1
            brotensor::conv2d_forward(
                proj, r.resize_w, &r.resize_b, 1, c, gh, gw, c, 3, 3,
                r.stride, r.stride, 1, 1, 1, 1, resized);
            H = (gh - 1) / r.stride + 1;
            W = (gw - 1) / r.stride + 1;
        }

        // Neck conv (3x3, bias-free): c -> F, spatial preserved.
        Plane pl;
        pl.t = conv3x3(resized, w_->neck_convs[i], /*bias=*/nullptr, c, F, H, W);
        pl.C = F; pl.H = H; pl.W = W;
        feats[i] = std::move(pl);
    }

    // 2. RefineNet fusion — deepest stage first (reversed order).
    Plane fused;
    for (int idx = 0; idx < nstages; ++idx) {
        const FusionLayer& fl = w_->fusion[idx];
        const Plane& cur = feats[nstages - 1 - idx];   // reversed[idx]

        Tensor hidden;
        int H, W;
        if (idx == 0) {
            // Deepest layer: no skip.
            H = cur.H; W = cur.W;
            hidden = preact_residual(fl.rl2, cur.t, F, H, W);
        } else {
            // hidden = fused; skip = cur (interpolated to fused's size if needed).
            H = fused.H; W = fused.W;
            Tensor skip = cur.t;
            if (cur.H != H || cur.W != W) {
                Tensor s2;
                brotensor::interp2d_forward(cur.t, 1, F, cur.H, cur.W, H, W,
                                            /*bilinear=*/1, s2);
                skip = std::move(s2);
            }
            Tensor r1 = preact_residual(fl.rl1, skip, F, H, W);
            brotensor::add_inplace(fused.t, r1);
            hidden = preact_residual(fl.rl2, fused.t, F, H, W);
        }

        // Upsample: to the next-shallower stage's size, or x2 for the last layer.
        int oh, ow;
        if (idx + 1 < nstages) {
            const Plane& nxt = feats[nstages - 2 - idx];   // reversed[idx+1]
            oh = nxt.H; ow = nxt.W;
        } else {
            oh = 2 * H; ow = 2 * W;
        }
        Tensor up;
        brotensor::interp2d_align_corners_forward(hidden, 1, F, H, W, oh, ow,
                                                  /*bilinear=*/1, up);
        fused.t = conv1x1(up, fl.proj_w, fl.proj_b, F, F, oh, ow);
        fused.C = F; fused.H = oh; fused.W = ow;
    }

    // 3. Depth head: conv -> align-corners upsample to (gh,gw)*patch -> conv ->
    //    ReLU -> 1x1 conv -> ReLU.
    const int Fh = F / 2;
    const int Hh = cfg_.head_hidden_size;
    const int oh = gh * cfg_.patch_size;
    const int ow = gw * cfg_.patch_size;

    Tensor c1 = conv3x3(fused.t, w_->h1_w, &w_->h1_b, F, Fh, fused.H, fused.W);
    Tensor up;
    brotensor::interp2d_align_corners_forward(c1, 1, Fh, fused.H, fused.W, oh, ow,
                                              /*bilinear=*/1, up);
    Tensor c2 = conv3x3(up, w_->h2_w, &w_->h2_b, Fh, Hh, oh, ow);
    Tensor a  = relu(c2);
    Tensor c3 = conv1x1(a, w_->h3_w, w_->h3_b, Hh, 1, oh, ow);
    Tensor depth = relu(c3);   // activation2 (relative depth); max_depth == 1
    return depth;              // (1, oh*ow)
}

}  // namespace brovisionml::dpt
