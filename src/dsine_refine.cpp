#include "brovisionml/dsine_refine.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include "weights_util.h"

#include <algorithm>
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

const std::string kWho = "dsine::Refiner: ";

constexpr int kPs   = 5;   // NRN_prop_ps
constexpr int kPad  = 2;   // (ps-1)/2
constexpr int kPP   = kPs * kPs;  // 25 neighbors
constexpr int kK    = 8;   // downsample_ratio
constexpr int kIter = 5;   // num_iter_test
constexpr float kRayEps = 1e-2f;
constexpr float kPi = 3.14159265358979323846f;

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

// ── conv / activation helpers ────────────────────────────────────────────────

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

// get_prediction_head forward.
Tensor run_head(const PredHead& hd, const Tensor& x, int H, int W) {
    Tensor h = conv2d(hd.c0, x, H, W, /*pad=*/1);
    apply_relu(h);
    h = conv2d(hd.c2, h, H, W, /*pad=*/0);
    apply_relu(h);
    h = conv2d(hd.c4, h, H, W, /*pad=*/0);
    return h;
}

// Channel-concat two NCHW (1, C*H*W) tensors with the same (H,W).
Tensor cat_ch(const Tensor& a, int ca, const Tensor& b, int cb, int H, int W) {
    const int HW = H * W;
    Tensor out = Tensor::mat(1, (ca + cb) * HW);
    float* o = out.host_f32_mut();
    std::memcpy(o, a.host_f32(), static_cast<std::size_t>(ca) * HW * sizeof(float));
    std::memcpy(o + static_cast<std::size_t>(ca) * HW, b.host_f32(),
                static_cast<std::size_t>(cb) * HW * sizeof(float));
    return out;
}

void apply_sigmoid_inplace(float* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        p[i] = 1.0f / (1.0f + std::exp(-p[i]));
}

// L2-normalize a (C*H*W) buffer over the channel axis (per pixel), in place.
void l2norm_channels(float* p, int C, int HW) {
    for (int i = 0; i < HW; ++i) {
        double ss = 0.0;
        for (int c = 0; c < C; ++c) {
            const double v = p[static_cast<std::size_t>(c) * HW + i];
            ss += v * v;
        }
        const double inv = 1.0 / std::max(std::sqrt(ss), 1e-12);
        for (int c = 0; c < C; ++c)
            p[static_cast<std::size_t>(c) * HW + i] *= static_cast<float>(inv);
    }
}

// ── ConvGRU ──────────────────────────────────────────────────────────────────

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

// h: (64*HW) FP32, x: (66*HW) FP32. Returns updated h (64*HW).
Tensor run_gru(const ConvGRU& g, const Tensor& h, const Tensor& x,
               int hidden, int input, int H, int W) {
    const int HW = H * W;
    // hx = cat([h, x])  -> (hidden+input)*HW
    Tensor hx = cat_ch(h, hidden, x, input, H, W);
    // z = sigmoid(convz(hx)); r = sigmoid(convr(hx))
    Tensor z = conv2d(g.convz, hx, H, W, /*pad=*/2);
    Tensor r = conv2d(g.convr, hx, H, W, /*pad=*/2);
    apply_sigmoid_inplace(z.host_f32_mut(), static_cast<std::size_t>(hidden) * HW);
    apply_sigmoid_inplace(r.host_f32_mut(), static_cast<std::size_t>(hidden) * HW);
    // rh = r * h
    Tensor rh = Tensor::mat(1, hidden * HW);
    {
        const float* rp = r.host_f32();
        const float* hp = h.host_f32();
        float* op = rh.host_f32_mut();
        for (std::size_t i = 0; i < static_cast<std::size_t>(hidden) * HW; ++i)
            op[i] = rp[i] * hp[i];
    }
    // q = tanh(convq(cat([r*h, x])))
    Tensor rhx = cat_ch(rh, hidden, x, input, H, W);
    Tensor q = conv2d(g.convq, rhx, H, W, /*pad=*/2);
    {
        float* qp = q.host_f32_mut();
        for (std::size_t i = 0; i < static_cast<std::size_t>(hidden) * HW; ++i)
            qp[i] = std::tanh(qp[i]);
    }
    // h_new = (1-z)*h + z*q
    Tensor h_new = Tensor::mat(1, hidden * HW);
    {
        const float* zp = z.host_f32();
        const float* hp = h.host_f32();
        const float* qp = q.host_f32();
        float* np = h_new.host_f32_mut();
        for (std::size_t i = 0; i < static_cast<std::size_t>(hidden) * HW; ++i)
            np[i] = (1.0f - zp[i]) * hp[i] + zp[i] * qp[i];
    }
    return h_new;
}

// ── get_unfold: replicate-pad by 2, im2col into (C, 25, H, W) ─────────────────
//
// src: (C*HW). out[c, n, y, x] = src[c, clamp(y-2+ky), clamp(x-2+kx)] with
// n = ky*5 + kx (F.unfold row-major). Replicate padding clamps to [0,dim-1].
void get_unfold(const float* src, int C, int H, int W, std::vector<float>& out) {
    const int HW = H * W;
    out.assign(static_cast<std::size_t>(C) * kPP * HW, 0.0f);
    for (int c = 0; c < C; ++c) {
        const float* sc = src + static_cast<std::size_t>(c) * HW;
        for (int ky = 0; ky < kPs; ++ky) {
            for (int kx = 0; kx < kPs; ++kx) {
                const int n = ky * kPs + kx;
                float* oc = out.data() +
                    (static_cast<std::size_t>(c) * kPP + n) * HW;
                for (int y = 0; y < H; ++y) {
                    int sy = y - kPad + ky;
                    sy = std::min(std::max(sy, 0), H - 1);
                    for (int x = 0; x < W; ++x) {
                        int sx = x - kPad + kx;
                        sx = std::min(std::max(sx, 0), W - 1);
                        oc[static_cast<std::size_t>(y) * W + x] =
                            sc[static_cast<std::size_t>(sy) * W + sx];
                    }
                }
            }
        }
    }
}

// ── axis_angle_to_matrix (PyTorch3D port) ────────────────────────────────────
//
// ax = axis*angle (3 components). Fills R (row-major 3x3).
void axis_angle_to_matrix(float ax, float ay, float az, float R[9]) {
    const double angle = std::sqrt(static_cast<double>(ax) * ax +
                                   static_cast<double>(ay) * ay +
                                   static_cast<double>(az) * az);
    const double half = angle * 0.5;
    double s;  // sin(half)/angle
    if (angle < 1e-6) {
        s = 0.5 - (angle * angle) / 48.0;
    } else {
        s = std::sin(half) / angle;
    }
    // quaternion (r, i, j, k) = (cos(half), ax*s, ay*s, az*s)
    const double r = std::cos(half);
    const double i = ax * s;
    const double j = ay * s;
    const double k = az * s;
    const double two_s = 2.0 / (r * r + i * i + j * j + k * k);
    R[0] = static_cast<float>(1.0 - two_s * (j * j + k * k));
    R[1] = static_cast<float>(two_s * (i * j - k * r));
    R[2] = static_cast<float>(two_s * (i * k + j * r));
    R[3] = static_cast<float>(two_s * (i * j + k * r));
    R[4] = static_cast<float>(1.0 - two_s * (i * i + k * k));
    R[5] = static_cast<float>(two_s * (j * k - i * r));
    R[6] = static_cast<float>(two_s * (i * k - j * r));
    R[7] = static_cast<float>(two_s * (j * k + i * r));
    R[8] = static_cast<float>(1.0 - two_s * (i * i + j * j));
}

// ── RayReLU applied to one (3*HW) normal map against ray (3*HW), in place ─────
//
// cos = cosine_similarity(n, ray) per pixel; ray is already unit, but the port
// uses cosine_similarity which divides by |n||ray| (with torch's 1e-8 floor).
void ray_relu_inplace(float* n, const float* ray, int HW) {
    for (int p = 0; p < HW; ++p) {
        const double nx = n[p], ny = n[HW + p], nz = n[2 * HW + p];
        const double rx = ray[p], ry = ray[HW + p], rz = ray[2 * HW + p];
        const double nn = std::sqrt(nx * nx + ny * ny + nz * nz);
        const double rr = std::sqrt(rx * rx + ry * ry + rz * rz);
        const double denom = std::max(nn * rr, 1e-8);  // torch cosine_sim eps
        double cos = (nx * rx + ny * ry + nz * rz) / denom;
        // norm_along_view = ray*cos ; relu version = ray*(relu(cos-eps)+eps)
        const double relu_cm = std::max(cos - kRayEps, 0.0) + kRayEps;
        // diff = ray*relu_cm - ray*cos = ray*(relu_cm - cos)
        const double dcoef = relu_cm - cos;
        double mx = nx + rx * dcoef;
        double my = ny + ry * dcoef;
        double mz = nz + rz * dcoef;
        const double inv = 1.0 / std::max(std::sqrt(mx * mx + my * my + mz * mz), 1e-12);
        n[p]          = static_cast<float>(mx * inv);
        n[HW + p]     = static_cast<float>(my * inv);
        n[2 * HW + p] = static_cast<float>(mz * inv);
    }
}

// ── upsample_via_mask (convex upsample by k=8, padding='replicate') ──────────
//
// out: (C*HW) low-res. up_mask: (9*k*k * HW) raw logits. Returns (C * (kH)*(kW)).
// up_mask layout: index = (((m*k + sy)*k + sx) ... ) — torch view (1,9,k,k,H,W),
// i.e. flattened channel = m*(k*k) + sy*k + sx. softmax over m (the 9 axis).
// up_out (replicate-pad 1, im2col 3x3): for each low-res pixel and each of the 9
// 3x3 neighbors m (ny=m/3, nx=m%3 -> sample (y-1+ny, x-1+nx)). Output pixel at
// (k*y+sy, k*x+sx) = sum_m softmax(up_mask[m,sy,sx]) * lowres[neighbor m].
std::vector<float> upsample_via_mask(const float* out, int C, int H, int W,
                                     const float* up_mask) {
    const int HW = H * W;
    const int kk = kK * kK;          // 64
    const int kH = kK * H, kW = kK * W;
    const int oHW = kH * kW;
    std::vector<float> up(static_cast<std::size_t>(C) * oHW, 0.0f);

    // Precompute softmax weights: w[m][sy*k+sx][pixel]. 9 * 64 * HW.
    // We'll iterate per low-res pixel to keep memory bounded.
    std::vector<float> sm(9);  // per-(sy,sx)-per-pixel softmax over the 9 axis
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pix = y * W + x;
            for (int sy = 0; sy < kK; ++sy) {
                for (int sx = 0; sx < kK; ++sx) {
                    const int sub = sy * kK + sx;
                    // softmax over m of up_mask[(m*kk + sub)*HW + pix]
                    float mxv = -3.4e38f;
                    for (int m = 0; m < 9; ++m) {
                        const float v = up_mask[
                            (static_cast<std::size_t>(m) * kk + sub) * HW + pix];
                        if (v > mxv) mxv = v;
                    }
                    double sum = 0.0;
                    for (int m = 0; m < 9; ++m) {
                        const float v = up_mask[
                            (static_cast<std::size_t>(m) * kk + sub) * HW + pix];
                        const double e = std::exp(static_cast<double>(v) - mxv);
                        sm[m] = static_cast<float>(e);
                        sum += e;
                    }
                    const double invs = 1.0 / sum;
                    for (int m = 0; m < 9; ++m)
                        sm[m] = static_cast<float>(sm[m] * invs);

                    const int oy = kK * y + sy;
                    const int ox = kK * x + sx;
                    const std::size_t opix = static_cast<std::size_t>(oy) * kW + ox;
                    for (int c = 0; c < C; ++c) {
                        const float* lc = out + static_cast<std::size_t>(c) * HW;
                        double acc = 0.0;
                        for (int m = 0; m < 9; ++m) {
                            const int ny = m / 3, nx = m % 3;
                            int yy = y - 1 + ny;
                            int xx = x - 1 + nx;
                            yy = std::min(std::max(yy, 0), H - 1);  // replicate
                            xx = std::min(std::max(xx, 0), W - 1);
                            acc += static_cast<double>(sm[m]) *
                                   lc[static_cast<std::size_t>(yy) * W + xx];
                        }
                        up[static_cast<std::size_t>(c) * oHW + opix] =
                            static_cast<float>(acc);
                    }
                }
            }
        }
    }
    return up;
}

}  // namespace

// ── Impl ────────────────────────────────────────────────────────────────────

struct Refiner::Impl {
    bool loaded = false;
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

brotensor::Tensor Refiner::forward(const DecoderOutput& dec,
                                   const Intrinsics& intrins, int origH,
                                   int origW, const DsineTransform& tf) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("forward() called before load()");

    const int H = dec.h8, W = dec.w8;
    const int HW = H * W;

    // ray_8 / uv_8 (build_ray8/build_uv apply the +0.5 internally).
    UvMap ray8 = build_ray8(intrins, H, W, origH, origW);
    UvMap uv8  = build_uv(intrins, H, W, origH, origW);
    const float* ray = ray8.data.host_f32();

    // Scaled intrinsics on the /8 grid, from the SAME +0.5 intrinsics build_ray8
    // uses (consistency requirement).
    const double sw = static_cast<double>(W) / origW;
    const double sh = static_cast<double>(H) / origH;
    const double fu = intrins.fx * sw;
    const double cu = (intrins.cx + 0.5) * sw;
    const double fv = intrins.fy * sh;
    const double cv = (intrins.cy + 0.5) * sh;

    // pixel_coords (x+0.5, y+0.5), 2 channels at the /8 grid; unfold to 25.
    std::vector<float> pix2(static_cast<std::size_t>(2) * HW);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            pix2[static_cast<std::size_t>(0) * HW + y * W + x] = x + 0.5f;
            pix2[static_cast<std::size_t>(1) * HW + y * W + x] = y + 0.5f;
        }
    std::vector<float> pix_unf;  // (2, 25, H, W)
    get_unfold(pix2.data(), 2, H, W, pix_unf);

    // ── Step 4 init ──────────────────────────────────────────────────────────
    // pred_norm = RayReLU(decoder.normal, ray_8)
    Tensor pred_norm = Tensor::mat(1, 3 * HW);
    std::memcpy(pred_norm.host_f32_mut(), dec.normal.host_f32(),
                static_cast<std::size_t>(3) * HW * sizeof(float));
    ray_relu_inplace(pred_norm.host_f32_mut(), ray, HW);

    // feat_map = cat([decoder.feature, uv_8])  -> 66ch
    Tensor feat_map = cat_ch(dec.feature, 64, uv8.data, 2, H, W);

    // h = decoder.hidden
    Tensor h = Tensor::mat(1, 64 * HW);
    std::memcpy(h.host_f32_mut(), dec.hidden.host_f32(),
                static_cast<std::size_t>(64) * HW * sizeof(float));

    // up0 = upsample(h, pred_norm, uv_8) — pred_list[0], not used for the result.

    std::vector<float> up_pred_norm;  // (3 * (8H)*(8W)), filled each iter

    // ── 5 refinement iterations ──────────────────────────────────────────────
    for (int it = 0; it < kIter; ++it) {
        // h_new = gru(h, feat_map)
        Tensor h_new = run_gru(m.gru, h, feat_map, m.hidden_dim, m.input_dim, H, W);

        // head input: cat([h_new, uv_8]) -> 66ch
        Tensor hin = cat_ch(h_new, m.hidden_dim, uv8.data, 2, H, W);

        // nghbr_prob = sigmoid(prob_head(...))  (25, H, W)
        Tensor prob = run_head(m.prob_head, hin, H, W);
        apply_sigmoid_inplace(prob.host_f32_mut(),
                              static_cast<std::size_t>(kPP) * HW);
        const float* probp = prob.host_f32();

        // nghbr_normals = get_unfold(pred_norm)  (3, 25, H, W)
        std::vector<float> nrm_unf;
        get_unfold(pred_norm.host_f32(), 3, H, W, nrm_unf);

        // nghbr_xys = L2normalize(xy_head(...))  -> split (25 xs, 25 ys), norm
        Tensor xy = run_head(m.xy_head, hin, H, W);  // (50, H, W): [xs(25), ys(25)]
        {
            float* xp = xy.host_f32_mut();
            // F.normalize over the 2 (x,y) axis per (neighbor, pixel).
            for (int n = 0; n < kPP; ++n) {
                float* xs = xp + static_cast<std::size_t>(n) * HW;
                float* ys = xp + static_cast<std::size_t>(kPP + n) * HW;
                for (int p = 0; p < HW; ++p) {
                    const double a = xs[p], b = ys[p];
                    const double inv = 1.0 / std::max(std::sqrt(a * a + b * b), 1e-12);
                    xs[p] = static_cast<float>(a * inv);
                    ys[p] = static_cast<float>(b * inv);
                }
            }
        }
        const float* xyp = xy.host_f32();

        // nghbr_angle = sigmoid(angle_head(...)) * pi   (25, H, W)
        Tensor ang = run_head(m.angle_head, hin, H, W);
        {
            float* ap = ang.host_f32_mut();
            for (std::size_t i = 0; i < static_cast<std::size_t>(kPP) * HW; ++i)
                ap[i] = (1.0f / (1.0f + std::exp(-ap[i]))) * kPi;
        }
        const float* angp = ang.host_f32();

        // ── per-pixel-per-neighbor rotation -> prob-weighted sum ──────────────
        // accumulate rotated+RayReLU'd neighbor normals, weighted by prob.
        std::vector<float> acc(static_cast<std::size_t>(3) * HW, 0.0f);
        // We must apply RayReLU per neighbor BEFORE the weighted sum. Process one
        // neighbor at a time across all pixels so we can reuse ray_relu_inplace.
        std::vector<float> rot(static_cast<std::size_t>(3) * HW);  // one neighbor

        for (int n = 0; n < kPP; ++n) {
            const float* xs = xyp + static_cast<std::size_t>(n) * HW;            // nghbr_xys[0]
            const float* ys = xyp + static_cast<std::size_t>(kPP + n) * HW;      // nghbr_xys[1]
            const float* pix_x = pix_unf.data() +
                (static_cast<std::size_t>(0) * kPP + n) * HW;
            const float* pix_y = pix_unf.data() +
                (static_cast<std::size_t>(1) * kPP + n) * HW;
            const float* nxs = nrm_unf.data() + (static_cast<std::size_t>(0) * kPP + n) * HW;
            const float* nys = nrm_unf.data() + (static_cast<std::size_t>(1) * kPP + n) * HW;
            const float* nzs = nrm_unf.data() + (static_cast<std::size_t>(2) * kPP + n) * HW;
            const float* angn = angp + static_cast<std::size_t>(n) * HW;

            for (int p = 0; p < HW; ++p) {
                const double du_over_fu = xs[p] / fu;
                const double dv_over_fv = ys[p] / fv;
                const double term_u = (pix_x[p] + xs[p] - cu) / fu;
                const double term_v = (pix_y[p] + ys[p] - cv) / fv;
                const double nx = nxs[p], ny = nys[p], nz = nzs[p];

                const double num = -(du_over_fu * nx + dv_over_fv * ny);
                // delta_z_denom, clamped where |.|<1e-8 to 1e-8*sign(.). torch's
                // sign(0)=0, so a true zero denom stays 0 and num/0 -> inf, which
                // the nan/inf axis guard below then zeroes. Faithful to upstream.
                double dz;
                {
                    double dd = term_u * nx + term_v * ny + nz;
                    if (std::abs(dd) < 1e-8) {
                        const double sgn = (dd > 0.0) ? 1.0 : (dd < 0.0 ? -1.0 : 0.0);
                        dd = 1e-8 * sgn;
                    }
                    dz = num / dd;
                }

                double axx = du_over_fu + dz * term_u;
                double axy = dv_over_fv + dz * term_v;
                double axz = dz;
                // normalize axis (F.normalize over the 3 axis)
                const double an = std::sqrt(axx * axx + axy * axy + axz * axz);
                const double ainv = 1.0 / std::max(an, 1e-12);
                axx *= ainv; axy *= ainv; axz *= ainv;

                // zero invalid (nan/inf) axes -> identity rotation
                float fax = static_cast<float>(axx);
                float fay = static_cast<float>(axy);
                float faz = static_cast<float>(axz);
                if (!std::isfinite(fax) || !std::isfinite(fay) || !std::isfinite(faz)) {
                    fax = fay = faz = 0.0f;
                }

                const float theta = angn[p];
                float R[9];
                axis_angle_to_matrix(fax * theta, fay * theta, faz * theta, R);

                // rotate neighbor normal: R @ [nx,ny,nz]
                const double rnx = R[0] * nx + R[1] * ny + R[2] * nz;
                const double rny = R[3] * nx + R[4] * ny + R[5] * nz;
                const double rnz = R[6] * nx + R[7] * ny + R[8] * nz;
                // normalize
                const double rinv = 1.0 /
                    std::max(std::sqrt(rnx * rnx + rny * rny + rnz * rnz), 1e-12);
                rot[p]          = static_cast<float>(rnx * rinv);
                rot[HW + p]     = static_cast<float>(rny * rinv);
                rot[2 * HW + p] = static_cast<float>(rnz * rinv);
            }

            // RayReLU on this neighbor's rotated normals
            ray_relu_inplace(rot.data(), ray, HW);

            // weighted accumulate: acc += prob[n] * rot
            const float* pw = probp + static_cast<std::size_t>(n) * HW;
            for (int p = 0; p < HW; ++p) {
                const float w = pw[p];
                acc[p]          += w * rot[p];
                acc[HW + p]     += w * rot[HW + p];
                acc[2 * HW + p] += w * rot[2 * HW + p];
            }
        }

        // pred_norm = normalize(acc)
        l2norm_channels(acc.data(), 3, HW);
        std::memcpy(pred_norm.host_f32_mut(), acc.data(),
                    static_cast<std::size_t>(3) * HW * sizeof(float));

        // up_pred_norm = normalize(upsample_via_mask(pred_norm, up_prob_head(...)))
        Tensor up_mask = run_head(m.up_prob_head, hin, H, W);  // (576, H, W)
        up_pred_norm = upsample_via_mask(pred_norm.host_f32(), 3, H, W,
                                         up_mask.host_f32());
        l2norm_channels(up_pred_norm.data(), 3, kK * H * kK * W);

        h = std::move(h_new);
    }

    // ── crop the full-res result to the original (unpadded) image ─────────────
    const int fullH = kK * H, fullW = kK * W;   // padded full-res
    const int outH = tf.orig_h, outW = tf.orig_w;
    const int l = tf.pad_l, t = tf.pad_t;

    Tensor out = Tensor::mat(1, static_cast<std::size_t>(3) * outH * outW);
    float* op = out.host_f32_mut();
    for (int c = 0; c < 3; ++c) {
        const float* src = up_pred_norm.data() +
            static_cast<std::size_t>(c) * fullH * fullW;
        float* dst = op + static_cast<std::size_t>(c) * outH * outW;
        for (int y = 0; y < outH; ++y) {
            const float* srow = src + static_cast<std::size_t>(t + y) * fullW + l;
            std::memcpy(dst + static_cast<std::size_t>(y) * outW, srow,
                        static_cast<std::size_t>(outW) * sizeof(float));
        }
    }
    return out;
}

}  // namespace brovisionml::dsine
