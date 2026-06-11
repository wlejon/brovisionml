#include "brovisionml/openpose.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include "broimage/geometric.h"

#include "weights_util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brovisionml::openpose {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "openpose::OpenposeDetector: ";

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

// A plain conv: weight (C_out, C_in*k*k) OIHW + bias (C_out,1), k known.
struct Conv {
    Tensor weight, bias;
    int c_in = 0, c_out = 0, k = 1;
};

Conv load_conv(const st::File& f, const std::string& prefix) {
    const st::TensorView* v = f.find(prefix + ".weight");
    if (!v) fail("missing tensor '" + prefix + ".weight'");
    if (v->shape.size() != 4) fail("conv '" + prefix + "' weight not 4D");
    Conv c;
    c.c_out = static_cast<int>(v->shape[0]);
    c.c_in  = static_cast<int>(v->shape[1]);
    c.k     = static_cast<int>(v->shape[2]);
    c.weight = load_whole(f, kWho, prefix + ".weight",
                          c.c_out, c.c_in * c.k * c.k);
    c.bias   = load_whole(f, kWho, prefix + ".bias", c.c_out, 1);
    return c;
}

Tensor convf(const Conv& c, const Tensor& x, int H, int W, int pad) {
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

// A two-branch refinement stage: L1 (PAF, 38ch out) + L2 (heatmap, 19ch out).
// Each is a sequence of convs; all but the last get a ReLU; the last (Mconv7 /
// conv5_5) has no ReLU. Stored with per-conv pad so stage-1 (3x3 p1 + 1x1 p0)
// and stages 2..6 (7x7 p3 + 1x1 p0) share one runner.
struct Branch {
    std::vector<Conv> convs;
    std::vector<int>  pads;
};

Tensor run_branch(const Branch& br, const Tensor& in, int H, int W) {
    Tensor x = in;
    const std::size_t n = br.convs.size();
    for (std::size_t i = 0; i < n; ++i) {
        x = convf(br.convs[i], x, H, W, br.pads[i]);
        if (i + 1 < n) apply_relu(x);   // last conv has no ReLU
    }
    return x;
}

// ── separable Gaussian blur matching scipy.ndimage.gaussian_filter ──────────
// mode='reflect' (d c b a | a b c d | d c b a), truncate=4.0, sigma=3 ->
// radius = int(truncate*sigma + 0.5) = 13. Kernel exp(-0.5*(i/sigma)^2),
// normalized.
std::vector<float> gaussian_blur(const std::vector<float>& src, int H, int W,
                                 float sigma) {
    const int radius = static_cast<int>(4.0f * sigma + 0.5f);
    std::vector<float> kern(2 * radius + 1);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double v = std::exp(-0.5 * (static_cast<double>(i) / sigma) *
                                  (static_cast<double>(i) / sigma));
        kern[i + radius] = static_cast<float>(v);
        sum += v;
    }
    for (float& v : kern) v = static_cast<float>(v / sum);

    auto reflect = [](int p, int n) {
        // scipy 'reflect': (d c b a | a b c d | d c b a) — mirror without
        // repeating the edge sample.
        if (n == 1) return 0;
        while (p < 0 || p >= n) {
            if (p < 0) p = -p - 1;
            if (p >= n) p = 2 * n - p - 1;
        }
        return p;
    };

    std::vector<float> tmp(static_cast<std::size_t>(H) * W);
    // Horizontal pass.
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double acc = 0.0;
            for (int i = -radius; i <= radius; ++i) {
                const int xx = reflect(x + i, W);
                acc += static_cast<double>(kern[i + radius]) *
                       src[static_cast<std::size_t>(y) * W + xx];
            }
            tmp[static_cast<std::size_t>(y) * W + x] = static_cast<float>(acc);
        }
    }
    // Vertical pass.
    std::vector<float> out(static_cast<std::size_t>(H) * W);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double acc = 0.0;
            for (int i = -radius; i <= radius; ++i) {
                const int yy = reflect(y + i, H);
                acc += static_cast<double>(kern[i + radius]) *
                       tmp[static_cast<std::size_t>(yy) * W + x];
            }
            out[static_cast<std::size_t>(y) * W + x] = static_cast<float>(acc);
        }
    }
    return out;
}

// Resize an HWC float map (Co channels). smart_resize / smart_resize_k filter
// rule: Area when (Ht+Wt)/(Ho+Wo) < 1 else Lanczos. broimage has no Lanczos4;
// Lanczos3 is the accepted approximation (the raw-map parity gate isolates the
// net from this host resize).
std::vector<float> resize_hwc(const std::vector<float>& src, int Ho, int Wo,
                              int Co, int Ht, int Wt) {
    std::vector<float> out(static_cast<std::size_t>(Ht) * Wt * Co);
    const double k = static_cast<double>(Ht + Wt) / static_cast<double>(Ho + Wo);
    const broimage::Filter filt =
        (k < 1.0) ? broimage::Filter::Area : broimage::Filter::Lanczos3;
    broimage::resize_hwc_f32(src.data(), Wo, Ho, Co, out.data(), Wt, Ht, filt);
    return out;
}

// ── decode constants (1-indexed parts, verbatim from body.py) ───────────────
constexpr int kLimbSeq[19][2] = {
    {2, 3}, {2, 6}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {2, 9}, {9, 10},
    {10, 11}, {2, 12}, {12, 13}, {13, 14}, {2, 1}, {1, 15}, {15, 17},
    {1, 16}, {16, 18}, {3, 17}, {6, 18}};
constexpr int kMapIdx[19][2] = {
    {31, 32}, {39, 40}, {33, 34}, {35, 36}, {41, 42}, {43, 44}, {19, 20},
    {21, 22}, {23, 24}, {25, 26}, {27, 28}, {29, 30}, {47, 48}, {49, 50},
    {53, 54}, {51, 52}, {55, 56}, {37, 38}, {45, 46}};

struct Peak {
    int x, y;       // col, row in detect-res pixels
    float score;    // unblurred map_ori value
    int id;         // global peak id
};

struct Connection {
    int idA, idB;       // global candidate ids
    float score;        // score_with_dist_prior
    int i, j;           // local indices into candA / candB
};

}  // namespace

// ── Impl ────────────────────────────────────────────────────────────────────

struct OpenposeDetector::Impl {
    bool loaded = false;
    bool fp16 = false;
    brotensor::Device device = brotensor::Device::CPU;

    std::vector<Conv> model0;    // VGG trunk convs (in order); maxpools implicit
    std::vector<int>  model0_pad;
    std::vector<bool> model0_pool;  // true => 2x2 maxpool BEFORE this conv

    Branch s1_l1, s1_l2;            // stage 1
    Branch sN_l1[5], sN_l2[5];      // stages 2..6 (index 0..4)
};

OpenposeDetector::OpenposeDetector(OpenposeConfig cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>()) {}
OpenposeDetector::~OpenposeDetector() = default;
OpenposeDetector::OpenposeDetector(OpenposeDetector&&) noexcept = default;
OpenposeDetector& OpenposeDetector::operator=(OpenposeDetector&&) noexcept = default;

void OpenposeDetector::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void OpenposeDetector::load_file(const std::string& path) {
    st::File f = st::File::open(path);
    Impl m;

    // model0: VGG trunk. (conv_name, pad, pool_before).
    struct M0 { const char* name; int pad; bool pool; };
    const M0 m0[] = {
        {"conv1_1", 1, false}, {"conv1_2", 1, false},
        {"conv2_1", 1, true},  {"conv2_2", 1, false},   // pool1 before conv2_1
        {"conv3_1", 1, true},  {"conv3_2", 1, false},   // pool2 before conv3_1
        {"conv3_3", 1, false}, {"conv3_4", 1, false},
        {"conv4_1", 1, true},  {"conv4_2", 1, false},   // pool3 before conv4_1
        {"conv4_3_CPM", 1, false}, {"conv4_4_CPM", 1, false},
    };
    for (const M0& e : m0) {
        m.model0.push_back(load_conv(f, std::string("model0.") + e.name));
        m.model0_pad.push_back(e.pad);
        m.model0_pool.push_back(e.pool);
    }

    // Stage 1 branches: conv5_1..5_3 (3x3 p1) + conv5_4 (1x1 p0) + conv5_5 (1x1 p0).
    auto load_stage1 = [&](const char* mod, const char* suf, Branch& br) {
        const char* names3[] = {"conv5_1_CPM_", "conv5_2_CPM_", "conv5_3_CPM_"};
        for (const char* n : names3) {
            br.convs.push_back(load_conv(f, std::string(mod) + "." + n + suf));
            br.pads.push_back(1);
        }
        br.convs.push_back(load_conv(f, std::string(mod) + ".conv5_4_CPM_" + suf));
        br.pads.push_back(0);
        br.convs.push_back(load_conv(f, std::string(mod) + ".conv5_5_CPM_" + suf));
        br.pads.push_back(0);
    };
    load_stage1("model1_1", "L1", m.s1_l1);
    load_stage1("model1_2", "L2", m.s1_l2);

    // Stages 2..6: Mconv1..5 (7x7 p3) + Mconv6 (1x1 p0) + Mconv7 (1x1 p0).
    auto load_stageN = [&](int stage, const char* suf, Branch& br) {
        const std::string mod = "model" + std::to_string(stage) +
                                (std::string("_") + (suf[1] == '1' ? "1" : "2"));
        for (int c = 1; c <= 5; ++c) {
            br.convs.push_back(load_conv(
                f, mod + ".Mconv" + std::to_string(c) + "_stage" +
                       std::to_string(stage) + "_" + suf));
            br.pads.push_back(3);
        }
        br.convs.push_back(load_conv(
            f, mod + ".Mconv6_stage" + std::to_string(stage) + "_" + suf));
        br.pads.push_back(0);
        br.convs.push_back(load_conv(
            f, mod + ".Mconv7_stage" + std::to_string(stage) + "_" + suf));
        br.pads.push_back(0);
    };
    for (int s = 2; s <= 6; ++s) {
        load_stageN(s, "L1", m.sN_l1[s - 2]);
        load_stageN(s, "L2", m.sN_l2[s - 2]);
    }

    m.loaded = true;
    *impl_ = std::move(m);
}

void OpenposeDetector::to(brotensor::Device dev) {
    Impl& m = *impl_;
    if (!m.loaded) fail("to() called before load()");
    if (dev == m.device) return;
    // Mixed precision on a GPU backend: every conv is a WMMA-covered shape
    // (3x3 p1 VGG trunk + stage 1, 7x7 p3 refinement stages, 1x1 projections)
    // and max_pool / concat / relu all have FP16 paths, so the whole network
    // runs FP16 and only the final l1/l2 maps are widened before the host
    // postprocess. CPU stays all-FP32; to(CPU) widens back.
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
    auto mv_br   = [&](Branch& b) { for (Conv& c : b.convs) mv_conv(c); };
    for (Conv& c : m.model0) mv_conv(c);
    mv_br(m.s1_l1); mv_br(m.s1_l2);
    for (int s = 0; s < 5; ++s) { mv_br(m.sN_l1[s]); mv_br(m.sN_l2[s]); }
    m.device = dev;
    m.fp16 = fp16;
}

brotensor::Device OpenposeDetector::device() const { return impl_->device; }

// ── Run ─────────────────────────────────────────────────────────────────────

PafHeatmap OpenposeDetector::run(const PreprocessedImage& pp) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("infer_maps() called before load()");

    int H = pp.transform.pad_h, W = pp.transform.pad_w;
    Tensor x = (m.device == brotensor::Device::CPU) ? pp.pixels
                                                     : pp.pixels.to(m.device);
    if (m.fp16) {
        Tensor h;
        brotensor::cast(x, h, brotensor::Dtype::FP16);
        x = std::move(h);
    }

    // model0 trunk.
    int cur_c = 3;
    for (std::size_t i = 0; i < m.model0.size(); ++i) {
        if (m.model0_pool[i]) {
            Tensor pooled, idx;
            const int Ho = (H - 2) / 2 + 1, Wo = (W - 2) / 2 + 1;
            brotensor::max_pool2d_forward(x, /*N=*/1, cur_c, H, W,
                                          /*kH=*/2, /*kW=*/2, /*sh=*/2, /*sw=*/2,
                                          /*ph=*/0, /*pw=*/0, pooled, idx);
            x = std::move(pooled);
            H = Ho; W = Wo;
        }
        x = convf(m.model0[i], x, H, W, m.model0_pad[i]);
        apply_relu(x);
        cur_c = m.model0[i].c_out;
    }
    const Tensor out1 = x;        // 128ch trunk feature at (H,W) = padded/8
    const int trunk_c = cur_c;    // 128

    // Stage 1.
    Tensor l1 = run_branch(m.s1_l1, out1, H, W);   // 38ch
    Tensor l2 = run_branch(m.s1_l2, out1, H, W);   // 19ch

    // Stages 2..6: input = cat([prev_l1, prev_l2, out1]) = 185ch.
    for (int s = 0; s < 5; ++s) {
        std::vector<const Tensor*> parts{&l1, &l2, &out1};
        std::vector<int> cs{38, 19, trunk_c};
        Tensor cat;
        brotensor::concat_nchw_channels(parts, /*N=*/1, H, W, cs, cat);
        Tensor nl1 = run_branch(m.sN_l1[s], cat, H, W);
        Tensor nl2 = run_branch(m.sN_l2[s], cat, H, W);
        l1 = std::move(nl1);
        l2 = std::move(nl2);
    }

    // Widen the final maps back to FP32 and pull to host; the gaussian-blur /
    // peak-NMS postprocess stays FP32 on the host.
    auto to_host = [](Tensor t) -> Tensor {
        if (t.dtype != brotensor::Dtype::FP32) {
            Tensor f;
            brotensor::cast(t, f, brotensor::Dtype::FP32);
            t = std::move(f);
        }
        return (t.device == brotensor::Device::CPU) ? t
                                                     : t.to(brotensor::Device::CPU);
    };
    Tensor h1 = to_host(std::move(l1)), h2 = to_host(std::move(l2));

    PafHeatmap out;
    out.paf_c = 38;
    out.hm_c = 19;
    out.height = H;
    out.width = W;
    const std::size_t plane = static_cast<std::size_t>(H) * W;
    out.paf.assign(h1.host_f32(), h1.host_f32() + 38 * plane);
    out.heatmap.assign(h2.host_f32(), h2.host_f32() + 19 * plane);
    return out;
}

PafHeatmap OpenposeDetector::infer_maps(const uint8_t* rgb, int w, int h,
                                        int channels) const {
    PreprocessedImage pp = preprocess(rgb, w, h, channels, cfg_.detect_resolution);
    return run(pp);
}

// ── Decode ────────────────────────────────────────────────────────────────

PoseResult OpenposeDetector::detect(const uint8_t* rgb, int w, int h,
                                    int channels) const {
    int dw = 0, dh = 0;
    std::vector<uint8_t> det =
        resize_to_detect(rgb, w, h, channels, cfg_.detect_resolution, dw, dh);
    PreprocessedImage pp = preprocess_from_detect(det.data(), dw, dh);
    const PafHeatmap maps = run(pp);

    const OpenposeTransform& tf = pp.transform;
    const int H = tf.detect_h, W = tf.detect_w;       // detect-res dims (oriImg)
    const int nh = maps.height, nw = maps.width;       // network-res dims
    const int stride = 8;

    // Upsample maps x8 (HWC), crop the pad, then resize to (H,W). Replicates
    // body.py: smart_resize_k(fx=fy=8) -> crop [:Hp-pad_down, :Wp-pad_right] ->
    // smart_resize((H,W)). Single-scale, so heatmap_avg == heatmap directly.
    auto upsample_crop_resize = [&](const std::vector<float>& nchw, int C) {
        // NCHW -> HWC at network resolution.
        std::vector<float> hwc(static_cast<std::size_t>(nh) * nw * C);
        const std::size_t plane = static_cast<std::size_t>(nh) * nw;
        for (int c = 0; c < C; ++c)
            for (int y = 0; y < nh; ++y)
                for (int xx = 0; xx < nw; ++xx)
                    hwc[(static_cast<std::size_t>(y) * nw + xx) * C + c] =
                        nchw[c * plane + static_cast<std::size_t>(y) * nw + xx];
        // x8 upsample.
        const int uH = nh * stride, uW = nw * stride;
        std::vector<float> up = resize_hwc(hwc, nh, nw, C, uH, uW);
        // Crop the pad: keep [: uH - pad_down, : uW - pad_right].
        const int cH = uH - tf.pad_down, cW = uW - tf.pad_right;
        std::vector<float> crop(static_cast<std::size_t>(cH) * cW * C);
        for (int y = 0; y < cH; ++y)
            for (int xx = 0; xx < cW; ++xx)
                for (int c = 0; c < C; ++c)
                    crop[(static_cast<std::size_t>(y) * cW + xx) * C + c] =
                        up[(static_cast<std::size_t>(y) * uW + xx) * C + c];
        // Resize to (H,W).
        return resize_hwc(crop, cH, cW, C, H, W);
    };

    std::vector<float> heatmap = upsample_crop_resize(maps.heatmap, 19);  // HWC
    std::vector<float> paf     = upsample_crop_resize(maps.paf, 38);      // HWC

    auto hm_ch = [&](int part) {  // extract one heatmap channel as HxW
        std::vector<float> ch(static_cast<std::size_t>(H) * W);
        for (int i = 0; i < H * W; ++i)
            ch[i] = heatmap[static_cast<std::size_t>(i) * 19 + part];
        return ch;
    };
    auto paf_val = [&](int y, int x, int ch) {
        return paf[(static_cast<std::size_t>(y) * W + x) * 38 + ch];
    };

    // ── Peak detection over the 18 body parts ──
    std::vector<std::vector<Peak>> all_peaks(18);
    std::vector<Peak> candidate;   // global candidate array (id == index)
    int peak_counter = 0;
    for (int part = 0; part < 18; ++part) {
        const std::vector<float> map_ori = hm_ch(part);
        const std::vector<float> one = gaussian_blur(map_ori, H, W, 3.0f);
        std::vector<Peak>& peaks = all_peaks[part];
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const float v = one[static_cast<std::size_t>(y) * W + x];
                if (v <= cfg_.thre1) continue;
                // local-max NMS: >= 4-neighbors (zero-pad at borders, matching
                // map_left/right/up/down shifts).
                const float left  = (y > 0)     ? one[static_cast<std::size_t>(y - 1) * W + x] : 0.0f;
                const float right = (y < H - 1) ? one[static_cast<std::size_t>(y + 1) * W + x] : 0.0f;
                const float up    = (x > 0)     ? one[static_cast<std::size_t>(y) * W + (x - 1)] : 0.0f;
                const float down  = (x < W - 1) ? one[static_cast<std::size_t>(y) * W + (x + 1)] : 0.0f;
                if (v >= left && v >= right && v >= up && v >= down) {
                    Peak p;
                    p.x = x; p.y = y;
                    p.score = map_ori[static_cast<std::size_t>(y) * W + x];
                    p.id = peak_counter++;
                    peaks.push_back(p);
                    candidate.push_back(p);
                }
            }
        }
    }

    // ── Limb connection via PAF line integrals ──
    const int mid_num = 10;
    std::vector<std::vector<Connection>> connection_all(19);
    std::vector<bool> special_k(19, false);
    for (int k = 0; k < 19; ++k) {
        const int ch0 = kMapIdx[k][0] - 19;
        const int ch1 = kMapIdx[k][1] - 19;
        const std::vector<Peak>& candA = all_peaks[kLimbSeq[k][0] - 1];
        const std::vector<Peak>& candB = all_peaks[kLimbSeq[k][1] - 1];
        const int nA = static_cast<int>(candA.size());
        const int nB = static_cast<int>(candB.size());
        if (nA == 0 || nB == 0) { special_k[k] = true; continue; }

        std::vector<Connection> cands;   // [i,j,score, total] candidates
        std::vector<float> cand_total;
        for (int i = 0; i < nA; ++i) {
            for (int j = 0; j < nB; ++j) {
                const float vx = static_cast<float>(candB[j].x - candA[i].x);
                const float vy = static_cast<float>(candB[j].y - candA[i].y);
                float norm = std::sqrt(vx * vx + vy * vy);
                norm = std::max(0.001f, norm);
                const float ux = vx / norm, uy = vy / norm;

                float sum_mid = 0.0f;
                int n_above = 0;
                for (int t = 0; t < mid_num; ++t) {
                    // np.linspace inclusive endpoints.
                    const float frac = (mid_num == 1) ? 0.0f
                        : static_cast<float>(t) / static_cast<float>(mid_num - 1);
                    const int sx = static_cast<int>(std::lround(
                        candA[i].x + frac * (candB[j].x - candA[i].x)));
                    const int sy = static_cast<int>(std::lround(
                        candA[i].y + frac * (candB[j].y - candA[i].y)));
                    const float vxs = paf_val(sy, sx, ch0);
                    const float vys = paf_val(sy, sx, ch1);
                    const float sc = vxs * ux + vys * uy;
                    sum_mid += sc;
                    if (sc > cfg_.thre2) ++n_above;
                }
                const float mean_mid = sum_mid / static_cast<float>(mid_num);
                const float prior = std::min(
                    0.5f * static_cast<float>(H) / norm - 1.0f, 0.0f);
                const float swdp = mean_mid + prior;
                const bool crit1 = n_above > static_cast<int>(0.8f * mid_num);
                const bool crit2 = swdp > 0.0f;
                if (crit1 && crit2) {
                    Connection c;
                    c.i = i; c.j = j; c.score = swdp;
                    c.idA = candA[i].id; c.idB = candB[j].id;
                    cands.push_back(c);
                    cand_total.push_back(swdp + candA[i].score + candB[j].score);
                }
            }
        }
        // Sort by score (col-2) desc; greedily accept while i,j unused.
        std::vector<int> order(cands.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return cands[a].score > cands[b].score; });
        std::vector<Connection>& conn = connection_all[k];
        std::vector<bool> usedA(nA, false), usedB(nB, false);
        for (int oi : order) {
            const Connection& c = cands[oi];
            if (usedA[c.i] || usedB[c.j]) continue;
            conn.push_back(c);
            usedA[c.i] = true; usedB[c.j] = true;
            if (static_cast<int>(conn.size()) >= std::min(nA, nB)) break;
        }
    }

    // ── Subset assembly (people) ── port of body.py lines 174-217.
    // Each subset row: 18 candidate ids (-1 absent) + total_score + total_parts.
    struct Row { std::array<float, 18> idx; float score; float parts; };
    std::vector<Row> subset;
    auto new_row = [&]() {
        Row r; r.idx.fill(-1.0f); r.score = 0; r.parts = 0; return r;
    };

    for (int k = 0; k < 19; ++k) {
        if (special_k[k]) continue;
        const int indexA = kLimbSeq[k][0] - 1;
        const int indexB = kLimbSeq[k][1] - 1;
        const std::vector<Connection>& conn = connection_all[k];
        for (const Connection& c : conn) {
            const float partA = static_cast<float>(c.idA);
            const float partB = static_cast<float>(c.idB);
            int found = 0;
            int subset_idx[2] = {-1, -1};
            for (int j = 0; j < static_cast<int>(subset.size()); ++j) {
                if (subset[j].idx[indexA] == partA ||
                    subset[j].idx[indexB] == partB) {
                    if (found < 2) subset_idx[found] = j;
                    ++found;
                }
            }
            if (found == 1) {
                const int j = subset_idx[0];
                if (subset[j].idx[indexB] != partB) {
                    subset[j].idx[indexB] = partB;
                    subset[j].parts += 1;
                    subset[j].score += candidate[c.idB].score + c.score;
                }
            } else if (found == 2) {
                const int j1 = subset_idx[0], j2 = subset_idx[1];
                int overlap = 0;
                for (int p = 0; p < 18; ++p)
                    if (subset[j1].idx[p] >= 0 && subset[j2].idx[p] >= 0)
                        ++overlap;
                if (overlap == 0) {  // disjoint -> merge
                    // body.py: subset[j1][:-2] += subset[j2][:-2] + 1. With the
                    // -1/id encoding and disjoint membership, each present part
                    // is -1 on the other side, so id + (-1) + 1 == id (absent
                    // stays -1 + -1 + 1 == -1). Apply that additive rule verbatim.
                    for (int p = 0; p < 18; ++p)
                        subset[j1].idx[p] = subset[j1].idx[p] + subset[j2].idx[p] + 1.0f;
                    subset[j1].score += subset[j2].score + c.score;
                    subset[j1].parts += subset[j2].parts;
                    subset.erase(subset.begin() + j2);
                } else {  // like found == 1
                    subset[j1].idx[indexB] = partB;
                    subset[j1].parts += 1;
                    subset[j1].score += candidate[c.idB].score + c.score;
                }
            } else if (found == 0 && k < 17) {
                Row r = new_row();
                r.idx[indexA] = partA;
                r.idx[indexB] = partB;
                r.parts = 2;
                r.score = candidate[c.idA].score + candidate[c.idB].score +
                          c.score;
                subset.push_back(r);
            }
        }
    }

    // ── Prune ──
    PoseResult result;
    result.width = W;
    result.height = H;
    for (const Row& r : subset) {
        if (r.parts < 4 || r.score / r.parts < 0.4f) continue;
        BodyPose body;
        for (int p = 0; p < 18; ++p) {
            const int id = static_cast<int>(r.idx[p]);
            if (id < 0 || id >= static_cast<int>(candidate.size())) {
                body.keypoints[p] = Keypoint{};   // absent
            } else {
                Keypoint kp;
                kp.x = static_cast<float>(candidate[id].x) /
                       static_cast<float>(W);
                kp.y = static_cast<float>(candidate[id].y) /
                       static_cast<float>(H);
                kp.score = candidate[id].score;
                kp.present = true;
                body.keypoints[p] = kp;
            }
        }
        body.total_score = r.score;
        body.total_parts = static_cast<int>(r.parts);
        result.bodies.push_back(body);
    }
    return result;
}

// ── Draw ────────────────────────────────────────────────────────────────────

std::vector<uint8_t> OpenposeDetector::draw(const PoseResult& r) {
    const int W = r.width, H = r.height;
    std::vector<uint8_t> canvas(static_cast<std::size_t>(W) * H * 3, 0);
    if (W <= 0 || H <= 0) return canvas;

    static const int limb17[17][2] = {
        {2, 3}, {2, 6}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {2, 9}, {9, 10},
        {10, 11}, {2, 12}, {12, 13}, {13, 14}, {2, 1}, {1, 15}, {15, 17},
        {1, 16}, {16, 18}};
    static const int colors[18][3] = {
        {255, 0, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {170, 255, 0},
        {85, 255, 0}, {0, 255, 0}, {0, 255, 85}, {0, 255, 170}, {0, 255, 255},
        {0, 170, 255}, {0, 85, 255}, {0, 0, 255}, {85, 0, 255}, {170, 0, 255},
        {255, 0, 255}, {255, 0, 170}, {255, 0, 85}};
    const int stickwidth = 4;

    auto put = [&](int x, int y, const int col[3]) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        uint8_t* px = &canvas[(static_cast<std::size_t>(y) * W + x) * 3];
        px[0] = static_cast<uint8_t>(col[0]);
        px[1] = static_cast<uint8_t>(col[1]);
        px[2] = static_cast<uint8_t>(col[2]);
    };

    for (const BodyPose& body : r.bodies) {
        // Limb sticks: filled rotated ellipse (length/2 x stickwidth).
        for (int li = 0; li < 17; ++li) {
            const Keypoint& k1 = body.keypoints[limb17[li][0] - 1];
            const Keypoint& k2 = body.keypoints[limb17[li][1] - 1];
            if (!k1.present || !k2.present) continue;
            // Reference: Y = x*W, X = y*H; midpoint (mY,mX); angle atan2(dX,dY).
            const float Y0 = k1.x * W, Y1 = k2.x * W;
            const float X0 = k1.y * H, X1 = k2.y * H;
            const float mX = 0.5f * (X0 + X1);
            const float mY = 0.5f * (Y0 + Y1);
            const float length =
                std::sqrt((X0 - X1) * (X0 - X1) + (Y0 - Y1) * (Y0 - Y1));
            const float angle = std::atan2(X0 - X1, Y0 - Y1);  // radians
            const float a = length / 2.0f;     // semi-major along the limb
            const float b = stickwidth;        // semi-minor
            const int dimcol[3] = {static_cast<int>(colors[li][0] * 0.6f),
                                   static_cast<int>(colors[li][1] * 0.6f),
                                   static_cast<int>(colors[li][2] * 0.6f)};
            const float ca = std::cos(angle), sa = std::sin(angle);
            // ellipse2Poly center is (mY,mX) in (col,row); the polygon axis is
            // along `angle`. Rasterize by scanning the bounding box and testing
            // the rotated-ellipse implicit equation.
            const float maxr = std::max(a, b) + 1.0f;
            const int x0 = static_cast<int>(mY - maxr), x1 = static_cast<int>(mY + maxr);
            const int y0 = static_cast<int>(mX - maxr), y1 = static_cast<int>(mX + maxr);
            for (int py = y0; py <= y1; ++py) {
                for (int px = x0; px <= x1; ++px) {
                    const float dx = px - mY, dy = py - mX;
                    // Rotate point into the ellipse frame (axis along angle).
                    const float u = dx * ca + dy * sa;
                    const float v = -dx * sa + dy * ca;
                    if ((u * u) / (a * a) + (v * v) / (b * b) <= 1.0f)
                        put(px, py, dimcol);
                }
            }
        }
        // Joint circles radius 4.
        for (int p = 0; p < 18; ++p) {
            const Keypoint& kp = body.keypoints[p];
            if (!kp.present) continue;
            const int cx = static_cast<int>(kp.x * W);
            const int cy = static_cast<int>(kp.y * H);
            for (int dy = -4; dy <= 4; ++dy)
                for (int dx = -4; dx <= 4; ++dx)
                    if (dx * dx + dy * dy <= 16)
                        put(cx + dx, cy + dy, colors[p]);
        }
    }
    return canvas;
}

}  // namespace brovisionml::openpose
