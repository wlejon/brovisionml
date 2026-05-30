#include "brovisionml/sam_prompt_encoder.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include "sam_weights_util.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace brovisionml::sam {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using detail::load_whole;

constexpr const char* kWho = "sam::PromptEncoder: ";

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

constexpr float kTwoPi = 6.28318530717958647692f;

}  // namespace

// ─── Weight table (host FP32; mask-path convs migrate with to()) ─────────────

struct PromptEncoderWeights {
    bool   loaded = false;
    Tensor gaussian;                          // (2, num_pos_feats)
    Tensor point_embed[4];                    // each (1, hidden)
    Tensor not_a_point;                        // (1, hidden)
    Tensor no_mask;                            // (1, hidden)
    // Mask downscaler (only touched on the mask-prompt path).
    Tensor mc1_w, mc1_b;                       // conv1 (mic, 1*2*2) / (mic,1)
    Tensor mln1_w, mln1_b;                     // LayerNorm2d(mic)
    Tensor mc2_w, mc2_b;                       // conv2 (mic*4, mic*2*2) / (.,1)
    Tensor mln2_w, mln2_b;                     // LayerNorm2d(mic*4)
    Tensor mc3_w, mc3_b;                       // conv3 (hidden, mic*4*1*1) / (.,1)
    // Image-grid positional encoding, (1, hidden*grid*grid) NCHW. Migrated by
    // to(); everything above except the mask convs stays host-resident.
    Tensor dense_pe;
};

// ─── Construction / loading ─────────────────────────────────────────────────

PromptEncoder::PromptEncoder(PromptEncoderConfig cfg)
    : cfg_(std::move(cfg)), w_(std::make_unique<PromptEncoderWeights>()) {
    if (cfg_.hidden_size % 2 != 0)
        fail("hidden_size must be even (sin/cos split)");
}

PromptEncoder::~PromptEncoder() = default;
PromptEncoder::PromptEncoder(PromptEncoder&&) noexcept = default;
PromptEncoder& PromptEncoder::operator=(PromptEncoder&&) noexcept = default;

namespace {

// pe_encoding for one coordinate already normalized to [0,1]. Writes
// hidden_size = 2*num_pos_feats values: [sin(t_0..F-1), cos(t_0..F-1)] where
// t_j = 2*pi * ((2*xn-1)*g(0,j) + (2*yn-1)*g(1,j)).
void pe_encode(const float* gaussian, int F, float xn, float yn, float* out) {
    const float cx = 2.0f * xn - 1.0f;
    const float cy = 2.0f * yn - 1.0f;
    for (int j = 0; j < F; ++j) {
        const float proj = cx * gaussian[j] + cy * gaussian[F + j];
        const float t = kTwoPi * proj;
        out[j]     = std::sin(t);
        out[F + j] = std::cos(t);
    }
}

}  // namespace

void PromptEncoder::load(const std::string& dir) {
    load_file(dir + "/model.safetensors");
}

void PromptEncoder::load_file(const std::string& path) {
    st::File f = st::File::open(path);

    const int H   = cfg_.hidden_size;
    const int F   = cfg_.num_pos_feats();
    const int g   = cfg_.grid();
    const int mic = cfg_.mask_input_channels / 4;   // first conv width (4)
    const std::string pre = "prompt_encoder.";

    PromptEncoderWeights w;

    w.gaussian = load_whole(f, kWho, pre + "shared_embedding.positional_embedding",
                            2, F);
    for (int i = 0; i < 4; ++i)
        w.point_embed[i] =
            load_whole(f, kWho, pre + "point_embed." + std::to_string(i) + ".weight",
                       1, H);
    w.not_a_point = load_whole(f, kWho, pre + "not_a_point_embed.weight", 1, H);
    w.no_mask     = load_whole(f, kWho, pre + "no_mask_embed.weight", 1, H);

    // Mask downscaler: conv1 1->mic (k2,s2), conv2 mic->mic*4 (k2,s2),
    // conv3 mic*4->hidden (k1). Two LayerNorm2d between the conv/activation.
    w.mc1_w  = load_whole(f, kWho, pre + "mask_embed.conv1.weight", mic, 1 * 2 * 2);
    w.mc1_b  = load_whole(f, kWho, pre + "mask_embed.conv1.bias",   mic, 1);
    w.mln1_w = load_whole(f, kWho, pre + "mask_embed.layer_norm1.weight", mic, 1);
    w.mln1_b = load_whole(f, kWho, pre + "mask_embed.layer_norm1.bias",   mic, 1);
    w.mc2_w  = load_whole(f, kWho, pre + "mask_embed.conv2.weight",
                          mic * 4, mic * 2 * 2);
    w.mc2_b  = load_whole(f, kWho, pre + "mask_embed.conv2.bias",   mic * 4, 1);
    w.mln2_w = load_whole(f, kWho, pre + "mask_embed.layer_norm2.weight", mic * 4, 1);
    w.mln2_b = load_whole(f, kWho, pre + "mask_embed.layer_norm2.bias",   mic * 4, 1);
    w.mc3_w  = load_whole(f, kWho, pre + "mask_embed.conv3.weight",
                          H, mic * 4 * 1 * 1);
    w.mc3_b  = load_whole(f, kWho, pre + "mask_embed.conv3.bias",   H, 1);

    // Build the image-grid positional encoding once, host-side. For grid cell
    // (i=row=y, j=col=x): coords normalized as (idx+0.5)/grid (cumsum-0.5 then
    // /size, input_shape=None so no second divide). NCHW channel order is the
    // [sin..,cos..] of pe_encode.
    w.dense_pe = Tensor::mat(1, H * g * g);
    {
        float* dp = w.dense_pe.host_f32_mut();
        const float* gptr = w.gaussian.host_f32();
        std::vector<float> pe(static_cast<std::size_t>(H));
        const int plane = g * g;
        for (int i = 0; i < g; ++i) {
            const float yn = (static_cast<float>(i) + 0.5f) / static_cast<float>(g);
            for (int j = 0; j < g; ++j) {
                const float xn =
                    (static_cast<float>(j) + 0.5f) / static_cast<float>(g);
                pe_encode(gptr, F, xn, yn, pe.data());
                const int spatial = i * g + j;
                for (int c = 0; c < H; ++c)
                    dp[static_cast<std::size_t>(c) * plane + spatial] = pe[c];
            }
        }
    }

    w.loaded = true;
    *w_ = std::move(w);
}

void PromptEncoder::to(brotensor::Device dev) {
    if (!w_->loaded) fail("to() called before load()");
    if (dev == device_) return;
    auto mv = [dev](Tensor& t) { if (t.data) t = t.to(dev); };
    // The mask-path convs run on-device, and dense_pe meets the image keys on
    // device. The Gaussian matrix and the per-type embeddings stay host-side:
    // they feed the host positional-encoding loop in encode().
    mv(w_->mc1_w); mv(w_->mc1_b); mv(w_->mln1_w); mv(w_->mln1_b);
    mv(w_->mc2_w); mv(w_->mc2_b); mv(w_->mln2_w); mv(w_->mln2_b);
    mv(w_->mc3_w); mv(w_->mc3_b);
    mv(w_->dense_pe);
    device_ = dev;
}

brotensor::Tensor PromptEncoder::dense_pe() const {
    if (!w_->loaded) fail("dense_pe() called before load()");
    return w_->dense_pe;
}

// ─── Encode ─────────────────────────────────────────────────────────────────

namespace {

// LayerNorm2d (channels-first), as in the image encoder's neck: normalize over
// the C channels at each pixel, then per-channel affine. X is (1, C*H*W).
void layer_norm_2d(Tensor& X, const Tensor& gamma, const Tensor& beta,
                   int C, int H, int W, float eps) {
    Tensor seq;
    brotensor::nchw_to_sequence(X, 1, C, H, W, seq);
    Tensor normed;
    brotensor::layernorm_forward_inference_batched(seq, gamma, beta, normed, eps);
    brotensor::sequence_to_nchw(normed, 1, C, H, W, X);
}

}  // namespace

PromptEmbeddings PromptEncoder::encode(const PromptInput& prompt) const {
    if (!w_->loaded) fail("encode() called before load()");
    const int H = cfg_.hidden_size;
    const int F = cfg_.num_pos_feats();
    const int g = cfg_.grid();
    const float in = static_cast<float>(cfg_.input_image_size);

    if (prompt.labels.size() != prompt.points.size())
        fail("points and labels must have equal length");

    // ── Sparse tokens (host FP32), built in SAM's order: points, then a
    //    "not a point" pad token when no box is present, then box corners. ──
    const bool has_points = !prompt.points.empty();
    const bool pad        = prompt.boxes.empty();   // SAM pads when no box
    const int  n_point_tok = has_points ? static_cast<int>(prompt.points.size()) +
                                               (pad ? 1 : 0)
                                         : 0;
    const int  n_box_tok   = 2 * static_cast<int>(prompt.boxes.size());
    const int  n_sparse    = n_point_tok + n_box_tok;

    PromptEmbeddings out;
    if (n_sparse > 0) {
        Tensor sparse = Tensor::mat(n_sparse, H);
        float* sp = sparse.host_f32_mut();
        const float* gptr = w_->gaussian.host_f32();
        int row = 0;

        auto add_embed = [&](float* dst, const Tensor& e) {
            const float* ev = e.host_f32();
            for (int c = 0; c < H; ++c) dst[c] += ev[c];
        };

        if (has_points) {
            for (std::size_t k = 0; k < prompt.points.size(); ++k) {
                float* dst = sp + static_cast<std::size_t>(row) * H;
                const int label = prompt.labels[k];
                if (label == -1) {
                    // "not a point": replaces the positional encoding entirely.
                    const float* nv = w_->not_a_point.host_f32();
                    for (int c = 0; c < H; ++c) dst[c] = nv[c];
                } else {
                    const float xn = (prompt.points[k][0] + 0.5f) / in;
                    const float yn = (prompt.points[k][1] + 0.5f) / in;
                    pe_encode(gptr, F, xn, yn, dst);
                    // label 0 = background -> embed[0]; 1 = foreground -> embed[1]
                    add_embed(dst, w_->point_embed[label == 1 ? 1 : 0]);
                }
                ++row;
            }
            if (pad) {  // trailing not_a_point padding token
                float* dst = sp + static_cast<std::size_t>(row) * H;
                const float* nv = w_->not_a_point.host_f32();
                for (int c = 0; c < H; ++c) dst[c] = nv[c];
                ++row;
            }
        }

        for (const auto& box : prompt.boxes) {
            // Two corner tokens: top-left + embed[2], bottom-right + embed[3].
            const float corners[2][2] = {{box[0], box[1]}, {box[2], box[3]}};
            for (int ci = 0; ci < 2; ++ci) {
                float* dst = sp + static_cast<std::size_t>(row) * H;
                const float xn = (corners[ci][0] + 0.5f) / in;
                const float yn = (corners[ci][1] + 0.5f) / in;
                pe_encode(gptr, F, xn, yn, dst);
                add_embed(dst, w_->point_embed[ci == 0 ? 2 : 3]);
                ++row;
            }
        }
        out.sparse = sparse.to(device_);
    }

    // ── Dense embedding ──
    if (prompt.mask.empty()) {
        // No mask: broadcast no_mask_embed (H,) over the grid -> (1, H*g*g).
        Tensor dense = Tensor::mat(1, H * g * g);
        float* dp = dense.host_f32_mut();
        const float* nm = w_->no_mask.host_f32();
        const int plane = g * g;
        for (int c = 0; c < H; ++c) {
            float* ch = dp + static_cast<std::size_t>(c) * plane;
            for (int s = 0; s < plane; ++s) ch[s] = nm[c];
        }
        out.dense = dense.to(device_);
    } else {
        // Mask path: strided-conv downscaler on device(). Expects the mask to
        // downscale exactly to the feature grid (stock SAM: 256x256 -> 64x64).
        if (prompt.mask_h <= 0 || prompt.mask_w <= 0 ||
            static_cast<int>(prompt.mask.size()) != prompt.mask_h * prompt.mask_w)
            fail("mask buffer size must equal mask_h*mask_w");
        if (prompt.mask_h != 4 * g || prompt.mask_w != 4 * g)
            fail("mask must be 4x the feature grid (e.g. 256x256 for grid 64)");

        const int mic  = cfg_.mask_input_channels / 4;
        const float eps = 1e-6f;
        const int mh = prompt.mask_h, mw = prompt.mask_w;

        Tensor x = Tensor::from_host_on(brotensor::Device::CPU, prompt.mask.data(),
                                        1, mh * mw).to(device_);

        Tensor c1;
        brotensor::conv2d_forward(x, w_->mc1_w, &w_->mc1_b,
                                  1, 1, mh, mw, mic, 2, 2, 2, 2, 0, 0, 1, 1, c1);
        layer_norm_2d(c1, w_->mln1_w, w_->mln1_b, mic, mh / 2, mw / 2, eps);
        Tensor a1; brotensor::gelu_exact_forward(c1, a1);

        Tensor c2;
        brotensor::conv2d_forward(a1, w_->mc2_w, &w_->mc2_b,
                                  1, mic, mh / 2, mw / 2, mic * 4, 2, 2, 2, 2,
                                  0, 0, 1, 1, c2);
        layer_norm_2d(c2, w_->mln2_w, w_->mln2_b, mic * 4, mh / 4, mw / 4, eps);
        Tensor a2; brotensor::gelu_exact_forward(c2, a2);

        Tensor c3;
        brotensor::conv2d_forward(a2, w_->mc3_w, &w_->mc3_b,
                                  1, mic * 4, mh / 4, mw / 4, H, 1, 1, 1, 1,
                                  0, 0, 1, 1, c3);
        out.dense = std::move(c3);  // (1, H*g*g) on device()
    }

    return out;
}

}  // namespace brovisionml::sam
