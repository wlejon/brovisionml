#include "brovisionml/segformer.h"

#include "brovisionml/segformer_preprocess.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"

#include "profile.h"
#include "weights_util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brovisionml::segformer {

namespace {

using brotensor::Tensor;
namespace st = brotensor::safetensors;
using brovisionml::detail::load_whole;

const std::string kWho = "segformer::SegformerDetector: ";

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(kWho + msg);
}

// ─── Minimal config.json reader ───────────────────────────────────────────────
//
// SegFormer's config.json is a flat JSON object whose values of interest are a
// few scalars and a few int arrays (hidden_sizes, depths, etc.). This is not a
// general JSON parser — it locates the key by name and reads the literal
// following the colon, which is all the SegFormer config layout needs.

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) fail("could not open '" + path + "'");
    const std::streamoff n = f.tellg();
    std::string buf(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
    f.seekg(0);
    if (n > 0) f.read(buf.data(), n);
    return buf;
}

// Find `"key"` and return the index just past the following colon.
std::size_t find_value_pos(const std::string& js, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    std::size_t k = js.find(pat);
    if (k == std::string::npos) fail("config.json missing key '" + key + "'");
    std::size_t c = js.find(':', k + pat.size());
    if (c == std::string::npos) fail("config.json malformed near '" + key + "'");
    return c + 1;
}

double json_number(const std::string& js, const std::string& key) {
    std::size_t p = find_value_pos(js, key);
    return std::strtod(js.c_str() + p, nullptr);
}

int json_int(const std::string& js, const std::string& key) {
    return static_cast<int>(std::lround(json_number(js, key)));
}

std::vector<int> json_int_array(const std::string& js, const std::string& key) {
    std::size_t p = find_value_pos(js, key);
    std::size_t lb = js.find('[', p);
    std::size_t rb = js.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos)
        fail("config.json key '" + key + "' is not an array");
    std::vector<int> out;
    const char* s = js.c_str() + lb + 1;
    const char* end = js.c_str() + rb;
    while (s < end) {
        while (s < end && (*s == ',' || *s == ' ' || *s == '\n' ||
                           *s == '\r' || *s == '\t'))
            ++s;
        if (s >= end) break;
        char* next = nullptr;
        long v = std::strtol(s, &next, 10);
        if (next == s) break;
        out.push_back(static_cast<int>(v));
        s = next;
    }
    return out;
}

// ─── Config (read from config.json) ───────────────────────────────────────────

struct MiTConfig {
    std::vector<int> hidden_sizes;          // C per stage
    std::vector<int> num_attention_heads;   // heads per stage
    std::vector<int> depths;                // blocks per stage
    std::vector<int> sr_ratios;             // spatial-reduction ratio per stage
    std::vector<int> patch_sizes;           // overlap-patch kernel per stage
    std::vector<int> strides;               // overlap-patch stride per stage
    std::vector<int> mlp_ratios;            // MixFFN expansion per stage
    int num_encoder_blocks = 4;
    int decoder_hidden_size = 256;
    int num_labels = 150;
    float layer_norm_eps = 1e-6f;

    int stages() const { return num_encoder_blocks; }
};

MiTConfig parse_config(const std::string& path) {
    const std::string js = read_file(path);
    MiTConfig c;
    c.hidden_sizes        = json_int_array(js, "hidden_sizes");
    c.num_attention_heads = json_int_array(js, "num_attention_heads");
    c.depths              = json_int_array(js, "depths");
    c.sr_ratios           = json_int_array(js, "sr_ratios");
    c.patch_sizes         = json_int_array(js, "patch_sizes");
    c.strides             = json_int_array(js, "strides");
    c.num_encoder_blocks  = json_int(js, "num_encoder_blocks");
    c.decoder_hidden_size = json_int(js, "decoder_hidden_size");
    // num_labels has no top-level key in SegFormer's config (it is the size of
    // id2label); it is derived from the classifier weight at load time below.
    c.layer_norm_eps      = static_cast<float>(json_number(js, "layer_norm_eps"));
    // SegFormer's config stores per-stage "mlp_ratios" (an int array).
    c.mlp_ratios          = json_int_array(js, "mlp_ratios");
    if (static_cast<int>(c.mlp_ratios.size()) != c.num_encoder_blocks ||
        static_cast<int>(c.hidden_sizes.size()) != c.num_encoder_blocks ||
        static_cast<int>(c.num_attention_heads.size()) != c.num_encoder_blocks ||
        static_cast<int>(c.depths.size()) != c.num_encoder_blocks ||
        static_cast<int>(c.sr_ratios.size()) != c.num_encoder_blocks ||
        static_cast<int>(c.patch_sizes.size()) != c.num_encoder_blocks ||
        static_cast<int>(c.strides.size()) != c.num_encoder_blocks)
        fail("config.json stage-array lengths disagree with num_encoder_blocks");
    return c;
}

// ─── Weight tables (host FP32) ────────────────────────────────────────────────

struct AttnWeights {
    Tensor q_w, q_b, k_w, k_b, v_w, v_b;   // (C,C) / (C,1)
    Tensor o_w, o_b;                       // (C,C) / (C,1)
    bool has_sr = false;
    Tensor sr_w, sr_b;                     // Conv2d(C->C, k=sr, s=sr): (C, C*sr*sr)/(C,1)
    Tensor sr_ln_w, sr_ln_b;               // (C,1)
    int sr = 1;
};

struct BlockWeights {
    Tensor ln1_w, ln1_b;     // (C,1)
    AttnWeights attn;
    Tensor ln2_w, ln2_b;     // (C,1)
    Tensor fc1_w, fc1_b;     // (mlp_hidden,C) / (mlp_hidden,1)
    Tensor dw_w, dw_b;       // depthwise (mlp_hidden, 9) / (mlp_hidden,1)
    Tensor fc2_w, fc2_b;     // (C,mlp_hidden) / (C,1)
};

struct StageWeights {
    Tensor proj_w, proj_b;   // OverlapPatchEmbed conv (C, in*k*k) / (C,1)
    Tensor pe_ln_w, pe_ln_b; // patch-embed token LayerNorm (C,1)
    std::vector<BlockWeights> blocks;
    Tensor final_ln_w, final_ln_b;   // per-stage final LayerNorm (C,1)
    int in_ch = 0, k = 0, stride = 0, pad = 0, C = 0, heads = 0;
};

struct LinearC {
    Tensor w, b;   // (dec, C) / (dec,1)
};

}  // namespace

// ─── Impl ─────────────────────────────────────────────────────────────────────

struct SegformerDetector::Impl {
    bool loaded = false;
    brotensor::Device device = brotensor::Device::CPU;
    MiTConfig mc;

    std::vector<StageWeights> stages;

    // Decode head.
    std::vector<LinearC> linear_c;     // one per stage
    Tensor fuse_w;                     // Conv2d(4*dec -> dec, 1x1, no bias): (dec, 4*dec)
    Tensor bn_w, bn_b, bn_mean, bn_var;  // (dec,1)
    Tensor cls_w, cls_b;               // Conv2d(dec -> num_labels, 1x1): (num_labels, dec)/(num_labels,1)
};

namespace {

// Run an attention block with explicit biased q/k/v/o projections + manual
// per-head scaled-dot-product attention. This is the bias-correct path
// (cross_attention_forward takes no biases). Q over the full grid, K/V over the
// (optionally) reduced grid.
Tensor attention_biased(const AttnWeights& a, const Tensor& x, int C, int heads,
                        int H, int W, float ln_eps) {
    // Build the K/V context sequence (reduced if sr>1).
    Tensor ctx;
    if (a.has_sr) {
        Tensor nchw;
        brotensor::sequence_to_nchw(x, 1, C, H, W, nchw);
        Tensor red;
        brotensor::conv2d_forward(nchw, a.sr_w, &a.sr_b, 1, C, H, W,
                                  C, a.sr, a.sr, a.sr, a.sr, 0, 0, 1, 1, red);
        const int rh = H / a.sr, rw = W / a.sr;
        Tensor red_seq;
        brotensor::nchw_to_sequence(red, 1, C, rh, rw, red_seq);
        brotensor::layernorm_forward_inference_batched(
            red_seq, a.sr_ln_w, a.sr_ln_b, ctx, ln_eps);
    } else {
        ctx = x;
    }

    const int Nq = x.rows;        // H*W
    const int Nk = ctx.rows;      // reduced grid (or H*W)
    const int hd = C / heads;

    // q = x @ Wq^T + bq  -> (Nq, C); k,v = ctx @ Wk^T + bk -> (Nk, C).
    Tensor q, k, v;
    brotensor::linear_forward_batched(a.q_w, a.q_b, x, q);
    brotensor::linear_forward_batched(a.k_w, a.k_b, ctx, k);
    brotensor::linear_forward_batched(a.v_w, a.v_b, ctx, v);

    // Single-sequence varlen cross-attention, FP32 on either backend: the full
    // grid's Nq queries attend the (optionally sr-reduced) Nk keys. cu_seqlens
    // are built host-side (Tensor::zeros would land on the default device) and
    // migrated alongside q/k/v.
    Tensor cu_q = Tensor::zeros_on(brotensor::Device::CPU, 2, 1,
                                   brotensor::Dtype::INT32);
    static_cast<int32_t*>(cu_q.data)[1] = Nq;
    Tensor cu_k = Tensor::zeros_on(brotensor::Device::CPU, 2, 1,
                                   brotensor::Dtype::INT32);
    static_cast<int32_t*>(cu_k.data)[1] = Nk;
    if (q.device != brotensor::Device::CPU) {
        cu_q = cu_q.to(q.device);
        cu_k = cu_k.to(q.device);
    }
    Tensor out;
    brotensor::flash_attention_varlen_forward(
        q, k, v,
        static_cast<const int32_t*>(cu_q.data),
        static_cast<const int32_t*>(cu_k.data),
        /*batch_size=*/1, /*max_seqlen_q=*/Nq, /*max_seqlen_k=*/Nk,
        heads, hd, /*causal=*/false, out);

    // Output projection: out @ Wo^T + bo.
    Tensor proj;
    brotensor::linear_forward_batched(a.o_w, a.o_b, out, proj);
    return proj;   // (Nq, C)
}

}  // namespace

// ─── Construction / loading ───────────────────────────────────────────────────

SegformerDetector::SegformerDetector(SegformerConfig cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>()) {}
SegformerDetector::~SegformerDetector() = default;
SegformerDetector::SegformerDetector(SegformerDetector&&) noexcept = default;
SegformerDetector& SegformerDetector::operator=(SegformerDetector&&) noexcept = default;

void SegformerDetector::load(const std::string& dir) {
    impl_->mc = parse_config(dir + "/config.json");
    load_file(dir + "/model.safetensors");
}

void SegformerDetector::load_file(const std::string& path) {
    Impl& m = *impl_;
    // If load() wasn't used, derive the config dir from the file's directory.
    if (m.mc.hidden_sizes.empty()) {
        const std::size_t slash = path.find_last_of("/\\");
        const std::string dir = (slash == std::string::npos)
                                    ? std::string(".") : path.substr(0, slash);
        m.mc = parse_config(dir + "/config.json");
    }
    st::File f = st::File::open(path);

    // num_labels is not a config key — derive it from the classifier weight
    // (num_labels, decoder_hidden_size, 1, 1).
    {
        const st::TensorView* cw = f.find("decode_head.classifier.weight");
        if (!cw || cw->shape.empty())
            fail("missing 'decode_head.classifier.weight'");
        m.mc.num_labels = static_cast<int>(cw->shape[0]);
    }
    const MiTConfig& mc = m.mc;

    const std::string SE = "segformer.encoder.";

    m.stages.clear();
    m.stages.resize(mc.stages());
    for (int i = 0; i < mc.stages(); ++i) {
        StageWeights& s = m.stages[i];
        s.C = mc.hidden_sizes[i];
        s.heads = mc.num_attention_heads[i];
        s.k = mc.patch_sizes[i];
        s.stride = mc.strides[i];
        s.pad = mc.patch_sizes[i] / 2;
        s.in_ch = (i == 0) ? 3 : mc.hidden_sizes[i - 1];

        const std::string pe = SE + "patch_embeddings." + std::to_string(i) + ".";
        s.proj_w = load_whole(f, kWho, pe + "proj.weight",
                              s.C, s.in_ch * s.k * s.k);
        s.proj_b = load_whole(f, kWho, pe + "proj.bias", s.C, 1);
        s.pe_ln_w = load_whole(f, kWho, pe + "layer_norm.weight", s.C, 1);
        s.pe_ln_b = load_whole(f, kWho, pe + "layer_norm.bias",   s.C, 1);

        const int mlp_hidden = mc.mlp_ratios[i] * s.C;
        const int sr = mc.sr_ratios[i];
        s.blocks.resize(mc.depths[i]);
        for (int j = 0; j < mc.depths[i]; ++j) {
            BlockWeights& b = s.blocks[j];
            const std::string bp =
                SE + "block." + std::to_string(i) + "." + std::to_string(j) + ".";

            b.ln1_w = load_whole(f, kWho, bp + "layer_norm_1.weight", s.C, 1);
            b.ln1_b = load_whole(f, kWho, bp + "layer_norm_1.bias",   s.C, 1);

            AttnWeights& a = b.attn;
            a.sr = sr;
            a.q_w = load_whole(f, kWho, bp + "attention.self.query.weight", s.C, s.C);
            a.q_b = load_whole(f, kWho, bp + "attention.self.query.bias",   s.C, 1);
            a.k_w = load_whole(f, kWho, bp + "attention.self.key.weight",   s.C, s.C);
            a.k_b = load_whole(f, kWho, bp + "attention.self.key.bias",     s.C, 1);
            a.v_w = load_whole(f, kWho, bp + "attention.self.value.weight", s.C, s.C);
            a.v_b = load_whole(f, kWho, bp + "attention.self.value.bias",   s.C, 1);
            a.o_w = load_whole(f, kWho, bp + "attention.output.dense.weight", s.C, s.C);
            a.o_b = load_whole(f, kWho, bp + "attention.output.dense.bias",   s.C, 1);
            if (sr > 1) {
                a.has_sr = true;
                a.sr_w = load_whole(f, kWho, bp + "attention.self.sr.weight",
                                    s.C, s.C * sr * sr);
                a.sr_b = load_whole(f, kWho, bp + "attention.self.sr.bias", s.C, 1);
                a.sr_ln_w = load_whole(f, kWho, bp + "attention.self.layer_norm.weight", s.C, 1);
                a.sr_ln_b = load_whole(f, kWho, bp + "attention.self.layer_norm.bias",   s.C, 1);
            }

            b.ln2_w = load_whole(f, kWho, bp + "layer_norm_2.weight", s.C, 1);
            b.ln2_b = load_whole(f, kWho, bp + "layer_norm_2.bias",   s.C, 1);

            b.fc1_w = load_whole(f, kWho, bp + "mlp.dense1.weight", mlp_hidden, s.C);
            b.fc1_b = load_whole(f, kWho, bp + "mlp.dense1.bias",   mlp_hidden, 1);
            // depthwise 3x3: weight (mlp_hidden, 1, 3, 3) -> (mlp_hidden, 9).
            b.dw_w = load_whole(f, kWho, bp + "mlp.dwconv.dwconv.weight", mlp_hidden, 9);
            b.dw_b = load_whole(f, kWho, bp + "mlp.dwconv.dwconv.bias",   mlp_hidden, 1);
            b.fc2_w = load_whole(f, kWho, bp + "mlp.dense2.weight", s.C, mlp_hidden);
            b.fc2_b = load_whole(f, kWho, bp + "mlp.dense2.bias",   s.C, 1);
        }

        s.final_ln_w = load_whole(f, kWho, SE + "layer_norm." + std::to_string(i) + ".weight", s.C, 1);
        s.final_ln_b = load_whole(f, kWho, SE + "layer_norm." + std::to_string(i) + ".bias",   s.C, 1);
    }

    // ── Decode head ──
    const int dec = mc.decoder_hidden_size;
    const std::string DH = "decode_head.";
    m.linear_c.resize(mc.stages());
    for (int i = 0; i < mc.stages(); ++i) {
        m.linear_c[i].w = load_whole(f, kWho, DH + "linear_c." + std::to_string(i) + ".proj.weight",
                                     dec, mc.hidden_sizes[i]);
        m.linear_c[i].b = load_whole(f, kWho, DH + "linear_c." + std::to_string(i) + ".proj.bias",
                                     dec, 1);
    }
    // linear_fuse: Conv2d(4*dec -> dec, 1x1, bias=False): weight (dec, 4*dec).
    m.fuse_w = load_whole(f, kWho, DH + "linear_fuse.weight", dec, mc.stages() * dec);
    m.bn_w   = load_whole(f, kWho, DH + "batch_norm.weight",       dec, 1);
    m.bn_b   = load_whole(f, kWho, DH + "batch_norm.bias",         dec, 1);
    m.bn_mean= load_whole(f, kWho, DH + "batch_norm.running_mean", dec, 1);
    m.bn_var = load_whole(f, kWho, DH + "batch_norm.running_var",  dec, 1);
    // classifier: Conv2d(dec -> num_labels, 1x1, bias=True): weight (num_labels, dec).
    m.cls_w = load_whole(f, kWho, DH + "classifier.weight", mc.num_labels, dec);
    m.cls_b = load_whole(f, kWho, DH + "classifier.bias",   mc.num_labels, 1);

    m.loaded = true;
}

// ─── Migration ────────────────────────────────────────────────────────────────

void SegformerDetector::to(brotensor::Device dev) {
    Impl& m = *impl_;
    if (!m.loaded) fail("to() called before load()");
    if (dev == m.device) return;
    auto mv = [dev](Tensor& t) { if (t.data) t = t.to(dev); };
    for (StageWeights& s : m.stages) {
        mv(s.proj_w); mv(s.proj_b); mv(s.pe_ln_w); mv(s.pe_ln_b);
        mv(s.final_ln_w); mv(s.final_ln_b);
        for (BlockWeights& b : s.blocks) {
            mv(b.ln1_w); mv(b.ln1_b);
            AttnWeights& a = b.attn;
            mv(a.q_w); mv(a.q_b); mv(a.k_w); mv(a.k_b); mv(a.v_w); mv(a.v_b);
            mv(a.o_w); mv(a.o_b);
            if (a.has_sr) { mv(a.sr_w); mv(a.sr_b); mv(a.sr_ln_w); mv(a.sr_ln_b); }
            mv(b.ln2_w); mv(b.ln2_b);
            mv(b.fc1_w); mv(b.fc1_b); mv(b.dw_w); mv(b.dw_b); mv(b.fc2_w); mv(b.fc2_b);
        }
    }
    for (LinearC& lc : m.linear_c) { mv(lc.w); mv(lc.b); }
    mv(m.fuse_w);
    mv(m.bn_w); mv(m.bn_b); mv(m.bn_mean); mv(m.bn_var);
    mv(m.cls_w); mv(m.cls_b);
    m.device = dev;
}

brotensor::Device SegformerDetector::device() const { return impl_->device; }

// ─── Encoder + decode head forward ────────────────────────────────────────────

namespace {

// One transformer block (pre-norm attention + MixFFN), in-place on the token
// sequence x (N=H*W, C).
void run_block(const BlockWeights& b, Tensor& x, int C, int heads,
               int H, int W, int mlp_hidden, float ln_eps) {
    // Attention residual.
    {
        Tensor h;
        brotensor::layernorm_forward_inference_batched(x, b.ln1_w, b.ln1_b, h, ln_eps);
        Tensor attn = attention_biased(b.attn, h, C, heads, H, W, ln_eps);
        brotensor::add_inplace(x, attn);
        detail::profile_mark(x.device, "blk attn");
    }
    // MixFFN residual: dense1 -> dwconv(3x3) -> gelu -> dense2.
    {
        Tensor h;
        brotensor::layernorm_forward_inference_batched(x, b.ln2_w, b.ln2_b, h, ln_eps);
        Tensor f1;
        brotensor::linear_forward_batched(b.fc1_w, b.fc1_b, h, f1);   // (N, mlp_hidden)
        detail::profile_mark(x.device, "blk fc1");
        // dwconv: reshape (N,mlp_hidden) -> NCHW (mlp_hidden,H,W), depthwise 3x3 pad1.
        Tensor nchw;
        brotensor::sequence_to_nchw(f1, 1, mlp_hidden, H, W, nchw);
        Tensor dwc;
        brotensor::conv2d_forward(nchw, b.dw_w, &b.dw_b, 1, mlp_hidden, H, W,
                                  mlp_hidden, 3, 3, 1, 1, 1, 1, 1, 1,
                                  /*groups=*/mlp_hidden, dwc);
        Tensor f1b;
        brotensor::nchw_to_sequence(dwc, 1, mlp_hidden, H, W, f1b);
        detail::profile_mark(x.device, "blk dwconv");
        Tensor act;
        brotensor::gelu_exact_forward(f1b, act);
        Tensor f2;
        brotensor::linear_forward_batched(b.fc2_w, b.fc2_b, act, f2);  // (N, C)
        brotensor::add_inplace(x, f2);
        detail::profile_mark(x.device, "blk fc2");
    }
}

}  // namespace

brotensor::Tensor SegformerDetector::infer_logits_dev_(
        const brotensor::Tensor& pixels, int* grid_h, int* grid_w) const {
    const Impl& m = *impl_;
    if (!m.loaded) fail("infer_logits called before load()");
    const MiTConfig& mc = m.mc;
    const int S = cfg_.model_size;
    if (pixels.rows != 1 || pixels.cols != 3 * S * S)
        fail("pixels must be (1, 3*model_size*model_size)");

    Tensor x = (pixels.device == m.device) ? pixels : pixels.to(m.device);
    detail::profile_mark(m.device, nullptr);

    // ── Encoder: 4 hierarchical stages ──
    std::vector<Tensor> stage_nchw(mc.stages());   // (C_i, H_i, W_i) per stage
    std::vector<int> Hs(mc.stages()), Ws(mc.stages());

    Tensor cur = x;            // NCHW input for stage 0 (1, 3*S*S)
    int curC = 3, curH = S, curW = S;
    for (int i = 0; i < mc.stages(); ++i) {
        const StageWeights& s = m.stages[i];

        // OverlapPatchEmbed conv (strided, padded).
        Tensor emb;
        brotensor::conv2d_forward(cur, s.proj_w, &s.proj_b, 1, curC, curH, curW,
                                  s.C, s.k, s.k, s.stride, s.stride, s.pad, s.pad,
                                  1, 1, emb);
        const int H = (curH + 2 * s.pad - s.k) / s.stride + 1;
        const int W = (curW + 2 * s.pad - s.k) / s.stride + 1;

        // Flatten to tokens (H*W, C) and apply patch-embed LayerNorm.
        Tensor tok;
        brotensor::nchw_to_sequence(emb, 1, s.C, H, W, tok);
        Tensor seq;
        brotensor::layernorm_forward_inference_batched(
            tok, s.pe_ln_w, s.pe_ln_b, seq, mc.layer_norm_eps);
        detail::profile_mark(m.device, "patch embed");

        const int mlp_hidden = mc.mlp_ratios[i] * s.C;
        for (const BlockWeights& b : s.blocks)
            run_block(b, seq, s.C, s.heads, H, W, mlp_hidden, mc.layer_norm_eps);

        // Per-stage final LayerNorm, then reshape to NCHW for the next stage/head.
        Tensor normed;
        brotensor::layernorm_forward_inference_batched(
            seq, s.final_ln_w, s.final_ln_b, normed, mc.layer_norm_eps);
        Tensor out_nchw;
        brotensor::sequence_to_nchw(normed, 1, s.C, H, W, out_nchw);

        stage_nchw[i] = out_nchw;   // keep for the head
        Hs[i] = H; Ws[i] = W;

        cur = std::move(out_nchw);
        curC = s.C; curH = H; curW = W;
    }

    // ── Decode head (all-MLP) ──
    const int dec = mc.decoder_hidden_size;
    const int H0 = Hs[0], W0 = Ws[0];   // stage-0 grid (largest)

    // Project + upsample each stage to (dec, H0, W0).
    std::vector<Tensor> ups(mc.stages());
    for (int i = 0; i < mc.stages(); ++i) {
        Tensor seq;
        brotensor::nchw_to_sequence(stage_nchw[i], 1, mc.hidden_sizes[i], Hs[i], Ws[i], seq);
        Tensor proj;
        brotensor::linear_forward_batched(m.linear_c[i].w, m.linear_c[i].b, seq, proj);  // (Hi*Wi, dec)
        Tensor proj_nchw;
        brotensor::sequence_to_nchw(proj, 1, dec, Hs[i], Ws[i], proj_nchw);
        Tensor up;
        if (Hs[i] == H0 && Ws[i] == W0) {
            up = std::move(proj_nchw);
        } else {
            brotensor::interp2d_forward(proj_nchw, 1, dec, Hs[i], Ws[i], H0, W0,
                                        /*bilinear=*/1, up);   // align_corners=False
        }
        ups[i] = std::move(up);
    }

    // Concatenate in REVERSED stage order (stage3, stage2, stage1, stage0).
    std::vector<const Tensor*> parts;
    std::vector<int> chans;
    parts.reserve(mc.stages());
    chans.reserve(mc.stages());
    for (int i = mc.stages() - 1; i >= 0; --i) {
        parts.push_back(&ups[i]);
        chans.push_back(dec);
    }
    Tensor cat;
    brotensor::concat_nchw_channels(parts, 1, H0, W0, chans, cat);

    // linear_fuse (1x1 conv, no bias) -> BN -> ReLU.
    Tensor fused;
    brotensor::conv2d_forward(cat, m.fuse_w, /*bias=*/nullptr, 1, mc.stages() * dec,
                              H0, W0, dec, 1, 1, 1, 1, 0, 0, 1, 1, fused);
    Tensor bn;
    brotensor::batch_norm_inference(fused, m.bn_w, m.bn_b, m.bn_mean, m.bn_var,
                                    1, dec, H0, W0, /*eps=*/1e-5f, bn);
    Tensor act;
    brotensor::relu_forward(bn, act);

    // classifier (1x1 conv) -> logits (num_labels, H0, W0).
    Tensor logits_dev;
    brotensor::conv2d_forward(act, m.cls_w, &m.cls_b, 1, dec, H0, W0,
                              mc.num_labels, 1, 1, 1, 1, 0, 0, 1, 1, logits_dev);
    detail::profile_mark(m.device, "decode head");

    *grid_h = H0;
    *grid_w = W0;
    return logits_dev;
}

Logits SegformerDetector::infer_logits_from_tensor(const brotensor::Tensor& pixels) const {
    int H0 = 0, W0 = 0;
    Tensor logits_dev = infer_logits_dev_(pixels, &H0, &W0);
    Tensor host = (logits_dev.device == brotensor::Device::CPU)
                      ? logits_dev : logits_dev.to(brotensor::Device::CPU);
    const float* d = host.host_f32();
    Logits out;
    out.channels = impl_->mc.num_labels;
    out.height = H0;
    out.width = W0;
    out.data.assign(d, d + static_cast<std::size_t>(out.channels) * H0 * W0);
    return out;
}

Logits SegformerDetector::infer_logits(const uint8_t* rgb, int w, int h,
                                       int channels) const {
    PreprocessedImage pp = preprocess(rgb, w, h, channels, cfg_.model_size);
    Tensor px = (impl_->device == brotensor::Device::CPU)
                    ? pp.pixels : pp.pixels.to(impl_->device);
    return infer_logits_from_tensor(px);
}

SegMap SegformerDetector::detect(const uint8_t* rgb, int w, int h,
                                 int channels) const {
    // HF upsamples the LOGITS (bilinear, align_corners=False) to the input size
    // and argmaxes there; do the same on the compute device — interp2d over the
    // (C, gH, gW) logits, per-pixel argmax over channels — and download only
    // the (h*w) INT32 class map.
    PreprocessedImage pp = preprocess(rgb, w, h, channels, cfg_.model_size);
    Tensor px = (impl_->device == brotensor::Device::CPU)
                    ? pp.pixels : pp.pixels.to(impl_->device);
    int gH = 0, gW = 0;
    Tensor logits_dev = infer_logits_dev_(px, &gH, &gW);
    const int C = impl_->mc.num_labels;

    Tensor up;
    brotensor::interp2d_forward(logits_dev, 1, C, gH, gW, h, w,
                                /*bilinear=*/1, up);
    Tensor seq;   // (h*w, C): one row per pixel, channels across the row
    brotensor::nchw_to_sequence(up, 1, C, h, w, seq);
    Tensor idx;   // (h*w, 1) INT32; ties break to the smaller class id, same
                  // as the previous host argmax
    brotensor::argmax_rows(seq, idx);
    detail::profile_mark(impl_->device, "upsample+argmax");

    Tensor idx_host = (idx.device == brotensor::Device::CPU)
                          ? idx : idx.to(brotensor::Device::CPU);
    const int32_t* ip = static_cast<const int32_t*>(idx_host.data);

    SegMap out;
    out.width = w;
    out.height = h;
    const std::size_t hw = static_cast<std::size_t>(w) * h;
    out.classes.resize(hw);
    for (std::size_t p = 0; p < hw; ++p)
        out.classes[p] = static_cast<uint8_t>(ip[p]);
    return out;
}

// ─── ADE20K palette ───────────────────────────────────────────────────────────
//
// The canonical 150-class ADE20K palette (the mmsegmentation / HF
// `ade_palette()` array, also what ControlNet's seg annotator uses). 150 RGB
// triples; class id c maps to (kAdePalette[c][0..2]).
namespace {
const uint8_t kAdePalette[150][3] = {
    {120,120,120},{180,120,120},{6,230,230},{80,50,50},{4,200,3},{120,120,80},
    {140,140,140},{204,5,255},{230,230,230},{4,250,7},{224,5,255},{235,255,7},
    {150,5,61},{120,120,70},{8,255,51},{255,6,82},{143,255,140},{204,255,4},
    {255,51,7},{204,70,3},{0,102,200},{61,230,250},{255,6,51},{11,102,255},
    {255,7,71},{255,9,224},{9,7,230},{220,220,220},{255,9,92},{112,9,255},
    {8,255,214},{7,255,224},{255,184,6},{10,255,71},{255,41,10},{7,255,255},
    {224,255,8},{102,8,255},{255,61,6},{255,194,7},{255,122,8},{0,255,20},
    {255,8,41},{255,5,153},{6,51,255},{235,12,255},{160,150,20},{0,163,255},
    {140,140,140},{250,10,15},{20,255,0},{31,255,0},{255,31,0},{255,224,0},
    {153,255,0},{0,0,255},{255,71,0},{0,235,255},{0,173,255},{31,0,255},
    {11,200,200},{255,82,0},{0,255,245},{0,61,255},{0,255,112},{0,255,133},
    {255,0,0},{255,163,0},{255,102,0},{194,255,0},{0,143,255},{51,255,0},
    {0,82,255},{0,255,41},{0,255,173},{10,0,255},{173,255,0},{0,255,153},
    {255,92,0},{255,0,255},{255,0,245},{255,0,102},{255,173,0},{255,0,20},
    {255,184,184},{0,31,255},{0,255,61},{0,71,255},{255,0,204},{0,255,194},
    {0,255,82},{0,10,255},{0,112,255},{51,0,255},{0,194,255},{0,122,255},
    {0,255,163},{255,153,0},{0,255,10},{255,112,0},{143,255,0},{82,0,255},
    {163,255,0},{255,235,0},{8,184,170},{133,0,255},{0,255,92},{184,0,255},
    {255,0,31},{0,184,255},{0,214,255},{255,0,112},{92,255,0},{0,224,255},
    {112,224,255},{70,184,160},{163,0,255},{153,0,255},{71,255,0},{255,0,163},
    {255,204,0},{255,0,143},{0,255,235},{133,255,0},{255,0,235},{245,0,255},
    {255,0,122},{255,245,0},{10,190,212},{214,255,0},{0,204,255},{20,0,255},
    {255,255,0},{0,153,255},{0,41,255},{0,255,204},{41,0,255},{41,255,0},
    {173,0,255},{0,245,255},{71,0,255},{122,0,255},{0,255,184},{0,92,255},
    {184,255,0},{0,133,255},{255,214,0},{25,194,194},{102,255,0},{92,0,255},
};
}  // namespace

std::vector<uint8_t> SegformerDetector::colorize(const SegMap& m) {
    std::vector<uint8_t> rgb(static_cast<std::size_t>(m.width) * m.height * 3);
    const std::size_t hw = static_cast<std::size_t>(m.width) * m.height;
    for (std::size_t p = 0; p < hw; ++p) {
        const uint8_t c = m.classes[p] < 150 ? m.classes[p] : 0;
        rgb[p * 3 + 0] = kAdePalette[c][0];
        rgb[p * 3 + 1] = kAdePalette[c][1];
        rgb[p * 3 + 2] = kAdePalette[c][2];
    }
    return rgb;
}

}  // namespace brovisionml::segformer
