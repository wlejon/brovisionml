// SAM mask decoder: structural / end-to-end CPU test with a synthesized tiny
// checkpoint. We write a HF-named mask_decoder.* checkpoint (8-dim, 2 heads,
// 2 transformer layers, 4 mask tokens, 4x4 image grid), load it through the
// real loader, and run decode() against synthetic image embedding / positional
// encoding / sparse + dense prompt embeddings. This exercises the whole path:
// token assembly, the two-way transformer (self + both cross attentions via
// flash_attention_varlen over pre-projected Q/K/V), the conv-transpose
// upscaler, the per-token hypernetwork filters, and the IoU head. Asserts
// output shapes, the multimask vs single-mask slicing, and finiteness, plus
// CPU/CUDA parity when a GPU is present.
#include "brovisionml/sam_mask_decoder.h"

#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"
#include "brotensor/runtime.h"

#include "test_device.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace st = brotensor::safetensors;

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

struct CheckpointBuilder {
    std::deque<std::vector<float>> store;
    std::vector<st::WriteEntry>    entries;
    void add(const std::string& name, std::vector<int64_t> shape) {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        std::vector<float> buf(static_cast<std::size_t>(n));
        const bool is_ln = name.find("layer_norm") != std::string::npos;
        const bool is_w  = name.size() >= 7 &&
                           name.compare(name.size() - 7, 7, ".weight") == 0;
        for (std::size_t i = 0; i < buf.size(); ++i) {
            if (is_ln) buf[i] = is_w ? 1.0f : 0.0f;
            else       buf[i] = (static_cast<float>(i % 7) - 3.0f) * 0.02f;
        }
        store.push_back(std::move(buf));
        const std::vector<float>& b = store.back();
        entries.push_back(st::WriteEntry{name, st::Dtype::F32, std::move(shape),
                                         b.data(), b.size() * sizeof(float)});
    }
};

}  // namespace

int main() {
    using namespace brovisionml::sam;

    MaskDecoderConfig cfg;
    cfg.hidden_size             = 8;
    cfg.image_embedding_size    = 4;     // grid 4 -> HW 16, mask 16x16
    cfg.num_multimask_outputs   = 3;     // K = 4
    cfg.num_hidden_layers       = 2;
    cfg.num_attention_heads     = 2;
    cfg.attention_downsample_rate = 2;   // cross internal = 4
    cfg.mlp_dim                 = 16;
    cfg.iou_head_depth          = 3;
    cfg.iou_head_hidden_dim     = 8;

    const int D    = cfg.hidden_size;       // 8
    const int K    = cfg.num_mask_tokens(); // 4
    const int g    = cfg.grid();            // 4
    const int cint = D / cfg.attention_downsample_rate;  // 4
    const int md   = cfg.mlp_dim;           // 16
    const int cup1 = D / 4, cup2 = D / 8;   // 2, 1
    const int ms   = cfg.mask_size();       // 16

    CheckpointBuilder cb;
    const std::string pre = "mask_decoder.";
    cb.add(pre + "iou_token.weight",   {1, D});
    cb.add(pre + "mask_tokens.weight", {K, D});

    auto add_attn = [&](const std::string& p, int internal) {
        cb.add(p + "q_proj.weight",   {internal, D});
        cb.add(p + "q_proj.bias",     {internal});
        cb.add(p + "k_proj.weight",   {internal, D});
        cb.add(p + "k_proj.bias",     {internal});
        cb.add(p + "v_proj.weight",   {internal, D});
        cb.add(p + "v_proj.bias",     {internal});
        cb.add(p + "out_proj.weight", {D, internal});
        cb.add(p + "out_proj.bias",   {D});
    };
    auto add_ln = [&](const std::string& p) {
        cb.add(p + ".weight", {D});
        cb.add(p + ".bias",   {D});
    };

    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string lp = pre + "transformer.layers." + std::to_string(i) + ".";
        add_attn(lp + "self_attn.", D);
        add_ln(lp + "layer_norm1");
        add_attn(lp + "cross_attn_token_to_image.", cint);
        add_ln(lp + "layer_norm2");
        cb.add(lp + "mlp.lin1.weight", {md, D});
        cb.add(lp + "mlp.lin1.bias",   {md});
        cb.add(lp + "mlp.lin2.weight", {D, md});
        cb.add(lp + "mlp.lin2.bias",   {D});
        add_ln(lp + "layer_norm3");
        add_attn(lp + "cross_attn_image_to_token.", cint);
        add_ln(lp + "layer_norm4");
    }
    add_attn(pre + "transformer.final_attn_token_to_image.", cint);
    add_ln(pre + "transformer.layer_norm_final_attn");

    cb.add(pre + "upscale_conv1.weight", {D, cup1, 2, 2});
    cb.add(pre + "upscale_conv1.bias",   {cup1});
    cb.add(pre + "upscale_layer_norm.weight", {cup1});
    cb.add(pre + "upscale_layer_norm.bias",   {cup1});
    cb.add(pre + "upscale_conv2.weight", {cup1, cup2, 2, 2});
    cb.add(pre + "upscale_conv2.bias",   {cup2});

    for (int i = 0; i < K; ++i) {
        const std::string hp = pre + "output_hypernetworks_mlps." + std::to_string(i) + ".";
        cb.add(hp + "proj_in.weight",  {D, D});
        cb.add(hp + "proj_in.bias",    {D});
        cb.add(hp + "layers.0.weight", {D, D});
        cb.add(hp + "layers.0.bias",   {D});
        cb.add(hp + "proj_out.weight", {cup2, D});
        cb.add(hp + "proj_out.bias",   {cup2});
    }
    cb.add(pre + "iou_prediction_head.proj_in.weight",  {cfg.iou_head_hidden_dim, D});
    cb.add(pre + "iou_prediction_head.proj_in.bias",    {cfg.iou_head_hidden_dim});
    cb.add(pre + "iou_prediction_head.layers.0.weight", {cfg.iou_head_hidden_dim,
                                                          cfg.iou_head_hidden_dim});
    cb.add(pre + "iou_prediction_head.layers.0.bias",   {cfg.iou_head_hidden_dim});
    cb.add(pre + "iou_prediction_head.proj_out.weight", {K, cfg.iou_head_hidden_dim});
    cb.add(pre + "iou_prediction_head.proj_out.bias",   {K});

    const std::string path = "sam_mask_decoder_test.safetensors";
    st::write_file(path, cb.entries);

    MaskDecoder dec(cfg);
    dec.load_file(path);

    // Synthetic inputs (host FP32), all finite and non-trivial.
    auto make_map = [&](float phase) {
        brotensor::Tensor t = brotensor::Tensor::mat(1, D * g * g);
        float* p = t.host_f32_mut();
        for (int i = 0; i < t.cols; ++i)
            p[i] = std::sin(static_cast<float>(i) * 0.01f + phase) * 0.5f;
        return t;
    };
    brotensor::Tensor img = make_map(0.0f);
    brotensor::Tensor pe  = make_map(1.0f);
    brotensor::Tensor den = make_map(2.0f);

    brotensor::Tensor sparse = brotensor::Tensor::mat(2, D);  // two prompt tokens
    {
        float* s = sparse.host_f32_mut();
        for (int i = 0; i < sparse.cols * sparse.rows; ++i)
            s[i] = std::cos(static_cast<float>(i) * 0.03f) * 0.1f;
    }

    // ── Multimask: 3 masks (skip the first), 3 IoU scores. ──
    DecodedMasks mm = dec.decode(img, pe, sparse, den, /*multimask=*/true);
    check(mm.num_out == K - 1 && mm.mask_size == ms, "multimask num_out / size");
    check(mm.masks.rows == K - 1 && mm.masks.cols == ms * ms, "multimask masks shape");
    check(mm.iou.rows == K - 1 && mm.iou.cols == 1, "multimask iou shape");
    {
        bool finite = true;
        const float* m = mm.masks.host_f32();
        for (int i = 0; i < mm.masks.rows * mm.masks.cols; ++i)
            if (!std::isfinite(m[i])) finite = false;
        const float* q = mm.iou.host_f32();
        for (int i = 0; i < mm.iou.rows; ++i) if (!std::isfinite(q[i])) finite = false;
        check(finite, "multimask outputs finite");
    }

    // ── Single mask. ──
    DecodedMasks sm = dec.decode(img, pe, sparse, den, /*multimask=*/false);
    check(sm.num_out == 1 && sm.masks.rows == 1 && sm.masks.cols == ms * ms,
          "single mask shape");
    check(sm.iou.rows == 1 && sm.iou.cols == 1, "single iou shape");

    // ── No sparse prompts (mask-only / dense-only decode still works). ──
    {
        brotensor::Tensor none;  // empty
        DecodedMasks dm = dec.decode(img, pe, none, den, /*multimask=*/true);
        check(dm.masks.rows == K - 1 && dm.masks.cols == ms * ms,
              "no-sparse masks shape");
    }

    // ── CUDA parity (skipped cleanly when no GPU). ──
    brotensor::init();
    const brotensor::Device gpu = brovisionml_test::preferred_gpu();
    if (gpu != brotensor::Device::CPU) {
        const char* dev = brovisionml_test::device_name(gpu);
        dec.to(gpu);
        check(dec.device() == gpu, "to(gpu) moved weights");
        brotensor::Tensor img_g = img.to(gpu);
        brotensor::Tensor pe_g  = pe.to(gpu);
        brotensor::Tensor den_g = den.to(gpu);
        brotensor::Tensor sp_g  = sparse.to(gpu);
        DecodedMasks gm = dec.decode(img_g, pe_g, sp_g, den_g, /*multimask=*/true);
        brotensor::Tensor gm_cpu = gm.masks.to(brotensor::Device::CPU);
        float worst = 0.0f;
        const float* a = mm.masks.host_f32();
        const float* b = gm_cpu.host_f32();
        for (int i = 0; i < mm.masks.rows * mm.masks.cols; ++i)
            worst = std::max(worst, std::fabs(a[i] - b[i]));
        if (worst > 1e-2f) {
            std::fprintf(stderr, "FAIL: CPU/%s mask diff %g > 1e-2\n", dev, worst);
            ++failures;
        }
        std::printf("sam_mask_decoder: %s parity max abs diff %g\n", dev, worst);
    } else {
        std::printf("sam_mask_decoder: no GPU device, GPU parity skipped\n");
    }

    std::remove(path.c_str());

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("sam_mask_decoder: all checks passed\n");
    return 0;
}
