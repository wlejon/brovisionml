// SAM end-to-end: the real integration test. We synthesize a single tiny
// checkpoint carrying all three tensor namespaces (vision_encoder.*,
// prompt_encoder.*, mask_decoder.*), load it through the Sam orchestrator, set
// a small image, and run segment() with a point and with a box. This drives
// the whole pipeline — preprocess -> ViT image encoder -> prompt encoder ->
// mask decoder -> mask postprocess back to the original image size — and
// asserts the masks come out at the original resolution and finite.
#include "brovisionml/sam.h"

#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"

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

    // ── Tiny but architecturally complete config (D=8, grid=4, img=64). ──
    SamConfig cfg;
    cfg.encoder.img_size    = 64;
    cfg.encoder.patch_size  = 16;   // grid 4
    cfg.encoder.in_chans    = 3;
    cfg.encoder.embed_dim   = 8;
    cfg.encoder.depth       = 2;
    cfg.encoder.num_heads   = 2;
    cfg.encoder.mlp_ratio   = 2.0f;
    cfg.encoder.out_chans   = 8;
    cfg.encoder.window_size = 2;
    cfg.encoder.global_attn_indexes = {1};

    cfg.prompt.hidden_size          = 8;
    cfg.prompt.image_embedding_size = 4;
    cfg.prompt.input_image_size     = 64;
    cfg.prompt.mask_input_channels  = 16;

    cfg.decoder.hidden_size             = 8;
    cfg.decoder.image_embedding_size    = 4;
    cfg.decoder.num_multimask_outputs   = 3;
    cfg.decoder.num_hidden_layers       = 2;
    cfg.decoder.num_attention_heads     = 2;
    cfg.decoder.attention_downsample_rate = 2;
    cfg.decoder.mlp_dim                 = 16;
    cfg.decoder.iou_head_depth          = 3;
    cfg.decoder.iou_head_hidden_dim     = 8;

    const int D = 8, g = 4, hd = 4, md = 16, out = 8, p = 16;
    const int K = cfg.decoder.num_mask_tokens();       // 4
    const int cint = D / 2;                            // 4
    const int F = D / 2;                               // 4 (pos feats)
    const int mic = cfg.prompt.mask_input_channels / 4;  // 4
    const int cup1 = D / 4, cup2 = D / 8;              // 2, 1

    CheckpointBuilder cb;

    // ── vision_encoder.* ──
    {
        const std::string pre = "vision_encoder.";
        cb.add(pre + "patch_embed.projection.weight", {D, 3, p, p});
        cb.add(pre + "patch_embed.projection.bias",   {D});
        cb.add(pre + "pos_embed",                     {1, g, g, D});
        for (int i = 0; i < cfg.encoder.depth; ++i) {
            const std::string lp = pre + "layers." + std::to_string(i) + ".";
            const bool global = (i == 1);
            const int rg = global ? g : cfg.encoder.window_size;
            cb.add(lp + "layer_norm1.weight", {D});
            cb.add(lp + "layer_norm1.bias",   {D});
            cb.add(lp + "attn.qkv.weight",    {3 * D, D});
            cb.add(lp + "attn.qkv.bias",      {3 * D});
            cb.add(lp + "attn.proj.weight",   {D, D});
            cb.add(lp + "attn.proj.bias",     {D});
            cb.add(lp + "attn.rel_pos_h",     {2 * rg - 1, hd});
            cb.add(lp + "attn.rel_pos_w",     {2 * rg - 1, hd});
            cb.add(lp + "layer_norm2.weight", {D});
            cb.add(lp + "layer_norm2.bias",   {D});
            cb.add(lp + "mlp.lin1.weight",    {md, D});
            cb.add(lp + "mlp.lin1.bias",      {md});
            cb.add(lp + "mlp.lin2.weight",    {D, md});
            cb.add(lp + "mlp.lin2.bias",      {D});
        }
        cb.add(pre + "neck.conv1.weight",       {out, D, 1, 1});
        cb.add(pre + "neck.layer_norm1.weight", {out});
        cb.add(pre + "neck.layer_norm1.bias",   {out});
        cb.add(pre + "neck.conv2.weight",       {out, out, 3, 3});
        cb.add(pre + "neck.layer_norm2.weight", {out});
        cb.add(pre + "neck.layer_norm2.bias",   {out});
    }

    // ── prompt_encoder.* ──
    {
        const std::string pre = "prompt_encoder.";
        // Tied weight: HF SamModel stores the shared positional-encoding
        // gaussian once at the top level, not under prompt_encoder.* — match
        // the real checkpoint layout so this test exercises the real key.
        cb.add("shared_image_embedding.positional_embedding", {2, F});
        for (int i = 0; i < 4; ++i)
            cb.add(pre + "point_embed." + std::to_string(i) + ".weight", {1, D});
        cb.add(pre + "not_a_point_embed.weight", {1, D});
        cb.add(pre + "no_mask_embed.weight",     {1, D});
        cb.add(pre + "mask_embed.conv1.weight",       {mic, 1, 2, 2});
        cb.add(pre + "mask_embed.conv1.bias",         {mic});
        cb.add(pre + "mask_embed.layer_norm1.weight", {mic});
        cb.add(pre + "mask_embed.layer_norm1.bias",   {mic});
        cb.add(pre + "mask_embed.conv2.weight",       {mic * 4, mic, 2, 2});
        cb.add(pre + "mask_embed.conv2.bias",         {mic * 4});
        cb.add(pre + "mask_embed.layer_norm2.weight", {mic * 4});
        cb.add(pre + "mask_embed.layer_norm2.bias",   {mic * 4});
        cb.add(pre + "mask_embed.conv3.weight",       {D, mic * 4, 1, 1});
        cb.add(pre + "mask_embed.conv3.bias",         {D});
    }

    // ── mask_decoder.* ──
    {
        const std::string pre = "mask_decoder.";
        auto add_attn = [&](const std::string& pp, int internal) {
            cb.add(pp + "q_proj.weight",   {internal, D});
            cb.add(pp + "q_proj.bias",     {internal});
            cb.add(pp + "k_proj.weight",   {internal, D});
            cb.add(pp + "k_proj.bias",     {internal});
            cb.add(pp + "v_proj.weight",   {internal, D});
            cb.add(pp + "v_proj.bias",     {internal});
            cb.add(pp + "out_proj.weight", {D, internal});
            cb.add(pp + "out_proj.bias",   {D});
        };
        auto add_ln = [&](const std::string& pp) {
            cb.add(pp + ".weight", {D}); cb.add(pp + ".bias", {D});
        };
        cb.add(pre + "iou_token.weight",   {1, D});
        cb.add(pre + "mask_tokens.weight", {K, D});
        for (int i = 0; i < cfg.decoder.num_hidden_layers; ++i) {
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
            const std::string hp = pre + "output_hypernetworks_mlps." +
                                   std::to_string(i) + ".";
            cb.add(hp + "proj_in.weight",  {D, D});
            cb.add(hp + "proj_in.bias",    {D});
            cb.add(hp + "layers.0.weight", {D, D});
            cb.add(hp + "layers.0.bias",   {D});
            cb.add(hp + "proj_out.weight", {cup2, D});
            cb.add(hp + "proj_out.bias",   {cup2});
        }
        cb.add(pre + "iou_prediction_head.proj_in.weight",  {8, D});
        cb.add(pre + "iou_prediction_head.proj_in.bias",    {8});
        cb.add(pre + "iou_prediction_head.layers.0.weight", {8, 8});
        cb.add(pre + "iou_prediction_head.layers.0.bias",   {8});
        cb.add(pre + "iou_prediction_head.proj_out.weight", {K, 8});
        cb.add(pre + "iou_prediction_head.proj_out.bias",   {K});
    }

    const std::string path = "sam_end_to_end_test.safetensors";
    st::write_file(path, cb.entries);

    Sam sam(cfg);
    sam.load_file(path);

    // A small non-square RGB image (so the resize / crop / unresize matters).
    const int W = 50, H = 40;
    std::vector<uint8_t> img(static_cast<std::size_t>(W) * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const std::size_t o = (static_cast<std::size_t>(y) * W + x) * 3;
            img[o + 0] = static_cast<uint8_t>((x * 5) & 0xff);
            img[o + 1] = static_cast<uint8_t>((y * 7) & 0xff);
            img[o + 2] = static_cast<uint8_t>(((x + y) * 3) & 0xff);
        }

    check(!sam.has_image(), "no image before set_image");
    sam.set_image(img.data(), W, H, 3);
    check(sam.has_image(), "image set");

    // ── Point prompt (multimask): 3 masks at the original 50x40 resolution. ──
    const std::vector<std::array<float, 2>> pt = {{25.0f, 20.0f}};
    const std::vector<int> pt_labels = {1};
    Segmentation seg = sam.segment(pt, pt_labels, {}, /*multimask=*/true);
    {
        check(seg.num == K - 1, "multimask returns 3 masks");
        check(seg.height == H && seg.width == W, "masks at original resolution");
        check(seg.logits.size() ==
              static_cast<std::size_t>(seg.num) * H * W, "logits buffer size");
        check(static_cast<int>(seg.iou.size()) == seg.num, "one iou per mask");
        bool finite = true;
        for (float v : seg.logits) if (!std::isfinite(v)) finite = false;
        for (float v : seg.iou)    if (!std::isfinite(v)) finite = false;
        check(finite, "outputs finite");
        check(seg.best() >= 0 && seg.best() < seg.num, "best() in range");
    }

    // ── Box prompt (single mask). ──
    {
        Segmentation bseg = sam.segment({}, {}, {{5.0f, 5.0f, 45.0f, 35.0f}},
                                        /*multimask=*/false);
        check(bseg.num == 1 && bseg.height == H && bseg.width == W,
              "box single-mask shape");
        bool finite = true;
        for (float v : bseg.logits) if (!std::isfinite(v)) finite = false;
        check(finite, "box mask finite");
    }

    // ── Full-pipeline CUDA parity: the same image + point prompt, run with the
    //    whole model migrated to the GPU (set_image encodes on-device, segment
    //    decodes on-device; mask postprocess is host either way). Skipped
    //    cleanly when no GPU is present. ──
    brotensor::init();
    if (brotensor::is_available(brotensor::Device::CUDA)) {
        Sam sam_gpu(cfg);
        sam_gpu.load_file(path);
        sam_gpu.to(brotensor::Device::CUDA);
        check(sam_gpu.device() == brotensor::Device::CUDA, "Sam migrated to CUDA");
        sam_gpu.set_image(img.data(), W, H, 3);
        Segmentation gseg = sam_gpu.segment(pt, pt_labels, {}, /*multimask=*/true);
        check(gseg.num == seg.num && gseg.height == H && gseg.width == W,
              "GPU segmentation shape matches CPU");
        float worst = 0.0f;
        for (std::size_t i = 0;
             i < seg.logits.size() && i < gseg.logits.size(); ++i)
            worst = std::max(worst, std::fabs(seg.logits[i] - gseg.logits[i]));
        if (worst > 1e-2f) {
            std::fprintf(stderr, "FAIL: CPU/CUDA SAM logit diff %g > 1e-2\n", worst);
            ++failures;
        }
        std::printf("sam (end-to-end): CUDA parity max abs diff %g\n", worst);
    } else {
        std::printf("sam (end-to-end): no CUDA device, GPU parity skipped\n");
    }

    std::remove(path.c_str());

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("sam (end-to-end): all checks passed\n");
    return 0;
}
