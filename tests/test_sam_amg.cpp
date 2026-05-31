// SAM automatic mask generator test. Two parts:
//
//   1. A CPU structural test on a synthesized tiny checkpoint (the same
//      all-namespaces builder test_sam.cpp uses). With the IoU/stability
//      thresholds opened up so nothing is filtered on the meaningless tiny
//      weights, it drives the whole AMG pipeline — point grid, per-point decode,
//      binarize, box-NMS, crop layers, and the min-region cleanup path — and
//      asserts the structural invariants of every returned mask (resolution,
//      area == popcount, bbox/point/crop-box bounds, area-sorted order).
//
//   2. A weights-gated correctness run: when a real facebook/sam-vit-base
//      checkpoint is present under weights/, segment-everything a procedurally
//      rendered disk and assert one of the generated masks recovers it with high
//      IoU. Skips cleanly (exit 0) when the checkpoint is absent.

#define _CRT_SECURE_NO_WARNINGS  // std::getenv, matching the other tests

#include "brovisionml/sam_amg.h"

#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

namespace st = brotensor::safetensors;
using namespace brovisionml::sam;

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

// Same synthetic checkpoint builder as test_sam.cpp: deterministic small values,
// identity layer-norms.
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

SamConfig tiny_config() {
    SamConfig cfg;
    cfg.encoder.img_size    = 64;
    cfg.encoder.patch_size  = 16;
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
    return cfg;
}

// Write a tiny but architecturally complete checkpoint, mirroring test_sam.cpp.
void write_tiny_checkpoint(const std::string& path, const SamConfig& cfg) {
    const int D = 8, g = 4, hd = 4, md = 16, out = 8, p = 16;
    const int K = cfg.decoder.num_mask_tokens();
    const int cint = D / 2;
    const int F = D / 2;
    const int mic = cfg.prompt.mask_input_channels / 4;
    const int cup1 = D / 4, cup2 = D / 8;

    CheckpointBuilder cb;
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
    {
        const std::string pre = "prompt_encoder.";
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
    st::write_file(path, cb.entries);
}

// Assert every structural invariant of a generated mask set for a W×H image.
void check_masks(const std::vector<GeneratedMask>& masks, int W, int H,
                 const char* tag) {
    long long prev_area = -1;
    bool sorted = true;
    for (const GeneratedMask& m : masks) {
        check(m.width == W && m.height == H, "mask at original resolution");
        check(m.mask.size() == static_cast<std::size_t>(W) * H, "mask buffer size");

        long long popcount = 0;
        for (uint8_t v : m.mask) {
            check(v == 0 || v == 1, "mask is binary");
            popcount += v;
        }
        check(popcount == m.area, "area == foreground popcount");

        if (m.area > 0) {
            check(m.bbox[0] >= 0 && m.bbox[1] >= 0, "bbox origin in image");
            check(m.bbox[0] + m.bbox[2] <= W && m.bbox[1] + m.bbox[3] <= H,
                  "bbox within image");
            check(m.bbox[2] > 0 && m.bbox[3] > 0, "bbox has positive size");
        }
        check(m.point[0] >= 0.0f && m.point[0] <= static_cast<float>(W) &&
              m.point[1] >= 0.0f && m.point[1] <= static_cast<float>(H),
              "point in image");
        check(m.crop_box[0] >= 0 && m.crop_box[1] >= 0 &&
              m.crop_box[0] + m.crop_box[2] <= W &&
              m.crop_box[1] + m.crop_box[3] <= H && m.crop_box[2] > 0 &&
              m.crop_box[3] > 0, "crop_box within image");

        if (prev_area >= 0 && m.area > prev_area) sorted = false;
        prev_area = m.area;
    }
    check(sorted, "masks sorted by descending area");
    std::printf("  [%s] %zu mask(s)\n", tag, masks.size());
}

float disk_iou(const GeneratedMask& m, const std::vector<uint8_t>& gt) {
    long long inter = 0, uni = 0;
    for (std::size_t i = 0; i < gt.size(); ++i) {
        const bool a = m.mask[i] != 0, b = gt[i] != 0;
        if (a && b) ++inter;
        if (a || b) ++uni;
    }
    return uni ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

std::vector<uint8_t> make_disk_image(int W, int H, int cx, int cy, int r,
                                     std::vector<uint8_t>& gt) {
    std::vector<uint8_t> img(static_cast<std::size_t>(W) * H * 3);
    gt.assign(static_cast<std::size_t>(W) * H, 0);
    const long long r2 = static_cast<long long>(r) * r;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const std::size_t px = static_cast<std::size_t>(y) * W + x;
            const long long dx = x - cx, dy = y - cy;
            const bool inside = dx * dx + dy * dy <= r2;
            const std::size_t o = px * 3;
            if (inside) {
                img[o + 0] = img[o + 1] = img[o + 2] = 245;
                gt[px] = 1;
            } else {
                img[o + 0] = static_cast<uint8_t>((x * 5) & 0xff);
                img[o + 1] = static_cast<uint8_t>((y * 7) & 0xff);
                img[o + 2] = static_cast<uint8_t>(((x + y) * 3) & 0xff);
            }
        }
    return img;
}

}  // namespace

int main() {
    brotensor::init();

    // ── Part 1: CPU structural test on the synthetic checkpoint. ──
    {
        const SamConfig cfg = tiny_config();
        const std::string path = "sam_amg_test.safetensors";
        write_tiny_checkpoint(path, cfg);

        Sam sam(cfg);
        sam.load_file(path);

        const int W = 50, H = 40;
        std::vector<uint8_t> img(static_cast<std::size_t>(W) * H * 3);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const std::size_t o = (static_cast<std::size_t>(y) * W + x) * 3;
                img[o + 0] = static_cast<uint8_t>((x * 5) & 0xff);
                img[o + 1] = static_cast<uint8_t>((y * 7) & 0xff);
                img[o + 2] = static_cast<uint8_t>(((x + y) * 3) & 0xff);
            }

        // Open the thresholds so the tiny weights' arbitrary scores don't filter
        // everything; keep the grid small for speed.
        AmgConfig amg;
        amg.points_per_side        = 4;
        amg.pred_iou_thresh        = 0.0f;
        amg.stability_score_thresh = 0.0f;

        {
            AutomaticMaskGenerator g(sam, amg);
            std::vector<GeneratedMask> masks = g.generate(img.data(), W, H, 3);
            check(!masks.empty(), "single-crop generate returns masks");
            for (const GeneratedMask& m : masks)
                check(m.crop_box == (std::array<int, 4>{0, 0, W, H}),
                      "single-crop masks carry the full-image crop box");
            check_masks(masks, W, H, "single-crop");
        }

        // Crop layers: exercises the crop pyramid + across-crop NMS path.
        {
            AmgConfig c = amg;
            c.crop_n_layers = 1;
            AutomaticMaskGenerator g(sam, c);
            std::vector<GeneratedMask> masks = g.generate(img.data(), W, H, 3);
            check_masks(masks, W, H, "crop-layers");
        }

        // min_mask_region_area: exercises connected-components cleanup + re-NMS.
        {
            AmgConfig c = amg;
            c.min_mask_region_area = 8;
            AutomaticMaskGenerator g(sam, c);
            std::vector<GeneratedMask> masks = g.generate(img.data(), W, H, 3);
            check_masks(masks, W, H, "min-region");
        }

        std::remove(path.c_str());
        std::printf("sam_amg (synthetic): structural checks done\n");
    }

    // ── Part 2: weights-gated correctness on a real checkpoint. ──
    {
        const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
        const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
        const std::string path = base + "/sam-vit-base/model.safetensors";

        if (!file_exists(path)) {
            std::printf("sam_amg (real weights): no checkpoint under '%s' "
                        "(run scripts/download-weights.sh sam-vit-base); skipping\n",
                        base.empty() ? "<unset>" : base.c_str());
        } else {
            Sam sam(SamConfig::vit_b());
            sam.load_file(path);
            if (brotensor::is_available(brotensor::Device::CUDA))
                sam.to(brotensor::Device::CUDA);

            const int W = 320, H = 256, cx = 160, cy = 128, r = 80;
            std::vector<uint8_t> gt;
            const std::vector<uint8_t> img = make_disk_image(W, H, cx, cy, r, gt);

            AmgConfig amg;
            amg.points_per_side = 16;  // 256 grid points; modest for runtime
            AutomaticMaskGenerator g(sam, amg);
            std::vector<GeneratedMask> masks = g.generate(img.data(), W, H, 3);

            check(!masks.empty(), "real-weights generate finds masks");
            check_masks(masks, W, H, "real-weights");

            float best = 0.0f;
            for (const GeneratedMask& m : masks)
                best = std::max(best, disk_iou(m, gt));
            std::printf("  real-weights: %zu masks, best disk IoU=%.3f\n",
                        masks.size(), best);
            check(best >= 0.85f, "a generated mask recovers the disk (IoU)");
        }
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("sam_amg: all checks passed\n");
    return 0;
}
