// SAM prompt encoder: structural test with a synthesized tiny checkpoint. We
// write a HF-named prompt_encoder.* checkpoint, load it through the real loader,
// and exercise point / box / mask prompts plus the dense image-grid positional
// encoding. Assertions cover token counts and the two exact-value properties
// that pin the wiring: a label -1 sparse token equals not_a_point_embed, and
// the no-mask dense embedding is no_mask_embed broadcast across the grid.
#include "brovisionml/sam_prompt_encoder.h"

#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

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
            else       buf[i] = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
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

    PromptEncoderConfig cfg;
    cfg.hidden_size          = 8;    // num_pos_feats = 4 -> pe is 8 wide
    cfg.image_embedding_size = 4;    // grid 4x4
    cfg.input_image_size     = 16;
    cfg.mask_input_channels  = 16;   // mic = 4 -> conv2 width 16
    cfg.num_point_embeddings = 4;

    const int H = cfg.hidden_size;   // 8
    const int F = cfg.num_pos_feats();  // 4
    const int g = cfg.grid();        // 4
    const int mic = cfg.mask_input_channels / 4;  // 4

    CheckpointBuilder cb;
    const std::string pre = "prompt_encoder.";
    // Tied weight stored at the top level in real HF checkpoints (see loader).
    cb.add("shared_image_embedding.positional_embedding", {2, F});
    for (int i = 0; i < 4; ++i)
        cb.add(pre + "point_embed." + std::to_string(i) + ".weight", {1, H});
    cb.add(pre + "not_a_point_embed.weight", {1, H});
    cb.add(pre + "no_mask_embed.weight",     {1, H});
    cb.add(pre + "mask_embed.conv1.weight",       {mic, 1, 2, 2});
    cb.add(pre + "mask_embed.conv1.bias",         {mic});
    cb.add(pre + "mask_embed.layer_norm1.weight", {mic});
    cb.add(pre + "mask_embed.layer_norm1.bias",   {mic});
    cb.add(pre + "mask_embed.conv2.weight",       {mic * 4, mic, 2, 2});
    cb.add(pre + "mask_embed.conv2.bias",         {mic * 4});
    cb.add(pre + "mask_embed.layer_norm2.weight", {mic * 4});
    cb.add(pre + "mask_embed.layer_norm2.bias",   {mic * 4});
    cb.add(pre + "mask_embed.conv3.weight",       {H, mic * 4, 1, 1});
    cb.add(pre + "mask_embed.conv3.bias",         {H});

    const std::string path = "sam_prompt_encoder_test.safetensors";
    st::write_file(path, cb.entries);

    PromptEncoder enc(cfg);
    enc.load_file(path);

    // dense_pe: (1, H*g*g), finite.
    {
        brotensor::Tensor dpe = enc.dense_pe();
        check(dpe.rows == 1 && dpe.cols == H * g * g, "dense_pe shape");
        bool finite = true;
        const float* d = dpe.host_f32();
        for (int i = 0; i < dpe.cols; ++i) if (!std::isfinite(d[i])) finite = false;
        check(finite, "dense_pe finite");
    }

    // Recover the loaded no_mask / not_a_point vectors from the raw file so we
    // can assert exact-value wiring below.
    std::vector<float> no_mask(H), not_a_point(H);
    {
        st::File f = st::File::open(path);
        const auto* nm = f.find(pre + "no_mask_embed.weight");
        const auto* nap = f.find(pre + "not_a_point_embed.weight");
        const float* nmp  = reinterpret_cast<const float*>(nm->data);
        const float* napp = reinterpret_cast<const float*>(nap->data);
        for (int c = 0; c < H; ++c) { no_mask[c] = nmp[c]; not_a_point[c] = napp[c]; }
    }

    // ── Points only: 2 clicks -> 2 + 1 pad token = 3 sparse rows. ──
    {
        PromptInput p;
        p.points = {{4.0f, 4.0f}, {8.0f, 8.0f}};
        p.labels = {1, 0};
        PromptEmbeddings e = enc.encode(p);
        check(e.sparse.rows == 3 && e.sparse.cols == H, "points-only sparse shape");

        // The pad row (index 2) must equal not_a_point_embed exactly.
        const float* sp = e.sparse.host_f32();
        bool pad_ok = true;
        for (int c = 0; c < H; ++c)
            if (std::fabs(sp[2 * H + c] - not_a_point[c]) > 1e-6f) pad_ok = false;
        check(pad_ok, "pad token == not_a_point_embed");

        // Dense (no mask) is (1, H*g*g) with each channel constant == no_mask[c].
        check(e.dense.rows == 1 && e.dense.cols == H * g * g, "no-mask dense shape");
        const float* dn = e.dense.host_f32();
        bool bc_ok = true;
        const int plane = g * g;
        for (int c = 0; c < H; ++c)
            for (int s = 0; s < plane; ++s)
                if (std::fabs(dn[c * plane + s] - no_mask[c]) > 1e-6f) bc_ok = false;
        check(bc_ok, "no-mask dense == broadcast(no_mask_embed)");
    }

    // ── Box only: 1 box -> 2 corner tokens, no pad. ──
    {
        PromptInput p;
        p.boxes = {{2.0f, 2.0f, 10.0f, 10.0f}};
        PromptEmbeddings e = enc.encode(p);
        check(e.sparse.rows == 2 && e.sparse.cols == H, "box-only sparse shape");
    }

    // ── Point + box: 1 click + 1 box -> 1 + 2 = 3, no pad. ──
    {
        PromptInput p;
        p.points = {{5.0f, 5.0f}};
        p.labels = {1};
        p.boxes  = {{1.0f, 1.0f, 9.0f, 9.0f}};
        PromptEmbeddings e = enc.encode(p);
        check(e.sparse.rows == 3 && e.sparse.cols == H, "point+box sparse shape");
    }

    // ── Mask prompt: 16x16 single-channel mask downscales to the 4x4 grid. ──
    {
        PromptInput p;
        p.mask_h = 4 * g; p.mask_w = 4 * g;
        p.mask.resize(static_cast<std::size_t>(p.mask_h) * p.mask_w);
        for (std::size_t i = 0; i < p.mask.size(); ++i)
            p.mask[i] = std::sin(static_cast<float>(i) * 0.05f);
        PromptEmbeddings e = enc.encode(p);
        check(e.dense.rows == 1 && e.dense.cols == H * g * g, "mask dense shape");
        bool finite = true;
        const float* dn = e.dense.host_f32();
        for (int i = 0; i < e.dense.cols; ++i)
            if (!std::isfinite(dn[i])) finite = false;
        check(finite, "mask dense finite");
        // No sparse prompts -> empty sparse tensor.
        check(e.sparse.rows == 0 && e.sparse.cols == 0, "mask-only sparse empty");
    }

    std::remove(path.c_str());

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("sam_prompt_encoder: all checks passed\n");
    return 0;
}
