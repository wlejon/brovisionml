#include "brovisionml/sam2_config.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brovisionml::sam2 {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("sam2::Sam2Config: " + msg);
}

// ─── A small, complete JSON reader ──────────────────────────────────────────
//
// SAM 2's config.json nests the Hiera trunk two objects deep and reuses key
// names (`hidden_size` appears at four levels), so a key-scan reader cannot
// tell them apart. This parses the whole document into a tree.

struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    const Json* get(const std::string& key) const {
        if (kind != Object) return nullptr;
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }
};

struct Parser {
    const std::string& s;
    std::size_t i = 0;

    void ws() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    char peek() { ws(); return i < s.size() ? s[i] : '\0'; }
    void expect(char c) {
        if (peek() != c) fail(std::string("config.json: expected '") + c + "' at byte " +
                              std::to_string(i));
        ++i;
    }
    std::string parse_string() {
        expect('"');
        std::string out;
        while (i < s.size() && s[i] != '"') {
            char c = s[i++];
            if (c == '\\' && i < s.size()) {
                const char e = s[i++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': i += 4; out += '?'; break;   // keys/values here are ASCII
                    default:  out += e; break;
                }
            } else {
                out += c;
            }
        }
        if (i >= s.size()) fail("config.json: unterminated string");
        ++i;
        return out;
    }
    Json parse_value(int depth) {
        if (depth > 64) fail("config.json: nested too deeply");
        Json v;
        const char c = peek();
        if (c == '{') {
            ++i;
            v.kind = Json::Object;
            if (peek() == '}') { ++i; return v; }
            for (;;) {
                std::string key = parse_string();
                expect(':');
                v.obj[key] = parse_value(depth + 1);
                const char d = peek();
                if (d == ',') { ++i; continue; }
                if (d == '}') { ++i; break; }
                fail("config.json: expected ',' or '}' at byte " + std::to_string(i));
            }
        } else if (c == '[') {
            ++i;
            v.kind = Json::Array;
            if (peek() == ']') { ++i; return v; }
            for (;;) {
                v.arr.push_back(parse_value(depth + 1));
                const char d = peek();
                if (d == ',') { ++i; continue; }
                if (d == ']') { ++i; break; }
                fail("config.json: expected ',' or ']' at byte " + std::to_string(i));
            }
        } else if (c == '"') {
            v.kind = Json::String;
            v.str = parse_string();
        } else if (s.compare(i, 4, "true") == 0) {
            i += 4; v.kind = Json::Bool; v.b = true;
        } else if (s.compare(i, 5, "false") == 0) {
            i += 5; v.kind = Json::Bool; v.b = false;
        } else if (s.compare(i, 4, "null") == 0) {
            i += 4; v.kind = Json::Null;
        } else {
            char* end = nullptr;
            v.num = std::strtod(s.c_str() + i, &end);
            if (end == s.c_str() + i) fail("config.json: bad value at byte " + std::to_string(i));
            i = static_cast<std::size_t>(end - s.c_str());
            v.kind = Json::Number;
        }
        return v;
    }
};

void read_int(const Json* o, const char* key, int& out) {
    if (const Json* v = o ? o->get(key) : nullptr; v && v->kind == Json::Number)
        out = static_cast<int>(v->num);
}
void read_float(const Json* o, const char* key, float& out) {
    if (const Json* v = o ? o->get(key) : nullptr; v && v->kind == Json::Number)
        out = static_cast<float>(v->num);
}
void read_bool(const Json* o, const char* key, bool& out) {
    if (const Json* v = o ? o->get(key) : nullptr; v && v->kind == Json::Bool)
        out = v->b;
}
void read_ints(const Json* o, const char* key, std::vector<int>& out) {
    const Json* v = o ? o->get(key) : nullptr;
    if (!v || v->kind != Json::Array) return;
    std::vector<int> r;
    for (const Json& e : v->arr) {
        if (e.kind != Json::Number) fail(std::string("config.json: '") + key + "' is not an int array");
        r.push_back(static_cast<int>(e.num));
    }
    out = std::move(r);
}
void read_pair(const Json* o, const char* key, std::array<int, 2>& out) {
    std::vector<int> v;
    read_ints(o, key, v);
    if (v.size() == 2) out = {v[0], v[1]};
}

}  // namespace

// ─── Presets ────────────────────────────────────────────────────────────────

Sam2Config Sam2Config::base_plus() { return Sam2Config{}; }

Sam2Config Sam2Config::tiny() {
    Sam2Config c;
    c.hiera.blocks_per_stage      = {1, 2, 7, 2};
    c.hiera.embed_dim_per_stage   = {96, 192, 384, 768};
    c.hiera.heads_per_stage       = {1, 2, 4, 8};
    c.hiera.window_size_per_stage = {8, 4, 14, 7};
    c.hiera.global_attention_blocks = {5, 7, 9};
    c.hiera.pos_embed_bg = {7, 7};
    c.backbone_channel_list = {768, 384, 192, 96};
    return c;
}

Sam2Config Sam2Config::small() {
    Sam2Config c = tiny();
    c.hiera.blocks_per_stage        = {1, 2, 11, 2};
    c.hiera.global_attention_blocks = {7, 10, 13};
    return c;
}

Sam2Config Sam2Config::large() {
    Sam2Config c;
    c.hiera.blocks_per_stage      = {2, 6, 36, 4};
    c.hiera.embed_dim_per_stage   = {144, 288, 576, 1152};
    c.hiera.heads_per_stage       = {2, 4, 8, 16};
    c.hiera.window_size_per_stage = {8, 4, 16, 8};
    c.hiera.global_attention_blocks = {23, 33, 43};
    c.hiera.pos_embed_bg = {7, 7};
    c.backbone_channel_list = {1152, 576, 288, 144};
    return c;
}

// ─── config.json ────────────────────────────────────────────────────────────

Sam2Config Sam2Config::from_json(const std::string& text) {
    Parser p{text};
    const Json root = p.parse_value(0);
    if (root.kind != Json::Object) fail("config.json: top level is not an object");

    Sam2Config c;
    read_int(&root, "image_size", c.image_size);

    const Json* vis = root.get("vision_config");
    const Json* bb  = vis ? vis->get("backbone_config") : nullptr;
    if (bb) {
        HieraConfig& h = c.hiera;
        read_ints(bb, "blocks_per_stage", h.blocks_per_stage);
        read_ints(bb, "embed_dim_per_stage", h.embed_dim_per_stage);
        read_ints(bb, "num_attention_heads_per_stage", h.heads_per_stage);
        read_ints(bb, "window_size_per_stage", h.window_size_per_stage);
        read_ints(bb, "global_attention_blocks", h.global_attention_blocks);
        read_pair(bb, "window_positional_embedding_background_size", h.pos_embed_bg);
        read_int(bb, "num_query_pool_stages", h.num_query_pool_stages);
        std::vector<int> qs;
        read_ints(bb, "query_stride", qs);
        if (!qs.empty()) h.query_stride = qs[0];
        std::vector<int> v;
        read_ints(bb, "patch_kernel_size", v);  if (!v.empty()) h.patch_kernel = v[0];
        v.clear(); read_ints(bb, "patch_stride", v);  if (!v.empty()) h.patch_stride = v[0];
        v.clear(); read_ints(bb, "patch_padding", v); if (!v.empty()) h.patch_padding = v[0];
        read_float(bb, "mlp_ratio", h.mlp_ratio);
        read_float(bb, "layer_norm_eps", h.layer_norm_eps);
    }
    if (vis) {
        read_ints(vis, "backbone_channel_list", c.backbone_channel_list);
        read_ints(vis, "fpn_top_down_levels", c.fpn_top_down_levels);
        read_int(vis, "fpn_hidden_size", c.fpn_hidden_size);
        read_int(vis, "num_feature_levels", c.num_feature_levels);
    }
    if (const Json* pe = root.get("prompt_encoder_config")) {
        read_int(pe, "hidden_size", c.hidden_size);
        read_int(pe, "mask_input_channels", c.mask_input_channels);
        read_float(pe, "layer_norm_eps", c.prompt_ln_eps);
    }
    if (const Json* md = root.get("mask_decoder_config")) {
        read_int(md, "num_attention_heads", c.decoder_heads);
        read_int(md, "num_hidden_layers", c.decoder_layers);
        read_int(md, "mlp_dim", c.decoder_mlp_dim);
        read_int(md, "attention_downsample_rate", c.attention_downsample);
        read_int(md, "iou_head_depth", c.iou_head_depth);
        read_int(md, "iou_head_hidden_dim", c.iou_head_hidden);
        read_int(md, "num_multimask_outputs", c.num_multimask_outputs);
        read_bool(md, "dynamic_multimask_via_stability", c.dynamic_multimask_via_stability);
        read_float(md, "dynamic_multimask_stability_delta", c.stability_delta);
        read_float(md, "dynamic_multimask_stability_thresh", c.stability_thresh);
    }
    read_int(&root, "num_maskmem", c.num_maskmem);
    read_int(&root, "memory_encoder_output_channels", c.mem_dim);
    read_int(&root, "memory_attention_num_layers", c.mem_attn_layers);
    read_int(&root, "memory_attention_num_attention_heads", c.mem_attn_heads);
    read_int(&root, "memory_attention_feed_forward_hidden_size", c.mem_attn_ff);
    read_int(&root, "memory_attention_downsample_rate", c.mem_attn_downsample);
    read_float(&root, "memory_attention_rope_theta", c.rope_theta);
    read_pair(&root, "memory_attention_rope_feat_sizes", c.rope_feat_sizes);
    read_int(&root, "memory_fuser_num_layers", c.fuser_layers);
    read_int(&root, "memory_fuser_intermediate_dim", c.fuser_intermediate);
    read_int(&root, "memory_fuser_kernel_size", c.fuser_kernel);
    read_int(&root, "mask_downsampler_kernel_size", c.mask_down_kernel);
    read_int(&root, "mask_downsampler_stride", c.mask_down_stride);
    read_int(&root, "mask_downsampler_padding", c.mask_down_padding);
    read_int(&root, "mask_downsampler_total_stride", c.mask_down_total_stride);
    read_int(&root, "max_object_pointers_in_encoder", c.max_obj_ptrs);
    read_float(&root, "sigmoid_scale_for_mem_enc", c.sigmoid_scale_for_mem_enc);
    read_float(&root, "sigmoid_bias_for_mem_enc", c.sigmoid_bias_for_mem_enc);
    read_bool(&root, "multimask_output_in_sam", c.multimask_output_in_sam);
    read_bool(&root, "multimask_output_for_tracking", c.multimask_output_for_tracking);
    read_int(&root, "multimask_min_pt_num", c.multimask_min_pt_num);
    read_int(&root, "multimask_max_pt_num", c.multimask_max_pt_num);
    read_bool(&root, "enable_temporal_pos_encoding_for_object_pointers",
              c.temporal_pos_for_obj_ptrs);
    read_bool(&root, "enable_occlusion_spatial_embedding", c.occlusion_spatial_embedding);

    // Structural sanity: everything below indexes these vectors by stage.
    const HieraConfig& h = c.hiera;
    const std::size_t S = h.blocks_per_stage.size();
    if (S == 0 || h.embed_dim_per_stage.size() != S || h.heads_per_stage.size() != S ||
        h.window_size_per_stage.size() != S || c.backbone_channel_list.size() != S)
        fail("config.json: Hiera per-stage arrays disagree in length");
    for (std::size_t s = 0; s < S; ++s) {
        if (h.heads_per_stage[s] <= 0 || h.embed_dim_per_stage[s] % h.heads_per_stage[s] != 0)
            fail("config.json: stage width not divisible by its head count");
        if (c.backbone_channel_list[s] != h.embed_dim_per_stage[S - 1 - s])
            fail("config.json: backbone_channel_list does not mirror embed_dim_per_stage");
    }
    if (c.image_size % 16 != 0) fail("config.json: image_size must be a multiple of 16");
    if (c.hidden_size % c.mem_dim != 0) fail("config.json: hidden size not a multiple of mem dim");
    return c;
}

Sam2Config Sam2Config::from_json_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) fail("could not open '" + path + "'");
    const std::streamoff n = f.tellg();
    std::string buf(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
    f.seekg(0);
    if (n > 0) f.read(buf.data(), n);
    return from_json(buf);
}

}  // namespace brovisionml::sam2
