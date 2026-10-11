#pragma once

// SAM 2.1 (Segment Anything 2) hyperparameters, read from the HuggingFace
// `Sam2VideoModel` config.json that ships beside model.safetensors in the
// facebook/sam2.1-hiera-{tiny,small,base-plus,large} repos. One implementation
// serves all four sizes: only the Hiera trunk differs between them (stage
// widths, depths, heads, windows, global blocks, the background position-embed
// size); the FPN neck, prompt encoder, mask decoder and the whole memory side
// are shared.

#include <array>
#include <string>
#include <vector>

namespace brovisionml::sam2 {

// Hiera trunk (HF Sam2HieraDetConfig).
struct HieraConfig {
    std::vector<int> blocks_per_stage      = {2, 3, 16, 3};
    std::vector<int> embed_dim_per_stage   = {112, 224, 448, 896};
    std::vector<int> heads_per_stage       = {2, 4, 8, 16};
    std::vector<int> window_size_per_stage = {8, 4, 14, 7};
    std::vector<int> global_attention_blocks = {12, 16, 20};
    std::array<int, 2> pos_embed_bg = {14, 14};  // window_positional_embedding_background_size
    int   num_query_pool_stages = 3;
    int   query_stride          = 2;
    int   patch_kernel  = 7;
    int   patch_stride  = 4;
    int   patch_padding = 3;
    float mlp_ratio      = 4.0f;
    float layer_norm_eps = 1e-6f;

    int num_stages() const { return static_cast<int>(blocks_per_stage.size()); }
    int total_blocks() const {
        int n = 0;
        for (int b : blocks_per_stage) n += b;
        return n;
    }
};

struct Sam2Config {
    int image_size = 1024;
    HieraConfig hiera;

    // FPN neck.
    std::vector<int> backbone_channel_list = {896, 448, 224, 112};  // deepest first
    std::vector<int> fpn_top_down_levels   = {2, 3};
    int fpn_hidden_size    = 256;
    int num_feature_levels = 3;

    // Prompt encoder + mask decoder (shared across sizes).
    int   hidden_size            = 256;
    int   decoder_heads          = 8;
    int   decoder_layers         = 2;
    int   decoder_mlp_dim        = 2048;
    int   attention_downsample   = 2;
    int   iou_head_depth         = 3;
    int   iou_head_hidden        = 256;
    int   num_multimask_outputs  = 3;
    int   mask_input_channels    = 16;
    bool  dynamic_multimask_via_stability = true;
    float stability_delta  = 0.05f;
    float stability_thresh = 0.98f;
    float prompt_ln_eps    = 1e-6f;

    // Memory side.
    int   num_maskmem         = 7;
    int   mem_dim             = 64;       // memory_encoder_output_channels
    int   mem_attn_layers     = 4;
    int   mem_attn_heads      = 1;
    int   mem_attn_ff         = 2048;
    int   mem_attn_downsample = 1;
    float rope_theta          = 10000.0f;
    std::array<int, 2> rope_feat_sizes = {64, 64};
    int   fuser_layers        = 2;
    int   fuser_intermediate  = 1024;
    int   fuser_kernel        = 7;
    int   mask_down_kernel    = 3;
    int   mask_down_stride    = 2;
    int   mask_down_padding   = 1;
    int   mask_down_total_stride = 16;
    int   max_obj_ptrs        = 16;
    float sigmoid_scale_for_mem_enc = 20.0f;
    float sigmoid_bias_for_mem_enc  = -10.0f;
    bool  multimask_output_in_sam      = true;
    bool  multimask_output_for_tracking = true;
    int   multimask_min_pt_num = 0;
    int   multimask_max_pt_num = 1;
    bool  temporal_pos_for_obj_ptrs = true;
    bool  occlusion_spatial_embedding = true;

    int grid() const { return image_size / 16; }          // 64: top feature side
    int mask_size() const { return 4 * grid(); }          // 256: low-res mask side
    int num_mask_tokens() const { return num_multimask_outputs + 1; }

    // The four published sizes (values as in their config.json).
    static Sam2Config tiny();
    static Sam2Config small();
    static Sam2Config base_plus();
    static Sam2Config large();

    // Parse a HF Sam2VideoModel / Sam2Model config.json. Keys absent from the
    // file keep the base-plus defaults above. Throws std::runtime_error on an
    // unreadable or malformed file.
    static Sam2Config from_json_file(const std::string& path);
    static Sam2Config from_json(const std::string& text);
};

}  // namespace brovisionml::sam2
