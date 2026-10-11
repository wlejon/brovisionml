#pragma once

// SAM 2 memory side: what turns the image predictor into a tracker.
//
//   * Memory encoder — a frame's predicted mask (downsampled 16x through four
//     strided convs) fused with that frame's stride-16 features (two ConvNeXt
//     blocks) and projected to a 64-channel memory, stored BF16-rounded as the
//     reference stores it.
//   * Object pointers — the decoder's chosen output token through a 3-layer
//     MLP (or the learned "no object" pointer when the object score says the
//     object is absent), split into 4 tokens of 64.
//   * Memory attention — 4 layers of RoPE self-attention over the current
//     frame, RoPE cross-attention into the concatenated memories (+ their sine
//     and temporal position encodings) and object pointers (+ their projected
//     1-D temporal encodings, no RoPE), and an MLP.
//
// Which memories and pointers take part is the tracker's business
// (sam2_tracker.cpp); this module computes.

#include "brovisionml/sam2_config.h"

#include "brotensor/tensor.h"

#include <memory>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brovisionml::sam2 {

struct MemoryWeights;

class MemoryModule {
public:
    explicit MemoryModule(const Sam2Config& cfg);
    ~MemoryModule();
    MemoryModule(MemoryModule&&) noexcept;
    MemoryModule& operator=(MemoryModule&&) noexcept;
    MemoryModule(const MemoryModule&) = delete;
    MemoryModule& operator=(const MemoryModule&) = delete;

    void load(const brotensor::safetensors::File& f);
    void to(brotensor::Device dev);
    brotensor::Device device() const { return device_; }
    bool mixed_precision() const { return fp16_; }

    // The conditioning a frame with no memory gets: top + no_memory_embedding.
    brotensor::Tensor no_memory(const brotensor::Tensor& top) const;

    // Encode one memory: `top` (grid^2, hidden) raw stride-16 tokens, `low_res`
    // (1, mask_size^2) the frame's mask logits (already -1024 where the object
    // is absent). `from_points` binarizes the mask (a prompted frame) instead
    // of taking its sigmoid. Returns (grid^2, mem_dim) FP32, BF16-rounded.
    brotensor::Tensor encode(const brotensor::Tensor& top, const brotensor::Tensor& low_res,
                             bool from_points, float object_score) const;

    // (1, hidden) object pointer from the chosen decoder token.
    brotensor::Tensor object_pointer(const brotensor::Tensor& sam_token, bool appearing) const;

    struct Memory {
        const brotensor::Tensor* features;   // (grid^2, mem_dim) from encode()
        int tpos;                            // temporal encoding row, 0..num_maskmem-1
    };
    struct Pointer {
        const brotensor::Tensor* ptr;        // (1, hidden) from object_pointer()
        float pos;                           // temporal distance / t_diff_max
    };

    // Condition the current frame on its memories: `top` / `top_pos` are the
    // frame's raw stride-16 tokens and their sine encoding. Returns
    // (grid^2, hidden) FP32 tokens for SamHeads::decode.
    brotensor::Tensor attend(const brotensor::Tensor& top, const brotensor::Tensor& top_pos,
                             const std::vector<Memory>& memories,
                             const std::vector<Pointer>& pointers) const;

private:
    Sam2Config cfg_;
    std::unique_ptr<MemoryWeights> w_;
    brotensor::Device device_ = brotensor::Device::CPU;
    bool fp16_ = false;
};

// get_1d_sine_pe: (n, dim) host FP32, sin of the first dim/2 then cos.
brotensor::Tensor sine_pe_1d(const std::vector<float>& pos, int dim);

}  // namespace brovisionml::sam2
