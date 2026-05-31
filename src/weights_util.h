#pragma once

// Shared private weight-loading helpers, model-neutral. NOT a public header — it
// lives in src/ and is included only by the .cpp files of the model families
// here (dinov2, dpt, …). Every weight is widened to host FP32 regardless of the
// checkpoint's on-disk dtype (F32 / F16 / BF16), so a module loads on the host
// and migrates with to(Device) afterwards. (The SAM modules carry an older,
// sam-namespaced copy of these same helpers; new models use this one.)

#include "brotensor/tensor.h"
#include "brotensor/safetensors.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brovisionml::detail {

namespace st = brotensor::safetensors;
using brotensor::Tensor;

// Widen a flat element range [elem_off, elem_off + rows*cols) of a safetensors
// view (F32 / F16 / BF16) into a fresh host FP32 (rows, cols) tensor. The
// element offset lets a packed block be sliced without a second copy. `who` is
// the caller's error prefix (e.g. "dinov2::Backbone: ").
inline Tensor load_f32(const st::TensorView& v, std::size_t elem_off,
                       int rows, int cols,
                       const std::string& who, const std::string& label) {
    const std::size_t need = static_cast<std::size_t>(rows) * cols;
    const std::size_t have = static_cast<std::size_t>(v.numel());
    if (elem_off + need > have)
        throw std::runtime_error(
            who + label + ": range [" + std::to_string(elem_off) + ", " +
            std::to_string(elem_off + need) + ") exceeds tensor numel " +
            std::to_string(have) + " (key '" + v.name + "')");

    Tensor out = Tensor::mat(rows, cols);  // host FP32, zero-filled
    float* dst = out.host_f32_mut();
    switch (v.dtype) {
        case st::Dtype::F32: {
            const float* src = reinterpret_cast<const float*>(v.data) + elem_off;
            for (std::size_t i = 0; i < need; ++i) dst[i] = src[i];
            break;
        }
        case st::Dtype::F16: {
            const uint16_t* src =
                reinterpret_cast<const uint16_t*>(v.data) + elem_off;
            for (std::size_t i = 0; i < need; ++i)
                dst[i] = brotensor::fp16_bits_to_fp32(src[i]);
            break;
        }
        case st::Dtype::BF16: {
            const uint16_t* src =
                reinterpret_cast<const uint16_t*>(v.data) + elem_off;
            for (std::size_t i = 0; i < need; ++i)
                dst[i] = brotensor::bf16_bits_to_fp32(src[i]);
            break;
        }
        default:
            throw std::runtime_error(
                who + label + ": unsupported dtype " +
                std::string(st::dtype_name(v.dtype)) + " (key '" + v.name + "')");
    }
    return out;
}

// Fetch a named view, asserting its total element count.
inline const st::TensorView& need_view(const st::File& f, const std::string& who,
                                       const std::string& name,
                                       int64_t expect_numel) {
    const st::TensorView* v = f.find(name);
    if (!v) throw std::runtime_error(who + "missing tensor '" + name + "'");
    if (v->numel() != expect_numel)
        throw std::runtime_error(
            who + "tensor '" + name + "' has " + std::to_string(v->numel()) +
            " elements, expected " + std::to_string(expect_numel));
    return *v;
}

// Whole-tensor widen-to-FP32 with an element-count check.
inline Tensor load_whole(const st::File& f, const std::string& who,
                         const std::string& name, int rows, int cols) {
    const st::TensorView& v =
        need_view(f, who, name, static_cast<int64_t>(rows) * cols);
    return load_f32(v, 0, rows, cols, who, name);
}

}  // namespace brovisionml::detail
