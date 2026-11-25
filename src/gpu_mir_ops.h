#pragma once
// Backend-neutral GPU Brass MIR ops for brovisionml (Vulkan SPIR-V, CUDA PTX,
// Metal MSL), dispatched through brotensor::interop without nvcc, glslc, or
// Objective-C++.

#include <brotensor/tensor.h>

#include <cstdint>
#include <string>
#include <vector>

namespace brovisionml::dpt::detail {

void dpt_preprocess_gpu(const std::uint8_t* host_rgb, int in_w, int in_h, int channels,
                        brotensor::Device dev, int out_w, int out_h,
                        brotensor::Tensor& out_pixels);

}  // namespace brovisionml::dpt::detail

namespace brovisionml::dsine::detail {

void ray_relu_gpu(brotensor::Tensor& normal, const brotensor::Tensor& ray, int H, int W);

void angmf_propagate_gpu(const brotensor::Tensor& pred_norm, const brotensor::Tensor& prob,
                         const brotensor::Tensor& xy, const brotensor::Tensor& angle,
                         const brotensor::Tensor& ray,
                         double fu, double cu, double fv, double cv,
                         int H, int W, brotensor::Tensor& out);

}  // namespace brovisionml::dsine::detail

namespace brovisionml::detail {

struct MirAuditShader {
    std::string name;
    std::vector<std::uint32_t> words;
};

/// Compiles all 5 brovisionml GPU MIR kernels to SPIR-V for static formal
/// verification in tests/test_vulkan_shader_audit.cpp, and verifies PTX (all 5)
/// and MSL (all 3 FP32) lowering succeed cleanly.
std::vector<MirAuditShader> compile_mir_spirv_for_audit();

}  // namespace brovisionml::detail
