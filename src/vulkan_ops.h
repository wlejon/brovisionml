#pragma once
//
// Vulkan twins of brovisionml's own GPU kernels (src/vulkan_ops.cpp, GLSL in
// src/vulkan/): the DSINE NRN ops (dsine_ops.cu) and the DPT bicubic
// preprocessor (dpt_preprocess_gpu.cu). Same contracts as the CUDA / HIP
// entry points; compiled only with BROTENSOR_WITH_VULKAN
// (BROVISIONML_WITH_VULKAN). The DSINE geometry runs in double when the device
// has shaderFloat64 (as on the CPU and CUDA), else in float (as on Metal).

#include "brotensor/tensor.h"

#include <cstdint>

namespace brovisionml::dsine::detail {

void ray_relu_vulkan(brotensor::Tensor& normal, const brotensor::Tensor& ray, int H, int W);

void angmf_propagate_vulkan(const brotensor::Tensor& pred_norm, const brotensor::Tensor& prob,
                            const brotensor::Tensor& xy, const brotensor::Tensor& angle,
                            const brotensor::Tensor& ray,
                            double fu, double cu, double fv, double cv,
                            int H, int W, brotensor::Tensor& out);

}  // namespace brovisionml::dsine::detail

namespace brovisionml::dpt::detail {

void dpt_preprocess_vulkan(const std::uint8_t* host_rgb, int in_w, int in_h, int channels,
                           brotensor::Device dev, int out_w, int out_h,
                           brotensor::Tensor& out_pixels);

}  // namespace brovisionml::dpt::detail
