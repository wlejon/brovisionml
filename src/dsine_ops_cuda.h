#pragma once

// Internal: CUDA launch wrappers for the DSINE NRN device ops. Defined in
// dsine_ops.cu (compiled only with BROTENSOR_WITH_CUDA); declared here so the
// device dispatch in dsine_ops.cpp can call them. Inputs/outputs are CUDA-
// resident FP32 brotensor tensors; kernels launch on the default stream — the
// same stream brotensor's own ops use — so no cross-stream sync is needed.

#include "brotensor/tensor.h"

namespace brovisionml::dsine::detail {

void ray_relu_cuda(brotensor::Tensor& normal, const brotensor::Tensor& ray,
                   int H, int W);

void angmf_propagate_cuda(const brotensor::Tensor& pred_norm,
                          const brotensor::Tensor& prob,
                          const brotensor::Tensor& xy,
                          const brotensor::Tensor& angle,
                          const brotensor::Tensor& ray,
                          double fu, double cu, double fv, double cv,
                          int H, int W,
                          brotensor::Tensor& out);

}  // namespace brovisionml::dsine::detail
