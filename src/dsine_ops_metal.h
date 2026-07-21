#pragma once

// Internal: Metal launch wrappers for the DSINE NRN device ops. Defined in
// dsine_ops.mm (compiled only with BROTENSOR_WITH_METAL); declared here so the
// device dispatch in dsine_ops.cpp can call them. Mirrors dsine_ops_cuda.h.
// Inputs/outputs are Metal-resident FP32 brotensor tensors; kernels submit on
// brotensor's shared command queue (the same queue its own ops use), so no
// cross-queue sync is needed. This header is plain C++ (no Obj-C), so the
// device-agnostic dsine_ops.cpp can include it.

#include "brotensor/tensor.h"

namespace brovisionml::dsine::detail {

void ray_relu_metal(brotensor::Tensor& normal, const brotensor::Tensor& ray,
                    int H, int W);

void angmf_propagate_metal(const brotensor::Tensor& pred_norm,
                           const brotensor::Tensor& prob,
                           const brotensor::Tensor& xy,
                           const brotensor::Tensor& angle,
                           const brotensor::Tensor& ray,
                           double fu, double cu, double fv, double cv,
                           int H, int W,
                           brotensor::Tensor& out);

}  // namespace brovisionml::dsine::detail
