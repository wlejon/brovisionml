#pragma once

// DSINE NRN device ops — the two pieces of the AngMF refinement that have no
// brotensor primitive because they are surface-normal *domain* math. Per the
// kernel-promotion principle, the generic NRN mechanics (neighborhood unfold,
// channel L2-normalize, convex upsample, channel concat) already live in
// brotensor; what stays here is DSINE's rotation/clamp geometry. These are
// brovisionml's first CUDA kernels.
//
// Each op has a CPU FP32 path and, when built with BROTENSOR_WITH_CUDA, a CUDA
// kernel; the public entry points dispatch on the input tensor's device. Both
// run their per-pixel geometry in double internally (matching the reference and
// the original host port), and operate on the /8 NRN grid in NCHW layout.

#include "brotensor/tensor.h"

namespace brovisionml::dsine {

// RayReLU (submodules.RayReLU, eps=1e-2): clamp a (1, 3*H*W) NCHW unit-normal
// map to face the camera, in place. `ray` is the L2-normalized 3ch ray_8
// (1, 3*H*W) on the same device as `normal`.
//   cos   = cosine_similarity(n, ray)            (n·ray / (|n||ray|), eps 1e-8)
//   diff  = ray*(relu(cos-eps)+eps) - ray*cos
//   n_new = normalize(n + diff)
void ray_relu(brotensor::Tensor& normal, const brotensor::Tensor& ray,
              int H, int W);

// Fused AngMF propagate — the rotation-based angular mean-field update; a port
// of the per-pixel-per-neighbor loop in DSINE_v02.refine. For each of the 25
// (5x5) neighbors of every pixel: build a rotation axis from the intrinsics and
// the predicted per-neighbor xy offset, rotate the neighbor's normal by
// (axis * angle), RayReLU the result against the pixel's ray, and accumulate a
// probability-weighted sum; the output is L2-normalized over channels.
//
// All maps are NCHW on the same device. Neighbor slot n = ky*5+kx samples input
// position (clamp(y-2+ky), clamp(x-2+kx)) — F.unfold row-major, replicate pad.
//   pred_norm : (1, 3*H*W)   current normals (read-only source)
//   prob      : (1, 25*H*W)  sigmoid(prob_head)         per-neighbor weight
//   xy        : (1, 50*H*W)  RAW xy_head: [xs(25), ys(25)] — each (xs,ys) pair
//                            is L2-normalized per neighbor INSIDE this op (no
//                            brotensor op covers the strided-pair normalize)
//   angle     : (1, 25*H*W)  sigmoid(angle_head)*pi     per-neighbor angle
//   ray       : (1, 3*H*W)   L2-normalized ray_8
//   fu,cu,fv,cv: intrinsics scaled to the (H,W) grid (the +0.5-on-cx/cy build).
//   out       : (1, 3*H*W)   (re)allocated on the inputs' device; the
//                            L2-normalized prob-weighted sum.
void angmf_propagate(const brotensor::Tensor& pred_norm,
                     const brotensor::Tensor& prob,
                     const brotensor::Tensor& xy,
                     const brotensor::Tensor& angle,
                     const brotensor::Tensor& ray,
                     double fu, double cu, double fv, double cv,
                     int H, int W,
                     brotensor::Tensor& out);

}  // namespace brovisionml::dsine
