#pragma once

// DSINE v02 iterative refinement (NRN) — the AngMF ("angular mean-field")
// rotation-based update that turns the decoder's coarse /8 normal/feature/hidden
// maps into the final full-resolution surface normals. This is the LAST stage of
// DSINE: encoder -> decoder -> *this*. It is a port of `DSINE_v02.refine`,
// `DSINE_v02.upsample`, and Step 4 + the refinement loop of `DSINE_v02.forward`
// (models/dsine/v02.py), plus the submodules they call.
//
// ── The big picture ──────────────────────────────────────────────────────────
// Everything below operates at the /8 grid (h8,w8). The decoder already produced
//   normal  : (3, h8, w8)  the coarse normal (RayReLU'd, L2-normalized)
//   feature : (64, h8, w8)
//   hidden  : (64, h8, w8)
// The refinement holds a recurrent hidden state `h` (init = decoder.hidden) and a
// running coarse normal `pred_norm` (init = RayReLU(decoder.normal)). Each of the
// 5 iterations runs a ConvGRU update on `h`, predicts per-pixel per-neighbor
// rotation parameters, rotates each of a pixel's 25 (5x5) neighbor normals toward
// it, takes a learned-probability-weighted average, and re-normalizes — an
// angular mean-field smoothing in SO(3). After each iter the coarse /8 normal is
// convex-upsampled by 8 to full res via a learned 9-tap mask. The final output is
// the up-sampled normal from the 5th iteration, cropped to the original
// (unpadded) image.
//
// ── Step 4 + loop (forward) ──────────────────────────────────────────────────
//   pred_norm = RayReLU(decoder.normal, ray_8)          # face-camera clamp
//   feat_map  = cat([decoder.feature, uv_8])            # 64+2 = 66ch
//   h         = decoder.hidden                          # 64ch
//   up0       = upsample(h, pred_norm, uv_8)            # pred_list[0]; discarded
//   for i in 0..4:                                      # num_iter_test = 5
//       (h, pred_norm, up_pred_norm) = refine(h, feat_map, pred_norm, ...)
//   final = up_pred_norm                                # from the 5th iter
// `up0` (the pre-refinement upsample) is computed in the reference but only the
// LAST `up_pred_norm` is returned (pred_list[-1]); we still compute up0 because it
// shares no state — it is a no-op for the result, so we skip it.
//
// ── ConvGRU (submodules.ConvGRU, ks=5 pad=2) ─────────────────────────────────
//   hx = cat([h, x])                  # 64 + 66 = 130ch
//   z  = sigmoid(convz(hx))
//   r  = sigmoid(convr(hx))
//   q  = tanh(convq(cat([r*h, x])))   # 64 + 66 = 130ch
//   h  = (1-z)*h + z*q
// convz/convr/convq are 5x5, pad 2, 130->64.
//
// ── refine() (the AngMF update; port models/dsine/v02.py::refine exactly) ─────
//   h_new      = gru(h, feat_map)
//   nghbr_prob = sigmoid(prob_head(cat([h_new, uv_8])))        # (25, h, w)
//   nghbr_norm = get_unfold(pred_norm)                         # (3, 25, h, w)
//   nghbr_xy   = L2normalize(xy_head(cat([h_new, uv_8])))      # (2, 25, h, w)
//   nghbr_ang  = sigmoid(angle_head(cat([h_new, uv_8]))) * pi  # (25, h, w)
//   nghbr_pix  = get_unfold(pixel_coords)                      # (3, 25, h, w)
//   # build per-neighbor rotation AXIS from the intrinsics (fu,cu,fv,cv scaled to
//   # the /8 grid from the SAME +0.5 intrinsics build_ray8 uses):
//   du_over_fu = xy_x / fu;  dv_over_fv = xy_y / fv
//   term_u = (pix_x + xy_x - cu) / fu;  term_v = (pix_y + xy_y - cv) / fv
//   delta_z_num   = -(du_over_fu*nx + dv_over_fv*ny)
//   delta_z_denom = term_u*nx + term_v*ny + nz   (clamp |.|<1e-8 to ±1e-8)
//   delta_z = num/denom
//   axis = normalize([du_over_fu + delta_z*term_u,
//                     dv_over_fv + delta_z*term_v,
//                     delta_z])
//   if any axis component is nan/inf -> axis = 0    (-> identity rotation)
//   R = axis_angle_to_matrix(axis * nghbr_ang)      # per neighbor, per pixel
//   nghbr_norm_rot = normalize(R @ nghbr_norm)
//   nghbr_norm_rot = RayReLU(nghbr_norm_rot, ray_8) # per neighbor
//   pred_norm  = normalize( sum_over_25( nghbr_prob * nghbr_norm_rot ) )
//   up_pred_norm = normalize( upsample_via_mask(pred_norm, up_prob_head(...)) )
//
// `get_unfold` (submodules.get_unfold, ps=5 pad=2): replicate-pad the (C,h,w) map
// by 2 on every side, then im2col into (C, 25, h, w) — the 25 axis is the 5x5
// window in ROW-MAJOR order (F.unfold convention: for ky in 0..4, kx in 0..4 the
// flattened index is ky*5+kx, sampling input (y-2+ky, x-2+kx)).
//
// `axis_angle_to_matrix` (utils/rotation.py, copied from PyTorch3D):
//   angle = ||axis_angle||;  half = angle/2
//   if angle < 1e-6:  s = 0.5 - angle^2/48        (Taylor of sin(half)/angle)
//   else:             s = sin(half)/angle
//   quat = [cos(half), axis_angle*s]              # (w, x, y, z)
//   two_s = 2 / (quat·quat)
//   R = [[1-two_s(jj+kk),  two_s(ij-kr),    two_s(ik+jr)  ],
//        [two_s(ij+kr),    1-two_s(ii+kk),  two_s(jk-ir)  ],
//        [two_s(ik-jr),    two_s(jk+ir),    1-two_s(ii+jj)]]
//   with (r,i,j,k) = quat. Ported exactly.
//
// `RayReLU` (submodules.RayReLU, eps=1e-2): clamp the normal to face the camera.
//   cos = cosine_similarity(n, ray)               # n·ray / (|n||ray|), per pixel
//   diff = ray*(relu(cos-eps)+eps) - ray*cos
//   n_new = normalize(n + diff)
// (ray is the L2-normalized 3ch ray_8; build_ray8 already normalizes it.)
//
// `upsample_via_mask` (conv_encoder_decoder/submodules, padding='replicate'):
//   convex upsample by k=8. up_mask = up_prob_head(cat([h,uv_8])) is (9*8*8, h, w).
//   view as (1, 9, 8, 8, h, w) and softmax over the 9 axis. replicate-pad the
//   low-res map by 1, im2col 3x3 -> (C, 9, h, w). weighted-combine over the 9
//   neighbors per output sub-pixel (8x8 per low-res cell), then scatter to
//   (C, 8h, 8w): output pixel (y8,x8) with y8=8*y+ky, x8=8*x+kx reads sub-cell
//   (ky,kx) of low-res pixel (y,x). Ported exactly.
//
// ── Heads (get_prediction_head, plain convs — NOT weight-standardized) ────────
//   [ Conv2d(in,64,3,pad1), ReLU, Conv2d(64,64,1), ReLU, Conv2d(64,out,1) ]
//   in = hidden_dim+2 = 66 for all four heads (h_new/h concatenated with uv_8).
//   out: prob_head 25 (ps*ps), xy_head 50 (ps*ps*2), angle_head 25,
//        up_prob_head 576 (9*8*8).
//
// ── Intrinsics convention ────────────────────────────────────────────────────
// The refine axis math uses fu/cu/fv/cv scaled to the /8 grid from the SAME
// +0.5-on-principal-point intrinsics that build_uv/build_ray8 use internally:
//   fu = fx*(w8/origW);  cu = (cx+0.5)*(w8/origW)
//   fv = fy*(h8/origH);  cv = (cy+0.5)*(h8/origH)
// origH,origW are the PADDED (multiple-of-32) dims. Pass the raw preprocess
// intrinsics; the +0.5 is applied here for consistency with build_ray8.
//
// The ConvGRU gates and the four heads run through brotensor conv2d (5x5 pad2 for
// the GRU, the head kernels for the heads); the per-pixel-per-neighbor rotation
// math, the unfold, RayReLU, and the convex upsample have no direct brotensor op
// and run as host (CPU FP32) routines — matching the decoder/encoder, which also
// run host FP32 for parity. std::sin/cos drive the rotation.
//
// Weights load from a `model.safetensors` under the TOP-LEVEL keys
// `gru.conv{z,r,q}.*`, `prob_head.*`, `xy_head.*`, `angle_head.*`,
// `up_prob_head.*` (NOT `decoder.*`), FP32, host-resident.

#include "brovisionml/dsine_decoder.h"
#include "brovisionml/dsine_preprocess.h"

#include "brotensor/tensor.h"

#include <memory>
#include <string>

namespace brovisionml::dsine {

class Refiner {
public:
    Refiner();
    ~Refiner();
    Refiner(Refiner&&) noexcept;
    Refiner& operator=(Refiner&&) noexcept;

    // Load from a checkpoint directory (reads `<dir>/model.safetensors`) or a
    // file. Reads the gru/*_head top-level namespaces. Weights land FP32.
    void load(const std::string& dir);
    void load_file(const std::string& path);

    // Run the 5-iteration refinement on the decoder output. `intrins` is the raw
    // preprocess intrinsics (pre-"+0.5"; +0.5 applied internally). origH,origW
    // are the PADDED dims (the /8 grid is dec.h8/w8). `tf` carries the pad
    // offsets used to crop the full-res result back to the original image.
    //
    // Returns the final surface normals as a flat NCHW (1, 3*H*W) host FP32
    // tensor at the ORIGINAL (unpadded) resolution (H = tf.orig_h, W = tf.orig_w),
    // L2-normalized per pixel.
    brotensor::Tensor forward(const DecoderOutput& dec, const Intrinsics& intrins,
                              int origH, int origW, const DsineTransform& tf) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brovisionml::dsine
