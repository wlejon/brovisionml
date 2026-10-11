#pragma once

// Private helpers shared by the SAM 2 modules (Hiera encoder, SAM heads,
// memory side). NOT a public header. Every activation the modules pass around
// is a row-major brotensor::Tensor: token sequences are (L, C), feature maps
// are NCHW (N, C*H*W). The mixed-precision convention follows SAM 1: on a GPU
// whose compute dtype is FP16, GEMM weights and attention run FP16 while the
// residual streams, LayerNorms and every layout op stay FP32; `fp16 == false`
// is the exact all-FP32 path (CPU, or a GPU migrated under a CPU DeviceScope).

#include "brotensor/tensor.h"
#include "brotensor/safetensors.h"

#include <string>

namespace brovisionml::sam2::detail {

using brotensor::Tensor;
namespace st = brotensor::safetensors;

// ── Weights ────────────────────────────────────────────────────────────────

// Whole tensor `name` widened to host FP32 as (rows, cols), element count
// checked. Throws with `who` as the message prefix.
Tensor load_w(const st::File& f, const std::string& who, const std::string& name,
            int rows, int cols);
// Rows [row0, row0+rows) of a (R, cols) tensor (e.g. one third of a packed qkv).
Tensor load_rows(const st::File& f, const std::string& who, const std::string& name,
                 int row0, int rows, int cols);
bool has(const st::File& f, const std::string& name);

// Move `t` to `dev` and cast it to `dt` (no-op for an empty tensor).
void move(Tensor& t, brotensor::Device dev, brotensor::Dtype dt);

// ── Precision ──────────────────────────────────────────────────────────────

Tensor to16(const Tensor& t, bool fp16);   // FP32 -> FP16 when fp16, else a view-free pass-through copy
Tensor to32(const Tensor& t);              // -> FP32 (pass-through when already FP32)

// Y = X W^T + b with X in the weights' working precision (FP16 when fp16).
void linear(const Tensor& W, const Tensor& b, const Tensor& X, Tensor& Y, bool fp16);
// FP32 in, FP32 out; drops to FP16 around the GEMM when fp16.
Tensor linear32(const Tensor& W, const Tensor& b, const Tensor& X, bool fp16);

// ── Attention ──────────────────────────────────────────────────────────────

// Multi-head attention over pre-projected Q/K/V packed as `batch` independent
// sequences of `sq` query rows and `sk` key rows each (Q: (batch*sq, heads*hd),
// K/V: (batch*sk, heads*hd)), scale 1/sqrt(hd). Output has Q's dtype/shape.
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                 int batch, int sq, int sk, int heads);

// ── Layout / norms ─────────────────────────────────────────────────────────

Tensor layer_norm(const Tensor& x, const Tensor& g, const Tensor& b, float eps);
Tensor to_nchw(const Tensor& tokens, int N, int C, int H, int W);   // (N*H*W, C) -> (N, C*H*W)
Tensor to_tokens(const Tensor& nchw, int N, int C, int H, int W);   // inverse
// LayerNorm2d (channels-first) on an NCHW map, eps as given.
Tensor layer_norm_2d(const Tensor& nchw, const Tensor& g, const Tensor& b,
                     int N, int C, int H, int W, float eps);

// Non-owning (rows, cols) view at element offset `off` (FP32 or FP16 storage).
Tensor view_rows(const Tensor& t, long long off, int rows, int cols);

// Host FP32 copy of a (small) tensor.
Tensor host(const Tensor& t);

}  // namespace brovisionml::sam2::detail
