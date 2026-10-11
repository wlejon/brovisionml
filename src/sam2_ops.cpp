#include "sam2_ops.h"

#include "weights_util.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace brovisionml::sam2::detail {

namespace bt = brotensor;

Tensor load_w(const st::File& f, const std::string& who, const std::string& name,
            int rows, int cols) {
    return brovisionml::detail::load_whole(f, who, name, rows, cols);
}

Tensor load_rows(const st::File& f, const std::string& who, const std::string& name,
                 int row0, int rows, int cols) {
    const st::TensorView* v = f.find(name);
    if (!v) throw std::runtime_error(who + "missing tensor '" + name + "'");
    return brovisionml::detail::load_f32(*v, static_cast<std::size_t>(row0) * cols,
                                         rows, cols, who, name);
}

bool has(const st::File& f, const std::string& name) { return f.find(name) != nullptr; }

void move(Tensor& t, bt::Device dev, bt::Dtype dt) {
    if (!t.data) return;
    if (t.device != dev) t = t.to(dev);
    if (t.dtype != dt) { Tensor c; bt::cast(t, c, dt); t = std::move(c); }
}

Tensor to16(const Tensor& t, bool fp16) {
    if (!fp16 || t.dtype == bt::Dtype::FP16) return t;
    Tensor c; bt::cast(t, c, bt::Dtype::FP16); return c;
}

Tensor to32(const Tensor& t) {
    if (t.dtype == bt::Dtype::FP32) return t;
    Tensor c; bt::cast(t, c, bt::Dtype::FP32); return c;
}

void linear(const Tensor& W, const Tensor& b, const Tensor& X, Tensor& Y, bool fp16) {
    if (fp16) bt::linear_forward_batched_fp16(W, &b, X, Y);
    else      bt::linear_forward_batched(W, b, X, Y);
}

Tensor linear32(const Tensor& W, const Tensor& b, const Tensor& X, bool fp16) {
    Tensor y;
    if (fp16) {
        Tensor x16 = to16(X, true);
        bt::linear_forward_batched_fp16(W, &b, x16, y);
        return to32(y);
    }
    bt::linear_forward_batched(W, b, X, y);
    return y;
}

namespace {
Tensor make_cu(bt::Device dev, int batch, int seqlen) {
    std::vector<int32_t> h(static_cast<std::size_t>(batch) + 1);
    for (int i = 0; i <= batch; ++i) h[static_cast<std::size_t>(i)] = i * seqlen;
    Tensor cpu = Tensor::view(bt::Device::CPU, h.data(), batch + 1, 1, bt::Dtype::INT32);
    return cpu.to(dev);
}
}  // namespace

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                 int batch, int sq, int sk, int heads) {
    const int hd = q.cols / heads;
    Tensor o;
    // A single sequence on a 16-bit GPU path takes the tensor-core flash
    // kernel; packed windows (and the exact FP32 path) take the varlen kernel.
    if (batch == 1 && q.device.is_gpu() &&
        (q.dtype == bt::Dtype::FP16 || q.dtype == bt::Dtype::BF16)) {
        bt::flash_attention_forward(q, k, v, nullptr, heads, /*causal=*/false, o);
        return o;
    }
    Tensor cu_q = make_cu(q.device, batch, sq);
    Tensor cu_k = make_cu(k.device, batch, sk);
    bt::flash_attention_varlen_forward(
        q, k, v, reinterpret_cast<const int32_t*>(cu_q.data),
        reinterpret_cast<const int32_t*>(cu_k.data), batch, sq, sk, heads, hd,
        /*causal=*/false, o);
    return o;
}

Tensor layer_norm(const Tensor& x, const Tensor& g, const Tensor& b, float eps) {
    Tensor y;
    bt::layernorm_forward_inference_batched(x, g, b, y, eps);
    return y;
}

Tensor to_nchw(const Tensor& tokens, int N, int C, int H, int W) {
    Tensor y;
    bt::sequence_to_nchw(tokens, N, C, H, W, y);
    return y;
}

Tensor to_tokens(const Tensor& nchw, int N, int C, int H, int W) {
    Tensor y;
    bt::nchw_to_sequence(nchw, N, C, H, W, y);
    return y;
}

Tensor layer_norm_2d(const Tensor& nchw, const Tensor& g, const Tensor& b,
                     int N, int C, int H, int W, float eps) {
    Tensor t = to_tokens(nchw, N, C, H, W);
    Tensor n = layer_norm(t, g, b, eps);
    return to_nchw(n, N, C, H, W);
}

Tensor view_rows(const Tensor& t, long long off, int rows, int cols) {
    const int es = bt::dtype_size_bytes(t.dtype);
    return Tensor::view(t.device, static_cast<char*>(t.data) + off * es, rows, cols, t.dtype);
}

Tensor host(const Tensor& t) {
    Tensor h = t.device == bt::Device::CPU ? t : t.to(bt::Device::CPU);
    return to32(h);
}

}  // namespace brovisionml::sam2::detail
