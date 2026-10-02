// Fast fused GPU image preprocessor for DPT / Depth-Anything-V2:
// Combines aspect-ratio bicubic resizing (Catmull-Rom a=-0.5), uint8 rounding,
// [0, 255] -> [0, 1] scaling, and per-channel ImageNet normalization directly
// on the GPU in a single kernel.

#include "dpt_preprocess_gpu.h"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brovisionml::dpt::detail {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("dpt cuda: ") + what + ": " +
                                 cudaGetErrorString(e));
}

__device__ inline float cubic_weight(float t) {
    const float a = -0.5f;
    const float at = fabsf(t);
    if (at < 1.0f) {
        return (a + 2.0f) * at * at * at - (a + 3.0f) * at * at + 1.0f;
    }
    if (at < 2.0f) {
        return a * at * at * at - 5.0f * a * at * at + 8.0f * a * at - 4.0f * a;
    }
    return 0.0f;
}

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void dpt_preprocess_bicubic_kernel(
    const uint8_t* __restrict__ src,
    int in_w, int in_h, int channels,
    float* __restrict__ dst,
    int out_w, int out_h) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    const float xs = (float)in_w / (float)out_w;
    const float ys = (float)in_h / (float)out_h;

    const float fy = (y + 0.5f) * ys - 0.5f;
    const int   iy = (int)floorf(fy);
    const float ty = fy - (float)iy;
    const float wy[4] = {
        cubic_weight(-1.0f - ty),
        cubic_weight( 0.0f - ty),
        cubic_weight( 1.0f - ty),
        cubic_weight( 2.0f - ty),
    };

    const float fx = (x + 0.5f) * xs - 0.5f;
    const int   ix = (int)floorf(fx);
    const float tx = fx - (float)ix;
    const float wx[4] = {
        cubic_weight(-1.0f - tx),
        cubic_weight( 0.0f - tx),
        cubic_weight( 1.0f - tx),
        cubic_weight( 2.0f - tx),
    };

    int sy[4];
    for (int j = 0; j < 4; ++j) {
        sy[j] = clampi(iy + j - 1, 0, in_h - 1);
    }
    int sx[4];
    for (int i = 0; i < 4; ++i) {
        sx[i] = clampi(ix + i - 1, 0, in_w - 1);
    }

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float inv_std[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
    const int plane = out_w * out_h;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float row = 0.0f;
            const int row_offset = sy[j] * in_w;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int pix_idx = (row_offset + sx[i]) * channels;
                uint8_t val;
                if (channels == 1) {
                    val = src[pix_idx];
                } else {
                    val = src[pix_idx + c];
                }
                row += wx[i] * (float)val;
            }
            acc += wy[j] * row;
        }

        acc = fminf(fmaxf(acc, 0.0f), 255.0f);
        const uint8_t u = (uint8_t)(acc + 0.5f);

        const float norm = ((float)u * (1.0f / 255.0f) - mean[c]) * inv_std[c];
        dst[c * plane + y * out_w + x] = norm;
    }
}

} // namespace

void dpt_preprocess_gpu(const uint8_t* host_rgb, int in_w, int in_h, int channels,
                        brotensor::Device dev,
                        int out_w, int out_h,
                        brotensor::Tensor& out_pixels) {
    if (!host_rgb || in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) {
        throw std::runtime_error("dpt_preprocess_gpu: invalid dimensions");
    }
    const size_t in_bytes = static_cast<size_t>(in_w) * in_h * channels;
    const int out_plane = out_w * out_h;

    static thread_local brotensor::Tensor s_dev_in;
    const int int32_words = static_cast<int>((in_bytes + sizeof(int32_t) - 1) / sizeof(int32_t));
    if (s_dev_in.device != dev || s_dev_in.rows != 1 || s_dev_in.cols < int32_words) {
        s_dev_in = brotensor::Tensor::empty_on(dev, 1, int32_words, brotensor::Dtype::INT32);
    }

    cuda_check(cudaMemcpy(s_dev_in.data, host_rgb, in_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy host_rgb to dev");

    if (out_pixels.rows != 1 || out_pixels.cols != 3 * out_plane ||
        out_pixels.device != dev || out_pixels.dtype != brotensor::Dtype::FP32) {
        out_pixels = brotensor::Tensor::empty_on(dev, 1, 3 * out_plane, brotensor::Dtype::FP32);
    }

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    dpt_preprocess_bicubic_kernel<<<grid, block>>>(
        static_cast<const uint8_t*>(s_dev_in.data),
        in_w, in_h, channels,
        static_cast<float*>(out_pixels.data),
        out_w, out_h);
    cuda_check(cudaGetLastError(), "dpt_preprocess_bicubic launch");
}

}  // namespace brovisionml::dpt::detail
