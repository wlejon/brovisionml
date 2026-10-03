// Host side of brovisionml's Vulkan kernels (declared in vulkan_ops.h, GLSL in
// src/vulkan/, embedded by brotensor_vulkan_add_shaders in CMakeLists.txt).
// Each op is one dispatch on the tensors' device stream through
// brotensor::vulkan::dispatch, ordered with brotensor's own ops.

#include "vulkan_ops.h"

#include "brovisionml_vk_shaders.h"   // generated: brovisionml::detail::vk

#include <brotensor/vulkan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brovisionml {

namespace {

namespace bt = ::brotensor;
namespace vk = ::brovisionml::detail::vk;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brovisionml vulkan: " + msg);
}

void require_f32(const bt::Tensor& t, const char* what) {
    if (t.dtype != bt::Dtype::FP32) fail(std::string(what) + " must be FP32");
    if (!t.device.is_vulkan()) fail(std::string(what) + " must be on a Vulkan device");
}

std::uint32_t groups_for(std::uint64_t n) {
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>((n + 255) / 256, 1, 65535));
}

bool f64(bt::Device d) {
    static thread_local int cached_index = -1;
    static thread_local bool cached = false;
    if (cached_index != d.index) {
        cached = bt::vulkan::device_info(d).shader_float64;
        cached_index = d.index;
    }
    return cached;
}

struct RayReluPush {
    std::uint64_t n, r;
    std::uint32_t hw, pad;
};

template <class Real>
struct AngmfPush {
    std::uint64_t pn, pr, xy, ang, ray, op;
    Real fu, cu, fv, cv;
    std::int32_t H, W;
};
static_assert(sizeof(AngmfPush<double>) == 88 && sizeof(AngmfPush<float>) == 72,
              "push blocks mirror src/vulkan/dsine_ops.comp");

struct DptPush {
    std::uint64_t src, dst;
    std::int32_t in_w, in_h, channels, out_w, out_h, pad;
};

}  // namespace

namespace dsine::detail {

void ray_relu_vulkan(bt::Tensor& normal, const bt::Tensor& ray, int H, int W) {
    require_f32(normal, "ray_relu: normal");
    require_f32(ray, "ray_relu: ray");
    const std::uint64_t HW = static_cast<std::uint64_t>(H) * W;
    if (HW == 0) return;
    if (normal.size() < 3 * HW || ray.size() < 3 * HW) fail("ray_relu: maps smaller than 3*H*W");
    RayReluPush pc{bt::vulkan::address(normal), bt::vulkan::address(ray),
                   static_cast<std::uint32_t>(HW), 0};
    const auto s = f64(normal.device) ? vk::Shader::dsine_ray_relu_f64 : vk::Shader::dsine_ray_relu;
    bt::vulkan::dispatch(normal.device, vk::handle(s), &pc, sizeof pc, groups_for(HW));
}

void angmf_propagate_vulkan(const bt::Tensor& pred_norm, const bt::Tensor& prob,
                            const bt::Tensor& xy, const bt::Tensor& angle,
                            const bt::Tensor& ray,
                            double fu, double cu, double fv, double cv,
                            int H, int W, bt::Tensor& out) {
    require_f32(pred_norm, "angmf_propagate: pred_norm");
    require_f32(prob, "angmf_propagate: prob");
    require_f32(xy, "angmf_propagate: xy");
    require_f32(angle, "angmf_propagate: angle");
    require_f32(ray, "angmf_propagate: ray");
    const std::uint64_t HW = static_cast<std::uint64_t>(H) * W;
    // As on CUDA: the output is allocated on the input's device.
    out = bt::Tensor::zeros_on(pred_norm.device, 1, static_cast<int>(3 * HW));
    if (HW == 0) return;
    if (pred_norm.size() < 3 * HW || ray.size() < 3 * HW || prob.size() < 25 * HW ||
        angle.size() < 25 * HW || xy.size() < 50 * HW) {
        fail("angmf_propagate: input maps smaller than their (C, H, W) shapes");
    }
    const bt::Device d = pred_norm.device;
    if (f64(d)) {
        AngmfPush<double> pc{bt::vulkan::address(pred_norm), bt::vulkan::address(prob),
                             bt::vulkan::address(xy), bt::vulkan::address(angle),
                             bt::vulkan::address(ray), bt::vulkan::address(out),
                             fu, cu, fv, cv, H, W};
        bt::vulkan::dispatch(d, vk::handle(vk::Shader::dsine_angmf_f64), &pc, sizeof pc, groups_for(HW));
    } else {
        AngmfPush<float> pc{bt::vulkan::address(pred_norm), bt::vulkan::address(prob),
                            bt::vulkan::address(xy), bt::vulkan::address(angle),
                            bt::vulkan::address(ray), bt::vulkan::address(out),
                            static_cast<float>(fu), static_cast<float>(cu),
                            static_cast<float>(fv), static_cast<float>(cv), H, W};
        bt::vulkan::dispatch(d, vk::handle(vk::Shader::dsine_angmf), &pc, sizeof pc, groups_for(HW));
    }
}

}  // namespace dsine::detail

namespace dpt::detail {

void dpt_preprocess_vulkan(const std::uint8_t* host_rgb, int in_w, int in_h, int channels,
                           bt::Device dev, int out_w, int out_h, bt::Tensor& out_pixels) {
    if (!host_rgb || in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) {
        throw std::runtime_error("dpt_preprocess_vulkan: invalid dimensions");
    }
    if (!dev.is_vulkan()) fail("dpt_preprocess: not a Vulkan device");
    const std::size_t in_bytes = static_cast<std::size_t>(in_w) * in_h * channels;
    const int out_plane = out_w * out_h;

    // The image bytes in an INT32-typed carrier, reused across calls.
    static thread_local bt::Tensor s_dev_in;
    const int words = static_cast<int>((in_bytes + 3) / 4);
    if (s_dev_in.device != dev || s_dev_in.rows != 1 || s_dev_in.cols < words) {
        s_dev_in = bt::Tensor::empty_on(dev, 1, words, bt::Dtype::INT32);
    }
    s_dev_in.copy_from_host_raw(host_rgb, in_bytes);

    if (out_pixels.rows != 1 || out_pixels.cols != 3 * out_plane || out_pixels.device != dev ||
        out_pixels.dtype != bt::Dtype::FP32) {
        out_pixels = bt::Tensor::empty_on(dev, 1, 3 * out_plane, bt::Dtype::FP32);
    }
    DptPush pc{bt::vulkan::address(s_dev_in), bt::vulkan::address(out_pixels),
               in_w, in_h, channels, out_w, out_h, 0};
    const auto gx = static_cast<std::uint32_t>((out_w + 15) / 16);
    const auto gy = static_cast<std::uint32_t>((out_h + 15) / 16);
    if (gx > 65535 || gy > 65535) fail("dpt_preprocess: output too large");
    bt::vulkan::dispatch(dev, vk::handle(vk::Shader::dpt_preprocess), &pc, sizeof pc, gx, gy);
}

}  // namespace dpt::detail

}  // namespace brovisionml
