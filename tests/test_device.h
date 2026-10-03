#pragma once
//
// Test-only helper for GPU parity blocks. The model tests build a module on the
// CPU, then re-run it on a GPU backend and assert the two agree. Historically
// each test hard-coded Device::CUDA; this helper makes the same block exercise
// whichever GPU backend the binary was built with — HIP on a ROCm build, CUDA on
// a CUDA build, Metal on a Metal build — so the one parity test covers every
// platform.
//
// Call brotensor::init() first (it performs the HIP / CUDA / Metal driver probe),
// then preferred_gpu(): it returns the backend to test against, or Device::CPU when
// no GPU backend is registered (meaning "skip the parity block").
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

namespace brovisionml_test {

inline brotensor::Device preferred_gpu() {
    // The default device when it is a GPU (BROTENSOR_DEFAULT_DEVICE=vulkan runs
    // the parity blocks on Vulkan), else the first registered GPU, Vulkan last.
    const brotensor::Device d = brotensor::default_device();
    if (d.is_gpu()) return d;
    if (brotensor::is_available(brotensor::Device::HIP))   return brotensor::Device::HIP;
    if (brotensor::is_available(brotensor::Device::CUDA))  return brotensor::Device::CUDA;
    if (brotensor::is_available(brotensor::Device::Metal)) return brotensor::Device::Metal;
    if (brotensor::is_available(brotensor::Device::VULKAN)) return brotensor::Device::VULKAN;
    return brotensor::Device::CPU;
}

inline const char* device_name(brotensor::Device d) {
    switch (d.type) {
        case brotensor::DeviceType::HIP:   return "HIP";
        case brotensor::DeviceType::CUDA:  return "CUDA";
        case brotensor::DeviceType::Metal: return "Metal";
        case brotensor::DeviceType::VULKAN: return "Vulkan";
        default:                           return "CPU";
    }
}

} // namespace brovisionml_test
