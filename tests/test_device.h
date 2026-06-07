#pragma once
//
// Test-only helper for GPU parity blocks. The model tests build a module on the
// CPU, then re-run it on a GPU backend and assert the two agree. Historically
// each test hard-coded Device::CUDA; this helper makes the same block exercise
// whichever GPU backend the binary was built with — CUDA on a CUDA build, Metal
// on a Metal build — so the one parity test covers both platforms.
//
// Call brotensor::init() first (it performs the CUDA / Metal driver probe), then
// preferred_gpu(): it returns the backend to test against, or Device::CPU when
// no GPU backend is registered (meaning "skip the parity block").
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

namespace brovisionml_test {

inline brotensor::Device preferred_gpu() {
    if (brotensor::is_available(brotensor::Device::CUDA))  return brotensor::Device::CUDA;
    if (brotensor::is_available(brotensor::Device::Metal)) return brotensor::Device::Metal;
    return brotensor::Device::CPU;
}

inline const char* device_name(brotensor::Device d) {
    switch (d) {
        case brotensor::Device::CUDA:  return "CUDA";
        case brotensor::Device::Metal: return "Metal";
        default:                       return "CPU";
    }
}

} // namespace brovisionml_test
