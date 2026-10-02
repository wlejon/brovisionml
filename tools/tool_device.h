#pragma once
//
// Shared device selection for the brovisionml CLI tools.
//
// Every tool accepts
//     --device D   cpu | cuda | hip | rocm | metal | gpu   (default: cpu)
//     --cuda       legacy spelling of --device gpu
// "gpu" (and so --cuda) means the best GPU backend this binary registered —
// brotensor::default_device() after init() when that is a GPU (HIP > CUDA >
// Metal, honouring BROTENSOR_DEFAULT_DEVICE), else the first GPU available. On a
// CUDA build that is CUDA, exactly what --cuda always selected. A named backend
// that is not available falls back to the CPU with a note on stderr.
//
// Built standalone only, alongside the tools; not part of the library.

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdio>
#include <string>

namespace brovisionml_tools {

// The best GPU backend registered after brotensor::init(), or Device::CPU when
// none is.
inline brotensor::Device best_gpu() {
    const brotensor::Device d = brotensor::default_device();
    if (d.is_gpu()) return d;
    if (brotensor::is_available(brotensor::Device::HIP))   return brotensor::Device::HIP;
    if (brotensor::is_available(brotensor::Device::CUDA))  return brotensor::Device::CUDA;
    if (brotensor::is_available(brotensor::Device::Metal)) return brotensor::Device::Metal;
    return brotensor::Device::CPU;
}

// A --device / --cuda request, parsed at argument time and resolved once the
// model is loaded.
struct DeviceRequest {
    enum class Kind { Cpu, Gpu, Cuda, Hip, Metal };
    Kind kind = Kind::Cpu;

    // Parses a --device value; false for an unknown name.
    bool parse(const std::string& s) {
        if (s == "cpu")                     kind = Kind::Cpu;
        else if (s == "gpu")                kind = Kind::Gpu;
        else if (s == "cuda")               kind = Kind::Cuda;
        else if (s == "hip" || s == "rocm") kind = Kind::Hip;
        else if (s == "metal")              kind = Kind::Metal;
        else return false;
        return true;
    }

    bool wants_gpu() const { return kind != Kind::Cpu; }

    // Initializes brotensor (GPU requests only) and returns the device to run
    // on, printing it. An unavailable backend yields Device::CPU.
    brotensor::Device resolve() const {
        if (kind == Kind::Cpu) return brotensor::Device::CPU;
        brotensor::init();
        brotensor::Device want = brotensor::Device::CPU;
        const char* what = "GPU";
        switch (kind) {
            case Kind::Gpu:   want = best_gpu(); break;
            case Kind::Cuda:  want = brotensor::Device::CUDA;  what = "CUDA";  break;
            case Kind::Hip:   want = brotensor::Device::HIP;   what = "HIP";   break;
            case Kind::Metal: want = brotensor::Device::Metal; what = "Metal"; break;
            case Kind::Cpu:   break;
        }
        if (want.is_gpu() && brotensor::is_available(want)) {
            std::printf("running on %s\n", brotensor::device_name(want));
            return want;
        }
        std::fprintf(stderr, "%s requested but unavailable; using CPU\n", what);
        return brotensor::Device::CPU;
    }
};

}  // namespace brovisionml_tools
