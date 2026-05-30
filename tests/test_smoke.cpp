// Smoke test: brovisionml links, versions, and can reach both of its compute
// siblings — brotensor (tensors + ops + safetensors loader) and broimage
// (host-side decode / resize / normalize, including the SAM preprocessing
// presets vision models build on).
#include "brovisionml/version.h"

#include <brotensor/tensor.h>
#include <broimage/version.h>
#include <broimage/presets.h>

#include <cstdio>
#include <cstring>

int main() {
    const char* v = brovisionml::version_string();
    if (!v || std::strlen(v) == 0) {
        std::fprintf(stderr, "version_string() returned empty\n");
        return 1;
    }

    // Prove the brotensor dependency is wired in and usable: a 2x3 host tensor.
    brotensor::Tensor t = brotensor::Tensor::mat(2, 3);
    if (t.rows != 2 || t.cols != 3) {
        std::fprintf(stderr, "brotensor tensor has wrong shape: %dx%d\n",
                     t.rows, t.cols);
        return 1;
    }

    // Prove the broimage dependency is wired in: reach a real symbol (the SAM
    // normalize preset) so the link is exercised, not just the header.
    const char* iv = broimage::version_string();
    if (!iv || std::strlen(iv) == 0) {
        std::fprintf(stderr, "broimage::version_string() returned empty\n");
        return 1;
    }
    if (!(broimage::SAM_STD[0] > 0.0f)) {
        std::fprintf(stderr, "broimage SAM preset looks wrong: SAM_STD[0]=%f\n",
                     broimage::SAM_STD[0]);
        return 1;
    }

    std::printf("brovisionml %s (brotensor + broimage %s reachable)\n", v, iv);
    return 0;
}
