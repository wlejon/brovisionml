#pragma once

#include "brotensor/tensor.h"
#include <cstdint>

namespace brovisionml::dpt::detail {

void dpt_preprocess_gpu(const uint8_t* host_rgb, int in_w, int in_h, int channels,
                        brotensor::Device dev,
                        int out_w, int out_h,
                        brotensor::Tensor& out_pixels);

}  // namespace brovisionml::dpt::detail
