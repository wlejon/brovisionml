#pragma once

#include "embed/embed.h"
#include "host_class.h"
#include "object_builder.h"
#include "arg_reader.h"

#include <brovisionml/version.h>
#include <brovisionml/depth_anything.h>
#include <brovisionml/sam.h>
#include <brovisionml/sam_amg.h>
#include <brovisionml/dsine.h>
#include <brovisionml/hed.h>
#include <brovisionml/lineart.h>
#include <brovisionml/mlsd.h>
#include <brovisionml/openpose.h>
#include <brovisionml/segformer.h>
#include <brovisionml/birefnet.h>
#include <brovisionml/stylegan3.h>
#include <brovisionml/dinov2.h>
#include <brovisionml/dinov3.h>
#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include <broimage/decode.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace brovisionml::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

inline constexpr uint32_t kHostDepthEstimatorTag  = 0x56444550u; // 'VDEP'
inline constexpr uint32_t kHostSamTag             = 0x5653414Du; // 'VSAM'
inline constexpr uint32_t kHostNormalEstimatorTag = 0x564E4F52u; // 'VNOR'
inline constexpr uint32_t kHostHedTag             = 0x56484544u; // 'VHED'
inline constexpr uint32_t kHostLineartTag         = 0x564C494Eu; // 'VLIN'
inline constexpr uint32_t kHostMlsdTag            = 0x564D4C53u; // 'VMLS'
inline constexpr uint32_t kHostOpenposeTag        = 0x56504F53u; // 'VPOS'
inline constexpr uint32_t kHostSegformerTag       = 0x56534547u; // 'VSEG'
inline constexpr uint32_t kHostBirefnetTag        = 0x56424952u; // 'VBIR'
inline constexpr uint32_t kHostStyleGAN3Tag       = 0x56535459u; // 'VSTY'
inline constexpr uint32_t kHostDinov2Tag          = 0x56444932u; // 'VDI2'
inline constexpr uint32_t kHostDinov3Tag          = 0x56444933u; // 'VDI3'
inline constexpr uint32_t kHostVisionModelTag     = 0x564D4F44u; // 'VMOD'

struct DepthEstimatorWrapper {
    uint32_t tag = kHostDepthEstimatorTag;
    std::unique_ptr<brovisionml::depth::DepthEstimator> estimator;
    brotensor::Device device = brotensor::Device::CPU;
    bool loaded = false;
    std::string path;
};

struct SamWrapper {
    uint32_t tag = kHostSamTag;
    std::unique_ptr<brovisionml::sam::Sam> sam;
    brotensor::Device device = brotensor::Device::CPU;
    bool hasImage = false;
    bool loaded = false;
    int imageW = 0;
    int imageH = 0;
    std::string path;
};

struct NormalEstimatorWrapper {
    uint32_t tag = kHostNormalEstimatorTag;
    std::unique_ptr<brovisionml::dsine::NormalEstimator> estimator;
    brotensor::Device device = brotensor::Device::CPU;
    bool loaded = false;
    std::string path;
};

struct HedWrapper {
    uint32_t tag = kHostHedTag;
    std::unique_ptr<brovisionml::hed::SoftEdgeDetector> detector;
    brotensor::Device device = brotensor::Device::CPU;
    bool loaded = false;
    std::string path;
};

struct LineartWrapper {
    uint32_t tag = kHostLineartTag;
    std::unique_ptr<brovisionml::lineart::LineartDetector> detector;
    brotensor::Device device = brotensor::Device::CPU;
    bool loaded = false;
    std::string path;
};

struct MlsdWrapper {
    uint32_t tag = kHostMlsdTag;
    std::unique_ptr<brovisionml::mlsd::MLSDdetector> detector;
    brotensor::Device device = brotensor::Device::CPU;
    bool loaded = false;
    std::string path;
};

struct OpenposeWrapper {
    uint32_t tag = kHostOpenposeTag;
    std::unique_ptr<brovisionml::openpose::OpenposeDetector> detector;
    brotensor::Device device = brotensor::Device::CPU;
    bool loaded = false;
    std::string path;
};

struct SegformerWrapper {
    uint32_t tag = kHostSegformerTag;
    std::unique_ptr<brovisionml::segformer::SegformerDetector> detector;
    brotensor::Device device = brotensor::Device::CPU;
    bool loaded = false;
    std::string path;
};

struct BirefnetWrapper {
    uint32_t tag = kHostBirefnetTag;
    std::unique_ptr<brovisionml::birefnet::BiRefNet> net;
    brotensor::Device device = brotensor::Device::CPU;
    bool loaded = false;
    std::string path;
};

struct StyleGAN3Wrapper {
    uint32_t tag = kHostStyleGAN3Tag;
    std::unique_ptr<brovisionml::stylegan3::Generator> generator;
    brotensor::Device device = brotensor::Device::CPU;
    int32_t zDim = 512;
    int32_t cDim = 0;
    int32_t wDim = 512;
    int32_t imgResolution = 1024;
    int32_t imgChannels = 3;
    bool loaded = false;
    std::string path;
};

struct Dinov2Wrapper {
    uint32_t tag = kHostDinov2Tag;
    std::unique_ptr<brovisionml::dinov2::Backbone> backbone;
    brotensor::Device device = brotensor::Device::CPU;
    bool loaded = false;
    std::string path;
};

struct Dinov3Wrapper {
    uint32_t tag = kHostDinov3Tag;
    std::unique_ptr<brovisionml::dinov3::Backbone> backbone;
    brotensor::Device device = brotensor::Device::CPU;
    int size = 1024;
    bool loaded = false;
    std::string path;
};

enum class VisionTaskType {
    Generic,
    Depth,
    Sam,
    Detection,
    Pose,
    OCR,
    Normal,
    Edge,
    Lineart,
    Mlsd,
    Segformer,
    Birefnet
};

struct VisionModelWrapper {
    uint32_t tag = kHostVisionModelTag;
    VisionTaskType task = VisionTaskType::Generic;
    std::string taskName = "generic";
    std::string modelPath;
    brotensor::Device device = brotensor::Device::CPU;
    bool loaded = false;
    std::vector<std::string> labels;
    float confThreshold = 0.25f;
    float iouThreshold = 0.45f;

    // Optional underlying handles
    std::unique_ptr<DepthEstimatorWrapper> depthEstimator;
    std::unique_ptr<SamWrapper> sam;
    std::unique_ptr<NormalEstimatorWrapper> normalEstimator;
    std::unique_ptr<HedWrapper> hed;
    std::unique_ptr<LineartWrapper> lineart;
    std::unique_ptr<MlsdWrapper> mlsd;
    std::unique_ptr<OpenposeWrapper> openpose;
    std::unique_ptr<SegformerWrapper> segformer;
    std::unique_ptr<BirefnetWrapper> birefnet;
};

extern HostClass g_depthEstimatorClass;
extern HostClass g_samClass;
extern HostClass g_normalEstimatorClass;
extern HostClass g_hedClass;
extern HostClass g_lineartClass;
extern HostClass g_mlsdClass;
extern HostClass g_openposeClass;
extern HostClass g_segformerClass;
extern HostClass g_birefnetClass;
extern HostClass g_stylegan3Class;
extern HostClass g_dinov2Class;
extern HostClass g_dinov3Class;
extern HostClass g_visionModelClass;

brotensor::Device parseDevice(Value opts);
const char* deviceName(brotensor::Device dev);

bool readFloat32Array(Value val, const float*& outData, size_t& outCount);
bool readUint8Array(Value val, const uint8_t*& outData, size_t& outCount);
bool readInt32Array(Value val, const int32_t*& outData, size_t& outCount);
bool readImageInput(Value val, std::vector<uint8_t>& rgba, int& w, int& h, std::string& err);

Value makeFloat32Array(const float* data, size_t count);
Value makeUint8Array(const uint8_t* data, size_t count);
Value makeUint8ClampedArray(const uint8_t* data, size_t count);
Value makeInt32Array(const int32_t* data, size_t count);

inline Value makeEmptyArray() {
    ev::CallResult parsed = ev::parseJson("[]");
    return (!parsed.thrown && ev::isObject(parsed.value)) ? parsed.value : ev::undefined();
}

inline Value hostArrayOf(size_t count, const std::function<Value(size_t)>& make) {
    ev::CallResult parsed = ev::parseJson("[]");
    if (parsed.thrown || !ev::isObject(parsed.value)) return ev::undefined();
    ev::Persistent arr(parsed.value);
    if (count == 0) return arr.get();

    ev::Persistent push(ev::getProperty(arr.get(), "push"));
    if (!ev::isFunction(push.get())) return arr.get();

    for (size_t i = 0; i < count; ++i) {
        Value v = make(i);
        ev::call(push.get(), arr.get(), std::span<const Value>(&v, 1));
    }
    return arr.get();
}

inline bool isJsArray(Value v) {
    if (!ev::isObject(v)) return false;
    Value lenV = ev::getProperty(v, "length");
    return !ev::isUndefined(lenV) && !ev::isObject(lenV);
}

inline uint32_t getJsArrayLength(Value v) {
    if (!ev::isObject(v)) return 0;
    Value lenV = ev::getProperty(v, "length");
    if (ev::isUndefined(lenV) || ev::isObject(lenV)) return 0;
    double d = ev::toDouble(lenV);
    return (d > 0.0) ? static_cast<uint32_t>(d) : 0;
}

void ensureVisionClassesInstalled();
Value makeVisionNamespace();
void mountVisionOps(ObjectBuilder& visionObj);

} // namespace brovisionml::api
