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
    int modelSize = 1024;          // square resolution the matte runs at
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
    int32_t numWs = 16;            // mapping rows / W+ row count
    int32_t imgResolution = 1024;
    int32_t imgChannels = 3;
    char variant = 'r';            // 'r' = config-R, 't' = config-T
    bool loaded = false;
    std::string path;
};

struct Dinov2Wrapper {
    uint32_t tag = kHostDinov2Tag;
    std::unique_ptr<brovisionml::dinov2::Backbone> backbone;
    brotensor::Device device = brotensor::Device::CPU;
    int size = 518;                // default square encode resolution
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

// Resolve a loader's device from opts.device (see native_vision_ops.cpp):
// init() + best available backend by default; false with a pending JS
// exception in `thrown` on a bad or unavailable device. `opts` must be
// current at the call (a rooted slot or a Persistent read).
bool resolveDevice(const char* fnName, Value opts, brotensor::Device& dev, Value& thrown);
const char* deviceName(brotensor::Device dev);

bool readFloat32Array(Value val, const float*& outData, size_t& outCount);
bool readUint8Array(Value val, const uint8_t*& outData, size_t& outCount);
bool readInt32Array(Value val, const int32_t*& outData, size_t& outCount);
bool readImageInput(Value val, std::vector<uint8_t>& rgba, int& w, int& h, std::string& err);

// Path resolution for the model loaders and the image-by-filename argument
// (api.h setPathResolver). Every string path a binding hands to brovisionml
// or to broimage's decoder goes through this first.
std::string resolvePath(const std::string& path);

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

// ── option reads ──────────────────────────────────────────────────────────
//
// THE GC RULE these exist for (bronze/src/embed/embed.h): getProperty may
// allocate, and any allocation may move every heap value, so a `Value opts`
// copied out of the argument span is stale after its first property read.
// Pass these a value that is current at the call — a slot of the rooted
// `args` span (`args[1]`) or a `Persistent::get()` — and read one property
// per call; never cache the receiver in a local across two of them.

// obj[key] as an int when it is a finite number in int range, else `def`.
inline int optInt(Value obj, const char* key, int def) {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, key);
    if (!ev::isNumber(v)) return def;
    const double d = ev::toDouble(v);
    if (!(d >= -2147483648.0 && d <= 2147483647.0)) return def;   // NaN, ±inf, out of range
    return static_cast<int>(d);
}

// obj[key] as a float when it is a number, else `def`.
inline float optFloat(Value obj, const char* key, float def) {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, key);
    return ev::isNumber(v) ? static_cast<float>(ev::toDouble(v)) : def;
}

// obj[key] as a double when it is a number, else `def`.
inline double optDouble(Value obj, const char* key, double def) {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, key);
    return ev::isNumber(v) ? ev::toDouble(v) : def;
}

// arr[i] as a float when it is a number, else 0.
inline float numElem(Value arr, uint32_t i) {
    Value v = ev::getElement(arr, i);
    return ev::isNumber(v) ? static_cast<float>(ev::toDouble(v)) : 0.0f;
}

// obj[key] truthiness; undefined / null / absent give `def`.
inline bool optBool(Value obj, const char* key, bool def) {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, key);
    if (ev::isUndefined(v) || ev::isNull(v)) return def;
    return ev::toBool(v);
}

// obj[key] as a string when it is a string, else `def`.
inline std::string optString(Value obj, const char* key, const std::string& def) {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, key);
    return ev::isString(v) ? ev::toUtf8(v) : def;
}

// Both of these read `length` (an allocating call) and so leave `v` stale:
// the caller must hold `v` in a Persistent or a rooted slot and re-read it.
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

// Prototype decoration is split across translation units so each file stays
// small: SAM in native_vision_sam.cpp, StyleGAN3 + BiRefNet in
// native_vision_generative.cpp, the ControlNet annotators in
// native_vision_annotators.cpp, everything else in native_vision_models.cpp.
void decorateSamProto(ObjectBuilder& proto);
void decorateStyleGAN3Proto(ObjectBuilder& proto);
void decorateBirefnetProto(ObjectBuilder& proto);

// The two ViT backbones, in native_vision_backbones.cpp.
void decorateDinov2Proto(ObjectBuilder& proto);
void decorateDinov3Proto(ObjectBuilder& proto);
bool loadDinov2Backbone(const std::string& dir, Value opts, Dinov2Wrapper& w, std::string& err);
bool loadDinov3Backbone(const std::string& path, Dinov3Wrapper& w, std::string& err);

// The five ControlNet annotators, in native_vision_annotators.cpp.
void decorateHedProto(ObjectBuilder& proto);
void decorateLineartProto(ObjectBuilder& proto);
void decorateMlsdProto(ObjectBuilder& proto);
void decorateOpenposeProto(ObjectBuilder& proto);
void decorateSegformerProto(ObjectBuilder& proto);

Value runHedDetect(HedWrapper* w, std::span<const Value> args);
Value runLineartDetect(LineartWrapper* w, std::span<const Value> args);
Value runMlsdDetect(MlsdWrapper* w, std::span<const Value> args);
Value runOpenposeDetect(OpenposeWrapper* w, std::span<const Value> args);
Value runSegformerDetect(SegformerWrapper* w, std::span<const Value> args);
Value runSamSegment(SamWrapper* w, std::span<const Value> args);
Value runBirefnet(BirefnetWrapper* w, std::span<const Value> args);

// { num, width, height, best, masks: [{ iou, data: Uint8Array }] } — the shape
// the pre-transition Sam.segment() returned (minus the ImageBitmap, which is
// an engine-side type; callers rasterize `data` themselves).
Value buildSegmentation(const brovisionml::sam::Segmentation& seg);

// Build a StyleGAN3 generator for `path`. opts.resolution (256/512/1024) and
// opts.variant ('r'/'t') must match the checkpoint. Fills the wrapper's
// dimensions; returns false and sets `err` on a bad option or a load failure.
bool loadStyleGAN3Generator(const std::string& path, Value opts,
                            StyleGAN3Wrapper& w, std::string& err);

void ensureVisionClassesInstalled();
Value makeVisionNamespace();
void mountVisionOps(ObjectBuilder& visionObj);

} // namespace brovisionml::api
