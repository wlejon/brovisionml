#include "host_vision_internal.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace brovisionml::api {

HostClass g_depthEstimatorClass;
HostClass g_samClass;
HostClass g_normalEstimatorClass;
HostClass g_hedClass;
HostClass g_lineartClass;
HostClass g_mlsdClass;
HostClass g_openposeClass;
HostClass g_segformerClass;
HostClass g_birefnetClass;
HostClass g_stylegan3Class;
HostClass g_dinov2Class;
HostClass g_dinov3Class;
HostClass g_visionModelClass;

// Template to add the standard "device" getter to any host class prototype
template <typename WrapperT>
static void addDeviceAccessor(ObjectBuilder& proto, const HostClass& cls) {
    proto.accessor("device", [&cls](Value thisVal, std::span<const Value>) -> Value {
        void* p = cls.unwrap(thisVal);
        if (!p) return ev::fromUtf8("CPU");
        auto* w = static_cast<WrapperT*>(p);
        return ev::fromUtf8(deviceName(w->device));
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// DepthEstimator
// ═══════════════════════════════════════════════════════════════════════════

static Value runDepthEstimator(DepthEstimatorWrapper* w, std::span<const Value> args) {
    if (!w || !w->loaded || !w->estimator) {
        return ev::throwError("DepthEstimator.estimate: model is not initialized/loaded");
    }
    if (args.empty()) {
        return ev::throwTypeError("DepthEstimator.estimate: image argument required");
    }

    const bool invert = args.size() > 1 && ev::isObject(args[1])
                            ? ev::toBool(ev::getProperty(args[1], "invert"))
                            : false;

    std::vector<uint8_t> rgba;
    int width = 0, height = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, width, height, err)) {
        return ev::throwTypeError(std::string("DepthEstimator.estimate: ") + err);
    }

    std::vector<float> depth;
    try {
        brotensor::DeviceScope scope(w->device);
        auto dm = w->estimator->estimate(rgba.data(), width, height, 4);
        depth = std::move(dm.depth);
        width = dm.width;
        height = dm.height;
    } catch (const std::exception& e) {
        return ev::throwError(std::string("DepthEstimator estimate failed: ") + e.what());
    }

    float minV = 0.0f, maxV = 1.0f;
    if (!depth.empty()) {
        minV = *std::min_element(depth.begin(), depth.end());
        maxV = *std::max_element(depth.begin(), depth.end());
    }
    const float span = (maxV > minV) ? (maxV - minV) : 1.0f;
    std::vector<uint8_t> gray(depth.size());
    for (size_t i = 0; i < depth.size(); ++i) {
        float t = (depth[i] - minV) / span;
        if (invert) t = 1.0f - t;
        gray[i] = static_cast<uint8_t>(std::clamp(t * 255.0f, 0.0f, 255.0f));
    }

    ObjectBuilder res;
    res.set("width", static_cast<double>(width));
    res.set("height", static_cast<double>(height));
    {
        ev::Persistent dep(makeFloat32Array(depth.data(), depth.size()));
        res.set("depth", dep.get());
    }
    {
        ev::Persistent g(makeUint8Array(gray.data(), gray.size()));
        res.set("gray", g.get());
    }
    res.set("min", static_cast<double>(minV));
    res.set("max", static_cast<double>(maxV));
    return res.build();
}

static void decorateDepthEstimatorProto(ObjectBuilder& proto) {
    addDeviceAccessor<DepthEstimatorWrapper>(proto, g_depthEstimatorClass);

    // estimate(image, opts?) -> { width, height, depth, gray, min, max }
    proto.def("estimate", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_depthEstimatorClass.unwrap(thisVal);
        auto* w = p ? static_cast<DepthEstimatorWrapper*>(p) : nullptr;
        if (!w || w->tag != kHostDepthEstimatorTag) {
            return ev::throwTypeError("DepthEstimator.prototype.estimate: not a DepthEstimator instance");
        }
        return runDepthEstimator(w, args);
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Sam
// ═══════════════════════════════════════════════════════════════════════════

// decorateSamProto lives in native_vision_sam.cpp — the restored setImage /
// segment(points, labels, boxes, multimask) / segmentEverything(image, cfg).

// ═══════════════════════════════════════════════════════════════════════════
// NormalEstimator (DSINE)
// ═══════════════════════════════════════════════════════════════════════════

static Value runNormalEstimator(NormalEstimatorWrapper* w, std::span<const Value> args) {
    if (!w || !w->loaded || !w->estimator) {
        return ev::throwError("NormalEstimator.estimate: model is not initialized/loaded");
    }
    if (args.empty()) {
        return ev::throwTypeError("NormalEstimator.estimate: image argument required");
    }

    bool hasIntrinsics = false;
    float fx = 0.0f, fy = 0.0f, cx = 0.0f, cy = 0.0f;
    if (args.size() > 1 && ev::isObject(args[1])) {
        Value opts = args[1];
        Value fxv = ev::getProperty(opts, "fx");
        if (ev::isNumber(fxv)) {
            hasIntrinsics = true;
            fx = static_cast<float>(ev::toDouble(fxv));
            Value v = ev::getProperty(opts, "fy");
            if (ev::isNumber(v)) fy = static_cast<float>(ev::toDouble(v));
            v = ev::getProperty(opts, "cx");
            if (ev::isNumber(v)) cx = static_cast<float>(ev::toDouble(v));
            v = ev::getProperty(opts, "cy");
            if (ev::isNumber(v)) cy = static_cast<float>(ev::toDouble(v));
        }
    }

    std::vector<uint8_t> rgba;
    int width = 0, height = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, width, height, err)) {
        return ev::throwTypeError(std::string("NormalEstimator.estimate: ") + err);
    }

    std::vector<float> normals;
    try {
        brotensor::DeviceScope scope(w->device);
        auto nm = hasIntrinsics
                      ? w->estimator->estimate(rgba.data(), width, height, 4,
                                               fx, fy, cx, cy)
                      : w->estimator->estimate(rgba.data(), width, height, 4);
        normals = std::move(nm.normals);
        width = nm.width;
        height = nm.height;
    } catch (const std::exception& e) {
        return ev::throwError(std::string("NormalEstimator estimate failed: ") + e.what());
    }

    ObjectBuilder res;
    res.set("width", static_cast<double>(width));
    res.set("height", static_cast<double>(height));
    ev::Persistent norm(makeFloat32Array(normals.data(), normals.size()));
    res.set("normals", norm.get());
    res.set("normal", norm.get());   // the port's name for the same plane
    return res.build();
}

static void decorateNormalEstimatorProto(ObjectBuilder& proto) {
    addDeviceAccessor<NormalEstimatorWrapper>(proto, g_normalEstimatorClass);

    // estimate(image, opts?) -> { width, height, normals, normal }
    proto.def("estimate", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_normalEstimatorClass.unwrap(thisVal);
        auto* w = p ? static_cast<NormalEstimatorWrapper*>(p) : nullptr;
        if (!w || w->tag != kHostNormalEstimatorTag) {
            return ev::throwTypeError("NormalEstimator.prototype.estimate: not a NormalEstimator instance");
        }
        return runNormalEstimator(w, args);
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge & Line Annotators: HED, Lineart, MLSD
// ═══════════════════════════════════════════════════════════════════════════

// The five ControlNet annotators (HED, Lineart, MLSD, OpenPose, SegFormer)
// live in native_vision_annotators.cpp, where detect() is restored alongside
// the port's estimate() and both spellings of the result keys are published.

// decorateBirefnetProto lives in native_vision_generative.cpp — the restored
// removeBackground() (alpha / matte / cutout) plus dispose().

// ═══════════════════════════════════════════════════════════════════════════
// StyleGAN3, DINOv2, DINOv3
// ═══════════════════════════════════════════════════════════════════════════

// decorateStyleGAN3Proto lives in native_vision_generative.cpp — the restored
// generate(seed/z/truncation/returnLatents), synthesize(w) and invert(image).

// decorateDinov2Proto / decorateDinov3Proto live in native_vision_backbones.cpp
// — the restored encode(image, opts) and dispose().

// ═══════════════════════════════════════════════════════════════════════════
// VisionModel Wrapper
// ═══════════════════════════════════════════════════════════════════════════

static void decorateVisionModelProto(ObjectBuilder& proto) {
    addDeviceAccessor<VisionModelWrapper>(proto, g_visionModelClass);

    proto.accessor("type", [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::fromUtf8("generic");
        auto* w = static_cast<VisionModelWrapper*>(p);
        return ev::fromUtf8(w->taskName);
    });

    proto.accessor("isLoaded", [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::fromBool(false);
        auto* w = static_cast<VisionModelWrapper*>(p);
        return ev::fromBool(w->loaded);
    });

    proto.def("predict", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.predict: not a VisionModel");
        auto* w = static_cast<VisionModelWrapper*>(p);
        if (w->depthEstimator) return runDepthEstimator(w->depthEstimator.get(), args);
        if (w->sam) return runSamSegment(w->sam.get(), args);
        if (w->normalEstimator) return runNormalEstimator(w->normalEstimator.get(), args);
        if (w->hed) return runHedDetect(w->hed.get(), args);
        if (w->lineart) return runLineartDetect(w->lineart.get(), args);
        if (w->mlsd) return runMlsdDetect(w->mlsd.get(), args);
        if (w->openpose) return runOpenposeDetect(w->openpose.get(), args);
        if (w->segformer) return runSegformerDetect(w->segformer.get(), args);
        if (w->birefnet) return runBirefnet(w->birefnet.get(), args);
        return ev::throwError(std::string("VisionModel: model weights not loaded for task '") + w->taskName + "'");
    });

    proto.def("detect", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.detect: not a VisionModel");
        auto* w = static_cast<VisionModelWrapper*>(p);
        if (w->openpose) return runOpenposeDetect(w->openpose.get(), args);
        if (w->hed) return runHedDetect(w->hed.get(), args);
        if (w->lineart) return runLineartDetect(w->lineart.get(), args);
        if (w->mlsd) return runMlsdDetect(w->mlsd.get(), args);
        if (w->segformer) return runSegformerDetect(w->segformer.get(), args);
        return ev::throwError("VisionModel: model weights not loaded for task 'detect'");
    });

    proto.def("segment", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.segment: not a VisionModel");
        auto* w = static_cast<VisionModelWrapper*>(p);
        if (w->sam) return runSamSegment(w->sam.get(), args);
        if (w->segformer) return runSegformerDetect(w->segformer.get(), args);
        if (w->birefnet) return runBirefnet(w->birefnet.get(), args);
        return ev::throwError("VisionModel: model weights not loaded for task 'segment'");
    });

    proto.def("depth", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.depth: not a VisionModel");
        auto* w = static_cast<VisionModelWrapper*>(p);
        if (w->depthEstimator) return runDepthEstimator(w->depthEstimator.get(), args);
        return ev::throwError("VisionModel: model weights not loaded for task 'depth'");
    });

    proto.def("pose", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.pose: not a VisionModel");
        auto* w = static_cast<VisionModelWrapper*>(p);
        if (w->openpose) return runOpenposeDetect(w->openpose.get(), args);
        return ev::throwError("VisionModel: model weights not loaded for task 'pose'");
    });

    proto.def("ocr", 2, [](Value thisVal, std::span<const Value> /*args*/) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.ocr: not a VisionModel");
        return ev::throwTypeError("VisionModel.prototype.ocr: OCR backend requires an OCR model checkpoint to be explicitly loaded");
    });

    proto.def("dispose", 0, [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::undefined();
        auto* w = static_cast<VisionModelWrapper*>(p);
        w->loaded = false;
        w->depthEstimator.reset();
        w->sam.reset();
        w->normalEstimator.reset();
        w->hed.reset();
        w->lineart.reset();
        w->mlsd.reset();
        w->openpose.reset();
        w->segformer.reset();
        w->birefnet.reset();
        return ev::undefined();
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Ensure Classes Installed
// ═══════════════════════════════════════════════════════════════════════════

void ensureVisionClassesInstalled() {
    // Once per THREAD: a class's constructor and prototype are the
    // installing thread's (host_class.h), so a Worker realm installs its own.
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    g_depthEstimatorClass.install("DepthEstimator", 0, nullptr, decorateDepthEstimatorProto);
    g_samClass.install("Sam", 0, nullptr, decorateSamProto);
    g_normalEstimatorClass.install("NormalEstimator", 0, nullptr, decorateNormalEstimatorProto);
    g_hedClass.install("Hed", 0, nullptr, decorateHedProto);
    g_lineartClass.install("Lineart", 0, nullptr, decorateLineartProto);
    g_mlsdClass.install("Mlsd", 0, nullptr, decorateMlsdProto);
    g_openposeClass.install("Openpose", 0, nullptr, decorateOpenposeProto);
    g_segformerClass.install("Segformer", 0, nullptr, decorateSegformerProto);
    g_birefnetClass.install("Birefnet", 0, nullptr, decorateBirefnetProto);
    g_stylegan3Class.install("StyleGAN3", 0, nullptr, decorateStyleGAN3Proto);
    g_dinov2Class.install("Dinov2", 0, nullptr, decorateDinov2Proto);
    g_dinov3Class.install("Dinov3", 0, nullptr, decorateDinov3Proto);
    g_visionModelClass.install("VisionModel", 0, nullptr, decorateVisionModelProto);
}

static bool validateLoaderPath(const char* name, std::span<const Value> args, std::string& path, Value& outError) {
    if (args.empty()) {
        outError = ev::throwTypeError(std::string("bro.vision.") + name + ": path is required");
        return false;
    }
    if (!ev::isString(args[0])) {
        outError = ev::throwTypeError(std::string("bro.vision.") + name + ": path must be a string");
        return false;
    }
    path = resolvePath(ev::toUtf8(args[0]));
    if (!std::filesystem::exists(path)) {
        outError = ev::throwError(std::string(name) + " failed: model dir not found: " + path);
        return false;
    }
    return true;
}

// Template loader helper for detector classes
template <typename WrapperT, typename NetT>
static Value loadVisionDetector(const char* name, const HostClass& hostCls, std::span<const Value> args) {
    std::string path;
    Value errVal;
    if (!validateLoaderPath(name, args, path, errVal)) return errVal;

    Value opts = args.size() > 1 ? args[1] : ev::undefined();
    auto dev = parseDevice(opts);
    auto w = std::make_unique<WrapperT>();
    w->path = path;
    w->device = dev;
    try {
        w->detector = std::make_unique<NetT>();
        w->detector->to(dev);
        if (std::filesystem::is_regular_file(path)) {
            w->detector->load_file(path);
        } else {
            w->detector->load(path);
        }
        w->loaded = true;
    } catch (const std::exception& e) {
        return ev::throwError(std::string(name) + " failed: " + e.what());
    } catch (...) {
        return ev::throwError(std::string(name) + " failed: unknown error");
    }
    return hostCls.createInstance(std::move(w));
}

Value makeVisionNamespace() {
    ensureVisionClassesInstalled();
    ObjectBuilder vision;

    vision.accessor("version", [](Value, std::span<const Value>) -> Value {
        return ev::fromUtf8(brovisionml::version_string());
    });

    vision.def("init", 0, [](Value, std::span<const Value>) -> Value {
        try {
            brotensor::init();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("bro.vision.init failed: ") + e.what());
        }
        return ev::undefined();
    });

    vision.def("loadModel", 2, [](Value, std::span<const Value> args) -> Value {
        std::string path;
        Value errVal;
        if (!validateLoaderPath("loadModel", args, path, errVal)) return errVal;

        Value opts = args.size() > 1 ? args[1] : ev::undefined();
        auto w = std::make_unique<VisionModelWrapper>();
        w->modelPath = path;
        w->device = parseDevice(opts);
        w->loaded = false;

        if (ev::isObject(opts)) {
            Value tv = ev::getProperty(opts, "type");
            if (ev::isString(tv)) w->taskName = ev::toUtf8(tv);
            Value cv = ev::getProperty(opts, "confThreshold");
            if (!ev::isUndefined(cv)) w->confThreshold = static_cast<float>(ev::toDouble(cv));
            Value iv = ev::getProperty(opts, "iouThreshold");
            if (!ev::isUndefined(iv)) w->iouThreshold = static_cast<float>(ev::toDouble(iv));
        }

        try {
            if (w->taskName == "depth") {
                auto dw = std::make_unique<DepthEstimatorWrapper>();
                dw->path = path;
                dw->device = w->device;
                dw->estimator = std::make_unique<brovisionml::depth::DepthEstimator>(
                    brovisionml::depth::DepthAnythingConfig::v2_small());
                dw->estimator->to(w->device);
                if (std::filesystem::is_regular_file(path)) {
                    dw->estimator->load_file(path);
                } else {
                    dw->estimator->load(path);
                }
                dw->loaded = true;
                w->depthEstimator = std::move(dw);
                w->task = VisionTaskType::Depth;
                w->loaded = true;
            } else if (w->taskName == "sam" || w->taskName == "segment") {
                auto sw = std::make_unique<SamWrapper>();
                sw->path = path;
                sw->device = w->device;
                sw->sam = std::make_unique<brovisionml::sam::Sam>(brovisionml::sam::SamConfig::vit_b());
                sw->sam->to(w->device);
                if (std::filesystem::is_regular_file(path)) {
                    sw->sam->load_file(path);
                } else {
                    sw->sam->load(path);
                }
                sw->loaded = true;
                w->sam = std::move(sw);
                w->task = VisionTaskType::Sam;
                w->loaded = true;
            } else if (w->taskName == "normal") {
                auto nw = std::make_unique<NormalEstimatorWrapper>();
                nw->path = path;
                nw->device = w->device;
                nw->estimator = std::make_unique<brovisionml::dsine::NormalEstimator>();
                nw->estimator->to(w->device);
                if (std::filesystem::is_regular_file(path)) {
                    nw->estimator->load_file(path);
                } else {
                    nw->estimator->load(path);
                }
                nw->loaded = true;
                w->normalEstimator = std::move(nw);
                w->task = VisionTaskType::Normal;
                w->loaded = true;
            } else if (w->taskName == "hed" || w->taskName == "edge") {
                auto hw = std::make_unique<HedWrapper>();
                hw->path = path;
                hw->device = w->device;
                hw->detector = std::make_unique<brovisionml::hed::SoftEdgeDetector>();
                hw->detector->to(w->device);
                if (std::filesystem::is_regular_file(path)) {
                    hw->detector->load_file(path);
                } else {
                    hw->detector->load(path);
                }
                hw->loaded = true;
                w->hed = std::move(hw);
                w->task = VisionTaskType::Edge;
                w->loaded = true;
            } else if (w->taskName == "lineart") {
                auto lw = std::make_unique<LineartWrapper>();
                lw->path = path;
                lw->device = w->device;
                lw->detector = std::make_unique<brovisionml::lineart::LineartDetector>();
                lw->detector->to(w->device);
                if (std::filesystem::is_regular_file(path)) {
                    lw->detector->load_file(path);
                } else {
                    lw->detector->load(path);
                }
                lw->loaded = true;
                w->lineart = std::move(lw);
                w->task = VisionTaskType::Lineart;
                w->loaded = true;
            } else if (w->taskName == "mlsd") {
                auto mw = std::make_unique<MlsdWrapper>();
                mw->path = path;
                mw->device = w->device;
                mw->detector = std::make_unique<brovisionml::mlsd::MLSDdetector>();
                mw->detector->to(w->device);
                if (std::filesystem::is_regular_file(path)) {
                    mw->detector->load_file(path);
                } else {
                    mw->detector->load(path);
                }
                mw->loaded = true;
                w->mlsd = std::move(mw);
                w->task = VisionTaskType::Mlsd;
                w->loaded = true;
            } else if (w->taskName == "openpose" || w->taskName == "pose") {
                auto ow = std::make_unique<OpenposeWrapper>();
                ow->path = path;
                ow->device = w->device;
                ow->detector = std::make_unique<brovisionml::openpose::OpenposeDetector>();
                ow->detector->to(w->device);
                if (std::filesystem::is_regular_file(path)) {
                    ow->detector->load_file(path);
                } else {
                    ow->detector->load(path);
                }
                ow->loaded = true;
                w->openpose = std::move(ow);
                w->task = VisionTaskType::Pose;
                w->loaded = true;
            } else if (w->taskName == "segformer") {
                auto sfw = std::make_unique<SegformerWrapper>();
                sfw->path = path;
                sfw->device = w->device;
                sfw->detector = std::make_unique<brovisionml::segformer::SegformerDetector>();
                sfw->detector->to(w->device);
                if (std::filesystem::is_regular_file(path)) {
                    sfw->detector->load_file(path);
                } else {
                    sfw->detector->load(path);
                }
                sfw->loaded = true;
                w->segformer = std::move(sfw);
                w->task = VisionTaskType::Segformer;
                w->loaded = true;
            } else if (w->taskName == "birefnet") {
                auto bw = std::make_unique<BirefnetWrapper>();
                bw->path = path;
                bw->device = w->device;
                bw->net = std::make_unique<brovisionml::birefnet::BiRefNet>();
                bw->net->load(path);
                bw->net->to(w->device);
                bw->loaded = true;
                w->birefnet = std::move(bw);
                w->task = VisionTaskType::Birefnet;
                w->loaded = true;
            }
        } catch (const std::exception& e) {
            return ev::throwError(std::string("loadModel failed: ") + e.what());
        } catch (...) {
            return ev::throwError("loadModel failed: unknown error");
        }

        return g_visionModelClass.createInstance(std::move(w));
    });

    vision.def("loadDepth", 2, [](Value, std::span<const Value> args) -> Value {
        std::string path;
        Value errVal;
        if (!validateLoaderPath("loadDepth", args, path, errVal)) return errVal;

        Value opts = args.size() > 1 ? args[1] : ev::undefined();
        auto dev = parseDevice(opts);
        auto w = std::make_unique<DepthEstimatorWrapper>();
        w->path = path;
        w->device = dev;
        try {
            w->estimator = std::make_unique<brovisionml::depth::DepthEstimator>(
                brovisionml::depth::DepthAnythingConfig::v2_small());
            w->estimator->to(dev);
            if (std::filesystem::is_regular_file(path)) {
                w->estimator->load_file(path);
            } else {
                w->estimator->load(path);
            }
            w->loaded = true;
        } catch (const std::exception& e) {
            return ev::throwError(std::string("loadDepth failed: ") + e.what());
        } catch (...) {
            return ev::throwError("loadDepth failed: unknown error");
        }
        return g_depthEstimatorClass.createInstance(std::move(w));
    });

    vision.def("loadSam", 2, [](Value, std::span<const Value> args) -> Value {
        std::string path;
        Value errVal;
        if (!validateLoaderPath("loadSam", args, path, errVal)) return errVal;

        Value opts = args.size() > 1 ? args[1] : ev::undefined();
        auto dev = parseDevice(opts);
        auto w = std::make_unique<SamWrapper>();
        w->path = path;
        w->device = dev;
        try {
            w->sam = std::make_unique<brovisionml::sam::Sam>(brovisionml::sam::SamConfig::vit_b());
            w->sam->to(dev);
            if (std::filesystem::is_regular_file(path)) {
                w->sam->load_file(path);
            } else {
                w->sam->load(path);
            }
            w->loaded = true;
        } catch (const std::exception& e) {
            return ev::throwError(std::string("loadSam failed: ") + e.what());
        } catch (...) {
            return ev::throwError("loadSam failed: unknown error");
        }
        return g_samClass.createInstance(std::move(w));
    });

    vision.def("loadNormal", 2, [](Value, std::span<const Value> args) -> Value {
        std::string path;
        Value errVal;
        if (!validateLoaderPath("loadNormal", args, path, errVal)) return errVal;

        Value opts = args.size() > 1 ? args[1] : ev::undefined();
        auto dev = parseDevice(opts);
        auto w = std::make_unique<NormalEstimatorWrapper>();
        w->path = path;
        w->device = dev;
        try {
            w->estimator = std::make_unique<brovisionml::dsine::NormalEstimator>();
            w->estimator->to(dev);
            if (std::filesystem::is_regular_file(path)) {
                w->estimator->load_file(path);
            } else {
                w->estimator->load(path);
            }
            w->loaded = true;
        } catch (const std::exception& e) {
            return ev::throwError(std::string("loadNormal failed: ") + e.what());
        } catch (...) {
            return ev::throwError("loadNormal failed: unknown error");
        }
        return g_normalEstimatorClass.createInstance(std::move(w));
    });

    vision.def("loadHed", 2, [](Value, std::span<const Value> args) -> Value {
        return loadVisionDetector<HedWrapper, brovisionml::hed::SoftEdgeDetector>(
            "loadHed", g_hedClass, args);
    });

    vision.def("loadLineart", 2, [](Value, std::span<const Value> args) -> Value {
        return loadVisionDetector<LineartWrapper, brovisionml::lineart::LineartDetector>(
            "loadLineart", g_lineartClass, args);
    });

    vision.def("loadMlsd", 2, [](Value, std::span<const Value> args) -> Value {
        return loadVisionDetector<MlsdWrapper, brovisionml::mlsd::MLSDdetector>(
            "loadMlsd", g_mlsdClass, args);
    });

    vision.def("loadOpenpose", 2, [](Value, std::span<const Value> args) -> Value {
        return loadVisionDetector<OpenposeWrapper, brovisionml::openpose::OpenposeDetector>(
            "loadOpenpose", g_openposeClass, args);
    });

    vision.def("loadSegformer", 2, [](Value, std::span<const Value> args) -> Value {
        return loadVisionDetector<SegformerWrapper, brovisionml::segformer::SegformerDetector>(
            "loadSegformer", g_segformerClass, args);
    });

    vision.def("loadBirefnet", 2, [](Value, std::span<const Value> args) -> Value {
        std::string path;
        Value errVal;
        if (!validateLoaderPath("loadBirefnet", args, path, errVal)) return errVal;

        Value opts = args.size() > 1 ? args[1] : ev::undefined();
        auto dev = parseDevice(opts);
        auto w = std::make_unique<BirefnetWrapper>();
        w->path = path;
        w->device = dev;
        if (ev::isObject(opts)) {
            Value msv = ev::getProperty(opts, "modelSize");
            if (ev::isNumber(msv)) w->modelSize = static_cast<int>(ev::toDouble(msv));
        }
        try {
            w->net = std::make_unique<brovisionml::birefnet::BiRefNet>();
            w->net->load(path);
            w->net->to(dev);
            w->loaded = true;
        } catch (const std::exception& e) {
            return ev::throwError(std::string("loadBirefnet failed: ") + e.what());
        } catch (...) {
            return ev::throwError("loadBirefnet failed: unknown error");
        }
        return g_birefnetClass.createInstance(std::move(w));
    });

    vision.def("loadStyleGAN3", 2, [](Value, std::span<const Value> args) -> Value {
        std::string path;
        Value errVal;
        if (!validateLoaderPath("loadStyleGAN3", args, path, errVal)) return errVal;

        Value opts = args.size() > 1 ? args[1] : ev::undefined();
        auto dev = parseDevice(opts);
        auto w = std::make_unique<StyleGAN3Wrapper>();
        w->path = path;
        w->device = dev;
        // Actually construct and load the Generator: the bronze port only
        // recorded the path, which is why generate() could not do anything.
        std::string loadErr;
        if (!loadStyleGAN3Generator(path, opts, *w, loadErr)) {
            return ev::throwError(loadErr);
        }
        return g_stylegan3Class.createInstance(std::move(w));
    });

    vision.def("loadDinov2", 2, [](Value, std::span<const Value> args) -> Value {
        std::string path;
        Value errVal;
        if (!validateLoaderPath("loadDinov2", args, path, errVal)) return errVal;

        Value opts = args.size() > 1 ? args[1] : ev::undefined();
        auto dev = parseDevice(opts);
        auto w = std::make_unique<Dinov2Wrapper>();
        w->path = path;
        w->device = dev;
        std::string loadErr;
        if (!loadDinov2Backbone(path, opts, *w, loadErr)) {
            return ev::throwError("loadDinov2 failed: " + loadErr);
        }
        return g_dinov2Class.createInstance(std::move(w));
    });

    vision.def("loadDinov3", 2, [](Value, std::span<const Value> args) -> Value {
        std::string path;
        Value errVal;
        if (!validateLoaderPath("loadDinov3", args, path, errVal)) return errVal;

        Value opts = args.size() > 1 ? args[1] : ev::undefined();
        auto dev = parseDevice(opts);
        auto w = std::make_unique<Dinov3Wrapper>();
        w->path = path;
        w->device = dev;
        std::string loadErr;
        if (!loadDinov3Backbone(path, *w, loadErr)) {
            return ev::throwError("loadDinov3 failed: " + loadErr);
        }
        return g_dinov3Class.createInstance(std::move(w));
    });

    // Direct constructors
    vision.set("DepthEstimator", g_depthEstimatorClass.constructor());
    vision.set("Sam", g_samClass.constructor());
    vision.set("NormalEstimator", g_normalEstimatorClass.constructor());
    vision.set("Hed", g_hedClass.constructor());
    vision.set("Lineart", g_lineartClass.constructor());
    vision.set("Mlsd", g_mlsdClass.constructor());
    vision.set("Openpose", g_openposeClass.constructor());
    vision.set("Segformer", g_segformerClass.constructor());
    vision.set("Birefnet", g_birefnetClass.constructor());
    vision.set("StyleGAN3", g_stylegan3Class.constructor());
    vision.set("Dinov2", g_dinov2Class.constructor());
    vision.set("Dinov3", g_dinov3Class.constructor());
    vision.set("VisionModel", g_visionModelClass.constructor());

    // Ops and helpers
    mountVisionOps(vision);

    return vision.build();
}

} // namespace brovisionml::api
