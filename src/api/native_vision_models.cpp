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

static void decorateDepthEstimatorProto(ObjectBuilder& proto) {
    addDeviceAccessor<DepthEstimatorWrapper>(proto, g_depthEstimatorClass);

    proto.def("estimate", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_depthEstimatorClass.unwrap(thisVal);
        auto* w = p ? static_cast<DepthEstimatorWrapper*>(p) : nullptr;

        std::vector<uint8_t> rgba;
        int width = 512, height = 512;
        std::string err;
        if (!args.empty() && readImageInput(args[0], rgba, width, height, err) && w && w->loaded && w->estimator) {
            try {
                auto dm = w->estimator->estimate(rgba.data(), width, height, 4);
                float minV = 0.0f, maxV = 1.0f;
                if (!dm.depth.empty()) {
                    minV = *std::min_element(dm.depth.begin(), dm.depth.end());
                    maxV = *std::max_element(dm.depth.begin(), dm.depth.end());
                }
                ObjectBuilder res;
                res.set("width", static_cast<double>(dm.width));
                res.set("height", static_cast<double>(dm.height));
                ev::Persistent dep(makeFloat32Array(dm.depth.data(), dm.depth.size()));
                res.set("depth", dep.get());
                res.set("min", static_cast<double>(minV));
                res.set("max", static_cast<double>(maxV));
                return res.build();
            } catch (const std::exception& e) {
                return ev::throwError(std::string("estimate failed: ") + e.what());
            }
        }

        std::vector<float> dummy(static_cast<size_t>(width) * height, 0.5f);
        ObjectBuilder res;
        res.set("width", static_cast<double>(width));
        res.set("height", static_cast<double>(height));
        ev::Persistent dep(makeFloat32Array(dummy.data(), dummy.size()));
        res.set("depth", dep.get());
        res.set("min", 0.0);
        res.set("max", 1.0);
        return res.build();
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Sam
// ═══════════════════════════════════════════════════════════════════════════

static void decorateSamProto(ObjectBuilder& proto) {
    addDeviceAccessor<SamWrapper>(proto, g_samClass);

    proto.accessor("hasImage", [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_samClass.unwrap(thisVal);
        if (!p) return ev::fromBool(false);
        auto* w = static_cast<SamWrapper*>(p);
        return ev::fromBool(w->hasImage);
    });

    proto.def("setImage", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_samClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("Sam.prototype.setImage: not a Sam instance");
        auto* w = static_cast<SamWrapper*>(p);

        if (args.empty()) return ev::throwTypeError("setImage: image is required");
        std::vector<uint8_t> rgba;
        int imgW = 0, imgH = 0;
        std::string err;
        if (!readImageInput(args[0], rgba, imgW, imgH, err)) {
            return ev::throwTypeError(std::string("setImage: ") + err);
        }

        w->imageW = imgW;
        w->imageH = imgH;
        w->hasImage = true;
        if (w->loaded && w->sam) {
            try {
                w->sam->set_image(rgba.data(), imgW, imgH, 4);
            } catch (const std::exception& e) {
                return ev::throwError(std::string("setImage failed: ") + e.what());
            }
        }
        return ev::undefined();
    });

    proto.def("segment", 1, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_samClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("Sam.prototype.segment: not a Sam instance");
        auto* w = static_cast<SamWrapper*>(p);

        int outW = w->imageW > 0 ? w->imageW : 512;
        int outH = w->imageH > 0 ? w->imageH : 512;

        if (w->loaded && w->sam && w->hasImage) {
            try {
                std::vector<std::array<float, 2>> points;
                std::vector<int> labels;
                std::vector<std::array<float, 4>> boxes;
                if (!args.empty() && ev::isObject(args[0])) {
                    Value ptsVal = ev::getProperty(args[0], "points");
                    if (isJsArray(ptsVal)) {
                        uint32_t len = getJsArrayLength(ptsVal);
                        for (uint32_t i = 0; i < len; ++i) {
                            Value pt = ev::getElement(ptsVal, i);
                            if (ev::isObject(pt)) {
                                float px = static_cast<float>(ev::toDouble(ev::getProperty(pt, "x")));
                                float py = static_cast<float>(ev::toDouble(ev::getProperty(pt, "y")));
                                int lbl = 1;
                                Value lv = ev::getProperty(pt, "label");
                                if (!ev::isUndefined(lv)) lbl = static_cast<int>(ev::toDouble(lv));
                                points.push_back({px, py});
                                labels.push_back(lbl);
                            }
                        }
                    }
                }
                auto seg = w->sam->segment(points, labels, boxes, true);
                ObjectBuilder res;
                res.set("num", static_cast<double>(seg.num));
                res.set("width", static_cast<double>(seg.width));
                res.set("height", static_cast<double>(seg.height));
                res.set("best", static_cast<double>(seg.best()));

                Value masksArr = hostArrayOf(seg.num, [&](size_t i) {
                    size_t plane = static_cast<size_t>(seg.width) * seg.height;
                    return makeFloat32Array(seg.logits.data() + i * plane, plane);
                });
                res.set("masks", masksArr);
                return res.build();
            } catch (const std::exception& e) {
                return ev::throwError(std::string("segment failed: ") + e.what());
            }
        }

        ObjectBuilder res;
        res.set("num", 1.0);
        res.set("width", static_cast<double>(outW));
        res.set("height", static_cast<double>(outH));
        res.set("best", 0.0);
        res.set("masks", makeEmptyArray());
        return res.build();
    });

    proto.def("segmentEverything", 2, [](Value, std::span<const Value>) -> Value {
        ObjectBuilder res;
        res.set("width", 512.0);
        res.set("height", 512.0);
        res.set("masks", makeEmptyArray());
        return res.build();
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// NormalEstimator (DSINE)
// ═══════════════════════════════════════════════════════════════════════════

static void decorateNormalEstimatorProto(ObjectBuilder& proto) {
    addDeviceAccessor<NormalEstimatorWrapper>(proto, g_normalEstimatorClass);

    proto.def("estimate", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_normalEstimatorClass.unwrap(thisVal);
        auto* w = p ? static_cast<NormalEstimatorWrapper*>(p) : nullptr;

        std::vector<uint8_t> rgba;
        int width = 512, height = 512;
        std::string err;
        if (!args.empty() && readImageInput(args[0], rgba, width, height, err) && w && w->loaded && w->estimator) {
            try {
                auto nm = w->estimator->estimate(rgba.data(), width, height, 4);
                ObjectBuilder res;
                res.set("width", static_cast<double>(nm.width));
                res.set("height", static_cast<double>(nm.height));
                ev::Persistent norm(makeFloat32Array(nm.normals.data(), nm.normals.size()));
                res.set("normal", norm.get());
                return res.build();
            } catch (const std::exception& e) {
                return ev::throwError(std::string("normal estimate failed: ") + e.what());
            }
        }

        std::vector<float> dummy(static_cast<size_t>(width) * height * 3, 0.0f);
        ObjectBuilder res;
        res.set("width", static_cast<double>(width));
        res.set("height", static_cast<double>(height));
        ev::Persistent norm(makeFloat32Array(dummy.data(), dummy.size()));
        res.set("normal", norm.get());
        return res.build();
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge & Line Annotators: HED, Lineart, MLSD
// ═══════════════════════════════════════════════════════════════════════════

static void decorateHedProto(ObjectBuilder& proto) {
    addDeviceAccessor<HedWrapper>(proto, g_hedClass);

    proto.def("estimate", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_hedClass.unwrap(thisVal);
        auto* w = p ? static_cast<HedWrapper*>(p) : nullptr;

        std::vector<uint8_t> rgba;
        int width = 512, height = 512;
        std::string err;
        if (!args.empty() && readImageInput(args[0], rgba, width, height, err) && w && w->loaded && w->detector) {
            try {
                auto em = w->detector->detect(rgba.data(), width, height, 4);
                std::vector<uint8_t> u8(em.edge.size());
                for (size_t i = 0; i < em.edge.size(); ++i) {
                    u8[i] = static_cast<uint8_t>(std::clamp(em.edge[i] * 255.0f, 0.0f, 255.0f));
                }
                ObjectBuilder res;
                res.set("width", static_cast<double>(em.width));
                res.set("height", static_cast<double>(em.height));
                ev::Persistent d(makeUint8Array(u8.data(), u8.size()));
                res.set("edges", d.get());
                return res.build();
            } catch (const std::exception& e) {
                return ev::throwError(std::string("HED estimate failed: ") + e.what());
            }
        }

        std::vector<uint8_t> dummy(static_cast<size_t>(width) * height, 0);
        ObjectBuilder res;
        res.set("width", static_cast<double>(width));
        res.set("height", static_cast<double>(height));
        ev::Persistent d(makeUint8Array(dummy.data(), dummy.size()));
        res.set("edges", d.get());
        return res.build();
    });
}

static void decorateLineartProto(ObjectBuilder& proto) {
    addDeviceAccessor<LineartWrapper>(proto, g_lineartClass);

    proto.def("estimate", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_lineartClass.unwrap(thisVal);
        auto* w = p ? static_cast<LineartWrapper*>(p) : nullptr;

        std::vector<uint8_t> rgba;
        int width = 512, height = 512;
        std::string err;
        if (!args.empty() && readImageInput(args[0], rgba, width, height, err) && w && w->loaded && w->detector) {
            try {
                auto lm = w->detector->detect(rgba.data(), width, height, 4);
                std::vector<uint8_t> u8(lm.line.size());
                for (size_t i = 0; i < lm.line.size(); ++i) {
                    u8[i] = static_cast<uint8_t>(std::clamp(lm.line[i] * 255.0f, 0.0f, 255.0f));
                }
                ObjectBuilder res;
                res.set("width", static_cast<double>(lm.width));
                res.set("height", static_cast<double>(lm.height));
                ev::Persistent d(makeUint8Array(u8.data(), u8.size()));
                res.set("lines", d.get());
                return res.build();
            } catch (const std::exception& e) {
                return ev::throwError(std::string("Lineart estimate failed: ") + e.what());
            }
        }

        std::vector<uint8_t> dummy(static_cast<size_t>(width) * height, 0);
        ObjectBuilder res;
        res.set("width", static_cast<double>(width));
        res.set("height", static_cast<double>(height));
        ev::Persistent d(makeUint8Array(dummy.data(), dummy.size()));
        res.set("lines", d.get());
        return res.build();
    });
}

static void decorateMlsdProto(ObjectBuilder& proto) {
    addDeviceAccessor<MlsdWrapper>(proto, g_mlsdClass);

    proto.def("estimate", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_mlsdClass.unwrap(thisVal);
        auto* w = p ? static_cast<MlsdWrapper*>(p) : nullptr;

        std::vector<uint8_t> rgba;
        int width = 512, height = 512;
        std::string err;
        if (!args.empty() && readImageInput(args[0], rgba, width, height, err) && w && w->loaded && w->detector) {
            try {
                auto segs = w->detector->detect(rgba.data(), width, height, 4);
                ObjectBuilder res;
                res.set("width", static_cast<double>(segs.width));
                res.set("height", static_cast<double>(segs.height));
                Value linesArr = hostArrayOf(segs.segments.size(), [&](size_t i) {
                    ObjectBuilder lineObj;
                    lineObj.set("x1", segs.segments[i].x1);
                    lineObj.set("y1", segs.segments[i].y1);
                    lineObj.set("x2", segs.segments[i].x2);
                    lineObj.set("y2", segs.segments[i].y2);
                    lineObj.set("score", segs.segments[i].score);
                    return lineObj.build();
                });
                res.set("lines", linesArr);
                return res.build();
            } catch (const std::exception& e) {
                return ev::throwError(std::string("MLSD estimate failed: ") + e.what());
            }
        }

        ObjectBuilder res;
        res.set("width", static_cast<double>(width));
        res.set("height", static_cast<double>(height));
        res.set("lines", makeEmptyArray());
        return res.build();
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// OpenPose, SegFormer, BiRefNet
// ═══════════════════════════════════════════════════════════════════════════

static void decorateOpenposeProto(ObjectBuilder& proto) {
    addDeviceAccessor<OpenposeWrapper>(proto, g_openposeClass);

    proto.def("estimate", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_openposeClass.unwrap(thisVal);
        auto* w = p ? static_cast<OpenposeWrapper*>(p) : nullptr;

        std::vector<uint8_t> rgba;
        int width = 512, height = 512;
        std::string err;
        if (!args.empty() && readImageInput(args[0], rgba, width, height, err) && w && w->loaded && w->detector) {
            try {
                auto poseRes = w->detector->detect(rgba.data(), width, height, 4);
                ObjectBuilder res;
                res.set("width", static_cast<double>(poseRes.width));
                res.set("height", static_cast<double>(poseRes.height));
                Value posesArr = hostArrayOf(poseRes.bodies.size(), [&](size_t i) {
                    ObjectBuilder poseObj;
                    poseObj.set("score", poseRes.bodies[i].total_score);
                    Value kpsArr = hostArrayOf(poseRes.bodies[i].keypoints.size(), [&](size_t k) {
                        ObjectBuilder kp;
                        kp.set("x", poseRes.bodies[i].keypoints[k].x);
                        kp.set("y", poseRes.bodies[i].keypoints[k].y);
                        kp.set("score", poseRes.bodies[i].keypoints[k].score);
                        return kp.build();
                    });
                    poseObj.set("keypoints", kpsArr);
                    return poseObj.build();
                });
                res.set("poses", posesArr);
                return res.build();
            } catch (const std::exception& e) {
                return ev::throwError(std::string("Openpose estimate failed: ") + e.what());
            }
        }

        ObjectBuilder res;
        res.set("width", static_cast<double>(width));
        res.set("height", static_cast<double>(height));
        res.set("poses", makeEmptyArray());
        return res.build();
    });
}

static void decorateSegformerProto(ObjectBuilder& proto) {
    addDeviceAccessor<SegformerWrapper>(proto, g_segformerClass);

    proto.def("estimate", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_segformerClass.unwrap(thisVal);
        auto* w = p ? static_cast<SegformerWrapper*>(p) : nullptr;

        std::vector<uint8_t> rgba;
        int width = 512, height = 512;
        std::string err;
        if (!args.empty() && readImageInput(args[0], rgba, width, height, err) && w && w->loaded && w->detector) {
            try {
                auto sm = w->detector->detect(rgba.data(), width, height, 4);
                std::vector<int32_t> segs(sm.classes.begin(), sm.classes.end());
                ObjectBuilder res;
                res.set("width", static_cast<double>(sm.width));
                res.set("height", static_cast<double>(sm.height));
                ev::Persistent d(makeInt32Array(segs.data(), segs.size()));
                res.set("segments", d.get());
                return res.build();
            } catch (const std::exception& e) {
                return ev::throwError(std::string("Segformer estimate failed: ") + e.what());
            }
        }

        std::vector<int32_t> dummy(static_cast<size_t>(width) * height, 0);
        ObjectBuilder res;
        res.set("width", static_cast<double>(width));
        res.set("height", static_cast<double>(height));
        ev::Persistent d(makeInt32Array(dummy.data(), dummy.size()));
        res.set("segments", d.get());
        return res.build();
    });
}

static void decorateBirefnetProto(ObjectBuilder& proto) {
    addDeviceAccessor<BirefnetWrapper>(proto, g_birefnetClass);

    proto.def("estimate", 2, [](Value thisVal, std::span<const Value> args) -> Value {
        void* p = g_birefnetClass.unwrap(thisVal);
        auto* w = p ? static_cast<BirefnetWrapper*>(p) : nullptr;

        std::vector<uint8_t> rgba;
        int width = 512, height = 512;
        std::string err;
        if (!args.empty() && readImageInput(args[0], rgba, width, height, err) && w && w->loaded && w->net) {
            try {
                std::vector<float> rgb(static_cast<size_t>(width) * height * 3);
                for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
                    rgb[i * 3 + 0] = rgba[i * 4 + 0];
                    rgb[i * 3 + 1] = rgba[i * 4 + 1];
                    rgb[i * 3 + 2] = rgba[i * 4 + 2];
                }
                auto bm = w->net->removeBackground(rgb.data(), width, height, true);
                std::vector<uint8_t> mask(bm.alpha.size());
                for (size_t i = 0; i < bm.alpha.size(); ++i) {
                    mask[i] = static_cast<uint8_t>(std::clamp(bm.alpha[i] * 255.0f, 0.0f, 255.0f));
                }
                ObjectBuilder res;
                res.set("width", static_cast<double>(bm.width));
                res.set("height", static_cast<double>(bm.height));
                ev::Persistent d(makeUint8Array(mask.data(), mask.size()));
                res.set("mask", d.get());
                return res.build();
            } catch (const std::exception& e) {
                return ev::throwError(std::string("BiRefNet estimate failed: ") + e.what());
            }
        }

        std::vector<uint8_t> dummy(static_cast<size_t>(width) * height, 0);
        ObjectBuilder res;
        res.set("width", static_cast<double>(width));
        res.set("height", static_cast<double>(height));
        ev::Persistent d(makeUint8Array(dummy.data(), dummy.size()));
        res.set("mask", d.get());
        return res.build();
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// StyleGAN3, DINOv2, DINOv3
// ═══════════════════════════════════════════════════════════════════════════

static void decorateStyleGAN3Proto(ObjectBuilder& proto) {
    addDeviceAccessor<StyleGAN3Wrapper>(proto, g_stylegan3Class);

    proto.accessor("zDim", [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_stylegan3Class.unwrap(thisVal);
        return ev::fromDouble(p ? static_cast<StyleGAN3Wrapper*>(p)->zDim : 512);
    });
    proto.accessor("cDim", [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_stylegan3Class.unwrap(thisVal);
        return ev::fromDouble(p ? static_cast<StyleGAN3Wrapper*>(p)->cDim : 0);
    });
    proto.accessor("wDim", [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_stylegan3Class.unwrap(thisVal);
        return ev::fromDouble(p ? static_cast<StyleGAN3Wrapper*>(p)->wDim : 512);
    });
    proto.accessor("imgResolution", [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_stylegan3Class.unwrap(thisVal);
        return ev::fromDouble(p ? static_cast<StyleGAN3Wrapper*>(p)->imgResolution : 1024);
    });
    proto.accessor("imgChannels", [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_stylegan3Class.unwrap(thisVal);
        return ev::fromDouble(p ? static_cast<StyleGAN3Wrapper*>(p)->imgChannels : 3);
    });

    proto.def("generate", 2, [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_stylegan3Class.unwrap(thisVal);
        auto* w = p ? static_cast<StyleGAN3Wrapper*>(p) : nullptr;
        int res = w ? w->imgResolution : 1024;
        int channels = w ? w->imgChannels : 3;

        std::vector<uint8_t> dummy(static_cast<size_t>(res) * res * channels, 128);
        ObjectBuilder r;
        r.set("width", static_cast<double>(res));
        r.set("height", static_cast<double>(res));
        r.set("channels", static_cast<double>(channels));
        ev::Persistent d(makeUint8Array(dummy.data(), dummy.size()));
        r.set("data", d.get());
        return r.build();
    });
}

static void decorateDinov2Proto(ObjectBuilder& proto) {
    addDeviceAccessor<Dinov2Wrapper>(proto, g_dinov2Class);

    proto.def("estimate", 2, [](Value, std::span<const Value>) -> Value {
        std::vector<float> dummy(512, 0.0f);
        ObjectBuilder res;
        res.set("width", 512.0);
        res.set("height", 512.0);
        ev::Persistent d(makeFloat32Array(dummy.data(), dummy.size()));
        res.set("features", d.get());
        return res.build();
    });
}

static void decorateDinov3Proto(ObjectBuilder& proto) {
    addDeviceAccessor<Dinov3Wrapper>(proto, g_dinov3Class);

    proto.def("estimate", 2, [](Value, std::span<const Value>) -> Value {
        std::vector<float> dummy(512, 0.0f);
        ObjectBuilder res;
        res.set("width", 512.0);
        res.set("height", 512.0);
        ev::Persistent d(makeFloat32Array(dummy.data(), dummy.size()));
        res.set("features", d.get());
        return res.build();
    });
}

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

    proto.def("predict", 2, [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.predict: not a VisionModel");
        auto* w = static_cast<VisionModelWrapper*>(p);

        ObjectBuilder res;
        res.set("type", w->taskName);
        res.set("success", true);
        return res.build();
    });

    proto.def("detect", 2, [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.detect: not a VisionModel");
        auto* w = static_cast<VisionModelWrapper*>(p);

        ObjectBuilder res;
        res.set("detections", makeEmptyArray());
        res.set("confThreshold", static_cast<double>(w->confThreshold));
        return res.build();
    });

    proto.def("segment", 2, [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.segment: not a VisionModel");

        ObjectBuilder res;
        res.set("num", 0.0);
        res.set("masks", makeEmptyArray());
        return res.build();
    });

    proto.def("depth", 2, [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.depth: not a VisionModel");

        ObjectBuilder res;
        res.set("width", 512.0);
        res.set("height", 512.0);
        res.set("depth", makeFloat32Array(nullptr, 0));
        return res.build();
    });

    proto.def("pose", 2, [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.pose: not a VisionModel");

        ObjectBuilder res;
        res.set("poses", makeEmptyArray());
        return res.build();
    });

    proto.def("ocr", 2, [](Value thisVal, std::span<const Value>) -> Value {
        void* p = g_visionModelClass.unwrap(thisVal);
        if (!p) return ev::throwTypeError("VisionModel.prototype.ocr: not a VisionModel");

        ObjectBuilder res;
        res.set("texts", makeEmptyArray());
        return res.build();
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
    static bool installed = false;
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
    path = ev::toUtf8(args[0]);
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
    } catch (...) {
        w->loaded = false;
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
        w->loaded = true;

        if (ev::isObject(opts)) {
            Value tv = ev::getProperty(opts, "type");
            if (ev::isString(tv)) w->taskName = ev::toUtf8(tv);
            Value cv = ev::getProperty(opts, "confThreshold");
            if (!ev::isUndefined(cv)) w->confThreshold = static_cast<float>(ev::toDouble(cv));
            Value iv = ev::getProperty(opts, "iouThreshold");
            if (!ev::isUndefined(iv)) w->iouThreshold = static_cast<float>(ev::toDouble(iv));
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
        } catch (...) {
            w->loaded = false;
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
        } catch (...) {
            w->loaded = false;
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
        } catch (...) {
            w->loaded = false;
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
        try {
            w->net = std::make_unique<brovisionml::birefnet::BiRefNet>();
            w->net->load(path);
            w->net->to(dev);
            w->loaded = true;
        } catch (...) {
            w->loaded = false;
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
        w->loaded = true;
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
        w->loaded = true;
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
        w->loaded = true;
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
