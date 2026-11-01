// ControlNet annotators: HED (soft edges), Lineart, MLSD (line segments),
// OpenPose and SegFormer.
//
// Every annotator answers to both detect() and estimate(), and each result
// carries two spellings of its planes: edge / edges, line / lines, segments /
// lines (MLSD), bodies / poses, classes / segments (SegFormer). OpenPose
// keypoints carry a `present` flag, and each body its totalScore / totalParts
// (also published as `score`).
//
// A standalone sibling cannot mint a DOM ImageBitmap, so there is no `image`
// result key: the pixel planes come back as typed arrays and the caller
// rasterizes.

#include "host_vision_internal.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace brovisionml::api {

namespace {

template <typename WrapperT>
WrapperT* selfOf(const HostClass& cls, Value thisVal, uint32_t tag) {
    void* p = cls.unwrap(thisVal);
    if (!p) return nullptr;
    auto* w = static_cast<WrapperT*>(p);
    return w->tag == tag ? w : nullptr;
}

// Unit-scalar plane -> bytes, for the `edges` / `lines` keys.
std::vector<uint8_t> toBytes(const std::vector<float>& unit) {
    std::vector<uint8_t> out(unit.size());
    for (size_t i = 0; i < unit.size(); ++i) {
        out[i] = static_cast<uint8_t>(std::clamp(unit[i] * 255.0f, 0.0f, 255.0f));
    }
    return out;
}

// A scalar-plane result carrying both key spellings: `floatKey` as the FP32
// plane, `byteKey` as the same plane in bytes.
Value scalarPlaneResult(int w, int h, const std::vector<float>& plane,
                        const char* floatKey, const char* byteKey) {
    const std::vector<uint8_t> bytes = toBytes(plane);
    ObjectBuilder res;
    res.set("width", static_cast<double>(w));
    res.set("height", static_cast<double>(h));
    {
        ev::Persistent f(makeFloat32Array(plane.data(), plane.size()));
        res.set(floatKey, f.get());
    }
    {
        ev::Persistent b(makeUint8Array(bytes.data(), bytes.size()));
        res.set(byteKey, b.get());
    }
    return res.build();
}

} // namespace

// ─── HED ──────────────────────────────────────────────────────────────────

Value runHedDetect(HedWrapper* w, std::span<const Value> args) {
    if (!w || !w->loaded || !w->detector) {
        return ev::throwError("Hed: detector is uninitialized or model weights not loaded");
    }
    if (args.empty()) {
        return ev::throwTypeError("Hed.detect: image argument required");
    }

    std::vector<uint8_t> rgba;
    int width = 0, height = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, width, height, err)) {
        return ev::throwTypeError(std::string("Hed.detect: ") + err);
    }
    try {
        brotensor::DeviceScope scope(w->device);
        auto em = w->detector->detect(rgba.data(), width, height, 4);
        return scalarPlaneResult(em.width, em.height, em.edge, "edge", "edges");
    } catch (const std::exception& e) {
        return ev::throwError(std::string("HED detect failed: ") + e.what());
    }
}

Value hedDetect(Value thisVal, std::span<const Value> args) {
    auto* w = selfOf<HedWrapper>(g_hedClass, thisVal, kHostHedTag);
    if (!w) return ev::throwTypeError("Hed: not a Hed instance");
    return runHedDetect(w, args);
}

// ─── Lineart ──────────────────────────────────────────────────────────────

Value runLineartDetect(LineartWrapper* w, std::span<const Value> args) {
    if (!w || !w->loaded || !w->detector) {
        return ev::throwError("Lineart: detector is uninitialized or model weights not loaded");
    }
    if (args.empty()) {
        return ev::throwTypeError("Lineart.detect: image argument required");
    }

    std::vector<uint8_t> rgba;
    int width = 0, height = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, width, height, err)) {
        return ev::throwTypeError(std::string("Lineart.detect: ") + err);
    }
    try {
        brotensor::DeviceScope scope(w->device);
        auto lm = w->detector->detect(rgba.data(), width, height, 4);
        return scalarPlaneResult(lm.width, lm.height, lm.line, "line", "lines");
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Lineart detect failed: ") + e.what());
    }
}

Value lineartDetect(Value thisVal, std::span<const Value> args) {
    auto* w = selfOf<LineartWrapper>(g_lineartClass, thisVal, kHostLineartTag);
    if (!w) return ev::throwTypeError("Lineart: not a Lineart instance");
    return runLineartDetect(w, args);
}

// ─── MLSD ─────────────────────────────────────────────────────────────────

Value runMlsdDetect(MlsdWrapper* w, std::span<const Value> args) {
    if (!w || !w->loaded || !w->detector) {
        return ev::throwError("Mlsd: detector is uninitialized or model weights not loaded");
    }
    if (args.empty()) {
        return ev::throwTypeError("Mlsd.detect: image argument required");
    }

    std::vector<uint8_t> rgba;
    int width = 0, height = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, width, height, err)) {
        return ev::throwTypeError(std::string("Mlsd.detect: ") + err);
    }
    try {
        brotensor::DeviceScope scope(w->device);
        auto segs = w->detector->detect(rgba.data(), width, height, 4);
        ev::Persistent arr(hostArrayOf(segs.segments.size(), [&segs](size_t i) {
            ObjectBuilder o;
            o.set("x1", static_cast<double>(segs.segments[i].x1));
            o.set("y1", static_cast<double>(segs.segments[i].y1));
            o.set("x2", static_cast<double>(segs.segments[i].x2));
            o.set("y2", static_cast<double>(segs.segments[i].y2));
            o.set("score", static_cast<double>(segs.segments[i].score));
            return o.build();
        }));
        ObjectBuilder res;
        res.set("width", static_cast<double>(segs.width));
        res.set("height", static_cast<double>(segs.height));
        res.set("segments", arr.get());
        res.set("lines", arr.get());   // second name for the same array
        return res.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("MLSD detect failed: ") + e.what());
    }
}

Value mlsdDetect(Value thisVal, std::span<const Value> args) {
    auto* w = selfOf<MlsdWrapper>(g_mlsdClass, thisVal, kHostMlsdTag);
    if (!w) return ev::throwTypeError("Mlsd: not an Mlsd instance");
    return runMlsdDetect(w, args);
}

// ─── OpenPose ─────────────────────────────────────────────────────────────

Value runOpenposeDetect(OpenposeWrapper* w, std::span<const Value> args) {
    if (!w || !w->loaded || !w->detector) {
        return ev::throwError("Openpose: detector is uninitialized or model weights not loaded");
    }
    if (args.empty()) {
        return ev::throwTypeError("Openpose.detect: image argument required");
    }

    std::vector<uint8_t> rgba;
    int width = 0, height = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, width, height, err)) {
        return ev::throwTypeError(std::string("Openpose.detect: ") + err);
    }
    try {
        brotensor::DeviceScope scope(w->device);
        auto pose = w->detector->detect(rgba.data(), width, height, 4);
        ev::Persistent bodies(hostArrayOf(pose.bodies.size(), [&pose](size_t b) {
            const auto& body = pose.bodies[b];
            ev::Persistent kps(hostArrayOf(body.keypoints.size(), [&body](size_t k) {
                const auto& kp = body.keypoints[k];
                ObjectBuilder ko;
                ko.set("x", static_cast<double>(kp.x));
                ko.set("y", static_cast<double>(kp.y));
                ko.set("score", static_cast<double>(kp.score));
                ko.set("present", kp.present);
                return ko.build();
            }));
            ObjectBuilder bo;
            bo.set("keypoints", kps.get());
            bo.set("totalScore", static_cast<double>(body.total_score));
            bo.set("totalParts", static_cast<double>(body.total_parts));
            // The body score again, as `score`.
            bo.set("score", static_cast<double>(body.total_score));
            return bo.build();
        }));
        ObjectBuilder res;
        res.set("width", static_cast<double>(pose.width));
        res.set("height", static_cast<double>(pose.height));
        res.set("bodies", bodies.get());
        res.set("poses", bodies.get());   // second name for the same array
        return res.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Openpose detect failed: ") + e.what());
    }
}

Value openposeDetect(Value thisVal, std::span<const Value> args) {
    auto* w = selfOf<OpenposeWrapper>(g_openposeClass, thisVal, kHostOpenposeTag);
    if (!w) return ev::throwTypeError("Openpose: not an Openpose instance");
    return runOpenposeDetect(w, args);
}

// ─── SegFormer ────────────────────────────────────────────────────────────

Value runSegformerDetect(SegformerWrapper* w, std::span<const Value> args) {
    if (!w || !w->loaded || !w->detector) {
        return ev::throwError("Segformer: detector is uninitialized or model weights not loaded");
    }
    if (args.empty()) {
        return ev::throwTypeError("Segformer.detect: image argument required");
    }

    std::vector<uint8_t> rgba;
    int width = 0, height = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, width, height, err)) {
        return ev::throwTypeError(std::string("Segformer.detect: ") + err);
    }
    std::vector<uint8_t> classes;
    int outW = width, outH = height;
    try {
        brotensor::DeviceScope scope(w->device);
        auto sm = w->detector->detect(rgba.data(), width, height, 4);
        classes = std::move(sm.classes);
        outW = sm.width;
        outH = sm.height;
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Segformer detect failed: ") + e.what());
    }

    std::vector<int32_t> asI32(classes.begin(), classes.end());
    ObjectBuilder res;
    res.set("width", static_cast<double>(outW));
    res.set("height", static_cast<double>(outH));
    {
        // `classes`: the raw class-id bytes, ids in [0, 149].
        ev::Persistent c(makeUint8Array(classes.data(), classes.size()));
        res.set("classes", c.get());
    }
    {
        // `segments`: the same class ids as Int32.
        ev::Persistent s(makeInt32Array(asI32.data(), asI32.size()));
        res.set("segments", s.get());
    }
    return res.build();
}

Value segformerDetect(Value thisVal, std::span<const Value> args) {
    auto* w = selfOf<SegformerWrapper>(g_segformerClass, thisVal, kHostSegformerTag);
    if (!w) return ev::throwTypeError("Segformer: not a Segformer instance");
    return runSegformerDetect(w, args);
}

// Both names for the same body: `detect` and `estimate`.
template <typename Fn>
void defDetect(ObjectBuilder& proto, Fn fn) {
    proto.def("detect", 2, fn);
    proto.def("estimate", 2, fn);
}

void decorateHedProto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = selfOf<HedWrapper>(g_hedClass, thisVal, kHostHedTag);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    defDetect(proto, hedDetect);
}

void decorateLineartProto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = selfOf<LineartWrapper>(g_lineartClass, thisVal, kHostLineartTag);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    defDetect(proto, lineartDetect);
}

void decorateMlsdProto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = selfOf<MlsdWrapper>(g_mlsdClass, thisVal, kHostMlsdTag);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    defDetect(proto, mlsdDetect);
}

void decorateOpenposeProto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = selfOf<OpenposeWrapper>(g_openposeClass, thisVal, kHostOpenposeTag);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    defDetect(proto, openposeDetect);
}

void decorateSegformerProto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = selfOf<SegformerWrapper>(g_segformerClass, thisVal, kHostSegformerTag);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    defDetect(proto, segformerDetect);
}

} // namespace brovisionml::api
