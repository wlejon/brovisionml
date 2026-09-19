// ControlNet annotators: HED (soft edges), Lineart, MLSD (line segments),
// OpenPose and SegFormer.
//
// Ported from the QuickJS-era src/js/vision_bindings.cpp (js_hed_detect,
// js_lineart_detect, js_mlsd_detect, js_openpose_detect, js_segformer_detect).
// The bronze port renamed every detect() to estimate() and renamed the result
// planes with it (edge -> edges, line -> lines, segments -> lines, bodies ->
// poses, classes -> segments), and dropped the per-keypoint `present` flag and
// the per-body totalScore / totalParts. Both names are published here: the old
// one because callers and docs use it, the new one so the port's callers keep
// working.
//
// The old `image` result key was an ImageBitmap built by bro's DOM; a
// standalone sibling cannot mint one, so the pixel planes come back as typed
// arrays and the caller rasterizes.

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

// Unit-scalar plane -> bytes, for the port's `edges` / `lines` keys.
std::vector<uint8_t> toBytes(const std::vector<float>& unit) {
    std::vector<uint8_t> out(unit.size());
    for (size_t i = 0; i < unit.size(); ++i) {
        out[i] = static_cast<uint8_t>(std::clamp(unit[i] * 255.0f, 0.0f, 255.0f));
    }
    return out;
}

// A scalar-plane result carrying both key spellings: `oldKey` as the FP32 plane
// the pre-transition binding returned, `newKey` as the port's byte plane.
Value scalarPlaneResult(int w, int h, const std::vector<float>& plane,
                        const char* oldKey, const char* newKey) {
    const std::vector<uint8_t> bytes = toBytes(plane);
    ObjectBuilder res;
    res.set("width", static_cast<double>(w));
    res.set("height", static_cast<double>(h));
    {
        ev::Persistent f(makeFloat32Array(plane.data(), plane.size()));
        res.set(oldKey, f.get());
    }
    {
        ev::Persistent b(makeUint8Array(bytes.data(), bytes.size()));
        res.set(newKey, b.get());
    }
    return res.build();
}

// ─── HED ──────────────────────────────────────────────────────────────────

Value hedDetect(Value thisVal, std::span<const Value> args) {
    auto* w = selfOf<HedWrapper>(g_hedClass, thisVal, kHostHedTag);

    std::vector<uint8_t> rgba;
    int width = 512, height = 512;
    std::string err;
    if (!args.empty() && readImageInput(args[0], rgba, width, height, err) &&
        w && w->loaded && w->detector) {
        try {
            auto em = w->detector->detect(rgba.data(), width, height, 4);
            return scalarPlaneResult(em.width, em.height, em.edge, "edge", "edges");
        } catch (const std::exception& e) {
            return ev::throwError(std::string("HED detect failed: ") + e.what());
        }
    }
    std::vector<float> dummy(static_cast<size_t>(width) * height, 0.0f);
    return scalarPlaneResult(width, height, dummy, "edge", "edges");
}

// ─── Lineart ──────────────────────────────────────────────────────────────

Value lineartDetect(Value thisVal, std::span<const Value> args) {
    auto* w = selfOf<LineartWrapper>(g_lineartClass, thisVal, kHostLineartTag);

    std::vector<uint8_t> rgba;
    int width = 512, height = 512;
    std::string err;
    if (!args.empty() && readImageInput(args[0], rgba, width, height, err) &&
        w && w->loaded && w->detector) {
        try {
            auto lm = w->detector->detect(rgba.data(), width, height, 4);
            return scalarPlaneResult(lm.width, lm.height, lm.line, "line", "lines");
        } catch (const std::exception& e) {
            return ev::throwError(std::string("Lineart detect failed: ") + e.what());
        }
    }
    std::vector<float> dummy(static_cast<size_t>(width) * height, 0.0f);
    return scalarPlaneResult(width, height, dummy, "line", "lines");
}

// ─── MLSD ─────────────────────────────────────────────────────────────────

Value mlsdDetect(Value thisVal, std::span<const Value> args) {
    auto* w = selfOf<MlsdWrapper>(g_mlsdClass, thisVal, kHostMlsdTag);

    std::vector<uint8_t> rgba;
    int width = 512, height = 512;
    std::string err;
    if (!args.empty() && readImageInput(args[0], rgba, width, height, err) &&
        w && w->loaded && w->detector) {
        try {
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
            res.set("lines", arr.get());   // the port's name for the same array
            return res.build();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("MLSD detect failed: ") + e.what());
        }
    }
    ObjectBuilder res;
    res.set("width", static_cast<double>(width));
    res.set("height", static_cast<double>(height));
    ev::Persistent empty(makeEmptyArray());
    res.set("segments", empty.get());
    res.set("lines", empty.get());
    return res.build();
}

// ─── OpenPose ─────────────────────────────────────────────────────────────

Value openposeDetect(Value thisVal, std::span<const Value> args) {
    auto* w = selfOf<OpenposeWrapper>(g_openposeClass, thisVal, kHostOpenposeTag);

    std::vector<uint8_t> rgba;
    int width = 512, height = 512;
    std::string err;
    if (!args.empty() && readImageInput(args[0], rgba, width, height, err) &&
        w && w->loaded && w->detector) {
        try {
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
                // The port published the body score as `score`; keep it.
                bo.set("score", static_cast<double>(body.total_score));
                return bo.build();
            }));
            ObjectBuilder res;
            res.set("width", static_cast<double>(pose.width));
            res.set("height", static_cast<double>(pose.height));
            res.set("bodies", bodies.get());
            res.set("poses", bodies.get());   // the port's name for the same array
            return res.build();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("Openpose detect failed: ") + e.what());
        }
    }
    ObjectBuilder res;
    res.set("width", static_cast<double>(width));
    res.set("height", static_cast<double>(height));
    ev::Persistent empty(makeEmptyArray());
    res.set("bodies", empty.get());
    res.set("poses", empty.get());
    return res.build();
}

// ─── SegFormer ────────────────────────────────────────────────────────────

Value segformerDetect(Value thisVal, std::span<const Value> args) {
    auto* w = selfOf<SegformerWrapper>(g_segformerClass, thisVal, kHostSegformerTag);

    std::vector<uint8_t> rgba;
    int width = 512, height = 512;
    std::string err;
    std::vector<uint8_t> classes;
    int outW = width, outH = height;
    if (!args.empty() && readImageInput(args[0], rgba, width, height, err) &&
        w && w->loaded && w->detector) {
        try {
            auto sm = w->detector->detect(rgba.data(), width, height, 4);
            classes = std::move(sm.classes);
            outW = sm.width;
            outH = sm.height;
        } catch (const std::exception& e) {
            return ev::throwError(std::string("Segformer detect failed: ") + e.what());
        }
    } else {
        classes.assign(static_cast<size_t>(width) * height, 0);
    }

    std::vector<int32_t> asI32(classes.begin(), classes.end());
    ObjectBuilder res;
    res.set("width", static_cast<double>(outW));
    res.set("height", static_cast<double>(outH));
    {
        // `classes` is the old key: the raw class-id bytes, ids in [0, 149].
        ev::Persistent c(makeUint8Array(classes.data(), classes.size()));
        res.set("classes", c.get());
    }
    {
        // `segments` is the port's name, as Int32 — kept for its callers.
        ev::Persistent s(makeInt32Array(asI32.data(), asI32.size()));
        res.set("segments", s.get());
    }
    return res.build();
}

// Both names for the same body: `detect` is what the pre-transition binding
// published, `estimate` is what the bronze port renamed it to.
template <typename Fn>
void defDetect(ObjectBuilder& proto, Fn fn) {
    proto.def("detect", 2, fn);
    proto.def("estimate", 2, fn);
}

} // namespace

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
