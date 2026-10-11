// Sam2 / Sam2Tracker — SAM 2.1 image segmentation and one-object tracking
// through caller-supplied frames.
//
//   const sam2 = bro.vision.loadSam2(dir, { device });
//   sam2.setImage(image);
//   sam2.segment({ points, labels, box, multimask, maskInput });
//   const trk = sam2.tracker();
//   trk.addFrame(image);                 // -> frame index
//   trk.prompt(0, { points, labels, box });
//   trk.track(1, { reverse });           // or trk.propagate()
//
// These are the synchronous bodies; bro re-drives every one of them on its
// job machine (host_vision_sam2.cpp) and adds an ImageBitmap to each mask.

#include "host_vision_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace brovisionml::api {

HostClass g_sam2Class;
HostClass g_sam2TrackerClass;

Sam2Wrapper* sam2Self(Value thisVal) {
    void* p = g_sam2Class.unwrap(thisVal);
    if (!p) return nullptr;
    auto* w = static_cast<Sam2Wrapper*>(p);
    return w->tag == kHostSam2Tag ? w : nullptr;
}

Sam2TrackerWrapper* sam2TrackerSelf(Value thisVal) {
    void* p = g_sam2TrackerClass.unwrap(thisVal);
    if (!p) return nullptr;
    auto* w = static_cast<Sam2TrackerWrapper*>(p);
    return w->tag == kHostSam2TrackerTag ? w : nullptr;
}

bool readSam2Prompt(Value optsIn, brovisionml::Sam2Prompt& prompt, std::string& err) {
    prompt = {};
    if (!ev::isObject(optsIn)) {
        err = "a prompt object { points, labels, box } is required";
        return false;
    }
    ev::Persistent opts(optsIn);
    std::vector<int> inlineLabels;
    {
        ev::Persistent arr(ev::getProperty(opts.get(), "points"));
        if (isJsArray(arr.get())) {
            const uint32_t n = getJsArrayLength(arr.get());
            if (n > 4096) { err = "too many points"; return false; }
            for (uint32_t i = 0; i < n; ++i) {
                ev::Persistent e(ev::getElement(arr.get(), i));
                if (!ev::isObject(e.get())) { err = "points must be [x, y] or {x, y}"; return false; }
                if (isJsArray(e.get())) {
                    const float x = numElem(e.get(), 0);
                    const float y = numElem(e.get(), 1);
                    prompt.points.push_back({x, y});
                    inlineLabels.push_back(1);
                } else {
                    const float x = optFloat(e.get(), "x", 0.0f);
                    const float y = optFloat(e.get(), "y", 0.0f);
                    const int label = optInt(e.get(), "label", 1);
                    prompt.points.push_back({x, y});
                    inlineLabels.push_back(label);
                }
            }
        }
    }
    {
        ev::Persistent arr(ev::getProperty(opts.get(), "labels"));
        if (isJsArray(arr.get())) {
            const uint32_t n = getJsArrayLength(arr.get());
            std::vector<int> labels;
            for (uint32_t i = 0; i < n && i < 4096; ++i)
                labels.push_back(static_cast<int>(numElem(arr.get(), i)));
            if (!labels.empty()) inlineLabels = std::move(labels);
        }
    }
    inlineLabels.resize(prompt.points.size(), 1);
    for (int l : inlineLabels) {
        if (l != 0 && l != 1) { err = "labels must be 1 (foreground) or 0 (background)"; return false; }
    }
    prompt.labels = std::move(inlineLabels);
    {
        ev::Persistent b(ev::getProperty(opts.get(), "box"));
        if (ev::isObject(b.get())) {
            if (isJsArray(b.get())) {
                if (getJsArrayLength(b.get()) != 4) { err = "box must be [x0, y0, x1, y1]"; return false; }
                for (uint32_t k = 0; k < 4; ++k) prompt.box[k] = numElem(b.get(), k);
            } else {
                static const char* kKeys[4] = {"x1", "y1", "x2", "y2"};
                for (int k = 0; k < 4; ++k)
                    prompt.box[static_cast<std::size_t>(k)] = optFloat(b.get(), kKeys[k], 0.0f);
            }
            if (!(prompt.box[2] > prompt.box[0] && prompt.box[3] > prompt.box[1])) {
                err = "box must have x1 > x0 and y1 > y0";
                return false;
            }
            prompt.has_box = true;
        }
    }
    for (const auto& p : prompt.points) {
        if (!std::isfinite(p[0]) || !std::isfinite(p[1])) { err = "points must be finite"; return false; }
    }
    if (prompt.points.empty() && !prompt.has_box) {
        err = "opts.points or opts.box is required";
        return false;
    }
    return true;
}

brovisionml::Sam2Segmentation sam2Segment(const brovisionml::Sam2& sam,
                                          const brovisionml::Sam2Prompt& prompt, bool multimask,
                                          const std::vector<float>* mask_input,
                                          std::vector<float>& low_res) {
    brotensor::Tensor mi;
    if (mask_input) {
        const int ms = sam.config().mask_size();
        if (static_cast<int>(mask_input->size()) != ms * ms)
            throw std::runtime_error("maskInput must hold 256*256 low-res logits");
        mi = brotensor::Tensor::from_host(mask_input->data(), 1, ms * ms).to(sam.device());
    }
    brovisionml::Sam2Segmentation seg = sam.segment(prompt, multimask, mask_input ? &mi : nullptr);
    brotensor::Tensor h = seg.low_res.to(brotensor::Device::CPU);
    low_res.assign(h.host_f32(), h.host_f32() + h.size());
    return seg;
}

namespace {

Value boxValue(bool has, const std::array<int, 4>& box) {
    if (!has) return ev::null();
    return hostArrayOf(4, [&box](size_t k) { return ev::fromDouble(static_cast<double>(box[k])); });
}

}  // namespace

Value buildSam2Segmentation(const brovisionml::Sam2Segmentation& seg,
                            const std::vector<float>& low_res) {
    const size_t plane = static_cast<size_t>(seg.width) * static_cast<size_t>(seg.height);
    const size_t lplane = 256 * 256;
    ev::Persistent masks(hostArrayOf(static_cast<size_t>(seg.num), [&](size_t m) -> Value {
        const float* lg = seg.logits.data() + m * plane;
        std::vector<uint8_t> bin(plane);
        for (size_t i = 0; i < plane; ++i) bin[i] = lg[i] > 0.0f ? 1 : 0;
        std::array<int, 4> box{};
        const bool has = brovisionml::sam2_mask_box(lg, seg.width, seg.height, box);
        ObjectBuilder mo;
        mo.set("iou", m < seg.iou.size() ? static_cast<double>(seg.iou[m]) : 0.0);
        { ev::Persistent v(makeUint8Array(bin.data(), bin.size())); mo.set("data", v.get()); }
        { ev::Persistent v(makeFloat32Array(lg, plane)); mo.set("logits", v.get()); }
        if (low_res.size() >= (m + 1) * lplane) {
            ev::Persistent v(makeFloat32Array(low_res.data() + m * lplane, lplane));
            mo.set("lowRes", v.get());
        }
        { ev::Persistent v(boxValue(has, box)); mo.set("box", v.get()); }
        return mo.build();
    }));
    ObjectBuilder res;
    res.set("num", static_cast<double>(seg.num));
    res.set("width", static_cast<double>(seg.width));
    res.set("height", static_cast<double>(seg.height));
    res.set("best", static_cast<double>(seg.num > 0 ? seg.best() : 0));
    res.set("objectScore", static_cast<double>(seg.object_score));
    res.set("masks", masks.get());
    return res.build();
}

Value buildSam2Frame(const brovisionml::Sam2FrameResult& r) {
    const size_t plane = static_cast<size_t>(r.width) * static_cast<size_t>(r.height);
    ObjectBuilder o;
    o.set("frame", static_cast<double>(r.frame));
    o.set("width", static_cast<double>(r.width));
    o.set("height", static_cast<double>(r.height));
    const bool has_output = r.logits.size() == plane && plane > 0;
    o.set("tracked", has_output);
    o.set("present", r.present);
    o.set("objectScore", static_cast<double>(r.object_score));
    o.set("iou", static_cast<double>(r.iou));
    o.set("conditioning", r.conditioning);
    { ev::Persistent v(boxValue(r.has_box, r.box)); o.set("box", v.get()); }
    if (has_output) {
        std::vector<uint8_t> bin(plane);
        for (size_t i = 0; i < plane; ++i) bin[i] = r.logits[i] > 0.0f ? 1 : 0;
        { ev::Persistent v(makeUint8Array(bin.data(), bin.size())); o.set("data", v.get()); }
        { ev::Persistent v(makeFloat32Array(r.logits.data(), plane)); o.set("logits", v.get()); }
    } else {
        o.set("data", ev::null());
        o.set("logits", ev::null());
    }
    return o.build();
}

namespace {

// ── Sam2 ────────────────────────────────────────────────────────────────────

Value sam2SetImage(Value thisVal, std::span<const Value> args) {
    auto* w = sam2Self(thisVal);
    if (!w) return ev::throwTypeError("Sam2.prototype.setImage: not a Sam2 instance");
    if (args.empty()) return ev::throwTypeError("setImage(image, opts?): image is required");
    std::vector<uint8_t> rgba;
    int iw = 0, ih = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, iw, ih, err)) return ev::throwTypeError("setImage: " + err);
    if (!(w->loaded && w->sam)) return ev::throwError("Sam2.setImage: model is not loaded");
    try {
        brotensor::DeviceScope scope(w->device);
        w->sam->set_image(rgba.data(), iw, ih, 4);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("setImage failed: ") + e.what());
    }
    return ev::undefined();
}

Value sam2SegmentFn(Value thisVal, std::span<const Value> args) {
    auto* w = sam2Self(thisVal);
    if (!w) return ev::throwTypeError("Sam2.prototype.segment: not a Sam2 instance");
    brovisionml::Sam2Prompt prompt;
    std::string err;
    if (!readSam2Prompt(args.empty() ? ev::undefined() : args[0], prompt, err))
        return ev::throwTypeError("segment: " + err);
    const bool multimask = optBool(args[0], "multimask", true);
    std::vector<float> maskInput;
    {
        ev::Persistent mv(ev::getProperty(args[0], "maskInput"));
        const float* d = nullptr;
        size_t n = 0;
        if (readFloat32Array(mv.get(), d, n) && d) maskInput.assign(d, d + n);
    }
    if (!(w->loaded && w->sam)) return ev::throwError("Sam2.segment: model is not loaded");
    if (!w->sam->has_image()) return ev::throwError("segment: call setImage() before segment()");
    try {
        brotensor::DeviceScope scope(w->device);
        std::vector<float> low;
        auto seg = sam2Segment(*w->sam, prompt, multimask, maskInput.empty() ? nullptr : &maskInput, low);
        return buildSam2Segmentation(seg, low);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("segment failed: ") + e.what());
    }
}

Value sam2Tracker(Value thisVal, std::span<const Value>) {
    auto* w = sam2Self(thisVal);
    if (!w) return ev::throwTypeError("Sam2.prototype.tracker: not a Sam2 instance");
    if (!(w->loaded && w->sam)) return ev::throwError("Sam2.tracker: model is not loaded");
    auto t = std::make_unique<Sam2TrackerWrapper>();
    t->sam = w->sam;
    t->device = w->device;
    t->tracker = std::make_unique<brovisionml::Sam2Tracker>(*w->sam);
    return g_sam2TrackerClass.createInstance(std::move(t));
}

// ── Sam2Tracker ─────────────────────────────────────────────────────────────

int frameArg(Sam2TrackerWrapper* t, Value v, std::string& err) {
    if (!ev::isNumber(v)) { err = "frame index must be a number"; return -1; }
    const double d = ev::toDouble(v);
    if (!(d >= 0 && d < t->tracker->num_frames()) || d != std::floor(d)) {
        err = "frame index out of range";
        return -1;
    }
    return static_cast<int>(d);
}

Value trkAddFrame(Value thisVal, std::span<const Value> args) {
    auto* t = sam2TrackerSelf(thisVal);
    if (!t) return ev::throwTypeError("Sam2Tracker.prototype.addFrame: not a Sam2Tracker");
    if (args.empty()) return ev::throwTypeError("addFrame(image): image is required");
    std::vector<uint8_t> rgba;
    int iw = 0, ih = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, iw, ih, err)) return ev::throwTypeError("addFrame: " + err);
    try {
        brotensor::DeviceScope scope(t->device);
        return ev::fromDouble(t->tracker->add_frame(rgba.data(), iw, ih, 4));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("addFrame failed: ") + e.what());
    }
}

Value trkPrompt(Value thisVal, std::span<const Value> args) {
    auto* t = sam2TrackerSelf(thisVal);
    if (!t) return ev::throwTypeError("Sam2Tracker.prototype.prompt: not a Sam2Tracker");
    std::string err;
    const int f = frameArg(t, args.empty() ? ev::undefined() : args[0], err);
    if (f < 0) return ev::throwRangeError("prompt: " + err);
    brovisionml::Sam2Prompt prompt;
    if (!readSam2Prompt(args.size() > 1 ? args[1] : ev::undefined(), prompt, err))
        return ev::throwTypeError("prompt: " + err);
    try {
        brotensor::DeviceScope scope(t->device);
        return buildSam2Frame(t->tracker->prompt(f, prompt));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("prompt failed: ") + e.what());
    }
}

Value trkTrack(Value thisVal, std::span<const Value> args) {
    auto* t = sam2TrackerSelf(thisVal);
    if (!t) return ev::throwTypeError("Sam2Tracker.prototype.track: not a Sam2Tracker");
    std::string err;
    const int f = frameArg(t, args.empty() ? ev::undefined() : args[0], err);
    if (f < 0) return ev::throwRangeError("track: " + err);
    const bool reverse = args.size() > 1 && optBool(args[1], "reverse", false);
    try {
        brotensor::DeviceScope scope(t->device);
        return buildSam2Frame(t->tracker->track(f, reverse));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("track failed: ") + e.what());
    }
}

Value trkPropagate(Value thisVal, std::span<const Value>) {
    auto* t = sam2TrackerSelf(thisVal);
    if (!t) return ev::throwTypeError("Sam2Tracker.prototype.propagate: not a Sam2Tracker");
    std::vector<brovisionml::Sam2FrameResult> all;
    try {
        brotensor::DeviceScope scope(t->device);
        all = t->tracker->propagate();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("propagate failed: ") + e.what());
    }
    return hostArrayOf(all.size(), [&all](size_t i) { return buildSam2Frame(all[i]); });
}

Value trkResult(Value thisVal, std::span<const Value> args) {
    auto* t = sam2TrackerSelf(thisVal);
    if (!t) return ev::throwTypeError("Sam2Tracker.prototype.result: not a Sam2Tracker");
    std::string err;
    const int f = frameArg(t, args.empty() ? ev::undefined() : args[0], err);
    if (f < 0) return ev::throwRangeError("result: " + err);
    try {
        brotensor::DeviceScope scope(t->device);
        return buildSam2Frame(t->tracker->result(f));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("result failed: ") + e.what());
    }
}

}  // namespace

void decorateSam2Proto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sam2Self(thisVal);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    proto.accessor("hasImage", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sam2Self(thisVal);
        return ev::fromBool(w && w->sam && w->sam->has_image());
    });
    proto.def("setImage", 2, sam2SetImage);
    proto.def("segment", 1, sam2SegmentFn);
    proto.def("tracker", 0, sam2Tracker);
}

void decorateSam2TrackerProto(ObjectBuilder& proto) {
    proto.accessor("numFrames", [](Value thisVal, std::span<const Value>) -> Value {
        auto* t = sam2TrackerSelf(thisVal);
        return ev::fromDouble(t ? t->tracker->num_frames() : 0);
    });
    proto.def("addFrame", 2, trkAddFrame);
    proto.def("prompt", 3, trkPrompt);
    proto.def("track", 2, trkTrack);
    proto.def("propagate", 1, trkPropagate);
    proto.def("result", 1, trkResult);
    proto.def("reset", 0, [](Value thisVal, std::span<const Value>) -> Value {
        auto* t = sam2TrackerSelf(thisVal);
        if (!t) return ev::throwTypeError("Sam2Tracker.prototype.reset: not a Sam2Tracker");
        t->tracker->reset();
        return ev::undefined();
    });
}

}  // namespace brovisionml::api
