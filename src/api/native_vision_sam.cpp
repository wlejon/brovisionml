// Sam — promptable segmentation. setImage() caches the ViT embedding,
// segment() is the cheap per-click decode, segmentEverything() drives the
// AutomaticMaskGenerator.
//
// Ported from the QuickJS-era src/js/vision_bindings.cpp (js_sam_setImage /
// js_sam_segment / js_sam_segmentEverything). The bronze port had lost
// segment()'s labels / boxes / multimask options and left segmentEverything()
// as a fixed 512x512 empty result.

#include "host_vision_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace brovisionml::api {

namespace {

SamWrapper* samSelf(Value thisVal) {
    void* p = g_samClass.unwrap(thisVal);
    if (!p) return nullptr;
    auto* w = static_cast<SamWrapper*>(p);
    return w->tag == kHostSamTag ? w : nullptr;
}

// [[x, y], ...] or [{x, y, label?}, ...]. The object form also carries the
// per-point label, which the bronze port read but the array form did not.
void readPointsAndLabels(Value v, std::vector<std::array<float, 2>>& points,
                         std::vector<int>& inlineLabels) {
    if (!isJsArray(v)) return;
    ev::Persistent arr(v);
    const uint32_t n = getJsArrayLength(arr.get());
    for (uint32_t i = 0; i < n; ++i) {
        ev::Persistent e(ev::getElement(arr.get(), i));
        if (!ev::isObject(e.get())) continue;
        if (isJsArray(e.get())) {
            float x = static_cast<float>(ev::toDouble(ev::getElement(e.get(), 0)));
            float y = static_cast<float>(ev::toDouble(ev::getElement(e.get(), 1)));
            points.push_back({x, y});
            inlineLabels.push_back(1);
        } else {
            float x = static_cast<float>(ev::toDouble(ev::getProperty(e.get(), "x")));
            float y = static_cast<float>(ev::toDouble(ev::getProperty(e.get(), "y")));
            Value lv = ev::getProperty(e.get(), "label");
            points.push_back({x, y});
            inlineLabels.push_back(ev::isNumber(lv) ? static_cast<int>(ev::toDouble(lv)) : 1);
        }
    }
}

std::vector<int> readInts(Value v) {
    std::vector<int> out;
    if (!isJsArray(v)) return out;
    ev::Persistent arr(v);
    const uint32_t n = getJsArrayLength(arr.get());
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        out.push_back(static_cast<int>(ev::toDouble(ev::getElement(arr.get(), i))));
    }
    return out;
}

// [[x1, y1, x2, y2], ...] or [{x1, y1, x2, y2}, ...].
std::vector<std::array<float, 4>> readBoxes(Value v) {
    std::vector<std::array<float, 4>> out;
    if (!isJsArray(v)) return out;
    ev::Persistent arr(v);
    const uint32_t n = getJsArrayLength(arr.get());
    for (uint32_t i = 0; i < n; ++i) {
        ev::Persistent e(ev::getElement(arr.get(), i));
        if (!ev::isObject(e.get())) continue;
        std::array<float, 4> b{};
        if (isJsArray(e.get())) {
            for (uint32_t k = 0; k < 4; ++k) {
                b[k] = static_cast<float>(ev::toDouble(ev::getElement(e.get(), k)));
            }
        } else {
            static const char* kKeys[4] = {"x1", "y1", "x2", "y2"};
            for (int k = 0; k < 4; ++k) {
                b[static_cast<size_t>(k)] =
                    static_cast<float>(ev::toDouble(ev::getProperty(e.get(), kKeys[k])));
            }
        }
        out.push_back(b);
    }
    return out;
}

Value samSetImage(Value thisVal, std::span<const Value> args) {
    auto* w = samSelf(thisVal);
    if (!w) return ev::throwTypeError("Sam.prototype.setImage: not a Sam instance");
    if (args.empty()) return ev::throwTypeError("setImage(image, opts?): image is required");

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
            brotensor::DeviceScope scope(w->device);
            w->sam->set_image(rgba.data(), imgW, imgH, 4);
        } catch (const std::exception& e) {
            return ev::throwError(std::string("setImage failed: ") + e.what());
        }
    }
    return ev::undefined();
}

// segment(opts) -> { num, width, height, best, masks }
//   opts.points    [[x,y], ...] or [{x, y, label?}, ...] in image pixels
//   opts.labels    [1, 0, ...]  1 = foreground, 0 = background (default all 1)
//   opts.boxes     [[x1,y1,x2,y2], ...]
//   opts.multimask bool (default true)
Value samSegment(Value thisVal, std::span<const Value> args) {
    auto* w = samSelf(thisVal);
    if (!w) return ev::throwTypeError("Sam.prototype.segment: not a Sam instance");

    std::vector<std::array<float, 2>> points;
    std::vector<int> inlineLabels;
    std::vector<int> labels;
    std::vector<std::array<float, 4>> boxes;
    bool multimask = true;

    if (!args.empty() && ev::isObject(args[0])) {
        ev::Persistent opts(args[0]);
        readPointsAndLabels(ev::getProperty(opts.get(), "points"), points, inlineLabels);
        labels = readInts(ev::getProperty(opts.get(), "labels"));
        boxes = readBoxes(ev::getProperty(opts.get(), "boxes"));
        Value mm = ev::getProperty(opts.get(), "multimask");
        if (!ev::isUndefined(mm) && !ev::isNull(mm)) multimask = ev::toBool(mm);
    }
    // An explicit labels array wins; otherwise the per-point `label` fields
    // (all 1 when the caller passed bare [x, y] pairs).
    if (labels.empty()) labels = inlineLabels;
    if (labels.size() < points.size()) labels.resize(points.size(), 1);

    // Argument validation runs before the model check so a prompt-less call is
    // a TypeError whether or not weights happen to be loaded.
    if (points.empty() && boxes.empty()) {
        return ev::throwTypeError("segment: opts.points or opts.boxes required");
    }
    if (!(w->loaded && w->sam)) {
        return ev::throwError("Sam.segment: model is not initialized/loaded");
    }
    if (!w->sam->has_image()) {
        return ev::throwError("segment: call setImage() before segment()");
    }

    try {
        brotensor::DeviceScope scope(w->device);
        auto seg = w->sam->segment(points, labels, boxes, multimask);
        return buildSegmentation(seg);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("segment failed: ") + e.what());
    }
}

// segmentEverything(image, opts?) -> { width, height, masks }
//   masks sorted by descending area; each is
//   { data, bbox, area, predictedIou, stabilityScore, point }.
//   opts.pointsPerSide / pointsPerBatch / predIouThresh / stabilityThresh /
//   boxNmsThresh / cropNLayers / minMaskRegionArea — defaults mirror upstream.
Value samSegmentEverything(Value thisVal, std::span<const Value> args) {
    auto* w = samSelf(thisVal);
    if (!w) return ev::throwTypeError("Sam.prototype.segmentEverything: not a Sam instance");
    if (args.empty()) {
        return ev::throwTypeError("segmentEverything(image, opts?): image required");
    }

    std::vector<uint8_t> rgba;
    int inW = 0, inH = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, inW, inH, err)) {
        return ev::throwTypeError(std::string("segmentEverything: ") + err);
    }

    brovisionml::sam::AmgConfig cfg;
    if (args.size() > 1 && ev::isObject(args[1])) {
        ev::Persistent o(args[1]);
        Value v = ev::getProperty(o.get(), "pointsPerSide");
        if (ev::isNumber(v)) cfg.points_per_side = static_cast<int>(ev::toDouble(v));
        v = ev::getProperty(o.get(), "pointsPerBatch");
        if (ev::isNumber(v)) cfg.points_per_batch = static_cast<int>(ev::toDouble(v));
        v = ev::getProperty(o.get(), "predIouThresh");
        if (ev::isNumber(v)) cfg.pred_iou_thresh = static_cast<float>(ev::toDouble(v));
        v = ev::getProperty(o.get(), "stabilityThresh");
        if (ev::isNumber(v)) cfg.stability_score_thresh = static_cast<float>(ev::toDouble(v));
        v = ev::getProperty(o.get(), "boxNmsThresh");
        if (ev::isNumber(v)) cfg.box_nms_thresh = static_cast<float>(ev::toDouble(v));
        v = ev::getProperty(o.get(), "cropNLayers");
        if (ev::isNumber(v)) cfg.crop_n_layers = static_cast<int>(ev::toDouble(v));
        v = ev::getProperty(o.get(), "minMaskRegionArea");
        if (ev::isNumber(v)) cfg.min_mask_region_area = static_cast<int>(ev::toDouble(v));
    }

    if (!(w->loaded && w->sam)) {
        return ev::throwError("Sam.segmentEverything: model is not initialized/loaded");
    }

    std::vector<brovisionml::sam::GeneratedMask> masks;
    try {
        brotensor::DeviceScope scope(w->device);
        brovisionml::sam::AutomaticMaskGenerator gen(*w->sam, cfg);
        masks = gen.generate(rgba.data(), inW, inH, 4);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("segmentEverything failed: ") + e.what());
    }
    // generate() leaves its own image cached on the model.
    w->hasImage = true;
    w->imageW = inW;
    w->imageH = inH;

    ev::Persistent arr(hostArrayOf(masks.size(), [&masks](size_t i) -> Value {
        const auto& gm = masks[i];
        ObjectBuilder mo;
        {
            ev::Persistent d(makeUint8Array(gm.mask.data(), gm.mask.size()));
            mo.set("data", d.get());
        }
        mo.set("width", static_cast<double>(gm.width));
        mo.set("height", static_cast<double>(gm.height));
        {
            ev::Persistent bbox(hostArrayOf(4, [&gm](size_t k) {
                return ev::fromDouble(static_cast<double>(gm.bbox[k]));
            }));
            mo.set("bbox", bbox.get());
        }
        mo.set("area", static_cast<double>(gm.area));
        mo.set("predictedIou", static_cast<double>(gm.predicted_iou));
        mo.set("stabilityScore", static_cast<double>(gm.stability_score));
        {
            ev::Persistent pt(hostArrayOf(2, [&gm](size_t k) {
                return ev::fromDouble(static_cast<double>(gm.point[k]));
            }));
            mo.set("point", pt.get());
        }
        return mo.build();
    }));

    ObjectBuilder res;
    res.set("width", static_cast<double>(inW));
    res.set("height", static_cast<double>(inH));
    res.set("masks", arr.get());
    return res.build();
}

} // namespace

Value buildSegmentation(const brovisionml::sam::Segmentation& seg) {
    const int W = seg.width, H = seg.height;
    const size_t plane = static_cast<size_t>(W) * static_cast<size_t>(H);

    ev::Persistent masks(hostArrayOf(static_cast<size_t>(seg.num), [&](size_t m) -> Value {
        std::vector<uint8_t> bin(plane);
        const float* lg = seg.logits.data() + m * plane;
        for (size_t i = 0; i < plane; ++i) bin[i] = lg[i] > 0.0f ? 1 : 0;
        ObjectBuilder mo;
        mo.set("iou", m < seg.iou.size() ? static_cast<double>(seg.iou[m]) : 0.0);
        {
            ev::Persistent d(makeUint8Array(bin.data(), bin.size()));
            mo.set("data", d.get());
        }
        // The raw per-mask logits, which the bronze port returned under
        // `masks`; keeping them means neither caller has to change.
        {
            ev::Persistent lo(makeFloat32Array(lg, plane));
            mo.set("logits", lo.get());
        }
        return mo.build();
    }));

    ObjectBuilder res;
    res.set("num", static_cast<double>(seg.num));
    res.set("width", static_cast<double>(W));
    res.set("height", static_cast<double>(H));
    res.set("best", static_cast<double>(seg.best()));
    res.set("masks", masks.get());
    return res.build();
}

void decorateSamProto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = samSelf(thisVal);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });

    proto.accessor("hasImage", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = samSelf(thisVal);
        return ev::fromBool(w && w->hasImage);
    });

    proto.def("setImage", 2, samSetImage);
    proto.def("segment", 1, samSegment);
    proto.def("segmentEverything", 2, samSegmentEverything);
}

} // namespace brovisionml::api
