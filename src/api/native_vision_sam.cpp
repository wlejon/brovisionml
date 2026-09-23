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
// Every reader below roots its argument before the first allocating call
// (isJsArray reads `length`), and each element before its second read.
void readPointsAndLabels(Value v, std::vector<std::array<float, 2>>& points,
                         std::vector<int>& inlineLabels) {
    ev::Persistent arr(v);
    if (auto tinfo = ev::typedArrayInfo(arr.get())) {
        if (tinfo.elementKind == ev::elements::Float32 && tinfo.data) {
            const float* p = reinterpret_cast<const float*>(tinfo.data);
            const size_t n = tinfo.elementCount / 2;
            points.reserve(points.size() + n);
            inlineLabels.reserve(inlineLabels.size() + n);
            for (size_t i = 0; i < n; ++i) {
                points.push_back({p[2 * i], p[2 * i + 1]});
                inlineLabels.push_back(1);
            }
            return;
        }
    }
    if (!isJsArray(arr.get())) return;
    const uint32_t n = getJsArrayLength(arr.get());
    points.reserve(points.size() + std::min<uint32_t>(n, 1u << 16));
    inlineLabels.reserve(inlineLabels.size() + std::min<uint32_t>(n, 1u << 16));
    for (uint32_t i = 0; i < n; ++i) {
        ev::Persistent e(ev::getElement(arr.get(), i));
        if (!ev::isObject(e.get())) continue;
        if (isJsArray(e.get())) {
            const float x = numElem(e.get(), 0);
            const float y = numElem(e.get(), 1);
            points.push_back({x, y});
            inlineLabels.push_back(1);
        } else {
            const float x = optFloat(e.get(), "x", 0.0f);
            const float y = optFloat(e.get(), "y", 0.0f);
            const int label = optInt(e.get(), "label", 1);
            points.push_back({x, y});
            inlineLabels.push_back(label);
        }
    }
}

std::vector<int> readInts(Value v) {
    std::vector<int> out;
    ev::Persistent arr(v);
    if (auto tinfo = ev::typedArrayInfo(arr.get())) {
        if (tinfo.elementKind == ev::elements::Int32 && tinfo.data) {
            const int32_t* p = reinterpret_cast<const int32_t*>(tinfo.data);
            out.assign(p, p + tinfo.elementCount);
            return out;
        }
    }
    if (!isJsArray(arr.get())) return out;
    const uint32_t n = getJsArrayLength(arr.get());
    out.reserve(std::min<uint32_t>(n, 1u << 16));
    for (uint32_t i = 0; i < n; ++i) {
        out.push_back(static_cast<int>(numElem(arr.get(), i)));
    }
    return out;
}

// [[x1, y1, x2, y2], ...] or [{x1, y1, x2, y2}, ...].
std::vector<std::array<float, 4>> readBoxes(Value v) {
    std::vector<std::array<float, 4>> out;
    ev::Persistent arr(v);
    if (auto tinfo = ev::typedArrayInfo(arr.get())) {
        if (tinfo.elementKind == ev::elements::Float32 && tinfo.data) {
            const float* p = reinterpret_cast<const float*>(tinfo.data);
            const size_t n = tinfo.elementCount / 4;
            out.reserve(out.size() + n);
            for (size_t i = 0; i < n; ++i) {
                out.push_back({p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]});
            }
            return out;
        }
    }
    if (!isJsArray(arr.get())) return out;
    const uint32_t n = getJsArrayLength(arr.get());
    out.reserve(std::min<uint32_t>(n, 1u << 16));
    for (uint32_t i = 0; i < n; ++i) {
        ev::Persistent e(ev::getElement(arr.get(), i));
        if (!ev::isObject(e.get())) continue;
        std::array<float, 4> b{};
        if (isJsArray(e.get())) {
            for (uint32_t k = 0; k < 4; ++k) {
                b[k] = numElem(e.get(), k);
            }
        } else {
            static const char* kKeys[4] = {"x1", "y1", "x2", "y2"};
            for (int k = 0; k < 4; ++k) {
                b[static_cast<size_t>(k)] = optFloat(e.get(), kKeys[k], 0.0f);
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
static Value samSegmentDirect(SamWrapper* w, std::span<const Value> args) {
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

Value samSegment(Value thisVal, std::span<const Value> args) {
    auto* w = samSelf(thisVal);
    if (!w) return ev::throwTypeError("Sam.prototype.segment: not a Sam instance");
    return samSegmentDirect(w, args);
}

// segmentEverything(image, opts?) -> { width, height, masks }
//   masks sorted by descending area; each is
//   { data, bbox, area, predictedIou, stabilityScore, point }.
//   opts.pointsPerSide / pointsPerBatch / predIouThresh / stabilityThresh /
//   boxNmsThresh / cropNLayers / minMaskRegionArea — defaults mirror upstream.
static Value samSegmentEverythingDirect(SamWrapper* w, std::span<const Value> args) {
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
        cfg.points_per_side = optInt(args[1], "pointsPerSide", cfg.points_per_side);
        cfg.points_per_batch = optInt(args[1], "pointsPerBatch", cfg.points_per_batch);
        cfg.pred_iou_thresh = optFloat(args[1], "predIouThresh", cfg.pred_iou_thresh);
        cfg.stability_score_thresh = optFloat(args[1], "stabilityThresh", cfg.stability_score_thresh);
        cfg.box_nms_thresh = optFloat(args[1], "boxNmsThresh", cfg.box_nms_thresh);
        cfg.crop_n_layers = optInt(args[1], "cropNLayers", cfg.crop_n_layers);
        cfg.min_mask_region_area = optInt(args[1], "minMaskRegionArea", cfg.min_mask_region_area);
    }
    // The grid is pointsPerSide^2 prompts per crop and there are up to
    // 4^cropNLayers crops per layer: bound both before the generator sizes
    // anything from them.
    if (cfg.points_per_side < 1 || cfg.points_per_side > 256 || cfg.points_per_batch < 1 ||
        cfg.crop_n_layers < 0 || cfg.crop_n_layers > 4 || cfg.min_mask_region_area < 0) {
        return ev::throwRangeError(
            "segmentEverything: pointsPerSide must be in [1, 256], pointsPerBatch >= 1, "
            "cropNLayers in [0, 4] and minMaskRegionArea >= 0");
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
        mo.set("data", makeUint8Array(gm.mask.data(), gm.mask.size()));
        mo.set("width", static_cast<double>(gm.width));
        mo.set("height", static_cast<double>(gm.height));
        mo.set("bbox", hostArrayOf(4, [&gm](size_t k) {
            return ev::fromDouble(static_cast<double>(gm.bbox[k]));
        }));
        mo.set("area", static_cast<double>(gm.area));
        mo.set("predictedIou", static_cast<double>(gm.predicted_iou));
        mo.set("stabilityScore", static_cast<double>(gm.stability_score));
        mo.set("point", hostArrayOf(2, [&gm](size_t k) {
            return ev::fromDouble(static_cast<double>(gm.point[k]));
        }));
        return mo.build();
    }));

    ObjectBuilder res;
    res.set("num", static_cast<double>(masks.size()));
    res.set("width", static_cast<double>(inW));
    res.set("height", static_cast<double>(inH));
    res.set("masks", arr.get());
    return res.build();
}

Value samSegmentEverything(Value thisVal, std::span<const Value> args) {
    auto* w = samSelf(thisVal);
    if (!w) return ev::throwTypeError("Sam.prototype.segmentEverything: not a Sam instance");
    return samSegmentEverythingDirect(w, args);
}

} // namespace

Value runSamSegment(SamWrapper* w, std::span<const Value> args) {
    if (!w) return ev::throwTypeError("Sam: not a Sam instance");
    if (!(w->loaded && w->sam)) {
        return ev::throwError("Sam: model is not initialized/loaded");
    }
    if (args.empty()) {
        return ev::throwTypeError("Sam.segment: image or prompt argument required");
    }

    std::vector<uint8_t> rgba;
    int inW = 0, inH = 0;
    std::string err;
    if (readImageInput(args[0], rgba, inW, inH, err)) {
        if (args.size() > 1 && ev::isObject(args[1])) {
            // One read, one check, before the next read: isJsArray allocates.
            bool hasPrompt = false;
            {
                ev::Persistent pv(ev::getProperty(args[1], "points"));
                hasPrompt = isJsArray(pv.get());
            }
            if (!hasPrompt) {
                ev::Persistent bv(ev::getProperty(args[1], "boxes"));
                hasPrompt = isJsArray(bv.get());
            }
            if (hasPrompt) {
                try {
                    brotensor::DeviceScope scope(w->device);
                    w->sam->set_image(rgba.data(), inW, inH, 4);
                    w->hasImage = true;
                    w->imageW = inW;
                    w->imageH = inH;
                } catch (const std::exception& e) {
                    return ev::throwError(std::string("Sam setImage failed: ") + e.what());
                }
                std::span<const Value> promptArgs = args.subspan(1);
                return samSegmentDirect(w, promptArgs);
            }
        }
        return samSegmentEverythingDirect(w, args);
    }
    return samSegmentDirect(w, args);
}

Value buildSegmentation(const brovisionml::sam::Segmentation& seg) {
    const int W = seg.width, H = seg.height;
    const size_t plane = static_cast<size_t>(W) * static_cast<size_t>(H);

    ev::Persistent masks(hostArrayOf(static_cast<size_t>(seg.num), [&](size_t m) -> Value {
        std::vector<uint8_t> bin(plane);
        const float* lg = seg.logits.data() + m * plane;
        for (size_t i = 0; i < plane; ++i) bin[i] = lg[i] > 0.0f ? 1 : 0;
        ObjectBuilder mo;
        mo.set("iou", m < seg.iou.size() ? static_cast<double>(seg.iou[m]) : 0.0);
        mo.set("data", makeUint8Array(bin.data(), bin.size()));
        // The raw per-mask logits, which the bronze port returned under
        // `masks`; keeping them means neither caller has to change.
        mo.set("logits", makeFloat32Array(lg, plane));
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
