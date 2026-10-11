// OWLv2 open-vocabulary detector — bro.vision.loadOwlv2 and the Owlv2 class
// (embedQueries / detect / dispose).
//
// The text queries' embeddings are cached on the instance by query string, so
// a caller that detects the same few queries in many photos pays for the text
// tower once; detect() itself runs the vision tower and the heads.

#include "host_vision_internal.h"

#include <brovisionml/owlv2.h>

#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace brovisionml::api {

namespace {

inline constexpr uint32_t kHostOwlv2Tag = 0x564F574Cu;   // 'VOWL'

struct Owlv2Wrapper {
    uint32_t tag = kHostOwlv2Tag;
    std::unique_ptr<brovisionml::owlv2::Detector> det;
    brotensor::Device device = brotensor::Device::CPU;
    std::map<std::string, std::vector<float>> queryCache;
    std::string path;
};

HostClass g_owlv2Class;

Owlv2Wrapper* owlSelf(Value thisVal) {
    void* p = g_owlv2Class.unwrap(thisVal);
    if (!p) return nullptr;
    auto* w = static_cast<Owlv2Wrapper*>(p);
    return w->tag == kHostOwlv2Tag ? w : nullptr;
}

// A string or an array of strings; false (and `err`) on anything else.
bool readQueries(Value v, std::vector<std::string>& out, std::string& err) {
    ev::Persistent arr(v);
    if (ev::isString(arr.get())) {
        out.push_back(ev::toUtf8(arr.get()));
        return true;
    }
    if (!isJsArray(arr.get())) {
        err = "queries must be a string or an array of strings";
        return false;
    }
    const uint32_t n = getJsArrayLength(arr.get());
    for (uint32_t i = 0; i < n; ++i) {
        Value e = ev::getElement(arr.get(), i);
        if (!ev::isString(e)) {
            err = "queries[" + std::to_string(i) + "] is not a string";
            return false;
        }
        out.push_back(ev::toUtf8(e));
    }
    if (out.empty()) err = "queries is empty";
    return !out.empty();
}

// The query embeddings, from the cache or the text tower; (n, 512) row-major.
std::vector<float> embedQueries(Owlv2Wrapper& w, const std::vector<std::string>& queries) {
    std::vector<std::string> missing;
    for (const std::string& q : queries)
        if (!w.queryCache.count(q)) missing.push_back(q);
    if (!missing.empty()) {
        const std::vector<float> e = w.det->embed_text(missing);
        const std::size_t E = static_cast<std::size_t>(w.det->config().embed_dim);
        for (std::size_t i = 0; i < missing.size(); ++i)
            w.queryCache[missing[i]].assign(e.begin() + i * E, e.begin() + (i + 1) * E);
    }
    std::vector<float> out;
    for (const std::string& q : queries) {
        const std::vector<float>& e = w.queryCache[q];
        out.insert(out.end(), e.begin(), e.end());
    }
    return out;
}

// A file path, or { width, height, data | pixels, channels? } of 8-bit RGB or
// RGBA rows (ImageData, a bro.image decode result).
bool readRgbImage(Value val, std::vector<uint8_t>& px, int& w, int& h, int& ch, std::string& err) {
    ev::Persistent img(val);
    if (ev::isString(img.get()) || !ev::isObject(img.get())) {
        ch = 4;
        return readImageInput(img.get(), px, w, h, err);
    }
    const double wd = optDouble(img.get(), "width", 0.0);
    const double hd = optDouble(img.get(), "height", 0.0);
    int channels = optInt(img.get(), "channels", 0);
    if (!(wd >= 1.0 && hd >= 1.0 && wd <= 65536.0 && hd <= 65536.0)) {
        err = "image { width, height } must be positive integers (at most 65536)";
        return false;
    }
    w = static_cast<int>(wd);
    h = static_cast<int>(hd);
    Value data = ev::getProperty(img.get(), "data");
    if (ev::isUndefined(data) || ev::isNull(data)) data = ev::getProperty(img.get(), "pixels");
    // The data pointer is read right after the last allocating call.
    const uint8_t* bytes = nullptr;
    size_t count = 0;
    if (!readUint8Array(data, bytes, count) || !bytes) {
        err = "image.data (or .pixels) must be a Uint8Array / Uint8ClampedArray";
        return false;
    }
    const size_t n = static_cast<size_t>(w) * h;
    if (channels == 0) channels = count >= n * 4 ? 4 : 3;
    if ((channels != 3 && channels != 4) || count < n * channels) {
        err = "image data too small for width * height * channels (3 or 4)";
        return false;
    }
    ch = channels;
    px.assign(bytes, bytes + n * channels);
    return true;
}

Value owlEmbedQueries(Value thisVal, std::span<const Value> args) {
    Owlv2Wrapper* w = owlSelf(thisVal);
    if (!w) return ev::throwTypeError("Owlv2.prototype.embedQueries: not an Owlv2 instance");
    if (!w->det) return ev::throwError("Owlv2.embedQueries: model is disposed");
    std::vector<std::string> queries;
    std::string err;
    if (args.empty() || !readQueries(args[0], queries, err))
        return ev::throwTypeError("Owlv2.embedQueries: " + (err.empty() ? std::string("queries required") : err));
    try {
        brotensor::DeviceScope scope(w->device);
        const std::vector<float> e = embedQueries(*w, queries);
        return makeFloat32Array(e.data(), e.size());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Owlv2.embedQueries failed: ") + e.what());
    }
}

// detect(image, { queries, threshold = 0.1, nms = 0.3, maxDetections = 100 })
//   -> [{ box: [x, y, w, h], score, label, query, objectness }], best first.
Value owlDetect(Value thisVal, std::span<const Value> args) {
    Owlv2Wrapper* w = owlSelf(thisVal);
    if (!w) return ev::throwTypeError("Owlv2.prototype.detect: not an Owlv2 instance");
    if (args.size() < 2 || !ev::isObject(args[1]))
        return ev::throwTypeError("Owlv2.detect(image, { queries }): options with queries required");
    if (!w->det) return ev::throwError("Owlv2.detect: model is disposed");

    std::vector<std::string> queries;
    std::string err;
    {
        ev::Persistent q(ev::getProperty(args[1], "queries"));
        if (!readQueries(q.get(), queries, err)) return ev::throwTypeError("Owlv2.detect: " + err);
    }
    brovisionml::owlv2::DetectOptions o;
    o.threshold = optFloat(args[1], "threshold", o.threshold);
    o.nms_iou = optFloat(args[1], "nms", o.nms_iou);
    o.max_detections = optInt(args[1], "maxDetections", o.max_detections);
    if (o.max_detections < 1) return ev::throwRangeError("Owlv2.detect: maxDetections must be >= 1");

    std::vector<uint8_t> px;
    int iw = 0, ih = 0, ch = 4;
    if (!readRgbImage(args[0], px, iw, ih, ch, err)) return ev::throwTypeError("Owlv2.detect: " + err);

    std::vector<brovisionml::owlv2::Detection> dets;
    try {
        brotensor::DeviceScope scope(w->device);
        const std::vector<float> q = embedQueries(*w, queries);
        dets = w->det->detect(px.data(), iw, ih, ch, q.data(), static_cast<int>(queries.size()), o);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Owlv2.detect failed: ") + e.what());
    }
    return hostArrayOf(dets.size(), [&](size_t i) {
        const auto& d = dets[i];
        ObjectBuilder r;
        {
            const double b[4] = {d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0};
            ev::Persistent box(hostArrayOf(4, [&](size_t k) { return ev::fromDouble(b[k]); }));
            r.set("box", box.get());
        }
        r.set("score", d.score);
        r.set("label", static_cast<int32_t>(d.label));
        r.set("query", queries[static_cast<size_t>(d.label)]);
        r.set("objectness", d.objectness);
        return r.build();
    });
}

void decorateOwlv2Proto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        Owlv2Wrapper* w = owlSelf(thisVal);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    proto.accessor("imageSize", [](Value thisVal, std::span<const Value>) -> Value {
        Owlv2Wrapper* w = owlSelf(thisVal);
        return ev::fromDouble(w && w->det ? w->det->config().image_size : 960);
    });
    proto.def("embedQueries", 1, owlEmbedQueries);
    proto.def("detect", 2, owlDetect);
    // dispose() — the weights are GPU-resident and invisible to the JS heap
    // accounting; a caller done with the model frees them now.
    proto.def("dispose", 0, [](Value thisVal, std::span<const Value>) -> Value {
        Owlv2Wrapper* w = owlSelf(thisVal);
        if (!w) return ev::throwTypeError("Owlv2.prototype.dispose: not an Owlv2 instance");
        w->det.reset();
        w->queryCache.clear();
        return ev::undefined();
    });
}

}  // namespace

void mountOwlv2(ObjectBuilder& vision) {
    static thread_local bool installed = false;   // per realm thread (host_class.h)
    if (!installed) {
        installed = true;
        g_owlv2Class.install("Owlv2", 0, nullptr, decorateOwlv2Proto);
    }

    // loadOwlv2(dir, { device }) — dir holds the HF release: model.safetensors,
    // vocab.json, merges.txt.
    vision.def("loadOwlv2", 2, [](Value, std::span<const Value> args) -> Value {
        if (args.empty() || !ev::isString(args[0]))
            return ev::throwTypeError("bro.vision.loadOwlv2: path is required (a directory)");
        const std::string path = resolvePath(ev::toUtf8(args[0]));
        std::error_code ec;
        if (!std::filesystem::is_directory(path, ec))
            return ev::throwError("loadOwlv2 failed: model dir not found: " + path);
        brotensor::Device dev = brotensor::Device::CPU;
        Value thrown;
        if (!resolveDevice("bro.vision.loadOwlv2", args.size() > 1 ? args[1] : ev::undefined(), dev, thrown))
            return thrown;
        auto w = std::make_unique<Owlv2Wrapper>();
        w->path = path;
        w->device = dev;
        try {
            w->det = std::make_unique<brovisionml::owlv2::Detector>();
            w->det->load(path);
            w->det->to(dev);
        } catch (const std::exception& e) {
            return ev::throwError(std::string("loadOwlv2 failed: ") + e.what());
        }
        return g_owlv2Class.createInstance(std::move(w));
    });
    vision.set("Owlv2", g_owlv2Class.constructor());
}

}  // namespace brovisionml::api
