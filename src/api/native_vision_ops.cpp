#include "host_vision_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace brovisionml::api {

// ═══════════════════════════════════════════════════════════════════════════
// TypedArray & Device Helpers
// ═══════════════════════════════════════════════════════════════════════════

brotensor::Device parseDevice(Value opts) {
    if (!ev::isObject(opts)) return brotensor::Device::CPU;
    Value devVal = ev::getProperty(opts, "device");
    if (!ev::isString(devVal)) return brotensor::Device::CPU;
    std::string s = ev::toUtf8(devVal);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "cuda" || s == "gpu") {
        if (brotensor::is_available(brotensor::Device::CUDA)) {
            return brotensor::Device::CUDA;
        }
    } else if (s == "metal") {
        if (brotensor::is_available(brotensor::Device::Metal)) {
            return brotensor::Device::Metal;
        }
    }
    return brotensor::Device::CPU;
}

const char* deviceName(brotensor::Device dev) {
    switch (dev.type) {
        case brotensor::DeviceType::CUDA:  return "CUDA";
        case brotensor::DeviceType::Metal: return "Metal";
        case brotensor::DeviceType::CPU:   return "CPU";
    }
    return "CPU";
}

bool readFloat32Array(Value val, const float*& outData, size_t& outCount) {
    outData = nullptr;
    outCount = 0;
    ev::TypedArrayInfo info = ev::typedArrayInfo(val);
    if (!info || info.elementKind != ev::elements::Float32) return false;
    outData = reinterpret_cast<const float*>(info.data);
    outCount = info.elementCount;
    return true;
}

bool readUint8Array(Value val, const uint8_t*& outData, size_t& outCount) {
    outData = nullptr;
    outCount = 0;
    ev::TypedArrayInfo info = ev::typedArrayInfo(val);
    if (!info || (info.elementKind != ev::elements::Uint8 && info.elementKind != ev::elements::Uint8Clamped)) {
        return false;
    }
    outData = reinterpret_cast<const uint8_t*>(info.data);
    outCount = info.elementCount;
    return true;
}

bool readInt32Array(Value val, const int32_t*& outData, size_t& outCount) {
    outData = nullptr;
    outCount = 0;
    ev::TypedArrayInfo info = ev::typedArrayInfo(val);
    if (!info || info.elementKind != ev::elements::Int32) return false;
    outData = reinterpret_cast<const int32_t*>(info.data);
    outCount = info.elementCount;
    return true;
}

bool readImageInput(Value val, std::vector<uint8_t>& rgba, int& w, int& h, std::string& err) {
    if (ev::isString(val)) {
        std::string path = resolvePath(ev::toUtf8(val));
        broimage::Image img;
        if (!broimage::decode_file(path, img, &err)) {
            if (err.empty()) err = "Failed to decode image file: " + path;
            return false;
        }
        w = img.width;
        h = img.height;
        rgba = std::move(img.pixels);
        return true;
    }

    if (ev::isObject(val)) {
        Value wVal = ev::getProperty(val, "width");
        Value hVal = ev::getProperty(val, "height");
        if (!ev::isUndefined(wVal) && !ev::isUndefined(hVal)) {
            w = static_cast<int>(ev::toDouble(wVal));
            h = static_cast<int>(ev::toDouble(hVal));
        }
        Value dataVal = ev::getProperty(val, "data");
        const uint8_t* u8Data = nullptr;
        size_t u8Count = 0;
        if (readUint8Array(dataVal, u8Data, u8Count) && u8Data) {
            if (w <= 0 || h <= 0) {
                err = "Image { width, height } must be positive";
                return false;
            }
            const size_t need = static_cast<size_t>(w) * h * 4;
            if (u8Count < need) {
                err = "image.data too small for width*height*4 RGBA";
                return false;
            }
            rgba.assign(u8Data, u8Data + need);
            return true;
        }
    }

    err = "Invalid image input: expected file path string or { width, height, data: Uint8Array }";
    return false;
}

Value makeFloat32Array(const float* data, size_t count) {
    Value arr = ev::createTypedArray(ev::elements::Float32, static_cast<uint32_t>(count));
    if (data && count > 0) {
        ev::fillTypedArray(arr, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(data), count * sizeof(float)));
    }
    return arr;
}

Value makeUint8Array(const uint8_t* data, size_t count) {
    Value arr = ev::createTypedArray(ev::elements::Uint8, static_cast<uint32_t>(count));
    if (data && count > 0) {
        ev::fillTypedArray(arr, std::span<const uint8_t>(data, count));
    }
    return arr;
}

Value makeUint8ClampedArray(const uint8_t* data, size_t count) {
    Value arr = ev::createTypedArray(ev::elements::Uint8Clamped, static_cast<uint32_t>(count));
    if (data && count > 0) {
        ev::fillTypedArray(arr, std::span<const uint8_t>(data, count));
    }
    return arr;
}

Value makeInt32Array(const int32_t* data, size_t count) {
    Value arr = ev::createTypedArray(ev::elements::Int32, static_cast<uint32_t>(count));
    if (data && count > 0) {
        ev::fillTypedArray(arr, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(data), count * sizeof(int32_t)));
    }
    return arr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Vision Operations: Bounding Boxes, NMS, Rasterization, Colormaps
// ═══════════════════════════════════════════════════════════════════════════

struct BoundingBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int classId = 0;
    int index = 0;
};

static float computeIoU(const BoundingBox& a, const BoundingBox& b) {
    float interX1 = std::max(a.x1, b.x1);
    float interY1 = std::max(a.y1, b.y1);
    float interX2 = std::min(a.x2, b.x2);
    float interY2 = std::min(a.y2, b.y2);

    float interW = std::max(0.0f, interX2 - interX1);
    float interH = std::max(0.0f, interY2 - interY1);
    float interArea = interW * interH;

    float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float unionArea = areaA + areaB - interArea;

    return unionArea > 1e-6f ? (interArea / unionArea) : 0.0f;
}

static std::vector<BoundingBox> runNMS(std::vector<BoundingBox>& boxes, float iouThreshold, int maxDetections, bool perClass) {
    std::sort(boxes.begin(), boxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
        return a.score > b.score;
    });

    std::vector<BoundingBox> result;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);
        if (static_cast<int>(result.size()) >= maxDetections) break;

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (perClass && boxes[i].classId != boxes[j].classId) continue;

            if (computeIoU(boxes[i], boxes[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

// Turbo Colormap Approximation (Google Turbo)
static void turboColormap(float x, uint8_t& r, uint8_t& g, uint8_t& b) {
    x = std::clamp(x, 0.0f, 1.0f);
    const float r_coeffs[5] = {0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f, -152.94239396f};
    const float g_coeffs[5] = {0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f, 4.27729857f};
    const float b_coeffs[5] = {0.10667330f, 12.64194608f, -60.58204836f, 110.36275817f, -89.90310912f};

    auto evalPoly = [x](const float* c) {
        return c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])));
    };

    float rf = std::clamp(evalPoly(r_coeffs), 0.0f, 1.0f);
    float gf = std::clamp(evalPoly(g_coeffs), 0.0f, 1.0f);
    float bf = std::clamp(evalPoly(b_coeffs), 0.0f, 1.0f);

    r = static_cast<uint8_t>(rf * 255.0f + 0.5f);
    g = static_cast<uint8_t>(gf * 255.0f + 0.5f);
    b = static_cast<uint8_t>(bf * 255.0f + 0.5f);
}

// Viridis Colormap Approximation
static void viridisColormap(float x, uint8_t& r, uint8_t& g, uint8_t& b) {
    x = std::clamp(x, 0.0f, 1.0f);
    float rf = std::clamp(0.267f + x * (0.004f + x * (2.82f - x * 2.09f)), 0.0f, 1.0f);
    float gf = std::clamp(0.003f + x * (1.404f - x * (0.41f - x * 0.0f)), 0.0f, 1.0f);
    float bf = std::clamp(0.329f + x * (1.428f - x * (2.95f - x * 1.55f)), 0.0f, 1.0f);

    r = static_cast<uint8_t>(rf * 255.0f + 0.5f);
    g = static_cast<uint8_t>(gf * 255.0f + 0.5f);
    b = static_cast<uint8_t>(bf * 255.0f + 0.5f);
}

// Discrete ADE20K 150-class Palette generator
static void getPaletteColor(int classId, uint8_t& r, uint8_t& g, uint8_t& b) {
    if (classId <= 0) {
        r = 0; g = 0; b = 0;
        return;
    }
    // High-contrast distinct golden-ratio palette
    uint32_t hash = static_cast<uint32_t>(classId) * 2654435761u;
    r = static_cast<uint8_t>((hash >> 16) & 0xFF);
    g = static_cast<uint8_t>((hash >> 8) & 0xFF);
    b = static_cast<uint8_t>(hash & 0xFF);
    // Ensure sufficient brightness
    if (r < 40 && g < 40 && b < 40) {
        r = static_cast<uint8_t>(r + 80);
        g = static_cast<uint8_t>(g + 80);
        b = static_cast<uint8_t>(b + 80);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Native JS Bindings for Vision Ops
// ═══════════════════════════════════════════════════════════════════════════

static Value js_decodeBoxes(Value, std::span<const Value> args) {
    if (args.empty()) {
        return ev::throwTypeError("bro.vision.decodeBoxes: predictions array is required");
    }

    const float* preds = nullptr;
    size_t count = 0;
    if (!readFloat32Array(args[0], preds, count) || !preds) {
        return ev::throwTypeError("bro.vision.decodeBoxes: first argument must be a Float32Array");
    }

    Value opts = args.size() > 1 ? args[1] : ev::undefined();
    int numClasses = 80;
    int numBoxes = 0;
    float confThreshold = 0.25f;
    float iouThreshold = 0.45f;
    bool doNms = true;
    bool transposed = false; // YOLOv8 [4+C, N] vs [N, 4+C]

    if (ev::isObject(opts)) {
        Value ncVal = ev::getProperty(opts, "numClasses");
        if (!ev::isUndefined(ncVal)) numClasses = static_cast<int>(ev::toDouble(ncVal));

        Value ctVal = ev::getProperty(opts, "confThreshold");
        if (!ev::isUndefined(ctVal)) confThreshold = static_cast<float>(ev::toDouble(ctVal));

        Value itVal = ev::getProperty(opts, "iouThreshold");
        if (!ev::isUndefined(itVal)) iouThreshold = static_cast<float>(ev::toDouble(itVal));

        Value nmsVal = ev::getProperty(opts, "nms");
        if (!ev::isUndefined(nmsVal)) doNms = ev::toBool(nmsVal);

        Value trVal = ev::getProperty(opts, "transposed");
        if (!ev::isUndefined(trVal)) transposed = ev::toBool(trVal);
    }

    const int channels = 4 + numClasses;
    if (channels <= 4) {
        return ev::throwTypeError("decodeBoxes: numClasses must be > 0");
    }

    std::vector<BoundingBox> candidates;
    if (transposed) {
        numBoxes = static_cast<int>(count / channels);
        for (int b = 0; b < numBoxes; ++b) {
            float cx = preds[0 * numBoxes + b];
            float cy = preds[1 * numBoxes + b];
            float w  = preds[2 * numBoxes + b];
            float h  = preds[3 * numBoxes + b];

            float maxScore = -1.0f;
            int bestClass = -1;
            for (int c = 0; c < numClasses; ++c) {
                float s = preds[(4 + c) * numBoxes + b];
                if (s > maxScore) {
                    maxScore = s;
                    bestClass = c;
                }
            }

            if (maxScore >= confThreshold) {
                BoundingBox box;
                box.x1 = cx - w * 0.5f;
                box.y1 = cy - h * 0.5f;
                box.x2 = cx + w * 0.5f;
                box.y2 = cy + h * 0.5f;
                box.score = maxScore;
                box.classId = bestClass;
                candidates.push_back(box);
            }
        }
    } else {
        numBoxes = static_cast<int>(count / channels);
        for (int b = 0; b < numBoxes; ++b) {
            const float* row = preds + static_cast<size_t>(b) * channels;
            float cx = row[0];
            float cy = row[1];
            float w  = row[2];
            float h  = row[3];

            float maxScore = -1.0f;
            int bestClass = -1;
            for (int c = 0; c < numClasses; ++c) {
                float s = row[4 + c];
                if (s > maxScore) {
                    maxScore = s;
                    bestClass = c;
                }
            }

            if (maxScore >= confThreshold) {
                BoundingBox box;
                box.x1 = cx - w * 0.5f;
                box.y1 = cy - h * 0.5f;
                box.x2 = cx + w * 0.5f;
                box.y2 = cy + h * 0.5f;
                box.score = maxScore;
                box.classId = bestClass;
                candidates.push_back(box);
            }
        }
    }

    if (doNms && !candidates.empty()) {
        candidates = runNMS(candidates, iouThreshold, 100, false);
    }

    return hostArrayOf(candidates.size(), [&](size_t i) {
        ObjectBuilder boxObj;
        boxObj.set("x1", candidates[i].x1);
        boxObj.set("y1", candidates[i].y1);
        boxObj.set("x2", candidates[i].x2);
        boxObj.set("y2", candidates[i].y2);
        boxObj.set("score", candidates[i].score);
        boxObj.set("classId", candidates[i].classId);
        return boxObj.build();
    });
}

static void extractBoxesFromFlatFloats(const float* data, size_t totalFloats, size_t stride,
                                       float scoreThreshold, std::vector<BoundingBox>& out) {
    if (!data || totalFloats < 4) return;
    if (stride < 4) {
        if (totalFloats % 6 == 0) stride = 6;
        else if (totalFloats % 5 == 0) stride = 5;
        else stride = 4;
    }
    const size_t numBoxes = totalFloats / stride;
    out.reserve(out.size() + numBoxes);
    for (size_t b = 0; b < numBoxes; ++b) {
        const float* row = data + b * stride;
        BoundingBox box;
        box.x1 = row[0];
        box.y1 = row[1];
        box.x2 = row[2];
        box.y2 = row[3];
        box.score = (stride >= 5) ? row[4] : 1.0f;
        box.classId = (stride >= 6) ? static_cast<int>(row[5]) : 0;
        box.index = static_cast<int>(b);
        if (box.score >= scoreThreshold) {
            out.push_back(box);
        }
    }
}

static Value js_nms(Value, std::span<const Value> args) {
    if (args.empty()) {
        return ev::throwTypeError("bro.vision.nms: boxes array is required");
    }

    float iouThreshold = 0.45f;
    int maxDetections = 100;
    bool perClass = false;
    float scoreThreshold = 0.0f;
    int optStride = 0;
    bool asTypedArray = false;
    bool returnIndices = false;

    if (args.size() > 1 && ev::isObject(args[1])) {
        Value it = ev::getProperty(args[1], "iouThreshold");
        if (!ev::isUndefined(it)) iouThreshold = static_cast<float>(ev::toDouble(it));

        Value md = ev::getProperty(args[1], "maxDetections");
        if (!ev::isUndefined(md)) maxDetections = static_cast<int>(ev::toDouble(md));

        Value pc = ev::getProperty(args[1], "perClass");
        if (!ev::isUndefined(pc)) perClass = ev::toBool(pc);

        Value st = ev::getProperty(args[1], "scoreThreshold");
        if (!ev::isUndefined(st)) scoreThreshold = static_cast<float>(ev::toDouble(st));

        Value strVal = ev::getProperty(args[1], "stride");
        if (!ev::isUndefined(strVal)) optStride = static_cast<int>(ev::toDouble(strVal));

        Value ata = ev::getProperty(args[1], "asTypedArray");
        if (!ev::isUndefined(ata)) asTypedArray = ev::toBool(ata);

        Value idxVal = ev::getProperty(args[1], "returnIndices");
        if (!ev::isUndefined(idxVal)) returnIndices = ev::toBool(idxVal);
    }

    std::vector<BoundingBox> boxes;
    Value input = args[0];

    // Case 1: Direct TypedArray buffer view (Float32Array)
    if (auto tinfo = ev::typedArrayInfo(input)) {
        if (tinfo.elementKind == ev::elements::Float32 && tinfo.data) {
            extractBoxesFromFlatFloats(reinterpret_cast<const float*>(tinfo.data),
                                       tinfo.elementCount, static_cast<size_t>(optStride),
                                       scoreThreshold, boxes);
        }
    }
    // Case 2: ArrayBuffer
    else if (auto ab = ev::arrayBufferInfo(input)) {
        if (ab.data) {
            extractBoxesFromFlatFloats(reinterpret_cast<const float*>(ab.data),
                                       ab.byteLength / sizeof(float), static_cast<size_t>(optStride),
                                       scoreThreshold, boxes);
        }
    }
    // Case 3: Object containing { data/boxes/buffer: TypedArray/ArrayBuffer }
    else if (ev::isObject(input) && !isJsArray(input)) {
        Value sub = ev::getProperty(input, "data");
        if (ev::isUndefined(sub) || ev::isNull(sub)) sub = ev::getProperty(input, "boxes");
        if (ev::isUndefined(sub) || ev::isNull(sub)) sub = ev::getProperty(input, "buffer");

        if (auto subInfo = ev::typedArrayInfo(sub)) {
            if (subInfo.elementKind == ev::elements::Float32 && subInfo.data) {
                extractBoxesFromFlatFloats(reinterpret_cast<const float*>(subInfo.data),
                                           subInfo.elementCount, static_cast<size_t>(optStride),
                                           scoreThreshold, boxes);
            }
        } else if (auto subAb = ev::arrayBufferInfo(sub)) {
            if (subAb.data) {
                extractBoxesFromFlatFloats(reinterpret_cast<const float*>(subAb.data),
                                           subAb.byteLength / sizeof(float), static_cast<size_t>(optStride),
                                           scoreThreshold, boxes);
            }
        } else {
            return ev::throwTypeError("bro.vision.nms: first argument must be an Array or TypedArray of boxes");
        }
    }
    // Case 4: JS Array (elements may be typed array views, JS arrays, or box objects)
    else if (isJsArray(input)) {
        uint32_t len = getJsArrayLength(input);
        boxes.reserve(len);

        for (uint32_t i = 0; i < len; ++i) {
            Value elem = ev::getElement(input, i);
            if (!ev::isObject(elem)) continue;

            // 4a. Element is a TypedArray view (e.g. new Float32Array([x1, y1, x2, y2, score?, classId?]))
            if (auto eInfo = ev::typedArrayInfo(elem)) {
                if (eInfo.elementKind == ev::elements::Float32 && eInfo.data) {
                    const float* ep = reinterpret_cast<const float*>(eInfo.data);
                    const uint32_t ec = eInfo.elementCount;
                    BoundingBox b;
                    b.x1 = ec > 0 ? ep[0] : 0.0f;
                    b.y1 = ec > 1 ? ep[1] : 0.0f;
                    b.x2 = ec > 2 ? ep[2] : 0.0f;
                    b.y2 = ec > 3 ? ep[3] : 0.0f;
                    b.score = ec > 4 ? ep[4] : 1.0f;
                    b.classId = ec > 5 ? static_cast<int>(ep[5]) : 0;
                    b.index = static_cast<int>(i);
                    if (b.score >= scoreThreshold) {
                        boxes.push_back(b);
                    }
                    continue;
                }
            }

            // 4b. Element is a JS Array [x1, y1, x2, y2, score?, classId?]
            if (isJsArray(elem)) {
                const uint32_t elen = getJsArrayLength(elem);
                BoundingBox b;
                b.x1 = elen > 0 ? static_cast<float>(ev::toDouble(ev::getElement(elem, 0))) : 0.0f;
                b.y1 = elen > 1 ? static_cast<float>(ev::toDouble(ev::getElement(elem, 1))) : 0.0f;
                b.x2 = elen > 2 ? static_cast<float>(ev::toDouble(ev::getElement(elem, 2))) : 0.0f;
                b.y2 = elen > 3 ? static_cast<float>(ev::toDouble(ev::getElement(elem, 3))) : 0.0f;
                b.score = elen > 4 ? static_cast<float>(ev::toDouble(ev::getElement(elem, 4))) : 1.0f;
                b.classId = elen > 5 ? static_cast<int>(ev::toDouble(ev::getElement(elem, 5))) : 0;
                b.index = static_cast<int>(i);
                if (b.score >= scoreThreshold) {
                    boxes.push_back(b);
                }
                continue;
            }

            // 4c. Element is a JS Object: { x1, y1, x2, y2, score, classId }
            BoundingBox b;
            Value x1v = ev::getProperty(elem, "x1");
            Value y1v = ev::getProperty(elem, "y1");
            Value x2v = ev::getProperty(elem, "x2");
            Value y2v = ev::getProperty(elem, "y2");
            Value sv  = ev::getProperty(elem, "score");
            Value cv  = ev::getProperty(elem, "classId");

            b.x1 = ev::isUndefined(x1v) ? 0.0f : static_cast<float>(ev::toDouble(x1v));
            b.y1 = ev::isUndefined(y1v) ? 0.0f : static_cast<float>(ev::toDouble(y1v));
            b.x2 = ev::isUndefined(x2v) ? 0.0f : static_cast<float>(ev::toDouble(x2v));
            b.y2 = ev::isUndefined(y2v) ? 0.0f : static_cast<float>(ev::toDouble(y2v));
            b.score = ev::isUndefined(sv) ? 1.0f : static_cast<float>(ev::toDouble(sv));
            b.classId = ev::isUndefined(cv) ? 0 : static_cast<int>(ev::toDouble(cv));
            b.index = static_cast<int>(i);

            if (b.score >= scoreThreshold) {
                boxes.push_back(b);
            }
        }
    } else {
        return ev::throwTypeError("bro.vision.nms: first argument must be an Array or TypedArray of boxes");
    }

    std::vector<BoundingBox> filtered = runNMS(boxes, iouThreshold, maxDetections, perClass);

    if (returnIndices) {
        if (asTypedArray) {
            std::vector<int32_t> idxs(filtered.size());
            for (size_t i = 0; i < filtered.size(); ++i) {
                idxs[i] = filtered[i].index;
            }
            return makeInt32Array(idxs.data(), idxs.size());
        }
        return hostArrayOf(filtered.size(), [&](size_t i) {
            return ev::fromDouble(filtered[i].index);
        });
    }

    if (asTypedArray) {
        std::vector<float> flat(filtered.size() * 6);
        for (size_t i = 0; i < filtered.size(); ++i) {
            flat[i * 6 + 0] = filtered[i].x1;
            flat[i * 6 + 1] = filtered[i].y1;
            flat[i * 6 + 2] = filtered[i].x2;
            flat[i * 6 + 3] = filtered[i].y2;
            flat[i * 6 + 4] = filtered[i].score;
            flat[i * 6 + 5] = static_cast<float>(filtered[i].classId);
        }
        return makeFloat32Array(flat.data(), flat.size());
    }

    return hostArrayOf(filtered.size(), [&](size_t i) {
        ObjectBuilder boxObj;
        boxObj.set("x1", filtered[i].x1);
        boxObj.set("y1", filtered[i].y1);
        boxObj.set("x2", filtered[i].x2);
        boxObj.set("y2", filtered[i].y2);
        boxObj.set("score", filtered[i].score);
        boxObj.set("classId", filtered[i].classId);
        boxObj.set("index", filtered[i].index);
        return boxObj.build();
    });
}

static Value js_rasterizeMask(Value, std::span<const Value> args) {
    if (args.empty()) {
        return ev::throwTypeError("bro.vision.rasterizeMask: mask data is required");
    }

    Value opts = args.size() > 1 ? args[1] : ev::undefined();
    int srcW = 0, srcH = 0;
    int targetW = 0, targetH = 0;
    float threshold = 0.0f;

    if (ev::isObject(opts)) {
        Value sw = ev::getProperty(opts, "width");
        if (!ev::isUndefined(sw)) srcW = static_cast<int>(ev::toDouble(sw));
        Value sh = ev::getProperty(opts, "height");
        if (!ev::isUndefined(sh)) srcH = static_cast<int>(ev::toDouble(sh));
        Value tw = ev::getProperty(opts, "targetWidth");
        if (!ev::isUndefined(tw)) targetW = static_cast<int>(ev::toDouble(tw));
        Value th = ev::getProperty(opts, "targetHeight");
        if (!ev::isUndefined(th)) targetH = static_cast<int>(ev::toDouble(th));
        Value tv = ev::getProperty(opts, "threshold");
        if (!ev::isUndefined(tv)) threshold = static_cast<float>(ev::toDouble(tv));
    }

    if (targetW <= 0) targetW = srcW > 0 ? srcW : 512;
    if (targetH <= 0) targetH = srcH > 0 ? srcH : 512;
    if (srcW <= 0) srcW = targetW;
    if (srcH <= 0) srcH = targetH;

    const float* f32Data = nullptr;
    const uint8_t* u8Data = nullptr;
    size_t count = 0;

    std::vector<uint8_t> outMask(static_cast<size_t>(targetW) * targetH, 0);

    if (readFloat32Array(args[0], f32Data, count) && f32Data) {
        // Bilinear sample or nearest neighbor threshold
        for (int y = 0; y < targetH; ++y) {
            float sy = (static_cast<float>(y) + 0.5f) * srcH / targetH - 0.5f;
            int iy = std::clamp(static_cast<int>(std::floor(sy)), 0, srcH - 1);
            for (int x = 0; x < targetW; ++x) {
                float sx = (static_cast<float>(x) + 0.5f) * srcW / targetW - 0.5f;
                int ix = std::clamp(static_cast<int>(std::floor(sx)), 0, srcW - 1);
                float val = f32Data[static_cast<size_t>(iy) * srcW + ix];
                outMask[static_cast<size_t>(y) * targetW + x] = (val > threshold) ? 255 : 0;
            }
        }
    } else if (readUint8Array(args[0], u8Data, count) && u8Data) {
        for (int y = 0; y < targetH; ++y) {
            int iy = std::clamp(y * srcH / targetH, 0, srcH - 1);
            for (int x = 0; x < targetW; ++x) {
                int ix = std::clamp(x * srcW / targetW, 0, srcW - 1);
                uint8_t val = u8Data[static_cast<size_t>(iy) * srcW + ix];
                outMask[static_cast<size_t>(y) * targetW + x] = val ? 255 : 0;
            }
        }
    } else if (isJsArray(args[0])) {
        // Polygon point list [{x, y}, ...]
        uint32_t polyLen = getJsArrayLength(args[0]);
        std::vector<std::pair<float, float>> poly;
        poly.reserve(polyLen);
        for (uint32_t i = 0; i < polyLen; ++i) {
            Value p = ev::getElement(args[0], i);
            if (ev::isObject(p)) {
                float px = static_cast<float>(ev::toDouble(ev::getProperty(p, "x")));
                float py = static_cast<float>(ev::toDouble(ev::getProperty(p, "y")));
                poly.emplace_back(px, py);
            }
        }

        if (poly.size() >= 3) {
            // Point-in-polygon ray-casting test
            for (int y = 0; y < targetH; ++y) {
                float testY = static_cast<float>(y) + 0.5f;
                for (int x = 0; x < targetW; ++x) {
                    float testX = static_cast<float>(x) + 0.5f;
                    bool inside = false;
                    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
                        if (((poly[i].second > testY) != (poly[j].second > testY)) &&
                            (testX < (poly[j].first - poly[i].first) * (testY - poly[i].second) /
                                     (poly[j].second - poly[i].second) + poly[i].first)) {
                            inside = !inside;
                        }
                    }
                    if (inside) {
                        outMask[static_cast<size_t>(y) * targetW + x] = 255;
                    }
                }
            }
        }
    }

    ObjectBuilder res;
    res.set("width", static_cast<double>(targetW));
    res.set("height", static_cast<double>(targetH));
    ev::Persistent d(makeUint8Array(outMask.data(), outMask.size()));
    res.set("data", d.get());
    return res.build();
}

static Value js_colorMap(Value, std::span<const Value> args) {
    if (args.empty()) {
        return ev::throwTypeError("bro.vision.colorMap: input data is required");
    }

    Value opts = args.size() > 1 ? args[1] : ev::undefined();
    int width = 512, height = 512;
    std::string mapType = "turbo";
    float minVal = 0.0f, maxVal = 1.0f;
    bool autoRange = true;

    if (ev::isObject(opts)) {
        Value wv = ev::getProperty(opts, "width");
        if (!ev::isUndefined(wv)) width = static_cast<int>(ev::toDouble(wv));
        Value hv = ev::getProperty(opts, "height");
        if (!ev::isUndefined(hv)) height = static_cast<int>(ev::toDouble(hv));
        Value mv = ev::getProperty(opts, "map");
        if (ev::isString(mv)) mapType = ev::toUtf8(mv);
        Value minv = ev::getProperty(opts, "min");
        if (!ev::isUndefined(minv)) { minVal = static_cast<float>(ev::toDouble(minv)); autoRange = false; }
        Value maxv = ev::getProperty(opts, "max");
        if (!ev::isUndefined(maxv)) { maxVal = static_cast<float>(ev::toDouble(maxv)); autoRange = false; }
    }

    const size_t totalPixels = static_cast<size_t>(width) * height;
    std::vector<uint8_t> rgba(totalPixels * 4, 255);

    const float* f32Data = nullptr;
    const uint8_t* u8Data = nullptr;
    const int32_t* i32Data = nullptr;
    size_t count = 0;

    if (readFloat32Array(args[0], f32Data, count) && f32Data) {
        if (autoRange && count > 0) {
            minVal = std::numeric_limits<float>::infinity();
            maxVal = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < std::min(count, totalPixels); ++i) {
                minVal = std::min(minVal, f32Data[i]);
                maxVal = std::max(maxVal, f32Data[i]);
            }
        }
        float range = (maxVal > minVal) ? (maxVal - minVal) : 1.0f;

        for (size_t i = 0; i < std::min(count, totalPixels); ++i) {
            float norm = (f32Data[i] - minVal) / range;
            uint8_t r = 0, g = 0, b = 0;
            if (mapType == "viridis") {
                viridisColormap(norm, r, g, b);
            } else if (mapType == "grayscale") {
                uint8_t v = static_cast<uint8_t>(std::clamp(norm * 255.0f, 0.0f, 255.0f));
                r = g = b = v;
            } else {
                turboColormap(norm, r, g, b);
            }
            rgba[i * 4 + 0] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = 255;
        }
    } else if (readInt32Array(args[0], i32Data, count) && i32Data) {
        for (size_t i = 0; i < std::min(count, totalPixels); ++i) {
            uint8_t r = 0, g = 0, b = 0;
            getPaletteColor(i32Data[i], r, g, b);
            rgba[i * 4 + 0] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = 255;
        }
    } else if (readUint8Array(args[0], u8Data, count) && u8Data) {
        for (size_t i = 0; i < std::min(count, totalPixels); ++i) {
            uint8_t r = 0, g = 0, b = 0;
            if (mapType == "palette") {
                getPaletteColor(u8Data[i], r, g, b);
            } else {
                float norm = static_cast<float>(u8Data[i]) / 255.0f;
                turboColormap(norm, r, g, b);
            }
            rgba[i * 4 + 0] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = 255;
        }
    }

    ObjectBuilder res;
    res.set("width", static_cast<double>(width));
    res.set("height", static_cast<double>(height));
    ev::Persistent d(makeUint8ClampedArray(rgba.data(), rgba.size()));
    res.set("data", d.get());
    return res.build();
}

void mountVisionOps(ObjectBuilder& visionObj) {
    visionObj.def("decodeBoxes", 2, js_decodeBoxes);
    visionObj.def("nms", 2, js_nms);
    visionObj.def("rasterizeMask", 2, js_rasterizeMask);
    visionObj.def("colorMap", 2, js_colorMap);
    visionObj.def("colorizeDepth", 2, js_colorMap);
    visionObj.def("colorizeSegmentation", 2, js_colorMap);

    ObjectBuilder opsObj;
    opsObj.def("decodeBoxes", 2, js_decodeBoxes);
    opsObj.def("nms", 2, js_nms);
    opsObj.def("rasterizeMask", 2, js_rasterizeMask);
    opsObj.def("colorMap", 2, js_colorMap);
    opsObj.def("colorizeDepth", 2, js_colorMap);
    opsObj.def("colorizeSegmentation", 2, js_colorMap);
    visionObj.set("ops", opsObj.build());
}

} // namespace brovisionml::api
