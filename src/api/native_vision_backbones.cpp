// DINOv2 / DINOv3 ViT backbones — encode() and dispose().
//
// Ported from the QuickJS-era src/js/vision_bindings.cpp (js_dinov2_encode,
// js_dinov3_encode, js_dinov3_dispose, dinoPreprocess, downloadF32,
// js_loadDinov2, js_loadDinov3). The bronze port had both classes returning a
// fixed array of 512 zeros from an estimate() that never touched the model,
// and neither loader constructed a Backbone.

#include "host_vision_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace brovisionml::api {

namespace {

Dinov2Wrapper* v2Self(Value thisVal) {
    void* p = g_dinov2Class.unwrap(thisVal);
    if (!p) return nullptr;
    auto* w = static_cast<Dinov2Wrapper*>(p);
    return w->tag == kHostDinov2Tag ? w : nullptr;
}

Dinov3Wrapper* v3Self(Value thisVal) {
    void* p = g_dinov3Class.unwrap(thisVal);
    if (!p) return nullptr;
    auto* w = static_cast<Dinov3Wrapper*>(p);
    return w->tag == kHostDinov3Tag ? w : nullptr;
}

std::vector<float> downloadF32(const brotensor::Tensor& t) {
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (size_t i = 0; i < bits.size(); ++i) {
            out[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

// Resize RGBA8 to size x size (bilinear stretch), ImageNet-normalize, write
// planar NCHW FP32. Aspect ratio is not preserved — the source is stretched to
// the square the backbone consumes.
void dinoPreprocess(const std::vector<uint8_t>& rgba, int w, int h, int size,
                    std::vector<float>& nchw) {
    static const float kMean[3] = {0.485f, 0.456f, 0.406f};
    static const float kStd[3] = {0.229f, 0.224f, 0.225f};
    const size_t plane = static_cast<size_t>(size) * size;
    nchw.assign(plane * 3, 0.0f);
    auto px = [&](int x, int y, int c) -> float {
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        return static_cast<float>(rgba[(static_cast<size_t>(y) * w + x) * 4 + c]);
    };
    const float sx = static_cast<float>(w) / size;
    const float sy = static_cast<float>(h) / size;
    for (int ty = 0; ty < size; ++ty) {
        for (int tx = 0; tx < size; ++tx) {
            const float fx = (tx + 0.5f) * sx - 0.5f;
            const float fy = (ty + 0.5f) * sy - 0.5f;
            const int ix = static_cast<int>(std::floor(fx));
            const int iy = static_cast<int>(std::floor(fy));
            const float ax = fx - ix, ay = fy - iy;
            for (int c = 0; c < 3; ++c) {
                const float c00 = px(ix, iy, c), c10 = px(ix + 1, iy, c);
                const float c01 = px(ix, iy + 1, c), c11 = px(ix + 1, iy + 1, c);
                const float v = (c00 * (1 - ax) + c10 * ax) * (1 - ay) +
                                (c01 * (1 - ax) + c11 * ax) * ay;
                nchw[static_cast<size_t>(c) * plane +
                     static_cast<size_t>(ty) * size + tx] =
                    (v / 255.0f - kMean[c]) / kStd[c];
            }
        }
    }
}

// encode(image, opts?) -> { features, stages, tokens, dim, patchH, patchW,
//                           numPrefixTokens }
//   features is one Float32Array per DPT out-stage, each (1 + patchH*patchW)
//   rows of `dim`; row 0 is the cls token.
//   opts.size — square encode resolution, a positive multiple of patchSize.
Value v2Encode(Value thisVal, std::span<const Value> args) {
    auto* w = v2Self(thisVal);
    if (!w) return ev::throwTypeError("Dinov2Backbone.prototype.encode: not a backbone");
    if (!w->backbone) return ev::throwError("encode: disposed");
    if (args.empty()) return ev::throwTypeError("encode(image, opts?): image required");

    int size = w->size;
    if (args.size() > 1 && ev::isObject(args[1])) {
        Value sv = ev::getProperty(args[1], "size");
        if (ev::isNumber(sv)) size = static_cast<int>(ev::toDouble(sv));
    }
    const int ps = w->backbone->config().patch_size;
    if (size <= 0 || size % ps != 0) {
        return ev::throwTypeError("encode: opts.size must be a positive multiple of patchSize");
    }

    std::vector<uint8_t> rgba;
    int inW = 0, inH = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, inW, inH, err)) {
        return ev::throwTypeError(std::string("encode: ") + err);
    }

    std::vector<std::vector<float>> maps;
    std::vector<int> stages;
    int tokens = 0, dim = 0, patchH = 0, patchW = 0;
    try {
        std::vector<float> nchw;
        dinoPreprocess(rgba, inW, inH, size, nchw);
        brotensor::DeviceScope scope(w->device);
        brotensor::Tensor pxT = brotensor::Tensor::from_host_on(
            w->device, nchw.data(), 1, 3 * size * size);
        auto out = w->backbone->encode(pxT, size, size);
        for (const auto& fm : out.feature_maps) {
            maps.push_back(downloadF32(fm));
            tokens = fm.rows;
            dim = fm.cols;
        }
        patchH = out.patch_h;
        patchW = out.patch_w;
        stages = w->backbone->config().out_stages;
    } catch (const std::exception& e) {
        return ev::throwError(std::string("encode failed: ") + e.what());
    }

    ev::Persistent feats(hostArrayOf(maps.size(), [&maps](size_t i) {
        return makeFloat32Array(maps[i].data(), maps[i].size());
    }));
    ev::Persistent stagesArr(hostArrayOf(stages.size(), [&stages](size_t i) {
        return ev::fromDouble(static_cast<double>(stages[i]));
    }));

    ObjectBuilder res;
    res.set("features", feats.get());
    res.set("stages", stagesArr.get());
    res.set("tokens", static_cast<double>(tokens));
    res.set("dim", static_cast<double>(dim));
    res.set("patchH", static_cast<double>(patchH));
    res.set("patchW", static_cast<double>(patchW));
    res.set("numPrefixTokens", 1.0);
    return res.build();
}

// encode(image, opts?) -> { features, tokens, dim, patchH, patchW,
//                           numPrefixTokens }
//   A single final hidden state: rows [0, numPrefixTokens) are CLS + the four
//   register tokens, the rest are patch tokens.
Value v3Encode(Value thisVal, std::span<const Value> args) {
    auto* w = v3Self(thisVal);
    if (!w) return ev::throwTypeError("Dinov3Backbone.prototype.encode: not a backbone");
    if (!w->backbone) return ev::throwError("encode: disposed");
    if (args.empty()) return ev::throwTypeError("encode(image, opts?): image required");

    int size = w->size;
    if (args.size() > 1 && ev::isObject(args[1])) {
        Value sv = ev::getProperty(args[1], "size");
        if (ev::isNumber(sv)) size = static_cast<int>(ev::toDouble(sv));
    }
    const int ps = w->backbone->config().patch_size;
    if (size <= 0 || size % ps != 0) {
        return ev::throwTypeError("encode: opts.size must be a positive multiple of patchSize");
    }

    std::vector<uint8_t> rgba;
    int inW = 0, inH = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, inW, inH, err)) {
        return ev::throwTypeError(std::string("encode: ") + err);
    }

    std::vector<float> feats;
    int tokens = 0, dim = 0, patchH = 0, patchW = 0, prefix = 0;
    try {
        std::vector<float> nchw;
        dinoPreprocess(rgba, inW, inH, size, nchw);
        brotensor::DeviceScope scope(w->device);
        brotensor::Tensor pxT = brotensor::Tensor::from_host_on(
            w->device, nchw.data(), 1, 3 * size * size);
        auto out = w->backbone->encode(pxT, size, size);
        feats = downloadF32(out.last_hidden_state);
        tokens = out.last_hidden_state.rows;
        dim = out.last_hidden_state.cols;
        patchH = out.patch_h;
        patchW = out.patch_w;
        prefix = out.num_prefix_tokens;
    } catch (const std::exception& e) {
        return ev::throwError(std::string("encode failed: ") + e.what());
    }

    ObjectBuilder res;
    {
        ev::Persistent f(makeFloat32Array(feats.data(), feats.size()));
        res.set("features", f.get());
    }
    res.set("tokens", static_cast<double>(tokens));
    res.set("dim", static_cast<double>(dim));
    res.set("patchH", static_cast<double>(patchH));
    res.set("patchW", static_cast<double>(patchW));
    res.set("numPrefixTokens", static_cast<double>(prefix));
    return res.build();
}

} // namespace

brovisionml::dinov2::Config dinov2ConfigForVariant(const std::string& v) {
    if (v == "base" || v == "vit_b") return brovisionml::dinov2::Config::vit_b();
    if (v == "large" || v == "vit_l") return brovisionml::dinov2::Config::vit_l();
    return brovisionml::dinov2::Config::vit_s();
}

void decorateDinov2Proto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v2Self(thisVal);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    proto.accessor("patchSize", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v2Self(thisVal);
        return ev::fromDouble(w && w->backbone ? w->backbone->config().patch_size : 0);
    });
    proto.accessor("embedDim", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v2Self(thisVal);
        return ev::fromDouble(w && w->backbone ? w->backbone->config().embed_dim : 0);
    });
    proto.accessor("defaultSize", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v2Self(thisVal);
        return ev::fromDouble(w ? w->size : 518);
    });

    proto.def("encode", 2, v2Encode);
    // The bronze port named this estimate(); it is the same operation.
    proto.def("estimate", 2, v2Encode);
    proto.def("dispose", 0, [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v2Self(thisVal);
        if (!w) return ev::throwTypeError("Dinov2Backbone.prototype.dispose: not a backbone");
        w->backbone.reset();
        w->loaded = false;
        return ev::undefined();
    });
}

void decorateDinov3Proto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v3Self(thisVal);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    proto.accessor("patchSize", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v3Self(thisVal);
        return ev::fromDouble(w && w->backbone ? w->backbone->config().patch_size : 0);
    });
    proto.accessor("embedDim", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v3Self(thisVal);
        return ev::fromDouble(w && w->backbone ? w->backbone->config().embed_dim : 0);
    });
    proto.accessor("numRegisterTokens", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v3Self(thisVal);
        return ev::fromDouble(w && w->backbone ? w->backbone->config().num_register_tokens : 0);
    });
    proto.accessor("defaultSize", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v3Self(thisVal);
        return ev::fromDouble(w ? w->size : 224);
    });

    proto.def("encode", 2, v3Encode);
    proto.def("estimate", 2, v3Encode);
    proto.def("dispose", 0, [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = v3Self(thisVal);
        if (!w) return ev::throwTypeError("Dinov3Backbone.prototype.dispose: not a backbone");
        w->backbone.reset();
        w->loaded = false;
        return ev::undefined();
    });
}

void loadDinov2Backbone(const std::string& dir, Value opts, Dinov2Wrapper& w) {
    std::string variant = "small";
    if (ev::isObject(opts)) {
        Value v = ev::getProperty(opts, "variant");
        if (!ev::isUndefined(v) && !ev::isNull(v) && !ev::isObject(v)) {
            variant = ev::toUtf8(v);
        }
    }
    const brovisionml::dinov2::Config cfg = dinov2ConfigForVariant(variant);
    w.size = cfg.img_size;
    try {
        w.backbone = std::make_unique<brovisionml::dinov2::Backbone>(cfg);
        brotensor::DeviceScope scope(w.device);
        w.backbone->load(dir);
        w.backbone->to(w.device);
        w.loaded = true;
    } catch (...) {
        // A missing checkpoint leaves the handle usable for surface probes,
        // as every other loader in this binding does.
        w.backbone.reset();
        w.loaded = false;
    }
}

void loadDinov3Backbone(const std::string& path, Dinov3Wrapper& w) {
    const brovisionml::dinov3::Config cfg = brovisionml::dinov3::Config::vit_h();
    w.size = cfg.img_size;
    try {
        w.backbone = std::make_unique<brovisionml::dinov3::Backbone>(cfg);
        brotensor::DeviceScope scope(w.device);
        const std::string ext = ".safetensors";
        const bool isFile = path.size() >= ext.size() &&
                            path.compare(path.size() - ext.size(), ext.size(), ext) == 0;
        if (isFile) {
            w.backbone->load_file(path);
        } else {
            w.backbone->load(path);
        }
        w.backbone->to(w.device);
        w.loaded = true;
    } catch (...) {
        w.backbone.reset();
        w.loaded = false;
    }
}

} // namespace brovisionml::api
