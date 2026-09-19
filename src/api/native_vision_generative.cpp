// StyleGAN3 (generate / synthesize / invert) and BiRefNet (removeBackground /
// dispose) — the two generative / matting classes.
//
// Ported from the QuickJS-era src/js/vision_bindings.cpp (js_stylegan3_generate
// / _synthesize / _invert, sg3BuildResult, js_loadStyleGAN3, js_birefnet_remove
// / _dispose). The bronze port had reduced StyleGAN3.generate() to a flat gray
// buffer, never constructed a Generator at all, and replaced
// BiRefNet.removeBackground() with an estimate() that returned a byte mask.
//
// The old bodies returned ImageBitmaps (`image`, `matte`) built by bro's DOM;
// a standalone sibling cannot mint one, so pixels come back as typed arrays
// (`data` / `alpha`) exactly as native_vision_ops.cpp already does.

#include "host_vision_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace brovisionml::api {

namespace sg3 = brovisionml::stylegan3;

namespace {

StyleGAN3Wrapper* sgSelf(Value thisVal) {
    void* p = g_stylegan3Class.unwrap(thisVal);
    if (!p) return nullptr;
    auto* w = static_cast<StyleGAN3Wrapper*>(p);
    return w->tag == kHostStyleGAN3Tag ? w : nullptr;
}

BirefnetWrapper* brSelf(Value thisVal) {
    void* p = g_birefnetClass.unwrap(thisVal);
    if (!p) return nullptr;
    auto* w = static_cast<BirefnetWrapper*>(p);
    return w->tag == kHostBirefnetTag ? w : nullptr;
}

// A Float32Array option, copied out immediately: the backing pointer does not
// survive the next allocating embed call.
bool readFloatsProp(Value obj, const char* key, std::vector<float>& out) {
    if (!ev::isObject(obj)) return false;
    Value v = ev::getProperty(obj, key);
    if (!ev::isObject(v)) return false;
    const float* data = nullptr;
    size_t count = 0;
    if (!readFloat32Array(v, data, count) || !data) return false;
    out.assign(data, data + count);
    return true;
}

void intProp(Value obj, const char* key, int& dst) {
    if (!ev::isObject(obj)) return;
    Value v = ev::getProperty(obj, key);
    if (ev::isNumber(v)) dst = static_cast<int>(ev::toDouble(v));
}

void floatProp(Value obj, const char* key, float& dst) {
    if (!ev::isObject(obj)) return;
    Value v = ev::getProperty(obj, key);
    if (ev::isNumber(v)) dst = static_cast<float>(ev::toDouble(v));
}

bool boolProp(Value obj, const char* key, bool def) {
    if (!ev::isObject(obj)) return def;
    Value v = ev::getProperty(obj, key);
    if (ev::isUndefined(v) || ev::isNull(v)) return def;
    return ev::toBool(v);
}

sg3::Config sg3ConfigFor(int res, char variant) {
    if (variant == 't') {
        if (res == 1024) return sg3::Config::t1024();
        if (res == 512) return sg3::Config::t512();
        return sg3::Config::t256();
    }
    if (res == 1024) return sg3::Config::r1024();
    if (res == 512) return sg3::Config::r512();
    return sg3::Config::r256();
}

// { data, width, height, channels, [seed], [w, numWs, wDim], [loss, lossCurve] }
// — the old sg3BuildResult with its ImageBitmap replaced by the pixel bytes.
Value sg3Result(const StyleGAN3Wrapper& w, const sg3::Image& img, int64_t seed,
                const std::vector<float>& ws, bool hasLoss, float loss,
                const std::vector<float>& lossCurve) {
    ObjectBuilder res;
    {
        ev::Persistent d(makeUint8Array(img.rgb.data(), img.rgb.size()));
        res.set("data", d.get());
    }
    res.set("width", static_cast<double>(img.width));
    res.set("height", static_cast<double>(img.height));
    res.set("channels", static_cast<double>(img.channels));
    if (seed >= 0) res.set("seed", static_cast<double>(seed));
    if (!ws.empty()) {
        {
            ev::Persistent wv(makeFloat32Array(ws.data(), ws.size()));
            res.set("w", wv.get());
        }
        res.set("numWs", static_cast<double>(w.numWs));
        res.set("wDim", static_cast<double>(w.wDim));
    }
    if (hasLoss) res.set("loss", static_cast<double>(loss));
    if (!lossCurve.empty()) {
        ev::Persistent lc(makeFloat32Array(lossCurve.data(), lossCurve.size()));
        res.set("lossCurve", lc.get());
    }
    return res.build();
}

// An un-loaded generator still answers, so a surface probe works without
// weights: a mid-gray image of the model resolution, no latents.
Value sg3Placeholder(const StyleGAN3Wrapper* w, int64_t seed) {
    const int res = w ? w->imgResolution : 1024;
    const int ch = w ? w->imgChannels : 3;
    std::vector<uint8_t> gray(static_cast<size_t>(res) * res * ch, 128);
    ObjectBuilder r;
    {
        ev::Persistent d(makeUint8Array(gray.data(), gray.size()));
        r.set("data", d.get());
    }
    r.set("width", static_cast<double>(res));
    r.set("height", static_cast<double>(res));
    r.set("channels", static_cast<double>(ch));
    if (seed >= 0) r.set("seed", static_cast<double>(seed));
    return r.build();
}

// ─────────────────────────────────────────────────────────────────────────
// StyleGAN3
// ─────────────────────────────────────────────────────────────────────────

// generate(opts?) -> { data, width, height, channels, seed?, w? }
//   opts.seed             sample z ~ N(0,1) from this seed (default 0)
//   opts.z                Float32Array(zDim) — used instead of a seed
//   opts.truncation       psi toward w_avg (default 1.0); < 1 = tamer
//   opts.truncationCutoff rows to truncate; -1 = all (default)
//   opts.returnLatents    also return the mapped W+ as a Float32Array
Value sgGenerate(Value thisVal, std::span<const Value> args) {
    auto* w = sgSelf(thisVal);
    if (!w) return ev::throwTypeError("StyleGAN3.prototype.generate: not a generator");

    Value opts = args.empty() ? ev::undefined() : args[0];
    float psi = 1.0f;
    int cutoff = -1;
    floatProp(opts, "truncation", psi);
    intProp(opts, "truncationCutoff", cutoff);
    const bool returnLatents = boolProp(opts, "returnLatents", false);

    // An explicit z wins over a seed.
    std::vector<float> z;
    bool gotZ = false;
    if (readFloatsProp(opts, "z", z)) {
        if (static_cast<int>(z.size()) == w->zDim) {
            gotZ = true;
        } else if (!z.empty()) {
            return ev::throwTypeError("generate: opts.z must have length zDim");
        }
    }
    int64_t seed = -1;
    if (!gotZ) {
        seed = 0;
        if (ev::isObject(opts)) {
            Value sv = ev::getProperty(opts, "seed");
            if (ev::isNumber(sv)) seed = static_cast<int64_t>(ev::toDouble(sv));
        }
        z.resize(static_cast<size_t>(w->zDim));
        std::mt19937_64 rng(static_cast<unsigned long long>(seed));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : z) v = nd(rng);
    }

    if (!(w->loaded && w->generator)) return sg3Placeholder(w, seed);

    try {
        brotensor::DeviceScope scope(w->device);
        brotensor::Tensor zt = brotensor::Tensor::mat(1, w->zDim);
        for (int i = 0; i < w->zDim; ++i) zt[i] = z[static_cast<size_t>(i)];
        if (w->device != brotensor::Device::CPU) zt = zt.to(w->device);
        brotensor::Tensor ws = w->generator->map(zt, psi, cutoff);
        sg3::Image img = w->generator->render(ws);
        std::vector<float> wsOut;
        if (returnLatents) {
            brotensor::Tensor wc = ws.to(brotensor::Device::CPU);
            const float* p = wc.host_f32();
            wsOut.assign(p, p + static_cast<size_t>(w->numWs) * w->wDim);
        }
        return sg3Result(*w, img, seed, wsOut, false, 0.0f, {});
    } catch (const std::exception& e) {
        return ev::throwError(std::string("generate failed: ") + e.what());
    }
}

// synthesize(w) -> { data, width, height, channels }
//   w: Float32Array — the full W+ (numWs*wDim) or a single w (wDim), broadcast
//   across all rows. Renders a latent edited from generate({returnLatents:true}).
Value sgSynthesize(Value thisVal, std::span<const Value> args) {
    auto* w = sgSelf(thisVal);
    if (!w) return ev::throwTypeError("StyleGAN3.prototype.synthesize: not a generator");
    if (args.empty() || !ev::isObject(args[0])) {
        return ev::throwTypeError("synthesize(w, opts?): w Float32Array required");
    }

    std::vector<float> win;
    {
        const float* data = nullptr;
        size_t count = 0;
        if (!readFloat32Array(args[0], data, count) || !data) {
            return ev::throwTypeError("synthesize: w must be a Float32Array");
        }
        win.assign(data, data + count);
    }
    const int full = w->numWs * w->wDim;
    if (static_cast<int>(win.size()) != full && static_cast<int>(win.size()) != w->wDim) {
        return ev::throwTypeError(
            "synthesize: w must have length numWs*wDim (W+) or wDim (single w)");
    }

    if (!(w->loaded && w->generator)) return sg3Placeholder(w, -1);

    try {
        brotensor::DeviceScope scope(w->device);
        brotensor::Tensor ws = brotensor::Tensor::mat(w->numWs, w->wDim);
        const bool single = static_cast<int>(win.size()) == w->wDim;
        for (int r = 0; r < w->numWs; ++r) {
            for (int c = 0; c < w->wDim; ++c) {
                ws[r * w->wDim + c] =
                    win[static_cast<size_t>(single ? c : r * w->wDim + c)];
            }
        }
        if (w->device != brotensor::Device::CPU) ws = ws.to(w->device);
        sg3::Image img = w->generator->render(ws);
        return sg3Result(*w, img, -1, {}, false, 0.0f, {});
    } catch (const std::exception& e) {
        return ev::throwError(std::string("synthesize failed: ") + e.what());
    }
}

// invert(image, opts?) -> { data, width, height, channels, w, numWs, wDim,
//                           loss, lossCurve }
//   Optimization-based GAN inversion: Adam on the W+ rows minimizing
//   image-space MSE through the frozen synthesis.
//   opts.steps (350) / lr (0.05) / regW (0) / initNoise (0) / seed (0)
//   opts.initW  Float32Array(numWs*wDim) start latent (resume / refine)
Value sgInvert(Value thisVal, std::span<const Value> args) {
    auto* w = sgSelf(thisVal);
    if (!w) return ev::throwTypeError("StyleGAN3.prototype.invert: not a generator");
    if (args.empty()) return ev::throwTypeError("invert(image, opts?): image required");

    std::vector<uint8_t> rgba;
    int iw = 0, ih = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, iw, ih, err)) {
        return ev::throwTypeError(std::string("invert: ") + err);
    }
    if (iw != w->imgResolution || ih != w->imgResolution) {
        return ev::throwTypeError(
            "invert: image must be square at the model resolution; "
            "resize the source first");
    }

    Value opts = args.size() > 1 ? args[1] : ev::undefined();
    int steps = 350;
    float lr = 0.05f, regW = 0.0f, initNoise = 0.0f;
    int64_t seed = 0;
    intProp(opts, "steps", steps);
    floatProp(opts, "lr", lr);
    floatProp(opts, "regW", regW);
    floatProp(opts, "initNoise", initNoise);
    if (ev::isObject(opts)) {
        Value sv = ev::getProperty(opts, "seed");
        if (ev::isNumber(sv)) seed = static_cast<int64_t>(ev::toDouble(sv));
    }
    if (steps < 1) steps = 1;

    std::vector<float> initW;
    if (readFloatsProp(opts, "initW", initW) && !initW.empty() &&
        static_cast<int>(initW.size()) != w->numWs * w->wDim) {
        return ev::throwTypeError("invert: opts.initW must have length numWs*wDim");
    }

    if (!(w->loaded && w->generator)) {
        return ev::throwError("invert: no weights loaded");
    }

    try {
        brotensor::DeviceScope scope(w->device);
        sg3::Image target;
        target.width = iw;
        target.height = ih;
        target.channels = 3;
        target.rgb.resize(static_cast<size_t>(iw) * ih * 3);
        for (size_t i = 0, n = static_cast<size_t>(iw) * ih; i < n; ++i) {
            target.rgb[3 * i + 0] = rgba[4 * i + 0];
            target.rgb[3 * i + 1] = rgba[4 * i + 1];
            target.rgb[3 * i + 2] = rgba[4 * i + 2];
        }

        sg3::Generator::InvertOptions io;
        io.num_steps = steps;
        io.lr = lr;
        io.reg_w = regW;
        io.init_noise = initNoise;
        io.seed = static_cast<uint64_t>(seed);
        if (!initW.empty()) {
            io.init_w = brotensor::Tensor::from_host_on(w->device, initW.data(),
                                                        w->numWs, w->wDim);
        }
        std::vector<float> curve;
        curve.reserve(static_cast<size_t>(steps));
        io.on_step = [&curve](int, float l) { curve.push_back(l); };

        sg3::Generator::InvertResult r = w->generator->invert(target, io);
        brotensor::Tensor wc = r.w.to(brotensor::Device::CPU);
        const float* p = wc.host_f32();
        std::vector<float> wsOut(p, p + static_cast<size_t>(w->numWs) * w->wDim);
        sg3::Image img = w->generator->render(r.w);
        return sg3Result(*w, img, -1, wsOut, true, r.loss, curve);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("invert failed: ") + e.what());
    }
}

// ─────────────────────────────────────────────────────────────────────────
// BiRefNet
// ─────────────────────────────────────────────────────────────────────────

// removeBackground(image, opts?) -> { width, height, alpha, matte, data }
//   alpha  Float32Array(h*w) in [0, 1]
//   matte  Uint8Array(h*w)   the same matte as bytes (the old `matte` bitmap)
//   data   Uint8Array(h*w*4) the input RGBA with its alpha replaced — the
//          ready-to-draw cutout the old binding returned as `image`.
Value brRemoveBackground(Value thisVal, std::span<const Value> args) {
    auto* w = brSelf(thisVal);
    if (!w) {
        return ev::throwTypeError(
            "BiRefNet.prototype.removeBackground: not a BiRefNet instance");
    }
    if (args.empty()) {
        return ev::throwTypeError("removeBackground(image, opts?): image required");
    }

    std::vector<uint8_t> rgba;
    int inW = 0, inH = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, inW, inH, err)) {
        return ev::throwTypeError(std::string("removeBackground: ") + err);
    }

    int modelSize = w->modelSize;
    if (args.size() > 1) intProp(args[1], "modelSize", modelSize);

    std::vector<float> alpha;
    int outW = inW, outH = inH;
    if (w->loaded && w->net) {
        try {
            // removeBackground() consumes interleaved float RGB; the decoded
            // alpha plane does not participate in matting.
            const size_t n = static_cast<size_t>(inW) * inH;
            std::vector<float> rgb(n * 3);
            for (size_t i = 0; i < n; ++i) {
                rgb[3 * i + 0] = rgba[4 * i + 0];
                rgb[3 * i + 1] = rgba[4 * i + 1];
                rgb[3 * i + 2] = rgba[4 * i + 2];
            }
            brotensor::DeviceScope scope(w->device);
            auto matte = w->net->removeBackground(rgb.data(), inW, inH,
                                                  /*rgbIs255=*/true, modelSize);
            alpha = std::move(matte.alpha);
            outW = matte.width;
            outH = matte.height;
        } catch (const std::exception& e) {
            return ev::throwError(std::string("removeBackground failed: ") + e.what());
        }
    } else {
        alpha.assign(static_cast<size_t>(inW) * inH, 0.0f);
    }

    std::vector<uint8_t> bytes(alpha.size());
    for (size_t i = 0; i < alpha.size(); ++i) {
        bytes[i] = static_cast<uint8_t>(
            std::lround(std::clamp(alpha[i], 0.0f, 1.0f) * 255.0f));
    }
    std::vector<uint8_t> cut = rgba;
    for (size_t i = 0; i < alpha.size() && i * 4 + 3 < cut.size(); ++i) {
        cut[4 * i + 3] = bytes[i];
    }

    ObjectBuilder res;
    res.set("width", static_cast<double>(outW));
    res.set("height", static_cast<double>(outH));
    {
        ev::Persistent a(makeFloat32Array(alpha.data(), alpha.size()));
        res.set("alpha", a.get());
    }
    {
        ev::Persistent m(makeUint8Array(bytes.data(), bytes.size()));
        res.set("matte", m.get());
        // `mask` is what the bronze port's estimate() called this plane; keep
        // it so callers written against the port keep working.
        ev::Persistent m2(makeUint8Array(bytes.data(), bytes.size()));
        res.set("mask", m2.get());
    }
    {
        ev::Persistent c(makeUint8Array(cut.data(), cut.size()));
        res.set("data", c.get());
    }
    return res.build();
}

} // namespace

bool loadStyleGAN3Generator(const std::string& path, Value opts,
                            StyleGAN3Wrapper& w, std::string& err) {
    int res = 256;
    intProp(opts, "resolution", res);
    if (res != 256 && res != 512 && res != 1024) {
        err = "loadStyleGAN3: resolution must be 256, 512, or 1024";
        return false;
    }
    std::string variant = "r";
    if (ev::isObject(opts)) {
        Value v = ev::getProperty(opts, "variant");
        if (!ev::isUndefined(v) && !ev::isNull(v) && !ev::isObject(v)) {
            variant = ev::toUtf8(v);
        }
    }
    if (variant != "r" && variant != "t") {
        err = "loadStyleGAN3: variant must be 'r' or 't'";
        return false;
    }

    const sg3::Config cfg = sg3ConfigFor(res, variant[0]);
    w.imgResolution = res;
    w.variant = variant[0];
    w.zDim = cfg.z_dim;
    w.wDim = cfg.w_dim;
    w.numWs = cfg.num_ws();
    w.imgChannels = cfg.img_channels;
    try {
        w.generator = std::make_unique<sg3::Generator>(cfg);
        brotensor::DeviceScope scope(w.device);
        w.generator->load(path);
        w.generator->to(w.device);
        w.loaded = true;
    } catch (const std::exception& e) {
        // A missing checkpoint is not fatal: the handle stays usable for
        // surface probes, exactly as the other loaders in this binding.
        w.generator.reset();
        w.loaded = false;
        err = e.what();
    }
    return true;
}

void decorateStyleGAN3Proto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sgSelf(thisVal);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    proto.accessor("zDim", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sgSelf(thisVal);
        return ev::fromDouble(w ? w->zDim : 512);
    });
    proto.accessor("cDim", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sgSelf(thisVal);
        return ev::fromDouble(w ? w->cDim : 0);
    });
    proto.accessor("wDim", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sgSelf(thisVal);
        return ev::fromDouble(w ? w->wDim : 512);
    });
    proto.accessor("numWs", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sgSelf(thisVal);
        return ev::fromDouble(w ? w->numWs : 16);
    });
    proto.accessor("resolution", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sgSelf(thisVal);
        return ev::fromDouble(w ? w->imgResolution : 1024);
    });
    // imgResolution / imgChannels are the bronze port's names for the same two
    // numbers; kept so callers written against the port keep working.
    proto.accessor("imgResolution", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sgSelf(thisVal);
        return ev::fromDouble(w ? w->imgResolution : 1024);
    });
    proto.accessor("imgChannels", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sgSelf(thisVal);
        return ev::fromDouble(w ? w->imgChannels : 3);
    });
    proto.accessor("variant", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = sgSelf(thisVal);
        return ev::fromUtf8(std::string(1, w ? w->variant : 'r'));
    });

    proto.def("generate", 1, sgGenerate);
    proto.def("synthesize", 2, sgSynthesize);
    proto.def("invert", 2, sgInvert);
}

void decorateBirefnetProto(ObjectBuilder& proto) {
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = brSelf(thisVal);
        return ev::fromUtf8(w ? deviceName(w->device) : "CPU");
    });
    proto.accessor("modelSize", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = brSelf(thisVal);
        return ev::fromDouble(w ? w->modelSize : 1024);
    });

    proto.def("removeBackground", 2, brRemoveBackground);
    // The bronze port named this estimate(); it is the same operation.
    proto.def("estimate", 2, brRemoveBackground);

    // dispose() — the weights are GPU-resident and invisible to the JS heap
    // accounting, so a caller done with the model must not wait for the GC.
    proto.def("dispose", 0, [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = brSelf(thisVal);
        if (!w) return ev::throwTypeError("BiRefNet.prototype.dispose: not a BiRefNet");
        w->net.reset();
        w->loaded = false;
        return ev::undefined();
    });
}

} // namespace brovisionml::api
