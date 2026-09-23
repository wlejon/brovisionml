// brovisionml_test_api — boots a bronze realm, installs bro.vision the way
// bro's host does, and checks the weights-free surface from JavaScript.
//
// Every check is a JS block that answers "OK" or a description of what went
// wrong: the harness holds no bronze Value across an allocating call (the
// test runs a second time under BRONZE_GC_STRESS=1, where a collection at
// every allocation moves every heap value, so a Value cached in C++ would be
// read after it moved). Failures are counted, never assert()ed — assert is a
// no-op in the Release configuration this runs in.

#include <brovisionml/version.h>
#include "../src/api/api.h"
#include <eval/eval.h>

#include <cstdio>
#include <string>

// tests/test_vision_api_restored.cpp — the bro.vision members the bronze port
// dropped or renamed (bro's docs/transition-drift.md row H7).
void brovisionmlTestRestoredSurface();

// tests/test_vision_api_paths.cpp — setPathResolver: the host's resolver is
// consulted for every model path the loaders take.
void brovisionmlTestPathResolver();

namespace {

namespace ev = bronze::embed;

int g_failures = 0;

void runJs(const char* name, const std::string& script) {
    std::printf("  %s...\n", name);
    ev::CallResult r = bronze::eval::evalScript(script);
    if (r.thrown) {
        std::printf("    FAIL: threw %s\n", ev::toUtf8(r.value).c_str());
        ++g_failures;
        return;
    }
    const std::string out = ev::isString(r.value) ? ev::toUtf8(r.value) : std::string("<non-string>");
    if (out != "OK") {
        std::printf("    FAIL: %s\n", out.c_str());
        ++g_failures;
    }
}

// Shared prelude for every block: a throw-expectation helper.
const char* kPrelude = R"JS(
    const V = bro.vision;
    const throwsWith = (f, pred) => {
        try { f(); } catch (e) { return pred(e) ? null : ("wrong error: " + e); }
        return "did not throw";
    };
)JS";

std::string block(const char* body) {
    return std::string("(() => {") + kPrelude + body + "\n})()";
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("Installing Vision API into Bronze realm...\n");
    brovisionml::api::installVision();

    runJs("namespace, version, init", block(R"JS(
        if (typeof bro !== "object" || typeof V !== "object") return "bro.vision missing";
        if (typeof V.version !== "string" || V.version.length === 0) return "version: " + V.version;
        if (V.init() !== undefined) return "init() did not return undefined";
        return "OK";
    )JS"));

    runJs("loaders validate their path", block(R"JS(
        const loaders = ["loadDepth", "loadSam", "loadNormal", "loadHed", "loadLineart",
                         "loadMlsd", "loadOpenpose", "loadSegformer", "loadBirefnet",
                         "loadStyleGAN3", "loadDinov2", "loadDinov3", "loadModel"];
        for (const name of loaders) {
            if (typeof V[name] !== "function") return name + " is not a function";
            let r = throwsWith(() => V[name](),
                               (e) => e instanceof TypeError && String(e.message).includes("path"));
            if (r) return name + "(): " + r;
            r = throwsWith(() => V[name]("tests/vision/__no_such_dir__"),
                           (e) => !(e instanceof TypeError) && String(e.message).includes(name));
            if (r) return name + "(missing dir): " + r;
        }
        return "OK";
    )JS"));

    // The device is resolved after the path, so these use a directory that
    // exists (".") and fail on the device before any weights are read.
    runJs("loaders validate opts.device", block(R"JS(
        for (const name of ["loadDepth", "loadSam", "loadHed", "loadBirefnet", "loadModel"]) {
            let r = throwsWith(() => V[name](".", { device: "tpu" }),
                               (e) => e instanceof TypeError && String(e.message).includes("opts.device"));
            if (r) return name + "({device:'tpu'}): " + r;
            r = throwsWith(() => V[name](".", { device: 3 }),
                           (e) => e instanceof TypeError && String(e.message).includes("opts.device"));
            if (r) return name + "({device:3}): " + r;
        }
        // An explicit CPU request is honoured: the load gets as far as the
        // missing checkpoint and says which loader failed.
        const r = throwsWith(() => V.loadDepth(".", { device: "cpu" }),
                             (e) => !(e instanceof TypeError) && String(e.message).includes("loadDepth failed"));
        if (r) return "loadDepth({device:'cpu'}) on a weightless dir: " + r;
        return "OK";
    )JS"));

    runJs("loadBirefnet validates modelSize", block(R"JS(
        const r = throwsWith(() => V.loadBirefnet(".", { device: "cpu", modelSize: 1000 }),
                             (e) => e instanceof TypeError && String(e.message).includes("modelSize"));
        return r ? "modelSize 1000: " + r : "OK";
    )JS"));

    runJs("host class constructors refuse construction", block(R"JS(
        const classes = ["DepthEstimator", "Sam", "NormalEstimator", "Hed", "Lineart", "Mlsd",
                         "Openpose", "Segformer", "Birefnet", "StyleGAN3", "Dinov2", "Dinov3",
                         "VisionModel"];
        for (const name of classes) {
            if (typeof V[name] !== "function") return name + " constructor missing";
            const r = throwsWith(() => V[name](),
                                 (e) => String(e.message).includes("is not a constructor") ||
                                        String(e).includes("TypeError"));
            if (r) return name + "(): " + r;
        }
        return "OK";
    )JS"));

    runJs("nms over box objects", block(R"JS(
        const boxes = [
            { x1: 10, y1: 10, x2: 50, y2: 50, score: 0.9, classId: 0 },
            { x1: 12, y1: 12, x2: 48, y2: 48, score: 0.8, classId: 0 },
            { x1: 100, y1: 100, x2: 150, y2: 150, score: 0.7, classId: 1 },
        ];
        const kept = V.nms(boxes, { iouThreshold: 0.5 });
        if (kept.length !== 2) return "kept " + kept.length;
        if (kept[0].x1 !== 10 || kept[1].x1 !== 100) return "kept the wrong boxes";
        return "OK";
    )JS"));

    runJs("nms over flat / typed / array-of-array inputs", block(R"JS(
        const flat = new Float32Array([
            10, 10, 50, 50, 0.9, 0,
            12, 12, 48, 48, 0.8, 0,
            100, 100, 150, 150, 0.7, 1,
        ]);
        const keptFlat = V.nms(flat, { iouThreshold: 0.5 });
        if (keptFlat.length !== 2) return "flat kept " + keptFlat.length;
        if (keptFlat[0].index !== 0 || keptFlat[1].index !== 2) return "flat indices";
        const indices = V.nms(flat, { iouThreshold: 0.5, returnIndices: true });
        if (!Array.isArray(indices) || indices.length !== 2 || indices[0] !== 0 || indices[1] !== 2) return "returnIndices";
        const typedIdx = V.nms(flat, { iouThreshold: 0.5, returnIndices: true, asTypedArray: true });
        if (!(typedIdx instanceof Int32Array) || typedIdx.length !== 2 || typedIdx[1] !== 2) return "typed returnIndices";
        const keptTyped = V.nms(flat, { iouThreshold: 0.5, asTypedArray: true });
        if (!(keptTyped instanceof Float32Array) || keptTyped.length !== 12) return "asTypedArray";
        const views = [
            new Float32Array([10, 10, 50, 50, 0.9, 0]),
            new Float32Array([12, 12, 48, 48, 0.8, 0]),
            new Float32Array([100, 100, 150, 150, 0.7, 1]),
        ];
        if (V.nms(views, { iouThreshold: 0.5 }).length !== 2) return "typed views";
        const arrays = [[10, 10, 50, 50, 0.9, 0], [12, 12, 48, 48, 0.8, 0], [100, 100, 150, 150, 0.7, 1]];
        const keptArr = V.nms(arrays, { iouThreshold: 0.5 });
        if (keptArr.length !== 2 || keptArr[1].x2 !== 150) return "array of arrays";
        const wrapped = V.nms({ boxes: flat }, { iouThreshold: 0.5 });
        if (wrapped.length !== 2) return "{ boxes } wrapper";
        const r = throwsWith(() => V.nms(flat, { stride: -1 }), (e) => e instanceof RangeError);
        if (r) return "negative stride: " + r;
        return "OK";
    )JS"));

    runJs("decodeBoxes", block(R"JS(
        const preds = new Float32Array([100, 100, 40, 40, 0.1, 0.85]);
        const boxes = V.decodeBoxes(preds, { numClasses: 2, confThreshold: 0.5 });
        if (boxes.length !== 1 || boxes[0].x1 !== 80 || boxes[0].classId !== 1) return "decode: " + JSON.stringify(boxes);
        const r = throwsWith(() => V.decodeBoxes(preds, { numClasses: 0 }), (e) => e instanceof TypeError);
        if (r) return "numClasses 0: " + r;
        return "OK";
    )JS"));

    runJs("rasterizeMask", block(R"JS(
        const poly = [{ x: 10, y: 10 }, { x: 50, y: 10 }, { x: 50, y: 50 }, { x: 10, y: 50 }];
        const mask = V.rasterizeMask(poly, { width: 64, height: 64 });
        if (mask.width !== 64 || mask.height !== 64 || !(mask.data instanceof Uint8Array)) return "polygon result shape";
        if (mask.data[30 * 64 + 30] !== 255 || mask.data[5 * 64 + 5] !== 0) return "polygon fill";
        const logits = new Float32Array(4 * 4); logits[5] = 1;
        const up = V.rasterizeMask(logits, { width: 4, height: 4, targetWidth: 8, targetHeight: 8 });
        // Source pixel (1, 1) covers target rows/cols 3..4 at 2x.
        if (up.data.length !== 64 || up.data[3 * 8 + 3] !== 255 || up.data[0] !== 0) return "logit upsample";
        // A typed mask smaller than width*height is refused, not over-read.
        let r = throwsWith(() => V.rasterizeMask(new Float32Array(4), { width: 64, height: 64 }),
                           (e) => e instanceof RangeError);
        if (r) return "undersized Float32Array: " + r;
        r = throwsWith(() => V.rasterizeMask(new Uint8Array(10), { targetWidth: 32, targetHeight: 32 }),
                       (e) => e instanceof RangeError);
        if (r) return "undersized Uint8Array: " + r;
        r = throwsWith(() => V.rasterizeMask(poly, { width: 1e9, height: 1e9 }), (e) => e instanceof RangeError);
        if (r) return "huge size: " + r;
        return "OK";
    )JS"));

    runJs("colorMap", block(R"JS(
        const depth = new Float32Array([0.1, 0.5, 0.9, 0.2]);
        const colored = V.colorMap(depth, { width: 2, height: 2, map: "turbo" });
        if (colored.width !== 2 || colored.height !== 2 || colored.data.length !== 16) return "shape";
        const gray = V.colorMap(depth, { width: 2, height: 2, map: "grayscale", min: 0, max: 1 });
        if (gray.data[8] !== 229 || gray.data[3] !== 255) return "grayscale " + Array.from(gray.data);
        let r = throwsWith(() => V.colorMap(depth, { width: -2, height: 2 }), (e) => e instanceof RangeError);
        if (r) return "negative width: " + r;
        r = throwsWith(() => V.colorMap(depth, { width: 100000, height: 100000 }), (e) => e instanceof RangeError);
        if (r) return "huge size: " + r;
        return "OK";
    )JS"));

    runJs("ops sub-namespace", block(R"JS(
        for (const n of ["nms", "decodeBoxes", "rasterizeMask", "colorMap", "colorizeDepth", "colorizeSegmentation"]) {
            if (typeof V.ops[n] !== "function") return "ops." + n + " missing";
        }
        return "OK";
    )JS"));

    brovisionmlTestRestoredSurface();
    brovisionmlTestPathResolver();

    if (g_failures > 0) {
        std::printf("FAILED: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("All brovisionml_api standalone tests passed successfully!\n");
    return 0;
}
