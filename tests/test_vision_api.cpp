#include <brovisionml/version.h>
#include "../src/api/api.h"
#include <eval/eval.h>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main() {
    namespace ev = bronze::embed;

    std::cout << "Installing Vision API into Bronze realm..." << std::endl;
    brovisionml::api::installVision();

    auto g = ev::globalValue("bro");
    assert(g.found);
    assert(ev::isObject(g.value));

    // Check bro.vision namespace
    auto vision = ev::getProperty(g.value, "vision");
    assert(ev::isObject(vision));

    // Check version
    auto ver = ev::getProperty(vision, "version");
    assert(ev::isString(ver));
    assert(!ev::toUtf8(ver).empty());
    std::cout << "  bro.vision.version = " << ev::toUtf8(ver) << std::endl;

    // Check init()
    auto visionInit = ev::getProperty(vision, "init");
    assert(ev::isObject(visionInit));
    auto initRes = ev::call(visionInit, vision, {});
    assert(!initRes.thrown);
    assert(ev::isUndefined(initRes.value));

    // Verify all loaders exist and are functions
    const std::vector<std::string> loaders = {
        "loadDepth", "loadSam", "loadNormal", "loadHed", "loadLineart",
        "loadMlsd", "loadOpenpose", "loadSegformer", "loadBirefnet",
        "loadStyleGAN3", "loadDinov2", "loadDinov3", "loadModel"
    };

    for (const auto& name : loaders) {
        auto fn = ev::getProperty(vision, name.c_str());
        assert(ev::isObject(fn));

        // Calling with no args must throw TypeError mentioning 'path'
        auto badCall = ev::call(fn, vision, {});
        assert(badCall.thrown);
        std::string err = ev::toUtf8(badCall.value);
        assert(err.find("path") != std::string::npos);

        // Calling with nonexistent dir must throw runtime Error (not TypeError) with loader name
        ev::Persistent bogusArg(ev::fromUtf8("tests/vision/__no_such_dir__"));
        const ev::Value callArgs[1] = {bogusArg.get()};
        auto missingCall = ev::call(fn, vision, std::span<const ev::Value>(callArgs, 1));
        assert(missingCall.thrown);
        std::string missErr = ev::toUtf8(missingCall.value);
        assert(missErr.find(name) != std::string::npos);
    }
    std::cout << "  All loaders validated successfully (TypeError on empty, runtime Error on nonexistent path)" << std::endl;

    // Verify constructors exist on bro.vision
    const std::vector<std::string> constructors = {
        "DepthEstimator", "Sam", "NormalEstimator", "Hed", "Lineart",
        "Mlsd", "Openpose", "Segformer", "Birefnet", "StyleGAN3",
        "Dinov2", "Dinov3", "VisionModel"
    };

    for (const auto& name : constructors) {
        auto ctor = ev::getProperty(vision, name.c_str());
        assert(ev::isObject(ctor));

        // Calling constructor directly should throw TypeError
        auto badCtor = ev::call(ctor, ev::undefined(), {});
        assert(badCtor.thrown);
        std::string ctorErr = ev::toUtf8(badCtor.value);
        assert(ctorErr.find("is not a constructor") != std::string::npos ||
               ctorErr.find("TypeError") != std::string::npos);
    }
    std::cout << "  All host class constructors verified" << std::endl;

    // ── Test Vision Ops: decodeBoxes, NMS, rasterizeMask, colorMap ─────────
    auto ops = ev::getProperty(vision, "ops");
    assert(ev::isObject(ops));

    // 1. NMS
    auto nmsFn = ev::getProperty(vision, "nms");
    assert(ev::isObject(nmsFn));

    // Test NMS through Bronze eval
    auto evalRes = bronze::eval::evalScript(
        "(() => {"
        "  const boxes = ["
        "    { x1: 10, y1: 10, x2: 50, y2: 50, score: 0.9, classId: 0 },"
        "    { x1: 12, y1: 12, x2: 48, y2: 48, score: 0.8, classId: 0 }," // highly overlapping -> should suppress
        "    { x1: 100, y1: 100, x2: 150, y2: 150, score: 0.7, classId: 1 }" // far away -> should keep
        "  ];"
        "  const kept = bro.vision.nms(boxes, { iouThreshold: 0.5 });"
        "  return kept.length;"
        "})()"
    );
    assert(!evalRes.thrown);
    assert(ev::toDouble(evalRes.value) == 2.0);
    std::cout << "  bro.vision.nms correctly suppressed overlapping box (kept 2)" << std::endl;

    // 2. decodeBoxes
    auto decodeRes = bronze::eval::evalScript(
        "(() => {\n"
        "  const preds = new Float32Array([100, 100, 40, 40, 0.1, 0.85]);\n"
        "  const boxes = bro.vision.decodeBoxes(preds, { numClasses: 2, confThreshold: 0.5 });\n"
        "  return { len: boxes.length, x1: boxes[0].x1, classId: boxes[0].classId };\n"
        "})()"
    );
    if (decodeRes.thrown) {
        std::cerr << "decodeRes thrown: " << ev::toUtf8(decodeRes.value) << std::endl;
    }
    assert(!decodeRes.thrown);
    assert(ev::isObject(decodeRes.value));
    auto decLen = ev::getProperty(decodeRes.value, "len");
    auto decX1 = ev::getProperty(decodeRes.value, "x1");
    auto decClass = ev::getProperty(decodeRes.value, "classId");
    assert(ev::toDouble(decLen) == 1.0);
    assert(ev::toDouble(decX1) == 80.0); // 100 - 40/2 = 80
    assert(ev::toDouble(decClass) == 1.0);
    std::cout << "  bro.vision.decodeBoxes decoded YOLO format correctly" << std::endl;

    // 3. rasterizeMask
    auto maskRes = bronze::eval::evalScript(
        "(() => {"
        "  const poly = [{x: 10, y: 10}, {x: 50, y: 10}, {x: 50, y: 50}, {x: 10, y: 50}];"
        "  const mask = bro.vision.rasterizeMask(poly, { width: 64, height: 64 });"
        "  return { w: mask.width, h: mask.height, hasData: mask.data instanceof Uint8Array };"
        "})()"
    );
    assert(!maskRes.thrown);
    assert(ev::isObject(maskRes.value));
    assert(ev::toDouble(ev::getProperty(maskRes.value, "w")) == 64.0);
    assert(ev::toDouble(ev::getProperty(maskRes.value, "h")) == 64.0);
    assert(ev::toBool(ev::getProperty(maskRes.value, "hasData")) == true);
    std::cout << "  bro.vision.rasterizeMask generated binary mask" << std::endl;

    // 4. colorMap
    auto colorRes = bronze::eval::evalScript(
        "(() => {"
        "  const depth = new Float32Array([0.1, 0.5, 0.9, 0.2]);"
        "  const colored = bro.vision.colorMap(depth, { width: 2, height: 2, map: 'turbo' });"
        "  return { w: colored.width, h: colored.height, len: colored.data.length };"
        "})()"
    );
    assert(!colorRes.thrown);
    assert(ev::isObject(colorRes.value));
    assert(ev::toDouble(ev::getProperty(colorRes.value, "w")) == 2.0);
    assert(ev::toDouble(ev::getProperty(colorRes.value, "len")) == 16.0); // 2 * 2 * 4 bytes RGBA
    std::cout << "  bro.vision.colorMap colored depth map to RGBA" << std::endl;

    // 5. Check ops sub-namespace equivalence
    auto opsCheck = bronze::eval::evalScript(
        "typeof bro.vision.ops.nms === 'function' && "
        "typeof bro.vision.ops.decodeBoxes === 'function' && "
        "typeof bro.vision.ops.rasterizeMask === 'function' && "
        "typeof bro.vision.ops.colorMap === 'function'"
    );
    assert(!opsCheck.thrown);
    assert(ev::toBool(opsCheck.value) == true);
    std::cout << "  bro.vision.ops sub-namespace fully populated" << std::endl;

    std::cout << "All brovisionml_api standalone tests passed successfully!" << std::endl;
    return 0;
}
