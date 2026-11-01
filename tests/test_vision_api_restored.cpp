// Coverage for the bro.vision class prototypes: SAM's setImage / segment /
// segmentEverything, the annotators' detect() and estimate(), the depth and
// normal estimators, BiRefNet, StyleGAN3 and the DINOv2 / DINOv3 backbones.
//
// The heavyweight paths need checkpoints this repo does not ship, so the
// checks run against the class prototypes: every name must be a callable
// member with its declared arity, the model-bound ones must reject a foreign
// receiver, and the accessors must be present.
//
// Linked into brovisionml_test_api; called from its main(). Failures exit the
// process: assert() is a no-op in the Release configuration this test runs in.

#include "embed/embed.h"
#include <eval/eval.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
namespace ev = bronze::embed;
} // namespace

void brovisionmlTestClassSurface() {
    std::cout << "Checking the bro.vision class surface..." << std::endl;

    const char* script = R"JS(
        const V = bro.vision;
        const proto = (n) => V[n].prototype;
        const fail = (m) => { throw new Error(m); };

        const isFn = (o, n) => typeof o[n] === "function";
        const needFn = (o, cls, n, arity) => {
            if (!isFn(o, n)) fail(cls + ".prototype." + n + " is missing");
            if (arity !== undefined && o[n].length !== arity) {
                fail(cls + ".prototype." + n + " arity " + o[n].length + ", expected " + arity);
            }
        };
        const needThrows = (o, cls, n, args) => {
            let threw = false;
            try { o[n].apply(undefined, args || []); } catch (e) { threw = true; }
            if (!threw) fail(cls + "." + n + " accepted a foreign receiver");
        };
        const needGetter = (o, cls, n) => {
            if (Object.getOwnPropertyDescriptor(o, n) === undefined) {
                fail(cls + ".prototype." + n + " accessor is missing");
            }
        };

        // A 2x2 RGBA image, enough to drive the model-free fallbacks.
        const img = { width: 2, height: 2, data: new Uint8Array(16) };

        // ── Sam: segment() reads labels/boxes/multimask, and
        //    segmentEverything() takes the AMG config ─────────────────────
        const sam = proto("Sam");
        needFn(sam, "Sam", "setImage", 2);
        needFn(sam, "Sam", "segment", 1);
        needFn(sam, "Sam", "segmentEverything", 2);
        needThrows(sam, "Sam", "segment", [{ points: [[1, 1]], labels: [1], multimask: false }]);
        needThrows(sam, "Sam", "segmentEverything", [img, { pointsPerSide: 8 }]);
        needThrows(sam, "Sam", "setImage", [img]);
        needGetter(sam, "Sam", "hasImage");

        // ── Annotators: detect() and estimate() are both published ────────
        const scalarPlanes = [["Hed", "edge", "edges"], ["Lineart", "line", "lines"]];
        for (const [cls, floatKey, byteKey] of scalarPlanes) {
            const p = proto(cls);
            needFn(p, cls, "detect", 2);
            needFn(p, cls, "estimate", 2);
            needThrows(p, cls, "detect", [img]);
            needThrows(p, cls, "estimate", [img]);
        }

        const mlsd = proto("Mlsd");
        needFn(mlsd, "Mlsd", "detect", 2);
        needFn(mlsd, "Mlsd", "estimate", 2);
        needThrows(mlsd, "Mlsd", "detect", [img]);
        needThrows(mlsd, "Mlsd", "estimate", [img]);

        const op = proto("Openpose");
        needFn(op, "Openpose", "detect", 2);
        needFn(op, "Openpose", "estimate", 2);
        needThrows(op, "Openpose", "detect", [img]);
        needThrows(op, "Openpose", "estimate", [img]);

        const sf = proto("Segformer");
        needFn(sf, "Segformer", "detect", 2);
        needFn(sf, "Segformer", "estimate", 2);
        needThrows(sf, "Segformer", "detect", [img]);
        needThrows(sf, "Segformer", "estimate", [img]);

        // ── DepthEstimator.estimate(image, { invert }) ────────────────────
        const dep = proto("DepthEstimator");
        needFn(dep, "DepthEstimator", "estimate", 2);
        needThrows(dep, "DepthEstimator", "estimate", [img]);

        // ── NormalEstimator.estimate(image, { fx, fy, cx, cy }) ───────────
        const nrm = proto("NormalEstimator");
        needFn(nrm, "NormalEstimator", "estimate", 2);
        needThrows(nrm, "NormalEstimator", "estimate", [img]);

        // ── BiRefNet.removeBackground / dispose ───────────────────────────
        const br = proto("Birefnet");
        needFn(br, "Birefnet", "removeBackground", 2);
        needFn(br, "Birefnet", "estimate", 2);
        needFn(br, "Birefnet", "dispose", 0);
        needThrows(br, "Birefnet", "removeBackground", [img]);
        needThrows(br, "Birefnet", "dispose", []);
        needGetter(br, "Birefnet", "modelSize");

        // ── StyleGAN3.generate / synthesize / invert ──────────────────────
        const sg = proto("StyleGAN3");
        needFn(sg, "StyleGAN3", "generate", 1);
        needFn(sg, "StyleGAN3", "synthesize", 2);
        needFn(sg, "StyleGAN3", "invert", 2);
        needThrows(sg, "StyleGAN3", "generate", [{ seed: 7, truncation: 0.7, returnLatents: true }]);
        needThrows(sg, "StyleGAN3", "synthesize", [new Float32Array(512)]);
        needThrows(sg, "StyleGAN3", "invert", [img, { steps: 2 }]);
        for (const g of ["zDim", "wDim", "numWs", "resolution", "variant"]) needGetter(sg, "StyleGAN3", g);

        // ── DINOv2 / DINOv3 encode + dispose ──────────────────────────────
        const d2 = proto("Dinov2");
        needFn(d2, "Dinov2", "encode", 2);
        needFn(d2, "Dinov2", "dispose", 0);
        needThrows(d2, "Dinov2", "encode", [img]);
        needThrows(d2, "Dinov2", "dispose", []);
        for (const g of ["patchSize", "embedDim", "defaultSize"]) needGetter(d2, "Dinov2", g);

        const d3 = proto("Dinov3");
        needFn(d3, "Dinov3", "encode", 2);
        needFn(d3, "Dinov3", "dispose", 0);
        needThrows(d3, "Dinov3", "encode", [img, { size: 224 }]);
        needThrows(d3, "Dinov3", "dispose", []);
        for (const g of ["patchSize", "embedDim", "numRegisterTokens", "defaultSize"]) {
            needGetter(d3, "Dinov3", g);
        }

        "SUCCESS";
    )JS";

    auto res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::cerr << "  class-surface script threw: " << ev::toUtf8(res.value) << std::endl;
        std::exit(1);
    }
    if (ev::toUtf8(res.value) != "SUCCESS") {
        std::cerr << "  class-surface script returned: " << ev::toUtf8(res.value) << std::endl;
        std::exit(1);
    }
    std::cout << "  bro.vision class surface OK." << std::endl;
}
