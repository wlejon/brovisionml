// OWLv2 pieces that need no weights: preprocessing (square pad with 0.5 grey,
// anti-aliased resize, CLIP normalisation) and decode (best query, sigmoid,
// threshold, NMS, padded-square -> image pixels). The model itself is checked
// against transformers in test_owlv2_parity.cpp.
#include "brovisionml/owlv2.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace brovisionml::owlv2;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

constexpr float kMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float kStd[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

float norm(float v01, int c) { return (v01 - kMean[c]) / kStd[c]; }

void test_preprocess() {
    // A flat white 200 x 100 image: the top half of the square is white, the
    // bottom half (the pad) is 0.5 grey; the seam is the only blend.
    const int w = 200, h = 100, S = 64;
    std::vector<uint8_t> rgba(static_cast<std::size_t>(w) * h * 4, 255);
    const Pixels px = preprocess(rgba.data(), w, h, 4, S);
    check(px.padded == 200 && px.width == w && px.height == h, "padded side");
    check(px.nchw.rows == 1 && px.nchw.cols == 3 * S * S, "tensor shape");
    const float* d = px.nchw.host_f32();
    bool white = true, grey = true;
    for (int c = 0; c < 3; ++c) {
        for (int x = 0; x < S; ++x) {
            white = white && std::fabs(d[c * S * S + 5 * S + x] - norm(1.0f, c)) < 1e-4f;
            grey = grey && std::fabs(d[c * S * S + (S - 5) * S + x] - norm(0.5f, c)) < 1e-4f;
        }
    }
    check(white, "image rows stay white through the blur");
    check(grey, "pad rows are 0.5 grey");

    // Upscaling (no blur): a 2 x 2 RGB checker resampled bilinearly.
    const uint8_t rgb[12] = {0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0};
    const Pixels up = preprocess(rgb, 2, 2, 3, 8);
    const float* u = up.nchw.host_f32();
    check(std::fabs(u[0] - norm(0.f, 0)) < 1e-5f, "corner keeps its value");
    // Column 3 samples source x = 0.25 * 3.5 - 0.5 = 0.375 (align_corners=False).
    check(std::fabs(u[3] - norm(0.375f, 0)) < 1e-5f, "bilinear tap");
}

void test_decode() {
    // Three patches, two queries. Patch 0 and 1 overlap heavily (NMS keeps the
    // better one); patch 2 is below the threshold.
    Predictions p;
    p.num_patches = 3;
    p.num_queries = 2;
    p.logits = {2.0f, -1.0f,     // query 0, sigmoid 0.881
                -3.0f, 1.0f,     // query 1, sigmoid 0.731
                -4.0f, -5.0f};   // below 0.1
    p.boxes = {0.25f, 0.25f, 0.2f, 0.2f,
               0.26f, 0.25f, 0.2f, 0.2f,
               0.75f, 0.75f, 0.1f, 0.1f};
    p.objectness = {0.f, 0.f, 0.f};
    Pixels px;
    px.width = 400;
    px.height = 200;
    px.padded = 400;
    DetectOptions o;
    std::vector<Detection> d = Detector::decode(p, px, o);
    check(d.size() == 1, "NMS drops the overlapping box, threshold the weak one");
    if (!d.empty()) {
        check(d[0].label == 0 && std::fabs(d[0].score - 0.880797f) < 1e-5f, "best label and score");
        check(std::fabs(d[0].x0 - 60.f) < 1e-3f && std::fabs(d[0].y1 - 140.f) < 1e-3f, "pixels of the padded square");
        check(std::fabs(d[0].objectness - 0.5f) < 1e-6f, "objectness sigmoid");
    }
    o.nms_iou = 0.f;   // disabled
    d = Detector::decode(p, px, o);
    check(d.size() == 2 && d[1].label == 1, "without NMS both boxes stay, sorted");
    o.threshold = 0.f;
    d = Detector::decode(p, px, o);
    check(d.size() == 3 && d[2].y1 == 200.f, "boxes clip to the image");
    o.max_detections = 1;
    check(Detector::decode(p, px, o).size() == 1, "max_detections");
}

}  // namespace

int main() {
    test_preprocess();
    test_decode();
    if (g_failures) {
        std::fprintf(stderr, "owlv2: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("owlv2 OK\n");
    return 0;
}
