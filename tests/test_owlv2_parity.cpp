// OWLv2 parity against transformers' Owlv2ForObjectDetection
// (tests/ref/gen_owlv2_fixture.py -> tests/fixtures/owlv2.bin), on the CPU and
// on the default GPU when there is one.
//
// Weights: the HF release google/owlv2-base-patch16-ensemble as-is
// (model.safetensors, vocab.json, merges.txt), from BROVISIONML_OWLV2_DIR, else
// the brovisionml-data sibling (../brovisionml-data/owlv2). Skips (passes) when
// the weights or the fixture are absent.
//
// Stages: the tokenizer's ids; preprocessing (pad, anti-aliased resize,
// normalise) of a 1280 x 800 synthetic image; then, fed the reference pixels so
// each stage measures the model alone, the token embeddings, block 0, the last
// block, the image embeddings, the text embeddings, the class / box /
// objectness heads, and the decoded detections (best query, sigmoid, NMS at
// IoU 0.3, top 20, original-image pixels). The CPU is FP32 and held tight; a
// GPU runs the vision blocks in FP16 and is held to what half precision
// through 12 residual blocks allows.
#include "brovisionml/owlv2.h"
#include "test_device.h"

#include "brotensor/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace brovisionml::owlv2;

namespace {

int g_failures = 0;

struct Rec {
    int code = 0;
    std::vector<int> dims;
    std::vector<float> f;
    std::vector<int> i;
    std::string s;
};
using Fixture = std::map<std::string, Rec>;

Fixture read_fixture(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    Fixture out;
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "OWL2FIX1", 8) != 0) throw std::runtime_error("bad fixture magic");
    auto rd = [&](void* d, std::size_t n) {
        in.read(static_cast<char*>(d), static_cast<std::streamsize>(n));
        if (!in) throw std::runtime_error("truncated fixture");
    };
    int count = 0;
    rd(&count, 4);
    for (int r = 0; r < count; ++r) {
        int nl = 0;
        rd(&nl, 4);
        std::string name(static_cast<std::size_t>(nl), '\0');
        rd(name.data(), name.size());
        Rec rec;
        int nd = 0;
        rd(&rec.code, 4);
        rd(&nd, 4);
        rec.dims.resize(static_cast<std::size_t>(nd));
        rd(rec.dims.data(), 4u * static_cast<std::size_t>(nd));
        std::size_t n = 1;
        for (int d : rec.dims) n *= static_cast<std::size_t>(d);
        if (rec.code == 0) { rec.f.resize(n); rd(rec.f.data(), 4 * n); }
        else if (rec.code == 1) { rec.i.resize(n); rd(rec.i.data(), 4 * n); }
        else { rec.s.resize(n); rd(rec.s.data(), n); }
        out.emplace(name, std::move(rec));
    }
    return out;
}

const Rec& need(const Fixture& fx, std::string_view name) {
    auto it = fx.find(std::string(name));
    if (it == fx.end()) throw std::runtime_error("fixture has no record " + std::string(name));
    return it->second;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

// 99th percentile of |a - b| and max |b|: context for a max-abs bound.
std::string spread(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return "";
    std::vector<float> d(a.size());
    double peak = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d[i] = std::fabs(a[i] - b[i]);
        peak = std::max(peak, std::fabs(static_cast<double>(b[i])));
    }
    std::nth_element(d.begin(), d.begin() + d.size() * 99 / 100, d.end());
    char buf[64];
    std::snprintf(buf, sizeof buf, "p99 %.1e, max|ref| %.1e", d[d.size() * 99 / 100], peak);
    return buf;
}

void report(const char* what, double v, double tol, const std::string& note = "") {
    std::printf("    %-34s %.3e  (tol %.0e)  %s\n", what, v, tol, note.c_str());
    if (!(v <= tol)) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s = %g > %g\n", what, v, tol);
    }
}

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

struct Tol {
    double embeddings, block0, last_hidden, image_embeds, text, logits, boxes, objectness, scores;
    double det_box, det_score;
    int det_compared;   // leading detections compared one to one
};
constexpr Tol kCpuTol{2e-5, 5e-5, 1e-2, 3e-3, 2e-6, 1e-2, 2e-3, 1e-2, 1e-4, 0.05, 2e-5, 20};
constexpr Tol kGpuTol{5e-3, 1e-2, 4.0, 1.0, 2e-6, 3.0, 0.2, 2.5, 1e-3, 1.0, 2e-3, 5};

void run_device(const char* name, Detector& det, const Fixture& fx, const Tol& tol,
                const std::vector<float>& text_ref) {
    std::printf("  %s\n", name);
    const Rec& prgb = need(fx, "image/pixels");
    Pixels px;
    const Rec& rgb = need(fx, "image/rgb");
    px.height = rgb.dims[0];
    px.width = rgb.dims[1];
    px.padded = std::max(px.width, px.height);
    px.nchw = brotensor::Tensor::from_host(prgb.f.data(), 1, static_cast<int>(prgb.f.size()));

    const std::vector<std::string> queries = [&] {
        std::vector<std::string> q;
        std::stringstream ss(need(fx, "text/queries").s);
        std::string line;
        while (std::getline(ss, line)) q.push_back(line);
        return q;
    }();
    const std::vector<float> text = det.embed_text(queries);
    report("text embeddings", max_abs(text, text_ref), tol.text);

    VisionTrace tr;
    const Predictions pred = det.predict(px, text.data(), static_cast<int>(queries.size()), &tr);
    auto stage = [&](const char* what, const std::vector<float>& got, const char* rec, double t) {
        const std::vector<float>& ref = need(fx, rec).f;
        report(what, max_abs(got, ref), t, spread(got, ref));
    };
    stage("token embeddings", tr.embeddings, "vision/embeddings", tol.embeddings);
    stage("block 0", tr.block0, "vision/block0", tol.block0);
    stage("last block", tr.last_hidden, "vision/last_hidden", tol.last_hidden);
    stage("image embeddings", tr.image_embeds, "vision/image_embeds", tol.image_embeds);
    stage("class logits", pred.logits, "heads/logits", tol.logits);
    stage("boxes (cxcywh)", pred.boxes, "heads/boxes", tol.boxes);
    stage("objectness logits", pred.objectness, "heads/objectness", tol.objectness);
    // What a caller thresholds: the per-query scores, sigmoid of the logits.
    {
        auto sig = [](std::vector<float> v) {
            for (float& x : v) x = 1.0f / (1.0f + std::exp(-x));
            return v;
        };
        const std::vector<float> got = sig(pred.logits), ref = sig(need(fx, "heads/logits").f);
        report("scores (sigmoid of logits)", max_abs(got, ref), tol.scores, spread(got, ref));
    }

    // Detections: the leading ones one to one (later ranks hold near-ties a
    // GPU may reorder).
    const Rec& params = need(fx, "detect/params");
    DetectOptions o;
    o.threshold = 0.f;
    o.nms_iou = params.f[0];
    o.max_detections = static_cast<int>(params.f[1]);
    const std::vector<Detection> dets = Detector::decode(pred, px, o);
    const Rec& rb = need(fx, "detect/boxes");
    const Rec& rs = need(fx, "detect/scores");
    const Rec& rl = need(fx, "detect/labels");
    const int n = std::min<int>(tol.det_compared, static_cast<int>(rs.f.size()));
    check(static_cast<int>(dets.size()) >= n, std::string(name) + ": too few detections");
    double dbox = 0.0, dscore = 0.0;
    int label_miss = 0;
    for (int i = 0; i < n && i < static_cast<int>(dets.size()); ++i) {
        const Detection& d = dets[i];
        const float b[4] = {d.x0, d.y0, d.x1, d.y1};
        for (int c = 0; c < 4; ++c) dbox = std::max(dbox, std::fabs(double(b[c]) - rb.f[i * 4 + c]));
        dscore = std::max(dscore, std::fabs(double(d.score) - rs.f[i]));
        if (d.label != rl.i[i]) ++label_miss;
    }
    char what[64];
    std::snprintf(what, sizeof what, "top-%d detection boxes (px)", n);
    report(what, dbox, tol.det_box);
    std::snprintf(what, sizeof what, "top-%d detection scores", n);
    report(what, dscore, tol.det_score);
    check(label_miss == 0, std::string(name) + ": detection labels differ");
    std::printf("    best: %s %.4f [%.1f %.1f %.1f %.1f]\n", queries[dets[0].label].c_str(),
                dets[0].score, dets[0].x0, dets[0].y0, dets[0].x1, dets[0].y1);
}

}  // namespace

int main() {
    const char* env = std::getenv("BROVISIONML_OWLV2_DIR");
    const fs::path dir = env ? fs::path(env) : fs::path(BROVISIONML_DATA_DIR) / "owlv2";
    const fs::path fixture = fs::path(BROVISIONML_FIXTURE_DIR) / "owlv2.bin";
    if (!fs::exists(dir / "model.safetensors") || !fs::exists(fixture)) {
        std::printf("SKIP owlv2 parity: need %s and %s (tests/ref/gen_owlv2_fixture.py)\n",
                    (dir / "model.safetensors").string().c_str(), fixture.string().c_str());
        return 0;
    }
    brotensor::init();
    const Fixture fx = read_fixture(fixture);

    Detector det;
    det.load(dir.string());

    // Tokenizer.
    {
        std::stringstream ss(need(fx, "text/queries").s);
        std::string line;
        for (int i = 0; std::getline(ss, line); ++i) {
            const auto ids = det.tokenizer().encode(line);
            check(ids == need(fx, "text/ids/" + std::to_string(i)).i, "token ids of '" + line + "'");
        }
    }

    // Preprocessing.
    {
        const Rec& rgb = need(fx, "image/rgb");
        std::vector<uint8_t> bytes(rgb.i.begin(), rgb.i.end());
        const Pixels px = preprocess(bytes.data(), rgb.dims[1], rgb.dims[0], 3);
        std::printf("  preprocess (%dx%d -> pad %d -> 960)\n", px.width, px.height, px.padded);
        report("pixels", max_abs(px.nchw.to_host_vector(), need(fx, "image/pixels").f), 2e-4);
    }

    const std::vector<float>& text_ref = need(fx, "text/embeds").f;
    run_device("CPU (FP32)", det, fx, kCpuTol, text_ref);

    const brotensor::Device gpu = brovisionml_test::preferred_gpu();
    if (gpu != brotensor::Device::CPU) {
        brotensor::DeviceScope scope(gpu);
        det.to(gpu);
        std::string label = std::string(brovisionml_test::device_name(gpu)) + " (mixed FP16)";
        run_device(label.c_str(), det, fx, kGpuTol, text_ref);

        // Timing: the vision tower + heads for one image, queries cached.
        const Rec& rgb = need(fx, "image/rgb");
        std::vector<uint8_t> bytes(rgb.i.begin(), rgb.i.end());
        const std::vector<float> q = det.embed_text({"a bird", "a squirrel", "a deer"});
        const Pixels px = preprocess(bytes.data(), rgb.dims[1], rgb.dims[0], 3);
        det.predict(px, q.data(), 3);   // warm: pipelines, allocations
        using clk = std::chrono::steady_clock;
        const int reps = 5;
        auto t0 = clk::now();
        for (int i = 0; i < reps; ++i) det.predict(px, q.data(), 3);
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / reps;
        t0 = clk::now();
        for (int i = 0; i < reps; ++i) preprocess(bytes.data(), rgb.dims[1], rgb.dims[0], 3);
        const double pre = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / reps;
        std::printf("  %s: predict %.1f ms/image at 960, preprocess %.1f ms (1280x800)\n",
                    brovisionml_test::device_name(gpu), ms, pre);
    } else {
        std::printf("  no GPU backend: GPU parity skipped\n");
    }

    if (g_failures) {
        std::fprintf(stderr, "owlv2 parity: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("owlv2 parity OK\n");
    return 0;
}
