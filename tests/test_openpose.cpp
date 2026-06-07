// OpenPose body-pose OpenposeDetector: real-weights END-TO-END parity test.
//
// Weights-gated (skips cleanly when weights/openpose/model.safetensors or
// golden_openpose.bin is absent). Reads the stored detect-res RGB bytes, runs
// the body network, and validates two gates:
//
//   Gate 1 (the tight neural gate): infer_maps() raw stage-6 PAF + heatmap
//     (network resolution) vs the golden raw arrays — pure conv parity like the
//     other annotators. The PAF/heatmap logits are O(1), so the bar is
//     mean-abs < 1e-3 and max-abs < 5e-2.
//
//   Gate 2 (end-to-end): detect() recovered people + keypoints vs golden. The
//     host postproc uses Lanczos3 (broimage has no Lanczos4) and a hand-rolled
//     scipy-'reflect' Gaussian, so keypoint pixel locations may drift a little.
//     Per the execution-not-research bar we match people count and require
//     keypoint recall@6px >= 0.8 over present keypoints for each matched person.
//     6px is ~1% of the 700px detect image — a tight pixel tolerance that still
//     absorbs the resampling/blur-kernel approximation; a real bug (BGR flip,
//     normalize, resize convention) would tank recall far below 0.8, not nudge
//     it. Actual recall + deltas are printed so the numbers are visible.
//
// A CUDA device, when present, repeats both gates and asserts CUDA-vs-CPU
// agreement (raw maps tiny mean-abs; identical people decode).

#define _CRT_SECURE_NO_WARNINGS

#include "brovisionml/openpose.h"

#include "brotensor/runtime.h"

#include "test_device.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifndef BROVISIONML_WEIGHTS_DIR
#define BROVISIONML_WEIGHTS_DIR ""
#endif

using brovisionml::openpose::BodyPose;
using brovisionml::openpose::Keypoint;
using brovisionml::openpose::OpenposeConfig;
using brovisionml::openpose::OpenposeDetector;
using brovisionml::openpose::PafHeatmap;
using brovisionml::openpose::PoseResult;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

namespace {

template <class T>
bool read_one(std::ifstream& f, T& x) {
    f.read(reinterpret_cast<char*>(&x), sizeof(T));
    return static_cast<bool>(f);
}
template <class T>
bool read_vec(std::ifstream& f, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(f);
}
bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

struct GKeypoint { float x, y, score; int present; };
struct GPerson { std::array<GKeypoint, 18> kp; float total_score; int total_parts; };

struct Golden {
    int W = 0, H = 0;
    std::vector<uint8_t> input;           // H*W*3 HWC RGB
    int Wp = 0, Hp = 0, pad_down = 0, pad_right = 0;
    float scale = 0;
    int paf_c = 0, nh = 0, nw = 0, hm_c = 0;
    std::vector<float> paf, heatmap;       // NCHW network-res
    std::vector<GPerson> people;
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 8) != "BVMLOPN1") return false;
    int version = 0;
    if (!read_one(f, version)) return false;
    if (!read_one(f, g.W) || !read_one(f, g.H)) return false;
    if (!read_vec(f, g.input, static_cast<std::size_t>(g.H) * g.W * 3)) return false;
    if (!read_one(f, g.Wp) || !read_one(f, g.Hp) ||
        !read_one(f, g.pad_down) || !read_one(f, g.pad_right)) return false;
    if (!read_one(f, g.scale)) return false;
    if (!read_one(f, g.paf_c) || !read_one(f, g.nh) || !read_one(f, g.nw)) return false;
    if (!read_vec(f, g.paf, static_cast<std::size_t>(g.paf_c) * g.nh * g.nw)) return false;
    if (!read_one(f, g.hm_c)) return false;
    if (!read_vec(f, g.heatmap, static_cast<std::size_t>(g.hm_c) * g.nh * g.nw)) return false;
    int n_people = 0;
    if (!read_one(f, n_people)) return false;
    g.people.resize(n_people);
    for (int i = 0; i < n_people; ++i) {
        for (int p = 0; p < 18; ++p) {
            GKeypoint k;
            if (!read_one(f, k.x) || !read_one(f, k.y) ||
                !read_one(f, k.score) || !read_one(f, k.present)) return false;
            g.people[i].kp[p] = k;
        }
        if (!read_one(f, g.people[i].total_score) ||
            !read_one(f, g.people[i].total_parts)) return false;
    }
    return true;
}

void map_diff(const std::vector<float>& a, const std::vector<float>& b,
              double& max_abs, double& mean_abs) {
    double m = 0.0, s = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        m = std::max(m, d);
        s += d;
    }
    max_abs = m;
    mean_abs = n ? s / (double)n : 0.0;
}

// Match each golden person to its nearest detected person (by mean keypoint
// distance over jointly-present parts) and report keypoint recall@tol over the
// golden person's present keypoints. Prints per-person recall.
double people_recall(const Golden& g, const PoseResult& r, float tol_px) {
    const float W = static_cast<float>(g.W), H = static_cast<float>(g.H);
    std::size_t total_present = 0, total_hit = 0;
    std::vector<bool> used(r.bodies.size(), false);

    for (const GPerson& gp : g.people) {
        // Find the best-matching detected body (most keypoints within tol).
        int best = -1, best_hit = -1;
        for (std::size_t b = 0; b < r.bodies.size(); ++b) {
            if (used[b]) continue;
            int hit = 0;
            for (int p = 0; p < 18; ++p) {
                if (!gp.kp[p].present || !r.bodies[b].keypoints[p].present) continue;
                const float dx = (gp.kp[p].x - r.bodies[b].keypoints[p].x) * W;
                const float dy = (gp.kp[p].y - r.bodies[b].keypoints[p].y) * H;
                if (std::sqrt(dx * dx + dy * dy) <= tol_px) ++hit;
            }
            if (hit > best_hit) { best_hit = hit; best = static_cast<int>(b); }
        }
        int present = 0;
        for (int p = 0; p < 18; ++p) if (gp.kp[p].present) ++present;
        total_present += present;
        if (best >= 0) {
            used[best] = true;
            total_hit += best_hit;
            std::printf("    matched golden person (parts=%d) -> detected #%d  "
                        "recall=%d/%d\n", present, best, best_hit, present);
        } else {
            std::printf("    golden person (parts=%d) UNMATCHED\n", present);
        }
    }
    return total_present ? (double)total_hit / (double)total_present : 1.0;
}

void run_gates(OpenposeDetector& det, const Golden& g, const char* tag,
               PafHeatmap& maps_out) {
    PafHeatmap maps = det.infer_maps(g.input.data(), g.W, g.H, 3);
    std::printf("  [%s] raw maps: paf %dx%dx%d  hm %dx%dx%d (golden %dx%d)\n",
                tag, maps.paf_c, maps.height, maps.width,
                maps.hm_c, maps.height, maps.width, g.nh, g.nw);
    CHECK(maps.height == g.nh && maps.width == g.nw);

    if (maps.paf.size() == g.paf.size() && maps.heatmap.size() == g.heatmap.size()) {
        double pmx = 0, pmn = 0, hmx = 0, hmn = 0;
        map_diff(maps.paf, g.paf, pmx, pmn);
        map_diff(maps.heatmap, g.heatmap, hmx, hmn);
        std::printf("    [%s] PAF vs golden: max-abs=%.3e mean-abs=%.3e\n", tag, pmx, pmn);
        std::printf("    [%s] HM  vs golden: max-abs=%.3e mean-abs=%.3e\n", tag, hmx, hmn);
        CHECK(pmn < 1e-3); CHECK(pmx < 5e-2);
        CHECK(hmn < 1e-3); CHECK(hmx < 5e-2);
    } else {
        CHECK(false);
    }

    PoseResult pose = det.detect(g.input.data(), g.W, g.H, 3);
    std::printf("    [%s] detected %zu people (golden %zu) at %dx%d\n",
                tag, pose.bodies.size(), g.people.size(), pose.width, pose.height);
    CHECK(pose.bodies.size() == g.people.size());
    const double recall = people_recall(g, pose, /*tol_px=*/6.0f);
    std::printf("    [%s] keypoint recall@6px = %.3f\n", tag, recall);
    CHECK(recall >= 0.8);

    maps_out = std::move(maps);
}

}  // namespace

int main() {
    std::printf("test_openpose:\n");

    const char* env = std::getenv("BROVISIONML_WEIGHTS_DIR");
    const std::string base = (env && *env) ? env : BROVISIONML_WEIGHTS_DIR;
    const std::string dir = base + "/openpose";
    const std::string ckpt = dir + "/model.safetensors";
    const std::string gpath = dir + "/golden_openpose.bin";

    if (!file_exists(ckpt) || !file_exists(gpath)) {
        std::printf("  no checkpoint/golden under '%s' — skipping "
                    "(weights-gated).\n", dir.c_str());
        return 0;
    }

    Golden g;
    if (!load_golden(gpath, g)) {
        std::printf("  FAIL  could not load/parse golden %s\n", gpath.c_str());
        return 1;
    }
    std::printf("  golden: in %dx%d  net %dx%d  people=%zu\n",
                g.W, g.H, g.nh, g.nw, g.people.size());

    try {
        OpenposeConfig cfg;
        OpenposeDetector det(cfg);
        det.load(dir);

        PafHeatmap cpu_maps;
        run_gates(det, g, "CPU", cpu_maps);

        brotensor::init();
        const brotensor::Device gpu = brovisionml_test::preferred_gpu();
        if (gpu != brotensor::Device::CPU) {
            const char* dev = brovisionml_test::device_name(gpu);
            det.to(gpu);
            CHECK(det.device() == gpu);
            PafHeatmap gpu_maps;
            run_gates(det, g, dev, gpu_maps);
            // GPU-vs-CPU agreement on the raw maps.
            double pmx = 0, pmn = 0, hmx = 0, hmn = 0;
            map_diff(gpu_maps.paf, cpu_maps.paf, pmx, pmn);
            map_diff(gpu_maps.heatmap, cpu_maps.heatmap, hmx, hmn);
            std::printf("    %s vs CPU: PAF max-abs=%.3e mean-abs=%.3e  "
                        "HM max-abs=%.3e mean-abs=%.3e\n", dev, pmx, pmn, hmx, hmn);
            CHECK(pmn < 1e-3); CHECK(hmn < 1e-3);
        } else {
            std::printf("  (no GPU device available — on-device check skipped)\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  error: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) {
        std::printf("  OK  openpose parity checks passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
