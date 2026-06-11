#include "brovisionml/sam_amg.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace brovisionml::sam {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("sam::AutomaticMaskGenerator: " + msg);
}

// ── Point grid ──────────────────────────────────────────────────────────────
// segment_anything build_point_grid: a regular n×n grid of points in [0,1]^2,
// each cell sampled at its center. Row-major (y outer, x inner).
std::vector<std::array<float, 2>> build_point_grid(int n) {
    std::vector<std::array<float, 2>> pts;
    if (n <= 0) return pts;
    pts.reserve(static_cast<std::size_t>(n) * n);
    const float offset = 1.0f / (2.0f * static_cast<float>(n));
    auto coord = [&](int i) {
        // np.linspace(offset, 1 - offset, n)
        if (n == 1) return offset;
        return offset + (1.0f - 2.0f * offset) * static_cast<float>(i) /
                            static_cast<float>(n - 1);
    };
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            pts.push_back({coord(x), coord(y)});
    return pts;
}

// ── Crop boxes ──────────────────────────────────────────────────────────────
// segment_anything generate_crop_boxes. Boxes are XYXY in image coords; the
// first is always the whole image (layer 0). Returns boxes paired with layer.
struct CropBox {
    int x0, y0, x1, y1;
    int layer;
};

int crop_len(int orig_len, int n_crops, int overlap) {
    // ceil((overlap*(n_crops-1) + orig_len) / n_crops)
    const long long num =
        static_cast<long long>(overlap) * (n_crops - 1) + orig_len;
    return static_cast<int>((num + n_crops - 1) / n_crops);
}

std::vector<CropBox> generate_crop_boxes(int im_w, int im_h, int n_layers,
                                         float overlap_ratio) {
    std::vector<CropBox> boxes;
    boxes.push_back({0, 0, im_w, im_h, 0});
    const int short_side = std::min(im_w, im_h);

    for (int i_layer = 0; i_layer < n_layers; ++i_layer) {
        const int n_per_side = 1 << (i_layer + 1);  // 2^(layer+1)
        const int overlap = static_cast<int>(overlap_ratio *
                                              static_cast<float>(short_side) *
                                              (2.0f / static_cast<float>(n_per_side)));
        const int cw = crop_len(im_w, n_per_side, overlap);
        const int ch = crop_len(im_h, n_per_side, overlap);

        for (int xi = 0; xi < n_per_side; ++xi) {
            const int x0 = (cw - overlap) * xi;
            for (int yi = 0; yi < n_per_side; ++yi) {
                const int y0 = (ch - overlap) * yi;
                boxes.push_back({x0, y0, std::min(x0 + cw, im_w),
                                 std::min(y0 + ch, im_h), i_layer + 1});
            }
        }
    }
    return boxes;
}

// ── Run-length encoding (compact candidate storage) ───────────────────────────
// Row-major alternating runs starting with a (possibly empty) background run,
// so we can hold thousands of pre-NMS candidate masks without materializing a
// full-resolution buffer per candidate.
struct Rle {
    std::vector<int> counts;  // counts[0]=bg run, [1]=fg run, alternating
    int n = 0;                // total pixels encoded
};

Rle rle_encode(const std::vector<uint8_t>& mask) {
    Rle r;
    r.n = static_cast<int>(mask.size());
    uint8_t last = 0;
    int run = 0;
    for (uint8_t v : mask) {
        if (v == last) {
            ++run;
        } else {
            r.counts.push_back(run);
            run = 1;
            last = v;
        }
    }
    r.counts.push_back(run);
    return r;
}

void rle_decode(const Rle& r, std::vector<uint8_t>& out) {
    out.assign(static_cast<std::size_t>(r.n), 0);
    std::size_t i = 0;
    uint8_t v = 0;
    for (int c : r.counts) {
        if (v) {
            for (int k = 0; k < c; ++k) out[i + static_cast<std::size_t>(k)] = 1;
        }
        i += static_cast<std::size_t>(c);
        v = static_cast<uint8_t>(1 - v);
    }
}

long long rle_area(const Rle& r) {
    long long a = 0;
    for (std::size_t i = 1; i < r.counts.size(); i += 2) a += r.counts[i];
    return a;
}

// ── Geometry helpers ──────────────────────────────────────────────────────────
using Box = std::array<int, 4>;  // XYXY, inclusive pixel extents

// Tight box of a binary mask (XYXY inclusive). Empty mask -> {0,0,0,0}.
Box mask_to_box(const std::vector<uint8_t>& mask, int w, int h) {
    int minx = w, miny = h, maxx = -1, maxy = -1;
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = mask.data() + static_cast<std::size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            if (row[x]) {
                if (x < minx) minx = x;
                if (x > maxx) maxx = x;
                if (y < miny) miny = y;
                if (y > maxy) maxy = y;
            }
        }
    }
    if (maxx < 0) return {0, 0, 0, 0};
    return {minx, miny, maxx, maxy};
}

float box_iou(const Box& a, const Box& b) {
    const float aw = static_cast<float>(a[2] - a[0]);
    const float ah = static_cast<float>(a[3] - a[1]);
    const float bw = static_cast<float>(b[2] - b[0]);
    const float bh = static_cast<float>(b[3] - b[1]);
    const float area_a = std::max(0.0f, aw) * std::max(0.0f, ah);
    const float area_b = std::max(0.0f, bw) * std::max(0.0f, bh);
    const float ix = static_cast<float>(std::min(a[2], b[2]) - std::max(a[0], b[0]));
    const float iy = static_cast<float>(std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const float inter = std::max(0.0f, ix) * std::max(0.0f, iy);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// Greedy NMS over XYXY boxes by descending score (ties broken by lower index,
// matching a stable score sort). Returns kept indices in descending-score order.
std::vector<int> nms(const std::vector<Box>& boxes,
                     const std::vector<float>& scores, float thresh) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int i, int j) { return scores[i] > scores[j]; });
    std::vector<int> keep;
    for (int i : order) {
        bool suppressed = false;
        for (int j : keep) {
            if (box_iou(boxes[i], boxes[j]) > thresh) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) keep.push_back(i);
    }
    return keep;
}

// ── Connected components (8-connectivity), cv2.connectedComponentsWithStats ────
// Labels the non-zero pixels of `img`; label 0 is the zero-pixel background.
// Returns per-pixel labels and per-label pixel counts (sizes[0] = background).
struct CC {
    std::vector<int> labels;  // w*h, 0 = background
    std::vector<int> sizes;   // index by label; sizes[0] = background area
    int n_labels = 0;         // including background
};

CC connected_components(const std::vector<uint8_t>& img, int w, int h) {
    CC cc;
    cc.labels.assign(static_cast<std::size_t>(w) * h, -1);
    long long bg = 0;
    int next = 1;
    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    std::queue<int> q;  // flat indices
    for (int s = 0; s < w * h; ++s) {
        if (img[static_cast<std::size_t>(s)] == 0) {
            cc.labels[static_cast<std::size_t>(s)] = 0;
            ++bg;
            continue;
        }
        if (cc.labels[static_cast<std::size_t>(s)] != -1) continue;
        const int label = next++;
        int count = 0;
        cc.labels[static_cast<std::size_t>(s)] = label;
        q.push(s);
        while (!q.empty()) {
            const int p = q.front();
            q.pop();
            ++count;
            const int px = p % w, py = p / w;
            for (int k = 0; k < 8; ++k) {
                const int nx = px + dx[k], ny = py + dy[k];
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                const int np = ny * w + nx;
                if (img[static_cast<std::size_t>(np)] &&
                    cc.labels[static_cast<std::size_t>(np)] == -1) {
                    cc.labels[static_cast<std::size_t>(np)] = label;
                    q.push(np);
                }
            }
        }
        cc.sizes.push_back(count);
    }
    cc.n_labels = next;  // labels 0..next-1
    cc.sizes.insert(cc.sizes.begin(), static_cast<int>(bg));  // sizes[0] = background
    return cc;
}

// segment_anything remove_small_regions. mode "holes" fills background regions
// smaller than `area`; mode "islands" drops foreground regions smaller than
// `area`. Returns true if the mask changed.
bool remove_small_regions(std::vector<uint8_t>& mask, int w, int h, int area,
                          bool holes) {
    std::vector<uint8_t> working(mask.size());
    for (std::size_t i = 0; i < mask.size(); ++i)
        working[i] = static_cast<uint8_t>((holes ? 1 : 0) ^ mask[i]);

    CC cc = connected_components(working, w, h);

    // Foreground (non-background) regions of `working` smaller than `area`.
    std::vector<char> fill(static_cast<std::size_t>(cc.n_labels), 0);
    std::vector<int> small;
    for (int lbl = 1; lbl < cc.n_labels; ++lbl)
        if (cc.sizes[static_cast<std::size_t>(lbl)] < area) small.push_back(lbl);
    if (small.empty()) return false;

    if (holes) {
        // Keep working's background (label 0 == original foreground) + small holes.
        fill[0] = 1;
        for (int lbl : small) fill[static_cast<std::size_t>(lbl)] = 1;
    } else {
        // Keep the large foreground regions (everything but bg and small islands).
        std::vector<char> is_small(static_cast<std::size_t>(cc.n_labels), 0);
        for (int lbl : small) is_small[static_cast<std::size_t>(lbl)] = 1;
        int kept = 0;
        for (int lbl = 1; lbl < cc.n_labels; ++lbl) {
            if (!is_small[static_cast<std::size_t>(lbl)]) {
                fill[static_cast<std::size_t>(lbl)] = 1;
                ++kept;
            }
        }
        if (kept == 0) {
            // Every region below threshold: keep the largest one.
            int best = 1, best_sz = -1;
            for (int lbl = 1; lbl < cc.n_labels; ++lbl)
                if (cc.sizes[static_cast<std::size_t>(lbl)] > best_sz) {
                    best_sz = cc.sizes[static_cast<std::size_t>(lbl)];
                    best = lbl;
                }
            fill[static_cast<std::size_t>(best)] = 1;
        }
    }

    for (std::size_t i = 0; i < mask.size(); ++i)
        mask[i] = fill[static_cast<std::size_t>(cc.labels[i])];
    return true;
}

// Stability score: fraction of the high-threshold mask area retained at the
// low threshold, i.e. how stable the binarization is to a ±offset logit nudge.
float stability_score(const float* logits, int n, float offset) {
    long long inter = 0, uni = 0;
    for (int i = 0; i < n; ++i) {
        if (logits[i] > offset) ++inter;
        if (logits[i] > -offset) ++uni;
    }
    return uni > 0 ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

// is_box_near_crop_edge: box (image-frame XYXY) touches a crop edge (within
// atol) that is not also the image edge.
bool near_crop_edge(const Box& box, const CropBox& crop, int im_w, int im_h) {
    const float atol = 20.0f;
    auto close = [&](int a, int b) {
        return std::fabs(static_cast<float>(a - b)) <= atol;
    };
    const int crop_e[4] = {crop.x0, crop.y0, crop.x1, crop.y1};
    const int img_e[4]  = {0, 0, im_w, im_h};
    for (int i = 0; i < 4; ++i)
        if (close(box[static_cast<std::size_t>(i)], crop_e[i]) &&
            !close(box[static_cast<std::size_t>(i)], img_e[i]))
            return true;
    return false;
}

// One pre-NMS candidate mask, stored compactly in crop-frame RLE.
struct Candidate {
    Rle  rle;                 // crop-frame mask (cw*ch)
    int  cx0, cy0, cw, ch;    // crop placement + size, for uncropping
    Box  box;                 // image-frame XYXY
    std::array<float, 2> point;  // image-frame originating point
    float iou, stab;
    CropBox crop;
};

}  // namespace

// ── AutomaticMaskGenerator ────────────────────────────────────────────────────

AutomaticMaskGenerator::AutomaticMaskGenerator(Sam& model, AmgConfig cfg)
    : model_(model), cfg_(cfg) {
    if (cfg_.points_per_side <= 0) fail("points_per_side must be positive");
    if (cfg_.points_per_batch <= 0) fail("points_per_batch must be positive");
    if (cfg_.crop_n_layers < 0) fail("crop_n_layers must be >= 0");
    if (cfg_.crop_n_points_downscale_factor < 1)
        fail("crop_n_points_downscale_factor must be >= 1");
}

std::vector<GeneratedMask> AutomaticMaskGenerator::generate(const uint8_t* pixels,
                                                            int w, int h,
                                                            int channels) {
    if (!pixels || w <= 0 || h <= 0) fail("empty image");
    if (channels != 1 && channels != 3 && channels != 4)
        fail("channels must be 1, 3, or 4");

    const std::vector<CropBox> crops =
        generate_crop_boxes(w, h, cfg_.crop_n_layers, cfg_.crop_overlap_ratio);

    std::vector<Candidate> kept;  // survivors of within-crop NMS, all crops

    std::vector<uint8_t> crop_buf;  // reused scratch for non-trivial crops
    std::vector<uint8_t> bin;       // reused scratch for a crop-frame binary mask

    for (const CropBox& crop : crops) {
        const int cw = crop.x1 - crop.x0;
        const int ch = crop.y1 - crop.y0;
        if (cw <= 0 || ch <= 0) continue;

        // Set the image (the whole crop, or the original buffer for layer 0).
        if (crop.x0 == 0 && crop.y0 == 0 && cw == w && ch == h) {
            model_.set_image(pixels, w, h, channels);
        } else {
            crop_buf.assign(static_cast<std::size_t>(cw) * ch * channels, 0);
            for (int y = 0; y < ch; ++y) {
                const uint8_t* src =
                    pixels + (static_cast<std::size_t>(crop.y0 + y) * w + crop.x0) *
                                 channels;
                uint8_t* dst =
                    crop_buf.data() + static_cast<std::size_t>(y) * cw * channels;
                std::copy(src, src + static_cast<std::size_t>(cw) * channels, dst);
            }
            model_.set_image(crop_buf.data(), cw, ch, channels);
        }

        // Coarser grid per crop layer.
        int n = cfg_.points_per_side;
        for (int i = 0; i < crop.layer; ++i)
            n /= cfg_.crop_n_points_downscale_factor;
        if (n < 1) n = 1;
        const std::vector<std::array<float, 2>> grid = build_point_grid(n);

        std::vector<Candidate> crop_cands;

        // Grid points in crop-pixel coords; decoded in independent batches.
        std::vector<std::array<float, 2>> pts;
        pts.reserve(grid.size());
        for (const auto& g : grid)
            pts.push_back({g[0] * static_cast<float>(cw),
                           g[1] * static_cast<float>(ch)});

        const int bs = std::max(1, cfg_.points_per_batch);
        for (std::size_t off = 0; off < pts.size(); off += static_cast<std::size_t>(bs)) {
            const std::size_t end =
                std::min(pts.size(), off + static_cast<std::size_t>(bs));
            std::vector<std::array<float, 2>> chunk(pts.begin() + static_cast<std::ptrdiff_t>(off),
                                                    pts.begin() + static_cast<std::ptrdiff_t>(end));

            // The predicted-IoU threshold is applied inside segment_points,
            // before the full-resolution upscale — rejected masks are never
            // upscaled or downloaded. The per-mask check below then only sees
            // survivors.
            const float min_iou =
                cfg_.pred_iou_thresh > 0.0f ? cfg_.pred_iou_thresh : -1.0f;
            const std::vector<Segmentation> segs =
                model_.segment_points(chunk, /*multimask=*/true, min_iou);

            for (std::size_t si = 0; si < segs.size(); ++si) {
                const Segmentation& seg = segs[si];
                const float px = chunk[si][0], py = chunk[si][1];
                const int plane = seg.height * seg.width;  // == ch*cw

                for (int m = 0; m < seg.num; ++m) {
                    const float iou = seg.iou[static_cast<std::size_t>(m)];
                    if (cfg_.pred_iou_thresh > 0.0f && iou <= cfg_.pred_iou_thresh)
                        continue;

                    const float* logit =
                        seg.logits.data() + static_cast<std::size_t>(m) * plane;
                    const float stab =
                        stability_score(logit, plane, cfg_.stability_score_offset);
                    if (cfg_.stability_score_thresh > 0.0f &&
                        stab < cfg_.stability_score_thresh)
                        continue;

                    bin.assign(static_cast<std::size_t>(plane), 0);
                    for (int i = 0; i < plane; ++i)
                        bin[static_cast<std::size_t>(i)] = logit[i] > 0.0f ? 1 : 0;

                    const Box cbox = mask_to_box(bin, cw, ch);
                    const Box ibox = {cbox[0] + crop.x0, cbox[1] + crop.y0,
                                      cbox[2] + crop.x0, cbox[3] + crop.y0};
                    if (near_crop_edge(ibox, crop, w, h)) continue;

                    Candidate cand;
                    cand.rle = rle_encode(bin);
                    cand.cx0 = crop.x0;
                    cand.cy0 = crop.y0;
                    cand.cw = cw;
                    cand.ch = ch;
                    cand.box = ibox;
                    cand.point = {px + static_cast<float>(crop.x0),
                                  py + static_cast<float>(crop.y0)};
                    cand.iou = iou;
                    cand.stab = stab;
                    cand.crop = crop;
                    crop_cands.push_back(std::move(cand));
                }
            }
        }

        // Within-crop NMS by predicted IoU.
        std::vector<Box> boxes;
        std::vector<float> scores;
        boxes.reserve(crop_cands.size());
        scores.reserve(crop_cands.size());
        for (const auto& c : crop_cands) {
            boxes.push_back(c.box);
            scores.push_back(c.iou);
        }
        for (int idx : nms(boxes, scores, cfg_.box_nms_thresh))
            kept.push_back(std::move(crop_cands[static_cast<std::size_t>(idx)]));
    }

    // Across-crop NMS: prefer masks from smaller (finer) crops via 1/crop-area.
    if (crops.size() > 1 && !kept.empty()) {
        std::vector<Box> boxes;
        std::vector<float> scores;
        boxes.reserve(kept.size());
        scores.reserve(kept.size());
        for (const auto& c : kept) {
            boxes.push_back(c.box);
            const float carea =
                static_cast<float>(c.crop.x1 - c.crop.x0) *
                static_cast<float>(c.crop.y1 - c.crop.y0);
            scores.push_back(carea > 0.0f ? 1.0f / carea : 0.0f);
        }
        std::vector<Candidate> filtered;
        for (int idx : nms(boxes, scores, cfg_.crop_nms_thresh))
            filtered.push_back(std::move(kept[static_cast<std::size_t>(idx)]));
        kept = std::move(filtered);
    }

    // Materialize full-resolution masks for the survivors.
    std::vector<GeneratedMask> out;
    out.reserve(kept.size());
    std::vector<uint8_t> cm;  // reused crop-frame decode scratch
    for (const auto& c : kept) {
        GeneratedMask gm;
        gm.height = h;
        gm.width = w;
        gm.mask.assign(static_cast<std::size_t>(w) * h, 0);
        rle_decode(c.rle, cm);
        for (int y = 0; y < c.ch; ++y) {
            const uint8_t* src = cm.data() + static_cast<std::size_t>(y) * c.cw;
            uint8_t* dst =
                gm.mask.data() + (static_cast<std::size_t>(c.cy0 + y) * w + c.cx0);
            std::copy(src, src + c.cw, dst);
        }
        gm.predicted_iou = c.iou;
        gm.stability_score = c.stab;
        gm.point = c.point;
        gm.crop_box = {c.crop.x0, c.crop.y0, c.crop.x1 - c.crop.x0,
                       c.crop.y1 - c.crop.y0};
        out.push_back(std::move(gm));
    }

    // Optional small-region / hole cleanup, then re-NMS to drop the duplicates
    // the fix-up creates (preferring masks that were left unchanged).
    if (cfg_.min_mask_region_area > 0 && !out.empty()) {
        std::vector<Box> boxes(out.size());
        std::vector<float> scores(out.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            bool changed = remove_small_regions(out[i].mask, w, h,
                                                 cfg_.min_mask_region_area,
                                                 /*holes=*/true);
            changed |= remove_small_regions(out[i].mask, w, h,
                                            cfg_.min_mask_region_area,
                                            /*holes=*/false);
            boxes[i] = mask_to_box(out[i].mask, w, h);
            scores[i] = changed ? 0.0f : 1.0f;  // prefer unchanged masks
        }
        const float thresh = std::max(cfg_.box_nms_thresh, cfg_.crop_nms_thresh);
        std::vector<GeneratedMask> filtered;
        for (int idx : nms(boxes, scores, thresh))
            filtered.push_back(std::move(out[static_cast<std::size_t>(idx)]));
        out = std::move(filtered);
    }

    // Finalize per-mask box + area, then sort by descending area.
    for (auto& gm : out) {
        const Box b = mask_to_box(gm.mask, w, h);
        long long a = 0;
        for (uint8_t v : gm.mask) a += v;
        gm.area = a;
        gm.bbox = {b[0], b[1], b[2] - b[0] + 1, b[3] - b[1] + 1};
        if (a == 0) gm.bbox = {0, 0, 0, 0};
    }
    std::sort(out.begin(), out.end(),
              [](const GeneratedMask& a, const GeneratedMask& b) {
                  return a.area > b.area;
              });
    return out;
}

}  // namespace brovisionml::sam
