// Sam2Tracker: the reference SAM2VideoPredictor's single-object bookkeeping
// (add_new_points_or_box / propagate_in_video_preflight / propagate_in_video /
// _prepare_memory_conditioned_features) over frames the caller supplies.
//
// Where Hugging Face's port and facebookresearch/sam2 differ, this follows the
// original: a correction on a tracked frame is stored as a conditioning frame
// (add_all_frames_to_correct_as_cond) and feeds the frame's previous mask
// logits back in as the dense prompt.

#include "brovisionml/sam2.h"

#include "sam2_ops.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace brovisionml {

namespace bt = brotensor;
using bt::Tensor;

namespace {
constexpr float kNoObjScore = -1024.0f;
}

Sam2Tracker::Sam2Tracker(const Sam2& model) : model_(&model) {}

void Sam2Tracker::check(int frame) const {
    if (frame < 0 || frame >= num_frames())
        throw std::runtime_error("Sam2Tracker: frame " + std::to_string(frame) + " out of range");
}

int Sam2Tracker::add_frame(const uint8_t* pixels, int w, int h, int channels) {
    Frame f;
    f.w = w;
    f.h = h;
    f.feats = model_->encode(pixels, w, h, channels);
    frames_.push_back(std::move(f));
    return num_frames() - 1;
}

void Sam2Tracker::reset() {
    for (Frame& f : frames_) {
        f.has_output = f.cond = f.tracked = f.reverse = false;
        f.low_res = Tensor();
        f.obj_ptr = Tensor();
        f.memory = Tensor();
        f.iou = f.object_score = 0.0f;
    }
}

Sam2Tracker::Step Sam2Tracker::run_frame(int frame, bool is_init,
                                         const std::vector<std::array<float, 2>>& points,
                                         const std::vector<int>& labels,
                                         const Tensor* mask_input, bool reverse) const {
    const sam2::Sam2Config& cfg = model_->config();
    const Frame& fr = frames_[static_cast<std::size_t>(frame)];
    const int N = num_frames();

    // ── Condition the frame's features ──
    Tensor image;
    if (is_init) {
        image = model_->memory().no_memory(fr.feats.top);
    } else {
        std::vector<sam2::MemoryModule::Memory> mems;
        std::vector<sam2::MemoryModule::Pointer> ptrs;
        const int nmm = cfg.num_maskmem;
        // Every conditioning frame (max_cond_frames_in_attn = -1), then the
        // nmm-1 previous frames in tracking order, nearest last.
        for (int t = 0; t < N; ++t) {
            const Frame& c = frames_[static_cast<std::size_t>(t)];
            if (c.has_output && c.cond) {
                if (!c.memory.data) throw std::logic_error("Sam2Tracker: unencoded conditioning frame");
                mems.push_back({&c.memory, nmm - 1});
            }
        }
        for (int t_pos = 1; t_pos < nmm; ++t_pos) {
            const int t_rel = nmm - t_pos;
            const int prev = reverse ? frame + t_rel : frame - t_rel;
            if (prev < 0 || prev >= N) continue;
            const Frame& p = frames_[static_cast<std::size_t>(prev)];
            if (p.has_output && !p.cond) mems.push_back({&p.memory, nmm - t_pos - 1});
        }
        // Object pointers: the conditioning frames not ahead of this one, then
        // the previous non-conditioning frames up to max_obj_ptrs - 1 back.
        const int max_ptrs = std::min(N, cfg.max_obj_ptrs);
        const float t_diff_max = static_cast<float>(std::max(max_ptrs - 1, 1));
        const int sign = reverse ? -1 : 1;
        for (int t = 0; t < N; ++t) {
            const Frame& c = frames_[static_cast<std::size_t>(t)];
            if (!(c.has_output && c.cond)) continue;
            if (reverse ? t < frame : t > frame) continue;
            ptrs.push_back({&c.obj_ptr, static_cast<float>((frame - t) * sign) / t_diff_max});
        }
        for (int t_diff = 1; t_diff < max_ptrs; ++t_diff) {
            const int t = reverse ? frame + t_diff : frame - t_diff;
            if (t < 0 || t >= N) break;
            const Frame& p = frames_[static_cast<std::size_t>(t)];
            if (p.has_output && !p.cond)
                ptrs.push_back({&p.obj_ptr, static_cast<float>(t_diff) / t_diff_max});
        }
        image = model_->memory().attend(fr.feats.top, model_->encoder().top_pos(), mems, ptrs);
    }

    // ── SAM heads ──
    const int num_pts = static_cast<int>(points.size());
    const bool multimask = cfg.multimask_output_in_sam &&
                           (is_init || cfg.multimask_output_for_tracking) &&
                           num_pts >= cfg.multimask_min_pt_num && num_pts <= cfg.multimask_max_pt_num;
    sam2::HeadsPrompt hp;
    if (points.empty()) {
        hp.points = {{0.0f, 0.0f}};
        hp.labels = {-1};
    } else {
        hp.points = points;
        hp.labels = labels;
    }
    hp.mask_input = mask_input;
    sam2::HeadsOutput o = model_->heads().decode(image, fr.feats.feat_s0, fr.feats.feat_s1, hp, multimask);

    int pick = 0;
    for (int i = 1; i < o.num; ++i)
        if (o.iou[static_cast<std::size_t>(i)] > o.iou[static_cast<std::size_t>(pick)]) pick = i;
    const int ms = cfg.mask_size();
    const bool appearing = o.object_score > 0.0f;
    Step s;
    s.iou = o.iou[static_cast<std::size_t>(pick)];
    s.object_score = o.object_score;
    if (appearing) {
        s.low_res = sam2::detail::view_rows(o.low_res, static_cast<long long>(pick) * ms * ms, 1, ms * ms).clone();
    } else {
        s.low_res = Tensor::zeros_on(o.low_res.device, 1, ms * ms);
        bt::add_scalar_inplace(s.low_res, kNoObjScore);
    }
    const int tok = o.sam_tokens.rows > 1 ? pick : 0;
    Tensor token = sam2::detail::view_rows(o.sam_tokens, static_cast<long long>(tok) * cfg.hidden_size,
                                           1, cfg.hidden_size);
    s.obj_ptr = model_->memory().object_pointer(token, appearing);
    return s;
}

void Sam2Tracker::preflight() {
    // Conditioning frames get their memory from the (binarized) prompted mask.
    for (Frame& f : frames_) {
        if (f.has_output && f.cond && !f.memory.data)
            f.memory = model_->memory().encode(f.feats.top, f.low_res, /*from_points=*/true,
                                               f.object_score);
    }
}

Sam2FrameResult Sam2Tracker::prompt(int frame, const Sam2Prompt& prompt) {
    check(frame);
    Frame& f = frames_[static_cast<std::size_t>(frame)];
    std::vector<std::array<float, 2>> pts;
    std::vector<int> labels;
    model_->map_prompt(prompt, f.w, f.h, pts, labels);
    if (pts.empty()) throw std::runtime_error("Sam2Tracker: prompt needs points or a box");

    const bool is_init = !f.tracked;
    const bool reverse = is_init ? false : f.reverse;
    // A conditioning frame other than this one must already be encoded when
    // this one is memory-conditioned.
    if (!is_init) preflight();
    Tensor prev;
    if (f.has_output) {
        prev = f.low_res.clone();
        bt::clamp(prev, -32.0f, 32.0f);
    }
    Step s = run_frame(frame, is_init, pts, labels, prev.data ? &prev : nullptr, reverse);
    f.has_output = true;
    f.cond = true;
    f.low_res = std::move(s.low_res);
    f.obj_ptr = std::move(s.obj_ptr);
    f.iou = s.iou;
    f.object_score = s.object_score;
    f.memory = Tensor();   // re-encoded from the new mask at the next track
    return make_result(frame);
}

Sam2FrameResult Sam2Tracker::track(int frame, bool reverse) {
    check(frame);
    bool any_cond = false;
    for (const Frame& f : frames_) any_cond = any_cond || (f.has_output && f.cond);
    if (!any_cond) throw std::runtime_error("Sam2Tracker: track() before any prompt()");
    preflight();
    Frame& f = frames_[static_cast<std::size_t>(frame)];
    if (!(f.has_output && f.cond)) {
        Step s = run_frame(frame, /*is_init=*/false, {}, {}, nullptr, reverse);
        f.memory = model_->memory().encode(f.feats.top, s.low_res, /*from_points=*/false,
                                           s.object_score);
        f.has_output = true;
        f.cond = false;
        f.low_res = std::move(s.low_res);
        f.obj_ptr = std::move(s.obj_ptr);
        f.iou = s.iou;
        f.object_score = s.object_score;
    }
    f.tracked = true;
    f.reverse = reverse;
    return make_result(frame);
}

std::vector<Sam2FrameResult> Sam2Tracker::propagate() {
    int start = -1;
    for (int t = 0; t < num_frames(); ++t)
        if (frames_[static_cast<std::size_t>(t)].has_output && frames_[static_cast<std::size_t>(t)].cond) {
            start = t;
            break;
        }
    if (start < 0) throw std::runtime_error("Sam2Tracker: propagate() before any prompt()");
    for (int t = start; t < num_frames(); ++t) track(t, false);
    if (start > 0)
        for (int t = start; t >= 0; --t) track(t, true);
    std::vector<Sam2FrameResult> out;
    for (int t = 0; t < num_frames(); ++t) out.push_back(make_result(t));
    return out;
}

Sam2FrameResult Sam2Tracker::result(int frame) const {
    check(frame);
    return make_result(frame);
}

Sam2FrameResult Sam2Tracker::make_result(int frame) const {
    const Frame& f = frames_[static_cast<std::size_t>(frame)];
    Sam2FrameResult r;
    r.frame = frame;
    r.width = f.w;
    r.height = f.h;
    if (!f.has_output) return r;
    r.iou = f.iou;
    r.object_score = f.object_score;
    r.present = f.object_score > 0.0f;
    r.conditioning = f.cond;
    r.logits = model_->upscale(f.low_res, 1, f.w, f.h);
    r.has_box = sam2_mask_box(r.logits.data(), f.w, f.h, r.box);
    return r;
}

}  // namespace brovisionml
