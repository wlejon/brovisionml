// Single-source Brass MIR GPU kernels for brovisionml:
//   - DPT bicubic resize + ImageNet normalize (uint8 HWC -> FP32 CHW)
//   - DSINE RayReLU (in-place 3-channel normal clamp against camera ray)
//   - DSINE AngMF 5x5 rotation-propagated normal refinement
//
// Each kernel is authored once in Brass MIR and lowered on demand to SPIR-V
// (Vulkan), PTX (CUDA), or MSL (Metal), then dispatched on brotensor's device
// stream via brotensor::interop.

#include "gpu_mir_ops.h"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/msl_target.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv_target.hpp>

#include <brotensor/kernel_interop.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brovisionml {

namespace {

namespace bt = ::brotensor;
using brass::BasicBlock;
using brass::Function;
using brass::Builder;
using brass::Module;
using brass::Type;
using brass::Value;
using brass::codegen::KernelBuilder;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("brovisionml gpu: " + msg);
}

void require_f32_gpu(const bt::Tensor& t, const char* what) {
    if (t.dtype != bt::Dtype::FP32) fail(std::string(what) + " must be FP32");
    if (!t.device.is_gpu()) fail(std::string(what) + " must be on a GPU device");
}

std::uint32_t groups_1d(std::uint64_t n) {
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>((n + 255u) / 256u, 1u, 65535u));
}

// ── Precision-polymorphic MIR math helpers (FP64 or FP32) ───────────────────

struct RealOps {
    KernelBuilder& kb;
    Builder& b;
    bool f64;

    Type type() const { return f64 ? Type::f64() : Type::f32(); }
    Value* k(double v) const {
        return f64 ? b.build_fconst_f64(v) : b.build_fconst_f32(static_cast<float>(v));
    }
    Value* i32(std::int32_t v) const { return b.build_iconst_i32(v); }
    Value* add(Value* x, Value* y) const { return b.build_add(x, y); }
    Value* sub(Value* x, Value* y) const { return b.build_sub(x, y); }
    Value* mul(Value* x, Value* y) const { return b.build_mul(x, y); }
    Value* div(Value* x, Value* y) const { return b.build_sdiv(x, y); }
    Value* neg(Value* x) const { return b.build_neg(x); }
    Value* fma(Value* x, Value* y, Value* z) const {
        return f64 ? b.build_fma_f64(x, y, z) : b.build_fma_f32(x, y, z);
    }
    Value* abs(Value* x) const { return f64 ? b.build_fabs_f64(x) : kb.fabs(x); }
    Value* sqrt(Value* x) const {
        return f64 ? b.build_sqrt_f64(x) : b.build_call("sqrtf", Type::f32(), {x});
    }
    Value* fmax(Value* x, Value* y) const {
        return f64 ? b.build_select(b.build_sgt(x, y), x, y) : kb.fmax(x, y);
    }
    Value* from_i32(Value* x) const {
        return f64 ? b.build_sitofp_f64_i32(x) : kb.i32_to_f32(x);
    }
    Value* load(Value* ptr, Value* idx64) const {
        Value* v = kb.load_f32_indexed(ptr, idx64);
        return f64 ? b.build_fpext_f64_f32(v) : v;
    }
    void store(Value* ptr, Value* idx64, Value* val) const {
        Value* v = f64 ? b.build_fptrunc_f32_f64(val) : val;
        kb.store_f32_indexed(ptr, idx64, v);
    }
    Value* is_finite(Value* v) const {
        const double max_v = f64 ? 1.7976931348623157e+308 : 3.4028234663852886e+38;
        return b.build_sle(abs(v), k(max_v));
    }

    template <std::size_t N>
    Value* horner(Value* x, const double (&coeffs)[N]) const {
        Value* acc = k(coeffs[N - 1]);
        for (std::size_t i = N - 1; i > 0; --i) {
            acc = fma(acc, x, k(coeffs[i - 1]));
        }
        return acc;
    }

    std::pair<Value*, Value*> sincos(Value* x) const {
        if (f64) {
            Value* j = b.build_round_f64(mul(x, k(0.63661977236758134308)));
            Value* y = sub(x, mul(j, k(1.5707963267341256141662598)));
            y = sub(y, mul(j, k(6.0771005065061922518926e-11)));
            Value* z = mul(y, y);
            static constexpr double kSin[] = {
                -1.0 / 6.0,
                 1.0 / 120.0,
                -1.0 / 5040.0,
                 1.0 / 362880.0,
                -1.0 / 39916800.0,
                 1.0 / 6227020800.0,
                -1.0 / 1307674368000.0,
                 1.0 / 355687428096000.0,
                -1.0 / 121645100408832000.0,
            };
            static constexpr double kCos[] = {
                 0.5,
                -1.0 / 24.0,
                 1.0 / 720.0,
                -1.0 / 40320.0,
                 1.0 / 3628800.0,
                -1.0 / 479001600.0,
                 1.0 / 87178291200.0,
                -1.0 / 20922789888000.0,
                 1.0 / 6402373705728000.0,
            };
            Value* sv = add(y, mul(mul(y, z), horner(z, kSin)));
            Value* cv = sub(k(1.0), mul(z, horner(z, kCos)));
            Value* q = b.build_and(b.build_fptosi_i32(j), i32(3));
            Value* nsv = neg(sv);
            Value* ncv = neg(cv);
            Value* q0 = b.build_eq(q, i32(0));
            Value* q1 = b.build_eq(q, i32(1));
            Value* q2 = b.build_eq(q, i32(2));
            Value* s = b.build_select(q0, sv, b.build_select(q1, cv, b.build_select(q2, nsv, ncv)));
            Value* c = b.build_select(q0, cv, b.build_select(q1, nsv, b.build_select(q2, ncv, sv)));
            return {s, c};
        }
        // FP32 range-reduced sincos matching brotensor's sincos_acc (~1 ulp).
        Value* j = b.build_round_f32(mul(x, k(0.636619772367581343)));
        Value* nj = neg(j);
        Value* y = fma(nj, k(1.57079637050628662), x);
        y = fma(nj, k(-4.37113882867379e-08), y);
        y = fma(nj, k(-1.71512451e-15), y);
        Value* z = mul(y, y);
        static constexpr double kSinF[] = {-1.6666654611e-1, 8.3321608736e-3, -1.9515295891e-4};
        static constexpr double kCosF[] = {4.166664568298827e-2, -1.388731625493765e-3, 2.443315711809948e-5};
        Value* sp = fma(mul(y, z), horner(z, kSinF), y);
        Value* cp = fma(mul(z, z), horner(z, kCosF), fma(k(-0.5), z, k(1.0)));
        Value* nsp = neg(sp);
        Value* ncp = neg(cp);
        Value* q = b.build_and(kb.f32_to_i32(j), i32(3));
        Value* q0 = b.build_eq(q, i32(0));
        Value* q1 = b.build_eq(q, i32(1));
        Value* q2 = b.build_eq(q, i32(2));
        Value* s = b.build_select(q0, sp, b.build_select(q1, cp, b.build_select(q2, nsp, ncp)));
        Value* c = b.build_select(q0, cp, b.build_select(q1, nsp, b.build_select(q2, ncp, sp)));
        return {s, c};
    }

    void ray_relu3(Value*& nx, Value*& ny, Value*& nz,
                   Value* rx, Value* ry, Value* rz) const {
        Value* nn = sqrt(add(add(mul(nx, nx), mul(ny, ny)), mul(nz, nz)));
        Value* rr = sqrt(add(add(mul(rx, rx), mul(ry, ry)), mul(rz, rz)));
        Value* denom = fmax(mul(nn, rr), k(1e-8));
        Value* cosv = div(add(add(mul(nx, rx), mul(ny, ry)), mul(nz, rz)), denom);
        Value* relu_cm = add(fmax(sub(cosv, k(1e-2)), k(0.0)), k(1e-2));
        Value* dcoef = sub(relu_cm, cosv);
        Value* mx = add(nx, mul(rx, dcoef));
        Value* my = add(ny, mul(ry, dcoef));
        Value* mz = add(nz, mul(rz, dcoef));
        Value* inv = div(k(1.0), fmax(sqrt(add(add(mul(mx, mx), mul(my, my)), mul(mz, mz))), k(1e-12)));
        nx = mul(mx, inv);
        ny = mul(my, inv);
        nz = mul(mz, inv);
    }

    std::array<Value*, 9> axis_angle_to_matrix(Value* ax, Value* ay, Value* az) const {
        Value* angle = sqrt(add(add(mul(ax, ax), mul(ay, ay)), mul(az, az)));
        Value* hang = mul(angle, k(0.5));
        auto [sh, ch] = sincos(hang);
        Value* s_small = sub(k(0.5), div(mul(angle, angle), k(48.0)));
        Value* safe_angle = fmax(angle, k(1e-12));
        Value* s = b.build_select(b.build_slt(angle, k(1e-6)), s_small, div(sh, safe_angle));
        Value* r = ch;
        Value* i = mul(ax, s);
        Value* j = mul(ay, s);
        Value* kv = mul(az, s);
        Value* qn = add(add(mul(r, r), mul(i, i)), add(mul(j, j), mul(kv, kv)));
        Value* two_s = div(k(2.0), qn);
        Value* ii = mul(i, i), *jj = mul(j, j), *kk = mul(kv, kv);
        Value* ij = mul(i, j), *ik = mul(i, kv), *jk = mul(j, kv);
        Value* ir = mul(i, r), *jr = mul(j, r), *kr = mul(kv, r);
        return {
            sub(k(1.0), mul(two_s, add(jj, kk))),
            mul(two_s, sub(ij, kr)),
            mul(two_s, add(ik, jr)),
            mul(two_s, add(ij, kr)),
            sub(k(1.0), mul(two_s, add(ii, kk))),
            mul(two_s, sub(jk, ir)),
            mul(two_s, sub(ik, jr)),
            mul(two_s, add(jk, ir)),
            sub(k(1.0), mul(two_s, add(ii, jj))),
        };
    }
};

Value* clamp_i32(Builder& b, Value* v, Value* lo, Value* hi) {
    return b.build_select(b.build_slt(v, lo), lo, b.build_select(b.build_sgt(v, hi), hi, v));
}

// ── Kernel 1: DPT bicubic resize + ImageNet normalize ───────────────────────

Function* build_dpt_preprocess_mir(Module& mod) {
    std::vector<Type> ptypes = {
        Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function("dpt_preprocess", Type::void_type(),
                                       brass::Span<const Type>(ptypes));
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* body = b.append_block("body");
    BasicBlock* done = b.append_block("done");
    b.position_at_end(entry);

    Value* p_src = b.add_param(Type::ptr());
    Value* p_dst = b.add_param(Type::ptr());
    Value* p_in_w = b.add_param(Type::i32());
    Value* p_in_h = b.add_param(Type::i32());
    Value* p_ch = b.add_param(Type::i32());
    Value* p_out_w = b.add_param(Type::i32());
    Value* p_out_h = b.add_param(Type::i32());

    auto ci = [&](std::int32_t v) { return b.build_iconst_i32(v); };
    auto cf = [&](float v) { return b.build_fconst_f32(v); };

    Value* x = b.build_add(b.build_mul(kb.ctaid_x(), ci(16)), kb.tid_x());
    Value* y = b.build_add(b.build_mul(kb.ctaid_y(), ci(16)), kb.tid_y());
    Value* in_bounds = b.build_and(b.build_slt(x, p_out_w), b.build_slt(y, p_out_h));
    b.build_br_if(in_bounds, body, done);

    b.position_at_end(body);
    Value* xs = b.build_sdiv(kb.i32_to_f32(p_in_w), kb.i32_to_f32(p_out_w));
    Value* ys = b.build_sdiv(kb.i32_to_f32(p_in_h), kb.i32_to_f32(p_out_h));

    Value* fy = b.build_sub(b.build_mul(b.build_add(kb.i32_to_f32(y), cf(0.5f)), ys), cf(0.5f));
    Value* iy = b.build_fptosi_i32_f32(b.build_floor_f32(fy));
    Value* ty = b.build_sub(fy, kb.i32_to_f32(iy));

    Value* fx = b.build_sub(b.build_mul(b.build_add(kb.i32_to_f32(x), cf(0.5f)), xs), cf(0.5f));
    Value* ix = b.build_fptosi_i32_f32(b.build_floor_f32(fx));
    Value* tx = b.build_sub(fx, kb.i32_to_f32(ix));

    auto cubic = [&](Value* t) -> Value* {
        Value* at = kb.fabs(t);
        Value* at2 = b.build_mul(at, at);
        Value* at3 = b.build_mul(at2, at);
        Value* w1 = b.build_add(b.build_sub(b.build_mul(cf(1.5f), at3), b.build_mul(cf(2.5f), at2)), cf(1.0f));
        Value* w2 = b.build_add(
            b.build_sub(b.build_add(b.build_mul(cf(-0.5f), at3), b.build_mul(cf(2.5f), at2)),
                        b.build_mul(cf(4.0f), at)),
            cf(2.0f));
        return b.build_select(b.build_slt(at, cf(1.0f)), w1,
                              b.build_select(b.build_slt(at, cf(2.0f)), w2, cf(0.0f)));
    };

    Value* zero_i = ci(0);
    Value* max_y = b.build_sub(p_in_h, ci(1));
    Value* max_x = b.build_sub(p_in_w, ci(1));
    Value* wy[4];
    Value* wx[4];
    Value* sy[4];
    Value* sx[4];
    for (int k = 0; k < 4; ++k) {
        wy[k] = cubic(b.build_sub(cf(static_cast<float>(k - 1)), ty));
        wx[k] = cubic(b.build_sub(cf(static_cast<float>(k - 1)), tx));
        sy[k] = clamp_i32(b, b.build_add(iy, ci(k - 1)), zero_i, max_y);
        sx[k] = clamp_i32(b, b.build_add(ix, ci(k - 1)), zero_i, max_x);
    }

    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float kInvStd[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
    Value* plane = b.build_mul(p_out_w, p_out_h);
    Value* yx_off = b.build_add(b.build_mul(y, p_out_w), x);
    Value* is_gray = b.build_eq(p_ch, ci(1));

    for (int c = 0; c < 3; ++c) {
        Value* acc = cf(0.0f);
        for (int j = 0; j < 4; ++j) {
            Value* row = cf(0.0f);
            Value* row_off = b.build_mul(sy[j], p_in_w);
            for (int i = 0; i < 4; ++i) {
                Value* pix = b.build_mul(b.build_add(row_off, sx[i]), p_ch);
                Value* bidx = b.build_select(is_gray, pix, b.build_add(pix, ci(c)));
                Value* ptr = b.build_add(p_src, b.build_zext_i64(bidx));
                Value* v = kb.u32_to_f32(kb.load_u8(ptr, 0));
                row = b.build_fma_f32(wx[i], v, row);
            }
            acc = b.build_fma_f32(wy[j], row, acc);
        }
        acc = kb.clamp_f32(acc, cf(0.0f), cf(255.0f));
        Value* u = kb.i32_to_f32(b.build_fptosi_i32_f32(b.build_add(acc, cf(0.5f))));
        Value* norm = b.build_mul(
            b.build_sub(b.build_mul(u, cf(1.0f / 255.0f)), cf(kMean[c])),
            cf(kInvStd[c]));
        Value* dst_idx = b.build_zext_i64(b.build_add(b.build_mul(ci(c), plane), yx_off));
        kb.store_f32_indexed(p_dst, dst_idx, norm);
    }

    b.build_br(done);
    b.position_at_end(done);
    b.build_ret_void();
    return fn;
}

// ── Kernel 2: DSINE RayReLU ─────────────────────────────────────────────────

Function* build_dsine_ray_relu_mir(Module& mod, bool use_f64) {
    const std::string name = use_f64 ? "dsine_ray_relu_f64" : "dsine_ray_relu";
    std::vector<Type> ptypes = {Type::ptr(), Type::ptr(), Type::i32()};
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(ptypes));
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    RealOps ro{kb, b, use_f64};

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* hdr = b.append_block("hdr");
    BasicBlock* body = b.append_block("body");
    BasicBlock* done = b.append_block("done");
    b.position_at_end(entry);

    Value* p_n = b.add_param(Type::ptr());
    Value* p_r = b.add_param(Type::ptr());
    Value* p_hw = b.add_param(Type::i32());

    Value* gid = kb.global_tid_x();
    Value* stride = b.build_mul(kb.nctaid_x(), b.build_iconst_i32(256));
    Value* hw64 = b.build_zext_i64(p_hw);
    Value* hw2_64 = b.build_add(hw64, hw64);
    b.build_br(hdr, {gid});

    b.position_at_end(hdr);
    Value* phi_p = b.add_param(Type::i32());
    b.build_br_if(b.build_ult(phi_p, p_hw), body, done);

    b.position_at_end(body);
    Value* p0 = b.build_zext_i64(phi_p);
    Value* p1 = b.build_add(p0, hw64);
    Value* p2 = b.build_add(p0, hw2_64);
    Value* nx = ro.load(p_n, p0);
    Value* ny = ro.load(p_n, p1);
    Value* nz = ro.load(p_n, p2);
    Value* rx = ro.load(p_r, p0);
    Value* ry = ro.load(p_r, p1);
    Value* rz = ro.load(p_r, p2);
    ro.ray_relu3(nx, ny, nz, rx, ry, rz);
    ro.store(p_n, p0, nx);
    ro.store(p_n, p1, ny);
    ro.store(p_n, p2, nz);

    Value* next_p = b.build_add(phi_p, stride);
    b.build_br(hdr, {next_p});

    b.position_at_end(done);
    b.build_ret_void();
    return fn;
}

// ── Kernel 3: DSINE AngMF 5x5 propagation ───────────────────────────────────

Function* build_dsine_angmf_mir(Module& mod, bool use_f64) {
    const std::string name = use_f64 ? "dsine_angmf_f64" : "dsine_angmf";
    const Type rt = use_f64 ? Type::f64() : Type::f32();
    std::vector<Type> ptypes = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        rt, rt, rt, rt,
        Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(ptypes));
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    RealOps ro{kb, b, use_f64};

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* phdr = b.append_block("phdr");
    BasicBlock* pbody = b.append_block("pbody");
    BasicBlock* nhdr = b.append_block("nhdr");
    BasicBlock* nbody = b.append_block("nbody");
    BasicBlock* pstore = b.append_block("pstore");
    BasicBlock* done = b.append_block("done");
    b.position_at_end(entry);

    Value* p_pn = b.add_param(Type::ptr());
    Value* p_pr = b.add_param(Type::ptr());
    Value* p_xy = b.add_param(Type::ptr());
    Value* p_ang = b.add_param(Type::ptr());
    Value* p_ray = b.add_param(Type::ptr());
    Value* p_op = b.add_param(Type::ptr());
    Value* p_fu = b.add_param(rt);
    Value* p_cu = b.add_param(rt);
    Value* p_fv = b.add_param(rt);
    Value* p_cv = b.add_param(rt);
    Value* p_H = b.add_param(Type::i32());
    Value* p_W = b.add_param(Type::i32());

    auto ci = [&](std::int32_t v) { return b.build_iconst_i32(v); };
    Value* gid = kb.global_tid_x();
    Value* stride = b.build_mul(kb.nctaid_x(), ci(256));
    Value* hw = b.build_mul(p_H, p_W);
    Value* hw64 = b.build_zext_i64(hw);
    Value* hw2_64 = b.build_add(hw64, hw64);
    Value* max_y = b.build_sub(p_H, ci(1));
    Value* max_x = b.build_sub(p_W, ci(1));
    Value* zero_i = ci(0);
    Value* c25_64 = b.build_iconst_i64(25);
    b.build_br(phdr, {gid});

    b.position_at_end(phdr);
    Value* phi_p = b.add_param(Type::i32());
    b.build_br_if(b.build_ult(phi_p, hw), pbody, done);

    b.position_at_end(pbody);
    Value* y = b.build_udiv(phi_p, p_W);
    Value* x = b.build_sub(phi_p, b.build_mul(y, p_W));
    Value* p0 = b.build_zext_i64(phi_p);
    Value* p1 = b.build_add(p0, hw64);
    Value* p2 = b.build_add(p0, hw2_64);
    Value* rx = ro.load(p_ray, p0);
    Value* ry = ro.load(p_ray, p1);
    Value* rz = ro.load(p_ray, p2);
    Value* z0 = ro.k(0.0);
    b.build_br(nhdr, {zero_i, z0, z0, z0});

    b.position_at_end(nhdr);
    Value* phi_nn = b.add_param(Type::i32());
    Value* phi_a0 = b.add_param(rt);
    Value* phi_a1 = b.add_param(rt);
    Value* phi_a2 = b.add_param(rt);
    b.build_br_if(b.build_slt(phi_nn, ci(25)), nbody, pstore);

    b.position_at_end(nbody);
    Value* ky = b.build_sdiv(phi_nn, ci(5));
    Value* kx = b.build_sub(phi_nn, b.build_mul(ky, ci(5)));
    Value* sy = clamp_i32(b, b.build_add(b.build_sub(y, ci(2)), ky), zero_i, max_y);
    Value* sx = clamp_i32(b, b.build_add(b.build_sub(x, ci(2)), kx), zero_i, max_x);
    Value* sp = b.build_zext_i64(b.build_add(b.build_mul(sy, p_W), sx));

    Value* nx = ro.load(p_pn, sp);
    Value* ny = ro.load(p_pn, b.build_add(sp, hw64));
    Value* nz = ro.load(p_pn, b.build_add(sp, hw2_64));
    Value* pix_x = ro.add(ro.from_i32(sx), ro.k(0.5));
    Value* pix_y = ro.add(ro.from_i32(sy), ro.k(0.5));

    Value* nn64 = b.build_zext_i64(phi_nn);
    Value* ch_off = b.build_add(b.build_mul(nn64, hw64), p0);
    Value* ch_off2 = b.build_add(b.build_mul(b.build_add(nn64, c25_64), hw64), p0);
    Value* xs = ro.load(p_xy, ch_off);
    Value* ys = ro.load(p_xy, ch_off2);
    Value* xyinv = ro.div(ro.k(1.0), ro.fmax(ro.sqrt(ro.add(ro.mul(xs, xs), ro.mul(ys, ys))), ro.k(1e-12)));
    xs = ro.mul(xs, xyinv);
    ys = ro.mul(ys, xyinv);
    Value* theta = ro.load(p_ang, ch_off);
    Value* w = ro.load(p_pr, ch_off);

    Value* du_over_fu = ro.div(xs, p_fu);
    Value* dv_over_fv = ro.div(ys, p_fv);
    Value* term_u = ro.div(ro.sub(ro.add(pix_x, xs), p_cu), p_fu);
    Value* term_v = ro.div(ro.sub(ro.add(pix_y, ys), p_cv), p_fv);

    Value* num = ro.neg(ro.add(ro.mul(du_over_fu, nx), ro.mul(dv_over_fv, ny)));
    Value* dd0 = ro.add(ro.add(ro.mul(term_u, nx), ro.mul(term_v, ny)), nz);
    Value* sgn = b.build_select(
        b.build_sgt(dd0, ro.k(0.0)), ro.k(1.0),
        b.build_select(b.build_slt(dd0, ro.k(0.0)), ro.k(-1.0), ro.k(0.0)));
    Value* dd = b.build_select(b.build_slt(ro.abs(dd0), ro.k(1e-8)), ro.mul(ro.k(1e-8), sgn), dd0);
    Value* dz = ro.div(num, dd);

    Value* axx = ro.add(du_over_fu, ro.mul(dz, term_u));
    Value* axy = ro.add(dv_over_fv, ro.mul(dz, term_v));
    Value* axz = dz;
    Value* ainv = ro.div(
        ro.k(1.0),
        ro.fmax(ro.sqrt(ro.add(ro.add(ro.mul(axx, axx), ro.mul(axy, axy)), ro.mul(axz, axz))), ro.k(1e-12)));
    axx = ro.mul(axx, ainv);
    axy = ro.mul(axy, ainv);
    axz = ro.mul(axz, ainv);
    Value* valid = b.build_and(b.build_and(ro.is_finite(axx), ro.is_finite(axy)), ro.is_finite(axz));
    axx = b.build_select(valid, axx, ro.k(0.0));
    axy = b.build_select(valid, axy, ro.k(0.0));
    axz = b.build_select(valid, axz, ro.k(0.0));

    auto R = ro.axis_angle_to_matrix(ro.mul(axx, theta), ro.mul(axy, theta), ro.mul(axz, theta));
    Value* rnx = ro.add(ro.add(ro.mul(R[0], nx), ro.mul(R[1], ny)), ro.mul(R[2], nz));
    Value* rny = ro.add(ro.add(ro.mul(R[3], nx), ro.mul(R[4], ny)), ro.mul(R[5], nz));
    Value* rnz = ro.add(ro.add(ro.mul(R[6], nx), ro.mul(R[7], ny)), ro.mul(R[8], nz));
    Value* rinv = ro.div(
        ro.k(1.0),
        ro.fmax(ro.sqrt(ro.add(ro.add(ro.mul(rnx, rnx), ro.mul(rny, rny)), ro.mul(rnz, rnz))), ro.k(1e-12)));
    rnx = ro.mul(rnx, rinv);
    rny = ro.mul(rny, rinv);
    rnz = ro.mul(rnz, rinv);

    ro.ray_relu3(rnx, rny, rnz, rx, ry, rz);
    Value* na0 = ro.add(phi_a0, ro.mul(w, rnx));
    Value* na1 = ro.add(phi_a1, ro.mul(w, rny));
    Value* na2 = ro.add(phi_a2, ro.mul(w, rnz));
    Value* next_nn = b.build_add(phi_nn, ci(1));
    b.build_br(nhdr, {next_nn, na0, na1, na2});

    b.position_at_end(pstore);
    Value* inv = ro.div(
        ro.k(1.0),
        ro.fmax(ro.sqrt(ro.add(ro.add(ro.mul(phi_a0, phi_a0), ro.mul(phi_a1, phi_a1)),
                               ro.mul(phi_a2, phi_a2))),
                ro.k(1e-12)));
    ro.store(p_op, p0, ro.mul(phi_a0, inv));
    ro.store(p_op, p1, ro.mul(phi_a1, inv));
    ro.store(p_op, p2, ro.mul(phi_a2, inv));

    Value* next_p = b.build_add(phi_p, stride);
    b.build_br(phdr, {next_p});

    b.position_at_end(done);
    b.build_ret_void();
    return fn;
}

// ── Backend-neutral compilation cache & dispatch ────────────────────────────

enum class KernelId : std::uint32_t {
    DptPreprocess = 0,
    DsineRayReluF32 = 1,
    DsineRayReluF64 = 2,
    DsineAngmfF32 = 3,
    DsineAngmfF64 = 4,
};

enum class BackendKind : std::uint8_t { Vulkan = 0, Cuda = 1, Metal = 2 };

struct CompiledKernel {
    brass::target::SpirvKernel spv;
    std::string ptx;
    std::string ptx_entry;
    brass::target::MslKernel msl;
    std::uint32_t lx = 256, ly = 1, lz = 1;
};

Function* build_kernel(Module& mod, KernelId id, std::uint32_t& lx, std::uint32_t& ly) {
    switch (id) {
        case KernelId::DptPreprocess:   lx = 16;  ly = 16; return build_dpt_preprocess_mir(mod);
        case KernelId::DsineRayReluF32: lx = 256; ly = 1;  return build_dsine_ray_relu_mir(mod, false);
        case KernelId::DsineRayReluF64: lx = 256; ly = 1;  return build_dsine_ray_relu_mir(mod, true);
        case KernelId::DsineAngmfF32:   lx = 256; ly = 1;  return build_dsine_angmf_mir(mod, false);
        case KernelId::DsineAngmfF64:   lx = 256; ly = 1;  return build_dsine_angmf_mir(mod, true);
    }
    fail("unknown kernel id");
}

const CompiledKernel& get_kernel(KernelId id, BackendKind bk) {
    struct Cache {
        std::mutex mu;
        std::unordered_map<std::uint64_t, CompiledKernel> map;
    };
    static auto* cache = new Cache();
    const std::uint64_t key = (static_cast<std::uint64_t>(bk) << 32) | static_cast<std::uint32_t>(id);
    {
        std::lock_guard<std::mutex> lk(cache->mu);
        if (auto it = cache->map.find(key); it != cache->map.end()) return it->second;
    }

    Module mod("brovisionml_mir");
    std::uint32_t lx = 256, ly = 1;
    Function* fn = build_kernel(mod, id, lx, ly);

    CompiledKernel ck;
    ck.lx = lx;
    ck.ly = ly;
    ck.lz = 1;
    if (bk == BackendKind::Vulkan) {
        brass::target::SpirvOptions opts;
        opts.stage = brass::target::SpirvStage::Compute;
        opts.local_size_x = lx;
        opts.local_size_y = ly;
        opts.local_size_z = 1;
        opts.local_size_spec_constants = false;
        ck.spv = brass::target::SpirvTarget::compile(*fn, opts);
    } else if (bk == BackendKind::Cuda) {
        brass::target::PtxOptions opts;
        opts.sm_arch = "sm_70";
        opts.ptx_version_major = 7;
        opts.ptx_version_minor = 0;
        ck.ptx = brass::target::PtxTarget::emit_function(*fn, opts);
        ck.ptx_entry = std::string(fn->name());
    } else {
        brass::target::MslOptions opts;
        opts.workgroup_x = lx;
        opts.workgroup_y = ly;
        opts.workgroup_z = 1;
        ck.msl = brass::target::MslTarget::compile(*fn, opts);
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    auto [it, _] = cache->map.emplace(key, std::move(ck));
    return it->second;
}

BackendKind backend_of(bt::Device dev) {
    if (dev.is_vulkan()) return BackendKind::Vulkan;
    if (dev.is_cuda()) return BackendKind::Cuda;
    if (dev.is_metal()) return BackendKind::Metal;
    fail("unsupported GPU backend");
}

struct ArgSlot {
    bool is_ptr = false;
    std::array<std::uint8_t, 8> bytes{};
    std::uint32_t size = 0;

    static ArgSlot ptr(const bt::Tensor& t) {
        ArgSlot s;
        s.is_ptr = true;
        s.size = 8;
        const std::uint64_t addr = bt::interop::view(t).device_address;
        std::memcpy(s.bytes.data(), &addr, 8);
        return s;
    }
    template <typename T>
    static ArgSlot scalar(const T& v) {
        static_assert(sizeof(T) <= 8);
        ArgSlot s;
        s.is_ptr = false;
        s.size = static_cast<std::uint32_t>(sizeof(T));
        std::memcpy(s.bytes.data(), &v, sizeof(T));
        return s;
    }
};

void dispatch_mir(bt::Device dev, KernelId id,
                  std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                  const ArgSlot* args, std::size_t nargs) {
    const BackendKind bk = backend_of(dev);
    const CompiledKernel& ck = get_kernel(id, bk);
    std::string err;

    if (bk == BackendKind::Vulkan) {
        std::vector<std::uint8_t> push(ck.spv.push_constant_bytes, 0);
        if (ck.spv.params.size() != nargs) fail("SPIR-V param count mismatch");
        for (std::size_t i = 0; i < nargs; ++i) {
            std::memcpy(push.data() + ck.spv.params[i].offset, args[i].bytes.data(), args[i].size);
        }
        bt::interop::SpirvDispatchDesc desc;
        desc.words = ck.spv.words.data();
        desc.word_count = ck.spv.words.size();
        desc.entry_name = ck.spv.entry.c_str();
        desc.local_size[0] = ck.lx;
        desc.local_size[1] = ck.ly;
        desc.local_size[2] = ck.lz;
        desc.groups[0] = gx;
        desc.groups[1] = gy;
        desc.groups[2] = gz;
        desc.push_data = push.data();
        desc.push_bytes = static_cast<std::uint32_t>(push.size());
        desc.local_size_spec_constants = false;
        if (!bt::interop::dispatch_spirv(dev, desc, &err)) fail(err);
        return;
    }

    if (bk == BackendKind::Cuda) {
        std::array<std::uint64_t, 16> storage{};
        std::array<void*, 16> kparams{};
        for (std::size_t i = 0; i < nargs; ++i) {
            std::memcpy(&storage[i], args[i].bytes.data(), args[i].size);
            kparams[i] = &storage[i];
        }
        bt::interop::PtxDispatchDesc desc;
        desc.ptx = ck.ptx.data();
        desc.ptx_bytes = ck.ptx.size();
        desc.entry_name = ck.ptx_entry.c_str();
        desc.local_size[0] = ck.lx;
        desc.local_size[1] = ck.ly;
        desc.local_size[2] = ck.lz;
        desc.groups[0] = gx;
        desc.groups[1] = gy;
        desc.groups[2] = gz;
        desc.kernel_params = kparams.data();
        if (!bt::interop::dispatch_ptx(dev, desc, &err)) fail(err);
        return;
    }

    std::array<const void*, 16> bufs{};
    std::uint32_t nbufs = 0;
    std::array<std::uint8_t, 128> scalars{};
    std::size_t sidx = 0;
    for (std::size_t i = 0; i < nargs; ++i) {
        if (args[i].is_ptr) {
            std::uint64_t addr = 0;
            std::memcpy(&addr, args[i].bytes.data(), 8);
            bufs[nbufs++] = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(addr));
        } else {
            if (sidx < ck.msl.scalar_param_offsets.size()) {
                std::memcpy(scalars.data() + ck.msl.scalar_param_offsets[sidx],
                            args[i].bytes.data(), args[i].size);
            }
            ++sidx;
        }
    }
    bt::interop::MslDispatchDesc desc;
    desc.msl = ck.msl.source.data();
    desc.msl_bytes = ck.msl.source.size();
    desc.entry_name = ck.msl.entry_name.c_str();
    desc.local_size[0] = ck.lx;
    desc.local_size[1] = ck.ly;
    desc.local_size[2] = ck.lz;
    desc.groups[0] = gx;
    desc.groups[1] = gy;
    desc.groups[2] = gz;
    desc.buffer_ptrs = bufs.data();
    desc.buffer_count = nbufs;
    desc.scalar_data = scalars.data();
    desc.scalar_bytes = static_cast<std::uint32_t>(ck.msl.scalar_params_bytes);
    desc.scalar_buffer_index = static_cast<std::uint32_t>(ck.msl.num_buffers);
    if (!bt::interop::dispatch_msl(dev, desc, &err)) fail(err);
}

}  // namespace

// ── Public internal entry points ────────────────────────────────────────────

namespace dpt::detail {

void dpt_preprocess_gpu(const std::uint8_t* host_rgb, int in_w, int in_h, int channels,
                        bt::Device dev, int out_w, int out_h, bt::Tensor& out_pixels) {
    if (!host_rgb || in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) {
        fail("dpt_preprocess_gpu: invalid dimensions");
    }
    if (!dev.is_gpu()) fail("dpt_preprocess_gpu: not a GPU device");

    const std::size_t in_bytes = static_cast<std::size_t>(in_w) * in_h * channels;
    const int out_plane = out_w * out_h;

    static thread_local bt::Tensor s_dev_in;
    const int words = static_cast<int>((in_bytes + 3u) / 4u);
    if (s_dev_in.device != dev || s_dev_in.rows != 1 || s_dev_in.cols < words) {
        s_dev_in = bt::Tensor::empty_on(dev, 1, words, bt::Dtype::INT32);
    }
    std::string err;
    if (!bt::interop::copy_bytes_to_tensor(host_rgb, in_bytes, s_dev_in, &err)) {
        fail("dpt_preprocess_gpu: H2D upload failed: " + err);
    }

    if (out_pixels.rows != 1 || out_pixels.cols != 3 * out_plane || out_pixels.device != dev ||
        out_pixels.dtype != bt::Dtype::FP32) {
        out_pixels = bt::Tensor::empty_on(dev, 1, 3 * out_plane, bt::Dtype::FP32);
    }

    const auto gx = static_cast<std::uint32_t>((out_w + 15) / 16);
    const auto gy = static_cast<std::uint32_t>((out_h + 15) / 16);
    if (gx > 65535u || gy > 65535u) fail("dpt_preprocess_gpu: output too large");

    const ArgSlot args[7] = {
        ArgSlot::ptr(s_dev_in),
        ArgSlot::ptr(out_pixels),
        ArgSlot::scalar<std::int32_t>(in_w),
        ArgSlot::scalar<std::int32_t>(in_h),
        ArgSlot::scalar<std::int32_t>(channels),
        ArgSlot::scalar<std::int32_t>(out_w),
        ArgSlot::scalar<std::int32_t>(out_h),
    };
    dispatch_mir(dev, KernelId::DptPreprocess, gx, gy, 1u, args, 7);
}

}  // namespace dpt::detail

namespace dsine::detail {

void ray_relu_gpu(bt::Tensor& normal, const bt::Tensor& ray, int H, int W) {
    require_f32_gpu(normal, "ray_relu: normal");
    require_f32_gpu(ray, "ray_relu: ray");
    const std::uint64_t HW = static_cast<std::uint64_t>(H) * W;
    if (HW == 0) return;
    const auto u64_sz = [](const bt::Tensor& t) {
        return static_cast<std::uint64_t>(std::max(0, t.size()));
    };
    if (u64_sz(normal) < 3 * HW || u64_sz(ray) < 3 * HW) {
        fail("ray_relu: maps smaller than 3*H*W");
    }
    const bool use_f64 = bt::interop::device_has_float64(normal.device);
    const ArgSlot args[3] = {
        ArgSlot::ptr(normal),
        ArgSlot::ptr(ray),
        ArgSlot::scalar<std::uint32_t>(static_cast<std::uint32_t>(HW)),
    };
    dispatch_mir(normal.device,
                 use_f64 ? KernelId::DsineRayReluF64 : KernelId::DsineRayReluF32,
                 groups_1d(HW), 1u, 1u, args, 3);
}

void angmf_propagate_gpu(const bt::Tensor& pred_norm, const bt::Tensor& prob,
                         const bt::Tensor& xy, const bt::Tensor& angle,
                         const bt::Tensor& ray,
                         double fu, double cu, double fv, double cv,
                         int H, int W, bt::Tensor& out) {
    require_f32_gpu(pred_norm, "angmf_propagate: pred_norm");
    require_f32_gpu(prob, "angmf_propagate: prob");
    require_f32_gpu(xy, "angmf_propagate: xy");
    require_f32_gpu(angle, "angmf_propagate: angle");
    require_f32_gpu(ray, "angmf_propagate: ray");
    const std::uint64_t HW = static_cast<std::uint64_t>(H) * W;
    out = bt::Tensor::zeros_on(pred_norm.device, 1, static_cast<int>(3 * HW));
    if (HW == 0) return;
    const auto u64_sz = [](const bt::Tensor& t) {
        return static_cast<std::uint64_t>(std::max(0, t.size()));
    };
    if (u64_sz(pred_norm) < 3 * HW || u64_sz(ray) < 3 * HW || u64_sz(prob) < 25 * HW ||
        u64_sz(angle) < 25 * HW || u64_sz(xy) < 50 * HW) {
        fail("angmf_propagate: input maps smaller than their (C, H, W) shapes");
    }
    const bt::Device dev = pred_norm.device;
    const bool use_f64 = bt::interop::device_has_float64(dev);
    if (use_f64) {
        const ArgSlot args[12] = {
            ArgSlot::ptr(pred_norm), ArgSlot::ptr(prob), ArgSlot::ptr(xy),
            ArgSlot::ptr(angle), ArgSlot::ptr(ray), ArgSlot::ptr(out),
            ArgSlot::scalar<double>(fu), ArgSlot::scalar<double>(cu),
            ArgSlot::scalar<double>(fv), ArgSlot::scalar<double>(cv),
            ArgSlot::scalar<std::int32_t>(H), ArgSlot::scalar<std::int32_t>(W),
        };
        dispatch_mir(dev, KernelId::DsineAngmfF64, groups_1d(HW), 1u, 1u, args, 12);
    } else {
        const ArgSlot args[12] = {
            ArgSlot::ptr(pred_norm), ArgSlot::ptr(prob), ArgSlot::ptr(xy),
            ArgSlot::ptr(angle), ArgSlot::ptr(ray), ArgSlot::ptr(out),
            ArgSlot::scalar<float>(static_cast<float>(fu)),
            ArgSlot::scalar<float>(static_cast<float>(cu)),
            ArgSlot::scalar<float>(static_cast<float>(fv)),
            ArgSlot::scalar<float>(static_cast<float>(cv)),
            ArgSlot::scalar<std::int32_t>(H), ArgSlot::scalar<std::int32_t>(W),
        };
        dispatch_mir(dev, KernelId::DsineAngmfF32, groups_1d(HW), 1u, 1u, args, 12);
    }
}

}  // namespace dsine::detail

namespace detail {

std::vector<MirAuditShader> compile_mir_spirv_for_audit() {
    static constexpr std::pair<const char*, KernelId> kAll[] = {
        {"dsine_ray_relu",     KernelId::DsineRayReluF32},
        {"dsine_angmf",        KernelId::DsineAngmfF32},
        {"dsine_ray_relu_f64", KernelId::DsineRayReluF64},
        {"dsine_angmf_f64",    KernelId::DsineAngmfF64},
        {"dpt_preprocess",     KernelId::DptPreprocess},
    };
    std::vector<MirAuditShader> out;
    out.reserve(5);
    for (const auto& [name, id] : kAll) {
        const CompiledKernel& vk_ck = get_kernel(id, BackendKind::Vulkan);
        const CompiledKernel& cu_ck = get_kernel(id, BackendKind::Cuda);
        if (cu_ck.ptx.empty()) fail(std::string("PTX emit failed for ") + name);
        if (id != KernelId::DsineRayReluF64 && id != KernelId::DsineAngmfF64) {
            const CompiledKernel& mtl_ck = get_kernel(id, BackendKind::Metal);
            if (mtl_ck.msl.source.empty()) fail(std::string("MSL emit failed for ") + name);
        }
        out.push_back(MirAuditShader{name, vk_ck.spv.words});
    }
    return out;
}

}  // namespace detail

}  // namespace brovisionml
