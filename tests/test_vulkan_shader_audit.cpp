// Static SPIR-V verification & memory-safety audit of brovisionml Brass MIR GPU kernels.

#include "gpu_mir_ops.h"

#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

namespace {

bool run_mem_safety(
    const brass::spirv::Module& mod,
    const std::string& label,
    const brass::verify::GpuProverOptions& opts,
    bool require_exact_safe) {
    brass::verify::GpuMemorySafetyResult mem = brass::verify::verify_spirv_memory_safety(mod, opts);
    const bool bounded_ok =
        !require_exact_safe &&
        mem.status == brass::verify::EquivStatus::Unknown &&
        mem.violations.empty() &&
        mem.global_accesses_proven > 0;
    const bool ok =
        !mem.is_counterexample() &&
        mem.violations.empty() &&
        (mem.is_safe() || bounded_ok) &&
        mem.global_accesses_proven > 0;
    std::printf("    %-28s status=%d global=%u align=%u shared=%u diag=\"%s\"\n",
                label.c_str(),
                static_cast<int>(mem.status),
                mem.global_accesses_proven,
                mem.alignments_proven,
                mem.shared_accesses_proven,
                mem.diagnostic.c_str());
    return ok;
}

} // namespace

int main() {
    const auto shaders = brovisionml::detail::compile_mir_spirv_for_audit();
    const std::size_t n = shaders.size();
    std::printf("Auditing %zu brovisionml Brass MIR SPIR-V kernels...\n", n);
    CHECK(n == 5);

    for (std::size_t i = 0; i < n; ++i) {
        const auto& info = shaders[i];
        CHECK(!info.words.empty());
        if (info.words.empty()) continue;

        const std::string& name = info.name;
        std::printf("  [%zu/%zu] %s (%zu words)\n", i + 1, n, name.c_str(), info.words.size());

        brass::spirv::Module mod;
        std::string parse_err;
        const bool parsed = brass::spirv::parse_binary(
            std::span<const std::uint32_t>(info.words.data(), info.words.size()), mod, &parse_err);
        if (!parsed) std::printf("    parse error: %s\n", parse_err.c_str());
        CHECK(parsed);
        if (!parsed) continue;

        std::string struct_err;
        const bool struct_ok = brass::spirv::verify_spirv_structural(mod, &struct_err);
        if (!struct_ok) std::printf("    structural error: %s\n", struct_err.c_str());
        CHECK(struct_ok);

        const brass::spirv::ComputeShaderAuditReport rep = brass::verify::prove_spirv_shader_safety(mod);
        CHECK(rep.structural_ok);
        CHECK(rep.dominance_ok);
        CHECK(rep.shared_bounds_ok);
        CHECK(rep.barriers_uniform);
        CHECK(rep.div_sqrt_ok);

        if (name == "dsine_ray_relu" || name == "dsine_ray_relu_f64") {
            brass::verify::GpuProverOptions opts;
            opts.local_size_x = std::max(1u, rep.local_size[0]);
            opts.loop_unroll_bound = 2;
            opts.max_sat_conflicts = 4000;
            opts.buffers = {
                {0, 0, 2, 4, 16, -1, -1, 3},
                {1, 0, 2, 4, 16, -1, -1, 3},
            };
            opts.param_constraints = {{2, UINT32_MAX, 1, 0}};
            CHECK(run_mem_safety(mod, name, opts, false));
        } else if (name == "dsine_angmf" || name == "dsine_angmf_f64") {
            brass::verify::GpuProverOptions opts;
            opts.local_size_x = std::max(1u, rep.local_size[0]);
            opts.loop_unroll_bound = 2;
            opts.max_sat_conflicts = 4000;
            opts.buffers = {
                {0, 0, 10, 4, 16, 11, -1, 3},
                {1, 0, 10, 4, 16, 11, -1, 25},
                {2, 0, 10, 4, 16, 11, -1, 50},
                {3, 0, 10, 4, 16, 11, -1, 25},
                {4, 0, 10, 4, 16, 11, -1, 3},
                {5, 0, 10, 4, 16, 11, -1, 3},
            };
            opts.param_constraints = {
                {10, UINT32_MAX, 1, 0},
                {11, UINT32_MAX, 1, 0},
            };
            CHECK(run_mem_safety(mod, name, opts, false));
        } else if (name == "dpt_preprocess") {
            brass::verify::GpuProverOptions base_opts;
            base_opts.local_size_x = std::max(1u, rep.local_size[0]);
            base_opts.local_size_y = std::max(1u, rep.local_size[1]);
            base_opts.loop_unroll_bound = 4;
            base_opts.max_sat_conflicts = 4000;
            base_opts.buffers = {
                {0, 0, 2, 1, 1, 3, 4, 1},
                {1, 0, 5, 4, 16, 6, -1, 3},
            };
            base_opts.param_constraints = {
                {2, UINT32_MAX, 1, 0},
                {3, UINT32_MAX, 1, 0},
                {5, UINT32_MAX, 1, 0},
                {6, UINT32_MAX, 1, 0},
            };

            auto opts_c1 = base_opts;
            opts_c1.param_constraints.push_back({4, UINT32_MAX, 1, 1});
            CHECK(run_mem_safety(mod, name + "[c=1]", opts_c1, true));

            auto opts_c34 = base_opts;
            opts_c34.param_constraints.push_back({4, UINT32_MAX, 3, 4});
            CHECK(run_mem_safety(mod, name + "[c=3..4]", opts_c34, true));
        } else {
            CHECK(false && "Unexpected brovisionml Vulkan shader");
        }
    }

    if (g_failures != 0) {
        std::printf("FAILED (%d checks)\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
