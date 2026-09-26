// Example harness: production vec_add_inplace (in-tree .hip -> .hsaco, unmodified) vs a float4
// candidate, at the decode shape n = 5120 (one residual row). Copy this layout for real work.
//   build.sh  -> base_<arch>.hsaco, cand_<arch>.hsaco, harness
//   run.sh    -> gpu_run.sh --dev ... -- ./harness <arch>
#include "kbench.h"

int main(int argc, char** argv) {
    const std::string arch = argc > 1 ? argv[1] : "gfx1151";
    const std::string dir = argc > 2 ? argv[2] : ".";
    kb::init();
    kb::Module base(dir + "/base_" + arch + ".hsaco"), cand(dir + "/cand_" + arch + ".hsaco");
    hipFunction_t f0 = base.fn("vec_add_inplace"), f1 = cand.fn("vec_add_inplace_v4");

    const unsigned n = 5120;
    float* out = kb::dalloc<float>(n);
    float* rhs = kb::dalloc<float>(n);
    float* out_ref = kb::dalloc<float>(n);
    kb::fill_f32(rhs, n, 1);

    // ---- correctness: one call each from identical inputs
    kb::fill_f32(out_ref, n, 2);
    KB_CHECK(hipMemcpy(out, out_ref, n * 4, hipMemcpyDeviceToDevice));
    kb::launch(f0, dim3((n + 255) / 256), dim3(256), 0, 0, out_ref, (const float*)rhs, n);
    kb::launch(f1, dim3((n / 4 + 255) / 256), dim3(256), 0, 0, out, (const float*)rhs, n);
    KB_CHECK(hipDeviceSynchronize());
    auto a = kb::d2h(out_ref, n), b = kb::d2h(out, n);
    kb::print_cmp("vec_add v4 vs base", kb::compare_f32(a.data(), b.data(), n));

    // ---- timing: graph mode (decode stages are graph-captured in production), warm caches
    std::vector<kb::Variant> vs = {
        {"base vec_add_inplace", [&](hipStream_t s) { kb::launch(f0, dim3((n + 255) / 256), dim3(256), 0, s, out, (const float*)rhs, n); }, 3.0 * 4 * n},
        {"cand vec_add_inplace_v4", [&](hipStream_t s) { kb::launch(f1, dim3((n / 4 + 255) / 256), dim3(256), 0, s, out, (const float*)rhs, n); }, 3.0 * 4 * n},
    };
    kb::AbOpts o;
    o.graph = true;
    o.inner = 20;
    o.tag = "vec_add n=5120 graph warm";
    kb::ab(vs, o);
    return 0;
}
