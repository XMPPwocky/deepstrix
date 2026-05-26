// WMMA throughput probe — measures realized f16 WMMA TFLOPs on
// RDNA3.5 (gfx1151 iGPU) and RDNA4 (gfx1201 dGPU). Used to decide
// whether the iq2 WMMA rewrite is worth doing: theoretical peak says
// ~32× the dp4a rate, but realized often falls short due to issue-rate
// limits, register pressure, fragment-load overhead. This probe tells
// us the actual rate before we commit to the rewrite.
//
// Per WMMA op on RDNA3+: 16×16×16 fragments = 4096 multiplies + 4096
// adds = 8192 ops. Per warp-cycle (1 issue/cycle for w32 wave).
//
// Strategy: each warp does N_ITERS back-to-back WMMA accumulates into
// the same accumulator (creating a serial dependency chain that
// stresses the WMMA issue rate). Total ops = n_warps × N_ITERS × 8192.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));

extern "C" __global__ void wmma_f16_throughput_probe(
        float        * __restrict__ out,     // [n_warps × 8] sink for c_frag
        const _Float16 * __restrict__ a_in,  // [16] just one A fragment we keep in regs
        const _Float16 * __restrict__ b_in,  // [16] just one B fragment
        unsigned int   n_iters) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp_id_in_block = threadIdx.x >> 5u;
    const unsigned int n_warps_per_block = blockDim.x >> 5u;
    const unsigned int global_warp_id =
        blockIdx.x * n_warps_per_block + warp_id_in_block;

    // Load fragments — every lane loads the same 16 elements. In a real
    // WMMA op the load is interpreted per-fragment-mapping; here we
    // just need values in registers so the compiler emits the WMMA insn.
    half16 a;
    half16 b;
    for (int i = 0; i < 16; i++) {
        a[i] = a_in[i];
        b[i] = b_in[i];
    }
    float8 c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // The dependency chain: each iter feeds c back into itself. This
    // measures issue rate per warp, not parallel throughput. To measure
    // parallel throughput, increase n_warps_per_block + total WGs.
    for (unsigned int i = 0; i < n_iters; i++) {
#if defined(__gfx1151__) || defined(__gfx1150__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
        // gfx1201 (RDNA4) uses a different intrinsic naming; not
        // exercised here since iq2 lives on iGPU only. Stub out so
        // the kernel still compiles but the probe shouldn't be run.
        c[0] += (float)a[0] * (float)b[0];
#endif
    }

    // Sink to prevent DCE. Each warp's lane 0 writes 8 floats.
    if (lane == 0) {
        const size_t base = (size_t)global_warp_id * 8;
        out[base + 0] = c[0];
        out[base + 1] = c[1];
        out[base + 2] = c[2];
        out[base + 3] = c[3];
        out[base + 4] = c[4];
        out[base + 5] = c[5];
        out[base + 6] = c[6];
        out[base + 7] = c[7];
    }
}

// Variant: 8 INDEPENDENT accumulators per warp. The 8 WMMAs in each
// iteration only depend on the SAME accumulator n iters back, not on
// each other. The compiler/scheduler can pipeline up to 8 WMMAs in
// flight per warp → measures actual peak throughput, not latency rate.
extern "C" __global__ void wmma_f16_throughput_probe_parallel(
        float        * __restrict__ out,
        const _Float16 * __restrict__ a_in,
        const _Float16 * __restrict__ b_in,
        unsigned int   n_iters) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp_id_in_block = threadIdx.x >> 5u;
    const unsigned int n_warps_per_block = blockDim.x >> 5u;
    const unsigned int global_warp_id =
        blockIdx.x * n_warps_per_block + warp_id_in_block;

    half16 a;
    half16 b;
    for (int i = 0; i < 16; i++) { a[i] = a_in[i]; b[i] = b_in[i]; }
    float8 c0 = {0.0f}, c1 = {0.0f}, c2 = {0.0f}, c3 = {0.0f};
    float8 c4 = {0.0f}, c5 = {0.0f}, c6 = {0.0f}, c7 = {0.0f};

    for (unsigned int i = 0; i < n_iters; i++) {
#if defined(__gfx1151__) || defined(__gfx1150__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c7);
#else
        c0[0] += (float)a[0] * (float)b[0];
#endif
    }

    // Reduce to one accumulator and sink.
    float8 c = {0.0f};
    for (int i = 0; i < 8; i++) {
        c[i] = c0[i]+c1[i]+c2[i]+c3[i]+c4[i]+c5[i]+c6[i]+c7[i];
    }
    if (lane == 0) {
        const size_t base = (size_t)global_warp_id * 8;
        out[base+0]=c[0]; out[base+1]=c[1]; out[base+2]=c[2]; out[base+3]=c[3];
        out[base+4]=c[4]; out[base+5]=c[5]; out[base+6]=c[6]; out[base+7]=c[7];
    }
}
