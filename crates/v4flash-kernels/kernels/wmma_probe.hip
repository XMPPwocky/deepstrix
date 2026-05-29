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
typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

// RDNA3/3.5 (gfx11xx) and RDNA4 (gfx12xx) use different WMMA intrinsics.
// RDNA3 packs the K=16 operand as 16 halfs/lane (replicated across the two
// halves of the w32 wave); RDNA4 dropped that replication, so its A/B
// fragments are 8 halfs/lane (verified by compiling against ROCm 7.2.3
// clang — the gfx12 builtin rejects a 16-wide vector). Op count is the
// same 16×16×16 = 8192 flops either way.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
#define WMMA_RDNA3 1
#elif defined(__gfx1200__) || defined(__gfx1201__)
#define WMMA_RDNA4 1
#endif

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

    // Load fragments — every lane loads the same elements. In a real
    // WMMA op the load is interpreted per-fragment-mapping; here we
    // just need values in registers so the compiler emits the WMMA insn.
    // RDNA4 A/B fragments are 8 halfs/lane; RDNA3 are 16 (replicated).
#if defined(WMMA_RDNA4)
    half8 a, b;
    for (int i = 0; i < 8; i++) { a[i] = a_in[i]; b[i] = b_in[i]; }
#else
    half16 a, b;
    for (int i = 0; i < 16; i++) { a[i] = a_in[i]; b[i] = b_in[i]; }
#endif
    float8 c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // The dependency chain: each iter feeds c back into itself. This
    // measures issue rate per warp, not parallel throughput. To measure
    // parallel throughput, increase n_warps_per_block + total WGs.
    for (unsigned int i = 0; i < n_iters; i++) {
#if defined(WMMA_RDNA4)
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#elif defined(WMMA_RDNA3)
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
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

#if defined(WMMA_RDNA4)
    half8 a, b;
    for (int i = 0; i < 8; i++) { a[i] = a_in[i]; b[i] = b_in[i]; }
#else
    half16 a, b;
    for (int i = 0; i < 16; i++) { a[i] = a_in[i]; b[i] = b_in[i]; }
#endif
    float8 c0 = {0.0f}, c1 = {0.0f}, c2 = {0.0f}, c3 = {0.0f};
    float8 c4 = {0.0f}, c5 = {0.0f}, c6 = {0.0f}, c7 = {0.0f};

    for (unsigned int i = 0; i < n_iters; i++) {
#if defined(WMMA_RDNA4)
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c7);
#elif defined(WMMA_RDNA3)
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

// IU8 (int8 -> int32) WMMA throughput. Same parallel-accumulator
// pattern as wmma_f16_throughput_probe_parallel. Per-op count:
// 16*16*16 = 4096 muls + 4096 adds = 8192 int-ops.
typedef int8_t  i8x16  __attribute__((ext_vector_type(16)));
typedef int8_t  i8x8   __attribute__((ext_vector_type(8)));
typedef int     i32x8 __attribute__((ext_vector_type(8)));

extern "C" __global__ void wmma_iu8_throughput_probe_parallel(
        int          * __restrict__ out,
        const int8_t * __restrict__ a_in,
        const int8_t * __restrict__ b_in,
        unsigned int   n_iters) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp_id_in_block = threadIdx.x >> 5u;
    const unsigned int n_warps_per_block = blockDim.x >> 5u;
    const unsigned int global_warp_id =
        blockIdx.x * n_warps_per_block + warp_id_in_block;

    // RDNA4 int8 A/B fragments are 8 int8/lane (packed as 2×int32);
    // RDNA3 are 16/lane. Op count is the same 16×16×16.
#if defined(WMMA_RDNA4)
    i8x8 a, b;
    for (int i = 0; i < 8; i++) { a[i] = a_in[i]; b[i] = b_in[i]; }
#else
    i8x16 a, b;
    for (int i = 0; i < 16; i++) { a[i] = a_in[i]; b[i] = b_in[i]; }
#endif
    i32x8 c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    i32x8 c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};

    // IU8: 6 args (A-signedness, A, B-signedness, B, C, clamp). false =
    // signed interpretation for both operands.
    for (unsigned int i = 0; i < n_iters; i++) {
#if defined(WMMA_RDNA4)
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c3, false);
        c4 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c4, false);
        c5 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c5, false);
        c6 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c6, false);
        c7 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c7, false);
#elif defined(WMMA_RDNA3)
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c3, false);
        c4 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c4, false);
        c5 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c5, false);
        c6 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c6, false);
        c7 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c7, false);
#else
        c0[0] += (int)a[0] * (int)b[0];
#endif
    }

    i32x8 c = {0};
    for (int i = 0; i < 8; i++) {
        c[i] = c0[i]+c1[i]+c2[i]+c3[i]+c4[i]+c5[i]+c6[i]+c7[i];
    }
    if (lane == 0) {
        const size_t base = (size_t)global_warp_id * 8;
        out[base+0]=c[0]; out[base+1]=c[1]; out[base+2]=c[2]; out[base+3]=c[3];
        out[base+4]=c[4]; out[base+5]=c[5]; out[base+6]=c[6]; out[base+7]=c[7];
    }
}

// ============================================================================
// Non-WMMA baseline: plain vector-ALU FMA throughput. This is the path the
// attention/dense kernels use today (scalar f32). Measuring it gives the
// honest "what does WMMA buy over what we run now" denominator.
//
// Each lane keeps FMA_ACC independent accumulator chains so the SIMD pipeline
// stays full (hides FMA latency) and the compiler is free to dual-issue
// (VOPD) where it can. flops = n_warps × 32 lanes × n_iters × FMA_ACC × (mul+add).
// ============================================================================
#define FMA_ACC 32

extern "C" __global__ void fma_f32_throughput_probe(
        float       * __restrict__ out,
        const float * __restrict__ a_in,
        const float * __restrict__ b_in,
        unsigned int  n_iters) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int gwarp = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);

    const float b = b_in[0];
    float acc[FMA_ACC];
    #pragma unroll
    for (int k = 0; k < FMA_ACC; k++) acc[k] = a_in[0] + (float)k;

    // Plain mul+add (NOT fmaf): hipcc -O3 contracts this to one hardware
    // v_fma_f32. Calling fmaf() can route through OCML's correctly-rounded
    // software fma (a libcall) and tanks throughput ~10×.
    for (unsigned int i = 0; i < n_iters; i++) {
        #pragma unroll
        for (int k = 0; k < FMA_ACC; k++) acc[k] = acc[k] * b + b;
    }

    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < FMA_ACC; k++) s += acc[k];
    if (lane == 0) out[gwarp] = s;
}

// Packed f16x2 FMA (v_pk_fma_f16): 2 elements per op → 2× the f32 flop rate
// when the data is half. flops = ... × FMA_ACC × 2 elems × (mul+add).
typedef _Float16 half2v __attribute__((ext_vector_type(2)));

extern "C" __global__ void fma_f16x2_throughput_probe(
        float          * __restrict__ out,
        const _Float16 * __restrict__ a_in,
        const _Float16 * __restrict__ b_in,
        unsigned int     n_iters) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int gwarp = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);

    const half2v b = {b_in[0], b_in[0]};
    half2v acc[FMA_ACC];
    #pragma unroll
    for (int k = 0; k < FMA_ACC; k++) acc[k] = half2v{a_in[0], a_in[0]} + (_Float16)k;

    for (unsigned int i = 0; i < n_iters; i++) {
        #pragma unroll
        for (int k = 0; k < FMA_ACC; k++) acc[k] = acc[k] * b + b;
    }

    _Float16 s = 0.0f;
    #pragma unroll
    for (int k = 0; k < FMA_ACC; k++) s += acc[k][0] + acc[k][1];
    if (lane == 0) out[gwarp] = (float)s;
}
