// Candidate: q8_0_gemv_bpack_warp8 with a COMPILE-TIME batch (tB) and an optional K-unroll (ku).
//
// Why: the production kernel's `for (b = 0; b < batch; ++b)` is a runtime loop over a register
// array `acc[16]`; the gfx1201 ISA shows `v_movrels_b32 / v_movreld_b32` (M0-relative register
// indexing) per (block, b), and an `s_wait_loadcnt 0x0` INSIDE the b-loop, so every weight block's
// DRAM loads are followed by `batch` serialised L2 round trips per lane. At b=8 that is what makes
// the same bytes take ~2x the b=1 time. With a compile-time B the b-loop unrolls: all 2B+B
// activation loads are issued together with the weight loads, one wait, no movrel.
//
// NUMERICS: identical expression per (row, b), same `bl = lane; bl += 32` order, same
// `acc[b] += ws * xscale * (float)dot`, same warp_sum tree -> bit-identical to production.
// The K-unroll (KU blocks per lane per iteration) only reorders LOADS; the float accumulation
// order over bl is unchanged (block bl is accumulated before bl+32).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define WARP_LANES 32u
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ int32_t dp4a_sudot(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
__device__ __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    return v;
}
__device__ __forceinline__ int32_t dot8(const uint4& a0, const uint4& a1, const uint4& b0, const uint4& b1) {
    int32_t dot = 0;
    dot = dp4a_sudot((int32_t)a0.x, (int32_t)b0.x, dot);
    dot = dp4a_sudot((int32_t)a0.y, (int32_t)b0.y, dot);
    dot = dp4a_sudot((int32_t)a0.z, (int32_t)b0.z, dot);
    dot = dp4a_sudot((int32_t)a0.w, (int32_t)b0.w, dot);
    dot = dp4a_sudot((int32_t)a1.x, (int32_t)b1.x, dot);
    dot = dp4a_sudot((int32_t)a1.y, (int32_t)b1.y, dot);
    dot = dp4a_sudot((int32_t)a1.z, (int32_t)b1.z, dot);
    dot = dp4a_sudot((int32_t)a1.w, (int32_t)b1.w, dot);
    return dot;
}

// One block `bl` for all B rows: weight loaded once, B activation loads, B dots. Same math order.
template <int B>
__device__ __forceinline__ void bpack_block(float* acc, const int8_t* quants, const __half* scales,
                                            const int8_t* xq, const float* xscale, uint32_t in_dim,
                                            uint32_t blocks, uint32_t bl) {
    uint4 av0, av1;
    __builtin_memcpy(&av0, quants + (size_t)bl * 32u, 16);
    __builtin_memcpy(&av1, quants + (size_t)bl * 32u + 16, 16);
    const float ws = __half2float(scales[bl]);
    uint4 bv0[B], bv1[B];
    float xs[B];
#pragma unroll
    for (int b = 0; b < B; ++b) {
        const int8_t* xqb = xq + (size_t)b * in_dim + (size_t)bl * 32u;
        __builtin_memcpy(&bv0[b], xqb, 16);
        __builtin_memcpy(&bv1[b], xqb + 16, 16);
        xs[b] = xscale[(size_t)b * blocks + bl];
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        int32_t dot = dot8(av0, av1, bv0[b], bv1[b]);
        acc[b] += ws * xs[b] * (float)dot;
    }
}

template <int B, int KU>
__device__ __forceinline__ void bpack_body(float* __restrict__ out, const unsigned char* __restrict__ w,
                                           const int8_t* __restrict__ xq, const float* __restrict__ xscale,
                                           uint32_t in_dim, uint32_t out_dim, uint32_t blocks) {
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char* wr = w + (size_t)row * blocks * 34u;
    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)blocks * 2u);

    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;

    uint32_t bl = lane;
    if constexpr (KU > 1) {
        // KU blocks per lane per iteration: loads for all KU blocks issued together, then the
        // accumulation in the ORIGINAL bl order (bl, bl+32, ...).
        for (; bl + (KU - 1) * WARP_LANES < blocks; bl += KU * WARP_LANES) {
            uint4 av0[KU], av1[KU];
            float ws[KU];
            uint4 bv0[KU][B], bv1[KU][B];
            float xs[KU][B];
#pragma unroll
            for (int u = 0; u < KU; ++u) {
                const uint32_t blu = bl + u * WARP_LANES;
                __builtin_memcpy(&av0[u], quants + (size_t)blu * 32u, 16);
                __builtin_memcpy(&av1[u], quants + (size_t)blu * 32u + 16, 16);
                ws[u] = __half2float(scales[blu]);
#pragma unroll
                for (int b = 0; b < B; ++b) {
                    const int8_t* xqb = xq + (size_t)b * in_dim + (size_t)blu * 32u;
                    __builtin_memcpy(&bv0[u][b], xqb, 16);
                    __builtin_memcpy(&bv1[u][b], xqb + 16, 16);
                    xs[u][b] = xscale[(size_t)b * blocks + blu];
                }
            }
#pragma unroll
            for (int u = 0; u < KU; ++u) {
#pragma unroll
                for (int b = 0; b < B; ++b) {
                    int32_t dot = dot8(av0[u], av1[u], bv0[u][b], bv1[u][b]);
                    acc[b] += ws[u] * xs[u][b] * (float)dot;
                }
            }
        }
    }
    for (; bl < blocks; bl += WARP_LANES) bpack_block<B>(acc, quants, scales, xq, xscale, in_dim, blocks, bl);

#pragma unroll
    for (int b = 0; b < B; ++b) {
        float a = warp_sum_f32(acc[b]);
        if (lane == 0) out[(size_t)b * out_dim + row] = a;
    }
}

// Same argument list as production (batch is ignored; the caller picks the instantiation).
#define BPACK_KERNEL(NAME, B, KU)                                                              \
    extern "C" __global__ void NAME(float* __restrict__ out, const unsigned char* __restrict__ w, \
                                    const int8_t* __restrict__ xq, const float* __restrict__ xscale, \
                                    uint32_t in_dim, uint32_t out_dim, uint32_t blocks, uint32_t batch) { \
        (void)batch;                                                                           \
        bpack_body<B, KU>(out, w, xq, xscale, in_dim, out_dim, blocks);                        \
    }

BPACK_KERNEL(q8_0_gemv_bpack_tB1, 1, 1)
BPACK_KERNEL(q8_0_gemv_bpack_tB2, 2, 1)
BPACK_KERNEL(q8_0_gemv_bpack_tB3, 3, 1)
BPACK_KERNEL(q8_0_gemv_bpack_tB4, 4, 1)
BPACK_KERNEL(q8_0_gemv_bpack_tB5, 5, 1)
BPACK_KERNEL(q8_0_gemv_bpack_tB6, 6, 1)
BPACK_KERNEL(q8_0_gemv_bpack_tB8, 8, 1)
BPACK_KERNEL(q8_0_gemv_bpack_tB10, 10, 1)

BPACK_KERNEL(q8_0_gemv_bpack_tB1_ku2, 1, 2)
BPACK_KERNEL(q8_0_gemv_bpack_tB2_ku2, 2, 2)
BPACK_KERNEL(q8_0_gemv_bpack_tB4_ku2, 4, 2)
BPACK_KERNEL(q8_0_gemv_bpack_tB5_ku2, 5, 2)
BPACK_KERNEL(q8_0_gemv_bpack_tB8_ku2, 8, 2)
BPACK_KERNEL(q8_0_gemv_bpack_tB10_ku2, 10, 2)

BPACK_KERNEL(q8_0_gemv_bpack_tB1_ku4, 1, 4)
BPACK_KERNEL(q8_0_gemv_bpack_tB4_ku4, 4, 4)
BPACK_KERNEL(q8_0_gemv_bpack_tB8_ku4, 8, 4)

// ---------------------------------------------------------------------------------------------
// Grouped twin (wo_a: 8 groups x (1024 x 4096)): same compile-time-B unroll, production args.
template <int B>
__device__ __forceinline__ void gbpack_body(float* __restrict__ out, const unsigned char* __restrict__ w,
                                            const int8_t* __restrict__ xq, const float* __restrict__ xscale,
                                            unsigned group_dim, unsigned rank, unsigned bpg, unsigned n_groups) {
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;
    uint32_t g = row / rank;
    const unsigned char* wr = w + (size_t)row * (size_t)bpg * 34u;
    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)bpg * 2u);
    const uint32_t xq_stride = n_groups * group_dim, xs_stride = n_groups * bpg;
    const int8_t* xq_g = xq + (size_t)g * group_dim;
    const float* xs_g = xscale + (size_t)g * bpg;
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;
    for (uint32_t bl = lane; bl < bpg; bl += WARP_LANES) {
        uint4 av0, av1;
        __builtin_memcpy(&av0, quants + (size_t)bl * 32u, 16);
        __builtin_memcpy(&av1, quants + (size_t)bl * 32u + 16, 16);
        const float ws = __half2float(scales[bl]);
        uint4 bv0[B], bv1[B];
        float xs[B];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const int8_t* xqb = xq_g + (size_t)b * xq_stride + (size_t)bl * 32u;
            __builtin_memcpy(&bv0[b], xqb, 16);
            __builtin_memcpy(&bv1[b], xqb + 16, 16);
            xs[b] = xs_g[(size_t)b * xs_stride + bl];
        }
#pragma unroll
        for (int b = 0; b < B; ++b) {
            int32_t dot = dot8(av0, av1, bv0[b], bv1[b]);
            acc[b] += ws * xs[b] * (float)dot;
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        float a = warp_sum_f32(acc[b]);
        if (lane == 0) out[(size_t)b * (size_t)out_dim + row] = a;
    }
}
#define GBPACK_KERNEL(NAME, B)                                                                 \
    extern "C" __global__ void NAME(float* __restrict__ out, const unsigned char* __restrict__ w, \
                                    const int8_t* __restrict__ xq, const float* __restrict__ xscale, \
                                    unsigned group_dim, unsigned rank, unsigned bpg, unsigned n_groups, unsigned batch) { \
        (void)batch;                                                                           \
        gbpack_body<B>(out, w, xq, xscale, group_dim, rank, bpg, n_groups);                    \
    }
GBPACK_KERNEL(q8_0_grouped_gemv_bpack_tB1, 1)
GBPACK_KERNEL(q8_0_grouped_gemv_bpack_tB2, 2)
GBPACK_KERNEL(q8_0_grouped_gemv_bpack_tB4, 4)
GBPACK_KERNEL(q8_0_grouped_gemv_bpack_tB5, 5)
GBPACK_KERNEL(q8_0_grouped_gemv_bpack_tB8, 8)

// ---------------------------------------------------------------------------------------------
// Quantize candidate: one WAVE per 32-element block (lane i owns element i), 8 blocks per WG.
// Production launches (blocks) x 32 threads where EVERY thread serially loops all 32 elements
// (32x redundant loads/stores per block). amax via fmaxf is order-independent (exact), d and
// id identical, lrintf identical -> bit-identical xq / xscale.
extern "C" __global__ void q8_0_quantize_f32_wave(int8_t* __restrict__ xq, float* __restrict__ xscale,
                                                  const float* __restrict__ x, uint32_t blocks) {
    uint32_t b = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (b >= blocks) return;
    float v = x[(size_t)b * 32u + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULL_WARP_MASK, amax, off));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0) xscale[b] = d;
    int q = (int)lrintf(v * id);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    xq[(size_t)b * 32u + lane] = (int8_t)q;
}
