// Candidate: shared-expert fused chain for decode (dGPU, gfx1201).
//
// Production (FP:1847-1901) at b<=8 runs 5 graph nodes per lane-layer:
//   gate = bpack(Wg, xq)   [288 WGs x 8 waves, 12.5 MB cold]
//   up   = bpack(Wu, xq)   [same]
//   mid  = swiglu(gate, up, clamp=10)          [elementwise]
//   (mid_xq, mid_xs) = q8_0_quantize_f32(mid)  [72*b WGs x 32 thr]
//   out  = bpack(Wd, mid_xq)                   [640 WGs, 12.5 MB cold]
//
// This file fuses the first three (A) or the first four (B) into one kernel:
//   A: `shared_gateup_swiglu_tB<B>`      each wave owns ONE row r and streams BOTH Wg[r] and Wu[r]
//      block-by-block (4 x b128 weight loads per block instead of 2), 2B accumulators, and applies
//      the clamped swiglu in the epilogue -> writes mid[b][r] (f32). Same WG geometry as production
//      (8 waves, 288 WGs). Removes the `up` and `swiglu` nodes and the gate/up f32 round trip.
//   B: `shared_gateup_swiglu_q8_tB<B>_r<RPW>`  additionally quantizes mid: a WG owns 32 consecutive
//      rows (= exactly one Q8_0 block of mid per b), with RPW rows per wave and 32/RPW waves per WG
//      (RPW=1: 1024 thr / 72 WGs; RPW=2: 512 thr; RPW=4: 256 thr). The 32 mid values per b go through
//      LDS, wave b reduces amax (fmaxf, order-free -> exact), writes mid_xq[b][r0..r0+31] and
//      mid_xs[b][blockIdx.x]. Removes the `quantize_mid` node too and the f32 mid round trip.
//
// NUMERICS: per (row, b) the GEMV accumulation is the production expression in the production order
// (`bl = lane; bl += 32`, `acc += ws * xs * (float)dot`, same warp_sum tree), swiglu is the
// production formula, the quantize is the production formula (amax via fmaxf: any order is exact;
// d = amax/127; id = 1/d; lrintf(v*id) clamp) -> expected bit-identical mid / mid_xq / mid_xs.
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
// Production swiglu.hip formula (clamp > 1e-6 -> one-sided on gate, two-sided on up).
__device__ __forceinline__ float swiglu_clamped(float g, float u, float clamp) {
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    const float sig = 1.0f / (1.0f + expf(-g));
    return g * sig * u;
}

// Streams RPW gate rows and RPW up rows per wave; writes the swiglu'd mid for each (row, b) to `sink`
// (a functor: either global mid[] or LDS).
template <int B, int RPW, int WAVES, bool QUANT>
__device__ __forceinline__ void shared_body(float* __restrict__ mid, int8_t* __restrict__ mid_xq,
                                            float* __restrict__ mid_xs,
                                            const unsigned char* __restrict__ wg,
                                            const unsigned char* __restrict__ wu,
                                            const int8_t* __restrict__ xq, const float* __restrict__ xscale,
                                            uint32_t in_dim, uint32_t n_ff, uint32_t blocks, float clamp) {
    constexpr int ROWS_PER_WG = RPW * WAVES;
    const uint32_t wave = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t row0 = blockIdx.x * ROWS_PER_WG + wave * RPW;
    __shared__ float mid_s[QUANT ? B : 1][QUANT ? 32 : 1];

    const int8_t* qg[RPW]; const int8_t* qu[RPW];
    const __half* sg[RPW]; const __half* su[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const unsigned char* wrg = wg + (size_t)(row0 + i) * blocks * 34u;
        const unsigned char* wru = wu + (size_t)(row0 + i) * blocks * 34u;
        sg[i] = (const __half*)wrg; qg[i] = (const int8_t*)(wrg + (size_t)blocks * 2u);
        su[i] = (const __half*)wru; qu[i] = (const int8_t*)(wru + (size_t)blocks * 2u);
    }
    float accg[RPW][B], accu[RPW][B];
#pragma unroll
    for (int i = 0; i < RPW; ++i)
#pragma unroll
        for (int b = 0; b < B; ++b) { accg[i][b] = 0.0f; accu[i][b] = 0.0f; }

    if (row0 < n_ff) {  // n_ff % ROWS_PER_WG == 0 at the production shape (2304); whole-wave guard
        for (uint32_t bl = lane; bl < blocks; bl += WARP_LANES) {
            uint4 ag0[RPW], ag1[RPW], au0[RPW], au1[RPW];
            float wsg[RPW], wsu[RPW];
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                __builtin_memcpy(&ag0[i], qg[i] + (size_t)bl * 32u, 16);
                __builtin_memcpy(&ag1[i], qg[i] + (size_t)bl * 32u + 16, 16);
                __builtin_memcpy(&au0[i], qu[i] + (size_t)bl * 32u, 16);
                __builtin_memcpy(&au1[i], qu[i] + (size_t)bl * 32u + 16, 16);
                wsg[i] = __half2float(sg[i][bl]);
                wsu[i] = __half2float(su[i][bl]);
            }
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
            for (int i = 0; i < RPW; ++i) {
#pragma unroll
                for (int b = 0; b < B; ++b) {
                    int32_t dg = dot8(ag0[i], ag1[i], bv0[b], bv1[b]);
                    accg[i][b] += wsg[i] * xs[b] * (float)dg;
                    int32_t du = dot8(au0[i], au1[i], bv0[b], bv1[b]);
                    accu[i][b] += wsu[i] * xs[b] * (float)du;
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float g = warp_sum_f32(accg[i][b]);
            float u = warp_sum_f32(accu[i][b]);
            if (lane == 0) {
                float m = swiglu_clamped(g, u, clamp);
                if constexpr (QUANT) mid_s[b][wave * RPW + i] = m;
                else mid[(size_t)b * n_ff + row0 + i] = m;
            }
        }
    }
    if constexpr (QUANT) {
        __syncthreads();
        // wave w quantizes the 32-row block of batch row w (production q8_0_quantize_f32 math).
        if (wave < (uint32_t)B) {
            const uint32_t b = wave;
            const float v = mid_s[b][lane];
            float amax = fabsf(v);
            for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULL_WARP_MASK, amax, off));
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            int q = (int)lrintf(v * id);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            const uint32_t r0 = blockIdx.x * ROWS_PER_WG;
            mid_xq[(size_t)b * n_ff + r0 + lane] = (int8_t)q;
            if (lane == 0) mid_xs[(size_t)b * (n_ff / 32u) + blockIdx.x] = d;
        }
    }
}

// A: fused gate+up+swiglu, production geometry (8 waves x 1 row, grid n_ff/8), writes mid f32.
#define SHARED_A(NAME, B)                                                                       \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                    \
        float* __restrict__ mid, const unsigned char* __restrict__ wg,                         \
        const unsigned char* __restrict__ wu, const int8_t* __restrict__ xq,                   \
        const float* __restrict__ xscale, uint32_t in_dim, uint32_t n_ff, uint32_t blocks,     \
        float clamp) {                                                                         \
        shared_body<B, 1, 8, false>(mid, nullptr, nullptr, wg, wu, xq, xscale, in_dim, n_ff, blocks, clamp); \
    }
SHARED_A(shared_gateup_swiglu_tB1, 1)
SHARED_A(shared_gateup_swiglu_tB2, 2)
SHARED_A(shared_gateup_swiglu_tB3, 3)
SHARED_A(shared_gateup_swiglu_tB4, 4)
SHARED_A(shared_gateup_swiglu_tB5, 5)
SHARED_A(shared_gateup_swiglu_tB8, 8)

// B: + fused Q8_0 quantize of mid. WG owns 32 rows: RPW rows per wave, 32/RPW waves. grid n_ff/32.
#define SHARED_B(NAME, B, RPW)                                                                  \
    extern "C" __global__ void __launch_bounds__(32 * (32 / RPW)) NAME(                        \
        int8_t* __restrict__ mid_xq, float* __restrict__ mid_xs,                               \
        const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,             \
        const int8_t* __restrict__ xq, const float* __restrict__ xscale, uint32_t in_dim,      \
        uint32_t n_ff, uint32_t blocks, float clamp) {                                         \
        shared_body<B, RPW, 32 / RPW, true>(nullptr, mid_xq, mid_xs, wg, wu, xq, xscale, in_dim, n_ff, blocks, clamp); \
    }
SHARED_B(shared_gateup_swiglu_q8_tB1_r1, 1, 1)
SHARED_B(shared_gateup_swiglu_q8_tB1_r2, 1, 2)
SHARED_B(shared_gateup_swiglu_q8_tB1_r4, 1, 4)
SHARED_B(shared_gateup_swiglu_q8_tB2_r1, 2, 1)
SHARED_B(shared_gateup_swiglu_q8_tB2_r2, 2, 2)
SHARED_B(shared_gateup_swiglu_q8_tB2_r4, 2, 4)
SHARED_B(shared_gateup_swiglu_q8_tB4_r1, 4, 1)
SHARED_B(shared_gateup_swiglu_q8_tB4_r2, 4, 2)
SHARED_B(shared_gateup_swiglu_q8_tB4_r4, 4, 4)
SHARED_B(shared_gateup_swiglu_q8_tB5_r1, 5, 1)
SHARED_B(shared_gateup_swiglu_q8_tB5_r2, 5, 2)
SHARED_B(shared_gateup_swiglu_q8_tB5_r4, 5, 4)
SHARED_B(shared_gateup_swiglu_q8_tB8_r1, 8, 1)
SHARED_B(shared_gateup_swiglu_q8_tB8_r2, 8, 2)
