// Shared-expert fused chain for dGPU decode (gfx1201): `shared_gateup_swiglu_q8_tB<B>_r1`,
// B = 1..10.
//
// Production (forward_prefill.rs issue_shared_expert_prefill, b <= 8 dp4a arm) runs 5 graph
// nodes per lane-layer: gate = bpack(Wg, xq); up = bpack(Wu, xq); mid = swiglu(gate, up, 10);
// (mid_xq, mid_xs) = q8_0_quantize_f32(mid); out = bpack(Wd, mid_xq). This kernel fuses the
// first FOUR into one launch: a 1024-thread WG (32 waves) owns 32 consecutive FFN rows (= one
// Q8_0 block of `mid` per batch row); each wave owns one row r and streams BOTH Wg[r] and
// Wu[r] block-by-block (2B accumulators), applies the clamped swiglu in the epilogue, hands
// the 32 mid values per b through LDS, and wave b quantizes block blockIdx.x of batch row b
// to Q8_0 -> writes mid_xq / mid_xs directly for the down GEMV. The f32 gate/up/mid round
// trips and 3 graph nodes (~2.5 us each in-graph) disappear. grid = n_ff/32, block = 1024.
//
// NUMERICS: per (row, b) the GEMV accumulation is the production expression in the
// production order (`bl = lane; bl += 32`, `acc += ws * xs * (float)dot`, same warp_sum
// tree); swiglu is the swiglu.hip formula; the quantize is the q8_0_quantize_f32 formula
// (amax via fmaxf, order-free -> exact; d = amax/127; id = 1/d; lrintf; clamp) ->
// BIT-IDENTICAL mid_xq / mid_xs / down output vs the 5-kernel chain
// (tests/q8_0_tb_bitexact.rs, b = 1..10, saturating AND non-saturating activations; sweep
// review 3 runs x b in {1,2,3,4,5,8}, plus C1_XS_SCALE 0.01 / 0.003 = no clamp hit).
//
// 2026-09-26 sweep (C1_dense_decode/shared_expert_fused_chain, dGPU, cold weights, graph):
// chain b=4 105.9 -> 78.6 us with the tB down (x1.35, reviewer x1.347); b=1 86.1 -> 76.6
// (x1.12); b=3 99.3 -> 77.9 (x1.27); b=5 113 -> 79.8 (x1.42). Of the 27 us at b=4 about
// 4-7 us is the fusion itself, the rest is the tB gate/up/down GEMVs. At b=8 its 100 VGPRs
// leave one 32-wave WG per WGP (72 WGs on 32 WGPs = tail) and the plain tB 5-node chain is
// faster (92.5 vs 100.7 us) -> the wrapper takes this kernel for b <= 5 only.
// Requires n_ff % 32 == 0 (whole-wave row guard; the quantize writes 32 rows per WG
// unguarded) and k % 32 == 0; the wrapper asserts both. Assumes wave32.
// `V41_SHARED_FUSED=0` (forward_prefill.rs) restores the 5-node chain.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define WARP_LANES 32u
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ int32_t sdp4a_sudot(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
__device__ __forceinline__ float swarp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    return v;
}
__device__ __forceinline__ int32_t sdot8(const uint4& a0, const uint4& a1, const uint4& b0, const uint4& b1) {
    int32_t dot = 0;
    dot = sdp4a_sudot((int32_t)a0.x, (int32_t)b0.x, dot);
    dot = sdp4a_sudot((int32_t)a0.y, (int32_t)b0.y, dot);
    dot = sdp4a_sudot((int32_t)a0.z, (int32_t)b0.z, dot);
    dot = sdp4a_sudot((int32_t)a0.w, (int32_t)b0.w, dot);
    dot = sdp4a_sudot((int32_t)a1.x, (int32_t)b1.x, dot);
    dot = sdp4a_sudot((int32_t)a1.y, (int32_t)b1.y, dot);
    dot = sdp4a_sudot((int32_t)a1.z, (int32_t)b1.z, dot);
    dot = sdp4a_sudot((int32_t)a1.w, (int32_t)b1.w, dot);
    return dot;
}
// swiglu.hip formula (clamp > 1e-6 -> one-sided on gate, two-sided on up).
__device__ __forceinline__ float swiglu_clamped(float g, float u, float clamp) {
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    const float sig = 1.0f / (1.0f + expf(-g));
    return g * sig * u;
}

// One row per wave, 32 waves per WG (1024 threads), WG owns rows [32*blockIdx.x, +32).
template <int B>
__device__ __forceinline__ void shared_gateup_swiglu_q8_body(
    int8_t* __restrict__ mid_xq, float* __restrict__ mid_xs,
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
    const int8_t* __restrict__ xq, const float* __restrict__ xscale,
    uint32_t in_dim, uint32_t n_ff, uint32_t blocks, float clamp)
{
    constexpr int ROWS_PER_WG = 32;
    const uint32_t wave = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t row0 = blockIdx.x * ROWS_PER_WG + wave;
    __shared__ float mid_s[B][32];

    const unsigned char* wrg = wg + (size_t)row0 * blocks * 34u;
    const unsigned char* wru = wu + (size_t)row0 * blocks * 34u;
    const __half* sg = (const __half*)wrg;
    const int8_t* qg = (const int8_t*)(wrg + (size_t)blocks * 2u);
    const __half* su = (const __half*)wru;
    const int8_t* qu = (const int8_t*)(wru + (size_t)blocks * 2u);

    float accg[B], accu[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { accg[b] = 0.0f; accu[b] = 0.0f; }

    if (row0 < n_ff) {  // n_ff % 32 == 0 (wrapper asserts): whole-wave guard
        for (uint32_t bl = lane; bl < blocks; bl += WARP_LANES) {
            uint4 ag0, ag1, au0, au1;
            __builtin_memcpy(&ag0, qg + (size_t)bl * 32u, 16);
            __builtin_memcpy(&ag1, qg + (size_t)bl * 32u + 16, 16);
            __builtin_memcpy(&au0, qu + (size_t)bl * 32u, 16);
            __builtin_memcpy(&au1, qu + (size_t)bl * 32u + 16, 16);
            const float wsg = __half2float(sg[bl]);
            const float wsu = __half2float(su[bl]);
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
                int32_t dg = sdot8(ag0, ag1, bv0[b], bv1[b]);
                accg[b] += wsg * xs[b] * (float)dg;
                int32_t du = sdot8(au0, au1, bv0[b], bv1[b]);
                accu[b] += wsu * xs[b] * (float)du;
            }
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        float g = swarp_sum_f32(accg[b]);
        float u = swarp_sum_f32(accu[b]);
        if (lane == 0) mid_s[b][wave] = swiglu_clamped(g, u, clamp);
    }
    __syncthreads();
    // wave b quantizes the 32-row block of batch row b (q8_0_quantize_f32 math).
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

#define SHARED_GATEUP_SWIGLU_Q8(NAME, B)                                                       \
    extern "C" __global__ void __launch_bounds__(1024) NAME(                                   \
        int8_t* __restrict__ mid_xq, float* __restrict__ mid_xs,                               \
        const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,             \
        const int8_t* __restrict__ xq, const float* __restrict__ xscale, uint32_t in_dim,      \
        uint32_t n_ff, uint32_t blocks, float clamp) {                                         \
        shared_gateup_swiglu_q8_body<B>(mid_xq, mid_xs, wg, wu, xq, xscale, in_dim, n_ff, blocks, clamp); \
    }
SHARED_GATEUP_SWIGLU_Q8(shared_gateup_swiglu_q8_tB1_r1, 1)
SHARED_GATEUP_SWIGLU_Q8(shared_gateup_swiglu_q8_tB2_r1, 2)
SHARED_GATEUP_SWIGLU_Q8(shared_gateup_swiglu_q8_tB3_r1, 3)
SHARED_GATEUP_SWIGLU_Q8(shared_gateup_swiglu_q8_tB4_r1, 4)
SHARED_GATEUP_SWIGLU_Q8(shared_gateup_swiglu_q8_tB5_r1, 5)
SHARED_GATEUP_SWIGLU_Q8(shared_gateup_swiglu_q8_tB6_r1, 6)
SHARED_GATEUP_SWIGLU_Q8(shared_gateup_swiglu_q8_tB7_r1, 7)
SHARED_GATEUP_SWIGLU_Q8(shared_gateup_swiglu_q8_tB8_r1, 8)
SHARED_GATEUP_SWIGLU_Q8(shared_gateup_swiglu_q8_tB9_r1, 9)
SHARED_GATEUP_SWIGLU_Q8(shared_gateup_swiglu_q8_tB10_r1, 10)
