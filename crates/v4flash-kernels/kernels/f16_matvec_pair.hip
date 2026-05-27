// f16_matvec_pair — paired F16 matvec sharing one F32 activation across
// two weight matrices (M14h). Used by the V4-Flash compressor where
// `kv = W_kv @ attn_norm` and `gate = W_gate @ attn_norm` are both
// (comp_width=1024, n_embd=4096) F16 matmuls reading the same input.
//
// Two wins over calling f16_matvec twice:
//   1. Activation reads are shared in cache (lane reads x[i] once, used
//      for both kv and gate dot products).
//   2. Single kernel launch + only one kernel-boundary sync.
//   3. Per-lane interleaved FMAs from independent weight streams give
//      the compiler more ILP to hide load latency.
//
// Layout matches f16_matvec wide: 8 rows per workgroup, 1 warp per row.
// Each warp computes both kv[row] and gate[row]. (Same row index for
// both outputs — the kernel processes row r of kv_w and row r of
// gate_w together.)
//
// Vectorized via half2 / float2 loads — 64 elements per lane-iteration
// (down from 32) and 2 independent FMAs per iteration.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define WARP 32u
#define FULL_MASK 0xFFFFFFFFFFFFFFFFull
#define ROWS_PER_BLOCK 8u

__device__ __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;
}

extern "C" __global__ void f16_matvec_pair(
        float          * __restrict__ kv,         // [n_rows]
        float          * __restrict__ gate,       // [n_rows]
        const __half   * __restrict__ kv_w,       // [n_rows, k]
        const __half   * __restrict__ gate_w,     // [n_rows, k]
        const float    * __restrict__ x,          // [k]
        unsigned int     k,
        unsigned int     n_rows) {
    const uint32_t row  = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= n_rows) return;

    const __half *kv_row   = kv_w   + (size_t)row * (size_t)k;
    const __half *gate_row = gate_w + (size_t)row * (size_t)k;

    // 4-way unrolled inner loop: read 4 halves + 4 floats per lane per
    // iteration (uint64 + float4 loads), 8 independent FMAs per output.
    // For k=4096 this gives k/4/32 = 32 loop iterations per lane (vs 64
    // for the simpler half2 version), letting the compiler schedule
    // around half→float conversion latency.
    // Pre-condition: k % 4 == 0 (always true for n_embd=4096).
    const uint64_t *kv_u64   = reinterpret_cast<const uint64_t*>(kv_row);
    const uint64_t *gate_u64 = reinterpret_cast<const uint64_t*>(gate_row);
    const float4   *x_f4     = reinterpret_cast<const float4  *>(x);
    const uint32_t  k4       = k >> 2u;  // # of 4-half (uint64) blocks

    // 4 independent accumulators per output for FMA ILP.
    float kv_a0 = 0.0f, kv_a1 = 0.0f, kv_a2 = 0.0f, kv_a3 = 0.0f;
    float g_a0  = 0.0f, g_a1  = 0.0f, g_a2  = 0.0f, g_a3  = 0.0f;

    for (uint32_t i = lane; i < k4; i += WARP) {
        const uint64_t kv_pack = kv_u64[i];
        const uint64_t g_pack  = gate_u64[i];
        const float4   xv      = x_f4[i];

        const __half2 kv_lo = *reinterpret_cast<const __half2*>(&kv_pack);
        const __half2 kv_hi = *(reinterpret_cast<const __half2*>(&kv_pack) + 1);
        const __half2 g_lo  = *reinterpret_cast<const __half2*>(&g_pack);
        const __half2 g_hi  = *(reinterpret_cast<const __half2*>(&g_pack) + 1);

        const float2 fkv_lo = __half22float2(kv_lo);
        const float2 fkv_hi = __half22float2(kv_hi);
        const float2 fg_lo  = __half22float2(g_lo);
        const float2 fg_hi  = __half22float2(g_hi);

        kv_a0 += fkv_lo.x * xv.x;
        kv_a1 += fkv_lo.y * xv.y;
        kv_a2 += fkv_hi.x * xv.z;
        kv_a3 += fkv_hi.y * xv.w;
        g_a0  += fg_lo.x  * xv.x;
        g_a1  += fg_lo.y  * xv.y;
        g_a2  += fg_hi.x  * xv.z;
        g_a3  += fg_hi.y  * xv.w;
    }

    float kv_acc = (kv_a0 + kv_a1) + (kv_a2 + kv_a3);
    float g_acc  = (g_a0  + g_a1)  + (g_a2  + g_a3);
    kv_acc = warp_sum_f32(kv_acc);
    g_acc  = warp_sum_f32(g_acc);
    if (lane == 0) {
        kv[row]   = kv_acc;
        gate[row] = g_acc;
    }
}

// M50 batched: B independent matvec_pair calls in one launch. Per-batch
// x[B, k] → kv[B, n_rows], gate[B, n_rows]. Weights shared (same
// kv_w / gate_w across all B). Grid (n_rows/ROWS, 1, B).
extern "C" __global__ void f16_matvec_pair_batched(
        float          * __restrict__ kv,         // [B, n_rows]
        float          * __restrict__ gate,       // [B, n_rows]
        const __half   * __restrict__ kv_w,       // [n_rows, k]
        const __half   * __restrict__ gate_w,     // [n_rows, k]
        const float    * __restrict__ x,          // [B, k]
        unsigned int     k,
        unsigned int     n_rows) {
    const uint32_t row  = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t b    = blockIdx.z;
    if (row >= n_rows) return;

    const __half *kv_row   = kv_w   + (size_t)row * (size_t)k;
    const __half *gate_row = gate_w + (size_t)row * (size_t)k;
    const float  *x_b      = x      + (size_t)b   * (size_t)k;

    const uint64_t *kv_u64   = reinterpret_cast<const uint64_t*>(kv_row);
    const uint64_t *gate_u64 = reinterpret_cast<const uint64_t*>(gate_row);
    const float4   *x_f4     = reinterpret_cast<const float4  *>(x_b);
    const uint32_t  k4       = k >> 2u;

    float kv_a0 = 0.0f, kv_a1 = 0.0f, kv_a2 = 0.0f, kv_a3 = 0.0f;
    float g_a0  = 0.0f, g_a1  = 0.0f, g_a2  = 0.0f, g_a3  = 0.0f;

    for (uint32_t i = lane; i < k4; i += WARP) {
        const uint64_t kv_pack = kv_u64[i];
        const uint64_t g_pack  = gate_u64[i];
        const float4   xv      = x_f4[i];

        const __half2 kv_lo = *reinterpret_cast<const __half2*>(&kv_pack);
        const __half2 kv_hi = *(reinterpret_cast<const __half2*>(&kv_pack) + 1);
        const __half2 g_lo  = *reinterpret_cast<const __half2*>(&g_pack);
        const __half2 g_hi  = *(reinterpret_cast<const __half2*>(&g_pack) + 1);

        const float2 fkv_lo = __half22float2(kv_lo);
        const float2 fkv_hi = __half22float2(kv_hi);
        const float2 fg_lo  = __half22float2(g_lo);
        const float2 fg_hi  = __half22float2(g_hi);

        kv_a0 += fkv_lo.x * xv.x;
        kv_a1 += fkv_lo.y * xv.y;
        kv_a2 += fkv_hi.x * xv.z;
        kv_a3 += fkv_hi.y * xv.w;
        g_a0  += fg_lo.x  * xv.x;
        g_a1  += fg_lo.y  * xv.y;
        g_a2  += fg_hi.x  * xv.z;
        g_a3  += fg_hi.y  * xv.w;
    }

    float kv_acc = (kv_a0 + kv_a1) + (kv_a2 + kv_a3);
    float g_acc  = (g_a0  + g_a1)  + (g_a2  + g_a3);
    kv_acc = warp_sum_f32(kv_acc);
    g_acc  = warp_sum_f32(g_acc);
    if (lane == 0) {
        kv  [(size_t)b * (size_t)n_rows + row] = kv_acc;
        gate[(size_t)b * (size_t)n_rows + row] = g_acc;
    }
}
