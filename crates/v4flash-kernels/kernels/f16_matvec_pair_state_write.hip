// f16_matvec_pair_state_write — fused compressor matvec_pair +
// state_write. Replaces two back-to-back direct launches per layer
// (f16_matvec_pair → compressor_state_write) with a single launch.
//
// Original sequence in `forward_layer.rs` for every layer with ratio>0:
//   1. f16_matvec_pair(kv_cur, sc_cur, kv_w, gate_w, attn_input_norm)
//   2. compressor_state_write(state_kv, state_score, kv_cur, sc_cur,
//                              ape, width, row, pos_mod)
//
// Fused: each warp computes one row of the pair matvec, then lane 0
// adds the APE term and writes directly to state_kv / state_score —
// no intermediate kv_cur / sc_cur global memory traffic.
//
// Lane 0 issues the ape global-memory load BEFORE the matvec inner loop
// runs. The matvec loop's k/4/32 FMA-heavy iterations hide the ~200-cycle
// ape load latency. (A version that issued the ape load AFTER warp_sum
// regressed p50 by 2 ms because the load latency landed on the critical
// path of every warp's retirement.)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define WARP 32u
#define FULL_MASK 0xFFFFFFFFFFFFFFFFull
#define ROWS_PER_BLOCK 8u

__device__ __forceinline__ float warp_sum_f32_fused(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;
}

extern "C" __global__ void f16_matvec_pair_state_write(
        float          * __restrict__ state_kv,    // [ratio * width]
        float          * __restrict__ state_score, // [ratio * width]
        const __half   * __restrict__ kv_w,        // [width, k]
        const __half   * __restrict__ gate_w,      // [width, k]
        const __half   * __restrict__ ape,         // [ratio, width]
        const float    * __restrict__ x,           // [k]
        unsigned int     k,
        unsigned int     width,                    // = n_rows of matvec
        unsigned int     row,                      // per-token state row
        unsigned int     pos_mod                   // per-token APE column
) {
    const uint32_t j    = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (j >= width) return;

    // Prefetch ape on lane 0 BEFORE the matvec loop. The 200+ cycle
    // load latency overlaps with the matvec's k/4/32 iterations.
    float ape_val = 0.0f;
    if (lane == 0) {
        ape_val = __half2float(ape[(size_t)pos_mod * (size_t)width + (size_t)j]);
    }

    const __half *kv_row   = kv_w   + (size_t)j * (size_t)k;
    const __half *gate_row = gate_w + (size_t)j * (size_t)k;

    const uint64_t *kv_u64   = reinterpret_cast<const uint64_t*>(kv_row);
    const uint64_t *gate_u64 = reinterpret_cast<const uint64_t*>(gate_row);
    const float4   *x_f4     = reinterpret_cast<const float4  *>(x);
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
    kv_acc = warp_sum_f32_fused(kv_acc);
    g_acc  = warp_sum_f32_fused(g_acc);

    if (lane == 0) {
        const size_t state_off = (size_t)row * (size_t)width + (size_t)j;
        state_kv[state_off]    = kv_acc;
        state_score[state_off] = g_acc + ape_val;
    }
}
