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

    // Each lane handles k/32 elements; pair into half2/float2 for 2x
    // throughput per memory transaction.
    // Pre-condition: k % 2 == 0 (always true for n_embd = 4096).
    const __half2 *kv_h2   = reinterpret_cast<const __half2 *>(kv_row);
    const __half2 *gate_h2 = reinterpret_cast<const __half2 *>(gate_row);
    const float2  *x_f2    = reinterpret_cast<const float2  *>(x);
    const uint32_t k2 = k >> 1u;  // # of half2 elements

    // Two independent accumulators per output → more ILP.
    float kv_acc0 = 0.0f, kv_acc1 = 0.0f;
    float g_acc0  = 0.0f, g_acc1  = 0.0f;

    for (uint32_t i = lane; i < k2; i += WARP) {
        const __half2 wkv = kv_h2[i];
        const __half2 wg  = gate_h2[i];
        const float2  xv  = x_f2[i];
        const float2  fwkv = __half22float2(wkv);
        const float2  fwg  = __half22float2(wg);
        kv_acc0 += fwkv.x * xv.x;
        kv_acc1 += fwkv.y * xv.y;
        g_acc0  += fwg.x  * xv.x;
        g_acc1  += fwg.y  * xv.y;
    }

    float kv_acc = kv_acc0 + kv_acc1;
    float g_acc  = g_acc0  + g_acc1;
    kv_acc = warp_sum_f32(kv_acc);
    g_acc  = warp_sum_f32(g_acc);
    if (lane == 0) {
        kv[row]   = kv_acc;
        gate[row] = g_acc;
    }
}
