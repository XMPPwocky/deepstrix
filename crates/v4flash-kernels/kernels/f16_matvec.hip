// f16_matvec — out[r] = sum_i f32(weight_f16[r, i]) * x_f32[i] for r in 0..n_rows.
// Mirrors ds4's `matvec_any` fallback path for F16 weights (the V4 Flash
// compressor + indexer use this — they're all F16, not Q8_0).
//
// Layout: weight tensor is row-major [n_rows, k], each element a `__half`.
// Reading row r col i: weight[r * k + i].
//
// 8 rows per workgroup (256 threads = 8 warps of 32 lanes). Same launch
// geometry as q8_0_matvec to keep the dispatch consistent across the
// crate. F16 → F32 via `__half2float`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define F16_WARP_LANES 32u
#define F16_FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float f16_warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(F16_FULL_WARP_MASK, v, offset);
    }
    return v;
}

extern "C" __global__ void f16_matvec(
        float          * __restrict__ out,        // [n_rows]
        const __half   * __restrict__ weight,     // [n_rows, k]
        const float    * __restrict__ x,          // [k]
        unsigned int     k,
        unsigned int     n_rows) {
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= n_rows) return;

    const __half *wr = weight + (size_t)row * (size_t)k;

    float acc = 0.0f;
    for (uint32_t i = lane; i < k; i += F16_WARP_LANES) {
        acc += __half2float(wr[i]) * x[i];
    }

    acc = f16_warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}
