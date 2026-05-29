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

// Batched wide variant: grid.z = B. For each batch element b, computes
// out[b, r] = sum_i f32(W[r,i]) * x[b, i]. Weight is shared across the
// batch; x and out are indexed by blockIdx.z. The per-row reduction is the
// identical single-warp shuffle tree as f16_matvec, so each (b, row) output
// is bit-identical to calling f16_matvec once per batch element — only the
// launch count changes (1 launch vs B).
extern "C" __global__ void f16_matvec_batched(
        float          * __restrict__ out,        // [B, n_rows]
        const __half   * __restrict__ weight,     // [n_rows, k]
        const float    * __restrict__ x,          // [B, k]
        unsigned int     k,
        unsigned int     n_rows) {
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= n_rows) return;
    const uint32_t bz = blockIdx.z;

    const __half *wr = weight + (size_t)row * (size_t)k;
    const float  *xb = x + (size_t)bz * (size_t)k;

    float acc = 0.0f;
    for (uint32_t i = lane; i < k; i += F16_WARP_LANES) {
        acc += __half2float(wr[i]) * xb[i];
    }

    acc = f16_warp_sum_f32(acc);
    if (lane == 0) out[(size_t)bz * (size_t)n_rows + row] = acc;
}

// M40-P4.5: 2-wide pair variant. One weight matrix, two input vectors,
// two output vectors. Per-lane: load W[i] once, multiply by x_a[i] and
// x_b[i], accumulate into two scalars. Halves W bandwidth.
//
// NB: this is the OPPOSITE pattern from `f16_matvec_pair` (in a separate
// kernel file). That one shares ONE input across TWO weights (used by
// compressor kv+gate). This new kernel shares ONE weight across TWO
// inputs (used by pair-forward where t0 and t1 multiply against the
// same hc_attn_fn / hc_ffn_fn / ffn_gate_inp weights).
extern "C" __global__ void f16_matvec_two_inputs(
        float          * __restrict__ out_a,      // [n_rows]
        float          * __restrict__ out_b,      // [n_rows]
        const __half   * __restrict__ weight,     // [n_rows, k]
        const float    * __restrict__ x_a,        // [k]
        const float    * __restrict__ x_b,        // [k]
        unsigned int     k,
        unsigned int     n_rows) {
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= n_rows) return;

    const __half *wr = weight + (size_t)row * (size_t)k;

    float acc_a = 0.0f;
    float acc_b = 0.0f;
    for (uint32_t i = lane; i < k; i += F16_WARP_LANES) {
        float wv = __half2float(wr[i]);
        acc_a += wv * x_a[i];
        acc_b += wv * x_b[i];
    }

    acc_a = f16_warp_sum_f32(acc_a);
    acc_b = f16_warp_sum_f32(acc_b);
    if (lane == 0) {
        out_a[row] = acc_a;
        out_b[row] = acc_b;
    }
}
