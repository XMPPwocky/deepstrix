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

// ============================================================================
// f16_matvec_wide_vec — bandwidth-driven B=1 projection matvec (Laguna decode).
//
// Same math as f16_matvec (warp-per-row, f32 accumulate), but the weight row
// is streamed with 128-bit vector loads (uint4 == 8× __half) instead of scalar
// __half loads. Two effects that raise achieved DRAM bandwidth on the
// (read-once, no-reuse) weight stream:
//   1. Coalesced transaction width: 32 lanes × 16 B = 512 B per warp load
//      (vs 32 × 2 B = 64 B for the scalar path) → fewer, fatter DRAM bursts.
//   2. Memory-level parallelism: the inner loop is 4-way unrolled over
//      independent 128-bit loads, so 4 loads per lane are in flight before any
//      is consumed — hides DRAM latency without adding occupancy pressure.
// No LDS: the weights are used exactly once, so staging them through LDS would
// only add a round-trip. The activation x[] is tiny (k·4 B) and stays L1/L2
// resident, reused across the 8 rows of the block and all blocks.
//
// Requires k % 8 == 0 (every Laguna projection K is a multiple of 128) and the
// weight row base 16-B aligned (row*k*2 is a multiple of 16 when k%8==0 and the
// buffer base is device-aligned). Accumulation is grouped in 8s, so the result
// is NOT bit-identical to the scalar path (reduction order differs) but matches
// within f16-dequant tolerance — argmax-preserving.
typedef _Float16 f16x8_t __attribute__((ext_vector_type(8)));

extern "C" __global__ void f16_matvec_wide_vec(
        float          * __restrict__ out,        // [n_rows]
        const __half   * __restrict__ weight,     // [n_rows, k]
        const float    * __restrict__ x,          // [k]
        unsigned int     k,
        unsigned int     n_rows) {
    uint32_t row  = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= n_rows) return;

    const f16x8_t *wr8 = reinterpret_cast<const f16x8_t *>(
        weight + (size_t)row * (size_t)k);
    const uint32_t n8 = k >> 3u;   // number of 8-wide chunks in the row

    float acc = 0.0f;
    uint32_t j = lane;
    // 4-way unrolled main body: 4 independent 128-bit loads in flight per lane.
    for (; j + 96u < n8; j += 128u) {
        f16x8_t w0 = wr8[j];
        f16x8_t w1 = wr8[j + 32u];
        f16x8_t w2 = wr8[j + 64u];
        f16x8_t w3 = wr8[j + 96u];
        const float *x0 = x + ((size_t)j        << 3u);
        const float *x1 = x + ((size_t)(j + 32u) << 3u);
        const float *x2 = x + ((size_t)(j + 64u) << 3u);
        const float *x3 = x + ((size_t)(j + 96u) << 3u);
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            acc += (float)w0[t] * x0[t];
            acc += (float)w1[t] * x1[t];
            acc += (float)w2[t] * x2[t];
            acc += (float)w3[t] * x3[t];
        }
    }
    // Tail: remaining chunks, one 128-bit load per iter.
    for (; j < n8; j += 32u) {
        f16x8_t w = wr8[j];
        const float *xb = x + ((size_t)j << 3u);
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            acc += (float)w[t] * xb[t];
        }
    }

    acc = f16_warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

// Fused pre-scaled variant: identical to f16_matvec but each x[i] is
// implicitly multiplied by a *scalar* `pre_scale` read once from a 1-element
// device buffer. Pairs with a multi-WG RMS that just computes inv_rms
// without writing the applied-vector — the apply pass and its DRAM
// round-trip get eliminated (each output reads residual + scalar instead
// of a separate flat[] buffer).
//   out[r] = pre_scale * sum_i f32(weight[r, i]) * x[i]
// Algebraically identical to f16_matvec(weight, x * pre_scale, ...) but
// without materializing the x * pre_scale buffer.
extern "C" __global__ void f16_matvec_pre_scaled(
        float          * __restrict__ out,        // [n_rows]
        const __half   * __restrict__ weight,     // [n_rows, k]
        const float    * __restrict__ x,          // [k]
        const float    * __restrict__ pre_scale,  // [1]
        unsigned int     k,
        unsigned int     n_rows) {
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= n_rows) return;

    const __half *wr = weight + (size_t)row * (size_t)k;
    // Read the scalar once per WG (broadcast-cached by L1).
    const float s = pre_scale[0];

    float acc = 0.0f;
    for (uint32_t i = lane; i < k; i += F16_WARP_LANES) {
        acc += __half2float(wr[i]) * x[i];
    }

    acc = f16_warp_sum_f32(acc);
    if (lane == 0) out[row] = acc * s;
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
