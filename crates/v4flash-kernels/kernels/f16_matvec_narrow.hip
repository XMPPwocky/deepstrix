// f16_matvec_narrow — f16 matvec variant for "narrow output" shapes
// (typically n_rows < 64, k large). Sibling of f16_matvec which uses
// 8-rows-per-block + 1-warp-per-row and is fast for wide outputs but
// becomes memory-latency-bound for very narrow ones: with n_rows=24
// the original kernel launches only 3 blocks total, and each row's
// dot product is computed by a single warp reading k/32 elements
// serially (deep load-dependence chain).
//
// This variant launches one block per row and lets all `BLOCK_THREADS`
// threads cooperate on that single row's dot product, dramatically
// shrinking the per-thread strided-load chain and increasing CU
// occupancy. Used by mhc_pre_attn / mhc_pre_ffn (n_rows=24, k=16384)
// in V4-Flash, where it cuts wall time several-x.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define F16N_BLOCK_THREADS 256u
#define F16N_WARP 32u
#define F16N_MAX_WARPS 8u  // 256/32

extern "C" __global__ void f16_matvec_narrow(
        float          * __restrict__ out,        // [n_rows]
        const __half   * __restrict__ weight,     // [n_rows, k]
        const float    * __restrict__ x,          // [k]
        unsigned int     k,
        unsigned int     n_rows) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;

    const __half *wr = weight + (size_t)row * (size_t)k;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (F16N_WARP - 1u);
    const uint32_t warp = tid >> 5u;

    float acc = 0.0f;
    for (uint32_t i = tid; i < k; i += F16N_BLOCK_THREADS) {
        acc += __half2float(wr[i]) * x[i];
    }

    // Warp reduce.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, F16N_WARP);
    }

    __shared__ float per_warp[F16N_MAX_WARPS];
    if (lane == 0) per_warp[warp] = acc;
    __syncthreads();

    // First warp does the final reduction across `per_warp`.
    if (warp == 0) {
        float v = (lane < F16N_MAX_WARPS) ? per_warp[lane] : 0.0f;
        for (int offset = F16N_MAX_WARPS / 2; offset > 0; offset >>= 1) {
            v += __shfl_down(v, offset, F16N_WARP);
        }
        if (lane == 0) {
            out[row] = v;
        }
    }
}

// M40-P7: narrow variant for the 2-input matvec pattern. Used by
// `mhc_pre_attn`/`mhc_pre_ffn` in pair-forward where the n_rows=24
// `f16_matvec_two_inputs` wide variant launched only 3 WGs and ran at
// ~1.5% peak (78 µs vs 1.2 µs floor). This narrow form mirrors
// `f16_matvec_narrow` (1 block per row, 256 threads cooperating per
// row) but accumulates two outputs for two input vectors against
// the same W row.
extern "C" __global__ void f16_matvec_narrow_two_inputs(
        float          * __restrict__ out_a,
        float          * __restrict__ out_b,
        const __half   * __restrict__ weight,
        const float    * __restrict__ x_a,
        const float    * __restrict__ x_b,
        unsigned int     k,
        unsigned int     n_rows) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;

    const __half *wr = weight + (size_t)row * (size_t)k;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (F16N_WARP - 1u);
    const uint32_t warp = tid >> 5u;

    float acc_a = 0.0f;
    float acc_b = 0.0f;
    for (uint32_t i = tid; i < k; i += F16N_BLOCK_THREADS) {
        float wv = __half2float(wr[i]);
        acc_a += wv * x_a[i];
        acc_b += wv * x_b[i];
    }

    // Warp reduce both.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_a += __shfl_down(acc_a, offset, F16N_WARP);
        acc_b += __shfl_down(acc_b, offset, F16N_WARP);
    }

    __shared__ float per_warp_a[F16N_MAX_WARPS];
    __shared__ float per_warp_b[F16N_MAX_WARPS];
    if (lane == 0) {
        per_warp_a[warp] = acc_a;
        per_warp_b[warp] = acc_b;
    }
    __syncthreads();

    if (warp == 0) {
        float va = (lane < F16N_MAX_WARPS) ? per_warp_a[lane] : 0.0f;
        float vb = (lane < F16N_MAX_WARPS) ? per_warp_b[lane] : 0.0f;
        for (int offset = F16N_MAX_WARPS / 2; offset > 0; offset >>= 1) {
            va += __shfl_down(va, offset, F16N_WARP);
            vb += __shfl_down(vb, offset, F16N_WARP);
        }
        if (lane == 0) {
            out_a[row] = va;
            out_b[row] = vb;
        }
    }
}
