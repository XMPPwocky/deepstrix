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
