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

// Pre-scaled narrow variant — out[r] = pre_scale[0] * sum_i W[r,i] * x[i].
// Folds the rms_norm apply step into the matvec: the input vector is
// implicitly multiplied by a scalar (read once at the end) instead of
// reading a pre-multiplied flat[] buffer. Eliminates one full N=k DRAM
// round-trip and one kernel launch from the mhc_pre chain.
extern "C" __global__ void f16_matvec_narrow_pre_scaled(
        float          * __restrict__ out,        // [n_rows]
        const __half   * __restrict__ weight,     // [n_rows, k]
        const float    * __restrict__ x,          // [k]
        const float    * __restrict__ pre_scale,  // [1]
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
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, F16N_WARP);
    }

    __shared__ float per_warp[F16N_MAX_WARPS];
    if (lane == 0) per_warp[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < F16N_MAX_WARPS) ? per_warp[lane] : 0.0f;
        for (int offset = F16N_MAX_WARPS / 2; offset > 0; offset >>= 1) {
            v += __shfl_down(v, offset, F16N_WARP);
        }
        if (lane == 0) {
            out[row] = v * pre_scale[0];
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

// === K-split narrow matvec ==============================================
// For narrow M (e.g., HC_MIX_DIM=24, HC_DIM=16384). Each WG processes
// ALL n_rows outputs for a K-slice. Eliminates the 24× redundant x reads
// of f16_matvec_narrow (which had Grid(n_rows,1,1) each WG re-reading
// the full x). x is read ONCE total (each WG reads a unique K-chunk).
// ATT on the legacy narrow showed 89% wave time waiting on global loads;
// k-split converts those to one cooperative LDS stage per K-chunk.
//
//   pass 1 partial: grid (n_k_split, 1, 1). Each WG stages K_chunk
//     elements of x in LDS, then computes ALL n_rows outputs against
//     that x. Writes partials[(k_chunk_idx, row)].
//   pass 2 reduce + pre_scale: sums n_k_split partials per row,
//     multiplies by pre_scale[0] scalar.
//
// Constraints: k % n_k_split == 0, k_chunk ≤ 1024 (LDS 4 KB), n_rows ≤ 32.

#define KMV_BLOCK 256u
#define KMV_LANES 32u
#define KMV_WAVES 8u

__device__ __forceinline__ float kmv_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down(v, o, KMV_LANES);
    }
    return v;
}

extern "C" __global__ void f16_matvec_narrow_ksplit_partial(
        float          * __restrict__ partials,  // [n_k_split, n_rows]
        const __half   * __restrict__ weight,    // [n_rows, k_total]
        const float    * __restrict__ x,         // [k_total]
        unsigned int     k_total,
        unsigned int     k_chunk,
        unsigned int     n_rows) {
    __shared__ float s_x[1024];

    const uint32_t tid          = threadIdx.x;
    const uint32_t lane         = tid & (KMV_LANES - 1u);
    const uint32_t wave         = tid >> 5u;
    const uint32_t k_chunk_idx  = blockIdx.x;
    const uint32_t k_start      = k_chunk_idx * k_chunk;

    for (uint32_t i = tid; i < k_chunk; i += KMV_BLOCK) {
        s_x[i] = x[k_start + i];
    }
    __syncthreads();

    const uint32_t out_per_wave = (n_rows + KMV_WAVES - 1u) / KMV_WAVES;
    for (uint32_t oi = 0; oi < out_per_wave; oi++) {
        const uint32_t row = wave * out_per_wave + oi;
        if (row >= n_rows) break;
        const __half *wr = weight + (size_t)row * (size_t)k_total + (size_t)k_start;
        float acc = 0.0f;
        for (uint32_t k = lane; k < k_chunk; k += KMV_LANES) {
            acc += __half2float(wr[k]) * s_x[k];
        }
        acc = kmv_warp_sum(acc);
        if (lane == 0) {
            partials[(size_t)k_chunk_idx * (size_t)n_rows + (size_t)row] = acc;
        }
    }
}

extern "C" __global__ void f16_matvec_narrow_ksplit_reduce_pre_scaled(
        float          * __restrict__ out,
        const float    * __restrict__ partials,
        const float    * __restrict__ pre_scale,
        unsigned int     n_k_split,
        unsigned int     n_rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    float acc = 0.0f;
    for (uint32_t kc = 0; kc < n_k_split; kc++) {
        acc += partials[(size_t)kc * (size_t)n_rows + (size_t)row];
    }
    out[row] = acc * pre_scale[0];
}

extern "C" __global__ void f16_matvec_narrow_ksplit_reduce(
        float          * __restrict__ out,
        const float    * __restrict__ partials,
        unsigned int     n_k_split,
        unsigned int     n_rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    float acc = 0.0f;
    for (uint32_t kc = 0; kc < n_k_split; kc++) {
        acc += partials[(size_t)kc * (size_t)n_rows + (size_t)row];
    }
    out[row] = acc;
}

// M50 Phase 2: batched narrow f16 matvec. Grid (n_rows, 1, B), block (256).
// Per-batch x[B, k], out[B, n_rows]. Weight shared.
extern "C" __global__ void f16_matvec_narrow_batched(
        float          * __restrict__ out,        // [B, n_rows]
        const __half   * __restrict__ weight,     // [n_rows, k]
        const float    * __restrict__ x,          // [B, k]
        unsigned int     k,
        unsigned int     n_rows) {
    const uint32_t row = blockIdx.x;
    const uint32_t b = blockIdx.z;
    if (row >= n_rows) return;

    const __half *wr = weight + (size_t)row * (size_t)k;
    const float  *xb = x + (size_t)b * (size_t)k;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (F16N_WARP - 1u);
    const uint32_t warp = tid >> 5u;

    float acc = 0.0f;
    for (uint32_t i = tid; i < k; i += F16N_BLOCK_THREADS) {
        acc += __half2float(wr[i]) * xb[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, F16N_WARP);
    }
    __shared__ float per_warp[F16N_MAX_WARPS];
    if (lane == 0) per_warp[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < F16N_MAX_WARPS) ? per_warp[lane] : 0.0f;
        for (int offset = F16N_MAX_WARPS / 2; offset > 0; offset >>= 1) {
            v += __shfl_down(v, offset, F16N_WARP);
        }
        if (lane == 0) {
            out[(size_t)b * (size_t)n_rows + row] = v;
        }
    }
}
