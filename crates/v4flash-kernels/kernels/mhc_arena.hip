// mhc_arena — the arena (multistream decode) mHC MIXES (RMS scalar + matvec +
// Sinkhorn split) in 1 launch per sub-block instead of 3-5, BIT-IDENTICAL to the kernel chains they replace
// (tests/mhc_arena_bitexact.rs). Every value is produced by the same
// per-thread loop, the same shared-memory tree and the same finalize order as
// the original kernel, only inside one launch:
//
// mhc_mix_batched  grid (n_rows = HC_MIX_DIM, 1, B), block 256
//   mode 0 (pre-attn, `mhc_pre_scaled_for`):
//     rms_norm_partial_sum_sq (n_rms_wgs slices) + rms_norm_finalize_inv
//     + f16_matvec_narrow_pre_scaled, per row. The partial kernel's thread t of
//     slice w reads x[w*ws + t + 256 j]; with ws = k / n_rms_wgs a multiple of
//     256 that is exactly an element this WG's thread t reads for the dot, so
//     every WG recomputes the 16 partials (16 trees in lock-step) and the
//     finalize itself.
//   mode 1 (pre-ffn, `mhc_narrow_fallback_for`):
//     rms_norm_no_weight_batched + f16_matvec_narrow_batched: the WG computes
//     the row's scale with rms_nw's single tree and dots W against x * scale on
//     the fly -- the same f32 products the kernel read back from `flat`.
//   Then the LAST WG of each batch row (atomic counter, self-resetting, so
//   graph replays are safe) runs hc_sinkhorn_par_batched verbatim on the row's
//   HC_MIX_DIM values -> split.
//
// (A fused hc_weighted + rms_w kernel was measured SLOWER, 30 vs 23.6 us at
// b=1: one WG per row is less parallel than hc_weighted's 20 WGs. Dropped.)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define MA_BLOCK 256u
#define MA_WARP 32u
#define MA_MAX_WARPS 8u
#define MA_MAX_RMS_WGS 16u
#define MA_HC 4u
#define MA_MIX (2u * MA_HC + MA_HC * MA_HC)   // 24

extern "C" __global__ void mhc_mix_batched(
        float          * __restrict__ split_out,   // [B, MA_MIX]
        float          *              mix_out,     // [B, MA_MIX] (read back by the last WG)
        unsigned int   *              counters,    // [B], zero between launches
        const __half   * __restrict__ weight,      // [n_rows, k]
        const float    * __restrict__ x,           // [B, k]
        const float    * __restrict__ scale,       // [3]
        const float    * __restrict__ base,        // [MA_MIX]
        unsigned int     k,
        unsigned int     n_rows,                   // == MA_MIX
        unsigned int     mode,                     // 0 pre-scaled, 1 normed
        unsigned int     n_rms_wgs,                // mode 0: 16
        float            rms_eps,
        unsigned int     n_hc,                     // == MA_HC
        unsigned int     iters,
        float            eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t b = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (MA_WARP - 1u);
    const uint32_t warp = tid >> 5u;
    const float *xb = x + (size_t)b * (size_t)k;

    __shared__ float sq[MA_MAX_RMS_WGS][MA_BLOCK];
    __shared__ float inv_s;
    __shared__ float per_warp[MA_MAX_WARPS];
    __shared__ unsigned int is_last;

    const __half *wr = weight + (size_t)row * (size_t)k;
    float acc = 0.0f;
    // ---- the row's RMS scalar, exactly as the replaced kernels compute it
    if (mode == 0u) {
        // rms_norm_partial_sum_sq: slice w = [w*ws, (w+1)*ws), thread-strided
        // double accumulation, cast, then a 256-wide tree per slice. ONE pass:
        // with ws a multiple of 256, slice w's j-th element for thread t is
        // x[t + 256*(w*tps + j)], i.e. the dot loop's elements in the same
        // order, so the dot accumulates alongside (f16_matvec_narrow_pre_scaled
        // reads the raw residual).
        const uint32_t ws = k / n_rms_wgs;
        const uint32_t tps = ws / MA_BLOCK;
        for (uint32_t w = 0; w < n_rms_wgs; w++) {
            double partial = 0.0;
            for (uint32_t j = 0; j < tps; j++) {
                const uint32_t i = w * ws + j * MA_BLOCK + tid;
                const float v = xb[i];
                partial += (double)v * (double)v;
                acc += __half2float(wr[i]) * v;
            }
            sq[w][tid] = (float)partial;
        }
        __syncthreads();
        for (uint32_t s = MA_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                for (uint32_t w = 0; w < n_rms_wgs; w++) sq[w][tid] += sq[w][tid + s];
            }
            __syncthreads();
        }
        // rms_norm_finalize_inv: one thread, partials summed in order.
        if (tid == 0) {
            float s = 0.0f;
            for (uint32_t w = 0; w < n_rms_wgs; w++) s += sq[w][0];
            const float mean_sq = s / (float)k;
            inv_s = rsqrtf(mean_sq + rms_eps);
        }
    } else {
        // rms_norm_no_weight_batched: one 256-wide tree over the whole row.
        double partial = 0.0;
        for (uint32_t i = tid; i < k; i += MA_BLOCK) {
            const float v = xb[i];
            partial += (double)v * (double)v;
        }
        sq[0][tid] = (float)partial;
        __syncthreads();
        for (uint32_t s = MA_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sq[0][tid] += sq[0][tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            const float mean_sq = sq[0][0] / (float)k;
            inv_s = rsqrtf(mean_sq + rms_eps);
        }
    }
    __syncthreads();

    // ---- the dot product, exactly as f16_matvec_narrow(_pre_scaled|_batched)
    // (mode 0 accumulated it in the RMS pass above)
    if (mode != 0u) {
        const float sc = inv_s;
        for (uint32_t i = tid; i < k; i += MA_BLOCK) {
            const float f = xb[i] * sc;   // the value rms_nw stored in `flat`
            acc += __half2float(wr[i]) * f;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset, MA_WARP);
    }
    if (lane == 0) per_warp[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < MA_MAX_WARPS) ? per_warp[lane] : 0.0f;
        for (int offset = MA_MAX_WARPS / 2; offset > 0; offset >>= 1) {
            v += __shfl_down(v, offset, MA_WARP);
        }
        if (lane == 0) {
            mix_out[(size_t)b * (size_t)n_rows + row] = (mode == 0u) ? v * inv_s : v;
        }
    }

    // ---- the last WG of this batch row runs the Sinkhorn split
    if (tid == 0) {
        __threadfence();
        const unsigned int prev = atomicAdd(&counters[b], 1u);
        is_last = (prev == n_rows - 1u) ? 1u : 0u;
    }
    __syncthreads();
    if (is_last == 0u) return;
    // Sinkhorn needs 16 threads: keep one wave so its ~80 barriers are
    // single-wave (as in hc_sinkhorn_par_batched's 16-thread block).
    if (tid >= MA_WARP) return;
    __threadfence();
    if (tid == 0) counters[b] = 0u;

    __shared__ float mix_s[MA_MIX];
    if (tid < MA_MIX) {
        const volatile float *mv = mix_out + (size_t)b * (size_t)n_rows;
        mix_s[tid] = mv[tid];
    }
    __syncthreads();

    // hc_sinkhorn_par_batched, verbatim for threads 0..15; threads 16..31 of
    // this wave only take part in the barriers.
    const bool act = tid < 16u && n_hc == MA_HC;
    float *out_b = split_out + (size_t)b * (size_t)MA_MIX;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    if (act) {
        if (tid < MA_HC) {
            const float z = mix_s[tid] * pre_scale + base[tid];
            float sig;
            if (z >= 0.0f) sig = 1.0f / (1.0f + expf(-z));
            else           sig = expf(z) / (1.0f + expf(z));
            out_b[tid] = sig + eps;
        } else if (tid < 2 * MA_HC) {
            const unsigned int off = tid;
            const float z = mix_s[off] * post_scale + base[off];
            float sig;
            if (z >= 0.0f) sig = 1.0f / (1.0f + expf(-z));
            else           sig = expf(z) / (1.0f + expf(z));
            out_b[off] = 2.0f * sig;
        }
    }
    __shared__ float c[16];
    __shared__ float row_sum[MA_HC];
    __shared__ float col_sum[MA_HC];
    const unsigned int src = tid % MA_HC;
    const unsigned int dst = (tid / MA_HC) % MA_HC;
    const unsigned int idx = src + dst * MA_HC;
    if (act) {
        const unsigned int off = 2u * MA_HC + idx;
        c[idx] = mix_s[off] * comb_scale + base[off];
    }
    __syncthreads();
    if (act && tid < MA_HC) {
        float m = -INFINITY;
        for (unsigned int s = 0; s < MA_HC; s++) {
            const float v = c[s + tid * MA_HC];
            if (v > m) m = v;
        }
        row_sum[tid] = m;
    }
    __syncthreads();
    if (act) c[idx] = expf(c[idx] - row_sum[dst]);
    __syncthreads();
    if (act && tid < MA_HC) {
        float s = 0.0f;
        for (unsigned int xx = 0; xx < MA_HC; xx++) s += c[xx + tid * MA_HC];
        row_sum[tid] = s;
    }
    __syncthreads();
    if (act) c[idx] = c[idx] / row_sum[dst] + eps;
    __syncthreads();
    if (act && tid < MA_HC) {
        float s = 0.0f;
        for (unsigned int y = 0; y < MA_HC; y++) s += c[tid + y * MA_HC];
        col_sum[tid] = s;
    }
    __syncthreads();
    if (act) c[idx] = c[idx] / (col_sum[src] + eps);
    __syncthreads();
    for (unsigned int iter = 1; iter < iters; iter++) {
        if (act && tid < MA_HC) {
            float s = 0.0f;
            for (unsigned int xx = 0; xx < MA_HC; xx++) s += c[xx + tid * MA_HC];
            row_sum[tid] = s;
        }
        __syncthreads();
        if (act) c[idx] = c[idx] / (row_sum[dst] + eps);
        __syncthreads();
        if (act && tid < MA_HC) {
            float s = 0.0f;
            for (unsigned int y = 0; y < MA_HC; y++) s += c[tid + y * MA_HC];
            col_sum[tid] = s;
        }
        __syncthreads();
        if (act) c[idx] = c[idx] / (col_sum[src] + eps);
        __syncthreads();
    }
    if (act) out_b[2u * MA_HC + idx] = c[idx];
}
