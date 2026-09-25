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

// Sync for the one-wave tails (after `if (tid >= MA_WARP) return;` only wave 0
// is alive): wavefront-scope fences around a wave barrier order this wave's
// LDS accesses. No workgroup barrier runs after the other waves exit, so
// nothing depends on how exited waves count at `s_barrier` (formally
// undefined in the HIP model); the arithmetic is unchanged.
#define MA_WAVE_SYNC() do { \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); \
        __builtin_amdgcn_wave_barrier(); \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront"); \
    } while (0)

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
    // From here on only wave 0 runs, and it syncs with MA_WAVE_SYNC (wave-level,
    // fully defined) -- never a workgroup barrier after the other waves exit.
    if (tid >= MA_WARP) return;
    __threadfence();
    if (tid == 0) counters[b] = 0u;

    __shared__ float mix_s[MA_MIX];
    if (tid < MA_MIX) {
        const volatile float *mv = mix_out + (size_t)b * (size_t)n_rows;
        mix_s[tid] = mv[tid];
    }
    MA_WAVE_SYNC();

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
    MA_WAVE_SYNC();
    if (act && tid < MA_HC) {
        float m = -INFINITY;
        for (unsigned int s = 0; s < MA_HC; s++) {
            const float v = c[s + tid * MA_HC];
            if (v > m) m = v;
        }
        row_sum[tid] = m;
    }
    MA_WAVE_SYNC();
    if (act) c[idx] = expf(c[idx] - row_sum[dst]);
    MA_WAVE_SYNC();
    if (act && tid < MA_HC) {
        float s = 0.0f;
        for (unsigned int xx = 0; xx < MA_HC; xx++) s += c[xx + tid * MA_HC];
        row_sum[tid] = s;
    }
    MA_WAVE_SYNC();
    if (act) c[idx] = c[idx] / row_sum[dst] + eps;
    MA_WAVE_SYNC();
    if (act && tid < MA_HC) {
        float s = 0.0f;
        for (unsigned int y = 0; y < MA_HC; y++) s += c[tid + y * MA_HC];
        col_sum[tid] = s;
    }
    MA_WAVE_SYNC();
    if (act) c[idx] = c[idx] / (col_sum[src] + eps);
    MA_WAVE_SYNC();
    for (unsigned int iter = 1; iter < iters; iter++) {
        if (act && tid < MA_HC) {
            float s = 0.0f;
            for (unsigned int xx = 0; xx < MA_HC; xx++) s += c[xx + tid * MA_HC];
            row_sum[tid] = s;
        }
        MA_WAVE_SYNC();
        if (act) c[idx] = c[idx] / (row_sum[dst] + eps);
        MA_WAVE_SYNC();
        if (act && tid < MA_HC) {
            float s = 0.0f;
            for (unsigned int y = 0; y < MA_HC; y++) s += c[tid + y * MA_HC];
            col_sum[tid] = s;
        }
        MA_WAVE_SYNC();
        if (act) c[idx] = c[idx] / (col_sum[src] + eps);
        MA_WAVE_SYNC();
    }
    if (act) out_b[2u * MA_HC + idx] = c[idx];
}

// mhc_mix_ksplit_batched -- TIER 2, NOT bit-identical: the same mix + split,
// but each row's dot is split over S = gridDim.x / n_rows K-chunks (one WG
// each) so ~10x more WGs share the work. Chunk partials land in `dotp`
// [B, n_rows, S] and the row's sum-of-squares partials (row-0 WGs only) in
// `sqp` [B, S]; the LAST WG of the batch row sums both in fixed chunk order,
// forms mix = dot * rsqrt(mean_sq + eps) -- dot-then-scale for BOTH sub-blocks
// (decode's formulation; the pre-ffn path normalized first) -- and runs the
// Sinkhorn split on one wave. Deterministic; differs from the chains by
// summation order only (see tests/mhc_arena_bitexact.rs, tier-2 error stats).
//
// MEASURED 2026-09-25, NOT WIRED: 25.0 / 28.5 us at b = 1 / 2 vs 37 for the
// bit-identical mix (gfx1201, graph replay, min) -- ~0.7 ms/step at 4 rows,
// the pre-ffn mixes being off the router path already -- while the split
// lands up to 6e-4 relative off the exact chains (mix 1.9e-7: Sinkhorn
// amplifies ~3000x, cf. KNOWN_BUGS #0b). Not worth a numerics change.
extern "C" __global__ void mhc_mix_ksplit_batched(
        float          * __restrict__ split_out,   // [B, MA_MIX]
        float          *              mix_out,     // [B, MA_MIX]
        float          *              dotp,        // [B, n_rows, S]
        float          *              sqp,         // [B, S]
        unsigned int   *              counters,    // [B]
        const __half   * __restrict__ weight,      // [n_rows, k]
        const float    * __restrict__ x,           // [B, k]
        const float    * __restrict__ scale,
        const float    * __restrict__ base,
        unsigned int     k,
        unsigned int     n_rows,
        unsigned int     n_split,                  // S; k % (S * 256) == 0
        float            rms_eps,
        unsigned int     n_hc,
        unsigned int     iters,
        float            eps) {
    const uint32_t row = blockIdx.x / n_split;
    const uint32_t ks = blockIdx.x % n_split;
    const uint32_t b = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (MA_WARP - 1u);
    const uint32_t warp = tid >> 5u;
    const uint32_t chunk = k / n_split;
    const float *xb = x + (size_t)b * (size_t)k + (size_t)ks * chunk;
    const __half *wr = weight + (size_t)row * (size_t)k + (size_t)ks * chunk;

    __shared__ float red[MA_MAX_WARPS];
    __shared__ double redd[MA_MAX_WARPS];
    __shared__ unsigned int is_last;

    float acc = 0.0f;
    double sq = 0.0;
    for (uint32_t i = tid; i < chunk; i += MA_BLOCK) {
        const float v = xb[i];
        acc += __half2float(wr[i]) * v;
        if (row == 0) sq += (double)v * (double)v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, MA_WARP);
        if (row == 0) sq += __shfl_down(sq, off, MA_WARP);
    }
    if (lane == 0) { red[warp] = acc; redd[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float a = 0.0f;
        double q = 0.0;
        for (uint32_t w = 0; w < MA_MAX_WARPS; w++) { a += red[w]; q += redd[w]; }
        dotp[((size_t)b * n_rows + row) * n_split + ks] = a;
        if (row == 0) sqp[(size_t)b * n_split + ks] = (float)q;
        __threadfence();
        const unsigned int prev = atomicAdd(&counters[b], 1u);
        is_last = (prev == n_rows * n_split - 1u) ? 1u : 0u;
    }
    __syncthreads();
    if (is_last == 0u) return;
    if (tid >= MA_WARP) return;   // one wave on: MA_WAVE_SYNC only, see mhc_mix_batched
    __threadfence();
    if (tid == 0) counters[b] = 0u;

    __shared__ float mix_s[MA_MIX];
    __shared__ float inv_s;
    if (tid == 0) {
        const volatile float *qv = sqp + (size_t)b * n_split;
        float s = 0.0f;
        for (uint32_t c = 0; c < n_split; c++) s += qv[c];
        inv_s = rsqrtf(s / (float)k + rms_eps);
    }
    MA_WAVE_SYNC();
    if (tid < MA_MIX) {
        const volatile float *dv = dotp + ((size_t)b * n_rows + tid) * n_split;
        float d = 0.0f;
        for (uint32_t c = 0; c < n_split; c++) d += dv[c];
        mix_s[tid] = d * inv_s;
        mix_out[(size_t)b * n_rows + tid] = mix_s[tid];
    }
    MA_WAVE_SYNC();

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
    MA_WAVE_SYNC();
    if (act && tid < MA_HC) {
        float m = -INFINITY;
        for (unsigned int s = 0; s < MA_HC; s++) {
            const float v = c[s + tid * MA_HC];
            if (v > m) m = v;
        }
        row_sum[tid] = m;
    }
    MA_WAVE_SYNC();
    if (act) c[idx] = expf(c[idx] - row_sum[dst]);
    MA_WAVE_SYNC();
    if (act && tid < MA_HC) {
        float s = 0.0f;
        for (unsigned int xx = 0; xx < MA_HC; xx++) s += c[xx + tid * MA_HC];
        row_sum[tid] = s;
    }
    MA_WAVE_SYNC();
    if (act) c[idx] = c[idx] / row_sum[dst] + eps;
    MA_WAVE_SYNC();
    if (act && tid < MA_HC) {
        float s = 0.0f;
        for (unsigned int y = 0; y < MA_HC; y++) s += c[tid + y * MA_HC];
        col_sum[tid] = s;
    }
    MA_WAVE_SYNC();
    if (act) c[idx] = c[idx] / (col_sum[src] + eps);
    MA_WAVE_SYNC();
    for (unsigned int iter = 1; iter < iters; iter++) {
        if (act && tid < MA_HC) {
            float s = 0.0f;
            for (unsigned int xx = 0; xx < MA_HC; xx++) s += c[xx + tid * MA_HC];
            row_sum[tid] = s;
        }
        MA_WAVE_SYNC();
        if (act) c[idx] = c[idx] / (row_sum[dst] + eps);
        MA_WAVE_SYNC();
        if (act && tid < MA_HC) {
            float s = 0.0f;
            for (unsigned int y = 0; y < MA_HC; y++) s += c[tid + y * MA_HC];
            col_sum[tid] = s;
        }
        MA_WAVE_SYNC();
        if (act) c[idx] = c[idx] / (col_sum[src] + eps);
        MA_WAVE_SYNC();
    }
    if (act) out_b[2u * MA_HC + idx] = c[idx];
}
