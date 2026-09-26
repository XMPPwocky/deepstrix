// mhc_fast -- generation 2 of the arena mHC sub-block (`MhcArena::launch_fast`,
// knob V41_MHC_FAST): the MIX (RMS scalar + [24, K] f16 matvec + Sinkhorn
// split) AND the COLLAPSE (hc_weighted_sum + rms_norm_weighted) in ONE launch,
// every value BIT-IDENTICAL to the kernels it replaces:
//   mix      : mhc_mix_batched (kernels/mhc_arena.hip), both modes -- itself
//              bit-identical to the rms_nw(_mw) + f16 matvec + sinkhorn chains;
//   collapse : hc_weighted_sum_batched + rms_norm_weighted_batched;
//   carry    : the `carry := split` D2D memcpy node.
// (tests/mhc_arena_bitexact.rs::mhc_fast_is_bit_identical, gfx1201 + gfx1151.)
//
// Grid (n_mix + do_collapse + rms_wg, 1, B), block 256, rms_wg = (n_mix > 0 &&
// mode == 0). Roles per batch row b:
//   blockIdx.x <  n_mix (24): mix row. Mode 1 (normed): exactly
//                  mhc_mix_batched's WG. Mode 0 (pre-scaled): only its dot --
//                  the raw dot goes to mix_out and the tail scales it (the same
//                  `v * inv_s` float op the WG did), so the 24 WGs no longer
//                  each recompute the row's fp64 sum of squares (80 DFMAs per
//                  thread at 1/32 rate + a 16-slice tree: most of the WG's time,
//                  and 24x redundant; at b >= 2 two WGs share a WGP and it
//                  doubles);
//   blockIdx.x == n_mix     : the collapse (one WG, exactly the
//                             rms_norm_weighted_batched WG, fed by the
//                             hc_weighted_sum values it computes itself);
//   last (mode 0 + mix)     : the RMS WG: rms_norm_partial_sum_sq's 16 slices +
//                             rms_norm_finalize_inv, exactly as
//                             mhc_mix_batched computed them -> inv_rows[b].
// Either the mix or the collapse can be absent (n_mix = 0: collapse only;
// do_collapse = 0: mix only). When the mix is present, the LAST of the row's
// WGs runs the Sinkhorn tail (atomic counter, self-resetting, so graph replays
// are safe) and writes the split to `split_out` and, if write_carry, to
// `carry`. V4.1 collapses with the PREVIOUS sub-block's pre-weights (the
// carry): the collapse WG reads them before its counter increment (release),
// the tail writes the new carry only after the last increment (acquire), so
// the collapse always sees the OLD carry -- the order the chain had
// (collapse, then carry := split).
//
// ROOFLINE (K = 20480, V4.1): per row the mix reads W [24, K] f16 = 0.98 MB
// (shared across rows) + x [K] f32 = 80 KB (L2-shared by the 24 WGs); the
// collapse reads 4 x 5120 f32 + the norm weight = 100 KB. ~1.1 MB at b = 1 =
// 1.7 us at 640 GB/s. The replaced kernels are LATENCY-bound, not bandwidth-
// bound (tests/bench_decode_latency_breakdown.rs, gfx1201, per graph node):
// mhc_mix_batched ~25 us (pre-scaled) / ~35 us (normed) because each thread
// issues its 80 (x, w) loads one loop iteration at a time; rms_norm_weighted
// ~10 us for its 20; each extra node ~2.2 us of inter-node gap. Here each
// thread issues ALL its loads up front (fully unrolled register arrays) and
// then runs the old per-thread arithmetic in the old order -- the same
// expressions, the same trees, the same finalize -- so only the schedule
// changes; the pre-scaled RMS moves to its own WG (see the roles above).
// MEASURED (tests/bench_decode_latency_ab.rs, gfx1201, graph replay, 1000
// paired rounds, b = 1 / 2 / 3): g.mhc_pre_attn 51.8 -> 22.8 / 23.6 / 24.2 us,
// g.mhc_pre_ffn 24.7 -> 15.9, g.mhc_mix_ffn_late 49.6 -> 22.7 / 26.2 / 29.7;
// one lane-layer -65 / -60 / -57 us. Of the ~23 us pre-attn stage ~11 is the
// event pair + graph launch; the kernel is ~11 us (loads, the RMS WG's fp64
// sum at 1/32 rate + 16-slice tree, the ~3 us Sinkhorn tail). The normed mix
// (mix_ffn_late) still recomputes the row RMS in every WG (its dot needs the
// scale first) and grows with b as two WGs share a WGP; it runs off the router
// path.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define MF_BLOCK 256u
#define MF_WARP 32u
#define MF_MAX_WARPS 8u
#define MF_RMS_WGS 16u
#define MF_HC 4u
#define MF_MIX (2u * MF_HC + MF_HC * MF_HC)   // 24

#ifndef MHC_HC_DIM
#define MHC_HC_DIM 16384
#endif
#ifndef MHC_N_EMBD
#define MHC_N_EMBD 4096
#endif
#define MF_K   ((unsigned int)MHC_HC_DIM)
#define MF_NE  ((unsigned int)MHC_N_EMBD)
#define MF_PT  (MF_K / MF_BLOCK)            // mix elements per thread (80 at V4.1)
#define MF_PE  (MF_NE / MF_BLOCK)           // collapse outputs per thread (20 at V4.1)
#define MF_TPS (MF_PT / MF_RMS_WGS)         // pre-scaled: elements per thread per slice (5)
#if (MHC_HC_DIM % (256 * 16)) != 0 || (MHC_N_EMBD % 256) != 0 || (MHC_HC_DIM != 4 * MHC_N_EMBD)
#error "mhc_fast: K must be a multiple of 4096 and K == 4 * N_EMBD"
#endif

// Wave-level sync for the one-wave Sinkhorn tail (see mhc_arena.hip).
#define MF_WAVE_SYNC() do { \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); \
        __builtin_amdgcn_wave_barrier(); \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront"); \
    } while (0)

// Sinkhorn split of one row: hc_sinkhorn_par_batched verbatim, as in
// mhc_mix_batched's tail, plus the optional carry copy of every output.
// `inv` (mode 0): the row's RMS scalar; the mix values are raw dots, scaled
// here with the same `v * inv_s` the mix WG used to do, and written back to
// `mix_row` (the chain's `mix` holds scaled values). Only wave 0 calls this,
// all 32 lanes.
__device__ __forceinline__ static void mf_sinkhorn_tail(
        const unsigned int tid, float *mix_row, const float *inv, float *out_b, float *carry_b,
        const float *scale, const float *base, unsigned int n_hc, unsigned int iters, float eps) {
    __shared__ float mix_s[MF_MIX];
    if (tid < MF_MIX) {
        volatile float *mv = mix_row;
        float v = mv[tid];
        if (inv != nullptr) {
            const float inv_s = *(const volatile float *)inv;
            v = v * inv_s;
            mv[tid] = v;
        }
        mix_s[tid] = v;
    }
    MF_WAVE_SYNC();
    const bool act = tid < 16u && n_hc == MF_HC;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    if (act) {
        if (tid < MF_HC) {
            const float z = mix_s[tid] * pre_scale + base[tid];
            float sig;
            if (z >= 0.0f) sig = 1.0f / (1.0f + expf(-z));
            else           sig = expf(z) / (1.0f + expf(z));
            const float o = sig + eps;
            out_b[tid] = o;
            if (carry_b) carry_b[tid] = o;
        } else if (tid < 2 * MF_HC) {
            const unsigned int off = tid;
            const float z = mix_s[off] * post_scale + base[off];
            float sig;
            if (z >= 0.0f) sig = 1.0f / (1.0f + expf(-z));
            else           sig = expf(z) / (1.0f + expf(z));
            const float o = 2.0f * sig;
            out_b[off] = o;
            if (carry_b) carry_b[off] = o;
        }
    }
    __shared__ float c[16];
    __shared__ float row_sum[MF_HC];
    __shared__ float col_sum[MF_HC];
    const unsigned int src = tid % MF_HC;
    const unsigned int dst = (tid / MF_HC) % MF_HC;
    const unsigned int idx = src + dst * MF_HC;
    if (act) {
        const unsigned int off = 2u * MF_HC + idx;
        c[idx] = mix_s[off] * comb_scale + base[off];
    }
    MF_WAVE_SYNC();
    if (act && tid < MF_HC) {
        float m = -INFINITY;
        for (unsigned int s = 0; s < MF_HC; s++) {
            const float v = c[s + tid * MF_HC];
            if (v > m) m = v;
        }
        row_sum[tid] = m;
    }
    MF_WAVE_SYNC();
    if (act) c[idx] = expf(c[idx] - row_sum[dst]);
    MF_WAVE_SYNC();
    if (act && tid < MF_HC) {
        float s = 0.0f;
        for (unsigned int xx = 0; xx < MF_HC; xx++) s += c[xx + tid * MF_HC];
        row_sum[tid] = s;
    }
    MF_WAVE_SYNC();
    if (act) c[idx] = c[idx] / row_sum[dst] + eps;
    MF_WAVE_SYNC();
    if (act && tid < MF_HC) {
        float s = 0.0f;
        for (unsigned int y = 0; y < MF_HC; y++) s += c[tid + y * MF_HC];
        col_sum[tid] = s;
    }
    MF_WAVE_SYNC();
    if (act) c[idx] = c[idx] / (col_sum[src] + eps);
    MF_WAVE_SYNC();
    for (unsigned int iter = 1; iter < iters; iter++) {
        if (act && tid < MF_HC) {
            float s = 0.0f;
            for (unsigned int xx = 0; xx < MF_HC; xx++) s += c[xx + tid * MF_HC];
            row_sum[tid] = s;
        }
        MF_WAVE_SYNC();
        if (act) c[idx] = c[idx] / (row_sum[dst] + eps);
        MF_WAVE_SYNC();
        if (act && tid < MF_HC) {
            float s = 0.0f;
            for (unsigned int y = 0; y < MF_HC; y++) s += c[tid + y * MF_HC];
            col_sum[tid] = s;
        }
        MF_WAVE_SYNC();
        if (act) c[idx] = c[idx] / (col_sum[src] + eps);
        MF_WAVE_SYNC();
    }
    if (act) {
        out_b[2u * MF_HC + idx] = c[idx];
        if (carry_b) carry_b[2u * MF_HC + idx] = c[idx];
    }
}

extern "C" __global__ __launch_bounds__(MF_BLOCK) void mhc_fast_batched(
        float          *              split_out,   // [B, 24]            (mix)
        float          *              mix_out,     // [B, 24]            (mix; read back by the tail)
        unsigned int   *              counters,    // [B], zero between launches (mix)
        float          *              inv_rows,    // [B] pre-scaled RMS scalar (mode 0 + mix)
        const __half   * __restrict__ weight,      // [24, K]            (mix)
        const float    * __restrict__ x,           // [B, K] mix input
        const float    * __restrict__ scale,       // [3]
        const float    * __restrict__ base,        // [24]
        float          *              carry,       // [B, 24]: read by the collapse, written by the tail if write_carry
        const float    * __restrict__ cx,          // [B, K] collapse input (4 copies x n_embd)
        float          * __restrict__ cur_out,     // [B, n_embd] hc_weighted_sum output
        float          * __restrict__ norm_out,    // [B, n_embd] rms_norm_weighted output
        const float    * __restrict__ norm_w,      // [n_embd]
        unsigned int     k,                        // == MF_K (runtime, as the replaced kernels divide by it)
        unsigned int     ne,                       // == MF_NE (idem)
        unsigned int     n_mix,                    // 24 or 0
        unsigned int     do_collapse,              // 0 / 1
        unsigned int     write_carry,              // 0 / 1
        unsigned int     mode,                     // 0 pre-scaled, 1 normed
        float            rms_eps,
        unsigned int     n_hc,
        unsigned int     iters,
        float            eps) {
    const uint32_t role = blockIdx.x;
    const uint32_t b = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (MF_WARP - 1u);
    const uint32_t warp = tid >> 5u;

    __shared__ float sq[MF_RMS_WGS][MF_BLOCK];
    __shared__ float inv_s;
    __shared__ float per_warp[MF_MAX_WARPS];
    __shared__ unsigned int is_last;

    if (role < n_mix) {
        // ------------------------------------------------ mix row `role`
        const float *xb = x + (size_t)b * (size_t)MF_K;
        const __half *wr = weight + (size_t)role * (size_t)MF_K;
        float xv[MF_PT];
        float wv[MF_PT];
        #pragma unroll
        for (uint32_t j = 0; j < MF_PT; j++) xv[j] = xb[tid + MF_BLOCK * j];
        #pragma unroll
        for (uint32_t j = 0; j < MF_PT; j++) wv[j] = __half2float(wr[tid + MF_BLOCK * j]);
        float acc = 0.0f;
        if (mode == 0u) {
            // f16_matvec_narrow_pre_scaled's dot over the RAW residual, in the
            // dot's order; the RMS WG computes the scalar, the tail applies it.
            #pragma unroll
            for (uint32_t j = 0; j < MF_PT; j++) acc += wv[j] * xv[j];
        } else {
            // rms_norm_no_weight_batched: one 256-wide tree over the row.
            double partial = 0.0;
            #pragma unroll
            for (uint32_t j = 0; j < MF_PT; j++) {
                const float v = xv[j];
                partial += (double)v * (double)v;
            }
            sq[0][tid] = (float)partial;
            __syncthreads();
            for (uint32_t s = MF_BLOCK / 2; s > 0; s >>= 1) {
                if (tid < s) sq[0][tid] += sq[0][tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                const float mean_sq = sq[0][0] / (float)k;
                inv_s = rsqrtf(mean_sq + rms_eps);
            }
        }
        __syncthreads();
        if (mode != 0u) {
            const float sc = inv_s;
            #pragma unroll
            for (uint32_t j = 0; j < MF_PT; j++) {
                const float f = xv[j] * sc;   // the value rms_nw stored in `flat`
                acc += wv[j] * f;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down(acc, offset, MF_WARP);
        }
        if (lane == 0) per_warp[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < MF_MAX_WARPS) ? per_warp[lane] : 0.0f;
            for (int offset = MF_MAX_WARPS / 2; offset > 0; offset >>= 1) {
                v += __shfl_down(v, offset, MF_WARP);
            }
            if (lane == 0) {
                // Mode 0: the RAW dot; the tail applies `* inv_s`.
                mix_out[(size_t)b * (size_t)MF_MIX + role] = v;
            }
        }
    } else if (mode == 0u && n_mix != 0u && role == n_mix + do_collapse) {
        // ------------------------------------------------ RMS WG (mode 0)
        // rms_norm_partial_sum_sq over 16 slices, as mhc_mix_batched computed
        // it: slice w's jj-th element for thread t is x[t + 256 (w TPS + jj)].
        const float *xb = x + (size_t)b * (size_t)MF_K;
        float xv[MF_PT];
        #pragma unroll
        for (uint32_t j = 0; j < MF_PT; j++) xv[j] = xb[tid + MF_BLOCK * j];
        #pragma unroll
        for (uint32_t w = 0; w < MF_RMS_WGS; w++) {
            double partial = 0.0;
            #pragma unroll
            for (uint32_t jj = 0; jj < MF_TPS; jj++) {
                const float v = xv[w * MF_TPS + jj];
                partial += (double)v * (double)v;
            }
            sq[w][tid] = (float)partial;
        }
        __syncthreads();
        for (uint32_t s = MF_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                for (uint32_t w = 0; w < MF_RMS_WGS; w++) sq[w][tid] += sq[w][tid + s];
            }
            __syncthreads();
        }
        // rms_norm_finalize_inv: one thread, partials summed in order.
        if (tid == 0) {
            float s = 0.0f;
            for (uint32_t w = 0; w < MF_RMS_WGS; w++) s += sq[w][0];
            const float mean_sq = s / (float)k;
            inv_rows[b] = rsqrtf(mean_sq + rms_eps);
        }
    } else if (do_collapse != 0u && role == n_mix) {
        // ------------------------------------------------ collapse
        // hc_weighted_sum_batched (weights = the carry) for this thread's
        // outputs d = tid + 256 j -- exactly the elements rms_norm_weighted's
        // thread tid owns -- then that kernel's passes on the values in
        // registers (the same f32 it would have re-read from `cur`).
        const float *cxb = cx + (size_t)b * (size_t)MF_K;
        const float *cwb = carry + (size_t)b * (size_t)MF_MIX;
        float xs[MF_HC][MF_PE];
        #pragma unroll
        for (uint32_t h = 0; h < MF_HC; h++) {
            #pragma unroll
            for (uint32_t j = 0; j < MF_PE; j++) xs[h][j] = cxb[(size_t)h * MF_NE + tid + MF_BLOCK * j];
        }
        float nw[MF_PE];
        #pragma unroll
        for (uint32_t j = 0; j < MF_PE; j++) nw[j] = norm_w[tid + MF_BLOCK * j];
        float cw[MF_HC];
        #pragma unroll
        for (uint32_t h = 0; h < MF_HC; h++) cw[h] = cwb[h];
        float cv[MF_PE];
        #pragma unroll
        for (uint32_t j = 0; j < MF_PE; j++) {
            float a = 0.0f;
            #pragma unroll
            for (uint32_t h = 0; h < MF_HC; h++) a += xs[h][j] * cw[h];
            cv[j] = a;
            cur_out[(size_t)b * MF_NE + tid + MF_BLOCK * j] = a;
        }
        double partial = 0.0;
        #pragma unroll
        for (uint32_t j = 0; j < MF_PE; j++) {
            const float v = cv[j];
            partial += (double)v * (double)v;
        }
        sq[0][tid] = (float)partial;
        __syncthreads();
        for (uint32_t s = MF_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sq[0][tid] += sq[0][tid + s];
            __syncthreads();
        }
        const float mean_sq = sq[0][0] / (float)ne;
        const float nscale = rsqrtf(mean_sq + rms_eps);
        #pragma unroll
        for (uint32_t j = 0; j < MF_PE; j++) {
            norm_out[(size_t)b * MF_NE + tid + MF_BLOCK * j] = cv[j] * nscale * nw[j];
        }
    } else {
        return;
    }
    if (n_mix == 0u) return;   // collapse only: no split, no counter

    // ---- the LAST WG of this batch row runs the Sinkhorn split
    const unsigned int n_part = n_mix + do_collapse + (mode == 0u ? 1u : 0u);
    if (tid == 0) {
        __threadfence();
        const unsigned int prev = atomicAdd(&counters[b], 1u);
        is_last = (prev == n_part - 1u) ? 1u : 0u;
    }
    __syncthreads();
    if (is_last == 0u) return;
    // One wave from here on (MF_WAVE_SYNC only, never a workgroup barrier
    // after the other waves exit -- see mhc_arena.hip).
    if (tid >= MF_WARP) return;
    __threadfence();
    if (tid == 0) counters[b] = 0u;
    mf_sinkhorn_tail(tid, mix_out + (size_t)b * (size_t)MF_MIX, mode == 0u ? inv_rows + b : nullptr,
                     split_out + (size_t)b * (size_t)MF_MIX,
                     write_carry ? carry + (size_t)b * (size_t)MF_MIX : nullptr,
                     scale, base, n_hc, iters, eps);
}
