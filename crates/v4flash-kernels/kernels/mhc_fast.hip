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
#include "mhc_fast_body.inc"
}

// The same body as a device function for the `_ind` twin: inlined, its __restrict__ parameters
// become scoped-noalias metadata on the resolved pointers.
__device__ __forceinline__ static void mhc_fast_body(
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
#include "mhc_fast_body.inc"
}

// `_ind` twin: operands 0..12 (split_out .. norm_w, in order) resolve through the arena context
// when their `ind_mask` bit is set (arena_ctx.inc); the canary's loads, the same body, the canary's
// record.
#include "arena_ctx.inc"
extern "C" __global__ __launch_bounds__(MF_BLOCK) void mhc_fast_batched_ind(
        uint32_t ind_mask, const ArenaCtx* canary, uint32_t canary_tag,
        float* split_out, float* mix_out, unsigned int* counters, float* inv_rows,
        const __half* weight, const float* x, const float* scale, const float* base,
        float* carry, const float* cx, float* cur_out, float* norm_out, const float* norm_w,
        unsigned int k, unsigned int ne, unsigned int n_mix, unsigned int do_collapse,
        unsigned int write_carry, unsigned int mode, float rms_eps, unsigned int n_hc,
        unsigned int iters, float eps) {
    ARENA_CTX_DEREF(ind_mask, 0, float*, split_out);
    ARENA_CTX_DEREF(ind_mask, 1, float*, mix_out);
    ARENA_CTX_DEREF(ind_mask, 2, unsigned int*, counters);
    ARENA_CTX_DEREF(ind_mask, 3, float*, inv_rows);
    ARENA_CTX_DEREF(ind_mask, 4, const __half*, weight);
    ARENA_CTX_DEREF(ind_mask, 5, const float*, x);
    ARENA_CTX_DEREF(ind_mask, 6, const float*, scale);
    ARENA_CTX_DEREF(ind_mask, 7, const float*, base);
    ARENA_CTX_DEREF(ind_mask, 8, float*, carry);
    ARENA_CTX_DEREF(ind_mask, 9, const float*, cx);
    ARENA_CTX_DEREF(ind_mask, 10, float*, cur_out);
    ARENA_CTX_DEREF(ind_mask, 11, float*, norm_out);
    ARENA_CTX_DEREF(ind_mask, 12, const float*, norm_w);
    ARENA_CTX_CANARY_LOAD(canary, ARENA_CTX_U64(split_out) ^ ARENA_CTX_U64(mix_out) ^ ARENA_CTX_U64(counters) ^
                                  ARENA_CTX_U64(inv_rows) ^ ARENA_CTX_U64(weight) ^ ARENA_CTX_U64(x) ^
                                  ARENA_CTX_U64(scale) ^ ARENA_CTX_U64(base) ^ ARENA_CTX_U64(carry) ^
                                  ARENA_CTX_U64(cx) ^ ARENA_CTX_U64(cur_out) ^ ARENA_CTX_U64(norm_out) ^
                                  ARENA_CTX_U64(norm_w));
    mhc_fast_body(split_out, mix_out, counters, inv_rows, weight, x, scale, base, carry, cx, cur_out, norm_out, norm_w,
                  k, ne, n_mix, do_collapse, write_carry, mode, rms_eps, n_hc, iters, eps);
    ARENA_CTX_CANARY_STORE(canary_tag);
}
