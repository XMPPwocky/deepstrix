// g6_boundary -- dGPU fusion group G6 (knob V41_DGPU_G6_BOUNDARY, default off): the decode layer
// BOUNDARY as ONE launch. It replaces, per lane-layer l -> l+1, three queue operations on de.compute:
//   POST(l)    the pinned H2D of box 2's partial (SDMA, b x 5120 f32) -- `early`: issued on the upload
//              stream at the Post instead (de.compute waits on its event); `zc`: read zero-copy here
//              hc_post_from_split_batched_add2 / _add (the FFN combine) -> residual_next
//   CHAIN(l+1) mhc_fast_batched (pre-attn mix PRE_SCALED + collapse + RMS WG + Sinkhorn + carry)
// every output BIT-IDENTICAL to that chain: residual_next (the combine), cur / attn_input_norm (the
// collapse), mix / split / carry (the mix + Sinkhorn), inv_rows (the RMS scalar)
// (tests/g6_boundary_bitexact.rs).
//
// ROLES (ticketed: every workgroup draws a ticket on entry and takes its role from it, so a
// workgroup that waits only ever waits on a LOWER ticket, drawn by a workgroup that is already
// resident -- forward progress needs no assumption about dispatch order; the G1 protocol):
//   t <  b      LEAD of row z = t. Thread tid owns d = tid + 256 j (j < 20) for all 4 copies, i.e. the
//               residual elements e = tid + 256 m (m = 20 h + j < 80) -- exactly the elements the old
//               mix / RMS / collapse threads tid owned. It issues box 2's partial first (zc: system-
//               scope loads from pinned host memory), the MoE / shared rows and after_attn_hc, then
//               the combine with hc_post_from_split_batched_add2's expression in its order:
//               bv = (moe + shared) [+ remote]; acc = bv * post[dst]; acc = fma(comb[dst + 4 src],
//               rh[src], acc) for src = 0..3. It stores residual_next, RELEASES ready[z], then runs
//               the collapse on the values still in registers (hc_weighted_sum with the OLD carry +
//               rms_norm_weighted, its fp64 partial and 256-wide tree).
//   t >= b      MIX workgroup k of row z (MR = 1 / 2 / 3 mix rows each: g6_boundary_m{MR}): prefetches
//               its f16 weight rows (they do not depend on the combine), waits for ready[z], reads the
//               row from L2 and runs mhc_fast's mode-0 dot in its order (the raw dot; the tail scales
//               it). It also carries the pre-scaled RMS scalar's slices w = k, k + 24/MR, .. (the old
//               RMS WG's fp64 slice partial and its 256-wide tree: levels 128 / 64 in LDS, 32 .. 1 by
//               wave shuffles over the same pairs) -> rms_part[z][w].
//   last of row the last of a row's 1 + 24/MR workgroups (atomic counter, as mhc_fast) sums the 16
//               slices in order (rms_norm_finalize_inv), then the Sinkhorn split IN REGISTERS (same
//               expressions and order as mf_sinkhorn_tail: a row's values in one DPP quad, a column's
//               gathered with v_permlane16) and split (+ carry := split). The lead reads the OLD carry
//               before its increment; the tail writes the new one after the last increment.
//   last of all resets the ticket and ready flags (self-resetting: zero between launches).
// Launches that share one sync buffer must be STREAM-ORDERED (production: de.compute only).
//
// Every poll is bounded (G6_SPIN_CAP, 2 s at 100 MHz): on expiry it sets the sticky S_ERR word and
// proceeds, so a broken protocol fails the tests instead of hanging the GPU.
//
// MEASURED (tests/bench_g6_boundary.rs, in-kernel trace V41_DGPU_G6_TS, MR = 2, b = 4, cold, partial in
// VRAM): lead stores 3.6 us, flag 4.1, mix wait over 4.6, dots 5.6, slice trees 6.3 -> 6.8, tail 7.3 ->
// 9.2, done 9.4 us; the chain it replaces is ~35 us of de.compute time. The zero-copy partial (zc) costs
// ~3 us per row of PCIe inside the lead (the link reads ~6.5 GB/s); the SDMA copy the same, but on
// the upload stream it leaves de.compute's timeline.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef MHC_HC_DIM
#define MHC_HC_DIM 16384
#endif
#ifndef MHC_N_EMBD
#define MHC_N_EMBD 4096
#endif

#define G6_BLOCK 256u
#define G6_WARP 32u
#define G6_WARPS 8u
#define G6_HC 4u
#define G6_MIX (2u * G6_HC + G6_HC * G6_HC)   // 24
#define G6_RMS_WGS 16u
#define G6_K  ((unsigned)MHC_HC_DIM)
#define G6_NE ((unsigned)MHC_N_EMBD)
#define G6_PT (G6_K / G6_BLOCK)              // 80 residual elements per thread
#define G6_PE (G6_NE / G6_BLOCK)             // 20 embd positions per thread
#define G6_TPS (G6_PT / G6_RMS_WGS)          // 5
#define G6_MAXB 16u
#if (MHC_HC_DIM != 4 * MHC_N_EMBD) || (MHC_N_EMBD % 256) != 0 || (MHC_HC_DIM % (256 * 16)) != 0
#error "g6_boundary: K must be 4 * N_EMBD, a multiple of 4096"
#endif

// sync words (u32), each on its own 128-byte line
#define S_TICKET 0u
#define S_FINISH 32u
#define S_READY(z) (64u + 32u * (z))
#define S_ERR (64u + 32u * G6_MAXB)
#define S_WORDS (S_ERR + 32u)

#define G6_SPIN_CAP 200000000ull             // 2 s at 100 MHz

struct G6Args {
    // combine
    float* resid_out;          // [b, K] residual_next (the next layer's residual)
    const float* moe;          // [b, NE] ffn_moe_recv
    const float* shared;       // [b, NE] ffn_shared (the first addend)
    const float* remote;       // [b, NE] box 2's partial in PINNED host memory (second addend) or null
    const float* after_attn;   // [b, K] after_attn_hc (the combine's residual_hc)
    const float* split_ffn;    // [b, 24] the FFN split (post at +4, comb at +8)
    // mhc pre-attn (mode 0 + collapse)
    const __half* weight;      // [24, K] hc_attn_fn
    const float* scale;        // [3]
    const float* base;         // [24]
    float* split_out;          // [b, 24]
    float* mix_out;            // [b, 24]
    unsigned* counters;        // [b] zero between launches
    float* inv_rows;           // [b]
    float* rms_part;           // [b, 16] scratch: the RMS slice sums (mix WGs -> the tail)
    float* carry;              // [b, 24] read (old) by the collapse, written by the tail
    float* cur_out;            // [b, NE]
    float* norm_out;           // [b, NE]
    const float* norm_w;       // [NE]
    unsigned* sync;            // S_WORDS, zero between launches
    unsigned long long* ts;    // debug phase timestamps or null: [ticket][8]
    unsigned b;
    unsigned write_carry;
    float rms_eps;
    unsigned n_hc;
    unsigned iters;
    float eps;
    unsigned dbg;              // bench ablations: 1 = no remote read (zeros)
    unsigned remote_sys;       // 1: `remote` is pinned HOST memory (system-scope loads, zero-copy over PCIe);
                               // 0: `remote` is a device buffer (box 2's partial already uploaded)
};

__device__ __forceinline__ unsigned long long g6_now() { return __builtin_readsteadycounter(); }
#define G6_TS(k) do { if (ts != nullptr) ts[k] = g6_now(); } while (0)

// Wave-uniform scalar load at device scope (s_dcache_inv first: a scoped s_load alone can return a
// stale line from the scalar cache -- the G1 lesson).
__device__ __forceinline__ unsigned g6_sload_dev(const unsigned* pv) {
    const unsigned long long pa = (unsigned long long)pv;
    const unsigned lo = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)pa);
    const unsigned hi = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)(pa >> 32));
    const unsigned* p = (const unsigned*)(((unsigned long long)hi << 32) | lo);
    unsigned r;
#if defined(__gfx1200__) || defined(__gfx1201__)
    asm volatile("s_dcache_inv\n\ts_wait_kmcnt 0x0\n\ts_load_b32 %0, %1, 0x0 scope:SCOPE_DEV\n\ts_wait_kmcnt 0x0" : "=s"(r) : "s"(p) : "memory");
#else
    asm volatile("s_dcache_inv\n\ts_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(r) : "s"(p) : "memory");
#endif
    return r;
}

// Wave-uniform bounded poll until *p != 0.
__device__ __forceinline__ void g6_spoll(const unsigned* p, unsigned* err) {
    const unsigned long long t0 = g6_now();
    while (g6_sload_dev(p) == 0u) {
        if (g6_now() - t0 > G6_SPIN_CAP) {
            if ((threadIdx.x & 31u) == 0u) atomicOr(err, 1u);
            break;
        }
        __builtin_amdgcn_s_sleep(1);
    }
}

// Quad broadcast of lane (4 * (lane / 4) + K) -- DPP quad_perm [K, K, K, K].
template <int K>
__device__ __forceinline__ float g6_quad_bcast(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), K | (K << 2) | (K << 4) | (K << 6), 0xf, 0xf, false));
}

// Column gather: lane i (< 16) takes lane (i & 3) + 4 Y of its 16-lane row (v_permlane16_b32, a VALU op:
// no LDS round trip).
template <unsigned Y>
__device__ __forceinline__ float g6_col_get(float v) {
    constexpr unsigned lo = (0u + 4u * Y) | ((1u + 4u * Y) << 4) | ((2u + 4u * Y) << 8) | ((3u + 4u * Y) << 12)
                          | ((0u + 4u * Y) << 16) | ((1u + 4u * Y) << 20) | ((2u + 4u * Y) << 24) | ((3u + 4u * Y) << 28);
    return __int_as_float(__builtin_amdgcn_permlane16(__float_as_int(v), __float_as_int(v), lo, lo, false, false));
}

// The Sinkhorn split of one row in REGISTERS: the same expressions in the same order as
// mf_sinkhorn_tail (and so hc_sinkhorn_par_batched), lane i < 16 holding c[i] (src = i % 4, dst = i / 4):
// a row's values sit in one DPP quad (broadcast in index order), a column's are gathered with
// v_permlane16 (index order). `mix_row` holds the RAW dots; `inv_s` is the row's RMS scalar (applied with
// the same `v * inv_s`, written back as the chain's `mix`). Wave 0 only, all 32 lanes.
__device__ __forceinline__ static void g6_sinkhorn_regs(
        const unsigned int tid, float *mix_row, const float inv_s, float *out_b, float *carry_b,
        const float *scale, const float *base, unsigned int n_hc, unsigned int iters, float eps) {
    float v = 0.0f;
    if (tid < G6_MIX) {
        volatile float *mv = mix_row;
        v = mv[tid];
        v = v * inv_s;
        mv[tid] = v;
    }
    const bool act = tid < 16u && n_hc == G6_HC;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    if (act) {
        if (tid < G6_HC) {
            const float z = v * pre_scale + base[tid];
            float sig;
            if (z >= 0.0f) sig = 1.0f / (1.0f + expf(-z));
            else           sig = expf(z) / (1.0f + expf(z));
            const float o = sig + eps;
            out_b[tid] = o;
            if (carry_b) carry_b[tid] = o;
        } else if (tid < 2 * G6_HC) {
            const unsigned int off = tid;
            const float z = v * post_scale + base[off];
            float sig;
            if (z >= 0.0f) sig = 1.0f / (1.0f + expf(-z));
            else           sig = expf(z) / (1.0f + expf(z));
            const float o = 2.0f * sig;
            out_b[off] = o;
            if (carry_b) carry_b[off] = o;
        }
    }
    // c[idx] = mix[8 + idx] * comb_scale + base[8 + idx], idx = lane (lanes 8..23 hold those mix values)
    const float mc = __shfl(v, (int)((tid + 2u * G6_HC) & 31u), 32);
    const unsigned int idx = tid & 15u;
    float c = mc * comb_scale + base[2u * G6_HC + idx];
    // row max (row = dst = this lane's quad), elements in index order
    float m = -INFINITY;
    {
        const float r0 = g6_quad_bcast<0>(c), r1 = g6_quad_bcast<1>(c), r2 = g6_quad_bcast<2>(c), r3 = g6_quad_bcast<3>(c);
        if (r0 > m) m = r0;
        if (r1 > m) m = r1;
        if (r2 > m) m = r2;
        if (r3 > m) m = r3;
    }
    c = expf(c - m);
    auto row_sum = [&](float x) {
        const float r0 = g6_quad_bcast<0>(x), r1 = g6_quad_bcast<1>(x), r2 = g6_quad_bcast<2>(x), r3 = g6_quad_bcast<3>(x);
        float s = 0.0f;
        s += r0;
        s += r1;
        s += r2;
        s += r3;
        return s;
    };
    auto col_sum = [&](float x) {
        const float c0 = g6_col_get<0>(x), c1 = g6_col_get<1>(x), c2 = g6_col_get<2>(x), c3 = g6_col_get<3>(x);
        float s = 0.0f;
        s += c0;
        s += c1;
        s += c2;
        s += c3;
        return s;
    };
    c = c / row_sum(c) + eps;
    c = c / (col_sum(c) + eps);
    for (unsigned int iter = 1; iter < iters; iter++) {
        c = c / (row_sum(c) + eps);
        c = c / (col_sum(c) + eps);
    }
    if (act) {
        out_b[2u * G6_HC + idx] = c;
        if (carry_b) carry_b[2u * G6_HC + idx] = c;
    }
}

// Mix workgroups per row and the mix rows each one computes.

template <unsigned G6_MR>
__device__ __forceinline__ void g6_body(const G6Args& a) {
    constexpr unsigned G6_NMIX = G6_MIX / G6_MR;
    constexpr unsigned G6_SPW = (G6_RMS_WGS + G6_NMIX - 1u) / G6_NMIX;
    __shared__ float sq[G6_SPW][G6_BLOCK];
    __shared__ float per_warp[G6_MR][G6_WARPS];
    __shared__ unsigned s_ticket;
    __shared__ unsigned is_last;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & (G6_WARP - 1u);
    const unsigned warp = tid >> 5u;
    const unsigned long long t_start = g6_now();
    if (tid == 0) s_ticket = atomicAdd(a.sync + S_TICKET, 1u);
    __syncthreads();
    const unsigned t = s_ticket;
    const unsigned b = a.b;
    unsigned long long* ts = (a.ts != nullptr && tid == 0) ? a.ts + (size_t)t * 16u : nullptr;
    if (ts != nullptr) { ts[0] = t_start; ts[1] = g6_now(); ts[7] = 0ull; }
    unsigned z;

    if (t < b) {
        // =========================================================== LEAD of row z
        z = t;
        const float* rm = a.remote != nullptr ? a.remote + (size_t)z * G6_NE : nullptr;
        const float* bo = a.moe + (size_t)z * G6_NE;
        const float* sh = a.shared + (size_t)z * G6_NE;
        const float* rh = a.after_attn + (size_t)z * G6_K;
        // box 2's partial first: the PCIe round trip is the longest latency (zero-copy: system scope,
        // never a stale cached line of the pinned staging)
        float rv[G6_PE];
        if (rm != nullptr && (a.dbg & 1u) == 0u && a.remote_sys != 0u) {
            #pragma unroll
            for (unsigned j = 0; j < G6_PE; j++)
                rv[j] = __hip_atomic_load(rm + tid + G6_BLOCK * j, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        } else if (rm != nullptr && (a.dbg & 1u) == 0u) {
            #pragma unroll
            for (unsigned j = 0; j < G6_PE; j++) rv[j] = rm[tid + G6_BLOCK * j];
        } else {
            #pragma unroll
            for (unsigned j = 0; j < G6_PE; j++) rv[j] = 0.0f;
        }
        float bov[G6_PE], shv[G6_PE];
        #pragma unroll
        for (unsigned j = 0; j < G6_PE; j++) bov[j] = bo[tid + G6_BLOCK * j];
        #pragma unroll
        for (unsigned j = 0; j < G6_PE; j++) shv[j] = sh[tid + G6_BLOCK * j];
        float xs[G6_HC][G6_PE];
        #pragma unroll
        for (unsigned h = 0; h < G6_HC; h++) {
            #pragma unroll
            for (unsigned j = 0; j < G6_PE; j++) xs[h][j] = rh[(size_t)h * G6_NE + tid + G6_BLOCK * j];
        }
        const float* sp = a.split_ffn + (size_t)z * G6_MIX;
        float pb[G6_HC], cb[G6_HC * G6_HC];
        #pragma unroll
        for (unsigned h = 0; h < G6_HC; h++) pb[h] = sp[G6_HC + h];
        #pragma unroll
        for (unsigned i = 0; i < G6_HC * G6_HC; i++) cb[i] = sp[2u * G6_HC + i];
        // the collapse's inputs that do not depend on the combine (the OLD carry)
        const float* cwb = a.carry + (size_t)z * G6_MIX;
        float cw[G6_HC];
        #pragma unroll
        for (unsigned h = 0; h < G6_HC; h++) cw[h] = cwb[h];
        float nw[G6_PE];
        #pragma unroll
        for (unsigned j = 0; j < G6_PE; j++) nw[j] = a.norm_w[tid + G6_BLOCK * j];

        // ---- the combine (hc_post_from_split_batched_add2 / _add), in place over xs
        float* ob = a.resid_out + (size_t)z * G6_K;
        #pragma unroll
        for (unsigned j = 0; j < G6_PE; j++) {
            float bv = bov[j];
            bv += shv[j];
            if (rm != nullptr) bv += rv[j];
            float o[G6_HC];
            #pragma unroll
            for (unsigned dst = 0; dst < G6_HC; dst++) {
                float acc = bv * pb[dst];
                #pragma unroll
                for (unsigned s = 0; s < G6_HC; s++) acc = __builtin_fmaf(cb[dst + s * G6_HC], xs[s][j], acc);
                o[dst] = acc;
            }
            #pragma unroll
            for (unsigned dst = 0; dst < G6_HC; dst++) {
                xs[dst][j] = o[dst];
                ob[(size_t)dst * G6_NE + tid + G6_BLOCK * j] = o[dst];
            }
        }
        G6_TS(11);
        // ---- RELEASE ready[z]: every thread's stores fenced, then one flag store
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            __hip_atomic_store(a.sync + S_READY(z), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        G6_TS(2);
        // ---- the collapse (mhc_fast's collapse WG on the values in registers)
        float cv[G6_PE];
        #pragma unroll
        for (unsigned j = 0; j < G6_PE; j++) {
            float acc = 0.0f;
            #pragma unroll
            for (unsigned h = 0; h < G6_HC; h++) acc = __builtin_fmaf(xs[h][j], cw[h], acc);
            cv[j] = acc;
            a.cur_out[(size_t)z * G6_NE + tid + G6_BLOCK * j] = acc;
        }
        {
            double partial = 0.0;
            #pragma unroll
            for (unsigned j = 0; j < G6_PE; j++) {
                const double v = (double)cv[j];
                partial = __builtin_fma(v, v, partial);
            }
            sq[0][tid] = (float)partial;
        }
        __syncthreads();
        for (unsigned s = G6_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sq[0][tid] += sq[0][tid + s];
            __syncthreads();
        }
        const float mean_sq = sq[0][0] / (float)G6_NE;
        const float nscale = rsqrtf(mean_sq + a.rms_eps);
        #pragma unroll
        for (unsigned j = 0; j < G6_PE; j++) {
            a.norm_out[(size_t)z * G6_NE + tid + G6_BLOCK * j] = cv[j] * nscale * nw[j];
        }
        G6_TS(3);
    } else {
        // =========================================================== MIX WG k of row z: mix rows
        // k * MR .. + MR (mhc_fast's mode-0 dot, raw) + the RMS slices w = k, k + NMIX, ... (the old RMS
        // WG's slice partial and its 256-wide tree; the tail sums the 16 slices in order).
        const unsigned q = t - b;
        z = q / G6_NMIX;
        const unsigned k = q - z * G6_NMIX;
        float wv[G6_MR][G6_PT];
        #pragma unroll
        for (unsigned r = 0; r < G6_MR; r++) {
            const __half* wr = a.weight + (size_t)(k * G6_MR + r) * G6_K;
            #pragma unroll
            for (unsigned j = 0; j < G6_PT; j++) wv[r][j] = __half2float(wr[tid + G6_BLOCK * j]);
        }
        if (warp == 0) g6_spoll(a.sync + S_READY(z), a.sync + S_ERR);
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        G6_TS(2);
        const float* xb = a.resid_out + (size_t)z * G6_K;
        float xv[G6_PT];
        #pragma unroll
        for (unsigned j = 0; j < G6_PT; j++) xv[j] = xb[tid + G6_BLOCK * j];
        #pragma unroll
        for (unsigned r = 0; r < G6_MR; r++) {
            float acc = 0.0f;
            #pragma unroll
            for (unsigned j = 0; j < G6_PT; j++) acc = __builtin_fmaf(wv[r][j], xv[j], acc);
            for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset, G6_WARP);
            if (lane == 0) per_warp[r][warp] = acc;
        }
        G6_TS(8);
        // the RMS slice partials (rms_norm_partial_sum_sq: slice w's jj-th element for thread t is
        // x[t + 256 (5 w + jj)])
        unsigned n_sl = 0;
        #pragma unroll
        for (unsigned i = 0; i < G6_SPW; i++) {
            const unsigned w = k + i * G6_NMIX;
            if (w < G6_RMS_WGS) {
                double partial = 0.0;
                #pragma unroll
                for (unsigned jj = 0; jj < G6_TPS; jj++) {
                    // w is not a compile-time constant: pick xv[5 w + jj] by constant indices
                    float vv = 0.0f;
                    #pragma unroll
                    for (unsigned ww = 0; ww < G6_RMS_WGS; ww++) if (ww == w) vv = xv[ww * G6_TPS + jj];
                    const double v = (double)vv;
                    partial = __builtin_fma(v, v, partial);
                }
                sq[i][tid] = (float)partial;
                n_sl = i + 1;
            }
        }
        __syncthreads();
        G6_TS(9);
        // the slice trees' cross-wave levels (s = 128, 64) in LDS; s = 32 .. 1 below in wave 0: the
        // same pairs (tid += tid + s for tid < s), the in-wave levels by shuffles
        if (n_sl > 0) {
            #pragma unroll
            for (unsigned s = G6_BLOCK / 2; s >= 64u; s >>= 1) {
                if (tid < s) {
                    #pragma unroll
                    for (unsigned i = 0; i < G6_SPW; i++) if (i < n_sl) sq[i][tid] += sq[i][tid + s];
                }
                __syncthreads();
            }
        }
        if (warp == 0) {
            #pragma unroll
            for (unsigned r = 0; r < G6_MR; r++) {
                float v = (lane < G6_WARPS) ? per_warp[r][lane] : 0.0f;
                for (int offset = G6_WARPS / 2; offset > 0; offset >>= 1) v += __shfl_down(v, offset, G6_WARP);
                if (lane == 0) a.mix_out[(size_t)z * G6_MIX + k * G6_MR + r] = v;
            }
            #pragma unroll
            for (unsigned i = 0; i < G6_SPW; i++) {
                if (i < n_sl) {
                    float v = sq[i][lane];
                    v += sq[i][lane + 32u];
                    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, G6_WARP);
                    if (lane == 0) a.rms_part[(size_t)z * G6_RMS_WGS + k + i * G6_NMIX] = v;
                }
            }
        }
        G6_TS(3);
    }

    // ---- the LAST of the row's 1 + NMIX workgroups: the RMS finalize + the Sinkhorn split
    if (tid == 0) {
        __threadfence();
        const unsigned prev = atomicAdd(&a.counters[z], 1u);
        is_last = (prev == G6_NMIX) ? 1u : 0u;
    }
    __syncthreads();
    if (is_last != 0u && tid < G6_WARP) {
        __threadfence();
        if (tid == 0) a.counters[z] = 0u;
        // rms_norm_finalize_inv: the 16 slice sums in order
        const float pv = tid < G6_RMS_WGS ? __hip_atomic_load(a.rms_part + (size_t)z * G6_RMS_WGS + tid, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : 0.0f;
        float s = 0.0f;
        #pragma unroll
        for (unsigned w = 0; w < G6_RMS_WGS; w++) s += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(pv), (int)w));
        const float ms = s / (float)G6_K;
        const float inv = rsqrtf(ms + a.rms_eps);
        if (tid == 0) a.inv_rows[z] = inv;
        G6_TS(10);
        g6_sinkhorn_regs(tid, a.mix_out + (size_t)z * G6_MIX, inv,
                         a.split_out + (size_t)z * G6_MIX,
                         a.write_carry ? a.carry + (size_t)z * G6_MIX : nullptr,
                         a.scale, a.base, a.n_hc, a.iters, a.eps);
        if (ts != nullptr) ts[7] = g6_now();
    }
    // ---- finish: the last WG of the launch resets the sync words (not S_ERR)
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned prev = atomicAdd(a.sync + S_FINISH, 1u);
        if (ts != nullptr) { ts[4] = g6_now(); ts[5] = z; ts[6] = t < b ? 0u : 2u; }
        if (prev == b * (1u + G6_NMIX) - 1u) {
            __threadfence();
            a.sync[S_TICKET] = 0u;
            for (unsigned zz = 0; zz < b; ++zz) a.sync[S_READY(zz)] = 0u;
            a.sync[S_FINISH] = 0u;
            __threadfence();
        }
    }
}

extern "C" __global__ __launch_bounds__(G6_BLOCK) void g6_boundary_m1(G6Args a) { g6_body<1u>(a); }
extern "C" __global__ __launch_bounds__(G6_BLOCK) void g6_boundary_m2(G6Args a) { g6_body<2u>(a); }
extern "C" __global__ __launch_bounds__(G6_BLOCK) void g6_boundary_m3(G6Args a) { g6_body<3u>(a); }
