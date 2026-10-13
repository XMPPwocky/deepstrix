// g1_router_block -- the decode lane-layer's whole ROUTER BLOCK as ONE launch (dGPU fusion group G1,
// knob V41_DGPU_G1_FUSED, default off). It replaces, on `de.compute` between the attention's hc_post
// and `selected_ready`, the nine queue operations of the production chain:
//
//   mhc_fast_batched (pre-ffn collapse: hc_weighted_sum + rms_norm_weighted)   -> cur, ffn_input_norm
//   f16_matvec_batched_h20_x2 (router of layer l and the k1 look-ahead's l+1) -> logits, logits2
//   pinned H2D of the cache prior (V41_SUB=3)                                   -> d_prior
//   router_topk_wfred (prior, protect, orig_sel, range)                         -> sel, ew, orig, range
//   router_topk_wfred (look-ahead, plain)                                       -> look_sel, look_ew
//   q8_k_quantize (box 2's / the iGPU's Q8_K rows)                             -> xq
//   readback_pack (everything host-bound -> the pinned pack)                   -> rb_pack
//   expert_sel_count (M62 bank)                                                -> sel bank
//
// with every value BIT-IDENTICAL (tests/g1_router_block_bitexact.rs): each stage keeps its kernel's
// per-output arithmetic -- the collapse is mhc_fast's collapse branch verbatim, the router keeps
// h20's per-lane element sequence (i = lane + 32 j, j ascending, one f32 accumulator per output, the
// same h2f(w) * x FMA) and its shfl_down tree, the router's probabilities / scores are wfred's
// expressions, the top-k is an exact selection under wfred's key (score order, then the lower
// expert index) with the weights summed in pick order, and each Q8_K block keeps q8_k_quantize's
// expressions and its tree's tie rule (below).
//
// Two kernel families share the roles and the arithmetic below: the round-1 forms (g1_body: symbols
// _z / _p) and FAMILY "r" (g1r_body, the default since round 2: see its header further down for the
// measured hardware behaviour it is scheduled around).
//
// DEPENDENCY GRAPH -> ROLES. Every workgroup draws a TICKET (atomic) on entry and takes its role from
// it, so a workgroup that waits only ever waits on LOWER tickets, which were drawn by workgroups that
// are already resident: forward progress needs no assumption about the hardware's dispatch order
// (rocPRIM's ordered block id). Roles, in ticket order:
//   [0, b)          COLLAPSE of batch row z (one 256-thread WG = mhc_fast's collapse WG): cur and
//                   ffn_input_norm, then RELEASE row_ready[z]; then the row's 20 Q8_K blocks (wave w:
//                   blocks w, w + 8, w + 16) -> xq + its pack words, beside the router.
//   [b]             PRIOR (with a cache prior): pinned host -> d_prior, RELEASE prior_ready.
//   [b + p, ..)     ROUTER: 8 waves = 8 weight rows of matrix m (this layer / the k1 look-ahead), ALL
//                   b batch rows per wave (each weight byte is read once; <= 112 WGs, all resident).
//                   Each wave issues its first weight elements BEFORE it waits on the rows (the
//                   weights do not depend on the collapse), then runs h20's schedule; its epilogue
//                   writes the logit and wfred's prob / score (lane z: row z). The LAST router WG of
//                   a matrix (counter) runs that matrix's selection, one wave per batch row.
//   any             the LAST WG to finish resets the scratch (self-resetting: replay-safe) and, when
//                   asked, writes `done_value` to a pinned word after a system-scope fence.
//
// LATENCY NOTES (in-kernel phase trace, tests/bench_g1_router_block.rs V41_DGPU_G1_TS=1): one wave
// runs serial code at ~5 cycles per dependent VALU op; gfx1201's 64-bit integer compares cost ~25;
// a storm of waves polling one flag line saturates its L2 channel; a pinned-host read queues behind
// the posted pack writes. Hence: 32-bit selection (max trees + DPP butterflies), one polling thread
// per WG with every flag on its own line, the prior read before any pack write, and the per-expert
// transcendentals spread over the router waves instead of the tail.
//
// Q8_K TIE RULE. q8_k_quantize finds the signed value at max |x| with a 256 -> 1 tree in which slot
// tid keeps its value unless slot tid + stride is STRICTLY greater (strides 128, 64, .., 1). Among
// equal maxima that tree keeps the element whose index has the smallest BIT-REVERSED value (the last
// level prefers bit 0 = 0, the one before bit 1 = 0, ...). Here: max |x|, then the min bitrev8(index)
// among the maxima, then that element's sign -- exact.
//
// TOP-K. wfred's passes pick the max of (ord(score), ~expert) and set the winner to -INF (it stays a
// candidate at ord(-INF)). Here each pass: a lane's best by a 12-way max tree on ord (strictly
// greater wins, so the lower index keeps ties), the wave max by DPP, and ties between lanes resolved
// to the lower expert by a second reduction (only when they happen). Same winner every pass.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef MHC_N_EMBD
#define MHC_N_EMBD 4096
#endif
#define G1_NE ((unsigned)MHC_N_EMBD)       // 5120 at V4.1
#define G1_BLOCK 256u
#define G1_WARP 32u
#define G1_HC 4u
#define G1_MIX 24u
#define G1_PE (G1_NE / G1_BLOCK)           // 20 collapse outputs per thread = Q8_K blocks per row
#define G1_NEXP 384u                       // router rows per matrix (V4.1 N_EXPERT)
#define G1_EPL (G1_NEXP / G1_WARP)         // 12 experts per lane in the one-wave top-k
#define G1_RWG (G1_NEXP / 8u)              // 48 router WGs per matrix
#define G1_MAXB 16u
#define G1_EPW (G1_NE / G1_WARP)           // 160 weight elements per lane
#define G1_PFE 40u                         // weight elements per lane prefetched before the row wait
#define G1_QBW 73u                         // block_q8_K = 292 bytes = 73 words
#define G1_NONE 0xffffffffu

// scratch words (`sync`, zeroed once at allocation; the kernel leaves it zeroed), each on its own
// 128-byte line: a polled flag must not share an L2 line (or channel queue) with the others.
#define S_LINE 32u
#define S_TICKET 0u
#define S_FINISH (1u * S_LINE)
#define S_RDONE(m) ((2u + (m)) * S_LINE)
#define S_PRIOR (4u * S_LINE)
#define S_READY(z) ((5u + (z)) * S_LINE)
#define S_WORDS (S_READY(G1_MAXB))

struct G1Args {
    // collapse
    const float* cx;          // [b, 4 * ne] after_attn_hc
    const float* carry;       // [b, 24] hc_pre_carry (the previous sub-block's pre-weights)
    const float* norm_w;      // [ne]
    float* cur_out;           // [b, ne]
    float* norm_out;          // [b, ne] ffn_input_norm
    // router
    const __half* w0;         // [384, ne] this layer's gate
    const __half* w1;         // [384, ne] look-ahead gate (layer l+1) or null
    float* logits0;           // [b, 384]
    float* logits1;           // [b, 384] (when w1)
    const float* bias0;       // [384] or null
    const float* bias1;       // [384] or null
    const float* prior_host;  // [384] pinned (device-visible) or null = no prior
    float* prior_dev;         // [384] d_prior (when prior_host)
    int* sel;                 // [b, n_used]
    float* ew;                // [b, n_used]
    int* orig_sel;            // [b, n_used] or null
    float* range_out;         // [b] or null
    int* look_sel;            // [b, n_used] (when w1)
    float* look_ew;           // [b, n_used] (when w1)
    unsigned* xq;             // [b, 20 * 73] words or null
    unsigned* pack;           // pinned readback pack or null
    unsigned* sel_bank;       // M62 bank or null
    unsigned* sync;           // [S_WORDS] scratch
    float* scr;               // [2 (prob, score)][2 mat][16][384] scratch
    unsigned* done_word;      // pinned completion word or null
    unsigned long long* ts;   // debug phase timestamps (V41_DGPU_G1_TS) or null: [ticket][16]
    // scalars
    unsigned b;
    unsigned n_used;
    unsigned n_protect;
    unsigned prior_dry;
    unsigned bank_off;        // layer * n_expert
    unsigned zb;              // batch rows per router wave (the symbol's ZB, >= b)
    unsigned n_mat;           // 1 or 2
    unsigned done_value;
    unsigned pk_sel, pk_look, pk_look_ew, pk_orig, pk_range, pk_ew, pk_xq;   // pack word offsets or G1_NONE
    float rms_eps;
    float scale;
    float weight_eps;
    unsigned n_pf;            // PREFETCH workgroups (0 = none)
    unsigned dbg;             // bench ablations (V41_DGPU_G1_DBG): 1 no prior, 2 no q8k, 4 no top-k tail, 8 no row wait, 16 no ticket
};

#define G1_SCR_PROB(m, z) (((0u * 2u + (m)) * G1_MAXB + (z)) * G1_NEXP)
#define G1_SCR_SCORE(m, z) (((1u * 2u + (m)) * G1_MAXB + (z)) * G1_NEXP)

__device__ inline float g1_softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ unsigned int g1_ord(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float g1_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, offset);
    return v;
}

__device__ __forceinline__ unsigned long long g1_now() { return __builtin_readsteadycounter(); }
#define G1_TS(k) do { if (ts != nullptr) ts[k] = g1_now(); } while (0)

__device__ __forceinline__ unsigned ld_relaxed(const unsigned* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// WG-wide wait until the flags of rows [z0, z0 + nz) (and S_PRIOR when `prior_flag`) are set: ONE
// thread polls, then the barrier, then every thread acquires.
__device__ __forceinline__ void g1_wg_wait(const unsigned* sync, unsigned z0, unsigned nz, bool prior_flag) {
    if (threadIdx.x == 0) {
        for (;;) {
            bool ok = !prior_flag || ld_relaxed(sync + S_PRIOR) != 0u;
            for (unsigned z = 0; z < nz; ++z) ok = ok && ld_relaxed(sync + S_READY(z0 + z)) != 0u;
            if (ok) break;
            __builtin_amdgcn_s_sleep(2);
        }
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

// ---- 32-bit wave32 butterflies: DPP row_xmask 1/2/4/8 inside each 16-lane row, then permlanex16
// across the rows. Every lane ends with the result.
template <int CTRL>
__device__ __forceinline__ unsigned g1_dpp(unsigned v) {
    return (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, CTRL, 0xf, 0xf, false);
}
__device__ __forceinline__ unsigned g1_x16(unsigned v) {
    return (unsigned)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210u, 0xfedcba98u, false, false);
}
__device__ __forceinline__ unsigned g1_wmax_u32(unsigned v) {
    v = max(v, g1_dpp<0x161>(v));
    v = max(v, g1_dpp<0x162>(v));
    v = max(v, g1_dpp<0x164>(v));
    v = max(v, g1_dpp<0x168>(v));
    return max(v, g1_x16(v));
}
__device__ __forceinline__ unsigned g1_wmin_u32(unsigned v) {
    v = min(v, g1_dpp<0x161>(v));
    v = min(v, g1_dpp<0x162>(v));
    v = min(v, g1_dpp<0x164>(v));
    v = min(v, g1_dpp<0x168>(v));
    return min(v, g1_x16(v));
}
__device__ __forceinline__ float g1_wmax_f32(float f) {
    f = fmaxf(f, __uint_as_float(g1_dpp<0x161>(__float_as_uint(f))));
    f = fmaxf(f, __uint_as_float(g1_dpp<0x162>(__float_as_uint(f))));
    f = fmaxf(f, __uint_as_float(g1_dpp<0x164>(__float_as_uint(f))));
    f = fmaxf(f, __uint_as_float(g1_dpp<0x168>(__float_as_uint(f))));
    return fmaxf(f, __uint_as_float(g1_x16(__float_as_uint(f))));
}
__device__ __forceinline__ float g1_wmin_f32(float f) {
    f = fminf(f, __uint_as_float(g1_dpp<0x161>(__float_as_uint(f))));
    f = fminf(f, __uint_as_float(g1_dpp<0x162>(__float_as_uint(f))));
    f = fminf(f, __uint_as_float(g1_dpp<0x164>(__float_as_uint(f))));
    f = fminf(f, __uint_as_float(g1_dpp<0x168>(__float_as_uint(f))));
    return fminf(f, __uint_as_float(g1_x16(__float_as_uint(f))));
}

// --------------------------------------------------------------------------------------------
// One selection pass (see TOP-K above): the winner expert (uniform) and its prob; the winner is
// masked to ord(-INF) in its owner lane. `p` = this lane's 12 probs.
// --------------------------------------------------------------------------------------------
__device__ __forceinline__ int g1_pass(unsigned (&h)[G1_EPL], const float (&p)[G1_EPL], unsigned lane, float& pw) {
    // lane-local max tree on (h, i): the right side wins only when strictly greater
    unsigned h1[6], i1[6];
    #pragma unroll
    for (unsigned k = 0; k < 6; ++k) {
        const bool g = h[2 * k + 1] > h[2 * k];
        h1[k] = g ? h[2 * k + 1] : h[2 * k];
        i1[k] = g ? 2 * k + 1 : 2 * k;
    }
    unsigned h2[3], i2[3];
    #pragma unroll
    for (unsigned k = 0; k < 3; ++k) {
        const bool g = h1[2 * k + 1] > h1[2 * k];
        h2[k] = g ? h1[2 * k + 1] : h1[2 * k];
        i2[k] = g ? i1[2 * k + 1] : i1[2 * k];
    }
    const bool g3 = h2[1] > h2[0];
    unsigned hb = g3 ? h2[1] : h2[0], ib = g3 ? i2[1] : i2[0];
    const bool g4 = h2[2] > hb;
    hb = g4 ? h2[2] : hb;
    ib = g4 ? i2[2] : ib;
    const unsigned mh = g1_wmax_u32(hb);
    const unsigned win = g1_wmin_u32(hb == mh ? lane + G1_WARP * ib : 0xffffffffu);
    const unsigned wi = win >> 5, wl = win & 31u;
    float pl = 0.0f;
    #pragma unroll
    for (unsigned i = 0; i < G1_EPL; ++i) pl = (i == wi) ? p[i] : pl;
    pw = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(pl), wl));
    if (wl == lane) {
        #pragma unroll
        for (unsigned i = 0; i < G1_EPL; ++i)
            if (i == wi) h[i] = 0x007fffffu;   // ord(-INF): still a candidate, as wfred's masked score
    }
    return (int)win;
}

// LDS of the selection tasks: per row and list (0 = the plain scores, 1 = scores + prior), the
// n_used winners in pass order (expert, prob).
struct G1Sel {
    int e[G1_MAXB][2][8];
    float p[G1_MAXB][2][8];
};

// TASK: one row's n_used passes over one list (kind 0: score; kind 1: score + prior, UNMASKED --
// the emit skips the protected experts instead of masking them: the same sequence of unprotected
// winners as wfred's masked passes, and n_used of them leave at least n_used - n_protect).
__device__ __forceinline__ void g1_sel_task(const G1Args& a, unsigned mat, unsigned z, unsigned kind, unsigned lane, G1Sel& sl) {
    const float* pr = a.scr + G1_SCR_PROB(mat, z);
    const float* scv = a.scr + G1_SCR_SCORE(mat, z);
    float sc[G1_EPL], pv[G1_EPL], pp[G1_EPL];
    #pragma unroll
    for (unsigned i = 0; i < G1_EPL; ++i) sc[i] = scv[lane + G1_WARP * i];
    #pragma unroll
    for (unsigned i = 0; i < G1_EPL; ++i) pp[i] = pr[lane + G1_WARP * i];
    if (kind == 1u) {
        #pragma unroll
        for (unsigned i = 0; i < G1_EPL; ++i) pv[i] = a.prior_dev[lane + G1_WARP * i];
    }
    unsigned h[G1_EPL];
    #pragma unroll
    for (unsigned i = 0; i < G1_EPL; ++i) h[i] = g1_ord(kind == 1u ? (sc[i] + pv[i]) : sc[i]);
    #pragma nounroll
    for (unsigned k = 0; k < a.n_used; ++k) {
        float pw;
        const int w = g1_pass(h, pp, lane, pw);
        if (lane == 0) { sl.e[z][kind][k] = w; sl.p[z][kind][k] = pw; }
    }
    // the row's score range (max - min of the plain scores), from the plain task
    if (kind == 0u && mat == 0 && a.range_out != nullptr) {
        float mx = -INFINITY, mn = INFINITY;
        #pragma unroll
        for (unsigned i = 0; i < G1_EPL; ++i) { mx = fmaxf(mx, sc[i]); mn = fminf(mn, sc[i]); }
        const float range = g1_wmax_f32(mx) - g1_wmin_f32(mn);
        if (lane == 0) {
            a.range_out[z] = range;
            if (a.pack != nullptr && a.pk_range != G1_NONE) a.pack[a.pk_range + z] = __float_as_uint(range);
        }
    }
}

// EMIT: row z's picks, weights, device outputs, pack words and M62 counts from the tasks' lists.
__device__ __forceinline__ void g1_emit_row(const G1Args& a, unsigned mat, unsigned z, unsigned lane, const G1Sel& sl) {
    const bool prior = (mat == 0) && (a.prior_host != nullptr);
    const unsigned n_used = a.n_used;
    const unsigned n_protect = a.n_protect < n_used ? a.n_protect : n_used;
    int my_orig = -1, my_sel = -1;
    float p_orig = 0.0f, p_sel = 0.0f;
    if (lane < n_used) { my_orig = sl.e[z][0][lane]; p_orig = sl.p[z][0][lane]; }
    my_sel = my_orig;
    p_sel = p_orig;
    if (prior) {
        if (lane >= n_protect) { my_sel = -1; p_sel = 0.0f; }
        unsigned cnt = n_protect;
        #pragma nounroll
        for (unsigned t = 0; t < n_used; ++t) {
            const int e = sl.e[z][1][t];
            bool prot = false;
            #pragma nounroll
            for (unsigned q = 0; q < n_protect; ++q) prot = prot || (e == sl.e[z][0][q]);
            if (!prot && cnt < n_used) {
                if (lane == cnt) { my_sel = e; p_sel = sl.p[z][1][t]; }
                ++cnt;
            }
        }
    }
    const int plain = my_orig;   // the router's own top-n_used (what M62 counts)
    if (prior && a.prior_dry != 0u) {
        const int t = my_sel; my_sel = my_orig; my_orig = t;
        const float tp = p_sel; p_sel = p_orig; p_orig = tp;
    }
    // Weights: lane j's prob, summed in pick order (wfred: p = s >= 0 ? prob : 0).
    const float pj = (lane < n_used && my_sel >= 0) ? p_sel : 0.0f;
    float sum = 0.0f;
    #pragma nounroll
    for (unsigned j = 0; j < n_used; ++j) sum += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(pj), j));
    if (sum < a.weight_eps) sum = a.weight_eps;
    if (lane < n_used) {
        const unsigned o = z * n_used + lane;
        const float wj = pj / sum * a.scale;
        if (mat == 0) {
            a.sel[o] = my_sel;
            a.ew[o] = wj;
            if (a.orig_sel != nullptr) a.orig_sel[o] = my_orig;
            if (a.pack != nullptr) {
                if (a.pk_sel != G1_NONE) a.pack[a.pk_sel + o] = (unsigned)my_sel;
                if (a.pk_ew != G1_NONE) a.pack[a.pk_ew + o] = __float_as_uint(wj);
                if (a.pk_orig != G1_NONE) a.pack[a.pk_orig + o] = (unsigned)my_orig;
            }
            if (a.sel_bank != nullptr && plain >= 0 && (unsigned)plain < G1_NEXP)
                atomicAdd(&a.sel_bank[a.bank_off + (unsigned)plain], 1u);
        } else {
            a.look_sel[o] = my_sel;
            a.look_ew[o] = wj;
            if (a.pack != nullptr) {
                if (a.pk_look != G1_NONE) a.pack[a.pk_look + o] = (unsigned)my_sel;
                if (a.pk_look_ew != G1_NONE) a.pack[a.pk_look_ew + o] = __float_as_uint(wj);
            }
        }
    }
}

// The tail of matrix `mat` (the WG whose router part finished last): the selection tasks spread
// over the 8 waves (a row's plain and prior passes run in different waves), then one wave per row
// composes and emits.
__device__ __forceinline__ void g1_tail(const G1Args& a, unsigned mat, unsigned warp, unsigned lane, G1Sel& sl, unsigned long long* ts) {
    const unsigned b = a.b;
    const unsigned kinds = (mat == 0 && a.prior_host != nullptr) ? 2u : 1u;
    for (unsigned t = warp; t < b * kinds; t += 8u) g1_sel_task(a, mat, t / kinds, t % kinds, lane, sl);
    __syncthreads();
    G1_TS(10);
    for (unsigned z = warp; z < b; z += 8u) g1_emit_row(a, mat, z, lane, sl);
    G1_TS(11);
}

// --------------------------------------------------------------------------------------------
// Q8_K of one 256-element block (row z, block j) by one wave: lane l holds t = l + 32 k, k < 8.
// --------------------------------------------------------------------------------------------
__device__ __forceinline__ void g1_q8k_block(const G1Args& a, unsigned z, unsigned j, unsigned lane) {
    const float* xb = a.norm_out + (size_t)z * G1_NE + j * 256u;
    float x[8];
    #pragma unroll
    for (unsigned k = 0; k < 8; ++k) x[k] = xb[lane + 32u * k];
    unsigned la = 0u;
    #pragma unroll
    for (unsigned k = 0; k < 8; ++k) la = max(la, __float_as_uint(x[k]) & 0x7fffffffu);
    const unsigned amax_bits = g1_wmax_u32(la);
    const float amax = __uint_as_float(amax_bits);
    // the winner among the maxima: min bitrev8(t), and its sign
    unsigned lr = 0xffffffffu, lsign = 0u;
    #pragma unroll
    for (unsigned k = 0; k < 8; ++k) {
        const unsigned r = __brev(lane + 32u * k) >> 24;
        const bool m = (__float_as_uint(x[k]) & 0x7fffffffu) == amax_bits;
        if (m && r < lr) { lr = r; lsign = __float_as_uint(x[k]) & 0x80000000u; }
    }
    const unsigned rmin = g1_wmin_u32(lr);
    const unsigned sbit = (lr == rmin) ? lsign : 0u;
    const unsigned wsign = g1_wmax_u32(sbit);   // only the winner contributes
    const float mxv = __uint_as_float(amax_bits | wsign);

    unsigned* dst_d = a.xq != nullptr ? a.xq + (size_t)(z * G1_PE + j) * G1_QBW : nullptr;
    unsigned* dst_p = (a.pack != nullptr && a.pk_xq != G1_NONE) ? a.pack + a.pk_xq + (z * G1_PE + j) * G1_QBW : nullptr;
    int q[8];
    float dval = 0.0f;
    if (amax == 0.0f) {
        #pragma unroll
        for (unsigned k = 0; k < 8; ++k) q[k] = 0;
    } else {
        const float iscale = -127.0f / mxv;
        #pragma unroll
        for (unsigned k = 0; k < 8; ++k) {
            int qv = (int)rintf(iscale * x[k]);
            if (qv > 127) qv = 127;
            if (qv < -128) qv = -128;
            q[k] = qv;
        }
        dval = 1.0f / iscale;
    }
    // qs: byte t at 4 + t -> word 1 + t / 4 = 1 + lane / 4 + 8 k (lanes 4m..4m+3, same k)
    #pragma unroll
    for (unsigned k = 0; k < 8; ++k) {
        const unsigned b0 = (unsigned)q[k] & 0xffu;
        const unsigned b1 = (unsigned)__shfl_down((int)b0, 1, 32);
        const unsigned b2 = (unsigned)__shfl_down((int)b0, 2, 32);
        const unsigned b3 = (unsigned)__shfl_down((int)b0, 3, 32);
        if ((lane & 3u) == 0u) {
            const unsigned w = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            const unsigned o = 1u + (lane >> 2) + 8u * k;
            if (dst_d) dst_d[o] = w;
            if (dst_p) dst_p[o] = w;
        }
    }
    // bsums: group g = t / 16 = 2 k + (lane >= 16), int16 at byte 260 + 2 g -> word 65 + k
    #pragma unroll
    for (unsigned k = 0; k < 8; ++k) {
        int s = q[k];
        s += __shfl_xor(s, 8, 16);
        s += __shfl_xor(s, 4, 16);
        s += __shfl_xor(s, 2, 16);
        s += __shfl_xor(s, 1, 16);
        const int hi = __shfl(s, 16, 32);
        if (lane == 0) {
            const unsigned w = ((unsigned)s & 0xffffu) | (((unsigned)hi & 0xffffu) << 16);
            if (dst_d) dst_d[65u + k] = w;
            if (dst_p) dst_p[65u + k] = w;
        }
    }
    if (lane == 0) {
        if (dst_d) dst_d[0] = __float_as_uint(dval);
        if (dst_p) dst_p[0] = __float_as_uint(dval);
    }
}

// --------------------------------------------------------------------------------------------
// Router wave: weight row `row` of matrix `mat`, batch rows [0, b), b <= ZB. CH = elements per lane
// per chunk (h20: 20; smaller at larger ZB to bound the x registers).
// --------------------------------------------------------------------------------------------
template <unsigned ZB, unsigned CH>
__device__ __forceinline__ void g1_router_wave(const G1Args& a, unsigned mat, unsigned row, unsigned lane,
                                               unsigned long long* ts) {
    constexpr unsigned NCH = G1_EPW / CH;     // chunks
    constexpr unsigned PF = G1_PFE / CH;      // prefetched chunks
    const unsigned b = a.b;
    const __half* __restrict__ wr = (mat == 0 ? a.w0 : a.w1) + (size_t)row * G1_NE;
    // Prefetch: this lane's first G1_PFE weight elements (independent of the collapse).
    if (a.dbg & 64u) {   // bench: start the prefetch ~1.5 us late (the collapse's loads first)
        const unsigned long long t0 = g1_now();
        while (g1_now() - t0 < 150ull) __builtin_amdgcn_s_sleep(2);
    }
    if (a.dbg & 128u) g1_wg_wait(a.sync, 0, (a.dbg & 8u) ? 0u : b, false);   // bench: no prefetch
    __half wp[PF][CH];
    #pragma unroll
    for (unsigned c = 0; c < PF; ++c) {
        #pragma unroll
        for (unsigned t = 0; t < CH; ++t) wp[c][t] = wr[c * (G1_WARP * CH) + lane + G1_WARP * t];
    }
    G1_TS(2);
    if (!(a.dbg & 128u)) g1_wg_wait(a.sync, 0, (a.dbg & 8u) ? 0u : b, false);
    G1_TS(3);
    const float* xb = a.norm_out;
    float acc[ZB];
    #pragma unroll
    for (unsigned z = 0; z < ZB; ++z) acc[z] = 0.0f;
    #pragma unroll
    for (unsigned c = 0; c < PF; ++c) {
        const unsigned base = c * (G1_WARP * CH) + lane;
        #pragma unroll
        for (unsigned z = 0; z < ZB; ++z) {
            if (z < b) {
                float xv[CH];
                #pragma unroll
                for (unsigned t = 0; t < CH; ++t) xv[t] = xb[(size_t)z * G1_NE + base + G1_WARP * t];
                #pragma unroll
                for (unsigned t = 0; t < CH; ++t) acc[z] += __half2float(wp[c][t]) * xv[t];
            }
        }
    }
    // The rest: h20's double-buffered schedule (next chunk's loads before this chunk's FMAs) as a
    // RUNTIME loop over chunk pairs (fully unrolled, the scheduler hoists every load and spills).
    {
        __half wv[2][CH];
        float xv[2][ZB][CH];
        auto issue = [&](unsigned c, unsigned slot) {
            const unsigned base = c * (G1_WARP * CH) + lane;
            #pragma unroll
            for (unsigned t = 0; t < CH; ++t) wv[slot][t] = wr[base + G1_WARP * t];
            #pragma unroll
            for (unsigned z = 0; z < ZB; ++z)
                if (z < b) {
                    #pragma unroll
                    for (unsigned t = 0; t < CH; ++t) xv[slot][z][t] = xb[(size_t)z * G1_NE + base + G1_WARP * t];
                }
        };
        auto fma_slot = [&](unsigned slot) {
            #pragma unroll
            for (unsigned z = 0; z < ZB; ++z)
                if (z < b) {
                    #pragma unroll
                    for (unsigned t = 0; t < CH; ++t) acc[z] += __half2float(wv[slot][t]) * xv[slot][z][t];
                }
        };
        unsigned nch = NCH;
        asm volatile("" : "+s"(nch));   // a runtime trip count, as h20's
        issue(PF, 0);
        #pragma nounroll
        for (unsigned c = PF; c < nch; c += 2) {
            if (c + 1 < nch) issue(c + 1, 1);
            fma_slot(0);
            if (c + 1 < nch) {
                if (c + 2 < nch) issue(c + 2, 0);
                fma_slot(1);
            }
        }
    }
    // Epilogue: logit (h20's shfl_down tree, lane 0), then lane z: wfred's prob and score of row z.
    float mine = 0.0f;
    #pragma unroll
    for (unsigned z = 0; z < ZB; ++z) {
        if (z < b) {
            const float s = g1_warp_sum(acc[z]);
            const float s0 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(s), 0));
            if (lane == z) mine = s0;
        }
    }
    if (lane < b) {
        float* lg = mat == 0 ? a.logits0 : a.logits1;
        const float* bias = mat == 0 ? a.bias0 : a.bias1;
        lg[(size_t)lane * G1_NEXP + row] = mine;
        const float p = sqrtf(g1_softplus_stable(mine));
        a.scr[G1_SCR_PROB(mat, lane) + row] = p;
        a.scr[G1_SCR_SCORE(mat, lane) + row] = (bias != nullptr) ? (p + bias[row]) : p;
    }
    G1_TS(4);
}

// --------------------------------------------------------------------------------------------
// Router wave, PHASED-LDS form (ZB >= b rows, PHE weight elements per lane per phase): the WG
// stages each K-slab of x (all rows, 32 * PHE elements per row) in LDS ONCE (instead of every wave
// reading all of x from L2: 768 waves x b x 20 KB), and each wave keeps a ring of PFE of its weight
// elements in registers: the first PFE are issued before the row wait -- the router's DRAM stream
// runs under the collapse -- and each finished phase's slot is refilled with the phase PFE / PHE
// ahead. Per (row, z) the accumulation is h20's: one f32 accumulator over j = 0..159 ascending, the
// same h2f(w) * x FMA, the same shfl_down tree.
// --------------------------------------------------------------------------------------------
template <unsigned ZB, unsigned PHE, unsigned PFE>
__device__ __forceinline__ void g1_router_wave_ph(const G1Args& a, unsigned mat, unsigned row, unsigned lane,
                                                  unsigned tid, float* sx, unsigned long long* ts) {
    constexpr unsigned NPH = G1_EPW / PHE;               // phases
    constexpr unsigned RPH = PFE / PHE;                  // phases resident in the ring
    constexpr unsigned SLAB = G1_WARP * PHE;             // x elements per row per phase
    constexpr unsigned ST4 = ZB * SLAB / 4u / G1_BLOCK;  // float4 per thread per phase
    static_assert(G1_EPW % PHE == 0 && PFE % PHE == 0 && (ZB * SLAB) % (4u * G1_BLOCK) == 0, "g1 phase shape");
    const unsigned b = a.b;
    const __half* __restrict__ wr = (mat == 0 ? a.w0 : a.w1) + (size_t)row * G1_NE;
    __half wq[PFE];   // ring: phase p lives at slot (p % RPH) * PHE
    #pragma unroll
    for (unsigned e = 0; e < PFE; ++e) wq[e] = wr[lane + G1_WARP * e];
    G1_TS(2);
    g1_wg_wait(a.sync, 0, (a.dbg & 8u) ? 0u : b, false);
    G1_TS(3);
    const float4* xg = reinterpret_cast<const float4*>(a.norm_out);
    float4* sx4 = reinterpret_cast<float4*>(sx);
    float mine = 0.0f;
    // Batch rows in slices of ZB (b > ZB: one slice after the other; a later slice re-reads its
    // weights, from the caches the first slice filled).
    constexpr unsigned SL = (ZB == 8u) ? G1_MAXB / ZB : 1u;   // row slices (only the ZB = 8 forms loop)
    #pragma unroll
    for (unsigned si = 0; si < SL; ++si) {
        const unsigned z0 = si * ZB;
        if (z0 >= b) break;
        const unsigned nz = min(ZB, b - z0);
        if (z0 != 0u) {
            #pragma unroll
            for (unsigned e = 0; e < PFE; ++e) wq[e] = wr[lane + G1_WARP * e];
        }
        // staging map: thread tid's i-th float4 of the phase = row zz, float4 c of the slab
        auto stage_load = [&](unsigned ph, float4 (&st)[ST4]) {
            #pragma unroll
            for (unsigned i = 0; i < ST4; ++i) {
                const unsigned f = tid + G1_BLOCK * i;            // float4 index in [ZB][SLAB / 4]
                const unsigned zz = f / (SLAB / 4u), c = f % (SLAB / 4u);
                st[i] = (zz < nz) ? xg[(size_t)(z0 + zz) * (G1_NE / 4u) + ph * (SLAB / 4u) + c] : make_float4(0.f, 0.f, 0.f, 0.f);
            }
        };
        float4 st[ST4];
        stage_load(0, st);
        float acc[ZB];
        #pragma unroll
        for (unsigned z = 0; z < ZB; ++z) acc[z] = 0.0f;
        #pragma unroll
        for (unsigned ph = 0; ph < NPH; ++ph) {
            #pragma unroll
            for (unsigned i = 0; i < ST4; ++i) sx4[tid + G1_BLOCK * i] = st[i];
            __syncthreads();
            if (ph + 1 < NPH) stage_load(ph + 1, st);
            const unsigned slot = (ph % RPH) * PHE;
            #pragma unroll
            for (unsigned z = 0; z < ZB; ++z) {
                if (z < nz) {
                    #pragma unroll
                    for (unsigned jj = 0; jj < PHE; ++jj) acc[z] += __half2float(wq[slot + jj]) * sx[z * SLAB + lane + G1_WARP * jj];
                }
            }
            // refill this slot with the phase RPH ahead
            if (ph + RPH < NPH) {
                #pragma unroll
                for (unsigned jj = 0; jj < PHE; ++jj) wq[slot + jj] = wr[lane + G1_WARP * ((ph + RPH) * PHE + jj)];
            }
            __syncthreads();
        }
        #pragma unroll
        for (unsigned z = 0; z < ZB; ++z) {
            if (z < nz) {
                const float s = g1_warp_sum(acc[z]);
                const float s0 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(s), 0));
                if (lane == z0 + z) mine = s0;
            }
        }
    }
    if (lane < b) {
        float* lg = mat == 0 ? a.logits0 : a.logits1;
        const float* bias = mat == 0 ? a.bias0 : a.bias1;
        lg[(size_t)lane * G1_NEXP + row] = mine;
        const float p = sqrtf(g1_softplus_stable(mine));
        a.scr[G1_SCR_PROB(mat, lane) + row] = p;
        a.scr[G1_SCR_SCORE(mat, lane) + row] = (bias != nullptr) ? (p + bias[row]) : p;
    }
    G1_TS(4);
}

// --------------------------------------------------------------------------------------------
// PREFETCH WG (`n_pf` of them, right after the collapse / prior tickets): streams both gate matrices
// DRAM -> L2 / MALL with 16-byte loads, in the order the router waves consume them (the first 2560
// bytes of every row, then the next 2560, ...), so the DRAM stream runs at full width from the
// kernel's start whatever the collapse and the router waves' in-order load counters do; the router
// waves' own loads then mostly hit the caches (or merge with these in flight). The loaded words are
// folded into a value nothing reads (kept alive by an impossible store).
// --------------------------------------------------------------------------------------------
__device__ __forceinline__ void g1_prefetch(const G1Args& a, unsigned idx, unsigned tid) {
    constexpr unsigned UNIT = 2560u;                      // bytes per (row, super-chunk): 5 wave loads
    constexpr unsigned NSC = G1_NE * 2u / UNIT;           // 4 super-chunks per row
    const unsigned rows = G1_NEXP * a.n_mat;
    const unsigned units = NSC * rows;
    const unsigned nw = a.n_pf * 8u;
    const unsigned gw = idx * 8u + (tid >> 5u);
    const unsigned lane = tid & 31u;
    unsigned acc = 0u;
    #pragma nounroll
    for (unsigned u0 = gw; u0 < units; u0 += 4u * nw) {
        uint4 v[4][5];
        #pragma unroll
        for (unsigned q = 0; q < 4; ++q) {
            const unsigned u = u0 + q * nw;
            if (u < units) {
                const unsigned sc = u / rows, r = u % rows;
                const __half* w = (r < G1_NEXP ? a.w0 + (size_t)r * G1_NE : a.w1 + (size_t)(r - G1_NEXP) * G1_NE);
                const uint4* p = reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned char*>(w) + sc * UNIT) + lane;
                #pragma unroll
                for (unsigned k = 0; k < 5; ++k) v[q][k] = p[32u * k];
            } else {
                #pragma unroll
                for (unsigned k = 0; k < 5; ++k) v[q][k] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        #pragma unroll
        for (unsigned q = 0; q < 4; ++q) {
            #pragma unroll
            for (unsigned k = 0; k < 5; ++k) acc ^= v[q][k].x ^ v[q][k].y ^ v[q][k].z ^ v[q][k].w;
        }
    }
    if (acc == 0x9e3779b9u && a.dbg == 0xfffffffeu) a.sync[S_TICKET + 1u] = acc;   // never: keeps the loads
}

// --------------------------------------------------------------------------------------------
// Collapse WG (mhc_fast's collapse branch, verbatim arithmetic), then the prior, then Q8_K.
// --------------------------------------------------------------------------------------------
// The cache prior's WG (ticket b, when there is a prior): pinned host -> d_prior, RELEASE prior_ready.
// Its own WG: a PCIe read is slow, and the in-order load counter would make a WG wait for it before
// any later load of its own.
__device__ __forceinline__ void g1_prior(const G1Args& a, unsigned tid) {
    const float p0 = a.prior_host[tid];
    const float p1 = (tid + G1_BLOCK < G1_NEXP) ? a.prior_host[tid + G1_BLOCK] : 0.0f;
    a.prior_dev[tid] = p0;
    if (tid + G1_BLOCK < G1_NEXP) a.prior_dev[tid + G1_BLOCK] = p1;
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        __hip_atomic_store(a.sync + S_PRIOR, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

__device__ __forceinline__ void g1_collapse(const G1Args& a, unsigned z, unsigned tid, float* sq, unsigned long long* ts) {
    if (a.dbg & 256u) __builtin_amdgcn_s_setprio(3);
    const float* cxb = a.cx + (size_t)z * (size_t)(G1_HC * G1_NE);
    const float* cwb = a.carry + (size_t)z * (size_t)G1_MIX;
    // Load order = need order (the load counter is in order): the carry and x feed cur and the
    // RMS reduction; the (cold) norm weight is only needed after the reduction, so its DRAM
    // latency hides under it instead of gating cur.
    float cw[G1_HC];
    #pragma unroll
    for (uint32_t h = 0; h < G1_HC; h++) cw[h] = cwb[h];
    float xs[G1_HC][G1_PE];
    #pragma unroll
    for (uint32_t h = 0; h < G1_HC; h++) {
        #pragma unroll
        for (uint32_t j = 0; j < G1_PE; j++) xs[h][j] = cxb[(size_t)h * G1_NE + tid + G1_BLOCK * j];
    }
    float nw[G1_PE];
    #pragma unroll
    for (uint32_t j = 0; j < G1_PE; j++) nw[j] = a.norm_w[tid + G1_BLOCK * j];
    __builtin_amdgcn_sched_barrier(0);   // keep this issue order
    float cv[G1_PE];
    #pragma unroll
    for (uint32_t j = 0; j < G1_PE; j++) {
        float acc = 0.0f;
        #pragma unroll
        for (uint32_t h = 0; h < G1_HC; h++) acc += xs[h][j] * cw[h];
        cv[j] = acc;
        a.cur_out[(size_t)z * G1_NE + tid + G1_BLOCK * j] = acc;
    }
    G1_TS(8);
    double partial = 0.0;
    #pragma unroll
    for (uint32_t j = 0; j < G1_PE; j++) {
        const float v = cv[j];
        partial += (double)v * (double)v;
    }
    sq[tid] = (float)partial;
    __syncthreads();
    for (uint32_t s = G1_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sq[tid] += sq[tid + s];
        __syncthreads();
    }
    G1_TS(9);
    const float mean_sq = sq[0] / (float)G1_NE;
    const float nscale = rsqrtf(mean_sq + a.rms_eps);
    #pragma unroll
    for (uint32_t j = 0; j < G1_PE; j++) a.norm_out[(size_t)z * G1_NE + tid + G1_BLOCK * j] = cv[j] * nscale * nw[j];
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        __hip_atomic_store(a.sync + S_READY(z), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    G1_TS(2);
    // Q8_K of the row: wave w does blocks w, w + 8, w + 16 (reading the row back: this WG's own
    // writes, ordered by the barrier above).
    const bool want_xq = (a.xq != nullptr) || (a.pack != nullptr && a.pk_xq != G1_NONE);
    if (want_xq && !(a.dbg & 2u)) {
        const unsigned lane = tid & (G1_WARP - 1u), warp = tid >> 5u;
        #pragma nounroll
        for (unsigned j = warp; j < G1_PE; j += 8u) g1_q8k_block(a, z, j, lane);
    }
    G1_TS(3);
}

// PHE = 0: the register form (g1_router_wave<ZB, CH>); else the phased-LDS form <ZB, PHE, PFE = CH>.
template <unsigned ZB, unsigned CH, unsigned PHE>
__device__ void g1_body(const G1Args& a) {
    __shared__ unsigned s_ticket;
    __shared__ unsigned s_last;
    // collapse: sq[256]; router (phased-LDS form): the x slab [ZB][32 * PHE]; tails: probs [8][384]
    constexpr unsigned SBUF = (PHE != 0 && ZB * G1_WARP * PHE > 8 * G1_NEXP) ? ZB * G1_WARP * PHE : 8 * G1_NEXP;
    __shared__ __attribute__((aligned(16))) float s_buf[SBUF];
    static_assert(sizeof(G1Sel) <= sizeof(float) * SBUF, "g1 tail LDS");
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & (G1_WARP - 1u);
    const unsigned warp = tid >> 5u;
    const unsigned long long t_start = g1_now();
    if (tid == 0) s_ticket = (a.dbg & 16u) ? blockIdx.x : atomicAdd(a.sync + S_TICKET, 1u);
    __syncthreads();
    const unsigned t = s_ticket;
    unsigned long long* ts = (a.ts != nullptr && tid == 0) ? a.ts + (size_t)t * 16u : nullptr;
    G1_TS(1);
    if (ts != nullptr) ts[0] = t_start;
    const unsigned b = a.b;
    const unsigned np = (a.prior_host != nullptr) ? 1u : 0u;
    const unsigned total = b + np + a.n_pf + a.n_mat * G1_RWG;

    if (t < b) {
        g1_collapse(a, t, tid, s_buf, ts);
    } else if (t >= b + np && t < b + np + a.n_pf) {
        g1_prefetch(a, t - b - np, tid);
        G1_TS(3);
    } else if (t < b + np) {
        g1_prior(a, tid);
        G1_TS(3);
    } else {
        const unsigned r = t - b - np - a.n_pf;
        const unsigned mat = r / G1_RWG;
        const unsigned row = (r - mat * G1_RWG) * 8u + warp;
        if constexpr (PHE != 0) g1_router_wave_ph<ZB, (PHE ? PHE : 1), CH>(a, mat, row, lane, tid, s_buf, ts);
        else g1_router_wave<ZB, CH>(a, mat, row, lane, ts);
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            const unsigned prev = atomicAdd(a.sync + S_RDONE(mat), 1u);
            s_last = (prev == G1_RWG - 1u) ? 1u : 0u;
        }
        __syncthreads();
        if (s_last != 0u) {
            g1_wg_wait(a.sync, 0, 0, mat == 0 && a.prior_host != nullptr);
            G1_TS(5);
            if (!(a.dbg & 4u)) g1_tail(a, mat, warp, lane, *reinterpret_cast<G1Sel*>(s_buf), ts);
            __syncthreads();
            G1_TS(6);
        }
    }

    // ---- finish: the last WG resets the scratch and posts the completion word.
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned prev = atomicAdd(a.sync + S_FINISH, 1u);
        G1_TS(7);
        if (prev == total - 1u) {
            __threadfence();
            a.sync[S_TICKET] = 0u;
            a.sync[S_RDONE(0)] = 0u;
            a.sync[S_RDONE(1)] = 0u;
            a.sync[S_PRIOR] = 0u;
            for (unsigned z = 0; z < G1_MAXB; ++z) a.sync[S_READY(z)] = 0u;
            a.sync[S_FINISH] = 0u;
            if (a.done_word != nullptr) {
                __threadfence_system();
                __hip_atomic_store(a.done_word, a.done_value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
    }
}

// ============================================================================================
// FAMILY "r" (G1-optimal, symbols g1_router_block_r{ZB}_{CPP}_{P0}_{KW}_{RX}_p{KP}_w{NW}): the same
// roles, outputs and per-output arithmetic as the families above (BIT-IDENTICAL to the production
// chain: tests/g1_router_block_bitexact.rs), with the schedule rebuilt from what the hardware does.
// Measured on gfx1201 beside the hub (tests/g1o_probe.rs + the in-kernel trace of
// tests/bench_g1_router_block.rs, 10-13):
//   (1) A CU's vector memory path is a FIFO: an L2-hot load issued behind a WG's 80 outstanding
//       gate loads per lane waits ~10 us. Scalar loads do not queue there (0.35 us vs 0.09 idle).
//   (2) The gate stream reaches the byte roof with only 8 u32 words per lane in flight (603 GB/s,
//       the same as a whole-row prefetch); u16 lane-strided loads (h20's) peak at 533 GB/s.
//   (3) Outstanding bytes slow every other access: store + fence 0.16 us at 10 words per lane in
//       flight, 2.4 us at 40, 3.2 us at 80; scalar loads 0.23 -> 1.17 us. The collapse runs under
//       the stream, so a deep prefetch delays its flag one-for-one.
//   (4) `__syncthreads` drains every outstanding load in WGP mode (s_wait_loadcnt 0), so phase
//       barriers here are LDS-only; and a scoped s_load can hit a stale scalar-cache line, so every
//       poll invalidates the scalar cache first.
// Hence:
//   * ROUTER WG = NW waves, one gate row each, all b batch rows (each weight byte read once). Gate
//     words are loaded as u32 (lane 2k+s loads word 32c + 16s + k = elements 64c + 32s + 2k, +1) and
//     re-dealt to h20's lanes by ONE DPP quad swap + ONE v_perm per word: every lane still
//     accumulates i = lane + 32 j, j ascending, with h20's FMA and shfl_down tree.
//   * P0 words per lane are prefetched at entry, PACED (word c is issued once word c - KP landed):
//     the stream runs at the roof under the collapse without queueing in front of it.
//   * One wave per WG polls the collapse flags (scalar loads, not behind the stream); then each wave
//     stages ITS share of x (ffn_input_norm) for RX phases ahead, issued in consumption order with
//     its gate words (the FIFO returns x when the phase needs it). Per phase (CPP words = 64 CPP
//     elements per row): write the phase's x share into an LDS slab, one LDS-only barrier, issue x
//     for phase + RX and the gate words up to the ring front, then the phase's FMAs (LDS reads in
//     batches, no branch on b in the loop: rows past b compute clamped copies that are discarded --
//     a guarded row puts each LDS read in its own block and the waitcnt pass then drains the ring).
//   * COLLAPSE WG: mhc_fast's collapse, its tree with LDS-only barriers and wave shuffles (the same
//     adds in the same order); PRIOR WG: pulls the norm weight into L2 first, then copies the prior.
//   * TAIL WGs (highest tickets): one selection task per wave, all in parallel. The selection is
//     THRESHOLD + RANK: T = the n_used-th largest of the 32 lane maxima (every top-n_used element is
//     >= T), the candidates >= T rank themselves under wfred's key; the serial passes remain as the
//     fallback (> 32 candidates, or T <= ord(-INF)). The emit's prior compose is lane-parallel.
// Every poll is bounded (G1R_SPIN_CAP ticks of the 100 MHz counter, 2 s): on expiry it sets the
// sticky S_ERR word (G1RouterBlock::error_word) and proceeds, so a broken flag protocol fails the
// tests instead of hanging the GPU. Forward progress as before: every wait is on lower tickets.
// --------------------------------------------------------------------------------------------
#define G1R_NWC (G1_NE / 64u)              // 80 u32 words per lane per gate row

// A barrier that waits for this wave's LDS operations only (no global-load drain).
__device__ __forceinline__ void g1r_lds_barrier() {
#if defined(__gfx1200__) || defined(__gfx1201__)
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
#else
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#endif
}

#define G1R_SPIN_CAP 200000000ull          // 2 s at 100 MHz

// (s_dcache_inv first: a scoped s_load alone was measured to return a stale flag from the scalar
// cache -- polls timed out at the cap in tests/g1_router_block_bitexact.rs.)
__device__ __forceinline__ unsigned g1_sload_dev(const unsigned* pv) {
    // the address is uniform; make that explicit (an LDS-derived index looks divergent)
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

// Invalidate this CU's vector caches (L0 / GL1) WITHOUT waiting on outstanding loads (a C++ acquire
// fence waits for every outstanding load: the whole prefetched ring).
__device__ __forceinline__ void g1_vinv() {
#if defined(__gfx1200__) || defined(__gfx1201__)
    asm volatile("global_inv scope:SCOPE_DEV" ::: "memory");
#else
    asm volatile("buffer_gl0_inv\n\tbuffer_gl1_inv" ::: "memory");
#endif
}

// Wave-uniform bounded scalar poll until *p >= v (SLEEP: s_sleep units between polls).
template <unsigned SLEEP = 2>
__device__ __forceinline__ void g1_spoll_ge(const unsigned* p, unsigned v, unsigned* err) {
    const unsigned long long t0 = g1_now();
    while (g1_sload_dev(p) < v) {
        if (g1_now() - t0 > G1R_SPIN_CAP) {
            if ((threadIdx.x & 31u) == 0u) atomicOr(err, 1u);
            break;
        }
        if constexpr (SLEEP > 0) __builtin_amdgcn_s_sleep(SLEEP);
    }
}

#define S_ERR (S_WORDS)                    // sticky error word (its own line, after the flags)

__device__ __forceinline__ void g1r_pin() { asm volatile("" ::: "memory"); }

// Compile-time loop: f(g1_uc<I>{}) for I in [I0, N) (each body sees a constant index).
template <unsigned V> struct g1_uc { static constexpr unsigned value = V; };
template <unsigned I, unsigned N, class F>
__device__ __forceinline__ void g1_static_for(F&& f) {
    if constexpr (I < N) {
        f(g1_uc<I>{});
        g1_static_for<I + 1u, N>(f);
    }
}

template <unsigned P0, unsigned CPP, unsigned KW>
__host__ __device__ constexpr unsigned g1r_front(int ph) {
    // gate words issued once phase ph's issue point has run (ph = -1: the entry prefetch).
    // KW > 0: a ring KW words ahead of consumption (never below P0); KW = 0: CONTINUOUS -- P0 ahead and
    // CPP more at every phase, so the stream does not pause while the phases catch up after the flag.
    return ph < 0 ? (P0 < G1R_NWC ? P0 : G1R_NWC)
         : KW == 0u ? (P0 + ((unsigned)ph + 1u) * CPP < G1R_NWC ? P0 + ((unsigned)ph + 1u) * CPP : G1R_NWC)
                    : ((((unsigned)ph + 1u) * CPP + KW > P0 ? ((unsigned)ph + 1u) * CPP + KW : P0) < G1R_NWC
                           ? (((unsigned)ph + 1u) * CPP + KW > P0 ? ((unsigned)ph + 1u) * CPP + KW : P0)
                           : G1R_NWC);
}

template <unsigned ZB, unsigned CPP, unsigned P0, unsigned KW, unsigned RX, unsigned KP, unsigned NW, class BAR>
__device__ __forceinline__ void g1r_router(const G1Args& a, unsigned mat, unsigned row, unsigned lane, unsigned warp,
                                           float* sx, unsigned long long* ts, BAR&& bar) {
    constexpr unsigned NPH = G1R_NWC / CPP;
    constexpr unsigned PHE = 64u * CPP;
    constexpr unsigned SLAB = ZB * PHE;
    constexpr unsigned PRS = ZB * CPP;                 // (row, 64-element chunk) pairs per phase
    constexpr unsigned NI = (PRS + NW - 1u) / NW;      // pairs staged per wave (the last ones clamped)
    static_assert(G1R_NWC % CPP == 0 && RX >= 1 && RX <= NPH, "g1r shape");
    const unsigned b = a.b;
    const unsigned* __restrict__ wr = reinterpret_cast<const unsigned*>((mat == 0 ? a.w0 : a.w1) + (size_t)row * G1_NE);
    const unsigned kk = lane >> 1, ss = lane & 1u;
    const unsigned psel = ss ? 0x03020706u : 0x05040100u;
    const float* __restrict__ xg = a.norm_out;
    unsigned wv[G1R_NWC];
    float xr[RX][NI][2];
    // this wave's x share: pairs warp + NW i (clamped to the last pair: a duplicate load + the same
    // LDS write, no branch); rows past b load row b - 1 (results discarded)
    auto load_x = [&](unsigned q, float (&r)[NI][2]) {
        #pragma unroll
        for (unsigned i = 0; i < NI; ++i) {
            const unsigned pr = min(warp + NW * i, PRS - 1u);
            const unsigned z = min(pr / CPP, b - 1u), cs = pr % CPP;
            const float* p = xg + (size_t)z * G1_NE + q * PHE + 64u * cs + lane;
            r[i][0] = p[0];
            r[i][1] = p[32];
        }
    };
    auto write_x = [&](unsigned q, const float (&r)[NI][2]) {
        float2* d = reinterpret_cast<float2*>(sx + (q & 1u) * SLAB);
        #pragma unroll
        for (unsigned i = 0; i < NI; ++i) {
            const unsigned pr = min(warp + NW * i, PRS - 1u);
            d[pr * 32u + lane] = make_float2(r[i][0], r[i][1]);   // pair (z, cs) at z * CPP + cs
        }
    };
    // ---- entry prefetch: the first P0 words, PACED (word c issued once word c - KP has landed: the
    // stream keeps ~KP words per lane in flight -- enough for the byte roof -- instead of queueing P0
    // words at once, which delays the collapse's loads and stores by the queue's drain time)
    g1_static_for<0u, g1r_front<P0, CPP, KW>(-1)>([&](auto cc_) {
        constexpr unsigned c = decltype(cc_)::value;
        if constexpr (KP > 0 && c >= KP) {
#if defined(__gfx1200__) || defined(__gfx1201__)
            asm volatile("s_wait_loadcnt %0" :: "n"(KP - 1) : "memory");
#else
            asm volatile("s_waitcnt vmcnt(%0)" :: "n"(KP - 1) : "memory");
#endif
        }
        wv[c] = wr[32u * c + 16u * ss + kk];
    });
    g1r_pin();
    G1_TS(2);
    // ---- the rows' flags: ONE wave per WG polls (scalar loads are not behind the ring; a scalar-cache
    // invalidate per poll -- every wave polling thrashed the WGP's scalar cache); the rest wait at an
    // LDS-only barrier with their rings in flight
    if (warp == 0 && !(a.dbg & 8u)) {
        for (unsigned z = 0; z < b; ++z) g1_spoll_ge(a.sync + S_READY(z), 1u, a.sync + S_ERR);
    }
    bar();
    g1_vinv();
    G1_TS(3);
    g1_static_for<0u, RX>([&](auto qc) { load_x(decltype(qc)::value, xr[decltype(qc)::value]); });
    g1r_pin();
    float acc[ZB];
    #pragma unroll
    for (unsigned z = 0; z < ZB; ++z) acc[z] = 0.0f;
    unsigned long long* tph = (ts != nullptr) ? ts + 1024u * 16u : nullptr;   // bench: per-phase completion times
    g1_static_for<0u, NPH>([&](auto phc) {
        constexpr unsigned ph = decltype(phc)::value;
        if (tph != nullptr && ph > 0) tph[ph - 1] = g1_now();
        write_x(ph, xr[ph % RX]);
        bar();
        if constexpr (ph + RX < NPH) load_x(ph + RX, xr[ph % RX]);
        constexpr unsigned c0 = g1r_front<P0, CPP, KW>((int)ph - 1), c1 = g1r_front<P0, CPP, KW>((int)ph);
        g1_static_for<c0, c1>([&](auto cc_) {
            constexpr unsigned c = decltype(cc_)::value;
            wv[c] = wr[32u * c + 16u * ss + kk];
        });
        g1r_pin();
        const float2* xs = reinterpret_cast<const float2*>(sx + (ph & 1u) * SLAB);
        // Half-phases: all LDS reads of HC chunks x ZB rows first (one batch in flight), then their
        // FMAs. Every row 0..ZB-1 is computed (rows past b: clamped copies, discarded): no branch in
        // the loop -- a guarded row puts each LDS read in its own block, waited one at a time.
        constexpr unsigned HC0 = CPP >= 4u ? CPP / 2u : CPP;
        constexpr unsigned HC = (ZB >= 16u && HC0 > 1u) ? 1u : ((ZB >= 8u && HC0 > 2u) ? 2u : HC0);   // bound the LDS batch registers
        g1_static_for<0u, CPP / HC>([&](auto hcc) {
            constexpr unsigned h0 = decltype(hcc)::value * HC;
            float2 xv[HC][ZB];
            g1_static_for<0u, HC>([&](auto i_) {
                constexpr unsigned i = decltype(i_)::value;
                #pragma unroll
                for (unsigned z = 0; z < ZB; ++z) xv[i][z] = xs[(z * CPP + h0 + i) * 32u + lane];
            });
            g1_static_for<0u, HC>([&](auto i_) {
                constexpr unsigned i = decltype(i_)::value;
                const unsigned u = wv[ph * CPP + h0 + i];
                const unsigned o = (unsigned)__builtin_amdgcn_mov_dpp((int)u, 0xB1, 0xf, 0xf, false);   // quad_perm [1,0,3,2]
                const unsigned e = __builtin_amdgcn_perm(o, u, psel);
                const float w0 = __half2float(__ushort_as_half((unsigned short)(e & 0xffffu)));
                const float w1 = __half2float(__ushort_as_half((unsigned short)(e >> 16)));
                #pragma unroll
                for (unsigned z = 0; z < ZB; ++z) {
                    acc[z] += w0 * xv[i][z].x;
                    acc[z] += w1 * xv[i][z].y;
                }
            });
            // pin the FMA chain here (pure ALU is not ordered by the memory clobbers: the scheduler
            // otherwise runs every chunk's DPP/perm first and the serial FMA chain last -> spills)
            #pragma unroll
            for (unsigned z = 0; z < ZB; ++z) asm volatile("" : "+v"(acc[z]));
            __builtin_amdgcn_sched_barrier(0);
        });
    });
    if (tph != nullptr) tph[NPH - 1] = g1_now();
    float mine = 0.0f;
    #pragma unroll
    for (unsigned z = 0; z < ZB; ++z) {
        if (z < b) {
            const float s = g1_warp_sum(acc[z]);
            const float s0 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(s), 0));
            if (lane == z) mine = s0;
        }
    }
    if (lane < b) {
        float* lg = mat == 0 ? a.logits0 : a.logits1;
        const float* bias = mat == 0 ? a.bias0 : a.bias1;
        lg[(size_t)lane * G1_NEXP + row] = mine;
        const float p = sqrtf(g1_softplus_stable(mine));
        a.scr[G1_SCR_PROB(mat, lane) + row] = p;
        a.scr[G1_SCR_SCORE(mat, lane) + row] = (bias != nullptr) ? (p + bias[row]) : p;
    }
    G1_TS(4);
}

// Collapse WG of family "r" (32 NW threads): mhc_fast's collapse arithmetic verbatim on threads
// 0..255; with NW > 8 the extra waves bring the (cold) norm weight into LDS meanwhile (no x load
// queues behind a DRAM read in its wave's in-order counter); NW = 8 reads it last, from L2 (the
// prior WG pulled it there first).
template <unsigned NW>
__device__ __forceinline__ void g1r_collapse(const G1Args& a, unsigned z, unsigned tid, float* sq, float* snw, unsigned long long* ts) {
    float cv[G1_PE];
    float nw[G1_PE];
    if (tid >= G1_BLOCK) {
        if constexpr (NW > 8u) {
            const float4* nw4 = reinterpret_cast<const float4*>(a.norm_w);
            float4* s4 = reinterpret_cast<float4*>(snw);
            for (unsigned i = tid - G1_BLOCK; i < G1_NE / 4u; i += 32u * NW - G1_BLOCK) s4[i] = nw4[i];
        }
    } else {
        const float* cxb = a.cx + (size_t)z * (size_t)(G1_HC * G1_NE);
        const float* cwb = a.carry + (size_t)z * (size_t)G1_MIX;
        float cw[G1_HC];
        #pragma unroll
        for (uint32_t h = 0; h < G1_HC; h++) cw[h] = cwb[h];
        float xs[G1_HC][G1_PE];
        #pragma unroll
        for (uint32_t h = 0; h < G1_HC; h++) {
            #pragma unroll
            for (uint32_t j = 0; j < G1_PE; j++) xs[h][j] = cxb[(size_t)h * G1_NE + tid + G1_BLOCK * j];
        }
        if constexpr (NW == 8u) {
            #pragma unroll
            for (uint32_t j = 0; j < G1_PE; j++) nw[j] = a.norm_w[tid + G1_BLOCK * j];
        }
        __builtin_amdgcn_sched_barrier(0);   // keep this issue order
        #pragma unroll
        for (uint32_t j = 0; j < G1_PE; j++) {
            float acc = 0.0f;
            #pragma unroll
            for (uint32_t h = 0; h < G1_HC; h++) acc += xs[h][j] * cw[h];
            cv[j] = acc;
            a.cur_out[(size_t)z * G1_NE + tid + G1_BLOCK * j] = acc;
        }
        G1_TS(8);
        double partial = 0.0;
        #pragma unroll
        for (uint32_t j = 0; j < G1_PE; j++) {
            const float v = cv[j];
            partial += (double)v * (double)v;
        }
        sq[tid] = (float)partial;
    }
    // mhc_fast's tree (sq[tid] += sq[tid + s], s = 128 .. 1) with LDS-only barriers (no wait on the cur
    // stores) for the cross-wave levels and wave 0's shuffles for s <= 16: the same adds, same order.
    if (NW > 8u) __syncthreads();   // the norm weight's LDS copy (extra waves) is complete too
    g1r_lds_barrier();
    if (tid < 128u) sq[tid] += sq[tid + 128u];
    g1r_lds_barrier();
    if (tid < 64u) sq[tid] += sq[tid + 64u];
    g1r_lds_barrier();
    if (tid < 32u) {
        float v = sq[tid] + sq[tid + 32u];
        #pragma unroll
        for (unsigned s = 16u; s > 0u; s >>= 1) {
            const float o = __shfl_down(v, s, 32);
            if (tid < s) v = v + o;
        }
        if (tid == 0) sq[0] = v;
    }
    g1r_lds_barrier();
    G1_TS(9);
    if (tid < G1_BLOCK) {
        const float mean_sq = sq[0] / (float)G1_NE;
        const float nscale = rsqrtf(mean_sq + a.rms_eps);
        #pragma unroll
        for (uint32_t j = 0; j < G1_PE; j++) {
            const float w = (NW > 8u) ? snw[tid + G1_BLOCK * j] : nw[j];
            a.norm_out[(size_t)z * G1_NE + tid + G1_BLOCK * j] = cv[j] * nscale * w;
        }
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        __hip_atomic_store(a.sync + S_READY(z), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    G1_TS(2);
    const bool want_xq = (a.xq != nullptr) || (a.pack != nullptr && a.pk_xq != G1_NONE);
    if (want_xq && !(a.dbg & 2u)) {
        const unsigned lane = tid & (G1_WARP - 1u), warp = tid >> 5u;
        #pragma nounroll
        for (unsigned j = warp; j < G1_PE; j += NW) g1_q8k_block(a, z, j, lane);
    }
    G1_TS(3);
}

// ---- family "r" selection: the same winners as g1_pass (wfred's key: score order, then the lower
// expert index), with a shorter dependent chain per pass: the lane's max by a max tree, its first
// index by a bit-mask + find-first, the wave max by DPP, and the lower-index tie resolution only when
// two lanes hold the max (ballot fast path). Winners stay in registers (lane k = pass k).
__device__ __forceinline__ void g1r_sel_task(const G1Args& a, unsigned mat, unsigned z, unsigned kind, unsigned lane, G1Sel& sl,
                                             float* sprob, unsigned long long* ts) {
    const float* pr = a.scr + G1_SCR_PROB(mat, z);
    const float* scv = a.scr + G1_SCR_SCORE(mat, z);
    float sc[G1_EPL], pv[G1_EPL], pp[G1_EPL];
    #pragma unroll
    for (unsigned i = 0; i < G1_EPL; ++i) sc[i] = scv[lane + G1_WARP * i];
    #pragma unroll
    for (unsigned i = 0; i < G1_EPL; ++i) pp[i] = pr[lane + G1_WARP * i];
    if (kind == 1u) {
        #pragma unroll
        for (unsigned i = 0; i < G1_EPL; ++i) pv[i] = a.prior_dev[lane + G1_WARP * i];
    }
    unsigned h[G1_EPL];
    #pragma unroll
    for (unsigned i = 0; i < G1_EPL; ++i) h[i] = g1_ord(kind == 1u ? (sc[i] + pv[i]) : sc[i]);
    #pragma unroll
    for (unsigned i = 0; i < G1_EPL; ++i) sprob[lane + G1_WARP * i] = pp[i];   // this task's probs (own wave only)
    if (ts != nullptr) {   // bench trace: the scores are in (the loads' latency is behind us)
        unsigned x = 0u;
        #pragma unroll
        for (unsigned i = 0; i < G1_EPL; ++i) x ^= h[i] ^ __float_as_uint(pp[i]);
        asm volatile("" :: "v"(x));
        ts[13] = g1_now();
    }
    int my_e = -1;
    float my_p = 0.0f;
    const unsigned kn = a.n_used;
    // ---- THRESHOLD + RANK (exact, the same winners in the same order as the serial passes):
    //   T = the kn-th largest of the 32 lane maxima: at least kn elements are >= T, so every one of the
    //   top kn is >= T. The candidates (>= T; typically kn..12 of them) rank themselves against each
    //   other under wfred's key (higher score, then the lower expert index); rank r -> pick r.
    // Falls back to the serial passes when T <= ord(-INF) (the masked-to--INF re-pick semantics of the
    // passes) or more than 32 candidates tie at or above T.
    unsigned m01 = max(h[0], h[1]), m23 = max(h[2], h[3]), m45 = max(h[4], h[5]);
    unsigned m67 = max(h[6], h[7]), m89 = max(h[8], h[9]), mab = max(h[10], h[11]);
    unsigned mv = max(max(max(m01, m23), max(m45, m67)), max(m89, mab));
    unsigned T = 0u;
    #pragma nounroll
    for (unsigned p = 0; p < kn; ++p) {
        T = g1_wmax_u32(mv);
        const unsigned bl = (unsigned)__builtin_amdgcn_ballot_w32(mv == T);
        if (lane == (unsigned)__builtin_ctz(bl)) mv = 0u;
    }
    unsigned cm = 0u;
    #pragma unroll
    for (unsigned i = 0; i < G1_EPL; ++i) cm |= (h[i] >= T ? 1u : 0u) << i;
    const unsigned cnt = (unsigned)__builtin_popcount(cm);
    // exclusive prefix of cnt over the lanes (cnt <= 12: 4 bit planes x mbcnt) and the total
    unsigned off = 0u, C = 0u;
    #pragma unroll
    for (unsigned bit = 0; bit < 4u; ++bit) {
        const unsigned bb = (unsigned)__builtin_amdgcn_ballot_w32((cnt >> bit) & 1u);
        off += __builtin_amdgcn_mbcnt_lo(bb, 0u) << bit;
        C += (unsigned)__builtin_popcount(bb) << bit;
    }
    unsigned* ck = reinterpret_cast<unsigned*>(sprob + G1_NEXP);   // [32] keys, [32] indices, [8] picks
    unsigned* ci = ck + 32;
    unsigned* cw = ck + 64;
    if (C <= 32u && T > 0x007fffffu) {
        #pragma unroll
        for (unsigned i = 0; i < G1_EPL; ++i) {
            if ((cm >> i) & 1u) {
                ck[off] = h[i];
                ci[off] = lane + G1_WARP * i;
                ++off;
            }
        }
        const unsigned kl = lane < C ? ck[lane] : 0u;
        const unsigned il = lane < C ? ci[lane] : 0xffffffffu;
        unsigned rank = 0u;
        #pragma nounroll
        for (unsigned q = 0; q < C; ++q) {
            const unsigned kq = (unsigned)__builtin_amdgcn_readlane((int)kl, (int)q);
            const unsigned iq = (unsigned)__builtin_amdgcn_readlane((int)il, (int)q);
            rank += (kq > kl || (kq == kl && iq < il)) ? 1u : 0u;
        }
        if (lane < C && rank < kn) cw[rank] = il;
        if (lane < kn) my_e = (int)cw[lane];
    } else {
        #pragma nounroll
        for (unsigned k = 0; k < kn; ++k) {
            unsigned n01 = max(h[0], h[1]), n23 = max(h[2], h[3]), n45 = max(h[4], h[5]);
            unsigned n67 = max(h[6], h[7]), n89 = max(h[8], h[9]), nab = max(h[10], h[11]);
            const unsigned m = max(max(max(n01, n23), max(n45, n67)), max(n89, nab));
            unsigned bits = 0u;
            #pragma unroll
            for (unsigned i = 0; i < G1_EPL; ++i) bits |= (h[i] == m ? 1u : 0u) << i;
            const unsigned ib = (unsigned)__builtin_ctz(bits | (1u << 31));   // the lane's first maximum
            const unsigned mh = g1_wmax_u32(m);
            const unsigned bal = (unsigned)__builtin_amdgcn_ballot_w32(m == mh);
            unsigned wl, wi;
            if (__builtin_popcount(bal) == 1) {
                wl = (unsigned)__builtin_ctz(bal);
                wi = (unsigned)__builtin_amdgcn_readlane((int)ib, (int)wl);
            } else {
                const unsigned win = g1_wmin_u32(m == mh ? lane + G1_WARP * ib : 0xffffffffu);
                wl = win & 31u;
                wi = win >> 5;
            }
            if (lane == k) my_e = (int)(wl + G1_WARP * wi);
            if (lane == wl) {
                #pragma unroll
                for (unsigned i = 0; i < G1_EPL; ++i)
                    if (i == wi) h[i] = 0x007fffffu;   // ord(-INF): still a candidate, as wfred's masked score
            }
        }
    }
    if (lane < a.n_used) {
        my_p = sprob[my_e];   // the winner's prob (written by this wave above: in-order LDS)
        sl.e[z][kind][lane] = my_e;
        sl.p[z][kind][lane] = my_p;
    }
    if (kind == 0u && mat == 0 && a.range_out != nullptr) {
        float mx = -INFINITY, mn = INFINITY;
        #pragma unroll
        for (unsigned i = 0; i < G1_EPL; ++i) { mx = fmaxf(mx, sc[i]); mn = fminf(mn, sc[i]); }
        const float range = g1_wmax_f32(mx) - g1_wmin_f32(mn);
        if (lane == 0) {
            a.range_out[z] = range;
            if (a.pack != nullptr && a.pk_range != G1_NONE) a.pack[a.pk_range + z] = __float_as_uint(range);
        }
    }
}

// ---- family "r" emit: g1_emit_row's outputs; the prior compose is lane-parallel (each lane holds
// one pick of each list; the protect test is a ballot) instead of LDS reads in a nested loop.
__device__ __forceinline__ void g1r_emit_row(const G1Args& a, unsigned mat, unsigned z, unsigned lane, const G1Sel& sl) {
    const bool prior = (mat == 0) && (a.prior_host != nullptr);
    const unsigned n_used = a.n_used;
    const unsigned n_protect = a.n_protect < n_used ? a.n_protect : n_used;
    int e0 = -1, e1 = -1;
    float p0 = 0.0f, p1 = 0.0f;
    if (lane < n_used) {
        e0 = sl.e[z][0][lane];
        p0 = sl.p[z][0][lane];
        if (prior) { e1 = sl.e[z][1][lane]; p1 = sl.p[z][1][lane]; }
    }
    int my_orig = e0, my_sel = e0;
    float p_orig = p0, p_sel = p0;
    if (prior) {
        if (lane >= n_protect) { my_sel = -1; p_sel = 0.0f; }
        unsigned cnt = n_protect;
        #pragma nounroll
        for (unsigned t = 0; t < n_used; ++t) {
            const int e = __builtin_amdgcn_readlane(e1, (int)t);
            const bool prot = __builtin_amdgcn_ballot_w32(lane < n_protect && e0 == e) != 0u;
            if (!prot && cnt < n_used) {
                const float pt = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(p1), (int)t));
                if (lane == cnt) { my_sel = e; p_sel = pt; }
                ++cnt;
            }
        }
    }
    const int plain = my_orig;   // the router's own top-n_used (what M62 counts)
    if (prior && a.prior_dry != 0u) {
        const int t = my_sel; my_sel = my_orig; my_orig = t;
        const float tp = p_sel; p_sel = p_orig; p_orig = tp;
    }
    const float pj = (lane < n_used && my_sel >= 0) ? p_sel : 0.0f;
    float sum = 0.0f;
    #pragma nounroll
    for (unsigned j = 0; j < n_used; ++j) sum += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(pj), j));
    if (sum < a.weight_eps) sum = a.weight_eps;
    if (lane < n_used) {
        const unsigned o = z * n_used + lane;
        const float wj = pj / sum * a.scale;
        if (mat == 0) {
            a.sel[o] = my_sel;
            a.ew[o] = wj;
            if (a.orig_sel != nullptr) a.orig_sel[o] = my_orig;
            if (a.pack != nullptr) {
                if (a.pk_sel != G1_NONE) a.pack[a.pk_sel + o] = (unsigned)my_sel;
                if (a.pk_ew != G1_NONE) a.pack[a.pk_ew + o] = __float_as_uint(wj);
                if (a.pk_orig != G1_NONE) a.pack[a.pk_orig + o] = (unsigned)my_orig;
            }
            if (a.sel_bank != nullptr && plain >= 0 && (unsigned)plain < G1_NEXP)
                atomicAdd(&a.sel_bank[a.bank_off + (unsigned)plain], 1u);
        } else {
            a.look_sel[o] = my_sel;
            a.look_ew[o] = wj;
            if (a.pack != nullptr) {
                if (a.pk_look != G1_NONE) a.pack[a.pk_look + o] = (unsigned)my_sel;
                if (a.pk_look_ew != G1_NONE) a.pack[a.pk_look_ew + o] = __float_as_uint(wj);
            }
        }
    }
}

// Tail WG `ti` (NWV waves): (mat, rows z0..) selection tasks, one wave each, then the emits.
template <unsigned NWV, unsigned NRW>
__device__ __forceinline__ void g1r_tail(const G1Args& a, unsigned ti, unsigned tid, G1Sel& sl, unsigned long long* ts) {
    const unsigned b = a.b;
    const bool prior = a.prior_host != nullptr;
    const unsigned k0 = prior ? 2u : 1u;
    const unsigned rpw0 = NWV / k0;
    const unsigned nt0 = (b + rpw0 - 1u) / rpw0;
    unsigned mat, z0, kinds, rpw;
    if (ti < nt0) { mat = 0; z0 = ti * rpw0; kinds = k0; rpw = rpw0; }
    else { mat = 1; z0 = (ti - nt0) * NWV; kinds = 1; rpw = NWV; }
    const unsigned nz = min(rpw, b - z0);
    const unsigned lane = tid & (G1_WARP - 1u), warp = tid >> 5u;
    if (warp == 0) {
        g1_spoll_ge<0>(a.sync + S_RDONE(mat), NRW, a.sync + S_ERR);
        if (mat == 0 && prior) g1_spoll_ge(a.sync + S_PRIOR, 1u, a.sync + S_ERR);
    }
    __syncthreads();
    g1_vinv();
    G1_TS(5);
    if (!(a.dbg & 4u)) {
        if (warp < nz * kinds)
            g1r_sel_task(a, mat, z0 + warp / kinds, warp % kinds, lane, sl, reinterpret_cast<float*>(&sl + 1) + warp * (G1_NEXP + 80u), ts);
        __syncthreads();
        G1_TS(10);
        if (warp < nz) g1r_emit_row(a, mat, z0 + warp, lane, sl);
        G1_TS(11);
    }
    G1_TS(6);
}

template <unsigned NW>
__device__ __forceinline__ unsigned g1r_number_of_tails(unsigned b, unsigned n_mat, bool prior) {
    const unsigned rpw0 = prior ? NW / 2u : NW;
    return (b + rpw0 - 1u) / rpw0 + (n_mat == 2u ? (b + NW - 1u) / NW : 0u);
}

template <unsigned ZB, unsigned CPP, unsigned P0, unsigned KW, unsigned RX, unsigned KP, unsigned NW>
__device__ __forceinline__ void g1r_body(const G1Args& a) {
    static_assert(NW >= 8u && G1_NEXP % NW == 0u, "g1r: NW waves per WG, a divisor of 384");
    constexpr unsigned NRW = G1_NEXP / NW;                         // router WGs per matrix
    __shared__ unsigned s_ticket;
    constexpr unsigned SXF = 2u * ZB * 64u * CPP;                  // router x slabs
    constexpr unsigned CLF = G1_BLOCK + (NW > 8u ? G1_NE : 0u);    // collapse: sq (+ norm weight)
    constexpr unsigned SB0 = SXF > CLF ? SXF : CLF;
    constexpr unsigned TLF = sizeof(G1Sel) / 4u + NW * (G1_NEXP + 80u);   // tail: G1Sel + per wave: probs + candidates
    constexpr unsigned SBUF = SB0 > TLF ? SB0 : TLF;
    __shared__ __attribute__((aligned(16))) float s_buf[SBUF];
    static_assert(sizeof(G1Sel) % 16u == 0u, "g1r tail LDS");
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & (G1_WARP - 1u), warp = tid >> 5u;
    const unsigned long long t_start = g1_now();
    if (tid == 0) s_ticket = atomicAdd(a.sync + S_TICKET, 1u);
    __syncthreads();
    const unsigned t = s_ticket;
    unsigned long long* ts = (a.ts != nullptr && tid == 0) ? a.ts + (size_t)t * 16u : nullptr;
    G1_TS(1);
    if (ts != nullptr) {
        ts[0] = t_start;
#if defined(__gfx1200__) || defined(__gfx1201__)
        unsigned hw;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
        ts[14] = hw;   // bench trace: where this WG runs (WGP / SA / SE)
#endif
    }
    const unsigned b = a.b;
    const bool prior = a.prior_host != nullptr;
    const unsigned np = 1u;                                         // the prior / L2-warm WG (always)
    const unsigned nr = a.n_mat * NRW;
    const unsigned total = b + np + nr + g1r_number_of_tails<NW>(b, a.n_mat, prior);

    if (t < b) {
        g1r_collapse<NW>(a, t, tid, s_buf, s_buf + G1_BLOCK, ts);
    } else if (t < b + np) {
        // the norm weight into L2 first (the collapse reads it last when NW = 8), then the prior
        if (tid < G1_BLOCK) {
            float s = 0.0f;
            #pragma unroll
            for (unsigned j = 0; j < G1_PE; ++j) s += a.norm_w[tid + G1_BLOCK * j];
            if (s == 1234.5f && a.dbg == 0xfffffffeu) a.sync[S_TICKET + 1u] = 1u;   // never: keeps the loads
        }
        if (prior) {
            for (unsigned i = tid; i < G1_NEXP; i += 32u * NW) a.prior_dev[i] = a.prior_host[i];
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                __hip_atomic_store(a.sync + S_PRIOR, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        G1_TS(3);
    } else if (t < b + np + nr) {
        const unsigned r = t - b - np;
        const unsigned mat = r / NRW;
        g1r_router<ZB, CPP, P0, KW, RX, KP, NW>(a, mat, (r - mat * NRW) * NW + warp, lane, warp, s_buf, ts, []() { g1r_lds_barrier(); });
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            atomicAdd(a.sync + S_RDONE(mat), 1u);
        }
    } else {
        g1r_tail<NW, NRW>(a, t - b - np - nr, tid, *reinterpret_cast<G1Sel*>(s_buf), ts);
    }

    // ---- finish: the last WG resets the scratch (not S_ERR) and posts the completion word.
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned prev = atomicAdd(a.sync + S_FINISH, 1u);
        G1_TS(7);
        if (prev == total - 1u) {
            __threadfence();
            a.sync[S_TICKET] = 0u;
            a.sync[S_RDONE(0)] = 0u;
            a.sync[S_RDONE(1)] = 0u;
            a.sync[S_PRIOR] = 0u;
            for (unsigned z = 0; z < G1_MAXB; ++z) a.sync[S_READY(z)] = 0u;
            a.sync[S_FINISH] = 0u;
            if (a.done_word != nullptr) {
                __threadfence_system();
                __hip_atomic_store(a.done_word, a.done_value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
    }
}

// g1_router_block_r{ZB}_{CPP}_{P0}_{KW}_{RX}_p{KP}_w{NW}: ZB batch rows (b <= ZB), CPP u32 words per
// phase, P0 words prefetched (paced, KP in flight), a ring KW words ahead (0: continuous issue), x
// staged RX phases ahead, NW router waves (rows) per WG. The set G1_VARIANTS lists; chosen per b by
// g1_variant_for (cold group wall, tests/bench_g1_router_block.rs, 10-13).
#define G1R_KERNEL(ZB, CPP, P0, KW, RX, KP, NW)                                                                    \
extern "C" __global__ __launch_bounds__(32 * NW) void                                                              \
g1_router_block_r##ZB##_##CPP##_##P0##_##KW##_##RX##_p##KP##_w##NW(G1Args a) {                                    \
    g1r_body<ZB, CPP, P0, KW, RX, KP, NW>(a);                                                                      \
}
G1R_KERNEL(1, 8, 24, 16, 2, 8, 8)
G1R_KERNEL(2, 8, 24, 16, 2, 8, 8)
G1R_KERNEL(3, 8, 24, 16, 2, 8, 8)
G1R_KERNEL(4, 8, 24, 16, 2, 8, 8)
G1R_KERNEL(4, 8, 32, 16, 2, 8, 8)
G1R_KERNEL(4, 8, 40, 0, 3, 8, 8)
G1R_KERNEL(8, 4, 24, 16, 4, 8, 8)
G1R_KERNEL(16, 2, 24, 16, 8, 8, 8)

// Symbols g1_router_block_z{ZB}_c{CH}: ZB = batch rows per router wave (>= b), CH = weight
// elements per lane per chunk (bounds the x double-buffer: ZB * CH * 2 registers).
#define G1_KERNEL(ZB, CH)                                                                       \
extern "C" __global__ __launch_bounds__(256) void g1_router_block_z##ZB##_c##CH(G1Args a) {     \
    g1_body<ZB, CH, 0>(a);                                                                      \
}
// Phased-LDS forms: g1_router_block_p{ZB}_{PHE}_{PFE}.
#define G1_KERNEL_PH(ZB, PHE, PFE)                                                              \
extern "C" __global__ __launch_bounds__(256) void g1_router_block_p##ZB##_##PHE##_##PFE(G1Args a) { \
    g1_body<ZB, PFE, PHE>(a);                                                                   \
}
G1_KERNEL_PH(2, 32, 64)
G1_KERNEL_PH(4, 40, 80)
G1_KERNEL_PH(8, 20, 80)
G1_KERNEL_PH(8, 20, 40)
G1_KERNEL_PH(16, 10, 40)
G1_KERNEL(1, 20)
G1_KERNEL(2, 10)
G1_KERNEL(4, 8)
G1_KERNEL(8, 4)
G1_KERNEL(16, 4)

