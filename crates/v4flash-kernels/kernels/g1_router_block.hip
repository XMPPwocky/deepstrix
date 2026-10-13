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
// Router wave, FULL-PREFETCH form (ZB >= b rows, x staged per K-slab of 32 * PHE in LDS): all 160
// weight elements of the lane are requested at entry, packed two per VGPR by global_load_d16_b16 /
// global_load_d16_hi_b16 (inline asm: the compiler loads each half into its own VGPR, 170 VGPRs),
// so the router's whole DRAM stream starts with the kernel and never waits on the collapse or on
// this wave's progress. The loads are opaque to the compiler: ONE s_wait_loadcnt 0 before the first
// FMA (after the row wait) covers them. Accumulation: h20's per-lane sequence, as the other forms.
// --------------------------------------------------------------------------------------------
__device__ __forceinline__ void g1_ld_lo(unsigned& r, const __half* p) {
    asm volatile("global_load_d16_b16 %0, %1, off" : "+v"(r) : "v"(p) : "memory");
}
__device__ __forceinline__ void g1_ld_hi(unsigned& r, const __half* p) {
    asm volatile("global_load_d16_hi_b16 %0, %1, off" : "+v"(r) : "v"(p) : "memory");
}
__device__ __forceinline__ void g1_wait_vm0() {
#if defined(__gfx1200__) || defined(__gfx1201__)
    asm volatile("s_wait_loadcnt 0x0" ::: "memory");
#else
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
__device__ __forceinline__ float g1_h_lo(unsigned r) { return __half2float(__ushort_as_half((unsigned short)(r & 0xffffu))); }
__device__ __forceinline__ float g1_h_hi(unsigned r) { return __half2float(__ushort_as_half((unsigned short)(r >> 16))); }

template <unsigned ZB, unsigned PHE>
__device__ __forceinline__ void g1_router_wave_full(const G1Args& a, unsigned mat, unsigned row, unsigned lane,
                                                    unsigned tid, float* sx, unsigned long long* ts) {
    constexpr unsigned NPH = G1_EPW / PHE;
    constexpr unsigned SLAB = G1_WARP * PHE;
    constexpr unsigned ST4 = ZB * SLAB / 4u / G1_BLOCK;
    static_assert(G1_EPW % PHE == 0 && PHE % 2 == 0 && (ZB * SLAB) % (4u * G1_BLOCK) == 0, "g1 phase shape");
    const unsigned b = a.b;
    const __half* wr = (mat == 0 ? a.w0 : a.w1) + (size_t)row * G1_NE + lane;
    unsigned w2[G1_EPW / 2];   // w2[k] = { w[lane + 32 (2k)], w[lane + 32 (2k + 1)] }
    #pragma unroll
    for (unsigned k = 0; k < G1_EPW / 2; ++k) {
        w2[k] = 0u;
        g1_ld_lo(w2[k], wr + 64u * k);
        g1_ld_hi(w2[k], wr + 64u * k + 32u);
    }
    G1_TS(2);
    g1_wg_wait(a.sync, 0, (a.dbg & 8u) ? 0u : b, false);
    G1_TS(3);
    const float4* xg = reinterpret_cast<const float4*>(a.norm_out);
    float4* sx4 = reinterpret_cast<float4*>(sx);
    auto stage_load = [&](unsigned ph, float4 (&st)[ST4]) {
        #pragma unroll
        for (unsigned i = 0; i < ST4; ++i) {
            const unsigned f = tid + G1_BLOCK * i;
            const unsigned zz = f / (SLAB / 4u), c = f % (SLAB / 4u);
            st[i] = (zz < b) ? xg[(size_t)zz * (G1_NE / 4u) + ph * (SLAB / 4u) + c] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    float4 st[ST4];
    stage_load(0, st);
    g1_wait_vm0();   // the weights (and phase 0's x)
    float acc[ZB];
    #pragma unroll
    for (unsigned z = 0; z < ZB; ++z) acc[z] = 0.0f;
    #pragma unroll
    for (unsigned ph = 0; ph < NPH; ++ph) {
        #pragma unroll
        for (unsigned i = 0; i < ST4; ++i) sx4[tid + G1_BLOCK * i] = st[i];
        __syncthreads();
        if (ph + 1 < NPH) stage_load(ph + 1, st);
        #pragma unroll
        for (unsigned z = 0; z < ZB; ++z) {
            if (z < b) {
                #pragma unroll
                for (unsigned jj = 0; jj < PHE; ++jj) {
                    const unsigned j = ph * PHE + jj;
                    const float w = (j & 1u) ? g1_h_hi(w2[j >> 1]) : g1_h_lo(w2[j >> 1]);
                    acc[z] += w * sx[z * SLAB + lane + G1_WARP * jj];
                }
            }
        }
        __syncthreads();
    }
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
__device__ void g1_prior(const G1Args& a, unsigned tid) {
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

__device__ void g1_collapse(const G1Args& a, unsigned z, unsigned tid, float* sq, unsigned long long* ts) {
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
        if constexpr (PHE != 0 && CH == G1_EPW) g1_router_wave_full<ZB, (PHE ? PHE : 2)>(a, mat, row, lane, tid, s_buf, ts);
        else if constexpr (PHE != 0) g1_router_wave_ph<ZB, (PHE ? PHE : 1), CH>(a, mat, row, lane, tid, s_buf, ts);
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
G1_KERNEL_PH(1, 32, 64)
G1_KERNEL_PH(2, 32, 64)
G1_KERNEL_PH(4, 40, 80)
G1_KERNEL_PH(8, 20, 80)
G1_KERNEL_PH(8, 20, 40)
G1_KERNEL_PH(16, 10, 80)
G1_KERNEL_PH(16, 10, 40)
// Full-prefetch forms (PFE = 160): g1_router_block_p{ZB}_{PHE}_160.
G1_KERNEL_PH(1, 32, 160)
G1_KERNEL_PH(2, 32, 160)
G1_KERNEL_PH(4, 40, 160)
G1_KERNEL_PH(8, 20, 160)
G1_KERNEL_PH(8, 40, 160)
G1_KERNEL_PH(16, 10, 160)
G1_KERNEL_PH(16, 20, 160)
G1_KERNEL(1, 20)
G1_KERNEL(2, 20)
G1_KERNEL(2, 10)
G1_KERNEL(4, 10)
G1_KERNEL(4, 8)
G1_KERNEL(8, 5)
G1_KERNEL(8, 4)
G1_KERNEL(16, 4)

