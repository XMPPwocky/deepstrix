// attn_dec_g3o -- "G3-optimal" (round 2 of fusion group G3): the decode attention BLOCK
// (attn_meta_fill + indexer_gather + attention_dec_fused + rope_inv_quant_q8) as ONE launch, every
// output BIT-IDENTICAL to that chain and to attn_dec_g3 (tests/attn_g3o.rs: 10080 comparisons, b =
// 1..16, two lanes sharing one stream, 3 rope sets, 5 key layouts, steady / ragged / edge rows,
// subnormal and signed-zero KV, with and without heads_out; a negative control).
//
// PRODUCTION KERNELS (`V41_ATTN_G3_KERNEL`, attn_dec_g3o.rs `auto_mode`): the "W" family at the end of
// this file -- one work-group per (row, 16-head tile, 1/DPW of the dims), no inter-WG traffic.
// Compared with attn_dec_g3 (one WG per row and head tile) it pipelines every load (the selection
// first, then q, then the first K tile; the next K tile's fragment u loaded into the register the
// chain just consumed; the W.V transposed-load ring started before the softmax; runtime loops with
// branch-free bodies -- per-tile branches made the waitcnt pass serialize every load), runs a
// branch-free phase A (attn_dec_g3's arithmetic: blocks of 32 keys past n_total are skipped
// uniformly, a lane past n_total reads -INF so w = +0 adds exactly nothing, fmaxf for the max), and
// computes each Q8_0 block's scale once per lane instead of 8 serial divisions. DPW = 2 / 4 replicate
// the scores + softmax per dim part to put 2x / 4x more WGPs to work at small b.
//
// BENCH-ONLY (kept to reproduce what was measured; never reachable from the production wiring): the
// verifier's suggested split -- scores spread over the whole GPU, softmax + W.V per (row, head tile,
// dim part) -- as two launches (attn_g3o_s / attn_g3o_v) and as one ticketed launch (attn_g3o_t).
// Bit-identical too, but slower than attn_dec_g3 from b = 3 on (it moves as many bytes through the
// WGPs: every K / V tile is still read once per head tile, plus the score and V^T scratch round trips
// and the 16-head softmax replicated per V work-group); see docs/v41/DECODE_OPT_KNOBS.md.
//   S (scores)  WG = (row, HGP head tiles, chunk of NWS key tiles), NWS warps; warp = one key tile, HGP
//               chains over the same K fragments: per (head, key) the same 32-step WMMA chain, kq_scale
//               multiply and f16 cast, stored to a score scratch (unpadded key positions); then (G3O_VT)
//               the tile's V^T fragments: kf[u] is already the A fragment of V[16 keys][16 dims], and
//               D = V x I (WMMA against the identity: one exact x * 1 product per output, every other
//               term +-0) is the W.V B fragment (global_load_tr_b128 was ~4x slower per byte than the
//               fragment pattern: tests/attn_g3o.rs g3o_load_probe).
//   V (W.V)     WG = (row, head tile, 1/DP of the dims), NWV = 32/DP warps (one 16-dim tile each): the
//               16 heads' softmax (replicated per V WG), the weights into LDS, the 40-step chains, the
//               epilogue (1 / sum, rope_tail's inverse rotation on dims 448..511, Q8_0 per 32-dim block).
//   attn_g3o_t: ticketed roles -- every WG draws a ticket; tickets [0, nS) are S work, the rest V
//               work. A V WG waits only on arrival counts of LOWER tickets (resident WGs that never
//               wait): forward progress without assumptions on dispatch order or co-residency; bounded
//               polls (an error word, not a hang); the last WG resets the scratch.
//
// Sentinel rows (selection < 0) inside n_comp read a ZERO key (as attn_dec_g3; the chain reads stale
// gather bytes there -- undefined, never produced by consistent decode state).
// HARD LIMITS (the launcher enforces them): b <= 16, n_comp <= 512, n_raw + n_comp <= 640 per row,
// n_head 64, head_dim 512, n_rot 64.
//
// Variant macros (bench iteration through code objects; the defaults are the shipped kernels):
//   split: G3O_HGP head tiles per S WG (1, 2), G3O_NWS warps per S WG, G3O_DP dim parts (2, 4),
//          G3O_VT, G3O_VPF; W: G3O_WPD (W.V ring depth), G3O_WMAP (dispatch order), G3O_WPF (K
//          prefetch), G3O_WSTG (staged Q8 stores); timing only: G3O_TSW, G3O_EMPTY, G3O_WABL.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <type_traits>

#include "act_quant_fp8.inc"

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));
typedef short    short8 __attribute__((ext_vector_type(8)));

#define HD            512u
#define HT            16u
#define NHEAD         64u
#define G3_MAX_KEYS   640u
#define G3_MAX_ROWS   16u
#define MAX_PER_LANE  (G3_MAX_KEYS / 32u)
#define WSTRIDE       664u               // weights row stride (f16): >= 41 tiles * 16, 16-B multiple
#define QSTRIDE       520u               // q_lds row stride (f16): 16-B skew per head row
#define TBL_N         672u
#define MAXT          41u                // most key tiles of a row: ceil(n_raw/16) + ceil(n_comp/16)
#define SC_STRIDE     640u               // score scratch row stride (f16, unpadded key positions)

#ifndef G3O_HGP
#define G3O_HGP 2
#endif
#ifndef G3O_NWS
#define G3O_NWS 16
#endif
#ifndef G3O_DP
#define G3O_DP 2
#endif
#ifndef G3O_VT
#define G3O_VT 1                         // 1: the S warps write V^T fragments, the V WGs read them
#endif
#ifndef G3O_VPF
#define G3O_VPF 41                       // V tiles transposed into registers before the wait
#endif
#ifndef G3O_VPD
#define G3O_VPD 8                        // the rest: raw loads issued this many tiles ahead
#endif

#define STHREADS      (G3O_NWS * 32u)
#define KTP           G3O_NWS                          // key tiles per S WG (one per warp)
#define NCH           ((MAXT + KTP - 1u) / KTP)         // S chunks per (row, head-tile group)
#define NHG_S         (4u / G3O_HGP)                   // head-tile groups per row
#define NWV           (32u / G3O_DP)                   // warps (dim tiles) per V WG
#define VTHREADS      (NWV * 32u)
static_assert(HT % G3O_NWS == 0 || G3O_NWS % HT == 0, "softmax heads per S warp");

// sync scratch (u32 words, each polled / counted word on its own 128-byte line)
#define S_LINE        32u
#define S_TICKET      0u
#define S_FINISH      (1u * S_LINE)
#define S_ERR         (2u * S_LINE)
#define S_CNT(i)      ((3u + (i)) * S_LINE)                          // i = row * 4 + head tile
#define S_VCNT(i)     ((3u + G3_MAX_ROWS * 4u + (i)) * S_LINE)            // V^T arrivals
#define S_WORDS       (S_VCNT(G3_MAX_ROWS * 4u))
#define G3O_SPIN_CAP  200000000ull                                   // 2 s at 100 MHz

struct G3Meta {
    int n_raw[G3_MAX_ROWS];
    int raw_off[G3_MAX_ROWS];
    int n_comp[G3_MAX_ROWS];
};

struct G3OArgs {
    float* heads_out;              // nullable [B, 64, 512]
    int8_t* xq;                    // [B, 32768]
    float* xscale;                 // [B, 1024]
    const float* sinks;            // [64]
    const float* q;                // [B, 64, 512]
    const _Float16* raw_kv;
    const _Float16* comp_kv;       // nullable
    const int* sel;                // nullable
    const int* comp_base_per;      // nullable
    const int* pos_per_b;
    const _Float16* zero_row;
    _Float16* sc;                  // [B, 64, SC_STRIDE] scores
    _Float16* vt;                  // [B, MAXT, 32, 256] V^T fragments (G3O_VT)
    unsigned* sync;                // [S_WORDS + 32] (ticketed entry)
    unsigned long long* ts;        // nullable: debug timestamps [WG][8]
    unsigned sel_stride;
    unsigned comp_kv_batch_stride;
    unsigned b;
    float theta_scale, freq_scale, ext_factor, mscale_eff, corr_low, corr_high, kq_scale;
    G3Meta meta;
};

#if defined(__gfx1200__) || defined(__gfx1201__)

__device__ __forceinline__ unsigned long long g3o_now() { return __builtin_readsteadycounter(); }
#define G3O_TS(slot, k) do { if (a.ts != nullptr && threadIdx.x == 0) a.ts[(size_t)(slot) * 16u + (k)] = g3o_now(); } while (0)
#ifdef G3O_TSW
#define G3O_TSL(slot, k) do { if (a.ts != nullptr) { asm volatile("s_wait_loadcnt 0x0" ::: "memory"); G3O_TS(slot, k); } } while (0)
#else
#define G3O_TSL(slot, k) G3O_TS(slot, k)
#endif

// Lane exchange "value of lane (lane ^ M)" by DPP row_xmask / v_permlanex16 (the pairing of
// __shfl_xor: every tree below is attn_dec_g3's, which is the production kernel's).
template <int M>
__device__ __forceinline__ float g3_xor(float v) {
    const int x = __float_as_int(v);
    int r;
    if constexpr (M == 16) r = __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false);
    else r = __builtin_amdgcn_update_dpp(x, x, 0x160 + M, 0xf, 0xf, false);
    return __int_as_float(r);
}

// A barrier that waits for this wave's LDS operations only (a __syncthreads also waits for every
// outstanding load in WGP mode, e.g. the prefetched V fragments).
__device__ __forceinline__ void g3o_lds_barrier() {
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

// Uniform scalar load at device scope (scalar cache invalidated first: a scoped s_load alone can
// return a stale line -- g1_router_block.hip).
__device__ __forceinline__ unsigned g3o_sload_dev(const unsigned* pv) {
    const unsigned long long pa = (unsigned long long)pv;
    const unsigned lo = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)pa);
    const unsigned hi = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)(pa >> 32));
    const unsigned* p = (const unsigned*)(((unsigned long long)hi << 32) | lo);
    unsigned r;
    asm volatile("s_dcache_inv\n\ts_wait_kmcnt 0x0\n\ts_load_b32 %0, %1, 0x0 scope:SCOPE_DEV\n\ts_wait_kmcnt 0x0" : "=s"(r) : "s"(p) : "memory");
    return r;
}

// Invalidate this CU's vector caches WITHOUT waiting for outstanding loads.
__device__ __forceinline__ void g3o_vinv() {
    asm volatile("global_inv scope:SCOPE_DEV" ::: "memory");
}

// ---- rope_tail.hip's rf_theta / rf_rot_pair arithmetic (as attn_dec_g3) ------------------------
__device__ __forceinline__ float g3_theta(int pos, unsigned l, float theta_scale, float freq_scale,
                                          float ext_factor, float corr_low, float corr_high) {
    const float theta_extrap = (float)pos * powf(theta_scale, (float)l);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float y = ((float)l - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        const float ramp_mix = ramp * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    }
    return theta;
}
__device__ __forceinline__ float g3_rot(float v, float partner, unsigned e, float c, float s) {
    const float x0 = (e & 1u) ? partner : v;
    const float x1 = (e & 1u) ? v : partner;
    // rope_tail.hip compiles `x0 * c - x1 * s` / `x0 * s + x1 * c` to exactly these contractions
    return (e & 1u) ? __builtin_fmaf(x1, c, x0 * s) : __builtin_fmaf(x0, c, -(x1 * s));
}

struct RowMeta {
    unsigned n_raw, raw_off, n_comp, n_tr, n_tiles, base;
};
__device__ __forceinline__ RowMeta g3o_row(const G3OArgs& a, unsigned b) {
    RowMeta m;
    m.n_raw = (unsigned)a.meta.n_raw[b];
    m.raw_off = (unsigned)a.meta.raw_off[b];
    m.n_comp = a.comp_kv ? (unsigned)a.meta.n_comp[b] : 0u;
    m.n_tr = (m.n_raw + 15u) / 16u;
    m.n_tiles = m.n_tr + (m.n_comp + 15u) / 16u;
    m.base = a.comp_base_per ? (unsigned)a.comp_base_per[b] : 0u;   // loaded once
    return m;
}

// The 512-dim row of key k of tile t (attn_dec_g3's key table entry): raw window row, the selected /
// dense compressed row, or the zero row for an invalid key (past n_raw / n_comp, or a sentinel).
__device__ __forceinline__ const _Float16* g3o_key(const G3OArgs& a, unsigned b, const RowMeta& m,
                                                   unsigned t, unsigned k) {
    const _Float16* e = a.zero_row;
    if (t < m.n_tr) {
        const unsigned key = t * 16u + k;
        if (key < m.n_raw) e = a.raw_kv + (size_t)(m.raw_off + key) * HD;
    } else {
        const unsigned j = (t - m.n_tr) * 16u + k;
        if (j < m.n_comp) {
            const unsigned base = m.base;
            if (a.sel) {
                const int sv = a.sel[(size_t)b * a.sel_stride + j];
                if (sv >= 0) e = a.comp_kv + (size_t)(base + (unsigned)sv) * HD;
            } else {
                e = a.comp_kv + (size_t)(a.comp_base_per ? base + j : b * a.comp_kv_batch_stride + j) * HD;
            }
        }
    }
    return e;
}

// The same key row as g3o_key, split so the selection load carries no branch: g3o_sel_ld issues it
// unconditionally (an in-bounds clamped index; 0 when there is no selection), g3o_key_at is ALU only.
__device__ __forceinline__ int g3o_sel_ld(const G3OArgs& a, unsigned b, const RowMeta& m, unsigned t, unsigned k) {
    if (a.sel == nullptr) return 0;   // uniform
    const unsigned j = t >= m.n_tr ? (t - m.n_tr) * 16u + k : 0u;
    return a.sel[(size_t)b * a.sel_stride + min(j, a.sel_stride - 1u)];
}
__device__ __forceinline__ const _Float16* g3o_key_at(const G3OArgs& a, unsigned b, const RowMeta& m, unsigned t,
                                                      unsigned k, int sv) {
    const unsigned key = t * 16u + k;
    const unsigned j = (t - m.n_tr) * 16u + k;
    const bool raw = t < m.n_tr;
    const bool raw_ok = raw && key < m.n_raw;
    const bool comp_ok = !raw && j < m.n_comp && (a.sel == nullptr || sv >= 0);
    const unsigned base = m.base;
    const size_t crow = a.sel ? (size_t)(base + (unsigned)sv)
                              : (size_t)(a.comp_base_per ? base + j : b * a.comp_kv_batch_stride + j);
    return raw_ok ? a.raw_kv + (size_t)(m.raw_off + key) * HD : (comp_ok ? a.comp_kv + crow * HD : a.zero_row);
}

template <unsigned I, unsigned N, class F>
__device__ __forceinline__ void g3o_for(F&& f) {
    if constexpr (I < N) {
        f(std::integral_constant<unsigned, I>{});
        g3o_for<I + 1u, N>(f);
    }
}

// ---- the softmax of one head (attn_dec_g3's phase A; scores from the scratch, weights into LDS) ---
// Branch-free: rr >= n_total reads a score that is replaced by -INF, so w = expf(-INF) = +0 and
// local_sum + 0 = local_sum (local_sum >= +0): the sum, the max and every valid weight are those of the
// guarded loop. Its weight store lands at pos >= comp_end (a padding position: the same zero).
// NU = the number of 32-key blocks processed (n_total <= 32 NU), a compile-time constant so the body
// is one basic block the scheduler can interleave (per-block branches serialized the exp chains).
template <unsigned NU>
__device__ __forceinline__ void g3o_softmax_nu(float (&v)[MAX_PER_LANE], _Float16* wrow, float* sinv_h,
                                               unsigned lane, const RowMeta& m, float sink_h, unsigned pad_end) {
    const unsigned n_total = m.n_raw + m.n_comp;
    const unsigned n_raw = m.n_raw, n_tr = m.n_tr, n_comp = m.n_comp, n_tiles = m.n_tiles;
    const unsigned shift = n_tr * 16u - n_raw;      // comp key rr lands at rr + shift
    // a lane past n_total in the last blocks reads -INF (bit select, no compare chain): the max is
    // unchanged and its w = expf(-INF - max) = +0 adds exactly nothing to the sum
    #pragma unroll
    for (unsigned u = (NU > 4u ? NU - 4u : 0u); u < NU; u++) {
        const unsigned rr = lane + u * 32u;
        const unsigned inval = (unsigned)((int)(n_total - 1u - rr) >> 31);      // ~0 when rr >= n_total
        v[u] = __uint_as_float((__float_as_uint(v[u]) & ~inval) | (0xff800000u & inval));
    }
    // fmaxf: the same maximum as the `if (v > m) m = v` scan (no NaN; a +-0 tie only flips the sign
    // of a zero max, and v - (+-0) differs only for v = +-0, where expf gives 1 either way)
    float local_max = sink_h;
    #pragma unroll
    for (unsigned u = 0; u < NU; u++) local_max = fmaxf(local_max, v[u]);
    local_max = fmaxf(local_max, g3_xor<16>(local_max));
    local_max = fmaxf(local_max, g3_xor<8>(local_max));
    local_max = fmaxf(local_max, g3_xor<4>(local_max));
    local_max = fmaxf(local_max, g3_xor<2>(local_max));
    local_max = fmaxf(local_max, g3_xor<1>(local_max));
    const float max_score = local_max;
    float w[NU];
    #pragma unroll
    for (unsigned u = 0; u < NU; u++) w[u] = expf(v[u] - max_score);
    float local_sum = (lane == 0) ? expf(sink_h - max_score) : 0.0f;
    #pragma unroll
    for (unsigned u = 0; u < NU; u++) local_sum += w[u];
    #pragma unroll
    for (unsigned u = 0; u < NU; u++) {
        const unsigned rr = lane + u * 32u;
        const unsigned comp = (unsigned)((int)(n_raw - 1u - rr) >> 31);        // ~0 when rr >= n_raw
        wrow[rr + (shift & comp)] = (_Float16)w[u];                           // pos < WSTRIDE (rr < 640)
    }
    if (lane < shift) wrow[n_raw + lane] = (_Float16)0;
    const unsigned comp_end = n_tr * 16u + n_comp;
    for (unsigned i = comp_end + lane; i < pad_end; i += 32u) wrow[i] = (_Float16)0;   // comp pad (+ ring pad)
    (void)n_tiles;
    local_sum += g3_xor<16>(local_sum); local_sum += g3_xor<8>(local_sum); local_sum += g3_xor<4>(local_sum);
    local_sum += g3_xor<2>(local_sum);  local_sum += g3_xor<1>(local_sum);
    if (lane == 0) *sinv_h = 1.0f / local_sum;
}
// attn_dec_g3's phase A from the scores in v[] (blocks >= nu are never read)
__device__ __forceinline__ void g3o_softmax_regs(float (&v)[MAX_PER_LANE], _Float16* wrow, float* sinv_h,
                                                 unsigned lane, const RowMeta& m, float sink_h, unsigned pad_end = 0u) {
    const unsigned nu = (m.n_raw + m.n_comp + 31u) >> 5;
    const unsigned pe = pad_end ? pad_end : m.n_tiles * 16u;
    if (nu <= 4u) g3o_softmax_nu<4>(v, wrow, sinv_h, lane, m, sink_h, pe);
    else if (nu <= 8u) g3o_softmax_nu<8>(v, wrow, sinv_h, lane, m, sink_h, pe);
    else if (nu <= 12u) g3o_softmax_nu<12>(v, wrow, sinv_h, lane, m, sink_h, pe);
    else if (nu <= 16u) g3o_softmax_nu<16>(v, wrow, sinv_h, lane, m, sink_h, pe);
    else g3o_softmax_nu<20>(v, wrow, sinv_h, lane, m, sink_h, pe);
}
__device__ __forceinline__ void g3o_softmax_head(const _Float16* srow, _Float16* wrow, float* sinv_h,
                                                 unsigned lane, const RowMeta& m, float sink_h) {
    float v[MAX_PER_LANE];
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) if (u < (((m.n_raw + m.n_comp + 127u) >> 7) << 2)) v[u] = (float)srow[lane + u * 32u];
    g3o_softmax_regs(v, wrow, sinv_h, lane, m, sink_h);
}

// ================================================================================================
// S role: WG (row b, head-tile group hgp, chunk ch), NWS warps. Warp w: key tile ch * NWS + w, all
// HGP head tiles of the group (HGP chains over the same K fragments: each K tile is read once per
// group). Then the arrival count; the last arrival of a (row, head tile) runs its softmax.
// ================================================================================================
__device__ __forceinline__ void g3o_s_arrive_at(const G3OArgs& a, unsigned b, unsigned hgp, unsigned base_word);
template <bool ARRIVE>
__device__ __forceinline__ void g3o_s_body(const G3OArgs& a, unsigned b, unsigned hgp, unsigned ch,
                                           _Float16* q_lds, unsigned slot) {
    const unsigned tid = threadIdx.x, wave = tid >> 5, lane = tid & 31u;
    const RowMeta m = g3o_row(a, b);
    const unsigned t = ch * KTP + wave;
    const unsigned col = lane & 15u, kh = (lane >> 4) * 8u;
    half8 kf[32];
    if (ch * KTP < m.n_tiles) {   // WG-uniform (an empty chunk only counts its arrival, ticketed)
        // Issue order (the vector memory path returns in order): the selection entry (its value is
        // the K address), then q (L2-hot in production), then the 32 cold K fragments -- the q cast
        // waits only for q.
        const bool active = t < m.n_tiles;   // wave-uniform
        const _Float16* krow = active ? g3o_key(a, b, m, t, col) : a.zero_row;
        G3O_TSL(slot, 2);
        constexpr unsigned QF4 = G3O_HGP * HT * HD / 4u / STHREADS;   // float4 of q per thread
        float4 qx[QF4];
        const float4* qsrc = (const float4*)(a.q + ((size_t)b * NHEAD + hgp * G3O_HGP * HT) * HD);
        #pragma unroll
        for (unsigned i = 0; i < QF4; i++) qx[i] = qsrc[tid + i * STHREADS];
        #pragma unroll
        for (unsigned u = 0; u < 32u; u++) kf[u] = *(const half8*)(krow + u * 16u + kh);
        // the same (_Float16) cast as attn_dec_g3 / the production kernel
        #pragma unroll
        for (unsigned i = 0; i < QF4; i++) {
            const unsigned f = tid + i * STHREADS, qh = f >> 7, qd = (f & 127u) * 4u;
            typedef _Float16 half4 __attribute__((ext_vector_type(4)));
            half4 h;
            h[0] = (_Float16)qx[i].x; h[1] = (_Float16)qx[i].y; h[2] = (_Float16)qx[i].z; h[3] = (_Float16)qx[i].w;
            *(half4*)(q_lds + qh * QSTRIDE + qd) = h;
        }
        g3o_lds_barrier();
        G3O_TS(slot, 1);
        if (active) {
            // HGP independent 32-step chains over this tile's K fragments; A fragments from LDS in
            // groups of SAG, one group ahead (the asm pins each group's WMMAs before the next group's
            // loads, so the compiler does not hoist all of them next to the 32 K fragments)
            float8 c[G3O_HGP];
            #pragma unroll
            for (int r = 0; r < G3O_HGP; r++) c[r] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
            constexpr unsigned SAG = G3O_HGP >= 2 ? 2u : 4u;
            half8 acur[G3O_HGP][SAG], anext[G3O_HGP][SAG];
            #pragma unroll
            for (int r = 0; r < G3O_HGP; r++) {
                #pragma unroll
                for (unsigned i = 0; i < SAG; i++) acur[r][i] = *(const half8*)(q_lds + ((unsigned)r * 16u + col) * QSTRIDE + kh + i * 16u);
            }
            G3O_TSL(slot, 3);
            #pragma unroll
            for (unsigned g = 0; g < 32u; g += SAG) {
                if (g + SAG < 32u) {
                    #pragma unroll
                    for (int r = 0; r < G3O_HGP; r++) {
                        #pragma unroll
                        for (unsigned i = 0; i < SAG; i++)
                            anext[r][i] = *(const half8*)(q_lds + ((unsigned)r * 16u + col) * QSTRIDE + kh + (g + SAG + i) * 16u);
                    }
                }
                #pragma unroll
                for (unsigned i = 0; i < SAG; i++) {
                    #pragma unroll
                    for (int r = 0; r < G3O_HGP; r++) c[r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(acur[r][i], kf[g + i], c[r]);
                }
                #pragma unroll
                for (int r = 0; r < G3O_HGP; r++) asm volatile("" : "+v"(c[r]) :: "memory");
                #pragma unroll
                for (int r = 0; r < G3O_HGP; r++) {
                    #pragma unroll
                    for (unsigned i = 0; i < SAG; i++) acur[r][i] = anext[r][i];
                }
            }
            G3O_TS(slot, 4);
            const unsigned n_avail = t < m.n_tr ? min(16u, m.n_raw - t * 16u) : min(16u, m.n_comp - (t - m.n_tr) * 16u);
            const unsigned woff = t < m.n_tr ? t * 16u : m.n_raw + (t - m.n_tr) * 16u;
            if (col < n_avail) {
                #pragma unroll
                for (int r = 0; r < G3O_HGP; r++) {
                    _Float16* dst = a.sc + ((size_t)b * NHEAD + (hgp * G3O_HGP + (unsigned)r) * HT + kh) * SC_STRIDE + woff + col;
                    #pragma unroll
                    for (int e2 = 0; e2 < 8; e2++) dst[(size_t)e2 * SC_STRIDE] = (_Float16)(c[r][e2] * a.kq_scale);
                }
            }
            G3O_TS(slot, 5);
        }
    }
    // the scores are out: count this WG's arrival (ticketed), before the V^T work
    if constexpr (ARRIVE) g3o_s_arrive_at(a, b, hgp, S_CNT(0));
#if G3O_VT
    {
        const bool active = ch * KTP + wave < m.n_tiles;   // (wave-uniform; false for an empty chunk)
        const unsigned t = ch * KTP + wave;
        if (active) {
            // V^T: kf[u] is an A fragment of V[16 keys][dims 16u..16u+16) (lane = key col, 8 dims at kh);
            // D = V x I puts dim (lane & 15) of keys 8h..8h+8 in lane (dim, h) -- the W.V B fragment of
            // (tile t, dim tile u), stored as 512 contiguous bytes. Each head-tile group transposes
            // its 32 / NHG_S dim tiles (every group holds the same K tile).
            {
                half8 ident;
                #pragma unroll
                for (int i = 0; i < 8; i++) ident[i] = (kh + (unsigned)i == col) ? (_Float16)1.0f : (_Float16)0.0f;
                _Float16* vtb = a.vt + ((size_t)b * MAXT + t) * 32u * 256u + lane * 8u;
                constexpr unsigned DTS = 32u / NHG_S;
                #pragma unroll
                for (unsigned u = 0; u < 32u; u++) {
                    if (u / DTS == hgp) {   // wave-uniform
                        const float8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf[u], ident, (float8){0, 0, 0, 0, 0, 0, 0, 0});
                        half8 h;
                        #pragma unroll
                        for (int i = 0; i < 8; i++) h[i] = (_Float16)d[i];
                        *(half8*)(vtb + u * 256u) = h;
                    }
                }
            }
        }
    }
    if constexpr (ARRIVE) g3o_s_arrive_at(a, b, hgp, S_VCNT(0));
#endif
}

// Ticketed entry: release this S WG's stores and count its arrival for each of its head tiles
// (base_word = S_CNT(0) for the scores, S_VCNT(0) for the V^T fragments).
__device__ __forceinline__ void g3o_s_arrive_at(const G3OArgs& a, unsigned b, unsigned hgp, unsigned base_word) {
    __syncthreads();
    if (threadIdx.x < G3O_HGP) {
        __threadfence();
        atomicAdd(a.sync + base_word + (b * 4u + hgp * G3O_HGP + threadIdx.x) * S_LINE, 1u);
    }
}

// ================================================================================================
// V role: WG (row b, head tile hg, dim part dp), NWV warps; warp w = dim tile dp * NWV + w.
// ================================================================================================
struct VLds {
    _Float16 w[HT * WSTRIDE];
    const _Float16* tbl[TBL_N];
    float s_inv[HT];
    float rope[64];
    float amax[NWV][HT];
    float dq[NWV][HT][2];
};

template <bool WAIT>
__device__ __forceinline__ void g3o_v_body(const G3OArgs& a, unsigned b, unsigned hg, unsigned dp, VLds& L,
                                           unsigned slot) {
    const unsigned tid = threadIdx.x, wave = tid >> 5, lane = tid & 31u;
    const RowMeta m = g3o_row(a, b);
    const unsigned h0 = hg * HT;
    const unsigned dt = dp * NWV + wave;
    const unsigned col = lane & 15u, kh = (lane >> 4) * 8u;
    const unsigned n_tiles = m.n_tiles;

    // ---- the rope tail's (cos, -sin) pairs (dims 448..511 are in the last part) ----------------
    if (dp == G3O_DP - 1u && wave == 0u) {
        const float th = g3_theta(a.pos_per_b[b], lane, a.theta_scale, a.freq_scale, a.ext_factor, a.corr_low, a.corr_high);
        L.rope[lane] = cosf(th) * a.mscale_eff;
        L.rope[32u + lane] = -1.0f * sinf(th) * a.mscale_eff;
    }
#if G3O_VT
    (void)col;
    constexpr unsigned R = G3O_VPF < MAXT ? G3O_VPF : MAXT;           // V fragments in registers
    short8 vf[R];
    const _Float16* vtb = a.vt + (size_t)b * MAXT * 32u * 256u + dt * 256u + lane * 8u;
    auto vload = [&](unsigned t) -> short8 { return *(const short8*)(vtb + (size_t)t * 32u * 256u); };
    G3O_TS(slot, 5);
    // ---- wait for the scores of hg (its group's S WGs), load them --------------------------------
    auto poll = [&](unsigned word) {
        if (wave == 0u) {
            const unsigned* p = a.sync + word;
            const unsigned long long t0 = g3o_now();
            while (g3o_sload_dev(p) < NCH) {
                if (g3o_now() - t0 > G3O_SPIN_CAP) { if (lane == 0u) atomicOr(a.sync + S_ERR, 1u); break; }
                __builtin_amdgcn_s_sleep(1);
            }
        }
    };
    if constexpr (WAIT) {
        poll(S_CNT(b * 4u + (hg / G3O_HGP) * G3O_HGP));
        g3o_lds_barrier();
        g3o_vinv();
    }
    G3O_TS(slot, 2);
    float sv[HT / NWV][MAX_PER_LANE];
    #pragma unroll
    for (unsigned j = 0; j < HT / NWV; j++) {
        const _Float16* srow = a.sc + ((size_t)b * NHEAD + h0 + wave * (HT / NWV) + j) * SC_STRIDE;
        const unsigned nu4 = ((m.n_raw + m.n_comp + 127u) >> 7) << 2;   // blocks the class reads
        #pragma unroll
        for (unsigned u = 0; u < MAX_PER_LANE; u++) if (u < nu4) sv[j][u] = (float)srow[lane + u * 32u];
    }
    // ---- the V^T fragments of these dims (their groups' second arrival), the first half now ------
    if constexpr (WAIT) {
        constexpr unsigned DTS = 32u / NHG_S;
        const unsigned g0 = (dp * NWV) / DTS, g1 = (dp * NWV + NWV - 1u) / DTS;
        for (unsigned g = g0; g <= g1; g++) poll(S_VCNT(b * 4u + g * G3O_HGP));
        g3o_lds_barrier();
        g3o_vinv();
    }
    constexpr unsigned VH = R / 2u;
    g3o_for<0u, VH>([&](auto I) {
        constexpr unsigned t = decltype(I)::value;
        if (t < n_tiles) vf[t] = vload(t);
    });
    G3O_TS(slot, 1);
    #pragma unroll
    for (unsigned j = 0; j < HT / NWV; j++) {
        const unsigned hl = wave * (HT / NWV) + j;
        g3o_softmax_regs(sv[j], L.w + hl * WSTRIDE, &L.s_inv[hl], lane, m, a.sinks[h0 + hl]);
    }
    g3o_for<VH, R>([&](auto I) {
        constexpr unsigned t = decltype(I)::value;
        if (t < n_tiles) vf[t] = vload(t);
    });
    g3o_lds_barrier();
    G3O_TS(slot, 3);
#else
    // ---- key table (the selection read once per WG) ------------------------------------------
    for (unsigned i = tid; i < TBL_N; i += VTHREADS) L.tbl[i] = g3o_key(a, b, m, i >> 4, i & 15u);
    g3o_lds_barrier();
    G3O_TS(slot, 5);

    // ---- V: fragment-pattern loads, transposed in place by WMMA against the identity -----------
    // (independent of the scores: all of it runs before the weights are ready)
    constexpr unsigned R = G3O_VPF < MAXT ? G3O_VPF : MAXT;           // transposed ring
    constexpr unsigned PD = R < MAXT ? (G3O_VPD < MAXT - R ? G3O_VPD : MAXT - R) : 1u;   // raw ring
    short8 vf[R];
    short8 vraw[PD];
    half8 ident;
    #pragma unroll
    for (int i = 0; i < 8; i++) ident[i] = (kh + (unsigned)i == col) ? (_Float16)1.0f : (_Float16)0.0f;
    auto vload = [&](unsigned t) -> short8 { return *(const short8*)(L.tbl[t * 16u + col] + dt * 16u + kh); };
    auto vtrans = [&](short8 x) -> short8 {
        const float8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(half8, x), ident,
                                                                        (float8){0, 0, 0, 0, 0, 0, 0, 0});
        half8 h;
        #pragma unroll
        for (int i = 0; i < 8; i++) h[i] = (_Float16)d[i];
        return __builtin_bit_cast(short8, h);
    };
    g3o_for<0u, R>([&](auto I) {
        constexpr unsigned t = decltype(I)::value;
        if (t < n_tiles) vf[t] = vload(t);
    });
    if constexpr (R < MAXT) {
        g3o_for<0u, PD>([&](auto I) {
            constexpr unsigned t = R + decltype(I)::value;
            if (t < n_tiles) vraw[t % PD] = vload(t);
        });
    }
    G3O_TSL(slot, 1);
    g3o_for<0u, R>([&](auto I) {
        constexpr unsigned t = decltype(I)::value;
        if (t < n_tiles) vf[t] = vtrans(vf[t]);
    });

    // ---- wait for the scores of (b, hg): every S WG of its group arrived ------------------------
    if constexpr (WAIT) {
        if (wave == 0u) {
            const unsigned* p = a.sync + S_CNT(b * 4u + hg);
            const unsigned long long t0 = g3o_now();
            while (g3o_sload_dev(p) < NCH) {
                if (g3o_now() - t0 > G3O_SPIN_CAP) { if (lane == 0u) atomicOr(a.sync + S_ERR, 1u); break; }
                __builtin_amdgcn_s_sleep(1);
            }
        }
        g3o_lds_barrier();
        g3o_vinv();
    }
    G3O_TS(slot, 2);

    // ---- the softmax of the 16 heads (HT / NWV per warp), weights + 1 / sum into LDS -------------
    #pragma unroll
    for (unsigned j = 0; j < HT / NWV; j++) {
        const unsigned hl = wave * (HT / NWV) + j;
        g3o_softmax_head(a.sc + ((size_t)b * NHEAD + h0 + hl) * SC_STRIDE, L.w + hl * WSTRIDE, &L.s_inv[hl], lane, m,
                         a.sinks[h0 + hl]);
    }
    g3o_lds_barrier();
    G3O_TS(slot, 3);

#endif

    // ---- W.V: one 16-dim tile, the chain over the key tiles in order ----------------------------
    const _Float16* wr = L.w + col * WSTRIDE + kh;
    float8 c = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    g3o_for<0u, MAXT>([&](auto I) {
        constexpr unsigned t = decltype(I)::value;
        if (t < n_tiles) {
            const half8 af = *(const half8*)(wr + t * 16u);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, __builtin_bit_cast(half8, vf[t % R]), c);
#if G3O_VT
            if constexpr (R < MAXT && t + R < MAXT) {
                if (t + R < n_tiles) vf[t % R] = vload(t + R);   // slot t % R streams tile t + R
            }
#else
            if constexpr (R < MAXT && t + R < MAXT) {
                // slot t % R is free: the next tile it holds (t + R) arrives transposed from the raw
                // ring, whose slot then takes the raw load of tile t + R + PD
                if (t + R < n_tiles) {
                    vf[t % R] = vtrans(vraw[(t + R) % PD]);
                    if constexpr (t + R + PD < MAXT) {
                        if (t + R + PD < n_tiles) vraw[(t + R) % PD] = vload(t + R + PD);
                    }
                }
            }
#endif
        }
    });
    G3O_TS(slot, 4);

    // ---- epilogue: 1/sum, inverse rope (dim tiles 28..31), Q8_0 per 32-dim block ----------------
    const unsigned d = dt * 16u + col;
    float v[8];
    #pragma unroll
    for (int e = 0; e < 8; e++) v[e] = c[e] * L.s_inv[kh + (unsigned)e];
    if (dt >= 28u) {   // wave-uniform: the rope tail
        const float cr = L.rope[(d - 448u) >> 1], sr = L.rope[32u + ((d - 448u) >> 1)];
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const float p = g3_xor<1>(v[e]);
            v[e] = g3_rot(v[e], p, d, cr, sr);
        }
    }
    float am[8];
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        float x = fabsf(v[e]);
        x = fmaxf(x, g3_xor<8>(x));
        x = fmaxf(x, g3_xor<4>(x));
        x = fmaxf(x, g3_xor<2>(x));
        x = fmaxf(x, g3_xor<1>(x));
        am[e] = x;
    }
    if (col == 0u) {
        #pragma unroll
        for (int e = 0; e < 8; e++) L.amax[wave][kh + (unsigned)e] = am[e];
    }
    g3o_lds_barrier();
    G3O_TS(slot, 6);
    // the block's scale: lane col < 8 of each half for head kh + col (q8_0's d = amax / 127,
    // id = 1 / d; the pair of warps of a block computes the same values)
    if (col < 8u) {
        const float amax = fmaxf(L.amax[wave][kh + col], L.amax[wave ^ 1u][kh + col]);
        const float dd = amax / 127.0f;
        L.dq[wave][kh + col][0] = dd;
        L.dq[wave][kh + col][1] = dd != 0.0f ? 1.0f / dd : 0.0f;
        if ((dt & 1u) == 0u) a.xscale[(size_t)b * 1024u + (h0 + kh + col) * 16u + (dt >> 1)] = dd;
    }
    __builtin_amdgcn_wave_barrier();
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    const size_t row = (size_t)b * 32768u;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned head = h0 + kh + (unsigned)e;
        const float id = L.dq[wave][kh + (unsigned)e][1];
        int qv = (int)lrintf(v[e] * id);
        qv = qv > 127 ? 127 : (qv < -128 ? -128 : qv);
        const size_t hbase = row + (size_t)head * 512u;
        a.xq[hbase + d] = (int8_t)qv;
        if (a.heads_out) a.heads_out[hbase + d] = v[e];
    }
}


// ================================================================================================
// "W" family: ONE work-group per (row, head tile, 1/DPW of the dims), no inter-WG communication.
// Each WG computes every score of its 16 heads (the K tiles of warp w: w, w + 16, w + 32, each K
// tile's 32 fragments prefetched into the registers the previous tile's chain just consumed), the
// softmax (attn_dec_g3's phase A, the fast form above), and the W.V chains + epilogue of its dims
// (V by global_load_tr_b128, a WPD-deep ring started BEFORE the softmax). DPW > 1 replicates the
// scores + softmax per dim part (more WGPs busy at small b; every output keeps its chain).
// ================================================================================================
#ifndef G3O_WPD
#define G3O_WPD 4
#endif
#ifndef G3O_WABL
#define G3O_WABL 0
#endif
#ifndef G3O_WSTG
#define G3O_WSTG 0
#endif
#ifndef G3O_WPF
#define G3O_WPF 0                        // 1: vector L2 prefetch of the later K tiles, 2: scalar prefetch
#endif
#ifndef G3O_WPFL
#define G3O_WPFL 15                      // s_prefetch_data length field
#endif
#ifndef G3O_WMAP
#define G3O_WMAP 0                       // 1: dispatch order row-fastest (a row's head tiles b WGs apart)
#endif

#define WSTRIDE_W 712u                   // W weights row: >= 44 tiles * 16 (ring pad), 16-B multiple
struct WLds {
    _Float16 q[HT * QSTRIDE];
    _Float16 w[HT * WSTRIDE_W];
    const _Float16* tbl[TBL_N];
    float s_inv[HT];
    float rope[64];
    float amax[32][HT];
    float dq[32][HT][2];
};

// Epilogue of one 16-dim tile, part 1: 1/sum, inverse rope, the tile's per-head amax into LDS slot
__device__ __forceinline__ void g3o_epi1(const float8& c, unsigned dt, unsigned slot, unsigned col, unsigned kh,
                                         const float* s_inv, const float* rope, float (*amax)[HT], float (&v)[8]) {
    const unsigned d = dt * 16u + col;
    #pragma unroll
    for (int e = 0; e < 8; e++) v[e] = c[e] * s_inv[kh + (unsigned)e];
    if (dt >= 28u) {   // wave-uniform: the rope tail
        const float cr = rope[(d - 448u) >> 1], sr = rope[32u + ((d - 448u) >> 1)];
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const float p = g3_xor<1>(v[e]);
            v[e] = g3_rot(v[e], p, d, cr, sr);
        }
    }
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        float x = fabsf(v[e]);
        x = fmaxf(x, g3_xor<8>(x));
        x = fmaxf(x, g3_xor<4>(x));
        x = fmaxf(x, g3_xor<2>(x));
        x = fmaxf(x, g3_xor<1>(x));
        if (col == 0u) amax[slot][kh + (unsigned)e] = x;
    }
}
// part 2 (after a barrier): the block's scale from the pair (slot, slot ^ 1), quantize, store.
// O8 (`V41_ATTN_O_FP8`): the act_quant scale and e4m3 codes instead (see g3o_v_body).
template <bool O8 = false>
__device__ __forceinline__ void g3o_epi2(const G3OArgs& a, unsigned b, unsigned h0, unsigned dt, unsigned slot,
                                         unsigned col, unsigned kh, float (*amax)[HT], float (*dq)[HT][2],
                                         const float (&v)[8], int8_t* stg = nullptr, unsigned stg_stride = 0u,
                                         unsigned stg_d0 = 0u) {
    if (col < 8u) {
        const float am = fmaxf(amax[slot][kh + col], amax[slot ^ 1u][kh + col]);
        if constexpr (O8) {
            const float s = aq_pow2(aq_scale_exp(am));
            dq[slot][kh + col][0] = s;
            if ((dt & 1u) == 0u) a.xscale[(size_t)b * 1024u + (h0 + kh + col) * 16u + (dt >> 1)] = s;
        } else {
        const float dd = am / 127.0f;
        dq[slot][kh + col][0] = dd;
        dq[slot][kh + col][1] = dd != 0.0f ? 1.0f / dd : 0.0f;
        if ((dt & 1u) == 0u) a.xscale[(size_t)b * 1024u + (h0 + kh + col) * 16u + (dt >> 1)] = dd;
        }
    }
    __builtin_amdgcn_wave_barrier();
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    const unsigned d = dt * 16u + col;
    const size_t row = (size_t)b * 32768u;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned head = h0 + kh + (unsigned)e;
        if constexpr (O8) {
            const int ex = (int)(__float_as_uint(dq[slot][kh + (unsigned)e][0]) >> 23) - 127;
            const int8_t qv = (int8_t)aq_e4m3_code(v[e], ex);
            const size_t hbase = row + (size_t)head * 512u;
            if (stg) stg[(kh + (unsigned)e) * stg_stride + d - stg_d0] = qv;
            else a.xq[hbase + d] = qv;
            if (a.heads_out) a.heads_out[hbase + d] = v[e];
            continue;
        }
        const float id = dq[slot][kh + (unsigned)e][1];
        int qv = (int)lrintf(v[e] * id);
        qv = qv > 127 ? 127 : (qv < -128 ? -128 : qv);
        const size_t hbase = row + (size_t)head * 512u;
        if (stg) stg[(kh + (unsigned)e) * stg_stride + d - stg_d0] = (int8_t)qv;   // staged: one 16-B store later
        else a.xq[hbase + d] = (int8_t)qv;
        if (a.heads_out) a.heads_out[hbase + d] = v[e];
    }
}

template <unsigned DPW, unsigned NW, bool O8 = false>
__device__ __forceinline__ void g3o_w_body(const G3OArgs& a, unsigned hg, unsigned b, unsigned dp, WLds& L) {
    constexpr unsigned NT = NW * 32u;                    // threads
    constexpr unsigned NDT = 32u / DPW;                  // dim tiles of this WG
    constexpr unsigned DTW = NDT > NW ? NDT / NW : 1u;   // dim tiles per W.V warp
    constexpr unsigned NWW = NDT / DTW;                  // W.V warps
    constexpr unsigned NTI = (MAXT + NW - 1u) / NW;      // most score tiles per warp (3 or 2)
    constexpr unsigned PD = G3O_WPD;
    const unsigned tid = threadIdx.x, wave = tid >> 5, lane = tid & 31u;
    const unsigned col = lane & 15u, kh = (lane >> 4) * 8u;
    const RowMeta m = g3o_row(a, b);
    const unsigned h0 = hg * HT, n_tiles = m.n_tiles;
    const unsigned slot = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    G3O_TS(slot, 0);

    // ---- every address-producing load first (the selection), then q, then the first K tile ------
    // (the vector memory path returns in order: the table stores and the q cast then wait only for
    // loads older than the 32 cold K fragments)
    const unsigned i0 = min(tid, TBL_N - 1u), i1 = min(tid + NT, TBL_N - 1u);
    const int st0 = g3o_sel_ld(a, b, m, i0 >> 4, i0 & 15u);
    const int st1 = g3o_sel_ld(a, b, m, i1 >> 4, i1 & 15u);
    int sk[NTI];
    #pragma unroll
    for (unsigned i = 0; i < NTI; i++) sk[i] = g3o_sel_ld(a, b, m, wave + NW * i, col);
    const unsigned ntw = n_tiles > wave ? (n_tiles - wave + NW - 1u) / NW : 0u;   // this warp's S tiles
    const _Float16* kr[NTI + 1];
    #pragma unroll
    for (unsigned i = 0; i < NTI; i++) kr[i] = (G3O_WABL & 1) ? a.zero_row : g3o_key_at(a, b, m, wave + NW * i, col, sk[i]);
    kr[NTI] = a.zero_row;
#if G3O_WPF == 1
    // L2 prefetch of this warp's later K tiles (one 4-byte load per 128-byte line, results unused but
    // kept live until a wait below; older than the real loads that follow, so the compiler's in-order
    // wait counts can only over-wait, never under-wait)
    unsigned pf_sink = 0;
    #pragma unroll
    for (unsigned i = 1; i < NTI; i++) {
        #pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            const _Float16* pa = kr[i] + ((lane >> 4) + 2u * j) * 64u;
            asm volatile("global_load_b32 %0, %1, off" : "+v"(pf_sink) : "v"(pa) : "memory");
        }
    }
#endif
    const _Float16* te[2] = {g3o_key_at(a, b, m, i0 >> 4, i0 & 15u, st0), g3o_key_at(a, b, m, i1 >> 4, i1 & 15u, st1)};
    constexpr unsigned QF4 = HT * HD / 4u / NT;          // float4 of q per thread (4 or 2)
    float4 qx[QF4];
    {
        const float4* src = (const float4*)(a.q + ((size_t)b * NHEAD + h0) * HD) + tid * QF4;
        #pragma unroll
        for (unsigned i = 0; i < QF4; i++) qx[i] = src[i];
    }
    half8 kf[32];
    #pragma unroll
    for (unsigned u = 0; u < 32u; u++) kf[u] = *(const half8*)(kr[0] + u * 16u + kh);
#if G3O_WPF == 2
    // the later tiles' K rows into the cache hierarchy now (scalar prefetch: 1 KB per row, the
    // row address read from the lane that holds it)
    #pragma unroll
    for (unsigned i = 1; i < NTI; i++) {
        if (wave + NW * i < n_tiles) {   // wave-uniform
            const unsigned long long ra = (unsigned long long)kr[i];
            #pragma unroll
            for (unsigned r = 0; r < 16u; r++) {
                const unsigned lo = (unsigned)__builtin_amdgcn_readlane((int)(unsigned)ra, (int)r);
                const unsigned hi = (unsigned)__builtin_amdgcn_readlane((int)(unsigned)(ra >> 32), (int)r);
                __builtin_amdgcn_s_prefetch_data((const void*)(((unsigned long long)hi << 32) | lo), G3O_WPFL);
            }
        }
    }
#endif
    if (tid < TBL_N) L.tbl[tid] = (G3O_WABL & 2) ? a.zero_row : te[0];
    if (tid + NT < TBL_N) L.tbl[tid + NT] = (G3O_WABL & 2) ? a.zero_row : te[1];
    {
        const unsigned f0 = tid * QF4 * 4u, qh = f0 / HD, qd = f0 % HD;
        #pragma unroll
        for (unsigned i = 0; i < QF4; i += 2u) {
            half8 h;
            h[0] = (_Float16)qx[i].x;     h[1] = (_Float16)qx[i].y;     h[2] = (_Float16)qx[i].z;     h[3] = (_Float16)qx[i].w;
            h[4] = (_Float16)qx[i + 1].x; h[5] = (_Float16)qx[i + 1].y; h[6] = (_Float16)qx[i + 1].z; h[7] = (_Float16)qx[i + 1].w;
            *(half8*)(L.q + qh * QSTRIDE + qd + i * 4u) = h;
        }
    }
    g3o_lds_barrier();
    G3O_TS(slot, 1);

    // ---- scores: this warp's tiles wave, wave + NW, .. (a runtime loop with a branch-free body);
    // the next tile's fragment u is loaded into kf[u] as soon as the chain consumed it -------------
    const _Float16* qa = L.q + col * QSTRIDE + kh;
    for (unsigned i = 0; i < ntw; i++) {
        const unsigned t = wave + NW * i;
        const _Float16* kn = i == 0u ? kr[1] : (NTI > 2u && i == 1u ? kr[NTI > 2u ? 2u : 1u] : kr[NTI]);
        float8 c = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        constexpr unsigned SAG = 4u;
        half8 acur[SAG], anext[SAG];
        #pragma unroll
        for (unsigned j = 0; j < SAG; j++) acur[j] = *(const half8*)(qa + j * 16u);
        #pragma unroll
        for (unsigned g = 0; g < 32u; g += SAG) {
            if (g + SAG < 32u) {
                #pragma unroll
                for (unsigned j = 0; j < SAG; j++) anext[j] = *(const half8*)(qa + (g + SAG + j) * 16u);
            }
            #pragma unroll
            for (unsigned j = 0; j < SAG; j++) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(acur[j], kf[g + j], c);
            asm volatile("" : "+v"(c) :: "memory");
            #pragma unroll
            for (unsigned j = 0; j < SAG; j++) kf[g + j] = *(const half8*)(kn + (g + j) * 16u + kh);
            #pragma unroll
            for (unsigned j = 0; j < SAG; j++) acur[j] = anext[j];
        }
        const unsigned n_avail = t < m.n_tr ? min(16u, m.n_raw - t * 16u) : min(16u, m.n_comp - (t - m.n_tr) * 16u);
        const unsigned woff = t < m.n_tr ? t * 16u : m.n_raw + (t - m.n_tr) * 16u;
        if (col < n_avail) {
            #pragma unroll
            for (int e2 = 0; e2 < 8; e2++) L.w[(kh + (unsigned)e2) * WSTRIDE_W + woff + col] = (_Float16)(c[e2] * a.kq_scale);
        }
    }
#if G3O_WPF == 1
    asm volatile("s_wait_loadcnt 0x0" ::: "memory");
    if (pf_sink == 0x9e3779b9u && lane == 99u) a.xscale[0] = 0.0f;   // never: keeps the prefetch register live
#endif
    // the rope tail's (cos, -sin) pairs by the last warp (fewest tiles), when this part holds dims 448..
    if (wave == NW - 1u && dp == DPW - 1u) {
        const float th = g3_theta(a.pos_per_b[b], lane, a.theta_scale, a.freq_scale, a.ext_factor, a.corr_low, a.corr_high);
        L.rope[lane] = cosf(th) * a.mscale_eff;
        L.rope[32u + lane] = -1.0f * sinf(th) * a.mscale_eff;
    }
    g3o_lds_barrier();
    G3O_TS(slot, 2);

    // ---- V: the first PD tiles' transposed loads issued BEFORE the softmax (table entries past
    // n_tiles are the zero row: every load below is unconditional) ---------------------------------
    const unsigned kk = (lane >> 4) * 8u + (lane & 7u);
    const unsigned dtb = dp * NDT + wave * DTW;          // this warp's first dim tile
    const unsigned dcol = dtb * 16u + ((lane >> 3) & 1u) * 8u;
    const bool vw = wave < NWW;                          // wave-uniform: does W.V
    const unsigned ntp = (n_tiles + PD - 1u) / PD * PD;  // tiles the ring walks (weights zero past n_tiles)
    short8 rv[PD][DTW];
    auto issue = [&](unsigned t, short8 (&dst)[DTW]) {
        const _Float16* p = L.tbl[min(t, TBL_N / 16u - 1u) * 16u + kk] + dcol;
        #pragma unroll
        for (unsigned k = 0; k < DTW; k++)
            dst[k] = __builtin_amdgcn_global_load_tr_b128_v8i16((short8 __attribute__((address_space(1)))*)(p + 16u * k));
    };
    if (vw) {
        #pragma unroll
        for (unsigned p = 0; p < PD; p++) issue(p, rv[p]);
    }

    // ---- softmax: warp w < 16 = head w (scores read from the row, weights written padded in place) --
    if (wave < HT && !(G3O_WABL & 4)) {
        float sv[MAX_PER_LANE];
        _Float16* wrow = L.w + wave * WSTRIDE_W;
        const unsigned nu4 = ((m.n_raw + m.n_comp + 127u) >> 7) << 2;
        #pragma unroll
        for (unsigned u = 0; u < MAX_PER_LANE; u++) if (u < nu4) sv[u] = (float)wrow[lane + u * 32u];
        asm volatile("s_wait_dscnt 0x0" ::: "memory");   // every read of the row lands before a write
        g3o_softmax_regs(sv, wrow, &L.s_inv[wave], lane, m, a.sinks[h0 + wave], ntp * 16u);
    }
    g3o_lds_barrier();
    G3O_TS(slot, 3);

    // ---- W.V: DTW chains over the key tiles in order, V from the PD-deep ring ----------------------
    float8 c[DTW];
    #pragma unroll
    for (unsigned k = 0; k < DTW; k++) c[k] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    if (vw && !(G3O_WABL & 8)) {
        const _Float16* wr = L.w + col * WSTRIDE_W + kh;
        for (unsigned blk = 0; blk < ntp; blk += PD) {
            #pragma unroll
            for (unsigned p = 0; p < PD; p++) {
                const half8 af = *(const half8*)(wr + (blk + p) * 16u);
                #pragma unroll
                for (unsigned k = 0; k < DTW; k++)
                    c[k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, __builtin_bit_cast(half8, rv[p][k]), c[k]);
                issue(blk + p + PD, rv[p]);
            }
        }
    }
    G3O_TS(slot, 4);

    // ---- epilogue (Q8_0 blocks = dim tile pairs: slots k + DTW * wave, pairs (2j, 2j + 1)) --------
    float v[DTW][8];
    if (vw) {
        #pragma unroll
        for (unsigned k = 0; k < DTW; k++) g3o_epi1(c[k], dtb + k, wave * DTW + k, col, kh, L.s_inv, L.rope, L.amax, v[k]);
    }
    g3o_lds_barrier();
#if G3O_WSTG
    // the quantized bytes through the (dead) q buffer, one 16-B store per thread (measured slower:
    // +1.5-2 us at b = 1..16 -- the extra barrier costs more than the byte stores)
    int8_t* stg = (int8_t*)L.q;
    constexpr unsigned SW = NDT * 16u;
    if (vw) {
        #pragma unroll
        for (unsigned k = 0; k < DTW; k++)
            g3o_epi2<O8>(a, b, h0, dtb + k, wave * DTW + k, col, kh, L.amax, L.dq, v[k], stg, SW, dp * SW);
    }
    g3o_lds_barrier();
    for (unsigned i = tid; i < HT * SW / 16u; i += NT) {
        const unsigned hl = i / (SW / 16u), c16 = i - hl * (SW / 16u);
        *(uint4*)(a.xq + (size_t)b * 32768u + (size_t)(h0 + hl) * 512u + dp * SW + c16 * 16u) =
            *(const uint4*)(stg + hl * SW + c16 * 16u);
    }
#else
    if (vw) {
        #pragma unroll
        for (unsigned k = 0; k < DTW; k++) g3o_epi2<O8>(a, b, h0, dtb + k, wave * DTW + k, col, kh, L.amax, L.dq, v[k]);
    }
#endif
    G3O_TS(slot, 7);
}

#endif  // gfx12

#if defined(G3O_EMPTY) && (defined(__gfx1200__) || defined(__gfx1201__))
#define G3O_BODY(...) { if (a.b == 0xdeadu) a.xscale[0] = 0.0f; }   // timing experiments ONLY: no work
#elif defined(__gfx1200__) || defined(__gfx1201__)
#define G3O_BODY(...) __VA_ARGS__
#else
#define G3O_BODY(...) (void)a;
#endif

// Two launches: S then V.
extern "C" __global__ __launch_bounds__(STHREADS) void attn_g3o_s(G3OArgs a) {
    G3O_BODY(
    __shared__ __attribute__((aligned(16))) _Float16 q_lds[G3O_HGP * HT * QSTRIDE];
    const unsigned slot = (blockIdx.z * NHG_S + blockIdx.y) * NCH + blockIdx.x;
    G3O_TS(slot, 0);
    g3o_s_body<false>(a, blockIdx.z, blockIdx.y, blockIdx.x, q_lds, slot);
    G3O_TS(slot, 7);
    )
}

extern "C" __global__ __launch_bounds__(VTHREADS) void attn_g3o_v(G3OArgs a) {
    G3O_BODY(
    __shared__ __attribute__((aligned(16))) VLds L;
    const unsigned slot = 1024u + (blockIdx.z * 4u + blockIdx.y) * G3O_DP + blockIdx.x;
    G3O_TS(slot, 0);
    g3o_v_body<false>(a, blockIdx.z, blockIdx.y, blockIdx.x, L, slot);
    G3O_TS(slot, 7);
    )
}

// One launch, ticketed roles (grid = nS + nV WGs of STHREADS = VTHREADS threads).
#ifndef G3O_TWPE
#define G3O_TWPE 1
#endif
// G3O_TWPE: minimum waves per SIMD for the ticketed entry (8 caps it at 192 VGPRs: two 16-wave WGs
// per WGP, so a V WG can sit beside an S WG and issue its V loads during the score phase)
extern "C" __global__ __launch_bounds__(STHREADS) __attribute__((amdgpu_waves_per_eu(G3O_TWPE))) void attn_g3o_t(G3OArgs a) {
#if STHREADS != VTHREADS
    (void)a;   // the ticketed entry needs S and V WGs of the same size: every wave meets every barrier
#else
    G3O_BODY(
    constexpr size_t SB = sizeof(_Float16) * G3O_HGP * HT * QSTRIDE;
    constexpr size_t LB = SB > sizeof(VLds) ? SB : sizeof(VLds);
    __shared__ __attribute__((aligned(16))) unsigned char lds[LB];
    __shared__ unsigned s_ticket;
    const unsigned tid = threadIdx.x;
    if (tid == 0) s_ticket = atomicAdd(a.sync + S_TICKET, 1u);
    g3o_lds_barrier();
    const unsigned t = s_ticket;
    G3O_TS(t, 0);
    const unsigned nS = a.b * NHG_S * NCH;
    const unsigned nV = a.b * 4u * G3O_DP;
    if (t < nS) {
        const unsigned b = t / (NHG_S * NCH), r = t - b * (NHG_S * NCH);
        const unsigned hgp = r / NCH, ch = r - hgp * NCH;
        g3o_s_body<true>(a, b, hgp, ch, (_Float16*)lds, t);
    } else {
        const unsigned v = t - nS;
        const unsigned b = v / (4u * G3O_DP), r = v - b * (4u * G3O_DP);
        const unsigned hg = r / G3O_DP, dp = r - hg * G3O_DP;
        g3o_v_body<true>(a, b, hg, dp, *(VLds*)lds, t);
    }
    G3O_TS(t, 7);
    // the last WG resets the scratch (every other WG has made all its sync accesses)
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned prev = atomicAdd(a.sync + S_FINISH, 1u);
        if (prev == nS + nV - 1u) {
            __threadfence();
            a.sync[S_TICKET] = 0u;
            for (unsigned i = 0; i < a.b * 4u; i++) { a.sync[S_CNT(i)] = 0u; a.sync[S_VCNT(i)] = 0u; }
            a.sync[S_FINISH] = 0u;
            __threadfence();
        }
    }
    )
#endif
}

// W family: one WG per (head tile, row, dim part) -- grid (4, b, DPW) x 512.
#define G3O_W_ENTRY(DPW, NW, NAME)                                                                   \
extern "C" __global__ __launch_bounds__(NW * 32) void NAME(G3OArgs a) {                              \
    G3O_BODY(                                                                                        \
    __shared__ __attribute__((aligned(16))) WLds L;                                                  \
    const unsigned lin_ = blockIdx.y * 4u + blockIdx.x;                                               \
    const unsigned hg_ = G3O_WMAP ? lin_ / a.b : blockIdx.x, row_ = G3O_WMAP ? lin_ % a.b : blockIdx.y; \
    g3o_w_body<DPW, NW>(a, hg_, row_, blockIdx.z, L);                                                \
    )                                                                                                \
}
G3O_W_ENTRY(1, 16, attn_g3o_w1)
G3O_W_ENTRY(2, 16, attn_g3o_w2)
G3O_W_ENTRY(4, 16, attn_g3o_w4)
G3O_W_ENTRY(1, 32, attn_g3o_x1)
G3O_W_ENTRY(2, 32, attn_g3o_x2)

// `V41_ATTN_O_FP8`: the W family with the act_quant epilogue (g3o_epi2<true>): xq gets the e4m3
// codes, xscale the per-32 power-of-two scales. (The two-launch / ticketed entries have no twin:
// production's `V41_ATTN_G3_KERNEL` picks only W modes; `AttnDecG3o::launch` refuses the others.)
#define G3O_W_ENTRY_O8(DPW, NW, NAME)                                                                \
extern "C" __global__ __launch_bounds__(NW * 32) void NAME(G3OArgs a) {                              \
    G3O_BODY(                                                                                        \
    __shared__ __attribute__((aligned(16))) WLds L;                                                  \
    const unsigned lin_ = blockIdx.y * 4u + blockIdx.x;                                               \
    const unsigned hg_ = G3O_WMAP ? lin_ / a.b : blockIdx.x, row_ = G3O_WMAP ? lin_ % a.b : blockIdx.y; \
    g3o_w_body<DPW, NW, true>(a, hg_, row_, blockIdx.z, L);                                          \
    )                                                                                                \
}
G3O_W_ENTRY_O8(1, 16, attn_g3o_w1_o8)
G3O_W_ENTRY_O8(2, 16, attn_g3o_w2_o8)
G3O_W_ENTRY_O8(4, 16, attn_g3o_w4_o8)
G3O_W_ENTRY_O8(1, 32, attn_g3o_x1_o8)
G3O_W_ENTRY_O8(2, 32, attn_g3o_x2_o8)

// The compiled variant's geometry (the host derives grids from it).
extern "C" __global__ void attn_g3o_cfg(unsigned* out) {
    if (threadIdx.x == 0) {
        out[0] = G3O_HGP; out[1] = KTP; out[2] = NCH; out[3] = G3O_DP; out[4] = NWV; out[5] = G3O_VPF < MAXT ? G3O_VPF : MAXT;
        out[6] = S_WORDS + 32u; out[7] = SC_STRIDE; out[8] = G3O_NWS; out[9] = WSTRIDE; out[10] = G3O_VT;
    }
}
