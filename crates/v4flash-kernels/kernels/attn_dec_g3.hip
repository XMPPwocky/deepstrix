// attn_dec_g3 -- the decode attention BLOCK as ONE launch (dGPU fusion group G3, 2026-10-13):
//
//   attn_meta_fill (n_raw / raw_off / n_comp per row)          -> kernel ARGUMENTS
//   indexer_gather (top-512 rows into attn_active_comp_kv)      -> rows read IN PLACE through the selection
//   attention_dec_fused_vt_qreg_sp_d4 (score + softmax + W.V)   -> same arithmetic, same order
//   rope_inv_quant_q8 (inverse rope on 448..511 + Q8_0 quant)  -> per-head epilogue
//
// BIT-IDENTICAL to that chain (heads, xq, xscale) for every row whose selection holds no sentinel
// inside its n_comp (tests/attn_dec_g3_bitexact.rs).
//
// What stays identical, and why:
//   * scores: per (head, key) the same 32-step WMMA chain over the 512 dims (q cast to f16 by the
//     same (_Float16) conversion -- once per WG into LDS instead of once per warp from global), the
//     same K bytes (the selected row itself instead of its gathered copy), the same kq_scale
//     multiply and f16 cast, written to the same unpadded LDS position.
//   * softmax: the production phase A verbatim (lane-strided max / exp / sum, xor trees).
//   * W.V: per (head, dim) the same 40-step WMMA chain over the same key tiles (raw tiles first,
//     the raw tail zero-padded, then the comp tiles), same A fragments (tile-major weights). Only the
//     B fragments are loaded differently: `global_load_tr_b128` transposes a 16 x 16 V block on the
//     way into the WMMA B layout, so V never goes through LDS (production staged every tile
//     TRANSPOSED into a warp-private LDS ring: a store -> load round trip per tile). A warp owns
//     dims [32 w, 32 w + 32) -- one Q8_0 block per head -- instead of [16 w, +16) and [16 (w+16), +16);
//     each output element's chain is unchanged.
//   * epilogue: out = c * (1 / sum) (the same f32 the chain stored), rope_tail's inverse rotation on
//     dims 448..511 (same theta / cosf / sinf expressions, computed once per pair by an idle warp
//     during phase S; the rotation's two contractions pinned to what rope_tail.hip compiles to),
//     and q8_0_quantize_f32_wave's block math (amax by fmaxf -- order-free --, d = amax / 127,
//     id = 1 / d, lrintf, clamp).
//
// Sentinel rows (selection < 0) inside n_comp: the chain reads whatever the gather buffer held from an
// earlier call (undefined); this kernel reads a ZERO key there (score 0, V 0). Consistent decode
// state never produces one (the top-K emits min(n_idx, 512) valid indices and n_comp is the same
// min()).
//
// Grid (n_head / 16, b) x 512 (one WG per 16-head tile of a row, like the production kernel).
// HARD LIMITS (the launcher enforces them; no device-side guard): n_comp <= 512 and
// n_raw + n_comp <= 640 per row, b <= 16, head_dim 512, n_head 64, n_rot 64.
//
// G3_DBG_STOP=0..4 (timing experiments ONLY, wrong output): return after the prologue / q + key
// table / phase S / phase A / phase B.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));
typedef short    short8 __attribute__((ext_vector_type(8)));

#define HD            512u
#define HT            16u
#define G3_MAX_KEYS   640u
#define G3_MAX_ROWS   16u
#define MAX_PER_LANE  (G3_MAX_KEYS / 32u)
#define WSTRIDE       664u
#define QSTRIDE       520u               // q_lds row stride (f16): 16-B skew per head row
#define TBL_N         672u               // key-table entries: n_tiles * 16 <= 640 + 2 partial tiles

#ifndef G3_PD
#define G3_PD 2
#endif

struct G3Meta {
    int n_raw[G3_MAX_ROWS];
    int raw_off[G3_MAX_ROWS];
    int n_comp[G3_MAX_ROWS];
};

#if defined(__gfx1200__) || defined(__gfx1201__)


// Lane exchange "value of lane (lane ^ m)" without the LDS round trip of __shfl_xor (ds_bpermute):
// DPP row_xmask for m < 16 (within a 16-lane row), v_permlanex16 for m == 16 (the other row, same
// position). Same pairing as __shfl_xor, so every reduction built on it is bit-identical.
#ifndef G3_DPP
#define G3_DPP 1
#endif
template <int M>
__device__ __forceinline__ float g3_xor(float v) {
#if G3_DPP
    const int x = __float_as_int(v);
    int r;
    if constexpr (M == 16) r = __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false);
    else r = __builtin_amdgcn_update_dpp(x, x, 0x160 + M, 0xf, 0xf, false);
    return __int_as_float(r);
#else
    return __shfl_xor(v, M);
#endif
}

// ---- phase A: production `phase_a_tm` verbatim (attention_dec_fused.hip) ----------------------
__device__ __forceinline__ void g3_phase_a(_Float16* wrow, unsigned wave, unsigned lane, unsigned n_raw,
                                           unsigned n_comp, unsigned n_tr, unsigned n_tiles,
                                           float sink_h, float* s_inv) {
    const unsigned n_total = n_raw + n_comp;
    float v[MAX_PER_LANE];
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) { const unsigned rr = lane + u * 32u; v[u] = (rr < n_total) ? (float)wrow[rr] : -INFINITY; }
    float local_max = sink_h;
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) if (v[u] > local_max) local_max = v[u];
    { float o;
      o = g3_xor<16>(local_max); if (o > local_max) local_max = o;
      o = g3_xor<8>(local_max);  if (o > local_max) local_max = o;
      o = g3_xor<4>(local_max);  if (o > local_max) local_max = o;
      o = g3_xor<2>(local_max);  if (o > local_max) local_max = o;
      o = g3_xor<1>(local_max);  if (o > local_max) local_max = o; }
    const float max_score = local_max;
    float local_sum = (lane == 0) ? expf(sink_h - max_score) : 0.0f;
    float w[MAX_PER_LANE];
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) { const unsigned rr = lane + u * 32u; w[u] = 0.0f; if (rr < n_total) { w[u] = expf(v[u] - max_score); local_sum += w[u]; } }
    __asm__ volatile("" ::: "memory");
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) {
        const unsigned rr = lane + u * 32u;
        if (rr < n_total) { const unsigned pos = (rr < n_raw) ? rr : n_tr * 16u + (rr - n_raw); wrow[pos] = (_Float16)w[u]; }
    }
    const unsigned raw_pad = n_tr * 16u - n_raw;
    if (lane < raw_pad) wrow[n_raw + lane] = (_Float16)0;
    const unsigned comp_end = n_tr * 16u + n_comp;
    const unsigned comp_pad = n_tiles * 16u - comp_end;
    if (lane < comp_pad) wrow[comp_end + lane] = (_Float16)0;
    local_sum += g3_xor<16>(local_sum); local_sum += g3_xor<8>(local_sum); local_sum += g3_xor<4>(local_sum);
    local_sum += g3_xor<2>(local_sum);  local_sum += g3_xor<1>(local_sum);
    if (lane == 0) s_inv[wave] = 1.0f / local_sum;
}

// ---- rope_tail.hip's rf_theta / rf_rot_pair arithmetic, split so theta is computed once per pair --
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
    // (in this kernel the default contraction picked the other operand for the odd lane: 1-ulp
    // differences on ~3% of the tail -- pinned explicitly, tests/attn_dec_g3_bitexact.rs).
    return (e & 1u) ? __builtin_fmaf(x1, c, x0 * s) : __builtin_fmaf(x0, c, -(x1 * s));
}

// ---- the kernel body ----------------------------------------------------------------------------
template <int PD>
__device__ __forceinline__ void g3_body(
        float* heads_out, int8_t* xq, float* xscale,
        const float* sinks, const float* q, const _Float16* raw_kv,
        const _Float16* comp_kv, const int* sel, unsigned sel_stride,
        const int* comp_base_per, unsigned comp_kv_batch_stride,
        const int* pos_per_b, float theta_scale, float freq_scale, float ext_factor,
        float mscale_eff, float corr_low, float corr_high, float kq_scale, unsigned n_head,
        const _Float16* zero_row, const G3Meta& meta, _Float16* w_lds, const _Float16** tbl, float* s_inv,
        _Float16* q_lds, float* rope_lds) {
    const unsigned hg = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const unsigned h0 = hg * HT;
    if (h0 >= n_head) return;
#if defined(G3_DBG_STOP) && G3_DBG_STOP == 0
    if (tid == 0) xscale[blockIdx.y * 1024u + hg] = 0.0f; return;
#endif
    const unsigned wave = tid >> 5, lane = tid & 31u;
    const unsigned n_raw = (unsigned)meta.n_raw[b];
    const unsigned raw_off = (unsigned)meta.raw_off[b];
    const unsigned n_comp_eff = comp_kv ? (unsigned)meta.n_comp[b] : 0u;
    const unsigned n_tr = (n_raw + 15u) / 16u;
    const unsigned n_tiles = n_tr + (n_comp_eff + 15u) / 16u;

    // ---- key table: entry t * 16 + k = the address of key k of tile t's 512-dim row -------------
    // Raw tiles: window row raw_off + key (key < n_raw). Comp tiles: comp key j's store row -- the
    // gather's `base + sel[j]` (sel set), or the dense store at `base + j` / `b * stride + j`. An
    // invalid key (past n_raw / n_comp in its tile, or a sentinel) points at `zero_row` (1 KB of
    // zeros): its K and V fragments load as zeros, exactly what the chain's guarded loads produce,
    // with no mask or branch in the loops. Phases S and B read only this table, so no global ->
    // address dependency sits inside their loops. The selection loads are issued before the q
    // loads so their latency hides under q's.
    const _Float16* ent[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        const unsigned i = tid + 512u * (unsigned)r;
        const unsigned t = i >> 4, k = i & 15u;
        const _Float16* e = zero_row;
        if (t < n_tr) {
            const unsigned key = t * 16u + k;
            if (key < n_raw) e = raw_kv + (size_t)(raw_off + key) * HD;
        } else {
            const unsigned j = (t - n_tr) * 16u + k;
            if (j < n_comp_eff) {
                const unsigned base = comp_base_per ? (unsigned)comp_base_per[b] : 0u;
                if (sel) { const int sv = sel[(size_t)b * sel_stride + j]; if (sv >= 0) e = comp_kv + (size_t)(base + (unsigned)sv) * HD; }
                else e = comp_kv + (size_t)(comp_base_per ? base + j : b * comp_kv_batch_stride + j) * HD;
            }
        }
        ent[r] = e;
    }

    // ---- q of the 16 heads cast to f16 ONCE per WG into LDS (each thread 16 floats) --------------
    // (the production kernel had all 16 warps load the same 32 KB of f32 q through L0); same cast.
    {
        const unsigned qh_i = tid >> 5, qd = (tid & 31u) * 16u;
        const float4* src = (const float4*)(q + ((size_t)b * n_head + h0 + qh_i) * HD + qd);
        const float4 x0 = src[0], x1 = src[1], x2 = src[2], x3 = src[3];
        half8 h0v, h1v;
        h0v[0] = (_Float16)x0.x; h0v[1] = (_Float16)x0.y; h0v[2] = (_Float16)x0.z; h0v[3] = (_Float16)x0.w;
        h0v[4] = (_Float16)x1.x; h0v[5] = (_Float16)x1.y; h0v[6] = (_Float16)x1.z; h0v[7] = (_Float16)x1.w;
        h1v[0] = (_Float16)x2.x; h1v[1] = (_Float16)x2.y; h1v[2] = (_Float16)x2.z; h1v[3] = (_Float16)x2.w;
        h1v[4] = (_Float16)x3.x; h1v[5] = (_Float16)x3.y; h1v[6] = (_Float16)x3.z; h1v[7] = (_Float16)x3.w;
        *(half8*)(q_lds + qh_i * QSTRIDE + qd) = h0v;
        *(half8*)(q_lds + qh_i * QSTRIDE + qd + 8u) = h1v;
    }
    tbl[tid] = ent[0];
    if (tid + 512u < TBL_N) tbl[tid + 512u] = ent[1];
    __syncthreads();
    const unsigned col = lane & 15u, kh = (lane >> 4) * 8u;
    half8 af[32];
    #pragma unroll
    for (unsigned u = 0; u < 32u; u++) af[u] = *(const half8*)(q_lds + col * QSTRIDE + u * 16u + kh);
#if defined(G3_DBG_STOP) && G3_DBG_STOP == 1
    { float acc = 0; for (int u = 0; u < 32; u++) acc += (float)af[u][0]; xscale[blockIdx.y * 1024u + tid] = acc + (float)tbl[tid & 255u][0]; return; }
#endif

    // ---- phase S, production structure: tiles wave, wave + 16, ...; per tile two half-tile groups
    // of 16 K fragments (64 VGPRs) -> 32-step WMMA chain -> scaled f16 scores into LDS ------------
    for (unsigned t = wave; t < n_tiles; t += 16u) {
        const _Float16* brow = tbl[t * 16u + col];
        const unsigned n_avail = t < n_tr ? min(16u, n_raw - t * 16u) : min(16u, n_comp_eff - (t - n_tr) * 16u);
        const unsigned woff = t < n_tr ? t * 16u : n_raw + (t - n_tr) * 16u;
        float8 c = (float8){0,0,0,0,0,0,0,0};
        #pragma unroll 1
        for (unsigned g = 0; g < 32u; g += 16u) {
            half8 bf[16];
            #pragma unroll
            for (unsigned u = 0; u < 16u; u++) {
                const unsigned dim0 = (g + u) * 16u + kh;
                bf[u] = *(const half8*)(brow + dim0);
            }
            if (g == 0) {
                #pragma unroll
                for (unsigned u = 0; u < 16u; u++) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[u], bf[u], c);
            } else {
                #pragma unroll
                for (unsigned u = 0; u < 16u; u++) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[16 + u], bf[u], c);
            }
        }
        if (col < n_avail) {
            #pragma unroll
            for (int e2 = 0; e2 < 8; e2++) w_lds[(kh + (unsigned)e2) * WSTRIDE + woff + col] = (_Float16)(c[e2] * kq_scale);
        }
    }
    // The 32 (cos, -sin) pairs of this row's rope tail, by the LAST warp (it has the fewest tiles:
    // 2 of 40) while the others finish phase S: rope_tail's per-element theta / cosf / sinf
    // expressions, once per pair instead of in the epilogue's critical path.
    if (wave == 15u) {
        const float th = g3_theta(pos_per_b[b], lane, theta_scale, freq_scale, ext_factor, corr_low, corr_high);
        rope_lds[lane] = cosf(th) * mscale_eff;
        rope_lds[32u + lane] = -1.0f * sinf(th) * mscale_eff;
    }
    __syncthreads();
#if defined(G3_DBG_STOP) && G3_DBG_STOP == 2
    xscale[blockIdx.y * 1024u + tid] = (float)w_lds[tid]; return;
#endif
    g3_phase_a(w_lds + wave * WSTRIDE, wave, lane, n_raw, n_comp_eff, n_tr, n_tiles, sinks[h0 + wave], s_inv);
    __syncthreads();
#if defined(G3_DBG_STOP) && G3_DBG_STOP == 3
    xscale[blockIdx.y * 1024u + tid] = (float)w_lds[tid] + s_inv[tid & 15u]; return;
#endif

    // ---- phase B: W.V with transposed global loads, PD tiles in flight ------------------------------
    // Lane L of the transposed load addresses key kk = 8 (L >> 4) + (L & 7) of the tile at dim offset
    // 8 ((L >> 3) & 1); after the load it holds dim (L & 15) of keys 8 (L >> 4) .. + 8 = the WMMA B
    // fragment. Warp w: dims [32 w, 32 w + 16) (lo) and [32 w + 16, 32 w + 32) (hi). Ring slot p holds
    // its tile's V fragments, A fragment and the table entry of the tile PD later, all issued PD
    // tiles ahead: no memory round trip sits between two WMMAs of the chain, and an invalid key is
    // the zero row (no mask).
    const unsigned arow = col;
    const _Float16* wr = w_lds + arow * WSTRIDE;
    const unsigned kk = (lane >> 4) * 8u + (lane & 7u);
    const unsigned dcol = wave * 32u + ((lane >> 3) & 1u) * 8u;
    auto issue = [&](const _Float16* ek, short8& lo, short8& hi) {
        const _Float16* p = ek + dcol;
        lo = __builtin_amdgcn_global_load_tr_b128_v8i16((short8 __attribute__((address_space(1)))*)p);
        hi = __builtin_amdgcn_global_load_tr_b128_v8i16((short8 __attribute__((address_space(1)))*)(p + 16));
    };
    float8 c_lo = {0,0,0,0,0,0,0,0}, c_hi = {0,0,0,0,0,0,0,0};
    short8 rlo[PD], rhi[PD]; const _Float16* enext[PD]; half8 af_b[PD];
    #pragma unroll
    for (int p = 0; p < PD; p++) {
        const unsigned t = (unsigned)p;
        if (t < n_tiles) {
            issue(tbl[t * 16u + kk], rlo[p], rhi[p]);
            af_b[p] = *(const half8*)(wr + t * 16u + kh);
            enext[p] = tbl[min(t + (unsigned)PD, TBL_N / 16u - 1u) * 16u + kk];
        }
    }
    for (unsigned blk = 0; blk < n_tiles; blk += (unsigned)PD) {
        #pragma unroll
        for (int p = 0; p < PD; p++) {
            const unsigned t = blk + (unsigned)p;
            if (t < n_tiles) {
                c_lo = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af_b[p], __builtin_bit_cast(half8, rlo[p]), c_lo);
                c_hi = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af_b[p], __builtin_bit_cast(half8, rhi[p]), c_hi);
                const unsigned tn = t + (unsigned)PD;
                if (tn < n_tiles) {
                    issue(enext[p], rlo[p], rhi[p]);
                    af_b[p] = *(const half8*)(wr + tn * 16u + kh);
                    enext[p] = tbl[min(tn + (unsigned)PD, TBL_N / 16u - 1u) * 16u + kk];
                }
            }
        }
    }
#if defined(G3_DBG_STOP) && G3_DBG_STOP == 4
    xscale[blockIdx.y * 1024u + tid] = c_lo[0] + c_hi[3]; return;
#endif

    // ---- epilogue: 1/sum, inverse rope (dims 448..511 = warps 14, 15), Q8_0 block per head ----------
    const unsigned d_lo = wave * 32u + arow, d_hi = d_lo + 16u;
    float v_lo[8], v_hi[8];
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const float sc = s_inv[kh + (unsigned)e];
        v_lo[e] = c_lo[e] * sc;
        v_hi[e] = c_hi[e] * sc;
    }
    if (wave >= 14u) {   // wave-uniform: the rope tail
        const float c_l = rope_lds[(d_lo - 448u) >> 1], s_l = rope_lds[32u + ((d_lo - 448u) >> 1)];
        const float c_h = rope_lds[(d_hi - 448u) >> 1], s_h = rope_lds[32u + ((d_hi - 448u) >> 1)];
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const float p_lo = g3_xor<1>(v_lo[e]), p_hi = g3_xor<1>(v_hi[e]);
            v_lo[e] = g3_rot(v_lo[e], p_lo, d_lo, c_l, s_l);
            v_hi[e] = g3_rot(v_hi[e], p_hi, d_hi, c_h, s_h);
        }
    }
    const size_t row = (size_t)b * 32768u;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned head = h0 + kh + (unsigned)e;
        float amax = fmaxf(fabsf(v_lo[e]), fabsf(v_hi[e]));
        amax = fmaxf(amax, g3_xor<8>(amax));
        amax = fmaxf(amax, g3_xor<4>(amax));
        amax = fmaxf(amax, g3_xor<2>(amax));
        amax = fmaxf(amax, g3_xor<1>(amax));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        int q_lo = (int)lrintf(v_lo[e] * id), q_hi = (int)lrintf(v_hi[e] * id);
        q_lo = q_lo > 127 ? 127 : (q_lo < -128 ? -128 : q_lo);
        q_hi = q_hi > 127 ? 127 : (q_hi < -128 ? -128 : q_hi);
        const size_t hbase = row + (size_t)head * 512u;
        xq[hbase + d_lo] = (int8_t)q_lo;
        xq[hbase + d_hi] = (int8_t)q_hi;
        if (arow == 0u) xscale[(size_t)b * 1024u + head * 16u + wave] = d;
        if (heads_out) { heads_out[hbase + d_lo] = v_lo[e]; heads_out[hbase + d_hi] = v_hi[e]; }
    }
}

#endif  // gfx12

// The symbol exists on every arch (the module must load on gfx1151); the body is gfx12 only.
extern "C" __global__ __launch_bounds__(512) void attn_dec_g3(
        float          * heads_out,         // nullable [B, 64, 512] f32: the chain's final `heads`
        int8_t         * xq,                // [B, 32768] Q8_0 quants
        float          * xscale,            // [B, 1024] Q8_0 scales
        const float    * sinks,             // [n_head]
        const float    * q,                 // [B, n_head, 512]
        const _Float16 * raw_kv,            // window cache (rows raw_off + i)
        const _Float16 * comp_kv,           // nullable: the compressed store (ratio-0 layers: null)
        const int      * sel,               // nullable [B, sel_stride]: top-K selection (null = dense)
        unsigned int     sel_stride,
        const int      * comp_base_per,     // nullable [B]: per-row comp store base ROW
        unsigned int     comp_kv_batch_stride,
        const int      * pos_per_b,         // [B]
        float theta_scale, float freq_scale, float ext_factor, float mscale_eff, float corr_low,
        float corr_high, float kq_scale, unsigned int n_head,
        const _Float16 * zero_row,          // >= 1 KB of zeros, 16-B aligned: the invalid keys' row
        G3Meta meta) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    __shared__ __attribute__((aligned(16))) _Float16 w_lds[HT * WSTRIDE];
    __shared__ __attribute__((aligned(16))) _Float16 q_lds[HT * QSTRIDE];
    __shared__ const _Float16* tbl[TBL_N];
    __shared__ float s_inv[HT];
    __shared__ float rope_lds[64];
    g3_body<G3_PD>(heads_out, xq, xscale, sinks, q, raw_kv, comp_kv, sel, sel_stride, comp_base_per,
                   comp_kv_batch_stride, pos_per_b, theta_scale, freq_scale, ext_factor, mscale_eff,
                   corr_low, corr_high, kq_scale, n_head, zero_row, meta, w_lds, tbl, s_inv, q_lds, rope_lds);
#else
    (void)heads_out; (void)xq; (void)xscale; (void)sinks; (void)q; (void)raw_kv; (void)comp_kv; (void)sel;
    (void)sel_stride; (void)comp_base_per; (void)comp_kv_batch_stride; (void)pos_per_b; (void)theta_scale;
    (void)freq_scale; (void)ext_factor; (void)mscale_eff; (void)corr_low; (void)corr_high; (void)kq_scale;
    (void)n_head; (void)zero_row; (void)meta;
#endif
}

