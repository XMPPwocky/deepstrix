// D_attention candidates, attempt 2c (gfx1201).
//
//  attention_dec_fused_qlds_pf<PF>r<RP>   BIT-EXACT fused pair, ONE launch, grid (4, b) x 512:
//      phase S like cand2's fused_direct (q staged once per WG as f16 in LDS, production chain and
//      rounding, 16 K-chunk loads in flight), phase A production order, phase B = attempt-1's pf2
//      LDS-V pipeline (PF tiles per LDS group, RP groups prefetched into registers, weights in LDS).
//      Combines the two halves that each measured well (fused_direct's S, pf2's B).
//  attention_dec_flash3_part1_k64/k128 + attention_dec_flash3_combine   flash split-K, v3:
//      ALL 16 warps score: NT key tiles x DS dim-slices (DS = 16/NT), partial accumulators reduced
//      through LDS in f32 (scores no longer bit-exact with production's chain: f32 re-association
//      only); V fragments: one 64-bit base per tile per lane + immediate offsets on the common
//      (non-tail, non-straddling) path. NOT bit-exact (as flash2).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define HD            512u
#define HT            16u
#define DEC_MAX_KEYS  640u
#define MAX_PER_LANE  (DEC_MAX_KEYS / 32u)
#define WSTRIDE       648u
#define QSTRIDE       520u
#define VSTRIDE       512u

#if defined(__gfx1200__) || defined(__gfx1201__)

struct TileDesc {
    const _Float16* base;
    unsigned row0;
    unsigned n_avail;
    unsigned woff;
};
__device__ __forceinline__ TileDesc tile_desc(unsigned t, unsigned n_tr, const _Float16* raw_kv,
                                              unsigned raw_off, unsigned n_raw,
                                              const _Float16* comp_kv_b, unsigned n_comp) {
    TileDesc d;
    if (t < n_tr) {
        d.base = raw_kv; d.row0 = raw_off + t * 16u;
        d.n_avail = min(16u, n_raw - t * 16u); d.woff = t * 16u;
    } else {
        const unsigned tc = t - n_tr;
        d.base = comp_kv_b; d.row0 = tc * 16u;
        d.n_avail = min(16u, n_comp - tc * 16u); d.woff = n_raw + tc * 16u;
    }
    return d;
}

__device__ __forceinline__ void stage_q_lds(const float* q, unsigned b, unsigned n_head, unsigned h0,
                                            unsigned tid, _Float16* q_lds) {
    const float* qb = q + ((size_t)b * n_head + h0) * HD;
    float4 v[4];
    #pragma unroll
    for (int it = 0; it < 4; it++) {
        const unsigned e = tid + (unsigned)it * 512u;
        v[it] = ((const float4*)(qb + (e >> 7) * HD))[e & 127u];
    }
    #pragma unroll
    for (int it = 0; it < 4; it++) {
        const unsigned e = tid + (unsigned)it * 512u;
        _Float16* dst = q_lds + (e >> 7) * QSTRIDE + (e & 127u) * 4u;
        dst[0] = (_Float16)v[it].x; dst[1] = (_Float16)v[it].y; dst[2] = (_Float16)v[it].z; dst[3] = (_Float16)v[it].w;
    }
}

// production score chain for one 16-key tile: A from f16 q row in LDS, B = key row (nullptr = zero)
template <int GROUP>
__device__ __forceinline__ float8 score_chain_lds_q(const _Float16* qrow, const _Float16* brow, unsigned kh,
                                                    unsigned c0 = 0u, unsigned c1 = 32u) {
    float8 c = (float8){0,0,0,0,0,0,0,0};
    #pragma unroll 1
    for (unsigned g = c0; g < c1; g += (unsigned)GROUP) {
        half8 bf[GROUP];
        #pragma unroll
        for (int u = 0; u < GROUP; u++) {
            const unsigned dim0 = (g + (unsigned)u) * 16u + kh;
            half8 tt = {0,0,0,0,0,0,0,0};
            if (brow) {
                #pragma unroll
                for (int e = 0; e < 8; e++) tt[e] = brow[dim0 + (unsigned)e];
            }
            bf[u] = tt;
        }
        #pragma unroll
        for (int u = 0; u < GROUP; u++) {
            const unsigned dim0 = (g + (unsigned)u) * 16u + kh;
            const half8 af = *(const half8*)(qrow + dim0);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf[u], c);
        }
    }
    return c;
}

__device__ __forceinline__ void phase_a(const _Float16* src, _Float16* wrow, unsigned wave, unsigned lane,
                                        unsigned n_total, float sink_h, float* s_inv) {
    float v[MAX_PER_LANE];
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) {
        const unsigned rr = lane + u * 32u;
        v[u] = (rr < n_total) ? (float)src[rr] : -INFINITY;
    }
    float local_max = sink_h;
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) if (v[u] > local_max) local_max = v[u];
    {
        float o;
        o = __shfl_xor(local_max, 16); if (o > local_max) local_max = o;
        o = __shfl_xor(local_max,  8); if (o > local_max) local_max = o;
        o = __shfl_xor(local_max,  4); if (o > local_max) local_max = o;
        o = __shfl_xor(local_max,  2); if (o > local_max) local_max = o;
        o = __shfl_xor(local_max,  1); if (o > local_max) local_max = o;
    }
    const float max_score = local_max;
    float local_sum = (lane == 0) ? expf(sink_h - max_score) : 0.0f;
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) {
        const unsigned rr = lane + u * 32u;
        if (rr < n_total) {
            const float w = expf(v[u] - max_score);
            wrow[rr] = (_Float16)w;
            local_sum += w;
        }
    }
    local_sum += __shfl_xor(local_sum, 16);
    local_sum += __shfl_xor(local_sum,  8);
    local_sum += __shfl_xor(local_sum,  4);
    local_sum += __shfl_xor(local_sum,  2);
    local_sum += __shfl_xor(local_sum,  1);
    if (lane == 0) s_inv[wave] = 1.0f / local_sum;
}

// attempt-1 pf2 phase B (LDS-V ring, register prefetch), unchanged numerics
__device__ __forceinline__ void load_tile_regs(const TileDesc& d, unsigned tid, ulong2 (&r)[2]) {
    #pragma unroll
    for (unsigned it = 0; it < 2u; it++) {
        const unsigned s = tid + it * 512u;
        const unsigned k_idx = s >> 6, d_chunk = s & 63u;
        if (k_idx < d.n_avail) r[it] = *(const ulong2*)(d.base + (size_t)(d.row0 + k_idx) * HD + d_chunk * 8u);
        else { r[it].x = 0; r[it].y = 0; }
    }
}
__device__ __forceinline__ void store_tile_lds(_Float16* v_lds, unsigned tid, const ulong2 (&r)[2]) {
    #pragma unroll
    for (unsigned it = 0; it < 2u; it++) {
        const unsigned s = tid + it * 512u;
        const unsigned k_idx = s >> 6, d_chunk = s & 63u;
        *(ulong2*)(v_lds + k_idx * VSTRIDE + d_chunk * 8u) = r[it];
    }
}
template <int PF, int RP>
__device__ __forceinline__ void phase_b(_Float16* v_lds, const _Float16* w_lds, unsigned tid,
                                        unsigned n_tiles, unsigned n_tr, const _Float16* raw_kv,
                                        unsigned raw_off, unsigned n_raw, const _Float16* comp_kv_b,
                                        unsigned n_comp, float8& c_a, float8& c_b) {
    const unsigned lane = tid & 31u, warp = tid >> 5;
    const unsigned arow = lane & 15u, kh = (lane >> 4) * 8u;
    const unsigned d_a = warp * 16u + arow, d_b = (warp + 16u) * 16u + arow;
    const unsigned n_groups = (n_tiles + PF - 1) / PF;
    ulong2 rg[RP][PF][2];
    #pragma unroll
    for (int i = 0; i < RP; i++) {
        #pragma unroll
        for (int p = 0; p < PF; p++) {
            const unsigned t = (unsigned)i * PF + p;
            if (t < n_tiles) load_tile_regs(tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp), tid, rg[i][p]);
        }
    }
    for (unsigned g0 = 0; g0 < n_groups; g0 += RP) {
        #pragma unroll
        for (int i = 0; i < RP; i++) {
            const unsigned g = g0 + (unsigned)i;
            if (g >= n_groups) break;
            __syncthreads();
            #pragma unroll
            for (int p = 0; p < PF; p++) if (g * PF + p < n_tiles) store_tile_lds(v_lds + p * (16u * VSTRIDE), tid, rg[i][p]);
            const unsigned gn = g + RP;
            if (gn < n_groups) {
                #pragma unroll
                for (int p = 0; p < PF; p++) {
                    const unsigned t = gn * PF + p;
                    if (t < n_tiles) load_tile_regs(tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp), tid, rg[i][p]);
                }
            }
            __syncthreads();
            #pragma unroll
            for (int p = 0; p < PF; p++) {
                const unsigned t = g * PF + p;
                if (t >= n_tiles) break;
                const TileDesc d = tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
                const _Float16* vt = v_lds + p * (16u * VSTRIDE);
                const _Float16* wr = w_lds + arow * WSTRIDE + d.woff;
                half8 a, ba, bb;
                if (d.n_avail == 16u) {
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        const unsigned k_idx = kh + (unsigned)e;
                        a[e] = wr[k_idx]; ba[e] = vt[k_idx * VSTRIDE + d_a]; bb[e] = vt[k_idx * VSTRIDE + d_b];
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        const unsigned k_idx = kh + (unsigned)e;
                        if (k_idx < d.n_avail) { a[e] = wr[k_idx]; ba[e] = vt[k_idx * VSTRIDE + d_a]; bb[e] = vt[k_idx * VSTRIDE + d_b]; }
                        else { a[e] = 0; ba[e] = 0; bb[e] = 0; }
                    }
                }
                c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
                c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
            }
        }
    }
}
__device__ __forceinline__ void store_out(float* out, unsigned b, unsigned n_head, unsigned h0, unsigned tid,
                                          const float* s_inv, const float8& c_a, const float8& c_b) {
    const unsigned lane = tid & 31u, warp = tid >> 5, arow = lane & 15u;
    const unsigned d_a = warp * 16u + arow, d_b = (warp + 16u) * 16u + arow;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned mh = (lane >> 4) * 8u + (unsigned)e;
        const float scale = s_inv[mh];
        const size_t obase = ((size_t)b * n_head + h0 + mh) * HD;
        out[obase + d_a] = c_a[e] * scale;
        out[obase + d_b] = c_b[e] * scale;
    }
}

// ---------------------------------------------------------------------------------------------
template <int PF, int RP>
__device__ __forceinline__ void fused_qlds_body(
        float* out, const float* sinks, const float* q, const _Float16* raw_kv,
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,
        const int* n_comp_per, unsigned n_head, float kq_scale, unsigned comp_kv_batch_stride,
        const int* comp_base_per, _Float16* w_lds, _Float16* q_lds, _Float16* v_lds, float* s_inv) {
    const unsigned hg = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const unsigned h0 = hg * HT;
    if (h0 >= n_head) return;
    const unsigned n_raw = (unsigned)n_raw_per[b];
    const unsigned raw_off = (unsigned)n_raw_offset_per[b];
    const unsigned n_comp = (unsigned)n_comp_per[b];
    const unsigned n_total = n_raw + n_comp;
    const _Float16* comp_kv_b = (comp_kv && comp_base_per)
        ? comp_kv + (size_t)(unsigned)comp_base_per[b] * HD
        : (comp_kv && comp_kv_batch_stride != 0u) ? comp_kv + (size_t)b * comp_kv_batch_stride * HD : comp_kv;
    const unsigned wave = tid >> 5, lane = tid & 31u;
    const unsigned n_tr = (n_raw + 15u) / 16u;
    const unsigned n_tiles = n_tr + ((comp_kv_b ? n_comp : 0u) + 15u) / 16u;

    stage_q_lds(q, b, n_head, h0, tid, q_lds);
    __syncthreads();
    {
        const unsigned col = lane & 15u, kh = (lane >> 4) * 8u;
        const _Float16* qrow = q_lds + col * QSTRIDE;
        for (unsigned t = wave; t < n_tiles; t += 16u) {
            const TileDesc d = tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
            const _Float16* brow = (col < d.n_avail) ? d.base + (size_t)(d.row0 + col) * HD : nullptr;
            const float8 c = score_chain_lds_q<16>(qrow, brow, kh);
            if (col < d.n_avail) {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    w_lds[(kh + (unsigned)e) * WSTRIDE + d.woff + col] = (_Float16)(c[e] * kq_scale);
            }
        }
    }
    __syncthreads();
    phase_a(w_lds + wave * WSTRIDE, w_lds + wave * WSTRIDE, wave, lane, n_total, sinks[h0 + wave], s_inv);
    // (phase_b's first __syncthreads publishes the weights)
    float8 c_a = {0,0,0,0,0,0,0,0}, c_b = {0,0,0,0,0,0,0,0};
    phase_b<PF, RP>(v_lds, w_lds, tid, n_tiles, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp, c_a, c_b);
    __syncthreads();   // s_inv visible (phase_a wrote it before phase_b's barriers; keep explicit)
    store_out(out, b, n_head, h0, tid, s_inv, c_a, c_b);
}
#define DEC_FUSED_QLDS(NAME, PF, RP)                                                           \
extern "C" __global__ __launch_bounds__(512) void NAME(                                        \
        float* out, const float* sinks, const float* q, const _Float16* raw_kv,                \
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,            \
        const int* n_comp_per, unsigned n_head, float kq_scale, unsigned comp_kv_batch_stride,  \
        const int* comp_base_per) {                                                            \
    __shared__ _Float16 w_lds[HT * WSTRIDE];                                                   \
    /* q (phase S only) and the V ring (phase B only) alias: q is dead after the post-S barrier */ \
    __shared__ _Float16 qv_lds[(HT * QSTRIDE > PF * 16u * VSTRIDE) ? HT * QSTRIDE : PF * 16u * VSTRIDE]; \
    __shared__ float s_inv[HT];                                                                \
    fused_qlds_body<PF, RP>(out, sinks, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per,       \
                            n_comp_per, n_head, kq_scale, comp_kv_batch_stride, comp_base_per, \
                            w_lds, qv_lds, qv_lds, s_inv);                                     \
}
DEC_FUSED_QLDS(attention_dec_fused_qlds_pf2r2, 2, 2)
DEC_FUSED_QLDS(attention_dec_fused_qlds_pf2r4, 2, 4)
DEC_FUSED_QLDS(attention_dec_fused_qlds_pf1r4, 1, 4)
DEC_FUSED_QLDS(attention_dec_fused_qlds_pf1r8, 1, 8)

// ---------------------------------------------------------------------------------------------
// flash3: grid (n_head/16, S, b) x 512, KPS keys per WG, all 16 warps score (NT tiles x DS slices).
template <int KPS>
__device__ __forceinline__ void flash3_part1_body(
        float* part_o, float* part_ml, const float* q, const _Float16* raw_kv,
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,
        const int* n_comp_per, const float* sinks, unsigned n_head, float kq_scale,
        unsigned comp_kv_batch_stride, const int* comp_base_per, unsigned max_splits,
        _Float16* q_lds, _Float16* w_lds, float* red_lds, float* s_m, float* s_l) {
    constexpr int NT = KPS / 16;
    constexpr int DS = 16 / NT;              // dim slices per tile
    constexpr unsigned CPS = 32u / (unsigned)DS;  // K-chunks per slice
    constexpr unsigned WS = (unsigned)KPS + 8u;
    const unsigned hg = blockIdx.x, s = blockIdx.y, b = blockIdx.z, tid = threadIdx.x;
    const unsigned h0 = hg * HT;
    if (h0 >= n_head) return;
    const unsigned n_raw = (unsigned)n_raw_per[b];
    const unsigned raw_off = (unsigned)n_raw_offset_per[b];
    const unsigned n_comp = (unsigned)n_comp_per[b];
    const unsigned n_total = n_raw + n_comp;
    const unsigned key0 = s * (unsigned)KPS;
    if (key0 >= n_total) return;
    const unsigned nk = min((unsigned)KPS, n_total - key0);
    const size_t comp_kv_b_off = comp_base_per ? (size_t)(unsigned)comp_base_per[b] : (size_t)b * comp_kv_batch_stride;
    const unsigned wave = tid >> 5, lane = tid & 31u;
    const unsigned arow = lane & 15u, kh = (lane >> 4) * 8u;
    const unsigned d_a = wave * 16u + arow, d_b = (wave + 16u) * 16u + arow;
    auto key_row = [&](unsigned key) -> const _Float16* {
        return (key < n_raw) ? raw_kv + (size_t)(key + raw_off) * HD
                             : comp_kv + (comp_kv_b_off + (key - n_raw)) * HD;
    };

    // ---- q loads
    float4 qv[4];
    {
        const float* qb = q + ((size_t)b * n_head + h0) * HD;
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            const unsigned e = tid + (unsigned)it * 512u;
            qv[it] = ((const float4*)(qb + (e >> 7) * HD))[e & 127u];
        }
    }
    // ---- V fragments
    half8 va[NT], vb[NT];
    const bool simple = (nk == (unsigned)KPS) && (key0 >= n_raw || key0 + (unsigned)KPS <= n_raw);
    if (simple) {
        #pragma unroll
        for (int p = 0; p < NT; p++) {
            const _Float16* base = key_row(key0 + (unsigned)p * 16u + kh);
            #pragma unroll
            for (int e = 0; e < 8; e++) { va[p][e] = base[(unsigned)e * HD + d_a]; vb[p][e] = base[(unsigned)e * HD + d_b]; }
        }
    } else {
        #pragma unroll
        for (int p = 0; p < NT; p++) {
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned kl = (unsigned)p * 16u + kh + (unsigned)e;
                const _Float16* row = key_row(key0 + min(kl, nk - 1u));
                const _Float16 x = row[d_a], y = row[d_b];
                va[p][e] = (kl < nk) ? x : (_Float16)0;
                vb[p][e] = (kl < nk) ? y : (_Float16)0;
            }
        }
    }
    // ---- q -> LDS
    #pragma unroll
    for (int it = 0; it < 4; it++) {
        const unsigned e = tid + (unsigned)it * 512u;
        _Float16* dst = q_lds + (e >> 7) * QSTRIDE + (e & 127u) * 4u;
        dst[0] = (_Float16)qv[it].x; dst[1] = (_Float16)qv[it].y; dst[2] = (_Float16)qv[it].z; dst[3] = (_Float16)qv[it].w;
    }
    __syncthreads();
    // ---- score: warp = tile p (wave % NT) x slice ds (wave / NT)
    {
        const unsigned p = wave % (unsigned)NT, ds = wave / (unsigned)NT;
        const unsigned col = lane & 15u;
        const unsigned kl = p * 16u + col;
        const _Float16* brow = (kl < nk) ? key_row(key0 + kl) : nullptr;
        const _Float16* qrow = q_lds + col * QSTRIDE;
        const float8 c = score_chain_lds_q<(CPS < 16u ? (int)CPS : 16)>(qrow, brow, kh, ds * CPS, (ds + 1u) * CPS);
        if (ds == 0) {
            // slice 0 keeps its partial in registers; others publish to LDS
        }
        if (ds != 0) {
            float* r = red_lds + ((ds - 1u) * (unsigned)NT + p) * 256u;   // [16 x 16] f32 per (slice, tile)
            #pragma unroll
            for (int e = 0; e < 8; e++) r[(kh + (unsigned)e) * 16u + col] = c[e];
        }
        __syncthreads();
        if (ds == 0) {
            float8 t = c;
            #pragma unroll
            for (int d2 = 1; d2 < DS; d2++) {
                const float* r = red_lds + (((unsigned)d2 - 1u) * (unsigned)NT + p) * 256u;
                #pragma unroll
                for (int e = 0; e < 8; e++) t[e] += r[(kh + (unsigned)e) * 16u + col];
            }
            if (kl < nk) {
                #pragma unroll
                for (int e = 0; e < 8; e++) w_lds[(kh + (unsigned)e) * WS + kl] = (_Float16)(t[e] * kq_scale);
            }
        }
    }
    __syncthreads();
    // ---- local softmax
    {
        constexpr int PER = KPS / 32;
        _Float16* wrow = w_lds + wave * WS;
        const float sink_h = sinks[h0 + wave];
        float v[PER];
        #pragma unroll
        for (int u = 0; u < PER; u++) { const unsigned rr = lane + (unsigned)u * 32u; v[u] = (rr < nk) ? (float)wrow[rr] : -INFINITY; }
        float m = sink_h;
        #pragma unroll
        for (int u = 0; u < PER; u++) if (v[u] > m) m = v[u];
        { float o;
          o = __shfl_xor(m, 16); if (o > m) m = o;  o = __shfl_xor(m, 8); if (o > m) m = o;
          o = __shfl_xor(m, 4);  if (o > m) m = o;  o = __shfl_xor(m, 2); if (o > m) m = o;
          o = __shfl_xor(m, 1);  if (o > m) m = o; }
        float l = 0.0f;
        #pragma unroll
        for (int u = 0; u < PER; u++) {
            const unsigned rr = lane + (unsigned)u * 32u;
            if (rr < nk) { const float w = expf(v[u] - m); wrow[rr] = (_Float16)w; l += w; }
        }
        l += __shfl_xor(l, 16); l += __shfl_xor(l, 8); l += __shfl_xor(l, 4); l += __shfl_xor(l, 2); l += __shfl_xor(l, 1);
        if (lane == 0) { s_m[wave] = m; s_l[wave] = l; }
    }
    __syncthreads();
    // ---- W.V
    float8 c_a = {0,0,0,0,0,0,0,0}, c_b = {0,0,0,0,0,0,0,0};
    const _Float16* wr = w_lds + arow * WS;
    #pragma unroll
    for (int p = 0; p < NT; p++) {
        if ((unsigned)p * 16u >= nk) break;
        half8 a;
        #pragma unroll
        for (int e = 0; e < 8; e++) { const unsigned kl = (unsigned)p * 16u + kh + (unsigned)e; a[e] = (kl < nk) ? wr[kl] : (_Float16)0; }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, va[p], c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, vb[p], c_b);
    }
    const size_t pbase = ((size_t)b * max_splits + s) * n_head + h0;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned mh = (lane >> 4) * 8u + (unsigned)e;
        float* o = part_o + (pbase + mh) * HD;
        o[d_a] = c_a[e]; o[d_b] = c_b[e];
    }
    if (wave == 0 && lane < HT) { part_ml[(pbase + lane) * 2u] = s_m[lane]; part_ml[(pbase + lane) * 2u + 1u] = s_l[lane]; }
}
#define DEC_FLASH3_PART1(NAME, KPS)                                                            \
extern "C" __global__ __launch_bounds__(512) void NAME(                                        \
        float* part_o, float* part_ml, const float* q, const _Float16* raw_kv,                 \
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,            \
        const int* n_comp_per, const float* sinks, unsigned n_head, float kq_scale,            \
        unsigned comp_kv_batch_stride, const int* comp_base_per, unsigned max_splits) {        \
    __shared__ _Float16 q_lds[HT * QSTRIDE];                                                   \
    __shared__ _Float16 w_lds[HT * ((KPS) + 8u)];                                              \
    __shared__ float red_lds[(16 / ((KPS) / 16) - 1) * ((KPS) / 16) * 256];                    \
    __shared__ float s_m[HT], s_l[HT];                                                         \
    flash3_part1_body<KPS>(part_o, part_ml, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per,   \
                           n_comp_per, sinks, n_head, kq_scale, comp_kv_batch_stride,          \
                           comp_base_per, max_splits, q_lds, w_lds, red_lds, s_m, s_l);        \
}
DEC_FLASH3_PART1(attention_dec_flash3_part1_k64, 64)
DEC_FLASH3_PART1(attention_dec_flash3_part1_k128, 128)

extern "C" __global__ __launch_bounds__(512) void attention_dec_flash3_combine(
        float* out, const float* part_o, const float* part_ml, const float* sinks,
        const int* n_raw_per, const int* n_comp_per, unsigned n_head, unsigned kps, unsigned max_splits) {
    const unsigned h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
    const unsigned n_total = (unsigned)n_raw_per[b] + (unsigned)n_comp_per[b];
    const unsigned S = (n_total + kps - 1u) / kps;
    const size_t pbase = (size_t)b * max_splits * n_head + h;
    float M = sinks[h];
    for (unsigned s = 0; s < S; s++) M = fmaxf(M, part_ml[(pbase + (size_t)s * n_head) * 2u]);
    float L = expf(sinks[h] - M), o = 0.0f;
    for (unsigned s = 0; s < S; s++) {
        const size_t pi = pbase + (size_t)s * n_head;
        const float f = expf(part_ml[pi * 2u] - M);
        L += part_ml[pi * 2u + 1u] * f;
        o += part_o[pi * HD + d] * f;
    }
    out[((size_t)b * n_head + h) * HD + d] = o / L;
}

#endif  // gfx12
