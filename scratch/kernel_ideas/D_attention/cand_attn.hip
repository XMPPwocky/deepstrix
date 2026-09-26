// D_attention candidates (gfx1201 only). Decode-shape twins of the production attention pair
//   attention_dec_score_htiled_wmma_f16s  +  attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s
// for n_total <= DEC_MAX_KEYS (640 = 128 window + 512 gathered) keys, b <= 16, 64 heads x 512.
//
// Every candidate keeps the production arithmetic order exactly (same f16 score rounding, same
// per-lane softmax order and shuffle tree, same expf, same per-tile WMMA chain on the same
// fragments), so the outputs are BIT-IDENTICAL to the production pair; only the schedule
// (what is in flight, where the weights live) changes. Kernels:
//
//   attention_dec_smwsum_pf2   smwsum twin: phase-A loads batched in registers, f16 weights kept
//                              in LDS (not written back to the scores buffer), V tiles through a
//                              2-slot LDS ring with register prefetch 2 groups (4 tiles) ahead.
//   attention_dec_smwsum_regpf same with a 1-slot ring (2 tiles ahead).
//   attention_dec_fused_f16s   score + softmax + wsum in ONE launch: grid (4, b) x 512, the scores
//                              never leave LDS.
//   attention_dec_score_blk256 production dec score with 8 warps per WG (128 keys per WG).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define HD            512u
#define HT            16u
#define DEC_MAX_KEYS  640u
#define MAX_PER_LANE  (DEC_MAX_KEYS / 32u)   // 20 scores per lane in phase A
#define WSTRIDE       648u                   // f16 per head row in the LDS weight tile (bank spread)
#define VSTRIDE       512u                   // f16 per key row in the LDS V tile (as production)

#if defined(__gfx1200__) || defined(__gfx1201__)

// ---------------------------------------------------------------------------------------------
// Tile descriptor: tile t of the concatenated (raw, comp) key list, 16 keys each.
struct TileDesc {
    const _Float16* base;   // kv rows
    unsigned row0;          // first row in `base`
    unsigned n_avail;       // 1..16 valid keys
    unsigned woff;          // first key index in the weights row
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

// One thread's share of a 16-key V tile: 2 x 16 B (as production's stage_v).
__device__ __forceinline__ void load_tile_regs(const TileDesc& d, unsigned tid, ulong2 (&r)[2]) {
    #pragma unroll
    for (unsigned it = 0; it < 2u; it++) {
        const unsigned s = tid + it * 512u;
        const unsigned k_idx = s >> 6;          // / 64
        const unsigned d_chunk = s & 63u;
        if (k_idx < d.n_avail) {
            r[it] = *(const ulong2*)(d.base + (size_t)(d.row0 + k_idx) * HD + d_chunk * 8u);
        } else {
            r[it].x = 0; r[it].y = 0;
        }
    }
}
__device__ __forceinline__ void store_tile_lds(_Float16* v_lds, unsigned tid, const ulong2 (&r)[2]) {
    #pragma unroll
    for (unsigned it = 0; it < 2u; it++) {
        const unsigned s = tid + it * 512u;
        const unsigned k_idx = s >> 6;
        const unsigned d_chunk = s & 63u;
        *(ulong2*)(v_lds + k_idx * VSTRIDE + d_chunk * 8u) = r[it];
    }
}

// Phase A (production order): one wave per head, scores as f16 from `src` (global or LDS, stride
// `sstride`), computes in f32, writes f16 weights to `w_lds` row `wave` (stride WSTRIDE) and
// s_inv[wave]. Loads batched: all <= 20 per lane issued before the first use.
template <typename SrcT>
__device__ __forceinline__ void phase_a(const SrcT* src, unsigned wave, unsigned lane, unsigned n_total,
                                        float sink_h, _Float16* w_lds, float* s_inv) {
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
    _Float16* wrow = w_lds + wave * WSTRIDE;
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

// Phase B: pipelined W.V over all tiles. PF = tiles per LDS group, RP = groups prefetched in
// registers. Every tile's WMMA pair is the production one (a = 8 weights of head `arow`,
// ba/bb = V columns d_a/d_b of the tile's 16 keys, zeros beyond n_avail).
template <int PF, int RP>
__device__ __forceinline__ void phase_b(_Float16* v_lds, const _Float16* w_lds, unsigned tid,
                                        unsigned n_tiles, unsigned n_tr, const _Float16* raw_kv,
                                        unsigned raw_off, unsigned n_raw, const _Float16* comp_kv_b,
                                        unsigned n_comp, float8& c_a, float8& c_b) {
    const unsigned lane = tid & 31u, warp = tid >> 5;
    const unsigned arow = lane & 15u;
    const unsigned kh   = (lane >> 4) * 8u;
    const unsigned d_a  = warp * 16u + arow;
    const unsigned d_b  = (warp + 16u) * 16u + arow;
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
            __syncthreads();                       // group g-1's LDS reads are done
            #pragma unroll
            for (int p = 0; p < PF; p++) {
                if (g * PF + p < n_tiles) store_tile_lds(v_lds + p * (16u * VSTRIDE), tid, rg[i][p]);
            }
            const unsigned gn = g + RP;            // refill this register slot from RP groups ahead
            if (gn < n_groups) {
                #pragma unroll
                for (int p = 0; p < PF; p++) {
                    const unsigned t = gn * PF + p;
                    if (t < n_tiles) load_tile_regs(tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp), tid, rg[i][p]);
                }
            }
            __syncthreads();                       // group g visible in LDS
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
                        a[e]  = wr[k_idx];
                        ba[e] = vt[k_idx * VSTRIDE + d_a];
                        bb[e] = vt[k_idx * VSTRIDE + d_b];
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        const unsigned k_idx = kh + (unsigned)e;
                        if (k_idx < d.n_avail) {
                            a[e]  = wr[k_idx];
                            ba[e] = vt[k_idx * VSTRIDE + d_a];
                            bb[e] = vt[k_idx * VSTRIDE + d_b];
                        } else {
                            a[e] = 0; ba[e] = 0; bb[e] = 0;
                        }
                    }
                }
                c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
                c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
            }
        }
    }
}

__device__ __forceinline__ void store_out(float* out, unsigned b, unsigned n_head, unsigned h0,
                                          unsigned ht, unsigned tid, const float* s_inv,
                                          const float8& c_a, const float8& c_b) {
    const unsigned lane = tid & 31u, warp = tid >> 5;
    const unsigned arow = lane & 15u;
    const unsigned d_a  = warp * 16u + arow;
    const unsigned d_b  = (warp + 16u) * 16u + arow;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned mh = (lane >> 4) * 8u + (unsigned)e;
        if (mh < ht) {
            const float scale = s_inv[mh];
            const size_t obase = ((size_t)b * n_head + h0 + mh) * HD;
            out[obase + d_a] = c_a[e] * scale;
            out[obase + d_b] = c_b[e] * scale;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// smwsum decode twin. Same signature as the production smwsum kernel (head_dim must be 512).
template <int PF, int RP>
__device__ __forceinline__ void dec_smwsum_body(
        float* out, _Float16* scores_g, const float* sinks, const _Float16* raw_kv,
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,
        const int* n_comp_per, unsigned n_head, unsigned max_keys, unsigned comp_kv_batch_stride,
        const int* comp_base_per, _Float16* w_lds, _Float16* v_lds, float* s_inv) {
    const unsigned hg = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const unsigned h0 = hg * HT;
    if (h0 >= n_head) return;
    unsigned ht = HT;
    if (h0 + ht > n_head) ht = n_head - h0;
    const unsigned n_raw = (unsigned)n_raw_per[b];
    const unsigned raw_off = (unsigned)n_raw_offset_per[b];
    const unsigned n_comp = (unsigned)n_comp_per[b];
    const unsigned n_total = n_raw + n_comp;
    const _Float16* comp_kv_b = (comp_kv && comp_base_per)
        ? comp_kv + (size_t)(unsigned)comp_base_per[b] * HD
        : (comp_kv && comp_kv_batch_stride != 0u) ? comp_kv + (size_t)b * comp_kv_batch_stride * HD : comp_kv;
    const unsigned wave = tid >> 5, lane = tid & 31u;
    if (wave < ht) {
        const _Float16* sw = scores_g + ((size_t)b * n_head + h0 + wave) * max_keys;
        phase_a(sw, wave, lane, n_total, sinks[h0 + wave], w_lds, s_inv);
    }
    __syncthreads();
    const unsigned n_tr = (n_raw + 15u) / 16u;
    const unsigned n_tiles = n_tr + ((comp_kv_b ? n_comp : 0u) + 15u) / 16u;
    float8 c_a = {0,0,0,0,0,0,0,0}, c_b = {0,0,0,0,0,0,0,0};
    phase_b<PF, RP>(v_lds, w_lds, tid, n_tiles, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp, c_a, c_b);
    store_out(out, b, n_head, h0, ht, tid, s_inv, c_a, c_b);
}

#define DEC_SMWSUM_KERNEL(NAME, PF, RP)                                                        \
extern "C" __global__ __launch_bounds__(512) void NAME(                                        \
        float* out, _Float16* scores_g, const float* sinks, const _Float16* raw_kv,             \
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,            \
        const int* n_comp_per, unsigned n_head, unsigned head_dim, unsigned max_keys,           \
        unsigned comp_kv_batch_stride, const int* comp_base_per) {                             \
    __shared__ _Float16 w_lds[HT * WSTRIDE];                                                   \
    __shared__ _Float16 v_lds[PF * 16u * VSTRIDE];                                             \
    __shared__ float s_inv[HT];                                                                \
    (void)head_dim;                                                                            \
    dec_smwsum_body<PF, RP>(out, scores_g, sinks, raw_kv, comp_kv, n_raw_per, n_raw_offset_per, \
                            n_comp_per, n_head, max_keys, comp_kv_batch_stride, comp_base_per, \
                            w_lds, v_lds, s_inv);                                              \
}
DEC_SMWSUM_KERNEL(attention_dec_smwsum_pf2, 2, 2)
DEC_SMWSUM_KERNEL(attention_dec_smwsum_regpf, 1, 2)

// ---------------------------------------------------------------------------------------------
// Fused decode attention: grid (n_head/16, b) x 512. Scores -> LDS (f16, production rounding),
// softmax in place, W.V pipelined. No scores buffer, no mask (decode passes none).
template <int PF, int RP>
__device__ __forceinline__ void dec_fused_body(
        float* out, const float* sinks, const float* q, const _Float16* raw_kv,
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,
        const int* n_comp_per, unsigned n_head, float kq_scale, unsigned comp_kv_batch_stride,
        const int* comp_base_per, _Float16* w_lds, _Float16* v_lds, float* s_inv) {
    const unsigned hg = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const unsigned h0 = hg * HT;
    if (h0 >= n_head) return;
    unsigned ht = HT;
    if (h0 + ht > n_head) ht = n_head - h0;
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

    // ---- phase S: scores, warp `wave` takes tiles wave, wave+16, ... (production dec-score chain)
    {
        const unsigned col = lane & 15u;
        const unsigned kh  = (lane >> 4) * 8u;
        const float* qh = q + ((size_t)b * n_head + h0 + col) * HD;
        for (unsigned t = wave; t < n_tiles; t += 16u) {
            const TileDesc d = tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
            const unsigned bkey = d.woff + col;
            const _Float16* brow = (col < d.n_avail) ? d.base + (size_t)(d.row0 + col) * HD : nullptr;
            float8 c = (float8){0,0,0,0,0,0,0,0};
            #pragma unroll 1
            for (unsigned g = 0; g < 32u; g += 16u) {
                half8 bf[16], af[16];
                #pragma unroll
                for (unsigned u = 0; u < 16u; u++) {
                    const unsigned dim0 = (g + u) * 16u + kh;
                    half8 tt = {0,0,0,0,0,0,0,0};
                    if (brow) {
                        #pragma unroll
                        for (int e = 0; e < 8; e++) tt[e] = brow[dim0 + (unsigned)e];
                    }
                    bf[u] = tt;
                    half8 a;
                    #pragma unroll
                    for (int e = 0; e < 8; e++) a[e] = (_Float16)qh[dim0 + (unsigned)e];
                    af[u] = a;
                }
                #pragma unroll
                for (unsigned u = 0; u < 16u; u++) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[u], bf[u], c);
            }
            if (col < d.n_avail) {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const unsigned hl = kh + (unsigned)e;
                    if (hl < ht) w_lds[hl * WSTRIDE + bkey] = (_Float16)(c[e] * kq_scale);
                }
            }
        }
    }
    __syncthreads();
    // ---- phase A from LDS
    if (wave < ht) phase_a(w_lds + wave * WSTRIDE, wave, lane, n_total, sinks[h0 + wave], w_lds, s_inv);
    __syncthreads();
    float8 c_a = {0,0,0,0,0,0,0,0}, c_b = {0,0,0,0,0,0,0,0};
    phase_b<PF, RP>(v_lds, w_lds, tid, n_tiles, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp, c_a, c_b);
    store_out(out, b, n_head, h0, ht, tid, s_inv, c_a, c_b);
}

#define DEC_FUSED_KERNEL(NAME, PF, RP)                                                         \
extern "C" __global__ __launch_bounds__(512) void NAME(                                        \
        float* out, const float* sinks, const float* q, const _Float16* raw_kv,                \
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,            \
        const int* n_comp_per, unsigned n_head, float kq_scale, unsigned comp_kv_batch_stride,  \
        const int* comp_base_per) {                                                            \
    __shared__ _Float16 w_lds[HT * WSTRIDE];                                                   \
    __shared__ _Float16 v_lds[PF * 16u * VSTRIDE];                                             \
    __shared__ float s_inv[HT];                                                                \
    dec_fused_body<PF, RP>(out, sinks, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per,        \
                           n_comp_per, n_head, kq_scale, comp_kv_batch_stride, comp_base_per,  \
                           w_lds, v_lds, s_inv);                                               \
}
DEC_FUSED_KERNEL(attention_dec_fused_f16s, 2, 2)

// ---------------------------------------------------------------------------------------------
// Production dec score with 8 warps per WG (grid.x = ceil(n_total/128)). Same per-warp chain.
extern "C" __global__ __launch_bounds__(256) void attention_dec_score_blk256(
        _Float16* scores_g, const float* q, const _Float16* raw_kv, const _Float16* comp_kv,
        const int* n_raw_per, const int* n_raw_offset_per, const int* n_comp_per,
        const unsigned int* comp_allowed_bits, unsigned max_keys_words, unsigned n_head,
        unsigned max_keys, float kq_scale, unsigned comp_kv_batch_stride, const int* comp_base_per) {
    const unsigned b = blockIdx.z, mt = blockIdx.y, tid = threadIdx.x;
    const unsigned warp = tid >> 5, lane = tid & 31u;
    const unsigned n_raw = (unsigned)n_raw_per[b];
    const unsigned raw_off = (unsigned)n_raw_offset_per[b];
    const unsigned n_comp = (unsigned)n_comp_per[b];
    const unsigned n_total = n_raw + n_comp;
    const unsigned n0 = blockIdx.x * 128u + warp * 16u;
    if (n0 >= n_total) return;
    const unsigned col = lane & 15u;
    const unsigned kh = (lane >> 4) * 8u;
    const unsigned bkey = n0 + col;
    const size_t comp_kv_b_off = comp_base_per ? (size_t)(unsigned)comp_base_per[b] : (size_t)b * comp_kv_batch_stride;
    const _Float16* brow = nullptr;
    if (bkey < n_total) {
        brow = (bkey < n_raw) ? raw_kv + (size_t)(bkey + raw_off) * HD
                              : comp_kv + (comp_kv_b_off + (bkey - n_raw)) * HD;
    }
    bool key_masked = false;
    if (comp_allowed_bits != nullptr && bkey >= n_raw && bkey < n_total) {
        const unsigned c = bkey - n_raw;
        const unsigned word_idx = (size_t)b * max_keys_words + (c >> 5);
        key_masked = ((comp_allowed_bits[word_idx] >> (c & 31)) & 1u) == 0u;
    }
    const float* qh = q + ((size_t)b * n_head + mt * 16u + col) * HD;
    float8 c = (float8){0,0,0,0,0,0,0,0};
    #pragma unroll 1
    for (unsigned g = 0; g < 32u; g += 16u) {
        half8 bf[16], af[16];
        #pragma unroll
        for (unsigned u = 0; u < 16u; u++) {
            const unsigned dim0 = (g + u) * 16u + kh;
            half8 t = {0,0,0,0,0,0,0,0};
            if (brow) {
                #pragma unroll
                for (int e = 0; e < 8; e++) t[e] = brow[dim0 + (unsigned)e];
            }
            bf[u] = t;
            half8 a;
            #pragma unroll
            for (int e = 0; e < 8; e++) a[e] = (_Float16)qh[dim0 + (unsigned)e];
            af[u] = a;
        }
        #pragma unroll
        for (unsigned u = 0; u < 16u; u++) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[u], bf[u], c);
    }
    const _Float16 NEG_INF_F16 = (_Float16)(-INFINITY);
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned h = mt * 16u + (lane >> 4) * 8u + (unsigned)e;
        if (h < n_head && bkey < n_total)
            scores_g[((size_t)b * n_head + h) * max_keys + bkey] = key_masked ? NEG_INF_F16 : (_Float16)(c[e] * kq_scale);
    }
}

#endif  // gfx12
