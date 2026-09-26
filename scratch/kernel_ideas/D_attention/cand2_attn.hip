// D_attention candidates, attempt 2 (gfx1201 only). Decode-shaped twins of the production pair
//   attention_dec_score_htiled_wmma_f16s + attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s
// for n_total <= 640 keys (128 window + <= 512 gathered), 64 heads x 512, b <= 16.
//
//  attention_dec_fused_direct_g16 / _g32   ONE launch, grid (4, b) x 512, BIT-EXACT with the pair:
//      q staged once per WG as f16 in LDS (same RN cast as production's per-warp recast), scores
//      -> LDS (production chain + rounding), production phase A, then W.V with the V fragments
//      loaded DIRECTLY from global into WMMA B registers (no LDS V tile, no barriers), DEPTH
//      tiles in flight, 2-slot register ring. _g16 issues the K-chunk loads of the score chain in
//      2 groups of 16 (production), _g32 all 32 at once (q no longer needs registers).
//  attention_dec_flash_part1_k32/k64/k128 + attention_dec_flash_combine   flash-decoding split-K:
//      grid (4, S, b) x 512 with S = ceil(n_total_max / KPS). Each WG scores its KPS keys
//      (production chain, bit-exact scores), local softmax (max includes the sink as production),
//      partial W.V, writes (o_s, m_s, l_s); combine rescales. NOT bit-exact: f16 weights are
//      rounded relative to the split max, accumulation re-associated.
//  attention_dec_score_blk256 / blk128 / blk64   production dec score with 8 / 4 / 2 warps per WG
//      (grid.x = ceil(n_total / (warps*16))). Bit-exact (same per-warp chain).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define HD            512u
#define HT            16u
#define DEC_MAX_KEYS  640u
#define MAX_PER_LANE  (DEC_MAX_KEYS / 32u)   // 20 scores per lane in phase A
#define WSTRIDE       648u                   // f16 per head row in the LDS weight tile (bank spread)
#define QSTRIDE       520u                   // f16 per head row of the staged q (1040 B: 16 B bank shift per row)

#if defined(__gfx1200__) || defined(__gfx1201__)

// ---------------------------------------------------------------------------------------------
struct TileDesc {
    const _Float16* base;   // kv rows
    unsigned row0;          // first row in `base`
    unsigned n_avail;       // 1..16 valid keys
    unsigned woff;          // first key index in the weights row
};
// Tile t of the concatenated (raw, comp) key list, PRODUCTION tiling: raw tiles first (tail padded),
// then comp tiles.
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

// Phase A (production order): one wave per head; scores f16 from `src` (stride 1), f32 math, f16
// weights written to `wrow`, s_inv[wave] = 1/sum. All <= 20 loads per lane issued before use.
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

__device__ __forceinline__ void store_out(float* out, unsigned b, unsigned n_head, unsigned h0,
                                          unsigned tid, const float* s_inv,
                                          const float8& c_a, const float8& c_b) {
    const unsigned lane = tid & 31u, warp = tid >> 5;
    const unsigned arow = lane & 15u;
    const unsigned d_a  = warp * 16u + arow;
    const unsigned d_b  = (warp + 16u) * 16u + arow;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned mh = (lane >> 4) * 8u + (unsigned)e;
        const float scale = s_inv[mh];
        const size_t obase = ((size_t)b * n_head + h0 + mh) * HD;
        out[obase + d_a] = c_a[e] * scale;
        out[obase + d_b] = c_b[e] * scale;
    }
}

// Production score chain for one 16-key tile, A from an f16 q row (LDS), B rows from `brow`
// (nullptr = key absent -> zero fragment). GROUP = K-chunks whose loads are in flight together.
template <int GROUP>
__device__ __forceinline__ float8 score_chain_lds_q(const _Float16* qrow, const _Float16* brow, unsigned kh) {
    float8 c = (float8){0,0,0,0,0,0,0,0};
    #pragma unroll 1
    for (unsigned g = 0; g < 32u; g += (unsigned)GROUP) {
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

// ---------------------------------------------------------------------------------------------
// Fused bit-exact decode attention. grid (n_head/16, b) x 512. Requires n_head % 16 == 0.
template <int GROUP, int DEPTH>
__device__ __forceinline__ void fused_direct_body(
        float* out, const float* sinks, const float* q, const _Float16* raw_kv,
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,
        const int* n_comp_per, unsigned n_head, float kq_scale, unsigned comp_kv_batch_stride,
        const int* comp_base_per, _Float16* w_lds, _Float16* q_lds, float* s_inv) {
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

    // ---- stage q[16 heads][512] f32 -> f16 (RN, as production's per-warp cast)
    {
        const float* qb = q + ((size_t)b * n_head + h0) * HD;
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            const unsigned e = tid + (unsigned)it * 512u;   // float4 index 0..2047
            const unsigned h = e >> 7, d4 = e & 127u;
            const float4 v = ((const float4*)(qb + h * HD))[d4];
            _Float16* dst = q_lds + h * QSTRIDE + d4 * 4u;
            dst[0] = (_Float16)v.x; dst[1] = (_Float16)v.y; dst[2] = (_Float16)v.z; dst[3] = (_Float16)v.w;
        }
    }
    __syncthreads();

    // ---- phase S: warp w scores tiles w, w+16, ... (production chain, production rounding)
    {
        const unsigned col = lane & 15u, kh = (lane >> 4) * 8u;
        const _Float16* qrow = q_lds + col * QSTRIDE;
        for (unsigned t = wave; t < n_tiles; t += 16u) {
            const TileDesc d = tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
            const _Float16* brow = (col < d.n_avail) ? d.base + (size_t)(d.row0 + col) * HD : nullptr;
            const float8 c = score_chain_lds_q<GROUP>(qrow, brow, kh);
            if (col < d.n_avail) {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    w_lds[(kh + (unsigned)e) * WSTRIDE + d.woff + col] = (_Float16)(c[e] * kq_scale);
            }
        }
    }
    __syncthreads();
    // ---- phase A in place (production order)
    phase_a(w_lds + wave * WSTRIDE, w_lds + wave * WSTRIDE, wave, lane, n_total, sinks[h0 + wave], s_inv);
    __syncthreads();

    // ---- phase B: direct-global V fragments, DEPTH tiles per group, 2 register slots
    const unsigned arow = lane & 15u, kh = (lane >> 4) * 8u;
    const unsigned d_a = wave * 16u + arow, d_b = (wave + 16u) * 16u + arow;
    const _Float16* wr = w_lds + arow * WSTRIDE;
    float8 c_a = {0,0,0,0,0,0,0,0}, c_b = {0,0,0,0,0,0,0,0};

    auto load_frag = [&](unsigned t, half8& ba, half8& bb) {
        // straight-line: clamp the row so every load is in bounds, zero-select afterwards
        const TileDesc d = tile_desc(min(t, n_tiles - 1u), n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned k = kh + (unsigned)e;
            const unsigned kc = min(k, d.n_avail - 1u);
            const _Float16* row = d.base + (size_t)(d.row0 + kc) * HD;
            const _Float16 va = row[d_a], vb = row[d_b];
            ba[e] = (k < d.n_avail) ? va : (_Float16)0;
            bb[e] = (k < d.n_avail) ? vb : (_Float16)0;
        }
    };
    auto load_group = [&](unsigned g, half8 (&va)[DEPTH], half8 (&vb)[DEPTH]) {
        #pragma unroll
        for (int p = 0; p < DEPTH; p++) load_frag(g * (unsigned)DEPTH + (unsigned)p, va[p], vb[p]);
    };
    auto compute_group = [&](unsigned g, const half8 (&va)[DEPTH], const half8 (&vb)[DEPTH]) {
        #pragma unroll
        for (int p = 0; p < DEPTH; p++) {
            const unsigned t = g * (unsigned)DEPTH + (unsigned)p;
            if (t >= n_tiles) break;
            const TileDesc d = tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
            half8 a;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned k = kh + (unsigned)e;
                a[e] = (k < d.n_avail) ? wr[d.woff + k] : (_Float16)0;
            }
            c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, va[p], c_a);
            c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, vb[p], c_b);
        }
    };
    const unsigned n_groups = (n_tiles + (unsigned)DEPTH - 1u) / (unsigned)DEPTH;
    half8 va0[DEPTH], vb0[DEPTH], va1[DEPTH], vb1[DEPTH];
    if (n_tiles > 0u) load_group(0u, va0, vb0);
    for (unsigned g = 0; g < n_groups; g += 2u) {
        if (g + 1u < n_groups) load_group(g + 1u, va1, vb1);
        compute_group(g, va0, vb0);
        if (g + 1u >= n_groups) break;
        if (g + 2u < n_groups) load_group(g + 2u, va0, vb0);
        compute_group(g + 1u, va1, vb1);
    }
    store_out(out, b, n_head, h0, tid, s_inv, c_a, c_b);
}

#define DEC_FUSED_DIRECT(NAME, GROUP, DEPTH)                                                   \
extern "C" __global__ __launch_bounds__(512) void NAME(                                        \
        float* out, const float* sinks, const float* q, const _Float16* raw_kv,                \
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,            \
        const int* n_comp_per, unsigned n_head, float kq_scale, unsigned comp_kv_batch_stride,  \
        const int* comp_base_per) {                                                            \
    __shared__ _Float16 w_lds[HT * WSTRIDE];                                                   \
    __shared__ _Float16 q_lds[HT * QSTRIDE];                                                   \
    __shared__ float s_inv[HT];                                                                \
    fused_direct_body<GROUP, DEPTH>(out, sinks, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per, \
                           n_comp_per, n_head, kq_scale, comp_kv_batch_stride, comp_base_per,  \
                           w_lds, q_lds, s_inv);                                               \
}
DEC_FUSED_DIRECT(attention_dec_fused_direct_g16, 16, 8)
DEC_FUSED_DIRECT(attention_dec_fused_direct_g32, 32, 8)
DEC_FUSED_DIRECT(attention_dec_fused_direct_g32d4, 32, 4)

// ---------------------------------------------------------------------------------------------
// Flash-decoding split-K. part1 grid (n_head/16, S, b) x 512, KPS keys per WG (NT = KPS/16 tiles).
// Partials: part_o[((b*max_splits + s)*n_head + h)*512 + d] (f32, unnormalised),
//           part_ml[((b*max_splits + s)*n_head + h)*2 + {0:m, 1:l}].
template <int KPS>
__device__ __forceinline__ void flash_part1_body(
        float* part_o, float* part_ml, const float* q, const _Float16* raw_kv,
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,
        const int* n_comp_per, const float* sinks, unsigned n_head, float kq_scale,
        unsigned comp_kv_batch_stride, const int* comp_base_per, unsigned max_splits,
        _Float16* w_lds, float* s_m, float* s_l) {
    constexpr int NT = KPS / 16;
    constexpr unsigned WS = (unsigned)KPS + 8u;   // LDS weight row stride (f16)
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

    auto key_row = [&](unsigned key) -> const _Float16* {   // key < n_total
        return (key < n_raw) ? raw_kv + (size_t)(key + raw_off) * HD
                             : comp_kv + (comp_kv_b_off + (key - n_raw)) * HD;
    };

    // ---- V fragments for all NT tiles, issued first (in flight during the score phase)
    half8 va[NT], vb[NT];
    #pragma unroll
    for (int p = 0; p < NT; p++) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned kl = (unsigned)p * 16u + kh + (unsigned)e;
            const unsigned key = key0 + min(kl, nk - 1u);
            const _Float16* row = key_row(key);
            const _Float16 x = row[d_a], y = row[d_b];
            va[p][e] = (kl < nk) ? x : (_Float16)0;
            vb[p][e] = (kl < nk) ? y : (_Float16)0;
        }
    }

    // ---- phase S: warps 0..NT-1, production chain (q f32 -> f16 per warp, 2 groups of 16 chunks)
    if (wave < (unsigned)NT) {
        const unsigned col = lane & 15u;
        const unsigned kl = wave * 16u + col;
        const _Float16* brow = (kl < nk) ? key_row(key0 + kl) : nullptr;
        const float* qh = q + ((size_t)b * n_head + h0 + col) * HD;
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
        if (kl < nk) {
            #pragma unroll
            for (int e = 0; e < 8; e++) w_lds[(kh + (unsigned)e) * WS + kl] = (_Float16)(c[e] * kq_scale);
        }
    }
    __syncthreads();
    // ---- phase A (local): wave = head, nk <= 128 scores -> <= 4 per lane
    {
        constexpr int PER = KPS / 32;
        _Float16* wrow = w_lds + wave * WS;
        const float sink_h = sinks[h0 + wave];
        float v[PER];
        #pragma unroll
        for (int u = 0; u < PER; u++) {
            const unsigned rr = lane + (unsigned)u * 32u;
            v[u] = (rr < nk) ? (float)wrow[rr] : -INFINITY;
        }
        float m = sink_h;
        #pragma unroll
        for (int u = 0; u < PER; u++) if (v[u] > m) m = v[u];
        {
            float o;
            o = __shfl_xor(m, 16); if (o > m) m = o;
            o = __shfl_xor(m,  8); if (o > m) m = o;
            o = __shfl_xor(m,  4); if (o > m) m = o;
            o = __shfl_xor(m,  2); if (o > m) m = o;
            o = __shfl_xor(m,  1); if (o > m) m = o;
        }
        float l = 0.0f;
        #pragma unroll
        for (int u = 0; u < PER; u++) {
            const unsigned rr = lane + (unsigned)u * 32u;
            if (rr < nk) {
                const float w = expf(v[u] - m);
                wrow[rr] = (_Float16)w;
                l += w;
            }
        }
        l += __shfl_xor(l, 16);
        l += __shfl_xor(l,  8);
        l += __shfl_xor(l,  4);
        l += __shfl_xor(l,  2);
        l += __shfl_xor(l,  1);
        if (lane == 0) { s_m[wave] = m; s_l[wave] = l; }
    }
    __syncthreads();
    // ---- phase B: NT tiles from registers
    float8 c_a = {0,0,0,0,0,0,0,0}, c_b = {0,0,0,0,0,0,0,0};
    const _Float16* wr = w_lds + arow * WS;
    #pragma unroll
    for (int p = 0; p < NT; p++) {
        if ((unsigned)p * 16u >= nk) break;
        half8 a;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned kl = (unsigned)p * 16u + kh + (unsigned)e;
            a[e] = (kl < nk) ? wr[kl] : (_Float16)0;
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, va[p], c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, vb[p], c_b);
    }
    // ---- partials
    const size_t pbase = ((size_t)b * max_splits + s) * n_head + h0;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned mh = (lane >> 4) * 8u + (unsigned)e;
        float* o = part_o + (pbase + mh) * HD;
        o[d_a] = c_a[e];
        o[d_b] = c_b[e];
    }
    if (wave == 0 && lane < HT) {
        part_ml[(pbase + lane) * 2u + 0u] = s_m[lane];
        part_ml[(pbase + lane) * 2u + 1u] = s_l[lane];
    }
}

#define DEC_FLASH_PART1(NAME, KPS)                                                             \
extern "C" __global__ __launch_bounds__(512) void NAME(                                        \
        float* part_o, float* part_ml, const float* q, const _Float16* raw_kv,                 \
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,            \
        const int* n_comp_per, const float* sinks, unsigned n_head, float kq_scale,            \
        unsigned comp_kv_batch_stride, const int* comp_base_per, unsigned max_splits) {        \
    __shared__ _Float16 w_lds[HT * ((KPS) + 8u)];                                              \
    __shared__ float s_m[HT], s_l[HT];                                                         \
    flash_part1_body<KPS>(part_o, part_ml, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per,    \
                          n_comp_per, sinks, n_head, kq_scale, comp_kv_batch_stride,           \
                          comp_base_per, max_splits, w_lds, s_m, s_l);                         \
}
DEC_FLASH_PART1(attention_dec_flash_part1_k32, 32)
DEC_FLASH_PART1(attention_dec_flash_part1_k64, 64)
DEC_FLASH_PART1(attention_dec_flash_part1_k128, 128)

// combine: grid (n_head, b) x 512 (thread = d). out[b,h,d] = sum_s o_s e^{m_s-M} / (sum_s l_s e^{m_s-M} + e^{sink-M})
extern "C" __global__ __launch_bounds__(512) void attention_dec_flash_combine(
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

// ---------------------------------------------------------------------------------------------
// Production dec score with WARPS warps per WG (grid.x = ceil(n_total / (WARPS*16))). Same chain.
template <int WARPS>
__device__ __forceinline__ void dec_score_body(
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
    const unsigned n0 = blockIdx.x * ((unsigned)WARPS * 16u) + warp * 16u;
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
#define DEC_SCORE_BLK(NAME, WARPS)                                                             \
extern "C" __global__ __launch_bounds__((WARPS) * 32) void NAME(                               \
        _Float16* scores_g, const float* q, const _Float16* raw_kv, const _Float16* comp_kv,   \
        const int* n_raw_per, const int* n_raw_offset_per, const int* n_comp_per,              \
        const unsigned int* comp_allowed_bits, unsigned max_keys_words, unsigned n_head,       \
        unsigned max_keys, float kq_scale, unsigned comp_kv_batch_stride, const int* comp_base_per) { \
    dec_score_body<WARPS>(scores_g, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per, n_comp_per, \
                          comp_allowed_bits, max_keys_words, n_head, max_keys, kq_scale,       \
                          comp_kv_batch_stride, comp_base_per);                                \
}
DEC_SCORE_BLK(attention_dec_score_blk256, 8)
DEC_SCORE_BLK(attention_dec_score_blk128, 4)
DEC_SCORE_BLK(attention_dec_score_blk64, 2)

#endif  // gfx12
