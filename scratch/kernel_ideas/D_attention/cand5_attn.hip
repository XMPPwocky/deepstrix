// D_attention candidates, attempt 2d (gfx1201): BIT-EXACT fused decode attention with a
// WARP-PRIVATE V staging ring (no WG-wide barriers in phase B).
//
//  attention_dec_fused_wpriv_d8     phase S = attempt-1 fused_f16s (q f32 from global, per warp,
//                                   production chain), phase A production, phase B: every warp
//                                   stages ONLY the 2 x 16 columns it needs of each 16-key V tile
//                                   (1 KB) into its own LDS slot (2-slot ring, 80 B key stride =
//                                   conflict-free fragment reads), 8 tiles of global loads in
//                                   flight per lane, no __syncthreads after phase A.
//  attention_dec_fused_wpriv_s2_d8  same, phase S processes tile pairs (t, t+16) per warp with the
//                                   q fragments loaded once: 4 dependent round trips per warp
//                                   instead of 6 (chains unchanged per tile -> bit-exact).
//
// LDS: w 20,736 B + 16 warps x 2 slots x 1,280 B = 40,960 B + s_inv -> 61.8 KB.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define HD            512u
#define HT            16u
#define DEC_MAX_KEYS  640u
#define MAX_PER_LANE  (DEC_MAX_KEYS / 32u)
#define WSTRIDE       648u
#define KSTRIDE_B     80u                    // bytes per key in a warp slot (64 B data + 16 B pad)
#define SLOT_B        (16u * KSTRIDE_B)      // 1280 B per tile slot
#define SLOT_H        (SLOT_B / 2u)          // in f16 units

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

// production dec-score chain for one tile (q f32 from global, cast per element), as attempt 1
__device__ __forceinline__ void score_tile(const float* qh, const _Float16* brow, unsigned kh, float8& c) {
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
}
// two tiles sharing the q fragments (each tile's chain identical to score_tile)
__device__ __forceinline__ void score_tile2(const float* qh, const _Float16* brow1, const _Float16* brow2,
                                            unsigned kh, float8& c1, float8& c2) {
    #pragma unroll 1
    for (unsigned g = 0; g < 32u; g += 16u) {
        half8 b1[16], b2[16], af[16];
        #pragma unroll
        for (unsigned u = 0; u < 16u; u++) {
            const unsigned dim0 = (g + u) * 16u + kh;
            half8 t1 = {0,0,0,0,0,0,0,0}, t2 = {0,0,0,0,0,0,0,0};
            if (brow1) {
                #pragma unroll
                for (int e = 0; e < 8; e++) t1[e] = brow1[dim0 + (unsigned)e];
            }
            if (brow2) {
                #pragma unroll
                for (int e = 0; e < 8; e++) t2[e] = brow2[dim0 + (unsigned)e];
            }
            b1[u] = t1; b2[u] = t2;
            half8 a;
            #pragma unroll
            for (int e = 0; e < 8; e++) a[e] = (_Float16)qh[dim0 + (unsigned)e];
            af[u] = a;
        }
        #pragma unroll
        for (unsigned u = 0; u < 16u; u++) {
            c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[u], b1[u], c1);
            c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[u], b2[u], c2);
        }
    }
}

template <int S2, int DEPTH>
__device__ __forceinline__ void fused_wpriv_body(
        float* out, const float* sinks, const float* q, const _Float16* raw_kv,
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,
        const int* n_comp_per, unsigned n_head, float kq_scale, unsigned comp_kv_batch_stride,
        const int* comp_base_per, _Float16* w_lds, _Float16* slots, float* s_inv) {
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

    // ---- phase S (production chain per tile)
    {
        const unsigned col = lane & 15u, kh = (lane >> 4) * 8u;
        const float* qh = q + ((size_t)b * n_head + h0 + col) * HD;
        auto store_scores = [&](const TileDesc& d, const float8& c) {
            if (col < d.n_avail) {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    w_lds[(kh + (unsigned)e) * WSTRIDE + d.woff + col] = (_Float16)(c[e] * kq_scale);
            }
        };
        if (S2) {
            for (unsigned t = wave; t < n_tiles; t += 32u) {
                const unsigned t2 = t + 16u;
                const bool has2 = t2 < n_tiles;
                const TileDesc d1 = tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
                const TileDesc d2 = tile_desc(has2 ? t2 : t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
                const _Float16* brow1 = (col < d1.n_avail) ? d1.base + (size_t)(d1.row0 + col) * HD : nullptr;
                const _Float16* brow2 = (has2 && col < d2.n_avail) ? d2.base + (size_t)(d2.row0 + col) * HD : nullptr;
                float8 c1 = (float8){0,0,0,0,0,0,0,0}, c2 = (float8){0,0,0,0,0,0,0,0};
                score_tile2(qh, brow1, brow2, kh, c1, c2);
                store_scores(d1, c1);
                if (has2) store_scores(d2, c2);
            }
        } else {
            for (unsigned t = wave; t < n_tiles; t += 16u) {
                const TileDesc d = tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
                const _Float16* brow = (col < d.n_avail) ? d.base + (size_t)(d.row0 + col) * HD : nullptr;
                float8 c = (float8){0,0,0,0,0,0,0,0};
                score_tile(qh, brow, kh, c);
                store_scores(d, c);
            }
        }
    }
    __syncthreads();
    phase_a(w_lds + wave * WSTRIDE, w_lds + wave * WSTRIDE, wave, lane, n_total, sinks[h0 + wave], s_inv);
    __syncthreads();   // weights + s_inv visible; the LAST WG-wide barrier

    // ---- phase B: warp-private 2-slot ring
    const unsigned arow = lane & 15u, kh = (lane >> 4) * 8u;
    const unsigned d_a = wave * 16u + arow, d_b = (wave + 16u) * 16u + arow;
    const _Float16* wr = w_lds + arow * WSTRIDE;
    _Float16* my_slots = slots + wave * (2u * SLOT_H);
    // loader: lane l -> key l>>1, column block l&1 (32 B = 2 x ulong2)
    const unsigned lk = lane >> 1, lblk = lane & 1u;
    const unsigned lcol = (lblk ? (wave + 16u) : wave) * 16u;      // first f16 column of the block
    auto load_tile = [&](unsigned t, ulong2 (&r)[2]) {
        const TileDesc d = tile_desc(min(t, n_tiles - 1u), n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
        const unsigned kc = min(lk, d.n_avail - 1u);
        const ulong2* src = (const ulong2*)(d.base + (size_t)(d.row0 + kc) * HD + lcol);
        const ulong2 x0 = src[0], x1 = src[1];
        const bool ok = (lk < d.n_avail) && (t < n_tiles);
        r[0].x = ok ? x0.x : 0; r[0].y = ok ? x0.y : 0;
        r[1].x = ok ? x1.x : 0; r[1].y = ok ? x1.y : 0;
    };
    auto load_block = [&](unsigned blk, ulong2 (&r)[DEPTH][2]) {
        #pragma unroll
        for (int p = 0; p < DEPTH; p++) load_tile(blk * (unsigned)DEPTH + (unsigned)p, r[p]);
    };
    float8 c_a = {0,0,0,0,0,0,0,0}, c_b = {0,0,0,0,0,0,0,0};
    auto consume = [&](unsigned t, const ulong2 (&r)[2]) {
        _Float16* slot = my_slots + (t & 1u) * SLOT_H;
        // stage: key lk, block lblk -> bytes [lk*80 + lblk*32, +32)
        ulong2* dst = (ulong2*)((char*)slot + lk * KSTRIDE_B + lblk * 32u);
        dst[0] = r[0]; dst[1] = r[1];
        __asm__ volatile("" ::: "memory");     // keep the fragment reads after the stores (same wave: LDS is in order)
        const TileDesc d = tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp);
        half8 a, ba, bb;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned k = kh + (unsigned)e;
            const _Float16* krow = (const _Float16*)((const char*)slot + k * KSTRIDE_B);
            const _Float16 va = krow[arow], vb = krow[16u + arow];
            const bool ok = k < d.n_avail;
            a[e]  = ok ? wr[d.woff + k] : (_Float16)0;
            ba[e] = ok ? va : (_Float16)0;
            bb[e] = ok ? vb : (_Float16)0;
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    };
    const unsigned nblk = (n_tiles + (unsigned)DEPTH - 1u) / (unsigned)DEPTH;
    ulong2 r0[DEPTH][2], r1[DEPTH][2];
    if (nblk > 0u) load_block(0u, r0);
    for (unsigned blk = 0; blk < nblk; blk += 2u) {
        if (blk + 1u < nblk) load_block(blk + 1u, r1);
        #pragma unroll
        for (int p = 0; p < DEPTH; p++) {
            const unsigned t = blk * (unsigned)DEPTH + (unsigned)p;
            if (t >= n_tiles) break;
            consume(t, r0[p]);
        }
        if (blk + 1u >= nblk) break;
        if (blk + 2u < nblk) load_block(blk + 2u, r0);
        #pragma unroll
        for (int p = 0; p < DEPTH; p++) {
            const unsigned t = (blk + 1u) * (unsigned)DEPTH + (unsigned)p;
            if (t >= n_tiles) break;
            consume(t, r1[p]);
        }
    }
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned mh = (lane >> 4) * 8u + (unsigned)e;
        const float scale = s_inv[mh];
        const size_t obase = ((size_t)b * n_head + h0 + mh) * HD;
        out[obase + d_a] = c_a[e] * scale;
        out[obase + d_b] = c_b[e] * scale;
    }
}

#define DEC_FUSED_WPRIV(NAME, S2, DEPTH)                                                       \
extern "C" __global__ __launch_bounds__(512) void NAME(                                        \
        float* out, const float* sinks, const float* q, const _Float16* raw_kv,                \
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,            \
        const int* n_comp_per, unsigned n_head, float kq_scale, unsigned comp_kv_batch_stride,  \
        const int* comp_base_per) {                                                            \
    __shared__ _Float16 w_lds[HT * WSTRIDE];                                                   \
    __shared__ __attribute__((aligned(16))) _Float16 slots[16u * 2u * SLOT_H];                 \
    __shared__ float s_inv[HT];                                                                \
    fused_wpriv_body<S2, DEPTH>(out, sinks, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per,   \
                                n_comp_per, n_head, kq_scale, comp_kv_batch_stride,            \
                                comp_base_per, w_lds, slots, s_inv);                           \
}
DEC_FUSED_WPRIV(attention_dec_fused_wpriv_d8, 0, 8)
DEC_FUSED_WPRIV(attention_dec_fused_wpriv_s2_d8, 1, 8)
DEC_FUSED_WPRIV(attention_dec_fused_wpriv_d4, 0, 4)

#endif  // gfx12
