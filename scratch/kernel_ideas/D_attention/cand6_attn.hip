// D_attention candidates, attempt 2e (gfx1201): BIT-EXACT fused decode attention whose phase B
// stages the V tile TRANSPOSED ([dim][key]) in a warp-private LDS slot so that every WMMA B
// fragment is 2 x ds_load_b64 instead of 8 x ds_load_u16, and whose weights are kept tile-major
// (raw tail zero-padded) so the A fragment is 1 x ds_load_b128 instead of 8 x ds_load_u16.
//
//  attention_dec_fused_vt_d8       phase S as attempt-1 fused_f16s (q f32 from global per chunk)
//  attention_dec_fused_vt_qreg_d8  phase S with the 32 q fragments cast ONCE per warp into 128 VGPRs
//                                  (same per-tile chain -> bit-exact), K loads only per tile
//
// Per tile per lane, phase B: 2 global b128 loads, 8 pack VALU, 8 ds_store_b32, 4 ds_load_b64,
// 1 ds_load_b128, 2 WMMA (vs 2 + 2 ds_store_b128 + 16 ds_load_u16 + 8 ds_load_u16 + 2 before).
// LDS: weights 16 x 664 x 2 = 21,248 B + 16 warps x 2 slots x 1,280 B = 40,960 B -> 62.3 KB.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define HD            512u
#define HT            16u
#define DEC_MAX_KEYS  640u
#define MAX_PER_LANE  (DEC_MAX_KEYS / 32u)
#define WSTRIDE       664u                   // f16 per head row: 41 tiles x 16 + 8; 1328 B = 83 x 16 B, bank shift 12
#define VT_ROW_B      40u                    // bytes per dim row in a slot: 16 keys x 2 B + 8 B pad (bank shift 10)
#define VT_SLOT_B     (32u * VT_ROW_B)       // 32 dims (d_a block + d_b block) -> 1280 B
#define VT_SLOT_H     (VT_SLOT_B / 2u)

#if defined(__gfx1200__) || defined(__gfx1201__)

struct TileDesc {
    const _Float16* base;
    unsigned row0;
    unsigned n_avail;
    unsigned woff;      // compact key index of the tile's first key (phase S scores layout)
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

// Phase A: reads the COMPACT scores row, writes weights TILE-MAJOR (raw tail zero-padded, comp
// tail zero-padded) in place. Same arithmetic and order as production.
__device__ __forceinline__ void phase_a_tm(_Float16* wrow, unsigned wave, unsigned lane, unsigned n_raw,
                                           unsigned n_comp, unsigned n_tr, unsigned n_tiles,
                                           float sink_h, float* s_inv) {
    const unsigned n_total = n_raw + n_comp;
    float v[MAX_PER_LANE];
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) {
        const unsigned rr = lane + u * 32u;
        v[u] = (rr < n_total) ? (float)wrow[rr] : -INFINITY;
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
    float w[MAX_PER_LANE];
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) {
        const unsigned rr = lane + u * 32u;
        w[u] = 0.0f;
        if (rr < n_total) { w[u] = expf(v[u] - max_score); local_sum += w[u]; }
    }
    // every lane has read its 20 values: now write tile-major (same wave, LDS ops stay in order)
    __asm__ volatile("" ::: "memory");
    #pragma unroll
    for (unsigned u = 0; u < MAX_PER_LANE; u++) {
        const unsigned rr = lane + u * 32u;
        if (rr < n_total) {
            const unsigned pos = (rr < n_raw) ? rr : n_tr * 16u + (rr - n_raw);
            wrow[pos] = (_Float16)w[u];
        }
    }
    const unsigned raw_pad = n_tr * 16u - n_raw;                 // 0..15
    if (lane < raw_pad) wrow[n_raw + lane] = (_Float16)0;
    const unsigned comp_end = n_tr * 16u + n_comp;
    const unsigned comp_pad = n_tiles * 16u - comp_end;          // 0..15
    if (lane < comp_pad) wrow[comp_end + lane] = (_Float16)0;
    local_sum += __shfl_xor(local_sum, 16);
    local_sum += __shfl_xor(local_sum,  8);
    local_sum += __shfl_xor(local_sum,  4);
    local_sum += __shfl_xor(local_sum,  2);
    local_sum += __shfl_xor(local_sum,  1);
    if (lane == 0) s_inv[wave] = 1.0f / local_sum;
}

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
__device__ __forceinline__ void score_tile_qreg(const half8 (&af)[32], const _Float16* brow, unsigned kh, float8& c) {
    #pragma unroll 1
    for (unsigned g = 0; g < 32u; g += 16u) {
        half8 bf[16];
        #pragma unroll
        for (unsigned u = 0; u < 16u; u++) {
            const unsigned dim0 = (g + u) * 16u + kh;
            half8 tt = {0,0,0,0,0,0,0,0};
            if (brow) {
                #pragma unroll
                for (int e = 0; e < 8; e++) tt[e] = brow[dim0 + (unsigned)e];
            }
            bf[u] = tt;
        }
        if (g == 0) {
            #pragma unroll
            for (unsigned u = 0; u < 16u; u++) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[u], bf[u], c);
        } else {
            #pragma unroll
            for (unsigned u = 0; u < 16u; u++) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[16 + u], bf[u], c);
        }
    }
}

template <int QREG, int DEPTH>
__device__ __forceinline__ void fused_vt_body(
        float* out, const float* sinks, const float* q, const _Float16* raw_kv,
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,
        const int* n_comp_per, unsigned n_head, float kq_scale, unsigned comp_kv_batch_stride,
        const int* comp_base_per, _Float16* w_lds, char* slots, float* s_inv) {
    const unsigned hg = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const unsigned h0 = hg * HT;
    if (h0 >= n_head) return;
    const unsigned n_raw = (unsigned)n_raw_per[b];
    const unsigned raw_off = (unsigned)n_raw_offset_per[b];
    const unsigned n_comp = (unsigned)n_comp_per[b];
    const _Float16* comp_kv_b = (comp_kv && comp_base_per)
        ? comp_kv + (size_t)(unsigned)comp_base_per[b] * HD
        : (comp_kv && comp_kv_batch_stride != 0u) ? comp_kv + (size_t)b * comp_kv_batch_stride * HD : comp_kv;
    const unsigned n_comp_eff = comp_kv_b ? n_comp : 0u;
    const unsigned wave = tid >> 5, lane = tid & 31u;
    const unsigned n_tr = (n_raw + 15u) / 16u;
    const unsigned n_tiles = n_tr + (n_comp_eff + 15u) / 16u;

    // ---- phase S (compact scores layout)
    {
        const unsigned col = lane & 15u, kh = (lane >> 4) * 8u;
        const float* qh = q + ((size_t)b * n_head + h0 + col) * HD;
        half8 af[32];
        if (QREG) {
            #pragma unroll
            for (unsigned u = 0; u < 32u; u++) {
                const unsigned dim0 = u * 16u + kh;
                half8 a;
                #pragma unroll
                for (int e = 0; e < 8; e++) a[e] = (_Float16)qh[dim0 + (unsigned)e];
                af[u] = a;
            }
        }
        for (unsigned t = wave; t < n_tiles; t += 16u) {
            const TileDesc d = tile_desc(t, n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp_eff);
            const _Float16* brow = (col < d.n_avail) ? d.base + (size_t)(d.row0 + col) * HD : nullptr;
            float8 c = (float8){0,0,0,0,0,0,0,0};
            if (QREG) score_tile_qreg(af, brow, kh, c); else score_tile(qh, brow, kh, c);
            if (col < d.n_avail) {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    w_lds[(kh + (unsigned)e) * WSTRIDE + d.woff + col] = (_Float16)(c[e] * kq_scale);
            }
        }
    }
    __syncthreads();
    phase_a_tm(w_lds + wave * WSTRIDE, wave, lane, n_raw, n_comp_eff, n_tr, n_tiles, sinks[h0 + wave], s_inv);
    __syncthreads();   // the last WG-wide barrier

    // ---- phase B: warp-private transposed 2-slot ring
    const unsigned arow = lane & 15u, kh = (lane >> 4) * 8u;
    const unsigned d_a = wave * 16u + arow, d_b = (wave + 16u) * 16u + arow;
    const _Float16* wr = w_lds + arow * WSTRIDE;
    char* my_slots = slots + wave * (2u * VT_SLOT_B);
    // stager: lane -> keys (2kp, 2kp+1), 8 dims: local rows dq*8..+8 (rows 0-15 = d_a block, 16-31 = d_b block)
    const unsigned kp = lane & 7u, dq = lane >> 3;
    const unsigned scol = ((dq < 2u) ? wave * 16u + dq * 8u : (wave + 16u) * 16u + (dq - 2u) * 8u);
    auto load_tile = [&](unsigned t, uint4& r0, uint4& r1) {
        const TileDesc d = tile_desc(min(t, n_tiles - 1u), n_tr, raw_kv, raw_off, n_raw, comp_kv_b, n_comp_eff);
        const unsigned k0 = 2u * kp, k1 = k0 + 1u;
        const unsigned kc0 = min(k0, d.n_avail - 1u), kc1 = min(k1, d.n_avail - 1u);
        const uint4 x0 = *(const uint4*)(d.base + (size_t)(d.row0 + kc0) * HD + scol);
        const uint4 x1 = *(const uint4*)(d.base + (size_t)(d.row0 + kc1) * HD + scol);
        const bool ok0 = (k0 < d.n_avail) && (t < n_tiles), ok1 = (k1 < d.n_avail) && (t < n_tiles);
        r0.x = ok0 ? x0.x : 0u; r0.y = ok0 ? x0.y : 0u; r0.z = ok0 ? x0.z : 0u; r0.w = ok0 ? x0.w : 0u;
        r1.x = ok1 ? x1.x : 0u; r1.y = ok1 ? x1.y : 0u; r1.z = ok1 ? x1.z : 0u; r1.w = ok1 ? x1.w : 0u;
    };
    auto load_block = [&](unsigned blk, uint4 (&r0)[DEPTH], uint4 (&r1)[DEPTH]) {
        #pragma unroll
        for (int p = 0; p < DEPTH; p++) load_tile(blk * (unsigned)DEPTH + (unsigned)p, r0[p], r1[p]);
    };
    float8 c_a = {0,0,0,0,0,0,0,0}, c_b = {0,0,0,0,0,0,0,0};
    auto consume = [&](unsigned t, const uint4& r0, const uint4& r1) {
        char* slot = my_slots + (t & 1u) * VT_SLOT_B;
        // transpose-store: row (dq*8 + i) gets the u32 (V[k0][d], V[k1][d]) at key offset kp*4
        const unsigned w0[4] = {r0.x, r0.y, r0.z, r0.w}, w1[4] = {r1.x, r1.y, r1.z, r1.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned lo = (w0[i] & 0xffffu) | (w1[i] << 16);            // dim 2i
            const unsigned hi = (w0[i] >> 16) | (w1[i] & 0xffff0000u);        // dim 2i+1
            *(unsigned*)(slot + (dq * 8u + 2u * (unsigned)i) * VT_ROW_B + kp * 4u) = lo;
            *(unsigned*)(slot + (dq * 8u + 2u * (unsigned)i + 1u) * VT_ROW_B + kp * 4u) = hi;
        }
        __asm__ volatile("" ::: "memory");
        const half8 a = *(const half8*)(wr + t * 16u + kh);                      // tile-major weights
        const uint2 pa0 = *(const uint2*)(slot + arow * VT_ROW_B + kh * 2u);
        const uint2 pa1 = *(const uint2*)(slot + arow * VT_ROW_B + kh * 2u + 8u);
        const uint2 pb0 = *(const uint2*)(slot + (16u + arow) * VT_ROW_B + kh * 2u);
        const uint2 pb1 = *(const uint2*)(slot + (16u + arow) * VT_ROW_B + kh * 2u + 8u);
        const uint4 pa = {pa0.x, pa0.y, pa1.x, pa1.y}, pb = {pb0.x, pb0.y, pb1.x, pb1.y};
        const half8 ba = __builtin_bit_cast(half8, pa), bb = __builtin_bit_cast(half8, pb);
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    };
    const unsigned nblk = (n_tiles + (unsigned)DEPTH - 1u) / (unsigned)DEPTH;
    uint4 ra0[DEPTH], ra1[DEPTH], rb0[DEPTH], rb1[DEPTH];
    if (nblk > 0u) load_block(0u, ra0, ra1);
    for (unsigned blk = 0; blk < nblk; blk += 2u) {
        if (blk + 1u < nblk) load_block(blk + 1u, rb0, rb1);
        #pragma unroll
        for (int p = 0; p < DEPTH; p++) {
            const unsigned t = blk * (unsigned)DEPTH + (unsigned)p;
            if (t >= n_tiles) break;
            consume(t, ra0[p], ra1[p]);
        }
        if (blk + 1u >= nblk) break;
        if (blk + 2u < nblk) load_block(blk + 2u, ra0, ra1);
        #pragma unroll
        for (int p = 0; p < DEPTH; p++) {
            const unsigned t = (blk + 1u) * (unsigned)DEPTH + (unsigned)p;
            if (t >= n_tiles) break;
            consume(t, rb0[p], rb1[p]);
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

#define DEC_FUSED_VT(NAME, QREG, DEPTH)                                                        \
extern "C" __global__ __launch_bounds__(512) void NAME(                                        \
        float* out, const float* sinks, const float* q, const _Float16* raw_kv,                \
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,            \
        const int* n_comp_per, unsigned n_head, float kq_scale, unsigned comp_kv_batch_stride,  \
        const int* comp_base_per) {                                                            \
    __shared__ __attribute__((aligned(16))) _Float16 w_lds[HT * WSTRIDE];                      \
    __shared__ __attribute__((aligned(16))) char slots[16u * 2u * VT_SLOT_B];                  \
    __shared__ float s_inv[HT];                                                                \
    fused_vt_body<QREG, DEPTH>(out, sinks, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per,    \
                               n_comp_per, n_head, kq_scale, comp_kv_batch_stride,             \
                               comp_base_per, w_lds, slots, s_inv);                            \
}
DEC_FUSED_VT(attention_dec_fused_vt_d8, 0, 8)
DEC_FUSED_VT(attention_dec_fused_vt_qreg_d8, 1, 8)
DEC_FUSED_VT(attention_dec_fused_vt_d4, 0, 4)

#endif  // gfx12
