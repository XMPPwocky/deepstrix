// attention_dec -- decode-shape (few rows) twin of the arena attention score
// kernel `attention_mixed_score_batched_htiled_wmma_f16s` (attention_mixed.hip),
// BIT-IDENTICAL output (tests/attention_dec_bitexact.rs).
//
// The batched kernel gives each warp 16 keys and loops ALL n_head/16 head
// tiles inside its 32-step K-chunk loop, one chunk's loads per iteration.
// At a decode row (128 window + 512 gathered keys, 64 heads) that is 3 WGs
// whose warps each walk 32 dependent-issue load round trips: ~16 us for
// ~0.8 MB (tests/bench_decode_latency_breakdown.rs, attn section).
//
// Here the head tile is a grid dimension (grid (key_blocks, n_head/16, B)),
// so there are n_head/16 times more warps, and each warp issues the loads of
// SCD_GROUP K-chunks before their WMMAs. Every score is still the same WMMA
// chain -- chunks 0..31 in order on the same A (q, cast to f16) and B (key)
// fragments into the same accumulator -- and the same scale / cast / mask at
// the store, so only the schedule changes.
//
// ROOFLINE (b = 1, 640 keys, 64 heads, head_dim 512): K 640 x 1 KB f16 =
// 655 KB + q 128 KB (L2-shared); 42 MFLOP of WMMA. ~1.3 us at 640 GB/s.
// Latency-bound: 2 groups of loads per warp instead of 32 round trips.
// MEASURED (tests/bench_decode_latency_ab.rs, paired, 1000 rounds, event
// pair ~4.6 us included): 21.0 -> 13.6 / 13.9 / 16.0 us at b = 1 / 2 / 3
// (group 8: 13.2 / 13.6 / 18.5). Staging the head tile's q in LDS once per
// WG was SLOWER (15.9 at b = 1: the warps then wait for the staging before
// their first key loads) -- not the bottleneck.
//
// Requires head_dim == 512 and n_head % 16 == 0 (checked by the launcher).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define SCD_KEYS_PER_BLK 256u   // 16 warps x 16 keys
#define SCD_HEAD_DIM     512u
#define SCD_N_KC         (SCD_HEAD_DIM / 16u)   // 32 K-chunks
#define SCD_GROUP        16u                     // K-chunks whose loads are in flight together

extern "C" __global__ __launch_bounds__(512) void attention_dec_score_htiled_wmma_f16s(
        _Float16       * scores_g,       // [B, n_head, max_keys]  (f16)
        const float    * q,              // [B, n_head, 512]
        const _Float16 * raw_kv,
        const _Float16 * comp_kv,
        const int      * n_raw_per,
        const int      * n_raw_offset_per,
        const int      * n_comp_per,
        const unsigned int * comp_allowed_bits, // nullable; [B, max_keys_words]
        unsigned int     max_keys_words,
        unsigned int     n_head,
        unsigned int     max_keys,
        float            kq_scale,
        unsigned int     comp_kv_batch_stride,
        const int      * comp_base_per) {
    const unsigned int b    = blockIdx.z;
    const unsigned int mt   = blockIdx.y;           // head tile
    const unsigned int tid  = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int raw_off= (unsigned int)n_raw_offset_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    const unsigned int n_total = n_raw + n_comp;
#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int n0   = blockIdx.x * SCD_KEYS_PER_BLK + warp * 16u;
    if (n0 >= n_total) return;                       // whole warp past the keys: no stores
    const unsigned int col  = lane & 15u;
    const unsigned int kh   = (lane >> 4) * 8u;
    const unsigned int bkey = n0 + col;
    const size_t comp_kv_b_off = comp_base_per
        ? (size_t)(unsigned int)comp_base_per[b]
        : (size_t)b * comp_kv_batch_stride;
    const _Float16 *brow = nullptr;
    if (bkey < n_total) {
        brow = (bkey < n_raw)
            ? raw_kv + (size_t)(bkey + raw_off) * SCD_HEAD_DIM
            : comp_kv + (comp_kv_b_off + (bkey - n_raw)) * SCD_HEAD_DIM;
    }
    bool key_masked = false;
    if (comp_allowed_bits != nullptr && bkey >= n_raw && bkey < n_total) {
        const unsigned int c = bkey - n_raw;
        const unsigned int word_idx = (size_t)b * max_keys_words + (c >> 5);
        key_masked = ((comp_allowed_bits[word_idx] >> (c & 31)) & 1u) == 0u;
    }
    const float *qh = q + ((size_t)b * n_head + mt * 16u + col) * SCD_HEAD_DIM;
    float8 c = (float8){0,0,0,0,0,0,0,0};
    #pragma unroll 1
    for (unsigned int g = 0; g < SCD_N_KC; g += SCD_GROUP) {
        half8 bf[SCD_GROUP];
        half8 af[SCD_GROUP];
        #pragma unroll
        for (unsigned int u = 0; u < SCD_GROUP; u++) {
            const unsigned int dim0 = (g + u) * 16u + kh;
            half8 t = {0,0,0,0,0,0,0,0};
            if (brow) {
                #pragma unroll
                for (int e = 0; e < 8; e++) t[e] = brow[dim0 + (unsigned int)e];
            }
            bf[u] = t;
            half8 a;
            #pragma unroll
            for (int e = 0; e < 8; e++) a[e] = (_Float16)qh[dim0 + (unsigned int)e];
            af[u] = a;
        }
        #pragma unroll
        for (unsigned int u = 0; u < SCD_GROUP; u++) {
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[u], bf[u], c);
        }
    }
    const _Float16 NEG_INF_F16 = (_Float16)(-INFINITY);
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned int h = mt * 16u + (lane >> 4) * 8u + (unsigned int)e;
        if (h < n_head && bkey < n_total) {
            scores_g[((size_t)b * n_head + h) * max_keys + bkey] =
                key_masked ? NEG_INF_F16 : (_Float16)(c[e] * kq_scale);
        }
    }
#else
    (void)scores_g; (void)q; (void)raw_kv; (void)comp_kv; (void)n_head; (void)mt;
    (void)max_keys; (void)kq_scale; (void)warp; (void)lane; (void)n_total;
    (void)comp_allowed_bits; (void)max_keys_words; (void)comp_kv_batch_stride; (void)comp_base_per;
    (void)raw_off; (void)tid;
#endif
}
