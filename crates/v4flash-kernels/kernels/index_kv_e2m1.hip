// index_kv_e2m1 — packed E2M1 (FP4) storage of the ratio-4 indexer key rows.
//
// Why: indexer_qat already snaps every key value to ±m x 2^e with
// m ∈ {0, .5, 1, 1.5, 2, 3, 4, 6} and one power-of-two exponent per
// 32-element block, then the row is stored as f16. So the f16 key cache
// holds exactly half_rn(±m x 2^e) and storing (sign, m-index, e) is
// bit-identical after expansion while cutting a 128-dim row from 256 B
// to 80 B (-69%). See docs/E2M1_INDEXER_KEYS_2026-09.md.
//
// Row layout (E2M1_KEY_ROW_BYTES = 80, 16-B aligned rows):
//   [0, 64)   nibbles: element i in byte i/2, LOW nibble for even i;
//             nibble = sign<<3 | idx (idx = e2m1fn_value table index)
//   [64, 68)  i8 block exponents for blocks 0..3
//   [68, 80)  zero pad
//
// Producer (index_kv_append_e2m1{,_batched}) takes the QAT'd f32 row —
// indexer_qat itself is untouched — and RE-DERIVES (idx, e) from it: the
// block max is {3, 4, 6} x 2^e0, so the recovered exponent is e0 (max 4
// or 6) or e0 - 1 (max 3, codes doubled, all still <= 6) and every value
// divided by 2^e' is exactly a table value: the nearest search is an exact
// match, never a rounding. Sign from the f32 SIGN BIT so a QAT'd -0.0
// (a negative value that snapped to code 0) stays 0x8000 like the old chain.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "e2m1_key_common.inc"

#define E2M1_KEY_DIM        128u
#define E2M1_KEY_N_BLOCKS   4u
#define E2M1_KEY_ROW_BYTES  80u
#define E2M1_KEY_OFF_EXP    64u
#define E2M1_KEY_OFF_PAD    68u

// One 128-thread workgroup per row.
__device__ __forceinline__ void index_kv_pack_row(
        unsigned char * __restrict__ packed,
        const float   * __restrict__ row,      // [128] QAT'd f32
        unsigned int row_idx) {
    const unsigned int t = threadIdx.x;        // 0..127
    __shared__ float s_abs[E2M1_KEY_DIM];
    __shared__ int   s_exp[E2M1_KEY_N_BLOCKS];
    __shared__ unsigned char s_nib[E2M1_KEY_DIM];

    const float v = row[t];
    s_abs[t] = fabsf(v);
    __syncthreads();
    // Per-32 block max, same fmaxf tree as indexer_qat.
    const unsigned int block_base = t & ~31u;
    const unsigned int lane = t & 31u;
    for (unsigned int stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            s_abs[block_base + lane] = fmaxf(s_abs[block_base + lane], s_abs[block_base + lane + stride]);
        }
        __syncthreads();
    }
    if (t < E2M1_KEY_N_BLOCKS) {
        int e = e2m1_block_exp(s_abs[t * 32u]);
        if (e < -128) e = -128;   // unreachable (floor gives e >= -127); i8 field
        if (e > 127) e = 127;
        s_exp[t] = e;
    }
    __syncthreads();

    const int e = s_exp[t >> 5];
    const float scale = ldexpf(1.0f, e);
    const float q = v / scale;                  // exactly ±table value (or ±0)
    const int idx = e2m1fn_nearest_index(fabsf(q));
    const unsigned int sign = __float_as_uint(v) >> 31;
    s_nib[t] = (unsigned char)((sign << 3) | (unsigned int)idx);
    __syncthreads();

    unsigned char * out = packed + (unsigned long long)row_idx * E2M1_KEY_ROW_BYTES;
    if (t < 64u) {
        out[t] = (unsigned char)(s_nib[2u * t] | (s_nib[2u * t + 1u] << 4));
    } else if (t < 64u + E2M1_KEY_N_BLOCKS) {
        out[t] = (unsigned char)(signed char)s_exp[t - 64u];
    } else if (t < E2M1_KEY_ROW_BYTES) {
        out[t] = 0;   // deterministic pad (snapshots are byte images)
    }
}

// Decode: one row appended at index n_comp. Grid (1), block (128).
extern "C" __global__ void index_kv_append_e2m1(
        unsigned char * __restrict__ packed,   // [n_comp_max * 80] bytes
        const float   * __restrict__ row,      // [128] QAT'd f32
        unsigned int                 n_comp) {
    index_kv_pack_row(packed, row, n_comp);
}

// Prefill: n_boundaries rows from rows_b[k*128] appended at n_comp_start+k.
// Grid (1, n_boundaries), block (128).
extern "C" __global__ void index_kv_append_e2m1_batched(
        unsigned char * __restrict__ packed,
        const float   * __restrict__ rows_b,   // [n_boundaries, 128]
        unsigned int                 n_comp_start) {
    const unsigned int k = blockIdx.y;
    index_kv_pack_row(packed, rows_b + (unsigned long long)k * E2M1_KEY_DIM, n_comp_start + k);
}

// dst[r, :] = expand(packed[r, :]) for r in [0, n_rows): 64 threads per
// row, thread t owns elements (2t, 2t+1) = one byte. Grid (n_rows),
// block (64). Used by the oracle dump and the tests; the score kernels
// expand inline.
extern "C" __global__ void index_kv_e2m1_expand(
        unsigned short      * __restrict__ dst,     // [n_rows, 128] f16 bits
        const unsigned char * __restrict__ packed,  // [n_rows, 80 B]
        unsigned int                       n_rows) {
    const unsigned int r = blockIdx.x;
    const unsigned int t = threadIdx.x;   // 0..63
    if (r >= n_rows || t >= 64u) return;
    const unsigned char * src = packed + (unsigned long long)r * E2M1_KEY_ROW_BYTES;
    const unsigned int byte = src[t];
    const int e = (int)(signed char)src[E2M1_KEY_OFF_EXP + (t >> 4)];   // element 2t is in block (2t)/32 = t/16
    // General (float) path on purpose: this is the reference the fast
    // byte-permute path (e2m1_key_expand8, used by the score kernels) is
    // checked against.
    const unsigned int lo = e2m1_key_expand_half_bits(byte & 0xFu, e);
    const unsigned int hi = e2m1_key_expand_half_bits(byte >> 4, e);
    *reinterpret_cast<unsigned int *>(dst + (unsigned long long)r * E2M1_KEY_DIM + 2u * t) = lo | (hi << 16);
}

// ---------------------------------------------------------------------------
// Test kernels (tests/e2m1_keys_format.rs)
// ---------------------------------------------------------------------------

// out_table[i] = e2m1fn_value(i) for i in [0, 8). Block (8).
extern "C" __global__ void e2m1_key_table_check(float * __restrict__ out_table) {
    const unsigned int i = threadIdx.x;
    if (i < 8u) out_table[i] = e2m1fn_value((int)i);
}

// The fast 8-nibble expansion versus the general per-nibble path: for
// every e in [e_lo, e_lo + n_e) and every word in `words[n_words]`,
// out_fast[(ei * n_words + wi) * 4 + j] = expand8(word, e).j and
// out_ref[...] = the per-nibble general expansion packed the same way.
extern "C" __global__ void e2m1_key_expand8_check(
        unsigned int       * __restrict__ out_fast,
        unsigned int       * __restrict__ out_ref,
        const unsigned int * __restrict__ words,
        unsigned int                      n_words,
        int                               e_lo,
        unsigned int                      n_e) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_e * n_words) return;
    const unsigned int wi = gid % n_words;
    const int e = e_lo + (int)(gid / n_words);
    const unsigned int w = words[wi];
    const uint4 f = e2m1_key_expand8(w, e);
    out_fast[gid * 4 + 0] = f.x; out_fast[gid * 4 + 1] = f.y; out_fast[gid * 4 + 2] = f.z; out_fast[gid * 4 + 3] = f.w;
    unsigned int ref[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const unsigned int lo = e2m1_key_expand_half_bits((w >> (8 * j)) & 15u, e);
        const unsigned int hi = e2m1_key_expand_half_bits((w >> (8 * j + 4)) & 15u, e);
        ref[j] = lo | (hi << 16);
    }
    out_ref[gid * 4 + 0] = ref[0]; out_ref[gid * 4 + 1] = ref[1]; out_ref[gid * 4 + 2] = ref[2]; out_ref[gid * 4 + 3] = ref[3];
}

// out[(e - e_lo) * 16 + nib] = expand(nib, e) for e in [e_lo, e_lo + n_e).
extern "C" __global__ void e2m1_key_expand_exhaustive(
        unsigned short * __restrict__ out,
        int                           e_lo,
        unsigned int                  n_e) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_e * 16u) return;
    out[gid] = e2m1_key_expand_half_bits(gid & 15u, e_lo + (int)(gid >> 4));
}
