// comp_kv_fp8 — packed FP8 storage of the ratio-4 compressed-KV rows.
//
// Why: the compressor already E4M3-rounds the 448 non-RoPE dims of every
// stored row with a power-of-two block-of-64 scale (fp8_e4m3fn_quantize),
// then stores the result as f16. So the f16 cache holds exactly
// `half_rn(sign * table[idx] * 2^e)` per value and storing (code, e)
// instead is bit-identical after expansion while cutting the row from
// 1024 B to 592 B (-42%). See docs/FP8_KV_IMPL_2026-09.md.
//
// Row layout (FP8_KV_ROW_BYTES = 592, 16-B aligned rows):
//   [0, 448)    u8 code per non-RoPE dim: sign<<7 | 7-bit E4M3FN index
//   [448, 456)  i8 block exponent e for blocks 0..6 (byte 455 = 0 pad)
//   [456, 584)  64 x f16 RoPE dims, verbatim
//   [584, 592)  zero pad
//
// Kernels:
//   comp_kv_append_fp8 / _batched   producer: post-RoPE f32 row -> packed
//                                   row (+ f16 head shadow for row < HEAD)
//   indexer_gather_fp8 / _batched   consumer: selected rows -> f16
//                                   active_comp_kv (what attention reads)
//   comp_kv_fp8_expand              rows [0, n) -> f16 (shadow rebuild on
//                                   snapshot restore, oracle dump, v3
//                                   snapshot conversion check)
//   fp8_kv_table_check, fp8_kv_expand_exhaustive   test kernels
//
// Every f32 -> f16 conversion here is __float2half_rn on the f32 product
// `value * 2^e`, the same as f16_roundtrip.hip; the numerics are pinned
// by fp8_e4m3fn_common.inc.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "fp8_e4m3fn_common.inc"

#define FP8_KV_HEAD_DIM   512u
#define FP8_KV_N_NOPE     448u
#define FP8_KV_N_ROT      64u
#define FP8_KV_N_BLOCKS   7u
#define FP8_KV_ROW_BYTES  592u
#define FP8_KV_OFF_EXP    448u
#define FP8_KV_OFF_ROPE   456u
#define FP8_KV_OFF_PAD    584u

// ---------------------------------------------------------------------------
// Producer
// ---------------------------------------------------------------------------

// One 512-thread workgroup per row. `row` is the post-RoPE f32 row
// (what fp8_e4m3fn_quantize used to see). Writes the packed row at
// `packed + row_idx * FP8_KV_ROW_BYTES` and, when `row_idx < head_rows`,
// the f16 row at `head + row_idx * 512` (the dense-path shadow; identical
// to what the f16 cache used to hold, computed by the SAME expand
// expression the gather uses).
__device__ __forceinline__ void comp_kv_pack_row(
        unsigned char * __restrict__ packed,
        unsigned short * __restrict__ head,
        unsigned int head_rows,
        const float * __restrict__ row,
        unsigned int row_idx) {
    const unsigned int t = threadIdx.x;   // 0..511
    __shared__ float s_red[FP8_KV_N_BLOCKS * 64u];
    __shared__ int   s_exp[FP8_KV_N_BLOCKS + 1u];

    const float v = row[t];
    if (t < FP8_KV_N_NOPE) s_red[t] = fabsf(v);
    __syncthreads();

    // Per-64 block max: same tree as fp8_e4m3fn_quantize (max is exact in
    // any order, this is just the shape).
    for (unsigned int stride = 32; stride > 0; stride >>= 1) {
        if (t < FP8_KV_N_NOPE) {
            const unsigned int lane = t & 63u;
            if (lane < stride) {
                const float other = s_red[t + stride];
                if (other > s_red[t]) s_red[t] = other;
            }
        }
        __syncthreads();
    }
    if (t < FP8_KV_N_BLOCKS) {
        // Clamp into the i8 field. Unreachable for finite rows (amax <=
        // FLT_MAX gives e <= 120; the 1e-4 floor gives e >= -22); a
        // non-finite amax makes (int)ceilf() implementation-defined and
        // the f16 path stores garbage for that block too — out of domain.
        int e = fp8_kv_block_exp(s_red[t * 64u]);
        if (e < -128) e = -128;
        if (e > 127) e = 127;
        s_exp[t] = e;
    }
    if (t == FP8_KV_N_BLOCKS) s_exp[t] = 0;   // pad exponent byte
    __syncthreads();

    unsigned char * out = packed + (unsigned long long)row_idx * FP8_KV_ROW_BYTES;

    if (t < FP8_KV_N_NOPE) {
        const int e = s_exp[t >> 6];
        const float scale = ldexpf(1.0f, e);
        float vs = v / scale;
        if (vs > 448.0f) vs = 448.0f;
        if (vs < -448.0f) vs = -448.0f;
        const int idx = e4m3fn_nearest_index(fabsf(vs));
        const unsigned int code = (vs < 0.0f ? 0x80u : 0u) | (unsigned int)idx;
        out[t] = (unsigned char)code;
        if (row_idx < head_rows) {
            head[(unsigned long long)row_idx * FP8_KV_HEAD_DIM + t] =
                fp8_kv_expand_half_bits(code, e);
        }
    } else {
        // RoPE dims: stored f16 verbatim (f16_roundtrip then the f16 store).
        const unsigned short h = __half_as_ushort(__float2half_rn(v));
        const unsigned int r = t - FP8_KV_N_NOPE;          // 0..63
        *reinterpret_cast<unsigned short *>(out + FP8_KV_OFF_ROPE + r * 2u) = h;
        if (row_idx < head_rows) {
            head[(unsigned long long)row_idx * FP8_KV_HEAD_DIM + t] = h;
        }
    }
    if (t < FP8_KV_N_BLOCKS + 1u) {
        out[FP8_KV_OFF_EXP + t] = (unsigned char)(signed char)s_exp[t];
    }
    if (t < FP8_KV_ROW_BYTES - FP8_KV_OFF_PAD) {
        out[FP8_KV_OFF_PAD + t] = 0;   // deterministic pad (snapshots are byte images)
    }
}

// Decode: one row appended at index n_comp. Grid (1), block (512).
extern "C" __global__ void comp_kv_append_fp8(
        unsigned char  * __restrict__ packed,     // [n_comp_max * 592] bytes
        unsigned short * __restrict__ head,       // [head_rows * 512] f16 bits
        const float    * __restrict__ row,        // [512] f32, post-RoPE
        unsigned int                  n_comp,
        unsigned int                  head_rows) {
    comp_kv_pack_row(packed, head, head_rows, row, n_comp);
}

// Prefill: n_boundaries rows from rows_b[k*512] appended at n_comp_start+k.
// Grid (1, n_boundaries), block (512).
extern "C" __global__ void comp_kv_append_fp8_batched(
        unsigned char  * __restrict__ packed,
        unsigned short * __restrict__ head,
        const float    * __restrict__ rows_b,     // [n_boundaries, 512] f32
        unsigned int                  n_comp_start,
        unsigned int                  head_rows) {
    const unsigned int k = blockIdx.y;
    comp_kv_pack_row(packed, head, head_rows,
                     rows_b + (unsigned long long)k * FP8_KV_HEAD_DIM,
                     n_comp_start + k);
}

// ---------------------------------------------------------------------------
// Consumers
// ---------------------------------------------------------------------------

// Expand one packed row into 512 f16. 256 threads, thread t owns dims
// (2t, 2t+1): both lie in the same 64-block, so one exponent.
__device__ __forceinline__ void comp_kv_expand_row(
        unsigned short * __restrict__ dst,          // [512] f16 bits
        const unsigned char * __restrict__ src) {   // packed row
    const unsigned int t = threadIdx.x;   // 0..255
    const unsigned int d = t * 2u;
    unsigned int lo, hi;
    if (d < FP8_KV_N_NOPE) {
        const unsigned short codes = *reinterpret_cast<const unsigned short *>(src + d);
        const int e = (int)(signed char)src[FP8_KV_OFF_EXP + (d >> 6)];
        lo = fp8_kv_expand_half_bits(codes & 0xFFu, e);
        hi = fp8_kv_expand_half_bits(codes >> 8, e);
    } else {
        const unsigned int r = d - FP8_KV_N_NOPE;   // 0, 2, .., 62
        const unsigned int w = *reinterpret_cast<const unsigned int *>(src + FP8_KV_OFF_ROPE + r * 2u);
        lo = w & 0xFFFFu;
        hi = w >> 16;
    }
    *reinterpret_cast<unsigned int *>(dst + d) = lo | (hi << 16);
}

// active_comp_kv[i, :] = expand(packed[selected[i], :]). Grid (top_k),
// block (256). Sentinel selected[i] < 0 rows are skipped (as
// indexer_gather does).
extern "C" __global__ void indexer_gather_fp8(
        unsigned short      * __restrict__ active_comp_kv,  // [top_k, 512]
        const unsigned char * __restrict__ packed,          // [n_comp, 592 B]
        const int           * __restrict__ selected,        // [top_k]
        unsigned int                       top_k) {
    const unsigned int i = blockIdx.x;
    if (i >= top_k) return;
    const int src_idx = selected[i];
    if (src_idx < 0) return;
    comp_kv_expand_row(active_comp_kv + (unsigned long long)i * FP8_KV_HEAD_DIM,
                       packed + (unsigned long long)(unsigned int)src_idx * FP8_KV_ROW_BYTES);
}

// Batched: grid (top_k, B), block (256). selected_b at stride top_k,
// destination at stride top_k*512. packed is shared across the batch.
extern "C" __global__ void indexer_gather_fp8_batched(
        unsigned short      * __restrict__ active_comp_kv_b, // [B, top_k, 512]
        const unsigned char * __restrict__ packed,
        const int           * __restrict__ selected_b,       // [B, top_k]
        unsigned int                       top_k) {
    const unsigned int i = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (i >= top_k) return;
    const int src_idx = selected_b[(unsigned long long)b * top_k + i];
    if (src_idx < 0) return;
    comp_kv_expand_row(active_comp_kv_b + ((unsigned long long)b * top_k + i) * FP8_KV_HEAD_DIM,
                       packed + (unsigned long long)(unsigned int)src_idx * FP8_KV_ROW_BYTES);
}

// dst[r, :] = expand(packed[r, :]) for r in [0, n_rows). Grid (n_rows),
// block (256).
extern "C" __global__ void comp_kv_fp8_expand(
        unsigned short      * __restrict__ dst,     // [n_rows, 512]
        const unsigned char * __restrict__ packed,  // [n_rows, 592 B]
        unsigned int                       n_rows) {
    const unsigned int r = blockIdx.x;
    if (r >= n_rows) return;
    comp_kv_expand_row(dst + (unsigned long long)r * FP8_KV_HEAD_DIM,
                       packed + (unsigned long long)r * FP8_KV_ROW_BYTES);
}

// ---------------------------------------------------------------------------
// Test kernels (tests/fp8_kv_format.rs)
// ---------------------------------------------------------------------------

// out_table[i] = c_e4m3fn_values[i], out_decode[i] = e4m3fn_decode_index(i),
// i in [0, 128). Block (128).
extern "C" __global__ void fp8_kv_table_check(
        float * __restrict__ out_table,
        float * __restrict__ out_decode) {
    const unsigned int i = threadIdx.x;
    if (i >= 128u) return;
    out_table[i] = c_e4m3fn_values[i];
    out_decode[i] = e4m3fn_decode_index((int)i);
}

// out[(e - e_lo) * 256 + code] = the NEW expand for every (code, e) with
// e in [e_lo, e_lo + n_e). Compared on the host against the host expand
// (the v3 snapshot converter's reference). The old-vs-new proof does NOT
// live here: a single-kernel replica of the old chain is not faithful
// (its f32 product never round-trips through memory, so the compiler may
// fold the sign of a zero product); tests/fp8_kv_format.rs pushes every
// reachable (code, e) through the REAL old kernels instead.
extern "C" __global__ void fp8_kv_expand_exhaustive(
        unsigned short * __restrict__ out,       // [n_e * 256]
        int                           e_lo,
        unsigned int                  n_e) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_e * 256u) return;
    const unsigned int code = gid & 255u;
    const int e = e_lo + (int)(gid >> 8);
    out[gid] = ((code & 0x7Fu) == 127u) ? (unsigned short)0 : fp8_kv_expand_half_bits(code, e);
}
