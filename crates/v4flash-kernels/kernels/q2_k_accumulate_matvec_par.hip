// q2_k_accumulate_matvec_par — parallel reimplementation of
// q2_k_accumulate_matvec (M14g).
//
// Original kernel uses 1 warp per row with each lane covering blocks
// `lane, lane + 32, ...`. For the routed-MoE down projection
// n_blocks_in = N_FF_EXP/256 = 8 — so only lanes 0..7 do work; 75% of
// the warp is idle.
//
// This variant maps **4 lanes per super-block** when n_blocks_in ≤ 8:
//   lane:    0  1  2  3   4  5  6  7  ...  28 29 30 31
//   block:   0  0  0  0   1  1  1  1  ...   7  7  7  7
//   sub:     0  1  2  3   0  1  2  3  ...   0  1  2  3
//
// Inner work split: each block's 16 dot_q2_16 calls (2 outer × 4 inner
// × 2 halves) are split across 4 lanes:
//   sub 0: k=0, j={0,1}  → 4 dot_q2_16
//   sub 1: k=0, j={2,3}  → 4 dot_q2_16
//   sub 2: k=1, j={0,1}  → 4 dot_q2_16
//   sub 3: k=1, j={2,3}  → 4 dot_q2_16
//
// The summs term (Σ_j bsums[j] * (sc[j]>>4), 16 mads) is also split
// 4 ways: each sub handles 4 of the 16 entries. The per-lane partial
// emits `dall * partial_isum - dmin * partial_summs`; summing all 4
// reproduces the original block_result formula exactly.
//
// For n_blocks_in > 8 the stride wraps with step 8, so each (block_lane)
// iterates ceil(n_blocks_in/8) times — generality preserved.
//
// Numerics: integer per-(k,j) contributions are identical to serial;
// the f32 reduction order differs (warp-tree across 32 partials instead
// of 8), so results match within f32-ULP.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q2_K_BYTES 84
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float f16_to_f32_par(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4_par(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t load_i32_unaligned_par(const uint8_t* p) {
    return (int32_t)((uint32_t)p[0] |
                     ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) |
                     ((uint32_t)p[3] << 24));
}

__device__ __forceinline__ int32_t load_i32_unaligned_i8_par(const int8_t* p) {
    return load_i32_unaligned_par((const uint8_t*)p);
}

__device__ __forceinline__ int32_t dot_q2_16_par(
        const uint8_t* q2, const int8_t* q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (load_i32_unaligned_par(q2 + i) >> shift) & 0x03030303;
        sum = sudot4_par(v, load_i32_unaligned_i8_par(q8 + i), sum);
    }
    return sum;
}

// Half-block dot: 4 of 16 dot_q2_16 calls (2 of 4 j-iterations for one
// of 2 k-outers). Returns this lane's contribution to block_result:
//   sub=0 (k=0,j=0,1):  dall * partial_isum  - dmin * (summs[0..4])
//   sub=1 (k=0,j=2,3):  dall * partial_isum  - dmin * (summs[4..8])
//   sub=2 (k=1,j=0,1):  dall * partial_isum  - dmin * (summs[8..12])
//   sub=3 (k=1,j=2,3):  dall * partial_isum  - dmin * (summs[12..16])
__device__ __forceinline__ float dot_block_quarter(
        const uint8_t* x_blk, const uint8_t* y_blk, unsigned int sub) {
    const uint8_t* sc       = x_blk;
    const uint8_t* q2_base  = x_blk + 16;
    const uint16_t xd_bits  = *(const uint16_t*)(x_blk + 80);
    const uint16_t xdm_bits = *(const uint16_t*)(x_blk + 82);

    const float yd  = *(const float*)(y_blk + 0);
    const int8_t* q8_base = (const int8_t*)(y_blk + 4);
    const int16_t* bsums  = (const int16_t*)(y_blk + 260);

    const float dall = yd * f16_to_f32_par(xd_bits);
    const float dmin = yd * f16_to_f32_par(xdm_bits);

    const unsigned int k     = sub >> 1u;   // 0 or 1
    const unsigned int j_lo  = (sub & 1u) * 2u; // 0 or 2

    // is base for this (k, j_lo): 2 * (4*k + j_lo)
    const unsigned int is_base = 2u * (4u * k + j_lo);
    // q2 base for this k: q2_base + 32*k
    const uint8_t* q2 = q2_base + (uint64_t)k * 32u;
    // q8 base for this (k, j_lo): k*128 + j_lo*32
    const int8_t* q8 = q8_base + (uint64_t)k * 128u + (uint64_t)j_lo * 32u;
    // shift for this j_lo: 2 * j_lo
    int shift = (int)(2u * j_lo);

    int32_t partial_isum = 0;
    #pragma unroll
    for (int jj = 0; jj < 2; jj++) {
        const int32_t d_a = (int32_t)(sc[is_base + 2u * jj] & 0x0f);
        partial_isum += d_a * dot_q2_16_par(q2, q8, shift);
        const int32_t d_b = (int32_t)(sc[is_base + 2u * jj + 1u] & 0x0f);
        partial_isum += d_b * dot_q2_16_par(q2 + 16, q8 + 16, shift);
        shift += 2;
        q8 += 32;
    }

    // 4 of 16 summs entries; sub picks which 4.
    int32_t partial_summs = 0;
    const unsigned int summs_lo = sub * 4u;
    #pragma unroll
    for (unsigned int t = 0; t < 4u; t++) {
        const unsigned int idx = summs_lo + t;
        partial_summs += (int32_t)bsums[idx] * (int32_t)(sc[idx] >> 4);
    }

    return dall * (float)partial_isum - dmin * (float)partial_summs;
}

__device__ __forceinline__ float warp_sum_f32_par(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

extern "C" __global__ void q2_k_accumulate_matvec_par(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_expert,
        const uint8_t  * __restrict__ xq,
        unsigned int     n_rows,
        unsigned int     n_blocks_in,
        unsigned int     zero_init) {
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;        // 0..7
    const unsigned int sub        = lane & 3u;         // 0..3

    const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q2_K_BYTES;

    float acc = 0.0f;
    // Each lane group of 4 strides over blocks with step 8.
    for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
        const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_Q2_K_BYTES;
        const uint8_t* y_blk = xq       + (uint64_t)b * BLOCK_Q8_K_BYTES;
        acc += dot_block_quarter(w_blk, y_blk, sub);
    }
    acc = warp_sum_f32_par(acc);

    if (lane == 0) {
        if (zero_init) {
            out[row] = acc;
        } else {
            out[row] += acc;
        }
    }
}
