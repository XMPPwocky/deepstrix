// q2_k_accumulate_matvec — Q2_K × Q8_K matvec for the routed-MoE down
// projection, accumulating across all selected experts.
// Mirrors ds4 `matvec_q2_K_down_accumulate_q8_K` (CPU) and the CUDA
// `dev_dot_q2_K_q8_K_block`-based down projection.
//
// For each output row r:
//   out[r] = sum_expert sum_b dot_q2K(W[expert, r, b], midq[expert, b])
// where:
//   block_q2_K (84 B):  uint8 scales[16] + uint8 qs[64] + uint16 d + uint16 dmin
//   block_q8_K (292 B): float d + int8 qs[256] + int16 bsums[16]
//
// Per-block math (matches dev_dot_q2_K_q8_K_block):
//   summs = sum_j y.bsums[j] * (sc[j] >> 4)
//   dall  = y.d * f16(x.d)
//   dmin  = y.d * f16(x.dmin)
//   isum: 2 outer × 4 inner; per inner pair:
//     d_a = sc[is++] & 0x0f; isum += d_a * dot_q2_16(q2,    q8,    shift)
//     d_b = sc[is++] & 0x0f; isum += d_b * dot_q2_16(q2+16, q8+16, shift)
//     shift += 2; q8 += 32
//   q2 += 32; (after 4-inner step that moves q8 by 128)
//   block_result = dall * isum - dmin * summs
//
// dot_q2_16: 4 dp4a of ((q2[i..i+4] >> shift) & 0x03030303) × q8[i..i+4].
//
// Launch:
//   grid:  (n_rows / 8, 1, 1)   8 rows per workgroup
//   block: (256, 1, 1)           warp-per-row (32 lanes)
// Each warp reduces 6 experts × n_blocks_in dot products (typically
// 6 × 8 = 48 contributions for in_dim=2048).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q2_K_BYTES 84
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float f16_to_f32(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// q2 + i and q8 + i are both 4-aligned. memcpy folds to a single
// dword load.
__device__ __forceinline__ int32_t load_dword(const uint8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

__device__ __forceinline__ int32_t load_dword_i8(const int8_t* p) {
    return load_dword((const uint8_t*)p);
}

__device__ __forceinline__ int32_t dot_q2_16(
        const uint8_t* q2, const int8_t* q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (load_dword(q2 + i) >> shift) & 0x03030303;
        sum = sudot4(v, load_dword_i8(q8 + i), sum);
    }
    return sum;
}

__device__ __forceinline__ float dot_block_q2k_q8k(
        const uint8_t* x_blk, const uint8_t* y_blk) {
    const uint8_t* sc = x_blk;                           // scales[16]
    const uint8_t* q2_base = x_blk + 16;                 // qs[64]
    const uint16_t xd_bits  = *(const uint16_t*)(x_blk + 80);
    const uint16_t xdm_bits = *(const uint16_t*)(x_blk + 82);

    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8 = (const int8_t*)(y_blk + 4);
    const int16_t* bsums = (const int16_t*)(y_blk + 260);

    int32_t summs = 0;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        summs += (int32_t)bsums[j] * (int32_t)(sc[j] >> 4);
    }
    const float dall = yd * f16_to_f32(xd_bits);
    const float dmin = yd * f16_to_f32(xdm_bits);

    int32_t isum = 0;
    int is = 0;
    const uint8_t* q2 = q2_base;
    for (int k = 0; k < QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int32_t d = (int32_t)(sc[is++] & 0x0f);
            isum += d * dot_q2_16(q2, q8, shift);
            d = (int32_t)(sc[is++] & 0x0f);
            isum += d * dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

__device__ __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// Per-expert Q2_K matvec contribution, accumulated into `out`.
// Weight layout (expert-base pointer): row-major [n_rows, n_blocks_in] of
// block_q2_K (84 bytes each). Activation `xq`: [n_blocks_in] of block_q8_K.
//
// The orchestrator launches this kernel 6 times (once per selected
// expert), with `out_init` controlling whether the first launch zeros
// the output. Subsequent launches add into existing `out`.
extern "C" __global__ void q2_k_accumulate_matvec(
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

    const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q2_K_BYTES;

    float acc = 0.0f;
    for (unsigned int b = lane; b < n_blocks_in; b += 32u) {
        const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_Q2_K_BYTES;
        const uint8_t* y_blk = xq        + (uint64_t)b * BLOCK_Q8_K_BYTES;
        acc += dot_block_q2k_q8k(w_blk, y_blk);
    }
    acc = warp_sum_f32(acc);

    if (lane == 0) {
        if (zero_init) {
            out[row] = acc;
        } else {
            out[row] += acc;
        }
    }
}
