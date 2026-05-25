// iq2_xxs_pair_matvec — paired IQ2_XXS×Q8_K matvec for one selected expert.
// Computes simultaneously gate[r] and up[r] = sum_b dot_iq2xxs(W_b, xq_b)
// for r in [0, n_rows), sharing one Q8_K-quantized activation across both
// weight matrices. Mirrors the CPU `ds4_vec_dot_iq2_xxs_pair_q8_K` fast
// path and the CUDA `dev_dot_iq2_xxs_q8_K_block_lut` / `dev_dot_iq2_xxs_q8_K_block`.
//
// Block layout:
//   block_iq2_xxs (66 B): uint16 d + uint16 qs[32]
//   block_q8_K   (292 B): float d + int8 qs[256] + int16 bsums[16]
//
// Algorithm per super-block (256 elements):
//   For ib32 in 0..QK_K/32:                              // 8 sub-blocks per super-block
//     aux0 = qs[0]|qs[1]<<16,  aux1 = qs[2]|qs[3]<<16   // 4 uint16s
//     ls   = 2*(aux1>>28) + 1
//     for l in {0,2}:                                    // two pairs per sub-block
//       sumi += dot_iq2_pair_16(
//                grid[(aux0 >> (8*l)) & 0xff],    sign[(aux1 >> (7*l)) & 0x7f],
//                grid[(aux0 >> (8*(l+1))) & 0xff], sign[(aux1 >> (7*(l+1))) & 0x7f],
//                q8 + (l*8))
//     bsum += sumi * ls
//   sumf += f16_to_f32(W.d) * Y.d * bsum
//   result = 0.125 * sumf
//
// dot_iq2_pair_16: dot of 16 quantized-i8 weights (2 grid entries × 8 i8s
// each, sign-flipped via ksigns_iq2xs) against 16 i8 activations.
//
// Launch: grid (n_rows / ROWS_PER_BLOCK, 1, 1), block (ROWS_PER_BLOCK * 32, 1, 1).
// 8 rows per workgroup, one warp (32 threads) per row.
// Each warp lane handles 1 of 32 ib32 sub-blocks (across all 16 super-blocks
// for in_dim=4096; ib32 step iterates over super-blocks in sequence).
//
// Two outputs per row: gate[row], up[row].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q8_K_BYTES 292
#define BLOCK_IQ2_XXS_BYTES 66
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

#include "iq2_xxs_tables.inc"

__device__ __forceinline__ float f16_to_f32(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4(int32_t a, int32_t b, int32_t c) {
    // RDNA3.5+: signed-signed dp4a via the sudot4 builtin (matches CUDA __dp4a)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// Q8_K q8 buffer (the only call site) starts at block offset 4 (after
// fp32 yd), so q8+0 and q8+4 are both 4-aligned. memcpy folds to a
// single dword load.
__device__ __forceinline__ int32_t load_dword(const int8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

// Expand a ksigns byte (already odd-parity-corrected by table lookup) into
// 4 byte-mask values where each byte is 0xff if the corresponding sign bit
// is set, 0x00 otherwise. Mirrors __vcmpne4(s & m, 0) followed by use.
__device__ __forceinline__ uint32_t expand_signs(uint32_t s, uint32_t mask) {
    const uint32_t m = s & mask;
    uint32_t out = 0;
    if (m & 0xffu)         out |= 0xffu;
    if (m & 0xff00u)       out |= 0xff00u;
    if (m & 0xff0000u)     out |= 0xff0000u;
    if (m & 0xff000000u)   out |= 0xff000000u;
    return out;
}

// Per-byte conditional negation: for each of 4 bytes in `pack`, if the
// corresponding byte-mask in `sm` is 0xff, negate that byte (else leave it).
// Equivalent to __vsub4(pack ^ sm, sm) in the CUDA path.
__device__ __forceinline__ int32_t cond_neg_bytes(int32_t pack, uint32_t sm) {
    union { int32_t i; int8_t b[4]; } in, out;
    in.i = pack;
    out.b[0] = (sm & 0xffu)       ? (int8_t)(-in.b[0]) : in.b[0];
    out.b[1] = (sm & 0xff00u)     ? (int8_t)(-in.b[1]) : in.b[1];
    out.b[2] = (sm & 0xff0000u)   ? (int8_t)(-in.b[2]) : in.b[2];
    out.b[3] = (sm & 0xff000000u) ? (int8_t)(-in.b[3]) : in.b[3];
    return out.i;
}

// Dot product of 8 signed-i8 weights (grid 64 bits) against 8 i8 activations.
// `sign_idx` indexes ksigns_iq2xs (0..127); the table value carries the
// extra parity bit so signed-i8 multiplication aligns with the IQ2 spec.
__device__ __forceinline__ int32_t iq2_dp4a_8(
        uint64_t grid_val, uint32_t sign_idx, const int8_t* q8, int32_t acc) {
    const uint32_t s = ksigns_iq2xs[sign_idx];
    const uint32_t signs = s * 0x01010101u;        // broadcast byte
    const uint32_t sm0 = expand_signs(signs, 0x08040201u);
    const uint32_t sm1 = expand_signs(signs, 0x80402010u);
    const int32_t g0 = cond_neg_bytes((int32_t)(uint32_t)grid_val, sm0);
    const int32_t g1 = cond_neg_bytes((int32_t)(uint32_t)(grid_val >> 32), sm1);
    acc = sudot4(g0, load_dword(q8 + 0), acc);
    acc = sudot4(g1, load_dword(q8 + 4), acc);
    return acc;
}

__device__ __forceinline__ int32_t dot_iq2_pair_16(
        uint8_t g0_idx, uint32_t s0_idx,
        uint8_t g1_idx, uint32_t s1_idx,
        const int8_t* q8) {
    int32_t sum = 0;
    sum = iq2_dp4a_8(iq2xxs_grid[g0_idx], s0_idx, q8 + 0, sum);
    sum = iq2_dp4a_8(iq2xxs_grid[g1_idx], s1_idx, q8 + 8, sum);
    return sum;
}

// Single-block dot product: one block_iq2_xxs × one block_q8_K → f32 partial.
__device__ __forceinline__ float dot_block(
        const uint8_t* w_blk, const uint8_t* y_blk) {
    const uint16_t wd = *(const uint16_t*)(w_blk + 0);
    const uint16_t* q2 = (const uint16_t*)(w_blk + 2);
    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8 = (const int8_t*)(y_blk + 4);
    const float d = f16_to_f32(wd) * yd;

    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        int32_t sumi = 0;
        sumi += dot_iq2_pair_16(
            (uint8_t)(aux0 & 0xffu),         (aux1 >>  0) & 127u,
            (uint8_t)((aux0 >>  8) & 0xffu), (aux1 >>  7) & 127u,
            q8 + 0);
        sumi += dot_iq2_pair_16(
            (uint8_t)((aux0 >> 16) & 0xffu), (aux1 >> 14) & 127u,
            (uint8_t)((aux0 >> 24) & 0xffu), (aux1 >> 21) & 127u,
            q8 + 16);
        bsum += sumi * (int32_t)ls;
        q8 += 32;
    }
    return 0.125f * d * (float)bsum;
}

__device__ __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// Paired matvec: gate[row] and up[row] for r in [0, n_rows), sharing xq.
// gate_w, up_w: row-major [n_rows, n_blocks] of block_iq2_xxs (66 B each).
// xq: [n_blocks] of block_q8_K (292 B each).
//
// Launch:
//   grid:  (n_rows / 8, 1, 1)
//   block: (256, 1, 1)  [8 rows × 32 lanes per warp]
extern "C" __global__ void iq2_xxs_pair_matvec(
        float        * __restrict__ gate,
        float        * __restrict__ up,
        const uint8_t * __restrict__ gate_w,
        const uint8_t * __restrict__ up_w,
        const uint8_t * __restrict__ xq,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const uint8_t* gate_row_base = gate_w + (uint64_t)row * n_blocks * BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_row_base   = up_w   + (uint64_t)row * n_blocks * BLOCK_IQ2_XXS_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;

    // Each lane sums every (lane + 32k)-th super-block; with n_blocks ≤ 16
    // (in_dim ≤ 4096), most lanes process at most 1 block.
    for (unsigned int b = lane; b < n_blocks; b += 32u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* y   = xq            + (uint64_t)b * BLOCK_Q8_K_BYTES;
        g_sum += dot_block(w_g, y);
        u_sum += dot_block(w_u, y);
    }

    g_sum = warp_sum_f32(g_sum);
    u_sum = warp_sum_f32(u_sum);

    if (lane == 0) {
        gate[row] = g_sum;
        up[row]   = u_sum;
    }
}
