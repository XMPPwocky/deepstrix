// q6_k_dense_matvec — dense (non-MoE) Q6_K weight × f32 activation GEMV.
//
// For the Laguna model port (Q4_K_M): the LM head (`output.weight`) and some
// `ffn_down` / `attn_v` tensors are stored as Q6_K. No Q6_K kernel existed;
// this provides the plain dense path: out[i] = sum_k dequant(W[i,k]) * x[k].
//
// CORRECTNESS-FIRST: no activation quantization (x stays f32), no WMMA, no
// dp4a, no fancy tiling. Each warp dequantizes its weight superblocks to
// f32 and does a scalar f32 multiply-accumulate against x. Perf comes later.
//
// Q6_K block layout (210 bytes per 256 elements) — identical to
// llama.cpp/ggml `block_q6_K`:
//   offset   0: uint8  ql[128]     (lower 4 bits of quants)
//   offset 128: uint8  qh[64]      (upper 2 bits of quants)
//   offset 192: int8   scales[16]  (per-16-element sub-block scales, signed)
//   offset 208: ggml_half d        (super-block scale, f16) -> 2 bytes
//                                   total 128+64+16+2 = 210 bytes
//
// Dequant (matches ggml dequantize_row_q6_K exactly): each superblock is two
// 128-element halves. For half h in {0,1} (ql += 64*h, qh += 32*h, sc += 8*h,
// y += 128*h) and l in [0,32):
//   is = l / 16
//   q1 = ((ql[l   ] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
//   q2 = ((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
//   q3 = ((ql[l   ] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32
//   q4 = ((ql[l+32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32
//   y[l +  0] = d * sc[is + 0] * q1
//   y[l + 32] = d * sc[is + 2] * q2
//   y[l + 64] = d * sc[is + 4] * q3
//   y[l + 96] = d * sc[is + 6] * q4
// so weight element k within the superblock maps directly to x[k].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q6_K_BYTES 210
#define WARP_LANES 32u
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float f16_to_f32_q6kd(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

// Dequantize one 256-element Q6_K superblock and dot it with 256 f32
// activations. Pure scalar f32 accumulate — correctness first.
__device__ __forceinline__ float dot_super_q6kd_f32(
        const uint8_t* blk, const float* x) {
    const uint8_t* ql   = blk + 0;
    const uint8_t* qh   = blk + 128;
    const int8_t*  sc   = (const int8_t*)(blk + 192);
    const uint16_t d_bits = *(const uint16_t*)(blk + 208);
    const float d = f16_to_f32_q6kd(d_bits);

    float sum = 0.0f;
    #pragma unroll
    for (int h = 0; h < 2; h++) {  // two 128-element halves per superblock
        const uint8_t* qlh = ql + (size_t)h * 64u;
        const uint8_t* qhh = qh + (size_t)h * 32u;
        const int8_t*  sch = sc + (size_t)h * 8u;
        const float*   xh  = x  + (size_t)h * 128u;
        #pragma unroll
        for (int l = 0; l < 32; l++) {
            const int is = l >> 4;  // l / 16
            const uint8_t lo = qlh[l];
            const uint8_t hi = qlh[l + 32];
            const uint8_t hb = qhh[l];
            const int q1 = (int)((lo & 0x0F) | (((hb >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((hi & 0x0F) | (((hb >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((lo  >> 4)  | (((hb >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((hi  >> 4)  | (((hb >> 6) & 3) << 4)) - 32;
            const float w1 = d * (float)sch[is + 0] * (float)q1;
            const float w2 = d * (float)sch[is + 2] * (float)q2;
            const float w3 = d * (float)sch[is + 4] * (float)q3;
            const float w4 = d * (float)sch[is + 6] * (float)q4;
            sum += w1 * xh[l +  0];
            sum += w2 * xh[l + 32];
            sum += w3 * xh[l + 64];
            sum += w4 * xh[l + 96];
        }
    }
    return sum;
}

__device__ __forceinline__ float warp_sum_f32_q6kd(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// Dense Q6_K GEMV: out[i] = sum_k dequant(W[i,k]) * x[k].
// Weight is [n_rows, K] Q6_K, row-major, row pitch = n_super*210 bytes.
// Activation x is [K] f32. K must be a multiple of 256.
//
// Layout mirrors q4_k_dense_gemv: 8 rows per workgroup, one warp (32 lanes)
// per output row. Lanes stride over the row's superblocks; each lane
// dequantizes whole superblocks and accumulates, then a warp reduce sums
// the partials.
extern "C" __global__ void q6_k_dense_gemv(
        float* __restrict__ out,           // [n_rows]
        const uint8_t* __restrict__ w,     // [n_rows * n_super * 210]
        const float* __restrict__ x,       // [K]
        uint32_t in_dim,                   // K
        uint32_t out_dim,                  // n_rows
        uint32_t n_super)                  // K / 256
{
    const uint32_t row  = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const uint8_t* wr = w + (size_t)row * n_super * BLOCK_Q6_K_BYTES;
    float acc = 0.0f;
    for (uint32_t b = lane; b < n_super; b += WARP_LANES) {
        const uint8_t* blk = wr + (size_t)b * BLOCK_Q6_K_BYTES;
        const float*   xb  = x  + (size_t)b * QK_K;
        acc += dot_super_q6kd_f32(blk, xb);
    }

    acc = warp_sum_f32_q6kd(acc);
    if (lane == 0) out[row] = acc;
}

// Batched dense Q6_K GEMV — grid.z = B parallel single-token kernels.
// Same per-row math as q6_k_dense_gemv; weight is shared across the batch,
// each batch element has its own x[B,K] and out[B,n_rows] slice.
extern "C" __global__ void q6_k_dense_gemv_batched(
        float* __restrict__ out,           // [B, n_rows]
        const uint8_t* __restrict__ w,     // [n_rows * n_super * 210] — shared
        const float* __restrict__ x,       // [B, K]
        uint32_t in_dim,                   // K
        uint32_t out_dim,                  // n_rows
        uint32_t n_super)                  // K / 256
{
    const uint32_t row  = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t bt   = blockIdx.z;   // batch element
    if (row >= out_dim) return;

    const uint8_t* wr = w + (size_t)row * n_super * BLOCK_Q6_K_BYTES;
    const float*   xb = x + (size_t)bt * in_dim;
    float acc = 0.0f;
    for (uint32_t b = lane; b < n_super; b += WARP_LANES) {
        const uint8_t* blk = wr + (size_t)b * BLOCK_Q6_K_BYTES;
        const float*   xs  = xb + (size_t)b * QK_K;
        acc += dot_super_q6kd_f32(blk, xs);
    }

    acc = warp_sum_f32_q6kd(acc);
    if (lane == 0) out[(size_t)bt * out_dim + row] = acc;
}
