// q4_k_dense_matvec — dense (non-MoE) Q4_K weight × f32 activation GEMV.
//
// For the Laguna model port: attention Q/K/V/O projections are stored as
// dense Q4_K weights. The existing Q4_K kernel (q4_k_matvec_par.hip) is
// MoE-batched-only (expert-indexed, Q8_K activations). This provides the
// plain dense path: out[i] = sum_k dequant(W[i,k]) * x[k].
//
// CORRECTNESS-FIRST: no activation quantization (x stays f32), no WMMA, no
// dp4a, no fancy tiling. Each warp dequantizes its weight superblocks to
// f32 and does a scalar f32 multiply-accumulate against x. Perf comes later.
//
// Q4_K block layout (144 bytes per 256 elements) — identical to
// llama.cpp/ggml `block_q4_K` and to q4_k_matvec_par.hip:
//   offset  0: ggml_half d       (super-block scale for sub-block scales)
//   offset  2: ggml_half dmin    (super-block scale for sub-block mins)
//   offset  4: uint8 scales[12]  (packed 6-bit (d_sub, m_sub) for 8 sub-blocks)
//   offset 16: uint8 qs[128]     (4-bit packed weights; 2 weights per byte)
//
// Dequant (matches ggml dequantize_row_q4_K exactly): for each of the four
// 64-element groups g in [0,4), q = qs + 32*g,
//   get_scale_min_k4(2g,   scales) -> (sc1,m1);  d1 = d*sc1;  min1 = dmin*m1
//   get_scale_min_k4(2g+1, scales) -> (sc2,m2);  d2 = d*sc2;  min2 = dmin*m2
//   y[64g +  l] = d1 * (q[l] & 0xF) - min1     for l in [0,32)
//   y[64g + 32 + l] = d2 * (q[l] >> 4)  - min2  for l in [0,32)
// so weight element k within the superblock maps directly to x[k].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q4_K_BYTES 144
#define WARP_LANES 32u
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float f16_to_f32_q4kd(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

// Decode 6-bit (d_sub, m_sub) for sub-block j in [0, 8). Matches ggml's
// get_scale_min_k4 exactly (same as q4_k_matvec_par.hip).
__device__ __forceinline__ void get_scale_min_q4kd(
        int j, const uint8_t* scales, uint8_t* d_out, uint8_t* m_out) {
    if (j < 4) {
        *d_out = scales[j]     & 0x3F;
        *m_out = scales[j + 4] & 0x3F;
    } else {
        *d_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m_out = (scales[j + 4] >>  4)  | ((scales[j - 0] >> 6) << 4);
    }
}

// Dequantize one 256-element Q4_K superblock and dot it with 256 f32
// activations. Pure scalar f32 accumulate — correctness first.
__device__ __forceinline__ float dot_super_q4kd_f32(
        const uint8_t* blk, const float* x) {
    const uint16_t d_bits    = *(const uint16_t*)(blk + 0);
    const uint16_t dmin_bits = *(const uint16_t*)(blk + 2);
    const uint8_t* scales     = blk + 4;
    const uint8_t* qs         = blk + 16;

    const float d    = f16_to_f32_q4kd(d_bits);
    const float dmin = f16_to_f32_q4kd(dmin_bits);

    float sum = 0.0f;
    #pragma unroll
    for (int g = 0; g < QK_K / 64; g++) {  // 4 groups of 64 elements
        uint8_t sc, m;
        get_scale_min_q4kd(2 * g, scales, &sc, &m);
        const float d1 = d * (float)sc;
        const float min1 = dmin * (float)m;
        get_scale_min_q4kd(2 * g + 1, scales, &sc, &m);
        const float d2 = d * (float)sc;
        const float min2 = dmin * (float)m;

        const uint8_t* q = qs + (size_t)g * 32u;
        const float* xlo = x + (size_t)g * 64u;
        const float* xhi = xlo + 32u;
        #pragma unroll
        for (int l = 0; l < 32; l++) {
            const float wlo = d1 * (float)(q[l] & 0x0F) - min1;
            const float whi = d2 * (float)(q[l] >> 4)   - min2;
            sum += wlo * xlo[l];
            sum += whi * xhi[l];
        }
    }
    return sum;
}

__device__ __forceinline__ float warp_sum_f32_q4kd(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// Dense Q4_K GEMV: out[i] = sum_k dequant(W[i,k]) * x[k].
// Weight is [n_rows, K] Q4_K, row-major, row pitch = n_super*144 bytes.
// Activation x is [K] f32. K must be a multiple of 256.
//
// Layout mirrors q8_0_gemv_warp8: 8 rows per workgroup, one warp (32 lanes)
// per output row. Lanes stride over the row's superblocks; each lane
// dequantizes whole superblocks and accumulates, then a warp reduce sums
// the partials. (For small n_super only a few lanes are active — fine for
// a correctness-first kernel.)
extern "C" __global__ void q4_k_dense_gemv(
        float* __restrict__ out,           // [n_rows]
        const uint8_t* __restrict__ w,     // [n_rows * n_super * 144]
        const float* __restrict__ x,       // [K]
        uint32_t in_dim,                   // K
        uint32_t out_dim,                  // n_rows
        uint32_t n_super)                  // K / 256
{
    const uint32_t row  = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const uint8_t* wr = w + (size_t)row * n_super * BLOCK_Q4_K_BYTES;
    float acc = 0.0f;
    for (uint32_t b = lane; b < n_super; b += WARP_LANES) {
        const uint8_t* blk = wr + (size_t)b * BLOCK_Q4_K_BYTES;
        const float*   xb  = x  + (size_t)b * QK_K;
        acc += dot_super_q4kd_f32(blk, xb);
    }

    acc = warp_sum_f32_q4kd(acc);
    if (lane == 0) out[row] = acc;
}

// Batched dense Q4_K GEMV — grid.z = B parallel single-token kernels.
// Same per-row math as q4_k_dense_gemv; weight is shared across the batch,
// each batch element has its own x[B,K] and out[B,n_rows] slice.
extern "C" __global__ void q4_k_dense_gemv_batched(
        float* __restrict__ out,           // [B, n_rows]
        const uint8_t* __restrict__ w,     // [n_rows * n_super * 144] — shared
        const float* __restrict__ x,       // [B, K]
        uint32_t in_dim,                   // K
        uint32_t out_dim,                  // n_rows
        uint32_t n_super)                  // K / 256
{
    const uint32_t row  = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t bt   = blockIdx.z;   // batch element
    if (row >= out_dim) return;

    const uint8_t* wr = w + (size_t)row * n_super * BLOCK_Q4_K_BYTES;
    const float*   xb = x + (size_t)bt * in_dim;
    float acc = 0.0f;
    for (uint32_t b = lane; b < n_super; b += WARP_LANES) {
        const uint8_t* blk = wr + (size_t)b * BLOCK_Q4_K_BYTES;
        const float*   xs  = xb + (size_t)b * QK_K;
        acc += dot_super_q4kd_f32(blk, xs);
    }

    acc = warp_sum_f32_q4kd(acc);
    if (lane == 0) out[(size_t)bt * out_dim + row] = acc;
}
