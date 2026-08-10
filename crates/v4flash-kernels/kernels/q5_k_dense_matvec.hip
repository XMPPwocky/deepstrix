// q5_k_dense_matvec — dense Q5_K weight × f32 activation GEMV.
//
// unsloth UD mix: ffn_{gate,up}_shexp + attn_q_a are Q5_K (Q8_0 in the
// antirez mix). Clone of q4_k_dense_matvec with the 5th bit plane (qh)
// added; scales/mins packing is identical to Q4_K.
//
// Q5_K block layout (176 bytes per 256 elements) — llama.cpp block_q5_K:
//   offset   0: ggml_half d
//   offset   2: ggml_half dmin
//   offset   4: uint8 scales[12]   (packed 6-bit (d_sub, m_sub) × 8)
//   offset  16: uint8 qh[32]       (high bits: bit (2g[+1]) of qh[l])
//   offset  48: uint8 qs[128]      (low 4 bits packed, 2 weights/byte)
//
// Dequant (matches ggml dequantize_row_q5_K):
//   group g in [0,4): q = qs + 32g, u1 = 1<<2g, u2 = 1<<(2g+1)
//     y[64g +  l]     = d1 * ((q[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - min1
//     y[64g + 32 + l] = d2 * ((q[l] >> 4)  + (qh[l] & u2 ? 16 : 0)) - min2

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q5_K_BYTES 176
#define WARP_LANES 32u
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float f16_to_f32_q5kd(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ void get_scale_min_q5kd(
        int j, const uint8_t* scales, uint8_t* d_out, uint8_t* m_out) {
    if (j < 4) {
        *d_out = scales[j]     & 0x3F;
        *m_out = scales[j + 4] & 0x3F;
    } else {
        *d_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m_out = (scales[j + 4] >>  4)  | ((scales[j - 0] >> 6) << 4);
    }
}

__device__ __forceinline__ float dot_super_q5kd_f32(
        const uint8_t* blk, const float* x) {
    const uint16_t d_bits    = *(const uint16_t*)(blk + 0);
    const uint16_t dmin_bits = *(const uint16_t*)(blk + 2);
    const uint8_t* scales     = blk + 4;
    const uint8_t* qh         = blk + 16;
    const uint8_t* qs         = blk + 48;

    const float d    = f16_to_f32_q5kd(d_bits);
    const float dmin = f16_to_f32_q5kd(dmin_bits);

    float sum = 0.0f;
    #pragma unroll
    for (int g = 0; g < QK_K / 64; g++) {
        uint8_t sc, m;
        get_scale_min_q5kd(2 * g, scales, &sc, &m);
        const float d1 = d * (float)sc;
        const float min1 = dmin * (float)m;
        get_scale_min_q5kd(2 * g + 1, scales, &sc, &m);
        const float d2 = d * (float)sc;
        const float min2 = dmin * (float)m;

        const uint8_t u1 = (uint8_t)(1u << (2 * g));
        const uint8_t u2 = (uint8_t)(1u << (2 * g + 1));
        const uint8_t* q = qs + (size_t)g * 32u;
        const float* xlo = x + (size_t)g * 64u;
        const float* xhi = xlo + 32u;
        #pragma unroll
        for (int l = 0; l < 32; l++) {
            const float wlo =
                d1 * (float)((q[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - min1;
            const float whi =
                d2 * (float)((q[l] >> 4)   + ((qh[l] & u2) ? 16 : 0)) - min2;
            sum += wlo * xlo[l];
            sum += whi * xhi[l];
        }
    }
    return sum;
}

__device__ __forceinline__ float warp_sum_f32_q5kd(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

extern "C" __global__ void q5_k_dense_gemv(
        float* __restrict__ out,           // [n_rows]
        const uint8_t* __restrict__ w,     // [n_rows * n_super * 176]
        const float* __restrict__ x,       // [K]
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_super)
{
    const uint32_t row  = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const uint8_t* wr = w + (size_t)row * n_super * BLOCK_Q5_K_BYTES;
    float acc = 0.0f;
    for (uint32_t b = lane; b < n_super; b += WARP_LANES) {
        acc += dot_super_q5kd_f32(wr + (size_t)b * BLOCK_Q5_K_BYTES,
                                  x + (size_t)b * QK_K);
    }
    acc = warp_sum_f32_q5kd(acc);
    if (lane == 0) out[row] = acc;
}

extern "C" __global__ void q5_k_dense_gemv_batched(
        float* __restrict__ out,           // [B, n_rows]
        const uint8_t* __restrict__ w,     // shared across batch
        const float* __restrict__ x,       // [B, K]
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_super)
{
    const uint32_t row  = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t bt   = blockIdx.z;
    if (row >= out_dim) return;

    const uint8_t* wr = w + (size_t)row * n_super * BLOCK_Q5_K_BYTES;
    const float*   xb = x + (size_t)bt * in_dim;
    float acc = 0.0f;
    for (uint32_t b = lane; b < n_super; b += WARP_LANES) {
        acc += dot_super_q5kd_f32(wr + (size_t)b * BLOCK_Q5_K_BYTES,
                                  xb + (size_t)b * QK_K);
    }
    acc = warp_sum_f32_q5kd(acc);
    if (lane == 0) out[(size_t)bt * out_dim + row] = acc;
}
