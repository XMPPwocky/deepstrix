#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Q8_0 GEMV — single-token, int8 SIMD inner loop.
//
// Approach adapted from antirez/ds4 ds4_cuda.cu:
//   - Weights stored as Q8_0: fp16 scale + 32 int8 quants per 32-element block
//   - Input pre-quantized to int8 + per-block fp32 scale (mirrors the
//     production pipeline so we measure the same memory pattern as Phase 1)
//   - Inner loop: int8×int8 → int32 dot product via __dp4a (RDNA 3.5/4)
//   - Each warp (32 lanes) computes one output row; 8 warps per workgroup
//
// Bandwidth-bound on weight reads. For M=4096 K=2048:
//   weight_bytes = 4096 * 2048 / 32 * 34 = 8,912,896 B ≈ 8.5 MiB.

#define WARP_LANES 32u
// AMD HIP requires a 64-bit mask for __shfl_down_sync (supports wave32 + wave64).
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ int32_t load_i8x4_aligned(const int8_t* p) {
    return *(const int32_t*)p;
}

__device__ __forceinline__ int32_t load_i8x4_unaligned(const int8_t* p) {
    const uint8_t* u = (const uint8_t*)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

// AMD-specific 4-byte int8 SIMD dot-product. RDNA 3.5 + RDNA 4 both
// expose this via `sudot4` (the `sdot4` form is older AMDs only).
// Mirrors ggml's ggml_cuda_dp4a path for RDNA3/4.
__device__ __forceinline__ int32_t dp4a_sudot(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t dot_i8x32(const int8_t* a, const int8_t* b) {
    int32_t dot = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        // a (weight quants) may be unaligned by 2 bytes due to the fp16
        // scale prefix in each Q8_0 block. b (pre-quantized input) is
        // 4-aligned by construction.
        dot = dp4a_sudot(load_i8x4_unaligned(a + i), load_i8x4_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// 8 rows per workgroup, 32 lanes per row.
extern "C" __global__ void q8_0_gemv_warp8(
    float* __restrict__ out,           // [M]
    const unsigned char* __restrict__ w,  // [M * blocks * 34]
    const int8_t* __restrict__ xq,       // [K] = [blocks * 32]
    const float* __restrict__ xscale,    // [blocks]
    uint32_t in_dim,                    // K
    uint32_t out_dim,                   // M
    uint32_t blocks)                    // K / 32
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const unsigned char* wr = w + (size_t)row * blocks * 34u;
    float acc = 0.0f;

    for (uint32_t b = lane; b < blocks; b += WARP_LANES) {
        const __half* scale_h = (const __half*)(wr + (size_t)b * 34u);
        const int8_t* qs = (const int8_t*)(wr + (size_t)b * 34u + 2u);
        const int8_t* xqb = xq + (size_t)b * 32u;
        int32_t dot = dot_i8x32(qs, xqb);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }

    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

// Quantize FP32 input → int8 + per-block fp32 scale. Run once per token
// before the GEMV; same approach as ds4 quantize_q8_0_f32_kernel.
extern "C" __global__ void q8_0_quantize_f32(
    int8_t* __restrict__ xq,
    float* __restrict__ xscale,
    const float* __restrict__ x,
    uint32_t blocks)
{
    uint32_t b = blockIdx.x;
    if (b >= blocks) return;

    const float* xb = x + (size_t)b * 32u;
    float amax = 0.0f;
    for (uint32_t i = 0; i < 32u; i++) {
        amax = fmaxf(amax, fabsf(xb[i]));
    }
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;

    if (threadIdx.x == 0) xscale[b] = d;

    int8_t* xqb = xq + (size_t)b * 32u;
    for (uint32_t i = 0; i < 32u; i++) {
        int q = (int)lrintf(xb[i] * id);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        xqb[i] = (int8_t)q;
    }
}
