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
    // M18: let the compiler emit the right instruction for the target.
    // RDNA3/4 hardware supports misaligned dword loads; the prior
    // byte-by-byte assembly defeated load coalescing and locked us to
    // ~60% of dGPU memory peak on output_proj.
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

// AMD-specific 4-byte int8 SIMD dot-product. RDNA 3.5 + RDNA 4 both
// expose this via `sudot4` (the `sdot4` form is older AMDs only).
// Mirrors ggml's ggml_cuda_dp4a path for RDNA3/4.
__device__ __forceinline__ int32_t dp4a_sudot(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t dot_i8x32(const int8_t* a, const int8_t* b) {
    // M18: load 32 int8s as 2 uint4s (16 bytes each), then dp4a them 4 at
    // a time. Both `a` (post-repack quants) and `b` (input quants) are
    // 4-byte aligned. uint4 loads cut the load instruction count from 8
    // to 2 per dot product.
    int32_t dot = 0;
    uint4 av0, av1, bv0, bv1;
    __builtin_memcpy(&av0, a, 16);
    __builtin_memcpy(&av1, a + 16, 16);
    __builtin_memcpy(&bv0, b, 16);
    __builtin_memcpy(&bv1, b + 16, 16);
    dot = dp4a_sudot((int32_t)av0.x, (int32_t)bv0.x, dot);
    dot = dp4a_sudot((int32_t)av0.y, (int32_t)bv0.y, dot);
    dot = dp4a_sudot((int32_t)av0.z, (int32_t)bv0.z, dot);
    dot = dp4a_sudot((int32_t)av0.w, (int32_t)bv0.w, dot);
    dot = dp4a_sudot((int32_t)av1.x, (int32_t)bv1.x, dot);
    dot = dp4a_sudot((int32_t)av1.y, (int32_t)bv1.y, dot);
    dot = dp4a_sudot((int32_t)av1.z, (int32_t)bv1.z, dot);
    dot = dp4a_sudot((int32_t)av1.w, (int32_t)bv1.w, dot);
    return dot;
}

__device__ __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// 8 rows per workgroup, 32 lanes per row.
//
// M18 repacked layout: each row is [scales(2*blocks bytes) |
// quants(32*blocks bytes)] with no interleaving. Quants start at a
// 4-byte-aligned offset within the row, so dot_i8x32's int8x4 loads
// hit hardware-aligned dword loads instead of the byte-by-byte
// unaligned path that capped us at ~60% memory peak.
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
    const __half*  scales = (const __half*)wr;
    const int8_t*  quants = (const int8_t*)(wr + (size_t)blocks * 2u);
    float acc = 0.0f;

    for (uint32_t b = lane; b < blocks; b += WARP_LANES) {
        const int8_t* qs = quants + (size_t)b * 32u;       // 4-byte aligned
        const int8_t* xqb = xq + (size_t)b * 32u;          // 4-byte aligned
        int32_t dot = dot_i8x32(qs, xqb);
        acc += __half2float(scales[b]) * xscale[b] * (float)dot;
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
