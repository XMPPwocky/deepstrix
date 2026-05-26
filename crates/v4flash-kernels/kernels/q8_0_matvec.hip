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

// 2-wide pair GEMV: same shape as q8_0_gemv_warp8 above but processes TWO
// input vectors against ONE weight matrix per call. Each block read of W
// is amortized across both input columns → halves W bandwidth vs running
// the single-input kernel twice. Used by M40-P4.5 for pair-forward where
// t0 and t1 share weights but have different activations.
//
// Per row, per lane: load W block (scales[b] + 32 quants[b]) once, do TWO
// dp4a dot products (one per input column), multiply by two different
// xscale[b] values, accumulate into two scalars. Final 2× warp reduce →
// two output writes.
//
// Reg pressure: ~doubles per-lane scalars (2 accumulators, 2 input pointers,
// 2 scales) but fits comfortably in RDNA's 256 VGPR/wave32 budget.
extern "C" __global__ void q8_0_gemv_pair_warp8(
    float* __restrict__ out_a,            // [M] — output for input column A
    float* __restrict__ out_b,            // [M] — output for input column B
    const unsigned char* __restrict__ w,  // [M * blocks * 34] — shared
    const int8_t* __restrict__ xq_a,      // [K] = [blocks * 32]
    const int8_t* __restrict__ xq_b,      // [K]
    const float* __restrict__ xscale_a,   // [blocks]
    const float* __restrict__ xscale_b,   // [blocks]
    uint32_t in_dim,
    uint32_t out_dim,
    uint32_t blocks)
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const unsigned char* wr = w + (size_t)row * blocks * 34u;
    const __half*  scales = (const __half*)wr;
    const int8_t*  quants = (const int8_t*)(wr + (size_t)blocks * 2u);
    float acc_a = 0.0f;
    float acc_b = 0.0f;

    for (uint32_t b = lane; b < blocks; b += WARP_LANES) {
        const int8_t* qs   = quants + (size_t)b * 32u;
        const int8_t* xqba = xq_a + (size_t)b * 32u;
        const int8_t* xqbb = xq_b + (size_t)b * 32u;
        // Load W block ONCE — both dots reuse the same `qs` cache line.
        int32_t dot_a = dot_i8x32(qs, xqba);
        int32_t dot_b = dot_i8x32(qs, xqbb);
        float ws = __half2float(scales[b]);
        acc_a += ws * xscale_a[b] * (float)dot_a;
        acc_b += ws * xscale_b[b] * (float)dot_b;
    }

    acc_a = warp_sum_f32(acc_a);
    acc_b = warp_sum_f32(acc_b);
    if (lane == 0) {
        out_a[row] = acc_a;
        out_b[row] = acc_b;
    }
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
// Q8_0 batched GEMV — Phase 2 v0: grid.z = B parallel single-token kernels.
//
// Initial version: same per-row math as q8_0_gemv_warp8, but launched
// with grid.z = batch_size so multiple batch elements run concurrently.
// Per-batch inputs (xq, xscale) and outputs (y) live at their own
// offsets:
//
//   xq    + batch * in_dim          // [B, K] int8
//   xscale + batch * (K / 32)       // [B, K/32] f32
//   out   + batch * out_dim         // [B, M] f32
//   W     // [M * blocks * 34] — SHARED across batch
//
// This does NOT amortize W reads across batch (each WG still re-reads W
// once per row group). A v1 kernel will pack B inputs through one
// W-load — register pressure permitting.

// Grid: (out_dim / 8, 1, batch_size). Block: (256, 1, 1) — 8 warps,
// each warp computes one output row of one batch element. Reuses the
// dp4a_sudot / dot_i8x32 / warp_sum_f32 helpers defined above.
extern "C" __global__ void q8_0_gemv_batched_warp8(
    float* __restrict__ out,             // [B, M]
    const unsigned char* __restrict__ w, // [M * blocks * 34] — shared across B
    const int8_t* __restrict__ xq,       // [B, K]
    const float* __restrict__ xscale,    // [B, K/32]
    uint32_t in_dim,                     // K
    uint32_t out_dim,                    // M
    uint32_t blocks)                     // K / 32
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t b = blockIdx.z;
    if (row >= out_dim) return;

    const unsigned char* wr = w + (size_t)row * blocks * 34u;
    const __half*  scales = (const __half*)wr;
    const int8_t*  quants = (const int8_t*)(wr + (size_t)blocks * 2u);
    const int8_t* xq_b = xq + (size_t)b * in_dim;
    const float*  xscale_b = xscale + (size_t)b * blocks;

    float acc = 0.0f;
    for (uint32_t bl = lane; bl < blocks; bl += WARP_LANES) {
        const int8_t* qs = quants + (size_t)bl * 32u;
        const int8_t* xqb = xq_b + (size_t)bl * 32u;
        int32_t dot = dot_i8x32(qs, xqb);
        acc += __half2float(scales[bl]) * xscale_b[bl] * (float)dot;
    }

    acc = warp_sum_f32(acc);
    if (lane == 0) {
        out[(size_t)b * out_dim + row] = acc;
    }
}
