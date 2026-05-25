// q8_0_grouped_gemv — variant of `q8_0_gemv_warp8` where each output row
// `idx` reads its group's input slice (`g = idx / rank`). Mirrors ds4's
// `matvec_q8_0_grouped_rows_decode_scratch` (ds4.c:3618).
//
// Used by V4 Flash for `attn_output_a`: maps `attn_heads_inv_rope`
// `[n_groups=8, group_dim=4096]` → `attn_out_low` `[n_groups=8, rank=1024]`.
//
// Block of 256 threads handles 8 output rows × 32 lanes per row (same
// shape as q8_0_matvec). Per-block input quantisation (q8_0_quantize_f32)
// works unchanged for the grouped case because Q8_0 blocks are 32-element
// chunks and group_dim is divisible by 32 — the per-block absmax of each
// chunk is identical whether we quantise per-group or as a flat 32k-element
// buffer. Same for xscale.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define WARP_LANES 32u
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ int32_t gdp4a_sudot(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t gdot_i8x32(const int8_t* a, const int8_t* b) {
    // M18: 2× uint4 loads per side, 8× dp4a. See q8_0_matvec.hip for rationale.
    int32_t dot = 0;
    uint4 av0, av1, bv0, bv1;
    __builtin_memcpy(&av0, a, 16);
    __builtin_memcpy(&av1, a + 16, 16);
    __builtin_memcpy(&bv0, b, 16);
    __builtin_memcpy(&bv1, b + 16, 16);
    dot = gdp4a_sudot((int32_t)av0.x, (int32_t)bv0.x, dot);
    dot = gdp4a_sudot((int32_t)av0.y, (int32_t)bv0.y, dot);
    dot = gdp4a_sudot((int32_t)av0.z, (int32_t)bv0.z, dot);
    dot = gdp4a_sudot((int32_t)av0.w, (int32_t)bv0.w, dot);
    dot = gdp4a_sudot((int32_t)av1.x, (int32_t)bv1.x, dot);
    dot = gdp4a_sudot((int32_t)av1.y, (int32_t)bv1.y, dot);
    dot = gdp4a_sudot((int32_t)av1.z, (int32_t)bv1.z, dot);
    dot = gdp4a_sudot((int32_t)av1.w, (int32_t)bv1.w, dot);
    return dot;
}

__device__ __forceinline__ float gwarp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

extern "C" __global__ void q8_0_grouped_gemv(
        float        * __restrict__ out,         // [n_groups * rank]
        const unsigned char * __restrict__ w,    // [n_groups * rank, blocks_per_group * 34]
        const int8_t * __restrict__ xq,          // [n_groups * group_dim]
        const float  * __restrict__ xscale,      // [n_groups * blocks_per_group]
        unsigned int   group_dim,                // K per group (e.g. 4096)
        unsigned int   rank,                     // rows per group (e.g. 1024)
        unsigned int   blocks_per_group,         // group_dim / 32 (e.g. 128)
        unsigned int   n_groups)                 // e.g. 8
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;

    uint32_t g = row / rank;
    const unsigned char* wr = w + (size_t)row * (size_t)blocks_per_group * 34u;
    const int8_t* xq_g = xq + (size_t)g * (size_t)group_dim;
    const float*  xs_g = xscale + (size_t)g * (size_t)blocks_per_group;

    // M18 repacked layout: [scales(2*blocks bytes) | quants(32*blocks bytes)] per row.
    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)blocks_per_group * 2u);
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks_per_group; b += WARP_LANES) {
        const int8_t* qs = quants + (size_t)b * 32u;     // 4-byte aligned
        const int8_t* xqb = xq_g + (size_t)b * 32u;
        int32_t dot = gdot_i8x32(qs, xqb);
        acc += __half2float(scales[b]) * xs_g[b] * (float)dot;
    }

    acc = gwarp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}
