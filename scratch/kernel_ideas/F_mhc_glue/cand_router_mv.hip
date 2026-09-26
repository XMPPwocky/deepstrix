// cand_router_mv -- the decode router matvec (f16_matvec_batched, W [384, 5120] f16, x [b, 5120]
// f32, grid (48, 1, b) x 256, one wave per output row) with the loads hoisted.
//
// Production (K/f16_matvec.hip:91): `for (i = lane; i < k; i += 32) acc += h2f(w[i]) * x[i]` --
// k is a runtime value, the loop body is one u16 load + one f32 load + one FMA, and the compiler
// keeps only a few iterations in flight: measured 22 us for 3.9 MB = 177 GB/s, 27% of the
// 640 GB/s roof (6.1 us), at b = 1..4 (graph mode, dGPU idle).
//
// Here the per-lane element sequence is UNCHANGED (i = lane + 32 j, j ascending, one f32
// accumulator, the same `h2f(w) * x` FMA expression), so the result is BIT-EXACT; only the
// schedule differs: k is required to be a multiple of 32 * CH (else the production loop runs),
// and each chunk of CH elements per lane is loaded into register arrays (CH u16 + CH f32 loads
// in flight per lane) before its CH FMAs -- with the next chunk's loads issued before the
// current chunk's FMAs (double buffer) so the memory pipe never drains.
//
// Same signature and grid as f16_matvec_batched.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define F16_WARP_LANES 32u
#define F16_FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float f16_warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(F16_FULL_WARP_MASK, v, offset);
    }
    return v;
}

template <unsigned CH>
__device__ __forceinline__ float mv_row_hoisted(const __half* __restrict__ wr, const float* __restrict__ xb,
                                                uint32_t k, uint32_t lane) {
    const uint32_t nchunks = k / (F16_WARP_LANES * CH);
    __half wv[2][CH];
    float xv[2][CH];
    auto issue = [&](uint32_t c, uint32_t slot) {
        const uint32_t base = c * (F16_WARP_LANES * CH) + lane;
        #pragma unroll
        for (unsigned j = 0; j < CH; ++j) wv[slot][j] = wr[base + F16_WARP_LANES * j];
        #pragma unroll
        for (unsigned j = 0; j < CH; ++j) xv[slot][j] = xb[base + F16_WARP_LANES * j];
    };
    float acc = 0.0f;
    issue(0, 0);
    for (uint32_t c = 0; c < nchunks; c += 2) {
        if (c + 1 < nchunks) issue(c + 1, 1);
        #pragma unroll
        for (unsigned j = 0; j < CH; ++j) acc += __half2float(wv[0][j]) * xv[0][j];
        if (c + 1 < nchunks) {
            if (c + 2 < nchunks) issue(c + 2, 0);
            #pragma unroll
            for (unsigned j = 0; j < CH; ++j) acc += __half2float(wv[1][j]) * xv[1][j];
        }
    }
    return acc;
}

template <unsigned CH>
__device__ __forceinline__ void mv_body(float* __restrict__ out, const __half* __restrict__ weight,
                                        const float* __restrict__ x, unsigned int k, unsigned int n_rows) {
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= n_rows) return;
    const uint32_t bz = blockIdx.z;
    const __half* wr = weight + (size_t)row * (size_t)k;
    const float* xb = x + (size_t)bz * (size_t)k;
    float acc;
    if (k % (F16_WARP_LANES * CH) == 0u) {
        acc = mv_row_hoisted<CH>(wr, xb, k, lane);
    } else {
        acc = 0.0f;
        for (uint32_t i = lane; i < k; i += F16_WARP_LANES) acc += __half2float(wr[i]) * xb[i];
    }
    acc = f16_warp_sum_f32(acc);
    if (lane == 0) out[(size_t)bz * (size_t)n_rows + row] = acc;
}

#define MV_KERNEL(name, CH)                                                                          \
    extern "C" __global__ __launch_bounds__(256) void name(float* __restrict__ out,                  \
                                                          const __half* __restrict__ weight,        \
                                                          const float* __restrict__ x,              \
                                                          unsigned int k, unsigned int n_rows) {    \
        mv_body<CH>(out, weight, x, k, n_rows);                                                     \
    }

MV_KERNEL(f16_matvec_batched_h20, 20)   // 8 chunks of 20 (k = 5120)
MV_KERNEL(f16_matvec_batched_h40, 40)   // 4 chunks of 40
MV_KERNEL(f16_matvec_batched_h80, 80)   // 2 chunks of 80
