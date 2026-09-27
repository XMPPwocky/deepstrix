// Round 2 item (c): grid.z = B -> grid.z = ceil(B / NB) for f16_matvec_batched.
// Production f16_matvec_batched: grid (ceil(n_rows/8), 1, B) x 256, one warp per (row, b), each
// warp re-reads its weight row (B times per row in total). Here one warp owns a row for up to NB
// batch rows: every weight element is loaded ONCE per warp and feeds NB accumulators.
// Bit-exact by construction: per (row, b) the accumulation is the production one -- one f32
// accumulator, elements i = lane + 32 j in ascending j, the same `acc += h2f(w) * x` expression,
// then the same shfl_down tree (f16_warp_sum_f32). f16_matvec_batched_h20 has the same per-lane
// order, so these are also bit-identical to the router kernel.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define F16_WARP_LANES 32u
#define F16_FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float z_warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(F16_FULL_WARP_MASK, v, offset);
    return v;
}

// Plain: one element per lane per step, NB accumulators (nb <= NB valid).
template <unsigned NB>
__device__ __forceinline__ void z_plain(float* __restrict__ out, const __half* __restrict__ weight,
                                        const float* __restrict__ x, unsigned k, unsigned n_rows, unsigned batch) {
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= n_rows) return;
    const uint32_t b0 = blockIdx.z * NB;
    if (b0 >= batch) return;
    const uint32_t nb = min(NB, batch - b0);
    const __half* wr = weight + (size_t)row * k;
    const float* xb = x + (size_t)b0 * k;
    float acc[NB];
    #pragma unroll
    for (unsigned j = 0; j < NB; ++j) acc[j] = 0.0f;
    for (uint32_t i = lane; i < k; i += F16_WARP_LANES) {
        const float w = __half2float(wr[i]);
        #pragma unroll
        for (unsigned j = 0; j < NB; ++j)
            if (j < nb) acc[j] += w * xb[(size_t)j * k + i];
    }
    #pragma unroll
    for (unsigned j = 0; j < NB; ++j) {
        if (j < nb) {
            const float s = z_warp_sum_f32(acc[j]);
            if (lane == 0) out[(size_t)(b0 + j) * n_rows + row] = s;
        }
    }
}

// Hoisted: when k % (32 * CH) == 0, CH weight elements per lane are loaded into registers, then
// for every batch row the CH matching x values are loaded and FMA'd in ascending element order
// (the per-accumulator sequence is unchanged). Other k fall back to z_plain's loop.
template <unsigned NB, unsigned CH>
__device__ __forceinline__ void z_hoist(float* __restrict__ out, const __half* __restrict__ weight,
                                        const float* __restrict__ x, unsigned k, unsigned n_rows, unsigned batch) {
    if (k % (F16_WARP_LANES * CH) != 0u) { z_plain<NB>(out, weight, x, k, n_rows, batch); return; }
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= n_rows) return;
    const uint32_t b0 = blockIdx.z * NB;
    if (b0 >= batch) return;
    const uint32_t nb = min(NB, batch - b0);
    const __half* wr = weight + (size_t)row * k;
    const float* xb = x + (size_t)b0 * k;
    float acc[NB];
    #pragma unroll
    for (unsigned j = 0; j < NB; ++j) acc[j] = 0.0f;
    const uint32_t nchunks = k / (F16_WARP_LANES * CH);
    __half wv[CH];
    for (uint32_t c = 0; c < nchunks; ++c) {
        const uint32_t base = c * (F16_WARP_LANES * CH) + lane;
        #pragma unroll
        for (unsigned t = 0; t < CH; ++t) wv[t] = wr[base + F16_WARP_LANES * t];
        #pragma unroll
        for (unsigned j = 0; j < NB; ++j) {
            if (j < nb) {
                float xv[CH];
                #pragma unroll
                for (unsigned t = 0; t < CH; ++t) xv[t] = xb[(size_t)j * k + base + F16_WARP_LANES * t];
                #pragma unroll
                for (unsigned t = 0; t < CH; ++t) acc[j] += __half2float(wv[t]) * xv[t];
            }
        }
    }
    #pragma unroll
    for (unsigned j = 0; j < NB; ++j) {
        if (j < nb) {
            const float s = z_warp_sum_f32(acc[j]);
            if (lane == 0) out[(size_t)(b0 + j) * n_rows + row] = s;
        }
    }
}

#define ZP(NB) \
extern "C" __global__ __launch_bounds__(256) void f16_mv_z_plain_n##NB(float* __restrict__ out, const __half* __restrict__ w, \
        const float* __restrict__ x, unsigned k, unsigned n_rows, unsigned batch) { z_plain<NB>(out, w, x, k, n_rows, batch); }
#define ZH(NB, CH) \
extern "C" __global__ __launch_bounds__(256) void f16_mv_z_h##CH##_n##NB(float* __restrict__ out, const __half* __restrict__ w, \
        const float* __restrict__ x, unsigned k, unsigned n_rows, unsigned batch) { z_hoist<NB, CH>(out, w, x, k, n_rows, batch); }

ZP(1) ZP(2) ZP(4) ZP(8) ZP(16)
ZH(1, 8) ZH(2, 8) ZH(4, 8) ZH(8, 8) ZH(16, 8)
ZH(1, 4) ZH(4, 4) ZH(16, 4)
ZH(4, 16)
