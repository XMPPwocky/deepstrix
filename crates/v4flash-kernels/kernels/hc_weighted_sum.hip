// hc_weighted_sum — out[d] = sum_h x[h*n_embd + d] * weights[h].
// Mirrors ds4 hc_weighted_sum_one (ds4.c:4301). n_hc is small (DS4_N_HC = 4);
// n_embd is large (DS4_N_EMBD = 4096) for the head, hc_dim/n_hc for hc_pre.
// One thread per output element; weights cached in registers since n_hc <= 4.

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_weighted_sum(
        float        * __restrict__ out,
        const float  * __restrict__ x,
        const float  * __restrict__ weights,
        unsigned int   n_embd,
        unsigned int   n_hc) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    float acc = 0.0f;
    for (unsigned int h = 0; h < n_hc; h++) {
        acc += x[(unsigned long long)h * n_embd + d] * weights[h];
    }
    out[d] = acc;
}

// M50 Phase 2: batched. x[B, n_hc, n_embd], weights[B, n_hc], out[B, n_embd].
// Grid (n_embd/256, 1, B).
extern "C" __global__ void hc_weighted_sum_batched(
        float        * __restrict__ out,
        const float  * __restrict__ x,
        const float  * __restrict__ weights,
        unsigned int   n_embd,
        unsigned int   n_hc) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int b = blockIdx.z;
    if (d >= n_embd) return;
    const float *xb = x + (size_t)b * (size_t)n_hc * (size_t)n_embd;
    const float *wb = weights + (size_t)b * (size_t)n_hc;
    float acc = 0.0f;
    for (unsigned int h = 0; h < n_hc; h++) {
        acc += xb[(size_t)h * (size_t)n_embd + d] * wb[h];
    }
    out[(size_t)b * (size_t)n_embd + d] = acc;
}
