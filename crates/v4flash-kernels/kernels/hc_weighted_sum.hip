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
