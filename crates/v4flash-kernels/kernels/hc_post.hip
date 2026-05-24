// hc_post — combine sublayer output (block_out) with the HC residual via
// the per-dst post[dst] scale and per-(dst,src) comb mixing matrix.
// Mirrors ds4 hc_post_one (ds4.c:4400):
//
//   for dst in 0..n_hc, d in 0..n_embd:
//     acc = block_out[d] * post[dst]
//     for src in 0..n_hc:
//       acc += comb[dst + src*n_hc] * residual_hc[src*n_embd + d]
//     out_hc[dst*n_embd + d] = acc
//
// One thread per output element. Grid (n_embd/block, n_hc).

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_post(
        float        * __restrict__ out_hc,        // [n_hc, n_embd]
        const float  * __restrict__ block_out,     // [n_embd]
        const float  * __restrict__ residual_hc,   // [n_hc, n_embd]
        const float  * __restrict__ post,          // [n_hc]
        const float  * __restrict__ comb,          // [n_hc, n_hc] addressed [dst + src*n_hc]
        unsigned int   n_embd,
        unsigned int   n_hc) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const unsigned int dst = blockIdx.y;

    float acc = block_out[d] * post[dst];
    for (unsigned int src = 0; src < n_hc; src++) {
        acc += comb[dst + src * n_hc]
             * residual_hc[(unsigned long long)src * n_embd + d];
    }
    out_hc[(unsigned long long)dst * n_embd + d] = acc;
}
