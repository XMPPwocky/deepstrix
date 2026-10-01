// vec_add — out[i] += rhs[i]. In-place accumulate. Used to combine
// ffn_moe + ffn_shared inside the forward orchestrator.

#include <hip/hip_runtime.h>

extern "C" __global__ void vec_add_inplace(
        float        * __restrict__ out,
        const float  * __restrict__ rhs,
        unsigned int   n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] += rhs[i];
}

// gather_f32 -- out[i] = src[idx[i]] for i < n (a negative idx yields -inf).
// The DSpark exit fetches its top-M candidate logits with it, so only the
// M (id, logit) pairs of each draft position cross to the host.
extern "C" __global__ void gather_f32(
        float        * __restrict__ out,
        const float  * __restrict__ src,
        const int    * __restrict__ idx,
        unsigned int   n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int j = idx[i];
    out[i] = j >= 0 ? src[j] : -INFINITY;
}
