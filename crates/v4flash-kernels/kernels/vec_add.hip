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
