// swiglu — out[i] = silu(gate[i]) * up[i]. Mirrors ds4.c:5083 swiglu().
// Used by both shared expert (no clamp) and routed experts (with clamp,
// handled by separate kernel variant when needed).

#include <hip/hip_runtime.h>

extern "C" __global__ void swiglu(
        float        * __restrict__ out,
        const float  * __restrict__ gate,
        const float  * __restrict__ up,
        unsigned int   n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float sig = 1.0f / (1.0f + expf(-g));
    out[i] = g * sig * up[i];
}
