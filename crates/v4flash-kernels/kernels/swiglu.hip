// swiglu — out[i] = silu(gate[i]) * up[i], with the ds4 swiglu_limit
// clamp. Mirrors ds4.c swiglu() post-5bc1e6d ("Flash graph correctness"):
//
//   if (clamp > 1e-6) {
//       if (gate > clamp)  gate = clamp;     // one-sided on gate
//       if (up   > clamp)  up   = clamp;     // two-sided on up
//       if (up  < -clamp)  up   = -clamp;
//   }
//   out[i] = silu(gate) * up
//
// The official V4-Flash graph applies the same swiglu_limit clamp to
// shared experts as to routed experts (DS4_SWIGLU_CLAMP_EXP = 10.0).
// clamp <= 1e-6 disables clamping (used by Laguna, which has no limit).

#include <hip/hip_runtime.h>

extern "C" __global__ void swiglu(
        float        * __restrict__ out,
        const float  * __restrict__ gate,
        const float  * __restrict__ up,
        unsigned int   n,
        float          clamp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    const float sig = 1.0f / (1.0f + expf(-g));
    out[i] = g * sig * u;
}
