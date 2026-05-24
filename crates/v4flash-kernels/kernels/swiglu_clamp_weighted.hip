// swiglu_clamp_weighted — per-expert SwiGLU with one-sided clamp on gate,
// two-sided clamp on up, multiplied by an expert weight scalar. Mirrors
// the inner of matvec_iq2_xxs_mid_worker (ds4.c:3868-3873):
//
//   if (clamp > 1e-6) {
//       if (gate > clamp)  gate = clamp;
//       if (up   > clamp)  up   = clamp;
//       if (up  < -clamp)  up   = -clamp;
//   }
//   mid[expert*ff_exp + r] = silu(gate) * up * expert_weight[expert]
//
// Launch: one thread per (expert, ff_row). Grid (ff_exp/block, n_experts).

#include <hip/hip_runtime.h>

extern "C" __global__ void swiglu_clamp_weighted(
        float        * __restrict__ mid,        // [n_experts, ff_exp]
        const float  * __restrict__ gate,       // [n_experts, ff_exp]
        const float  * __restrict__ up,         // [n_experts, ff_exp]
        const float  * __restrict__ expert_w,   // [n_experts]
        float          clamp,
        unsigned int   ff_exp) {
    const unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= ff_exp) return;
    const unsigned int e = blockIdx.y;
    const unsigned int idx = e * ff_exp + r;

    float g = gate[idx];
    float u = up[idx];
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    const float sig = 1.0f / (1.0f + expf(-g));
    mid[idx] = g * sig * u * expert_w[e];
}
