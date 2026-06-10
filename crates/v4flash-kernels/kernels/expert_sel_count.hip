// M62 — expert-selection histogram accumulator.
//
// counts[layer_off + sel[i]] += 1 for every (token, slot) router pick.
// Launched on de.compute right after router top-k (the dGPU holds
// d_selected natively in both prefill and decode), accumulating into a
// persistent per-bank u32[N_LAYER × N_EXPERT] buffer. Never read back on
// the hot path — the server harvests + zeros the bank at the existing
// snapshot-save points. Cost: B×n_used atomics per layer (6144 at B=1024)
// — noise.
//
// `layer_off` = layer × N_EXPERT, precomputed on the host so the kernel
// stays a pure histogram over one bank.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void expert_sel_count(
        unsigned int * __restrict__ counts,     // [n_layer * n_expert] bank
        const int    * __restrict__ d_selected, // [total] flat (B × n_used)
        unsigned int   layer_off,               // layer * n_expert
        unsigned int   n_expert,
        unsigned int   total) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int e = d_selected[i];
    if (e < 0 || (unsigned int)e >= n_expert) return;
    atomicAdd(&counts[layer_off + (unsigned int)e], 1u);
}
