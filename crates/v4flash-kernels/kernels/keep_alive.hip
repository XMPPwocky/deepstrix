// M28: clock-keep-alive. Single-block single-thread kernel that burns
// ~50µs of dummy ALU. Launched on a SECONDARY dGPU stream throughout
// forward_token so the dGPU doesn't idle and drop its clock.
//
// Hypothesis (from M27 variance analysis): per-token spikes (p99 sync
// ≈ 2× p50) come from dGPU clock dipping between brief idle windows
// (the ~13ms ffn_combine.wait per token, between tokens, etc). A
// concurrent kernel on a second stream signals the power manager
// that the GPU is busy → clock stays high → next token's sync runs
// at peak speed.
//
// Single-thread, no memory ops — won't contend with the real
// compute stream's BW. `volatile` keeps the compiler from optimizing
// the loop away.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void keep_alive(int iters) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    volatile int x = 0;
    for (int i = 0; i < iters; i++) {
        x = x * 3 + i;
    }
}
