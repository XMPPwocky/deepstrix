#include <hip/hip_runtime.h>
#include <stdint.h>

// Writes a deterministic pattern to `out[0..n]`: out[i] = (seed ^ i) * MULT.
// Used by Gate C to validate cross-device cache coherency: dGPU writes
// the pattern, we cross-device copy it, then verify on the host.
extern "C" __global__ void fill_pattern(uint32_t* out, uint32_t seed, uint32_t n) {
    const uint32_t MULT = 2654435761u;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (seed ^ i) * MULT;
    }
}

// No-op kernel used to put *something* on a stream when we want to
// measure event-sync round-trip without conflating DMA cost.
extern "C" __global__ void noop_kernel(uint32_t* /*scratch*/) {
    // intentionally empty
}
