#include <hip/hip_runtime.h>

// Writes 42 to *out from a single thread. Used by Phase 0 toolchain smoke
// test: load this hsaco on each device, launch it, verify the result.
extern "C" __global__ void hello(int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 42;
    }
}
