// broadcast_to_hc — replicate a length-n source vector to n_rows copies
// in a contiguous row-major output. Used by MTP draft to expand the
// embedding-projection output (N_EMBD) into the N_HC-row hidden state
// (N_HC × N_EMBD = HC_DIM) before adding the per-row h_proj output.
//
// Equivalent to `for r in 0..n_rows: out[r*n..(r+1)*n] = src[0..n]`.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void broadcast_to_hc(
        float       * __restrict__ out,   // [n_rows * n]
        const float * __restrict__ src,   // [n]
        unsigned int  n,
        unsigned int  n_rows) {
    const unsigned int j   = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y;
    if (j >= n || row >= n_rows) return;
    out[(size_t)row * (size_t)n + (size_t)j] = src[j];
}
