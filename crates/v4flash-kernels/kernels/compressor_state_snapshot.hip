// compressor_state_snapshot — copy the shared state_kv/state_score
// buffers to a per-boundary slot in scratch. Called at each boundary
// firing in the prefill compressor loop so that the post-boundary
// stages (pool, rms_w, rope, fp8, f16rt, append) can run as ONE
// batched pass over all boundaries instead of one launch chain per
// boundary.
//
// Total bytes copied per call:
//   ratio==4 main:    8 rows × 1024 cols ×  4B × 2 buffers =  64 KB
//   ratio==128 main:  128 × 512 × 4B × 2                    = 512 KB
//   ratio==4 indexer: 8 × 256 × 4B × 2                      =  16 KB
//
// Grid (ceil(elems/256), 1, 1), block 256. Vectorized 4 floats/thread.

#include <hip/hip_runtime.h>

extern "C" __global__ void compressor_state_snapshot(
        float       * __restrict__ snap_kv,        // [snap_elems] dst
        float       * __restrict__ snap_score,     // [snap_elems] dst
        const float * __restrict__ state_kv,       // [snap_elems] src
        const float * __restrict__ state_score,    // [snap_elems] src
        unsigned int   snap_elems) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= snap_elems) return;
    snap_kv[i]    = state_kv[i];
    snap_score[i] = state_score[i];
}
