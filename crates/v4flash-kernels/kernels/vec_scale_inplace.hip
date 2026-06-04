// vec_scale_inplace — single-block f32 in-place scalar multiply.
//
//   x[i] *= scalar  for i in 0..n
//
// Used by the indexer pipeline to apply 1/sqrt(head_dim*n_head) to
// head_weights[64] after the indexer.proj matvec. Tiny (~64 ops) but
// kept off the host to avoid a sync + copy_to_host + copy_from_host
// roundtrip per layer per token.

#include <hip/hip_runtime.h>

extern "C" __global__ void vec_scale_inplace(
        float       * __restrict__ x,
        float                      scalar,
        unsigned int               n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= scalar;
}
