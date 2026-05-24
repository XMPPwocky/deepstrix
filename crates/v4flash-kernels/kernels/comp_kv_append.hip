// comp_kv_append — append one compressor row into the comp_kv cache.
// Mirrors `kv_cache_append` but without SWA eviction (comp_kv grows
// unbounded as more boundaries fire). Used by the dGPU side of the
// M13.5 compressor migration to receive `comp_row` peer-pushed from
// the iGPU.
//
// Single-block kernel; launch with `block.x = head_dim`.

#include <hip/hip_runtime.h>

extern "C" __global__ void comp_kv_append(
        float        * __restrict__ comp_kv,    // [n_comp_max * head_dim]
        const float  * __restrict__ row,        // [head_dim]
        unsigned int   n_comp,                  // current row count
        unsigned int   head_dim) {
    const unsigned int d = threadIdx.x;
    if (d >= head_dim) return;
    comp_kv[(unsigned long long)n_comp * head_dim + d] = row[d];
}
