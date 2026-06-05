// comp_kv_append — append one compressor row into the comp_kv cache.
// Mirrors `kv_cache_append` but without SWA eviction (comp_kv grows
// unbounded as more boundaries fire). Used by the dGPU side of the
// M13.5 compressor migration to receive `comp_row` peer-pushed from
// the iGPU.
//
// Single-block kernel; launch with `block.x = head_dim`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void comp_kv_append(
        _Float16     * __restrict__ comp_kv,    // [n_comp_max * head_dim] (f16)
        const float  * __restrict__ row,        // [head_dim]              (f32 input, cast at store)
        unsigned int   n_comp,                  // current row count
        unsigned int   head_dim) {
    const unsigned int d = threadIdx.x;
    if (d >= head_dim) return;
    comp_kv[(unsigned long long)n_comp * head_dim + d] = (_Float16)row[d];
}

// Batched: append n_boundaries rows from rows_b[k*head_dim] to
// comp_kv[(n_comp_start + k)*head_dim] for k in 0..n_boundaries.
// Grid (1, n_boundaries, 1), block head_dim.
extern "C" __global__ void comp_kv_append_batched(
        _Float16     * __restrict__ comp_kv,
        const float  * __restrict__ rows_b,
        unsigned int   n_comp_start,
        unsigned int   head_dim) {
    const unsigned int d = threadIdx.x;
    const unsigned int k = blockIdx.y;
    if (d >= head_dim) return;
    const size_t out_off = (size_t)(n_comp_start + k) * head_dim;
    const size_t in_off  = (size_t)k * head_dim;
    comp_kv[out_off + d] = (_Float16)rows_b[in_off + d];
}
