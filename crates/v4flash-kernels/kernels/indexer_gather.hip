// indexer_gather — scatter-read selected rows of comp_kv into a dense
// contiguous buffer that the attention kernels can read like a normal
// comp_kv at a smaller n_comp.
//
// After IndexerTopk produces a sorted index list [k0, k1, ..., k_{K-1}]
// (sorted by selection order, descending score), this kernel copies:
//
//   active_comp_kv[i, d] = comp_kv[selected[i], d]   for i in 0..K, d in 0..head_dim
//
// One workgroup per (selected row, head_dim chunk). Block=256 threads
// cooperatively copy head_dim contiguous f16 elements per row. Coalesced
// destination writes; source reads are coalesced WITHIN a row (the
// dominant pattern at head_dim=512=256*2 → each thread loads two f16s).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void indexer_gather(
        _Float16     * __restrict__ active_comp_kv,  // [top_k, head_dim] f16
        const _Float16 * __restrict__ comp_kv,       // [n_comp, head_dim] f16
        const int    * __restrict__ selected,        // [top_k]
        unsigned int                top_k,
        unsigned int                head_dim) {
    const unsigned int i = blockIdx.x;             // 0..top_k
    if (i >= top_k) return;
    const int src_idx = selected[i];
    if (src_idx < 0) return;                       // sentinel from IndexerTopk

    const unsigned int tid = threadIdx.x;
    const unsigned int dim_chunk = blockIdx.y * blockDim.x;
    const unsigned int d = dim_chunk + tid;
    if (d >= head_dim) return;

    const size_t src_base = (size_t)(unsigned int)src_idx * head_dim;
    const size_t dst_base = (size_t)i * head_dim;
    active_comp_kv[dst_base + d] = comp_kv[src_base + d];
}

// Batched variant: one launch covers B tokens. Per-batch selected indices
// at stride `top_k`; per-batch destination strides `top_k * head_dim`.
// comp_kv is SHARED across the batch (one main compressor per layer).
// Grid: (top_k, B, ceil(head_dim/256)), block 256.
extern "C" __global__ void indexer_gather_batched(
        _Float16     * __restrict__ active_comp_kv_b,  // [B, top_k, head_dim] f16
        const _Float16 * __restrict__ comp_kv,         // [n_comp, head_dim] f16 (shared)
        const int    * __restrict__ selected_b,        // [B, top_k]
        unsigned int                top_k,
        unsigned int                head_dim) {
    const unsigned int i = blockIdx.x;                 // 0..top_k
    const unsigned int b = blockIdx.y;
    if (i >= top_k) return;

    const int src_idx = selected_b[(size_t)b * top_k + i];
    if (src_idx < 0) return;                           // sentinel

    const unsigned int tid = threadIdx.x;
    const unsigned int dim_chunk = blockIdx.z * blockDim.x;
    const unsigned int d = dim_chunk + tid;
    if (d >= head_dim) return;

    const size_t src_base = (size_t)(unsigned int)src_idx * head_dim;
    const size_t dst_base = ((size_t)b * top_k + i) * head_dim;
    active_comp_kv_b[dst_base + d] = comp_kv[src_base + d];
}
