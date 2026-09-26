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
        unsigned int                head_dim,
        const int    * __restrict__ comp_base_per) {   // nullable [B]: per-row comp_kv base row (multi-stream arena)
    const unsigned int i = blockIdx.x;                 // 0..top_k
    const unsigned int b = blockIdx.y;
    if (i >= top_k) return;

    const int src_idx = selected_b[(size_t)b * top_k + i];
    if (src_idx < 0) return;                           // sentinel

    const unsigned int tid = threadIdx.x;
    const unsigned int dim_chunk = blockIdx.z * blockDim.x;
    const unsigned int d = dim_chunk + tid;
    if (d >= head_dim) return;

    const size_t src_base = ((size_t)(comp_base_per ? (unsigned int)comp_base_per[b] : 0u)
                             + (unsigned int)src_idx) * head_dim;
    const size_t dst_base = ((size_t)b * top_k + i) * head_dim;
    active_comp_kv_b[dst_base + d] = comp_kv[src_base + d];
}

// gather_u4_r1 — indexer_gather_batched moving 16 B (b128) per thread instead
// of one f16: grid (top_k, B) x (head_dim*2/16) = 64 threads per 1 KB row.
// Same signature; pure copy, sentinel rows (selected < 0) skipped exactly as
// production (destination row untouched): bit-identical. 2026-09-26 sweep
// (E_indexer/gather_b128, dGPU, cold store): b=4 graph 25.0 -> 18.6 us, b=8
// 40.8 -> 24.4, b=512 prefill LOCAL picks 1417 -> 522 (x2.7). BUT at b=2 it is
// 1.5-1.8x SLOWER (reviewer, both u4 variants, not root-caused), so
// IndexerGather::launch_batched_rows selects it only for batch >= 4
// (V41_IDX_GATHER_B128). Requires head_dim*2 % 16 == 0, head_dim <= 512 (the
// launch bound) and 16-B aligned comp_kv / destination (checked in Rust).
// A sentinel / out-of-range row reads row `base` harmlessly and is not stored.
template <unsigned VEC, unsigned ROWS, unsigned IPT>
__device__ __forceinline__ void gather_body(
        _Float16* __restrict__ dst_b, const _Float16* __restrict__ comp_kv,
        const int* __restrict__ selected_b, unsigned top_k, unsigned head_dim,
        const int* __restrict__ comp_base_per) {
    const unsigned row_bytes = head_dim * 2u;
    const unsigned tpr       = row_bytes / VEC;              // threads per row (64 at b128)
    const unsigned rows_per_pass = ROWS / IPT;               // rows covered by one block pass
    const unsigned tid = threadIdx.x;
    const unsigned b   = blockIdx.y;
    const unsigned r_in = tid / tpr, seg = tid % tpr;
    const unsigned row0 = blockIdx.x * ROWS;
    const size_t base = (size_t)(comp_base_per ? (unsigned)comp_base_per[b] : 0u);
    const int* sel = selected_b + (size_t)b * top_k;

    // branch-free: all IPT index loads, then all IPT data loads (a sentinel / out-of-range row
    // reads row `base` harmlessly and is not stored), then the stores.
    uint4 v[IPT]; unsigned long long v2[IPT];
    bool ok[IPT]; unsigned rr[IPT]; int src[IPT];
    #pragma unroll
    for (unsigned i = 0; i < IPT; i++) {
        rr[i] = row0 + r_in + i * rows_per_pass;
        src[i] = (rr[i] < top_k) ? sel[rr[i]] : -1;
    }
    #pragma unroll
    for (unsigned i = 0; i < IPT; i++) {
        ok[i] = src[i] >= 0;
        const unsigned s = ok[i] ? (unsigned)src[i] : 0u;
        const unsigned char* p = reinterpret_cast<const unsigned char*>(comp_kv + (base + s) * head_dim) + seg * VEC;
        if (VEC == 16) v[i] = *reinterpret_cast<const uint4*>(p);
        else           v2[i] = *reinterpret_cast<const unsigned long long*>(p);
    }
    #pragma unroll
    for (unsigned i = 0; i < IPT; i++) {
        if (!ok[i]) continue;
        unsigned char* d = reinterpret_cast<unsigned char*>(dst_b + ((size_t)b * top_k + rr[i]) * head_dim) + seg * VEC;
        if (VEC == 16) *reinterpret_cast<uint4*>(d) = v[i];
        else           *reinterpret_cast<unsigned long long*>(d) = v2[i];
    }
}

extern "C" __global__ __launch_bounds__(64)
void gather_u4_r1(_Float16* __restrict__ dst, const _Float16* __restrict__ comp_kv,
                  const int* __restrict__ selected, unsigned top_k, unsigned head_dim,
                  const int* __restrict__ comp_base_per) {
    gather_body<16, 1, 1>(dst, comp_kv, selected, top_k, head_dim, comp_base_per);
}
