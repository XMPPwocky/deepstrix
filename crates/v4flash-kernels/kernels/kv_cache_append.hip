// kv_cache_append — device-side append (with SWA-window slide eviction)
// of a freshly-computed KV row into a per-layer cache. Replaces the
// host-side sync+copy_to_host+memmove+copy_from_host round-trip
// (M13.3).
//
// Cache layout: [swa_window × head_dim] f16, row-major (one cache slot
// per row). `kv_new` arrives as f32 from the per-token compute path —
// cast to f16 at store. `n_raw_before` is the current fill count
// (0..swa_window inclusive); on every call we either:
//   * not-full (n_raw_before < swa_window): write `kv_new` into slot
//     `n_raw_before`. All earlier slots stay put.
//   * full (n_raw_before == swa_window): shift slots [1, swa_window)
//     down by one then write `kv_new` into slot `swa_window - 1`.
//
// Single-block kernel: each thread owns one element of head_dim. The
// shift path uses __syncthreads() to make the read-then-write race-free
// within the block. Caller must launch with block = (head_dim, 1, 1).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void kv_cache_append(
        _Float16     * __restrict__ cache,         // [swa_window * head_dim] (f16)
        const float  * __restrict__ kv_new,        // [head_dim]              (f32 input, cast at store)
        unsigned int   n_raw_before,
        unsigned int   swa_window,
        unsigned int   head_dim) {
    const unsigned int d = threadIdx.x;
    if (d >= head_dim) return;

    if (n_raw_before < swa_window) {
        // Append at slot n_raw_before. No reads, just a write.
        cache[(unsigned long long)n_raw_before * head_dim + d] = (_Float16)kv_new[d];
        return;
    }

    // Evict-oldest slide: shift slots [1, swa_window) → [0, swa_window-1).
    // Read s+1 into a register, syncthreads, write to s. Race-free within
    // the single block.
    for (unsigned int s = 0; s < swa_window - 1; ++s) {
        _Float16 val = cache[(unsigned long long)(s + 1) * head_dim + d];
        __syncthreads();
        cache[(unsigned long long)s * head_dim + d] = val;
        __syncthreads();
    }
    cache[(unsigned long long)(swa_window - 1) * head_dim + d] = (_Float16)kv_new[d];
}

// M50 batched: append B contiguous rows to slots
// [n_raw_before .. n_raw_before + B). Requires n_raw_before + B <= swa_window
// (no eviction handling on the batched path — caller falls back to serial).
// Grid (B, 1, 1), block (head_dim, 1, 1).
extern "C" __global__ void kv_cache_append_batched(
        _Float16     * __restrict__ cache,         // [swa_window * head_dim] (f16)
        const float  * __restrict__ kv_new,        // [B, head_dim]           (f32 input)
        unsigned int   n_raw_before,
        unsigned int   head_dim) {
    const unsigned int d = threadIdx.x;
    const unsigned int b = blockIdx.x;
    if (d >= head_dim) return;
    const unsigned int slot = n_raw_before + b;
    cache[(unsigned long long)slot * head_dim + d] =
        (_Float16)kv_new[(unsigned long long)b * head_dim + d];
}


// M59: device-slot variant of the monotonic append — `slot` comes from a
// device u32 (one hipStreamWriteValue32 per token serves all layers, since
// every layer appends exactly once per token and the counters evolve in
// lockstep). No evict path: the M55 monotonic scheme guarantees
// slot < capacity; the host still owns raw_off/window bookkeeping.
extern "C" __global__ void kv_cache_append_slotdev(
        _Float16     * __restrict__ cache,         // [capacity * head_dim] (f16)
        const float  * __restrict__ kv_new,        // [head_dim]
        const unsigned int * __restrict__ slot_dev,
        unsigned int   head_dim) {
    const unsigned int d = threadIdx.x;
    if (d >= head_dim) return;
    const unsigned int slot = *slot_dev;
    cache[(unsigned long long)slot * head_dim + d] = (_Float16)kv_new[d];
}
