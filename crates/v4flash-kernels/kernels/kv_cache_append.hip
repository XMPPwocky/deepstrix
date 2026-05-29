// kv_cache_append — device-side append (with SWA-window slide eviction)
// of a freshly-computed KV row into a per-layer cache. Replaces the
// host-side sync+copy_to_host+memmove+copy_from_host round-trip
// (M13.3).
//
// Cache layout: [swa_window × head_dim] f32, row-major (one cache slot
// per row). `n_raw_before` is the current fill count (0..swa_window
// inclusive); on every call we either:
//   * not-full (n_raw_before < swa_window): write `kv_new` into slot
//     `n_raw_before`. All earlier slots stay put.
//   * full (n_raw_before == swa_window): shift slots [1, swa_window)
//     down by one then write `kv_new` into slot `swa_window - 1`.
//
// Single-block kernel: each thread owns one element of head_dim. The
// shift path uses __syncthreads() to make the read-then-write race-free
// within the block. Caller must launch with block = (head_dim, 1, 1).

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_append(
        float        * __restrict__ cache,         // [swa_window * head_dim]
        const float  * __restrict__ kv_new,        // [head_dim]
        unsigned int   n_raw_before,
        unsigned int   swa_window,
        unsigned int   head_dim) {
    const unsigned int d = threadIdx.x;
    if (d >= head_dim) return;

    if (n_raw_before < swa_window) {
        // Append at slot n_raw_before. No reads, just a write.
        cache[(unsigned long long)n_raw_before * head_dim + d] = kv_new[d];
        return;
    }

    // Evict-oldest slide: shift slots [1, swa_window) → [0, swa_window-1).
    // Read s+1 into a register, syncthreads, write to s. Race-free within
    // the single block.
    for (unsigned int s = 0; s < swa_window - 1; ++s) {
        float val = cache[(unsigned long long)(s + 1) * head_dim + d];
        __syncthreads();
        cache[(unsigned long long)s * head_dim + d] = val;
        __syncthreads();
    }
    cache[(unsigned long long)(swa_window - 1) * head_dim + d] = kv_new[d];
}

// M50 batched: append B contiguous rows to slots
// [n_raw_before .. n_raw_before + B). Requires n_raw_before + B <= swa_window
// (no eviction handling on the batched path — caller falls back to serial).
// Grid (B, 1, 1), block (head_dim, 1, 1).
extern "C" __global__ void kv_cache_append_batched(
        float        * __restrict__ cache,         // [swa_window * head_dim]
        const float  * __restrict__ kv_new,        // [B, head_dim]
        unsigned int   n_raw_before,
        unsigned int   head_dim) {
    const unsigned int d = threadIdx.x;
    const unsigned int b = blockIdx.x;
    if (d >= head_dim) return;
    const unsigned int slot = n_raw_before + b;
    cache[(unsigned long long)slot * head_dim + d] =
        kv_new[(unsigned long long)b * head_dim + d];
}

// M51 general eviction gather. Whenever the combined sequence
// [old_cache(R0 rows), kv_new(B rows)] overflows the window
// (R0 + B > swa_window), the serial slide loop leaves the cache holding
// exactly the LAST `swa_window` rows of that combined sequence, in order.
// Reproduce that final state directly: slot s ← combined[R0 + B - W + s],
// where combined[m] = old_cache[m] if m < R0 else kv_new[m - R0].
//
// In the eviction regime R0 + B > W, the source index m = R0+B-W+s ≥ 1 for
// all s ≥ 0, so there is no underflow even when B < W (partial eviction with
// surviving prior rows). Output is written to a SEPARATE scratch buffer, so
// the survivor reads from `cache` never race the writes — caller copies the
// scratch ring back into the cache afterwards.
//
// Grid (swa_window, 1, 1), block (head_dim, 1, 1). Caller ensures R0+B > W.
extern "C" __global__ void kv_cache_append_evict_gather(
        const float  * __restrict__ cache,         // [swa_window * head_dim] (old)
        const float  * __restrict__ kv_new,        // [B, head_dim]
        float        * __restrict__ out,           // [swa_window * head_dim] (new ring)
        unsigned int   r0,                         // prior fill count (old rows in cache)
        unsigned int   b,
        unsigned int   swa_window,
        unsigned int   head_dim) {
    const unsigned int d = threadIdx.x;
    const unsigned int s = blockIdx.x;
    if (d >= head_dim || s >= swa_window) return;
    // m ≥ 1 because r0 + b > swa_window in the eviction regime.
    const unsigned long long m = (unsigned long long)r0 + b - swa_window + s;
    const float val = (m < (unsigned long long)r0)
        ? cache[m * head_dim + d]
        : kv_new[(m - r0) * head_dim + d];
    out[(unsigned long long)s * head_dim + d] = val;
}
