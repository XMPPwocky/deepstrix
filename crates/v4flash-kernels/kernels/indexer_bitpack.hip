// indexer_bitpack — convert IndexerTopk's `selected[K]` sorted-index list
// into a per-(batch token) bitmap slice consumed by the prefill mask-
// aware attention score kernel.
//
//   For c in 0..n_comp:
//       bits[batch_offset + (c >> 5)] |= (1 << (c & 31)) iff c is in `selected`
//
// Equivalent to "mark allowed_bits from a sorted index list". K threads
// per launch; each thread atomically OR's one bit. Bitmap is assumed
// pre-zeroed to its `n_comp_words` extent (the all-1s alternative path
// uses a different fill kernel; see indexer_bitpack_all_ones).
//
// `selected` can contain sentinel `-1` entries (when fewer than K rows
// were selected; ds4's early-permit). Those are skipped.

#include <hip/hip_runtime.h>

extern "C" __global__ void indexer_bitpack_set(
        unsigned int * __restrict__ bits,       // [n_comp_words] — per-token slice
        const int    * __restrict__ selected,   // [k]
        unsigned int                k) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const int s = selected[i];
    if (s < 0) return;
    const unsigned int c = (unsigned int)s;
    atomicOr(&bits[c >> 5], 1u << (c & 31));
}

// Fill the first `n_words` words of `bits` with 0. Used to clear a
// per-token slice before `indexer_bitpack_set` starts OR'ing.
extern "C" __global__ void indexer_bitpack_zero(
        unsigned int * __restrict__ bits,
        unsigned int                n_words) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_words) return;
    bits[i] = 0u;
}

// Set bits [0, n_comp) to 1 — the early-permit branch where ds4 returns
// an all-allowed mask without running the indexer. Trailing bits in the
// last word past n_comp are left zero (don't matter — the score kernel
// only tests bits with index < n_comp via its bkey range check).
extern "C" __global__ void indexer_bitpack_all_ones(
        unsigned int * __restrict__ bits,
        unsigned int                n_comp) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int n_words = (n_comp + 31) >> 5;
    if (i >= n_words) return;
    const unsigned int word_lo = i * 32;
    if (word_lo + 32 <= n_comp) {
        bits[i] = 0xFFFFFFFFu;
    } else {
        // Partial last word — only set bits up to n_comp.
        const unsigned int valid = n_comp - word_lo;
        bits[i] = (1u << valid) - 1u;
    }
}
