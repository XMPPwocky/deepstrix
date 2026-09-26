// indexer_topk_bitonic — bitonic-sort based IndexerTopk, ported from
// ds4's reference design (ds4_cuda.cu: indexer_topk_chunk_pow2_kernel +
// indexer_topk_merge_pow2_kernel). Same first-index tie-break:
//
//   better(a, b) := a.score > b.score
//                || (a.score == b.score && a.idx < b.idx)
//
// Two kernels, two-stage pipeline:
//   1. chunk_4096 — one WG per chunk of up to 4096 scores. Each WG
//      bitonic-sorts its chunk in LDS and writes its top_k indices to a
//      per-chunk scratch slot. Designed to use the WHOLE GPU when there
//      are many chunks (ratio=4 at 64K → 4 chunks; at 128K → 8 chunks).
//   2. merge_4096 — one WG. Reads all (n_chunks × top_k) candidate
//      indices from scratch, re-reads their scores from the original
//      scores buffer (so we don't have to write scores out of the
//      chunk kernels), bitonic-sorts the combined candidate set, writes
//      the top_k selected indices AND the per-row allowed bitmap.
//
// The greedy single-WG fallback ate 6 ms/call at n_comp=16K (~1/60 of
// the GPU live for that wall). Bitonic at n_comp=4096 per chunk is
// O(N log²N) parallel work distributed across 1024 threads with no
// inter-iteration dependency, and the merge over 4×512=2048 candidates
// is the same shape at smaller N.

#include <hip/hip_runtime.h>

#define TOPK_BLOCK     1024u
#define TOPK_SORT_N    4096u
#define TOPK_NEG_INF   (-3.4028235e38f)

__device__ __forceinline__ static bool topk_better(float av, unsigned int ai,
                                                   float bv, unsigned int bi) {
    return av > bv || (av == bv && ai < bi);
}

// In-place bitonic sort of vals[]/idxs[] over SORT_N=TOPK_SORT_N. After
// completion, vals[]/idxs[] are sorted descending (vals[0] is the
// largest). Caller pre-fills slots past the valid range with
// (-INFINITY, UINT32_MAX) so they sort to the end naturally.
//
// Cooperative across TOPK_BLOCK threads. Each thread handles up to
// SORT_N/TOPK_BLOCK = 4 elements per stage via strided indexing.
__device__ __forceinline__ static void bitonic_sort_lds(
        float        * __restrict__ vals,
        unsigned int * __restrict__ idxs) {
    const unsigned int tid = threadIdx.x;
    #pragma unroll 1
    for (unsigned int k = 2u; k <= TOPK_SORT_N; k <<= 1u) {
        #pragma unroll 1
        for (unsigned int j = k >> 1u; j > 0u; j >>= 1u) {
            for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
                const unsigned int other = i ^ j;
                if (other > i && other < TOPK_SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const unsigned int ai = idxs[i];
                    const unsigned int bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_better(bv, bi, av, ai)
                        : topk_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i]     = bv;
                        idxs[i]     = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Per-chunk bitonic sort. Grid: (n_chunks, 1, 1). Each WG handles
// chunk = blockIdx.x ∈ [0, n_chunks). Writes top_k indices of this
// chunk to candidates[chunk * top_k .. (chunk+1) * top_k).
extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void indexer_topk_chunk_4096(
        unsigned int * __restrict__ candidates,  // [n_chunks * top_k]
        const float  * __restrict__ scores,      // [n_comp]
        unsigned int                n_comp,
        unsigned int                top_k) {
    const unsigned int chunk    = blockIdx.x;
    const unsigned int chunk_lo = chunk * TOPK_SORT_N;
    if (chunk_lo >= n_comp) return;
    const unsigned int chunk_hi = (chunk_lo + TOPK_SORT_N < n_comp)
        ? chunk_lo + TOPK_SORT_N
        : n_comp;
    const unsigned int chunk_n  = chunk_hi - chunk_lo;

    __shared__ float        vals[TOPK_SORT_N];
    __shared__ unsigned int idxs[TOPK_SORT_N];

    const unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
        if (i < chunk_n) {
            vals[i] = scores[chunk_lo + i];
            idxs[i] = chunk_lo + i;
        } else {
            vals[i] = TOPK_NEG_INF;
            idxs[i] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();

    bitonic_sort_lds(vals, idxs);

    // Output the top_k from this chunk. Sentinels (UINT32_MAX) for
    // chunks that had fewer than top_k valid elements are preserved
    // so the merge kernel can skip them.
    const unsigned int out_base = chunk * top_k;
    for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
        candidates[out_base + i] = idxs[i];
    }
}

// Final merge. Bitonic-sorts up to TOPK_SORT_N candidate scores
// (= n_chunks × top_k; caller must ensure ≤ TOPK_SORT_N) and writes
// out selected[] + sets per-row bits in allowed_bits[].
//
// Inputs:
//   candidates[n_chunks * top_k]: per-chunk selections from
//     indexer_topk_chunk_4096. We re-read their scores from `scores`
//     so the chunk kernels didn't have to write scores back.
//
// Self-zeros `allowed_bits` at WG start so callers don't have to.
// Matches the API contract of the old greedy IndexerTopk.
extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void indexer_topk_merge_4096(
        int          * __restrict__ selected,        // [top_k]
        unsigned int * __restrict__ allowed_bits,    // [ceil(n_comp/32)]
        const unsigned int * __restrict__ candidates,// [n_chunks * top_k]
        const float        * __restrict__ scores,    // [n_comp]
        unsigned int                       n_comp,
        unsigned int                       top_k,
        unsigned int                       n_candidates) {
    __shared__ float        vals[TOPK_SORT_N];
    __shared__ unsigned int idxs[TOPK_SORT_N];

    const unsigned int tid = threadIdx.x;
    // Zero allowed_bits before any atomicOr lands. `allowed_bits` may be
    // nullptr (no CSA bitmap consumer): skip the bitmap entirely.
    const unsigned int n_words = (n_comp + 31u) >> 5u;
    if (allowed_bits != nullptr) {
        for (unsigned int i = tid; i < n_words; i += TOPK_BLOCK) {
            allowed_bits[i] = 0u;
        }
    }
    __syncthreads();
    // Load candidates into the sort buffer. Padded slots get sentinels.
    for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
        if (i < n_candidates) {
            const unsigned int idx = candidates[i];
            if (idx < n_comp) {
                vals[i] = scores[idx];
                idxs[i] = idx;
            } else {
                vals[i] = TOPK_NEG_INF;
                idxs[i] = 0xFFFFFFFFu;
            }
        } else {
            vals[i] = TOPK_NEG_INF;
            idxs[i] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();

    bitonic_sort_lds(vals, idxs);

    // Emit selected[] + bitmap. Sentinels preserved as -1 in selected[].
    for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
        const unsigned int idx = idxs[i];
        if (idx < n_comp) {
            selected[i] = (int)idx;
            if (allowed_bits != nullptr) {
                atomicOr(&allowed_bits[idx >> 5], 1u << (idx & 31u));
            }
        } else {
            selected[i] = -1;
        }
    }
}

// Single-block bitonic sort for n_comp ≤ TOPK_SORT_N. Combines the
// chunk + merge of the two-stage pipeline into one launch — no scratch
// allocation, no separate merge kernel needed. Writes selected[] +
// bitmap directly.
extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void indexer_topk_bitonic_4096(
        int          * __restrict__ selected,      // [top_k]
        unsigned int * __restrict__ allowed_bits,  // [ceil(n_comp/32)]
        const float  * __restrict__ scores,        // [n_comp]
        unsigned int                n_comp,
        unsigned int                top_k) {
    __shared__ float        vals[TOPK_SORT_N];
    __shared__ unsigned int idxs[TOPK_SORT_N];

    const unsigned int tid = threadIdx.x;
    // Self-zero allowed_bits (nullptr = no bitmap wanted; skip it).
    const unsigned int n_words = (n_comp + 31u) >> 5u;
    if (allowed_bits != nullptr) {
        for (unsigned int i = tid; i < n_words; i += TOPK_BLOCK) {
            allowed_bits[i] = 0u;
        }
    }
    for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
        if (i < n_comp) {
            vals[i] = scores[i];
            idxs[i] = i;
        } else {
            vals[i] = TOPK_NEG_INF;
            idxs[i] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();

    bitonic_sort_lds(vals, idxs);

    for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
        const unsigned int idx = idxs[i];
        if (idx < n_comp) {
            selected[i] = (int)idx;
            if (allowed_bits != nullptr) {
                atomicOr(&allowed_bits[idx >> 5], 1u << (idx & 31u));
            }
        } else {
            selected[i] = -1;
        }
    }
}

// ---- Batched variants (one launch over B tokens) -----------------------
//
// Each batched kernel adds grid.y = B and strides scores/selected/bitmap/
// scratch by `bi`. Per-token `n_comp` comes from `n_idx_per[bi]`. Scores
// past valid range MUST already be -INF (the IndexerScoreWmma_batched
// kernel does this), so each WG can sort over a uniform TOPK_SORT_N range
// and the bitonic better-predicate naturally pushes invalid slots to the
// end.

extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void indexer_topk_bitonic_4096_batched(
        int          * __restrict__ selected_b,     // [B, top_k]
        unsigned int * __restrict__ allowed_bits_b, // [B, n_words_per_b]
        const float  * __restrict__ scores_b,       // [B, n_idx_stride]
        const unsigned int * __restrict__ n_idx_per,// [B]
        unsigned int                n_idx_stride,
        unsigned int                n_words_per_b,
        unsigned int                top_k) {
    const unsigned int bi = blockIdx.y;
    const unsigned int n_comp = n_idx_per[bi];

    int          * __restrict__ selected     = selected_b     + (size_t)bi * top_k;
    // nullptr bitmap = no CSA-mask consumer (batched prefill gathers the
    // selected rows instead); skip the self-zero and the atomicOr.
    unsigned int * __restrict__ allowed_bits = (allowed_bits_b != nullptr)
        ? allowed_bits_b + (size_t)bi * n_words_per_b
        : nullptr;
    const float  * __restrict__ scores       = scores_b       + (size_t)bi * n_idx_stride;

    __shared__ float        vals[TOPK_SORT_N];
    __shared__ unsigned int idxs[TOPK_SORT_N];

    const unsigned int tid = threadIdx.x;
    if (allowed_bits != nullptr) {
        for (unsigned int i = tid; i < n_words_per_b; i += TOPK_BLOCK) {
            allowed_bits[i] = 0u;
        }
    }
    if (n_comp == 0) {
        for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
            selected[i] = -1;
        }
        return;
    }
    for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
        if (i < n_comp) {
            vals[i] = scores[i];
            idxs[i] = i;
        } else {
            vals[i] = TOPK_NEG_INF;
            idxs[i] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();

    bitonic_sort_lds(vals, idxs);

    for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
        const unsigned int idx = idxs[i];
        if (idx < n_comp) {
            selected[i] = (int)idx;
            if (allowed_bits != nullptr) {
                atomicOr(&allowed_bits[idx >> 5], 1u << (idx & 31u));
            }
        } else {
            selected[i] = -1;
        }
    }
}

// Per-chunk batched. Grid: (n_chunks, B, 1).
extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void indexer_topk_chunk_4096_batched(
        unsigned int * __restrict__ candidates_b,   // [B, candidates_stride]
        const float  * __restrict__ scores_b,       // [B, n_idx_stride]
        const unsigned int * __restrict__ n_idx_per,// [B]
        unsigned int                n_idx_stride,
        unsigned int                candidates_stride, // = n_chunks * top_k
        unsigned int                top_k,
        const unsigned int * __restrict__ done_b) { // [B] or null: 1 = token already selected by the fast path
    const unsigned int chunk = blockIdx.x;
    const unsigned int bi    = blockIdx.y;
    if (done_b != nullptr && done_b[bi] != 0u) return;
    const unsigned int n_comp = n_idx_per[bi];
    const unsigned int chunk_lo = chunk * TOPK_SORT_N;
    unsigned int * __restrict__ candidates = candidates_b + (size_t)bi * candidates_stride;
    const float  * __restrict__ scores     = scores_b     + (size_t)bi * n_idx_stride;
    const unsigned int out_base = chunk * top_k;
    const unsigned int tid = threadIdx.x;
    if (chunk_lo >= n_comp) {
        for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
            candidates[out_base + i] = 0xFFFFFFFFu;
        }
        return;
    }
    const unsigned int chunk_hi = (chunk_lo + TOPK_SORT_N < n_comp)
        ? chunk_lo + TOPK_SORT_N
        : n_comp;
    const unsigned int chunk_n  = chunk_hi - chunk_lo;

    __shared__ float        vals[TOPK_SORT_N];
    __shared__ unsigned int idxs[TOPK_SORT_N];

    for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
        if (i < chunk_n) {
            vals[i] = scores[chunk_lo + i];
            idxs[i] = chunk_lo + i;
        } else {
            vals[i] = TOPK_NEG_INF;
            idxs[i] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();

    bitonic_sort_lds(vals, idxs);

    for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
        candidates[out_base + i] = idxs[i];
    }
}

// Final merge batched. Grid: (1, B, 1).
extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void indexer_topk_merge_4096_batched(
        int          * __restrict__ selected_b,     // [B, top_k]
        unsigned int * __restrict__ allowed_bits_b, // [B, n_words_per_b]
        const unsigned int * __restrict__ candidates_b, // [B, candidates_stride]
        const float        * __restrict__ scores_b,     // [B, n_idx_stride]
        const unsigned int * __restrict__ n_idx_per,    // [B]
        unsigned int                n_idx_stride,
        unsigned int                candidates_stride,
        unsigned int                n_words_per_b,
        unsigned int                top_k,
        unsigned int                n_candidates,
        const unsigned int * __restrict__ done_b) {
    if (done_b != nullptr && done_b[blockIdx.y] != 0u) return;
    const unsigned int bi = blockIdx.y;
    const unsigned int n_comp = n_idx_per[bi];

    int          * __restrict__ selected     = selected_b     + (size_t)bi * top_k;
    // nullptr bitmap = no CSA-mask consumer (batched prefill gathers the
    // selected rows instead); skip the self-zero and the atomicOr.
    unsigned int * __restrict__ allowed_bits = (allowed_bits_b != nullptr)
        ? allowed_bits_b + (size_t)bi * n_words_per_b
        : nullptr;
    const unsigned int * __restrict__ candidates = candidates_b + (size_t)bi * candidates_stride;
    const float        * __restrict__ scores     = scores_b     + (size_t)bi * n_idx_stride;

    __shared__ float        vals[TOPK_SORT_N];
    __shared__ unsigned int idxs[TOPK_SORT_N];

    const unsigned int tid = threadIdx.x;
    if (allowed_bits != nullptr) {
        for (unsigned int i = tid; i < n_words_per_b; i += TOPK_BLOCK) {
            allowed_bits[i] = 0u;
        }
    }
    if (n_comp == 0) {
        for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
            selected[i] = -1;
        }
        return;
    }
    for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
        if (i < n_candidates) {
            const unsigned int idx = candidates[i];
            if (idx < n_comp) {
                vals[i] = scores[idx];
                idxs[i] = idx;
            } else {
                vals[i] = TOPK_NEG_INF;
                idxs[i] = 0xFFFFFFFFu;
            }
        } else {
            vals[i] = TOPK_NEG_INF;
            idxs[i] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();

    bitonic_sort_lds(vals, idxs);

    for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
        const unsigned int idx = idxs[i];
        if (idx < n_comp) {
            selected[i] = (int)idx;
            if (allowed_bits != nullptr) {
                atomicOr(&allowed_bits[idx >> 5], 1u << (idx & 31u));
            }
        } else {
            selected[i] = -1;
        }
    }
}

// ---- Multi-level tree merge: intermediate "regroup" level ---------------
//
// When n_chunks * top_k > TOPK_SORT_N the final merge can't see every
// candidate in one bitonic sort. The regroup kernel reduces the candidate
// set by one level: it partitions the n_chunks*top_k candidates into
// contiguous GROUPS of `group_span` (= floor(TOPK_SORT_N/top_k)*top_k)
// candidates and emits each group's top_k. n_chunks*top_k candidates →
// n_groups*top_k, which the existing merge_4096 then sorts to the final
// top_k. Re-reads scores from `scores` like the merge (indices only flow
// through the candidate buffers). Two levels cover n_comp up to
// (TOPK_SORT_N/top_k)^2 * TOPK_SORT_N — far beyond any real ctx.
//
// Grid: (n_groups, 1, 1). One WG per group.
extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void indexer_topk_regroup_4096(
        unsigned int       * __restrict__ candidates_out,  // [n_groups * top_k]
        const unsigned int * __restrict__ candidates_in,   // [n_candidates_in]
        const float        * __restrict__ scores,          // [n_comp]
        unsigned int                      n_comp,
        unsigned int                      n_candidates_in,
        unsigned int                      top_k,
        unsigned int                      group_span) {
    const unsigned int group = blockIdx.x;
    const unsigned int lo = group * group_span;
    unsigned int hi = lo + group_span;
    if (hi > n_candidates_in) hi = n_candidates_in;
    const unsigned int cnt = (hi > lo) ? (hi - lo) : 0u;

    __shared__ float        vals[TOPK_SORT_N];
    __shared__ unsigned int idxs[TOPK_SORT_N];

    const unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
        if (i < cnt) {
            const unsigned int idx = candidates_in[lo + i];
            if (idx < n_comp) {
                vals[i] = scores[idx];
                idxs[i] = idx;
            } else {
                vals[i] = TOPK_NEG_INF;
                idxs[i] = 0xFFFFFFFFu;
            }
        } else {
            vals[i] = TOPK_NEG_INF;
            idxs[i] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();

    bitonic_sort_lds(vals, idxs);

    const unsigned int out_base = group * top_k;
    for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
        candidates_out[out_base + i] = idxs[i];
    }
}

// Batched regroup. Grid: (n_groups, B, 1). Per-token candidate count is
// uniform (= in_stride = n_chunks*top_k; chunks past a token's n_comp
// already hold sentinels), so the group partition is the same for every
// token; n_comp per token only gates the score re-read.
extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void indexer_topk_regroup_4096_batched(
        unsigned int       * __restrict__ candidates_out_b, // [B, out_stride]
        const unsigned int * __restrict__ candidates_in_b,  // [B, in_stride]
        const float        * __restrict__ scores_b,         // [B, n_idx_stride]
        const unsigned int * __restrict__ n_idx_per,        // [B]
        unsigned int                      n_idx_stride,
        unsigned int                      in_stride,        // = n_chunks * top_k
        unsigned int                      out_stride,       // = n_groups * top_k
        unsigned int                      top_k,
        unsigned int                      group_span,
        const unsigned int * __restrict__ done_b) {
    const unsigned int group = blockIdx.x;
    const unsigned int bi    = blockIdx.y;
    if (done_b != nullptr && done_b[bi] != 0u) return;
    const unsigned int n_comp = n_idx_per[bi];

    unsigned int       * __restrict__ candidates_out = candidates_out_b + (size_t)bi * out_stride;
    const unsigned int * __restrict__ candidates_in  = candidates_in_b  + (size_t)bi * in_stride;
    const float        * __restrict__ scores         = scores_b         + (size_t)bi * n_idx_stride;

    const unsigned int lo = group * group_span;
    unsigned int hi = lo + group_span;
    if (hi > in_stride) hi = in_stride;
    const unsigned int cnt = (hi > lo) ? (hi - lo) : 0u;

    __shared__ float        vals[TOPK_SORT_N];
    __shared__ unsigned int idxs[TOPK_SORT_N];

    const unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
        if (i < cnt) {
            const unsigned int idx = candidates_in[lo + i];
            if (idx < n_comp) {
                vals[i] = scores[idx];
                idxs[i] = idx;
            } else {
                vals[i] = TOPK_NEG_INF;
                idxs[i] = 0xFFFFFFFFu;
            }
        } else {
            vals[i] = TOPK_NEG_INF;
            idxs[i] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();

    bitonic_sort_lds(vals, idxs);

    const unsigned int out_base = group * top_k;
    for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
        candidates_out[out_base + i] = idxs[i];
    }
}

// =============================================================================
// indexer_topk_select_batched — exact threshold top-k (2026-09-08).
//
// The chunk/regroup/merge chain fully bitonic-sorts every 4096-score chunk
// to keep 512: O(n log^2 n) with ~90% of the compare-exchanges spent on
// elements that can never be selected. At 192K (12 chunks/token) it was
// 5.1 ms/launch, 8% of the prefill wall (trace 2026-09-08).
//
// This kernel produces the SAME selection (same total order: score desc,
// index asc) with O(n) work per token:
//   1. n <= 4096: one bitonic sort of everything (== the single-chunk path).
//   2. n > 4096: sort a strided 2048-element SAMPLE of the scores (keys
//      only), pick a deliberately LOW threshold t from the sample rank of
//      the top_k-th element, then one pass counts #(> t) and #(>= t)
//      (thread-local counts + one block reduce; no per-element atomics).
//      Adjust t along the sample (a few retries) until #(> t) <= 4096 and
//      #(>= t) >= top_k.
//   3. Compact indices with score > t into LDS; if fewer than top_k, append
//      the lowest-index elements with score == t (deterministic per-thread
//      contiguous ranges + block prefix) — exactly the reference tie-break.
//   4. bitonic_sort_lds the <= 4096 candidates with the reference comparator
//      and write the first top_k.
// If the sample cannot bracket a usable threshold (a >4096-element cluster
// of near-equal scores between two consecutive sample values) the token is
// left with done_b[bi] = 0 and the existing chain — which now early-outs
// on done_b[bi] = 1 — handles it. Correctness never depends on the sample.
// Grid (B), block 1024. LDS: 32 KB sort arrays + 8 KB sample + 4 KB scan.
// =============================================================================
#define TOPK_SEL_SAMPLE 2048u
#define TOPK_SEL_CAP    TOPK_SORT_N

__device__ __forceinline__ static unsigned int topk_key(float f) {
    unsigned int b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);   // monotonic: larger f -> larger key
}

// Descending bitonic sort of keys[TOPK_SEL_SAMPLE] in LDS (1024 threads).
__device__ __forceinline__ static void bitonic_sort_keys2048(unsigned int* keys) {
    const unsigned int tid = threadIdx.x;
    #pragma unroll 1
    for (unsigned int k = 2u; k <= TOPK_SEL_SAMPLE; k <<= 1u) {
        #pragma unroll 1
        for (unsigned int j = k >> 1u; j > 0u; j >>= 1u) {
            for (unsigned int i = tid; i < TOPK_SEL_SAMPLE; i += TOPK_BLOCK) {
                const unsigned int other = i ^ j;
                if (other > i) {
                    const unsigned int a = keys[i], b = keys[other];
                    const bool desc_half = (i & k) == 0u;
                    if (desc_half ? (b > a) : (a > b)) { keys[i] = b; keys[other] = a; }
                }
            }
            __syncthreads();
        }
    }
}

// TOPK_SEL_ILP (2026-09-26): loads in flight per thread in the two O(n) scans
// (count, compact). With U = 1 each thread walks its ~n/1024 strided scores one
// dependent-issue iteration at a time: at a decode row's n = 235K that is ~230
// serial round trips per pass, ~185 us per launch for ~1 MB of scores
// (tests/bench_decode_latency_breakdown.rs, gfx1201). U > 1 issues U loads,
// then consumes them in the same order. The selection is IDENTICAL: the counts
// are integer sums, and the compact pass already placed candidates in atomic
// (nondeterministic) slot order ahead of a full sort by the total order
// (score desc, index asc). `indexer_topk_select_batched` keeps U = 1 for
// rollback (V41_TOPK_SELECT_ILP=0).
#define TOPK_SEL_ILP 16u

template <unsigned int U>
__device__ __forceinline__ static void topk_select_impl(
        int          * __restrict__ selected_b,     // [B, top_k]
        unsigned int * __restrict__ done_b,         // [B]
        const float  * __restrict__ scores_b,       // [B, n_idx_stride]
        const unsigned int * __restrict__ n_idx_per,// [B]
        unsigned int                n_idx_stride,
        unsigned int                top_k) {
    const unsigned int bi  = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int n   = n_idx_per[bi];
    int          * __restrict__ selected = selected_b + (size_t)bi * top_k;
    const float  * __restrict__ scores   = scores_b   + (size_t)bi * n_idx_stride;

    __shared__ float        vals[TOPK_SORT_N];
    __shared__ unsigned int idxs[TOPK_SORT_N];
    __shared__ unsigned int sample[TOPK_SEL_SAMPLE];
    __shared__ unsigned int red_gt[TOPK_BLOCK / 32], red_ge[TOPK_BLOCK / 32];
    __shared__ unsigned int scan[TOPK_BLOCK];
    __shared__ unsigned int s_count;
    __shared__ unsigned int s_t, s_state;   // state: 0 = searching, 1 = ok, 2 = give up

    if (n == 0u) {
        for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) selected[i] = -1;
        if (tid == 0u) done_b[bi] = 1u;
        return;
    }

    unsigned int n_cand = 0u;
    if (n <= TOPK_SORT_N) {
        // --- small: sort everything (identical to the single-chunk path)
        for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
            if (i < n) { vals[i] = scores[i]; idxs[i] = i; }
            else       { vals[i] = TOPK_NEG_INF; idxs[i] = 0xFFFFFFFFu; }
        }
        n_cand = n;
    } else {
        // --- sample + sort (keys only)
        for (unsigned int j = tid; j < TOPK_SEL_SAMPLE; j += TOPK_BLOCK) {
            const unsigned int src = (unsigned int)(((unsigned long long)j * n) / TOPK_SEL_SAMPLE);
            sample[j] = topk_key(scores[src]);
        }
        __syncthreads();
        bitonic_sort_keys2048(sample);
        // rank of the top_k-th element in the sample, then a deliberately low threshold
        const unsigned int r = (unsigned int)(((unsigned long long)top_k * TOPK_SEL_SAMPLE + n - 1u) / n);
        unsigned int idx_t = min(TOPK_SEL_SAMPLE - 1u, 2u * r + 32u);
        unsigned int lo_idx = 0u, hi_idx = TOPK_SEL_SAMPLE - 1u;   // search bracket over sample ranks
        unsigned int t = 0u; unsigned int count_gt = 0u, count_ge = 0u;
        bool ok = false, giveup = false;
        #pragma unroll 1
        for (int attempt = 0; attempt < 8 && !ok && !giveup; attempt++) {
            if (tid == 0u) s_t = sample[idx_t];
            __syncthreads();
            t = s_t;
            // count > t and >= t
            unsigned int c_gt = 0u, c_ge = 0u;
            unsigned int i = tid;
            if (U > 1u) {
                for (; i + (U - 1u) * TOPK_BLOCK < n; i += U * TOPK_BLOCK) {
                    float f[U];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) f[u] = scores[i + u * TOPK_BLOCK];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) {
                        const unsigned int k = topk_key(f[u]);
                        c_gt += (k > t); c_ge += (k >= t);
                    }
                }
            }
            for (; i < n; i += TOPK_BLOCK) {
                const unsigned int k = topk_key(scores[i]);
                c_gt += (k > t); c_ge += (k >= t);
            }
            // block reduce
            #pragma unroll
            for (unsigned int o = 16; o > 0; o >>= 1) { c_gt += __shfl_down(c_gt, o); c_ge += __shfl_down(c_ge, o); }
            if ((tid & 31u) == 0u) { red_gt[tid >> 5] = c_gt; red_ge[tid >> 5] = c_ge; }
            __syncthreads();
            if (tid == 0u) {
                unsigned int g = 0u, e = 0u;
                for (unsigned int w = 0; w < TOPK_BLOCK / 32; w++) { g += red_gt[w]; e += red_ge[w]; }
                red_gt[0] = g; red_ge[0] = e;
                // decide
                unsigned int st = 0u;
                if (g <= TOPK_SEL_CAP && e >= top_k) st = 1u;
                else if (g > TOPK_SEL_CAP) {            // too many above: raise t (smaller sample rank)
                    hi_idx = idx_t;
                    if (idx_t == 0u) st = 2u; else idx_t = lo_idx + (idx_t - lo_idx) / 2u;
                } else {                                 // too few >= t: lower t (larger sample rank)
                    lo_idx = idx_t;
                    if (idx_t == TOPK_SEL_SAMPLE - 1u) { st = 2u; }
                    else idx_t = idx_t + (hi_idx - idx_t + 1u) / 2u;
                }
                if (st == 0u && lo_idx + 1u >= hi_idx && attempt > 2) st = 2u;   // bracket collapsed
                s_state = st;
            }
            __syncthreads();
            count_gt = red_gt[0]; count_ge = red_ge[0];
            ok = (s_state == 1u); giveup = (s_state == 2u);
            __syncthreads();
        }
        if (!ok) {
            // Last resort before giving up: t = key(-inf) selects everything finite;
            // usable only when n itself is <= CAP (it is not here) -> give up.
            if (tid == 0u) done_b[bi] = 0u;
            return;
        }
        // --- compact > t
        if (tid == 0u) s_count = 0u;
        for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) { vals[i] = TOPK_NEG_INF; idxs[i] = 0xFFFFFFFFu; }
        __syncthreads();
        {
            unsigned int i = tid;
            if (U > 1u) {
                for (; i + (U - 1u) * TOPK_BLOCK < n; i += U * TOPK_BLOCK) {
                    float f[U];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) f[u] = scores[i + u * TOPK_BLOCK];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) {
                        if (topk_key(f[u]) > t) {
                            const unsigned int slot = atomicAdd(&s_count, 1u);
                            vals[slot] = f[u]; idxs[slot] = i + u * TOPK_BLOCK;
                        }
                    }
                }
            }
            for (; i < n; i += TOPK_BLOCK) {
                const float f = scores[i];
                if (topk_key(f) > t) {
                    const unsigned int slot = atomicAdd(&s_count, 1u);
                    vals[slot] = f; idxs[slot] = i;
                }
            }
        }
        __syncthreads();
        n_cand = count_gt;
        if (count_gt < top_k) {
            // append the lowest-index elements with key == t: per-thread contiguous
            // ranges keep ascending order; block prefix gives global ranks.
            const unsigned int need  = top_k - count_gt;
            const unsigned int per   = (n + TOPK_BLOCK - 1u) / TOPK_BLOCK;
            const unsigned int my_lo = min(n, tid * per), my_hi = min(n, my_lo + per);
            unsigned int mine = 0u;
            for (unsigned int i = my_lo; i < my_hi; i++) mine += (topk_key(scores[i]) == t);
            scan[tid] = mine;
            __syncthreads();
            // Hillis-Steele inclusive scan over 1024
            for (unsigned int o = 1u; o < TOPK_BLOCK; o <<= 1u) {
                const unsigned int v = (tid >= o) ? scan[tid - o] : 0u;
                __syncthreads();
                scan[tid] += v;
                __syncthreads();
            }
            const unsigned int before = scan[tid] - mine;   // exclusive
            unsigned int rank = before;
            for (unsigned int i = my_lo; i < my_hi && rank < need; i++) {
                const float f = scores[i];
                if (topk_key(f) == t) { vals[count_gt + rank] = f; idxs[count_gt + rank] = i; rank++; }
            }
            n_cand = count_gt + need;
            __syncthreads();
        }
    }
    __syncthreads();
    bitonic_sort_lds(vals, idxs);
    for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
        selected[i] = (i < n_cand) ? (int)idxs[i] : -1;
    }
    if (tid == 0u) done_b[bi] = 1u;
}

extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void indexer_topk_select_batched(
        int          * __restrict__ selected_b,
        unsigned int * __restrict__ done_b,
        const float  * __restrict__ scores_b,
        const unsigned int * __restrict__ n_idx_per,
        unsigned int                n_idx_stride,
        unsigned int                top_k) {
    topk_select_impl<1u>(selected_b, done_b, scores_b, n_idx_per, n_idx_stride, top_k);
}

extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void indexer_topk_select_batched_ilp(
        int          * __restrict__ selected_b,
        unsigned int * __restrict__ done_b,
        const float  * __restrict__ scores_b,
        const unsigned int * __restrict__ n_idx_per,
        unsigned int                n_idx_stride,
        unsigned int                top_k) {
    topk_select_impl<TOPK_SEL_ILP>(selected_b, done_b, scores_b, n_idx_per, n_idx_stride, top_k);
}
