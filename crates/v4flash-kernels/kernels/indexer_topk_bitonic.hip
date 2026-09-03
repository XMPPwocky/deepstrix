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
        unsigned int                top_k) {
    const unsigned int chunk = blockIdx.x;
    const unsigned int bi    = blockIdx.y;
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
        unsigned int                n_candidates) {
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
        unsigned int                      group_span) {
    const unsigned int group = blockIdx.x;
    const unsigned int bi    = blockIdx.y;
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
