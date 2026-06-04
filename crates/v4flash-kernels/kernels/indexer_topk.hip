// indexer_topk — greedy K-of-N top-K selection for the CSA indexer.
//
// Mirrors ds4's iterative greedy max-find (ds4.c:7022-7032 and 7093-7103):
//
//   for (k = 0; k < top_k; k++) {
//       best = 0; best_score = -INF;
//       for (c = 0; c < n_comp; c++) {
//           if (!allowed[c] && scores[c] > best_score) {
//               best = c; best_score = scores[c];
//           }
//       }
//       allowed[best] = true;
//   }
//
// The strict `>` is load-bearing: ties go to the FIRST-encountered index.
// We preserve that by combining tournament reductions with (score, idx)
// ordered comparison: take the strictly-larger score, or if equal, the
// strictly-smaller index.
//
// Outputs:
//   selected[k]      = chosen comp row index for k = 0..min(top_k, n_comp)
//   allowed_bits[ b ] = packed bitmap; bit (c & 31) of word (c >> 5) is 1
//                      iff comp row c is selected.
//
// Single workgroup, BLOCK=256 (= 8 wave32). Sequential across K iterations;
// parallel within each "find max" pass via wave-shuffle + LDS cross-wave.
//
// For our production sizes (top_k=512, n_comp up to ~32K), the K-loop dominates:
// 512 iterations * (32K/256 strided scan + reduce + atomic) ≈ 100s of µs.
// If this becomes a hot spot, swap for a radix-select or threshold-then-select
// kernel; this implementation matches ds4 bit-exactly which is more important
// for the validation harness than the perf optimum.

#include <hip/hip_runtime.h>

#define ITOPK_BLOCK 256
#define ITOPK_NEG_INF (-3.4028235e38f)

// Comparator: return true if (sa, ia) STRICTLY OUTRANKS (sb, ib).
// Outranking = larger score; ties broken by smaller index.
__device__ inline bool itopk_better(float sa, int ia, float sb, int ib) {
    if (sa > sb) return true;
    if (sa < sb) return false;
    // Equal scores. Smaller index wins; -1 (initial) loses to any real index.
    if (ia < 0) return false;
    if (ib < 0) return true;
    return ia < ib;
}

// Wave-shuffle reduce over (score, idx) pairs preserving first-index ties.
// On gfx12/RDNA4 we're wave32; __shfl_xor sees a 32-wide lane mask.
__device__ inline void itopk_wave_reduce(float &score, int &idx) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        float os = __shfl_xor(score, off);
        int   oi = __shfl_xor(idx,   off);
        if (itopk_better(os, oi, score, idx)) {
            score = os;
            idx   = oi;
        }
    }
}

extern "C" __global__ __launch_bounds__(ITOPK_BLOCK)
void indexer_topk(
        int          * __restrict__ selected,       // [top_k]   sorted by selection order
        unsigned int * __restrict__ allowed_bits,   // [ceil(n_comp/32)] bitmap output
        const float  * __restrict__ scores,         // [n_comp]
        unsigned int                n_comp,
        unsigned int                top_k) {
    __shared__ float s_wave_score[8];   // ITOPK_BLOCK / 32
    __shared__ int   s_wave_idx[8];

    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;

    // Zero the bitmap. ceil(n_comp/32) words.
    const unsigned int n_words = (n_comp + 31u) >> 5;
    for (unsigned int i = tid; i < n_words; i += ITOPK_BLOCK) {
        allowed_bits[i] = 0u;
    }
    __syncthreads();

    const unsigned int k_actual = (top_k < n_comp) ? top_k : n_comp;

    for (unsigned int k = 0; k < k_actual; ++k) {
        // Per-thread strided scan of scores[], skipping already-taken indices.
        float my_score = ITOPK_NEG_INF;
        int   my_idx   = -1;
        for (unsigned int c = tid; c < n_comp; c += ITOPK_BLOCK) {
            const unsigned int word = c >> 5;
            const unsigned int bit  = c & 31u;
            const bool taken = ((allowed_bits[word] >> bit) & 1u) != 0u;
            if (!taken) {
                const float sc = scores[c];
                if (itopk_better(sc, (int)c, my_score, my_idx)) {
                    my_score = sc;
                    my_idx   = (int)c;
                }
            }
        }

        // Intra-wave reduce.
        itopk_wave_reduce(my_score, my_idx);

        // Stash wave-leader (lane 0) values, then cross-wave reduce in wave 0.
        if (lane == 0) {
            s_wave_score[wave] = my_score;
            s_wave_idx[wave]   = my_idx;
        }
        __syncthreads();

        if (wave == 0) {
            float w_score = (lane < (ITOPK_BLOCK / 32)) ? s_wave_score[lane] : ITOPK_NEG_INF;
            int   w_idx   = (lane < (ITOPK_BLOCK / 32)) ? s_wave_idx[lane]   : -1;
            // Reduce 8 lanes via the same primitive (last 3 levels do work; earlier are no-ops).
            itopk_wave_reduce(w_score, w_idx);
            if (lane == 0) {
                const int winner = (w_idx >= 0) ? w_idx : 0;
                selected[k] = winner;
                // Mark the winner. No atomic needed — single thread writes.
                const unsigned int word = (unsigned int)winner >> 5;
                const unsigned int bit  = (unsigned int)winner & 31u;
                allowed_bits[word] |= (1u << bit);
            }
        }
        __syncthreads();
    }

    // For k in [k_actual, top_k): write a sentinel so callers can detect
    // "no more rows" without re-reading n_comp / top_k. -1 fits Rust's i32.
    for (unsigned int k = k_actual + tid; k < top_k; k += ITOPK_BLOCK) {
        selected[k] = -1;
    }
}
