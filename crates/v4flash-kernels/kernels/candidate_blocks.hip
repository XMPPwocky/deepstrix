// ARCH_SPEC §1.5 — hierarchical candidate pool (level one of the two-level top-k).
//
// Mirrors `inference/model.py::select_candidate_blocks`:
//   block score = max over CB_SIZE consecutive positions (-inf pads the tail),
//   the block holding the newest position is pinned to +inf, keep the top
//   CB_TOPK blocks, drop any pick that came back -inf.
//
// Scores are the batched indexer layout: row b at `scores + b*stride`, with
// `n_per[b]` reachable positions.

#include <hip/hip_runtime.h>

#define CB_HIST 256
#define CB_THREADS 256

// Float -> order-preserving uint32, so a radix select can walk the bits.
// -inf maps low, +inf maps high, which is exactly the pin/drop behaviour we want.
__device__ __forceinline__ unsigned int cb_key(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One workgroup per (row, block-chunk). block_score[b*nb_stride + blk].
extern "C" __global__ __launch_bounds__(CB_THREADS)
void candidate_block_max(
    float* __restrict__ block_score,
    const float* __restrict__ scores,
    const unsigned int* __restrict__ n_per,
    unsigned int stride,
    unsigned int nb_stride,
    unsigned int cb_size)
{
    const unsigned int b = blockIdx.y;
    const unsigned int n = n_per[b];
    const unsigned int n_blocks = (n + cb_size - 1u) / cb_size;
    const unsigned int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= n_blocks) return;

    const unsigned int lo = blk * cb_size;
    const unsigned int hi = min(lo + cb_size, n);
    float m = -INFINITY;
    for (unsigned int p = lo; p < hi; ++p) {
        float v = scores[(size_t)b * stride + p];
        m = fmaxf(m, v);
    }
    // The newest position's block is only partly filled; pin it so it cannot be
    // outscored by an older, full block.
    if (n > 0u && blk == (n - 1u) / cb_size) m = INFINITY;
    block_score[(size_t)b * nb_stride + blk] = m;
}

// One workgroup per row. Radix-selects the CB_TOPK-th largest block key and
// writes it to `threshold[b]`. Blocks whose key >= threshold are the candidates.
extern "C" __global__ __launch_bounds__(CB_THREADS)
void candidate_threshold(
    unsigned int* __restrict__ threshold,
    const float* __restrict__ block_score,
    const unsigned int* __restrict__ n_per,
    unsigned int nb_stride,
    unsigned int cb_size,
    unsigned int topk)
{
    const unsigned int b = blockIdx.x;
    const unsigned int n = n_per[b];
    const unsigned int n_blocks = (n + cb_size - 1u) / cb_size;

    __shared__ unsigned int hist[CB_HIST];
    __shared__ unsigned int s_prefix, s_mask;
    __shared__ int s_remaining;
    __shared__ unsigned int s_done;

    if (threadIdx.x == 0) {
        s_prefix = 0u; s_mask = 0u; s_done = 0u;
        // Fewer blocks than topk: everything is kept, so the threshold is 0.
        s_remaining = (int)min(topk, n_blocks);
        if (n_blocks <= topk) s_done = 1u;
    }
    __syncthreads();
    if (s_done) {
        if (threadIdx.x == 0) threshold[b] = 0u;
        return;
    }

    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned int i = threadIdx.x; i < CB_HIST; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
            unsigned int k = cb_key(block_score[(size_t)b * nb_stride + i]);
            if ((k & s_mask) == s_prefix) {
                atomicAdd(&hist[(k >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int rem = s_remaining;
            for (int bin = CB_HIST - 1; bin >= 0; --bin) {
                int c = (int)hist[bin];
                if (c >= rem) {
                    s_prefix |= ((unsigned int)bin) << shift;
                    s_mask |= 0xFFu << shift;
                    break;
                }
                rem -= c;
            }
            s_remaining = rem;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) threshold[b] = s_prefix;
}

// candidate_threshold_ilp — candidate_threshold at block 1024 with U=8
// independent block-score loads per thread per radix pass (production: block
// 256, one dependent load per iteration). Same 4 passes, same histogram, same
// serial scan -> the same exact topk-th largest key: bit-identical thresholds.
// 2026-09-26 sweep (E_indexer/candidate_threshold_ilp, dGPU, L20 build): 235K
// 87.4 -> 43.8 us, 131K 57.3 -> 32.4, 65K 41.9 -> 30.0 (b = 1 and 4).
// BUT the reviewer measured it 2.0-2.7x SLOWER at prefill shapes (b = 512/1024
// rows: 15 of 16 waves idle through thread 0's serial bin scan in the
// throughput regime), so CandidateBlocks selects it only for batch <= 8
// (V41_CAND_THRESH_ILP); prefill keeps candidate_threshold. Only this kernel
// is 1024-wide: candidate_block_max / mask_apply stay __launch_bounds__(256).
template <unsigned BLOCK, unsigned U>
__device__ __forceinline__ void cand_threshold_body(
    unsigned int* __restrict__ threshold, const float* __restrict__ block_score,
    const unsigned int* __restrict__ n_per, unsigned int nb_stride, unsigned int cb_size, unsigned int topk) {
    const unsigned int b = blockIdx.x;
    const unsigned int n = n_per[b];
    const unsigned int n_blocks = (n + cb_size - 1u) / cb_size;
    const float* bs = block_score + (size_t)b * nb_stride;

    __shared__ unsigned int hist[CB_HIST];
    __shared__ unsigned int s_prefix, s_mask;
    __shared__ int s_remaining;
    __shared__ unsigned int s_done;

    if (threadIdx.x == 0) {
        s_prefix = 0u; s_mask = 0u; s_done = 0u;
        s_remaining = (int)min(topk, n_blocks);
        if (n_blocks <= topk) s_done = 1u;
    }
    __syncthreads();
    if (s_done) {
        if (threadIdx.x == 0) threshold[b] = 0u;
        return;
    }

    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned int i = threadIdx.x; i < CB_HIST; i += BLOCK) hist[i] = 0u;
        __syncthreads();
        const unsigned int mask = s_mask, prefix = s_prefix;
        unsigned int i = threadIdx.x;
        for (; i + (U - 1u) * BLOCK < n_blocks; i += U * BLOCK) {
            float f[U];
            #pragma unroll
            for (unsigned int u = 0; u < U; u++) f[u] = bs[i + u * BLOCK];
            #pragma unroll
            for (unsigned int u = 0; u < U; u++) {
                const unsigned int k = cb_key(f[u]);
                if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFFu], 1u);
            }
        }
        for (; i < n_blocks; i += BLOCK) {
            const unsigned int k = cb_key(bs[i]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int rem = s_remaining;
            for (int bin = CB_HIST - 1; bin >= 0; --bin) {
                int c = (int)hist[bin];
                if (c >= rem) {
                    s_prefix |= ((unsigned int)bin) << shift;
                    s_mask |= 0xFFu << shift;
                    break;
                }
                rem -= c;
            }
            s_remaining = rem;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) threshold[b] = s_prefix;
}

extern "C" __global__ __launch_bounds__(1024)
void candidate_threshold_ilp(unsigned int* __restrict__ threshold, const float* __restrict__ block_score,
                             const unsigned int* __restrict__ n_per, unsigned int nb_stride, unsigned int cb_size, unsigned int topk) {
    cand_threshold_body<1024u, 8u>(threshold, block_score, n_per, nb_stride, cb_size, topk);
}

// scores[b][p] = -inf for every position whose block missed the cut.
// A block whose score is -inf is dropped even if it clears the threshold
// ("leftover picks came back -inf"), which matters when n_blocks < topk only.
extern "C" __global__ __launch_bounds__(CB_THREADS)
void candidate_mask_apply(
    float* __restrict__ scores,
    const float* __restrict__ block_score,
    const unsigned int* __restrict__ threshold,
    const unsigned int* __restrict__ n_per,
    unsigned int stride,
    unsigned int nb_stride,
    unsigned int cb_size)
{
    const unsigned int b = blockIdx.y;
    const unsigned int n = n_per[b];
    const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const unsigned int t = threshold[b];
    if (t == 0u) return;                       // vacuous: every block kept
    const float bs = block_score[(size_t)b * nb_stride + (p / cb_size)];
    if (cb_key(bs) < t || bs == -INFINITY) {
        scores[(size_t)b * stride + p] = -INFINITY;
    }
}
