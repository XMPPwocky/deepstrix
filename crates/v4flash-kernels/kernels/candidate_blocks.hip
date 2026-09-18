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
