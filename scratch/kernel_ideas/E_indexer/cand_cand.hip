// Candidates for candidate_threshold (K/candidate_blocks.hip:54, gfx1201). Same signature:
//   (threshold[B], block_score[B, nb_stride], n_per[B], nb_stride, cb_size, topk), grid (B).
// Production: ONE 256-thread WG per row runs a 4-pass 8-bit radix select over n_blocks =
// ceil(n/8) block scores (29.4K at 235K), with a serial 256-bin scan by thread 0 per pass:
// 42-91 us per call at 65K-235K for ~120 KB of L2-resident data. Latency-bound.
//
// candidate_threshold_ilp: block 1024 (launch geometry knob), U independent loads per thread per
// pass (4x the threads x U loads in flight), otherwise the same passes / same histogram / same
// scan -> the same 32-bit key (the topk-th largest key is unique as a value; the select is
// exact), so bit-identical thresholds.
#include <hip/hip_runtime.h>

#define CB_HIST 256

__device__ __forceinline__ unsigned int cb_key(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

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
extern "C" __global__ __launch_bounds__(512)
void candidate_threshold_ilp512(unsigned int* __restrict__ threshold, const float* __restrict__ block_score,
                                const unsigned int* __restrict__ n_per, unsigned int nb_stride, unsigned int cb_size, unsigned int topk) {
    cand_threshold_body<512u, 8u>(threshold, block_score, n_per, nb_stride, cb_size, topk);
}
