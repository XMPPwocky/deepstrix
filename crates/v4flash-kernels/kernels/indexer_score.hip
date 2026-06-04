// indexer_score — per-(comp-row) score for the V4 Flash indexer. Mirrors
// the inner loop of ds4's indexer_allowed_decode_one_decode_scratch
// (ds4.c:7015-7025):
//
//   for c in 0..n_comp:
//       s = 0
//       for h in 0..n_head:
//           dot = sum_i kv[c, i] * q[h, i]        (i over head_dim)
//           if dot < 0: dot = 0                   (ReLU)
//           s += dot * head_weights[h]
//       scores[c] = s
//
// One workgroup per comp row. Each workgroup uses 256 threads to fan over
// (n_head=64 × head_dim=128 = 8192) head-dim entries; per-head dot products
// are reduced in shared mem, ReLU'd, then weighted-summed.
//
// For V4 Flash: n_head=DS4_N_INDEXER_HEAD=64, head_dim=DS4_N_INDEXER_HEAD_DIM=128.
//
// `index_comp_kv` is f16-stored (the indexer compressor emits f16
// post-roundtrip; ds4's CPU path matches this in its `index_comp_post_fp8`
// dump tag). Q and head_weights are f32 (live, no quantization).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define INDEXER_BLOCK 256

extern "C" __global__ void indexer_score(
        float          * __restrict__ scores,        // [n_comp]
        const float    * __restrict__ q,             // [n_head, head_dim]
        const float    * __restrict__ head_weights,  // [n_head]
        const _Float16 * __restrict__ index_comp_kv, // [n_comp, head_dim] f16
        unsigned int     n_comp,
        unsigned int     n_head,
        unsigned int     head_dim) {
    const unsigned int c = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (c >= n_comp) return;

    __shared__ float partial[INDEXER_BLOCK];

    const _Float16 *kv = index_comp_kv + (unsigned long long)c * head_dim;
    float total = 0.0f;

    for (unsigned int h = 0; h < n_head; h++) {
        const float *qh = q + (unsigned long long)h * head_dim;
        float p = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += INDEXER_BLOCK) {
            p += (float)kv[i] * qh[i];
        }
        partial[tid] = p;
        __syncthreads();

        for (unsigned int stride = INDEXER_BLOCK / 2; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            float dot = partial[0];
            if (dot < 0.0f) dot = 0.0f;
            total += dot * head_weights[h];
        }
    }

    if (tid == 0) {
        scores[c] = total;
    }
}
