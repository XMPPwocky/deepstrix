// router_topk_par — parallel reimplementation of router_topk (M14b).
//
// Original kernel computes the top-K via a serial insertion sort run by
// thread 0 of a 256-thread block, leaving 99.6% of the wavefront idle.
// This variant uses K=6 passes of a parallel argmax (tree reduction in
// shared memory), masking the winner after each pass.
//
// Numerics are identical to the serial version for tie-free inputs;
// ties are broken by lowest-index, matching the serial insertion sort's
// "first one that's >=, take it" behavior.

#include <hip/hip_runtime.h>

#define ROUTER_MAX_EXPERTS 256
#define ROUTER_MAX_USED    8

__device__ inline float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

extern "C" __global__ void router_topk_par(
        int          * __restrict__ selected,
        float        * __restrict__ weights,
        const float  * __restrict__ logits,
        const float  * __restrict__ bias,
        unsigned int   n_expert,
        unsigned int   n_used,
        float          expert_weight_scale,
        float          weight_eps) {
    __shared__ float s_probs[ROUTER_MAX_EXPERTS];
    __shared__ float s_scores[ROUTER_MAX_EXPERTS]; // mutable; winners set to -INF
    __shared__ float buf[ROUTER_MAX_EXPERTS];
    __shared__ int   idxs[ROUTER_MAX_EXPERTS];
    __shared__ int   s_sel[ROUTER_MAX_USED];
    __shared__ float s_w[ROUTER_MAX_USED];
    __shared__ float s_sum;

    const unsigned int tid = threadIdx.x;

    // Phase 1: probs + selection scores.
    if (tid < n_expert) {
        const float p = sqrtf(softplus_stable(logits[tid]));
        s_probs[tid] = p;
        s_scores[tid] = (bias != nullptr) ? (p + bias[tid]) : p;
    }
    __syncthreads();

    // Phase 2: K passes of parallel argmax.
    // The reduce is in [0, n_expert) so we pad with -INF for tid in
    // [n_expert, ROUTER_MAX_EXPERTS).
    for (unsigned int k = 0; k < n_used; ++k) {
        // Init buf + idxs.
        if (tid < n_expert) {
            buf[tid] = s_scores[tid];
            idxs[tid] = (int)tid;
        } else {
            buf[tid] = -INFINITY;
            idxs[tid] = -1;
        }
        __syncthreads();

        // Tree reduce: stride-halving. For ROUTER_MAX_EXPERTS=256:
        // strides 128, 64, 32, 16, 8, 4, 2, 1.
        for (unsigned int stride = ROUTER_MAX_EXPERTS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float bo = buf[tid + stride];
                const int io = idxs[tid + stride];
                const float bs = buf[tid];
                const int is = idxs[tid];
                // Strictly-greater wins; on ties keep the lower index to
                // match the serial insertion sort's tie-breaking.
                if (bo > bs || (bo == bs && io >= 0 && (is < 0 || io < is))) {
                    buf[tid] = bo;
                    idxs[tid] = io;
                }
            }
            __syncthreads();
        }

        // Thread 0 records the winner and masks it for the next pass.
        if (tid == 0) {
            const int w = idxs[0];
            s_sel[k] = w;
            if (w >= 0) s_scores[w] = -INFINITY;
        }
        __syncthreads();
    }

    // Phase 3: weights + sum.
    if (tid == 0) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < n_used; ++j) {
            const int w = s_sel[j];
            const float p = (w >= 0) ? s_probs[w] : 0.0f;
            s_w[j] = p;
            sum += p;
        }
        if (sum < weight_eps) sum = weight_eps;
        s_sum = sum;
    }
    __syncthreads();

    // Phase 4: emit.
    if (tid < n_used) {
        selected[tid] = s_sel[tid];
        weights[tid]  = s_w[tid] / s_sum * expert_weight_scale;
    }
}
