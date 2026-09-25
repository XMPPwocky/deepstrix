// router_topk — V4 Flash routed-MoE expert selection.
//
// Inputs:
//   logits[n_expert]            — pre-softplus router scores
//   bias[n_expert] (optional)   — per-expert selection bias (learned router)
//
// Outputs:
//   selected[n_used]            — chosen expert indices, top-K descending
//   weights[n_used]             — normalized expert weights, sum*scale = scale
//
// Computation (mirrors `forward.rs::softplus_stable / topk_desc` and
// ds4.c:5260-5290):
//   probs[i]     = sqrt(softplus_stable(logits[i]))
//   selection[i] = probs[i] + (bias ? bias[i] : 0)
//   selected     = top-K(selection)            // descending insertion sort
//   w_raw[j]     = probs[selected[j]]
//   sum          = max(sum(w_raw), weight_eps)
//   weights[j]   = w_raw[j] / sum * expert_weight_scale
//
// Single block, 256 threads (= N_EXPERT). Top-K is computed by thread 0
// since K=6 and the per-thread cost is dwarfed by everything else.

#include <hip/hip_runtime.h>

#ifndef ROUTER_MAX_EXPERTS
#define ROUTER_MAX_EXPERTS 256   // V4.1 build passes 512 (384 experts, padded)
#endif
#define ROUTER_MAX_USED    8
// Alternatives (ranks n_used+1..n_used+n_alt) -- see router_topk_par.hip.
#define ROUTER_MAX_ALT     4

__device__ inline float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

extern "C" __global__ void router_topk(
        int          * __restrict__ selected,
        float        * __restrict__ weights,
        const float  * __restrict__ logits,
        const float  * __restrict__ bias,         // may be nullptr
        unsigned int   n_expert,
        unsigned int   n_used,
        float          expert_weight_scale,
        float          weight_eps,
        int          * __restrict__ alts,          // may be nullptr; [n_alt]
        unsigned int   n_alt,
        float        * __restrict__ alt_w,         // may be nullptr; see router_topk_par.hip
        // Cache-prior arguments (see router_topk_par.hip). This serial REFERENCE
        // does not implement the prior: it must be nullptr here. `orig_sel`
        // (= the plain picks) and `range_out` are filled when given.
        const float  * __restrict__ prior,
        unsigned int   n_protect,
        unsigned int   prior_dry,
        int          * __restrict__ orig_sel,
        float        * __restrict__ range_out) {
    __shared__ float s_probs[ROUTER_MAX_EXPERTS];
    __shared__ float s_select[ROUTER_MAX_EXPERTS];
    __shared__ int   s_sel[ROUTER_MAX_USED + ROUTER_MAX_ALT];
    __shared__ float s_w[ROUTER_MAX_USED];
    __shared__ float s_sum;

    const unsigned int tid = threadIdx.x;
    if (alts == nullptr) n_alt = 0;
    if (n_alt > ROUTER_MAX_ALT) n_alt = ROUTER_MAX_ALT;
    // Sorting into n_used + n_alt slots leaves the first n_used unchanged: an
    // element that lands at slot >= n_used never shifts an earlier one.
    const unsigned int n_keep = n_used + n_alt;

    if (tid < n_expert) {
        const float p = sqrtf(softplus_stable(logits[tid]));
        s_probs[tid] = p;
        s_select[tid] = (bias != nullptr) ? (p + bias[tid]) : p;
    }
    __syncthreads();

    if (tid == 0) {
        // Initialize.
        for (unsigned int j = 0; j < n_keep; ++j) {
            s_sel[j] = -1;
        }
        // Descending insertion sort over n_expert items (mirrors
        // forward.rs::topk_desc).
        for (unsigned int i = 0; i < n_expert; ++i) {
            for (unsigned int j = 0; j < n_keep; ++j) {
                const bool slot_empty = (s_sel[j] < 0);
                const bool greater = !slot_empty && (s_select[i] > s_select[s_sel[j]]);
                if (slot_empty || greater) {
                    // Shift [j+1, n_used) right by one.
                    for (int m = (int)n_keep - 1; m > (int)j; --m) {
                        s_sel[m] = s_sel[m - 1];
                    }
                    s_sel[j] = (int)i;
                    break;
                }
            }
        }
        // Raw weights + sum.
        float sum = 0.0f;
        for (unsigned int j = 0; j < n_used; ++j) {
            const float p = s_probs[s_sel[j]];
            s_w[j] = p;
            sum += p;
        }
        if (sum < weight_eps) sum = weight_eps;
        if (range_out != nullptr) {
            float mx = -INFINITY, mn = INFINITY;
            for (unsigned int i = 0; i < n_expert; ++i) {
                mx = fmaxf(mx, s_select[i]);
                mn = fminf(mn, s_select[i]);
            }
            range_out[0] = mx - mn;
        }
        (void)prior;
        (void)n_protect;
        (void)prior_dry;
        s_sum = sum;
    }
    __syncthreads();

    if (tid < n_used) {
        selected[tid] = s_sel[tid];
        weights[tid]  = s_w[tid] / s_sum * expert_weight_scale;
        if (orig_sel != nullptr) orig_sel[tid] = s_sel[tid];
    } else if (tid < n_used + n_alt) {
        const int a = s_sel[tid];
        alts[tid - n_used] = a;
        if (alt_w != nullptr) {
            alt_w[tid - n_used] = ((a >= 0) ? s_probs[a] : 0.0f) / s_sum * expert_weight_scale;
        }
    }
}
