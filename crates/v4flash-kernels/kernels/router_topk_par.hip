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

#ifndef ROUTER_MAX_EXPERTS
#define ROUTER_MAX_EXPERTS 256   // V4.1 build passes 512 (384 experts, padded)
#endif
#define ROUTER_MAX_USED    8
// Ranks n_used+1 .. n_used+n_alt ("alternatives"), written to `alts` when it
// is non-null: box 2 substitutes one for a pick it does not hold resident
// (docs/v41/BOX2_MISS_SUBSTITUTION.md). They are extra argmax passes AFTER the
// first n_used, so `selected` / `weights` are bit-identical whatever n_alt is.
#define ROUTER_MAX_ALT     4

__device__ inline float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

// One parallel argmax over `scores[0, n_expert)` (every thread must call it).
// Returns the winner (lowest index on ties; -1 if all are -INF) and masks it
// to -INF in `scores`.
__device__ inline int argmax_pass(float* scores, float* buf, int* idxs, int* s_win, unsigned int n_expert, unsigned int tid) {
    if (tid < n_expert) {
        buf[tid] = scores[tid];
        idxs[tid] = (int)tid;
    } else {
        buf[tid] = -INFINITY;
        idxs[tid] = -1;
    }
    __syncthreads();
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
    if (tid == 0) {
        const int w = idxs[0];
        *s_win = w;
        if (w >= 0) scores[w] = -INFINITY;
    }
    __syncthreads();
    return *s_win;
}

extern "C" __global__ void router_topk_par(
        int          * __restrict__ selected,
        float        * __restrict__ weights,
        const float  * __restrict__ logits,
        const float  * __restrict__ bias,
        unsigned int   n_expert,
        unsigned int   n_used,
        float          expert_weight_scale,
        float          weight_eps,
        int          * __restrict__ alts,          // may be nullptr; [n_alt] per token
        unsigned int   n_alt,
        // May be nullptr; [n_alt] per token. prob(alt) / (top-n_used prob sum)
        // * scale: on the SAME scale as `weights`, so a caller that swaps pick
        // j for alt k renormalizes exactly (ref.Gate) by dividing everything by
        // 1 - weights[j]/scale + alt_w[k]/scale.
        float        * __restrict__ alt_w,
        // CACHE-PRIOR (Skliar et al. 2024, arXiv 2412.00099; box-2 miss
        // substitution V41_SUB=3). May be nullptr = off, and then everything
        // below is bit-identical to the plain router. [n_expert], shared by all
        // tokens: added to the SELECTION score of each expert (the caller puts
        // lambda * Delta_layer on experts held by the box that computes them,
        // 0 elsewhere). The first `n_protect` picks of the ORIGINAL ranking are
        // always kept; the remaining slots (and the alternatives) are the argmax
        // of score + prior. `weights` still come from the unbiased probs of the
        // final set: exact ref.Gate renormalization.
        const float  * __restrict__ prior,
        unsigned int   n_protect,
        // DRY RUN: compute the prior's selection but emit the PLAIN picks and
        // weights in `selected`/`weights`, and the prior's picks in `orig_sel`
        // (roles swapped), so the caller can count would-be swaps without
        // changing any numerics.
        unsigned int   prior_dry,
        // May be nullptr; [n_used] per token: the ORIGINAL top-n_used (no
        // prior), so the caller can see what the prior displaced.
        int          * __restrict__ orig_sel,
        // May be nullptr; [1] per token: max - min of the selection scores over
        // the n_expert experts (the paper's per-token logit range; the caller
        // keeps the running average Delta_layer).
        float        * __restrict__ range_out) {
    __shared__ float s_probs[ROUTER_MAX_EXPERTS];
    __shared__ float s_scores[ROUTER_MAX_EXPERTS]; // mutable; winners set to -INF
    __shared__ float s_boost[ROUTER_MAX_EXPERTS];  // score + prior (cache-prior only)
    __shared__ float buf[ROUTER_MAX_EXPERTS];
    __shared__ int   idxs[ROUTER_MAX_EXPERTS];
    __shared__ int   s_sel[ROUTER_MAX_USED + ROUTER_MAX_ALT];
    __shared__ int   s_orig[ROUTER_MAX_USED];
    __shared__ float s_w[ROUTER_MAX_USED];
    __shared__ float s_sum;
    __shared__ int   s_win;

    const unsigned int tid = threadIdx.x;

    // Batched over blockIdx.x: each block handles one token. logits/selected/
    // weights are indexed by the block; bias is shared. For the single-token
    // launch (grid.x == 1) blockIdx.x == 0, so these offsets are no-ops and
    // the result is identical to the unbatched kernel.
    logits   += (size_t)blockIdx.x * (size_t)n_expert;
    selected += (size_t)blockIdx.x * (size_t)n_used;
    weights  += (size_t)blockIdx.x * (size_t)n_used;
    if (alts == nullptr) n_alt = 0;
    if (n_alt > ROUTER_MAX_ALT) n_alt = ROUTER_MAX_ALT;
    if (alts != nullptr) alts += (size_t)blockIdx.x * (size_t)n_alt;
    if (alt_w != nullptr) alt_w += (size_t)blockIdx.x * (size_t)n_alt;
    if (orig_sel != nullptr) orig_sel += (size_t)blockIdx.x * (size_t)n_used;
    if (range_out != nullptr) range_out += blockIdx.x;
    if (n_protect > n_used) n_protect = n_used;

    // Phase 1: probs + selection scores (+ the cache-boosted copy).
    if (tid < n_expert) {
        const float p = sqrtf(softplus_stable(logits[tid]));
        s_probs[tid] = p;
        const float sc = (bias != nullptr) ? (p + bias[tid]) : p;
        s_scores[tid] = sc;
        if (prior != nullptr) s_boost[tid] = sc + prior[tid];
    }
    __syncthreads();

    // Phase 1b: the token's selection-score range, max - min.
    if (range_out != nullptr) {
        buf[tid] = (tid < n_expert) ? s_scores[tid] : -INFINITY;
        __syncthreads();
        for (unsigned int stride = ROUTER_MAX_EXPERTS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) buf[tid] = fmaxf(buf[tid], buf[tid + stride]);
            __syncthreads();
        }
        const float mx = buf[0];
        __syncthreads();
        buf[tid] = (tid < n_expert) ? s_scores[tid] : INFINITY;
        __syncthreads();
        for (unsigned int stride = ROUTER_MAX_EXPERTS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) buf[tid] = fminf(buf[tid], buf[tid + stride]);
            __syncthreads();
        }
        if (tid == 0) range_out[0] = mx - buf[0];
        __syncthreads();
    }

    if (prior == nullptr) {
        // Phase 2 (plain): K (+ alternatives) passes of parallel argmax.
        // Passes n_used.. are the alternatives; they only mask further
        // winners, so the first n_used are unchanged.
        for (unsigned int k = 0; k < n_used + n_alt; ++k) {
            const int w = argmax_pass(s_scores, buf, idxs, &s_win, n_expert, tid);
            if (tid == 0) s_sel[k] = w;
        }
        __syncthreads();
        if (orig_sel != nullptr && tid < n_used) orig_sel[tid] = s_sel[tid];
    } else {
        // Phase 2a (cache-prior): the ORIGINAL top-n_used, on the plain scores.
        for (unsigned int k = 0; k < n_used; ++k) {
            const int w = argmax_pass(s_scores, buf, idxs, &s_win, n_expert, tid);
            if (tid == 0) s_orig[k] = w;
        }
        __syncthreads();
        if (orig_sel != nullptr && tid < n_used) orig_sel[tid] = s_orig[tid];
        // Phase 2b: keep the original top-n_protect, take them out of the
        // boosted pool, fill the rest by score + prior.
        if (tid == 0) {
            for (unsigned int k = 0; k < n_protect; ++k) {
                const int w = s_orig[k];
                s_sel[k] = w;
                if (w >= 0) s_boost[w] = -INFINITY;
            }
        }
        __syncthreads();
        for (unsigned int k = n_protect; k < n_used + n_alt; ++k) {
            const int w = argmax_pass(s_boost, buf, idxs, &s_win, n_expert, tid);
            if (tid == 0) s_sel[k] = w;
        }
        __syncthreads();
    }

    // DRY RUN: swap roles, so the plain picks are emitted (and weighted) and
    // `orig_sel` carries what the prior would have chosen.
    if (prior != nullptr && prior_dry != 0) {
        if (tid == 0) {
            for (unsigned int j = 0; j < n_used; ++j) {
                const int t = s_sel[j];
                s_sel[j] = s_orig[j];
                s_orig[j] = t;
            }
        }
        __syncthreads();
        if (orig_sel != nullptr && tid < n_used) orig_sel[tid] = s_orig[tid];
        __syncthreads();
    }

    // Phase 3: weights + sum, from the unbiased probs of the FINAL set.
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
    } else if (tid < n_used + n_alt) {
        const int a = s_sel[tid];
        alts[tid - n_used] = a;
        if (alt_w != nullptr) {
            alt_w[tid - n_used] = ((a >= 0) ? s_probs[a] : 0.0f) / s_sum * expert_weight_scale;
        }
    }
}

// router_topk_wfred — router_topk_par with each argmax pass as ONE hardware
// wave reduction per wave (__ockl_wfred_max_u64: DPP / permlane, no LDS tree)
// + a 16-slot LDS merge: one __syncthreads per pass instead of ten.
// 2026-09-26 sweep (F_mhc_glue/topk_wfred): decode cache-prior config
// (n_protect 2, orig_sel, range_out) b=1 11.15 -> 9.20 us, b=4 11.15 -> 9.05
// (graph; direct launch -18% too); plain b=1 neutral; plain b=512 a 10% LOSS
// (33.0 -> 36.5), so RouterTopk::launch_batched_ex runs it only for b <= 16
// under V41_TOPK_WFRED. Bit-exact (reviewer: 243 config x input classes incl.
// all-equal, all -INF, extreme logits, injected -0.0).
//
// Layout = production's: 512 threads, thread tid owns expert tid (tid <
// n_expert). Per pass each thread forms a 64-bit SORT KEY = (monotonic(score)
// << 32) | ~tid, so "max key" == production's argmax predicate (strictly
// greater score wins; equal scores -> lower index; invalid (-1) last: key 0).
// Each wave reduces its 32 keys, lane 0 parks the wave max in LDS (double-
// buffered by pass parity), one __syncthreads, and EVERY thread reads the 16
// wave maxima and takes the max -> the winner is uniform in registers, no
// second barrier; the owner thread masks its score to -INF.
//
// monotonic(f32) orders every non-NaN float as `<` does except that it orders
// -0.0 below +0.0 (production's `==` treats them equal). Scores are
// sqrt(softplus(.)) + bias (+ prior): sqrt(softplus) is +0 or positive, so a
// -0.0 score cannot occur (x + y == -0 needs both -0). NaN logits are undefined
// in both. Weights, range: as production. Same signature and grid (b) x 512.
#define RT_NSEL (ROUTER_MAX_USED + ROUTER_MAX_ALT)
#define RT_WAVES (ROUTER_MAX_EXPERTS / 32)

extern "C" __device__ unsigned long long __ockl_wfred_max_u64(unsigned long long);
extern "C" __device__ float __ockl_wfred_max_f32(float);
extern "C" __device__ float __ockl_wfred_min_f32(float);

// monotonic map f32 -> u32 (order-preserving for all non-NaN values)
__device__ __forceinline__ unsigned int rt_ord(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One argmax over the block's scores (thread tid holds v, valid if tid < n_expert). Every thread
// must call it. Returns the winner (uniform), masks it in the owner's register.
__device__ __forceinline__ int rt_block_argmax(float& v, const bool valid, const unsigned int tid,
                                               unsigned long long* s_wk, const unsigned int pass) {
    const unsigned long long key = valid ? (((unsigned long long)rt_ord(v) << 32) | (unsigned int)(~tid)) : 0ull;
    const unsigned long long wmax = __ockl_wfred_max_u64(key);
    unsigned long long* slot = s_wk + (pass & 1u) * RT_WAVES;
    if ((tid & 31u) == 0u) slot[tid >> 5] = wmax;
    __syncthreads();
    unsigned long long best = 0ull;
    #pragma unroll
    for (int w = 0; w < RT_WAVES; ++w) { const unsigned long long k = slot[w]; best = (k > best) ? k : best; }
    const int win = (best == 0ull) ? -1 : (int)(~(unsigned int)best);
    if (win >= 0 && tid == (unsigned int)win) v = -INFINITY;
    return win;
}

extern "C" __global__ __launch_bounds__(ROUTER_MAX_EXPERTS) void router_topk_wfred(
        int          * __restrict__ selected,
        float        * __restrict__ weights,
        const float  * __restrict__ logits,
        const float  * __restrict__ bias,
        unsigned int   n_expert,
        unsigned int   n_used,
        float          expert_weight_scale,
        float          weight_eps,
        int          * __restrict__ alts,
        unsigned int   n_alt,
        float        * __restrict__ alt_w,
        const float  * __restrict__ prior,
        unsigned int   n_protect,
        unsigned int   prior_dry,
        int          * __restrict__ orig_sel,
        float        * __restrict__ range_out) {
    __shared__ float s_probs[ROUTER_MAX_EXPERTS];
    __shared__ unsigned long long s_wk[2 * RT_WAVES];
    __shared__ float s_wf[2 * RT_WAVES];

    const unsigned int tid = threadIdx.x;
    const bool valid = tid < n_expert;

    logits   += (size_t)blockIdx.x * (size_t)n_expert;
    selected += (size_t)blockIdx.x * (size_t)n_used;
    weights  += (size_t)blockIdx.x * (size_t)n_used;
    if (alts == nullptr) n_alt = 0;
    if (n_alt > ROUTER_MAX_ALT) n_alt = ROUTER_MAX_ALT;
    if (alts != nullptr) alts += (size_t)blockIdx.x * (size_t)n_alt;
    if (alt_w != nullptr) alt_w += (size_t)blockIdx.x * (size_t)n_alt;
    if (orig_sel != nullptr) orig_sel += (size_t)blockIdx.x * (size_t)n_used;
    if (range_out != nullptr) range_out += blockIdx.x;
    if (n_protect > n_used) n_protect = n_used;
    if (n_used > ROUTER_MAX_USED) n_used = ROUTER_MAX_USED;

    // Phase 1: production's expressions; scores stay in registers.
    float sc = -INFINITY, bo = -INFINITY;
    if (valid) {
        const float p = sqrtf(softplus_stable(logits[tid]));
        s_probs[tid] = p;
        sc = (bias != nullptr) ? (p + bias[tid]) : p;
        if (prior != nullptr) bo = sc + prior[tid];
    }
    unsigned int pass = 0;

    // Phase 1b: score range (fmaxf/fminf are order-independent).
    if (range_out != nullptr) {
        const float mx = __ockl_wfred_max_f32(valid ? sc : -INFINITY);
        const float mn = __ockl_wfred_min_f32(valid ? sc : INFINITY);
        if ((tid & 31u) == 0u) { s_wf[tid >> 5] = mx; s_wf[RT_WAVES + (tid >> 5)] = mn; }
        __syncthreads();
        if (tid == 0) {
            float gx = -INFINITY, gn = INFINITY;
            #pragma unroll
            for (int w = 0; w < RT_WAVES; ++w) { gx = fmaxf(gx, s_wf[w]); gn = fminf(gn, s_wf[RT_WAVES + w]); }
            range_out[0] = gx - gn;
        }
    } else {
        __syncthreads();   // s_probs visible before phase 3 (also covered by the pass barriers)
    }

    int sel[RT_NSEL], orig[ROUTER_MAX_USED];
    #pragma unroll
    for (int j = 0; j < RT_NSEL; ++j) sel[j] = -1;
    #pragma unroll
    for (int j = 0; j < ROUTER_MAX_USED; ++j) orig[j] = -1;

    if (prior == nullptr) {
        #pragma unroll
        for (int k = 0; k < RT_NSEL; ++k)
            if ((unsigned)k < n_used + n_alt) sel[k] = rt_block_argmax(sc, valid, tid, s_wk, pass++);
        if (orig_sel != nullptr) {
            #pragma unroll
            for (int j = 0; j < ROUTER_MAX_USED; ++j)
                if (tid == (unsigned)j && (unsigned)j < n_used) orig_sel[j] = sel[j];
        }
    } else {
        #pragma unroll
        for (int k = 0; k < ROUTER_MAX_USED; ++k)
            if ((unsigned)k < n_used) orig[k] = rt_block_argmax(sc, valid, tid, s_wk, pass++);
        if (orig_sel != nullptr) {
            #pragma unroll
            for (int j = 0; j < ROUTER_MAX_USED; ++j)
                if (tid == (unsigned)j && (unsigned)j < n_used) orig_sel[j] = orig[j];
        }
        #pragma unroll
        for (int k = 0; k < ROUTER_MAX_USED; ++k) {
            if ((unsigned)k < n_protect) {
                sel[k] = orig[k];
                if (orig[k] >= 0 && tid == (unsigned)orig[k]) bo = -INFINITY;
            }
        }
        #pragma unroll
        for (int k = 0; k < RT_NSEL; ++k)
            if ((unsigned)k >= n_protect && (unsigned)k < n_used + n_alt) sel[k] = rt_block_argmax(bo, valid, tid, s_wk, pass++);
    }

    if (prior != nullptr && prior_dry != 0u) {
        #pragma unroll
        for (int j = 0; j < ROUTER_MAX_USED; ++j)
            if ((unsigned)j < n_used) { const int t = sel[j]; sel[j] = orig[j]; orig[j] = t; }
        if (orig_sel != nullptr) {
            #pragma unroll
            for (int j = 0; j < ROUTER_MAX_USED; ++j)
                if (tid == (unsigned)j && (unsigned)j < n_used) orig_sel[j] = orig[j];
        }
    }

    // Phase 3 (uniform): weights from the unbiased probs of the final set, summed in j order.
    float w[ROUTER_MAX_USED];
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < ROUTER_MAX_USED; ++j) {
        w[j] = 0.0f;
        if ((unsigned)j < n_used) {
            const int s = sel[j];
            const float p = (s >= 0) ? s_probs[s] : 0.0f;
            w[j] = p;
            sum += p;
        }
    }
    if (sum < weight_eps) sum = weight_eps;

    // Phase 4: emit.
    #pragma unroll
    for (int j = 0; j < ROUTER_MAX_USED; ++j) {
        if (tid == (unsigned)j && (unsigned)j < n_used) {
            selected[j] = sel[j];
            weights[j] = w[j] / sum * expert_weight_scale;
        }
    }
    #pragma unroll
    for (int j = 0; j < RT_NSEL; ++j) {
        if (tid == (unsigned)j && (unsigned)j >= n_used && (unsigned)j < n_used + n_alt) {
            const int a = sel[j];
            alts[j - n_used] = a;
            if (alt_w != nullptr) alt_w[j - n_used] = ((a >= 0) ? s_probs[a] : 0.0f) / sum * expert_weight_scale;
        }
    }
}
