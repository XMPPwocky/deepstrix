// cand_topk -- router_topk_par with the K argmax passes done by ONE WAVE in registers
// (wave shuffles) instead of 512-thread LDS tree reductions with 10 __syncthreads each.
//
// Production (K/router_topk_par.hip, grid (b) x 512): phase 1 (sqrt(softplus) + bias [+ prior])
// by 384 threads, then per pass: LDS copy + 9-level tree + winner mask = 10 barriers; with the
// cache-prior on (production: V41_SUB=3, n_protect 2, range_out set) that is 6 + 4 argmax
// passes + 2 range reductions = ~120 barriers per token.
//
// Here: phase 1 unchanged (all threads of the block, one barrier), then wave 0 holds the 384
// scores as 12 registers per lane (expert e lives in lane e % 32, slot e / 32) and each argmax
// pass is 12 compares + a 5-step xor butterfly on (value, index) pairs -- no LDS, no barrier.
// The winner is masked in its owner's register.
//
// BIT-EXACT by construction: probs / scores / boost are the same float expressions; an argmax
// under the strict total order (value desc, index asc; index -1 = invalid last) has ONE answer
// whatever the reduction tree, and the production predicate is exactly that order (strictly
// greater wins, ties -> lower index, valid beats -1); range = fmaxf/fminf trees are exact;
// weights are s_probs[sel[j]] summed in j order / clamped / scaled as production.
//
// Same signature as router_topk_par. `router_topk_wave` is meant for block 512 (phase 1 in
// parallel); `router_topk_wave_b2` is the same body for a 32-thread block (phase 1 by the one
// wave, 12 transcendental chains per lane, no barrier at all).

#include <hip/hip_runtime.h>

#ifndef ROUTER_MAX_EXPERTS
#define ROUTER_MAX_EXPERTS 256
#endif
#define ROUTER_MAX_USED    8
#define ROUTER_MAX_ALT     4
#define RT_PL   (ROUTER_MAX_EXPERTS / 32)          // register slots per lane (16 at 512)
#define RT_NSEL (ROUTER_MAX_USED + ROUTER_MAX_ALT)  // 12

__device__ inline float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

// production's argmax predicate: does (bo, io) beat (bs, is)?
__device__ __forceinline__ bool rt_beats(float bo, int io, float bs, int is) {
    return bo > bs || (bo == bs && io >= 0 && (is < 0 || io < is));
}

// One argmax over the wave-distributed scores v[]; returns the winner (uniform across the
// wave, -1 if nothing valid) and masks it to -INF in its owner's slot.
__device__ __forceinline__ int rt_wave_argmax(float (&v)[RT_PL], const unsigned int lane,
                                              const unsigned int n_expert) {
    float bv = -INFINITY;
    int bi = -1;
    #pragma unroll
    for (int j = 0; j < RT_PL; ++j) {
        const int e = (int)lane + 32 * j;
        if ((unsigned)e < n_expert && rt_beats(v[j], e, bv, bi)) { bv = v[j]; bi = e; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor(bv, off, 32);
        const int oi = __shfl_xor(bi, off, 32);
        if (rt_beats(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    if (bi >= 0) {
        #pragma unroll
        for (int j = 0; j < RT_PL; ++j)
            if ((int)lane + 32 * j == bi) v[j] = -INFINITY;
    }
    return bi;
}

__device__ __forceinline__ void rt_mask(float (&v)[RT_PL], const unsigned int lane, const int w) {
    if (w < 0) return;
    #pragma unroll
    for (int j = 0; j < RT_PL; ++j)
        if ((int)lane + 32 * j == w) v[j] = -INFINITY;
}

__device__ __forceinline__ void router_topk_wave_body(
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
    __shared__ float s_scores[ROUTER_MAX_EXPERTS];
    __shared__ float s_boost[ROUTER_MAX_EXPERTS];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;

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

    // Phase 1 (all threads of the block): probs, scores, boosted scores -- production's expressions.
    for (unsigned int i = tid; i < n_expert; i += blockDim.x) {
        const float p = sqrtf(softplus_stable(logits[i]));
        s_probs[i] = p;
        const float sc = (bias != nullptr) ? (p + bias[i]) : p;
        s_scores[i] = sc;
        if (prior != nullptr) s_boost[i] = sc + prior[i];
    }
    __syncthreads();
    if (tid >= 32u) return;
    // ---- wave 0 only from here: no workgroup barrier below.

    float sc[RT_PL], bo[RT_PL];
    #pragma unroll
    for (int j = 0; j < RT_PL; ++j) {
        const unsigned int e = lane + 32u * j;
        sc[j] = (e < n_expert) ? s_scores[e] : -INFINITY;
        bo[j] = (e < n_expert && prior != nullptr) ? s_boost[e] : -INFINITY;
    }

    // Phase 1b: the score range (max - min), fmaxf/fminf are order-independent.
    if (range_out != nullptr) {
        float mx = -INFINITY, mn = INFINITY;
        #pragma unroll
        for (int j = 0; j < RT_PL; ++j) {
            const unsigned int e = lane + 32u * j;
            if (e < n_expert) { mx = fmaxf(mx, sc[j]); mn = fminf(mn, sc[j]); }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            mx = fmaxf(mx, __shfl_xor(mx, off, 32));
            mn = fminf(mn, __shfl_xor(mn, off, 32));
        }
        if (lane == 0) range_out[0] = mx - mn;
    }

    int sel[RT_NSEL], orig[ROUTER_MAX_USED];
    #pragma unroll
    for (int j = 0; j < RT_NSEL; ++j) sel[j] = -1;
    #pragma unroll
    for (int j = 0; j < ROUTER_MAX_USED; ++j) orig[j] = -1;

    if (prior == nullptr) {
        #pragma unroll
        for (int k = 0; k < RT_NSEL; ++k)
            if ((unsigned)k < n_used + n_alt) sel[k] = rt_wave_argmax(sc, lane, n_expert);
        if (orig_sel != nullptr) {
            #pragma unroll
            for (int j = 0; j < ROUTER_MAX_USED; ++j)
                if (lane == (unsigned)j && (unsigned)j < n_used) orig_sel[j] = sel[j];
        }
    } else {
        #pragma unroll
        for (int k = 0; k < ROUTER_MAX_USED; ++k)
            if ((unsigned)k < n_used) orig[k] = rt_wave_argmax(sc, lane, n_expert);
        if (orig_sel != nullptr) {
            #pragma unroll
            for (int j = 0; j < ROUTER_MAX_USED; ++j)
                if (lane == (unsigned)j && (unsigned)j < n_used) orig_sel[j] = orig[j];
        }
        #pragma unroll
        for (int k = 0; k < ROUTER_MAX_USED; ++k) {
            if ((unsigned)k < n_protect) { sel[k] = orig[k]; rt_mask(bo, lane, orig[k]); }
        }
        #pragma unroll
        for (int k = 0; k < RT_NSEL; ++k)
            if ((unsigned)k >= n_protect && (unsigned)k < n_used + n_alt) sel[k] = rt_wave_argmax(bo, lane, n_expert);
    }

    if (prior != nullptr && prior_dry != 0u) {
        #pragma unroll
        for (int j = 0; j < ROUTER_MAX_USED; ++j) {
            if ((unsigned)j < n_used) { const int t = sel[j]; sel[j] = orig[j]; orig[j] = t; }
        }
        if (orig_sel != nullptr) {
            #pragma unroll
            for (int j = 0; j < ROUTER_MAX_USED; ++j)
                if (lane == (unsigned)j && (unsigned)j < n_used) orig_sel[j] = orig[j];
        }
    }

    // Phase 3: weights from the unbiased probs of the final set, summed in j order (uniform).
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

    // Phase 4: emit (lane j <-> slot j, as production's tid j).
    #pragma unroll
    for (int j = 0; j < ROUTER_MAX_USED; ++j) {
        if (lane == (unsigned)j && (unsigned)j < n_used) {
            selected[j] = sel[j];
            weights[j] = w[j] / sum * expert_weight_scale;
        }
    }
    #pragma unroll
    for (int j = 0; j < RT_NSEL; ++j) {
        if (lane == (unsigned)j && (unsigned)j >= n_used && (unsigned)j < n_used + n_alt) {
            const int a = sel[j];
            alts[j - n_used] = a;
            if (alt_w != nullptr) alt_w[j - n_used] = ((a >= 0) ? s_probs[a] : 0.0f) / sum * expert_weight_scale;
        }
    }
}

#define RT_KERNEL(name)                                                                              \
    extern "C" __global__ void name(                                                                 \
            int* __restrict__ selected, float* __restrict__ weights, const float* __restrict__ logits, \
            const float* __restrict__ bias, unsigned int n_expert, unsigned int n_used,               \
            float expert_weight_scale, float weight_eps, int* __restrict__ alts, unsigned int n_alt,  \
            float* __restrict__ alt_w, const float* __restrict__ prior, unsigned int n_protect,       \
            unsigned int prior_dry, int* __restrict__ orig_sel, float* __restrict__ range_out) {      \
        router_topk_wave_body(selected, weights, logits, bias, n_expert, n_used, expert_weight_scale, \
                              weight_eps, alts, n_alt, alt_w, prior, n_protect, prior_dry, orig_sel,  \
                              range_out);                                                            \
    }

RT_KERNEL(router_topk_wave)      // block 512 (production launch geometry)
RT_KERNEL(router_topk_wave_b2)   // block 32: phase 1 by the single wave
