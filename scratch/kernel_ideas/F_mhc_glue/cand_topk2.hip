// cand_topk2 -- router_topk_par with each argmax pass as ONE hardware wave reduction per wave
// (__ockl_wfred_max_u64: DPP / permlane, no LDS traffic, no ds_bpermute) + a 16-slot LDS merge.
//
// Layout = production's: 512 threads, thread tid owns expert tid (tid < n_expert). Per pass each
// thread forms a 64-bit SORT KEY = (monotonic(score) << 32) | ~tid, so "max key" == production's
// argmax predicate (strictly greater score wins; equal scores -> lower index; invalid (-1) last:
// key 0). Each wave reduces its 32 keys with __ockl_wfred_max_u64, lane 0 parks the wave max in
// LDS (double-buffered by pass parity), one __syncthreads, and EVERY thread reads the 16 wave
// maxima (broadcast LDS reads) and takes the max -> the winner is uniform in registers, no second
// barrier; the owner thread masks its score to -INF. One barrier per pass instead of ten.
//
// cand_topk.hip (wave-register version, 5-step ds_bpermute butterflies + 16-slot serial scans on
// a single wave) measured SLOWER than production (14 vs 7 us at b=1): the serial chain of a lone
// wave is longer than production's 9-level 512-thread LDS tree. This one keeps the parallelism.
//
// BIT-EXACT vs router_topk_par for finite logits: monotonic(f32) orders every non-NaN float as
// `<` does except that it orders -0.0 below +0.0 (production's `==` treats them equal). Scores are
// sqrt(softplus(.)) + bias (+ prior): sqrt(softplus) is +0 or positive, so a -0.0 score cannot
// occur (x + y == -0 needs both -0). NaN logits are undefined in both. Weights, range: as production.
// Same signature and launch geometry (grid (b) x 512).

#include <hip/hip_runtime.h>

#ifndef ROUTER_MAX_EXPERTS
#define ROUTER_MAX_EXPERTS 256
#endif
#define ROUTER_MAX_USED    8
#define ROUTER_MAX_ALT     4
#define RT_NSEL (ROUTER_MAX_USED + ROUTER_MAX_ALT)
#define RT_WAVES (ROUTER_MAX_EXPERTS / 32)

extern "C" __device__ unsigned long long __ockl_wfred_max_u64(unsigned long long);
extern "C" __device__ float __ockl_wfred_max_f32(float);
extern "C" __device__ float __ockl_wfred_min_f32(float);

__device__ inline float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

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
