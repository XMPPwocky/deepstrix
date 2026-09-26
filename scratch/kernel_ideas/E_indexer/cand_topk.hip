// Candidates for indexer_topk_select_batched_ilp (gfx1201). Same signature and grid as
// production: grid (B) x 1024, args (selected, done, scores, n_idx_per, n_idx_stride, top_k).
//
// The production kernel (K/indexer_topk_bitonic.hip:600-796, TOPK_SEL_ILP=16) is a single
// 1024-thread WG per row: sample 2048 keys -> bitonic sort (66 LDS stages + barriers) ->
// 1-3 strided count passes over n -> compact >t into LDS -> [tie append] -> bitonic sort of
// 4096 (score, idx) pairs (78 LDS stages + barriers) -> write top_k. ATT: latency/barrier bound.
//
// Variants (all produce the IDENTICAL selection: same thresholds, same comparator, same
// deterministic total order; only the mechanics of the sort network and of the O(n) loads change):
//   topk_select_regsort  HYBRID bitonic sorts: stages with j < 32 exchange within a wave via
//                        __shfl_xor on registers (no LDS, no barrier); j >= 32 stages through LDS.
//                        Barriers per row: 66 -> 20 (sample), 78 -> 36 (final).
//   topk_select_v3       regsort + float4 loads in the count and compact passes (4x the bytes in
//                        flight per s_wait_loadcnt slot; the counts are integer sums so the
//                        result is identical; the compact slot order was already nondeterministic
//                        ahead of the total-order sort).
//   topk_select_v3_u8    v3 with U=8 float4 chunks (same bytes in flight as production's U=16
//                        dwords x 2).
#include <hip/hip_runtime.h>

#define TOPK_BLOCK     1024u
#define TOPK_SORT_N    4096u
#define TOPK_NEG_INF   (-3.4028235e38f)
#define TOPK_SEL_SAMPLE 2048u
#define TOPK_SEL_CAP    TOPK_SORT_N

__device__ __forceinline__ static bool topk_better(float av, unsigned int ai, float bv, unsigned int bi) {
    return av > bv || (av == bv && ai < bi);
}
__device__ __forceinline__ static unsigned int topk_key(float f) {
    unsigned int b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// ---------------------------------------------------------------------------------------------
// Hybrid descending bitonic sort of 2048 keys. Thread t owns elements t and t+1024.
// Stage (k, j): element i pairs with i^j; the element at the lower index keeps the larger value
// when (i & k) == 0 (descending half), the smaller otherwise. j >= 1024: both elements are in
// the same thread. 32 <= j < 1024: LDS exchange, double-buffered (one barrier per stage).
// j < 32: partner lane is in the same wave -> __shfl_xor. Result is written back to keys[].
// bufA/bufB: two 2048-word LDS buffers (bufA may be keys itself).
__device__ __forceinline__ static void bitonic_sort_keys2048_hyb(unsigned int* keys, unsigned int* bufB) {
    const unsigned int tid = threadIdx.x;
    unsigned int a = keys[tid], b = keys[tid + 1024u];
    const unsigned int ia = tid, ib = tid + 1024u;
    unsigned int* bufs[2] = {keys, bufB};
    unsigned int stage = 0u;
    #pragma unroll 1
    for (unsigned int k = 2u; k <= TOPK_SEL_SAMPLE; k <<= 1u) {
        #pragma unroll 1
        for (unsigned int j = k >> 1u; j > 0u; j >>= 1u) {
            unsigned int wa, wb;
            if (j >= 1024u) {
                wa = b; wb = a;
            } else if (j >= 32u) {
                unsigned int* buf = bufs[stage & 1u];
                stage++;
                buf[ia] = a; buf[ib] = b;
                __syncthreads();
                wa = buf[ia ^ j]; wb = buf[ib ^ j];
            } else {
                wa = __shfl_xor(a, (int)j); wb = __shfl_xor(b, (int)j);
            }
            const bool la = (ia & j) == 0u, da = (ia & k) == 0u;
            const bool lb = (ib & j) == 0u, db = (ib & k) == 0u;
            a = (la == da) ? max(a, wa) : min(a, wa);
            b = (lb == db) ? max(b, wb) : min(b, wb);
        }
    }
    __syncthreads();          // everyone done reading the last exchange buffer
    keys[ia] = a; keys[ib] = b;
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------
// Hybrid bitonic sort of 4096 (val, idx) pairs in LDS, descending by (val desc, idx asc).
// Thread t owns elements t + m*1024, m = 0..3. LDS stages (j >= 32) are done in place by the
// thread holding the lower index (as production); register phases handle j = 16..1 for every k
// (and all of k = 2..32) with __shfl_xor, then store back. Identical network and comparator.
// One register phase: load the thread's 4 pairs, run stages (k, j) for k in [k_lo, k_hi] with
// j = min(k/2, 16) .. 1 via __shfl_xor, store back, one barrier.
__device__ __forceinline__ static void reg_phase_pairs(float* __restrict__ vals, unsigned int* __restrict__ idxs,
                                                       unsigned int k_lo, unsigned int k_hi) {
    const unsigned int tid = threadIdx.x;
    float v[4]; unsigned int x[4];
    #pragma unroll
    for (unsigned int m = 0; m < 4u; m++) { v[m] = vals[tid + m * 1024u]; x[m] = idxs[tid + m * 1024u]; }
    #pragma unroll 1
    for (unsigned int k = k_lo; k <= k_hi; k <<= 1u) {
        #pragma unroll 1
        for (unsigned int j = min(k >> 1u, 16u); j > 0u; j >>= 1u) {
            #pragma unroll
            for (unsigned int m = 0; m < 4u; m++) {
                const unsigned int i = tid + m * 1024u;
                const float w = __shfl_xor(v[m], (int)j);
                const unsigned int wx = __shfl_xor(x[m], (int)j);
                const bool lower = (i & j) == 0u, desc = (i & k) == 0u;
                const bool mine_better = topk_better(v[m], x[m], w, wx);
                // lower&&desc or upper&&asc keeps the better element
                const bool keep_mine = (lower == desc) ? mine_better : !mine_better;
                if (!keep_mine) { v[m] = w; x[m] = wx; }
            }
        }
    }
    #pragma unroll
    for (unsigned int m = 0; m < 4u; m++) { vals[tid + m * 1024u] = v[m]; idxs[tid + m * 1024u] = x[m]; }
    __syncthreads();
}
__device__ __forceinline__ static void bitonic_sort_lds_hyb(float* __restrict__ vals, unsigned int* __restrict__ idxs) {
    const unsigned int tid = threadIdx.x;
    // k = 2..32 : one register phase covering every (k, j) with j < 32
    reg_phase_pairs(vals, idxs, 2u, 32u);
    #pragma unroll 1
    for (unsigned int k = 64u; k <= TOPK_SORT_N; k <<= 1u) {
        #pragma unroll 1
        for (unsigned int j = k >> 1u; j >= 32u; j >>= 1u) {
            for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
                const unsigned int other = i ^ j;
                if (other > i) {
                    const float av = vals[i], bv = vals[other];
                    const unsigned int ai = idxs[i], bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half ? topk_better(bv, bi, av, ai) : topk_better(av, ai, bv, bi);
                    if (swap) { vals[i] = bv; idxs[i] = bi; vals[other] = av; idxs[other] = ai; }
                }
            }
            __syncthreads();
        }
        reg_phase_pairs(vals, idxs, k, k);
    }
}

// ---------------------------------------------------------------------------------------------
template <unsigned int U, bool VEC4>
__device__ __forceinline__ static void topk_select_cand(
        int          * __restrict__ selected_b,
        unsigned int * __restrict__ done_b,
        const float  * __restrict__ scores_b,
        const unsigned int * __restrict__ n_idx_per,
        unsigned int                n_idx_stride,
        unsigned int                top_k) {
    const unsigned int bi  = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int n   = n_idx_per[bi];
    int          * __restrict__ selected = selected_b + (size_t)bi * top_k;
    const float  * __restrict__ scores   = scores_b   + (size_t)bi * n_idx_stride;

    __shared__ float        vals[TOPK_SORT_N];
    __shared__ unsigned int idxs[TOPK_SORT_N];
    __shared__ unsigned int sample[TOPK_SEL_SAMPLE];
    __shared__ unsigned int red_gt[TOPK_BLOCK / 32], red_ge[TOPK_BLOCK / 32];
    __shared__ unsigned int scan[TOPK_BLOCK];
    __shared__ unsigned int s_count;
    __shared__ unsigned int s_t, s_state;

    if (n == 0u) {
        for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) selected[i] = -1;
        if (tid == 0u) done_b[bi] = 1u;
        return;
    }

    unsigned int n_cand = 0u;
    if (n <= TOPK_SORT_N) {
        for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) {
            if (i < n) { vals[i] = scores[i]; idxs[i] = i; }
            else       { vals[i] = TOPK_NEG_INF; idxs[i] = 0xFFFFFFFFu; }
        }
        n_cand = n;
    } else {
        for (unsigned int j = tid; j < TOPK_SEL_SAMPLE; j += TOPK_BLOCK) {
            const unsigned int src = (unsigned int)(((unsigned long long)j * n) / TOPK_SEL_SAMPLE);
            sample[j] = topk_key(scores[src]);
        }
        __syncthreads();
        bitonic_sort_keys2048_hyb(sample, idxs);   // idxs (4096 words) is free here: second buffer
        const unsigned int r = (unsigned int)(((unsigned long long)top_k * TOPK_SEL_SAMPLE + n - 1u) / n);
        unsigned int idx_t = min(TOPK_SEL_SAMPLE - 1u, 2u * r + 32u);
        unsigned int lo_idx = 0u, hi_idx = TOPK_SEL_SAMPLE - 1u;
        unsigned int t = 0u; unsigned int count_gt = 0u, count_ge = 0u;
        bool ok = false, giveup = false;
        #pragma unroll 1
        for (int attempt = 0; attempt < 8 && !ok && !giveup; attempt++) {
            if (tid == 0u) s_t = sample[idx_t];
            __syncthreads();
            t = s_t;
            unsigned int c_gt = 0u, c_ge = 0u;
            unsigned int i;
            if (VEC4) {
                // float4 layout: chunk of 4096 elements, thread loads elements base + tid*4 .. +3
                unsigned int base = 0u;
                const float4* s4 = reinterpret_cast<const float4*>(scores);
                for (; base + U * 4096u <= n; base += U * 4096u) {
                    float4 f[U];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) f[u] = s4[(base >> 2) + u * 1024u + tid];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) {
                        unsigned int k0 = topk_key(f[u].x), k1 = topk_key(f[u].y), k2 = topk_key(f[u].z), k3 = topk_key(f[u].w);
                        c_gt += (k0 > t) + (k1 > t) + (k2 > t) + (k3 > t);
                        c_ge += (k0 >= t) + (k1 >= t) + (k2 >= t) + (k3 >= t);
                    }
                }
                for (; base + 4096u <= n; base += 4096u) {
                    const float4 f = s4[(base >> 2) + tid];
                    unsigned int k0 = topk_key(f.x), k1 = topk_key(f.y), k2 = topk_key(f.z), k3 = topk_key(f.w);
                    c_gt += (k0 > t) + (k1 > t) + (k2 > t) + (k3 > t);
                    c_ge += (k0 >= t) + (k1 >= t) + (k2 >= t) + (k3 >= t);
                }
                i = base + tid;
            } else {
                i = tid;
                for (; i + (U - 1u) * TOPK_BLOCK < n; i += U * TOPK_BLOCK) {
                    float f[U];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) f[u] = scores[i + u * TOPK_BLOCK];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) {
                        const unsigned int k = topk_key(f[u]);
                        c_gt += (k > t); c_ge += (k >= t);
                    }
                }
            }
            for (; i < n; i += TOPK_BLOCK) {
                const unsigned int k = topk_key(scores[i]);
                c_gt += (k > t); c_ge += (k >= t);
            }
            #pragma unroll
            for (unsigned int o = 16; o > 0; o >>= 1) { c_gt += __shfl_down(c_gt, o); c_ge += __shfl_down(c_ge, o); }
            if ((tid & 31u) == 0u) { red_gt[tid >> 5] = c_gt; red_ge[tid >> 5] = c_ge; }
            __syncthreads();
            if (tid == 0u) {
                unsigned int g = 0u, e = 0u;
                for (unsigned int w = 0; w < TOPK_BLOCK / 32; w++) { g += red_gt[w]; e += red_ge[w]; }
                red_gt[0] = g; red_ge[0] = e;
                unsigned int st = 0u;
                if (g <= TOPK_SEL_CAP && e >= top_k) st = 1u;
                else if (g > TOPK_SEL_CAP) {
                    hi_idx = idx_t;
                    if (idx_t == 0u) st = 2u; else idx_t = lo_idx + (idx_t - lo_idx) / 2u;
                } else {
                    lo_idx = idx_t;
                    if (idx_t == TOPK_SEL_SAMPLE - 1u) { st = 2u; }
                    else idx_t = idx_t + (hi_idx - idx_t + 1u) / 2u;
                }
                if (st == 0u && lo_idx + 1u >= hi_idx && attempt > 2) st = 2u;
                s_state = st;
            }
            __syncthreads();
            count_gt = red_gt[0]; count_ge = red_ge[0];
            ok = (s_state == 1u); giveup = (s_state == 2u);
            __syncthreads();
        }
        if (!ok) {
            if (tid == 0u) done_b[bi] = 0u;
            return;
        }
        if (tid == 0u) s_count = 0u;
        for (unsigned int i = tid; i < TOPK_SORT_N; i += TOPK_BLOCK) { vals[i] = TOPK_NEG_INF; idxs[i] = 0xFFFFFFFFu; }
        __syncthreads();
        {
            unsigned int i;
            if (VEC4) {
                unsigned int base = 0u;
                const float4* s4 = reinterpret_cast<const float4*>(scores);
                for (; base + U * 4096u <= n; base += U * 4096u) {
                    float4 f[U];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) f[u] = s4[(base >> 2) + u * 1024u + tid];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) {
                        const unsigned int e0 = base + u * 4096u + tid * 4u;
                        const float fv[4] = {f[u].x, f[u].y, f[u].z, f[u].w};
                        #pragma unroll
                        for (unsigned int q = 0; q < 4u; q++) {
                            if (topk_key(fv[q]) > t) {
                                const unsigned int slot = atomicAdd(&s_count, 1u);
                                vals[slot] = fv[q]; idxs[slot] = e0 + q;
                            }
                        }
                    }
                }
                for (; base + 4096u <= n; base += 4096u) {
                    const float4 f = s4[(base >> 2) + tid];
                    const unsigned int e0 = base + tid * 4u;
                    const float fv[4] = {f.x, f.y, f.z, f.w};
                    #pragma unroll
                    for (unsigned int q = 0; q < 4u; q++) {
                        if (topk_key(fv[q]) > t) {
                            const unsigned int slot = atomicAdd(&s_count, 1u);
                            vals[slot] = fv[q]; idxs[slot] = e0 + q;
                        }
                    }
                }
                i = base + tid;
            } else {
                i = tid;
                for (; i + (U - 1u) * TOPK_BLOCK < n; i += U * TOPK_BLOCK) {
                    float f[U];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) f[u] = scores[i + u * TOPK_BLOCK];
                    #pragma unroll
                    for (unsigned int u = 0; u < U; u++) {
                        if (topk_key(f[u]) > t) {
                            const unsigned int slot = atomicAdd(&s_count, 1u);
                            vals[slot] = f[u]; idxs[slot] = i + u * TOPK_BLOCK;
                        }
                    }
                }
            }
            for (; i < n; i += TOPK_BLOCK) {
                const float f = scores[i];
                if (topk_key(f) > t) {
                    const unsigned int slot = atomicAdd(&s_count, 1u);
                    vals[slot] = f; idxs[slot] = i;
                }
            }
        }
        __syncthreads();
        n_cand = count_gt;
        if (count_gt < top_k) {
            const unsigned int need  = top_k - count_gt;
            const unsigned int per   = (n + TOPK_BLOCK - 1u) / TOPK_BLOCK;
            const unsigned int my_lo = min(n, tid * per), my_hi = min(n, my_lo + per);
            unsigned int mine = 0u;
            for (unsigned int i = my_lo; i < my_hi; i++) mine += (topk_key(scores[i]) == t);
            scan[tid] = mine;
            __syncthreads();
            for (unsigned int o = 1u; o < TOPK_BLOCK; o <<= 1u) {
                const unsigned int v = (tid >= o) ? scan[tid - o] : 0u;
                __syncthreads();
                scan[tid] += v;
                __syncthreads();
            }
            const unsigned int before = scan[tid] - mine;
            unsigned int rank = before;
            for (unsigned int i = my_lo; i < my_hi && rank < need; i++) {
                const float f = scores[i];
                if (topk_key(f) == t) { vals[count_gt + rank] = f; idxs[count_gt + rank] = i; rank++; }
            }
            n_cand = count_gt + need;
            __syncthreads();
        }
    }
    __syncthreads();
    bitonic_sort_lds_hyb(vals, idxs);
    for (unsigned int i = tid; i < top_k; i += TOPK_BLOCK) {
        selected[i] = (i < n_cand) ? (int)idxs[i] : -1;
    }
    if (tid == 0u) done_b[bi] = 1u;
}

extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void topk_select_regsort(int* __restrict__ selected_b, unsigned int* __restrict__ done_b, const float* __restrict__ scores_b,
                         const unsigned int* __restrict__ n_idx_per, unsigned int n_idx_stride, unsigned int top_k) {
    topk_select_cand<16u, false>(selected_b, done_b, scores_b, n_idx_per, n_idx_stride, top_k);
}
extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void topk_select_v3(int* __restrict__ selected_b, unsigned int* __restrict__ done_b, const float* __restrict__ scores_b,
                    const unsigned int* __restrict__ n_idx_per, unsigned int n_idx_stride, unsigned int top_k) {
    topk_select_cand<16u, true>(selected_b, done_b, scores_b, n_idx_per, n_idx_stride, top_k);
}
extern "C" __global__ __launch_bounds__(TOPK_BLOCK)
void topk_select_v3_u8(int* __restrict__ selected_b, unsigned int* __restrict__ done_b, const float* __restrict__ scores_b,
                       const unsigned int* __restrict__ n_idx_per, unsigned int n_idx_stride, unsigned int top_k) {
    topk_select_cand<8u, true>(selected_b, done_b, scores_b, n_idx_per, n_idx_stride, top_k);
}
