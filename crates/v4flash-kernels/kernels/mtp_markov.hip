// mtp_markov -- the DSpark exit's markov loop on the device (`V41_MTP_DEVICE_MARKOV`): each draft position
// reads the PREVIOUS position's token from device memory, so the 5 positions enqueue back to back with no
// host round trip (the host loop synchronised and read back once per position).
//
// mtp_markov_gather_f16   grid (1), block (rank)
//   out[k] = f32(table[toks[idx] * rank + k]) -- the host's `markov_row` (an exact f16 -> f32 widening),
//   with the token taken from `toks[idx]`. An out-of-range token writes NaN (the host path errored; the
//   NaN bias makes the position's top-M empty of finite logits and the host check reports it).
//
// mtp_draft_draw          grid (1), block (64), thread 0 works
//   `draft_dist` (mtp.rs) in f64, operation for operation: candidates = the top-M (ids, logits) with
//   id >= 0 and a finite logit, sorted by logit descending, ties to the lower id; w = exp((l - max) / T);
//   min-p prefix cut; top-p prefix cut over the running sum; q = w / acc; the draw is the first candidate
//   whose cumulative q reaches u, else the last kept. Writes the draw to toks[out_idx]. The host still
//   computes q from the same (ids, logits) after the loop and checks it draws the same token (a
//   difference needs u within an ulp of a cumulative boundary: device and host exp may differ in the
//   last bit).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define MD_MAX_M 64

extern "C" __global__ void mtp_markov_gather_f16(
        float          * __restrict__ out,     // [rank]
        const __half   * __restrict__ table,   // [n_vocab, rank]
        const int      * __restrict__ toks,    // [>= idx + 1]
        unsigned int     idx,
        unsigned int     n_vocab,
        unsigned int     rank) {
    const int id = toks[idx];
    const unsigned int k = threadIdx.x;
    if (k >= rank) return;
    if (id < 0 || (unsigned int)id >= n_vocab) {
        out[k] = __int_as_float(0x7fc00000);
        return;
    }
    out[k] = __half2float(table[(size_t)id * (size_t)rank + k]);
}

extern "C" __global__ void mtp_draft_draw(
        int            * __restrict__ toks,    // the draw goes to toks[out_idx]
        const int      * __restrict__ ids,     // [m] top-M ids
        const float    * __restrict__ logits,  // [m] their biased logits
        unsigned int     m,
        float            temperature,
        float            top_p,
        float            min_p_rel,
        float            u,
        unsigned int     out_idx) {
    // Block of 64 lanes: lane 0 filters + sorts (the top-M arrives sorted, so the insertion sort is a
    // linear check), every lane then computes ONE w_i = exp((l_i - max) / T) -- the same f64 expression
    // the serial loop evaluated -- and lane 0 does the min-p / top-p / draw prefix scans in index order,
    // so every sum is the host's left-to-right sum.
    __shared__ int    cid[MD_MAX_M];
    __shared__ float  clg[MD_MAX_M];
    __shared__ double w[MD_MAX_M];
    __shared__ unsigned int n_sh;
    const unsigned int tid = threadIdx.x;
    if (tid == 0) {
        unsigned int n = 0;
        for (unsigned int i = 0; i < m && i < MD_MAX_M; i++) {
            const float l = logits[i];
            if (ids[i] >= 0 && isfinite(l)) {
                cid[n] = ids[i];
                clg[n] = l;
                n++;
            }
        }
        // Stable insertion sort: logit descending, ties to the lower id (the host's sort_by key).
        for (unsigned int i = 1; i < n; i++) {
            const int ti = cid[i];
            const float tl = clg[i];
            int j = (int)i - 1;
            while (j >= 0 && (clg[j] < tl || (clg[j] == tl && cid[j] > ti))) {
                cid[j + 1] = cid[j];
                clg[j + 1] = clg[j];
                j--;
            }
            cid[j + 1] = ti;
            clg[j + 1] = tl;
        }
        n_sh = n;
    }
    __syncthreads();
    const unsigned int n = n_sh;
    if (n == 0) {
        if (tid == 0) toks[out_idx] = -1;
        return;
    }
    if (tid < n) {
        w[tid] = exp(((double)clg[tid] - (double)clg[0]) / (double)temperature);
    }
    __syncthreads();
    if (tid != 0) return;
    // min-p: the survivors are a prefix; the top one always survives.
    double min_p = (double)min_p_rel;
    min_p = min_p < 0.0 ? 0.0 : (min_p > 1.0 ? 1.0 : min_p);
    unsigned int len = n;
    for (unsigned int i = 0; i < n; i++) {
        if (w[i] < min_p) {
            len = i;
            break;
        }
    }
    if (len < 1) len = 1;
    double z = 0.0;
    for (unsigned int i = 0; i < len; i++) z += w[i];
    double tp = (double)top_p;
    tp = tp < 0.0 ? 0.0 : (tp > 1.0 ? 1.0 : tp);
    unsigned int keep = 0;
    double acc = 0.0;
    while (keep < len && (keep == 0 || acc < tp * z)) {
        acc += w[keep];
        keep++;
    }
    double target = (double)u;
    target = target < 0.0 ? 0.0 : (target > 1.0 ? 1.0 : target);
    double cum = 0.0;
    int pick = cid[keep - 1];
    for (unsigned int i = 0; i < keep; i++) {
        cum += w[i] / acc;
        if (cum >= target) {
            pick = cid[i];
            break;
        }
    }
    toks[out_idx] = pick;
}

// mtp_argmax_partial   grid (n_wg), block (256)  +  mtp_argmax_final   grid (1), block (256)
//   `argmax_one` (softmax_sample.hip) over many WGs: its result is the lowest index among the maximal
//   non-NaN values (a -inf entry beats the sentinel n; NaN never wins; nothing found -> n - 1), and that
//   is a total order, so a two-stage reduce with the SAME comparison returns the same index. Writes
//   toks[out_idx]. argmax_one is one WG walking 129280 floats (~135 us); this is ~n_wg x faster.
#define AM_BLOCK 256
__device__ __forceinline__ void am_take(float v, int i, float &bv, int &bi) {
    if (v > bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
    }
}

extern "C" __global__ void mtp_argmax_partial(
        float          * __restrict__ part_v,  // [n_wg]
        int            * __restrict__ part_i,  // [n_wg]
        const float    * __restrict__ logits,  // [n]
        unsigned int     n) {
    __shared__ float s_val[AM_BLOCK];
    __shared__ int   s_idx[AM_BLOCK];
    const unsigned int tid = threadIdx.x;
    float bv = -INFINITY;
    int   bi = (int)n;
    for (unsigned int i = blockIdx.x * AM_BLOCK + tid; i < n; i += gridDim.x * AM_BLOCK) {
        am_take(logits[i], (int)i, bv, bi);
    }
    s_val[tid] = bv;
    s_idx[tid] = bi;
    __syncthreads();
    for (unsigned int stride = AM_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v = s_val[tid];
            int   i = s_idx[tid];
            am_take(s_val[tid + stride], s_idx[tid + stride], v, i);
            s_val[tid] = v;
            s_idx[tid] = i;
        }
        __syncthreads();
    }
    if (tid == 0) {
        part_v[blockIdx.x] = s_val[0];
        part_i[blockIdx.x] = s_idx[0];
    }
}

extern "C" __global__ void mtp_argmax_final(
        int            * __restrict__ toks,
        const float    * __restrict__ part_v,
        const int      * __restrict__ part_i,
        unsigned int     n_wg,
        unsigned int     n,
        unsigned int     out_idx) {
    __shared__ float s_val[AM_BLOCK];
    __shared__ int   s_idx[AM_BLOCK];
    const unsigned int tid = threadIdx.x;
    float bv = -INFINITY;
    int   bi = (int)n;
    for (unsigned int w = tid; w < n_wg; w += AM_BLOCK) {
        am_take(part_v[w], part_i[w], bv, bi);
    }
    s_val[tid] = bv;
    s_idx[tid] = bi;
    __syncthreads();
    for (unsigned int stride = AM_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v = s_val[tid];
            int   i = s_idx[tid];
            am_take(s_val[tid + stride], s_idx[tid + stride], v, i);
            s_val[tid] = v;
            s_idx[tid] = i;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int idx = s_idx[0];
        if (idx >= (int)n) idx = (int)n - 1;
        toks[out_idx] = idx;
    }
}

// mtp_topm_chunk   grid (TM_WG), block (TM_BLOCK)  +  mtp_topm_final   grid (1), block (1024)
//   The exit's top-M (M <= 64) over one vocabulary row, spread over TM_WG work-groups with FIXED work (no sample,
//   no give-up): each WG bitonic-sorts its chunk (<= TM_CHUNK) by a 64-bit composite and keeps its best 64; one WG
//   sorts the TM_WG x 64 survivors and writes the best M ids in order. The composite is
//   (key(score) << 32) | ~idx, key = the indexer's monotonic float key with -0.0 folded into +0.0 -- so larger
//   composite == better(a, b) of the bitonic chain (score desc, ties to the LOWER idx). The best M of a total order
//   are unique, so the ids and their order equal the chain's. Padding is composite 0 (worse than every element:
//   ~idx is never 0 for idx < 2^32 - 1); a padded slot that reaches the output writes -1 (the chain's sentinel).
#define TM_WG 64u
#define TM_CHUNK 2048u
#define TM_BLOCK 256u
#define TM_KEEP 64u

__device__ __forceinline__ static unsigned long long tm_composite(float f, unsigned int idx) {
    if (f == 0.0f) f = 0.0f;   // -0.0 -> +0.0: the chain's float compare treats them as equal
    unsigned int b = __float_as_uint(f);
    const unsigned int key = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((unsigned long long)key << 32) | (unsigned long long)(~idx);
}

// Descending bitonic sort of n (power of two) composites in LDS by nthreads threads.
__device__ __forceinline__ static void tm_sort_desc(unsigned long long* a, unsigned int n, unsigned int nthreads) {
    const unsigned int tid = threadIdx.x;
    for (unsigned int k = 2u; k <= n; k <<= 1u) {
        for (unsigned int j = k >> 1u; j > 0u; j >>= 1u) {
            for (unsigned int i = tid; i < n; i += nthreads) {
                const unsigned int o = i ^ j;
                if (o > i) {
                    const unsigned long long x = a[i], y = a[o];
                    const bool desc = (i & k) == 0u;
                    if (desc ? (y > x) : (x > y)) { a[i] = y; a[o] = x; }
                }
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void mtp_topm_chunk(
        unsigned long long * __restrict__ part,    // [TM_WG * TM_KEEP]
        const float        * __restrict__ logits,  // [n]
        unsigned int         n) {
    __shared__ unsigned long long a[TM_CHUNK];
    const unsigned int chunk = (n + TM_WG - 1u) / TM_WG;   // <= TM_CHUNK (launcher checks)
    const unsigned int base = blockIdx.x * chunk;
    for (unsigned int i = threadIdx.x; i < TM_CHUNK; i += TM_BLOCK) {
        const unsigned int g = base + i;
        a[i] = (i < chunk && g < n) ? tm_composite(logits[g], g) : 0ull;
    }
    __syncthreads();
    tm_sort_desc(a, TM_CHUNK, TM_BLOCK);
    for (unsigned int i = threadIdx.x; i < TM_KEEP; i += TM_BLOCK) part[blockIdx.x * TM_KEEP + i] = a[i];
}

extern "C" __global__ __launch_bounds__(1024) void mtp_topm_final(
        int                      * __restrict__ selected,   // [m]
        const unsigned long long * __restrict__ part,       // [TM_WG * TM_KEEP]
        unsigned int               m) {
    __shared__ unsigned long long a[TM_WG * TM_KEEP];
    for (unsigned int i = threadIdx.x; i < TM_WG * TM_KEEP; i += 1024u) a[i] = part[i];
    __syncthreads();
    tm_sort_desc(a, TM_WG * TM_KEEP, 1024u);
    for (unsigned int i = threadIdx.x; i < m; i += 1024u) {
        const unsigned long long c = a[i];
        selected[i] = c == 0ull ? -1 : (int)(~(unsigned int)(c & 0xFFFFFFFFull));
    }
}

// =====================================================================================================================
// mtp_rx_pass x3 + mtp_rx_compact -- the exit's top-M (M <= 64) over one vocabulary row by RADIX SELECT, exact:
// the result is the best M by (key desc, idx asc) where key = the monotonic float key (-0.0 folded into +0.0), i.e.
// the bitonic chain's better(a, b) (score desc, ties to the lower idx), in that order. Fixed work, no sampling.
//   pass p (grid RX_WG x RX_BLOCK): key bits [shift, shift + nbits) = 31..21, 20..10, 9..0. Each WG histograms its
//     chunk's keys that still match the threshold prefix fixed by earlier passes (LDS) and adds its non-zero bins
//     into ONE global histogram H[0 .. RX_BINS) with atomics (the key's top bits cluster: a few dozen per WG); pass 2
//     also writes its row to H[RX_BINS + w * 1024 ..]. The LAST WG to finish (atomic counter, self-resetting) reads
//     the global histogram (and zeroes it for the next pass) and picks the bin
//     where the count from the top reaches `need`: prefix |= bin << shift, need -= count above. After pass 2 the
//     prefix is the exact threshold key Kt, need = how many key == Kt elements to take (lowest idx first), and the
//     last WG also writes each WG's offset among the key == Kt elements (exclusive prefix of H[w][bin]).
//   compact (grid RX_WG x RX_BLOCK): key > Kt -> cand[atomic slot] (fewer than M of them); key == Kt -> taken in
//     index order (tiles of RX_BLOCK + a block scan + the WG's offset) while the global rank < need, into
//     cand[count_gt + rank]. The last WG sorts the M composites and writes the ids.
// State st[]: [0] prefix, [1] need, [2] pass counter, [3] compact counter, [4] gt slot counter, [8 + w] eq offsets.
// =====================================================================================================================
#define RX_WG 64u
#define RX_BLOCK 256u
#define RX_BINS 2048u
#define RX_ST_EQOFF 8u

__device__ __forceinline__ static unsigned int rx_key(float f) {
    if (f == 0.0f) f = 0.0f;
    const unsigned int b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

extern "C" __global__ __launch_bounds__(256) void mtp_rx_pass(
        unsigned int  * __restrict__ H,       // [RX_WG * RX_BINS]
        unsigned int  * __restrict__ st,      // state
        const float   * __restrict__ logits,  // [n]
        unsigned int    n,
        unsigned int    pass,                 // 0, 1, 2
        unsigned int    m) {
    __shared__ unsigned int hist[RX_BINS];
    __shared__ unsigned int part[RX_BLOCK];
    __shared__ unsigned int s_last, s_bin, s_cum;
    const unsigned int shift = pass == 0u ? 21u : (pass == 1u ? 10u : 0u);
    const unsigned int nbits = pass == 2u ? 10u : 11u;
    const unsigned int nb = 1u << nbits;
    const unsigned int hi = shift + nbits;                 // 32 for pass 0: no prefix filter
    const unsigned int tid = threadIdx.x;
    const unsigned int prefix = pass == 0u ? 0u : st[0];
    for (unsigned int b = tid; b < nb; b += RX_BLOCK) hist[b] = 0u;
    __syncthreads();
    const unsigned int chunk = (n + RX_WG - 1u) / RX_WG;
    const unsigned int lo = blockIdx.x * chunk;
    const unsigned int end = min(lo + chunk, n);
    for (unsigned int i = lo + tid; i < end; i += RX_BLOCK) {
        const unsigned int k = rx_key(logits[i]);
        if (hi == 32u || (k >> hi) == (prefix >> hi)) atomicAdd(&hist[(k >> shift) & (nb - 1u)], 1u);
    }
    __syncthreads();
    for (unsigned int b = tid; b < nb; b += RX_BLOCK) {
        const unsigned int c = hist[b];
        if (c != 0u) atomicAdd(&H[b], c);
        if (pass == 2u) H[RX_BINS + blockIdx.x * 1024u + b] = c;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0u) s_last = (atomicAdd(&st[2], 1u) == RX_WG - 1u) ? 1u : 0u;
    __syncthreads();
    if (s_last == 0u) return;
    __threadfence();
    // Last WG: totals per bin (into hist), then the crossing bin scanning from the top.
    for (unsigned int b = tid; b < nb; b += RX_BLOCK) {
        hist[b] = __atomic_load_n(&H[b], __ATOMIC_RELAXED);
        H[b] = 0u;   // the global histogram is clean for the next pass / the next call
    }
    __syncthreads();
    // Thread t owns the bins_per = nb / RX_BLOCK bins nb-1-t*bins_per .. down (descending order).
    const unsigned int per = nb / RX_BLOCK;
    unsigned int sum = 0u;
    for (unsigned int j = 0u; j < per; j++) sum += hist[nb - 1u - tid * per - j];
    part[tid] = sum;
    __syncthreads();
    if (tid == 0u) {
        const unsigned int need = pass == 0u ? m : st[1];
        unsigned int cum = 0u, t = 0u;
        while (t < RX_BLOCK - 1u && cum + part[t] < need) { cum += part[t]; t++; }
        unsigned int bin = nb - 1u - t * per;
        for (unsigned int j = 0u; j < per; j++) {
            bin = nb - 1u - t * per - j;
            if (cum + hist[bin] >= need || j == per - 1u) break;
            cum += hist[bin];
        }
        s_bin = bin;
        s_cum = cum;
        st[0] = (pass == 0u ? 0u : prefix) | (bin << shift);
        st[1] = need - cum;
        st[2] = 0u;
    }
    __syncthreads();
    if (pass == 2u && tid == 0u) {
        // Each WG's offset among the key == Kt elements, in WG (= index) order.
        unsigned int off = 0u;
        for (unsigned int w = 0u; w < RX_WG; w++) {
            st[RX_ST_EQOFF + w] = off;
            off += __atomic_load_n(&H[RX_BINS + w * 1024u + s_bin], __ATOMIC_RELAXED);
        }
        st[3] = 0u;
        st[4] = 0u;
    }
}

extern "C" __global__ __launch_bounds__(256) void mtp_rx_compact(
        int                * __restrict__ selected,  // [m]
        unsigned long long * __restrict__ cand,      // [64]
        unsigned int       * __restrict__ st,
        const float        * __restrict__ logits,
        unsigned int         n,
        unsigned int         m) {
    __shared__ unsigned int scan[RX_BLOCK];
    __shared__ unsigned int s_last;
    __shared__ unsigned long long srt[64];
    const unsigned int tid = threadIdx.x;
    const unsigned int kt = st[0];
    const unsigned int need = st[1];
    const unsigned int count_gt = m - need;
    const unsigned int eqoff = st[RX_ST_EQOFF + blockIdx.x];
    const unsigned int chunk = (n + RX_WG - 1u) / RX_WG;
    const unsigned int lo = blockIdx.x * chunk;
    const unsigned int end = min(lo + chunk, n);
    unsigned int running = 0u;
    for (unsigned int t0 = lo; t0 < end; t0 += RX_BLOCK) {
        const unsigned int i = t0 + tid;
        unsigned int k = 0u;
        bool valid = i < end;
        if (valid) k = rx_key(logits[i]);
        const unsigned long long comp = ((unsigned long long)k << 32) | (unsigned long long)(~i);
        if (valid && k > kt) {
            const unsigned int slot = atomicAdd(&st[4], 1u);
            if (slot < 64u) cand[slot] = comp;
        }
        const unsigned int eq = (valid && k == kt) ? 1u : 0u;
        // Rank of each tie in index order: a ballot per wave (lanes below this one), then the waves before it.
        const unsigned int lane = tid % warpSize, wave = tid / warpSize, n_waves = RX_BLOCK / warpSize;
        const unsigned long long bal = __ballot(eq);
        const unsigned int below = __popcll(lane == 0u ? 0ull : (bal & ((1ull << lane) - 1ull)));
        if (lane == 0u) scan[wave] = (unsigned int)__popcll(bal);
        __syncthreads();
        unsigned int wave_off = 0u, total = 0u;
        for (unsigned int w = 0u; w < n_waves; w++) {
            if (w < wave) wave_off += scan[w];
            total += scan[w];
        }
        if (eq) {
            const unsigned int rank = eqoff + running + wave_off + below;
            if (rank < need && count_gt + rank < 64u) cand[count_gt + rank] = comp;
        }
        running += total;
        __syncthreads();
    }
    __threadfence();
    __syncthreads();
    if (tid == 0u) s_last = (atomicAdd(&st[3], 1u) == RX_WG - 1u) ? 1u : 0u;
    __syncthreads();
    if (s_last == 0u) return;
    __threadfence();
    if (tid < 64u) srt[tid] = tid < m ? __atomic_load_n(&cand[tid], __ATOMIC_RELAXED) : 0ull;
    __syncthreads();
    // Descending bitonic sort of 64 composites by the first 64 threads.
    for (unsigned int k = 2u; k <= 64u; k <<= 1u) {
        for (unsigned int j = k >> 1u; j > 0u; j >>= 1u) {
            if (tid < 64u) {
                const unsigned int o = tid ^ j;
                if (o > tid) {
                    const unsigned long long x = srt[tid], y = srt[o];
                    const bool desc = (tid & k) == 0u;
                    if (desc ? (y > x) : (x > y)) { srt[tid] = y; srt[o] = x; }
                }
            }
            __syncthreads();
        }
    }
    if (tid < m) {
        const unsigned long long c = srt[tid];
        selected[tid] = c == 0ull ? -1 : (int)(~(unsigned int)(c & 0xFFFFFFFFull));
    }
    if (tid == 0u) {
        st[3] = 0u;
        st[4] = 0u;
    }
}
