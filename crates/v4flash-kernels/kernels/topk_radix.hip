// topk_radix -- batched top-K (K <= 512) by RADIX SELECT, exactly the bitonic chain's result (indexer_topk_bitonic.hip:
// better(a, b) := a.score > b.score || (a.score == b.score && a.idx < b.idx)), in that order, with -1 past min(K, n)
// (the chain's sentinel). Fixed work: no sample, no give-up. `V41_IDX_TOPK_RADIX` (the decode indexer's top-512).
//
// Per row r (blockIdx.y), n = n_per[r], m = min(K, n); the row's scores are scores[r * stride .. + n).
// key(score) = the monotonic float key with -0.0 folded into +0.0 (the chain's float compare calls them equal);
// composite = (key << 32) | ~idx: larger composite == better, all distinct, so the best m are unique.
//   topk_rx_pass x3 (grid RX_WG x rows, block 256): key bits 31..21, 20..10, 9..0. Each WG histograms (LDS) the keys of
//     its chunk that still match the threshold prefix and adds its non-zero bins into the row's global histogram with
//     atomics; pass 2 also writes its row to the per-WG rows. The LAST WG of the row (atomic counter, self-resetting)
//     reads + zeroes the global histogram and fixes the next threshold bits; after pass 2 the prefix is the exact
//     threshold key Kt, need = how many key == Kt elements to take (lowest idx first), and each WG's offset among the
//     key == Kt elements is written.
//   topk_rx_compact (grid RX_WG x rows): key > Kt -> cand (atomic slot; fewer than m); key == Kt -> taken in index order
//     (tiles + wave ballots + the WG offset) while rank < need. The row's last WG bitonic-sorts the m composites and
//     writes the ids (then -1 up to K).
// Row buffers: hist [TRX_HIST] u32, state [TRX_STATE] u32 (both zero-initialised once; the kernels leave them clean),
// cand [512] u64, selected [K] i32.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define TRX_WG 64u
#define TRX_BLOCK 256u
#define TRX_BINS 2048u
#define TRX_ROW_BINS 1024u
#define TRX_HIST (TRX_BINS + TRX_WG * TRX_ROW_BINS)
#define TRX_EQOFF 8u
#define TRX_STATE (TRX_EQOFF + TRX_WG)
#define TRX_KMAX 512u

__device__ __forceinline__ static unsigned int trx_key(float f) {
    if (f == 0.0f) f = 0.0f;
    const unsigned int b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

extern "C" __global__ __launch_bounds__(256) void topk_rx_pass(
        unsigned int       * __restrict__ hist_all,   // [rows * TRX_HIST]
        unsigned int       * __restrict__ state_all,  // [rows * TRX_STATE]
        const float        * __restrict__ scores,     // [rows * stride]
        const unsigned int * __restrict__ n_per,      // [rows]
        unsigned int         stride,
        unsigned int         pass,
        unsigned int         k) {
    __shared__ unsigned int hist[TRX_BINS];
    __shared__ unsigned int part[TRX_BLOCK];
    __shared__ unsigned int s_last, s_bin;
    const unsigned int row = blockIdx.y;
    const unsigned int n = n_per[row];
    const unsigned int m = min(k, n);
    if (m == 0u) return;                                   // the whole row skips: counters stay 0
    unsigned int * __restrict__ H = hist_all + (size_t)row * TRX_HIST;
    unsigned int * __restrict__ st = state_all + (size_t)row * TRX_STATE;
    const float * __restrict__ x = scores + (size_t)row * stride;
    const unsigned int shift = pass == 0u ? 21u : (pass == 1u ? 10u : 0u);
    const unsigned int nbits = pass == 2u ? 10u : 11u;
    const unsigned int nb = 1u << nbits;
    const unsigned int hi = shift + nbits;
    const unsigned int tid = threadIdx.x;
    const unsigned int prefix = pass == 0u ? 0u : st[0];
    for (unsigned int b = tid; b < nb; b += TRX_BLOCK) hist[b] = 0u;
    __syncthreads();
    const unsigned int chunk = (n + TRX_WG - 1u) / TRX_WG;
    const unsigned int lo = blockIdx.x * chunk;
    const unsigned int end = min(lo + chunk, n);
    for (unsigned int i = lo + tid; i < end; i += TRX_BLOCK) {
        const unsigned int kk = trx_key(x[i]);
        if (hi == 32u || (kk >> hi) == (prefix >> hi)) atomicAdd(&hist[(kk >> shift) & (nb - 1u)], 1u);
    }
    __syncthreads();
    for (unsigned int b = tid; b < nb; b += TRX_BLOCK) {
        const unsigned int c = hist[b];
        if (c != 0u) atomicAdd(&H[b], c);
        if (pass == 2u) H[TRX_BINS + blockIdx.x * TRX_ROW_BINS + b] = c;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0u) s_last = (atomicAdd(&st[2], 1u) == TRX_WG - 1u) ? 1u : 0u;
    __syncthreads();
    if (s_last == 0u) return;
    __threadfence();
    for (unsigned int b = tid; b < nb; b += TRX_BLOCK) {
        hist[b] = __atomic_load_n(&H[b], __ATOMIC_RELAXED);
        H[b] = 0u;
    }
    __syncthreads();
    const unsigned int per = nb / TRX_BLOCK;
    unsigned int sum = 0u;
    for (unsigned int j = 0u; j < per; j++) sum += hist[nb - 1u - tid * per - j];
    part[tid] = sum;
    __syncthreads();
    if (tid == 0u) {
        const unsigned int need = pass == 0u ? m : st[1];
        unsigned int cum = 0u, t = 0u;
        while (t < TRX_BLOCK - 1u && cum + part[t] < need) { cum += part[t]; t++; }
        unsigned int bin = nb - 1u - t * per;
        for (unsigned int j = 0u; j < per; j++) {
            bin = nb - 1u - t * per - j;
            if (cum + hist[bin] >= need || j == per - 1u) break;
            cum += hist[bin];
        }
        s_bin = bin;
        st[0] = prefix | (bin << shift);
        st[1] = need - cum;
        st[2] = 0u;
    }
    __syncthreads();
    if (pass == 2u && tid == 0u) {
        unsigned int off = 0u;
        for (unsigned int w = 0u; w < TRX_WG; w++) {
            st[TRX_EQOFF + w] = off;
            off += __atomic_load_n(&H[TRX_BINS + w * TRX_ROW_BINS + s_bin], __ATOMIC_RELAXED);
        }
        st[3] = 0u;
        st[4] = 0u;
    }
}

extern "C" __global__ __launch_bounds__(256) void topk_rx_compact(
        int                * __restrict__ selected,   // [rows * k]
        unsigned long long * __restrict__ cand_all,   // [rows * TRX_KMAX]
        unsigned int       * __restrict__ state_all,
        const float        * __restrict__ scores,
        const unsigned int * __restrict__ n_per,
        unsigned int         stride,
        unsigned int         k) {
    __shared__ unsigned int wsum[TRX_BLOCK / 32u];
    __shared__ unsigned int s_last;
    __shared__ unsigned long long srt[TRX_KMAX];
    const unsigned int row = blockIdx.y;
    const unsigned int n = n_per[row];
    const unsigned int m = min(k, n);
    const unsigned int tid = threadIdx.x;
    int * __restrict__ out = selected + (size_t)row * k;
    if (m == 0u) {
        if (blockIdx.x == 0u) for (unsigned int i = tid; i < k; i += TRX_BLOCK) out[i] = -1;
        return;
    }
    unsigned int * __restrict__ st = state_all + (size_t)row * TRX_STATE;
    unsigned long long * __restrict__ cand = cand_all + (size_t)row * TRX_KMAX;
    const float * __restrict__ x = scores + (size_t)row * stride;
    const unsigned int kt = st[0];
    const unsigned int need = st[1];
    const unsigned int count_gt = m - need;
    const unsigned int eqoff = st[TRX_EQOFF + blockIdx.x];
    const unsigned int chunk = (n + TRX_WG - 1u) / TRX_WG;
    const unsigned int lo = blockIdx.x * chunk;
    const unsigned int end = min(lo + chunk, n);
    unsigned int running = 0u;
    for (unsigned int t0 = lo; t0 < end; t0 += TRX_BLOCK) {
        const unsigned int i = t0 + tid;
        const bool valid = i < end;
        const unsigned int kk = valid ? trx_key(x[i]) : 0u;
        const unsigned long long comp = ((unsigned long long)kk << 32) | (unsigned long long)(~i);
        if (valid && kk > kt) {
            const unsigned int slot = atomicAdd(&st[4], 1u);
            if (slot < TRX_KMAX) cand[slot] = comp;
        }
        const bool eq = valid && kk == kt;
        const unsigned int lane = tid % warpSize, wave = tid / warpSize, n_waves = TRX_BLOCK / warpSize;
        const unsigned long long bal = __ballot(eq);
        const unsigned int below = __popcll(lane == 0u ? 0ull : (bal & ((1ull << lane) - 1ull)));
        if (lane == 0u) wsum[wave] = (unsigned int)__popcll(bal);
        __syncthreads();
        unsigned int wave_off = 0u, total = 0u;
        for (unsigned int w = 0u; w < n_waves; w++) {
            if (w < wave) wave_off += wsum[w];
            total += wsum[w];
        }
        if (eq) {
            const unsigned int rank = eqoff + running + wave_off + below;
            if (rank < need && count_gt + rank < TRX_KMAX) cand[count_gt + rank] = comp;
        }
        running += total;
        __syncthreads();
    }
    __threadfence();
    __syncthreads();
    if (tid == 0u) s_last = (atomicAdd(&st[3], 1u) == TRX_WG - 1u) ? 1u : 0u;
    __syncthreads();
    if (s_last == 0u) return;
    __threadfence();
    // Sort the m winners (padded to the next power of two <= 512 with 0 = worse than all) descending.
    unsigned int p2 = 1u;
    while (p2 < m) p2 <<= 1u;
    for (unsigned int i = tid; i < p2; i += TRX_BLOCK) srt[i] = i < m ? __atomic_load_n(&cand[i], __ATOMIC_RELAXED) : 0ull;
    __syncthreads();
    for (unsigned int kk = 2u; kk <= p2; kk <<= 1u) {
        for (unsigned int j = kk >> 1u; j > 0u; j >>= 1u) {
            for (unsigned int i = tid; i < p2; i += TRX_BLOCK) {
                const unsigned int o = i ^ j;
                if (o > i) {
                    const unsigned long long a = srt[i], b = srt[o];
                    const bool desc = (i & kk) == 0u;
                    if (desc ? (b > a) : (a > b)) { srt[i] = b; srt[o] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (unsigned int i = tid; i < k; i += TRX_BLOCK) {
        out[i] = i < m ? (int)(~(unsigned int)(srt[i] & 0xFFFFFFFFull)) : -1;
    }
    if (tid == 0u) {
        st[3] = 0u;
        st[4] = 0u;
    }
}
