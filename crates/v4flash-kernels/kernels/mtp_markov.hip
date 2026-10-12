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
