// On-device sampling for the decode head.
//
// Replaces the per-token `copy_to_host(logits)` + CPU argmax with a small
// chain of WG-cooperating kernels that emit a single i32 token id. The
// host only has to read back 4 bytes (the sampled id).
//
// Two paths, chosen by the host:
//
//   ARGMAX path:
//     argmax_one — single-WG parallel reduce on (val, idx) — deterministic.
//
//   MULTINOMIAL path (V4-Flash recommended: temperature = 1.0, top_p = 1.0):
//     logits_max_partial       — per-WG max over a slice of logits
//     logits_expsum_partial    — per-WG sum(exp((x*inv_T) - gmax_scaled))
//     softmax_sample_one       — single-WG chunked prefix-scan walk
//
// All sampling kernels operate on f32 logits sized N (= N_VOCAB = 129,280
// for V4-Flash). N must be divisible by N_WG (use N_WG=64 → 2020/WG, fine).
//
// Temperature is folded into the max + expsum + walk kernels via `inv_T`
// so callers don't have to pre-scale logits. For T = 1.0, inv_T = 1.0
// makes this a no-op fmul per element. min-p threshold (`min_p_rel`) is
// applied in the walk: any token whose unnormalized prob is below
// min_p_rel * exp(0) = min_p_rel is skipped during the cumulative walk.
// Default min_p_rel = 0.0 means no pruning, which matches the
// recommended sampler for this model.

#include <hip/hip_runtime.h>

#define SAMP_BLOCK 256

// ---------------------------------------------------------------------------
// argmax_one — deterministic. Single WG, tie-break by lowest index.
// Output: next_token_out[0] = arg max_i logits[i]
// ---------------------------------------------------------------------------
extern "C" __global__ void argmax_one(
        int *next_token_out,     // [1]
        const float *logits,     // [n]
        unsigned int n) {
    __shared__ float s_val[SAMP_BLOCK];
    __shared__ int   s_idx[SAMP_BLOCK];
    const unsigned int tid = threadIdx.x;

    float best_v = -INFINITY;
    int   best_i = (int)n; // sentinel; if everything is -inf this becomes n-1 after clamp
    for (unsigned int i = tid; i < n; i += SAMP_BLOCK) {
        const float v = logits[i];
        if (v > best_v || (v == best_v && (int)i < best_i)) {
            best_v = v;
            best_i = (int)i;
        }
    }
    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    __syncthreads();

    for (unsigned int stride = SAMP_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float v_other = s_val[tid + stride];
            const int   i_other = s_idx[tid + stride];
            if (v_other > s_val[tid] || (v_other == s_val[tid] && i_other < s_idx[tid])) {
                s_val[tid] = v_other;
                s_idx[tid] = i_other;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        int idx = s_idx[0];
        if (idx >= (int)n) idx = (int)n - 1;
        next_token_out[0] = idx;
    }
}

// ---------------------------------------------------------------------------
// logits_max_partial — per-WG max of (logits * inv_T) over a contiguous slice.
// Writes partials_max[wg].
// ---------------------------------------------------------------------------
extern "C" __global__ void logits_max_partial(
        float *partials_max,     // [n_wgs] f32
        const float *logits,     // [n]
        unsigned int n,
        float inv_T) {
    __shared__ float sdata[SAMP_BLOCK];
    const unsigned int tid     = threadIdx.x;
    const unsigned int wg      = blockIdx.x;
    const unsigned int n_wgs   = gridDim.x;
    const unsigned int wg_size = n / n_wgs;
    const unsigned int base    = wg * wg_size;

    float m = -INFINITY;
    for (unsigned int i = tid; i < wg_size; i += SAMP_BLOCK) {
        const float v = logits[base + i] * inv_T;
        if (v > m) m = v;
    }
    sdata[tid] = m;
    __syncthreads();

    for (unsigned int s = SAMP_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float o = sdata[tid + s];
            if (o > sdata[tid]) sdata[tid] = o;
        }
        __syncthreads();
    }
    if (tid == 0) partials_max[wg] = sdata[0];
}

// ---------------------------------------------------------------------------
// logits_expsum_partial — each WG first reduces partials_max[n_wgs] to get
// the global max `gmax`, then writes sum(exp(x*inv_T - gmax)) over its slice
// to partials_z[wg]. Tokens below min_p_rel are excluded so Z renormalises
// over the surviving set (ds4.c sample_full_vocab); min_p_rel == 0 excludes
// nothing.
// ---------------------------------------------------------------------------
extern "C" __global__ void logits_expsum_partial(
        float *partials_z,            // [n_wgs] f32
        const float *partials_max,    // [n_wgs] f32
        const float *logits,          // [n]
        unsigned int n,
        unsigned int n_wgs_in,        // n_wgs for partials_max[] (== gridDim.x)
        float inv_T,
        float min_p_rel) {
    __shared__ float sdata[SAMP_BLOCK];
    __shared__ float s_gmax;
    const unsigned int tid     = threadIdx.x;
    const unsigned int wg      = blockIdx.x;
    const unsigned int n_wgs   = gridDim.x;
    const unsigned int wg_size = n / n_wgs;
    const unsigned int base    = wg * wg_size;

    // n_wgs is small (typically 64). Thread 0 computes gmax over partials_max
    // and broadcasts via LDS.
    if (tid == 0) {
        float gm = -INFINITY;
        for (unsigned int i = 0; i < n_wgs_in; i++) {
            const float v = partials_max[i];
            if (v > gm) gm = v;
        }
        s_gmax = gm;
    }
    __syncthreads();
    const float gmax = s_gmax;

    double acc = 0.0;
    for (unsigned int i = tid; i < wg_size; i += SAMP_BLOCK) {
        const float v = logits[base + i] * inv_T;
        const float e = expf(v - gmax);
        if (e >= min_p_rel) acc += (double)e;
    }
    sdata[tid] = (float)acc;
    __syncthreads();

    for (unsigned int s = SAMP_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials_z[wg] = sdata[0];
}

// ---------------------------------------------------------------------------
// softmax_sample_one — single-WG cumulative walk.
//
// Reads partials_max[] and partials_z[] to recover gmax and Z. Computes
// target = u01[0] * Z. Then divides the vocab into SAMP_BLOCK contiguous
// chunks (one per thread); each thread computes its chunk's exp-sum and
// participates in a Hillis-Steele inclusive scan to get its prefix.
// Exactly one thread's chunk straddles `target` — that thread walks its
// chunk sequentially to find the exact index and writes the result.
//
// min_p_rel: if > 0, tokens with unnormalised prob exp(x*inv_T - gmax)
// below min_p_rel * 1.0 are excluded from the walk (since exp at gmax = 1,
// min_p_rel is the threshold relative to the most-likely token). Z from
// logits_expsum_partial excludes them with the same threshold, so the
// surviving set is renormalised (matches ds4.c sample_full_vocab).
// ---------------------------------------------------------------------------
extern "C" __global__ void softmax_sample_one(
        int *next_token_out,         // [1]
        const float *logits,         // [n]
        const float *partials_max,   // [n_wgs] f32
        const float *partials_z,     // [n_wgs] f32
        const float *u01,            // [1] uniform in [0,1)
        unsigned int n,
        unsigned int n_wgs,          // n_wgs for the partials buffers
        float inv_T,
        float min_p_rel) {
    __shared__ float s_chunk_sum[SAMP_BLOCK];
    __shared__ float s_scan[SAMP_BLOCK];
    __shared__ float s_gmax;
    __shared__ float s_Z;
    __shared__ int   s_winner;

    const unsigned int tid = threadIdx.x;

    // Thread 0 reduces partials_max[] and partials_z[]; broadcast via LDS.
    if (tid == 0) {
        float gm = -INFINITY;
        for (unsigned int i = 0; i < n_wgs; i++) {
            const float v = partials_max[i];
            if (v > gm) gm = v;
        }
        float Z = 0.0f;
        for (unsigned int i = 0; i < n_wgs; i++) Z += partials_z[i];
        s_gmax   = gm;
        s_Z      = Z;
        s_winner = -1;
    }
    __syncthreads();

    const float gmax  = s_gmax;
    const float Z     = s_Z;
    const float u     = u01[0];
    const float target = u * Z;

    // Each thread owns a contiguous chunk. Use ceil-div so the last chunk
    // may be short; clamp `end` to n.
    const unsigned int chunk = (n + SAMP_BLOCK - 1u) / SAMP_BLOCK;
    const unsigned int start = tid * chunk;
    const unsigned int end   = min(start + chunk, n);

    double chunk_sum = 0.0;
    for (unsigned int i = start; i < end; i++) {
        const float v = logits[i] * inv_T;
        const float e = expf(v - gmax);
        if (e >= min_p_rel) chunk_sum += (double)e;
    }
    s_chunk_sum[tid] = (float)chunk_sum;
    __syncthreads();

    // Hillis-Steele inclusive scan over s_chunk_sum → s_scan.
    s_scan[tid] = s_chunk_sum[tid];
    __syncthreads();
    for (unsigned int stride = 1; stride < SAMP_BLOCK; stride <<= 1) {
        float v = (tid >= stride) ? s_scan[tid - stride] : 0.0f;
        __syncthreads();
        s_scan[tid] += v;
        __syncthreads();
    }
    const float my_excl = s_scan[tid] - s_chunk_sum[tid];

    // Whichever thread's chunk straddles target wins.
    if (target >= my_excl && target < my_excl + s_chunk_sum[tid] && start < n) {
        double cum = (double)my_excl;
        int found = -1;
        for (unsigned int i = start; i < end; i++) {
            const float v = logits[i] * inv_T;
            const float e = expf(v - gmax);
            if (e < min_p_rel) continue;
            const double next_cum = cum + (double)e;
            if ((double)target < next_cum) { found = (int)i; break; }
            cum = next_cum;
        }
        if (found < 0) {
            // Floating-point edge: target == Z exactly, or chunk had only
            // sub-threshold entries. Fall back to last non-empty index in
            // this chunk; if even that fails, take last vocab slot.
            for (int i = (int)end - 1; i >= (int)start; i--) {
                const float v = logits[i] * inv_T;
                const float e = expf(v - gmax);
                if (e >= min_p_rel) { found = i; break; }
            }
            if (found < 0) found = (int)n - 1;
        }
        s_winner = found;
    }
    __syncthreads();

    if (tid == 0) {
        int idx = s_winner;
        if (idx < 0) {
            // No chunk claimed target. Z is now the surviving-set mass, so
            // this is purely a floating-point edge (rounding gap between
            // the WG-level expsum and the per-chunk re-sums).
            // Fall back to argmax-of-gmax: any index with logit==gmax.
            // Cheap scan in thread 0.
            for (unsigned int i = 0; i < n; i++) {
                if (logits[i] * inv_T >= gmax) { idx = (int)i; break; }
            }
            if (idx < 0) idx = (int)n - 1;
        }
        next_token_out[0] = idx;
    }
}
