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
        float min_p_rel,
        const float *thr_dev) {       // [1] or nullptr — see note below
    __shared__ float sdata[SAMP_BLOCK];
    __shared__ float s_gmax;
    // `thr_dev` is the top-p path's device-resident cutoff (already
    // max()'d with min_p_rel by topp_bracket_step). nullptr => use the
    // host scalar, which is what the legacy (top_p >= 1) path passes —
    // the arithmetic below is then bit-identical to the pre-top_p kernel.
    const float thr = (thr_dev != nullptr) ? thr_dev[0] : min_p_rel;
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
        if (e >= thr) acc += (double)e;
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
        float min_p_rel,
        const float *thr_dev) {      // [1] or nullptr (see logits_expsum_partial)
    __shared__ float s_chunk_sum[SAMP_BLOCK];
    __shared__ float s_scan[SAMP_BLOCK];
    __shared__ float s_gmax;
    __shared__ float s_Z;
    __shared__ int   s_winner;

    const unsigned int tid = threadIdx.x;
    // Effective survivor cutoff: device-resident when the top-p search ran,
    // otherwise the host scalar (bit-identical legacy path).
    const float thr = (thr_dev != nullptr) ? thr_dev[0] : min_p_rel;

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
        if (e >= thr) chunk_sum += (double)e;
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
            if (e < thr) continue;
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
                if (e >= thr) { found = i; break; }
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

// ===========================================================================
// top-p (nucleus) support
// ===========================================================================
//
// Goal: sample from the smallest set of tokens whose cumulative probability
// (taken in descending-probability order) reaches `top_p`, renormalised over
// that set — without a 129,280-wide sort.
//
// Every such set is threshold-closed: "the k most probable tokens" is exactly
// "every token with w >= w_(k)", where w_i = exp(x_i*inv_T - gmax) in (0, 1]
// (w = 1 at the argmax). So instead of sorting we search for the cutoff
// weight directly.
//
// Let M(t) = sum{ w_i : w_i >= t }, a non-increasing step function of t, and
// let target = top_p * Z_full (Z_full = M(0), i.e. the full-vocab mass). The
// wanted cutoff is
//
//     t* = max { t : M(t) >= target }
//
// and the nucleus is { i : w_i >= t* }. (For distinct weights t* = w_(k) with
// k the smallest index whose cumulative mass reaches target, i.e. exactly the
// standard nucleus definition (note the weights w_i are already
// temperature-scaled here — see topp_bracket_step for the ordering). When several tokens share the boundary
// weight, ALL of them are kept — a threshold cannot split a tie. That is a
// superset of a sorted implementation's arbitrary k-token cut, still satisfies
// "cumulative >= top_p", and is deterministic / order-independent.)
//
// The search runs in log space s_i = x_i*inv_T - gmax = log w_i, s_i <= 0,
// bracketed to [-TOPP_LOG_RANGE, 0]. Each level splits the current bracket
// into TOPP_NBINS equal sub-intervals and evaluates M at all NBINS+1 edges in
// ONE pass over the vocabulary (one accumulator per edge, held in registers,
// no dynamic indexing), then keeps the sub-interval [edge_j*, edge_j*+1] where
//
//     j* = max { j : M(edge_j) >= target }.
//
// The bracket invariant M(lo) >= target > M(hi) is preserved, so the final
// `lo` is always a threshold whose set really does carry >= top_p of the mass
// (the guarantee is never violated, only over-inclusion is possible), and
// t* in [lo, hi). After L levels the bracket width in log space is
// TOPP_LOG_RANGE / NBINS^L, so the returned cutoff is within a factor
// exp(TOPP_LOG_RANGE / NBINS^L) of the exact one: with the values below,
// 40 / 32^4 = 3.8e-5, i.e. 1.0000382x. Only tokens whose probability is
// within 0.004% of the true cutoff can be admitted spuriously.
//
// Tokens with w < exp(-TOPP_LOG_RANGE) = 4.2e-18 are outside the initial
// bracket and can never be admitted. Their total mass is < n*4.2e-18 = 5.4e-13
// of Z_full >= 1, so this only perturbs the nucleus when top_p > 1 - 5.4e-13
// (and top_p >= 1 is a host-side no-op fast path anyway).

#define TOPP_NBINS 32
#define TOPP_NEDGE (TOPP_NBINS + 1)
// Initial bracket half-width in log space. MUST match
// `SAMPLER_TOPP_LOG_RANGE` in src/sampler.rs. Both search kernels seed the
// bracket themselves when `level == 0`, so the host never has to upload it
// (a host->device copy mid-chain would be a blocking null-stream memcpy on
// the decode hot path).
#define TOPP_LOG_RANGE 40.0f

// ---------------------------------------------------------------------------
// topp_mass_partial — one refinement level's vocabulary pass.
//
// bracket[0] = s_lo, bracket[1] = s_hi (log-space, both <= 0).
// Writes partials_mass[wg*TOPP_NEDGE + k] = sum{ w_i : s_i >= edge_k } over
// this WG's slice, for edge_k = s_lo + k*(s_hi - s_lo)/TOPP_NBINS.
// ---------------------------------------------------------------------------
extern "C" __global__ void topp_mass_partial(
        float *partials_mass,        // [n_wgs * TOPP_NEDGE]
        const float *partials_max,   // [n_wgs_in] from logits_max_partial
        const float *logits,         // [n]
        const float *bracket,        // [2] { s_lo, s_hi }; ignored when level == 0
        unsigned int n,
        unsigned int n_wgs_in,
        float inv_T,
        unsigned int level) {        // refinement level; 0 seeds the bracket
    // [edge][wave] staging for the cross-wave half of the reduce. Sized for
    // the wave32 worst case (8 waves of 256 threads); wave64 uses 4.
    __shared__ float s_wave[TOPP_NEDGE][SAMP_BLOCK / 32];
    __shared__ float s_gmax;
    __shared__ float s_lo_sh, s_hi_sh;

    const unsigned int tid     = threadIdx.x;
    const unsigned int wg      = blockIdx.x;
    const unsigned int n_wgs   = gridDim.x;
    const unsigned int wg_size = n / n_wgs;
    const unsigned int base    = wg * wg_size;

    if (tid == 0) {
        float gm = -INFINITY;
        for (unsigned int i = 0; i < n_wgs_in; i++) {
            const float v = partials_max[i];
            if (v > gm) gm = v;
        }
        s_gmax  = gm;
        // Level 0 seeds [-TOPP_LOG_RANGE, 0] in-kernel; later levels read
        // the bracket the previous topp_bracket_step wrote.
        s_lo_sh = (level == 0u) ? -TOPP_LOG_RANGE : bracket[0];
        s_hi_sh = (level == 0u) ? 0.0f            : bracket[1];
    }
    __syncthreads();

    const float gmax = s_gmax;
    const float s_lo = s_lo_sh;
    const float s_hi = s_hi_sh;
    const float width = s_hi - s_lo;
    const float step  = width / (float)TOPP_NBINS;

    // acc[k] = mass at or above edge_k. Fixed-size, statically indexed
    // (the unrolled predicate below keeps it in registers — a dynamic
    // bucket index would spill the array to scratch memory).
    float acc[TOPP_NEDGE];
#pragma unroll
    for (int k = 0; k < TOPP_NEDGE; k++) acc[k] = 0.0f;

    for (unsigned int i = tid; i < wg_size; i += SAMP_BLOCK) {
        const float s = logits[base + i] * inv_T - gmax;
        if (s < s_lo) continue;              // below the bracket: never admitted
        const float w = expf(s);
#pragma unroll
        for (int k = 0; k < TOPP_NEDGE; k++) {
            const float edge = (k == TOPP_NBINS) ? s_hi : (s_lo + step * (float)k);
            acc[k] += (s >= edge) ? w : 0.0f;
        }
    }

    // Deterministic reduce of TOPP_NEDGE accumulators. An LDS tree per edge
    // would cost TOPP_NEDGE * log2(SAMP_BLOCK) barriers; a wave-level
    // shuffle-down followed by one cross-wave pass costs a single barrier.
    // The shuffle order is fixed, so the result is still bit-reproducible.
    const int lane    = (int)tid % warpSize;
    const int wave    = (int)tid / warpSize;
    const int n_waves = SAMP_BLOCK / warpSize;
#pragma unroll
    for (int k = 0; k < TOPP_NEDGE; k++) {
        float v = acc[k];
        for (int m = warpSize / 2; m > 0; m >>= 1) {
            v += __shfl_down(v, m, warpSize);
        }
        if (lane == 0) s_wave[k][wave] = v;
    }
    __syncthreads();
    if ((int)tid < TOPP_NEDGE) {
        float t = 0.0f;
        for (int wv = 0; wv < n_waves; wv++) t += s_wave[tid][wv];
        partials_mass[wg * TOPP_NEDGE + tid] = t;
    }
}

// ---------------------------------------------------------------------------
// topp_bracket_step — single WG. Reduces partials_mass across WGs, picks the
// sub-interval, rewrites `bracket` in place and publishes the effective
// survivor cutoff
//
//     thr_out[0] = max(exp(new_s_lo), min_p_rel)
//
// which is where top-p and min-p compose. The chain is OpenAI/vLLM ordering:
// TEMPERATURE first, then top-p over the tempered distribution, then min-p.
// (llama.cpp's DEFAULT chain differs: it places the temperature sampler LAST,
// so its nucleus is taken on the un-tempered T=1 probabilities. The two agree
// exactly at temperature == 1.0 and diverge otherwise; this implementation
// deliberately follows the OpenAI-compatible ordering, since that is what the
// HTTP endpoint advertises.)
//
// min-p is itself the threshold rule `w >= min_p` (logit >= max_logit +
// log(min_p)), so the intersection of the two filters is a single threshold
// max(t*, min_p_rel), and the final Z is renormalised over exactly that set
// by logits_expsum_partial(thr_dev).
// ---------------------------------------------------------------------------
extern "C" __global__ void topp_bracket_step(
        float *bracket,               // [2] in/out { s_lo, s_hi }; in ignored at level 0
        float *thr_out,               // [1] out
        const float *partials_mass,   // [n_wgs * TOPP_NEDGE]
        const float *partials_z,      // [n_wgs] full-vocab Z partials
        unsigned int n_wgs,
        float top_p,
        float min_p_rel,
        unsigned int level) {         // must match topp_mass_partial's level
    __shared__ double s_mass[TOPP_NEDGE];
    const unsigned int tid = threadIdx.x;

    if (tid < TOPP_NEDGE) {
        double acc = 0.0;
        for (unsigned int wg = 0; wg < n_wgs; wg++) {
            acc += (double)partials_mass[wg * TOPP_NEDGE + tid];
        }
        s_mass[tid] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        double Z = 0.0;
        for (unsigned int i = 0; i < n_wgs; i++) Z += (double)partials_z[i];
        const double target = (double)top_p * Z;

        // j* = max { j : M(edge_j) >= target }. M is non-increasing in j, and
        // the bracket invariant guarantees j = 0 qualifies; the clamp below
        // is a floating-point guard, not an expected path.
        int jstar = 0;
        for (int j = TOPP_NEDGE - 1; j >= 0; j--) {
            if (s_mass[j] >= target) { jstar = j; break; }
        }

        // Same seeding rule as topp_mass_partial: at level 0 the bracket
        // buffer holds whatever the previous token left behind, so the
        // constants are used instead of reading it.
        const float s_lo = (level == 0u) ? -TOPP_LOG_RANGE : bracket[0];
        const float s_hi = (level == 0u) ?  0.0f           : bracket[1];
        const float step = (s_hi - s_lo) / (float)TOPP_NBINS;
        float new_lo, new_hi;
        if (jstar >= TOPP_NBINS) {
            // Even the top edge carries the whole target (e.g. p_max >= top_p):
            // the nucleus is the argmax (plus exact ties). Collapse and stay.
            new_lo = s_hi;
            new_hi = s_hi;
        } else {
            new_lo = s_lo + step * (float)jstar;
            new_hi = new_lo + step;
            if (new_hi > s_hi) new_hi = s_hi;
        }
        bracket[0] = new_lo;
        bracket[1] = new_hi;
        // exp(new_lo) <= 1 always (new_lo <= s_hi <= 0), so the argmax token
        // always survives and the walk can never come up empty.
        //
        // The "never over-truncate" guarantee does NOT come from this shave:
        // it comes from returning the LOW end of the bracket, whose
        // invariant M(lo) >= target holds at every level. `new_lo` is one of
        // 33 discrete bucket edges, so an f32 error big enough to change the
        // outcome would move the cutoff by a whole bucket (3.8e-5 relative),
        // which no 1e-6 shave could absorb anyway.
        //
        // What the shave DOES cover is the f32-vs-f64 boundary comparison
        // between this kernel's weights and any host-side reference: the
        // device compares expf(logit*inv_T - gmax) in f32 while a reference
        // (the oracle in tests/sampler.rs, say) recomputes the same weight in
        // f64, and the two can straddle an exact `>=` at the boundary token.
        // 1e-6 is 40x tighter than the search's own convergence band
        // (3.8e-5), so the extra over-inclusion costs nothing measurable.
        // min_p_rel is applied after the shave so min-p stays exact.
        float thr = expf(new_lo) * (1.0f - 1e-6f);
        thr_out[0] = (min_p_rel > thr) ? min_p_rel : thr;
    }
}
