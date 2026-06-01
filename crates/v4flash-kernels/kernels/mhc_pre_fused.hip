// Fused mhc_pre kernel — replaces the 5-kernel chain
//
//   1. rms_nw(flat ← residual)               // [HC_DIM] f32
//   2. f16.matvec(mix ← hc_fn × flat)        // [HC_MIX_DIM × HC_DIM] × [HC_DIM] → [HC_MIX_DIM]
//   3. hc_sinkhorn(split ← mix, scale, base) // [HC_MIX_DIM] in-place-style
//   4. hc_weighted(post ← residual, split)   // [N_HC, N_EMBD] × [N_HC] → [N_EMBD]
//   5. rms_w(out ← post, norm)               // [N_EMBD] × [N_EMBD] → [N_EMBD]
//
// with all five steps run inside ONE WG, intermediate state kept in LDS,
// no per-step kernel-launch overhead. Same fused kernel handles both the
// mhc_pre_attn and mhc_pre_ffn chains (caller swaps weight pointers).
//
// Decode-time motivation (per [[decode-long-ctx-analysis]]):
// each chain ran ~73-86 µs/dispatch at decode B=1 despite <10 µs of real
// work, because 5 separate kernels in a captured graph each have
// non-zero start cost. Fusing them eliminates 4 of the 5 starts.
//
// Layout assumptions (V4-Flash):
//   HC_DIM == 16384, N_EMBD == 4096, HC_MIX_DIM == 24, N_HC == 4.
//   Sinkhorn output layout matches hc_sinkhorn_par:
//     split[0..N_HC]      = sigmoid_stable(mix[i]*pre_scale + base[i]) + eps
//     split[N_HC..2*N_HC] = 2 * sigmoid_stable(mix[N_HC+i]*post_scale + base[N_HC+i])
//     split[2*N_HC..]     = sinkhorn-normalized 4×4 comb (not consumed by step 4
//                          for the pre_attn / pre_ffn use, but kept for parity).
//   hc_weighted consumes split[0..N_HC] as the per-channel weights.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define MHC_HC_DIM      16384
#define MHC_N_EMBD      4096
#define MHC_HC_MIX_DIM  24
#define MHC_N_HC        4
#define MHC_BLOCK       512
#define MHC_WAVES       16
#define MHC_LANES       32

__device__ __forceinline__ float sigmoid_stable_inline(float z) {
    if (z >= 0.0f) {
        return 1.0f / (1.0f + expf(-z));
    } else {
        const float e = expf(z);
        return e / (1.0f + e);
    }
}

extern "C" __global__ void mhc_pre_fused(
        float        * __restrict__ out,           // [N_EMBD]
        const float  * __restrict__ residual,      // [HC_DIM]
        const __half * __restrict__ hc_fn_w,       // [HC_MIX_DIM, HC_DIM] f16, row-major
        const float  * __restrict__ sk_scale,      // [3] (pre_scale, post_scale, comb_scale)
        const float  * __restrict__ sk_base,       // [HC_MIX_DIM]
        const float  * __restrict__ norm_w,        // [N_EMBD]
        float          rms_eps,
        float          sinkhorn_eps,
        unsigned int   sinkhorn_iters) {
    const unsigned int tid  = threadIdx.x;
    const unsigned int wave = tid >> 5;
    const unsigned int lane = tid & 31u;

    __shared__ float s_reduce[MHC_BLOCK];          // 2 KB — RMS reductions
    __shared__ float s_mix[MHC_HC_MIX_DIM];        // 96 B
    __shared__ float s_split[MHC_HC_MIX_DIM];      // 96 B  (only [0..N_HC] used by step 4)
    __shared__ float s_post[MHC_N_EMBD];           // 16 KB — attn_cur / ffn_cur staging
    __shared__ float s_scalar[1];                  // for inv_rms broadcast

    // ===== Step 1: RMS over residual (HC_DIM=16384) =====
    // Per thread: 32 elements (HC_DIM / BLOCK = 32).
    float partial_sq = 0.0f;
    #pragma unroll
    for (unsigned int i = 0; i < 32; i++) {
        const float v = residual[tid + i * MHC_BLOCK];
        partial_sq += v * v;
    }
    // Reduce sum-of-squares across the WG.
    s_reduce[tid] = partial_sq;
    __syncthreads();
    #pragma unroll
    for (unsigned int s = MHC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        const float mean_sq = s_reduce[0] / (float)MHC_HC_DIM;
        s_scalar[0] = rsqrtf(mean_sq + rms_eps);
    }
    __syncthreads();
    const float inv_rms = s_scalar[0];

    // ===== Step 2: matvec mix[m] = Σ_k (residual[k] * inv_rms) * w[m, k] =====
    // HC_MIX_DIM=24 outputs. 16 waves cover outputs 0..15 in pass 1; pass 2
    // re-uses waves 0..7 for outputs 16..23. Each output: one wave reduces
    // its 32-lane partial sum over the full K=HC_DIM via 5-step warp shfl.
    {
        // Pass 1
        const unsigned int m1 = wave;            // 0..15
        if (m1 < MHC_HC_MIX_DIM) {
            const __half *w_row = hc_fn_w + (size_t)m1 * MHC_HC_DIM;
            float p = 0.0f;
            for (unsigned int k = lane; k < MHC_HC_DIM; k += MHC_LANES) {
                const float r = residual[k] * inv_rms;
                p += r * (float)w_row[k];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) p += __shfl_down(p, off);
            if (lane == 0) s_mix[m1] = p;
        }
        // Pass 2 (only waves 0..7 do useful work)
        const unsigned int m2 = 16u + wave;      // 16..23 for wave 0..7, else >= 24 (skipped)
        if (m2 < MHC_HC_MIX_DIM) {
            const __half *w_row = hc_fn_w + (size_t)m2 * MHC_HC_DIM;
            float p = 0.0f;
            for (unsigned int k = lane; k < MHC_HC_DIM; k += MHC_LANES) {
                const float r = residual[k] * inv_rms;
                p += r * (float)w_row[k];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) p += __shfl_down(p, off);
            if (lane == 0) s_mix[m2] = p;
        }
    }
    __syncthreads();

    // ===== Step 3: sinkhorn (only threads 0..15 in wave 0 work) =====
    // Direct port of hc_sinkhorn_par for n_hc=4. Threads 16+ idle here.
    if (wave == 0u) {
        const float pre_scale  = sk_scale[0];
        const float post_scale = sk_scale[1];
        const float comb_scale = sk_scale[2];

        // Sigmoid blocks (per hc_sinkhorn_par lines 45-52).
        if (tid < MHC_N_HC) {
            const float z = s_mix[tid] * pre_scale + sk_base[tid];
            s_split[tid] = sigmoid_stable_inline(z) + sinkhorn_eps;
        } else if (tid < 2u * MHC_N_HC) {
            const unsigned int off = tid;
            const float z = s_mix[off] * post_scale + sk_base[off];
            s_split[off] = 2.0f * sigmoid_stable_inline(z);
        }
        // The 4×4 comb is at s_mix[8..24]. Per hc_sinkhorn_par it iterates
        // row/col normalize sinkhorn_iters times. We use 16 threads (tid
        // 0..15) for the 16 comb cells. Other threads in wave wait at the
        // wave-level shuffles below.
        const bool comb_active = (tid < 16u);
        const unsigned int src = tid % MHC_N_HC;
        const unsigned int dst = tid / MHC_N_HC;
        const unsigned int idx = src + dst * MHC_N_HC;
        const unsigned int off_in = 2u * MHC_N_HC + idx;

        float c = comb_active ? (s_mix[off_in] * comb_scale + sk_base[off_in]) : 0.0f;

        // Initial per-row softmax via wave shfls on rows of 4 threads.
        // We can't __syncthreads inside a wave-local section while other
        // threads in the WG are idle (they are at __syncthreads above),
        // so use shfl_xor across 4-thread row groups.
        // Row leader (per dst): max over the row's 4 c values.
        float row_max;
        if (comb_active) {
            // Threads with same dst form a row. shfl_xor 1 / shfl_xor 2 over
            // group of 4 gives the row max.
            float m1 = c;
            float o = __shfl_xor(m1, 1); if (o > m1) m1 = o;
            o = __shfl_xor(m1, 2); if (o > m1) m1 = o;
            row_max = m1;
            float e = expf(c - row_max);
            float row_sum;
            {
                float t = e;
                t += __shfl_xor(t, 1);
                t += __shfl_xor(t, 2);
                row_sum = t;
            }
            c = e / row_sum + sinkhorn_eps;
        }
        // First half-iter: src (column) normalize.
        if (comb_active) {
            // Column group = threads with same src. With idx = src + dst*4,
            // threads (0, 4, 8, 12) share src=0; (1, 5, 9, 13) share src=1;
            // etc. shfl_xor 4 / shfl_xor 8 over 16-thread block gives column sum.
            float t = c;
            t += __shfl_xor(t, 4);
            t += __shfl_xor(t, 8);
            c = c / (t + sinkhorn_eps);
        }
        // Remaining iters
        for (unsigned int it = 1; it < sinkhorn_iters; it++) {
            if (comb_active) {
                float t = c;
                t += __shfl_xor(t, 1);
                t += __shfl_xor(t, 2);
                c = c / (t + sinkhorn_eps);
            }
            if (comb_active) {
                float t = c;
                t += __shfl_xor(t, 4);
                t += __shfl_xor(t, 8);
                c = c / (t + sinkhorn_eps);
            }
        }
        if (comb_active) {
            s_split[2u * MHC_N_HC + idx] = c;
        }
    }
    __syncthreads();

    // ===== Step 4: hc_weighted: post[d] = Σ_h split[h] * residual[h*N_EMBD + d] =====
    // Each thread handles N_EMBD/BLOCK = 8 outputs.
    // Cache split[0..N_HC] in registers (4 elements).
    const float w0 = s_split[0];
    const float w1 = s_split[1];
    const float w2 = s_split[2];
    const float w3 = s_split[3];
    #pragma unroll
    for (unsigned int i = 0; i < 8; i++) {
        const unsigned int d = tid + i * MHC_BLOCK;
        if (d < MHC_N_EMBD) {
            float acc = 0.0f;
            acc += w0 * residual[(size_t)0 * MHC_N_EMBD + d];
            acc += w1 * residual[(size_t)1 * MHC_N_EMBD + d];
            acc += w2 * residual[(size_t)2 * MHC_N_EMBD + d];
            acc += w3 * residual[(size_t)3 * MHC_N_EMBD + d];
            s_post[d] = acc;
        }
    }
    __syncthreads();

    // ===== Step 5: rms_w over post (N_EMBD=4096), multiply by norm_w, write to out =====
    float pa = 0.0f;
    #pragma unroll
    for (unsigned int i = 0; i < 8; i++) {
        const unsigned int d = tid + i * MHC_BLOCK;
        if (d < MHC_N_EMBD) {
            const float v = s_post[d];
            pa += v * v;
        }
    }
    s_reduce[tid] = pa;
    __syncthreads();
    #pragma unroll
    for (unsigned int s = MHC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        const float mean_sq = s_reduce[0] / (float)MHC_N_EMBD;
        s_scalar[0] = rsqrtf(mean_sq + rms_eps);
    }
    __syncthreads();
    const float inv_rms_b = s_scalar[0];

    #pragma unroll
    for (unsigned int i = 0; i < 8; i++) {
        const unsigned int d = tid + i * MHC_BLOCK;
        if (d < MHC_N_EMBD) {
            out[d] = s_post[d] * inv_rms_b * norm_w[d];
        }
    }
}
