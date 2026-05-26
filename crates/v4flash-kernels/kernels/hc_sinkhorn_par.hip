// hc_sinkhorn_par — parallel reimplementation of hc_sinkhorn specialized
// for n_hc=4 (the only configuration V4-Flash uses). Replaces the
// single-thread Sinkhorn loop with a 16-thread shared-memory variant.
//
// Layout matches hc_sinkhorn:
//   out[0..n_hc]               = sigmoid_stable(mix[i]*pre_scale + base[i]) + eps
//   out[n_hc..2*n_hc]          = 2 * sigmoid_stable(mix[n_hc+i]*post_scale + base[n_hc+i])
//   out[2*n_hc..2*n_hc + 16]   = Sinkhorn-normalized 4x4 comb matrix
//
// Launch: 1 block, 16 threads. Each thread owns one (src, dst) element
// of the comb matrix and cooperates via __syncthreads + shared-mem row/col
// reductions through the 20 alternating Sinkhorn iterations.

#include <hip/hip_runtime.h>

#define HC_PAR_N 4

__device__ __forceinline__ float sigmoid_stable(float z) {
    if (z >= 0.0f) {
        return 1.0f / (1.0f + expf(-z));
    } else {
        const float e = expf(z);
        return e / (1.0f + e);
    }
}

extern "C" __global__ void hc_sinkhorn_par(
        float        * __restrict__ out,
        const float  * __restrict__ mix,
        const float  * __restrict__ scale,
        const float  * __restrict__ base,
        unsigned int   n_hc,           // must equal HC_PAR_N (4)
        unsigned int   iters,
        float          eps) {
    const unsigned int tid = threadIdx.x;
    if (n_hc != HC_PAR_N) return;
    if (tid >= 16) return;

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    // Sigmoid blocks. Threads 0..3 handle pre-weights; threads 4..7
    // handle post-weights. Other threads skip.
    if (tid < HC_PAR_N) {
        const float z = mix[tid] * pre_scale + base[tid];
        out[tid] = sigmoid_stable(z) + eps;
    } else if (tid < 2 * HC_PAR_N) {
        const unsigned int off = tid;  // n_hc..2*n_hc maps to tid 4..7
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f * sigmoid_stable(z);
    }

    // Comb matrix: 4×4 = 16 elements, one per thread.
    __shared__ float c[16];
    __shared__ float row_sum[HC_PAR_N];
    __shared__ float col_sum[HC_PAR_N];

    const unsigned int src = tid % HC_PAR_N;
    const unsigned int dst = tid / HC_PAR_N;
    const unsigned int idx = src + dst * HC_PAR_N;

    {
        const unsigned int off = 2u * HC_PAR_N + idx;
        c[idx] = mix[off] * comb_scale + base[off];
    }
    __syncthreads();

    // Initial per-row softmax: subtract row max, exp, normalize by row sum.
    // Use the 4 row-leader threads (dst==tid for tid<4) to compute reductions.
    if (tid < HC_PAR_N) {
        float m = -INFINITY;
        for (unsigned int s = 0; s < HC_PAR_N; s++) {
            const float v = c[s + tid * HC_PAR_N];
            if (v > m) m = v;
        }
        // Reuse row_sum as a temp for max — fold both passes into one
        // shared array for register pressure (16 floats max either way).
        row_sum[tid] = m;
    }
    __syncthreads();

    c[idx] = expf(c[idx] - row_sum[dst]);
    __syncthreads();

    if (tid < HC_PAR_N) {
        float s = 0.0f;
        for (unsigned int x = 0; x < HC_PAR_N; x++) s += c[x + tid * HC_PAR_N];
        row_sum[tid] = s;
    }
    __syncthreads();
    c[idx] = c[idx] / row_sum[dst] + eps;
    __syncthreads();

    // First half-iter: src-normalize (column normalize).
    if (tid < HC_PAR_N) {
        float s = 0.0f;
        for (unsigned int y = 0; y < HC_PAR_N; y++) s += c[tid + y * HC_PAR_N];
        col_sum[tid] = s;
    }
    __syncthreads();
    c[idx] = c[idx] / (col_sum[src] + eps);
    __syncthreads();

    // Remaining iters: dst-normalize (rows) then src-normalize (cols).
    for (unsigned int iter = 1; iter < iters; iter++) {
        if (tid < HC_PAR_N) {
            float s = 0.0f;
            for (unsigned int x = 0; x < HC_PAR_N; x++) s += c[x + tid * HC_PAR_N];
            row_sum[tid] = s;
        }
        __syncthreads();
        c[idx] = c[idx] / (row_sum[dst] + eps);
        __syncthreads();
        if (tid < HC_PAR_N) {
            float s = 0.0f;
            for (unsigned int y = 0; y < HC_PAR_N; y++) s += c[tid + y * HC_PAR_N];
            col_sum[tid] = s;
        }
        __syncthreads();
        c[idx] = c[idx] / (col_sum[src] + eps);
        __syncthreads();
    }

    out[2u * HC_PAR_N + idx] = c[idx];
}

// M50 Phase 2 / Phase 5: batched hc_sinkhorn_par. Grid (B, 1, 1), block (16).
// Each WG handles one batch element's sinkhorn iteration independently.
// mix[B, 2*n_hc + n_hc*n_hc], out[B, 2*n_hc + n_hc*n_hc]. scale, base shared.
extern "C" __global__ void hc_sinkhorn_par_batched(
        float        * __restrict__ out,
        const float  * __restrict__ mix,
        const float  * __restrict__ scale,
        const float  * __restrict__ base,
        unsigned int   n_hc,
        unsigned int   iters,
        float          eps) {
    const unsigned int tid = threadIdx.x;
    const unsigned int b = blockIdx.x;
    if (n_hc != HC_PAR_N) return;
    if (tid >= 16) return;

    const unsigned int stride = 2u * HC_PAR_N + HC_PAR_N * HC_PAR_N;
    const float *mix_b = mix + (size_t)b * (size_t)stride;
    float       *out_b = out + (size_t)b * (size_t)stride;

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    if (tid < HC_PAR_N) {
        const float z = mix_b[tid] * pre_scale + base[tid];
        float sig;
        if (z >= 0.0f) sig = 1.0f / (1.0f + expf(-z));
        else           sig = expf(z) / (1.0f + expf(z));
        out_b[tid] = sig + eps;
    } else if (tid < 2 * HC_PAR_N) {
        const unsigned int off = tid;
        const float z = mix_b[off] * post_scale + base[off];
        float sig;
        if (z >= 0.0f) sig = 1.0f / (1.0f + expf(-z));
        else           sig = expf(z) / (1.0f + expf(z));
        out_b[off] = 2.0f * sig;
    }

    __shared__ float c[16];
    __shared__ float row_sum[HC_PAR_N];
    __shared__ float col_sum[HC_PAR_N];

    const unsigned int src = tid % HC_PAR_N;
    const unsigned int dst = tid / HC_PAR_N;
    const unsigned int idx = src + dst * HC_PAR_N;

    {
        const unsigned int off = 2u * HC_PAR_N + idx;
        c[idx] = mix_b[off] * comb_scale + base[off];
    }
    __syncthreads();

    if (tid < HC_PAR_N) {
        float m = -INFINITY;
        for (unsigned int s = 0; s < HC_PAR_N; s++) {
            const float v = c[s + tid * HC_PAR_N];
            if (v > m) m = v;
        }
        row_sum[tid] = m;
    }
    __syncthreads();
    c[idx] = expf(c[idx] - row_sum[dst]);
    __syncthreads();
    if (tid < HC_PAR_N) {
        float s = 0.0f;
        for (unsigned int x = 0; x < HC_PAR_N; x++) s += c[x + tid * HC_PAR_N];
        row_sum[tid] = s;
    }
    __syncthreads();
    c[idx] = c[idx] / row_sum[dst] + eps;
    __syncthreads();
    if (tid < HC_PAR_N) {
        float s = 0.0f;
        for (unsigned int y = 0; y < HC_PAR_N; y++) s += c[tid + y * HC_PAR_N];
        col_sum[tid] = s;
    }
    __syncthreads();
    c[idx] = c[idx] / (col_sum[src] + eps);
    __syncthreads();
    for (unsigned int iter = 1; iter < iters; iter++) {
        if (tid < HC_PAR_N) {
            float s = 0.0f;
            for (unsigned int x = 0; x < HC_PAR_N; x++) s += c[x + tid * HC_PAR_N];
            row_sum[tid] = s;
        }
        __syncthreads();
        c[idx] = c[idx] / (row_sum[dst] + eps);
        __syncthreads();
        if (tid < HC_PAR_N) {
            float s = 0.0f;
            for (unsigned int y = 0; y < HC_PAR_N; y++) s += c[tid + y * HC_PAR_N];
            col_sum[tid] = s;
        }
        __syncthreads();
        c[idx] = c[idx] / (col_sum[src] + eps);
        __syncthreads();
    }
    out_b[2u * HC_PAR_N + idx] = c[idx];
}
