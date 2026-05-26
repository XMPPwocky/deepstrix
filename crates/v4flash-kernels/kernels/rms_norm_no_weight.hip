// rms_norm_no_weight — RMS-normalize each of n_rows rows of length n in place,
// without a learned scale. Mirrors ds4.c's `rms_norm_no_weight` and is the
// per-head no-weight variant used by `head_rms_norm_inplace`:
//
//     out[row, i] = x[row, i] / sqrt(mean(x[row, :]^2) + eps)
//
// One workgroup per row. Block of 256 threads, double-precision partial sums
// reduced via shared-memory tree (identical to rms_norm_weighted). Works for
// arbitrary n up to whatever fits in the strided loop — the only bound is
// shared memory (256 floats).
//
// V4 Flash uses this at two scales:
//   - head_rms_norm:  n_rows = DS4_N_HEAD = 64, n = DS4_N_HEAD_DIM = 512
//   - output_flat:    n_rows = 1, n = n_hc * n_embd = 16384 (M10; not exercised in M4)

#include <hip/hip_runtime.h>

extern "C" __global__ void rms_norm_no_weight(
        float *out,
        const float *x,
        unsigned int n,
        float eps) {
    __shared__ float sdata[256];

    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;

    const float *xr = x + (unsigned long long)row * (unsigned long long)n;
    float       *outr = out + (unsigned long long)row * (unsigned long long)n;

    double partial = 0.0;
    for (unsigned int i = tid; i < n; i += block) {
        float v = xr[i];
        partial += (double)v * (double)v;
    }
    sdata[tid] = (float)partial;
    __syncthreads();

    for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    const float mean_sq = sdata[0] / (float)n;
    const float scale = rsqrtf(mean_sq + eps);

    for (unsigned int i = tid; i < n; i += block) {
        outr[i] = xr[i] * scale;
    }
}

// M50 Phase 2: batched rms_norm_no_weight. Grid (B, n_rows, 1), block (256).
// Each WG processes one (batch, row) pair, of length n.
extern "C" __global__ void rms_norm_no_weight_batched(
        float *out,                     // [B, n_rows, n]
        const float *x,                 // [B, n_rows, n]
        unsigned int n_rows,
        unsigned int n,
        float eps) {
    __shared__ float sdata[256];
    const unsigned int b = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;

    const size_t off = ((size_t)b * (size_t)n_rows + (size_t)row) * (size_t)n;
    const float *xr = x + off;
    float       *outr = out + off;

    double partial = 0.0;
    for (unsigned int i = tid; i < n; i += block) {
        float v = xr[i];
        partial += (double)v * (double)v;
    }
    sdata[tid] = (float)partial;
    __syncthreads();
    for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float mean_sq = sdata[0] / (float)n;
    const float scale = rsqrtf(mean_sq + eps);
    for (unsigned int i = tid; i < n; i += block) {
        outr[i] = xr[i] * scale;
    }
}
