// rms_norm_weighted — RMS-normalize an n-element row and multiply by per-element
// weight. Mirrors ds4.c:2709 rms_norm_weight; we use the same epsilon (1e-6).
//
//     out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
//
// One workgroup per row. Currently n is exactly DS4_N_EMBD = 4096 in V4 Flash;
// this kernel assumes n <= 4096 and uses a 256-thread block with two-stage
// shared-memory reduction.

#include <hip/hip_runtime.h>

extern "C" __global__ void rms_norm_weighted(
        float *out,
        const float *x,
        const float *weight,
        unsigned int n,
        float eps) {
    __shared__ float sdata[256];

    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;

    // First pass: sum of squares.
    double partial = 0.0;
    for (unsigned int i = tid; i < n; i += block) {
        float v = x[i];
        partial += (double)v * (double)v;
    }
    sdata[tid] = (float)partial;
    __syncthreads();

    // Tree reduction in shared memory.
    for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Compute scale once (all threads agree).
    const float mean_sq = sdata[0] / (float)n;
    const float scale = rsqrtf(mean_sq + eps);

    // Second pass: scale and weight.
    for (unsigned int i = tid; i < n; i += block) {
        out[i] = x[i] * scale * weight[i];
    }
}

// M50 Phase 2: batched rms_norm_weighted. Grid (B, 1, 1), block (256).
// Each WG processes one batch element's n-element row.
extern "C" __global__ void rms_norm_weighted_batched(
        float *out,                     // [B, n]
        const float *x,                 // [B, n]
        const float *weight,            // [n] — SHARED across batch
        unsigned int n,
        float eps) {
    __shared__ float sdata[256];
    const unsigned int b = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;

    const float *xb = x + (size_t)b * (size_t)n;
    float       *ob = out + (size_t)b * (size_t)n;

    double partial = 0.0;
    for (unsigned int i = tid; i < n; i += block) {
        float v = xb[i];
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
        ob[i] = xb[i] * scale * weight[i];
    }
}
