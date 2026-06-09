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

// M55: fused rms_norm_weighted + q8_0 input quantization. The decode
// q_chain runs rms_norm_weighted(qr_normed ← qr) immediately followed by
// q8_0_quantize_f32(qr_xq ← qr_normed) — two ~6 µs single-WG kernels per
// layer per token. This computes both in one pass: the normed vector is
// still written (the CSA indexer reads it), and each 32-element q8_0
// block is quantized warp-per-block right after its values are produced.
// Numerics are identical: the quantizer consumes exactly the f32 values
// the separate kernel would have re-read from memory.
//
// One workgroup, block = 256 (8 warps). n must be a multiple of 32.
extern "C" __global__ void rms_norm_weighted_quantize_q8(
        float *out,                 // [n] normed (still consumed downstream)
        signed char *xq,            // [n] int8
        float *xscale,              // [n/32] per-block scale
        const float *x,
        const float *weight,
        unsigned int n,
        float eps) {
    __shared__ float sdata[256];

    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;

    double partial = 0.0;
    for (unsigned int i = tid; i < n; i += block) {
        float v = x[i];
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

    // Warp-per-32-block: lane L owns element L of each block; amax via
    // shfl reduce, then quantize. Matches q8_0_quantize_f32 exactly
    // (lrintf + clamp, d = amax/127).
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int nblocks = n >> 5;
    for (unsigned int b = warp; b < nblocks; b += (block >> 5)) {
        const unsigned int i = b * 32u + lane;
        const float v = x[i] * scale * weight[i];
        out[i] = v;
        float amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, off));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        int q = (int)lrintf(v * id);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        xq[i] = (signed char)q;
        if (lane == 0) {
            xscale[b] = d;
        }
    }
}
