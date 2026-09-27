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

// rms_norm_weighted_batched_fast — rms_norm_weighted_batched with every load
// issued up front. 2026-09-26 sweep (F_mhc_glue/rms_fast): n=5120 b=1 10.46 ->
// 4.07 us, n=1280 4.91 -> 3.38, n=512 3.82 -> 3.25 (graph, dGPU), B=512 n=5120
// 31.9 -> 25.5; bit-exact (reviewer: 90 + 511 compares, incl. inf/denormal rows).
//
// The production loop `for (i = tid; i < n; i += 256) partial += (double)x*x`
// has a runtime n, so the compiler cannot hoist the loads: each iteration is one
// dependent global load (~1 us round trip at decode occupancy, one WG per row)
// then one DFMA, and the output pass RE-READS x. Here n == NPT * 256 dispatches
// to a template holding NPT x and NPT weight values per thread in registers (all
// loads issued first), the double accumulation runs in the SAME order i = tid,
// tid+256, ..., the LDS tree is the same, and the output uses the x values in
// registers (the same f32 the second pass re-read). Any other n (or a block that
// is not 256) runs the production loop. Same signature and grid as
// rms_norm_weighted_batched; grid (1) is the per-row rms_norm_weighted.
// Selected by RmsNorm (src/rms_norm.rs) under V41_RMS_FAST for n in
// {512, 1280, 5120} only.
template <unsigned NPT>
__device__ __forceinline__ void rms_w_fast(float* __restrict__ ob, const float* __restrict__ xb,
                                           const float* __restrict__ weight, unsigned int n, float eps,
                                           float* sdata) {
    const unsigned int tid = threadIdx.x;
    float xv[NPT], wv[NPT];
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) xv[j] = xb[tid + 256u * j];
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) wv[j] = weight[tid + 256u * j];
    double partial = 0.0;
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) {
        const float v = xv[j];
        partial += (double)v * (double)v;
    }
    sdata[tid] = (float)partial;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float mean_sq = sdata[0] / (float)n;
    const float scale = rsqrtf(mean_sq + eps);
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) ob[tid + 256u * j] = xv[j] * scale * wv[j];
}

__device__ __forceinline__ void rms_w_generic(float* __restrict__ ob, const float* __restrict__ xb,
                                              const float* __restrict__ weight, unsigned int n, float eps,
                                              float* sdata) {
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;
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
    for (unsigned int i = tid; i < n; i += block) ob[i] = xb[i] * scale * weight[i];
}

extern "C" __global__ __launch_bounds__(256) void rms_norm_weighted_batched_fast(
        float* out, const float* x, const float* weight, unsigned int n, float eps) {
    __shared__ float sdata[256];
    const unsigned int b = blockIdx.x;
    const float* xb = x + (size_t)b * (size_t)n;
    float* ob = out + (size_t)b * (size_t)n;
    if (blockDim.x != 256u) { rms_w_generic(ob, xb, weight, n, eps, sdata); return; }
    switch (n) {
        case 512u:  rms_w_fast<2>(ob, xb, weight, n, eps, sdata); break;    // kv / comp norms
        case 1280u: rms_w_fast<5>(ob, xb, weight, n, eps, sdata); break;    // q_a norm
        case 5120u: rms_w_fast<20>(ob, xb, weight, n, eps, sdata); break;   // attn/ffn input norm, head prep
        default:    rms_w_generic(ob, xb, weight, n, eps, sdata); break;
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

// rms_quant_q8_1280_batched — 2026-09-27 round 2 (d_fusion): the decode q_a norm
// rms_norm_weighted_batched_fast(n = 1280) followed by q8_0_quantize_f32_wave of its
// output (the qb GEMV's dp4a input) in ONE launch, grid (b) x 256. rms_w_fast<5>
// verbatim (same loads, double partials, LDS tree, output expression); element
// e = tid + 256 j sits in wave tid >> 5, register j, so the 32-block k = 8 j + wave is
// exactly one wave's register j and the quantize is the wave kernel's code
// (shfl_xor amax, d = amax / 127, lrintf + clamp). Writes qr_normed (still read by the
// indexer), xq and xscale BIT-IDENTICAL to the two-kernel chain
// (tests/decode_fusion_bitexact.rs). Selected by RmsNorm::launch_weighted_quant_q8_1280.
extern "C" __global__ __launch_bounds__(256) void rms_quant_q8_1280_batched(
        float* __restrict__ out, signed char* __restrict__ xq, float* __restrict__ xscale,
        const float* __restrict__ x, const float* __restrict__ weight, float eps) {
    __shared__ float sdata[256];
    const unsigned int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, w = tid >> 5;
    const float* xb = x + (size_t)b * 1280u;
    float xv[5], wv[5];
    #pragma unroll
    for (unsigned j = 0; j < 5; ++j) xv[j] = xb[tid + 256u * j];
    #pragma unroll
    for (unsigned j = 0; j < 5; ++j) wv[j] = weight[tid + 256u * j];
    double partial = 0.0;
    #pragma unroll
    for (unsigned j = 0; j < 5; ++j) {
        const float v = xv[j];
        partial += (double)v * (double)v;
    }
    sdata[tid] = (float)partial;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float mean_sq = sdata[0] / (float)1280u;
    const float scale = rsqrtf(mean_sq + eps);
    float* ob = out + (size_t)b * 1280u;
    #pragma unroll
    for (unsigned j = 0; j < 5; ++j) {
        const float v = xv[j] * scale * wv[j];
        ob[tid + 256u * j] = v;
        const unsigned k = 8u * j + w;
        float amax = fabsf(v);
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, amax, off));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        if (lane == 0) xscale[(size_t)b * 40u + k] = d;
        int q = (int)lrintf(v * id);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        xq[(size_t)b * 1280u + 32u * k + lane] = (signed char)q;
    }
}
