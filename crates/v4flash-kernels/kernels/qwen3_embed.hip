// Qwen3-Embedding forward, the small kernels around the GEMMs and attention
// (docs/v41/EMBED_PHASE_DESIGN.md §5.2). The projections run on the production
// Q8_0 x f16 WMMA GEMM (`q8_0_gemm_wmma_f16x`) and attention on
// `gqa_attn_prefill_flash_wmma_fa2`; these kernels move activations between
// them. Every f32 -> f16 conversion is round-to-nearest (`__float2half_rn`).
//
// Row-major throughout; "pitch" = elements between consecutive rows.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define QE_BLOCK 256u

__device__ __forceinline__ unsigned short qe_f16(float v) {
    return __half_as_ushort(__float2half_rn(v));
}

// Block-wide sum over QE_BLOCK threads (every thread gets the result).
__device__ __forceinline__ float qe_block_sum(float v, float *sdata) {
    const unsigned int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float r = sdata[0];
    __syncthreads();
    return r;
}

// out16[r, i] = f16(x[r, i] * rsqrt(mean_i(x[r, :]^2) + eps) * w[i]).
// Grid (rows), block QE_BLOCK. x is [rows, n] (pitch n).
extern "C" __global__ void qe_rmsnorm_f16(
        unsigned short *out16,
        unsigned int out_pitch,
        const float *x,
        const float *w,
        unsigned int n,
        float eps) {
    __shared__ float sdata[QE_BLOCK];
    const unsigned int r = blockIdx.x;
    const float *xr = x + (size_t)r * n;
    unsigned short *or_ = out16 + (size_t)r * out_pitch;
    double partial = 0.0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        partial += (double)v * (double)v;
    }
    const float ss = qe_block_sum((float)partial, sdata);
    const float scale = rsqrtf(ss / (float)n + eps);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        or_[i] = qe_f16(xr[i] * scale * w[i]);
    }
}

// Per (row, head) of the fused q|k|v GEMM output `qkv` [rows, (n_head +
// 2 n_kv_head) * head_dim]:
//   q heads: RMSNorm * q_norm, NeoX RoPE at pos[r]  -> q16[r, h, :]
//   k heads: RMSNorm * k_norm, NeoX RoPE at pos[r]  -> kc[row0 + r, h, :]
//   v heads: cast                                   -> vc[row0 + r, h, :]
// NeoX: element i < d/2 pairs with i + d/2, both rotated by
// pos * inv_freq[i], inv_freq[i] = 1 / theta^(2i/d) -- the HF numerics, in f32.
// Grid (rows, n_head + 2 n_kv_head), block = head_dim (<= QE_BLOCK, even,
// a power of two for the reduction).
extern "C" __global__ void qe_qk_norm_rope(
        const float *qkv,
        unsigned short *q16,
        unsigned short *kc,
        unsigned short *vc,
        const float *q_norm,
        const float *k_norm,
        const unsigned int *pos,
        unsigned int row0,
        unsigned int n_head,
        unsigned int n_kv_head,
        unsigned int head_dim,
        float theta,
        float eps) {
    __shared__ float sdata[QE_BLOCK];
    __shared__ float xs[QE_BLOCK];
    const unsigned int r = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int i = threadIdx.x;
    const unsigned int d = head_dim;
    const unsigned int half = d / 2u;
    const unsigned int width = (n_head + 2u * n_kv_head) * d;
    const float x = qkv[(size_t)r * width + (size_t)h * d + i];
    const size_t kv_width = (size_t)n_kv_head * d;
    if (h >= n_head + n_kv_head) {
        const unsigned int vh = h - n_head - n_kv_head;
        vc[(size_t)(row0 + r) * kv_width + (size_t)vh * d + i] = qe_f16(x);
        return;
    }
    const bool is_q = h < n_head;
    const float ss = qe_block_sum(x * x, sdata);
    const float scale = rsqrtf(ss / (float)d + eps);
    const float wn = is_q ? q_norm[i] : k_norm[i];
    xs[i] = x * scale * wn;
    __syncthreads();
    const unsigned int j = i < half ? i : i - half;
    const float inv_freq = 1.0f / powf(theta, (float)(2u * j) / (float)d);
    const float a = (float)pos[r] * inv_freq;
    float s, c;
    sincosf(a, &s, &c);
    const float y = i < half ? xs[i] * c - xs[i + half] * s
                             : xs[i] * c + xs[i - half] * s;
    if (is_q) {
        q16[(size_t)r * n_head * d + (size_t)h * d + i] = qe_f16(y);
    } else {
        const unsigned int kh = h - n_head;
        kc[(size_t)(row0 + r) * kv_width + (size_t)kh * d + i] = qe_f16(y);
    }
}

// out16[r, i] = f16(x[r, i]), x [rows, n]. Grid (ceil(n / QE_BLOCK), rows).
extern "C" __global__ void qe_cast_f16(
        unsigned short *out16,
        unsigned int out_pitch,
        const float *x,
        unsigned int n) {
    const unsigned int r = blockIdx.y;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out16[(size_t)r * out_pitch + i] = qe_f16(x[(size_t)r * n + i]);
}

// out16[r, i] = f16(silu(g) * u), g = gu[r, i], u = gu[r, n_ff + i] (the
// fused gate|up GEMM output, pitch 2 n_ff). Grid (ceil(n_ff / QE_BLOCK), rows).
extern "C" __global__ void qe_swiglu_f16(
        unsigned short *out16,
        unsigned int out_pitch,
        const float *gu,
        unsigned int n_ff) {
    const unsigned int r = blockIdx.y;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ff) return;
    const float *row = gu + (size_t)r * 2u * n_ff;
    const float g = row[i];
    const float u = row[n_ff + i];
    out16[(size_t)r * out_pitch + i] = qe_f16(g / (1.0f + expf(-g)) * u);
}

// resid[i] += delta[i], i < n. Grid-stride.
extern "C" __global__ void qe_add(
        float *resid,
        const float *delta,
        unsigned int n) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        resid[i] += delta[i];
    }
}

// out[c, :] = src[idx[c], :], rows of n. Grid (ceil(n / QE_BLOCK), count).
extern "C" __global__ void qe_gather_rows(
        float *out,
        const float *src,
        const unsigned int *idx,
        unsigned int n) {
    const unsigned int c = blockIdx.y;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[(size_t)c * n + i] = src[(size_t)idx[c] * n + i];
}
