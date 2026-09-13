// engram_gate_add — DeepSeek-V4.1 Engram gate + residual add (model.py
// `Engram.forward`, applied to the hc copies BEFORE layers 1 and 14):
//
//   per (token b, copy c), over dim:
//     rstd = rsqrt(mean(h²) + eps) · rsqrt(mean(key_c²) + eps)
//     dot  = Σ_d h[d] · qk[c][d] · key_c[d] · rstd · dim^-1/2      (qk = q_weight ⊙ k_weight)
//     gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
//     h[d] += gate · value[d]
//
// Layout: h [B][hc][dim] f32 (in/out), key [B][hc·dim] f32 (the first hc·dim
// outputs of wkv), value [B][dim] f32 (the last dim outputs; the caller passes
// the wkv output buffer twice with `value_off` = hc·dim), qk [hc][dim] f32.
// One 256-thread WG per (copy, token): three warp-reduced sums, then the add.
// Roofline: 3 reads + 1 write of dim floats per copy — ~80 KB per token, noise.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float warp_sum_f32_eg(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

extern "C" __global__ void engram_gate_add(
        float       * __restrict__ h,        // [B][hc][dim]
        const float * __restrict__ kv,       // [B][kv_stride]: key at 0, value at value_off
        const float * __restrict__ qk,       // [hc][dim]
        unsigned int   hc,
        unsigned int   dim,
        unsigned int   kv_stride,
        unsigned int   value_off,
        float          eps) {
    const unsigned int c = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    float* hrow = h + ((size_t)b * hc + c) * dim;
    const float* key = kv + (size_t)b * kv_stride + (size_t)c * dim;
    const float* val = kv + (size_t)b * kv_stride + value_off;
    const float* qkr = qk + (size_t)c * dim;

    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        const float hv = hrow[i];
        const float kvv = key[i];
        sh += hv * hv;
        sk += kvv * kvv;
        sd += hv * qkr[i] * kvv;
    }
    sh = warp_sum_f32_eg(sh);
    sk = warp_sum_f32_eg(sk);
    sd = warp_sum_f32_eg(sd);
    __shared__ float red[3][8];
    __shared__ float s_gate;
    const unsigned int warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) { red[0][warp] = sh; red[1][warp] = sk; red[2][warp] = sd; }
    __syncthreads();
    if (tid == 0) {
        float a = 0.0f, bb = 0.0f, d = 0.0f;
        const unsigned int nw = (blockDim.x + 31u) >> 5;
        for (unsigned int w = 0; w < nw; w++) { a += red[0][w]; bb += red[1][w]; d += red[2][w]; }
        const float inv_dim = 1.0f / (float)dim;
        const float rstd = rsqrtf(a * inv_dim + eps) * rsqrtf(bb * inv_dim + eps);
        const float dot = d * rstd * rsqrtf((float)dim);
        const float mag = fmaxf(fabsf(dot), 1.0e-6f);
        const float g = copysignf(sqrtf(mag), dot);
        s_gate = 1.0f / (1.0f + expf(-g));
    }
    __syncthreads();
    const float gate = s_gate;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        hrow[i] += gate * val[i];
    }
}
