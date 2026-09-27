// Round 2 item (d): decode launch-count fusions. Every kernel here reproduces a chain of
// production kernels element for element (same expressions, same reduction orders), so the
// outputs are bit-identical to the chain's.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "fp8_e4m3fn_common.inc"

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

// ---- shared pieces (verbatim from rms_norm.hip / rope_tail.hip / fp4_kv_quant.hip / q8_0_matvec.hip)
__device__ __forceinline__ float fz_rope_theta(int pos, unsigned lane, float theta_scale, float freq_scale,
                                               float ext_factor, float corr_low, float corr_high) {
    const float theta_extrap = (float)pos * powf(theta_scale, (float)lane);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float y = ((float)lane - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        const float ramp_mix = ramp * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    }
    return theta;
}

__device__ __forceinline__ float fz_fp8_ue8m0_scale(float amax_in) {
    const float amax = fmaxf(amax_in, 1.0e-4f);
    const float t = amax / 448.0f;
    int e;
    const float m = frexpf(t, &e);
    const int cexp = (m == 0.5f) ? e - 1 : e;
    return ldexpf(1.0f, cexp);
}

__device__ __forceinline__ float fz_wave_amax(float v) {
    float a = fabsf(v);
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 1));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 2));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 4));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 8));
    a = fmaxf(a, __shfl_xor_sync(FULL_WARP_MASK, a, 16));
    return a;
}

// q8_0_quantize_f32_wave on one 32-block held one element per lane
__device__ __forceinline__ void fz_q8_block(float v, uint32_t lane, int8_t* __restrict__ xqb, float* __restrict__ xs) {
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULL_WARP_MASK, amax, off));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0) *xs = d;
    int q = (int)lrintf(v * id);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    xqb[lane] = (int8_t)q;
}

// rms_w_fast<NPT> front half: returns the normalised values of elements tid + 256 j in v[]
template <unsigned NPT>
__device__ __forceinline__ void fz_rms(float (&v)[NPT], const float* __restrict__ xb, const float* __restrict__ weight,
                                       unsigned n, float eps, float* sdata) {
    const unsigned int tid = threadIdx.x;
    float xv[NPT], wv[NPT];
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) xv[j] = xb[tid + 256u * j];
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) wv[j] = weight[tid + 256u * j];
    double partial = 0.0;
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) {
        const float x = xv[j];
        partial += (double)x * (double)x;
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
    for (unsigned j = 0; j < NPT; ++j) v[j] = xv[j] * scale * wv[j];
}

// ---- (d1) kv chain: rms_norm_weighted_batched_fast(n=512) -> rope_tail_batched(1 head, 512, n_rot 64)
//      -> fp8_act_quant_inplace(width 512). grid (b) x 256; writes the final kv_normed row.
extern "C" __global__ __launch_bounds__(256) void kv_rms_rope_fp8(
        float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ weight, float eps,
        const int* __restrict__ pos_per_b, float theta_scale, float freq_scale, float ext_factor,
        float mscale_eff, float corr_low, float corr_high) {
    __shared__ float sdata[256];
    const unsigned b = blockIdx.x, tid = threadIdx.x, lane = tid & 31u;
    float v[2];
    fz_rms<2>(v, x + (size_t)b * 512u, weight, 512u, eps, sdata);
    // rope on elements 448..511 = register 1 of threads 192..255 (pairs = adjacent lanes)
    if (tid >= 192u) {
        const unsigned e = tid + 256u;              // 448..511
        const unsigned l = (e - 448u) >> 1;         // rope lane 0..31
        const float partner = __shfl_xor_sync(FULL_WARP_MASK, v[1], 1);  // waves 6-7 fully active
        const float theta = fz_rope_theta(pos_per_b[b], l, theta_scale, freq_scale, ext_factor, corr_low, corr_high);
        const float c = cosf(theta) * mscale_eff;
        const float s = 1.0f * sinf(theta) * mscale_eff;
        const float x0 = (e & 1u) ? partner : v[1];
        const float x1 = (e & 1u) ? v[1] : partner;
        v[1] = (e & 1u) ? (x0 * s + x1 * c) : (x0 * c - x1 * s);
    }
    float* ob = out + (size_t)b * 512u;
    #pragma unroll
    for (unsigned j = 0; j < 2; ++j) {
        const float a = fz_wave_amax(v[j]);
        const float scale = fz_fp8_ue8m0_scale(a);
        float q = v[j] / scale;
        q = fminf(fmaxf(q, -448.0f), 448.0f);
        ob[tid + 256u * j] = dsv4_e4m3fn_dequant(q) * scale;
    }
    (void)lane;
}

// ---- (d2) q_a norm + its Q8_0 quantize: rms_norm_weighted_batched_fast(n=1280) ->
//      q8_0_quantize_f32_wave(1280/32 blocks). grid (b) x 256. Element e = tid + 256 j sits in wave
//      (tid >> 5) register j, and a 32-block k = 8 j + wave is exactly one wave's register j.
extern "C" __global__ __launch_bounds__(256) void rms_quant_q8_1280(
        float* __restrict__ out, int8_t* __restrict__ xq, float* __restrict__ xscale,
        const float* __restrict__ x, const float* __restrict__ weight, float eps) {
    __shared__ float sdata[256];
    const unsigned b = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, w = tid >> 5;
    float v[5];
    fz_rms<5>(v, x + (size_t)b * 1280u, weight, 1280u, eps, sdata);
    float* ob = out + (size_t)b * 1280u;
    #pragma unroll
    for (unsigned j = 0; j < 5; ++j) {
        ob[tid + 256u * j] = v[j];
        const unsigned k = 8u * j + w;  // block index within the row
        fz_q8_block(v[j], lane, xq + (size_t)b * 1280u + 32u * k, xscale + (size_t)b * 40u + k);
    }
}

// ---- (d3) q -> q_normed copy + rope_tail_batched(64 heads, 512, n_rot 64) in one pass.
//      grid (n_head, 1, b) x 128: 112 threads copy the 448 nope floats as float4, lanes 0..31 rope.
extern "C" __global__ __launch_bounds__(128) void rope_tail_batched_copy(
        float* __restrict__ dst, const float* __restrict__ src, const int* __restrict__ pos_per_b,
        unsigned n_head, unsigned head_dim, unsigned n_rot, float theta_scale, float freq_scale,
        float ext_factor, float mscale_eff, float corr_low, float corr_high, int inverse) {
    const unsigned head = blockIdx.x, b = blockIdx.z, tid = threadIdx.x;
    if (head >= n_head) return;
    const unsigned n_nope = head_dim - n_rot;
    const size_t base = (size_t)b * n_head * head_dim + (size_t)head * head_dim;
    const float4* s4 = reinterpret_cast<const float4*>(src + base);
    float4* d4 = reinterpret_cast<float4*>(dst + base);
    for (unsigned i = tid; i < n_nope / 4u; i += blockDim.x) d4[i] = s4[i];
    if (tid < 32u && 2u * tid < n_rot) {
        const unsigned lane = tid, i = 2u * lane;
        const float theta = fz_rope_theta(pos_per_b[b], lane, theta_scale, freq_scale, ext_factor, corr_low, corr_high);
        const float sin_sign = (inverse != 0) ? -1.0f : 1.0f;
        const float c = cosf(theta) * mscale_eff;
        const float s = sin_sign * sinf(theta) * mscale_eff;
        const float x0 = src[base + n_nope + i + 0];
        const float x1 = src[base + n_nope + i + 1];
        dst[base + n_nope + i + 0] = x0 * c - x1 * s;
        dst[base + n_nope + i + 1] = x0 * s + x1 * c;
    }
}

// ---- (d4) output projection prep: rope_tail_batched INVERSE on heads (64 x 512, n_rot 64) then
//      q8_0_quantize_f32_wave of the whole [b, 32768] row. grid (64, b) x 128 (4 waves x 4 blocks).
extern "C" __global__ __launch_bounds__(128) void rope_inv_quant_q8(
        float* __restrict__ heads, int8_t* __restrict__ xq, float* __restrict__ xscale,
        const int* __restrict__ pos_per_b, float theta_scale, float freq_scale, float ext_factor,
        float mscale_eff, float corr_low, float corr_high) {
    const unsigned head = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, lane = tid & 31u, w = tid >> 5;
    if (head >= 64u) return;  // idle grid-pad column
    const size_t row = (size_t)b * 32768u;
    float* hb = heads + row + (size_t)head * 512u;
    #pragma unroll
    for (unsigned t = 0; t < 4; ++t) {
        const unsigned blk = 4u * w + t;             // 0..15 within the head
        const unsigned e = 32u * blk + lane;         // 0..511 within the head
        float v = hb[e];
        if (blk >= 14u) {                            // elements 448..511: the rope tail
            const unsigned l = (e - 448u) >> 1;
            const float partner = __shfl_xor_sync(FULL_WARP_MASK, v, 1);
            const float theta = fz_rope_theta(pos_per_b[b], l, theta_scale, freq_scale, ext_factor, corr_low, corr_high);
            const float c = cosf(theta) * mscale_eff;
            const float s = -1.0f * sinf(theta) * mscale_eff;
            const float x0 = (e & 1u) ? partner : v;
            const float x1 = (e & 1u) ? v : partner;
            v = (e & 1u) ? (x0 * s + x1 * c) : (x0 * c - x1 * s);
            hb[e] = v;
        }
        const size_t k = (size_t)head * 16u + blk;  // block index within the row
        fz_q8_block(v, lane, xq + row + 32u * k, xscale + (size_t)b * 1024u + k);
    }
}
