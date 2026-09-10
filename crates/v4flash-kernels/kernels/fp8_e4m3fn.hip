// fp8_e4m3fn_quantize — in-place block-of-64 E4M3FN round-trip over the
// non-RoPE portion of a head_dim row. Mirrors ds4's
// `dsv4_fp8_kv_quantize_row_inplace_cpu` (ds4.c:1635) and its lookup-based
// `dsv4_e4m3fn_dequant_cpu` (ds4.c:1605) bit-for-bit.
//
// For V4 Flash: only the main attention's compressor invokes this
// (head_dim=512, n_rot=64, n_nope=448 → 7 blocks of 64 elements each).
// The indexer compressor (head_dim=128) skips this op.
//
// One workgroup per block. block.x = 64 threads. Grid.x = n_nope / 64.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "fp8_e4m3fn_common.inc"

extern "C" __global__ void fp8_e4m3fn_quantize(
        float        * __restrict__ x,
        unsigned int   n_nope) {
    const unsigned int block_idx = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int off = block_idx * 64u;
    if (off >= n_nope) return;

    __shared__ float partial[64];
    __shared__ float scale_s;

    const float v = x[off + tid];
    const float av = fabsf(v);
    partial[tid] = av;
    __syncthreads();

    for (unsigned int stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = partial[tid + stride];
            if (other > partial[tid]) partial[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float amax = partial[0];
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        // scale = ldexpf(1, ceil(log2(amax/448)))
        const float l = log2f(amax / 448.0f);
        const int e = (int)ceilf(l);
        scale_s = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float scale = scale_s;

    float vs = v / scale;
    if (vs > 448.0f) vs = 448.0f;
    if (vs < -448.0f) vs = -448.0f;
    x[off + tid] = dsv4_e4m3fn_dequant(vs) * scale;
}

// M50 batched: B independent rows of `row_stride` length each, quantize
// the first `n_nope` elements of each (the nope part). Each block handles
// one (b, block_idx) chunk of 64 elements. Same compute, B-fold launch.
// Grid (n_nope/64, B), block (64).
extern "C" __global__ void fp8_e4m3fn_quantize_batched(
        float        * __restrict__ x,
        unsigned int   n_nope,
        unsigned int   row_stride) {
    const unsigned int block_idx = blockIdx.x;
    const unsigned int b         = blockIdx.y;
    const unsigned int tid       = threadIdx.x;
    const unsigned int off       = block_idx * 64u;
    if (off >= n_nope) return;
    float * row = x + (unsigned long long)b * row_stride;

    __shared__ float partial[64];
    __shared__ float scale_s;

    const float v = row[off + tid];
    const float av = fabsf(v);
    partial[tid] = av;
    __syncthreads();

    for (unsigned int stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = partial[tid + stride];
            if (other > partial[tid]) partial[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float amax = partial[0];
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float l = log2f(amax / 448.0f);
        const int e = (int)ceilf(l);
        scale_s = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float scale = scale_s;

    float vs = v / scale;
    if (vs > 448.0f) vs = 448.0f;
    if (vs < -448.0f) vs = -448.0f;
    row[off + tid] = dsv4_e4m3fn_dequant(vs) * scale;
}

// =============================================================================
// M59: fused decode KV tail. Replaces the 5-kernel chain
//   rms_norm_weighted → rope_tail(last 64) → fp8_e4m3fn_quantize(first 448)
//   → f16_roundtrip(512) → kv_cache_append
// with ONE 512-thread workgroup (each tiny kernel carried a ~3-6 µs device
// floor; the chain is elementwise over 512 values after one reduction).
// pos/slot come from device u32s (M59 scalars) so the kernel lives inside
// the per-layer qkv_chain graph.
//
// Numerics: identical formulas; the rms sum-of-squares reduces over 512
// lanes instead of 256×2 (reorder ≈ 1e-16 rel). rope math copied verbatim
// from rope_tail; fp8 per-64-group max/scale/nearest-even identical.
// Also writes the post-everything vector to kv_normed (downstream readers).
// =============================================================================
extern "C" __global__ void kv_post_fused(
        float        * __restrict__ kv_normed,     // [512] out (post all steps)
        _Float16     * __restrict__ cache,          // [capacity * 512]
        const float  * __restrict__ kv_raw,         // [512] matvec output
        const float  * __restrict__ weight,         // [512] rms weight
        const unsigned int * __restrict__ pos_dev,
        const unsigned int * __restrict__ slot_dev,
        unsigned int   head_dim,                    // 512
        unsigned int   n_rot,                       // 64
        float          rms_eps,
        float          theta_scale,
        float          freq_scale,
        float          ext_factor,
        float          mscale_eff,
        float          corr_low,
        float          corr_high) {
    const unsigned int t = threadIdx.x;
    __shared__ float s_red[512];
    __shared__ float s_v[512];

    // --- rms_norm_weighted ---
    const float v_in = (t < head_dim) ? kv_raw[t] : 0.0f;
    s_red[t] = (float)((double)v_in * (double)v_in);
    __syncthreads();
    for (unsigned int stride = 256; stride > 0; stride >>= 1) {
        if (t < stride) s_red[t] += s_red[t + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(s_red[0] / (float)head_dim + rms_eps);
    if (t < head_dim) {
        s_v[t] = v_in * scale * weight[t];
    }
    __syncthreads();

    // --- rope_tail on the last n_rot dims (verbatim rope_tail math) ---
    const unsigned int n_nope = head_dim - n_rot;
    if (2u * t < n_rot) {
        const unsigned int pos = *pos_dev;
        const unsigned int i = 2u * t;
        const float theta_extrap = (float)pos * powf(theta_scale, (float)t);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        if (ext_factor != 0.0f) {
            const float y = ((float)t - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
            const float ramp_mix = ramp * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        }
        const float c = cosf(theta) * mscale_eff;
        const float s = sinf(theta) * mscale_eff;
        const float x0 = s_v[n_nope + i];
        const float x1 = s_v[n_nope + i + 1u];
        s_v[n_nope + i]      = x0 * c - x1 * s;
        s_v[n_nope + i + 1u] = x0 * s + x1 * c;
    }
    __syncthreads();

    // --- fp8 e4m3fn on the first n_nope dims, per-64 groups (verbatim) ---
    const unsigned int g = t >> 6;          // group 0..7
    const unsigned int l = t & 63u;
    const bool in_nope = (t < n_nope);
    s_red[t] = in_nope ? fabsf(s_v[t]) : 0.0f;
    __syncthreads();
    for (unsigned int stride = 32; stride > 0; stride >>= 1) {
        if (l < stride) {
            const float other = s_red[g * 64u + l + stride];
            if (other > s_red[g * 64u + l]) s_red[g * 64u + l] = other;
        }
        __syncthreads();
    }
    if (in_nope) {
        float amax = s_red[g * 64u];
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float lg = log2f(amax / 448.0f);
        const int e = (int)ceilf(lg);
        const float qs = ldexpf(1.0f, e);
        float vs = s_v[t] / qs;
        if (vs > 448.0f) vs = 448.0f;
        if (vs < -448.0f) vs = -448.0f;
        s_v[t] = dsv4_e4m3fn_dequant(vs) * qs;
    }
    __syncthreads();

    // --- f16 roundtrip + outputs ---
    if (t < head_dim) {
        const float r = __half2float(__float2half_rn(s_v[t]));
        kv_normed[t] = r;
        const unsigned int slot = *slot_dev;
        cache[(unsigned long long)slot * head_dim + t] = (_Float16)r;
    }
}
