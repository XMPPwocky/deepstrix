// rope_tail — RoPE applied only to the tail `n_rot` dims of each `head_dim`-
// element head, mirroring ds4.c rope_tail_ext_inplace (ds4.c:4728).
//
// Each workgroup handles one head; each thread handles one dim-pair
// (lane = threadIdx.x ∈ [0, n_rot/2), dim-pair index i = 2*lane).
// The first head_dim - n_rot dims of each head are no-op ("nope" section).
//
// YaRN ramp interpolation between extrapolation and interpolation is active
// iff `ext_factor != 0` (the compressed-attention layers of V4 Flash).
// `corr_low` / `corr_high` are precomputed host-side from beta_fast/beta_slow
// and freq_base / n_ctx_orig; `mscale_eff` folds in the YaRN attn_factor
// adjustment so the kernel only multiplies once per dim-pair.
//
// In-place: `x` is read and overwritten. `inverse` flips the sin sign
// (used by the attention-output projection pre-multiply in M5+).

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_tail(
        float        * x,
        unsigned int   n_head,
        unsigned int   head_dim,
        unsigned int   n_rot,
        unsigned int   pos,
        float          theta_scale,    // freq_base^(-2 / n_rot), host-precomputed
        float          freq_scale,
        float          ext_factor,
        float          mscale_eff,     // attn_factor with YaRN adjust baked in
        float          corr_low,
        float          corr_high,
        int            inverse) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (head >= n_head) return;
    if (2u * lane >= n_rot) return;

    const unsigned int n_nope = head_dim - n_rot;
    const unsigned int i = 2u * lane;

    // Per-lane theta_extrap = pos * theta_scale^lane. Each thread computes
    // its own value rather than serialising the multiplicative recurrence
    // that the CPU loop uses; powf is cheap on RDNA3.
    const float theta_extrap = (float)pos * powf(theta_scale, (float)lane);
    const float theta_interp = freq_scale * theta_extrap;

    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        // rope_yarn_ramp inline: ((float)(i/2) - low) / max(.001, high - low)
        const float y = ((float)lane - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        const float ramp_mix = ramp * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    }

    const float sin_sign = (inverse != 0) ? -1.0f : 1.0f;
    const float c = cosf(theta) * mscale_eff;
    const float s = sin_sign * sinf(theta) * mscale_eff;

    float *tail = x + (unsigned long long)head * (unsigned long long)head_dim + n_nope;
    const float x0 = tail[i + 0];
    const float x1 = tail[i + 1];
    tail[i + 0] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

// M50: batched variant. Grid (n_head, 1, B), each block handles one
// (head, batch) pair at pos = pos_per_b[batch]. Eliminates the B× host
// launch overhead in the per-token serial loops of prefill.
extern "C" __global__ void rope_tail_batched(
        float        * x,                  // [B, n_head, head_dim]
        const int    * __restrict__ pos_per_b,  // [B]
        unsigned int   n_head,
        unsigned int   head_dim,
        unsigned int   n_rot,
        float          theta_scale,
        float          freq_scale,
        float          ext_factor,
        float          mscale_eff,
        float          corr_low,
        float          corr_high,
        int            inverse) {
    const unsigned int head = blockIdx.x;
    const unsigned int b    = blockIdx.z;
    const unsigned int lane = threadIdx.x;
    if (head >= n_head) return;
    if (2u * lane >= n_rot) return;

    const unsigned int n_nope = head_dim - n_rot;
    const unsigned int i = 2u * lane;
    const int pos = pos_per_b[b];

    const float theta_extrap = (float)pos * powf(theta_scale, (float)lane);
    const float theta_interp = freq_scale * theta_extrap;

    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float y = ((float)lane - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        const float ramp_mix = ramp * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    }

    const float sin_sign = (inverse != 0) ? -1.0f : 1.0f;
    const float c = cosf(theta) * mscale_eff;
    const float s = sin_sign * sinf(theta) * mscale_eff;

    const unsigned long long batch_stride = (unsigned long long)n_head * (unsigned long long)head_dim;
    float *tail = x + (unsigned long long)b * batch_stride
                    + (unsigned long long)head * (unsigned long long)head_dim
                    + n_nope;
    const float x0 = tail[i + 0];
    const float x1 = tail[i + 1];
    tail[i + 0] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

// M59: device-pos variant — identical math, but `pos` is read from a device
// u32 (written once per token via hipStreamWriteValue32). Removes the only
// host-side argument that kept the decode q/kv rope launches out of the
// per-layer HIP graphs.
extern "C" __global__ void rope_tail_pdev(
        float        * x,
        const unsigned int * __restrict__ pos_dev,
        unsigned int   n_head,
        unsigned int   head_dim,
        unsigned int   n_rot,
        float          theta_scale,
        float          freq_scale,
        float          ext_factor,
        float          mscale_eff,
        float          corr_low,
        float          corr_high,
        int            inverse) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (head >= n_head) return;
    if (2u * lane >= n_rot) return;
    const unsigned int pos = *pos_dev;

    const unsigned int n_nope = head_dim - n_rot;
    const unsigned int i = 2u * lane;

    // EXACT copy of rope_tail's math (bit-identical rotation).
    const float theta_extrap = (float)pos * powf(theta_scale, (float)lane);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float y = ((float)lane - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        const float ramp_mix = ramp * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    }

    const float sin_sign = (inverse != 0) ? -1.0f : 1.0f;
    const float c = cosf(theta) * mscale_eff;
    const float s = sin_sign * sinf(theta) * mscale_eff;

    float *tail = x + (unsigned long long)head * (unsigned long long)head_dim + n_nope;
    const float x0 = tail[i + 0];
    const float x1 = tail[i + 1];
    tail[i + 0] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

// ============================================================================
// 2026-09-27 round 2 (d_fusion): decode launch-count fusions around RoPE. Each
// kernel reproduces a chain of production kernels element for element (the SAME
// expressions and reduction orders, copied from rms_norm.hip rms_w_fast,
// rope_tail_batched above, fp4_kv_quant.hip fp8_act_quant_inplace and
// q8_0_matvec.hip q8_0_quantize_f32_wave), so every output is BIT-IDENTICAL to
// the chain's (tests/decode_fusion_bitexact.rs, plain and YaRN rope params).
// Measured as graph chains at decode b = 1..8 (warm, 5 runs), see RopeTail.
// ============================================================================
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "fp8_e4m3fn_common.inc"

#define RF_FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float rf_theta(int pos, unsigned lane, float theta_scale, float freq_scale,
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

// rotate the pair held by two adjacent lanes (element e even: x0, odd: x1) -- rope_tail_batched's
// `x0 * c - x1 * s` / `x0 * s + x1 * c` with sin_sign folded into s exactly as there
__device__ __forceinline__ float rf_rot_pair(float v, unsigned e, int pos, unsigned l, float sin_sign,
                                             float theta_scale, float freq_scale, float ext_factor,
                                             float mscale_eff, float corr_low, float corr_high) {
    const float partner = __shfl_xor_sync(RF_FULL_WARP_MASK, v, 1);
    const float theta = rf_theta(pos, l, theta_scale, freq_scale, ext_factor, corr_low, corr_high);
    const float c = cosf(theta) * mscale_eff;
    const float s = sin_sign * sinf(theta) * mscale_eff;
    const float x0 = (e & 1u) ? partner : v;
    const float x1 = (e & 1u) ? v : partner;
    return (e & 1u) ? (x0 * s + x1 * c) : (x0 * c - x1 * s);
}

// q8_0_quantize_f32_wave on one 32-element block held one element per lane
__device__ __forceinline__ void rf_q8_block(float v, uint32_t lane, int8_t* __restrict__ xqb, float* __restrict__ xs) {
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(RF_FULL_WARP_MASK, amax, off));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0) *xs = d;
    int q = (int)lrintf(v * id);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    xqb[lane] = (int8_t)q;
}

// kv_rms_rope_fp8 — the decode window-KV chain rms_norm_weighted_batched_fast(n = 512)
// -> rope_tail_batched(1 head, 512, n_rot 64) -> fp8_act_quant_inplace(512) in one launch,
// grid (b) x 256 (rms_w_fast<2>'s layout: thread t holds elements t and t + 256). The rope
// tail 448..511 is register 1 of waves 6-7 (a pair = two adjacent lanes); every fp8 32-block
// is one wave's register. Writes the chain's final kv_normed row. n_rot must be 64.
extern "C" __global__ __launch_bounds__(256) void kv_rms_rope_fp8(
        float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ weight, float eps,
        const int* __restrict__ pos_per_b, float theta_scale, float freq_scale, float ext_factor,
        float mscale_eff, float corr_low, float corr_high) {
#include "kv_rms_rope_fp8_body.inc"
}

// rope_tail_batched_copy — the decode `q_normed := q` D2D copy + rope_tail_batched(q_normed)
// in one launch: the nope part is copied (float4), lanes 0..31 rotate the tail from `src` into
// `dst`. grid (n_head [+ pad], 1, b) x 128; needs (head_dim - n_rot) % 4 == 0 and 16-B rows.
extern "C" __global__ __launch_bounds__(128) void rope_tail_batched_copy(
        float* __restrict__ dst, const float* __restrict__ src, const int* __restrict__ pos_per_b,
        unsigned int n_head, unsigned int head_dim, unsigned int n_rot, float theta_scale,
        float freq_scale, float ext_factor, float mscale_eff, float corr_low, float corr_high,
        int inverse) {
#include "rope_tail_batched_copy_body.inc"
}

// rope_inv_quant_q8 — the output projection's rope_tail_batched INVERSE on heads (64 x 512,
// n_rot 64, in place) + q8_0_quantize_f32_wave of the whole [b, 32768] row, one launch.
// grid (64 [+ pad], b) x 128: 4 waves x 4 32-blocks per head; blocks 14-15 hold the rope tail.
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
        const unsigned blk = 4u * w + t;          // 0..15 within the head
        const unsigned e = 32u * blk + lane;      // 0..511 within the head
        float v = hb[e];
        if (blk >= 14u) {                         // 448..511: the rope tail (wave-uniform branch)
            v = rf_rot_pair(v, e, pos_per_b[b], (e - 448u) >> 1, -1.0f, theta_scale, freq_scale, ext_factor,
                            mscale_eff, corr_low, corr_high);
            hb[e] = v;
        }
        const size_t k = (size_t)head * 16u + blk;
        rf_q8_block(v, lane, xq + row + 32u * k, xscale + (size_t)b * 1024u + k);
    }
}

// ---- `_ind` twins (GRAPH_KEYS_DESIGN.md 2.3, 2.11) ---------------------------------------------
// Pointer operands resolve through the arena context when their `ind_mask` bit is set; the six
// per-layer rope arguments come from three context slots holding f32 pairs (theta_scale |
// freq_scale, ext_factor | mscale_eff, corr_low | corr_high: `rope::arena_ctx_rope_words`) -- the
// `rope*` arguments are those slots' addresses, always. Production twins (no canary code, 2.8).
#include "arena_ctx.inc"

__device__ __forceinline__ static void kv_rms_rope_fp8_body(
        float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ weight, float eps,
        const int* __restrict__ pos_per_b, float theta_scale, float freq_scale, float ext_factor,
        float mscale_eff, float corr_low, float corr_high) {
#include "kv_rms_rope_fp8_body.inc"
}

// operands: 0 out, 1 x, 2 weight, 3 pos_per_b
extern "C" __global__ __launch_bounds__(256) void kv_rms_rope_fp8_ind(
        uint32_t ind_mask, const ArenaCtx* canary, uint32_t canary_tag, float* out, const float* x,
        const float* weight, float eps, const int* pos_per_b, const uint64_t* rope01, const uint64_t* rope23,
        const uint64_t* rope45) {
    (void)canary;
    (void)canary_tag;
    ARENA_CTX_DEREF(ind_mask, 0, float*, out);
    ARENA_CTX_DEREF(ind_mask, 1, const float*, x);
    ARENA_CTX_DEREF(ind_mask, 2, const float*, weight);
    ARENA_CTX_DEREF(ind_mask, 3, const int*, pos_per_b);
    float theta_scale, freq_scale, ext_factor, mscale_eff, corr_low, corr_high;
    ARENA_CTX_F32X2(rope01, theta_scale, freq_scale);
    ARENA_CTX_F32X2(rope23, ext_factor, mscale_eff);
    ARENA_CTX_F32X2(rope45, corr_low, corr_high);
    kv_rms_rope_fp8_body(out, x, weight, eps, pos_per_b, theta_scale, freq_scale, ext_factor, mscale_eff,
                         corr_low, corr_high);
}

__device__ __forceinline__ static void rope_tail_batched_copy_body(
        float* __restrict__ dst, const float* __restrict__ src, const int* __restrict__ pos_per_b,
        unsigned int n_head, unsigned int head_dim, unsigned int n_rot, float theta_scale,
        float freq_scale, float ext_factor, float mscale_eff, float corr_low, float corr_high,
        int inverse) {
#include "rope_tail_batched_copy_body.inc"
}

// operands: 0 dst, 1 src, 2 pos_per_b
extern "C" __global__ __launch_bounds__(128) void rope_tail_batched_copy_ind(
        uint32_t ind_mask, const ArenaCtx* canary, uint32_t canary_tag, float* dst, const float* src,
        const int* pos_per_b, unsigned int n_head, unsigned int head_dim, unsigned int n_rot,
        const uint64_t* rope01, const uint64_t* rope23, const uint64_t* rope45, int inverse) {
    (void)canary;
    (void)canary_tag;
    ARENA_CTX_DEREF(ind_mask, 0, float*, dst);
    ARENA_CTX_DEREF(ind_mask, 1, const float*, src);
    ARENA_CTX_DEREF(ind_mask, 2, const int*, pos_per_b);
    float theta_scale, freq_scale, ext_factor, mscale_eff, corr_low, corr_high;
    ARENA_CTX_F32X2(rope01, theta_scale, freq_scale);
    ARENA_CTX_F32X2(rope23, ext_factor, mscale_eff);
    ARENA_CTX_F32X2(rope45, corr_low, corr_high);
    rope_tail_batched_copy_body(dst, src, pos_per_b, n_head, head_dim, n_rot, theta_scale, freq_scale,
                                ext_factor, mscale_eff, corr_low, corr_high, inverse);
}
