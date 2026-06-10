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
