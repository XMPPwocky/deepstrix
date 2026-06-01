// Multi-WG rms_norm_no_weight: replaces the Grid(1,1,1) single-CU variant.
//
// The single-WG kernel was 150× off BW roofline (31 µs vs 0.2 µs floor for
// 128 KB DRAM traffic at HC_DIM=16384). Per ATT: 89% of kernel wall is
// dispatch + memory latency with 63/64 CUs idle.
//
// This version splits across N_WG WGs, each handling n/N_WG elements:
//   Pass 1 (rms_norm_partial_sum_sq): each WG computes its slice's
//          sum-of-squares, atomically accumulates to a single global f32.
//   Pass 2 (rms_norm_apply_scale): one WG reads the global sum, computes
//          inv_rms; THEN each WG multiplies its slice elements by inv_rms.
//
// Used at decode for the HC_DIM=16384 RMS on residual (mhc_pre_attn /
// mhc_pre_ffn). The two-launch pattern still beats single-WG because
// each launch is small AND parallel-capable.

#include <hip/hip_runtime.h>

#define RMSNW_BLOCK 256

// Writes per-WG partial to partials[wg] — no atomic, no pre-zeroing required.
extern "C" __global__ void rms_norm_partial_sum_sq(
        float *partials,        // [n_wgs] f32 — one slot per WG (no zero needed)
        const float *x,         // [n]
        unsigned int n) {
    __shared__ float sdata[RMSNW_BLOCK];
    const unsigned int tid     = threadIdx.x;
    const unsigned int wg      = blockIdx.x;
    const unsigned int n_wgs   = gridDim.x;
    const unsigned int wg_size = n / n_wgs;          // n must be divisible by n_wgs
    const unsigned int base    = wg * wg_size;

    double partial = 0.0;
    for (unsigned int i = tid; i < wg_size; i += RMSNW_BLOCK) {
        const float v = x[base + i];
        partial += (double)v * (double)v;
    }
    sdata[tid] = (float)partial;
    __syncthreads();

    for (unsigned int s = RMSNW_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        partials[wg] = sdata[0];
    }
}

// Reduce-only variant: 1 WG × 1 thread reads ALL partials, sums them,
// computes inv_rms, writes a single scalar. Pairs with
// `f16_matvec_pre_scaled` to fold the apply-pass into the next kernel —
// saves one full N=HC_DIM DRAM round-trip and one kernel launch vs the
// rms_norm_apply_scale pair.
extern "C" __global__ void rms_norm_finalize_inv(
        float       *inv_out,        // [1] f32
        const float *partials,       // [n_wgs]
        unsigned int n,
        unsigned int n_wgs,
        float eps) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0.0f;
    for (unsigned int i = 0; i < n_wgs; i++) s += partials[i];
    const float mean_sq = s / (float)n;
    inv_out[0] = rsqrtf(mean_sq + eps);
}

// Each WG reads ALL partials, sums them, computes inv_rms, applies to its slice.
// Caller MUST launch with the same n_wgs grid as the partial kernel.
extern "C" __global__ void rms_norm_apply_scale(
        float *out,                  // [n]
        const float *x,              // [n]
        const float *partials,       // [n_wgs] from rms_norm_partial_sum_sq
        unsigned int n,
        unsigned int n_wgs,
        float eps) {
    const unsigned int tid     = threadIdx.x;
    const unsigned int wg      = blockIdx.x;
    const unsigned int wg_size = n / n_wgs;
    const unsigned int base    = wg * wg_size;

    // Each thread sums the small partials array independently — cheaper
    // than coordinating a cross-WG reduce. n_wgs ≤ 64 so this is small.
    float sum_sq = 0.0f;
    for (unsigned int i = 0; i < n_wgs; i++) {
        sum_sq += partials[i];
    }
    const float mean_sq = sum_sq / (float)n;
    const float scale = rsqrtf(mean_sq + eps);

    for (unsigned int i = tid; i < wg_size; i += RMSNW_BLOCK) {
        out[base + i] = x[base + i] * scale;
    }
}
