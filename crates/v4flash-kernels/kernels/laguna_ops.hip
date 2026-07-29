// Laguna-S-2.1 device ops — the small elementwise/router kernels that move the
// spike's HOST ops onto the GPU. ADDITIVE: none of these touch a ds4 shared
// kernel; they are Laguna-specific variants.
//
// Kernels:
//   1. laguna_router        — folds the routed-MoE router: F32 matvec
//        (ffn_gate_inp · x) -> sigmoid -> +exp_probs_b bias -> top-k select ->
//        sum-normalized weights ×scale. Sigmoid gating (NOT the sqrt-softplus
//        of ds4's router_topk_par) and top-k up to 16 (Laguna uses 10).
//   2. laguna_softplus_gate — per-head softplus attention gate applied to the
//        attention output before o_proj: out[h,:] *= softplus(gate_logits[h]).
//   3. laguna_cast_f32_f16  — plumbing: contiguous f32 -> f16 (for Q and for the
//        contiguous KV-cache append at a given position).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define LAG_MAX_EXPERTS 256
#define LAG_MAX_USED    16
#define LAG_MAX_HIDDEN  3072

// --------------------------------------------------------------------------
// 1. Sigmoid router with folded F32 matvec + score-correction bias + top-k.
//    Launch: grid (1,1,1), block (LAG_MAX_EXPERTS = 256, 1, 1).
//    Numerics match laguna_spike.rs's host router for distinct scores (ties
//    broken by lowest index, matching router_topk_par).
// --------------------------------------------------------------------------
extern "C" __global__ void laguna_router(
        int          * __restrict__ selected,   // [n_used]
        float        * __restrict__ weights,    // [n_used]
        const float  * __restrict__ router_w,   // [n_expert, hidden] row-major
        const float  * __restrict__ x,          // [hidden]
        const float  * __restrict__ bias,       // [n_expert] or nullptr
        unsigned int   n_expert,
        unsigned int   hidden,
        unsigned int   n_used,
        float          weight_scale,
        float          weight_eps) {
    __shared__ float s_x[LAG_MAX_HIDDEN];
    __shared__ float s_probs[LAG_MAX_EXPERTS];
    __shared__ float s_scores[LAG_MAX_EXPERTS]; // winners masked to -INF
    __shared__ float buf[LAG_MAX_EXPERTS];
    __shared__ int   idxs[LAG_MAX_EXPERTS];
    __shared__ int   s_sel[LAG_MAX_USED];
    __shared__ float s_w[LAG_MAX_USED];
    __shared__ float s_sum;

    const unsigned int tid = threadIdx.x;

    // Stage x into LDS (reused by every expert's dot product).
    for (unsigned int i = tid; i < hidden; i += blockDim.x) s_x[i] = x[i];
    __syncthreads();

    // One thread per expert: dot(router_row, x) -> sigmoid -> +bias.
    if (tid < n_expert) {
        const float *wrow = router_w + (size_t)tid * (size_t)hidden;
        float acc = 0.0f;
        for (unsigned int k = 0; k < hidden; ++k) acc += wrow[k] * s_x[k];
        const float p = 1.0f / (1.0f + expf(-acc));
        s_probs[tid] = p;
        s_scores[tid] = (bias != nullptr) ? (p + bias[tid]) : p;
    }
    __syncthreads();

    // K passes of parallel argmax over the (biased) selection scores.
    for (unsigned int kk = 0; kk < n_used; ++kk) {
        if (tid < n_expert) { buf[tid] = s_scores[tid]; idxs[tid] = (int)tid; }
        else                { buf[tid] = -INFINITY;     idxs[tid] = -1; }
        __syncthreads();
        for (unsigned int stride = LAG_MAX_EXPERTS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float bo = buf[tid + stride];
                const int   io = idxs[tid + stride];
                const float bs = buf[tid];
                const int   is = idxs[tid];
                if (bo > bs || (bo == bs && io >= 0 && (is < 0 || io < is))) {
                    buf[tid] = bo; idxs[tid] = io;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int w = idxs[0];
            s_sel[kk] = w;
            if (w >= 0) s_scores[w] = -INFINITY;
        }
        __syncthreads();
    }

    // Weights = selected sigmoid probs, sum-normalized, × scale.
    if (tid == 0) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < n_used; ++j) {
            const int w = s_sel[j];
            const float p = (w >= 0) ? s_probs[w] : 0.0f;
            s_w[j] = p;
            sum += p;
        }
        if (sum < weight_eps) sum = weight_eps;
        s_sum = sum;
    }
    __syncthreads();

    if (tid < n_used) {
        selected[tid] = s_sel[tid];
        weights[tid]  = s_w[tid] / s_sum * weight_scale;
    }
}

// --------------------------------------------------------------------------
// 1b. Multi-WG router, split into two launches so the 256×hidden matvec runs
//     across MANY workgroups instead of one. `laguna_router` (above) does the
//     whole thing in a single WG (grid (1,1,1)) — the 256-expert dot products
//     serialize onto one CU (~204 µs at decode). These two kernels parallelize
//     the matvec over the grid, then do the (cheap) top-k in one WG.
//
//   laguna_router_scores: grid (n_expert,1,1), block (256,1,1). One WG per
//     expert cooperatively reduces dot(router_row_e, x), then thread 0 writes
//     probs[e] = sigmoid(dot) and scores[e] = probs[e] + bias[e].
// --------------------------------------------------------------------------
extern "C" __global__ void laguna_router_scores(
        float        * __restrict__ probs,     // [n_expert] out: sigmoid(dot)
        float        * __restrict__ scores,    // [n_expert] out: prob + bias
        const float  * __restrict__ router_w,  // [n_expert, hidden] row-major
        const float  * __restrict__ x,         // [hidden]
        const float  * __restrict__ bias,      // [n_expert] or nullptr
        unsigned int   n_expert,
        unsigned int   hidden) {
    __shared__ float sdata[256];
    const unsigned int e     = blockIdx.x;
    const unsigned int tid   = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (e >= n_expert) return;

    const float *wrow = router_w + (size_t)e * (size_t)hidden;
    float partial = 0.0f;
    for (unsigned int k = tid; k < hidden; k += block) {
        partial += wrow[k] * x[k];
    }
    sdata[tid] = partial;
    __syncthreads();
    for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        const float p = 1.0f / (1.0f + expf(-sdata[0]));
        probs[e]  = p;
        scores[e] = (bias != nullptr) ? (p + bias[e]) : p;
    }
}

// --------------------------------------------------------------------------
//   laguna_router_topk: grid (1,1,1), block (256,1,1). Consumes the precomputed
//     probs/scores, runs the K-pass parallel argmax + sum-normalized ×scale
//     weights. Identical selection/weight math to `laguna_router` (winners
//     masked to -INF, lowest-index tie-break, sum floored to weight_eps).
// --------------------------------------------------------------------------
extern "C" __global__ void laguna_router_topk(
        int          * __restrict__ selected,   // [n_used]
        float        * __restrict__ weights,    // [n_used]
        const float  * __restrict__ probs,      // [n_expert]
        const float  * __restrict__ scores,     // [n_expert] (prob + bias)
        unsigned int   n_expert,
        unsigned int   n_used,
        float          weight_scale,
        float          weight_eps) {
    __shared__ float s_probs[LAG_MAX_EXPERTS];
    __shared__ float s_scores[LAG_MAX_EXPERTS];
    __shared__ float buf[LAG_MAX_EXPERTS];
    __shared__ int   idxs[LAG_MAX_EXPERTS];
    __shared__ int   s_sel[LAG_MAX_USED];
    __shared__ float s_w[LAG_MAX_USED];
    __shared__ float s_sum;

    const unsigned int tid = threadIdx.x;
    if (tid < n_expert) {
        s_probs[tid]  = probs[tid];
        s_scores[tid] = scores[tid];
    }
    __syncthreads();

    for (unsigned int kk = 0; kk < n_used; ++kk) {
        if (tid < n_expert) { buf[tid] = s_scores[tid]; idxs[tid] = (int)tid; }
        else                { buf[tid] = -INFINITY;     idxs[tid] = -1; }
        __syncthreads();
        for (unsigned int stride = LAG_MAX_EXPERTS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float bo = buf[tid + stride];
                const int   io = idxs[tid + stride];
                const float bs = buf[tid];
                const int   is = idxs[tid];
                if (bo > bs || (bo == bs && io >= 0 && (is < 0 || io < is))) {
                    buf[tid] = bo; idxs[tid] = io;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int w = idxs[0];
            s_sel[kk] = w;
            if (w >= 0) s_scores[w] = -INFINITY;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < n_used; ++j) {
            const int w = s_sel[j];
            const float p = (w >= 0) ? s_probs[w] : 0.0f;
            s_w[j] = p;
            sum += p;
        }
        if (sum < weight_eps) sum = weight_eps;
        s_sum = sum;
    }
    __syncthreads();

    if (tid < n_used) {
        selected[tid] = s_sel[tid];
        weights[tid]  = s_w[tid] / s_sum * weight_scale;
    }
}

// --------------------------------------------------------------------------
//   BATCHED router (prefill): grid.z = B. Semantically B independent
//   laguna_router_scores / laguna_router_topk calls in one launch each.
//   Layouts are tightly packed: x[B,hidden], probs/scores[B,n_expert],
//   selected/weights[B,n_used]. Router weight + bias are shared across B.
// --------------------------------------------------------------------------
extern "C" __global__ void laguna_router_scores_batched(
        float        * __restrict__ probs,     // [B, n_expert]
        float        * __restrict__ scores,    // [B, n_expert]
        const float  * __restrict__ router_w,  // [n_expert, hidden]
        const float  * __restrict__ x,         // [B, hidden]
        const float  * __restrict__ bias,      // [n_expert] or nullptr
        unsigned int   n_expert,
        unsigned int   hidden) {
    __shared__ float sdata[256];
    const unsigned int e     = blockIdx.x;
    const unsigned int b     = blockIdx.z;
    const unsigned int tid   = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (e >= n_expert) return;

    const float *wrow = router_w + (size_t)e * (size_t)hidden;
    const float *xb   = x + (size_t)b * (size_t)hidden;
    float partial = 0.0f;
    for (unsigned int k = tid; k < hidden; k += block) {
        partial += wrow[k] * xb[k];
    }
    sdata[tid] = partial;
    __syncthreads();
    for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        const float p = 1.0f / (1.0f + expf(-sdata[0]));
        probs[(size_t)b * n_expert + e]  = p;
        scores[(size_t)b * n_expert + e] = (bias != nullptr) ? (p + bias[e]) : p;
    }
}

// --------------------------------------------------------------------------
//   WEIGHT-READ-ONCE batched router scores. Same math as
//   laguna_router_scores_batched, but the router weight row for an expert is
//   read from DRAM ONCE (into registers) and reused across all B token columns,
//   instead of being re-read for every token (grid.z=B re-streams the 3 MB
//   router weight B times → ~160 ms/prefill at B=512, ~12% of the iGPU wall).
//
//   grid (n_expert,1,1), block (256,1,1): one WG per expert. Each thread owns
//   its strided weight elements w[e][tid + j*256] (j in 0..hidden/256) in
//   registers, then loops b=0..B computing partial = sum_j w_j*x[b][tid+j*256]
//   with the SAME per-thread stride + tree reduction as the non-WRO kernel, so
//   the result is BIT-IDENTICAL. Only the DRAM traffic changes: router weight
//   read 1× (3 MB) instead of B× (1.5 GB @ B=512); x served from L2.
//   ROUTER_WRO_MAXJ caps hidden at 256*16 = 4096 (Laguna HIDDEN=3072 → 12).
// --------------------------------------------------------------------------
#define ROUTER_WRO_MAXJ 16
extern "C" __global__ void laguna_router_scores_batched_wro(
        float        * __restrict__ probs,     // [B, n_expert]
        float        * __restrict__ scores,    // [B, n_expert]
        const float  * __restrict__ router_w,  // [n_expert, hidden]
        const float  * __restrict__ x,         // [B, hidden]
        const float  * __restrict__ bias,      // [n_expert] or nullptr
        unsigned int   n_expert,
        unsigned int   hidden,
        unsigned int   B) {
    __shared__ float sdata[256];
    const unsigned int e     = blockIdx.x;
    const unsigned int tid   = threadIdx.x;
    const unsigned int block = blockDim.x;   // 256
    if (e >= n_expert) return;

    // Read this thread's strided weight elements ONCE into registers.
    const float *wrow = router_w + (size_t)e * (size_t)hidden;
    float wv[ROUTER_WRO_MAXJ];
    #pragma unroll
    for (unsigned int j = 0; j < ROUTER_WRO_MAXJ; j++) {
        const unsigned int k = tid + j * block;
        wv[j] = (k < hidden) ? wrow[k] : 0.0f;
    }

    for (unsigned int b = 0; b < B; b++) {
        const float *xb = x + (size_t)b * (size_t)hidden;
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < ROUTER_WRO_MAXJ; j++) {
            const unsigned int k = tid + j * block;
            if (k < hidden) partial += wv[j] * xb[k];
        }
        sdata[tid] = partial;
        __syncthreads();
        for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sdata[tid] += sdata[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float p = 1.0f / (1.0f + expf(-sdata[0]));
            probs[(size_t)b * n_expert + e]  = p;
            scores[(size_t)b * n_expert + e] = (bias != nullptr) ? (p + bias[e]) : p;
        }
        __syncthreads();  // guard sdata reuse for next b
    }
}

// --------------------------------------------------------------------------
//   WARP-SHUFFLE batched router scores (LAGUNA_ROUTER_WRO=warp). The router
//   matmul is neither compute- nor BW-bound (measured 0.47 TFLOP/s, 1.76 GB/s
//   — 100x below roofline); it is reduction/barrier-latency bound. This variant
//   keeps the full grid.z=B parallelism but replaces the 8-step LDS tree
//   reduction (+__syncthreads each step) with a warp-shuffle reduction, and
//   assigns one WARP (not a 256-thread block) per (expert, token) dot so 8
//   tokens are computed per WG. No LDS, no barriers on the hot path.
//   grid (n_expert, 1, ceil(B/8)), block (256,1,1) = 8 warps/WG.
//   NOT bit-identical (32-lane shuffle order != 256-thread tree order), but
//   greedy-stable + within the 5e-2 logit gate.
// --------------------------------------------------------------------------
extern "C" __global__ void laguna_router_scores_batched_warp(
        float        * __restrict__ probs,     // [B, n_expert]
        float        * __restrict__ scores,    // [B, n_expert]
        const float  * __restrict__ router_w,  // [n_expert, hidden]
        const float  * __restrict__ x,         // [B, hidden]
        const float  * __restrict__ bias,      // [n_expert] or nullptr
        unsigned int   n_expert,
        unsigned int   hidden,
        unsigned int   B) {
    const unsigned int e    = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;   // 0..7
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int b    = blockIdx.z * 8u + warp;
    if (e >= n_expert || b >= B) return;

    const float *wrow = router_w + (size_t)e * (size_t)hidden;
    const float *xb   = x        + (size_t)b * (size_t)hidden;
    float partial = 0.0f;
    if ((hidden & 3u) == 0u) {
        // Vectorized float4 loads (hidden multiple of 4; rows are 16B-aligned).
        const float4 *w4 = reinterpret_cast<const float4*>(wrow);
        const float4 *x4 = reinterpret_cast<const float4*>(xb);
        const unsigned int n4 = hidden >> 2;
        for (unsigned int k = lane; k < n4; k += 32u) {
            const float4 wv = w4[k];
            const float4 xv = x4[k];
            partial += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
    } else {
        for (unsigned int k = lane; k < hidden; k += 32u) {
            partial += wrow[k] * xb[k];
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) partial += __shfl_down(partial, o);
    if (lane == 0) {
        const float p = 1.0f / (1.0f + expf(-partial));
        probs[(size_t)b * n_expert + e]  = p;
        scores[(size_t)b * n_expert + e] = (bias != nullptr) ? (p + bias[e]) : p;
    }
}

extern "C" __global__ void laguna_router_topk_batched(
        int          * __restrict__ selected,   // [B, n_used]
        float        * __restrict__ weights,    // [B, n_used]
        const float  * __restrict__ probs,      // [B, n_expert]
        const float  * __restrict__ scores,     // [B, n_expert]
        unsigned int   n_expert,
        unsigned int   n_used,
        float          weight_scale,
        float          weight_eps) {
    __shared__ float s_probs[LAG_MAX_EXPERTS];
    __shared__ float s_scores[LAG_MAX_EXPERTS];
    __shared__ float buf[LAG_MAX_EXPERTS];
    __shared__ int   idxs[LAG_MAX_EXPERTS];
    __shared__ int   s_sel[LAG_MAX_USED];
    __shared__ float s_w[LAG_MAX_USED];
    __shared__ float s_sum;

    const unsigned int tid = threadIdx.x;
    const unsigned int b   = blockIdx.z;
    const float *probs_b  = probs  + (size_t)b * n_expert;
    const float *scores_b = scores + (size_t)b * n_expert;
    if (tid < n_expert) {
        s_probs[tid]  = probs_b[tid];
        s_scores[tid] = scores_b[tid];
    }
    __syncthreads();

    for (unsigned int kk = 0; kk < n_used; ++kk) {
        if (tid < n_expert) { buf[tid] = s_scores[tid]; idxs[tid] = (int)tid; }
        else                { buf[tid] = -INFINITY;     idxs[tid] = -1; }
        __syncthreads();
        for (unsigned int stride = LAG_MAX_EXPERTS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float bo = buf[tid + stride];
                const int   io = idxs[tid + stride];
                const float bs = buf[tid];
                const int   is = idxs[tid];
                if (bo > bs || (bo == bs && io >= 0 && (is < 0 || io < is))) {
                    buf[tid] = bo; idxs[tid] = io;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int w = idxs[0];
            s_sel[kk] = w;
            if (w >= 0) s_scores[w] = -INFINITY;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < n_used; ++j) {
            const int w = s_sel[j];
            const float p = (w >= 0) ? s_probs[w] : 0.0f;
            s_w[j] = p;
            sum += p;
        }
        if (sum < weight_eps) sum = weight_eps;
        s_sum = sum;
    }
    __syncthreads();

    if (tid < n_used) {
        selected[(size_t)b * n_used + tid] = s_sel[tid];
        weights[(size_t)b * n_used + tid]  = s_w[tid] / s_sum * weight_scale;
    }
}

// --------------------------------------------------------------------------
// 2. Per-head softplus attention gate, applied in place to the attention
//    output. Launch: grid (n_head,1,1), block (head_dim,1,1).
//    softplus(x) = log(1+exp(x)), numerically stable (x>20 -> x).
// --------------------------------------------------------------------------
extern "C" __global__ void laguna_softplus_gate(
        float        * attn_out,       // [n_head, head_dim] in place
        const float  * __restrict__ gate_logits, // [n_head]
        unsigned int   n_head,
        unsigned int   head_dim) {
    const unsigned int h = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (h >= n_head || d >= head_dim) return;
    const float g = gate_logits[h];
    const float sp = (g > 20.0f) ? g : log1pf(expf(g));
    attn_out[(size_t)h * head_dim + d] *= sp;
}

// --------------------------------------------------------------------------
// 3. Contiguous f32 -> f16 cast (plumbing). Used to build the f16 Q vector and
//    to append K/V into the [n_kv, n_kv_head, head_dim] f16 cache at a given
//    position (the caller passes a slice_view at pos * n_kv_head * head_dim).
// --------------------------------------------------------------------------
extern "C" __global__ void laguna_cast_f32_f16(
        unsigned short * out,               // f16 bits
        const float    * __restrict__ in,
        unsigned int     n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    union { __half h; unsigned short u; } cvt;
    cvt.h = __float2half_rn(in[i]);
    out[i] = cvt.u;
}

// --------------------------------------------------------------------------
// INT8 symmetric KV-cache quantize (LAGUNA_FP8_KV write path — the gate name is
// kept; the backing format is now int8, which strictly dominates e4m3 on KV
// quality at matched bytes, see docs/laguna/FP8_KV_REVIVE.md 2026-07-28). Layout
// is BYTE-IDENTICAL to the old e4m3 path: 1 byte/elem + one f32 per-row scale.
// Quantizes `n_rows` contiguous f32 rows of `head_dim` (<=128) into signed int8
// (two's complement, stored in the u8 buffer) + one f32 per-row symmetric scale
// = amax/127 (per-(token,kv_head) granularity — robust to per-vector outliers,
// self-contained, no calibration). q = round-to-nearest(v/scale) clamped to
// [-127,127]; dequant is (float)q * scale. Same amax/127 scheme as the ds4 q8_0
// path. amax/127.5 was measured slightly worse, so 127 is used. Portable — no
// arch intrinsics; identical ISA on gfx1200/1201 and the LUT-free fallback arch.
//   grid: (n_rows,1,1)  block: (128,1,1); one WG per (token,kv_head) row.
// --------------------------------------------------------------------------
#if !(defined(__gfx1200__) || defined(__gfx1201__))
__constant__ float c_lg_e4m3fn_values[128] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,
};
#endif

static __device__ __forceinline__ unsigned char laguna_f32_to_fp8(float x) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    int packed = __builtin_amdgcn_cvt_pk_fp8_f32(x, 0.0f, 0, false);
    return (unsigned char)(packed & 0xff);
#else
    // Nearest-value binary search over the magnitude table; returns index|sign.
    const unsigned int sgn = (x < 0.0f) ? 0x80u : 0x00u;
    float ax = fabsf(x);
    if (ax > 448.0f) ax = 448.0f;
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (c_lg_e4m3fn_values[mid] <= ax) lo = mid; else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - c_lg_e4m3fn_values[best]);
        const float nd = fabsf(ax - c_lg_e4m3fn_values[best + 1]);
        if (nd < bd) best++;
    }
    return (unsigned char)((unsigned int)best | sgn);
#endif
}

extern "C" __global__ void laguna_quantize_fp8_kv(
        unsigned char * __restrict__ out,   // [n_rows, head_dim] e4m3fn bytes
        float         * __restrict__ scale, // [n_rows] per-row f32 scale
        const float   * __restrict__ in,    // [n_rows, head_dim] f32
        unsigned int    n_rows,
        unsigned int    head_dim) {
    const unsigned int row = blockIdx.x;
    const unsigned int t   = threadIdx.x;
    if (row >= n_rows) return;
    const float * r = in + (unsigned long long)row * head_dim;

    __shared__ float red[128];
    const float v = (t < head_dim) ? r[t] : 0.0f;
    red[t] = (t < head_dim) ? fabsf(v) : 0.0f;
    __syncthreads();
    for (unsigned int s = 64u; s > 0u; s >>= 1) {
        if (t < s) { const float o = red[t + s]; if (o > red[t]) red[t] = o; }
        __syncthreads();
    }
    // Per-row symmetric scale = amax/127 (floored to avoid div-by-zero rows).
    float sc = red[0] * (1.0f / 127.0f);
    if (sc < 1.0e-12f) sc = 1.0e-12f;
    if (t == 0) scale[row] = sc;
    if (t < head_dim) {
        // Symmetric int8: round-half-away-from-zero, clamp to [-127,127], store
        // as two's-complement in the u8 buffer. Matches the LAGUNA_FP8_FAKE
        // fmt=2 fake-quant that validated the numerics (roundf, amax/127).
        float q = roundf(v / sc);
        if (q >  127.0f) q =  127.0f;
        if (q < -127.0f) q = -127.0f;
        out[(unsigned long long)row * head_dim + t] = (unsigned char)(signed char)(int)q;
    }
}

// --------------------------------------------------------------------------
// FP8 (e4m3fn) FAKE-QUANT round-trip for KV attribution experiments
// (LAGUNA_FP8_FAKE). Quantizes each row to e4m3fn + per-row amax/448 scale and
// immediately DEQUANTIZES back to f16 into the normal f16 KV cache, so the
// downstream attention reads use the *identical* f16 read kernels. This
// isolates the pure NUMERICAL error of fp8 (no read-kernel path change) and lets
// K and V be round-tripped independently to attribute drift. NOT a production
// path — diagnostic only.
typedef float v2f_lg __attribute__((ext_vector_type(2)));
static __device__ __forceinline__ float laguna_fp8_to_f32(unsigned char b) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    v2f_lg r = __builtin_amdgcn_cvt_pk_f32_fp8((int)(unsigned int)b, false);
    return r[0];
#else
    const float sign = (b & 0x80u) ? -1.0f : 1.0f;
    return sign * c_lg_e4m3fn_values[b & 0x7fu];
#endif
}

// e5m2 (bf8) round-trip via native converts (gfx1200/1201). Max normal 57344.
static __device__ __forceinline__ unsigned char laguna_f32_to_bf8(float x) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    int packed = __builtin_amdgcn_cvt_pk_bf8_f32(x, 0.0f, 0, false);
    return (unsigned char)(packed & 0xff);
#else
    return 0;
#endif
}
static __device__ __forceinline__ float laguna_bf8_to_f32(unsigned char b) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    v2f_lg r = __builtin_amdgcn_cvt_pk_f32_bf8((int)(unsigned int)b, false);
    return r[0];
#else
    return 0.0f;
#endif
}

// Generalized fake-quant round-trip with configurable scale block size.
// fmt: 0=e4m3 (amax scale = 448), 1=e5m2 (amax scale = 57344),
//      2=int8 symmetric (scale = amax/127, round-half-away, clamp [-127,127]),
//      3=int8 symmetric (scale = amax/127.5, round-half-away, full [-128,127]).
// blk: elements sharing one amax scale (<=head_dim; head_dim => per-row). The
// per-block amax reduction runs within each contiguous blk-chunk of the row.
extern "C" __global__ void laguna_fp8_roundtrip_kv(
        _Float16     * __restrict__ out,   // [n_rows, head_dim] f16
        const float  * __restrict__ in,    // [n_rows, head_dim] f32
        unsigned int    n_rows,
        unsigned int    head_dim,
        unsigned int    blk,
        unsigned int    fmt) {
    const unsigned int row = blockIdx.x;
    const unsigned int t   = threadIdx.x;
    if (row >= n_rows) return;
    const float * r = in + (unsigned long long)row * head_dim;

    const bool  is_int8 = (fmt == 2u || fmt == 3u);
    // Denominator that maps amax onto the format's max code.
    const float fmax = (fmt == 1u) ? 57344.0f
                     : (fmt == 2u) ? 127.0f
                     : (fmt == 3u) ? 127.5f
                     : 448.0f;
    __shared__ float red[128];
    const float v = (t < head_dim) ? r[t] : 0.0f;
    red[t] = (t < head_dim) ? fabsf(v) : 0.0f;
    __syncthreads();
    // Per-block amax: reduce within each contiguous [g*blk, g*blk+blk) chunk.
    const unsigned int g = t / blk;      // block index of this lane
    const unsigned int base = g * blk;
    for (unsigned int s = blk >> 1; s > 0u; s >>= 1) {
        const unsigned int l = t - base;
        if (t < head_dim && l < s) {
            const float o = red[t + s];
            if (o > red[t]) red[t] = o;
        }
        __syncthreads();
    }
    float sc = red[base] / fmax;
    if (sc < 1.0e-12f) sc = 1.0e-12f;
    if (t < head_dim) {
        float deq;
        if (is_int8) {
            // Symmetric int8: round-half-away-from-zero, then clamp to the
            // signed range and dequantize by the same per-block scale.
            float q = roundf(v / sc);
            const float lo = (fmt == 3u) ? -128.0f : -127.0f;
            if (q >  127.0f) q =  127.0f;
            if (q <  lo)     q =  lo;
            deq = q * sc;
            out[(unsigned long long)row * head_dim + t] = (_Float16)deq;
        } else {
            float q = v / sc;
            if (q > fmax) q = fmax;
            if (q < -fmax) q = -fmax;
            if (fmt == 1u) deq = laguna_bf8_to_f32(laguna_f32_to_bf8(q));
            else           deq = laguna_fp8_to_f32(laguna_f32_to_fp8(q));
            out[(unsigned long long)row * head_dim + t] = (_Float16)(deq * sc);
        }
    }
}

// --------------------------------------------------------------------------
// 4. Fused per-head QK RMSNorm. Normalizes each head's `head_dim` slice by its
//    OWN RMS and scales by a shared `weight[head_dim]` — replacing the 72 (Q) +
//    8 (K) per-slice `rms_norm_weighted` launches with a single grid launch.
//    Launch: grid (n_head,1,1), block (256,1,1); one WG per head.
//    Numerics mirror rms_norm.hip:rms_norm_weighted EXACTLY (double-precision
//    sum-of-squares, tree reduction, rsqrtf(mean_sq + eps)) for greedy parity.
//        out[h,i] = (in[h,i] / sqrt(mean(in[h,:]^2) + eps)) * weight[i]
// --------------------------------------------------------------------------
extern "C" __global__ void laguna_qk_rmsnorm(
        float        * __restrict__ out,      // [n_head, head_dim]
        const float  * __restrict__ in,       // [n_head, head_dim]
        const float  * __restrict__ weight,   // [head_dim] shared across heads
        unsigned int   n_head,
        unsigned int   head_dim,
        float          eps) {
    __shared__ float sdata[256];
    const unsigned int h     = blockIdx.x;
    const unsigned int tid   = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (h >= n_head) return;

    const float *xr = in  + (size_t)h * (size_t)head_dim;
    float       *orow = out + (size_t)h * (size_t)head_dim;

    // First pass: sum of squares (double accum, matches rms_norm_weighted).
    double partial = 0.0;
    for (unsigned int i = tid; i < head_dim; i += block) {
        float v = xr[i];
        partial += (double)v * (double)v;
    }
    sdata[tid] = (float)partial;
    __syncthreads();

    for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    const float mean_sq = sdata[0] / (float)head_dim;
    const float scale = rsqrtf(mean_sq + eps);

    for (unsigned int i = tid; i < head_dim; i += block) {
        orow[i] = xr[i] * scale * weight[i];
    }
}
