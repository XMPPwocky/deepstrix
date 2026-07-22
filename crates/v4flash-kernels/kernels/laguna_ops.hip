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
