// attention_swa — sink-aware causal softmax + weighted-sum over the SWA KV
// cache. Mirrors ds4.c `layer_attention_rows_one` (ds4.c:4955) for the
// pure-SWA layers (ratio==0). One workgroup per head, 256 threads per
// workgroup.
//
// Per head h:
//   max_score = max(sinks[h], max_r dot(qh, kv[r]) * (1/sqrt(head_dim)))
//   denom     = exp(sinks[h] - max_score) + sum_r exp(score[r] - max_score)
//   out[h, d] = (sum_r weight[r] * kv[r, d]) / denom
//
// Sinks contribute to the softmax denominator but have NO value vector;
// they're effectively "always-attendable" rows with a zero-vector value.
//
// Shared-memory budget per workgroup (head_dim=512, n_kv≤128):
//   partial[256] = 1024 bytes  (dot-product tree reduction)
//   scores[128]  =  512 bytes  (per-row pre-softmax scores)
//   weights[128] =  512 bytes  (per-row post-softmax weights)
//   inv_denom    =    4 bytes
//   ≈ 2 KB total

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define ATTN_SWA_MAX_KV 128
// Batched prefill rows inside a Vision-Exp image span see a widened raw
// window (back max(W-1, left), forward right; capped at W + 384 = 512 keys,
// see het/image_spans.rs). `attention_swa_batched` therefore takes its
// scores/weights arrays as DYNAMIC LDS sized `2 * max_n_kv` floats at
// launch: a text-only chunk passes max_n_kv <= 128 and gets exactly the
// 1024 B the old static arrays used, so occupancy is unchanged; only a
// chunk containing image rows pays the wider 4096 B. The single-token
// decode kernel above keeps its static 128-row arrays (decode rows are
// never image rows).
#define ATTN_SWA_BATCHED_MAX_KV 512

extern "C" __global__ void attention_swa(
        float        * out,            // [n_head, head_dim]
        const float  * q,              // [n_head, head_dim] post-RoPE
        const _Float16 * kv,           // [n_kv, head_dim]   cached rows (f16)
        const float  * sinks,          // [n_head]
        unsigned int   n_head,
        unsigned int   head_dim,
        unsigned int   n_kv,
        float          kq_scale) {     // 1 / sqrt(head_dim), host-precomputed
    const unsigned int h = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (h >= n_head) return;

    __shared__ float partial[256];
    __shared__ float scores[ATTN_SWA_MAX_KV];
    __shared__ float weights[ATTN_SWA_MAX_KV];
    __shared__ float inv_denom_s;

    const float *qh = q + (unsigned long long)h * head_dim;
    float *oh = out + (unsigned long long)h * head_dim;
    const float sink_h = sinks[h];

    // Phase 1: per-row dot product → scores[r] = dot(qh, kv[r]) * kq_scale.
    for (unsigned int r = 0; r < n_kv; r++) {
        const _Float16 *kv_row = kv + (unsigned long long)r * head_dim;
        float p = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block) {
            p += qh[i] * (float)kv_row[i];
        }
        partial[tid] = p;
        __syncthreads();

        for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            scores[r] = partial[0] * kq_scale;
        }
        __syncthreads();
    }

    // Phase 2 + 3: single-thread max + softmax denom (n_kv ≤ 128, fast).
    if (tid == 0) {
        float max_score = sink_h;
        for (unsigned int r = 0; r < n_kv; r++) {
            if (scores[r] > max_score) max_score = scores[r];
        }
        float denom = expf(sink_h - max_score);
        for (unsigned int r = 0; r < n_kv; r++) {
            float w = expf(scores[r] - max_score);
            weights[r] = w;
            denom += w;
        }
        inv_denom_s = 1.0f / denom;
    }
    __syncthreads();

    // Phase 4: weighted sum of value vectors, normalised.
    //   oh[d] = inv_denom * sum_r weights[r] * kv[r, d]
    // Iteration order over r matches the CPU's sequential axpy (per d), so
    // f32 reductions are bit-identical to CPU modulo phase-1 dot-product
    // ULPs flowing through softmax.
    const float inv = inv_denom_s;
    for (unsigned int d = tid; d < head_dim; d += block) {
        float sum = 0.0f;
        for (unsigned int r = 0; r < n_kv; r++) {
            sum += weights[r] * (float)kv[(unsigned long long)r * head_dim + d];
        }
        oh[d] = sum * inv;
    }
}

// M50 Phase 4: per-token causal SWA attention.
// Grid (n_head, B, 1) — each WG handles one (head, token) pair.
// Per-token q[B, n_head, head_dim], out[B, n_head, head_dim], n_raw_per[B].
// kv_cache is SHARED across tokens (causal: token b sees rows [0..n_raw_per[b]]).
extern "C" __global__ void attention_swa_batched(
        float        * out,                 // [B, n_head, head_dim]
        const float  * q,                   // [B, n_head, head_dim]
        const _Float16 * kv,                // oversized: [..., head_dim] shared (f16)
        const float  * sinks,               // [n_head]
        const int    * n_raw_per,           // [B] causal count for this token
        const int    * n_raw_offset_per,    // [B] starting slot into oversized cache
        unsigned int   n_head,
        unsigned int   head_dim,
        unsigned int   max_n_kv,            // dynamic-LDS stride: >= max(n_raw_per)
        float          kq_scale) {
    const unsigned int h   = blockIdx.x;
    const unsigned int b   = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (h >= n_head) return;

    __shared__ float partial[256];
    __shared__ float inv_denom_s;
    // Dynamic LDS: [scores[max_n_kv], weights[max_n_kv]].
    extern __shared__ float lds_sw[];
    float *scores  = lds_sw;
    float *weights = lds_sw + max_n_kv;

    const unsigned int n_kv = (unsigned int)n_raw_per[b];
    const unsigned int raw_off = (unsigned int)n_raw_offset_per[b];

    const float *qh = q + ((size_t)b * (size_t)n_head + (size_t)h) * (size_t)head_dim;
    float       *oh = out + ((size_t)b * (size_t)n_head + (size_t)h) * (size_t)head_dim;
    const float sink_h = sinks[h];

    for (unsigned int r = 0; r < n_kv; r++) {
        const _Float16 *kv_row = kv + (size_t)(r + raw_off) * (size_t)head_dim;
        float p = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block) {
            p += qh[i] * (float)kv_row[i];
        }
        partial[tid] = p;
        __syncthreads();
        for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0) scores[r] = partial[0] * kq_scale;
        __syncthreads();
    }

    if (tid == 0) {
        float max_score = sink_h;
        for (unsigned int r = 0; r < n_kv; r++) {
            if (scores[r] > max_score) max_score = scores[r];
        }
        float denom = expf(sink_h - max_score);
        for (unsigned int r = 0; r < n_kv; r++) {
            float w = expf(scores[r] - max_score);
            weights[r] = w;
            denom += w;
        }
        inv_denom_s = 1.0f / denom;
    }
    __syncthreads();

    const float inv = inv_denom_s;
    for (unsigned int d = tid; d < head_dim; d += block) {
        float sum = 0.0f;
        for (unsigned int r = 0; r < n_kv; r++) {
            sum += weights[r] * (float)kv[(size_t)(r + raw_off) * (size_t)head_dim + d];
        }
        oh[d] = sum * inv;
    }
}
