// attention_mixed — sink-aware causal softmax + weighted-sum over both the
// raw SWA cache AND the compressed-KV cache, with an optional per-comp-row
// allow mask. Mirrors ds4.c `layer_attention_mixed_one_decode_scratch`
// (ds4.c:6738). One workgroup per head, 256 threads per workgroup.
//
// When n_comp == 0 and mask is null, the math reduces exactly to
// `attention_swa` (M5) — that's how this kernel covers L=0,1 (pure SWA),
// L=3,5,…,41 (ratio==128 but n_comp==0 in our 57-tok prompt), and
// L=2,4,…,42 (ratio==4 with active comp rows + mask).
//
// Mask semantics: `mask[r] == 0` makes the r-th compressed row contribute
// `score = -INFINITY` to the softmax. `expf(-INFINITY - max_score)` is
// IEEE 754 zero, so the masked row drops out of both the denom and the
// weighted-sum axpy without any special-casing.
//
// Shared-mem budget per workgroup:
//   partial[256] = 1024 bytes
//   scores[144]  =  576 bytes  (ATTN_MIXED_MAX_KEYS)
//   weights[144] =  576 bytes
//   inv_denom    =    4 bytes
//   ≈ 2.2 KB total

#include <hip/hip_runtime.h>

#define ATTN_MIXED_MAX_KEYS 144  // 128 raw + 14 comp + headroom for M1 prompt

extern "C" __global__ void attention_mixed(
        float          * out,
        const float    * q,
        const float    * raw_kv,
        const float    * comp_kv,        // may be null when n_comp == 0
        const int32_t  * mask,           // may be null = all-permit
        const float    * sinks,
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     n_raw,
        unsigned int     n_comp,
        float            kq_scale) {
    const unsigned int h = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (h >= n_head) return;

    __shared__ float partial[256];
    __shared__ float scores[ATTN_MIXED_MAX_KEYS];
    __shared__ float weights[ATTN_MIXED_MAX_KEYS];
    __shared__ float inv_denom_s;

    const float *qh = q + (unsigned long long)h * head_dim;
    float *oh = out + (unsigned long long)h * head_dim;
    const float sink_h = sinks[h];

    // Phase 1a: raw_kv dot products.
    for (unsigned int r = 0; r < n_raw; r++) {
        const float *kv_row = raw_kv + (unsigned long long)r * head_dim;
        float p = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block) {
            p += qh[i] * kv_row[i];
        }
        partial[tid] = p;
        __syncthreads();
        for (unsigned int s = block / 2; s > 0; s >>= 1) {
            if (tid < s) partial[tid] += partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[r] = partial[0] * kq_scale;
        __syncthreads();
    }

    // Phase 1b: comp_kv dot products (skipped when n_comp == 0).
    for (unsigned int r = 0; r < n_comp; r++) {
        const unsigned int idx = n_raw + r;
        if (mask != nullptr && mask[r] == 0) {
            if (tid == 0) scores[idx] = -INFINITY;
            __syncthreads();
            continue;
        }
        const float *kv_row = comp_kv + (unsigned long long)r * head_dim;
        float p = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block) {
            p += qh[i] * kv_row[i];
        }
        partial[tid] = p;
        __syncthreads();
        for (unsigned int s = block / 2; s > 0; s >>= 1) {
            if (tid < s) partial[tid] += partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[idx] = partial[0] * kq_scale;
        __syncthreads();
    }

    // Phase 2 + 3: single-thread max + softmax denom. n_total ≤ 144, fast.
    if (tid == 0) {
        const unsigned int n_total = n_raw + n_comp;
        float max_score = sink_h;
        for (unsigned int r = 0; r < n_total; r++) {
            if (scores[r] > max_score) max_score = scores[r];
        }
        float denom = expf(sink_h - max_score);
        for (unsigned int r = 0; r < n_total; r++) {
            float w = expf(scores[r] - max_score);   // = 0 when scores[r] == -INF
            weights[r] = w;
            denom += w;
        }
        inv_denom_s = 1.0f / denom;
    }
    __syncthreads();

    // Phase 4: weighted sum. Two passes over the head_dim (raw then comp),
    // axpy order matches CPU per-d so f32 reductions are bit-identical
    // modulo phase-1 dot-product ULPs flowing through softmax.
    const float inv = inv_denom_s;
    for (unsigned int d = tid; d < head_dim; d += block) {
        float sum = 0.0f;
        for (unsigned int r = 0; r < n_raw; r++) {
            sum += weights[r] * raw_kv[(unsigned long long)r * head_dim + d];
        }
        for (unsigned int r = 0; r < n_comp; r++) {
            const float w = weights[n_raw + r];
            // w == 0 happens on masked rows (we still touch comp_kv[r,d] in
            // the legal-pointer case, but if n_comp > 0 then comp_kv is
            // guaranteed non-null by the caller).
            sum += w * comp_kv[(unsigned long long)r * head_dim + d];
        }
        oh[d] = sum * inv;
    }
}
