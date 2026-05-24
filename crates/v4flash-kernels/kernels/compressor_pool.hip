// compressor_pool — per-output-dim softmax-weighted average over the
// compressor state buffer. Mirrors ds4's `compressor_pool_decode_state`
// (ds4.c:6439-6490) exactly.
//
// For ratio==4 (state has 2*ratio=8 rows of width=2*head_dim, output is
// head_dim):
//   - Each output dim j samples from rows 0..3 cols 0..head_dim-1 AND
//     rows 4..7 cols head_dim..2*head_dim-1 (8 entries per output dim).
// For ratio==128 (state has ratio=128 rows of width=head_dim, output is
// head_dim):
//   - Each output dim j samples from rows 0..127 col j (128 entries).
//
// One thread per output dim, one workgroup. block.x = head_dim (512 main,
// 128 indexer).
//
// Per-thread sequential softmax — `ratio` ≤ 128, fast in registers.

#include <hip/hip_runtime.h>

#define POOL_NEG_INF (-3.4028235e38f)

extern "C" __global__ void compressor_pool(
        float       * __restrict__ out,           // [head_dim]
        const float * __restrict__ state_kv,      // [ratio * coff, width]
        const float * __restrict__ state_score,   // same shape
        unsigned int  head_dim,
        unsigned int  compress_ratio) {
    const unsigned int j = threadIdx.x;
    if (j >= head_dim) return;

    const unsigned int coff = (compress_ratio == 4u) ? 2u : 1u;
    const unsigned int width = coff * head_dim;

    float max_score = POOL_NEG_INF;
    if (compress_ratio == 4u) {
        // Split layout: rows 0..3 col j + rows 4..7 col head_dim+j.
        for (unsigned int r = 0; r < compress_ratio; r++) {
            const float sp = state_score[(unsigned long long)r * width + j];
            const float sc = state_score[(unsigned long long)(compress_ratio + r) * width + head_dim + j];
            if (sp > max_score) max_score = sp;
            if (sc > max_score) max_score = sc;
        }
    } else {
        for (unsigned int r = 0; r < compress_ratio; r++) {
            const float s = state_score[(unsigned long long)r * width + j];
            if (s > max_score) max_score = s;
        }
    }

    if (max_score <= POOL_NEG_INF * 0.5f) {
        out[j] = 0.0f;
        return;
    }

    float denom = 0.0f;
    float sum = 0.0f;
    if (compress_ratio == 4u) {
        for (unsigned int r = 0; r < compress_ratio; r++) {
            const float wp = expf(state_score[(unsigned long long)r * width + j] - max_score);
            const float wc = expf(state_score[(unsigned long long)(compress_ratio + r) * width + head_dim + j] - max_score);
            denom += wp + wc;
            sum += wp * state_kv[(unsigned long long)r * width + j];
            sum += wc * state_kv[(unsigned long long)(compress_ratio + r) * width + head_dim + j];
        }
    } else {
        for (unsigned int r = 0; r < compress_ratio; r++) {
            const float w = expf(state_score[(unsigned long long)r * width + j] - max_score);
            denom += w;
            sum += w * state_kv[(unsigned long long)r * width + j];
        }
    }

    out[j] = (denom > 0.0f) ? (sum / denom) : 0.0f;
}
