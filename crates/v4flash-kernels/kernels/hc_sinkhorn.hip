// hc_sinkhorn — split a 24-element "mix" vector into 3 sub-vectors:
//   out[0..n_hc]               = sigmoid_stable(mix[i] * scale[0] + base[i]) + eps    (n_hc weights for weighted_sum)
//   out[n_hc..2*n_hc]          = 2 * sigmoid_stable(mix[off] * scale[1] + base[off])  (post-scale weights for hc_post)
//   out[2*n_hc..2*n_hc+n_hc*n_hc] = Sinkhorn-normalized n_hc × n_hc comb matrix       (mixing matrix)
//
// Mirrors ds4 hc_split_sinkhorn_one (ds4.c:4220). For V4-Flash n_hc=4,
// iters=20, eps=1e-6. mix and base are hc_mix_dim=24, scale is 3.
//
// Single workgroup, single thread — the math is sequential per the
// row/col scaling alternation in Sinkhorn. 24 elements is tiny.

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_sinkhorn(
        float        * __restrict__ out,     // [2*n_hc + n_hc*n_hc]
        const float  * __restrict__ mix,     // [hc_mix_dim]
        const float  * __restrict__ scale,   // [3]
        const float  * __restrict__ base,    // [hc_mix_dim]
        unsigned int   n_hc,
        unsigned int   iters,
        float          eps) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    // out[0..n_hc]: sigmoid(mix*pre_scale + base) + eps
    for (unsigned int i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        float sig;
        if (z >= 0.0f) {
            const float e = expf(-z);
            sig = 1.0f / (1.0f + e);
        } else {
            const float e = expf(z);
            sig = e / (1.0f + e);
        }
        out[i] = sig + eps;
    }
    // out[n_hc..2*n_hc]: 2 * sigmoid(mix*post_scale + base) at offset n_hc
    for (unsigned int i = 0; i < n_hc; i++) {
        const unsigned int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        float sig;
        if (z >= 0.0f) {
            const float e = expf(-z);
            sig = 1.0f / (1.0f + e);
        } else {
            const float e = expf(z);
            sig = e / (1.0f + e);
        }
        out[off] = 2.0f * sig;
    }

    // Sinkhorn over n_hc × n_hc comb (n_hc≤16 so 256 floats fits in regs/local)
    float c[256];
    for (unsigned int dst = 0; dst < n_hc; dst++) {
        float row_max = -INFINITY;
        for (unsigned int src = 0; src < n_hc; src++) {
            const unsigned int idx = src + dst * n_hc;
            const unsigned int off = 2u * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (unsigned int src = 0; src < n_hc; src++) {
            const unsigned int idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv = 1.0f / row_sum;
        for (unsigned int src = 0; src < n_hc; src++) {
            const unsigned int idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    // First half-iteration: normalise per-source.
    for (unsigned int src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (unsigned int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
        const float inv = 1.0f / (sum + eps);
        for (unsigned int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }

    // Remaining iterations: dst-normalize then src-normalize.
    for (unsigned int iter = 1; iter < iters; iter++) {
        for (unsigned int dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (unsigned int src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (unsigned int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }
        for (unsigned int src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (unsigned int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (unsigned int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }

    for (unsigned int i = 0; i < n_hc * n_hc; i++) out[2u * n_hc + i] = c[i];
}
