// indexer_qat — in-place Hadamard128 rotation + E2M1 FP4 activation
// quantize-dequantize round trip on 128-wide indexer rows. Mirrors ds4's
// `dsv4_indexer_qat_rows_inplace_cpu` (ds4.c, added in 5bc1e6d "Flash
// graph correctness") and the CUDA `indexer_hadamard_fp4_kernel`.
//
// The official DeepSeek V4 graph rotates indexer activations with a
// 128-wide Hadamard transform (normalised by 1/sqrt(128)) and immediately
// runs the FP4 QAT simulation: per 32-element block,
//   scale = 2^ceil(log2(amax / 6))    (amax floored at 6 * 2^-126)
//   x     = e2m1_nearest_even(clamp(x / scale, ±6)) * scale
// This applies to both indexer Q rows and the ratio-4 indexer compressor
// KV rows, after RoPE and before top-k scoring / cache append; without it
// the top-k compressed-row selection is not the model's graph.
//
// Launch: grid (n_rows, 1, 1), block (128, 1, 1). One workgroup per row.

#include <hip/hip_runtime.h>

__device__ static float e2m1fn_value(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

// Nearest E2M1 value with ties broken toward the even mantissa index
// (matches ds4's dsv4_e2m1fn_dequant_cpu exactly).
__device__ static float e2m1fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - e2m1fn_value(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - e2m1fn_value(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * e2m1fn_value(best);
}

extern "C" __global__ void indexer_qat(
        float        * __restrict__ x,   // [n_rows, 128] in-place
        unsigned int   n_rows) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (row >= n_rows || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (unsigned long long)row * 128u;
    vals[tid] = xr[tid];
    __syncthreads();

    // Hadamard128 butterfly. Same pairwise add/sub tree as the CPU loop
    // (stride 1, 2, ..., 64) — bit-exact regardless of thread mapping.
    for (unsigned int stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            const unsigned int base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            const float a = vals[base];
            const float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    // 1/sqrt(128) normalisation, then per-32-block FP4 round trip.
    const float v = vals[tid] * 0.08838834764831845f;
    const unsigned int block_base = tid & ~31u;
    const unsigned int lane = tid & 31u;
    absbuf[tid] = fabsf(v);
    __syncthreads();
    for (unsigned int stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] =
                fmaxf(absbuf[block_base + lane], absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    const float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    // scale = 2^ceil(log2(amax / 6)). Computed exactly via frexpf instead
    // of log2f (device log2f is not correctly rounded; frexpf gives the
    // mathematically exact ceil for every finite input).
    const float t = amax / 6.0f;
    int e;
    const float m = frexpf(t, &e);      // t = m * 2^e, m in [0.5, 1)
    const int cexp = (m == 0.5f) ? e - 1 : e;
    const float scale = ldexpf(1.0f, cexp);
    float q = v / scale;
    if (q > 6.0f) q = 6.0f;
    if (q < -6.0f) q = -6.0f;
    xr[tid] = e2m1fn_dequant(q) * scale;
}
