// hc_sigmoid_bias — w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + DS4_HC_EPS.
// Mirrors the per-stream sigmoid in output_hc_head_one (ds4.c:8155-8157)
// and the identical body in output_logits_one_decode_scratch (ds4.c:8208).
// DS4_HC_EPS = 1.0e-6f (ds4.c:53). n is small (DS4_N_HC = 4).

#include <hip/hip_runtime.h>

#define DS4_HC_EPS 1.0e-6f

extern "C" __global__ void hc_sigmoid_bias(
        float        * __restrict__ out,
        const float  * __restrict__ pre,
        const float  * __restrict__ scale,
        const float  * __restrict__ base,
        unsigned int   n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = scale[0];
    const float x = pre[i] * s + base[i];
    float sig;
    if (x >= 0.0f) {
        const float e = expf(-x);
        sig = 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        sig = e / (1.0f + e);
    }
    out[i] = sig + DS4_HC_EPS;
}
