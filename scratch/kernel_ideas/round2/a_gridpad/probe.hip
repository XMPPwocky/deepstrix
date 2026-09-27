// Synthetic probe for the power-of-two dispatch pathology (round 2 item a).
// Every WG past n_valid exits at once (same guard shape as q8_0_quantize_f32).
#include <hip/hip_runtime.h>
extern "C" __global__ void probe_nop(float* out, unsigned n_valid) {
    const unsigned wg = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if (wg >= n_valid) return;
    if (threadIdx.x == 0) out[wg] = 1.0f;
}
// quantize-like traffic: every thread reads one float, writes one float.
extern "C" __global__ void probe_rw(float* __restrict__ out, const float* __restrict__ in, unsigned n_valid) {
    const unsigned wg = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if (wg >= n_valid) return;
    const size_t i = (size_t)wg * blockDim.x + threadIdx.x;
    out[i] = in[i] * 1.5f + 0.25f;
}
// rw + a per-thread dependent FMA loop of `iters` (kernel length knob)
extern "C" __global__ void probe_spin(float* __restrict__ out, const float* __restrict__ in, unsigned n_valid, unsigned iters) {
    const unsigned wg = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if (wg >= n_valid) return;
    const size_t i = (size_t)wg * blockDim.x + threadIdx.x;
    float v = in[i];
    for (unsigned k = 0; k < iters; ++k) v = v * 1.0000001f + 0.5f;
    out[i] = v;
}
