// cand_rms -- rms_norm_weighted_batched with every load issued up front.
//
// Production (K/rms_norm.hip:52, grid (B) x 256): `for (i = tid; i < n; i += 256) partial +=
// (double)x*x` with n a runtime value -> the compiler cannot hoist the loads; each iteration is
// a global load (~1 us round trip at decode occupancy: ONE workgroup per row, nothing else in
// flight) followed by one DFMA, then the 8-level LDS tree, then a second pass that RE-READS x.
// At n = 5120 that is 20 dependent load latencies (measured ~11 us direct at b = 1 for a
// 40 KB row; the launch floor is ~3 us).
//
// Here the per-thread element count is a template constant (n == NPT * 256 dispatches to it,
// anything else falls back to production's loop, so the kernel is a drop-in replacement):
// all NPT x loads and NPT weight loads are issued first (register arrays, fully unrolled), the
// double accumulation runs in the SAME order i = tid, tid+256, ..., the LDS tree is the same,
// and the output uses the x values already in registers -- the same f32 the second pass would
// have re-read. BIT-EXACT.
//
// Same signature and grid as rms_norm_weighted_batched. rms_norm_weighted (the per-row head
// prep kernel) is the B = 1 case of the same math: launch this with grid (1).

#include <hip/hip_runtime.h>

template <unsigned NPT>
__device__ __forceinline__ void rms_w_fast(float* __restrict__ ob, const float* __restrict__ xb,
                                           const float* __restrict__ weight, unsigned int n, float eps,
                                           float* sdata) {
    const unsigned int tid = threadIdx.x;
    float xv[NPT], wv[NPT];
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) xv[j] = xb[tid + 256u * j];
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) wv[j] = weight[tid + 256u * j];
    double partial = 0.0;
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) {
        const float v = xv[j];
        partial += (double)v * (double)v;
    }
    sdata[tid] = (float)partial;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float mean_sq = sdata[0] / (float)n;
    const float scale = rsqrtf(mean_sq + eps);
    #pragma unroll
    for (unsigned j = 0; j < NPT; ++j) ob[tid + 256u * j] = xv[j] * scale * wv[j];
}

__device__ __forceinline__ void rms_w_generic(float* __restrict__ ob, const float* __restrict__ xb,
                                              const float* __restrict__ weight, unsigned int n, float eps,
                                              float* sdata) {
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;
    double partial = 0.0;
    for (unsigned int i = tid; i < n; i += block) {
        float v = xb[i];
        partial += (double)v * (double)v;
    }
    sdata[tid] = (float)partial;
    __syncthreads();
    for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float mean_sq = sdata[0] / (float)n;
    const float scale = rsqrtf(mean_sq + eps);
    for (unsigned int i = tid; i < n; i += block) ob[i] = xb[i] * scale * weight[i];
}

extern "C" __global__ __launch_bounds__(256) void rms_norm_weighted_batched_fast(
        float* out, const float* x, const float* weight, unsigned int n, float eps) {
    __shared__ float sdata[256];
    const unsigned int b = blockIdx.x;
    const float* xb = x + (size_t)b * (size_t)n;
    float* ob = out + (size_t)b * (size_t)n;
    if (blockDim.x != 256u) { rms_w_generic(ob, xb, weight, n, eps, sdata); return; }
    switch (n) {
        case 512u:  rms_w_fast<2>(ob, xb, weight, n, eps, sdata); break;    // kv / comp norms
        case 1280u: rms_w_fast<5>(ob, xb, weight, n, eps, sdata); break;    // q_a norm
        case 5120u: rms_w_fast<20>(ob, xb, weight, n, eps, sdata); break;   // attn/ffn input norm, head prep
        default:    rms_w_generic(ob, xb, weight, n, eps, sdata); break;
    }
}
