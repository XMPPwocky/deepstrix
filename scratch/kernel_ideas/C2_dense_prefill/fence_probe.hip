// Probe: does the LDS-only workgroup fence emit s_wait_dscnt only (no s_wait_loadcnt / global_inv)?
#include <hip/hip_runtime.h>
extern "C" __global__ void fence_probe(float* out, const float* in) {
    __shared__ float s[256];
    float v = in[threadIdx.x + 256];      // an outstanding global load across the barrier
    s[threadIdx.x] = in[threadIdx.x];
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    out[threadIdx.x] = s[(threadIdx.x + 1) & 255] + v;
}
