// Probe: does this clang accept an LDS-only ("local") workgroup fence, and what does it emit?
#include <hip/hip_runtime.h>
extern "C" __global__ void probe_fence(float* out, const float* in) {
    __shared__ float s[256];
    float pre = in[threadIdx.x + 1024];           // a load we want to stay in flight across the barrier
    s[threadIdx.x] = in[threadIdx.x];
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    out[threadIdx.x] = s[(threadIdx.x + 1) & 255] + pre;
}
