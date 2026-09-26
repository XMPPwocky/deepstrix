// Example candidate: vec_add_inplace with float4 loads/stores (n % 4 == 0 assumed).
#include <hip/hip_runtime.h>

extern "C" __global__ void vec_add_inplace_v4(float* __restrict__ out, const float* __restrict__ rhs,
                                              unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 4 >= n) return;
    float4 a = reinterpret_cast<float4*>(out)[i];
    float4 b = reinterpret_cast<const float4*>(rhs)[i];
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    reinterpret_cast<float4*>(out)[i] = a;
}
