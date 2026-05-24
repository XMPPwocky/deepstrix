// f16_roundtrip — in-place x[i] = f32(f16(x[i])) per element. Mirrors
// ds4's `f16_round_inplace_cpu` (ds4.c:1586) used implicitly by
// `kv_cache_push_comp` (ds4.c:6385-6390) when storing compressed KV.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void f16_roundtrip(float *x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __half2float(__float2half_rn(x[i]));
}
