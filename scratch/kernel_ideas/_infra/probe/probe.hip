// Profiler capability probe: one streaming + LDS + VALU kernel, launched N times on HIP device argv[1].
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
__global__ void probe_kernel(const float* __restrict__ x, float* __restrict__ y, int n, float a) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? x[i] : 0.f;
    s[threadIdx.x] = v;
    __syncthreads();
    float acc = v;
#pragma unroll 8
    for (int k = 0; k < 64; ++k) acc = fmaf(acc, a, s[(threadIdx.x + k) & 255]);
    if (i < n) y[i] = acc;
}
int main(int argc, char** argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    int iters = argc > 2 ? atoi(argv[2]) : 5;
    hipSetDevice(dev);
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, dev);
    int n = 8 << 20;  // 32 MB per buffer
    float *x, *y;
    hipMalloc(&x, n * 4);
    hipMalloc(&y, n * 4);
    hipMemset(x, 0, n * 4);
    for (int it = 0; it < iters; ++it)
        hipLaunchKernelGGL(probe_kernel, dim3(n / 256), dim3(256), 0, 0, x, y, n, 0.999f);
    hipError_t e = hipDeviceSynchronize();
    printf("dev %d %s: %d launches, grid %d, waves(wave32)=%d -> %s\n", dev, p.gcnArchName, iters,
           n / 256, n / 32, hipGetErrorString(e));
    hipFree(x);
    hipFree(y);
    return e != hipSuccess;
}
