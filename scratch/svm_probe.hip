// Standalone probe: replicate llama.cpp's GGML_CUDA_ENABLE_UNIFIED_MEMORY path.
//   1. hipMallocManaged a big buffer
//   2. hipMemAdvise(SetCoarseGrain)
//   3. touch from a kernel (writes + reads)
//   4. report bytes_done / failure point
//
// Usage:  svm_probe <gib> [device]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "FAIL %s:%d  %s -> %d (%s)\n", \
                __FILE__, __LINE__, #expr, _e, hipGetErrorString(_e)); \
        return 1; \
    } \
} while (0)

__global__ void touch_kernel(uint32_t* p, size_t n_u32, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t k = i; k < n_u32; k += stride) {
        p[k] = (uint32_t)(k ^ seed);
    }
}

__global__ void verify_kernel(const uint32_t* p, size_t n_u32, uint32_t seed,
                              uint64_t* err_count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint64_t local = 0;
    for (size_t k = i; k < n_u32; k += stride) {
        if (p[k] != (uint32_t)(k ^ seed)) local++;
    }
    if (local) atomicAdd((unsigned long long*)err_count, (unsigned long long)local);
}

int main(int argc, char** argv) {
    double gib = (argc > 1) ? atof(argv[1]) : 8.0;
    int device = (argc > 2) ? atoi(argv[2]) : 0;
    size_t bytes = (size_t)(gib * (1024.0 * 1024.0 * 1024.0));
    size_t n_u32 = bytes / 4;

    int n_dev = 0;
    CK(hipGetDeviceCount(&n_dev));
    fprintf(stderr, "hipGetDeviceCount: %d\n", n_dev);
    CK(hipSetDevice(device));
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, device));
    fprintf(stderr, "device %d: %s (arch=%s)  totalGlobalMem=%.2f GiB\n",
            device, prop.name, prop.gcnArchName,
            (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    fprintf(stderr, "requesting %.2f GiB (%zu bytes) via hipMallocManaged\n", gib, bytes);
    uint32_t* p = nullptr;
    CK(hipMallocManaged((void**)&p, bytes));
    fprintf(stderr, "  alloc ok  ptr=%p\n", (void*)p);

    // Mirror llama.cpp: hint coarse-grain, swallow errors
    hipError_t adv = hipMemAdvise(p, bytes, hipMemAdviseSetCoarseGrain, device);
    fprintf(stderr, "  hipMemAdvise(SetCoarseGrain) -> %d (%s)\n", adv, hipGetErrorString(adv));
    (void)hipGetLastError();

    // 1. write from kernel
    fprintf(stderr, "launching touch kernel (write)…\n");
    uint32_t seed = 0xDEADBEEF;
    int block = 256;
    int grid = 4096;
    touch_kernel<<<grid, block, 0, 0>>>(p, n_u32, seed);
    CK(hipDeviceSynchronize());
    fprintf(stderr, "  touch ok\n");

    // 2. verify from kernel
    fprintf(stderr, "launching verify kernel…\n");
    uint64_t* d_err = nullptr;
    CK(hipMalloc((void**)&d_err, sizeof(uint64_t)));
    CK(hipMemset(d_err, 0, sizeof(uint64_t)));
    verify_kernel<<<grid, block, 0, 0>>>(p, n_u32, seed, d_err);
    CK(hipDeviceSynchronize());
    uint64_t err = 0;
    CK(hipMemcpy(&err, d_err, sizeof(uint64_t), hipMemcpyDeviceToHost));
    fprintf(stderr, "  verify ok  errors=%llu / %zu u32s\n",
            (unsigned long long)err, n_u32);

    // 3. spot-check from host
    fprintf(stderr, "host spot-check (first/middle/last)…\n");
    uint32_t expect_first = (uint32_t)(0 ^ seed);
    uint32_t expect_mid   = (uint32_t)((n_u32 / 2) ^ seed);
    uint32_t expect_last  = (uint32_t)((n_u32 - 1) ^ seed);
    fprintf(stderr, "  first: got 0x%08x  exp 0x%08x  %s\n",
            p[0], expect_first, p[0] == expect_first ? "OK" : "MISMATCH");
    fprintf(stderr, "  mid  : got 0x%08x  exp 0x%08x  %s\n",
            p[n_u32 / 2], expect_mid, p[n_u32 / 2] == expect_mid ? "OK" : "MISMATCH");
    fprintf(stderr, "  last : got 0x%08x  exp 0x%08x  %s\n",
            p[n_u32 - 1], expect_last, p[n_u32 - 1] == expect_last ? "OK" : "MISMATCH");

    CK(hipFree(p));
    CK(hipFree(d_err));
    fprintf(stderr, "DONE %.2f GiB ok\n", gib);
    return 0;
}
