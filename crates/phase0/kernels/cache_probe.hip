#include <hip/hip_runtime.h>
#include <stdint.h>

// Gate D — cache-probe kernels.
//
// Each kernel reads `n_words` u32 from `buf` in a coalesced strided
// pattern. The two variants differ in cache hint:
//   read_cached:       normal load (compiler / driver may cache in MALL/L2)
//   read_nontemporal:  __builtin_nontemporal_load (bypass hint)
//
// We accumulate via XOR + write to per-thread output slot so the
// compiler doesn't elide the loads.

__device__ __forceinline__ uint32_t do_read_cached(const uint32_t* p) {
    return *p;
}

__device__ __forceinline__ uint32_t do_read_nontemporal(const uint32_t* p) {
    return __builtin_nontemporal_load(p);
}

extern "C" __global__ void read_cached(
    const uint32_t* __restrict__ buf,
    uint32_t* __restrict__ out,
    uint32_t n_words)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (uint32_t i = tid; i < n_words; i += total) {
        acc ^= do_read_cached(buf + i);
    }
    out[tid] = acc;
}

extern "C" __global__ void read_nontemporal(
    const uint32_t* __restrict__ buf,
    uint32_t* __restrict__ out,
    uint32_t n_words)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (uint32_t i = tid; i < n_words; i += total) {
        acc ^= do_read_nontemporal(buf + i);
    }
    out[tid] = acc;
}

// Write-only "pollution" kernel — used between two read passes to push
// other data into the cache and force eviction of the read buffer.
extern "C" __global__ void pollute_write(
    uint32_t* __restrict__ buf,
    uint32_t n_words,
    uint32_t seed)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = gridDim.x * blockDim.x;
    for (uint32_t i = tid; i < n_words; i += total) {
        buf[i] = seed ^ i;
    }
}
