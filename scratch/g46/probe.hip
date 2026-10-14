// Floor probes for the G6 bench (tests/bench_g6_boundary.rs, loaded from G6_PROBE_HSACO).
// hipcc -O3 --genco --offload-arch=gfx1201 probe.hip -o probe.hsaco
#include <hip/hip_runtime.h>

// One WG per row, 256 threads x 20 floats: the lead's zero-copy read of box 2's partial, alone.
extern "C" __global__ __launch_bounds__(256) void zc_rows(const float* p, float* out, unsigned ne, unsigned sys) {
    const float* r = p + (size_t)blockIdx.x * ne;
    float v[20];
    if (sys) {
        #pragma unroll
        for (unsigned j = 0; j < 20; j++) v[j] = __hip_atomic_load(r + threadIdx.x + 256u * j, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    } else {
        #pragma unroll
        for (unsigned j = 0; j < 20; j++) v[j] = r[threadIdx.x + 256u * j];
    }
    float s = 0.0f;
    #pragma unroll
    for (unsigned j = 0; j < 20; j++) s += v[j];
    if (s == 1234.5f) out[blockIdx.x] = s;
}

// One WG per row: the lead's VRAM loads (moe, shared, 4 x after_attn) + the stores of 4 copies.
extern "C" __global__ __launch_bounds__(256) void vram_rows(const float* moe, const float* sh, const float* aa, float* outp, unsigned ne) {
    const unsigned z = blockIdx.x, t = threadIdx.x;
    float a[20], b[20], x[4][20];
    #pragma unroll
    for (unsigned j = 0; j < 20; j++) a[j] = moe[(size_t)z * ne + t + 256u * j];
    #pragma unroll
    for (unsigned j = 0; j < 20; j++) b[j] = sh[(size_t)z * ne + t + 256u * j];
    #pragma unroll
    for (unsigned h = 0; h < 4; h++) {
        #pragma unroll
        for (unsigned j = 0; j < 20; j++) x[h][j] = aa[(size_t)z * 4 * ne + h * ne + t + 256u * j];
    }
    #pragma unroll
    for (unsigned h = 0; h < 4; h++) {
        #pragma unroll
        for (unsigned j = 0; j < 20; j++) outp[(size_t)z * 4 * ne + h * ne + t + 256u * j] = x[h][j] + a[j] * b[j];
    }
}

extern "C" __global__ void empty_k(float* out) {
    if (threadIdx.x == 999) out[0] = 1.0f;
}
