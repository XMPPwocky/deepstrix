// Load-pattern probes for G3-optimal (bench only, never shipped): the same 16 KB key tile
// (16 rows x 1 KB f16) read by one warp in three patterns, from an L2-resident buffer:
//   frag : the WMMA fragment pattern (lane = row col, 16 B at dims u*16 + kh): 16 rows x 32 B per load
//   tr   : global_load_tr_b128 over 16 rows x 32 B per load (the W.V V pattern), 32 dim tiles
//   coal : 512 contiguous bytes per load (lane l: 16 B at row j/2, half j%2)
#include <hip/hip_runtime.h>
#include <stdint.h>
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef short short8 __attribute__((ext_vector_type(8)));

#define ITER_BODY(...)                                                                               \
    const unsigned tid = threadIdx.x, wave = tid >> 5, lane = tid & 31u;                             \
    const unsigned nw = blockDim.x >> 5;                                                             \
    unsigned acc = 0;                                                                                \
    for (unsigned i = 0; i < iters; i++) {                                                           \
        const unsigned t = (blockIdx.x * nw + wave + i * gridDim.x * nw) % n_tiles;                  \
        const _Float16* tb = buf + (size_t)t * 16u * 512u;                                           \
        short8 v[32];                                                                                \
        _Pragma("unroll") for (unsigned u = 0; u < 32u; u++) { __VA_ARGS__; }                         \
        _Pragma("unroll") for (unsigned u = 0; u < 32u; u++) acc ^= (unsigned)v[u][0];              \
    }                                                                                                \
    if (acc == 0x12345u) sink[tid] = acc;

extern "C" __global__ __launch_bounds__(512) void probe_frag(const _Float16* buf, unsigned n_tiles, unsigned iters, unsigned* sink) {
#if defined(__gfx1201__)
    ITER_BODY({ const unsigned col = lane & 15u, kh = (lane >> 4) * 8u;
                v[u] = *(const short8*)(tb + col * 512u + u * 16u + kh); })
#endif
}
extern "C" __global__ __launch_bounds__(512) void probe_tr(const _Float16* buf, unsigned n_tiles, unsigned iters, unsigned* sink) {
#if defined(__gfx1201__)
    ITER_BODY({ const unsigned kk = (lane >> 4) * 8u + (lane & 7u), dc = u * 16u + ((lane >> 3) & 1u) * 8u;
                v[u] = __builtin_amdgcn_global_load_tr_b128_v8i16((short8 __attribute__((address_space(1)))*)(tb + kk * 512u + dc)); })
#endif
}
extern "C" __global__ __launch_bounds__(512) void probe_coal(const _Float16* buf, unsigned n_tiles, unsigned iters, unsigned* sink) {
#if defined(__gfx1201__)
    ITER_BODY({ v[u] = *(const short8*)(tb + u * 256u + lane * 8u); })
#endif
}
// LDS-transposed variant cost is the coal cost + an LDS round trip; not probed here.

// Dependent-load latency: one lane follows next[] for `steps` hops (the buffer holds a permutation).
extern "C" __global__ void probe_chase(const unsigned* next, unsigned steps, unsigned* out) {
    unsigned i = 0;
    for (unsigned s = 0; s < steps; s++) { asm volatile("" : "+v"(i)); i = __builtin_nontemporal_load(next + i); }
    if (threadIdx.x == 0) out[blockIdx.x] = i;
}
extern "C" __global__ void probe_chase_c(const unsigned* next, unsigned steps, unsigned* out) {
    unsigned i = 0;
    for (unsigned s = 0; s < steps; s++) { asm volatile("" : "+v"(i)); i = next[i]; }
    if (threadIdx.x == 0) out[blockIdx.x] = i;
}
