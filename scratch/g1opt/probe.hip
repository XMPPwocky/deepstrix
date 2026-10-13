// G1-optimal probes (dGPU only, bench-only; loaded by tests/g1o_probe.rs from G1O_HSACO):
//   probe_m{mode}_r{rows}: 768 waves stream the router's two f16 gates (768 rows x 10240 B = 7.86 MB)
//     with each load width, the whole row in flight (u16 lane-strided 533 GB/s; u32 / b64 / b128 605);
//   probe_ring{K}_r1: a ring of K u32 words per lane in flight (K = 8 already at the roof);
//   probe_q{K}: the latency of a store + fence, scalar loads and dependent L2-hot vector loads from a
//     wave beside 8 waves holding K gate words per lane in flight (the CU FIFO / congestion data);
//   probe_pr* / prs* / pra*: a prefetch, a pause, then a ring (per-WG pause lengths vary with occupancy).
// Build: hipcc -O3 --genco --offload-arch=gfx1201 probe.hip -o /tmp/g1o/probe.hsaco
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWB 10240u
#define NROWS 768u

__device__ __forceinline__ unsigned long long now() { return __builtin_readsteadycounter(); }

// MODE: 0 = u16 lane-strided (h20), 1 = u32 pairs, 2 = b64, 3 = b128. R = rows per wave.
template <int MODE, int R>
__device__ void probe_body(const uint8_t* __restrict__ w, unsigned* sink, unsigned long long* ts, unsigned nwg) {
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    if (tid == 0) ts[blockIdx.x * 2] = now();
    unsigned acc = 0u;
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const unsigned row = (blockIdx.x * 8u + warp) * R + r;
        if (row >= NROWS) break;
        const uint8_t* wr = w + (size_t)row * ROWB;
        if constexpr (MODE == 0) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(wr);
            uint16_t v[160];
            #pragma unroll
            for (int j = 0; j < 160; ++j) v[j] = p[lane + 32 * j];
            #pragma unroll
            for (int j = 0; j < 160; ++j) acc = acc * 33u + v[j];
        } else if constexpr (MODE == 1) {
            const unsigned* p = reinterpret_cast<const unsigned*>(wr);
            unsigned v[80];
            #pragma unroll
            for (int c = 0; c < 80; ++c) v[c] = p[32 * c + lane];
            #pragma unroll
            for (int c = 0; c < 80; ++c) acc ^= v[c];
        } else if constexpr (MODE == 2) {
            const uint2* p = reinterpret_cast<const uint2*>(wr);
            uint2 v[40];
            #pragma unroll
            for (int c = 0; c < 40; ++c) v[c] = p[32 * c + lane];
            #pragma unroll
            for (int c = 0; c < 40; ++c) acc ^= v[c].x ^ v[c].y;
        } else {
            const uint4* p = reinterpret_cast<const uint4*>(wr);
            uint4 v[20];
            #pragma unroll
            for (int c = 0; c < 20; ++c) v[c] = p[32 * c + lane];
            #pragma unroll
            for (int c = 0; c < 20; ++c) acc ^= v[c].x ^ v[c].y ^ v[c].z ^ v[c].w;
        }
    }
    if (acc == 0x9e3779b9u) sink[tid] = acc;
    __syncthreads();
    if (tid == 0) ts[blockIdx.x * 2 + 1] = now();
}

#define PK(M, R) extern "C" __global__ __launch_bounds__(256) void probe_m##M##_r##R(const uint8_t* w, unsigned* sink, unsigned long long* ts, unsigned nwg) { probe_body<M, R>(w, sink, ts, nwg); }
PK(0, 1)
PK(1, 1)
PK(2, 1)
PK(3, 1)
PK(1, 2)
PK(3, 2)
PK(3, 3)

// flush: read n16 uint4 with the whole grid
extern "C" __global__ __launch_bounds__(256) void probe_flush(const uint4* p, unsigned n16, unsigned* sink) {
    unsigned acc = 0;
    for (unsigned i = blockIdx.x * 256u + threadIdx.x; i < n16; i += gridDim.x * 256u) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) sink[threadIdx.x] = acc;
}

extern "C" __global__ void probe_empty(unsigned* sink) {
    if (threadIdx.x == 999u) sink[0] = 1u;
}

// Queue probe: 96 WGs x 320 threads. Waves 0-7 stream one gate row each (80 u32 loads in flight) when
// STREAM; wave 8 measures, from its start: 8 dependent VECTOR loads of an L2-hot line, then 8 dependent
// SCALAR loads, then a store + device fence. Wave 9 (STREAM 2): the same vector chain but the WG's
// streaming waves wait for wave 9's first load to be ISSUED (LDS flag) before streaming.
template <int K>
__device__ void q_body(const uint8_t* __restrict__ w, unsigned* hot, unsigned* sink, unsigned long long* ts) {
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    unsigned long long* t = ts + blockIdx.x * 8u;
    if (warp < 8u) {
        if (K > 0) {
            const unsigned row = blockIdx.x * 8u + warp;
            const unsigned* p = reinterpret_cast<const unsigned*>(w + (size_t)row * ROWB);
            unsigned v[K > 0 ? K : 1];
            #pragma unroll
            for (int c = 0; c < K; ++c) v[c] = p[32 * c + lane];
            unsigned acc = 0;
            #pragma unroll
            for (int c = 0; c < K; ++c) acc ^= v[c];
            if (acc == 0x9e3779b9u) sink[tid] = acc;
        }
    } else if (warp == 8u) {
        const unsigned long long t0 = now();
        while (now() - t0 < 50ull) {}   // 0.5 us: let the stream get queued first
        const unsigned long long a = now();
        if (lane == 0) hot[16384u + blockIdx.x * 32u] = 7u;
        __threadfence();
        const unsigned long long bt = now();
        unsigned sidx = 0;
        #pragma unroll 1
        for (int i = 0; i < 8; ++i) {
            unsigned r;
            const unsigned* pp = hot + 8192u + blockIdx.x * 16u + (sidx & 1u);
            asm volatile("s_load_b32 %0, %1, 0x0 scope:SCOPE_DEV\n\ts_wait_kmcnt 0x0" : "=s"(r) : "s"(pp) : "memory");
            sidx = r;
        }
        const unsigned long long c = now();
        unsigned idx = sidx;
        #pragma unroll 1
        for (int i = 0; i < 8; ++i) idx = __hip_atomic_load(hot + (idx & 1u) + blockIdx.x * 64u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const unsigned long long d = now();
        if (lane == 0) { t[0] = t0; t[1] = a; t[2] = bt; t[3] = c; t[4] = d; t[5] = idx; }
    }
}
extern "C" __global__ __launch_bounds__(320) void probe_q0(const uint8_t* w, unsigned* hot, unsigned* sink, unsigned long long* ts) { q_body<0>(w, hot, sink, ts); }
extern "C" __global__ __launch_bounds__(320) void probe_q80(const uint8_t* w, unsigned* hot, unsigned* sink, unsigned long long* ts) { q_body<80>(w, hot, sink, ts); }
extern "C" __global__ __launch_bounds__(320) void probe_q40(const uint8_t* w, unsigned* hot, unsigned* sink, unsigned long long* ts) { q_body<40>(w, hot, sink, ts); }
extern "C" __global__ __launch_bounds__(320) void probe_q20(const uint8_t* w, unsigned* hot, unsigned* sink, unsigned long long* ts) { q_body<20>(w, hot, sink, ts); }
extern "C" __global__ __launch_bounds__(320) void probe_q10(const uint8_t* w, unsigned* hot, unsigned* sink, unsigned long long* ts) { q_body<10>(w, hot, sink, ts); }
extern "C" __global__ void probe_hot(unsigned* hot) { hot[blockIdx.x * 256u + threadIdx.x] = 0u; }

// Ring stream: each wave keeps K u32 words per lane in flight (word c + K issued after word c is consumed).
template <int K>
__device__ void ring_body(const uint8_t* __restrict__ w, unsigned* sink, unsigned long long* ts) {
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    if (tid == 0) ts[blockIdx.x * 2] = now();
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned* p = reinterpret_cast<const unsigned*>(w + (size_t)row * ROWB);
    unsigned v[K];
    #pragma unroll
    for (int c = 0; c < K; ++c) v[c] = p[32 * c + lane];
    unsigned acc = 0;
    #pragma unroll
    for (int c = 0; c < 80; ++c) {
        acc = acc * 33u + v[c % K];
        asm volatile("" : "+v"(acc) :: "memory");
        if (c + K < 80) v[c % K] = p[32 * (c + K) + lane];
    }
    if (acc == 0x9e3779b9u) sink[tid] = acc;
    __syncthreads();
    if (tid == 0) ts[blockIdx.x * 2 + 1] = now();
}
#define RK(K) extern "C" __global__ __launch_bounds__(256) void probe_ring##K##_r1(const uint8_t* w, unsigned* sink, unsigned long long* ts, unsigned nwg) { ring_body<K>(w, sink, ts); }
RK(4)
RK(8)
RK(12)
RK(16)
RK(24)
RK(32)
RK(48)

// Paused ring: P0 words prefetched at entry, a pause of PAUSE x 10 ns (from the WG's start), then a ring of
// K words ahead for the rest (issue order pinned).
template <int P0, int K, int PAUSE, int SLEEP = 1>
__device__ void pring_body(const uint8_t* __restrict__ w, unsigned* sink, unsigned long long* ts) {
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const unsigned long long t0 = now();
    if (tid == 0) ts[blockIdx.x * 2] = t0;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned* p = reinterpret_cast<const unsigned*>(w + (size_t)row * ROWB);
    unsigned v[80];
    #pragma unroll
    for (int c = 0; c < P0; ++c) v[c] = p[32 * c + lane];
    asm volatile("" ::: "memory");
    if (SLEEP == 2) {
        // ALU spin, no message traffic: PAUSE x 10 dependent v_fma (~5 cycles each)
        float f = (float)lane;
        for (int i = 0; i < PAUSE * 10; ++i) asm volatile("v_fma_f32 %0, %0, %0, 1.0" : "+v"(f));
        if (f == 1234.5f) sink[tid] = 1u;
    } else {
        while (now() - t0 < (unsigned long long)PAUSE) {
            if (SLEEP) __builtin_amdgcn_s_sleep(2);
        }
    }
    if (tid == 0) ts[2048 + blockIdx.x] = now();
    unsigned acc = 0;
    #pragma unroll
    for (int c = 0; c < 80; ++c) {
        if (c + K >= P0 && c + K < 80) v[c + K] = p[32 * (c + K) + lane];
        asm volatile("" ::: "memory");
        acc = acc * 33u + v[c];
        asm volatile("" : "+v"(acc) :: "memory");
    }
    if (acc == 0x9e3779b9u) sink[tid] = acc;
    __syncthreads();
    if (tid == 0) ts[blockIdx.x * 2 + 1] = now();
}
#define PR(P0, K, PA) extern "C" __global__ __launch_bounds__(256) void probe_pr_##P0##_##K##_##PA##_r1(const uint8_t* w, unsigned* sink, unsigned long long* ts, unsigned nwg) { pring_body<P0, K, PA>(w, sink, ts); }
PR(24, 16, 0)
PR(24, 16, 600)
PR(24, 32, 600)
PR(40, 16, 600)
PR(16, 16, 0)
#define PRS(P0, K, PA) extern "C" __global__ __launch_bounds__(256) void probe_prs_##P0##_##K##_##PA##_r1(const uint8_t* w, unsigned* sink, unsigned long long* ts, unsigned nwg) { pring_body<P0, K, PA, 0>(w, sink, ts); }
PRS(24, 16, 600)
PRS(24, 16, 200)
PR(24, 16, 200)
PRS(24, 16, 300)
PRS(24, 16, 400)
PRS(24, 16, 500)
PRS(0, 16, 300)
PRS(0, 16, 600)
PRS(0, 16, 1200)
#define PRA(P0, K, PA) extern "C" __global__ __launch_bounds__(256) void probe_pra_##P0##_##K##_##PA##_r1(const uint8_t* w, unsigned* sink, unsigned long long* ts, unsigned nwg) { pring_body<P0, K, PA, 2>(w, sink, ts); }
PRA(24, 16, 0)
PRA(24, 16, 300)
PRA(24, 16, 600)
PRA(24, 16, 1200)
