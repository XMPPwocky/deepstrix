// W0 microbenchmarks for qe_attn_g4 (docs/v41/EMBED_PHASE_DESIGN.md §19.11, the GPU
// window's first 10 minutes). Each pins one hardware fact the design's expected rate
// rests on; none changes the design, they move the expected % of peak. Built with
// -mcumode (build.rs CU_MODE_STEMS) like the kernel's primary cell; qe_attn_w0wgp.hip
// is the same source in WGP mode (probe d's CU-vs-WGP comparison).
//
//   (b) qw_mix_w*_v*_e*: W v_wmma_f32_16x16x16_f16 + V VALU (v_fma_f32) + E v_exp_f32
//       per iteration, interleaved round-robin, 16 waves per CU (8 per SIMD, 1 WG per
//       CU via 40 KB of LDS). The slope of clocks vs V says whether VALU co-issues
//       under WMMA (slope ~0) or serializes on the issue port (slope ~1 clk / op).
//   (c) qw_mix_w0_v0_e64 vs qw_mix_w0_v64_e0: the v_exp_f32 issue rate vs plain VALU.
//   (d) qw_lds_b128: lane-linear ds_load_b128 (the kernel's operand reads), B/clk/CU,
//       at 8 or 16 waves (blockDim 256 / 512), CU vs WGP module.
//   (e) qw_barrier_rearrive: the NW16 cell's exact signal/wait pattern (3-stage ring,
//       signal BEFORE compute) with one wave slept on chosen tiles so a fast wave
//       re-signals before the slow one executed its wait; every LDS read is checked
//       against the tile it must hold. Must neither hang nor read stale data.
//   (f) qw_fma_mix_vs_cvt: v_fma_mixlo_f16(x, 1.0, 0) vs v_cvt_f16_f32(x) bit identity
//       over all 2^32 f32 patterns (for the post-window P-pack polish only).
#include <hip/hip_runtime.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

#define QW_SIGNAL() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); \
                         __builtin_amdgcn_s_barrier_signal(-1); } while (0)
#define QW_WAIT()   do { __builtin_amdgcn_s_barrier_wait(-1); \
                         __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); } while (0)

#if defined(__gfx1200__) || defined(__gfx1201__)
#define QW_GFX12 1
#endif

// ---- (b) / (c) ----------------------------------------------------------------
template <int W, int V, int E>
__device__ __forceinline__ void qw_mix(float *out, const _Float16 *a_in, unsigned int n_iters) {
#ifdef QW_GFX12
    __shared__ unsigned int pad[10240];          // 40 KB: one WG per CU in CU mode
    if (n_iters == 0xffffffffu) pad[threadIdx.x] = threadIdx.x;
    half8 a, b;
#pragma unroll
    for (int i = 0; i < 8; ++i) { a[i] = a_in[i]; b[i] = a_in[8 + i]; }
    float8 c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = (float)threadIdx.x * 1e-6f + (float)j * 1e-3f;
    const float y = (float)a_in[16], z = (float)a_in[17];
    constexpr int WW = W > 0 ? W : 1;
    for (unsigned int it = 0; it < n_iters; ++it) {
#pragma unroll
        for (int w = 0; w < WW; ++w) {
            if constexpr (W > 0) c[w & 7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c[w & 7]);
#pragma unroll
            for (int j = w * V / WW; j < (w + 1) * V / WW; ++j)
                asm volatile("v_fma_f32 %0, %1, %2, %0" : "+v"(x[j & 7]) : "v"(y), "v"(z));
#pragma unroll
            for (int j = w * E / WW; j < (w + 1) * E / WW; ++j)
                asm volatile("v_exp_f32 %0, %0" : "+v"(x[(j + 3) & 7]));
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += c[i][0] + c[i][7] + x[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s + (float)pad[threadIdx.x & 7] * 0.0f;
#else
    __builtin_trap();
#endif
}

#define QW_MIX(W, V, E)                                                                      \
    extern "C" __global__ __launch_bounds__(512) void qw_mix_w##W##_v##V##_e##E(            \
            float *out, const _Float16 *a_in, unsigned int n_iters) {                        \
        qw_mix<W, V, E>(out, a_in, n_iters);                                                 \
    }
QW_MIX(32, 0, 0)
QW_MIX(32, 32, 0)
QW_MIX(32, 80, 0)
QW_MIX(32, 160, 0)
QW_MIX(32, 0, 16)
QW_MIX(32, 66, 16)   // the steady-path ledger of qe_attn_g4_n16 (66 VALU + 16 exp per 32 WMMA)
QW_MIX(0, 64, 0)
QW_MIX(0, 0, 64)

// ---- (d) ------------------------------------------------------------------------
// Every wave reads 16 lane-linear 512-B fragments per iteration from a 48 KB region
// (offsets move every iteration, so nothing is loop-invariant); out keeps the XOR.
extern "C" __global__ __launch_bounds__(512) void qw_lds_b128(unsigned int *out, unsigned int n_iters) {
    __shared__ u32x4 s[3072];                    // 48 KB, as qe_attn_g4_n16
    for (unsigned int i = threadIdx.x; i < 3072u; i += blockDim.x) s[i] = (u32x4){i, i ^ 1u, i ^ 2u, i ^ 3u};
    __syncthreads();
    const unsigned int lane = threadIdx.x & 31u;
    u32x4 acc = {0, 0, 0, 0};
    for (unsigned int it = 0; it < n_iters; ++it) {
        const u32x4 *base = s + ((it * 16u) % 96u) * 32u + lane;
#pragma unroll
        for (unsigned int k = 0; k < 16u; ++k) acc ^= base[k * 32u];
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc[0] ^ acc[1] ^ acc[2] ^ acc[3];
}

// ---- (e) ------------------------------------------------------------------------
// qe_attn_g4_n16's barrier pattern on its 3 x 16 KB ring: prologue store(0) + signal;
// per tile: wait, store(t+1 -> next stage), signal (if t < last), [one wave sleeps on
// the tiles in sleep_mask], read the WHOLE current stage lane-linear and check each
// chunk is tile t's. err += mismatches; done += 1 per WG that finished.
__device__ __forceinline__ u32x4 qw_tile_val(unsigned int t, unsigned int cid) {
    const unsigned int v = t * 4096u + cid;
    return (u32x4){v, ~v, t ^ (cid << 8), 0xA5A5A5A5u ^ t};
}

extern "C" __global__ __launch_bounds__(512) void qw_barrier_rearrive(
        unsigned int *err, unsigned int *done, unsigned int n_tiles, unsigned int sleep_wave, unsigned int sleep_mask) {
#ifdef QW_GFX12
    __shared__ u32x4 lds[3u * 1024u];
    const unsigned int tid = threadIdx.x, lane = tid & 31u;
    const unsigned int w = __builtin_amdgcn_readfirstlane(tid >> 5);
    if (blockDim.x != 512u) __builtin_trap();
    auto store = [&](unsigned int t, unsigned int stage) {
        lds[stage * 1024u + tid] = qw_tile_val(t, tid);
        lds[stage * 1024u + 512u + tid] = qw_tile_val(t, 512u + tid);
    };
    store(0u, 0u);
    QW_SIGNAL();
    unsigned int errs = 0, stage = 0;
    for (unsigned int t = 0; t < n_tiles; ++t) {
        const unsigned int nxt = stage == 2u ? 0u : stage + 1u;
        QW_WAIT();
        if (t + 1u < n_tiles) store(t + 1u, nxt);
        if (t + 1u < n_tiles) QW_SIGNAL();
        if (w == (sleep_wave + blockIdx.x) % 16u && ((sleep_mask >> (t & 31u)) & 1u)) {
#pragma unroll
            for (int i = 0; i < 8; ++i) __builtin_amdgcn_s_sleep(127);
        }
#pragma unroll 4
        for (unsigned int k = 0; k < 32u; ++k) {
            const u32x4 v = lds[stage * 1024u + k * 32u + lane];
            const u32x4 e = qw_tile_val(t, k * 32u + lane);
            errs += (v[0] != e[0]) | (v[1] != e[1]) | (v[2] != e[2]) | (v[3] != e[3]);
        }
        stage = nxt;
    }
    if (errs) atomicAdd(err, errs);
    __syncthreads();
    if (tid == 0) atomicAdd(done, 1u);
#else
    __builtin_trap();
#endif
}

// ---- (f) ------------------------------------------------------------------------
// One f32 bit pattern per thread: grid (65536, 256) x 256 threads = all 2^32.
extern "C" __global__ __launch_bounds__(256) void qw_fma_mix_vs_cvt(unsigned int *mism, unsigned int *first) {
#ifdef QW_GFX12
    const unsigned int bits = ((blockIdx.y * 65536u + blockIdx.x) << 8) | threadIdx.x;
    const float x = __uint_as_float(bits);
    const unsigned short cv = __builtin_bit_cast(unsigned short, (_Float16)x);   // v_cvt_f16_f32 (RNE)
    unsigned int mx = 0;
    asm volatile("v_fma_mixlo_f16 %0, %1, 1.0, 0" : "+v"(mx) : "v"(x));
    const unsigned short mv = (unsigned short)(mx & 0xffffu);
    const bool nan_c = (cv & 0x7c00u) == 0x7c00u && (cv & 0x3ffu), nan_m = (mv & 0x7c00u) == 0x7c00u && (mv & 0x3ffu);
    if (!(cv == mv || (nan_c && nan_m))) {
        atomicAdd(mism, 1u);
        atomicMin(first, bits);
    }
#else
    __builtin_trap();
#endif
}
