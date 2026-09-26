// C2_dense_prefill candidates. Copies of the production q8_0_gemm_wmma_f16x (K/q8_0_matvec_wmma.hip:574)
// with the changes named per kernel. Same argument lists as the production kernels where possible.
//
//   q8_0_gemm_wmma_f16x_lb    : production tile (128x128x32, 8 waves 2x4), __syncthreads replaced by an
//                               LDS-only fence + barrier (gfx12: s_wait_dscnt 0 instead of
//                               s_wait_loadcnt_dscnt 0), so the register prefetch overlaps the WMMAs.
//   q8_0_gemm_wmma_f16x_pf2   : lb + register prefetch TWO k-outers ahead (k-outer loop unrolled x2).
//   q8_0_gemm_wmma_f16x_bn64  : lb, tile 128(m) x 64(n), 8 waves as 4(m) x 2(n), wave tile 32x32.
//   q8_0_gemm_wmma_f16x_64x64 : lb, tile 64x64, 4 waves (128 threads) as 2x2, wave tile 32x32.
//   q8_0_gemm_wmma_i8x        : lb, production tile, activations as (int8 xq, f32 xscale) like
//                               q8_0_gemm_wmma_lds_tiled (Engram drop-in), dequantised at stage.
//   q8_0_gemm_wmma_i8x_bn64   : the same with the 128x64 tile.
//   q8_0_gemv_bpack64_warp8   : dp4a GEMV, weight read once, batch <= 64 in registers (replay twin).
//   q8_0_grouped_gemv_bpack64 : grouped variant.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));
typedef _Float16 half2v_q8 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ void q8x_magic4(uint32_t w, _Float16 scale, uint32_t& lo, uint32_t& hi) {
    const uint32_t u = w ^ 0x80808080u;
    const uint32_t h01 = __builtin_amdgcn_perm(u, 0x64646464u, 0x00050004u);
    const uint32_t h23 = __builtin_amdgcn_perm(u, 0x64646464u, 0x00070006u);
    half2v_q8 a, b;
    __builtin_memcpy(&a, &h01, 4);
    __builtin_memcpy(&b, &h23, 4);
    const half2v_q8 bias = {(_Float16)-1152.0f, (_Float16)-1152.0f};
    a = (a + bias) * scale;
    b = (b + bias) * scale;
    __builtin_memcpy(&lo, &a, 4);
    __builtin_memcpy(&hi, &b, 4);
}

// LDS-only workgroup barrier: release/acquire fences restricted to the local address space so the
// compiler emits s_wait_dscnt 0 (not s_wait_loadcnt_dscnt 0) and no global_inv. Global loads issued
// before it stay in flight across the barrier. Valid here because the only data exchanged through
// the barrier is LDS.
__device__ __forceinline__ void sync_lds() {
#if defined(__gfx1200__) || defined(__gfx1201__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}

// ---------------------------------------------------------------------------------------------
// Generic core. BM x BN x 32 tile, WM x WN waves, wave tile (BM/WM) x (BN/WN), PF-deep register
// prefetch. A (weights) staged as 16-byte int8 slots (BM*2 slots), dequantised into LDS f16 [row][40].
// B (activations): F16 -> 16-byte f16 slots (BN*4); I8 -> 16-byte int8 slots (BN*2) + f32 xscale,
// dequantised as (f16)((float)q * xscale) exactly like q8_0_gemm_wmma_lds_tiled.
// ---------------------------------------------------------------------------------------------
template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB>
__device__ __forceinline__ void f16x_core(
    float* __restrict__ out, const unsigned char* __restrict__ w,
    const uint16_t* __restrict__ x16, const int8_t* __restrict__ xq, const float* __restrict__ xscale,
    uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks, uint32_t ldx)
{
    constexpr int BK = 32;
    constexpr int LDS_STRIDE = BK + 8;               // 40 halves = 80 B
    constexpr int NT_ = WM * WN * 32;                // threads
    constexpr int MT = (BM / WM) / 16, NTL = (BN / WN) / 16;
    constexpr int A_SLOTS = BM * 2;                  // 16 B int8 slots per k-outer
    constexpr int A_PER = (A_SLOTS + NT_ - 1) / NT_;
    constexpr int B_SLOTS = BI8 ? BN * 2 : BN * 4;   // 16 B slots
    constexpr int B_PER = (B_SLOTS + NT_ - 1) / NT_;
    static_assert(A_SLOTS % NT_ == 0 || NT_ % A_SLOTS == 0, "A slots");
    static_assert(B_SLOTS % NT_ == 0 || NT_ % B_SLOTS == 0, "B slots");

    __shared__ __attribute__((aligned(16))) _Float16 A_tile[BM * LDS_STRIDE];
    __shared__ __attribute__((aligned(16))) _Float16 B_tile[BN * LDS_STRIDE];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t wave_m0 = (wave / WN) * (BM / WM);
    const uint32_t wave_n0 = (wave % WN) * (BN / WN);
    const uint32_t n0 = blockIdx.x * BN;
    const uint32_t m0 = blockIdx.y * BM;
    const uint32_t g  = blockIdx.z;
    if (m0 >= M) return;
    const uint32_t ldo = n_groups * M;
    const size_t rowbytes   = (size_t)blocks * 34u;
    const size_t scalebytes = (size_t)blocks * 2u;
    const unsigned char* w_g = w + (size_t)g * (size_t)M * rowbytes;

    // staging roles
    const unsigned char* a_src[A_PER]; uint32_t a_row[A_PER], a_koff[A_PER]; bool a_on[A_PER];
    #pragma unroll
    for (int i = 0; i < A_PER; ++i) {
        const uint32_t slot = tid + i * NT_;
        a_on[i] = slot < A_SLOTS;
        a_row[i] = a_on[i] ? slot >> 1 : 0u;
        a_koff[i] = (slot & 1u) * 16u;
        a_src[i] = w_g + (size_t)(m0 + a_row[i]) * rowbytes;    // m0+row < M by M % BM == 0
    }
    const void* b_src[B_PER]; uint32_t b_row[B_PER], b_koff[B_PER]; bool b_on[B_PER];
    const float* bs_src[B_PER];
    #pragma unroll
    for (int i = 0; i < B_PER; ++i) {
        const uint32_t slot = tid + i * NT_;
        const bool in = slot < B_SLOTS;
        if (BI8) { b_row[i] = in ? slot >> 1 : 0u; b_koff[i] = (slot & 1u) * 16u; }
        else     { b_row[i] = in ? slot >> 2 : 0u; b_koff[i] = (slot & 3u) * 8u; }
        b_on[i] = in && (n0 + b_row[i]) < batch;
        if (BI8) {
            b_src[i] = xq + (size_t)(n0 + b_row[i]) * (size_t)ldx + (size_t)g * K;
            bs_src[i] = xscale + (size_t)(n0 + b_row[i]) * (size_t)(ldx / 32u) + (size_t)g * blocks;
        } else {
            b_src[i] = x16 + (size_t)(n0 + b_row[i]) * (size_t)ldx + (size_t)g * K;
            bs_src[i] = nullptr;
        }
    }

    float8 acc[MT][NTL];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NTL; ++nt)
            acc[mt][nt] = (float8){0,0,0,0,0,0,0,0};

    // prefetch registers, PF stages deep
    uint4 qa[PF][A_PER]; uint16_t sbits[PF][A_PER];
    uint4 vb[PF][B_PER]; float xs[PF][B_PER];
    auto load_raw = [&](int p, uint32_t ko) {
        #pragma unroll
        for (int i = 0; i < A_PER; ++i) if (a_on[i]) {
            __builtin_memcpy(&qa[p][i], a_src[i] + scalebytes + ko + a_koff[i], 16);
            __builtin_memcpy(&sbits[p][i], a_src[i] + 2u * (ko >> 5), 2);
        }
        #pragma unroll
        for (int i = 0; i < B_PER; ++i) {
            vb[p][i] = make_uint4(0u, 0u, 0u, 0u);
            if (BI8) xs[p][i] = 0.f;
            if (b_on[i]) {
                if (BI8) {
                    __builtin_memcpy(&vb[p][i], (const int8_t*)b_src[i] + ko + b_koff[i], 16);
                    xs[p][i] = bs_src[i][ko >> 5];
                } else {
                    __builtin_memcpy(&vb[p][i], (const uint16_t*)b_src[i] + ko + b_koff[i], 16);
                }
            }
        }
    };
    auto stage = [&](int p) {
        #pragma unroll
        for (int i = 0; i < A_PER; ++i) if (a_on[i]) {
            _Float16 d; __builtin_memcpy(&d, &sbits[p][i], 2);
            uint32_t o[8];
            q8x_magic4(qa[p][i].x, d, o[0], o[1]);
            q8x_magic4(qa[p][i].y, d, o[2], o[3]);
            q8x_magic4(qa[p][i].z, d, o[4], o[5]);
            q8x_magic4(qa[p][i].w, d, o[6], o[7]);
            uint4* dst = (uint4*)(&A_tile[a_row[i] * LDS_STRIDE + a_koff[i]]);
            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
            dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
        }
        #pragma unroll
        for (int i = 0; i < B_PER; ++i) {
            const uint32_t slot = tid + i * NT_;
            if (slot >= B_SLOTS) continue;
            if (BI8) {
                // (f16)((float)q * xscale), the lds_tiled expression, 16 values -> 2 x b128
                int8_t q[16]; __builtin_memcpy(q, &vb[p][i], 16);
                const float sc = xs[p][i];
                _Float16 v[16];
                #pragma unroll
                for (int e = 0; e < 16; ++e) v[e] = (_Float16)((float)q[e] * sc);
                uint4* dst = (uint4*)(&B_tile[b_row[i] * LDS_STRIDE + b_koff[i]]);
                __builtin_memcpy(&dst[0], &v[0], 16);
                __builtin_memcpy(&dst[1], &v[8], 16);
            } else {
                uint4* dst = (uint4*)(&B_tile[b_row[i] * LDS_STRIDE + b_koff[i]]);
                dst[0] = vb[p][i];
            }
        }
    };
    auto compute = [&]() {
        #pragma unroll
        for (uint32_t k_inner = 0; k_inner < (uint32_t)BK; k_inner += 16u) {
            const uint32_t kf = k_inner + (lane >> 4) * 8u;
            half8 a_frags[MT];
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const uint32_t r = wave_m0 + mt * 16u + (lane & 15u);
                __builtin_memcpy(&a_frags[mt], &A_tile[r * LDS_STRIDE + kf], 16);
            }
            half8 b_frags[NTL];
            #pragma unroll
            for (int nt = 0; nt < NTL; ++nt) {
                const uint32_t r = wave_n0 + nt * 16u + (lane & 15u);
                __builtin_memcpy(&b_frags[nt], &B_tile[r * LDS_STRIDE + kf], 16);
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                #pragma unroll
                for (int nt = 0; nt < NTL; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frags[mt], b_frags[nt], acc[mt][nt]);
        }
    };
    auto sync = [&]() { if (LB) sync_lds(); else __syncthreads(); };

    #pragma unroll
    for (int p = 0; p < PF; ++p) if ((uint32_t)p * BK < K) load_raw(p, p * BK);
    for (uint32_t k0 = 0; k0 < K; k0 += PF * BK) {
        #pragma unroll
        for (int p = 0; p < PF; ++p) {
            const uint32_t ko = k0 + p * BK;
            if (ko >= K) break;
            stage(p);
            sync();
            if (ko + PF * BK < K) load_raw(p, ko + PF * BK);
            compute();
            sync();
        }
    }

    #pragma unroll
    for (int nt = 0; nt < NTL; ++nt) {
        const uint32_t n_idx = n0 + wave_n0 + nt * 16u + (lane & 15u);
        if (n_idx >= batch) continue;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const uint32_t m_idx = m0 + wave_m0 + mt * 16u + (lane >> 4) * 8u;
            float* dst = out + (size_t)n_idx * (size_t)ldo + (size_t)g * M + m_idx;
            const float8 c = acc[mt][nt];
            const float4 c0 = {c[0], c[1], c[2], c[3]};
            const float4 c1 = {c[4], c[5], c[6], c[7]};
            __builtin_memcpy(dst, &c0, 16);
            __builtin_memcpy(dst + 4, &c1, 16);
        }
    }
}

#define F16X_KERNEL(NAME, BM, BN, WM, WN, PF, LB)                                                     \
extern "C" __global__ void __launch_bounds__(WM * WN * 32) NAME(                                     \
    float* __restrict__ out, const unsigned char* __restrict__ w, const uint16_t* __restrict__ x16,   \
    uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks, uint32_t ldx)         \
{                                                                                                     \
    _Pragma("clang diagnostic push")                                                                  \
    f16x_core<BM, BN, WM, WN, PF, false, LB>(out, w, x16, nullptr, nullptr, K, M, n_groups, batch, blocks, ldx); \
    _Pragma("clang diagnostic pop")                                                                   \
}

#if defined(__gfx1200__) || defined(__gfx1201__)
F16X_KERNEL(q8_0_gemm_wmma_f16x_v2,    128, 128, 2, 4, 1, false)   // control: generic core, production barrier
F16X_KERNEL(q8_0_gemm_wmma_f16x_lb,    128, 128, 2, 4, 1, true)
F16X_KERNEL(q8_0_gemm_wmma_f16x_pf2,   128, 128, 2, 4, 2, true)
F16X_KERNEL(q8_0_gemm_wmma_f16x_bn64,  128,  64, 4, 2, 1, true)
F16X_KERNEL(q8_0_gemm_wmma_f16x_64x64,  64,  64, 2, 2, 1, true)

// int8-activation twins (Engram drop-in for q8_0_gemm_wmma_lds_tiled): xq [B, ldx] int8 with
// ldx = K (plain rows), xscale [B, K/32] f32. Signature = f16x's with (xq, xscale) in place of x16.
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 128, 2, 4, 1, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_bn64(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
    f16x_core<128, 64, 4, 2, 1, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
}
#endif

// ---------------------------------------------------------------------------------------------
// Replay twin: b-packed dp4a GEMV for batch <= 64. Bit-identical per (row, b) to
// q8_0_gemv_batched_warp8 / q8_0_gemv_bpack_warp8: same bl = lane + 32*i striding, same
// `ws * xscale * dot` expression, same warp_sum_f32 tree. The batch loop is unrolled at a fixed
// 64 with a predicate so acc[] stays in registers (a runtime-bounded loop over acc[64] spills).
// ---------------------------------------------------------------------------------------------
#define WARP_LANES 32u
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull
__device__ __forceinline__ int32_t dp4a_sudot(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
__device__ __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    return v;
}
#define GEMV_BPACK64_MAX 64u
extern "C" __global__ void q8_0_gemv_bpack64_warp8(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t in_dim, uint32_t out_dim, uint32_t blocks, uint32_t batch)
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    if (batch > GEMV_BPACK64_MAX) return;
    const unsigned char* wr = w + (size_t)row * blocks * 34u;
    const __half*  scales = (const __half*)wr;
    const int8_t*  quants = (const int8_t*)(wr + (size_t)blocks * 2u);
    float acc[GEMV_BPACK64_MAX];
    #pragma unroll
    for (uint32_t b = 0; b < GEMV_BPACK64_MAX; ++b) acc[b] = 0.0f;
    for (uint32_t bl = lane; bl < blocks; bl += WARP_LANES) {
        const int8_t* qs = quants + (size_t)bl * 32u;
        uint4 av0, av1;
        __builtin_memcpy(&av0, qs, 16);
        __builtin_memcpy(&av1, qs + 16, 16);
        const float ws = __half2float(scales[bl]);
        #pragma unroll
        for (uint32_t b = 0; b < GEMV_BPACK64_MAX; ++b) {
            if (b >= batch) break;
            const int8_t* xqb = xq + (size_t)b * in_dim + (size_t)bl * 32u;
            uint4 bv0, bv1;
            __builtin_memcpy(&bv0, xqb, 16);
            __builtin_memcpy(&bv1, xqb + 16, 16);
            int32_t dot = 0;
            dot = dp4a_sudot((int32_t)av0.x, (int32_t)bv0.x, dot);
            dot = dp4a_sudot((int32_t)av0.y, (int32_t)bv0.y, dot);
            dot = dp4a_sudot((int32_t)av0.z, (int32_t)bv0.z, dot);
            dot = dp4a_sudot((int32_t)av0.w, (int32_t)bv0.w, dot);
            dot = dp4a_sudot((int32_t)av1.x, (int32_t)bv1.x, dot);
            dot = dp4a_sudot((int32_t)av1.y, (int32_t)bv1.y, dot);
            dot = dp4a_sudot((int32_t)av1.z, (int32_t)bv1.z, dot);
            dot = dp4a_sudot((int32_t)av1.w, (int32_t)bv1.w, dot);
            acc[b] += ws * xscale[(size_t)b * blocks + bl] * (float)dot;
        }
    }
    #pragma unroll
    for (uint32_t b = 0; b < GEMV_BPACK64_MAX; ++b) {
        if (b >= batch) break;
        float a = warp_sum_f32(acc[b]);
        if (lane == 0) out[(size_t)b * out_dim + row] = a;
    }
}

extern "C" __global__ void q8_0_grouped_gemv_bpack64(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, unsigned int group_dim, unsigned int rank,
    unsigned int blocks_per_group, unsigned int n_groups, unsigned int batch)
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;
    if (batch > GEMV_BPACK64_MAX) return;
    uint32_t g = row / rank;
    const unsigned char* wr = w + (size_t)row * (size_t)blocks_per_group * 34u;
    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)blocks_per_group * 2u);
    const uint32_t xq_stride = n_groups * group_dim;
    const uint32_t xs_stride = n_groups * blocks_per_group;
    float acc[GEMV_BPACK64_MAX];
    #pragma unroll
    for (uint32_t b = 0; b < GEMV_BPACK64_MAX; ++b) acc[b] = 0.0f;
    for (uint32_t bl = lane; bl < blocks_per_group; bl += WARP_LANES) {
        const int8_t* qs = quants + (size_t)bl * 32u;
        uint4 av0, av1;
        __builtin_memcpy(&av0, qs, 16);
        __builtin_memcpy(&av1, qs + 16, 16);
        const float ws = __half2float(scales[bl]);
        #pragma unroll
        for (uint32_t b = 0; b < GEMV_BPACK64_MAX; ++b) {
            if (b >= batch) break;
            const int8_t* xqb = xq + (size_t)b * xq_stride + (size_t)g * group_dim + (size_t)bl * 32u;
            uint4 bv0, bv1;
            __builtin_memcpy(&bv0, xqb, 16);
            __builtin_memcpy(&bv1, xqb + 16, 16);
            int32_t dot = 0;
            dot = dp4a_sudot((int32_t)av0.x, (int32_t)bv0.x, dot);
            dot = dp4a_sudot((int32_t)av0.y, (int32_t)bv0.y, dot);
            dot = dp4a_sudot((int32_t)av0.z, (int32_t)bv0.z, dot);
            dot = dp4a_sudot((int32_t)av0.w, (int32_t)bv0.w, dot);
            dot = dp4a_sudot((int32_t)av1.x, (int32_t)bv1.x, dot);
            dot = dp4a_sudot((int32_t)av1.y, (int32_t)bv1.y, dot);
            dot = dp4a_sudot((int32_t)av1.z, (int32_t)bv1.z, dot);
            dot = dp4a_sudot((int32_t)av1.w, (int32_t)bv1.w, dot);
            acc[b] += ws * xscale[(size_t)b * xs_stride + (size_t)g * blocks_per_group + bl] * (float)dot;
        }
    }
    #pragma unroll
    for (uint32_t b = 0; b < GEMV_BPACK64_MAX; ++b) {
        if (b >= batch) break;
        float a = warp_sum_f32(acc[b]);
        if (lane == 0) out[(size_t)b * (size_t)out_dim + row] = a;
    }
}

// ---- patch10: dummy instantiations to grow the module (never launched) ----
#if defined(__gfx1200__) || defined(__gfx1201__)
F16X_KERNEL(dummy_f16x_1, 128, 128, 2, 4, 2, true)
F16X_KERNEL(dummy_f16x_2, 128, 128, 2, 4, 2, false)
F16X_KERNEL(dummy_f16x_3, 128, 128, 2, 4, 3, true)
F16X_KERNEL(dummy_f16x_4, 128, 128, 2, 4, 3, false)
F16X_KERNEL(dummy_f16x_5, 128, 128, 2, 4, 4, true)
F16X_KERNEL(dummy_f16x_6, 128, 128, 2, 4, 4, false)
F16X_KERNEL(dummy_f16x_7, 128, 64, 4, 2, 2, true)
F16X_KERNEL(dummy_f16x_8, 128, 64, 4, 2, 2, false)
F16X_KERNEL(dummy_f16x_9, 128, 64, 4, 2, 3, true)
F16X_KERNEL(dummy_f16x_10, 128, 64, 4, 2, 3, false)
F16X_KERNEL(dummy_f16x_11, 128, 64, 4, 2, 4, true)
F16X_KERNEL(dummy_f16x_12, 128, 64, 4, 2, 4, false)
F16X_KERNEL(dummy_f16x_13, 64, 64, 2, 2, 2, true)
F16X_KERNEL(dummy_f16x_14, 64, 64, 2, 2, 2, false)
F16X_KERNEL(dummy_f16x_15, 64, 64, 2, 2, 3, true)
F16X_KERNEL(dummy_f16x_16, 64, 64, 2, 2, 3, false)
F16X_KERNEL(dummy_f16x_17, 64, 64, 2, 2, 4, true)
F16X_KERNEL(dummy_f16x_18, 64, 64, 2, 2, 4, false)
extern "C" __global__ void __launch_bounds__(256) dummy_i8x_19(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{ f16x_core<128, 128, 2, 4, 2, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K); }
extern "C" __global__ void __launch_bounds__(256) dummy_i8x_20(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{ f16x_core<128, 64, 4, 2, 2, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K); }
extern "C" __global__ void __launch_bounds__(256) dummy_i8x_21(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{ f16x_core<128, 128, 2, 4, 3, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K); }
extern "C" __global__ void __launch_bounds__(256) dummy_i8x_22(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{ f16x_core<128, 64, 4, 2, 3, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K); }
#endif
