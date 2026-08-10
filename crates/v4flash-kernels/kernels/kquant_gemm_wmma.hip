// K-quant × Q8_K WMMA GEMM:  out[B, M] = W[M, K] · X[B, K]^T
//
// Same algorithmic shape as q8_0_gemm_wmma_lds_tiled (sister kernel): the
// weight tile is DEQUANTIZED TO f16 IN LDS once per (m0, n0, k_outer) and
// then consumed by f32_16x16x16_f16 WMMA, so the quant format only affects
// the LDS-fill step. Each A element is reused BN=64 times, so even a
// scalar per-element dequant is amortized.
//
// Motivation (2026-08-10): the unsloth UD mix puts shexp gate/up + attn_q_a
// on Q5_K and shexp-down on Q6_K. The dp4a register-tiled GEMM measured
// 0.70 ms vs Q8_0 WMMA's 0.37 ms at B=512 on [4096→2048] — ~2% of the
// prefill gap. Full dequant + WMMA recovers the matrix-core path.
//
// Activations are Q8_K ([B, n_blk*292] token-major: f32 d | 256×i8 | 16×i16
// bsums). Weights fully dequantize (scale AND min), so the plain product
// sum_k Wdeq*Xdeq is exact — no separate bsums min-term.
//
// Layouts (llama.cpp block_q{4,5,6}_K):
//   Q4_K 144 B: d(h) dmin(h) scales[12] qs[128]
//   Q5_K 176 B: d(h) dmin(h) scales[12] qh[32] qs[128]
//   Q6_K 210 B: ql[128] qh[64] sc[16](i8) d(h)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define KQW_Q4_K_BYTES 144
#define KQW_Q5_K_BYTES 176
#define KQW_Q6_K_BYTES 210
#define KQW_Q8_K_BYTES 292

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float kqw_f16_to_f32(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ void kqw_scale_min_k4(
        int j, const uint8_t* scales, uint8_t* d_out, uint8_t* m_out) {
    if (j < 4) {
        *d_out = scales[j]     & 0x3F;
        *m_out = scales[j + 4] & 0x3F;
    } else {
        *d_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m_out = (scales[j + 4] >>  4)  | ((scales[j - 0] >> 6) << 4);
    }
}

// Dequantize 16 CONSECUTIVE, 16-ALIGNED weights starting at element `k` of
// a K-quant row. 16-alignment keeps every scale/min lookup constant across
// the run for all three formats.
__device__ __forceinline__ void kqw_dequant16_q4k(
        const uint8_t* row, uint32_t k, _Float16* out) {
    const uint8_t* blk = row + (size_t)(k >> 8) * KQW_Q4_K_BYTES;
    const uint32_t kk = k & 255u;                 // element in superblock
    const float d    = kqw_f16_to_f32(*(const uint16_t*)(blk + 0));
    const float dmin = kqw_f16_to_f32(*(const uint16_t*)(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs     = blk + 16;
    const uint32_t g = kk >> 6;                   // 64-elem group 0..3
    const uint32_t l = kk & 63u;                  // 0..63 within group
    const uint32_t hi = l >> 5;                   // 0 = low nibble half
    uint8_t sc, m;
    kqw_scale_min_k4((int)(2u * g + hi), scales, &sc, &m);
    const float ds = d * (float)sc;
    const float mn = dmin * (float)m;
    const uint8_t* q = qs + (size_t)g * 32u + (l & 31u);
    #pragma unroll
    for (int e = 0; e < 16; ++e) {
        const uint32_t nib = hi ? (uint32_t)(q[e] >> 4) : (uint32_t)(q[e] & 0x0F);
        out[e] = (_Float16)(ds * (float)nib - mn);
    }
}

__device__ __forceinline__ void kqw_dequant16_q5k(
        const uint8_t* row, uint32_t k, _Float16* out) {
    const uint8_t* blk = row + (size_t)(k >> 8) * KQW_Q5_K_BYTES;
    const uint32_t kk = k & 255u;
    const float d    = kqw_f16_to_f32(*(const uint16_t*)(blk + 0));
    const float dmin = kqw_f16_to_f32(*(const uint16_t*)(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qh     = blk + 16;
    const uint8_t* qs     = blk + 48;
    const uint32_t g = kk >> 6;
    const uint32_t l = kk & 63u;
    const uint32_t hi = l >> 5;
    uint8_t sc, m;
    kqw_scale_min_k4((int)(2u * g + hi), scales, &sc, &m);
    const float ds = d * (float)sc;
    const float mn = dmin * (float)m;
    const uint32_t off = l & 31u;
    const uint8_t* q = qs + (size_t)g * 32u + off;
    const uint8_t  u = (uint8_t)(1u << (2u * g + hi));
    const uint8_t* qhp = qh + off;
    #pragma unroll
    for (int e = 0; e < 16; ++e) {
        const uint32_t nib = hi ? (uint32_t)(q[e] >> 4) : (uint32_t)(q[e] & 0x0F);
        const uint32_t v   = nib + ((qhp[e] & u) ? 16u : 0u);
        out[e] = (_Float16)(ds * (float)v - mn);
    }
}

__device__ __forceinline__ void kqw_dequant16_q6k(
        const uint8_t* row, uint32_t k, _Float16* out) {
    const uint8_t* blk = row + (size_t)(k >> 8) * KQW_Q6_K_BYTES;
    const uint32_t kk = k & 255u;
    const float d = kqw_f16_to_f32(*(const uint16_t*)(blk + 208));
    // Superblock = 2 halves of 128; within a half, quarters at +0/32/64/96.
    const uint32_t half = kk >> 7;                // 0..1
    const uint32_t r    = kk & 127u;              // 0..127 within half
    const uint32_t quad = r >> 5;                 // 0..3 (which of q1..q4)
    const uint32_t l    = r & 31u;                // 0..31
    const uint8_t* ql = blk + 0   + (size_t)half * 64u;
    const uint8_t* qh = blk + 128 + (size_t)half * 32u;
    const int8_t*  sc = (const int8_t*)(blk + 192) + (size_t)half * 8u;
    const uint32_t is = l >> 4;                   // 0..1, constant over a 16-run
    const float ds = d * (float)sc[is + 2u * quad];
    const uint8_t* qlp = ql + ((quad & 1u) ? 32u : 0u) + l;  // q1,q3: +0; q2,q4: +32
    const uint8_t* qhp = qh + l;
    const uint32_t shift = 2u * quad;             // q1:0 q2:2 q3:4 q4:6
    const bool upper = quad >= 2u;                // q3,q4 use the high nibble
    #pragma unroll
    for (int e = 0; e < 16; ++e) {
        const uint32_t lo = upper ? (uint32_t)(qlp[e] >> 4) : (uint32_t)(qlp[e] & 0x0F);
        const int32_t v = (int32_t)(lo | (((uint32_t)(qhp[e] >> shift) & 3u) << 4)) - 32;
        out[e] = (_Float16)(ds * (float)v);
    }
}

// Shared kernel body, parameterized by the 16-wide dequant + row pitch.
#define KQW_GEMM_KERNEL(NAME, DEQ16, ROW_BYTES)                                  \
extern "C" __global__ void NAME(                                                 \
    float* __restrict__ out,               /* [B, M] */                          \
    const uint8_t* __restrict__ w,         /* [M, n_blk*ROW_BYTES] */            \
    const uint8_t* __restrict__ xq,        /* [B, n_blk*292] Q8_K */             \
    uint32_t in_dim,                       /* K */                               \
    uint32_t out_dim,                      /* M */                               \
    uint32_t batch,                        /* N = B */                           \
    uint32_t n_blk)                        /* K / 256 */                          \
{                                                                                \
_Pragma("clang diagnostic push")                                                 \
    if (false) { (void)in_dim; }                                                 \
    constexpr uint32_t BM = 64u, BN = 64u, BK = 32u;                             \
    constexpr uint32_t M_WARPS = 2u, N_WARPS = 2u;                               \
    constexpr uint32_t A_STRIDE = BK + 1u;                                       \
    constexpr uint32_t M_TILES_PER_WARP = (BM / M_WARPS) / 16u;                  \
    constexpr uint32_t N_TILES_PER_WARP = (BN / N_WARPS) / 16u;                  \
    __shared__ _Float16 A_tile[BM * A_STRIDE];                                   \
    __shared__ _Float16 B_tile[BK * BN];                                         \
    const uint32_t tid     = threadIdx.x;                                        \
    const uint32_t lane    = tid & 31u;                                          \
    const uint32_t wave_id = tid >> 5;                                           \
    const uint32_t wave_m  = wave_id / N_WARPS;                                  \
    const uint32_t wave_n  = wave_id % N_WARPS;                                  \
    const uint32_t wave_m0 = wave_m * (BM / M_WARPS);                            \
    const uint32_t wave_n0 = wave_n * (BN / N_WARPS);                            \
    const uint32_t m0 = blockIdx.x * BM;                                         \
    const uint32_t n0 = blockIdx.y * BN;                                         \
    if (m0 >= out_dim) return;                                                   \
    const size_t rowbytes = (size_t)n_blk * (ROW_BYTES);                         \
    const size_t xrowbytes = (size_t)n_blk * KQW_Q8_K_BYTES;                     \
    float8 acc[M_TILES_PER_WARP][N_TILES_PER_WARP];                              \
    _Pragma("unroll")                                                            \
    for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt)                           \
        _Pragma("unroll")                                                        \
        for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt)                       \
            acc[mt][nt] = (float8){0,0,0,0,0,0,0,0};                             \
    for (uint32_t k_outer = 0; k_outer < n_blk * QK_K; k_outer += BK) {          \
        {                                                                        \
            const uint32_t m_local  = tid >> 1;                                  \
            const uint32_t k_start  = (tid & 1u) * 16u;                          \
            const uint32_t m_global = m0 + m_local;                              \
            _Float16 vals[16];                                                   \
            if (m_global < out_dim) {                                            \
                DEQ16(w + (size_t)m_global * rowbytes, k_outer + k_start, vals); \
            } else {                                                             \
                _Pragma("unroll")                                                \
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)0;              \
            }                                                                    \
            _Pragma("unroll")                                                    \
            for (int e = 0; e < 16; ++e)                                         \
                A_tile[m_local * A_STRIDE + k_start + e] = vals[e];              \
        }                                                                        \
        {                                                                        \
            const uint32_t n_local  = tid >> 1;                                  \
            const uint32_t k_start  = (tid & 1u) * 16u;                          \
            const uint32_t n_global = n0 + n_local;                              \
            _Float16 vals[16];                                                   \
            if (n_global < batch) {                                              \
                const uint32_t kk = k_outer + k_start;                           \
                const uint8_t* xb = xq + (size_t)n_global * xrowbytes            \
                                  + (size_t)(kk >> 8) * KQW_Q8_K_BYTES;          \
                const float xd = *(const float*)(xb + 0);                        \
                const int8_t* q8 = (const int8_t*)(xb + 4) + (kk & 255u);        \
                _Pragma("unroll")                                                \
                for (int e = 0; e < 16; ++e)                                     \
                    vals[e] = (_Float16)(xd * (float)q8[e]);                     \
            } else {                                                             \
                _Pragma("unroll")                                                \
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)0;              \
            }                                                                    \
            _Pragma("unroll")                                                    \
            for (int e = 0; e < 16; ++e)                                         \
                B_tile[(k_start + e) * BN + n_local] = vals[e];                  \
        }                                                                        \
        __syncthreads();                                                         \
        _Pragma("unroll")                                                        \
        for (uint32_t k_inner = 0; k_inner < BK; k_inner += 16u) {               \
            half8 a_frags[M_TILES_PER_WARP];                                     \
            _Pragma("unroll")                                                    \
            for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {                 \
                const uint32_t m_in_tile = wave_m0 + mt * 16u + (lane & 15u);    \
                const uint32_t k_in_tile = k_inner + (lane >> 4) * 8u;           \
                _Pragma("unroll")                                                \
                for (int e = 0; e < 8; ++e)                                      \
                    a_frags[mt][e] = A_tile[m_in_tile * A_STRIDE + k_in_tile + e];\
            }                                                                    \
            half8 b_frags[N_TILES_PER_WARP];                                     \
            _Pragma("unroll")                                                    \
            for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {                 \
                const uint32_t kb = k_inner + (lane >> 4) * 8u;                  \
                const uint32_t n_in_tile = wave_n0 + nt * 16u + (lane & 15u);    \
                _Pragma("unroll")                                                \
                for (int e = 0; e < 8; ++e)                                      \
                    b_frags[nt][e] = B_tile[(kb + e) * BN + n_in_tile];          \
            }                                                                    \
            _Pragma("unroll")                                                    \
            for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt)                   \
                _Pragma("unroll")                                                \
                for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt)               \
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(\
                        a_frags[mt], b_frags[nt], acc[mt][nt]);                  \
        }                                                                        \
        __syncthreads();                                                         \
    }                                                                            \
    _Pragma("unroll")                                                            \
    for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {                         \
        _Pragma("unroll")                                                        \
        for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {                     \
            const uint32_t n_idx = n0 + wave_n0 + nt * 16u + (lane & 15u);       \
            if (n_idx >= batch) continue;                                        \
            _Pragma("unroll")                                                    \
            for (int e = 0; e < 8; ++e) {                                        \
                const uint32_t m_idx =                                           \
                    m0 + wave_m0 + mt * 16u + (lane >> 4) * 8u + (uint32_t)e;    \
                if (m_idx < out_dim)                                             \
                    out[(size_t)n_idx * out_dim + m_idx] = acc[mt][nt][e];       \
            }                                                                    \
        }                                                                        \
    }                                                                            \
_Pragma("clang diagnostic pop")                                                  \
}

#if defined(__gfx1200__) || defined(__gfx1201__)
KQW_GEMM_KERNEL(q4_k_gemm_wmma_lds_tiled, kqw_dequant16_q4k, KQW_Q4_K_BYTES)
KQW_GEMM_KERNEL(q5_k_gemm_wmma_lds_tiled, kqw_dequant16_q5k, KQW_Q5_K_BYTES)
KQW_GEMM_KERNEL(q6_k_gemm_wmma_lds_tiled, kqw_dequant16_q6k, KQW_Q6_K_BYTES)
#else
// Non-gfx12 stubs so the gfx1151 compile succeeds (the iGPU never
// dispatches these — dense weights are dGPU-resident).
#define KQW_STUB(NAME)                                                           \
extern "C" __global__ void NAME(                                                 \
    float* __restrict__ out, const uint8_t* __restrict__ w,                      \
    const uint8_t* __restrict__ xq, uint32_t in_dim, uint32_t out_dim,           \
    uint32_t batch, uint32_t n_blk) {                                            \
    (void)out; (void)w; (void)xq; (void)in_dim; (void)out_dim;                   \
    (void)batch; (void)n_blk;                                                    \
}
KQW_STUB(q4_k_gemm_wmma_lds_tiled)
KQW_STUB(q5_k_gemm_wmma_lds_tiled)
KQW_STUB(q6_k_gemm_wmma_lds_tiled)
#endif
