// vit.hip — DeepSeek-V4-Flash-Vision-Exp ViT + aligner kernels for the
// Strix Halo iGPU (gfx1151, RDNA3.5, wave32).
//
// Design notes (see crates/v4flash-vision/src/tower.rs for the pipeline):
//
// * GEMM: `vit_gemm` — out[n_tok, M] = x[n_tok, K](f16) · w[M, K](f16)^T
//   (+ bias[M]) with f32 accumulate, optional GELU(erf) and optional
//   in-place accumulate into an f32 residual stream. On gfx11xx it runs
//   on the RDNA3 WMMA path (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`,
//   16-half replicated A/B fragments); every other arch (gfx12 dGPU is
//   compiled only so build.rs's per-arch loop succeeds) gets a scalar
//   LDS-tiled fallback that is bit-for-bit the same math modulo summation
//   order. `vit_gemm_scalar` exports the scalar path on every arch for
//   A/B checks. Tile: BM(=tokens)=64, BN(=out rows)=64, BK=32, 128 threads.
//
//   RDNA3 wave32 WMMA fragment layout (verified on gfx1151 by
//   `vit_wmma_probe`, see tests/tower_encode.rs::wmma_layout_probe):
//     A (16x16, row-major M×K): lane l holds A[l%16][0..16]  (lanes 16..31 replicate)
//     B (16x16, K×N):           lane l holds B[0..16][l%16]
//     C/D (16x16, f32×8/lane):  lane l, element i holds D[WMMA_C_ROW(i,l)][l%16]
//
// * `vit_attention`: full bidirectional MHA, 16 heads × hd 64, online
//   softmax, f32 math, K/V tiles of 64 keys staged in LDS as f16; 4 lanes
//   per query row (16 dims each), shuffle-reduced scores.
// * Elementwise glue: `vit_rmsnorm_f16` (f32 in → f16 out), `vit_rope_split`
//   (2-D RoPE on q/k of the fused qkv GEMM output, writes f16 q/k/v),
//   `vit_swiglu_f16`, `vit_unfold` (aligner 3×3 gather, channel-major).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define VIT_WMMA_RDNA3 1
#endif

// D-fragment row for element i of lane l (RDNA3 wave32, f32 accumulator).
#ifndef WMMA_C_ROW
#define WMMA_C_ROW(i, l) (2u * (i) + ((l) >> 4))
#endif

#define GEMM_BT   64u   // tokens per tile (rows of x)
#define GEMM_BM   64u   // output features per tile (rows of w)
#define GEMM_BK   32u
#define GEMM_PITCH 40u  // halves per LDS row (BK + 8): 80 B rows, 16-B aligned
#define GEMM_THREADS 128u

#define FLAG_ACCUM 1u
#define FLAG_GELU  2u

__device__ __forceinline__ float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// Load one 64×32 f16 tile (rows `r0..r0+64` of a `[rows][K]` matrix, cols
// `k0..k0+32`) into LDS with pitch GEMM_PITCH. Rows ≥ `rows` are zero.
// K % 8 == 0 and k0 % 32 == 0 guarantee 16-B aligned uint4 loads.
__device__ __forceinline__ void gemm_load_tile(
        _Float16* __restrict__ dst, const _Float16* __restrict__ src,
        uint32_t rows, uint32_t K, uint32_t r0, uint32_t k0, uint32_t tid) {
    const uint32_t r  = tid >> 1;
    const uint32_t kh = (tid & 1u) * 16u;
    uint4 v0, v1;
    const uint32_t rg = r0 + r;
    if (rg < rows) {
        const _Float16* p = src + (size_t)rg * K + k0 + kh;
        v0 = *reinterpret_cast<const uint4*>(p);
        v1 = *reinterpret_cast<const uint4*>(p + 8);
    } else {
        v0 = make_uint4(0u, 0u, 0u, 0u);
        v1 = v0;
    }
    uint4* d = reinterpret_cast<uint4*>(dst + r * GEMM_PITCH + kh);
    d[0] = v0;
    d[1] = v1;
}

__device__ __forceinline__ void gemm_store(
        float* __restrict__ out_f32, _Float16* __restrict__ out_f16,
        const float* __restrict__ bias, uint32_t flags,
        uint32_t n, uint32_t m, uint32_t n_tok, uint32_t M, float acc) {
    if (n >= n_tok || m >= M) return;
    float v = acc;
    if (bias) v += bias[m];
    if (flags & FLAG_GELU) v = gelu_erf(v);
    const size_t o = (size_t)n * M + m;
    if (out_f16) {
        out_f16[o] = (_Float16)v;
    } else if (flags & FLAG_ACCUM) {
        out_f32[o] += v;
    } else {
        out_f32[o] = v;
    }
}

// Scalar tile compute: thread owns 4 tokens × 8 features.
__device__ __forceinline__ void gemm_scalar_tile(
        const _Float16* __restrict__ Xs, const _Float16* __restrict__ Ws,
        float acc[4][8], uint32_t tid) {
    const uint32_t tn = (tid & 15u) * 4u;
    const uint32_t tm = (tid >> 4) * 8u;
    #pragma unroll 4
    for (uint32_t k = 0; k < GEMM_BK; ++k) {
        float xv[4], wv[8];
        #pragma unroll
        for (int a = 0; a < 4; ++a) xv[a] = (float)Xs[(tn + a) * GEMM_PITCH + k];
        #pragma unroll
        for (int b = 0; b < 8; ++b) wv[b] = (float)Ws[(tm + b) * GEMM_PITCH + k];
        #pragma unroll
        for (int a = 0; a < 4; ++a)
            #pragma unroll
            for (int b = 0; b < 8; ++b) acc[a][b] += xv[a] * wv[b];
    }
}

__device__ __forceinline__ void gemm_scalar_body(
        float* __restrict__ out_f32, _Float16* __restrict__ out_f16,
        const _Float16* __restrict__ x, const _Float16* __restrict__ w,
        const float* __restrict__ bias,
        uint32_t n_tok, uint32_t K, uint32_t M, uint32_t flags,
        _Float16* Xs, _Float16* Ws) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m0 = blockIdx.x * GEMM_BM;
    const uint32_t n0 = blockIdx.y * GEMM_BT;
    float acc[4][8];
    #pragma unroll
    for (int a = 0; a < 4; ++a)
        #pragma unroll
        for (int b = 0; b < 8; ++b) acc[a][b] = 0.0f;
    for (uint32_t k0 = 0; k0 < K; k0 += GEMM_BK) {
        gemm_load_tile(Xs, x, n_tok, K, n0, k0, tid);
        gemm_load_tile(Ws, w, M, K, m0, k0, tid);
        __syncthreads();
        gemm_scalar_tile(Xs, Ws, acc, tid);
        __syncthreads();
    }
    const uint32_t tn = (tid & 15u) * 4u;
    const uint32_t tm = (tid >> 4) * 8u;
    #pragma unroll
    for (int a = 0; a < 4; ++a)
        #pragma unroll
        for (int b = 0; b < 8; ++b)
            gemm_store(out_f32, out_f16, bias, flags, n0 + tn + a, m0 + tm + b, n_tok, M, acc[a][b]);
}

// Scalar GEMM, every arch. Grid (ceil(M/64), ceil(n_tok/64)), block 128.
extern "C" __global__ __launch_bounds__(GEMM_THREADS) void vit_gemm_scalar(
        float* __restrict__ out_f32, _Float16* __restrict__ out_f16,
        const _Float16* __restrict__ x, const _Float16* __restrict__ w,
        const float* __restrict__ bias,
        uint32_t n_tok, uint32_t K, uint32_t M, uint32_t flags) {
    __shared__ __attribute__((aligned(16))) _Float16 Xs[GEMM_BT * GEMM_PITCH];
    __shared__ __attribute__((aligned(16))) _Float16 Ws[GEMM_BM * GEMM_PITCH];
    gemm_scalar_body(out_f32, out_f16, x, w, bias, n_tok, K, M, flags, Xs, Ws);
}

// Main GEMM. Grid (ceil(M/64), ceil(n_tok/64)), block 128 (4 waves, 2×2).
// Requires K % 32 == 0 (host pads), 16-B aligned x/w rows.
extern "C" __global__ __launch_bounds__(GEMM_THREADS) void vit_gemm(
        float* __restrict__ out_f32, _Float16* __restrict__ out_f16,
        const _Float16* __restrict__ x, const _Float16* __restrict__ w,
        const float* __restrict__ bias,
        uint32_t n_tok, uint32_t K, uint32_t M, uint32_t flags) {
    __shared__ __attribute__((aligned(16))) _Float16 Xs[GEMM_BT * GEMM_PITCH];
    __shared__ __attribute__((aligned(16))) _Float16 Ws[GEMM_BM * GEMM_PITCH];
#if defined(VIT_WMMA_RDNA3)
    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t wave_n0 = (wave >> 1) * 32u;   // token sub-tile
    const uint32_t wave_m0 = (wave & 1u) * 32u;   // feature sub-tile
    const uint32_t m0 = blockIdx.x * GEMM_BM;
    const uint32_t n0 = blockIdx.y * GEMM_BT;
    const uint32_t l16 = lane & 15u;

    float8 acc[2][2];
    #pragma unroll
    for (int a = 0; a < 2; ++a)
        #pragma unroll
        for (int b = 0; b < 2; ++b) acc[a][b] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    for (uint32_t k0 = 0; k0 < K; k0 += GEMM_BK) {
        gemm_load_tile(Xs, x, n_tok, K, n0, k0, tid);
        gemm_load_tile(Ws, w, M, K, m0, k0, tid);
        __syncthreads();
        #pragma unroll
        for (uint32_t ks = 0; ks < GEMM_BK; ks += 16u) {
            half16 af[2], bf[2];
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint4* p = reinterpret_cast<const uint4*>(
                    Xs + (wave_n0 + nt * 16u + l16) * GEMM_PITCH + ks);
                uint4 u0 = p[0], u1 = p[1];
                __builtin_memcpy(&af[nt], &u0, 16);
                __builtin_memcpy(reinterpret_cast<char*>(&af[nt]) + 16, &u1, 16);
            }
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const uint4* p = reinterpret_cast<const uint4*>(
                    Ws + (wave_m0 + mt * 16u + l16) * GEMM_PITCH + ks);
                uint4 u0 = p[0], u1 = p[1];
                __builtin_memcpy(&bf[mt], &u0, 16);
                __builtin_memcpy(reinterpret_cast<char*>(&bf[mt]) + 16, &u1, 16);
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt)
                    acc[nt][mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[nt], bf[mt], acc[nt][mt]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt)
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                const uint32_t n = n0 + wave_n0 + nt * 16u + WMMA_C_ROW(i, lane);
                const uint32_t m = m0 + wave_m0 + mt * 16u + l16;
                gemm_store(out_f32, out_f16, bias, flags, n, m, n_tok, M, acc[nt][mt][i]);
            }
#else
    gemm_scalar_body(out_f32, out_f16, x, w, bias, n_tok, K, M, flags, Xs, Ws);
#endif
}

// WMMA layout probe: D = A·B with A[m][0]=m, A[m][1]=1, B[0][n]=16, B[1][n]=n
// → D[m][n] = 16m + n. Stores raw lane fragments: out[lane*8 + i].
// Block 32, grid 1. On non-RDNA3 archs writes -1.
extern "C" __global__ void vit_wmma_probe(float* __restrict__ out) {
    const uint32_t lane = threadIdx.x & 31u;
#if defined(VIT_WMMA_RDNA3)
    half16 a, b;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const uint32_t r = lane & 15u;
        a[k] = (_Float16)(k == 0 ? (float)r : (k == 1 ? 1.0f : 0.0f));
        b[k] = (_Float16)(k == 0 ? 16.0f : (k == 1 ? (float)r : 0.0f));
    }
    float8 c = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[lane * 8 + i] = c[i];
#else
    for (int i = 0; i < 8; ++i) out[lane * 8 + i] = -1.0f;
#endif
}

// RMSNorm rows of `dim` f32 → f16. Grid n_tok, block 256.
extern "C" __global__ void vit_rmsnorm_f16(
        _Float16* __restrict__ out, const float* __restrict__ x,
        const float* __restrict__ weight, uint32_t dim, float eps) {
    __shared__ float red[256];
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float* xr = x + (size_t)t * dim;
    float ss = 0.0f;
    for (uint32_t i = tid; i < dim; i += 256u) { const float v = xr[i]; ss += v * v; }
    red[tid] = ss;
    __syncthreads();
    for (uint32_t s = 128u; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(red[0] / (float)dim + eps);
    _Float16* o = out + (size_t)t * dim;
    for (uint32_t i = tid; i < dim; i += 256u) o[i] = (_Float16)(xr[i] * scale * weight[i]);
}

// 2-D RoPE on q,k from the fused qkv GEMM output (f32 [n][3072], order
// q|k|v, each [16 heads][64]) → f16 q,k,v [n][16][64]. Pairs (i, i+32),
// cos/sin [n][32]. Grid n_tok, block 512 (16 heads × 32 pairs).
extern "C" __global__ void vit_rope_split(
        _Float16* __restrict__ q_out, _Float16* __restrict__ k_out, _Float16* __restrict__ v_out,
        const float* __restrict__ qkv, const float* __restrict__ cos_t, const float* __restrict__ sin_t) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = threadIdx.x >> 5;
    const uint32_t i = threadIdx.x & 31u;
    const float c = cos_t[t * 32u + i];
    const float s = sin_t[t * 32u + i];
    const size_t base = (size_t)t * 3072u + h * 64u;
    const size_t ob = ((size_t)t * 16u + h) * 64u;
    {
        const float x1 = qkv[base + i], x2 = qkv[base + 32u + i];
        q_out[ob + i] = (_Float16)(x1 * c - x2 * s);
        q_out[ob + 32u + i] = (_Float16)(x2 * c + x1 * s);
    }
    {
        const float x1 = qkv[base + 1024u + i], x2 = qkv[base + 1024u + 32u + i];
        k_out[ob + i] = (_Float16)(x1 * c - x2 * s);
        k_out[ob + 32u + i] = (_Float16)(x2 * c + x1 * s);
    }
    v_out[ob + i] = (_Float16)qkv[base + 2048u + i];
    v_out[ob + 32u + i] = (_Float16)qkv[base + 2048u + 32u + i];
}

// Bidirectional MHA. q/k/v f16 [n][16][64] -> out f16 [n][16][64].
// Grid (16 heads, ceil(n/64)), block 256: row = tid>>2, seg = tid&3 (16 dims).
//
// FA2 shape: a 64-key tile is staged in LDS, then walked in 16-key chunks.
// Per chunk the 16 scores are computed first (phase A), the running max is
// updated ONCE, and the output accumulator is rescaled ONCE (phase B),
// then the PV accumulation runs without any per-key rescale (phase C).
// The old kernel rescaled o[16] on every key; that alone was ~1/3 of its
// VALU work. Phase A uses v_dot2_f32_f16 (`__builtin_amdgcn_fdot2`): 8
// instructions instead of 16 cvt + 16 fma for the 16-dim partial dot, with
// f32 accumulation, so the score math is bit-comparable to the f32 path.
// Softmax runs in log2 space (exp2f, one transcendental per score) with
// the 1/sqrt(hd) scale folded into the f32 multiply after the dot.
#define ATT_BR 64u
#define ATT_BC 64u
#define ATT_CH 16u   // keys per softmax chunk
#define ATT_PITCH 72u
#define ATT_LOG2E 1.4426950408889634f

typedef _Float16 half2v __attribute__((ext_vector_type(2)));

extern "C" __global__ __launch_bounds__(256) void vit_attention(
        _Float16* __restrict__ out, const _Float16* __restrict__ q,
        const _Float16* __restrict__ k, const _Float16* __restrict__ v,
        uint32_t n, float scale) {
    __shared__ __attribute__((aligned(16))) _Float16 Ks[ATT_BC * ATT_PITCH];
    __shared__ __attribute__((aligned(16))) _Float16 Vs[ATT_BC * ATT_PITCH];
    const uint32_t h = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t row = tid >> 2;
    const uint32_t seg = (tid & 3u) * 16u;
    const uint32_t qi = blockIdx.y * ATT_BR + row;
    const bool valid = qi < n;
    const float sc = scale * ATT_LOG2E;

    half2v qh[8];
    float o[16];
    #pragma unroll
    for (int d = 0; d < 8; ++d) qh[d] = (half2v){(_Float16)0, (_Float16)0};
    #pragma unroll
    for (int d = 0; d < 16; ++d) o[d] = 0.0f;
    if (valid) {
        const _Float16* qp = q + ((size_t)qi * 16u + h) * 64u + seg;
        #pragma unroll
        for (int d = 0; d < 8; ++d) qh[d] = *reinterpret_cast<const half2v*>(qp + 2 * d);
    }
    float m = -INFINITY, l = 0.0f;

    for (uint32_t kt = 0; kt < n; kt += ATT_BC) {
        // Stage the K/V tile: thread -> key row tid>>2, 16 halves at `seg`.
        {
            const uint32_t kr = tid >> 2;
            const uint32_t kg = kt + kr;
            uint4 k0, k1, v0, v1;
            if (kg < n) {
                const _Float16* kp = k + ((size_t)kg * 16u + h) * 64u + seg;
                const _Float16* vp = v + ((size_t)kg * 16u + h) * 64u + seg;
                k0 = reinterpret_cast<const uint4*>(kp)[0];
                k1 = reinterpret_cast<const uint4*>(kp)[1];
                v0 = reinterpret_cast<const uint4*>(vp)[0];
                v1 = reinterpret_cast<const uint4*>(vp)[1];
            } else {
                k0 = make_uint4(0u, 0u, 0u, 0u); k1 = k0; v0 = k0; v1 = k0;
            }
            uint4* kd = reinterpret_cast<uint4*>(Ks + kr * ATT_PITCH + seg);
            uint4* vd = reinterpret_cast<uint4*>(Vs + kr * ATT_PITCH + seg);
            kd[0] = k0; kd[1] = k1; vd[0] = v0; vd[1] = v1;
        }
        __syncthreads();
        const uint32_t nk = (n - kt < ATT_BC) ? (n - kt) : ATT_BC;
        if (valid) {
            for (uint32_t c0 = 0; c0 < nk; c0 += ATT_CH) {
                const uint32_t nc = (nk - c0 < ATT_CH) ? (nk - c0) : ATT_CH;
                // ---- phase A: 16 scores, dot2 + cross-lane reduce.
                float s[ATT_CH];
                float mc = -INFINITY;
                for (uint32_t j = 0; j < nc; ++j) {
                    const _Float16* kr = Ks + (c0 + j) * ATT_PITCH + seg;
                    // 4 independent accumulators: the 8 dot2s form 2 dependent
                    // steps instead of an 8-deep chain (v_dot2acc latency is
                    // ~4 cyc, so the naive chain stalls ~32 cyc per score).
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < 8; d += 4) {
                        a0 = __builtin_amdgcn_fdot2(qh[d + 0], *reinterpret_cast<const half2v*>(kr + 2 * (d + 0)), a0, false);
                        a1 = __builtin_amdgcn_fdot2(qh[d + 1], *reinterpret_cast<const half2v*>(kr + 2 * (d + 1)), a1, false);
                        a2 = __builtin_amdgcn_fdot2(qh[d + 2], *reinterpret_cast<const half2v*>(kr + 2 * (d + 2)), a2, false);
                        a3 = __builtin_amdgcn_fdot2(qh[d + 3], *reinterpret_cast<const half2v*>(kr + 2 * (d + 3)), a3, false);
                    }
                    float acc = (a0 + a1) + (a2 + a3);
                    acc += __shfl_xor(acc, 1);
                    acc += __shfl_xor(acc, 2);
                    s[j] = acc * sc;
                    mc = fmaxf(mc, s[j]);
                }
                // ---- phase B: one running-max update + one rescale of o[].
                const float m_new = fmaxf(m, mc);
                const float corr = exp2f(m - m_new);
                l *= corr;
                #pragma unroll
                for (int d = 0; d < 16; ++d) o[d] *= corr;
                m = m_new;
                // ---- phase C: PV accumulation, no rescale.
                for (uint32_t j = 0; j < nc; ++j) {
                    const float p = exp2f(s[j] - m_new);
                    l += p;
                    const _Float16* vr = Vs + (c0 + j) * ATT_PITCH + seg;
                    #pragma unroll
                    for (int d = 0; d < 16; ++d) o[d] += p * (float)vr[d];
                }
            }
        }
        __syncthreads();
    }
    if (valid) {
        const float inv = 1.0f / l;
        _Float16* op = out + ((size_t)qi * 16u + h) * 64u + seg;
        #pragma unroll
        for (int d = 0; d < 16; ++d) op[d] = (_Float16)(o[d] * inv);
    }
}

// h[t][c] = silu(gu[t][c]) * gu[t][ffn + c], f32 in → f16 out. Grid ceil(n*ffn/256).
extern "C" __global__ void vit_swiglu_f16(
        _Float16* __restrict__ out, const float* __restrict__ gu, uint32_t n_tok, uint32_t ffn) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tok * ffn) return;
    const uint32_t t = idx / ffn, c = idx - t * ffn;
    const float g = gu[(size_t)t * 2u * ffn + c];
    const float u = gu[(size_t)t * 2u * ffn + ffn + c];
    out[idx] = (_Float16)(g / (1.0f + expf(-g)) * u);
}

// Aligner unfold: src f16 [n_h*n_w][dim] → dst f16 [n_llm_h*n_llm_w][dim*9],
// dst feature = c*9 + ky*3 + kx, zero outside the grid. Grid n_llm, block 256.
extern "C" __global__ void vit_unfold(
        _Float16* __restrict__ dst, const _Float16* __restrict__ src,
        uint32_t n_h, uint32_t n_w, uint32_t n_llm_w, uint32_t dim) {
    const uint32_t tok = blockIdx.x;
    const uint32_t ty = tok / n_llm_w, tx = tok - ty * n_llm_w;
    _Float16* d = dst + (size_t)tok * dim * 9u;
    for (uint32_t c = threadIdx.x; c < dim; c += blockDim.x) {
        #pragma unroll
        for (uint32_t ky = 0; ky < 3; ++ky)
            #pragma unroll
            for (uint32_t kx = 0; kx < 3; ++kx) {
                const uint32_t y = ty * 3u + ky, x = tx * 3u + kx;
                _Float16 v = (_Float16)0;
                if (y < n_h && x < n_w) v = src[((size_t)y * n_w + x) * dim + c];
                d[c * 9u + ky * 3u + kx] = v;
            }
    }
}
