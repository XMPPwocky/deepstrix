// WMMA wsum de-risk prototype. Computes attention Phase B as a GEMM:
//   O[M,N] = inv[m] * sum_k W[m,k] * V[k,N]
// where M = n_head, N = head_dim (512), K = n_keys, W = post-softmax weights,
// V = the shared MLA latent. The MLA shared-V reuse falls out for free: V is
// the WMMA B-operand, loaded once per k-tile and reused across all head rows.
//
// Two kernels here so we can A/B the GEMM in isolation (no softmax):
//   wsum_wmma_f16  — RDNA4 16x16x16 f16 WMMA, f32 accumulate.
//   wsum_f32_ref   — faithful copy of attention_mixed_softmax_wsum_batched_htiled
//                    Phase B (the tuned f32 path), to compare against.
//
// RDNA4 (gfx12) wave32 16x16x16 fragment layout (8 halfs/lane, no RDNA3
// replication). Derived from the GPUOpen WMMA convention, halved for gfx12;
// validated empirically by the correctness test against a host f32 reference.
//   A[m,k]: lane l holds a_frag[e] = A[(l&15)][ (l>>4)*8 + e ]   e in 0..7
//   B[k,n]: lane l holds b_frag[e] = B[ (l>>4)*8 + e ][(l&15)]   e in 0..7
//   C[m,n]: lane l holds c_frag[e] = C[ e*2 + (l>>4) ][(l&15)]   e in 0..7

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#if !(defined(__gfx1200__) || defined(__gfx1201__))
// Non-RDNA4 archs: stub so the file still compiles for gfx1151 etc. The
// prototype only runs on the dGPU (gfx1201).
#endif

// KCHUNK independent accumulators break the K-reduction dependency chain so
// the warp can keep multiple WMMAs in flight (latency hiding).
#define WSUM_KCHUNK 4

extern "C" __global__ void wsum_wmma_f16(
        float          * __restrict__ out,   // [M, N] f32
        const _Float16 * __restrict__ W,     // [M, K] f16
        const _Float16 * __restrict__ V,     // [K, N] f16
        const float    * __restrict__ inv,   // [M] f32 per-row scale (1.0 = raw GEMM)
        unsigned int M, unsigned int N, unsigned int K) {
    const unsigned int nt   = blockIdx.x;          // n-tile
    const unsigned int mt   = blockIdx.y;          // m-tile
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int m0   = mt * 16u;
    const unsigned int n0   = nt * 16u;
    if (m0 >= M || n0 >= N) return;

    const unsigned int arow = lane & 15u;          // A row / B col index
    const unsigned int ko   = (lane >> 4) * 8u;    // K half: lanes 0..15 -> 0, 16..31 -> 8

#if defined(__gfx1200__) || defined(__gfx1201__)
    float8 acc[WSUM_KCHUNK];
    #pragma unroll
    for (int j = 0; j < WSUM_KCHUNK; j++)
        acc[j] = (float8){0,0,0,0,0,0,0,0};

    unsigned int k = 0;
    for (; k + 16u * WSUM_KCHUNK <= K; k += 16u * WSUM_KCHUNK) {
        #pragma unroll
        for (int j = 0; j < WSUM_KCHUNK; j++) {
            const unsigned int kk = k + (unsigned int)j * 16u;
            half8 a, b;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                a[e] = W[(size_t)(m0 + arow) * K + kk + ko + e];
                b[e] = V[(size_t)(kk + ko + e) * N + n0 + arow];
            }
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[j]);
        }
    }
    // whole-tile tail
    for (; k + 16u <= K; k += 16u) {
        half8 a, b;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            a[e] = W[(size_t)(m0 + arow) * K + k + ko + e];
            b[e] = V[(size_t)(k + ko + e) * N + n0 + arow];
        }
        acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[0]);
    }
    // partial-tile tail (K not a multiple of 16): zero-pad
    if (k < K) {
        half8 a = {0,0,0,0,0,0,0,0}, b = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int kk = k + ko + (unsigned int)e;
            if (kk < K) {
                a[e] = W[(size_t)(m0 + arow) * K + kk];
                b[e] = V[(size_t)kk * N + n0 + arow];
            }
        }
        acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[0]);
    }
    #pragma unroll
    for (int j = 1; j < WSUM_KCHUNK; j++)
        #pragma unroll
        for (int e = 0; e < 8; e++) acc[0][e] += acc[j][e];

    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned int r  = (lane >> 4) * 8u + (unsigned int)e;
        const unsigned int gm = m0 + r;
        const unsigned int gn = n0 + arow;
        if (gm < M && gn < N) {
            float v = acc[0][e];
            if (inv) v *= inv[gm];
            out[(size_t)gm * N + gn] = v;
        }
    }
#else
    (void)out; (void)W; (void)V; (void)inv; (void)arow; (void)ko;
#endif
}

// One-shot layout calibration. A[m][k] = m*100+k (assuming A-row=lane&15,
// k=(lane>>4)*8+e), B = identity (B[k][n]=1 iff k==n, B-col=lane&15). Then
// C[m][n] = A[m][n] = m*100+n. Dump raw c_frag[e] per lane so the host can
// decode which (m,n) each (lane,e) accumulator slot maps to.
extern "C" __global__ void wmma_layout_probe(float * __restrict__ raw_out /* [32*8] */) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int arow = lane & 15u;
    const unsigned int ko   = (lane >> 4) * 8u;
    half8 a, b;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned int k = ko + (unsigned int)e;
        a[e] = (_Float16)(float)(arow * 100u + k);
        b[e] = (_Float16)((k == arow) ? 1.0f : 0.0f);
    }
    float8 c = {0,0,0,0,0,0,0,0};
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    #pragma unroll
    for (int e = 0; e < 8; e++) raw_out[lane * 8 + e] = c[e];
#else
    (void)raw_out;
#endif
}

// Faithful f32 baseline: same math, mirrors the tuned htiled Phase B
// (thread owns dim d, 16-head group, V loaded once and reused across heads).
#define WSUM_REF_HT 16
#define WSUM_REF_U  2
extern "C" __global__ void wsum_f32_ref(
        float       * __restrict__ out,   // [M, N]
        const float * __restrict__ W,     // [M, K]
        const float * __restrict__ V,     // [K, N]
        const float * __restrict__ inv,   // [M]
        unsigned int M, unsigned int N, unsigned int K) {
    const unsigned int hg    = blockIdx.x;
    const unsigned int h0    = hg * WSUM_REF_HT;
    if (h0 >= M) return;
    unsigned int ht = WSUM_REF_HT;
    if (h0 + ht > M) ht = M - h0;
    const unsigned int tid   = threadIdx.x;
    const unsigned int block = blockDim.x;

    for (unsigned int d = tid; d < N; d += block) {
        constexpr unsigned int U = WSUM_REF_U;
        float acc[WSUM_REF_HT];
        #pragma unroll
        for (int hh = 0; hh < WSUM_REF_HT; hh++) acc[hh] = 0.0f;

        unsigned int k = 0;
        for (; k + U <= K; k += U) {
            float vv[U];
            #pragma unroll
            for (unsigned int u = 0; u < U; u++) vv[u] = V[(size_t)(k + u) * N + d];
            #pragma unroll
            for (int hh = 0; hh < WSUM_REF_HT; hh++) {
                const float *wh = W + (size_t)(h0 + hh) * K;
                #pragma unroll
                for (unsigned int u = 0; u < U; u++) acc[hh] += wh[k + u] * vv[u];
            }
        }
        for (; k < K; k++) {
            float v = V[(size_t)k * N + d];
            #pragma unroll
            for (int hh = 0; hh < WSUM_REF_HT; hh++)
                acc[hh] += W[(size_t)(h0 + hh) * K + k] * v;
        }
        #pragma unroll
        for (int hh = 0; hh < WSUM_REF_HT; hh++)
            if ((unsigned int)hh < ht) out[(size_t)(h0 + hh) * N + d] = acc[hh] * inv[h0 + hh];
    }
}
