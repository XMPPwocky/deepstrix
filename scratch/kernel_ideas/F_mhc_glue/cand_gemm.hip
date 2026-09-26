// cand_gemm — the prefill mHC pre-mix GEMM (out[b, m] = sum_k f16(W[m,k]) * f16(X[b,k]), M = 24,
// K = 20480, B = 512) rewritten for the narrow-M shape, BIT-IDENTICAL to f16_gemm_wmma_lds_tiled:
//
//   * every output element is still ONE f32 accumulator fed by the same sequence of 1280
//     wmma_f32_16x16x16_f16 ops over K in order 0..K, with the same A fragment (f16 W) and the
//     same B fragment (X cast f32 -> f16 with the same `(_Float16)` conversion) in the same lane
//     layout -- so which WG / wave computes a tile changes nothing;
//   * the production grid is (1, B/64) x 128 = 8 WGs, each 4 waves of which 2 (rows 32..63) do
//     nothing useful, with NO load/compute overlap: ATT shows 85% of stall on s_wait_loadcnt,
//     ~1430 cycles of pure memory wait per 32-wide K step (639 us / 478 us p10 at B=512).
//   * here: grid (1, B/NT) WGs of 256 threads, ONE n-tile of NT columns per WG (NT = 16 -> 32 WGs
//     at B=512), all 8 waves stream the X and W K-chunks (X cast to f16 at the LDS store),
//     PF chunks of global loads in flight per thread, double-buffered LDS, and the first
//     NT/16 * 2 waves own the (m-tile, n-tile) accumulator chains.
//
// Signature = production (out, w, x, in_dim, out_dim, batch). Requires in_dim % BK == 0 and
// out_dim <= 32 (mHC: 24). Grid rule: (1, ceil(batch / NT)), block 256.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float4v __attribute__((ext_vector_type(4)));

template <unsigned NT, unsigned BK, unsigned PF>
__device__ __forceinline__ void gemm_narrow_body(float* __restrict__ out, const unsigned char* __restrict__ w,
                                                 const float* __restrict__ x, uint32_t in_dim, uint32_t out_dim,
                                                 uint32_t batch) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr unsigned MT = 32;                 // padded M (2 m-tiles); rows >= out_dim are zero
    constexpr unsigned LDK = BK + 8;            // padded LDS row (halves): 16 B pad breaks the b128 bank pattern
    constexpr unsigned X_PER_STEP = NT * BK;    // f32 elements of X per K-chunk
    constexpr unsigned W_PER_STEP = MT * BK;    // f16 elements of W per K-chunk
    constexpr unsigned XV = X_PER_STEP / 256;   // f32 per thread per chunk (multiple of 4)
    constexpr unsigned WV = W_PER_STEP / 256;   // f16 per thread per chunk (multiple of 8)
    static_assert(XV % 4 == 0 && WV % 8 == 0, "chunk must split into b128 loads");
    constexpr unsigned XL = XV / 4;             // b128 X loads per thread per chunk
    constexpr unsigned WL = WV / 8;             // b128 W loads per thread per chunk
    constexpr unsigned N_CHAINS = (NT / 16) * 2;

    __shared__ __attribute__((aligned(16))) _Float16 Xs[2][NT * LDK];
    __shared__ __attribute__((aligned(16))) _Float16 Ws[2][MT * LDK];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t n0 = blockIdx.y * NT;
    const uint32_t nsteps = in_dim / BK;
    const _Float16* w16 = reinterpret_cast<const _Float16*>(w);

    // ---- this thread's slice of each K-chunk (fixed across steps)
    // X: element e = tid*4 + 256*4*i  ->  row e / BK, col e % BK  (BK multiple of 4 keeps the 4 in one row)
    // W: element e = tid*8 + 256*8*i  ->  row e / BK, col e % BK
    float4v xr[PF][XL];
    half8 wr[PF][WL];

    auto issue = [&](uint32_t step, uint32_t slot) {
        const uint32_t k0 = step * BK;
        #pragma unroll
        for (unsigned i = 0; i < XL; ++i) {
            const uint32_t e = tid * 4u + 1024u * i;
            const uint32_t r = e / BK, c = e % BK;
            const uint32_t n = n0 + r;
            if (n < batch) xr[slot][i] = *reinterpret_cast<const float4v*>(x + (size_t)n * in_dim + k0 + c);
            else xr[slot][i] = (float4v){0.f, 0.f, 0.f, 0.f};
        }
        #pragma unroll
        for (unsigned i = 0; i < WL; ++i) {
            const uint32_t e = tid * 8u + 2048u * i;
            const uint32_t r = e / BK, c = e % BK;
            if (r < out_dim) wr[slot][i] = *reinterpret_cast<const half8*>(w16 + (size_t)r * in_dim + k0 + c);
            else wr[slot][i] = (half8){0, 0, 0, 0, 0, 0, 0, 0};
        }
    };
    auto store = [&](uint32_t slot, uint32_t buf) {
        #pragma unroll
        for (unsigned i = 0; i < XL; ++i) {
            const uint32_t e = tid * 4u + 1024u * i;
            const uint32_t r = e / BK, c = e % BK;
            half4 h;
            h[0] = (_Float16)xr[slot][i][0];  // the production cast, element by element
            h[1] = (_Float16)xr[slot][i][1];
            h[2] = (_Float16)xr[slot][i][2];
            h[3] = (_Float16)xr[slot][i][3];
            *reinterpret_cast<half4*>(&Xs[buf][r * LDK + c]) = h;
        }
        #pragma unroll
        for (unsigned i = 0; i < WL; ++i) {
            const uint32_t e = tid * 8u + 2048u * i;
            const uint32_t r = e / BK, c = e % BK;
            *reinterpret_cast<half8*>(&Ws[buf][r * LDK + c]) = wr[slot][i];
        }
    };

    // ---- accumulator chain of this wave (waves >= N_CHAINS only load)
    const bool compute = wave < N_CHAINS;
    const uint32_t mt = wave & 1u;          // m-tile 0 / 1
    const uint32_t nt = wave >> 1;          // n-tile within the WG
    float8 acc = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t frag_row_a = mt * 16u + (lane & 15u);
    const uint32_t frag_row_b = nt * 16u + (lane & 15u);
    const uint32_t frag_k = (lane >> 4) * 8u;

    // prologue: PF-1 chunks in flight
    #pragma unroll
    for (unsigned j = 0; j + 1 < PF; ++j) if (j < nsteps) issue(j, j);

    for (uint32_t s0 = 0; s0 < nsteps; s0 += PF) {
        #pragma unroll
        for (unsigned j = 0; j < PF; ++j) {
            const uint32_t step = s0 + j;
            if (step >= nsteps) break;
            const uint32_t nxt = step + PF - 1;
            if (nxt < nsteps) issue(nxt, (j + PF - 1) % PF);
            const uint32_t buf = step & 1u;
            store(j, buf);
            __syncthreads();
            if (compute) {
                #pragma unroll
                for (unsigned kk = 0; kk < BK; kk += 16) {
                    const half8 a = *reinterpret_cast<const half8*>(&Ws[buf][frag_row_a * LDK + kk + frag_k]);
                    const half8 b = *reinterpret_cast<const half8*>(&Xs[buf][frag_row_b * LDK + kk + frag_k]);
                    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
                }
            }
        }
    }
    if (compute) {
        const uint32_t n_idx = n0 + nt * 16u + (lane & 15u);
        if (n_idx < batch) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t m_idx = mt * 16u + (lane >> 4) * 8u + (uint32_t)e;
                if (m_idx < out_dim) out[(size_t)n_idx * out_dim + m_idx] = acc[e];
            }
        }
    }
#else
    (void)out; (void)w; (void)x; (void)in_dim; (void)out_dim; (void)batch;
#endif
}

#define GEMM_VARIANT(name, NT, BK, PF)                                                                   \
    extern "C" __global__ __launch_bounds__(256) void name(float* __restrict__ out,                     \
                                                          const unsigned char* __restrict__ w,          \
                                                          const float* __restrict__ x, uint32_t in_dim, \
                                                          uint32_t out_dim, uint32_t batch) {           \
        if (out_dim > 32u || (in_dim % BK) != 0u) return;                                               \
        gemm_narrow_body<NT, BK, PF>(out, w, x, in_dim, out_dim, batch);                                \
    }

GEMM_VARIANT(f16_gemm_narrow_n16_bk64_pf4, 16, 64, 4)
GEMM_VARIANT(f16_gemm_narrow_n16_bk128_pf3, 16, 128, 3)
GEMM_VARIANT(f16_gemm_narrow_n16_bk128_pf2, 16, 128, 2)
GEMM_VARIANT(f16_gemm_narrow_n16_bk64_pf6, 16, 64, 6)
GEMM_VARIANT(f16_gemm_narrow_n32_bk64_pf4, 32, 64, 4)
GEMM_VARIANT(f16_gemm_narrow_n32_bk128_pf2, 32, 128, 2)
