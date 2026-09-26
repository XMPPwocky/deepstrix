// f16_gemm_wmma — LDS-tiled WMMA GEMM for f16 weights × f32 activations.
//
// Computes: out[b, m] = sum_k f16(weight[m, k]) * x[b, k]
//   weight: f16 row-major [M, K] (interpreted from u8 buffer)
//   x:      f32 row-major [B, K]
//   out:    f32 row-major [B, M]
//
// Same algorithmic shape as q8_0_gemm_wmma_lds_tiled (sister kernel),
// minus the q8 dequant: A_tile staged as f16 directly from weight, B_tile
// cast from f32→f16 at LDS load time.
//
// Tile shape (matches q8):
//   BM=64, BN=64, BK=32; WG = 4 warps × wave32 = 128 threads
//   2×2 wave grid → each warp owns 2×2 = 4 WMMA accumulators
//   LDS: 64×33 A f16 + 32×64 B f16 = ~8.2 KB (well under 64 KB)
//
// Designed for the V4-Flash indexer matvec_q at B=512 × M=8192 × K=1024
// (replaces f16_matvec_batched which was 7.4 ms/layer = 156 ms/chunk
// total at 21 ratio==4 layers — weight-BW-bound at the matvec-per-batch
// shape). At GEMM-with-LDS shape, weight loaded once per (m0,n0) tile,
// reused across the BN=64 output cols.
//
// Grid: (ceil(M/BM), ceil(N/BN)); block: 128.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

extern "C" __global__ void f16_gemm_wmma_lds_tiled(
    float* __restrict__ out,               // [B, M]
    const unsigned char* __restrict__ w,   // [M, K] f16 bytes
    const float* __restrict__ x,           // [B, K] f32
    uint32_t in_dim,                       // K
    uint32_t out_dim,                      // M
    uint32_t batch)                        // N = B
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr uint32_t BM = 64u, BN = 64u, BK = 32u;
    constexpr uint32_t M_WARPS = 2u, N_WARPS = 2u;
    constexpr uint32_t A_STRIDE = BK + 1u;            // 33: odd, breaks pow2 bank pattern
    constexpr uint32_t M_TILES_PER_WARP = (BM / M_WARPS) / 16u;   // 2
    constexpr uint32_t N_TILES_PER_WARP = (BN / N_WARPS) / 16u;   // 2

    __shared__ _Float16 A_tile[BM * A_STRIDE];        // [m][k_padded]
    __shared__ _Float16 B_tile[BK * BN];              // [k][n]

    const uint32_t tid     = threadIdx.x;
    const uint32_t lane    = tid & 31u;
    const uint32_t wave_id = tid >> 5;                // 0..3
    const uint32_t wave_m  = wave_id / N_WARPS;
    const uint32_t wave_n  = wave_id % N_WARPS;
    const uint32_t wave_m0 = wave_m * (BM / M_WARPS); // 0 or 32
    const uint32_t wave_n0 = wave_n * (BN / N_WARPS); // 0 or 32

    const uint32_t m0 = blockIdx.x * BM;
    const uint32_t n0 = blockIdx.y * BN;
    if (m0 >= out_dim) return;

    const _Float16* w_f16 = reinterpret_cast<const _Float16*>(w);

    float8 acc[M_TILES_PER_WARP][N_TILES_PER_WARP];
    #pragma unroll
    for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt)
            acc[mt][nt] = (float8){0,0,0,0,0,0,0,0};

    // === K-outer loop ===
    for (uint32_t k_outer = 0; k_outer < in_dim; k_outer += BK) {
        // --- Cooperative load A_tile[BM][BK]: BM*BK = 2048 elements,
        // 128 threads → 16 elements/thread. tid = m_local*2 + half.
        {
            const uint32_t m_local  = tid >> 1;
            const uint32_t k_start  = (tid & 1u) * 16u;
            const uint32_t m_global = m0 + m_local;
            _Float16 vals[16];
            if (m_global < out_dim) {
                const _Float16* w_row = w_f16 + (size_t)m_global * in_dim + k_outer + k_start;
                #pragma unroll
                for (int e = 0; e < 16; ++e) vals[e] = w_row[e];
            } else {
                #pragma unroll
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)0;
            }
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                A_tile[m_local * A_STRIDE + k_start + e] = vals[e];
            }
        }

        // --- Cooperative load B_tile[BK][BN]: cast f32→f16 at the boundary.
        // BK*BN = 2048 elements, 128 threads → 16 elements/thread.
        // tid = n_local*2 + half (half picks low/high 16 of K).
        {
            const uint32_t n_local  = tid >> 1;
            const uint32_t k_start  = (tid & 1u) * 16u;
            const uint32_t n_global = n0 + n_local;
            _Float16 vals[16];
            if (n_global < batch) {
                const float* x_row = x + (size_t)n_global * in_dim + k_outer + k_start;
                #pragma unroll
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)x_row[e];
            } else {
                #pragma unroll
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)0;
            }
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                B_tile[(k_start + e) * BN + n_local] = vals[e];
            }
        }

        __syncthreads();

        // --- Inner K loop: BK/16 = 2 WMMA K-tiles ---
        #pragma unroll
        for (uint32_t k_inner = 0; k_inner < BK; k_inner += 16u) {
            half8 a_frags[M_TILES_PER_WARP];
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {
                const uint32_t m_in_tile = wave_m0 + mt * 16u + (lane & 15u);
                const uint32_t k_in_tile = k_inner + (lane >> 4) * 8u;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    a_frags[mt][e] = A_tile[m_in_tile * A_STRIDE + k_in_tile + e];
                }
            }

            half8 b_frags[N_TILES_PER_WARP];
            #pragma unroll
            for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {
                const uint32_t k_in_tile_base = k_inner + (lane >> 4) * 8u;
                const uint32_t n_in_tile      = wave_n0 + nt * 16u + (lane & 15u);
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    b_frags[nt][e] = B_tile[(k_in_tile_base + e) * BN + n_in_tile];
                }
            }

            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {
                #pragma unroll
                for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a_frags[mt], b_frags[nt], acc[mt][nt]);
                }
            }
        }

        __syncthreads();
    }

    // --- Write C accumulators ---
    //   C[m,n]: lane l holds c[e] = C[(l>>4)*8 + e][l&15]
    #pragma unroll
    for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {
        #pragma unroll
        for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {
            const uint32_t n_idx = n0 + wave_n0 + nt * 16u + (lane & 15u);
            if (n_idx >= batch) continue;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t m_idx = m0 + wave_m0 + mt * 16u + (lane >> 4) * 8u + (uint32_t)e;
                if (m_idx < out_dim) {
                    out[(size_t)n_idx * out_dim + m_idx] = acc[mt][nt][e];
                }
            }
        }
    }
#else
    (void)out; (void)w; (void)x; (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// f16_gemm_narrow_n16_bk128_pf2 — the prefill mHC pre-mix GEMM (out[b, m] =
// sum_k f16(W[m,k]) * f16(X[b,k]), M = HC_MIX_DIM = 24, K = HC_DIM = 20480)
// re-tiled for narrow M, BIT-IDENTICAL to f16_gemm_wmma_lds_tiled.
// 2026-09-26 sweep (F_mhc_glue/gemm_narrow_m24): B=512 476 -> 80 us flushed
// (x5.4-6.0 reviewer, 3 regimes), ~7x at B=65..128; bit-exact at B = 1..1024.
//
//   * every output element is still ONE f32 accumulator fed by the same
//     sequence of K/16 wmma_f32_16x16x16_f16 ops over K in order 0..K, with the
//     same A fragment (f16 W) and the same B fragment (X cast f32 -> f16 with
//     the same `(_Float16)` conversion) in the same lane layout -- so which WG /
//     wave computes a tile changes nothing;
//   * lds_tiled at M=24 is grid (1, B/64) x 128 = 8 WGs at B=512, half of each
//     WG's waves idle (rows 32..63), no load/compute overlap (ATT: 85% of the
//     stall on s_wait_loadcnt, ~1430 cycles per 32-wide K step);
//   * here: grid (1, ceil(B/16)) WGs of 256 threads, ONE 16-column n-tile per
//     WG (32 WGs at B=512), all 8 waves stream the X and W K-chunks (BK=128,
//     b128 loads, X cast to f16 at the LDS store), PF=2 chunks of global loads
//     in flight, double-buffered LDS, waves 0-1 own the two (m-tile) chains.
//
// Signature = f16_gemm_wmma_lds_tiled (out, w, x, in_dim, out_dim, batch).
// Requires out_dim <= 32, in_dim % 128 == 0 and 16-byte aligned x / w (b128
// loads); it RETURNS WITHOUT WRITING otherwise, so the Rust wrapper
// (F16Matvec::gemm_batched_wmma, V41_MHC_GEMM_NARROW) gates on those shapes.
// gfx1151 compiles a no-op body, like the kernel above.
typedef _Float16 gn_half4 __attribute__((ext_vector_type(4)));
typedef float gn_float4 __attribute__((ext_vector_type(4)));

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
    gn_float4 xr[PF][XL];
    half8 wr[PF][WL];

    auto issue = [&](uint32_t step, uint32_t slot) {
        const uint32_t k0 = step * BK;
        #pragma unroll
        for (unsigned i = 0; i < XL; ++i) {
            const uint32_t e = tid * 4u + 1024u * i;
            const uint32_t r = e / BK, c = e % BK;
            const uint32_t n = n0 + r;
            if (n < batch) xr[slot][i] = *reinterpret_cast<const gn_float4*>(x + (size_t)n * in_dim + k0 + c);
            else xr[slot][i] = (gn_float4){0.f, 0.f, 0.f, 0.f};
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
            gn_half4 h;
            h[0] = (_Float16)xr[slot][i][0];  // the production cast, element by element
            h[1] = (_Float16)xr[slot][i][1];
            h[2] = (_Float16)xr[slot][i][2];
            h[3] = (_Float16)xr[slot][i][3];
            *reinterpret_cast<gn_half4*>(&Xs[buf][r * LDK + c]) = h;
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

extern "C" __global__ __launch_bounds__(256) void f16_gemm_narrow_n16_bk128_pf2(
        float* __restrict__ out, const unsigned char* __restrict__ w, const float* __restrict__ x,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    if (out_dim > 32u || (in_dim % 128u) != 0u) return;
    gemm_narrow_body<16, 128, 2>(out, w, x, in_dim, out_dim, batch);
}
