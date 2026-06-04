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
