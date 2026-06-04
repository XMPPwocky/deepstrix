// indexer_score_wmma — fused WMMA + per-row reduce variant of indexer_score.
//
// Math identical to indexer_score.hip:
//   scores[c] = sum_h max(0, dot(q[h], index_comp_kv[c])) * head_weights[h]
//
// Reshapes the per-row scalar scoring as a Q × K^T GEMM that the
// RDNA4 16×16×16 f16 WMMA tile fits naturally:
//   - Q  [N_HEAD=64,  HEAD_DIM=128]   f32 source (cast to f16 at frag load)
//   - K  [n_comp,    HEAD_DIM=128]    f16 (the indexer compressor's output dtype)
//   - S  [N_HEAD=64,  n_comp]         f32 fragments, never written to DRAM
//
// Fragment shape: 64 heads = 4 m-tiles, 128 dim = 8 k-tiles, 16 cols/n-tile.
// Per WG: one n-tile (16 comp rows). 4 m-tiles × 8 k-tiles = 32 WMMA calls.
// After accumulation each lane's c_frag[mt][e] holds f32 partial sums for
// 32 (head, col) pairs (4 m-tiles × 8 e); we apply ReLU + head_weights
// scaling + sum across heads IN REGISTERS, then cross-lane reduce within
// the wave to produce 16 scalar scores per WG. The 4 MB intermediate
// scores buffer that a naïve GEMM would materialise to DRAM never exists.
//
// LDS budget per WG (~20 KB; gfx12 has 64 KB):
//   Q  [64, 128]  f16  = 16 KB   (staged once at WG start, reused 1024×)
//   K  [16, 128]  f16  =  4 KB   (staged once per WG)
//   hw [64]       f32  =   256 B (broadcast across all (head, col) pairs)
//
// Grid: (n_comp / 16, 1, 1) — one WG per 16 comp rows. Block: 32 (1 wave).
//
// Single-wave WGs let many WGs land per CU (LDS not the limit, VGPR/wave
// count gates occupancy). At n_comp=16384 the grid is 1024 WGs across 60
// CUs ≈ 17 WGs/CU.
//
// Wave-fragment layout (verified against existing WMMA kernels):
//   A [m,k]: a_frag[e] = Q[h0 + (lane&15)][kbase + (lane>>4)*8 + e]
//   B [k,n]: b_frag[e] = K[kbase + (lane>>4)*8 + e][nt*16 + (lane&15)]
//   C [m,n]: c_frag[e] → head h0+((lane>>4)*8+e), col nt*16+(lane&15)
//
// For our single-wave-handles-all-m: lane l owns 4 (m-tile) × 8 (e) = 32
// c-fragment slots, each at col=(l&15) and head ∈ {mt*16 + (l>>4)*8 + e}.
// Lanes l and l^16 share col but cover the complementary head halves —
// the final cross-lane reduce is one __shfl_xor(_, 16).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef N_INDEXER_HEAD
#define N_INDEXER_HEAD 64
#endif
#ifndef N_INDEXER_HEAD_DIM
#define N_INDEXER_HEAD_DIM 128
#endif

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define ISW_BLOCK    32u
#define ISW_N_TILE   16u   // comp rows per WG
#define ISW_M_TILES   4u   // 64 heads / 16 per WMMA m-dim
#define ISW_K_TILES   8u   // 128 dim  / 16 per WMMA k-dim

extern "C" __global__ __launch_bounds__(ISW_BLOCK)
void indexer_score_wmma(
        float          * __restrict__ scores,        // [n_comp]
        const float    * __restrict__ q,             // [N_INDEXER_HEAD, N_INDEXER_HEAD_DIM]
        const float    * __restrict__ head_weights,  // [N_INDEXER_HEAD]
        const _Float16 * __restrict__ index_comp_kv, // [n_comp, N_INDEXER_HEAD_DIM] f16
        unsigned int     n_comp) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int n_tile = blockIdx.x;
    const unsigned int c_base = n_tile * ISW_N_TILE;

    // LDS layout: Q first (16 KB), then K (4 KB), then hw (256 B).
    __shared__ _Float16 lds_q[N_INDEXER_HEAD * N_INDEXER_HEAD_DIM];
    __shared__ _Float16 lds_k[ISW_N_TILE * N_INDEXER_HEAD_DIM];
    __shared__ float    lds_hw[N_INDEXER_HEAD];

    // Stage Q from DRAM as f32 → f16. 64 × 128 = 8192 elements, 32 lanes
    // → 256 elements per lane.
    #pragma unroll 8
    for (unsigned int i = lane; i < N_INDEXER_HEAD * N_INDEXER_HEAD_DIM; i += ISW_BLOCK) {
        lds_q[i] = (_Float16)q[i];
    }
    // Stage K (this WG's 16 comp rows). At c_base + 16 > n_comp, pad with
    // zeros so out-of-range cols contribute 0 (frag.relu still 0).
    const unsigned int k_elems = ISW_N_TILE * N_INDEXER_HEAD_DIM;
    #pragma unroll 4
    for (unsigned int i = lane; i < k_elems; i += ISW_BLOCK) {
        const unsigned int local_row = i / N_INDEXER_HEAD_DIM;
        const unsigned int d         = i % N_INDEXER_HEAD_DIM;
        const unsigned int c         = c_base + local_row;
        lds_k[i] = (c < n_comp)
            ? index_comp_kv[(size_t)c * N_INDEXER_HEAD_DIM + d]
            : (_Float16)0.0f;
    }
    // Stage head_weights.
    if (lane < N_INDEXER_HEAD) {
        lds_hw[lane] = head_weights[lane];
    }
    if (N_INDEXER_HEAD > ISW_BLOCK && lane < (N_INDEXER_HEAD - ISW_BLOCK)) {
        lds_hw[ISW_BLOCK + lane] = head_weights[ISW_BLOCK + lane];
    }
    __syncthreads();

#if defined(__gfx1200__) || defined(__gfx1201__)
    // Accumulators per m-tile. Each WMMA call writes/accumulates a 16×16
    // tile into 8 f32 elements per lane.
    float8 c_acc[ISW_M_TILES];
    #pragma unroll
    for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
        c_acc[mt] = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    const unsigned int col_in_tile = lane & 15u;
    const unsigned int half        = lane >> 4;   // 0 or 1
    const unsigned int k_off       = half * 8u;   // k-half offset within tile

    #pragma unroll
    for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
        const unsigned int kbase = kt * 16u;
        // B frag (K): lane (l) reads K[kbase+k_off+e, n_tile*16 + (l&15)]
        // — i.e. for comp_row=(c_base + col_in_tile), 8 contiguous f16
        // starting at dim=kbase+k_off. Since lds_k is row-major
        // [comp_row, dim], that's 8 contiguous f16 = 16 bytes (b128 load).
        half8 b_frag;
        const _Float16 *b_row = lds_k + (size_t)col_in_tile * N_INDEXER_HEAD_DIM
            + kbase + k_off;
        #pragma unroll
        for (int e = 0; e < 8; e++) b_frag[e] = b_row[e];

        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            const unsigned int h0 = mt * 16u;
            // A frag (Q): lane reads Q[h0 + col_in_tile, kbase + k_off + e]
            // — 8 contiguous f16 = 16 bytes per lane.
            half8 a_frag;
            const _Float16 *a_row = lds_q + (size_t)(h0 + col_in_tile) * N_INDEXER_HEAD_DIM
                + kbase + k_off;
            #pragma unroll
            for (int e = 0; e < 8; e++) a_frag[e] = a_row[e];
            c_acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                a_frag, b_frag, c_acc[mt]);
        }
    }

    // Fused tail: per-(head, col) ReLU + head_weights mult, sum across
    // heads owned by this lane, then cross-lane reduce.
    // c_frag layout: c_acc[mt][e] is at (head = mt*16 + half*8 + e,
    //                                    col = c_base + col_in_tile).
    float partial = 0.0f;
    #pragma unroll
    for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int h = mt * 16u + k_off + (unsigned int)e;
            float v = c_acc[mt][e];
            if (v < 0.0f) v = 0.0f;
            partial += v * lds_hw[h];
        }
    }
    // Cross-lane reduce: lane l and lane l^16 share col=(l&15); they
    // owned the lower-8-heads and upper-8-heads halves respectively.
    partial += __shfl_xor(partial, 16);
    // Lanes 0..15 now have the final sum for col c_base+lane.
    if (lane < ISW_N_TILE) {
        const unsigned int c = c_base + lane;
        if (c < n_comp) {
            scores[c] = partial;
        }
    }
#else
    (void)scores; (void)q; (void)head_weights; (void)index_comp_kv; (void)n_comp;
    (void)lane; (void)c_base;
#endif
}
