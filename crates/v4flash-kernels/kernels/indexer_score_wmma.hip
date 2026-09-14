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

#include "e2m1_key_common.inc"

#ifndef N_INDEXER_HEAD
// V4-Flash has 64 index heads; V4.1 has 32 (`index_n_heads` in config.json).
// The build passes -DDEEPSTRIX_V41=1 for the few kernels that bake model dims.
// Everything below derives from this and N_INDEXER_HEAD_DIM — see ISW_M_TILES.
#ifdef DEEPSTRIX_V41
#define N_INDEXER_HEAD 32
#else
#define N_INDEXER_HEAD 64
#endif
#endif
#ifndef N_INDEXER_HEAD_DIM
#define N_INDEXER_HEAD_DIM 128
#endif

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define ISW_BLOCK    32u
#define ISW_N_TILE   16u   // comp rows per WG
// heads / 16 per WMMA m-dim: 4 at 64 heads (V4-Flash), 2 at 32 (V4.1).
#define ISW_M_TILES   (N_INDEXER_HEAD / 16u)
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

// =============================================================================
// Multi-wave batched variant (M52, 2026-06-09).
//
// The 1-wave `_batched` kernel re-stages the token's FULL Q (8192 f32→f16,
// 256 serial iterations/lane) in EVERY 128-col WG. At 96K ctx (n_comp≈24.5K,
// 192 WGs/token, B=1024) that is ~6.3 GB of redundant Q reads per launch and
// ~4× more staging instructions than WMMAs — measured 28.7 ms/launch vs a
// ~3.4 ms compute roofline.
//
// This variant:
//   - 8 waves / WG (256 threads); Q + hw staged cooperatively ONCE per WG,
//     then each wave owns its own ISW_NT_PER_WG n-tiles → Q traffic ÷8,
//     staging amortised 8×.
//   - NO K LDS staging: the B-fragment is loaded straight from global
//     (16 contiguous bytes per lane; K is small and MALL-resident). Kills
//     the per-n-tile stage loop AND any cross-wave sync — the single
//     __syncthreads after the Q stage is the only barrier, so per-wave
//     early-outs can't deadlock.
//
// Grid: (ceil(n_tiles_max / (8 waves × ISW_NT_PER_WG)), B). Block: 256.
// LDS: 16 KB Q + 256 B hw.
// =============================================================================
#define ISWMW_WAVES 8u
#define ISW_NT_PER_WG 8u   // n-tiles per wave (mw) / per WG (1-wave batched)

extern "C" __global__ __launch_bounds__(ISWMW_WAVES * ISW_BLOCK)
void indexer_score_wmma_batched_mw(
        float          * __restrict__ scores,          // [B, n_idx_stride]
        const float    * __restrict__ q,               // [B, N_INDEXER_HEAD * N_INDEXER_HEAD_DIM]
        const float    * __restrict__ head_weights,    // [B, N_INDEXER_HEAD]
        const _Float16 * __restrict__ index_comp_kv,   // [n_idx_max, N_INDEXER_HEAD_DIM]
        const unsigned int * __restrict__ n_idx_per,   // [B]
        unsigned int     n_idx_stride) {
    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;
    const unsigned int bi   = blockIdx.y;
    const unsigned int n_comp = n_idx_per[bi];

    const unsigned int cols_per_wave = ISW_NT_PER_WG * ISW_N_TILE;      // 128
    const unsigned int cols_per_wg   = ISWMW_WAVES * cols_per_wave;     // 1024
    const unsigned int c_wg0         = blockIdx.x * cols_per_wg;

    float       * __restrict__ scores_b = scores + (size_t)bi * n_idx_stride;
    const float * __restrict__ q_b      = q      + (size_t)bi * (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM);
    const float * __restrict__ hw_b     = head_weights + (size_t)bi * N_INDEXER_HEAD;

    // Whole-WG early-out: entire 1024-col chunk is past valid range.
    if (c_wg0 >= n_comp) {
        for (unsigned int i = tid; i < cols_per_wg; i += ISWMW_WAVES * ISW_BLOCK) {
            const unsigned int c = c_wg0 + i;
            if (c < n_idx_stride) {
                scores_b[c] = -3.4028235e38f;
            }
        }
        return;
    }

    __shared__ _Float16 lds_q[N_INDEXER_HEAD * N_INDEXER_HEAD_DIM];
    __shared__ float    lds_hw[N_INDEXER_HEAD];

    // Cooperative Q stage: 8192 elements over 256 threads = 32 each.
    #pragma unroll 8
    for (unsigned int i = tid; i < N_INDEXER_HEAD * N_INDEXER_HEAD_DIM;
         i += ISWMW_WAVES * ISW_BLOCK) {
        lds_q[i] = (_Float16)q_b[i];
    }
    if (tid < N_INDEXER_HEAD) {
        lds_hw[tid] = hw_b[tid];
    }
    __syncthreads();   // the ONLY barrier — waves diverge freely below

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int col_in_tile = lane & 15u;
    const unsigned int half        = lane >> 4;
    const unsigned int k_off       = half * 8u;
    const unsigned int c_wave0     = c_wg0 + wave * cols_per_wave;

    #pragma unroll 1
    for (unsigned int nt_local = 0; nt_local < ISW_NT_PER_WG; nt_local++) {
        const unsigned int c_base = c_wave0 + nt_local * ISW_N_TILE;
        if (c_base >= n_idx_stride) break;

        if (c_base >= n_comp) {
            if (lane < ISW_N_TILE) {
                const unsigned int c = c_base + lane;
                if (c < n_idx_stride) {
                    scores_b[c] = -3.4028235e38f;
                }
            }
            continue;
        }

        float8 c_acc[ISW_M_TILES];
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            c_acc[mt] = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

        // B-fragment row for this lane: comp row c_base+col_in_tile.
        const unsigned int c_row = c_base + col_in_tile;
        const bool c_row_valid = (c_row < n_comp);
        const _Float16 *k_row = index_comp_kv + (size_t)c_row * N_INDEXER_HEAD_DIM;

        #pragma unroll
        for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
            const unsigned int kbase = kt * 16u;
            // b_frag straight from global: 8 contiguous f16 = one b128.
            half8 b_frag;
            if (c_row_valid) {
                const _Float16 *b_src = k_row + kbase + k_off;
                #pragma unroll
                for (int e = 0; e < 8; e++) b_frag[e] = b_src[e];
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) b_frag[e] = (_Float16)0.0f;
            }

            #pragma unroll
            for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
                const unsigned int h0 = mt * 16u;
                half8 a_frag;
                const _Float16 *a_row = lds_q + (size_t)(h0 + col_in_tile) * N_INDEXER_HEAD_DIM
                    + kbase + k_off;
                #pragma unroll
                for (int e = 0; e < 8; e++) a_frag[e] = a_row[e];
                c_acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_frag, b_frag, c_acc[mt]);
            }
        }

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
        partial += __shfl_xor(partial, 16);
        if (lane < ISW_N_TILE) {
            const unsigned int c = c_base + lane;
            if (c < n_idx_stride) {
                scores_b[c] = (c < n_comp) ? partial : -3.4028235e38f;
            }
        }
    }
#else
    (void)scores_b; (void)hw_b; (void)n_idx_stride; (void)wave;
#endif
}

// =============================================================================
// M58: multi-wave NON-batched variant (decode path). Same fix as the M52
// batched `_mw` kernel: the 1-wave `indexer_score_wmma` re-stages the full
// 16 KB Q in every 16-col WG — at decode depth 96K (n_comp ≈ 24.5K,
// ~1534 WGs) that's ~25 MB of Q re-reads per layer per token, measured
// ~55 µs/launch ≈ 1.2 ms/token across the 21 CSA layers. 8 waves share one
// cooperative Q stage; B-fragments load straight from global.
// Grid: (ceil(n_comp / 1024), 1, 1). Block: 256. Writes scores[0, n_comp).
// =============================================================================
extern "C" __global__ __launch_bounds__(ISWMW_WAVES * ISW_BLOCK)
void indexer_score_wmma_mw(
        float          * __restrict__ scores,        // [n_comp]
        const float    * __restrict__ q,             // [N_INDEXER_HEAD, N_INDEXER_HEAD_DIM]
        const float    * __restrict__ head_weights,  // [N_INDEXER_HEAD]
        const _Float16 * __restrict__ index_comp_kv, // [n_comp, N_INDEXER_HEAD_DIM]
        unsigned int     n_comp) {
    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;

    const unsigned int cols_per_wave = ISW_NT_PER_WG * ISW_N_TILE;      // 128
    const unsigned int cols_per_wg   = ISWMW_WAVES * cols_per_wave;     // 1024
    const unsigned int c_wg0         = blockIdx.x * cols_per_wg;
    if (c_wg0 >= n_comp) return;

    __shared__ _Float16 lds_q[N_INDEXER_HEAD * N_INDEXER_HEAD_DIM];
    __shared__ float    lds_hw[N_INDEXER_HEAD];

    #pragma unroll 8
    for (unsigned int i = tid; i < N_INDEXER_HEAD * N_INDEXER_HEAD_DIM;
         i += ISWMW_WAVES * ISW_BLOCK) {
        lds_q[i] = (_Float16)q[i];
    }
    if (tid < N_INDEXER_HEAD) {
        lds_hw[tid] = head_weights[tid];
    }
    __syncthreads();   // the ONLY barrier — waves diverge freely below

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int col_in_tile = lane & 15u;
    const unsigned int half        = lane >> 4;
    const unsigned int k_off       = half * 8u;
    const unsigned int c_wave0     = c_wg0 + wave * cols_per_wave;

    #pragma unroll 1
    for (unsigned int nt_local = 0; nt_local < ISW_NT_PER_WG; nt_local++) {
        const unsigned int c_base = c_wave0 + nt_local * ISW_N_TILE;
        if (c_base >= n_comp) break;

        float8 c_acc[ISW_M_TILES];
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            c_acc[mt] = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

        const unsigned int c_row = c_base + col_in_tile;
        const bool c_row_valid = (c_row < n_comp);
        const _Float16 *k_row = index_comp_kv + (size_t)c_row * N_INDEXER_HEAD_DIM;

        #pragma unroll
        for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
            const unsigned int kbase = kt * 16u;
            half8 b_frag;
            if (c_row_valid) {
                const _Float16 *b_src = k_row + kbase + k_off;
                #pragma unroll
                for (int e = 0; e < 8; e++) b_frag[e] = b_src[e];
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) b_frag[e] = (_Float16)0.0f;
            }
            #pragma unroll
            for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
                const unsigned int h0 = mt * 16u;
                half8 a_frag;
                const _Float16 *a_row = lds_q + (size_t)(h0 + col_in_tile) * N_INDEXER_HEAD_DIM
                    + kbase + k_off;
                #pragma unroll
                for (int e = 0; e < 8; e++) a_frag[e] = a_row[e];
                c_acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_frag, b_frag, c_acc[mt]);
            }
        }

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
        partial += __shfl_xor(partial, 16);
        if (lane < ISW_N_TILE) {
            const unsigned int c = c_base + lane;
            if (c < n_comp) {
                scores[c] = partial;
            }
        }
    }
#else
    (void)scores; (void)wave;
#endif
}

// Batched variant: one launch handles B tokens. Sharing pattern:
//   - K (index_comp_kv) is shared across all B tokens at this layer
//   - Q, head_weights, scores are per-token (strided by bi)
//   - n_idx_per[bi] gives per-token comp count (varies within batch).
//
// Each WG handles ISW_NT_PER_WG consecutive n-tiles (= ISW_NT_PER_WG×16
// score cols) for ONE token. Q is staged once into LDS and reused across
// all n-tiles in the chunk. K is re-staged per n-tile (4 KB each).
// At ISW_NT_PER_WG=8 this cuts Q DRAM traffic 8× vs one-n-tile-per-WG:
// for B=512 / n_idx=8192, 8 GB → 1 GB of Q reads per layer (the prior
// design was BW-bound on Q reads at ~14 ms/layer).
//
// Grid: (ceil(n_tiles_max / ISW_NT_PER_WG), B, 1). Block: 32 (1 wave).
// (ISW_NT_PER_WG defined above, shared with the mw variant.)

extern "C" __global__ __launch_bounds__(ISW_BLOCK)
void indexer_score_wmma_batched(
        float          * __restrict__ scores,          // [B, n_idx_stride]
        const float    * __restrict__ q,               // [B, N_INDEXER_HEAD * N_INDEXER_HEAD_DIM]
        const float    * __restrict__ head_weights,    // [B, N_INDEXER_HEAD]
        const _Float16 * __restrict__ index_comp_kv,   // [n_idx_max, N_INDEXER_HEAD_DIM]
        const unsigned int * __restrict__ n_idx_per,   // [B]
        unsigned int     n_idx_stride) {
    const unsigned int lane     = threadIdx.x & 31u;
    const unsigned int chunk_x  = blockIdx.x;
    const unsigned int bi       = blockIdx.y;
    const unsigned int nt_base  = chunk_x * ISW_NT_PER_WG;
    const unsigned int c_base0  = nt_base * ISW_N_TILE;
    const unsigned int n_comp   = n_idx_per[bi];

    float       * __restrict__ scores_b = scores + (size_t)bi * n_idx_stride;
    const float * __restrict__ q_b      = q      + (size_t)bi * (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM);
    const float * __restrict__ hw_b     = head_weights + (size_t)bi * N_INDEXER_HEAD;

    // Whole-WG early-out: entire 128-col chunk is past valid range.
    if (c_base0 >= n_comp) {
        const unsigned int chunk_cols = ISW_NT_PER_WG * ISW_N_TILE;
        for (unsigned int i = lane; i < chunk_cols; i += ISW_BLOCK) {
            const unsigned int c = c_base0 + i;
            if (c < n_idx_stride) {
                scores_b[c] = -3.4028235e38f;
            }
        }
        return;
    }

    __shared__ _Float16 lds_q[N_INDEXER_HEAD * N_INDEXER_HEAD_DIM];
    __shared__ _Float16 lds_k[ISW_N_TILE * N_INDEXER_HEAD_DIM];
    __shared__ float    lds_hw[N_INDEXER_HEAD];

    // Stage Q + hw ONCE per WG (amortised across all NT_PER_WG n-tiles).
    #pragma unroll 8
    for (unsigned int i = lane; i < N_INDEXER_HEAD * N_INDEXER_HEAD_DIM; i += ISW_BLOCK) {
        lds_q[i] = (_Float16)q_b[i];
    }
    if (lane < N_INDEXER_HEAD) {
        lds_hw[lane] = hw_b[lane];
    }
    if (N_INDEXER_HEAD > ISW_BLOCK && lane < (N_INDEXER_HEAD - ISW_BLOCK)) {
        lds_hw[ISW_BLOCK + lane] = hw_b[ISW_BLOCK + lane];
    }
    __syncthreads();

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int col_in_tile = lane & 15u;
    const unsigned int half        = lane >> 4;
    const unsigned int k_off       = half * 8u;

    // Loop over n-tiles in this WG's chunk. K is re-staged per iteration
    // (4 KB DRAM read per iter); Q stays in LDS the whole time.
    #pragma unroll 1
    for (unsigned int nt_local = 0; nt_local < ISW_NT_PER_WG; nt_local++) {
        const unsigned int c_base = c_base0 + nt_local * ISW_N_TILE;
        if (c_base >= n_idx_stride) break;

        if (c_base >= n_comp) {
            // Past valid range; stamp -INF and skip work.
            if (lane < ISW_N_TILE) {
                const unsigned int c = c_base + lane;
                if (c < n_idx_stride) {
                    scores_b[c] = -3.4028235e38f;
                }
            }
            continue;
        }

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
        __syncthreads();

        float8 c_acc[ISW_M_TILES];
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            c_acc[mt] = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

        #pragma unroll
        for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
            const unsigned int kbase = kt * 16u;
            half8 b_frag;
            const _Float16 *b_row = lds_k + (size_t)col_in_tile * N_INDEXER_HEAD_DIM
                + kbase + k_off;
            #pragma unroll
            for (int e = 0; e < 8; e++) b_frag[e] = b_row[e];

            #pragma unroll
            for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
                const unsigned int h0 = mt * 16u;
                half8 a_frag;
                const _Float16 *a_row = lds_q + (size_t)(h0 + col_in_tile) * N_INDEXER_HEAD_DIM
                    + kbase + k_off;
                #pragma unroll
                for (int e = 0; e < 8; e++) a_frag[e] = a_row[e];
                c_acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_frag, b_frag, c_acc[mt]);
            }
        }

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
        partial += __shfl_xor(partial, 16);
        if (lane < ISW_N_TILE) {
            const unsigned int c = c_base + lane;
            if (c < n_idx_stride) {
                scores_b[c] = (c < n_comp) ? partial : -3.4028235e38f;
            }
        }
        __syncthreads();  // before next iteration's K stage
    }
#else
    (void)c_base0; (void)scores_b; (void)q_b; (void)hw_b; (void)n_comp;
    (void)n_idx_stride;
#endif
}

// =============================================================================
// GEMM-shaped batched variant (2026-09-08).
//
// `_batched_mw` is one TOKEN per WG: every K (comp-row) fragment is fetched
// from L2/MALL once per token per launch — 6.4 GB per launch at 192K
// (B=512, 49K rows) — and the kernel sits at 22% of the matrix peak,
// bounded by that traffic (trace 2026-09-08: 9.6 ms/launch, the largest
// kernel on either GPU at 192K).
//
// This variant turns the batch into the reuse dimension:
//   - WG = 8 waves = 8 consecutive TOKENS (one per wave). Each wave loads
//     its token's Q ONCE into registers as 32 f16 A-fragments (4 m-tiles x
//     8 k-tiles, 128 VGPRs) and keeps them for the whole kernel.
//   - K rows are staged in LDS in tiles of ISG_TILE_ROWS (64 rows x 128
//     dims f16, row stride 136 halves so the 16-lane fragment read is
//     conflict-free) ONCE per WG and consumed by all 8 tokens -> K traffic
//     / 8, and no A-fragment loads at all in the inner loop (the old kernel
//     did one LDS b128 per WMMA; this one does one per FOUR WMMAs).
//   - grid = (n_splits, ceil(B/8)): each WG owns a contiguous 1/n_splits of
//     the row range so that ~4 WGs land per CU (a single split would give
//     64 WGs for B=512). The split multiplies Q traffic (64 MB/launch at
//     n_splits=4: nothing) and leaves K traffic at (B/8) x |K| ~ 0.8 GB.
//   - Register prefetch of the next K tile one iteration ahead, as in
//     q8_0_gemm_wmma_f16x.
//   - Epilogue is IDENTICAL to _batched_mw (same k-tile order, same
//     Σ_mt Σ_e relu(c)*hw order, same shfl_xor 16) so results are
//     bit-exact with the 1-wave reference and the existing oracle applies.
//   - Tail: cols in [n_comp(token), n_idx_stride) inside this WG's range
//     are stamped -inf; whole n-tiles past the token's n_comp skip the
//     WMMAs (waves diverge freely: the only barriers are the per-tile
//     stage barriers, which every wave reaches).
// LDS: 64 x 272 B = 17 KB K tile + 2 KB hw -> 3 WGs/CU (6 waves/SIMD).
// =============================================================================
#define ISG_WAVES      8u
#define ISG_TILE_ROWS  64u
#define ISG_K_STRIDE   136u   // halves per staged K row (128 + 8 pad)

extern "C" __global__ __launch_bounds__(ISG_WAVES * ISW_BLOCK)
void indexer_score_wmma_gemm(
        float          * __restrict__ scores,          // [B, n_idx_stride]
        const _Float16 * __restrict__ q16,             // [B, 64*128] f16 (cast once by the caller)
        const float    * __restrict__ head_weights,    // [B, 64]
        const _Float16 * __restrict__ index_comp_kv,   // [n_idx_max, 128]
        const unsigned int * __restrict__ n_idx_per,   // [B]
        unsigned int     n_idx_stride,
        unsigned int     n_idx_max,
        unsigned int     batch,
        unsigned int     span) {                       // rows per split (multiple of ISG_TILE_ROWS)
#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;
    const unsigned int t0   = blockIdx.y * ISG_WAVES;
    const unsigned int bi   = t0 + wave;
    const bool tok_valid    = bi < batch;
    const unsigned int n_comp = tok_valid ? n_idx_per[bi] : 0u;

    const unsigned int r0 = blockIdx.x * span;                       // this WG's row range
    const unsigned int r1 = min(r0 + span, n_idx_stride);
    if (r0 >= r1) return;

    __shared__ __attribute__((aligned(16))) _Float16 lds_k[ISG_TILE_ROWS * ISG_K_STRIDE];
    __shared__ float lds_hw[ISG_WAVES * N_INDEXER_HEAD];

    // hw for the WG's 8 tokens (zeros for tokens past the batch).
    for (unsigned int i = tid; i < ISG_WAVES * N_INDEXER_HEAD; i += ISG_WAVES * ISW_BLOCK) {
        const unsigned int t = t0 + i / N_INDEXER_HEAD;
        lds_hw[i] = (t < batch) ? head_weights[(size_t)t * N_INDEXER_HEAD + (i % N_INDEXER_HEAD)] : 0.0f;
    }

    const unsigned int col_in_tile = lane & 15u;
    const unsigned int half        = lane >> 4;
    const unsigned int k_off       = half * 8u;

    // --- Q fragments for this wave's token, held for the whole kernel.
    //     A[m,k]: a_frag[e] = Q[mt*16 + (lane&15)][kt*16 + k_off + e]
    //     q16 is f16 already, so each fragment is one b128 straight into its
    //     final registers (an f32 source made the compiler hold 64 float4
    //     temporaries: 123 spilled VGPRs).
    half8 a_frag[ISW_M_TILES][ISW_K_TILES];
    {
        const _Float16* q_b = q16 + (size_t)(tok_valid ? bi : 0u) * (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM);
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            const _Float16* q_row = q_b + (size_t)(mt * 16u + col_in_tile) * N_INDEXER_HEAD_DIM + k_off;
            #pragma unroll
            for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
                __builtin_memcpy(&a_frag[mt][kt], q_row + kt * 16u, 16);
            }
        }
    }
    const float* hw_w = lds_hw + wave * N_INDEXER_HEAD;
    float* scores_b = scores + (size_t)(tok_valid ? bi : 0u) * n_idx_stride;

    // Compute range: rows the WG's tokens actually need; the rest is -inf.
    // (n_comp differs by <= 2 across 8 consecutive tokens.)
    unsigned int n_comp_max = n_comp;
    for (unsigned int w = 0; w < ISG_WAVES; w++) {
        const unsigned int t = t0 + w;
        if (t < batch) n_comp_max = max(n_comp_max, n_idx_per[t]);
    }
    const unsigned int c_end = min(r1, min(n_comp_max, n_idx_max));   // rows to compute in [r0, c_end)

    // Staging role: thread -> (row tid>>2, quarter tid&3 = 32 halves = 4 b128).
    const unsigned int st_row = tid >> 2;
    const unsigned int st_off = (tid & 3u) * 32u;
    uint4 pre[4];
    auto load_tile = [&](unsigned int tile_r0) {
        const unsigned int r = tile_r0 + st_row;
        if (r < n_idx_max) {
            const _Float16* src = index_comp_kv + (size_t)r * N_INDEXER_HEAD_DIM + st_off;
            #pragma unroll
            for (int j = 0; j < 4; j++) __builtin_memcpy(&pre[j], src + 8 * j, 16);
        } else {
            #pragma unroll
            for (int j = 0; j < 4; j++) pre[j] = make_uint4(0u, 0u, 0u, 0u);
        }
    };

    if (r0 < c_end) load_tile(r0);
    __syncthreads();   // lds_hw visible

    for (unsigned int tile_r0 = r0; tile_r0 < c_end; tile_r0 += ISG_TILE_ROWS) {
        // publish the prefetched tile
        {
            uint4* dst = (uint4*)(&lds_k[st_row * ISG_K_STRIDE + st_off]);
            #pragma unroll
            for (int j = 0; j < 4; j++) dst[j] = pre[j];
        }
        __syncthreads();
        if (tile_r0 + ISG_TILE_ROWS < c_end) load_tile(tile_r0 + ISG_TILE_ROWS);

        #pragma unroll 1
        for (unsigned int nt = 0; nt < ISG_TILE_ROWS / ISW_N_TILE; nt++) {
            const unsigned int c_base = tile_r0 + nt * ISW_N_TILE;
            if (c_base >= r1) break;
            if (!tok_valid) continue;
            if (c_base >= n_comp) {
                if (lane < ISW_N_TILE) {
                    const unsigned int c = c_base + lane;
                    if (c < r1) scores_b[c] = -3.4028235e38f;
                }
                continue;
            }
            float8 c_acc[ISW_M_TILES];
            #pragma unroll
            for (unsigned int mt = 0; mt < ISW_M_TILES; mt++)
                c_acc[mt] = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const _Float16* k_row = lds_k + (nt * ISW_N_TILE + col_in_tile) * ISG_K_STRIDE + k_off;
            #pragma unroll
            for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
                half8 b_frag;
                __builtin_memcpy(&b_frag, k_row + kt * 16u, 16);
                #pragma unroll
                for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
                    c_acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a_frag[mt][kt], b_frag, c_acc[mt]);
                }
            }
            float partial = 0.0f;
            #pragma unroll
            for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const unsigned int h = mt * 16u + k_off + (unsigned int)e;
                    float v = c_acc[mt][e];
                    if (v < 0.0f) v = 0.0f;
                    partial += v * hw_w[h];
                }
            }
            partial += __shfl_xor(partial, 16);
            if (lane < ISW_N_TILE) {
                const unsigned int c = c_base + lane;
                if (c < r1) scores_b[c] = (c < n_comp) ? partial : -3.4028235e38f;
            }
        }
        __syncthreads();   // lds_k reuse
    }

    // -inf for this WG's rows past the computed range (per token).
    if (tok_valid) {
        const unsigned int from = max(r0, c_end);
        for (unsigned int c = from + lane; c < r1; c += 32u) scores_b[c] = -3.4028235e38f;
        // rows in [r0, c_end) past this token's own n_comp were stamped in the loop,
        // EXCEPT when c_end < n_comp is impossible (c_end >= n_comp by construction)
        // and when the loop never ran for a token whose n_comp <= r0: stamp those too.
        if (n_comp <= r0) {
            for (unsigned int c = r0 + lane; c < min(c_end, r1); c += 32u) scores_b[c] = -3.4028235e38f;
        }
    }
#else
    (void)scores; (void)q16; (void)head_weights; (void)index_comp_kv; (void)n_idx_per;
    (void)n_idx_stride; (void)n_idx_max; (void)batch; (void)span;
#endif
}

// =============================================================================
// Packed-E2M1 key twins (2026-09-10, docs/E2M1_INDEXER_KEYS_2026-09.md).
// GENERATED from the f16 kernels above by the same text substitution the
// commit describes: only the K-row pointer type and the B-fragment / tile
// loads differ (e2m1_key_b_frag / e2m1_key_expand8 expand the nibbles to
// the identical f16 the f16 kernels read), every epilogue is byte-for-byte
// the same, so scores are bit-identical (tests/indexer_score_e2m1.rs).
// =============================================================================
extern "C" __global__ __launch_bounds__(ISWMW_WAVES * ISW_BLOCK)
void indexer_score_wmma_mw_e2m1(
        float          * __restrict__ scores,        // [n_comp]
        const float    * __restrict__ q,             // [N_INDEXER_HEAD, N_INDEXER_HEAD_DIM]
        const float    * __restrict__ head_weights,  // [N_INDEXER_HEAD]
        const unsigned char * __restrict__ index_comp_kv, // [n_comp, N_INDEXER_HEAD_DIM]
        unsigned int     n_comp) {
    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;

    const unsigned int cols_per_wave = ISW_NT_PER_WG * ISW_N_TILE;      // 128
    const unsigned int cols_per_wg   = ISWMW_WAVES * cols_per_wave;     // 1024
    const unsigned int c_wg0         = blockIdx.x * cols_per_wg;
    if (c_wg0 >= n_comp) return;

    __shared__ _Float16 lds_q[N_INDEXER_HEAD * N_INDEXER_HEAD_DIM];
    __shared__ float    lds_hw[N_INDEXER_HEAD];

    #pragma unroll 8
    for (unsigned int i = tid; i < N_INDEXER_HEAD * N_INDEXER_HEAD_DIM;
         i += ISWMW_WAVES * ISW_BLOCK) {
        lds_q[i] = (_Float16)q[i];
    }
    if (tid < N_INDEXER_HEAD) {
        lds_hw[tid] = head_weights[tid];
    }
    __syncthreads();   // the ONLY barrier — waves diverge freely below

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int col_in_tile = lane & 15u;
    const unsigned int half        = lane >> 4;
    const unsigned int k_off       = half * 8u;
    const unsigned int c_wave0     = c_wg0 + wave * cols_per_wave;

    #pragma unroll 1
    for (unsigned int nt_local = 0; nt_local < ISW_NT_PER_WG; nt_local++) {
        const unsigned int c_base = c_wave0 + nt_local * ISW_N_TILE;
        if (c_base >= n_comp) break;

        float8 c_acc[ISW_M_TILES];
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            c_acc[mt] = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

        const unsigned int c_row = c_base + col_in_tile;
        const bool c_row_valid = (c_row < n_comp);
        const unsigned char *k_row = index_comp_kv + (size_t)c_row * E2M1_KEY_ROW_BYTES;
        // All 9 loads for this lane's row up front (see e2m1_key_load_row_words).
        e2m1_key_row_words kwords;
        if (c_row_valid) {
            kwords = e2m1_key_load_row_words(k_row, k_off);
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) kwords.w[i] = 0u;
            kwords.ex4 = 0u;
        }

        #pragma unroll
        for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
            const unsigned int kbase = kt * 16u;
            half8 b_frag;
            if (c_row_valid) {
                b_frag = __builtin_bit_cast(half8, e2m1_key_b_frag_kt(kwords, kt));
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) b_frag[e] = (_Float16)0.0f;
            }
            #pragma unroll
            for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
                const unsigned int h0 = mt * 16u;
                half8 a_frag;
                const _Float16 *a_row = lds_q + (size_t)(h0 + col_in_tile) * N_INDEXER_HEAD_DIM
                    + kbase + k_off;
                #pragma unroll
                for (int e = 0; e < 8; e++) a_frag[e] = a_row[e];
                c_acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_frag, b_frag, c_acc[mt]);
            }
        }

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
        partial += __shfl_xor(partial, 16);
        if (lane < ISW_N_TILE) {
            const unsigned int c = c_base + lane;
            if (c < n_comp) {
                scores[c] = partial;
            }
        }
    }
#else
    (void)scores; (void)wave;
#endif
}

extern "C" __global__ __launch_bounds__(ISWMW_WAVES * ISW_BLOCK)
void indexer_score_wmma_batched_mw_e2m1(
        float          * __restrict__ scores,          // [B, n_idx_stride]
        const float    * __restrict__ q,               // [B, N_INDEXER_HEAD * N_INDEXER_HEAD_DIM]
        const float    * __restrict__ head_weights,    // [B, N_INDEXER_HEAD]
        const unsigned char * __restrict__ index_comp_kv,   // [n_idx_max, N_INDEXER_HEAD_DIM]
        const unsigned int * __restrict__ n_idx_per,   // [B]
        unsigned int     n_idx_stride) {
    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;
    const unsigned int bi   = blockIdx.y;
    const unsigned int n_comp = n_idx_per[bi];

    const unsigned int cols_per_wave = ISW_NT_PER_WG * ISW_N_TILE;      // 128
    const unsigned int cols_per_wg   = ISWMW_WAVES * cols_per_wave;     // 1024
    const unsigned int c_wg0         = blockIdx.x * cols_per_wg;

    float       * __restrict__ scores_b = scores + (size_t)bi * n_idx_stride;
    const float * __restrict__ q_b      = q      + (size_t)bi * (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM);
    const float * __restrict__ hw_b     = head_weights + (size_t)bi * N_INDEXER_HEAD;

    // Whole-WG early-out: entire 1024-col chunk is past valid range.
    if (c_wg0 >= n_comp) {
        for (unsigned int i = tid; i < cols_per_wg; i += ISWMW_WAVES * ISW_BLOCK) {
            const unsigned int c = c_wg0 + i;
            if (c < n_idx_stride) {
                scores_b[c] = -3.4028235e38f;
            }
        }
        return;
    }

    __shared__ _Float16 lds_q[N_INDEXER_HEAD * N_INDEXER_HEAD_DIM];
    __shared__ float    lds_hw[N_INDEXER_HEAD];

    // Cooperative Q stage: 8192 elements over 256 threads = 32 each.
    #pragma unroll 8
    for (unsigned int i = tid; i < N_INDEXER_HEAD * N_INDEXER_HEAD_DIM;
         i += ISWMW_WAVES * ISW_BLOCK) {
        lds_q[i] = (_Float16)q_b[i];
    }
    if (tid < N_INDEXER_HEAD) {
        lds_hw[tid] = hw_b[tid];
    }
    __syncthreads();   // the ONLY barrier — waves diverge freely below

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int col_in_tile = lane & 15u;
    const unsigned int half        = lane >> 4;
    const unsigned int k_off       = half * 8u;
    const unsigned int c_wave0     = c_wg0 + wave * cols_per_wave;

    #pragma unroll 1
    for (unsigned int nt_local = 0; nt_local < ISW_NT_PER_WG; nt_local++) {
        const unsigned int c_base = c_wave0 + nt_local * ISW_N_TILE;
        if (c_base >= n_idx_stride) break;

        if (c_base >= n_comp) {
            if (lane < ISW_N_TILE) {
                const unsigned int c = c_base + lane;
                if (c < n_idx_stride) {
                    scores_b[c] = -3.4028235e38f;
                }
            }
            continue;
        }

        float8 c_acc[ISW_M_TILES];
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            c_acc[mt] = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

        // B-fragment row for this lane: comp row c_base+col_in_tile.
        const unsigned int c_row = c_base + col_in_tile;
        const bool c_row_valid = (c_row < n_comp);
        const unsigned char *k_row = index_comp_kv + (size_t)c_row * E2M1_KEY_ROW_BYTES;
        // All 9 loads for this lane's row up front (see e2m1_key_load_row_words).
        e2m1_key_row_words kwords;
        if (c_row_valid) {
            kwords = e2m1_key_load_row_words(k_row, k_off);
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) kwords.w[i] = 0u;
            kwords.ex4 = 0u;
        }

        #pragma unroll
        for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
            const unsigned int kbase = kt * 16u;
            // b_frag straight from global: 8 contiguous f16 = one b128.
            half8 b_frag;
            if (c_row_valid) {
                b_frag = __builtin_bit_cast(half8, e2m1_key_b_frag_kt(kwords, kt));
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) b_frag[e] = (_Float16)0.0f;
            }

            #pragma unroll
            for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
                const unsigned int h0 = mt * 16u;
                half8 a_frag;
                const _Float16 *a_row = lds_q + (size_t)(h0 + col_in_tile) * N_INDEXER_HEAD_DIM
                    + kbase + k_off;
                #pragma unroll
                for (int e = 0; e < 8; e++) a_frag[e] = a_row[e];
                c_acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_frag, b_frag, c_acc[mt]);
            }
        }

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
        partial += __shfl_xor(partial, 16);
        if (lane < ISW_N_TILE) {
            const unsigned int c = c_base + lane;
            if (c < n_idx_stride) {
                scores_b[c] = (c < n_comp) ? partial : -3.4028235e38f;
            }
        }
    }
#else
    (void)scores_b; (void)hw_b; (void)n_idx_stride; (void)wave;
#endif
}

extern "C" __global__ __launch_bounds__(ISG_WAVES * ISW_BLOCK)
void indexer_score_wmma_gemm_e2m1(
        float          * __restrict__ scores,          // [B, n_idx_stride]
        const _Float16 * __restrict__ q16,             // [B, 64*128] f16 (cast once by the caller)
        const float    * __restrict__ head_weights,    // [B, 64]
        const unsigned char * __restrict__ index_comp_kv,   // [n_idx_max, 80 B] packed E2M1
        const unsigned int * __restrict__ n_idx_per,   // [B]
        unsigned int     n_idx_stride,
        unsigned int     n_idx_max,
        unsigned int     batch,
        unsigned int     span) {                       // rows per split (multiple of ISG_TILE_ROWS)
#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;
    const unsigned int t0   = blockIdx.y * ISG_WAVES;
    const unsigned int bi   = t0 + wave;
    const bool tok_valid    = bi < batch;
    const unsigned int n_comp = tok_valid ? n_idx_per[bi] : 0u;

    const unsigned int r0 = blockIdx.x * span;                       // this WG's row range
    const unsigned int r1 = min(r0 + span, n_idx_stride);
    if (r0 >= r1) return;

    __shared__ __attribute__((aligned(16))) _Float16 lds_k[ISG_TILE_ROWS * ISG_K_STRIDE];
    __shared__ float lds_hw[ISG_WAVES * N_INDEXER_HEAD];

    // hw for the WG's 8 tokens (zeros for tokens past the batch).
    for (unsigned int i = tid; i < ISG_WAVES * N_INDEXER_HEAD; i += ISG_WAVES * ISW_BLOCK) {
        const unsigned int t = t0 + i / N_INDEXER_HEAD;
        lds_hw[i] = (t < batch) ? head_weights[(size_t)t * N_INDEXER_HEAD + (i % N_INDEXER_HEAD)] : 0.0f;
    }

    const unsigned int col_in_tile = lane & 15u;
    const unsigned int half        = lane >> 4;
    const unsigned int k_off       = half * 8u;

    // --- Q fragments for this wave's token, held for the whole kernel.
    //     A[m,k]: a_frag[e] = Q[mt*16 + (lane&15)][kt*16 + k_off + e]
    //     q16 is f16 already, so each fragment is one b128 straight into its
    //     final registers (an f32 source made the compiler hold 64 float4
    //     temporaries: 123 spilled VGPRs).
    half8 a_frag[ISW_M_TILES][ISW_K_TILES];
    {
        const _Float16* q_b = q16 + (size_t)(tok_valid ? bi : 0u) * (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM);
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            const _Float16* q_row = q_b + (size_t)(mt * 16u + col_in_tile) * N_INDEXER_HEAD_DIM + k_off;
            #pragma unroll
            for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
                __builtin_memcpy(&a_frag[mt][kt], q_row + kt * 16u, 16);
            }
        }
    }
    const float* hw_w = lds_hw + wave * N_INDEXER_HEAD;
    float* scores_b = scores + (size_t)(tok_valid ? bi : 0u) * n_idx_stride;

    // Compute range: rows the WG's tokens actually need; the rest is -inf.
    // (n_comp differs by <= 2 across 8 consecutive tokens.)
    unsigned int n_comp_max = n_comp;
    for (unsigned int w = 0; w < ISG_WAVES; w++) {
        const unsigned int t = t0 + w;
        if (t < batch) n_comp_max = max(n_comp_max, n_idx_per[t]);
    }
    const unsigned int c_end = min(r1, min(n_comp_max, n_idx_max));   // rows to compute in [r0, c_end)

    // Staging role: thread -> (row tid>>2, quarter tid&3 = 32 halves = 4 b128).
    const unsigned int st_row = tid >> 2;
    const unsigned int st_off = (tid & 3u) * 32u;
    // Packed: the quarter (32 halves) is exactly one 32-element block = 16 B
    // of nibbles (one aligned b128) + one exponent byte. Expansion happens
    // at publish so the prefetch stays one tile ahead.
    // Packed: the quarter (32 halves) is exactly one 32-element block = 16 B
    // of nibbles (one aligned b128) + one exponent byte. Expansion happens
    // at publish (~48 VALU per thread per tile, measured +7.5% on this
    // kernel at n=76800/B=512 = ~0.5% of prefill at 300K). Expanding in the
    // shadow of the previous tile's WMMAs was tried and is WORSE (+12%): it
    // forces an early wait on the just-issued prefetch.
    uint4 pre;
    int pre_e;
    auto load_tile = [&](unsigned int tile_r0) {
        const unsigned int r = tile_r0 + st_row;
        if (r < n_idx_max) {
            const unsigned char* src = index_comp_kv + (size_t)r * E2M1_KEY_ROW_BYTES;
            __builtin_memcpy(&pre, src + (st_off >> 1), 16);
            pre_e = (int)(signed char)src[E2M1_KEY_OFF_EXP + (st_off >> 5)];
        } else {
            pre = make_uint4(0u, 0u, 0u, 0u);
            pre_e = 0;
        }
    };

    if (r0 < c_end) load_tile(r0);
    __syncthreads();   // lds_hw visible

    for (unsigned int tile_r0 = r0; tile_r0 < c_end; tile_r0 += ISG_TILE_ROWS) {
        // publish the prefetched tile (expanded here)
        {
            uint4* dst = (uint4*)(&lds_k[st_row * ISG_K_STRIDE + st_off]);
            dst[0] = e2m1_key_expand8(pre.x, pre_e);
            dst[1] = e2m1_key_expand8(pre.y, pre_e);
            dst[2] = e2m1_key_expand8(pre.z, pre_e);
            dst[3] = e2m1_key_expand8(pre.w, pre_e);
        }
        __syncthreads();
        if (tile_r0 + ISG_TILE_ROWS < c_end) load_tile(tile_r0 + ISG_TILE_ROWS);

        #pragma unroll 1
        for (unsigned int nt = 0; nt < ISG_TILE_ROWS / ISW_N_TILE; nt++) {
            const unsigned int c_base = tile_r0 + nt * ISW_N_TILE;
            if (c_base >= r1) break;
            if (!tok_valid) continue;
            if (c_base >= n_comp) {
                if (lane < ISW_N_TILE) {
                    const unsigned int c = c_base + lane;
                    if (c < r1) scores_b[c] = -3.4028235e38f;
                }
                continue;
            }
            float8 c_acc[ISW_M_TILES];
            #pragma unroll
            for (unsigned int mt = 0; mt < ISW_M_TILES; mt++)
                c_acc[mt] = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const _Float16* k_row = lds_k + (nt * ISW_N_TILE + col_in_tile) * ISG_K_STRIDE + k_off;
            #pragma unroll
            for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
                half8 b_frag;
                __builtin_memcpy(&b_frag, k_row + kt * 16u, 16);
                #pragma unroll
                for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
                    c_acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a_frag[mt][kt], b_frag, c_acc[mt]);
                }
            }
            float partial = 0.0f;
            #pragma unroll
            for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const unsigned int h = mt * 16u + k_off + (unsigned int)e;
                    float v = c_acc[mt][e];
                    if (v < 0.0f) v = 0.0f;
                    partial += v * hw_w[h];
                }
            }
            partial += __shfl_xor(partial, 16);
            if (lane < ISW_N_TILE) {
                const unsigned int c = c_base + lane;
                if (c < r1) scores_b[c] = (c < n_comp) ? partial : -3.4028235e38f;
            }
        }
        __syncthreads();   // lds_k reuse
    }

    // -inf for this WG's rows past the computed range (per token).
    if (tok_valid) {
        const unsigned int from = max(r0, c_end);
        for (unsigned int c = from + lane; c < r1; c += 32u) scores_b[c] = -3.4028235e38f;
        // rows in [r0, c_end) past this token's own n_comp were stamped in the loop,
        // EXCEPT when c_end < n_comp is impossible (c_end >= n_comp by construction)
        // and when the loop never ran for a token whose n_comp <= r0: stamp those too.
        if (n_comp <= r0) {
            for (unsigned int c = r0 + lane; c < min(c_end, r1); c += 32u) scores_b[c] = -3.4028235e38f;
        }
    }
#else
    (void)scores; (void)q16; (void)head_weights; (void)index_comp_kv; (void)n_idx_per;
    (void)n_idx_stride; (void)n_idx_max; (void)batch; (void)span;
#endif
}
