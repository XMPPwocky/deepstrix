// Candidates for indexer_score_wmma_batched_mw_e2m1 (decode arena rows, gfx1201).
// Same signature and grid as production: grid (ceil(n_idx_max/1024), b) x 256.
// The WMMA / epilogue code is copied VERBATIM from the production kernel (same k-tile order,
// same relu*hw sum order, same shfl_xor 16), so every variant is bit-exact by construction;
// only HOW the packed key words reach the lane changes.
//
//   score_mw_pf      production strided dword loads, but the NEXT n-tile's 9 words are
//                    prefetched into registers before the current tile is computed.
//   score_mw_coal    the 16-row x 80 B n-tile is loaded as 3 contiguous b128 per lane
//                    (2.5 x 512 B, ~15 cache lines instead of ~180 line touches), staged
//                    through a per-wave 1280 B LDS buffer, one tile prefetched in registers.
//   score_mw_pf_qreg score_mw_pf + the wave's 16 Q A-fragments held in registers (no LDS
//                    A-fragment reads in the loop; 64 VGPRs).
//   score_mw_pf_coal score_mw_coal + Q A-fragments in registers.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "e2m1_key_common.inc"

#define N_INDEXER_HEAD 32
#define N_INDEXER_HEAD_DIM 128
typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));
#define ISW_BLOCK    32u
#define ISW_N_TILE   16u
#define ISW_M_TILES   (N_INDEXER_HEAD / 16u)
#define ISW_K_TILES  8u
#define ISWMW_WAVES  8u
#define ISW_NT_PER_WG 8u
#define TILE_BYTES   (ISW_N_TILE * E2M1_KEY_ROW_BYTES)   // 1280

// ---- production compute for one n-tile (verbatim math) ------------------------------------
template <bool QREG>
__device__ __forceinline__ float tile_partial(const e2m1_key_row_words& kwords, bool c_row_valid,
                                              const _Float16* __restrict__ lds_q, const float* __restrict__ lds_hw,
                                              const half8 (*a_reg)[ISW_K_TILES],
                                              unsigned col_in_tile, unsigned k_off) {
    float8 c_acc[ISW_M_TILES];
    #pragma unroll
    for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
        c_acc[mt] = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
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
            half8 a_frag;
            if (QREG) {
                a_frag = a_reg[mt][kt];
            } else {
                const unsigned int h0 = mt * 16u;
                const _Float16 *a_row = lds_q + (size_t)(h0 + col_in_tile) * N_INDEXER_HEAD_DIM + kbase + k_off;
                #pragma unroll
                for (int e = 0; e < 8; e++) a_frag[e] = a_row[e];
            }
            c_acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, c_acc[mt]);
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
    return partial;
}

__device__ __forceinline__ void zero_words(e2m1_key_row_words& r) {
    #pragma unroll
    for (int i = 0; i < 8; i++) r.w[i] = 0u;
    r.ex4 = 0u;
}

// ---- coalesced tile path -------------------------------------------------------------------
struct TileRegs { uint4 v[3]; };
// rows_valid = number of rows of this tile that are < n_comp (0..16). 16-B chunks never straddle
// rows (80 = 5 x 16). Chunks of rows past n_comp are zeroed (never read from memory).
__device__ __forceinline__ void tile_load(TileRegs& r, const unsigned char* __restrict__ tile_ptr,
                                          unsigned rows_valid, unsigned lane) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
        const unsigned off = (unsigned)j * 512u + lane * 16u;
        if (off < TILE_BYTES && (off / E2M1_KEY_ROW_BYTES) < rows_valid) {
            r.v[j] = *reinterpret_cast<const uint4*>(tile_ptr + off);
        } else {
            r.v[j] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}
__device__ __forceinline__ void tile_commit(const TileRegs& r, unsigned char* lds_tile, unsigned lane) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
        const unsigned off = (unsigned)j * 512u + lane * 16u;
        if (off < TILE_BYTES) *reinterpret_cast<uint4*>(lds_tile + off) = r.v[j];
    }
}
__device__ __forceinline__ e2m1_key_row_words tile_words(const unsigned char* lds_tile, unsigned row, unsigned k_off) {
    e2m1_key_row_words r;
    const unsigned char* p = lds_tile + row * E2M1_KEY_ROW_BYTES;
    #pragma unroll
    for (int kt = 0; kt < 8; kt++) {
        r.w[kt] = *reinterpret_cast<const unsigned int*>(p + ((kt * 16u + k_off) >> 1));
    }
    r.ex4 = *reinterpret_cast<const unsigned int*>(p + E2M1_KEY_OFF_EXP);
    return r;
}

// ---- the kernel body, parameterised --------------------------------------------------------
template <bool COAL, bool QREG>
__device__ __forceinline__ void score_body(
        float          * __restrict__ scores,
        const float    * __restrict__ q,
        const float    * __restrict__ head_weights,
        const unsigned char * __restrict__ index_comp_kv,
        const unsigned int * __restrict__ n_idx_per,
        unsigned int     n_idx_stride,
        const unsigned int * __restrict__ keys_base_per) {
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

    if (c_wg0 >= n_comp) {
        for (unsigned int i = tid; i < cols_per_wg; i += ISWMW_WAVES * ISW_BLOCK) {
            const unsigned int c = c_wg0 + i;
            if (c < n_idx_stride) scores_b[c] = -3.4028235e38f;
        }
        return;
    }

    __shared__ _Float16 lds_q[N_INDEXER_HEAD * N_INDEXER_HEAD_DIM];
    __shared__ float    lds_hw[N_INDEXER_HEAD];
    __shared__ __attribute__((aligned(16))) unsigned char lds_tiles[COAL ? ISWMW_WAVES : 1][TILE_BYTES];

    #pragma unroll 8
    for (unsigned int i = tid; i < N_INDEXER_HEAD * N_INDEXER_HEAD_DIM; i += ISWMW_WAVES * ISW_BLOCK) {
        lds_q[i] = (_Float16)q_b[i];
    }
    if (tid < N_INDEXER_HEAD) lds_hw[tid] = hw_b[tid];
    __syncthreads();

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int col_in_tile = lane & 15u;
    const unsigned int half        = lane >> 4;
    const unsigned int k_off       = half * 8u;
    const unsigned int c_wave0     = c_wg0 + wave * cols_per_wave;
    const unsigned int kbase_row   = keys_base_per ? keys_base_per[bi] : 0u;
    unsigned char* lds_tile = lds_tiles[COAL ? wave : 0];

    half8 a_reg[ISW_M_TILES][ISW_K_TILES];
    if (QREG) {
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            const _Float16* a_row = lds_q + (size_t)(mt * 16u + col_in_tile) * N_INDEXER_HEAD_DIM + k_off;
            #pragma unroll
            for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
                #pragma unroll
                for (int e = 0; e < 8; e++) a_reg[mt][kt][e] = a_row[kt * 16u + e];
            }
        }
    }

    // Number of n-tiles this wave computes (the rest are -inf stamps / breaks, as in production).
    // prefetch state: tile t+1's raw data while tile t is computed.
    e2m1_key_row_words kw_next;   // strided path
    TileRegs tr_next;             // coalesced path
    auto issue = [&](unsigned int nt) {
        const unsigned int c_base = c_wave0 + nt * ISW_N_TILE;
        if (COAL) {
            const unsigned int rows_valid = (c_base < n_comp) ? min(16u, n_comp - c_base) : 0u;
            tile_load(tr_next, index_comp_kv + (size_t)(kbase_row + c_base) * E2M1_KEY_ROW_BYTES, rows_valid, lane);
        } else {
            const unsigned int c_row = c_base + col_in_tile;
            if (c_row < n_comp) {
                kw_next = e2m1_key_load_row_words(index_comp_kv + (size_t)(kbase_row + c_row) * E2M1_KEY_ROW_BYTES, k_off);
            } else {
                zero_words(kw_next);
            }
        }
    };
    issue(0);

    #pragma unroll 1
    for (unsigned int nt_local = 0; nt_local < ISW_NT_PER_WG; nt_local++) {
        const unsigned int c_base = c_wave0 + nt_local * ISW_N_TILE;
        if (c_base >= n_idx_stride) break;
        if (c_base >= n_comp) {
            if (lane < ISW_N_TILE) {
                const unsigned int c = c_base + lane;
                if (c < n_idx_stride) scores_b[c] = -3.4028235e38f;
            }
            continue;
        }
        // take the prefetched tile, then issue the next one
        e2m1_key_row_words kwords;
        if (COAL) {
            tile_commit(tr_next, lds_tile, lane);
            kwords = tile_words(lds_tile, col_in_tile, k_off);
        } else {
            kwords = kw_next;
        }
        if (nt_local + 1u < ISW_NT_PER_WG) issue(nt_local + 1u);

        const unsigned int c_row = c_base + col_in_tile;
        const bool c_row_valid = (c_row < n_comp);
        if (!c_row_valid) zero_words(kwords);   // production: zero words for rows past n_comp

        const float partial = tile_partial<QREG>(kwords, c_row_valid, lds_q, lds_hw, a_reg, col_in_tile, k_off);
        if (lane < ISW_N_TILE) {
            const unsigned int c = c_base + lane;
            if (c < n_idx_stride) scores_b[c] = (c < n_comp) ? partial : -3.4028235e38f;
        }
    }
#else
    (void)scores_b; (void)hw_b; (void)n_idx_stride; (void)wave;
#endif
}

extern "C" __global__ __launch_bounds__(ISWMW_WAVES * ISW_BLOCK)
void score_mw_pf(float* __restrict__ scores, const float* __restrict__ q, const float* __restrict__ head_weights,
                 const unsigned char* __restrict__ index_comp_kv, const unsigned int* __restrict__ n_idx_per,
                 unsigned int n_idx_stride, const unsigned int* __restrict__ keys_base_per) {
    score_body<false, false>(scores, q, head_weights, index_comp_kv, n_idx_per, n_idx_stride, keys_base_per);
}
extern "C" __global__ __launch_bounds__(ISWMW_WAVES * ISW_BLOCK)
void score_mw_coal(float* __restrict__ scores, const float* __restrict__ q, const float* __restrict__ head_weights,
                   const unsigned char* __restrict__ index_comp_kv, const unsigned int* __restrict__ n_idx_per,
                   unsigned int n_idx_stride, const unsigned int* __restrict__ keys_base_per) {
    score_body<true, false>(scores, q, head_weights, index_comp_kv, n_idx_per, n_idx_stride, keys_base_per);
}
extern "C" __global__ __launch_bounds__(ISWMW_WAVES * ISW_BLOCK)
void score_mw_pf_qreg(float* __restrict__ scores, const float* __restrict__ q, const float* __restrict__ head_weights,
                      const unsigned char* __restrict__ index_comp_kv, const unsigned int* __restrict__ n_idx_per,
                      unsigned int n_idx_stride, const unsigned int* __restrict__ keys_base_per) {
    score_body<false, true>(scores, q, head_weights, index_comp_kv, n_idx_per, n_idx_stride, keys_base_per);
}
extern "C" __global__ __launch_bounds__(ISWMW_WAVES * ISW_BLOCK)
void score_mw_pf_coal(float* __restrict__ scores, const float* __restrict__ q, const float* __restrict__ head_weights,
                      const unsigned char* __restrict__ index_comp_kv, const unsigned int* __restrict__ n_idx_per,
                      unsigned int n_idx_stride, const unsigned int* __restrict__ keys_base_per) {
    score_body<true, true>(scores, q, head_weights, index_comp_kv, n_idx_per, n_idx_stride, keys_base_per);
}
