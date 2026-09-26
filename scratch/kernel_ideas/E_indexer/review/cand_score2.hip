// Score v2 candidates for indexer_score_wmma_batched_mw_e2m1 (gfx1201), building on the measured
// win of cand_score.hip::score_mw_pf_qreg (Q A-fragments in VGPRs: 274 -> 143 us at 235K/b=4).
// Same signature as production; grid (ceil(n_idx_max / cols_per_wg), b) x (WAVES*32), where
// cols_per_wg = WAVES * NT * 16 (production: WAVES=8, NT=8 -> 1024).
//
// Knobs (template):
//   WAVES, NT   WG geometry. Production LDS (8.3 KB Q + hw) limits residency to ~7 WGs/CU; with
//               8 waves per WG that is fine, but the per-WG Q staging prologue (16 KB f32 from L2,
//               convert, LDS, barrier, 16 fragment loads per lane) is paid once per 1024 keys.
//               Fewer waves x more tiles per wave changes the prologue/work ratio and residency.
//   PF          prefetch the next tile's 9 key words into registers (as pf_qreg).
//   HWREG       the 16 head weights each lane needs (h = mt*16 + k_off + e) in VGPRs, loaded
//               straight from global (4 x float4), no lds_hw reads in the loop.
//   VSTAGE      vectorised Q staging: float4 global loads, 4 cvt, one b64 LDS store (vs 16-32
//               scalar loads + ds_store_b16 each in production).
// The WMMA k-tile order, the relu*hw sum order and the shfl_xor(16) reduce are verbatim from
// production, so every variant is bit-exact by construction (checked by the harness).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "e2m1_key_common.inc"

#define N_INDEXER_HEAD 32
#define N_INDEXER_HEAD_DIM 128
typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));
#define ISW_N_TILE   16u
#define ISW_M_TILES   (N_INDEXER_HEAD / 16u)
#define ISW_K_TILES  8u

__device__ __forceinline__ void zero_words(e2m1_key_row_words& r) {
    #pragma unroll
    for (int i = 0; i < 8; i++) r.w[i] = 0u;
    r.ex4 = 0u;
}

template <unsigned WAVES, unsigned NT, bool PF, bool HWREG, bool VSTAGE>
__device__ __forceinline__ void score_body2(
        float          * __restrict__ scores,
        const float    * __restrict__ q,
        const float    * __restrict__ head_weights,
        const unsigned char * __restrict__ index_comp_kv,
        const unsigned int * __restrict__ n_idx_per,
        unsigned int     n_idx_stride,
        const unsigned int * __restrict__ keys_base_per) {
    constexpr unsigned BLOCK = WAVES * 32u;
    constexpr unsigned cols_per_wave = NT * ISW_N_TILE;
    constexpr unsigned cols_per_wg   = WAVES * cols_per_wave;
    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;
    const unsigned int bi   = blockIdx.y;
    const unsigned int n_comp = n_idx_per[bi];
    const unsigned int c_wg0  = blockIdx.x * cols_per_wg;

    float       * __restrict__ scores_b = scores + (size_t)bi * n_idx_stride;
    const float * __restrict__ q_b      = q      + (size_t)bi * (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM);
    const float * __restrict__ hw_b     = head_weights + (size_t)bi * N_INDEXER_HEAD;

    if (c_wg0 >= n_comp) {
        for (unsigned int i = tid; i < cols_per_wg; i += BLOCK) {
            const unsigned int c = c_wg0 + i;
            if (c < n_idx_stride) scores_b[c] = -3.4028235e38f;
        }
        return;
    }

    __shared__ __attribute__((aligned(16))) _Float16 lds_q[N_INDEXER_HEAD * N_INDEXER_HEAD_DIM];
    __shared__ float    lds_hw[N_INDEXER_HEAD];

    if (VSTAGE) {
        const float4* q4 = reinterpret_cast<const float4*>(q_b);
        constexpr unsigned N4 = N_INDEXER_HEAD * N_INDEXER_HEAD_DIM / 4u;   // 1024
        #pragma unroll
        for (unsigned int i = tid; i < N4; i += BLOCK) {
            const float4 v = q4[i];
            const _Float16 h0 = (_Float16)v.x, h1 = (_Float16)v.y, h2 = (_Float16)v.z, h3 = (_Float16)v.w;
            uint2 pk;
            pk.x = (unsigned int)__builtin_bit_cast(unsigned short, h0) | ((unsigned int)__builtin_bit_cast(unsigned short, h1) << 16);
            pk.y = (unsigned int)__builtin_bit_cast(unsigned short, h2) | ((unsigned int)__builtin_bit_cast(unsigned short, h3) << 16);
            *reinterpret_cast<uint2*>(lds_q + (size_t)i * 4u) = pk;
        }
    } else {
        #pragma unroll 8
        for (unsigned int i = tid; i < N_INDEXER_HEAD * N_INDEXER_HEAD_DIM; i += BLOCK) {
            lds_q[i] = (_Float16)q_b[i];
        }
    }
    if (!HWREG && tid < N_INDEXER_HEAD) lds_hw[tid] = hw_b[tid];
    __syncthreads();

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int col_in_tile = lane & 15u;
    const unsigned int half        = lane >> 4;
    const unsigned int k_off       = half * 8u;
    const unsigned int c_wave0     = c_wg0 + wave * cols_per_wave;
    const unsigned int kbase_row   = keys_base_per ? keys_base_per[bi] : 0u;

    half8 a_reg[ISW_M_TILES][ISW_K_TILES];
    #pragma unroll
    for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
        const _Float16* a_row = lds_q + (size_t)(mt * 16u + col_in_tile) * N_INDEXER_HEAD_DIM + k_off;
        #pragma unroll
        for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
            __builtin_memcpy(&a_reg[mt][kt], a_row + kt * 16u, 16);
        }
    }
    float hw_reg[ISW_M_TILES][8];
    if (HWREG) {
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            const float4 a = *reinterpret_cast<const float4*>(hw_b + mt * 16u + k_off);
            const float4 b = *reinterpret_cast<const float4*>(hw_b + mt * 16u + k_off + 4u);
            hw_reg[mt][0] = a.x; hw_reg[mt][1] = a.y; hw_reg[mt][2] = a.z; hw_reg[mt][3] = a.w;
            hw_reg[mt][4] = b.x; hw_reg[mt][5] = b.y; hw_reg[mt][6] = b.z; hw_reg[mt][7] = b.w;
        }
    }

    e2m1_key_row_words kw_next;
    auto issue = [&](unsigned int nt) {
        const unsigned int c_row = c_wave0 + nt * ISW_N_TILE + col_in_tile;
        if (c_row < n_comp) kw_next = e2m1_key_load_row_words(index_comp_kv + (size_t)(kbase_row + c_row) * E2M1_KEY_ROW_BYTES, k_off);
        else zero_words(kw_next);
    };
    if (PF) issue(0);

    #pragma unroll 1
    for (unsigned int nt_local = 0; nt_local < NT; nt_local++) {
        const unsigned int c_base = c_wave0 + nt_local * ISW_N_TILE;
        if (c_base >= n_idx_stride) break;
        if (c_base >= n_comp) {
            if (lane < ISW_N_TILE) {
                const unsigned int c = c_base + lane;
                if (c < n_idx_stride) scores_b[c] = -3.4028235e38f;
            }
            continue;
        }
        const unsigned int c_row = c_base + col_in_tile;
        const bool c_row_valid = (c_row < n_comp);
        e2m1_key_row_words kwords;
        if (PF) {
            kwords = kw_next;
            if (nt_local + 1u < NT) issue(nt_local + 1u);
        } else {
            if (c_row_valid) kwords = e2m1_key_load_row_words(index_comp_kv + (size_t)(kbase_row + c_row) * E2M1_KEY_ROW_BYTES, k_off);
            else zero_words(kwords);
        }

        float8 c_acc[ISW_M_TILES];
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) c_acc[mt] = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (unsigned int kt = 0; kt < ISW_K_TILES; kt++) {
            half8 b_frag;
            if (c_row_valid) {
                b_frag = __builtin_bit_cast(half8, e2m1_key_b_frag_kt(kwords, kt));
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) b_frag[e] = (_Float16)0.0f;
            }
            #pragma unroll
            for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
                c_acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg[mt][kt], b_frag, c_acc[mt]);
            }
        }
        float partial = 0.0f;
        #pragma unroll
        for (unsigned int mt = 0; mt < ISW_M_TILES; mt++) {
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                float v = c_acc[mt][e];
                if (v < 0.0f) v = 0.0f;
                const float w = HWREG ? hw_reg[mt][e] : lds_hw[mt * 16u + k_off + (unsigned int)e];
                partial += v * w;
            }
        }
        partial += __shfl_xor(partial, 16);
        if (lane < ISW_N_TILE) {
            const unsigned int c = c_base + lane;
            if (c < n_idx_stride) scores_b[c] = (c < n_comp) ? partial : -3.4028235e38f;
        }
    }
#else
    (void)scores_b; (void)hw_b; (void)n_idx_stride; (void)wave;
#endif
}

#define SCORE2_KERNEL(NAME, WAVES, NT, PF, HWREG, VSTAGE)                                              \
    extern "C" __global__ __launch_bounds__(WAVES * 32u)                                               \
    void NAME(float* __restrict__ scores, const float* __restrict__ q, const float* __restrict__ head_weights, \
              const unsigned char* __restrict__ index_comp_kv, const unsigned int* __restrict__ n_idx_per, \
              unsigned int n_idx_stride, const unsigned int* __restrict__ keys_base_per) {            \
        score_body2<WAVES, NT, PF, HWREG, VSTAGE>(scores, q, head_weights, index_comp_kv, n_idx_per, n_idx_stride, keys_base_per); \
    }

//            name                W  NT  PF     HWREG  VSTAGE   cols/WG
SCORE2_KERNEL(s2_w8n8_qreg,       8,  8, false, false, false)  // 1024: qreg without prefetch
SCORE2_KERNEL(s2_w8n8_pf_hw,      8,  8, true,  true,  false)  // 1024: pf_qreg + hw regs
SCORE2_KERNEL(s2_w8n8_pf_hw_vs,   8,  8, true,  true,  true)   // 1024: + vectorised staging
SCORE2_KERNEL(s2_w8n8_hw_vs,      8,  8, false, true,  true)   // 1024: no prefetch
SCORE2_KERNEL(s2_w4n16_pf_hw_vs,  4, 16, true,  true,  true)   // 1024: 4 waves x 16 tiles
SCORE2_KERNEL(s2_w4n8_pf_hw_vs,   4,  8, true,  true,  true)   //  512
SCORE2_KERNEL(s2_w8n16_pf_hw_vs,  8, 16, true,  true,  true)   // 2048
SCORE2_KERNEL(s2_w4n32_pf_hw_vs,  4, 32, true,  true,  true)   // 2048
SCORE2_KERNEL(s2_w2n32_pf_hw_vs,  2, 32, true,  true,  true)   // 1024: 2 waves x 32 tiles
SCORE2_KERNEL(s2_w16n4_pf_hw_vs, 16,  4, true,  true,  true)   // 1024: 16 waves x 4 tiles
