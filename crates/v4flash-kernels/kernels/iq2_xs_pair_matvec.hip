// iq2_xs_pair_matvec — gate+up fused-SwiGLU kernels for IQ2_XS weights.
//
// unsloth UD-Q2_K_XL mix: ffn_gate_exps + ffn_up_exps are IQ2_XS on 42 of
// 43 layers (blk.26 goes IQ3_XXS). Structure mirrors the iq2_s pair family
// — the same three load-bearing variants (decode batch, decode hetsplit,
// prefill chunked-by-expert) — since only the per-block dot differs.
//
// IQ2_XS block (74 B per 256 elements), llama.cpp block_iq2_xs:
//   d (half) | qs[32] u16 | scales[8]
//   qs[4*ib32 + l] packs BOTH fields of one 8-weight group:
//     bits 0..8  = 9-bit index into iq2xs_grid[512]
//     bits 9..15 = 7-bit index into the ksigns_iq2xs[128] codebook
//   scales[ib32] = two 4-bit half-scales: ls1 = 1+2*(s&0xf) (l=0,1),
//                  ls2 = 1+2*(s>>4) (l=2,3)  — one scale per 16 weights
//   block result = 0.125 * d * yd * Σ_ib32 (ls1*sumi1 + ls2*sumi2)
//
// Versus its two neighbours: IQ2_XXS has a 256-entry grid and packs its
// scale into an aux word; IQ2_S has a 1024-entry grid, a separate qh plane
// and explicit sign BYTES. IQ2_XS takes XXS's sign codebook and S's
// explicit nibble scales, which is why this file is S's skeleton with
// XXS's sign staging.
//
// LDS: s_sign_pair128 (1 KiB; 128 pre-expanded ksigns mask pairs) +
// s_grid2xs (4 KiB; 512×u64 codebook) + xq staging (4.7 KiB) ≈ 9.7 KiB/WG,
// comfortably under the iq2_s family's 15 KiB.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_IQ2_XS_BYTES 74
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

#include "iq2_xs_tables.inc"   // iq2xs_grid[512]

__device__ __forceinline__ float f16_to_f32_xs(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4_xs(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t load_dword_xs(const int8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

__device__ __forceinline__ uint32_t expand_signs_xs(uint32_t s, uint32_t mask) {
    const uint32_t m = s & mask;
    uint32_t out = 0;
    if (m & 0xffu)         out |= 0xffu;
    if (m & 0xff00u)       out |= 0xff00u;
    if (m & 0xff0000u)     out |= 0xff0000u;
    if (m & 0xff000000u)   out |= 0xff000000u;
    return out;
}

__device__ __forceinline__ int32_t cond_neg_bytes_fast(int32_t pack, uint32_t sm) {
    return (int32_t)(((uint32_t)pack ^ sm) + (sm & 0x01010101u));
}

// Stage the 128-entry sign-pair table + 512-entry grid into LDS.
//
// Sign source is the 7-bit ksigns CODEBOOK (as in IQ2_XXS), not IQ2_S's raw
// explicit byte — so the staged table is 128 entries indexed by qs>>9, and
// each entry must go through ksigns_iq2xs first.
//
// Staging is not optional: these lookups are data-dependent per lane, and
// divergent __constant__ access serializes (the M14k/M31 lesson that cost
// IQ3_XXS 3.7x before its tables moved to LDS). 128*8 + 512*8 = 5 KiB.
#define IQ2XS_STAGE_TABLES(s_sign_pair128, s_grid2xs)                         \
    {                                                                         \
        const unsigned int tid_ = threadIdx.x;                                \
        if (tid_ < 128u) {                                                    \
            const uint32_t signs_ = (uint32_t)ksigns_iq2xs[tid_] * 0x01010101u; \
            const uint32_t sm0_ = expand_signs_xs(signs_, 0x08040201u);       \
            const uint32_t sm1_ = expand_signs_xs(signs_, 0x80402010u);       \
            s_sign_pair128[tid_] = ((uint64_t)sm1_ << 32) | (uint64_t)sm0_;   \
        }                                                                     \
        for (unsigned int g_ = threadIdx.x; g_ < 512u; g_ += blockDim.x) {    \
            s_grid2xs[g_] = iq2xs_grid[g_];                                   \
        }                                                                     \
    }

__device__ __forceinline__ float warp_sum_f32_xs(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// This lane's half of one super-block: ib32 ∈ [ib32_lo, ib32_lo+4).
// Returns 0.125 * d * yd * partial_bsum.
__device__ __forceinline__ float dot_block_half_iq2xs(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int ib32_lo,
        const uint64_t* __restrict__ s_sign_pair128,
        const uint64_t* __restrict__ s_grid2xs) {
    // Layout (74 B): d(f16) | qs[32] u16 | scales[8] u8.
    // Each qs entry packs a 9-bit grid index (low) and a 7-bit ksigns index
    // (high) — IQ2_S's separate qh/sign-byte planes do not exist here.
    const uint16_t wd = *(const uint16_t*)(w_blk + 0);
    const uint16_t* qs = (const uint16_t*)(w_blk + 2);
    const uint8_t*  scales = w_blk + 66;

    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8_base = (const int8_t*)(y_blk + 4);
    const float d = f16_to_f32_xs(wd) * yd;

    int32_t bsum = 0;
    #pragma unroll
    for (unsigned int t = 0; t < 4u; t++) {
        const unsigned int ib32 = ib32_lo + t;
        const uint16_t* q2 = qs + 4u * ib32;
        const uint8_t  sc  = scales[ib32];
        // Two 4-bit scales per byte, each covering 16 weights (l<2 / l>=2).
        const int32_t ls1 = 1 + 2 * (int32_t)(sc & 0x0F);
        const int32_t ls2 = 1 + 2 * (int32_t)(sc >> 4);
        const int8_t* q8 = q8_base + ib32 * 32u;

        int32_t sumi1 = 0, sumi2 = 0;
        #pragma unroll
        for (unsigned int l = 0; l < 4u; l++) {
            // Blocks are 74 B (even) and qs starts at +2, so every u16 here
            // is 2-byte aligned — no split-load helper needed.
            const uint32_t qv = (uint32_t)q2[l];
            const uint64_t grid = s_grid2xs[qv & 511u];
            const uint64_t sp = s_sign_pair128[qv >> 9];
            int32_t sumi = 0;
            sumi = sudot4_xs(
                cond_neg_bytes_fast((int32_t)(uint32_t)grid, (uint32_t)sp),
                load_dword_xs(q8 + 8u * l), sumi);
            sumi = sudot4_xs(
                cond_neg_bytes_fast((int32_t)(uint32_t)(grid >> 32),
                                    (uint32_t)(sp >> 32)),
                load_dword_xs(q8 + 8u * l + 4u), sumi);
            if (l < 2u) { sumi1 += sumi; } else { sumi2 += sumi; }
        }
        bsum += ls1 * sumi1 + ls2 * sumi2;
    }
    return 0.125f * d * (float)bsum;
}

// Decode: batched over n_used slots, fused SwiGLU + expert weight.
// Contract identical to iq2_xxs_pair_matvec_fused_swiglu_batch.
extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_batch(
        float        * __restrict__ mid,           // [n_used, n_rows]
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ expert_w,
        const int    * __restrict__ selected,      // [n_used]
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    __shared__ uint64_t s_sign_pair128[128];
    __shared__ uint64_t s_grid2xs[512];
    __shared__ uint8_t  s_xq[16 * 292];
    IQ2XS_STAGE_TABLES(s_sign_pair128, s_grid2xs);
    const unsigned int tid = threadIdx.x;
    const unsigned int xq_bytes = n_blocks * 292u;
    for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) {
        s_xq[i] = xq[i];
    }
    __syncthreads();

    const unsigned int slot = blockIdx.y;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const int e = selected[slot];
    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XS_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ2_XS_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ2_XS_BYTES;
        const uint8_t* y   = s_xq          + (size_t)b   * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half_iq2xs(w_g, y, ib32_lo, s_sign_pair128, s_grid2xs);
        u_sum += dot_block_half_iq2xs(w_u, y, ib32_lo, s_sign_pair128, s_grid2xs);
    }

    g_sum = warp_sum_f32_xs(g_sum);
    u_sum = warp_sum_f32_xs(u_sum);

    if (lane == 0) {
        const float ew = expert_w[slot];
        if (clamp > 1.0e-6f) {
            if (g_sum > clamp)  g_sum = clamp;
            if (u_sum > clamp)  u_sum = clamp;
            if (u_sum < -clamp) u_sum = -clamp;
        }
        const float sig = 1.0f / (1.0f + expf(-g_sum));
        mid[(uint64_t)slot * (uint64_t)n_rows + (uint64_t)row] =
            g_sum * sig * u_sum * ew;
    }
}

// Decode het-split; contract identical to
// iq2_xxs_pair_matvec_fused_swiglu_batch_hetsplit.
extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_batch_hetsplit(
        float        * __restrict__ mid,           // [n_used, n_rows]
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ expert_w,
        const int    * __restrict__ selected,      // [n_used]
        const int    * __restrict__ remap,         // [256]
        unsigned int    mode,
        unsigned int    dgpu_cap,
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    __shared__ uint64_t s_sign_pair128[128];
    __shared__ uint64_t s_grid2xs[512];
    __shared__ uint8_t  s_xq[16 * 292];

    const unsigned int slot = blockIdx.y;
    const int sel = selected[slot];
    const int dense = remap[sel];
    const unsigned int tid = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;

    unsigned int res_rank = 0;
    for (unsigned int t2 = 0; t2 < slot; t2++) {
        res_rank += (remap[selected[t2]] >= 0) ? 1u : 0u;
    }
    const bool dgpu_takes = (dense >= 0) && (res_rank < dgpu_cap);
    const bool ours = (mode == 0u) ? !dgpu_takes : dgpu_takes;
    if (!ours) {
        if (row < n_rows && lane == 0) {
            mid[(uint64_t)slot * (uint64_t)n_rows + (uint64_t)row] = 0.0f;
        }
        return;
    }
    // M63 iGPU de-dup: remap encodes BOTH devices' slots — hot experts hold
    // their dGPU dense slot (>=0), cold experts hold -(igpu_slot+1). With the
    // full 256-expert iGPU buffer igpu_slot == sel, so this is identical to the
    // old `sel`; with IGPU_DEDUP_HOT it is the packed cold-only slot.
    // An OVER-CAP resident slot (dense >= 0, rank >= cap) also lands here in
    // mode 0; its iGPU copy sits at the raw id `sel` (the full 256-expert
    // buffer). `-dense - 1` would be negative for it. Under IGPU_DEDUP_HOT
    // the cap is pinned to N_EXPERT_USED, so over-cap cannot occur and the
    // packed buffer is never indexed by a raw id.
    const int e = (mode == 0u) ? ((dense >= 0) ? sel : (-dense - 1)) : dense;

    IQ2XS_STAGE_TABLES(s_sign_pair128, s_grid2xs);
    const unsigned int xq_bytes = n_blocks * 292u;
    for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) {
        s_xq[i] = xq[i];
    }
    __syncthreads();
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XS_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ2_XS_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ2_XS_BYTES;
        const uint8_t* y   = s_xq          + (size_t)b   * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half_iq2xs(w_g, y, ib32_lo, s_sign_pair128, s_grid2xs);
        u_sum += dot_block_half_iq2xs(w_u, y, ib32_lo, s_sign_pair128, s_grid2xs);
    }

    g_sum = warp_sum_f32_xs(g_sum);
    u_sum = warp_sum_f32_xs(u_sum);

    if (lane == 0) {
        const float ew = expert_w[slot];
        if (clamp > 1.0e-6f) {
            if (g_sum > clamp)  g_sum = clamp;
            if (u_sum > clamp)  u_sum = clamp;
            if (u_sum < -clamp) u_sum = -clamp;
        }
        const float sig = 1.0f / (1.0f + expf(-g_sum));
        mid[(uint64_t)slot * (uint64_t)n_rows + (uint64_t)row] =
            g_sum * sig * u_sum * ew;
    }
}

// Prefill: chunked by-expert (work-items interface — same arrays as the
// iq2_xxs chunked/kwide and iq3/q2k by_expert kernels). One WG per
// (row-tile, work item); per member the batch's xq is staged in LDS.
// Writes SwiGLU-fused mid directly (same semantics as the decode kernels,
// per (b, slot)) — no partials/reduce needed for the gate/up stage.
extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_chunked(
        float        * __restrict__ mid,           // [B, n_used, n_rows]
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,           // [B, n_blocks * 292]
        const float  * __restrict__ expert_w,      // [B, n_used]
        const int    * __restrict__ group_count,   // [n_expert]
        const int    * __restrict__ expert_members,// [n_expert * max_per_expert]
        const int    * __restrict__ work_items,    // [n_work_items]
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        unsigned int    n_used,
        unsigned int    max_per_expert,
        unsigned int    chunk_size,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    const unsigned int w_idx = blockIdx.y;
    const int packed_item = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;

    __shared__ uint64_t s_sign_pair128[128];
    __shared__ uint64_t s_grid2xs[512];
    __shared__ uint8_t  s_xq[16 * 292];
    IQ2XS_STAGE_TABLES(s_sign_pair128, s_grid2xs);
    __syncthreads();

    const unsigned int tid = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool row_valid = row < n_rows;

    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XS_BYTES;

    const unsigned int xq_bytes = n_blocks * 292u;
    for (unsigned int m = member_start; m < member_end; m++) {
        const int packed = expert_members[(size_t)e * (size_t)max_per_expert + (size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);

        const uint8_t* xq_b = xq + (size_t)b * (size_t)xq_bytes;
        for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) {
            s_xq[i] = xq_b[i];
        }
        __syncthreads();

        if (row_valid) {
            float g_sum = 0.0f;
            float u_sum = 0.0f;
            for (unsigned int bl = block_lane; bl < n_blocks; bl += 16u) {
                const uint8_t* w_g = gate_row_base + (uint64_t)bl * BLOCK_IQ2_XS_BYTES;
                const uint8_t* w_u = up_row_base   + (uint64_t)bl * BLOCK_IQ2_XS_BYTES;
                const uint8_t* y   = s_xq          + (size_t)bl   * BLOCK_Q8_K_BYTES;
                g_sum += dot_block_half_iq2xs(w_g, y, ib32_lo, s_sign_pair128, s_grid2xs);
                u_sum += dot_block_half_iq2xs(w_u, y, ib32_lo, s_sign_pair128, s_grid2xs);
            }

            g_sum = warp_sum_f32_xs(g_sum);
            u_sum = warp_sum_f32_xs(u_sum);

            if (lane == 0) {
                const float ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
                float g = g_sum, u = u_sum;
                if (clamp > 1.0e-6f) {
                    if (g > clamp)  g = clamp;
                    if (u > clamp)  u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float sig = 1.0f / (1.0f + expf(-g));
                const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                                     * (size_t)n_rows + (size_t)row;
                mid[out_off] = g * sig * u * ew;
            }
        }
        __syncthreads(); // before next member's xq overwrite
    }
}

// =============================================================================
// kwide — M51-structure port for IQ2_XS (2026-08-15).
//
// The `_chunked` kernel above iterates chunk members SERIALLY: for every
// member it re-reads AND re-dequantizes the expert's full weight rows from
// global memory (grid/sign LUT walk + cond-neg per member). That is the
// ejpir-fork failure mode the iq2_xxs kwide kernel (M51, +39% e2e) was
// built to kill: with chunk_size=32 the same weight block is dequantized
// up to 32x per WG.
//
// Structure (mirrors iq2_xxs_pair_matvec_fused_swiglu_kwide):
//   - warp owns one row; 32 lanes split a PAIR of super-blocks
//     (sbh = lane>>4 picks the sb, l15 = lane&15 picks 16 weights of it).
//   - per sb-pair iteration each lane dequantizes its 16 gate + 16 up
//     weights ONCE into 8 registers, then dp4a's them against every
//     member's LDS-staged q8 slice (one ds_load_b128 per member feeds
//     8 dot4s).
//   - members' q8 slices are cooperatively staged branch-free (the M53
//     rewrite): thread t owns member t>>3, dword lanes (t&7)+8j.
//
// IQ2_XS lane dequant vs IQ2_XXS: the lane's 16 weights are ib32 =
// sub_block, groups l = 2*half, 2*half+1 — one u32 of qs (two u16
// entries, each 9-bit grid index + 7-bit ksigns index) + one scale
// nibble (IQ2_XS half-scales cover exactly a lane's 16 weights, so the
// per-lane scale fold is exact).
//
// LDS: 1 KiB sign pairs + 4 KiB grid (512 x u64) + 16 KiB q8 stage +
// 256 B yd + 128 B members ~= 21.4 KiB/WG.
// =============================================================================

#define IQ2XS_KW_MAX_CHUNK 32u
#ifndef IQ2XS_KW_UNROLL
#define IQ2XS_KW_UNROLL 4
#endif

extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_kwide(
        float        * __restrict__ mid,           // [B, n_used, n_rows]
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,           // [B, n_blocks*292]
        const float  * __restrict__ expert_w,      // [B, n_used]
        const int    * __restrict__ group_count,   // [n_expert]
        const int    * __restrict__ expert_members,// [n_expert * max_per_expert]
        const int    * __restrict__ work_items,    // [n_work_items]
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        unsigned int    n_used,
        unsigned int    max_per_expert,
        unsigned int    chunk_size,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    const unsigned int w_idx = blockIdx.y;
    const int packed_item = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;
    const unsigned int n_members = member_end - member_start;

    __shared__ uint64_t s_sign_pair128[128];
    __shared__ uint64_t s_grid2xs[512];
    __shared__ uint4    s_q8v[IQ2XS_KW_MAX_CHUNK * 2 * 16]; // [mi][sbh][16]
    __shared__ float    s_yd[IQ2XS_KW_MAX_CHUNK * 2];       // [mi][sbh]
    __shared__ int      s_member_packed[IQ2XS_KW_MAX_CHUNK];

    const unsigned int tid = threadIdx.x;
    IQ2XS_STAGE_TABLES(s_sign_pair128, s_grid2xs);
    for (unsigned int i = tid; i < n_members; i += blockDim.x) {
        s_member_packed[i] = expert_members[(size_t)e * (size_t)max_per_expert
                                          + (size_t)(member_start + i)];
    }
    __syncthreads();

    const unsigned int row_in_block = tid >> 5;            // 0..7 (warp index)
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool row_valid = (row < n_rows);

    const unsigned int sbh       = lane >> 4;              // 0..1: sb of pair
    const unsigned int l15       = lane & 15u;
    const unsigned int sub_block = l15 >> 1;               // 0..7 (ib32)
    const unsigned int half      = l15 & 1u;               // 0..1

    const uint8_t* gate_e_base = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;
    const uint8_t* up_e_base   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;

    float lane_pacc_g[IQ2XS_KW_MAX_CHUNK];
    float lane_pacc_u[IQ2XS_KW_MAX_CHUNK];
    #pragma unroll
    for (unsigned int mi = 0; mi < IQ2XS_KW_MAX_CHUNK; mi++) {
        lane_pacc_g[mi] = 0.0f;
        lane_pacc_u[mi] = 0.0f;
    }

    const unsigned int n_pairs = n_blocks >> 1;   // wrapper asserts n_blocks even
    for (unsigned int sbp = 0; sbp < n_pairs; sbp++) {
        const unsigned int sb_lane = 2u * sbp + sbh;       // this lane's sb

        // --- Dequant: this lane's 16 weights x 2 matrices -> 8 dwords ---
        int32_t wg0 = 0, wg1 = 0, wg2 = 0, wg3 = 0;
        int32_t wu0 = 0, wu1 = 0, wu2 = 0, wu3 = 0;
        float gds = 0.0f, uds = 0.0f;
        if (row_valid) {
            const uint8_t* gate_block = gate_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XS_BYTES
                                      + (uint64_t)sb_lane * BLOCK_IQ2_XS_BYTES;
            const uint8_t* up_block   = up_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XS_BYTES
                                      + (uint64_t)sb_lane * BLOCK_IQ2_XS_BYTES;
            const uint16_t wd_g = *(const uint16_t*)(gate_block + 0);
            const uint16_t wd_u = *(const uint16_t*)(up_block   + 0);

            // Two u16 qs entries for this lane: q2[4*ib32 + 2*half + {0,1}].
            // 2-aligned dword load (gfx1151/gfx1201 handle unaligned dwords;
            // same pattern as the xxs kwide aux loads).
            const uint32_t qs_g = *(const uint32_t*)(gate_block + 2 + sub_block * 8u + half * 4u);
            const uint32_t qs_u = *(const uint32_t*)(up_block   + 2 + sub_block * 8u + half * 4u);
            const uint8_t  sc_g = gate_block[66u + sub_block];
            const uint8_t  sc_u = up_block[66u + sub_block];

            {
                const uint32_t e0 = qs_g & 0xffffu;
                const uint32_t e1 = qs_g >> 16;
                const uint64_t grid0 = s_grid2xs[e0 & 511u];
                const uint64_t sp0   = s_sign_pair128[e0 >> 9];
                wg0 = cond_neg_bytes_fast((int32_t)(uint32_t)grid0,         (uint32_t)sp0);
                wg1 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid0 >> 32), (uint32_t)(sp0 >> 32));
                const uint64_t grid1 = s_grid2xs[e1 & 511u];
                const uint64_t sp1   = s_sign_pair128[e1 >> 9];
                wg2 = cond_neg_bytes_fast((int32_t)(uint32_t)grid1,         (uint32_t)sp1);
                wg3 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid1 >> 32), (uint32_t)(sp1 >> 32));
            }
            {
                const uint32_t e0 = qs_u & 0xffffu;
                const uint32_t e1 = qs_u >> 16;
                const uint64_t grid0 = s_grid2xs[e0 & 511u];
                const uint64_t sp0   = s_sign_pair128[e0 >> 9];
                wu0 = cond_neg_bytes_fast((int32_t)(uint32_t)grid0,         (uint32_t)sp0);
                wu1 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid0 >> 32), (uint32_t)(sp0 >> 32));
                const uint64_t grid1 = s_grid2xs[e1 & 511u];
                const uint64_t sp1   = s_sign_pair128[e1 >> 9];
                wu2 = cond_neg_bytes_fast((int32_t)(uint32_t)grid1,         (uint32_t)sp1);
                wu3 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid1 >> 32), (uint32_t)(sp1 >> 32));
            }

            // IQ2_XS nibble half-scale: ls1 (half=0) covers l=0,1;
            // ls2 (half=1) covers l=2,3 — exactly this lane's 16 weights.
            const int32_t ls_g = 1 + 2 * (int32_t)(half ? (sc_g >> 4) : (sc_g & 0x0F));
            const int32_t ls_u = 1 + 2 * (int32_t)(half ? (sc_u >> 4) : (sc_u & 0x0F));
            gds = (0.125f * f16_to_f32_xs(wd_g)) * (float)ls_g;
            uds = (0.125f * f16_to_f32_xs(wd_u)) * (float)ls_u;
        }

        // --- Cooperative xq stage (branch-free; M53 pattern, verbatim from
        //     the xxs kwide: thread t owns member t>>3, dword lanes
        //     (t&7)+8j of the member's 2x64 q8 dwords). ---
        {
            uint32_t* s_q8_dw = (uint32_t*)s_q8v;
            const unsigned int src_blk0 = 2u * sbp;
            const unsigned int mi = tid >> 3;
            const unsigned int r  = tid & 7u;
            if (mi < n_members) {
                const int packed     = s_member_packed[mi];
                const unsigned int b = (unsigned int)(packed >> 16);
                const uint32_t* xq_src_dw = (const uint32_t*)(
                    xq + (size_t)b * (size_t)n_blocks * (size_t)BLOCK_Q8_K_BYTES
                       + (size_t)src_blk0 * (size_t)BLOCK_Q8_K_BYTES);
                #pragma unroll
                for (unsigned int j = 0; j < 8u; j++) {
                    const unsigned int ii = r + 8u * j;          // 0..63
                    s_q8_dw[(mi * 2u + 0u) * 64u + ii] = xq_src_dw[1u + ii];
                }
                #pragma unroll
                for (unsigned int j = 0; j < 8u; j++) {
                    const unsigned int ii = r + 8u * j;
                    s_q8_dw[(mi * 2u + 1u) * 64u + ii] = xq_src_dw[74u + ii];
                }
            }
            if (tid < 2u * n_members) {
                const unsigned int myd = tid >> 1;
                const unsigned int h   = tid & 1u;
                const int packed     = s_member_packed[myd];
                const unsigned int b = (unsigned int)(packed >> 16);
                const uint32_t* xq_src_dw = (const uint32_t*)(
                    xq + (size_t)b * (size_t)n_blocks * (size_t)BLOCK_Q8_K_BYTES
                       + (size_t)src_blk0 * (size_t)BLOCK_Q8_K_BYTES);
                s_yd[myd * 2u + h] = __uint_as_float(xq_src_dw[h * 73u]);
            }
        }
        __syncthreads();

        // --- Dot phase: one b128 per (member, lane) feeds 8 dot4 ---
        if (row_valid) {
            const uint4* q8_lane = &s_q8v[sbh * 16u + l15];   // member mi at +mi*32
            const float* yd_lane = &s_yd[sbh];                // member mi at +mi*2

            #pragma unroll IQ2XS_KW_UNROLL
            for (unsigned int mi = 0; mi < n_members; mi++) {
                const uint4 q = q8_lane[mi * 32u];
                int32_t sumi_g = 0;
                sumi_g = sudot4_xs(wg0, (int32_t)q.x, sumi_g);
                sumi_g = sudot4_xs(wg1, (int32_t)q.y, sumi_g);
                sumi_g = sudot4_xs(wg2, (int32_t)q.z, sumi_g);
                sumi_g = sudot4_xs(wg3, (int32_t)q.w, sumi_g);
                int32_t sumi_u = 0;
                sumi_u = sudot4_xs(wu0, (int32_t)q.x, sumi_u);
                sumi_u = sudot4_xs(wu1, (int32_t)q.y, sumi_u);
                sumi_u = sudot4_xs(wu2, (int32_t)q.z, sumi_u);
                sumi_u = sudot4_xs(wu3, (int32_t)q.w, sumi_u);
                const float yd = yd_lane[mi * 2u];
                lane_pacc_g[mi] += (float)sumi_g * (yd * gds);
                lane_pacc_u[mi] += (float)sumi_u * (yd * uds);
            }
        }
        __syncthreads();  // protect s_q8v/s_yd from next pair's overwrite
    }

    // --- Reduce + swiglu + output, fused ---
    if (row_valid) {
        for (unsigned int mi = 0; mi < n_members; mi++) {
            float g_sum = warp_sum_f32_xs(lane_pacc_g[mi]);
            float u_sum = warp_sum_f32_xs(lane_pacc_u[mi]);
            if (lane == 0u) {
                const int packed          = s_member_packed[mi];
                const unsigned int b      = (unsigned int)(packed >> 16);
                const unsigned int slot   = (unsigned int)(packed & 0xFFFFu);
                if (clamp > 1.0e-6f) {
                    if (g_sum > clamp)  g_sum = clamp;
                    if (u_sum > clamp)  u_sum = clamp;
                    if (u_sum < -clamp) u_sum = -clamp;
                }
                const float ew  = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
                const float sig = 1.0f / (1.0f + expf(-g_sum));
                const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                                     * (size_t)n_rows + (size_t)row;
                mid[out_off] = g_sum * sig * u_sum * ew;
            }
        }
    }
}
