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


// =============================================================================
// wmma — f16 WMMA batched-prefill gate/up for IQ2_XS (2026-09-08).
//
// Why: the kwide kernel above runs at ~15% of the dot4 peak (137 GMAC in
// ~32 ms per 1024-token launch vs ~28 TMAC/s). It is issue-bound: every
// wave dot4 buys 128 MACs and drags ~one ancillary instruction with it. A
// wave WMMA buys 4096 MACs for one B-fragment load. Same peak on gfx1151
// (WMMA "internally uses the DOT instructions" — RDNA3 ISA 7.9 — so it
// occupies the VALU and NOTHING overlaps it; the only lever is fewer VALU
// instructions per MAC).
//
// Structure:
//   - WG = 256 threads = 8 waves, each wave owns a 16-row M-tile
//     (IQ2XS_WM_ROWS_PER_WG = 128 rows per WG).
//   - work item = (expert, <=32 members) exactly as kwide; members map to
//     N-tiles of 16 (n_tiles = 1 or 2, a TEMPLATE parameter: a runtime tile
//     count inside the k loop cost ~64 accumulator v_movs per k-step).
//     Padded members multiply zeros and are not written.
//   - RDNA3 wave32 WMMA fragment layout (verified on gfx1151 by
//     vit_wmma_probe and iq2_xs_wmma_upper_lane_probe): A lane l holds
//     A[l%16][0..16], B lane l holds B[0..16][l%16], lanes 16-31 MUST
//     replicate lanes 0-15 (spec + probe: every element of both halves is
//     read); C lane l elem i holds C[2i + (l>>4)][l%16].
//   - Weights: each lane loads its row's whole 74-B block per super-block
//     (4 x b128 qs + b64 scales + u16 d = 6 wide loads per matrix) and the
//     16 k-steps are fully unrolled so all slicing is static. The first cut
//     loaded one u32 per k-step and re-touched the same lines 16x:
//     MemUnitBusy 99%, 25 ms.
//   - Dequant ("magic"): i8 grid LUT (u64 gathers) + BIASED sign-mask LUT
//     (kwide's tables, masks ^ 0x80..). Per dword of 4 weights:
//     u = (g ^ smb) + (smb & 0x01010101) -> bytes 128 +/- grid; two
//     v_perm_b32 build halves 0x64|u = 1024+u exactly; v_pk_add_f16 -1152
//     then v_pk_mul_f16 by 0.125*d*ls. 9 VALU per 4 weights, ~100 VALU per
//     k-step for both matrices (vs 170 for per-byte cvt, and 62 for an f16
//     LUT whose 16-B random gathers made the LDS the co-bottleneck).
//   - Activations f16 [B, K]; each super-block's 256 k of the <=32 members
//     is staged in LDS once per WG (row stride 264 halves = 528 B ≡ 4
//     banks: conflict-free B-fragment reads).
//   - Epilogue: SwiGLU + router weight straight from the C fragments.
//
// Measured (iGPU, B=1024 uniform 32-member items): kwide 32.2 ms, this
// kernel 11.7 ms (2.75x); WMMA-only ablation 5.9 ms = 84% of matrix peak.
// Real routing (expert_stats layer 20, hot-17 removed): B=512/lane 15.9 ->
// 9.0 ms (1.75x), B=1024 26.8 -> 12.1 ms (2.2x). Per-item cost is flat
// ~34 us for 1-16 members; kwide is ~30 us + 2.8 us/member, so WMMA wins
// or ties at every member count >= 2 (no small-expert fallback needed).
// Tried and neutral/negative: 128-k staging windows for 3 WG/CU (-15..20%),
// gather prefetch one k-step ahead (0), B-fragment prefetch (0), runtime
// loop of 4-k-step groups to shrink code below the 32 KB I-cache (-20%).
// Oracle: tests/iq2_xs_wmma_oracle.rs, rel 3e-5 vs CPU f32 (kwide/Q8_K
// is 2.7% rms on the same inputs).
//
// Bench-only twins (same body, other DEQ paths / ablations) keep their
// symbols for A/B; mode numbers in Iq2XsPairMatvec::launch_fused_swiglu_wmma.
// LDS: 5 KiB LUTs + 16.5 KiB x-stage + 128 B ≈ 22 KiB/WG (2 WG/CU).
// =============================================================================

#define IQ2XS_WM_MAX_CHUNK   32u
#define IQ2XS_WM_ROWS_PER_WG 128u
#ifndef IQ2XS_WM_KWIN
#define IQ2XS_WM_KWIN        256u   // halves of K staged per window; 128 measured 15-20% SLOWER (2026-09-08)
#endif
#define IQ2XS_WM_X_STRIDE    (IQ2XS_WM_KWIN + 8u)   // halves per staged member row (+8 pad = 4 banks)

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define IQ2XS_WMMA_RDNA3 1
#endif

typedef _Float16 iq2xs_half16 __attribute__((ext_vector_type(16)));
typedef _Float16 iq2xs_half8  __attribute__((ext_vector_type(8)));
typedef _Float16 iq2xs_half2  __attribute__((ext_vector_type(2)));
typedef float    iq2xs_float8 __attribute__((ext_vector_type(8)));

// DEQ=0 helper: 4 dwords of signed i8 (byte i of dword j = weight 4j+i)
// -> 16 halves scaled by `scale`.
__device__ __forceinline__ iq2xs_half16 iq2xs_i8x16_to_h16(
        int32_t w0, int32_t w1, int32_t w2, int32_t w3, float scale) {
    iq2xs_half16 r;
    const int32_t w[4] = {w0, w1, w2, w3};
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int32_t v = (w[j] << (24 - 8 * i)) >> 24;   // sign-extend byte i
            r[4 * j + i] = (_Float16)((float)v * scale);
        }
    }
    return r;
}

// DEQ=1 helper: one 8-weight group. grid16 = 8 halves of the unsigned grid
// values, sign16 = 8 halves of 0x0000/0x8000. Returns (grid ^ sign) * scale.
__device__ __forceinline__ uint4 iq2xs_lut_group_h8(uint4 grid16, uint4 sign16, _Float16 scale) {
    uint4 v;
    v.x = grid16.x ^ sign16.x;
    v.y = grid16.y ^ sign16.y;
    v.z = grid16.z ^ sign16.z;
    v.w = grid16.w ^ sign16.w;
    iq2xs_half8 h;
    __builtin_memcpy(&h, &v, 16);
    h = h * scale;
    uint4 o;
    __builtin_memcpy(&o, &h, 16);
    return o;
}

// Stage the DEQ=1 tables: f16 grid (512 x 8 halves) + f16 sign masks
// (128 x 8 halves). 256 threads: 2 grid entries each + threads < 128 one
// sign entry each.
#define IQ2XS_STAGE_TABLES_F16(s_grid16, s_sign16)                             \
    {                                                                          \
        const unsigned int tid_ = threadIdx.x;                                 \
        for (unsigned int g_ = tid_; g_ < 512u; g_ += blockDim.x) {            \
            const uint64_t gv_ = iq2xs_grid[g_];                               \
            iq2xs_half8 h_;                                                    \
            _Pragma("unroll")                                                  \
            for (int i_ = 0; i_ < 8; i_++) {                                   \
                h_[i_] = (_Float16)(float)((gv_ >> (8 * i_)) & 0xffu);        \
            }                                                                  \
            uint4 o_;                                                          \
            __builtin_memcpy(&o_, &h_, 16);                                    \
            s_grid16[g_] = o_;                                                 \
        }                                                                      \
        if (tid_ < 128u) {                                                     \
            const uint32_t k_ = (uint32_t)ksigns_iq2xs[tid_];                  \
            uint4 s_;                                                          \
            s_.x = ((k_ & 0x01u) ? 0x00008000u : 0u) | ((k_ & 0x02u) ? 0x80000000u : 0u); \
            s_.y = ((k_ & 0x04u) ? 0x00008000u : 0u) | ((k_ & 0x08u) ? 0x80000000u : 0u); \
            s_.z = ((k_ & 0x10u) ? 0x00008000u : 0u) | ((k_ & 0x20u) ? 0x80000000u : 0u); \
            s_.w = ((k_ & 0x40u) ? 0x00008000u : 0u) | ((k_ & 0x80u) ? 0x80000000u : 0u); \
            s_sign16[tid_] = s_;                                               \
        }                                                                      \
    }



// One k-step's raw LUT rows for both matrices (DEQ=1): 4 grid + 4 sign
// b128 gathers. Kept separate from the xor/scale so the caller can issue the
// gathers one k-step AHEAD and let the current k-step's WMMAs cover their
// latency (software pipelining; the compiler would not hoist them itself).
struct iq2xs_raw_frag { uint4 g0, s0, g1, s1, u0, t0, u1, t1; };
__device__ __forceinline__ iq2xs_raw_frag iq2xs_gather_raw(
        const uint4* s_grid16, const uint4* s_sign16, uint32_t qs_g, uint32_t qs_u) {
    iq2xs_raw_frag r;
    const uint32_t eg0 = qs_g & 0xffffu, eg1 = qs_g >> 16;
    const uint32_t eu0 = qs_u & 0xffffu, eu1 = qs_u >> 16;
    r.g0 = s_grid16[eg0 & 511u]; r.s0 = s_sign16[eg0 >> 9];
    r.g1 = s_grid16[eg1 & 511u]; r.s1 = s_sign16[eg1 >> 9];
    r.u0 = s_grid16[eu0 & 511u]; r.t0 = s_sign16[eu0 >> 9];
    r.u1 = s_grid16[eu1 & 511u]; r.t1 = s_sign16[eu1 >> 9];
    return r;
}
__device__ __forceinline__ void iq2xs_finish_frag(
        const iq2xs_raw_frag& r, _Float16 hsg, _Float16 hsu, iq2xs_half16& a_g, iq2xs_half16& a_u) {
    const uint4 g0 = iq2xs_lut_group_h8(r.g0, r.s0, hsg);
    const uint4 g1 = iq2xs_lut_group_h8(r.g1, r.s1, hsg);
    const uint4 u0 = iq2xs_lut_group_h8(r.u0, r.t0, hsu);
    const uint4 u1 = iq2xs_lut_group_h8(r.u1, r.t1, hsu);
    __builtin_memcpy(((char*)&a_g) +  0, &g0, 16);
    __builtin_memcpy(((char*)&a_g) + 16, &g1, 16);
    __builtin_memcpy(((char*)&a_u) +  0, &u0, 16);
    __builtin_memcpy(((char*)&a_u) + 16, &u1, 16);
}


// DEQ=2 ("magic"): i8 grid LUT (u64 gathers, kwide's s_grid2xs) + BIASED sign
// masks (kwide's s_sign_pair128 ^ 0x8080..). Per dword of 4 unsigned grid
// bytes g and biased mask smb (0x80 = +, 0x7F = -):
//   u   = (g ^ smb) + (smb & 0x01010101)      bytes = 128 +/- grid   (3 VALU)
//   h01 = perm(u, 0x64646464, 0x00050004)      halves 0x64|u0, 0x64|u1 = 1024+u (2 VALU)
//   h23 = perm(u, 0x64646464, 0x00070006)
//   v   = (h - 1152) * scale                   pk_add exact, then pk_mul   (4 VALU)
// = 9 VALU per 4 weights, no f16 LUT, 8-byte gathers only.
__device__ __forceinline__ void iq2xs_magic_dword(uint32_t g, uint32_t smb, _Float16 scale,
                                                  uint32_t& lo, uint32_t& hi) {
    const uint32_t u = (g ^ smb) + (smb & 0x01010101u);
    const uint32_t h01 = __builtin_amdgcn_perm(u, 0x64646464u, 0x00050004u);
    const uint32_t h23 = __builtin_amdgcn_perm(u, 0x64646464u, 0x00070006u);
    iq2xs_half2 a, b;
    __builtin_memcpy(&a, &h01, 4);
    __builtin_memcpy(&b, &h23, 4);
    const iq2xs_half2 bias = {(_Float16)-1152.0f, (_Float16)-1152.0f};
    a = (a + bias) * scale;
    b = (b + bias) * scale;
    __builtin_memcpy(&lo, &a, 4);
    __builtin_memcpy(&hi, &b, 4);
}
// 16 weights of one matrix: two grid entries (u64 each) + two biased sign masks.
__device__ __forceinline__ iq2xs_half16 iq2xs_magic_frag(
        uint64_t g0, uint64_t sm0, uint64_t g1, uint64_t sm1, _Float16 scale) {
    uint32_t d[8];
    iq2xs_magic_dword((uint32_t)g0,         (uint32_t)sm0,         scale, d[0], d[1]);
    iq2xs_magic_dword((uint32_t)(g0 >> 32), (uint32_t)(sm0 >> 32), scale, d[2], d[3]);
    iq2xs_magic_dword((uint32_t)g1,         (uint32_t)sm1,         scale, d[4], d[5]);
    iq2xs_magic_dword((uint32_t)(g1 >> 32), (uint32_t)(sm1 >> 32), scale, d[6], d[7]);
    iq2xs_half16 r;
    __builtin_memcpy(&r, d, 32);
    return r;
}


// DEQ=2 raw gathers for one k-step (both matrices): 4 grid u64 + 4 sign u64.
struct iq2xs_raw8 { uint64_t g0, s0, g1, s1, u0, t0, u1, t1; };
__device__ __forceinline__ iq2xs_raw8 iq2xs_gather_raw8(
        const uint64_t* s_grid2xs, const uint64_t* s_sign_pair128, uint32_t qs_g, uint32_t qs_u) {
    iq2xs_raw8 r;
    const uint32_t eg0 = qs_g & 0xffffu, eg1 = qs_g >> 16;
    const uint32_t eu0 = qs_u & 0xffffu, eu1 = qs_u >> 16;
    r.g0 = s_grid2xs[eg0 & 511u]; r.s0 = s_sign_pair128[eg0 >> 9];
    r.g1 = s_grid2xs[eg1 & 511u]; r.s1 = s_sign_pair128[eg1 >> 9];
    r.u0 = s_grid2xs[eu0 & 511u]; r.t0 = s_sign_pair128[eu0 >> 9];
    r.u1 = s_grid2xs[eu1 & 511u]; r.t1 = s_sign_pair128[eu1 >> 9];
    return r;
}

// ABL (ablation, bench-only): 0 = real kernel; 1 = no WMMA (fragments
// folded into acc with one add so the loads stay live); 2 = no dequant (A
// = lane constant, no qs loads, no LUT reads); 3 = no x staging (B reads
// whatever LDS holds); 4 = no B loads (B = constant).
template <int DEQ, int NT, int ABL>
__device__ __forceinline__ void iq2xs_wmma_mainloop(
        iq2xs_float8 (&acc_g)[2], iq2xs_float8 (&acc_u)[2],
        const uint8_t* __restrict__ gate_row, const uint8_t* __restrict__ up_row,
        const uint16_t* __restrict__ x16,
        uint16_t* s_x, const int* s_member_packed,
        const uint64_t* s_grid2xs, const uint64_t* s_sign_pair128,
        const uint4* s_grid16, const uint4* s_sign16,
        unsigned int n_members, unsigned int n_blocks, bool row_valid,
        unsigned int tid, unsigned int l16) {
    const size_t k_total = (size_t)n_blocks * 256u;

    for (unsigned int sb = 0; sb < n_blocks; sb++) {
        // --- Per window of IQ2XS_WM_KWIN k: stage every member's slice (zeros
        //     past n_members), then run its k-steps. 32 members x
        //     (KWIN/8) b128 chunks per window, spread over 256 threads.
        constexpr unsigned int KSTEPS_PER_WIN = IQ2XS_WM_KWIN / 16u;
        constexpr unsigned int CHUNKS_PER_ROW = IQ2XS_WM_KWIN / 8u;           // b128 per member row
        constexpr unsigned int CHUNKS_PER_WIN = IQ2XS_WM_MAX_CHUNK * CHUNKS_PER_ROW;
        constexpr unsigned int CHUNKS_PER_THREAD = (CHUNKS_PER_WIN + 255u) / 256u;
        // --- Per-super-block row scales (0.125 * d), one per matrix. The qs
        //     and scale bytes are loaded per WINDOW below (2 x b128 + 1 x b32
        //     per matrix at KWIN=128) so every register index inside the
        //     unrolled window body is static; loading the whole 74-B block per
        //     super-block and slicing it by a runtime window index made the
        //     compiler spill 466 dwords.
        const uint8_t* gate_block = gate_row + (uint64_t)sb * BLOCK_IQ2_XS_BYTES;
        const uint8_t* up_block   = up_row   + (uint64_t)sb * BLOCK_IQ2_XS_BYTES;
        float dsc_g = 0.0f, dsc_u = 0.0f;
        if (row_valid) {
            uint16_t wd_g, wd_u;
            __builtin_memcpy(&wd_g, gate_block, 2);
            __builtin_memcpy(&wd_u, up_block, 2);
            dsc_g = 0.125f * f16_to_f32_xs(wd_g);
            dsc_u = 0.125f * f16_to_f32_xs(wd_u);
        }

        constexpr unsigned int QS_U4_PER_WIN = IQ2XS_WM_KWIN / 64u;   // b128 of qs per window (2 at KWIN=128)
        #pragma unroll 1
        for (unsigned int win = 0; win < 256u / IQ2XS_WM_KWIN; win++) {
        // This window's qs (KWIN/16 u32 = KWIN/64 x b128) + its scale bytes
        // (KWIN/32 bytes, loaded as one u32) per matrix. Issued before the
        // staging barrier so the latency overlaps it.
        constexpr unsigned int SC_BYTES_PER_WIN = IQ2XS_WM_KWIN / 32u;   // scale bytes per window (8 at KWIN=256)
        static_assert(SC_BYTES_PER_WIN <= 8u, "scale window must fit a uint2");
        uint4 gq[QS_U4_PER_WIN], uq[QS_U4_PER_WIN];
        uint2 gsc = make_uint2(0u, 0u), usc = make_uint2(0u, 0u);
        #pragma unroll
        for (unsigned int i = 0; i < QS_U4_PER_WIN; i++) { gq[i] = make_uint4(0u, 0u, 0u, 0u); uq[i] = gq[i]; }
        if (row_valid) {
            #pragma unroll
            for (unsigned int i = 0; i < QS_U4_PER_WIN; i++) {
                __builtin_memcpy(&gq[i], gate_block + 2 + (win * QS_U4_PER_WIN + i) * 16u, 16);
                __builtin_memcpy(&uq[i], up_block   + 2 + (win * QS_U4_PER_WIN + i) * 16u, 16);
            }
            __builtin_memcpy(&gsc, gate_block + 66 + win * SC_BYTES_PER_WIN, SC_BYTES_PER_WIN);
            __builtin_memcpy(&usc, up_block   + 66 + win * SC_BYTES_PER_WIN, SC_BYTES_PER_WIN);
        }
        if (ABL != 3) {
            #pragma unroll
            for (unsigned int j = 0; j < CHUNKS_PER_THREAD; j++) {
                const unsigned int idx = tid + 256u * j;
                if (idx < CHUNKS_PER_WIN) {
                    const unsigned int mi  = idx / CHUNKS_PER_ROW;
                    const unsigned int c   = idx % CHUNKS_PER_ROW;
                    uint4 v = make_uint4(0u, 0u, 0u, 0u);
                    if (mi < n_members) {
                        const unsigned int b = (unsigned int)(s_member_packed[mi] >> 16);
                        const uint4* src = (const uint4*)(x16 + (size_t)b * k_total
                                                              + (size_t)sb * 256u + (size_t)win * IQ2XS_WM_KWIN
                                                              + (size_t)c * 8u);
                        v = *src;
                    }
                    *(uint4*)(&s_x[mi * IQ2XS_WM_X_STRIDE + c * 8u]) = v;
                }
            }
        }
        __syncthreads();
        iq2xs_raw_frag raw_cur, raw_nxt;
        iq2xs_raw8 r8_cur, r8_nxt;
        iq2xs_half16 bf_cur[2], bf_nxt[2];
        if (DEQ == 1) {
            raw_cur = iq2xs_gather_raw(s_grid16, s_sign16, gq[0].x, uq[0].x);   // k-step 0 of this window
        } else if (DEQ == 2) {
            r8_cur = iq2xs_gather_raw8(s_grid2xs, s_sign_pair128, gq[0].x, uq[0].x);
            #pragma unroll
            for (unsigned int nt = 0; nt < 2u; nt++) {
                if (nt < (unsigned int)NT && ABL != 4) {
                    const uint16_t* xb = &s_x[(nt * 16u + l16) * IQ2XS_WM_X_STRIDE];
                    __builtin_memcpy(&bf_cur[nt], xb, 32);
                }
            }
        }
        #pragma unroll
        for (unsigned int kw = 0; kw < KSTEPS_PER_WIN; kw++) {
            const unsigned int ks = win * KSTEPS_PER_WIN + kw;
            const unsigned int ib32 = ks >> 1;
            const unsigned int half = ks & 1u;

            // --- A fragments: this lane's 16 gate + 16 up weights of row_a.
            //     qs u32 #ks of the block = entries 4*ib32 + 2*half + {0,1};
            //     scale byte #ib32.
            iq2xs_half16 a_g, a_u;
            const uint32_t qs_g = (kw & 3u) == 0u ? gq[kw >> 2].x : (kw & 3u) == 1u ? gq[kw >> 2].y
                                : (kw & 3u) == 2u ? gq[kw >> 2].z : gq[kw >> 2].w;
            const uint32_t qs_u = (kw & 3u) == 0u ? uq[kw >> 2].x : (kw & 3u) == 1u ? uq[kw >> 2].y
                                : (kw & 3u) == 2u ? uq[kw >> 2].z : uq[kw >> 2].w;
            // scale byte (kw>>1) of this window's SC_BYTES_PER_WIN bytes (static: kw unrolled)
            const unsigned int scb = kw >> 1;
            const uint8_t sc_g = (uint8_t)(((scb < 4u) ? gsc.x : gsc.y) >> (8u * (scb & 3u)));
            const uint8_t sc_u = (uint8_t)(((scb < 4u) ? usc.x : usc.y) >> (8u * (scb & 3u)));
            (void)ib32;
            if (ABL == 2) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    a_g[i] = (_Float16)(0.001f * (float)(l16 + i));
                    a_u[i] = (_Float16)(0.002f * (float)(l16 + i));
                }
            } else {
            const int32_t ls_g = 1 + 2 * (int32_t)(half ? (sc_g >> 4) : (sc_g & 0x0F));
            const int32_t ls_u = 1 + 2 * (int32_t)(half ? (sc_u >> 4) : (sc_u & 0x0F));
            const float sg = dsc_g * (float)ls_g;   // 0 for invalid rows (dsc=0)
            const float su = dsc_u * (float)ls_u;

            if (DEQ == 2) {
                if (kw + 1u < KSTEPS_PER_WIN) {   // prefetch next k-step's rows (WMMAs below cover it)
                    const unsigned int kn = kw + 1u;
                    const uint32_t nqs_g = (kn & 3u) == 0u ? gq[kn >> 2].x : (kn & 3u) == 1u ? gq[kn >> 2].y
                                         : (kn & 3u) == 2u ? gq[kn >> 2].z : gq[kn >> 2].w;
                    const uint32_t nqs_u = (kn & 3u) == 0u ? uq[kn >> 2].x : (kn & 3u) == 1u ? uq[kn >> 2].y
                                         : (kn & 3u) == 2u ? uq[kn >> 2].z : uq[kn >> 2].w;
                    r8_nxt = iq2xs_gather_raw8(s_grid2xs, s_sign_pair128, nqs_g, nqs_u);
                }
                a_g = iq2xs_magic_frag(r8_cur.g0, r8_cur.s0, r8_cur.g1, r8_cur.s1, (_Float16)sg);
                a_u = iq2xs_magic_frag(r8_cur.u0, r8_cur.t0, r8_cur.u1, r8_cur.t1, (_Float16)su);
            } else if (DEQ == 1) {
                // Pipelined: raw_cur was gathered during the previous k-step
                // (or the prologue); gather the NEXT k-step's rows now so the
                // WMMAs below cover their latency.
                if (kw + 1u < KSTEPS_PER_WIN) {
                    const unsigned int kn = kw + 1u;
                    const uint32_t nqs_g = (kn & 3u) == 0u ? gq[kn >> 2].x : (kn & 3u) == 1u ? gq[kn >> 2].y
                                         : (kn & 3u) == 2u ? gq[kn >> 2].z : gq[kn >> 2].w;
                    const uint32_t nqs_u = (kn & 3u) == 0u ? uq[kn >> 2].x : (kn & 3u) == 1u ? uq[kn >> 2].y
                                         : (kn & 3u) == 2u ? uq[kn >> 2].z : uq[kn >> 2].w;
                    raw_nxt = iq2xs_gather_raw(s_grid16, s_sign16, nqs_g, nqs_u);
                }
                iq2xs_finish_frag(raw_cur, (_Float16)sg, (_Float16)su, a_g, a_u);
            } else {
                int32_t wg0, wg1, wg2, wg3, wu0, wu1, wu2, wu3;
                {
                    const uint32_t e0 = qs_g & 0xffffu, e1 = qs_g >> 16;
                    const uint64_t g0 = s_grid2xs[e0 & 511u], sp0 = s_sign_pair128[e0 >> 9];
                    const uint64_t g1 = s_grid2xs[e1 & 511u], sp1 = s_sign_pair128[e1 >> 9];
                    wg0 = cond_neg_bytes_fast((int32_t)(uint32_t)g0,         (uint32_t)sp0);
                    wg1 = cond_neg_bytes_fast((int32_t)(uint32_t)(g0 >> 32), (uint32_t)(sp0 >> 32));
                    wg2 = cond_neg_bytes_fast((int32_t)(uint32_t)g1,         (uint32_t)sp1);
                    wg3 = cond_neg_bytes_fast((int32_t)(uint32_t)(g1 >> 32), (uint32_t)(sp1 >> 32));
                }
                {
                    const uint32_t e0 = qs_u & 0xffffu, e1 = qs_u >> 16;
                    const uint64_t g0 = s_grid2xs[e0 & 511u], sp0 = s_sign_pair128[e0 >> 9];
                    const uint64_t g1 = s_grid2xs[e1 & 511u], sp1 = s_sign_pair128[e1 >> 9];
                    wu0 = cond_neg_bytes_fast((int32_t)(uint32_t)g0,         (uint32_t)sp0);
                    wu1 = cond_neg_bytes_fast((int32_t)(uint32_t)(g0 >> 32), (uint32_t)(sp0 >> 32));
                    wu2 = cond_neg_bytes_fast((int32_t)(uint32_t)g1,         (uint32_t)sp1);
                    wu3 = cond_neg_bytes_fast((int32_t)(uint32_t)(g1 >> 32), (uint32_t)(sp1 >> 32));
                }
                a_g = iq2xs_i8x16_to_h16(wg0, wg1, wg2, wg3, sg);
                a_u = iq2xs_i8x16_to_h16(wu0, wu1, wu2, wu3, su);
            }
            } // ABL != 2

            // --- B fragments + WMMA per N-tile.
            #pragma unroll
            for (unsigned int nt = 0; nt < 2u; nt++) {
                if (nt < (unsigned int)NT) {
                    iq2xs_half16 bf;
                    if (ABL == 4) {
                        #pragma unroll
                        for (int i = 0; i < 16; i++) bf[i] = (_Float16)(0.01f * (float)(i + ks));
                    } else if (DEQ == 2) {
                        // B for this k-step was loaded one k-step ago; load the next one now.
                        bf = bf_cur[nt];
                        if (kw + 1u < KSTEPS_PER_WIN) {
                            const uint16_t* xb = &s_x[(nt * 16u + l16) * IQ2XS_WM_X_STRIDE + (kw + 1u) * 16u];
                            __builtin_memcpy(&bf_nxt[nt], xb, 32);
                        }
                    } else {
                        const uint16_t* xb = &s_x[(nt * 16u + l16) * IQ2XS_WM_X_STRIDE + kw * 16u];
                        const uint4 b0 = *(const uint4*)(xb);
                        const uint4 b1 = *(const uint4*)(xb + 8);
                        __builtin_memcpy(((char*)&bf) +  0, &b0, 16);
                        __builtin_memcpy(((char*)&bf) + 16, &b1, 16);
                    }
                    if (ABL == 1) {
                        acc_g[nt][0] += (float)a_g[ks & 15u] + (float)bf[0];
                        acc_u[nt][0] += (float)a_u[ks & 15u] + (float)bf[1];
                    } else {
                        acc_g[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_g, bf, acc_g[nt]);
                        acc_u[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_u, bf, acc_u[nt]);
                    }
                }
            }
            if (DEQ == 1) { raw_cur = raw_nxt; }
            if (DEQ == 2) { r8_cur = r8_nxt; bf_cur[0] = bf_nxt[0]; bf_cur[1] = bf_nxt[1]; }
        }
        __syncthreads();   // s_x reuse by the next window
        }
    }

}

template <int DEQ, int ABL, int OUT16>
__device__ __forceinline__ void iq2xs_wmma_body(
        float          * __restrict__ mid,           // OUT16=0: f32 [B, n_used, n_rows]; OUT16=1: reinterpreted as f16
        const uint8_t  * __restrict__ gate_w_base,
        const uint8_t  * __restrict__ up_w_base,
        const uint16_t * __restrict__ x16,
        const float    * __restrict__ expert_w,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        const int      * __restrict__ work_items,
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        unsigned int    n_used,
        unsigned int    max_per_expert,
        unsigned int    chunk_size,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
#if defined(IQ2XS_WMMA_RDNA3)
    const unsigned int w_idx = blockIdx.y;
    const int packed_item = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;
    const unsigned int n_members = member_end - member_start;
    const unsigned int n_tiles = (n_members + 15u) >> 4;     // 1 or 2

    // DEQ=0 tables (i8 grid + byte sign masks) or DEQ=1 tables (f16 grid +
    // f16 sign masks). Only one set is instantiated per symbol.
    __shared__ uint64_t s_sign_pair128[DEQ == 1 ? 1 : 128];
    __shared__ uint64_t s_grid2xs[DEQ == 1 ? 1 : 512];
    __shared__ __attribute__((aligned(16))) uint4 s_grid16[DEQ == 1 ? 512 : 1];
    __shared__ __attribute__((aligned(16))) uint4 s_sign16[DEQ == 1 ? 128 : 1];
    __shared__ __attribute__((aligned(16))) uint16_t s_x[IQ2XS_WM_MAX_CHUNK * IQ2XS_WM_X_STRIDE];
    __shared__ int      s_member_packed[IQ2XS_WM_MAX_CHUNK];

    const unsigned int tid = threadIdx.x;
    if (DEQ == 1) {
        IQ2XS_STAGE_TABLES_F16(s_grid16, s_sign16);
    } else {
        IQ2XS_STAGE_TABLES(s_sign_pair128, s_grid2xs);
        if (DEQ == 2) {
            __syncthreads();
            if (tid < 128u) s_sign_pair128[tid] ^= 0x8080808080808080ull;   // biased masks
        }
    }
    for (unsigned int i = tid; i < IQ2XS_WM_MAX_CHUNK; i += blockDim.x) {
        s_member_packed[i] = (i < n_members)
            ? expert_members[(size_t)e * (size_t)max_per_expert + (size_t)(member_start + i)]
            : 0;
    }
    __syncthreads();

    const unsigned int wave = tid >> 5;                    // 0..7
    const unsigned int lane = tid & 31u;
    const unsigned int l16  = lane & 15u;
    const unsigned int hi   = lane >> 4;                   // C-fragment row parity
    const unsigned int m0   = blockIdx.x * IQ2XS_WM_ROWS_PER_WG + wave * 16u;
    const unsigned int row_a = m0 + l16;                   // row this lane dequantizes
    const bool row_valid = (row_a < n_rows);

    const uint8_t* gate_row = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe
                            + (uint64_t)row_a * (uint64_t)n_blocks * BLOCK_IQ2_XS_BYTES;
    const uint8_t* up_row   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe
                            + (uint64_t)row_a * (uint64_t)n_blocks * BLOCK_IQ2_XS_BYTES;

    iq2xs_float8 acc_g[2], acc_u[2];
    #pragma unroll
    for (unsigned int nt = 0; nt < 2u; nt++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) { acc_g[nt][i] = 0.0f; acc_u[nt][i] = 0.0f; }
    }

    // Tile count is a template parameter of the main loop: a runtime
    // `if (nt < n_tiles)` inside the k loop made the compiler materialise
    // ~64 accumulator copies per k-step (phi nodes across the branch).
    if (n_tiles == 2u) {
        iq2xs_wmma_mainloop<DEQ, 2, ABL>(acc_g, acc_u, gate_row, up_row, x16, s_x,
                                    s_member_packed, s_grid2xs, s_sign_pair128,
                                    s_grid16, s_sign16, n_members, n_blocks, row_valid, tid, l16);
    } else {
        iq2xs_wmma_mainloop<DEQ, 1, ABL>(acc_g, acc_u, gate_row, up_row, x16, s_x,
                                    s_member_packed, s_grid2xs, s_sign_pair128,
                                    s_grid16, s_sign16, n_members, n_blocks, row_valid, tid, l16);
    }

    // --- Epilogue: C lane l elem i = [row 2i+hi][member l16].
    #pragma unroll
    for (unsigned int nt = 0; nt < 2u; nt++) {
        const unsigned int mi = nt * 16u + l16;
        if (nt < n_tiles && mi < n_members) {
            const int packed        = s_member_packed[mi];
            const unsigned int b    = (unsigned int)(packed >> 16);
            const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
            const float ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
            float* out_base = mid + ((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const unsigned int row = m0 + 2u * (unsigned int)i + hi;
                if (row < n_rows) {
                    float g = acc_g[nt][i];
                    float u = acc_u[nt][i];
                    if (clamp > 1.0e-6f) {
                        if (g > clamp)  g = clamp;
                        if (u > clamp)  u = clamp;
                        if (u < -clamp) u = -clamp;
                    }
                    const float sig = 1.0f / (1.0f + expf(-g));
                    const float v = g * sig * u * ew;
                    if (OUT16) {
                        ((uint16_t*)mid)[((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows + row]
                            = __half_as_ushort(__float2half(v));
                    } else {
                        out_base[row] = v;
                    }
                }
            }
        }
    }
#else
    // Non-RDNA3 targets: no WMMA path. Trap loudly rather than return
    // garbage; the wrapper only dispatches it on gfx11.
    __builtin_trap();
#endif
}

extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_wmma_lut(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base,
        const uint8_t* __restrict__ up_w_base, const uint16_t* __restrict__ x16,
        const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,
        unsigned int n_rows, unsigned int n_blocks) {
    iq2xs_wmma_body<1, 0, 0>(mid, gate_w_base, up_w_base, x16, expert_w, group_count,
                       expert_members, work_items, gate_bpe, up_bpe, n_used,
                       max_per_expert, chunk_size, clamp, n_rows, n_blocks);
}

extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_wmma_cvt(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base,
        const uint8_t* __restrict__ up_w_base, const uint16_t* __restrict__ x16,
        const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,
        unsigned int n_rows, unsigned int n_blocks) {
    iq2xs_wmma_body<0, 0, 0>(mid, gate_w_base, up_w_base, x16, expert_w, group_count,
                       expert_members, work_items, gate_bpe, up_bpe, n_used,
                       max_per_expert, chunk_size, clamp, n_rows, n_blocks);
}

extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_wmma(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base,
        const uint8_t* __restrict__ up_w_base, const uint16_t* __restrict__ x16,
        const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,
        unsigned int n_rows, unsigned int n_blocks) {
    iq2xs_wmma_body<2, 0, 0>(mid, gate_w_base, up_w_base, x16, expert_w, group_count,
                       expert_members, work_items, gate_bpe, up_bpe, n_used,
                       max_per_expert, chunk_size, clamp, n_rows, n_blocks);
}

// Production twin: f16 mid out ([B*n_used][n_rows] halves, = the down
// kernel's B operand), otherwise identical to _wmma.
extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_wmma_h(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base,
        const uint8_t* __restrict__ up_w_base, const uint16_t* __restrict__ x16,
        const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,
        unsigned int n_rows, unsigned int n_blocks) {
    iq2xs_wmma_body<2, 0, 1>(mid, gate_w_base, up_w_base, x16, expert_w, group_count,
                       expert_members, work_items, gate_bpe, up_bpe, n_used,
                       max_per_expert, chunk_size, clamp, n_rows, n_blocks);
}

// bench-only ablation 1 (see ABL above)
extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_wmma_abl1(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base,
        const uint8_t* __restrict__ up_w_base, const uint16_t* __restrict__ x16,
        const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,
        unsigned int n_rows, unsigned int n_blocks) {
    iq2xs_wmma_body<2, 1, 0>(mid, gate_w_base, up_w_base, x16, expert_w, group_count,
                       expert_members, work_items, gate_bpe, up_bpe, n_used,
                       max_per_expert, chunk_size, clamp, n_rows, n_blocks);
}

// bench-only ablation 2 (see ABL above)
extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_wmma_abl2(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base,
        const uint8_t* __restrict__ up_w_base, const uint16_t* __restrict__ x16,
        const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,
        unsigned int n_rows, unsigned int n_blocks) {
    iq2xs_wmma_body<2, 2, 0>(mid, gate_w_base, up_w_base, x16, expert_w, group_count,
                       expert_members, work_items, gate_bpe, up_bpe, n_used,
                       max_per_expert, chunk_size, clamp, n_rows, n_blocks);
}

// bench-only ablation 3 (see ABL above)
extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_wmma_abl3(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base,
        const uint8_t* __restrict__ up_w_base, const uint16_t* __restrict__ x16,
        const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,
        unsigned int n_rows, unsigned int n_blocks) {
    iq2xs_wmma_body<2, 3, 0>(mid, gate_w_base, up_w_base, x16, expert_w, group_count,
                       expert_members, work_items, gate_bpe, up_bpe, n_used,
                       max_per_expert, chunk_size, clamp, n_rows, n_blocks);
}

// bench-only ablation 4 (see ABL above)
extern "C" __global__ void iq2_xs_pair_matvec_fused_swiglu_wmma_abl4(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base,
        const uint8_t* __restrict__ up_w_base, const uint16_t* __restrict__ x16,
        const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,
        unsigned int n_rows, unsigned int n_blocks) {
    iq2xs_wmma_body<2, 4, 0>(mid, gate_w_base, up_w_base, x16, expert_w, group_count,
                       expert_members, work_items, gate_bpe, up_bpe, n_used,
                       max_per_expert, chunk_size, clamp, n_rows, n_blocks);
}

// -----------------------------------------------------------------------------
// Layout probe (bench-only): which (lane-half, element-range) groups of the
// A / B operands does gfx11 wave32 WMMA actually read? For each operand and
// each of the 4 groups {lo lanes e0-7, lo lanes e8-15, hi lanes e0-7, hi lanes
// e8-15} the group is replaced by garbage; a D identical to the reference
// means that group is NOT read. out = 9 tiles of 256 f32, row-major:
// [0]=reference, [1..4]=A groups 0..3, [5..8]=B groups 0..3.
// -----------------------------------------------------------------------------
extern "C" __global__ void iq2_xs_wmma_upper_lane_probe(float* __restrict__ out) {
#if defined(IQ2XS_WMMA_RDNA3)
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int l16 = lane & 15u;
    const unsigned int hi = lane >> 4;
    iq2xs_half16 a, b;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        a[k] = (_Float16)(0.25f * (float)((l16 * 7 + k * 3) % 11) - 1.0f);   // A[m=l16][k]
        b[k] = (_Float16)(0.5f  * (float)((l16 * 5 + k * 2) % 7)  - 1.5f);   // B[k][n=l16]
    }
    iq2xs_float8 c;
    #pragma unroll
    for (int i = 0; i < 8; i++) c[i] = 0.0f;
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    #pragma unroll
    for (int i = 0; i < 8; i++) out[(2u * (unsigned int)i + hi) * 16u + l16] = c[i];
    #pragma unroll
    for (int op = 0; op < 2; op++) {
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            const unsigned int g_hi = (unsigned int)(g >> 1);   // which lane half
            const int e_lo = (g & 1) ? 8 : 0;                    // which element range
            iq2xs_half16 ag = a, bg = b;
            if (hi == g_hi) {
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (op == 0) ag[e_lo + k] = (_Float16)(41.0f + (float)k);
                    else         bg[e_lo + k] = (_Float16)(-23.0f - (float)k);
                }
            }
            iq2xs_float8 d;
            #pragma unroll
            for (int i = 0; i < 8; i++) d[i] = 0.0f;
            d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ag, bg, d);
            const unsigned int tile = 1u + (unsigned int)(op * 4 + g);
            #pragma unroll
            for (int i = 0; i < 8; i++) out[tile * 256u + (2u * (unsigned int)i + hi) * 16u + l16] = d[i];
        }
    }
#else
    if (threadIdx.x == 0) out[0] = -1.0f;
#endif
}
