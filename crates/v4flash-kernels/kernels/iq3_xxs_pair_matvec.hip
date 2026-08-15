// iq3_xxs_pair_matvec — gate+up fused-SwiGLU kernels for IQ3_XXS weights.
//
// unsloth UD-Q2_K_XL puts blk.26's ffn_gate_exps + ffn_up_exps at IQ3_XXS
// (every other layer's gate/up is IQ2_XS). IQ3_XXS already ships in tree
// for the DOWN projection (iq3_xxs_matvec.hip) — down is a single matrix,
// so it never needed the paired form. This file supplies exactly that:
// the same block dot, restructured so gate and up share one xq read, one
// LDS table staging and a fused SwiGLU.
//
// Block dot, tables and helpers are lifted verbatim from
// iq3_xxs_matvec.hip; the kernel skeletons mirror iq2_xs_pair_matvec.hip.
// Only dot_block_half_iq3 is new, and it is dot_block_quarter_iq3 widened
// from 2 to 4 ib32 groups to match the pair kernels' lane mapping.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_IQ3_XXS_BYTES 98
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

#include "iq2_xxs_tables.inc"   // ksigns_iq2xs (shared sign codebook)
#include "iq3_xxs_tables.inc"   // iq3xxs_grid

__device__ __forceinline__ float f16_to_f32_par(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4_par(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t load_dword_i8_par(const int8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

// Even-offset u16 load (all IQ3_XXS intra-block fields are 2-aligned).
__device__ __forceinline__ uint32_t load_u16_par(const uint8_t* p) {
    uint16_t v;
    __builtin_memcpy(&v, p, 2);
    return (uint32_t)v;
}

__device__ __forceinline__ uint32_t load_u32_2aligned_par(const uint8_t* p) {
    return load_u16_par(p) | (load_u16_par(p + 2) << 16);
}

__device__ __forceinline__ uint32_t expand_signs_par(uint32_t s, uint32_t mask) {
    const uint32_t m = s & mask;
    uint32_t out = 0;
    if (m & 0xffu)         out |= 0xffu;
    if (m & 0xff00u)       out |= 0xff00u;
    if (m & 0xff0000u)     out |= 0xff0000u;
    if (m & 0xff000000u)   out |= 0xff000000u;
    return out;
}

__device__ __forceinline__ int32_t cond_neg_bytes_par(int32_t pack, uint32_t sm) {
    union { int32_t i; int8_t b[4]; } in, out;
    in.i = pack;
    out.b[0] = (sm & 0xffu)       ? (int8_t)(-in.b[0]) : in.b[0];
    out.b[1] = (sm & 0xff00u)     ? (int8_t)(-in.b[1]) : in.b[1];
    out.b[2] = (sm & 0xff0000u)   ? (int8_t)(-in.b[2]) : in.b[2];
    out.b[3] = (sm & 0xff000000u) ? (int8_t)(-in.b[3]) : in.b[3];
    return out.i;
}

// LDS table staging (M14k/M31 lesson from the iq2 family: divergent
// indexed reads of __constant__ tables serialize; stage once per WG).
// s_sign_pair[i] = pre-expanded (sm1 << 32) | sm0 byte masks for sign
// index i; s_grid3 = the 1 KiB iq3xxs codebook.
#define IQ3_STAGE_TABLES(s_sign_pair, s_grid3)                              \
    {                                                                        \
        const unsigned int tid_ = threadIdx.x;                               \
        if (tid_ < 128u) {                                                   \
            const uint32_t signs_ = ksigns_iq2xs[tid_] * 0x01010101u;        \
            const uint32_t sm0_ = expand_signs_par(signs_, 0x08040201u);     \
            const uint32_t sm1_ = expand_signs_par(signs_, 0x80402010u);     \
            s_sign_pair[tid_] = ((uint64_t)sm1_ << 32) | (uint64_t)sm0_;     \
        }                                                                    \
        if (tid_ < 256u) s_grid3[tid_] = iq3xxs_grid[tid_];                  \
        __syncthreads();                                                     \
    }

// One l-subgroup (8 weights): two grid dwords + one 7-bit sign index.
__device__ __forceinline__ int32_t iq3_dp4a_8_par(
        uint32_t grid1, uint32_t grid2, uint64_t sign_pair,
        const int8_t* q8, int32_t acc) {
    const uint32_t sm0 = (uint32_t)sign_pair;
    const uint32_t sm1 = (uint32_t)(sign_pair >> 32);
    const int32_t g0 = cond_neg_bytes_par((int32_t)grid1, sm0);
    const int32_t g1 = cond_neg_bytes_par((int32_t)grid2, sm1);
    acc = sudot4_par(g0, load_dword_i8_par(q8 + 0), acc);
    acc = sudot4_par(g1, load_dword_i8_par(q8 + 4), acc);
    return acc;
}

// This lane's quarter of one super-block: ib32 ∈ {2·sub, 2·sub+1}.
// Returns 0.25 * d * partial_bsum; summing the 4 subs reproduces the
// reference block result exactly (integer parts are bit-identical).
__device__ __forceinline__ float dot_block_quarter_iq3(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int sub,
        const uint64_t* s_sign_pair, const uint32_t* s_grid3) {
    const uint16_t wd_bits = (uint16_t)load_u16_par(w_blk);
    const uint8_t* q3_base  = w_blk + 2;
    const uint8_t* gas_base = w_blk + 2 + 64;

    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8_base = (const int8_t*)(y_blk + 4);

    const float d = f16_to_f32_par(wd_bits) * yd;

    int32_t bsum = 0;
    #pragma unroll
    for (unsigned int t = 0; t < 2u; t++) {
        const unsigned int ib32 = 2u * sub + t;
        const uint32_t aux32 = load_u32_2aligned_par(gas_base + 4u * ib32);
        const int32_t ls = (int32_t)(2u * (aux32 >> 28) + 1u);
        const uint8_t* q3 = q3_base + ib32 * 8u;
        const int8_t*  q8 = q8_base + ib32 * 32u;
        int32_t sumi = 0;
        #pragma unroll
        for (unsigned int l = 0; l < 4u; l++) {
            sumi = iq3_dp4a_8_par(
                s_grid3[q3[2u * l]],
                s_grid3[q3[2u * l + 1u]],
                s_sign_pair[(aux32 >> (7u * l)) & 127u],
                q8 + 8u * l,
                sumi);
        }
        bsum += sumi * ls;
    }
    return 0.25f * d * (float)bsum;
}

// This lane's HALF of one super-block: ib32 in [ib32_lo, ib32_lo+4).
// The pair kernels split a block across a lane PAIR (2 halves), where the
// down-projection kernels split it across four lanes (quarters) — same
// arithmetic, different partition. Returns 0.25 * d * partial_bsum.
__device__ __forceinline__ float dot_block_half_iq3(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int ib32_lo,
        const uint64_t* s_sign_pair, const uint32_t* s_grid3) {
    const uint16_t wd_bits = (uint16_t)load_u16_par(w_blk);
    const uint8_t* q3_base  = w_blk + 2;
    const uint8_t* gas_base = w_blk + 2 + 64;

    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8_base = (const int8_t*)(y_blk + 4);
    const float d = f16_to_f32_par(wd_bits) * yd;

    int32_t bsum = 0;
    #pragma unroll
    for (unsigned int t = 0; t < 4u; t++) {
        const unsigned int ib32 = ib32_lo + t;
        const uint32_t aux32 = load_u32_2aligned_par(gas_base + 4u * ib32);
        const int32_t ls = (int32_t)(2u * (aux32 >> 28) + 1u);
        const uint8_t* q3 = q3_base + ib32 * 8u;
        const int8_t*  q8 = q8_base + ib32 * 32u;
        int32_t sumi = 0;
        #pragma unroll
        for (unsigned int l = 0; l < 4u; l++) {
            sumi = iq3_dp4a_8_par(
                s_grid3[q3[2u * l]],
                s_grid3[q3[2u * l + 1u]],
                s_sign_pair[(aux32 >> (7u * l)) & 127u],
                q8 + 8u * l,
                sumi);
        }
        bsum += sumi * ls;
    }
    return 0.25f * d * (float)bsum;
}

__device__ __forceinline__ float warp_sum_f32_par(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// Decode: batched over n_used slots, fused SwiGLU + expert weight.
// Contract identical to iq2_xxs_pair_matvec_fused_swiglu_batch.
extern "C" __global__ void iq3_xxs_pair_matvec_fused_swiglu_batch(
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
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint32_t s_grid3[256];
    __shared__ uint8_t  s_xq[16 * 292];
    IQ3_STAGE_TABLES(s_sign_pair, s_grid3);
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
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_XXS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_XXS_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ3_XXS_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ3_XXS_BYTES;
        const uint8_t* y   = s_xq          + (size_t)b   * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half_iq3(w_g, y, ib32_lo, s_sign_pair, s_grid3);
        u_sum += dot_block_half_iq3(w_u, y, ib32_lo, s_sign_pair, s_grid3);
    }

    g_sum = warp_sum_f32_par(g_sum);
    u_sum = warp_sum_f32_par(u_sum);

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
extern "C" __global__ void iq3_xxs_pair_matvec_fused_swiglu_batch_hetsplit(
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
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint32_t s_grid3[256];
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
    const int e = (mode == 0u) ? (-dense - 1) : dense;

    IQ3_STAGE_TABLES(s_sign_pair, s_grid3);
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
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_XXS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_XXS_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ3_XXS_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ3_XXS_BYTES;
        const uint8_t* y   = s_xq          + (size_t)b   * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half_iq3(w_g, y, ib32_lo, s_sign_pair, s_grid3);
        u_sum += dot_block_half_iq3(w_u, y, ib32_lo, s_sign_pair, s_grid3);
    }

    g_sum = warp_sum_f32_par(g_sum);
    u_sum = warp_sum_f32_par(u_sum);

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
extern "C" __global__ void iq3_xxs_pair_matvec_fused_swiglu_chunked(
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

    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint32_t s_grid3[256];
    __shared__ uint8_t  s_xq[16 * 292];
    IQ3_STAGE_TABLES(s_sign_pair, s_grid3);
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
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_XXS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_XXS_BYTES;

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
                const uint8_t* w_g = gate_row_base + (uint64_t)bl * BLOCK_IQ3_XXS_BYTES;
                const uint8_t* w_u = up_row_base   + (uint64_t)bl * BLOCK_IQ3_XXS_BYTES;
                const uint8_t* y   = s_xq          + (size_t)bl   * BLOCK_Q8_K_BYTES;
                g_sum += dot_block_half_iq3(w_g, y, ib32_lo, s_sign_pair, s_grid3);
                u_sum += dot_block_half_iq3(w_u, y, ib32_lo, s_sign_pair, s_grid3);
            }

            g_sum = warp_sum_f32_par(g_sum);
            u_sum = warp_sum_f32_par(u_sum);

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
// kwide — M51-structure port for IQ3_XXS gate/up (2026-08-15).
//
// Same rationale as iq2_xs_pair_matvec_fused_swiglu_kwide: the `_chunked`
// kernel re-reads + re-dequantizes the expert's weight rows once PER
// MEMBER; kwide dequantizes each lane's 16 weights once per sb-pair and
// amortizes across all staged members (blk.26 only in UD-Q2_K_XL, but it
// otherwise serializes the layer the same way).
//
// Lane dequant: ib32 = sub_block, l = 2*half, 2*half+1. One u32 of q3
// (4 grid bytes), one u32 aux (two 7-bit sign indices + shared 4-bit
// scale). ls is per-ib32 (shared by the lane pair) — folding it per lane
// is exact.
// =============================================================================

#define IQ3_KW_MAX_CHUNK 32u
#ifndef IQ3_KW_UNROLL
#define IQ3_KW_UNROLL 4
#endif

extern "C" __global__ void iq3_xxs_pair_matvec_fused_swiglu_kwide(
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

    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint32_t s_grid3[256];
    __shared__ uint4    s_q8v[IQ3_KW_MAX_CHUNK * 2 * 16]; // [mi][sbh][16]
    __shared__ float    s_yd[IQ3_KW_MAX_CHUNK * 2];       // [mi][sbh]
    __shared__ int      s_member_packed[IQ3_KW_MAX_CHUNK];

    const unsigned int tid = threadIdx.x;
    // IQ3_STAGE_TABLES ends with __syncthreads(); member staging joins it.
    for (unsigned int i = tid; i < n_members; i += blockDim.x) {
        s_member_packed[i] = expert_members[(size_t)e * (size_t)max_per_expert
                                          + (size_t)(member_start + i)];
    }
    IQ3_STAGE_TABLES(s_sign_pair, s_grid3);

    const unsigned int row_in_block = tid >> 5;            // 0..7 (warp index)
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool row_valid = (row < n_rows);

    const unsigned int sbh       = lane >> 4;              // 0..1: sb of pair
    const unsigned int l15       = lane & 15u;
    const unsigned int sub_block = l15 >> 1;               // 0..7 (ib32)
    const unsigned int half      = l15 & 1u;               // 0..1
    const unsigned int sh0       = 14u * half;             // sign shift, l = 2*half

    const uint8_t* gate_e_base = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;
    const uint8_t* up_e_base   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;

    float lane_pacc_g[IQ3_KW_MAX_CHUNK];
    float lane_pacc_u[IQ3_KW_MAX_CHUNK];
    #pragma unroll
    for (unsigned int mi = 0; mi < IQ3_KW_MAX_CHUNK; mi++) {
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
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_XXS_BYTES
                                      + (uint64_t)sb_lane * BLOCK_IQ3_XXS_BYTES;
            const uint8_t* up_block   = up_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_XXS_BYTES
                                      + (uint64_t)sb_lane * BLOCK_IQ3_XXS_BYTES;
            const uint16_t wd_g = (uint16_t)load_u16_par(gate_block);
            const uint16_t wd_u = (uint16_t)load_u16_par(up_block);

            // 4 grid bytes: q3[ib32*8 + 4*half .. +4)
            const uint32_t q3_g = load_u32_2aligned_par(gate_block + 2 + sub_block * 8u + half * 4u);
            const uint32_t q3_u = load_u32_2aligned_par(up_block   + 2 + sub_block * 8u + half * 4u);
            // aux word (signs + scale) for this ib32
            const uint32_t aux_g = load_u32_2aligned_par(gate_block + 66u + sub_block * 4u);
            const uint32_t aux_u = load_u32_2aligned_par(up_block   + 66u + sub_block * 4u);

            {
                const uint64_t sp0 = s_sign_pair[(aux_g >> sh0) & 127u];
                const uint64_t sp1 = s_sign_pair[(aux_g >> (sh0 + 7u)) & 127u];
                wg0 = cond_neg_bytes_par((int32_t)s_grid3[q3_g & 0xffu],         (uint32_t)sp0);
                wg1 = cond_neg_bytes_par((int32_t)s_grid3[(q3_g >> 8) & 0xffu],  (uint32_t)(sp0 >> 32));
                wg2 = cond_neg_bytes_par((int32_t)s_grid3[(q3_g >> 16) & 0xffu], (uint32_t)sp1);
                wg3 = cond_neg_bytes_par((int32_t)s_grid3[q3_g >> 24],           (uint32_t)(sp1 >> 32));
            }
            {
                const uint64_t sp0 = s_sign_pair[(aux_u >> sh0) & 127u];
                const uint64_t sp1 = s_sign_pair[(aux_u >> (sh0 + 7u)) & 127u];
                wu0 = cond_neg_bytes_par((int32_t)s_grid3[q3_u & 0xffu],         (uint32_t)sp0);
                wu1 = cond_neg_bytes_par((int32_t)s_grid3[(q3_u >> 8) & 0xffu],  (uint32_t)(sp0 >> 32));
                wu2 = cond_neg_bytes_par((int32_t)s_grid3[(q3_u >> 16) & 0xffu], (uint32_t)sp1);
                wu3 = cond_neg_bytes_par((int32_t)s_grid3[q3_u >> 24],           (uint32_t)(sp1 >> 32));
            }

            const int32_t ls_g = (int32_t)(2u * (aux_g >> 28) + 1u);
            const int32_t ls_u = (int32_t)(2u * (aux_u >> 28) + 1u);
            gds = (0.25f * f16_to_f32_par(wd_g)) * (float)ls_g;
            uds = (0.25f * f16_to_f32_par(wd_u)) * (float)ls_u;
        }

        // --- Cooperative xq stage (M53 pattern, verbatim) ---
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

            #pragma unroll IQ3_KW_UNROLL
            for (unsigned int mi = 0; mi < n_members; mi++) {
                const uint4 q = q8_lane[mi * 32u];
                int32_t sumi_g = 0;
                sumi_g = sudot4_par(wg0, (int32_t)q.x, sumi_g);
                sumi_g = sudot4_par(wg1, (int32_t)q.y, sumi_g);
                sumi_g = sudot4_par(wg2, (int32_t)q.z, sumi_g);
                sumi_g = sudot4_par(wg3, (int32_t)q.w, sumi_g);
                int32_t sumi_u = 0;
                sumi_u = sudot4_par(wu0, (int32_t)q.x, sumi_u);
                sumi_u = sudot4_par(wu1, (int32_t)q.y, sumi_u);
                sumi_u = sudot4_par(wu2, (int32_t)q.z, sumi_u);
                sumi_u = sudot4_par(wu3, (int32_t)q.w, sumi_u);
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
            float g_sum = warp_sum_f32_par(lane_pacc_g[mi]);
            float u_sum = warp_sum_f32_par(lane_pacc_u[mi]);
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
