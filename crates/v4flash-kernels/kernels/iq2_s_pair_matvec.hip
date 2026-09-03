// iq2_s_pair_matvec — gate+up fused-SwiGLU kernels for IQ2_S weights.
//
// unsloth UD mix: blk.26's ffn_gate_exps + ffn_up_exps are IQ2_S (every
// other layer stays IQ2_XXS). One layer only, so this family mirrors the
// iq2_xxs pair kernels' three load-bearing variants (decode batch, decode
// hetsplit, prefill chunked-by-expert) without the deeper staging zoo.
//
// IQ2_S block (82 B per 256 elements), llama.cpp block_iq2_s:
//   d (half) | qs[64] | qh[8] | scales[8]
//   qs[0..32)  = grid index low bytes (4 per ib32 group)
//   qs[32..64) = EXPLICIT sign bytes (4 per ib32; kmask bit domain)
//   qh[ib32]   = 2 high grid-index bits per l: idx = qs[l] | ((qh<<(8-2l)) & 0x300)
//   scales[ib32] = two 4-bit half-scales: ls1 = 1+2*(s&0xf) (l=0,1),
//                  ls2 = 1+2*(s>>4) (l=2,3)
//   block result = 0.125 * d * yd * Σ_ib32 (ls1*sumi1 + ls2*sumi2)
//
// LDS: s_sign_pair256 (2 KiB; signs are raw bytes → 256-entry pre-expanded
// mask pairs) + s_grid2s (8 KiB; 1024×u64 codebook) + xq staging (4.7 KiB)
// ≈ 15 KiB/WG — same class as the iq2 wstage variant's 16.1 KiB, which
// measured full occupancy.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_IQ2_S_BYTES 82
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

#include "iq2_s_tables.inc"   // iq2s_grid[1024]

__device__ __forceinline__ float f16_to_f32_par(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4_par(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t load_dword_par(const int8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
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

__device__ __forceinline__ int32_t cond_neg_bytes_fast(int32_t pack, uint32_t sm) {
    return (int32_t)(((uint32_t)pack ^ sm) + (sm & 0x01010101u));
}

// Stage the 256-entry sign-pair table + 1024-entry grid. Sign source is
// the RAW byte (kmask domain), not the 7-bit ksigns codebook.
#define IQ2S_STAGE_TABLES(s_sign_pair256, s_grid2s)                          \
    {                                                                         \
        const unsigned int tid_ = threadIdx.x;                                \
        if (tid_ < 256u) {                                                    \
            const uint32_t signs_ = (uint32_t)tid_ * 0x01010101u;             \
            const uint32_t sm0_ = expand_signs_par(signs_, 0x08040201u);      \
            const uint32_t sm1_ = expand_signs_par(signs_, 0x80402010u);      \
            s_sign_pair256[tid_] = ((uint64_t)sm1_ << 32) | (uint64_t)sm0_;   \
        }                                                                     \
        for (unsigned int g_ = threadIdx.x; g_ < 1024u; g_ += blockDim.x) {   \
            s_grid2s[g_] = iq2s_grid[g_];                                     \
        }                                                                     \
    }

__device__ __forceinline__ float warp_sum_f32_par(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// This lane's half of one super-block: ib32 ∈ [ib32_lo, ib32_lo+4).
// Returns 0.125 * d * yd * partial_bsum.
__device__ __forceinline__ float dot_block_half_iq2s(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int ib32_lo,
        const uint64_t* __restrict__ s_sign_pair256,
        const uint64_t* __restrict__ s_grid2s) {
    const uint16_t wd = *(const uint16_t*)(w_blk + 0);
    const uint8_t* qs_grid = w_blk + 2;
    const uint8_t* qs_sign = w_blk + 2 + 32;
    const uint8_t* qh      = w_blk + 2 + 64;
    const uint8_t* scales  = w_blk + 2 + 64 + 8;

    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8_base = (const int8_t*)(y_blk + 4);
    const float d = f16_to_f32_par(wd) * yd;

    int32_t bsum = 0;
    #pragma unroll
    for (unsigned int t = 0; t < 4u; t++) {
        const unsigned int ib32 = ib32_lo + t;
        const uint8_t* qg = qs_grid + 4u * ib32;
        const uint8_t* sg = qs_sign + 4u * ib32;
        const uint32_t qh_b = qh[ib32];
        const uint8_t  sc  = scales[ib32];
        const int32_t ls1 = 1 + 2 * (int32_t)(sc & 0x0F);
        const int32_t ls2 = 1 + 2 * (int32_t)(sc >> 4);
        const int8_t* q8 = q8_base + ib32 * 32u;

        int32_t sumi1 = 0, sumi2 = 0;
        #pragma unroll
        for (unsigned int l = 0; l < 4u; l++) {
            const uint32_t gidx =
                (uint32_t)qg[l] | ((qh_b << (8u - 2u * l)) & 0x300u);
            const uint64_t grid = s_grid2s[gidx];
            const uint64_t sp = s_sign_pair256[sg[l]];
            int32_t sumi = 0;
            sumi = sudot4_par(
                cond_neg_bytes_fast((int32_t)(uint32_t)grid, (uint32_t)sp),
                load_dword_par(q8 + 8u * l), sumi);
            sumi = sudot4_par(
                cond_neg_bytes_fast((int32_t)(uint32_t)(grid >> 32),
                                    (uint32_t)(sp >> 32)),
                load_dword_par(q8 + 8u * l + 4u), sumi);
            if (l < 2u) { sumi1 += sumi; } else { sumi2 += sumi; }
        }
        bsum += ls1 * sumi1 + ls2 * sumi2;
    }
    return 0.125f * d * (float)bsum;
}

// Decode: batched over n_used slots, fused SwiGLU + expert weight.
// Contract identical to iq2_xxs_pair_matvec_fused_swiglu_batch.
extern "C" __global__ void iq2_s_pair_matvec_fused_swiglu_batch(
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
    __shared__ uint64_t s_sign_pair256[256];
    __shared__ uint64_t s_grid2s[1024];
    __shared__ uint8_t  s_xq[16 * 292];
    IQ2S_STAGE_TABLES(s_sign_pair256, s_grid2s);
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
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_S_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_S_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ2_S_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ2_S_BYTES;
        const uint8_t* y   = s_xq          + (size_t)b   * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half_iq2s(w_g, y, ib32_lo, s_sign_pair256, s_grid2s);
        u_sum += dot_block_half_iq2s(w_u, y, ib32_lo, s_sign_pair256, s_grid2s);
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
extern "C" __global__ void iq2_s_pair_matvec_fused_swiglu_batch_hetsplit(
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
    __shared__ uint64_t s_sign_pair256[256];
    __shared__ uint64_t s_grid2s[1024];
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

    IQ2S_STAGE_TABLES(s_sign_pair256, s_grid2s);
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
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_S_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_S_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ2_S_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ2_S_BYTES;
        const uint8_t* y   = s_xq          + (size_t)b   * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half_iq2s(w_g, y, ib32_lo, s_sign_pair256, s_grid2s);
        u_sum += dot_block_half_iq2s(w_u, y, ib32_lo, s_sign_pair256, s_grid2s);
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
extern "C" __global__ void iq2_s_pair_matvec_fused_swiglu_chunked(
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

    __shared__ uint64_t s_sign_pair256[256];
    __shared__ uint64_t s_grid2s[1024];
    __shared__ uint8_t  s_xq[16 * 292];
    IQ2S_STAGE_TABLES(s_sign_pair256, s_grid2s);
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
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_S_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_S_BYTES;

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
                const uint8_t* w_g = gate_row_base + (uint64_t)bl * BLOCK_IQ2_S_BYTES;
                const uint8_t* w_u = up_row_base   + (uint64_t)bl * BLOCK_IQ2_S_BYTES;
                const uint8_t* y   = s_xq          + (size_t)bl   * BLOCK_Q8_K_BYTES;
                g_sum += dot_block_half_iq2s(w_g, y, ib32_lo, s_sign_pair256, s_grid2s);
                u_sum += dot_block_half_iq2s(w_u, y, ib32_lo, s_sign_pair256, s_grid2s);
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
