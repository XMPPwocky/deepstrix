// iq3_s_pair_matvec — gate+up fused-SwiGLU kernels for IQ3_S weights.
//
// unsloth UD-IQ3_XXS (original / Vision-Exp repos): blk.26's ffn_gate_exps
// + ffn_up_exps are IQ3_S (every other layer's gate/up is IQ2_S). One layer
// only, so this family mirrors the iq2_s pair kernels' three load-bearing
// decode/prefill variants (batch, hetsplit, chunked-by-expert) plus the
// iq3_xxs_pair kwide prefill kernel (weights dequantized once per lane and
// amortized across chunk members — the `_chunked` kernel re-dequants per
// member, the ae9bd61 trap).
//
// IQ3_S block (110 B per 256 elements), llama.cpp block_iq3_s:
//   d (half) @0 | qs[64] @2 | qh[8] @66 | signs[32] @74 | scales[4] @106
//   qs[8*ib32 + 2l+{0,1}] = grid index low 8 bits for subgroup l (2 grid
//                           entries x 4 weights = 8 weights per l)
//   qh[ib32]     = 9th index bit per grid entry: bit 2l -> grid1 of l,
//                  bit 2l+1 -> grid2 of l
//                  (grid1 = qs[2l]   | ((qh << (8-2l)) & 256),
//                   grid2 = qs[2l+1] | ((qh << (7-2l)) & 256))
//   signs[4*ib32 + l] = EXPLICIT sign byte for the 8 weights of subgroup l
//                       (kmask_iq2xs bit j flips weight j)
//   scales[ib32/2]    = two 4-bit nibbles, one per 64 weights:
//                       ls = 1 + 2*(nibble)   (low nibble: even ib32,
//                       high nibble: odd ib32)
//   block result = d * yd * Σ_ib32 ls(ib32) * sumi(ib32)   — NO prefactor
//                  (iq2_s: 0.125, iq3_xxs: 0.25)
//
// Alignment: 110 ≡ 2 (mod 4), so intra-block dword fields are only
// 2-aligned; every multi-byte load goes through the u16-pair helpers (the
// same rule the 98-byte iq3_xxs blocks follow).
//
// LDS (decode/chunked): s_sign_pair256 (2 KiB; raw sign bytes → 256-entry
// pre-expanded mask pairs, iq2_s's builder verbatim) + s_grid3s (2 KiB;
// 512×u32 codebook) + xq staging (4.7 KiB) ≈ 8.7 KiB/WG (iq2_s: 14.7).
// kwide: 2 + 2 + 16 (s_q8v) + 0.25 + 0.125 ≈ 20.4 KiB (iq3_xxs 18.4,
// iq2_xs 21.4) → LDS allows 3 WGs/CU; the limiter stays VGPRs as for the
// siblings.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_IQ3_S_BYTES 110
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

#include "iq3_s_tables.inc"   // iq3s_grid[512]

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

// Even-offset u16 load (every multi-byte IQ3_S intra-block field is
// 2-aligned, never 4-aligned: 110 ≡ 2 mod 4).
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

// Conditional byte negate via xor+carry: for a byte with mask 0xff,
// (b ^ 0xff) + 1 == -b mod 256. Safe here because grid magnitudes are
// 0x01..0x0f — no carry can propagate into the next byte.
__device__ __forceinline__ int32_t cond_neg_bytes_fast(int32_t pack, uint32_t sm) {
    return (int32_t)(((uint32_t)pack ^ sm) + (sm & 0x01010101u));
}

// Stage the 256-entry sign-pair table (raw sign byte → (sm1<<32)|sm0 byte
// masks for the 8 weights of a subgroup) + the 512-entry grid. No trailing
// __syncthreads (callers sync after their own xq/member staging).
#define IQ3S_STAGE_TABLES(s_sign_pair256, s_grid3s)                          \
    {                                                                         \
        const unsigned int tid_ = threadIdx.x;                                \
        if (tid_ < 256u) {                                                    \
            const uint32_t signs_ = (uint32_t)tid_ * 0x01010101u;             \
            const uint32_t sm0_ = expand_signs_par(signs_, 0x08040201u);      \
            const uint32_t sm1_ = expand_signs_par(signs_, 0x80402010u);      \
            s_sign_pair256[tid_] = ((uint64_t)sm1_ << 32) | (uint64_t)sm0_;   \
        }                                                                     \
        for (unsigned int g_ = threadIdx.x; g_ < 512u; g_ += blockDim.x) {    \
            s_grid3s[g_] = iq3s_grid[g_];                                     \
        }                                                                     \
    }

__device__ __forceinline__ float warp_sum_f32_par(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// This lane's half of one super-block: ib32 ∈ [ib32_lo, ib32_lo+4).
// Returns d * yd * partial_bsum (no prefactor). Summing the two halves
// reproduces ggml_vec_dot_iq3_s_q8_K's per-block value exactly (integer
// parts are bit-identical; only the f32 accumulation order differs).
__device__ __forceinline__ float dot_block_half_iq3s(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int ib32_lo,
        const uint64_t* __restrict__ s_sign_pair256,
        const uint32_t* __restrict__ s_grid3s) {
    const uint16_t wd = (uint16_t)load_u16_par(w_blk + 0);
    const uint8_t* qs_grid = w_blk + 2;
    const uint8_t* qh      = w_blk + 66;
    const uint8_t* signs   = w_blk + 74;
    const uint8_t* scales  = w_blk + 106;

    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8_base = (const int8_t*)(y_blk + 4);
    const float d = f16_to_f32_par(wd) * yd;

    int32_t bsum = 0;
    #pragma unroll
    for (unsigned int t = 0; t < 4u; t++) {
        const unsigned int ib32 = ib32_lo + t;
        const uint8_t* qg = qs_grid + 8u * ib32;
        const uint8_t* sg = signs   + 4u * ib32;
        const uint32_t qh_b = qh[ib32];
        const int32_t ls = 1 + 2 * (int32_t)((scales[ib32 >> 1] >> (4u * (ib32 & 1u))) & 0x0Fu);
        const int8_t* q8 = q8_base + ib32 * 32u;

        int32_t sumi = 0;
        #pragma unroll
        for (unsigned int l = 0; l < 4u; l++) {
            const uint32_t g1 = (uint32_t)qg[2u * l]      | ((qh_b << (8u - 2u * l)) & 256u);
            const uint32_t g2 = (uint32_t)qg[2u * l + 1u] | ((qh_b << (7u - 2u * l)) & 256u);
            const uint64_t sp = s_sign_pair256[sg[l]];
            sumi = sudot4_par(
                cond_neg_bytes_fast((int32_t)s_grid3s[g1], (uint32_t)sp),
                load_dword_par(q8 + 8u * l), sumi);
            sumi = sudot4_par(
                cond_neg_bytes_fast((int32_t)s_grid3s[g2], (uint32_t)(sp >> 32)),
                load_dword_par(q8 + 8u * l + 4u), sumi);
        }
        bsum += ls * sumi;
    }
    return d * (float)bsum;
}

// Decode: batched over n_used slots, fused SwiGLU + expert weight.
// Contract identical to iq2_s_pair_matvec_fused_swiglu_batch.
extern "C" __global__ void iq3_s_pair_matvec_fused_swiglu_batch(
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
    __shared__ uint32_t s_grid3s[512];
    __shared__ uint8_t  s_xq[16 * 292];
    IQ3S_STAGE_TABLES(s_sign_pair256, s_grid3s);
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
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_S_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_S_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ3_S_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ3_S_BYTES;
        const uint8_t* y   = s_xq          + (size_t)b   * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half_iq3s(w_g, y, ib32_lo, s_sign_pair256, s_grid3s);
        u_sum += dot_block_half_iq3s(w_u, y, ib32_lo, s_sign_pair256, s_grid3s);
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
// iq2_s_pair_matvec_fused_swiglu_batch_hetsplit.
extern "C" __global__ void iq3_s_pair_matvec_fused_swiglu_batch_hetsplit(
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
    __shared__ uint32_t s_grid3s[512];
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

    IQ3S_STAGE_TABLES(s_sign_pair256, s_grid3s);
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
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_S_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_S_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ3_S_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ3_S_BYTES;
        const uint8_t* y   = s_xq          + (size_t)b   * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half_iq3s(w_g, y, ib32_lo, s_sign_pair256, s_grid3s);
        u_sum += dot_block_half_iq3s(w_u, y, ib32_lo, s_sign_pair256, s_grid3s);
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
// Kept as the PAIR_VARIANT=chunked fallback; kwide below is the default.
extern "C" __global__ void iq3_s_pair_matvec_fused_swiglu_chunked(
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
    __shared__ uint32_t s_grid3s[512];
    __shared__ uint8_t  s_xq[16 * 292];
    IQ3S_STAGE_TABLES(s_sign_pair256, s_grid3s);
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
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_S_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_S_BYTES;

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
                const uint8_t* w_g = gate_row_base + (uint64_t)bl * BLOCK_IQ3_S_BYTES;
                const uint8_t* w_u = up_row_base   + (uint64_t)bl * BLOCK_IQ3_S_BYTES;
                const uint8_t* y   = s_xq          + (size_t)bl   * BLOCK_Q8_K_BYTES;
                g_sum += dot_block_half_iq3s(w_g, y, ib32_lo, s_sign_pair256, s_grid3s);
                u_sum += dot_block_half_iq3s(w_u, y, ib32_lo, s_sign_pair256, s_grid3s);
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
// kwide — M51-structure port for IQ3_S gate/up (2026-09-03), from
// iq3_xxs_pair_matvec_fused_swiglu_kwide. Only the per-lane unpack differs.
//
// Lane mapping: warp = row; lanes split a super-block PAIR — sbh = lane>>4
// picks the sb, l15 = lane&15 → ib32 = l15>>1, half = l15&1. The lane
// owns subgroups l = 2*half, 2*half+1 (16 weights = 4 grid entries):
//   q3   = qs[8*ib32 + 4*half .. +4)          (offset 2+8*ib32+4*half ≡ 2 mod 4
//                                              → 2-aligned u16-pair load)
//   hi   = (qh[ib32] >> 4*half) & 0xf         (bit i → 9th index bit of q3 byte i)
//   sg   = signs[4*ib32 + 2*half .. +2)        (one raw sign byte per l)
//   ls   = 1 + 2 * nibble(scales[ib32/2], ib32&1)
// ls covers 64 weights ⊇ the lane's 16, so folding gds = f16(d)*ls per lane
// is exact (same argument as the iq3_xxs per-ib32 fold).
// The staged s_q8v lane slice [16*l15, +16) of the block's 256 q8 bytes is
// exactly ib32*32 + 16*half → (q.x,q.y) = subgroup 2*half's (grid1,grid2),
// (q.z,q.w) = subgroup 2*half+1's.
// =============================================================================

#define IQ3S_KW_MAX_CHUNK 32u
#ifndef IQ3S_KW_UNROLL
#define IQ3S_KW_UNROLL 4
#endif

extern "C" __global__ void iq3_s_pair_matvec_fused_swiglu_kwide(
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

    __shared__ uint64_t s_sign_pair256[256];
    __shared__ uint32_t s_grid3s[512];
    __shared__ uint4    s_q8v[IQ3S_KW_MAX_CHUNK * 2 * 16]; // [mi][sbh][16]
    __shared__ float    s_yd[IQ3S_KW_MAX_CHUNK * 2];       // [mi][sbh]
    __shared__ int      s_member_packed[IQ3S_KW_MAX_CHUNK];

    const unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i < n_members; i += blockDim.x) {
        s_member_packed[i] = expert_members[(size_t)e * (size_t)max_per_expert
                                          + (size_t)(member_start + i)];
    }
    IQ3S_STAGE_TABLES(s_sign_pair256, s_grid3s);
    __syncthreads();

    const unsigned int row_in_block = tid >> 5;            // 0..7 (warp index)
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool row_valid = (row < n_rows);

    const unsigned int sbh       = lane >> 4;              // 0..1: sb of pair
    const unsigned int l15       = lane & 15u;
    const unsigned int sub_block = l15 >> 1;               // 0..7 (ib32)
    const unsigned int half      = l15 & 1u;               // 0..1
    const unsigned int hsh       = 4u * half;              // qh nibble shift

    const uint8_t* gate_e_base = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;
    const uint8_t* up_e_base   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;

    float lane_pacc_g[IQ3S_KW_MAX_CHUNK];
    float lane_pacc_u[IQ3S_KW_MAX_CHUNK];
    #pragma unroll
    for (unsigned int mi = 0; mi < IQ3S_KW_MAX_CHUNK; mi++) {
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
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_S_BYTES
                                      + (uint64_t)sb_lane * BLOCK_IQ3_S_BYTES;
            const uint8_t* up_block   = up_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ3_S_BYTES
                                      + (uint64_t)sb_lane * BLOCK_IQ3_S_BYTES;
            const uint16_t wd_g = (uint16_t)load_u16_par(gate_block);
            const uint16_t wd_u = (uint16_t)load_u16_par(up_block);

            // 4 grid-index low bytes: qs[ib32*8 + 4*half .. +4)
            const uint32_t q3_g = load_u32_2aligned_par(gate_block + 2 + sub_block * 8u + half * 4u);
            const uint32_t q3_u = load_u32_2aligned_par(up_block   + 2 + sub_block * 8u + half * 4u);
            // 9th index bits: qh[ib32] nibble `half` (bit i ↔ q3 byte i)
            const uint32_t hi_g = ((uint32_t)gate_block[66u + sub_block] >> hsh) & 0xfu;
            const uint32_t hi_u = ((uint32_t)up_block[66u + sub_block]   >> hsh) & 0xfu;
            // two raw sign bytes: signs[ib32*4 + 2*half .. +2) (even offset)
            const uint32_t sg_g = load_u16_par(gate_block + 74u + sub_block * 4u + half * 2u);
            const uint32_t sg_u = load_u16_par(up_block   + 74u + sub_block * 4u + half * 2u);
            // per-64-weight scale nibble
            const uint32_t nib_g = ((uint32_t)gate_block[106u + (sub_block >> 1)] >> (4u * (sub_block & 1u))) & 0xfu;
            const uint32_t nib_u = ((uint32_t)up_block[106u + (sub_block >> 1)]   >> (4u * (sub_block & 1u))) & 0xfu;

            {
                const uint64_t sp0 = s_sign_pair256[sg_g & 0xffu];
                const uint64_t sp1 = s_sign_pair256[sg_g >> 8];
                const uint32_t i0 = (q3_g & 0xffu)         | ((hi_g & 1u) << 8);
                const uint32_t i1 = ((q3_g >> 8) & 0xffu)  | ((hi_g & 2u) << 7);
                const uint32_t i2 = ((q3_g >> 16) & 0xffu) | ((hi_g & 4u) << 6);
                const uint32_t i3 = (q3_g >> 24)           | ((hi_g & 8u) << 5);
                wg0 = cond_neg_bytes_fast((int32_t)s_grid3s[i0], (uint32_t)sp0);
                wg1 = cond_neg_bytes_fast((int32_t)s_grid3s[i1], (uint32_t)(sp0 >> 32));
                wg2 = cond_neg_bytes_fast((int32_t)s_grid3s[i2], (uint32_t)sp1);
                wg3 = cond_neg_bytes_fast((int32_t)s_grid3s[i3], (uint32_t)(sp1 >> 32));
            }
            {
                const uint64_t sp0 = s_sign_pair256[sg_u & 0xffu];
                const uint64_t sp1 = s_sign_pair256[sg_u >> 8];
                const uint32_t i0 = (q3_u & 0xffu)         | ((hi_u & 1u) << 8);
                const uint32_t i1 = ((q3_u >> 8) & 0xffu)  | ((hi_u & 2u) << 7);
                const uint32_t i2 = ((q3_u >> 16) & 0xffu) | ((hi_u & 4u) << 6);
                const uint32_t i3 = (q3_u >> 24)           | ((hi_u & 8u) << 5);
                wu0 = cond_neg_bytes_fast((int32_t)s_grid3s[i0], (uint32_t)sp0);
                wu1 = cond_neg_bytes_fast((int32_t)s_grid3s[i1], (uint32_t)(sp0 >> 32));
                wu2 = cond_neg_bytes_fast((int32_t)s_grid3s[i2], (uint32_t)sp1);
                wu3 = cond_neg_bytes_fast((int32_t)s_grid3s[i3], (uint32_t)(sp1 >> 32));
            }

            const int32_t ls_g = (int32_t)(2u * nib_g + 1u);
            const int32_t ls_u = (int32_t)(2u * nib_u + 1u);
            gds = f16_to_f32_par(wd_g) * (float)ls_g;
            uds = f16_to_f32_par(wd_u) * (float)ls_u;
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

            #pragma unroll IQ3S_KW_UNROLL
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
