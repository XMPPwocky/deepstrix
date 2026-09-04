// iq2_s_pair_matvec — gate+up fused-SwiGLU kernels for IQ2_S weights.
//
// unsloth UD mix: blk.26's ffn_gate_exps + ffn_up_exps are IQ2_S in the
// UD-Q2_K_XL file; in the Vision-Exp UD-IQ3_XXS file IQ2_S is instead the
// gate/up type on 42 of 43 layers (blk.26 goes IQ3_S). This family mirrors
// the iq2_xxs pair kernels' three load-bearing variants (decode batch,
// decode hetsplit, prefill chunked-by-expert) plus the kwide prefill kernel
// at the bottom (weights dequantized once per lane and amortized across the
// chunk's members — the `_chunked` kernel re-dequants per member, the
// ae9bd61 trap).
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
// LDS (DECODE + CHUNKED kernels only): s_sign_pair256 (2 KiB; signs are raw
// bytes → 256-entry pre-expanded mask pairs) + s_grid2s (8 KiB; 1024×u64
// codebook) + xq staging (4.7 KiB) ≈ 15 KiB/WG — same class as the iq2
// wstage variant's 16.1 KiB, which measured full occupancy.
// The kwide PREFILL kernel is NOT in this class: it adds a 16 KiB s_q8v
// member-staging buffer for 26.4 KiB total → 2 WGs/CU. See its banner
// ("kwide prefill", below) for the footprint and the occupancy note.

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


// =============================================================================
// kwide — M51-structure port for IQ2_S gate/up (2026-09-04), from
// iq3_s_pair_matvec_fused_swiglu_kwide (itself the iq3_xxs kwide port).
//
// WHY: the `_chunked` kernel above iterates chunk members SERIALLY and
// re-reads AND re-dequantizes the expert's whole weight rows for every
// member (grid/sign LUT walk + cond-neg per member). With HOT_CHUNK=32 the
// same weight block is dequantized up to 32x per WG. That is exactly the
// ae9bd61 UD-Q2_K_XL trap: gate/up IQ2_S on 42 of 43 layers of the unsloth
// Vision-Exp UD-IQ3_XXS mix ran this path and pinned prefill at ~152 tok/s.
//
// Lane mapping: warp = row; lanes split a super-block PAIR — sbh = lane>>4
// picks the sb, l15 = lane&15 → ib32 = l15>>1, half = l15&1. The lane owns
// the 16 weights of subgroups l = 2*half, 2*half+1 (2 grid entries x 8):
//   qg   = qs[4*ib32 + 2*half .. +2)        (offset 2 + 4*ib32 + 2*half → even)
//   sg   = qs[32 + 4*ib32 + 2*half .. +2)   (offset 34 + ... → even; RAW sign
//                                            bytes, kmask domain)
//   qhn  = (qh[ib32] >> 4*half) & 0xf       (bits (0,1) → 9th/10th index bit
//                                            of l0, bits (2,3) → those of l1;
//                                            ggml: qs[l] | ((qh<<(8-2l))&0x300))
//   ls   = 1 + 2 * ((scales[ib32] >> 4*half) & 0xf)
//          (ggml's ls1 covers l=0,1 and ls2 covers l=2,3 — i.e. EXACTLY one
//           lane's 16 weights, so the per-lane scale fold is exact)
//   gds  = 0.125f * f16(d) * ls             (IQ2_S carries ggml's 0.125
//                                            prefactor; iq3_s has none,
//                                            iq3_xxs uses 0.25)
// Note the same nibble shift 4*half serves BOTH qh and scales.
//
// Alignment: 82 ≡ 2 (mod 4), so intra-block dword fields are only 2-aligned;
// every multi-byte load goes through load_u16_par, as in the 110-byte iq3_s
// and 98-byte iq3_xxs kernels.
//
// The staged s_q8v lane slice [16*l15, +16) of the block's 256 q8 bytes is
// exactly ib32*32 + 16*half → (q.x,q.y) = subgroup 2*half's two grid halves,
// (q.z,q.w) = subgroup 2*half+1's.
//
// LDS: s_sign_pair256 2 KiB + s_grid2s 8 KiB + s_q8v 16 KiB + s_yd 0.25 +
// s_member_packed 0.125 ≈ 26.4 KiB (iq2_xs 21.4, iq3_s 20.4, iq3_xxs 18.4)
// → 2 WGs/CU, the same occupancy class as the iq2_xs kwide kernel.
//
// s_q8v dominates that footprint and is the binding limiter: halving
// IQ2S_KW_MAX_CHUNK to 16 drops the kernel under the 3-WG/CU line and
// measured -12%/-14.5% in the isolated bench. That number is an OCCUPANCY
// CEILING, not the expected production delta — it was taken at BENCH_CHUNK=16
// (bench_iq2_xs_isolated.rs's default), where the whole member set still fits
// ONE staging pass, so it pays none of the restructuring cost. Production runs
// chunk_size=32 (HOT_CHUNK / CHUNK_SIZE), where keeping MAX_CHUNK=16 forces
// s_q8v to be staged twice per super-block pair and adds two extra
// __syncthreads() per pair on top of the two already in the member loop —
// 8 pairs × 2 barriers/WG the probe never paid. Re-bench at BENCH_CHUNK=32
// before believing any part of the -14.5%. MAX_CHUNK=32 is what ships.
// =============================================================================

// Even-offset u16 load (every multi-byte IQ2_S intra-block field is
// 2-aligned, never 4-aligned: 82 ≡ 2 mod 4).
__device__ __forceinline__ uint32_t load_u16_iq2s(const uint8_t* p) {
    uint16_t v;
    __builtin_memcpy(&v, p, 2);
    return (uint32_t)v;
}

// Upper bound on the runtime `chunk_size`: sizes s_q8v / s_yd /
// s_member_packed below and the lane_pacc_g|u register arrays. #ifndef so a
// DEEPSTRIX_KERNEL_CFLAGS=-DIQ2S_KW_MAX_CHUNK=... occupancy probe actually
// takes effect (an unguarded #define silently wins over the -D).
// The Rust launch guard (src/iq2_s.rs IQ2S_KW_MAX_CHUNK) tracks the same -D
// via build.rs, so a probe cannot leave it accepting a chunk_size this
// kernel has no LDS for. Editing this literal directly bypasses that — pass
// -D instead.
#ifndef IQ2S_KW_MAX_CHUNK
#define IQ2S_KW_MAX_CHUNK 32u
#endif
#ifndef IQ2S_KW_UNROLL
#define IQ2S_KW_UNROLL 4
#endif

extern "C" __global__ void iq2_s_pair_matvec_fused_swiglu_kwide(
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
    __shared__ uint64_t s_grid2s[1024];
    __shared__ uint4    s_q8v[IQ2S_KW_MAX_CHUNK * 2 * 16]; // [mi][sbh][16]
    __shared__ float    s_yd[IQ2S_KW_MAX_CHUNK * 2];       // [mi][sbh]
    __shared__ int      s_member_packed[IQ2S_KW_MAX_CHUNK];

    const unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i < n_members; i += blockDim.x) {
        s_member_packed[i] = expert_members[(size_t)e * (size_t)max_per_expert
                                          + (size_t)(member_start + i)];
    }
    IQ2S_STAGE_TABLES(s_sign_pair256, s_grid2s);
    __syncthreads();

    const unsigned int row_in_block = tid >> 5;            // 0..7 (warp index)
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool row_valid = (row < n_rows);

    const unsigned int sbh       = lane >> 4;              // 0..1: sb of pair
    const unsigned int l15       = lane & 15u;
    const unsigned int sub_block = l15 >> 1;               // 0..7 (ib32)
    const unsigned int half      = l15 & 1u;               // 0..1
    const unsigned int hsh       = 4u * half;              // qh + scale nibble shift

    const uint8_t* gate_e_base = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;
    const uint8_t* up_e_base   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;

    float lane_pacc_g[IQ2S_KW_MAX_CHUNK];
    float lane_pacc_u[IQ2S_KW_MAX_CHUNK];
    #pragma unroll
    for (unsigned int mi = 0; mi < IQ2S_KW_MAX_CHUNK; mi++) {
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
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_S_BYTES
                                      + (uint64_t)sb_lane * BLOCK_IQ2_S_BYTES;
            const uint8_t* up_block   = up_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_S_BYTES
                                      + (uint64_t)sb_lane * BLOCK_IQ2_S_BYTES;
            const uint16_t wd_g = (uint16_t)load_u16_iq2s(gate_block);
            const uint16_t wd_u = (uint16_t)load_u16_iq2s(up_block);

            // 2 grid-index low bytes: qs[4*ib32 + 2*half .. +2)
            const uint32_t qg_g = load_u16_iq2s(gate_block + 2u + sub_block * 4u + half * 2u);
            const uint32_t qg_u = load_u16_iq2s(up_block   + 2u + sub_block * 4u + half * 2u);
            // 2 RAW sign bytes: qs[32 + 4*ib32 + 2*half .. +2)
            const uint32_t sg_g = load_u16_iq2s(gate_block + 34u + sub_block * 4u + half * 2u);
            const uint32_t sg_u = load_u16_iq2s(up_block   + 34u + sub_block * 4u + half * 2u);
            // high index bits: qh[ib32] nibble `half` (bits 0,1 -> l0; 2,3 -> l1)
            const uint32_t qhn_g = ((uint32_t)gate_block[66u + sub_block] >> hsh) & 0xfu;
            const uint32_t qhn_u = ((uint32_t)up_block[66u + sub_block]   >> hsh) & 0xfu;
            // per-16-weight scale nibble (ls1 for half=0, ls2 for half=1)
            const uint32_t nib_g = ((uint32_t)gate_block[74u + sub_block] >> hsh) & 0xfu;
            const uint32_t nib_u = ((uint32_t)up_block[74u + sub_block]   >> hsh) & 0xfu;

            {
                const uint64_t gr0 = s_grid2s[(qg_g & 0xffu) | ((qhn_g & 3u) << 8)];
                const uint64_t gr1 = s_grid2s[(qg_g >> 8)    | ((qhn_g & 12u) << 6)];
                const uint64_t sp0 = s_sign_pair256[sg_g & 0xffu];
                const uint64_t sp1 = s_sign_pair256[sg_g >> 8];
                wg0 = cond_neg_bytes_fast((int32_t)(uint32_t)gr0,         (uint32_t)sp0);
                wg1 = cond_neg_bytes_fast((int32_t)(uint32_t)(gr0 >> 32), (uint32_t)(sp0 >> 32));
                wg2 = cond_neg_bytes_fast((int32_t)(uint32_t)gr1,         (uint32_t)sp1);
                wg3 = cond_neg_bytes_fast((int32_t)(uint32_t)(gr1 >> 32), (uint32_t)(sp1 >> 32));
            }
            {
                const uint64_t gr0 = s_grid2s[(qg_u & 0xffu) | ((qhn_u & 3u) << 8)];
                const uint64_t gr1 = s_grid2s[(qg_u >> 8)    | ((qhn_u & 12u) << 6)];
                const uint64_t sp0 = s_sign_pair256[sg_u & 0xffu];
                const uint64_t sp1 = s_sign_pair256[sg_u >> 8];
                wu0 = cond_neg_bytes_fast((int32_t)(uint32_t)gr0,         (uint32_t)sp0);
                wu1 = cond_neg_bytes_fast((int32_t)(uint32_t)(gr0 >> 32), (uint32_t)(sp0 >> 32));
                wu2 = cond_neg_bytes_fast((int32_t)(uint32_t)gr1,         (uint32_t)sp1);
                wu3 = cond_neg_bytes_fast((int32_t)(uint32_t)(gr1 >> 32), (uint32_t)(sp1 >> 32));
            }

            const int32_t ls_g = (int32_t)(2u * nib_g + 1u);
            const int32_t ls_u = (int32_t)(2u * nib_u + 1u);
            gds = (0.125f * f16_to_f32_par(wd_g)) * (float)ls_g;
            uds = (0.125f * f16_to_f32_par(wd_u)) * (float)ls_u;
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

            #pragma unroll IQ2S_KW_UNROLL
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
