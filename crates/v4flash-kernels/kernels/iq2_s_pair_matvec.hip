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


// =============================================================================
// wmma — f16 WMMA batched-prefill gate/up for IQ2_S (2026-09-08).
//
// Port of iq2_xs_pair_matvec_fused_swiglu_wmma (see that banner for the
// design, measurements and the dead ends). Everything about the tiling,
// the LDS activation staging, the software pipelining and the epilogue is
// identical; only the block decode differs:
//
//   IQ2_XS  74 B: d | qs u16[32] (9-bit grid idx + 7-bit ksigns idx) | scales[8]
//   IQ2_S   82 B: d | qs_grid u8[32] | qs_sign u8[32] | qh u8[8] | scales[8]
//           grid idx (10 bit) = qs_grid[4*ib32+l] | (((qh[ib32] >> 2l) & 3) << 8)
//           sign byte         = qs_sign[4*ib32+l]  (raw, one bit per weight)
//
// Per k-step (16 weights = ib32 = ks>>1, half = ks&1, groups l = 2*half+{0,1})
// the two grid bytes are bytes (2*half, 2*half+1) of dword ib32 of qs_grid,
// the two sign bytes the same bytes of dword ib32 of qs_sign, the two
// 2-bit highs are nibble `half` of qh[ib32], the scale nibble is `half`
// of scales[ib32] — all static after unrolling, exactly as in the XS
// kernel. The biased sign masks (0x80 = +, 0x7F = -) come from the 256-entry
// raw-byte table, the grid from the 1024-entry codebook (both LDS, 10 KiB
// vs the XS kernel's 5 KiB; total ~27 KiB/WG, still 2 WG/CU on gfx1151).
//
// Rows: n_rows must be a multiple of 16 (N_FF_EXP = 2048). Members: <= 32
// per work item (chunk_size), 1 or 2 N-tiles (template).
// Oracle: tests/iq2_s_wmma_oracle.rs. Bench: bench_iq2_xs_isolated
// (BENCH_FMT=iq2s BENCH_VARIANT=7).
// =============================================================================

#define IQ2S_WM_MAX_CHUNK   32u
#define IQ2S_WM_ROWS_PER_WG 128u
#define IQ2S_WM_KWIN        256u                       // one window per super-block
#define IQ2S_WM_X_STRIDE    (IQ2S_WM_KWIN + 8u)        // halves per staged member row (+8 pad = 4 banks)

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define IQ2S_WMMA_RDNA3 1
#endif

typedef _Float16 iq2s_half16 __attribute__((ext_vector_type(16)));
typedef _Float16 iq2s_half2  __attribute__((ext_vector_type(2)));
typedef float    iq2s_float8 __attribute__((ext_vector_type(8)));

// "magic" dequant of one dword of 4 unsigned grid bytes g with biased sign
// mask smb: u = 128 +/- grid, halves 0x64|u = 1024+u exactly, (h-1152)*scale.
__device__ __forceinline__ void iq2s_magic_dword(uint32_t g, uint32_t smb, _Float16 scale,
                                                 uint32_t& lo, uint32_t& hi) {
    const uint32_t u = (g ^ smb) + (smb & 0x01010101u);
    const uint32_t h01 = __builtin_amdgcn_perm(u, 0x64646464u, 0x00050004u);
    const uint32_t h23 = __builtin_amdgcn_perm(u, 0x64646464u, 0x00070006u);
    iq2s_half2 a, b;
    __builtin_memcpy(&a, &h01, 4);
    __builtin_memcpy(&b, &h23, 4);
    const iq2s_half2 bias = {(_Float16)-1152.0f, (_Float16)-1152.0f};
    a = (a + bias) * scale;
    b = (b + bias) * scale;
    __builtin_memcpy(&lo, &a, 4);
    __builtin_memcpy(&hi, &b, 4);
}
// 16 weights of one matrix: two grid entries (u64) + two biased sign masks.
__device__ __forceinline__ iq2s_half16 iq2s_magic_frag(
        uint64_t g0, uint64_t sm0, uint64_t g1, uint64_t sm1, _Float16 scale) {
    uint32_t d[8];
    iq2s_magic_dword((uint32_t)g0,         (uint32_t)sm0,         scale, d[0], d[1]);
    iq2s_magic_dword((uint32_t)(g0 >> 32), (uint32_t)(sm0 >> 32), scale, d[2], d[3]);
    iq2s_magic_dword((uint32_t)g1,         (uint32_t)sm1,         scale, d[4], d[5]);
    iq2s_magic_dword((uint32_t)(g1 >> 32), (uint32_t)(sm1 >> 32), scale, d[6], d[7]);
    iq2s_half16 r;
    __builtin_memcpy(&r, d, 32);
    return r;
}

// Raw gathers for one k-step of both matrices. gb/sb = the two grid / sign
// bytes of this k-step (low byte = group 2*half), qhn = nibble `half` of
// qh[ib32] (bits [1:0] -> group 2*half, [3:2] -> group 2*half+1).
struct iq2s_raw8 { uint64_t g0, s0, g1, s1, u0, t0, u1, t1; };
__device__ __forceinline__ iq2s_raw8 iq2s_gather_raw8(
        const uint64_t* s_grid2s, const uint64_t* s_sign256b,
        uint32_t gb_g, uint32_t sb_g, uint32_t qhn_g,
        uint32_t gb_u, uint32_t sb_u, uint32_t qhn_u) {
    iq2s_raw8 r;
    r.g0 = s_grid2s[(gb_g & 0xffu) | ((qhn_g & 3u) << 8)];
    r.g1 = s_grid2s[((gb_g >> 8) & 0xffu) | (((qhn_g >> 2) & 3u) << 8)];
    r.s0 = s_sign256b[sb_g & 0xffu];
    r.s1 = s_sign256b[(sb_g >> 8) & 0xffu];
    r.u0 = s_grid2s[(gb_u & 0xffu) | ((qhn_u & 3u) << 8)];
    r.u1 = s_grid2s[((gb_u >> 8) & 0xffu) | (((qhn_u >> 2) & 3u) << 8)];
    r.t0 = s_sign256b[sb_u & 0xffu];
    r.t1 = s_sign256b[(sb_u >> 8) & 0xffu];
    return r;
}

// Static dword select from a uint4 pair (index is a compile-time constant
// after unrolling).
#define IQ2S_DW(v2, i) ((i) == 0u ? (v2)[0].x : (i) == 1u ? (v2)[0].y : (i) == 2u ? (v2)[0].z : (i) == 3u ? (v2)[0].w \
                      : (i) == 4u ? (v2)[1].x : (i) == 5u ? (v2)[1].y : (i) == 6u ? (v2)[1].z : (v2)[1].w)

template <int NT>
__device__ __forceinline__ void iq2s_wmma_mainloop(
        iq2s_float8 (&acc_g)[2], iq2s_float8 (&acc_u)[2],
        const uint8_t* __restrict__ gate_row, const uint8_t* __restrict__ up_row,
        const uint16_t* __restrict__ x16,
        uint16_t* s_x, const int* s_member_packed,
        const uint64_t* s_grid2s, const uint64_t* s_sign256b,
        unsigned int n_members, unsigned int n_blocks, bool row_valid,
        unsigned int tid, unsigned int l16) {
    const size_t k_total = (size_t)n_blocks * 256u;
    constexpr unsigned int KSTEPS_PER_WIN    = IQ2S_WM_KWIN / 16u;               // 16
    constexpr unsigned int CHUNKS_PER_ROW    = IQ2S_WM_KWIN / 8u;                // b128 per member row
    constexpr unsigned int CHUNKS_PER_WIN    = IQ2S_WM_MAX_CHUNK * CHUNKS_PER_ROW;
    constexpr unsigned int CHUNKS_PER_THREAD = (CHUNKS_PER_WIN + 255u) / 256u;
    static_assert(IQ2S_WM_KWIN == 256u, "one window per super-block");

    for (unsigned int sb = 0; sb < n_blocks; sb++) {
        const uint8_t* gate_block = gate_row + (uint64_t)sb * BLOCK_IQ2_S_BYTES;
        const uint8_t* up_block   = up_row   + (uint64_t)sb * BLOCK_IQ2_S_BYTES;

        // --- This super-block's block bytes per matrix: d, 2 x b128 grid
        //     bytes, 2 x b128 sign bytes, qh (u64), scales (u64). Issued
        //     before the staging barrier so the latency overlaps it.
        float dsc_g = 0.0f, dsc_u = 0.0f;
        uint4 gq[2], gs[2], uq[2], us[2];
        uint2 gqh = make_uint2(0u, 0u), uqh = gqh, gsc = gqh, usc = gqh;
        #pragma unroll
        for (unsigned int i = 0; i < 2u; i++) {
            gq[i] = make_uint4(0u, 0u, 0u, 0u); gs[i] = gq[i]; uq[i] = gq[i]; us[i] = gq[i];
        }
        if (row_valid) {
            uint16_t wd_g, wd_u;
            __builtin_memcpy(&wd_g, gate_block, 2);
            __builtin_memcpy(&wd_u, up_block, 2);
            dsc_g = 0.125f * f16_to_f32_par(wd_g);
            dsc_u = 0.125f * f16_to_f32_par(wd_u);
            #pragma unroll
            for (unsigned int i = 0; i < 2u; i++) {
                __builtin_memcpy(&gq[i], gate_block + 2  + i * 16u, 16);
                __builtin_memcpy(&gs[i], gate_block + 34 + i * 16u, 16);
                __builtin_memcpy(&uq[i], up_block   + 2  + i * 16u, 16);
                __builtin_memcpy(&us[i], up_block   + 34 + i * 16u, 16);
            }
            __builtin_memcpy(&gqh, gate_block + 66, 8);
            __builtin_memcpy(&gsc, gate_block + 74, 8);
            __builtin_memcpy(&uqh, up_block   + 66, 8);
            __builtin_memcpy(&usc, up_block   + 74, 8);
        }

        // --- Stage every member's 256 k of this super-block (zeros past
        //     n_members): 32 members x 32 b128 chunks over 256 threads.
        #pragma unroll
        for (unsigned int j = 0; j < CHUNKS_PER_THREAD; j++) {
            const unsigned int idx = tid + 256u * j;
            if (idx < CHUNKS_PER_WIN) {
                const unsigned int mi = idx / CHUNKS_PER_ROW;
                const unsigned int c  = idx % CHUNKS_PER_ROW;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (mi < n_members) {
                    const unsigned int b = (unsigned int)(s_member_packed[mi] >> 16);
                    const uint4* src = (const uint4*)(x16 + (size_t)b * k_total
                                                          + (size_t)sb * 256u + (size_t)c * 8u);
                    v = *src;
                }
                *(uint4*)(&s_x[mi * IQ2S_WM_X_STRIDE + c * 8u]) = v;
            }
        }
        __syncthreads();

        // --- Software pipeline: k-step 0's gathers + B fragments up front.
        iq2s_raw8 r8_cur, r8_nxt;
        iq2s_half16 bf_cur[2], bf_nxt[2];
        r8_cur = iq2s_gather_raw8(s_grid2s, s_sign256b,
                                  gq[0].x & 0xffffu, gs[0].x & 0xffffu, gqh.x & 0xfu,
                                  uq[0].x & 0xffffu, us[0].x & 0xffffu, uqh.x & 0xfu);
        #pragma unroll
        for (unsigned int nt = 0; nt < 2u; nt++) {
            if (nt < (unsigned int)NT) {
                const uint16_t* xb = &s_x[(nt * 16u + l16) * IQ2S_WM_X_STRIDE];
                __builtin_memcpy(&bf_cur[nt], xb, 32);
            }
        }

        #pragma unroll
        for (unsigned int kw = 0; kw < KSTEPS_PER_WIN; kw++) {
            const unsigned int ib32 = kw >> 1;
            const unsigned int half = kw & 1u;

            // scale nibble `half` of scales[ib32]
            const uint8_t sc_g = (uint8_t)(((ib32 < 4u) ? gsc.x : gsc.y) >> (8u * (ib32 & 3u)));
            const uint8_t sc_u = (uint8_t)(((ib32 < 4u) ? usc.x : usc.y) >> (8u * (ib32 & 3u)));
            const int32_t ls_g = 1 + 2 * (int32_t)(half ? (sc_g >> 4) : (sc_g & 0x0F));
            const int32_t ls_u = 1 + 2 * (int32_t)(half ? (sc_u >> 4) : (sc_u & 0x0F));
            const float sg = dsc_g * (float)ls_g;   // 0 for invalid rows (dsc=0)
            const float su = dsc_u * (float)ls_u;

            // prefetch the next k-step's rows (the WMMAs below cover it)
            if (kw + 1u < KSTEPS_PER_WIN) {
                const unsigned int kn = kw + 1u;
                const unsigned int nib32 = kn >> 1;
                const unsigned int nhalf = kn & 1u;
                const uint32_t ngq = IQ2S_DW(gq, nib32) >> (16u * nhalf);
                const uint32_t ngs = IQ2S_DW(gs, nib32) >> (16u * nhalf);
                const uint32_t nuq = IQ2S_DW(uq, nib32) >> (16u * nhalf);
                const uint32_t nus = IQ2S_DW(us, nib32) >> (16u * nhalf);
                const uint32_t nqh_g = (((nib32 < 4u) ? gqh.x : gqh.y) >> (8u * (nib32 & 3u) + 4u * nhalf)) & 0xfu;
                const uint32_t nqh_u = (((nib32 < 4u) ? uqh.x : uqh.y) >> (8u * (nib32 & 3u) + 4u * nhalf)) & 0xfu;
                r8_nxt = iq2s_gather_raw8(s_grid2s, s_sign256b,
                                          ngq & 0xffffu, ngs & 0xffffu, nqh_g,
                                          nuq & 0xffffu, nus & 0xffffu, nqh_u);
            }
            const iq2s_half16 a_g = iq2s_magic_frag(r8_cur.g0, r8_cur.s0, r8_cur.g1, r8_cur.s1, (_Float16)sg);
            const iq2s_half16 a_u = iq2s_magic_frag(r8_cur.u0, r8_cur.t0, r8_cur.u1, r8_cur.t1, (_Float16)su);

            #pragma unroll
            for (unsigned int nt = 0; nt < 2u; nt++) {
                if (nt < (unsigned int)NT) {
                    const iq2s_half16 bf = bf_cur[nt];
                    if (kw + 1u < KSTEPS_PER_WIN) {
                        const uint16_t* xb = &s_x[(nt * 16u + l16) * IQ2S_WM_X_STRIDE + (kw + 1u) * 16u];
                        __builtin_memcpy(&bf_nxt[nt], xb, 32);
                    }
#if defined(IQ2S_WMMA_RDNA3)
                    acc_g[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_g, bf, acc_g[nt]);
                    acc_u[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_u, bf, acc_u[nt]);
#else
                    (void)bf; (void)a_g; (void)a_u;
#endif
                }
            }
            r8_cur = r8_nxt; bf_cur[0] = bf_nxt[0]; bf_cur[1] = bf_nxt[1];
        }
        __syncthreads();   // s_x reuse by the next super-block
    }
}

template <int OUT16>
__device__ __forceinline__ void iq2s_wmma_body(
        float          * __restrict__ mid,           // OUT16=0: f32 [B, n_used, n_rows]; OUT16=1: f16 halves
        const uint8_t  * __restrict__ gate_w_base,
        const uint8_t  * __restrict__ up_w_base,
        const uint16_t * __restrict__ x16,           // [B, K] f16
        const float    * __restrict__ expert_w,      // [B, n_used]
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
#if defined(IQ2S_WMMA_RDNA3)
    const unsigned int w_idx = blockIdx.y;
    const int packed_item = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;
    const unsigned int n_members = member_end - member_start;
    const unsigned int n_tiles = (n_members + 15u) >> 4;     // 1 or 2

    __shared__ uint64_t s_sign_pair256[256];
    __shared__ uint64_t s_grid2s[1024];
    __shared__ __attribute__((aligned(16))) uint16_t s_x[IQ2S_WM_MAX_CHUNK * IQ2S_WM_X_STRIDE];
    __shared__ int      s_member_packed[IQ2S_WM_MAX_CHUNK];

    const unsigned int tid = threadIdx.x;
    IQ2S_STAGE_TABLES(s_sign_pair256, s_grid2s);
    __syncthreads();
    if (tid < 256u) s_sign_pair256[tid] ^= 0x8080808080808080ull;   // biased masks
    for (unsigned int i = tid; i < IQ2S_WM_MAX_CHUNK; i += blockDim.x) {
        s_member_packed[i] = (i < n_members)
            ? expert_members[(size_t)e * (size_t)max_per_expert + (size_t)(member_start + i)]
            : 0;
    }
    __syncthreads();

    const unsigned int wave = tid >> 5;                    // 0..7
    const unsigned int lane = tid & 31u;
    const unsigned int l16  = lane & 15u;
    const unsigned int hi   = lane >> 4;                   // C-fragment row parity
    const unsigned int m0   = blockIdx.x * IQ2S_WM_ROWS_PER_WG + wave * 16u;
    const unsigned int row_a = m0 + l16;                   // row this lane dequantizes
    const bool row_valid = (row_a < n_rows);

    const uint8_t* gate_row = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe
                            + (uint64_t)row_a * (uint64_t)n_blocks * BLOCK_IQ2_S_BYTES;
    const uint8_t* up_row   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe
                            + (uint64_t)row_a * (uint64_t)n_blocks * BLOCK_IQ2_S_BYTES;

    iq2s_float8 acc_g[2], acc_u[2];
    #pragma unroll
    for (unsigned int nt = 0; nt < 2u; nt++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) { acc_g[nt][i] = 0.0f; acc_u[nt][i] = 0.0f; }
    }

    if (n_tiles == 2u) {
        iq2s_wmma_mainloop<2>(acc_g, acc_u, gate_row, up_row, x16, s_x, s_member_packed,
                              s_grid2s, s_sign_pair256, n_members, n_blocks, row_valid, tid, l16);
    } else {
        iq2s_wmma_mainloop<1>(acc_g, acc_u, gate_row, up_row, x16, s_x, s_member_packed,
                              s_grid2s, s_sign_pair256, n_members, n_blocks, row_valid, tid, l16);
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
            const size_t out_row0 = ((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows;
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
                        ((uint16_t*)mid)[out_row0 + row] = __half_as_ushort(__float2half(v));
                    } else {
                        mid[out_row0 + row] = v;
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

extern "C" __global__ __launch_bounds__(256) void iq2_s_pair_matvec_fused_swiglu_wmma(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base,
        const uint8_t* __restrict__ up_w_base, const uint16_t* __restrict__ x16,
        const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,
        unsigned int n_rows, unsigned int n_blocks) {
    iq2s_wmma_body<0>(mid, gate_w_base, up_w_base, x16, expert_w, group_count,
                      expert_members, work_items, gate_bpe, up_bpe, n_used,
                      max_per_expert, chunk_size, clamp, n_rows, n_blocks);
}

// Production twin: f16 mid out ([B*n_used][n_rows] halves, = the down
// kernel's B operand), otherwise identical to _wmma.
extern "C" __global__ __launch_bounds__(256) void iq2_s_pair_matvec_fused_swiglu_wmma_h(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base,
        const uint8_t* __restrict__ up_w_base, const uint16_t* __restrict__ x16,
        const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,
        unsigned int n_rows, unsigned int n_blocks) {
    iq2s_wmma_body<1>(mid, gate_w_base, up_w_base, x16, expert_w, group_count,
                      expert_members, work_items, gate_bpe, up_bpe, n_used,
                      max_per_expert, chunk_size, clamp, n_rows, n_blocks);
}
