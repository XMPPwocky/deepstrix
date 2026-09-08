// iq3_xxs_matvec — routed-MoE down-projection kernels for IQ3_XXS weights.
//
// The unsloth UD-IQ2_XXS mix stores ffn_down_exps as IQ3_XXS on 41 of 43
// layers (Q2_K in the antirez mix). Family mirrors q2_k_accumulate_matvec_par:
// same launch geometry, same partials/reduce contract (reuse
// q2_k_reduce_partials — it is dtype-agnostic), same 4-lanes-per-superblock
// split. Numerics mirror llama.cpp ggml_vec_dot_iq3_xxs_q8_K_generic:
//
//   block_result = 0.25 * d * yd * Σ_ib32 ls(ib32) * sumi(ib32)
//   ls  = 2*(aux32 >> 28) + 1
//   sumi = Σ_l grid[q3[..]] · q8, sign-flipped via ksigns_iq2xs
//
// Per-lane split (sub = lane & 3): each lane owns TWO of the 8 ib32 groups
// ({2·sub, 2·sub+1}) → 16 sudot4/lane per block, identical VALU count to the
// q2_k quarter-block split.
//
// Alignment: IQ3_XXS blocks are 98 B (2 mod 4), so intra-block dwords
// alternate alignment phase across blocks. All within-block multi-byte reads
// are assembled from u16 loads (offsets are always even); q8/bsums reads
// stay aligned dword loads (Q8_K blocks are 292 B, 4-aligned).

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

__device__ __forceinline__ float warp_sum_f32_par(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// Decode (single token, single expert), zero_init/accumulate contract as
// q2_k_accumulate_matvec_par. grid (n_rows/8, 1, 1), block 256.
extern "C" __global__ void iq3_xxs_accumulate_matvec_par(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_expert,
        const uint8_t  * __restrict__ xq,
        unsigned int     n_rows,
        unsigned int     n_blocks_in,
        unsigned int     zero_init) {
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint32_t s_grid3[256];
    IQ3_STAGE_TABLES(s_sign_pair, s_grid3);

    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_IQ3_XXS_BYTES;

    float acc = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
        const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_IQ3_XXS_BYTES;
        const uint8_t* y_blk = xq       + (uint64_t)b * BLOCK_Q8_K_BYTES;
        acc += dot_block_quarter_iq3(w_blk, y_blk, sub, s_sign_pair, s_grid3);
    }
    acc = warp_sum_f32_par(acc);

    if (lane == 0) {
        if (zero_init) {
            out[row] = acc;
        } else {
            out[row] += acc;
        }
    }
}

// Decode batched over the n_used selected experts (graph-captured core).
// Same contract as q2_k_matvec_par_batched.
extern "C" __global__ void iq3_xxs_matvec_par_batched(
        float          * __restrict__ out,           // [n_rows]
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,       // slot-major midq_cat
        const int      * __restrict__ selected,      // [n_used]
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint32_t s_grid3[256];
    IQ3_STAGE_TABLES(s_sign_pair, s_grid3);

    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    float acc = 0.0f;
    for (unsigned int s = 0; s < n_used; s++) {
        const int e = selected[s];
        const uint8_t* w_expert = w_base + (uint64_t)e * (uint64_t)dbpe;
        const uint8_t* xq_slot  = xq_base + (uint64_t)s * (uint64_t)xq_slot_stride;
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_IQ3_XXS_BYTES;
        for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
            const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_IQ3_XXS_BYTES;
            const uint8_t* y_blk = xq_slot  + (uint64_t)b * BLOCK_Q8_K_BYTES;
            acc += dot_block_quarter_iq3(w_blk, y_blk, sub, s_sign_pair, s_grid3);
        }
    }
    acc = warp_sum_f32_par(acc);

    if (lane == 0) {
        out[row] = acc;
    }
}

// Decode het-split (mode 0 = iGPU takes non-resident, 1 = dGPU takes
// resident up to cap). Same contract as q2_k_matvec_par_batched_hetsplit.
extern "C" __global__ void iq3_xxs_matvec_par_batched_hetsplit(
        float          * __restrict__ out,           // [n_rows] (device partial)
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ selected,      // [n_used]
        const int      * __restrict__ remap,         // [256]
        unsigned int     mode,
        unsigned int     dgpu_cap,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint32_t s_grid3[256];
    IQ3_STAGE_TABLES(s_sign_pair, s_grid3);

    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    float acc = 0.0f;
    unsigned int res_rank = 0;
    for (unsigned int s = 0; s < n_used; s++) {
        const int sel = selected[s];
        const int dense = remap[sel];
        const bool resident = dense >= 0;
        const bool dgpu_takes = resident && (res_rank < dgpu_cap);
        if (resident) res_rank++;
        const bool ours = (mode == 0u) ? !dgpu_takes : dgpu_takes;
        if (!ours) continue;
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
        const uint8_t* w_expert = w_base + (uint64_t)e * (uint64_t)dbpe;
        const uint8_t* xq_slot  = xq_base + (uint64_t)s * (uint64_t)xq_slot_stride;
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_IQ3_XXS_BYTES;
        for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
            const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_IQ3_XXS_BYTES;
            const uint8_t* y_blk = xq_slot  + (uint64_t)b * BLOCK_Q8_K_BYTES;
            acc += dot_block_quarter_iq3(w_blk, y_blk, sub, s_sign_pair, s_grid3);
        }
    }
    acc = warp_sum_f32_par(acc);

    if (lane == 0) {
        out[row] = acc;
    }
}

// =============================================================================
// by_expert_kwide2 — prefill production analog of q2_k_..._by_expert_kwide2:
// one WG per (16-row tile, expert chunk); each warp owns TWO adjacent rows;
// per block the lane's two ib32 groups are unpacked ONCE per row into signed
// dwords (grid + ksigns applied — the 4-bit ls CANNOT be byte-folded here:
// ls ≤ 31 × grid magnitude ≤ 62 overflows i8, so ls stays a per-group i32
// multiplier after the sudot4 chain); per member only 16 sudot4 + 2 imul
// + fma remain, with one q8 load shared across both rows.
//
// Grid (n_rows/16, n_work_items), block 256, chunk_size ≤ 32 (halves of 16),
// n_rows % 16 == 0. Partials contract identical to the q2_k family (pair
// with q2_k_reduce_partials / _hetsplit).
// =============================================================================
extern "C" __global__ void iq3_xxs_matvec_par_by_expert_kwide2(
        float          * __restrict__ partials,      // [B*n_used, n_rows]
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,       // [B, n_used*xq_slot_stride]
        const int      * __restrict__ group_count,   // [N_EXPERT]
        const int      * __restrict__ expert_members,// [N_EXPERT * max_per_expert]
        const int      * __restrict__ work_items,    // [n_work_items]
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     chunk_size,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint32_t s_grid3[256];
    IQ3_STAGE_TABLES(s_sign_pair, s_grid3);

    const unsigned int warp_in_block = threadIdx.x / 32u;
    const unsigned int lane          = threadIdx.x & 31u;
    const unsigned int row0          = blockIdx.x * 16u + warp_in_block * 2u;
    const unsigned int w_idx         = blockIdx.y;
    if (row0 >= n_rows) return;

    const int packed_item = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;
    const unsigned int n_members = member_end - member_start;

    const unsigned int block_lane = lane >> 2u;   // 0..7
    const unsigned int sub        = lane & 3u;    // 0..3 → ib32 {2sub, 2sub+1}
    const unsigned int q8_off     = sub * 64u;    // this lane's 64 q8 bytes

    const uint8_t* w_expert = w_base + (uint64_t)e * (uint64_t)dbpe;

    for (unsigned int half_lo = 0; half_lo < n_members; half_lo += 16u) {
        const unsigned int half_n = min(16u, n_members - half_lo);

        float pacc0[16];
        float pacc1[16];
        #pragma unroll
        for (unsigned int m = 0; m < 16u; m++) { pacc0[m] = 0.0f; pacc1[m] = 0.0f; }

        for (unsigned int bl = block_lane; bl < n_blocks_in; bl += 8u) {
            // Unpack BOTH rows' two ib32 groups once: signed grid dwords
            // (signs applied) + per-group ls + block d.
            int32_t w0[16], w1[16];
            int32_t ls0[2], ls1[2];
            float d0 = 0.0f, d1 = 0.0f;
            #pragma unroll
            for (unsigned int rr = 0; rr < 2u; rr++) {
                const uint8_t* w_blk = w_expert
                    + (uint64_t)(row0 + rr) * (uint64_t)n_blocks_in * BLOCK_IQ3_XXS_BYTES
                    + (uint64_t)bl * BLOCK_IQ3_XXS_BYTES;
                const float dd = f16_to_f32_par((uint16_t)load_u16_par(w_blk));
                int32_t* w  = rr ? w1  : w0;
                int32_t* ls = rr ? ls1 : ls0;
                unsigned int wi = 0;
                #pragma unroll
                for (unsigned int t = 0; t < 2u; t++) {
                    const unsigned int ib32 = 2u * sub + t;
                    const uint32_t aux32 =
                        load_u32_2aligned_par(w_blk + 2 + 64 + 4u * ib32);
                    ls[t] = (int32_t)(2u * (aux32 >> 28) + 1u);
                    const uint8_t* q3 = w_blk + 2 + ib32 * 8u;
                    #pragma unroll
                    for (unsigned int l = 0; l < 4u; l++) {
                        const uint64_t sp = s_sign_pair[(aux32 >> (7u * l)) & 127u];
                        w[wi++] = cond_neg_bytes_par(
                            (int32_t)s_grid3[q3[2u * l]], (uint32_t)sp);
                        w[wi++] = cond_neg_bytes_par(
                            (int32_t)s_grid3[q3[2u * l + 1u]], (uint32_t)(sp >> 32));
                    }
                }
                if (rr) { d1 = dd; } else { d0 = dd; }
            }

            #pragma unroll 2
            for (unsigned int m = 0; m < half_n; m++) {
                const int packed = expert_members[(size_t)e * (size_t)max_per_expert
                                                + (size_t)(member_start + half_lo + m)];
                const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
                const unsigned int slot = (unsigned int)(packed       ) & 0xffffu;
                const uint8_t* y_blk = xq_base
                                     + (size_t)b * (size_t)n_used * (size_t)xq_slot_stride
                                     + (size_t)slot * (size_t)xq_slot_stride
                                     + (size_t)bl * (size_t)BLOCK_Q8_K_BYTES;
                const float yd = *(const float*)(y_blk + 0);
                const int8_t* q8 = (const int8_t*)(y_blk + 4) + q8_off;

                // q8 loaded once, dotted against both rows.
                int32_t q[16];
                #pragma unroll
                for (unsigned int i = 0; i < 16u; i++) {
                    q[i] = load_dword_i8_par(q8 + 4u * i);
                }
                // Per ib32 group: 8 sudot4 → sumi, then × ls.
                int32_t bsum0 = 0, bsum1 = 0;
                #pragma unroll
                for (unsigned int t = 0; t < 2u; t++) {
                    int32_t s0 = 0, s1 = 0;
                    #pragma unroll
                    for (unsigned int i = 0; i < 8u; i++) {
                        s0 = sudot4_par(w0[8u * t + i], q[8u * t + i], s0);
                        s1 = sudot4_par(w1[8u * t + i], q[8u * t + i], s1);
                    }
                    bsum0 += s0 * ls0[t];
                    bsum1 += s1 * ls1[t];
                }
                pacc0[m] += 0.25f * d0 * yd * (float)bsum0;
                pacc1[m] += 0.25f * d1 * yd * (float)bsum1;
            }
        }

        for (unsigned int m = 0; m < half_n; m++) {
            const float a0 = warp_sum_f32_par(pacc0[m]);
            const float a1 = warp_sum_f32_par(pacc1[m]);
            if (lane == 0) {
                const int packed = expert_members[(size_t)e * (size_t)max_per_expert
                                                + (size_t)(member_start + half_lo + m)];
                const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
                const unsigned int slot = (unsigned int)(packed       ) & 0xffffu;
                const size_t out_base = ((size_t)b * (size_t)n_used + (size_t)slot)
                                      * (size_t)n_rows;
                partials[out_base + (size_t)row0]      = a0;
                partials[out_base + (size_t)row0 + 1u] = a1;
            }
        }
    }
}

// =============================================================================
// wmma — f16 WMMA batched-prefill down-projection for IQ3_XXS (2026-09-08).
// Twin of iq2_xs_pair_matvec_fused_swiglu_wmma (see its banner for the
// rationale, layout facts and the measured design decisions); one matrix,
// partials contract of the kwide2 kernel above (pair with
// q2_k_reduce_partials / _hetsplit).
//
//   - WG = 256 threads = 8 waves x 16-row M-tiles (128 rows per WG); work
//     item = (expert, <=32 members) -> N-tiles of 16 (template NT = 1|2).
//   - B = f16 mid [B*n_used][K]: member (b, slot) row at
//     (b*n_used + slot) * x_member_stride halves; each super-block's 256 k
//     of the <=32 members staged in LDS (row stride 264 halves).
//   - A: lane l dequantizes row (m0 + l%16); per super-block one u16 d +
//     4 x b128 q3 (64 grid bytes) + 2 x b128 aux (8 u32) loads; 16 k-steps
//     fully unrolled. k-step ks: ib32 = ks>>1, half = ks&1 -> the u32 #ks of
//     q3 holds the 4 grid indices, aux[ib32] holds the two 7-bit sign
//     indices at bits 7*(2*half+j) and the 4-bit scale at bits 28-31.
//   - Dequant (magic): grid u32 (4 unsigned magnitude bytes) ^ biased sign
//     mask dword -> +(smb & 0x01010101) -> bytes 128 +/- g; v_perm builds
//     halves 0x64|u = 1024+u; pk_add -1152; pk_mul by 0.25*d*ls.
//     9 VALU per 4 weights, ~55 VALU per k-step; WMMA 2 x 32 clk (NT=2).
//   - Epilogue: partials[(b*n_used+slot)*n_rows + row] = C[row][member].
//
// LDS: 1 KiB grid + 1 KiB biased signs + 16.5 KiB x-stage + 128 B.
// =============================================================================

#define IQ3_WM_MAX_CHUNK   32u
#define IQ3_WM_ROWS_PER_WG 128u
#define IQ3_WM_X_STRIDE    264u

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define IQ3_WMMA_RDNA3 1
#endif

typedef _Float16 iq3_half16 __attribute__((ext_vector_type(16)));
typedef _Float16 iq3_half2  __attribute__((ext_vector_type(2)));
typedef float    iq3_float8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void iq3_magic_dword(uint32_t g, uint32_t smb, _Float16 scale,
                                                uint32_t& lo, uint32_t& hi) {
    const uint32_t u = (g ^ smb) + (smb & 0x01010101u);
    const uint32_t h01 = __builtin_amdgcn_perm(u, 0x64646464u, 0x00050004u);
    const uint32_t h23 = __builtin_amdgcn_perm(u, 0x64646464u, 0x00070006u);
    iq3_half2 a, b;
    __builtin_memcpy(&a, &h01, 4);
    __builtin_memcpy(&b, &h23, 4);
    const iq3_half2 bias = {(_Float16)-1152.0f, (_Float16)-1152.0f};
    a = (a + bias) * scale;
    b = (b + bias) * scale;
    __builtin_memcpy(&lo, &a, 4);
    __builtin_memcpy(&hi, &b, 4);
}

template <int NT>
__device__ __forceinline__ void iq3_wmma_mainloop(
        iq3_float8 (&acc)[2],
        const uint8_t* __restrict__ w_row,          // this lane's row, expert base applied
        const uint16_t* __restrict__ x16,
        uint16_t* s_x, const int* s_member_packed,
        const uint32_t* s_grid3, const uint64_t* s_sign_bias,
        unsigned int n_members, unsigned int n_used, unsigned int x_member_stride,
        unsigned int n_blocks_in, bool row_valid, unsigned int tid, unsigned int l16) {
    for (unsigned int sb = 0; sb < n_blocks_in; sb++) {
        // --- stage this super-block's 256 k of every member (zeros past n_members)
        {
            #pragma unroll
            for (unsigned int j = 0; j < 4u; j++) {
                const unsigned int idx = tid + 256u * j;
                const unsigned int mi  = idx >> 5;
                const unsigned int c   = idx & 31u;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (mi < n_members) {
                    const int packed = s_member_packed[mi];
                    const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
                    const unsigned int slot = (unsigned int)(packed) & 0xffffu;
                    const uint4* src = (const uint4*)(x16
                        + ((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)x_member_stride
                        + (size_t)sb * 256u + (size_t)c * 8u);
                    v = *src;
                }
                *(uint4*)(&s_x[mi * IQ3_WM_X_STRIDE + c * 8u]) = v;
            }
        }
        __syncthreads();

        // --- this lane's whole 98-B block for row_a: d | q3[64] | aux[8]
        const uint8_t* blk = w_row + (uint64_t)sb * BLOCK_IQ3_XXS_BYTES;
        uint4 q3v[4], auxv[2];
        float dsc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) q3v[i] = make_uint4(0u, 0u, 0u, 0u);
        auxv[0] = make_uint4(0u, 0u, 0u, 0u); auxv[1] = auxv[0];
        if (row_valid) {
            uint16_t wd;
            __builtin_memcpy(&wd, blk, 2);
            #pragma unroll
            for (int i = 0; i < 4; i++) __builtin_memcpy(&q3v[i], blk + 2 + 16 * i, 16);
            __builtin_memcpy(&auxv[0], blk + 66, 16);
            __builtin_memcpy(&auxv[1], blk + 82, 16);
            dsc = 0.25f * f16_to_f32_par(wd);
        }

        #pragma unroll
        for (unsigned int ks = 0; ks < 16u; ks++) {
            const unsigned int ib32 = ks >> 1;
            const unsigned int half = ks & 1u;
            const uint32_t qs = (ks & 3u) == 0u ? q3v[ks >> 2].x : (ks & 3u) == 1u ? q3v[ks >> 2].y
                              : (ks & 3u) == 2u ? q3v[ks >> 2].z : q3v[ks >> 2].w;
            const uint32_t aux = (ib32 & 3u) == 0u ? auxv[ib32 >> 2].x : (ib32 & 3u) == 1u ? auxv[ib32 >> 2].y
                               : (ib32 & 3u) == 2u ? auxv[ib32 >> 2].z : auxv[ib32 >> 2].w;
            const float ls = (float)(2u * (aux >> 28) + 1u);
            const _Float16 hs = (_Float16)(dsc * ls);
            // sign groups l = 2*half, 2*half+1 ; grid bytes 0..3 of qs
            const uint64_t sm0 = s_sign_bias[(aux >> (7u * (2u * half)))      & 127u];
            const uint64_t sm1 = s_sign_bias[(aux >> (7u * (2u * half + 1u))) & 127u];
            const uint32_t g0 = s_grid3[qs & 0xffu];
            const uint32_t g1 = s_grid3[(qs >> 8) & 0xffu];
            const uint32_t g2 = s_grid3[(qs >> 16) & 0xffu];
            const uint32_t g3 = s_grid3[qs >> 24];
            uint32_t d[8];
            iq3_magic_dword(g0, (uint32_t)sm0,         hs, d[0], d[1]);
            iq3_magic_dword(g1, (uint32_t)(sm0 >> 32), hs, d[2], d[3]);
            iq3_magic_dword(g2, (uint32_t)sm1,         hs, d[4], d[5]);
            iq3_magic_dword(g3, (uint32_t)(sm1 >> 32), hs, d[6], d[7]);
            iq3_half16 a;
            __builtin_memcpy(&a, d, 32);

            #pragma unroll
            for (unsigned int nt = 0; nt < 2u; nt++) {
                if (nt < (unsigned int)NT) {
                    const uint16_t* xb = &s_x[(nt * 16u + l16) * IQ3_WM_X_STRIDE + ks * 16u];
                    iq3_half16 bf;
                    __builtin_memcpy(&bf, xb, 32);
                    acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bf, acc[nt]);
                }
            }
        }
        __syncthreads();   // s_x reuse by the next super-block
    }
}

extern "C" __global__ void iq3_xxs_matvec_par_by_expert_wmma(
        float          * __restrict__ partials,      // [B*n_used, n_rows]
        const uint8_t  * __restrict__ w_base,
        const uint16_t * __restrict__ x16,           // f16 mid [B*n_used, x_member_stride]
        const int      * __restrict__ group_count,   // [N_EXPERT]
        const int      * __restrict__ expert_members,// [N_EXPERT * max_per_expert]
        const int      * __restrict__ work_items,    // [n_work_items]
        unsigned int     dbpe,
        unsigned int     x_member_stride,            // halves per (b, slot) row (>= n_blocks_in*256)
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     chunk_size,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
#if defined(IQ3_WMMA_RDNA3)
    const unsigned int w_idx = blockIdx.y;
    const int packed_item = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;
    const unsigned int n_members = member_end - member_start;
    const unsigned int n_tiles = (n_members + 15u) >> 4;

    __shared__ uint64_t s_sign_bias[128];
    __shared__ uint32_t s_grid3[256];
    __shared__ __attribute__((aligned(16))) uint16_t s_x[IQ3_WM_MAX_CHUNK * IQ3_WM_X_STRIDE];
    __shared__ int s_member_packed[IQ3_WM_MAX_CHUNK];

    const unsigned int tid = threadIdx.x;
    IQ3_STAGE_TABLES(s_sign_bias, s_grid3);          // ends with __syncthreads
    if (tid < 128u) s_sign_bias[tid] ^= 0x8080808080808080ull;   // biased masks
    for (unsigned int i = tid; i < IQ3_WM_MAX_CHUNK; i += blockDim.x) {
        s_member_packed[i] = (i < n_members)
            ? expert_members[(size_t)e * (size_t)max_per_expert + (size_t)(member_start + i)]
            : 0;
    }
    __syncthreads();

    const unsigned int wave = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int l16  = lane & 15u;
    const unsigned int hi   = lane >> 4;
    const unsigned int m0   = blockIdx.x * IQ3_WM_ROWS_PER_WG + wave * 16u;
    const unsigned int row_a = m0 + l16;
    const bool row_valid = (row_a < n_rows);
    const uint8_t* w_row = w_base + (uint64_t)e * (uint64_t)dbpe
                         + (uint64_t)row_a * (uint64_t)n_blocks_in * BLOCK_IQ3_XXS_BYTES;

    iq3_float8 acc[2];
    #pragma unroll
    for (unsigned int nt = 0; nt < 2u; nt++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[nt][i] = 0.0f;
    }
    if (n_tiles == 2u) {
        iq3_wmma_mainloop<2>(acc, w_row, x16, s_x, s_member_packed, s_grid3, s_sign_bias,
                             n_members, n_used, x_member_stride, n_blocks_in, row_valid, tid, l16);
    } else {
        iq3_wmma_mainloop<1>(acc, w_row, x16, s_x, s_member_packed, s_grid3, s_sign_bias,
                             n_members, n_used, x_member_stride, n_blocks_in, row_valid, tid, l16);
    }

    #pragma unroll
    for (unsigned int nt = 0; nt < 2u; nt++) {
        const unsigned int mi = nt * 16u + l16;
        if (nt < n_tiles && mi < n_members) {
            const int packed        = s_member_packed[mi];
            const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
            const unsigned int slot = (unsigned int)(packed) & 0xffffu;
            float* out_base = partials + ((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const unsigned int row = m0 + 2u * (unsigned int)i + hi;
                if (row < n_rows) out_base[row] = acc[nt][i];
            }
        }
    }
#else
    __builtin_trap();
#endif
}
