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
        const int e = (mode == 0u) ? sel : dense;
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
