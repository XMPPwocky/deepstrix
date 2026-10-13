// mxfp4_matvec — routed-MoE down-projection kernels for MXFP4 weights
// (blk.26 + blk.42 in the unsloth UD mix — the official checkpoint's native
// expert format, so these two layers' weights are plausibly lossless).
//
// MXFP4 block: 17 B per 32 elements = 1 B e8m0 scale + 16 B packed fp4
// codes. Element order (llama.cpp block_mxfp4): elems 0..15 = LOW nibbles
// of qs[0..16], elems 16..31 = HIGH nibbles. Values are kvalues_mxfp4
// (E2M1 ×2: 0,±1,±2,±3,±4,±6,±8,±12 — exact int8), scale =
// e8m0_to_fp32_half(e) = 2^(e-128).
//
// Interface matches the q2_k/iq3_xxs down family (Q8_K activations,
// slot-major midq, same partials/reduce contract) so it drops into the
// same per-layer dispatch. n_blocks_in counts Q8_K SUPERBLOCKS (256
// elems); each superblock spans 8 MXFP4 blocks (136 B).
//
// Per-lane split (sub = lane & 3): 2 of the 8 mxfp4 sub-blocks
// ({2·sub, 2·sub+1}) → 16 sudot4/lane per superblock, same VALU count as
// the q2_k/iq3 quarter splits. Nibble→int8 via a 16-entry LDS LUT.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_MXFP4_BYTES 17
#define SUPER_MXFP4_BYTES 136   // 8 × 17 per 256 elements

// Super-block layout v2 — see the long note in mxfp4_pair_matvec.hip.
//   bytes   0..128  eight 16-byte nibble blocks (block k at k*16)
//   bytes 128..136  eight E8M0 scale bytes (block k at 128+k)
// Stride unchanged at 136; only the arrangement inside a super-block moved, so
// nibbles can be read 8 bytes at a time instead of one byte at a time.
#define MXFP4_NIB(w_super, b32)   ((w_super) + (b32) * 16u)
#define MXFP4_SCALE(w_super, b32) ((w_super)[128u + (b32)])

__device__ __forceinline__ void mxfp4_load16(const uint8_t* qs, uint32_t* w) {
    uint2 a_, b_;
    __builtin_memcpy(&a_, qs, 8);
    __builtin_memcpy(&b_, qs + 8, 8);
    w[0] = a_.x; w[1] = a_.y; w[2] = b_.x; w[3] = b_.y;
}
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __constant__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

__device__ __forceinline__ int32_t sudot4_par(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t load_dword_i8_par(const int8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

// 2^(e-128) via bit construction (ggml_e8m0_to_fp32_half): e<2 → denormal
// patterns, else exponent field e-1.
__device__ __forceinline__ float e8m0_half_to_f32(uint32_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

#define MXFP4_STAGE_LUT(s_lut)                                   \
    {                                                             \
        if (threadIdx.x < 16u) s_lut[threadIdx.x] = kvalues_mxfp4[threadIdx.x]; \
        __syncthreads();                                          \
    }

__device__ __forceinline__ float warp_sum_f32_par(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// This lane's 2 mxfp4 sub-blocks of one 256-elem superblock.
// Returns yd * Σ_b scale_b * sumi_b.
__device__ __forceinline__ float dot_super_quarter_mxfp4(
        const uint8_t* w_super, const uint8_t* y_blk, unsigned int sub,
        const int8_t* s_lut) {
    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8_base = (const int8_t*)(y_blk + 4);

    float acc = 0.0f;
    #pragma unroll
    for (unsigned int t = 0; t < 2u; t++) {
        const unsigned int b32 = 2u * sub + t;          // 0..7
        const float scale = e8m0_half_to_f32(MXFP4_SCALE(w_super, b32));
        uint32_t ws_[4];
        mxfp4_load16(MXFP4_NIB(w_super, b32), ws_);
        const int8_t* q8 = q8_base + b32 * 32u;

        int32_t sumi = 0;
        #pragma unroll
        for (unsigned int i = 0; i < 4u; i++) {         // 4 bytes → 1 lo + 1 hi dword
            // Assemble lo-nibble dword (elems 4i..4i+3) and hi-nibble dword
            // (elems 16+4i..16+4i+3).
            union { int32_t d; int8_t b[4]; } lo, hi;
            #pragma unroll
            for (unsigned int j = 0; j < 4u; j++) {
                const uint8_t q = (uint8_t)((ws_[i] >> (8u * j)) & 0xFFu);
                lo.b[j] = s_lut[q & 0x0F];
                hi.b[j] = s_lut[q >> 4];
            }
            sumi = sudot4_par(lo.d, load_dword_i8_par(q8 + 4u * i), sumi);
            sumi = sudot4_par(hi.d, load_dword_i8_par(q8 + 16u + 4u * i), sumi);
        }
        acc += scale * (float)sumi;
    }
    return yd * acc;
}

// Decode batched over n_used selected experts. Contract identical to
// iq3_xxs_matvec_par_batched / q2_k_matvec_par_batched.
extern "C" __global__ void mxfp4_matvec_par_batched(
        float          * __restrict__ out,           // [n_rows]
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,       // slot-major midq_cat
        const int      * __restrict__ selected,      // [n_used]
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    __shared__ int8_t s_lut[16];
    MXFP4_STAGE_LUT(s_lut);

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
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * SUPER_MXFP4_BYTES;
        for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
            acc += dot_super_quarter_mxfp4(
                row_base + (uint64_t)b * SUPER_MXFP4_BYTES,
                xq_slot + (uint64_t)b * BLOCK_Q8_K_BYTES, sub, s_lut);
        }
    }
    acc = warp_sum_f32_par(acc);
    if (lane == 0) out[row] = acc;
}

// Decode het-split; contract identical to the q2_k/iq3 hetsplit kernels.
extern "C" __global__ void mxfp4_matvec_par_batched_hetsplit(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ selected,
        const int      * __restrict__ remap,
        unsigned int     mode,
        unsigned int     dgpu_cap,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    __shared__ int8_t s_lut[16];
    MXFP4_STAGE_LUT(s_lut);

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
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * SUPER_MXFP4_BYTES;
        for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
            acc += dot_super_quarter_mxfp4(
                row_base + (uint64_t)b * SUPER_MXFP4_BYTES,
                xq_slot + (uint64_t)b * BLOCK_Q8_K_BYTES, sub, s_lut);
        }
    }
    acc = warp_sum_f32_par(acc);
    if (lane == 0) out[row] = acc;
}

// Prefill by-expert kwide2 — same geometry/contract as
// iq3_xxs_matvec_par_by_expert_kwide2 (2 rows/warp, halves of 16 members,
// weights unpacked once per (row, superblock), pair with
// q2_k_reduce_partials).
extern "C" __global__ void mxfp4_matvec_par_by_expert_kwide2(
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
        unsigned int     n_blocks_in,
        const int      * __restrict__ n_work_items_dev) { // nullptr = grid.y is exact
    // DEVICE-SIDE COUNT (`V41_MOE_WI_DEVCOUNT`): the grid is an upper bound and
    // work items past the builder's count hold stale entries from earlier
    // layers. Uniform per work-group (blockIdx only), so it may precede any
    // barrier.
    if (n_work_items_dev != nullptr && (int)blockIdx.y >= *n_work_items_dev) return;
    __shared__ int8_t s_lut[16];
    MXFP4_STAGE_LUT(s_lut);

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

    const unsigned int block_lane = lane >> 2u;   // 0..7 (superblock)
    const unsigned int sub        = lane & 3u;    // 0..3 → mxfp4 blocks {2sub, 2sub+1}
    const unsigned int q8_off     = sub * 64u;

    const uint8_t* w_expert = w_base + (uint64_t)e * (uint64_t)dbpe;

    for (unsigned int half_lo = 0; half_lo < n_members; half_lo += 16u) {
        const unsigned int half_n = min(16u, n_members - half_lo);

        float pacc0[16];
        float pacc1[16];
        #pragma unroll
        for (unsigned int m = 0; m < 16u; m++) { pacc0[m] = 0.0f; pacc1[m] = 0.0f; }

        for (unsigned int bl = block_lane; bl < n_blocks_in; bl += 8u) {
            // Unpack BOTH rows' 2 sub-blocks once: signed value dwords +
            // per-sub-block scales.
            int32_t w0[16], w1[16];
            float sc0[2], sc1[2];
            #pragma unroll
            for (unsigned int rr = 0; rr < 2u; rr++) {
                const uint8_t* w_super = w_expert
                    + (uint64_t)(row0 + rr) * (uint64_t)n_blocks_in * SUPER_MXFP4_BYTES
                    + (uint64_t)bl * SUPER_MXFP4_BYTES;
                int32_t* w  = rr ? w1  : w0;
                float*   sc = rr ? sc1 : sc0;
                unsigned int wi = 0;
                #pragma unroll
                for (unsigned int t = 0; t < 2u; t++) {
                    const unsigned int b32_ = 2u * sub + t;
                    sc[t] = e8m0_half_to_f32(MXFP4_SCALE(w_super, b32_));
                    uint32_t qw_[4];
                    mxfp4_load16(MXFP4_NIB(w_super, b32_), qw_);
                    // 4 lo dwords then 4 hi dwords (matches q8 layout order
                    // below: q[8t..8t+3] lo elems, q[8t+4..8t+7] hi elems).
                    #pragma unroll
                    for (unsigned int i = 0; i < 4u; i++) {
                        union { int32_t d; int8_t b[4]; } lo;
                        #pragma unroll
                        for (unsigned int j = 0; j < 4u; j++) {
                            lo.b[j] = s_lut[((qw_[i] >> (8u * j)) & 0xFFu) & 0x0F];
                        }
                        w[wi + i] = lo.d;
                    }
                    #pragma unroll
                    for (unsigned int i = 0; i < 4u; i++) {
                        union { int32_t d; int8_t b[4]; } hi;
                        #pragma unroll
                        for (unsigned int j = 0; j < 4u; j++) {
                            hi.b[j] = s_lut[((qw_[i] >> (8u * j)) & 0xFFu) >> 4];
                        }
                        w[wi + 4u + i] = hi.d;
                    }
                    wi += 8u;
                }
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
                const int8_t* q8b = (const int8_t*)(y_blk + 4) + q8_off;

                // Load this lane's 16 q8 dwords in kernel weight order:
                // per sub-block t: 4 lo dwords (elems 0..15) + 4 hi (16..31).
                int32_t q[16];
                #pragma unroll
                for (unsigned int t = 0; t < 2u; t++) {
                    #pragma unroll
                    for (unsigned int i = 0; i < 4u; i++) {
                        q[8u * t + i]      = load_dword_i8_par(q8b + 32u * t + 4u * i);
                        q[8u * t + 4u + i] = load_dword_i8_par(q8b + 32u * t + 16u + 4u * i);
                    }
                }
                float f0 = 0.0f, f1 = 0.0f;
                #pragma unroll
                for (unsigned int t = 0; t < 2u; t++) {
                    int32_t s0 = 0, s1 = 0;
                    #pragma unroll
                    for (unsigned int i = 0; i < 8u; i++) {
                        s0 = sudot4_par(w0[8u * t + i], q[8u * t + i], s0);
                        s1 = sudot4_par(w1[8u * t + i], q[8u * t + i], s1);
                    }
                    f0 += sc0[t] * (float)s0;
                    f1 += sc1[t] * (float)s1;
                }
                pacc0[m] += yd * f0;
                pacc1[m] += yd * f1;
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

// mxfp4_matvec_par_by_expert_smallb — small-b twin of
// mxfp4_matvec_par_by_expert_kwide2, member-OUTER (A_moe_smallb/dn2_member_outer,
// candidate cand_dn_v2.hip with DN_ROWS=2). Same interface, grid (n_rows/16,
// n_work_items) x 256 and lane -> data map (2 rows per warp, block_lane = lane>>2
// over the 9 super-blocks, sub = lane&3 owns mxfp4 blocks {2sub, 2sub+1}; lanes
// with block_lane == 0 also do super-block 8), same per-lane accumulation order
// (iteration 0 then 1, sc[t]*s_t sums, yd*f, warp_sum) => BIT-EXACT with kwide2.
// What changes: the warp's WHOLE 2-row weight stripe (both super-block
// iterations) is loaded up front and kept unpacked in registers, and the member
// loop is OUTER with scalar accumulators (no pacc[16] arrays, no dynamic VGPR
// indexing: 105 VGPR vs 173).
// 2026-09-26 sweep: kernel 178 -> 142 us at b=4 (bound grid, cold), chain x1.085
// at b=4 (reviewer), bit-exact at b=1..8 incl. 1-member groups and hub-like
// 16-20 distinct experts. Measured only at <= 8 rows / <= 8 members per group:
// dispatch::moe_down_mxfp4 selects it for rows <= 8 (V41_MOE_DOWN_DN2); prefill
// keeps kwide2 / the WMMA arm.
// n_blocks_in is COMPILE-TIME 9 (N_FF_EXP = 2304): the kernel returns WITHOUT
// WRITING otherwise, so the wrapper (Mxfp4Matvec::launch_by_expert_smallb_ex)
// refuses any other n_blocks_in, and n_rows % 16 != 0.
// Includes the work-item loop (w += gridDim.y): a no-op at the production
// grid.y = bound (n_items = *n_work_items_dev, or gridDim.y when null).
#define DN2_NB 9u     // n_blocks_in (2304 / 256)
#define DN2_ROWS 2u
#define DN2_MUNR 1

struct Dn2RawSub { uint2 a0, b0, a1, b1; uint8_t sc0, sc1; };  // this lane's 2 blocks of one super-block
__device__ __forceinline__ void dn2_load_sub(const uint8_t* w_super, unsigned sub, Dn2RawSub& r) {
    const unsigned k0 = 2u * sub, k1 = 2u * sub + 1u;
    __builtin_memcpy(&r.a0, w_super + k0 * 16u, 8);
    __builtin_memcpy(&r.b0, w_super + k0 * 16u + 8u, 8);
    __builtin_memcpy(&r.a1, w_super + k1 * 16u, 8);
    __builtin_memcpy(&r.b1, w_super + k1 * 16u + 8u, 8);
    r.sc0 = w_super[128u + k0];
    r.sc1 = w_super[128u + k1];
}
// Bit-identical to kwide2's unpack: w[8t + i] = lo dwords, w[8t + 4 + i] = hi dwords.
__device__ __forceinline__ void dn2_unpack_sub(const Dn2RawSub& r, const int8_t* s_lut, int32_t w[16], float sc[2]) {
    sc[0] = e8m0_half_to_f32(r.sc0);
    sc[1] = e8m0_half_to_f32(r.sc1);
    const uint32_t qw[2][4] = {{r.a0.x, r.a0.y, r.b0.x, r.b0.y}, {r.a1.x, r.a1.y, r.b1.x, r.b1.y}};
    #pragma unroll
    for (unsigned t = 0; t < 2u; t++) {
        #pragma unroll
        for (unsigned i = 0; i < 4u; i++) {
            union { int32_t d; int8_t b[4]; } lo, hi;
            #pragma unroll
            for (unsigned j = 0; j < 4u; j++) {
                const unsigned q = (qw[t][i] >> (8u * j)) & 0xFFu;
                lo.b[j] = s_lut[q & 0x0F];
                hi.b[j] = s_lut[q >> 4];
            }
            w[8u * t + i]      = lo.d;
            w[8u * t + 4u + i] = hi.d;
        }
    }
}

extern "C" __global__ void mxfp4_matvec_par_by_expert_smallb(
        float          * __restrict__ partials,      // [B*n_used, n_rows]
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,       // [B, n_used*xq_slot_stride]
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        const int      * __restrict__ work_items,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     chunk_size,
        unsigned int     n_rows,
        unsigned int     n_blocks_in,
        const int      * __restrict__ n_work_items_dev) {
    if (n_blocks_in != DN2_NB) return;
    __shared__ int8_t s_lut[16];
    MXFP4_STAGE_LUT(s_lut);

    const unsigned warp_in_block = threadIdx.x / 32u;
    const unsigned lane          = threadIdx.x & 31u;
    const unsigned row0          = blockIdx.x * (8u * DN2_ROWS) + warp_in_block * DN2_ROWS;
    if (row0 >= n_rows) return;

    const unsigned block_lane = lane >> 2u;   // 0..7
    const unsigned sub        = lane & 3u;
    const unsigned q8_off     = sub * 64u;
    const bool     second     = (block_lane + 8u) < DN2_NB;   // block_lane 0 only

    // Work-item loop: with grid.y = the 6b upper bound this runs once (production semantics).
    const unsigned n_items = (n_work_items_dev != nullptr) ? (unsigned)max(0, *n_work_items_dev) : gridDim.y;
    for (unsigned w_idx = blockIdx.y; w_idx < n_items; w_idx += gridDim.y) {
    const int packed_item = work_items[w_idx];
    const unsigned e            = (unsigned)(packed_item >> 16) & 0xffffu;
    const unsigned member_start = (unsigned)(packed_item) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned member_end = min((unsigned)group_n, member_start + chunk_size);
    if (member_end <= member_start) continue;

    const uint8_t* w_r0 = w_base + (uint64_t)e * dbpe + (uint64_t)row0 * DN2_NB * SUPER_MXFP4_BYTES;

    // The warp's whole weight stripe: DN2_ROWS rows x (1 or 2) super-blocks, all loads issued now.
    Dn2RawSub raw[DN2_ROWS][2];
    #pragma unroll
    for (unsigned rr = 0; rr < DN2_ROWS; rr++) {
        const uint8_t* w_r = w_r0 + (uint64_t)rr * DN2_NB * SUPER_MXFP4_BYTES;
        dn2_load_sub(w_r + block_lane * SUPER_MXFP4_BYTES, sub, raw[rr][0]);
        // lanes without a second super-block re-load their first one (valid address, result unused)
        dn2_load_sub(w_r + (second ? block_lane + 8u : block_lane) * SUPER_MXFP4_BYTES, sub, raw[rr][1]);
    }
    int32_t w[DN2_ROWS][2][16];
    float   sc[DN2_ROWS][2][2];
    #pragma unroll
    for (unsigned rr = 0; rr < DN2_ROWS; rr++) {
        dn2_unpack_sub(raw[rr][0], s_lut, w[rr][0], sc[rr][0]);
        dn2_unpack_sub(raw[rr][1], s_lut, w[rr][1], sc[rr][1]);
    }

    #pragma unroll DN2_MUNR
    for (unsigned mi = member_start; mi < member_end; mi++) {
        const int packed = expert_members[(size_t)e * max_per_expert + (size_t)mi];
        const unsigned b    = (unsigned)(packed >> 16) & 0xffffu;
        const unsigned slot = (unsigned)(packed) & 0xffffu;
        const uint8_t* y_m = xq_base + ((size_t)b * n_used + slot) * xq_slot_stride;
        float pacc[DN2_ROWS];
        #pragma unroll
        for (unsigned rr = 0; rr < DN2_ROWS; rr++) pacc[rr] = 0.0f;
        #pragma unroll
        for (unsigned it = 0; it < 2u; it++) {
            if (it == 1u && !second) break;   // divergent for lanes 4..31 (as kwide2's bl loop)
            const unsigned bl = block_lane + 8u * it;
            const uint8_t* y_blk = y_m + (size_t)bl * BLOCK_Q8_K_BYTES;
            float yd;
            __builtin_memcpy(&yd, y_blk, 4);
            const int8_t* q8b = (const int8_t*)(y_blk + 4) + q8_off;
            int32_t q[16];
            #pragma unroll
            for (unsigned t = 0; t < 2u; t++) {
                #pragma unroll
                for (unsigned i = 0; i < 4u; i++) {
                    q[8u * t + i]      = load_dword_i8_par(q8b + 32u * t + 4u * i);
                    q[8u * t + 4u + i] = load_dword_i8_par(q8b + 32u * t + 16u + 4u * i);
                }
            }
            #pragma unroll
            for (unsigned rr = 0; rr < DN2_ROWS; rr++) {
                float f = 0.0f;
                #pragma unroll
                for (unsigned t = 0; t < 2u; t++) {
                    int32_t s = 0;
                    #pragma unroll
                    for (unsigned i = 0; i < 8u; i++) s = sudot4_par(w[rr][it][8u * t + i], q[8u * t + i], s);
                    f += sc[rr][it][t] * (float)s;
                }
                pacc[rr] += yd * f;
            }
        }
        #pragma unroll
        for (unsigned rr = 0; rr < DN2_ROWS; rr++) {
            const float a = warp_sum_f32_par(pacc[rr]);
            if (lane == 0) partials[((size_t)b * n_used + slot) * n_rows + row0 + rr] = a;
        }
    }
    }  // work-item loop
}

// =============================================================================
// Drafter (DSpark) block down projection, GROUPED BY EXPERT
// (`V41_MTP_MOE_GROUPED`; gate+up twin: mxfp4_pair_matvec_fused_swiglu_grouped).
//
// ONE launch for the whole drafter block (n_tok tokens x n_used picks, pick
// p = token * n_used + slot, midq pick-major at p * xq_slot_stride) instead of
// n_tok `mxfp4_matvec_par_batched_hetsplit` launches that each re-read their 3
// experts: a work-group owns 8 rows (one warp per row, as the per-row kernel)
// for EVERY token, walks the block's DISTINCT resident experts once, and each
// lane decodes its share of an expert's row once and applies it to every pick
// of that expert. ~7.6 distinct of 15 picks (MEASURED) = half the bytes.
//
// BIT-IDENTICAL per (token, row) to the per-row kernel (mode 0, cap >= n_used)
// BY CONSTRUCTION. Its per-lane sum runs over slots, then this lane's
// super-blocks, and only then crosses lanes (warp_sum) -- so a token's 3
// experts are summed INSIDE each lane, in slot order, and a by-expert partials
// + reduce (kwide2/smallb + q2_k_reduce_partials) would re-associate it. Here
// the expert walk only fills `inner[p][it]` = `dot_super_quarter_mxfp4`'s
// `acc` (same lane split, same t order), and the epilogue replays the per-row
// chain verbatim: per token, slots in order, super-blocks in order,
// `acc += yd * inner` (the inlined `acc += yd * acc_inner`), then the same
// warp_sum. The walk order cannot leak into any sum.
//
// No LDS beyond the LUT, no host-built groups: each work-group derives the
// distinct set from `selected` itself (<= 16 picks, uniform). Grid
// (n_rows / 8) x 256. Refuses (returns without writing; the launcher checks)
// n_picks > MXFP4_GROUPED_MAX_PICKS or n_blocks_in > 16 (two super-block
// iterations per lane: 9 at N_FF_EXP = 2304).
// =============================================================================
#ifndef MXFP4_GROUPED_MAX_PICKS
#define MXFP4_GROUPED_MAX_PICKS 16u
#endif
#define MXFP4_GROUPED_ITERS 2u

extern "C" __global__ void mxfp4_matvec_par_grouped(
        float          * __restrict__ out,           // [n_tok, n_rows]
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,       // [n_picks, xq_slot_stride]
        const int      * __restrict__ selected,      // [n_picks]
        const int      * __restrict__ remap,         // [n_expert]
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     n_picks,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    if (n_picks > MXFP4_GROUPED_MAX_PICKS || n_blocks_in > 8u * MXFP4_GROUPED_ITERS || n_used == 0u) return;
    __shared__ int8_t s_lut[16];
    MXFP4_STAGE_LUT(s_lut);

    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    // inner[p][it]: this lane's `acc` for pick p, super-block block_lane + 8 * it.
    float inner[MXFP4_GROUPED_MAX_PICKS][MXFP4_GROUPED_ITERS];
    #pragma unroll
    for (unsigned int p = 0; p < MXFP4_GROUPED_MAX_PICKS; p++) {
        #pragma unroll
        for (unsigned int it = 0; it < MXFP4_GROUPED_ITERS; it++) inner[p][it] = 0.0f;
    }

    // Walk the distinct resident experts (first pick of each), uniform.
    for (unsigned int p0 = 0; p0 < n_picks; p0++) {
        const int sel_e = selected[p0];
        if (remap[sel_e] >= 0) continue;
        bool first = true;
        for (unsigned int q = 0; q < p0; q++) {
            if (selected[q] == sel_e) { first = false; break; }
        }
        if (!first) continue;
        const int e = -remap[sel_e] - 1;
        const uint8_t* row_base = w_base + (uint64_t)e * (uint64_t)dbpe
                                + (uint64_t)row * n_blocks_in * SUPER_MXFP4_BYTES;
        #pragma unroll
        for (unsigned int it = 0; it < MXFP4_GROUPED_ITERS; it++) {
            const unsigned int b = block_lane + 8u * it;
            if (b < n_blocks_in) {
                // This lane's 2 mxfp4 blocks of the super-block, decoded ONCE.
                const uint8_t* w_super = row_base + (uint64_t)b * SUPER_MXFP4_BYTES;
                float scale[2];
                int32_t w_lo[2][4], w_hi[2][4];
                #pragma unroll
                for (unsigned int t = 0; t < 2u; t++) {
                    const unsigned int b32 = 2u * sub + t;
                    scale[t] = e8m0_half_to_f32(MXFP4_SCALE(w_super, b32));
                    uint32_t ws_[4];
                    mxfp4_load16(MXFP4_NIB(w_super, b32), ws_);
                    #pragma unroll
                    for (unsigned int i = 0; i < 4u; i++) {
                        union { int32_t d; int8_t b[4]; } lo, hi;
                        #pragma unroll
                        for (unsigned int j = 0; j < 4u; j++) {
                            const uint8_t q = (uint8_t)((ws_[i] >> (8u * j)) & 0xFFu);
                            lo.b[j] = s_lut[q & 0x0F];
                            hi.b[j] = s_lut[q >> 4];
                        }
                        w_lo[t][i] = lo.d;
                        w_hi[t][i] = hi.d;
                    }
                }
                #pragma unroll
                for (unsigned int p = 0; p < MXFP4_GROUPED_MAX_PICKS; p++) {
                    if (p < n_picks && selected[p] == sel_e) {
                        const int8_t* q8_base = (const int8_t*)(xq_base
                            + (uint64_t)p * (uint64_t)xq_slot_stride
                            + (uint64_t)b * BLOCK_Q8_K_BYTES + 4u);
                        float acc = 0.0f;
                        #pragma unroll
                        for (unsigned int t = 0; t < 2u; t++) {
                            const int8_t* q8 = q8_base + (2u * sub + t) * 32u;
                            int32_t sumi = 0;
                            #pragma unroll
                            for (unsigned int i = 0; i < 4u; i++) {
                                sumi = sudot4_par(w_lo[t][i], load_dword_i8_par(q8 + 4u * i), sumi);
                                sumi = sudot4_par(w_hi[t][i], load_dword_i8_par(q8 + 16u + 4u * i), sumi);
                            }
                            acc += scale[t] * (float)sumi;
                        }
                        inner[p][it] = acc;
                    }
                }
            }
        }
    }

    // The per-row kernel's chain, per token.
    float acc = 0.0f;
    unsigned int s = 0;
    unsigned int tok = 0;
    #pragma unroll
    for (unsigned int p = 0; p < MXFP4_GROUPED_MAX_PICKS; p++) {
        if (p < n_picks) {
            if (remap[selected[p]] < 0) {
                #pragma unroll
                for (unsigned int it = 0; it < MXFP4_GROUPED_ITERS; it++) {
                    const unsigned int b = block_lane + 8u * it;
                    if (b < n_blocks_in) {
                        const float yd = *(const float*)(xq_base
                            + (uint64_t)p * (uint64_t)xq_slot_stride
                            + (uint64_t)b * BLOCK_Q8_K_BYTES);
                        acc += yd * inner[p][it];
                    }
                }
            }
            if (++s == n_used) {
                acc = warp_sum_f32_par(acc);
                if (lane == 0) out[(uint64_t)tok * (uint64_t)n_rows + (uint64_t)row] = acc;
                acc = 0.0f;
                s = 0;
                tok++;
            }
        }
    }
}

// =============================================================================
// mxfp4_matvec_par_by_expert_dn3_r{R}  --  DECODE by-expert down ("dn3",
// `V41_MOE_DOWN_DN3`, default OFF). Same arguments, partials and work-item /
// device-count contract as mxfp4_matvec_par_by_expert_smallb (dn2) and kwide2,
// BIT-IDENTICAL partials (tests/mxfp4_down_dn3.rs), different everything else.
//
// WHY. A decode down call moves D experts x 6,266,880 B of MXFP4 weights (5120
// rows x 9 super-blocks x 136 B; D = 4..20 distinct experts of 1-3 members) and
// nothing else of note (activations: 2.6 KB per member, L2-resident; partials:
// 20 KB per member). It is a pure weight stream. dn2 spends per 2-row wave 256
// ds_load_u8 (the nibble LUT, half of it for super-block 8 that only lanes 0-3
// own) + ~700 VALU on 2.4 KB of weights, and re-reads the member activations
// for every row pair. dn3 is built from the bytes:
//
//  * R rows per wave (R = 8: 9.8 KB contiguous per wave). The arithmetic of
//    every output is dn2's EXACTLY: lane (bl = lane>>2, sub = lane&3) owns
//    super-block bl, MXFP4 blocks {2sub, 2sub+1}; f = sc0*s0 + sc1*s1 in that
//    order; pacc = 0 + yd*f; lanes 0-3 then add yd8*f8 for super-block 8; the
//    32 lane values reduce by dn2's tree (offsets 16, 8, 4, 2, 1). What moved
//    is only WHERE each value is computed:
//      - super-block 8 of the wave's R rows is ONE pass (lane = 4*r + sub)
//        instead of R passes at 1/8 lane utilisation; each f8 moves to lane
//        `sub` of its row by a shuffle (exact data movement) before the fma;
//      - the R rows reduce by a TRANSPOSED butterfly: at offset o a lane with
//        bit o set adds row B's pair, a lane with it clear row A's pair --
//        the SAME operand pairs as dn2's shfl_down tree (IEEE + is
//        commutative), so R reductions cost ~R + 4 shuffles, not 5R, and the
//        row r sum ends on lane (32/R) * bitrev(r).
//  * Nibble -> int8 by v_perm (3 perms + 3 ALU per 4 values), no LDS at all.
//  * The whole weight stripe (R x 2 x b128 + R x u16 scales, + one pass for
//    super-block 8) is issued before any is consumed; member activations are
//    loaded once per wave (shared by its R rows) and the NEXT member's are
//    prefetched while the current one computes.
//
// n_blocks_in is COMPILE-TIME 9 (the kernel returns without writing otherwise;
// the launcher refuses it), n_rows % (rows per WG) == 0 (launcher-checked).
// Grid (n_rows / (W * R), n_work_items) x 32W. Includes dn2's work-item loop
// (w += gridDim.y) for the bound grid + device count.
// =============================================================================
#define DN3_NB 9u
#define DN3_ROW_BYTES (DN3_NB * SUPER_MXFP4_BYTES)   // 1224

// kvalues_mxfp4[idx & 15] for the 4 bytes of idx (bits 4..7 of each byte ignored).
// v_perm_b32(src0, src1, sel): sel byte 0..3 = src1 byte, 4..7 = src0 byte.
__device__ __forceinline__ uint32_t dn3_e2m1x4(uint32_t idx) {
    const uint32_t i3  = idx & 0x07070707u;
    const uint32_t pos = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, i3);  // {0,1,2,3,4,6,8,12}
    const uint32_t neg = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, i3);  // {0,-1,-2,-3,-4,-6,-8,-12}
    const uint32_t sel = ((idx & 0x08080808u) >> 1) | 0x03020100u;
    return __builtin_amdgcn_perm(neg, pos, sel);
}

__device__ __forceinline__ uint4 dn3_ld16_a8(const uint8_t* p) {
    uint4 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 8), 16);
    return v;
}
__device__ __forceinline__ uint4 dn3_ld16_a4(const uint8_t* p) {
    uint4 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 4), 16);
    return v;
}

// This lane's MXFP4 blocks {2sub, 2sub+1} of one super-block + their 2 scale bytes.
struct Dn3Raw { uint4 a, b; uint32_t sc; };
__device__ __forceinline__ void dn3_load(const uint8_t* w_super, unsigned sub, Dn3Raw& r) {
    r.a  = dn3_ld16_a8(w_super + 32u * sub);
    r.b  = dn3_ld16_a8(w_super + 32u * sub + 16u);
    // The aligned dword holding this lane's 2 scale bytes (a 16-bit load would be a
    // d16 load that merges into the old register value, i.e. waits on its last load).
    const uint32_t sc4 = *(const uint32_t*)__builtin_assume_aligned(w_super + 128u + 4u * (sub >> 1), 4);
    r.sc = (sc4 >> (16u * (sub & 1u))) & 0xffffu;
}
// An empty asm that "modifies" the raw stripe: keeps a LAZY unpack inside the member loop.
__device__ __forceinline__ void dn3_opaque(Dn3Raw& r) {
    asm volatile("" : "+v"(r.a.x), "+v"(r.a.y), "+v"(r.a.z), "+v"(r.a.w),
                      "+v"(r.b.x), "+v"(r.b.y), "+v"(r.b.z), "+v"(r.b.w), "+v"(r.sc));
}
// dn2_unpack_sub's output (w[8t + i] = lo nibbles of raw dword i of block 2sub+t,
// w[8t + 4 + i] = hi nibbles; sc[t] = 2^(e-128)), same int8 values.
__device__ __forceinline__ void dn3_unpack(const Dn3Raw& r, int32_t w[16], float sc[2]) {
    sc[0] = e8m0_half_to_f32(r.sc & 0xFFu);
    sc[1] = e8m0_half_to_f32(r.sc >> 8);
    const uint32_t q[8] = {r.a.x, r.a.y, r.a.z, r.a.w, r.b.x, r.b.y, r.b.z, r.b.w};
    #pragma unroll
    for (unsigned t = 0; t < 2u; t++) {
        #pragma unroll
        for (unsigned i = 0; i < 4u; i++) {
            w[8u * t + i]      = (int32_t)dn3_e2m1x4(q[4u * t + i]);
            w[8u * t + 4u + i] = (int32_t)dn3_e2m1x4(q[4u * t + i] >> 4);
        }
    }
}

// One member's activations as this lane uses them: super-block bl (q, yd) and
// super-block 8 (q9, yd9), 64 B each at sub's 2 blocks; dword k of q = Q8_K
// bytes 4k..4k+3 of those 64 (= dn2's q[] order).
struct Dn3Act { uint4 q[4], q9[4]; float yd, yd9; unsigned b, slot; };
__device__ __forceinline__ void dn3_load_act(const uint8_t* __restrict__ xq_base, const int* __restrict__ expert_members,
                                             unsigned e, unsigned max_per_expert, unsigned mi, unsigned n_used,
                                             unsigned xq_slot_stride, unsigned bl, unsigned sub, Dn3Act& a) {
    const int packed = expert_members[(size_t)e * max_per_expert + (size_t)mi];
    a.b    = (unsigned)(packed >> 16) & 0xffffu;
    a.slot = (unsigned)(packed) & 0xffffu;
    const uint8_t* y_m = xq_base + ((size_t)a.b * n_used + a.slot) * xq_slot_stride;
    const uint8_t* y0  = y_m + (size_t)bl * BLOCK_Q8_K_BYTES;
    const uint8_t* y8  = y_m + (size_t)8u * BLOCK_Q8_K_BYTES;
    #pragma unroll
    for (unsigned k = 0; k < 4u; k++) {
        a.q[k]  = dn3_ld16_a4(y0 + 4u + 64u * sub + 16u * k);
        a.q9[k] = dn3_ld16_a4(y8 + 4u + 64u * sub + 16u * k);
    }
    __builtin_memcpy(&a.yd, __builtin_assume_aligned(y0, 4), 4);
    __builtin_memcpy(&a.yd9, __builtin_assume_aligned(y8, 4), 4);
}

// dn2's per-row-per-iteration `f` (sc[0]*s0 then sc[1]*s1, s = 8 sudot4 each).
__device__ __forceinline__ float dn3_f(const int32_t w[16], const float sc[2], const uint4 q4[4]) {
    const int32_t q[16] = {(int32_t)q4[0].x, (int32_t)q4[0].y, (int32_t)q4[0].z, (int32_t)q4[0].w,
                           (int32_t)q4[1].x, (int32_t)q4[1].y, (int32_t)q4[1].z, (int32_t)q4[1].w,
                           (int32_t)q4[2].x, (int32_t)q4[2].y, (int32_t)q4[2].z, (int32_t)q4[2].w,
                           (int32_t)q4[3].x, (int32_t)q4[3].y, (int32_t)q4[3].z, (int32_t)q4[3].w};
    float f = 0.0f;
    #pragma unroll
    for (unsigned t = 0; t < 2u; t++) {
        int32_t s = 0;
        #pragma unroll
        for (unsigned i = 0; i < 8u; i++) s = sudot4_par(w[8u * t + i], q[8u * t + i], s);
        f += sc[t] * (float)s;
    }
    return f;
}

template <unsigned R, bool LAZY>
__device__ __forceinline__ void dn3_body(
        float          * __restrict__ partials,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        const int      * __restrict__ work_items,
        unsigned dbpe, unsigned xq_slot_stride, unsigned n_used, unsigned max_per_expert,
        unsigned chunk_size, unsigned n_rows, unsigned n_blocks_in,
        const int      * __restrict__ n_work_items_dev) {
    static_assert(R == 1u || R == 2u || R == 4u || R == 8u, "R must be 1, 2, 4 or 8");
    constexpr unsigned LOG_R = (R == 8u) ? 3u : (R == 4u) ? 2u : (R == 2u) ? 1u : 0u;
    constexpr unsigned GSZ = 32u / R;            // lanes per row after the transposed steps
    if (n_blocks_in != DN3_NB) return;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
    if (row0 + R > n_rows) return;
    const unsigned bl  = lane >> 2u;              // super-block (passes 0..R-1); row (pass 8)
    const unsigned sub = lane & 3u;
    const bool has8 = bl < R;                     // lane computes super-block 8 of row bl

    const unsigned n_items = (n_work_items_dev != nullptr) ? (unsigned)max(0, *n_work_items_dev) : gridDim.y;
    for (unsigned w_idx = blockIdx.y; w_idx < n_items; w_idx += gridDim.y) {
    const int packed_item = work_items[w_idx];
    const unsigned e            = (unsigned)(packed_item >> 16) & 0xffffu;
    const unsigned member_start = (unsigned)(packed_item) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned member_end = min((unsigned)group_n, member_start + chunk_size);
    if (member_end <= member_start) continue;

    const uint8_t* w_r0 = w_base + (uint64_t)e * dbpe + (uint64_t)row0 * DN3_ROW_BYTES;

    // First member's activations (L2-resident), then the whole weight stripe.
    Dn3Act cur;
    dn3_load_act(xq_base, expert_members, e, max_per_expert, member_start, n_used, xq_slot_stride, bl, sub, cur);
    Dn3Raw raw[R];
    #pragma unroll
    for (unsigned p = 0; p < R; p++) dn3_load(w_r0 + p * DN3_ROW_BYTES + bl * SUPER_MXFP4_BYTES, sub, raw[p]);
    // Lanes past the R rows (R < 8) re-load a valid row's super-block 8 (result unused):
    // no divergent load, so the compiler's vmcnt waits stay exact.
    Dn3Raw raw8;
    dn3_load(w_r0 + (has8 ? bl : (bl & (R - 1u))) * DN3_ROW_BYTES + 8u * SUPER_MXFP4_BYTES, sub, raw8);
    // Eager: unpack once (16 + 2 VGPRs per pass). LAZY: keep the raw stripe (9 VGPRs per
    // pass) and unpack per member -- the R = 8 register budget.
    constexpr unsigned RU = LAZY ? 1u : R;
    int32_t w[RU][16];
    float   sc[RU][2];
    int32_t w8[16];
    float   sc8[2];
    if constexpr (!LAZY) {
        #pragma unroll
        for (unsigned p = 0; p < R; p++) dn3_unpack(raw[p], w[p], sc[p]);
        dn3_unpack(raw8, w8, sc8);
    }

    for (unsigned mi = member_start; mi < member_end; mi++) {
        Dn3Act nxt;
        const bool more = mi + 1u < member_end;
        // Unconditional (the last member re-loads itself, result unused): exact vmcnt waits.
        dn3_load_act(xq_base, expert_members, e, max_per_expert, more ? mi + 1u : mi, n_used, xq_slot_stride, bl, sub, nxt);

        float v[R];
        #pragma unroll
        for (unsigned p = 0; p < R; p++) {
            float f;
            if constexpr (LAZY) {
                // Opaque per iteration so the unpack is not hoisted out of the member loop.
                dn3_opaque(raw[p]);
                dn3_unpack(raw[p], w[0], sc[0]);
                f = dn3_f(w[0], sc[0], cur.q);
            } else {
                f = dn3_f(w[p], sc[p], cur.q);
            }
            float pacc = 0.0f;
            pacc += cur.yd * f;
            v[p] = pacc;
        }
        // Super-block 8: lane 4r + sub holds row r's f8; lane sub of row r adds it (dn2's it = 1).
        if constexpr (LAZY) {
            dn3_opaque(raw8);
            dn3_unpack(raw8, w8, sc8);
        }
        const float f8 = dn3_f(w8, sc8, cur.q9);
        #pragma unroll
        for (unsigned p = 0; p < R; p++) {
            const float g = __shfl(f8, (int)(4u * p + sub), 32);
            if (bl == 0u) v[p] += cur.yd9 * g;
        }
        // Transposed butterfly (offsets 16 .. 32/R), then dn2's plain steps.
        #pragma unroll
        for (unsigned lev = 0; lev < LOG_R; lev++) {
            const unsigned o = 16u >> lev;
            const bool up = (lane & o) != 0u;
            #pragma unroll
            for (unsigned k = 0; k < (R >> (lev + 1u)); k++) {
                const float A = v[2u * k], B = v[2u * k + 1u];
                const float send = up ? A : B;
                const float keep = up ? B : A;
                v[k] = keep + __shfl_xor(send, (int)o, 32);
            }
        }
        #pragma unroll
        for (unsigned o = 16u >> LOG_R; o >= 1u; o >>= 1) v[0] = v[0] + __shfl_xor(v[0], (int)o, 32);
        if ((lane & (GSZ - 1u)) == 0u) {
            const unsigned g  = lane / GSZ;
            const unsigned rr = (LOG_R == 0u) ? 0u : (__builtin_bitreverse32(g) >> (32u - LOG_R));
            partials[((size_t)cur.b * n_used + cur.slot) * n_rows + row0 + rr] = v[0];
        }
        if (more) cur = nxt;
    }
    }  // work-item loop
}

#define DN3_KERNEL(NAME_, R_, LAZY_)                                                           \
extern "C" __global__ __launch_bounds__(256) void mxfp4_matvec_par_by_expert_dn3_##NAME_(        \
        float* __restrict__ partials, const uint8_t* __restrict__ w_base,                      \
        const uint8_t* __restrict__ xq_base, const int* __restrict__ group_count,              \
        const int* __restrict__ expert_members, const int* __restrict__ work_items,            \
        unsigned dbpe, unsigned xq_slot_stride, unsigned n_used, unsigned max_per_expert,      \
        unsigned chunk_size, unsigned n_rows, unsigned n_blocks_in,                            \
        const int* __restrict__ n_work_items_dev) {                                            \
    dn3_body<R_, LAZY_>(partials, w_base, xq_base, group_count, expert_members, work_items, dbpe,     \
                 xq_slot_stride, n_used, max_per_expert, chunk_size, n_rows, n_blocks_in,      \
                 n_work_items_dev);                                                            \
}
DN3_KERNEL(r2, 2, false)
DN3_KERNEL(r4, 4, false)
DN3_KERNEL(r8, 8, true)
DN3_KERNEL(r4l, 4, true)

// -----------------------------------------------------------------------------
// dn3p: the same per-output arithmetic on a PERSISTENT 1-D grid.
//
// The production launch sizes grid.y by the work-item UPPER BOUND (6b, the
// device count is only known on the device), so a b = 8 call over D = 4
// experts launches 48 work-item columns of which 44 exit at once -- after the
// real ones, as a tail of empty waves (~0.4 ns each, measured: +24 us at b = 8,
// D = 4 for a 16-row WG grid). dn3p launches G work-groups (G ~ resident WG
// slots) that walk the flattened (work item, row tile) list t = blockIdx.x,
// + G, ... up to n_items * tiles, n_items read from the device count: no empty
// waves, launch cost independent of the bound. Consecutive t are consecutive
// row tiles of one expert, so the G work-groups in flight stream one
// contiguous region.
//
// PF: the next tile's weight stripe is issued right after the current one is
// unpacked, so every wave keeps one stripe in flight while it computes. Loads
// return in issue order (vmcnt), so the current tile's first two members'
// activations are issued BEFORE that prefetch (members beyond the second load
// after it and wait for it -- correct, only slower; decode has 1-3 members).
// -----------------------------------------------------------------------------
template <unsigned R>
__device__ __forceinline__ void dn3_member(const int32_t (&w)[R][16], const float (&sc)[R][2],
                                           const int32_t (&w8)[16], const float (&sc8)[2], const Dn3Act& a,
                                           unsigned lane, unsigned bl, unsigned sub,
                                           float* __restrict__ partials, unsigned n_used, unsigned n_rows,
                                           unsigned row0) {
    constexpr unsigned LOG_R = (R == 8u) ? 3u : (R == 4u) ? 2u : (R == 2u) ? 1u : 0u;
    constexpr unsigned GSZ = 32u / R;
    float v[R];
    #pragma unroll
    for (unsigned p = 0; p < R; p++) {
        const float f = dn3_f(w[p], sc[p], a.q);
        float pacc = 0.0f;
        pacc += a.yd * f;
        v[p] = pacc;
    }
    const float f8 = dn3_f(w8, sc8, a.q9);
    #pragma unroll
    for (unsigned p = 0; p < R; p++) {
        const float g = __shfl(f8, (int)(4u * p + sub), 32);
        if (bl == 0u) v[p] += a.yd9 * g;
    }
    #pragma unroll
    for (unsigned lev = 0; lev < LOG_R; lev++) {
        const unsigned o = 16u >> lev;
        const bool up = (lane & o) != 0u;
        #pragma unroll
        for (unsigned k = 0; k < (R >> (lev + 1u)); k++) {
            const float A = v[2u * k], B = v[2u * k + 1u];
            const float send = up ? A : B;
            const float keep = up ? B : A;
            v[k] = keep + __shfl_xor(send, (int)o, 32);
        }
    }
    #pragma unroll
    for (unsigned o = 16u >> LOG_R; o >= 1u; o >>= 1) v[0] = v[0] + __shfl_xor(v[0], (int)o, 32);
    if ((lane & (GSZ - 1u)) == 0u) {
        const unsigned g  = lane / GSZ;
        const unsigned rr = (LOG_R == 0u) ? 0u : (__builtin_bitreverse32(g) >> (32u - LOG_R));
        partials[((size_t)a.b * n_used + a.slot) * n_rows + row0 + rr] = v[0];
    }
}

template <unsigned R>
__device__ __forceinline__ void dn3_load_stripe(const uint8_t* w_r0, unsigned bl, unsigned sub, bool has8,
                                                Dn3Raw (&raw)[R], Dn3Raw& raw8) {
    #pragma unroll
    for (unsigned p = 0; p < R; p++) dn3_load(w_r0 + p * DN3_ROW_BYTES + bl * SUPER_MXFP4_BYTES, sub, raw[p]);
    dn3_load(w_r0 + (has8 ? bl : (bl & (R - 1u))) * DN3_ROW_BYTES + 8u * SUPER_MXFP4_BYTES, sub, raw8);
}

struct Dn3Tile { unsigned e, ms, me, row0; };

// Sub-staging builds only (DEEPSTRIX_KERNEL_CFLAGS=-DDN3_ABLATE=1|2): 1 = no unpack /
// dot / reduce (XOR fold of the loaded data), 2 = also no activation loads.
#ifndef DN3_ABLATE
#define DN3_ABLATE 0
#endif

// PF: 0 = none (next stripe loaded after the members), 1 = after the current stripe
// is unpacked (reuses its registers), 2 = EARLY: before the unpack, into a second
// register set (one stripe always in flight; R = 2 register budget).
// ACTR: keep the members' activations in registers while consecutive tiles belong to
// the same work item (tiles G apart, 320 per item: most do) instead of re-loading.
template <unsigned R, unsigned PF, unsigned MODE, bool ACTR>
__device__ __forceinline__ void dn3p_body(
        float          * __restrict__ partials,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        const int      * __restrict__ work_items,
        unsigned dbpe, unsigned xq_slot_stride, unsigned n_used, unsigned max_per_expert,
        unsigned chunk_size, unsigned n_rows, unsigned n_blocks_in,
        const int      * __restrict__ n_work_items_dev, unsigned n_items_host) {
    // MODE 0: flat persistent walk t = blockIdx.x, + G, ...; 1 (COL): grid (tiles, Y),
    // the WG keeps its row tile and walks work items blockIdx.y, + Y, ...; 2: flat
    // persistent, WG g takes the CONTIGUOUS tiles [g * per, (g + 1) * per).
    constexpr bool COL = MODE == 1u;
    if (n_blocks_in != DN3_NB) return;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned rows_wg = (blockDim.x >> 5) * R;
    const unsigned tiles = n_rows / rows_wg;                // launcher: n_rows % rows_wg == 0
    const unsigned wave_row = (threadIdx.x >> 5) * R;
    const unsigned bl  = lane >> 2u;
    const unsigned sub = lane & 3u;
    const bool has8 = bl < R;
    // dn2's item count: the device count when given (the grid is then only an upper
    // bound), else the launch's.
    const unsigned n_items = (n_work_items_dev != nullptr) ? (unsigned)max(0, *n_work_items_dev) : n_items_host;
    // COL: grid (tiles, Y) -- the WG keeps its row tile blockIdx.x and walks work items
    // blockIdx.y, + Y, ... (Y = grid.y <= the bound); else the flat persistent walk.
    const unsigned total = COL ? n_items : n_items * tiles;
    // Consecutive tiles of a work-group are nearly always in the same work item (G <
    // tiles per item): reuse its (e, ms, me) instead of two dependent scalar loads that
    // the next tile's prefetch would wait on.
    unsigned memo_item = 0xffffffffu;
    Dn3Tile memo{};
    auto resolve = [&](unsigned t) -> Dn3Tile {
        const unsigned item = COL ? t : t / tiles;
        if (item != memo_item) {
            const int packed_item = work_items[item];
            memo.e  = (unsigned)(packed_item >> 16) & 0xffffu;
            memo.ms = (unsigned)(packed_item) & 0xffffu;
            memo.me = min((unsigned)group_count[memo.e], memo.ms + chunk_size);
            memo_item = item;
        }
        Dn3Tile x = memo;
        x.row0 = (COL ? blockIdx.x : (t - item * tiles)) * rows_wg + wave_row;
        return x;
    };
    auto stripe = [&](const Dn3Tile& x) -> const uint8_t* {
        return w_base + (uint64_t)x.e * dbpe + (uint64_t)x.row0 * DN3_ROW_BYTES;
    };
    const unsigned per = (MODE == 2u) ? (total + gridDim.x - 1u) / gridDim.x : 0u;
    const unsigned t_stride = COL ? gridDim.y : (MODE == 2u ? 1u : gridDim.x);
    unsigned t = COL ? blockIdx.y : (MODE == 2u ? blockIdx.x * per : blockIdx.x);
    const unsigned t_end = (MODE == 2u) ? min(total, t + per) : total;
    // First tile: its work item is read BEFORE the device-count check (the grid never
    // exceeds the host bound, so item < n_items_host <= work_items.len()) so the two
    // loads overlap; its stripe is issued as soon as the expert is known -- a builder
    // item always has >= 1 member and the address is valid whatever group_count says --
    // so it does not also wait for group_count[e].
    const unsigned item0 = COL ? t : t / tiles;
    // (MODE 2: t depends on the count, so this read follows it; t < total keeps it in range.)
    if (MODE == 2u && t >= t_end) return;
    const int packed0 = work_items[item0];
    if (t >= t_end) return;
    Dn3Tile cur;
    cur.e  = (unsigned)(packed0 >> 16) & 0xffffu;
    cur.ms = (unsigned)(packed0) & 0xffffu;
    cur.row0 = (COL ? blockIdx.x : (t - item0 * tiles)) * rows_wg + wave_row;
    Dn3Raw raw[R];
    Dn3Raw raw8;
    dn3_load_stripe<R>(stripe(cur), bl, sub, has8, raw, raw8);
    cur.me = min((unsigned)group_count[cur.e], cur.ms + chunk_size);
    memo = cur;
    memo_item = item0;
    Dn3Act a0, a1;
    unsigned act_e = 0xffffffffu, act_ms = 0xffffffffu;   // the item a0/a1 hold (ACTR)
    // One tile: `rc` holds its stripe (issued earlier), the next tile's stripe goes to
    // `rn` (PF 2: a second register set, the loop ping-pongs the two so no copy waits
    // on the prefetched data; PF 0/1: rn aliases rc). Returns false after the last tile.
    auto step = [&](Dn3Raw (&rc)[R], Dn3Raw& rc8, Dn3Raw (&rn)[R], Dn3Raw& rn8) -> bool {
        const unsigned tn = t + t_stride;
        const bool has_next = tn < t_end;
        Dn3Tile nxt = cur;
        if (has_next) nxt = resolve(tn);
        const bool pf_ok = has_next && nxt.me > nxt.ms;
        if (cur.me > cur.ms) {
            const bool two = cur.ms + 1u < cur.me;
            // Activations of the first two members, issued before any prefetch (loads
            // return in order). Unconditional loads (the second re-loads the first when
            // there is one member) keep the compiler's vmcnt waits exact.
            if ((!ACTR || cur.e != act_e || cur.ms != act_ms) && DN3_ABLATE < 2) {
                dn3_load_act(xq_base, expert_members, cur.e, max_per_expert, cur.ms, n_used, xq_slot_stride, bl, sub, a0);
                dn3_load_act(xq_base, expert_members, cur.e, max_per_expert, two ? cur.ms + 1u : cur.ms, n_used, xq_slot_stride, bl, sub, a1);
                act_e = cur.e;
                act_ms = cur.ms;
            }
            // The machine scheduler would hoist the prefetch above the activation loads
            // (the compute would then wait for the prefetch): pin the order.
            if constexpr (PF != 0u) __builtin_amdgcn_sched_barrier(0);
            if constexpr (PF == 2u) {
                // Branch-free (no next / empty next: re-load this stripe, L2 hits).
                dn3_load_stripe<R>(stripe(pf_ok ? nxt : cur), bl, sub, has8, rn, rn8);
                __builtin_amdgcn_sched_barrier(0);
            }
#if DN3_ABLATE >= 1
            // ABLATION (sub-staging only, never the default build): consume the stripe and
            // activations by an XOR fold instead of unpack + dot + reduce.
            {
                uint32_t x = rc8.a.x ^ rc8.a.y ^ rc8.a.z ^ rc8.a.w ^ rc8.b.x ^ rc8.b.y ^ rc8.b.z ^ rc8.b.w ^ rc8.sc;
                #pragma unroll
                for (unsigned p = 0; p < R; p++)
                    x ^= rc[p].a.x ^ rc[p].a.y ^ rc[p].a.z ^ rc[p].a.w ^ rc[p].b.x ^ rc[p].b.y ^ rc[p].b.z ^ rc[p].b.w ^ rc[p].sc;
                #pragma unroll
                for (unsigned k = 0; k < 4u; k++) x ^= a0.q[k].x ^ a0.q9[k].w ^ a1.q[k].y ^ a1.q9[k].z;
                if constexpr (PF == 1u) dn3_load_stripe<R>(stripe(pf_ok ? nxt : cur), bl, sub, has8, rn, rn8);
                if (x == 0x9e3779b9u) partials[cur.row0] = (float)x;
            }
            if (false) {
#else
            {
#endif
            int32_t w[R][16];
            float   sc[R][2];
            int32_t w8[16];
            float   sc8[2];
            #pragma unroll
            for (unsigned p = 0; p < R; p++) dn3_unpack(rc[p], w[p], sc[p]);
            dn3_unpack(rc8, w8, sc8);
            if constexpr (PF == 1u) {
                __builtin_amdgcn_sched_barrier(0);
                dn3_load_stripe<R>(stripe(pf_ok ? nxt : cur), bl, sub, has8, rn, rn8);
                __builtin_amdgcn_sched_barrier(0);
            }
            dn3_member<R>(w, sc, w8, sc8, a0, lane, bl, sub, partials, n_used, n_rows, cur.row0);
            if (two) dn3_member<R>(w, sc, w8, sc8, a1, lane, bl, sub, partials, n_used, n_rows, cur.row0);
            for (unsigned mi = cur.ms + 2u; mi < cur.me; mi++) {
                Dn3Act a;
                dn3_load_act(xq_base, expert_members, cur.e, max_per_expert, mi, n_used, xq_slot_stride, bl, sub, a);
                dn3_member<R>(w, sc, w8, sc8, a, lane, bl, sub, partials, n_used, n_rows, cur.row0);
            }
            }  // compute (DN3_ABLATE)
            if constexpr (PF == 0u) dn3_load_stripe<R>(stripe(pf_ok ? nxt : cur), bl, sub, has8, rn, rn8);
        } else if (pf_ok) {
            dn3_load_stripe<R>(stripe(nxt), bl, sub, has8, rn, rn8);
        }
        t = tn;
        cur = nxt;
        return has_next;
    };
    if constexpr (PF == 2u) {
        Dn3Raw rawn[R];
        Dn3Raw rawn8;
        while (step(raw, raw8, rawn, rawn8) && step(rawn, rawn8, raw, raw8)) {}
    } else {
        while (step(raw, raw8, raw, raw8)) {}
    }
}

// -----------------------------------------------------------------------------
// dn3l: dn3p (persistent, strided walk, early prefetch, activation reuse) with the
// weight stripe loaded FULLY COALESCED and staged through a wave-private LDS slice.
//
// The compute lane map is fixed by the bit-exact arithmetic: lane (bl, sub) needs
// bytes bl*136 + 32*sub .. +32 and its 2 scale bytes of each super-block. Loading
// that directly costs 3 loads per lane per 1088 B, each touching half of every
// 128 B line -- measured 2-4% below a coalesced read of the same bytes
// (ceil_read_slots_mx vs ceil_read_slots, tests/mxfp4_down_dn3.rs). Here a wave
// reads its R-row stripe (R * 1224 B, contiguous) as 16 B per lane per load, 512 B
// per instruction, writes it to its LDS slice and reads its lane's bytes back --
// the same bytes in the same lanes, so the arithmetic (and every bit) is dn3p's.
// No barriers: the slice is wave-private and a wave's LDS operations execute in
// order. R even (the stripe is a whole number of 16 B chunks).
// -----------------------------------------------------------------------------
// A native LLVM vector (not HIP's uint4 struct, whose copies lower to memcpys that
// kept the stripe in scratch).
typedef unsigned int dn3_v4u __attribute__((ext_vector_type(4)));

template <unsigned R>
struct Dn3Co {
    static constexpr unsigned NCH = R * DN3_ROW_BYTES / 16u;      // 16 B chunks (R = 2: 153)
    static constexpr unsigned NV  = (NCH + 31u) / 32u;            // loads per lane (R = 2: 5)
};

template <unsigned R>
__device__ __forceinline__ void dn3_load_co(const uint8_t* w_r0, unsigned lane, dn3_v4u (&v)[Dn3Co<R>::NV]) {
    static_assert((R * DN3_ROW_BYTES) % 16u == 0u, "dn3l needs an even R");
    #pragma unroll
    for (unsigned k = 0; k < Dn3Co<R>::NV; k++) {
        // Lanes past the stripe re-load its last chunk (result written to the same
        // LDS bytes with the same value): no divergent load, exact vmcnt waits.
        const unsigned ch = min(32u * k + lane, Dn3Co<R>::NCH - 1u);
        v[k] = *(const dn3_v4u*)__builtin_assume_aligned(w_r0 + 16u * ch, 16);
    }
}

// Stage the stripe in the wave's LDS slice / read this lane's dn3_load view back.
template <unsigned R>
__device__ __forceinline__ void dn3_lds_put(const dn3_v4u (&v)[Dn3Co<R>::NV], uint8_t* slice, unsigned lane) {
    #pragma unroll
    for (unsigned k = 0; k < Dn3Co<R>::NV; k++) {
        const unsigned ch = min(32u * k + lane, Dn3Co<R>::NCH - 1u);
        *(dn3_v4u*)__builtin_assume_aligned(slice + 16u * ch, 16) = v[k];
    }
}
template <unsigned R>
__device__ __forceinline__ void dn3_lds_get(const uint8_t* slice, unsigned bl, unsigned sub, bool has8,
                                            Dn3Raw (&raw)[R], Dn3Raw& raw8) {
    auto rd = [&](const uint8_t* w_super, Dn3Raw& r) {
        __builtin_memcpy(&r.a, __builtin_assume_aligned(w_super + 32u * sub, 8), 16);
        __builtin_memcpy(&r.b, __builtin_assume_aligned(w_super + 32u * sub + 16u, 8), 16);
        const uint32_t sc4 = *(const uint32_t*)__builtin_assume_aligned(w_super + 128u + 4u * (sub >> 1), 4);
        r.sc = (sc4 >> (16u * (sub & 1u))) & 0xffffu;
    };
    #pragma unroll
    for (unsigned p = 0; p < R; p++) rd(slice + p * DN3_ROW_BYTES + bl * SUPER_MXFP4_BYTES, raw[p]);
    rd(slice + (has8 ? bl : (bl & (R - 1u))) * DN3_ROW_BYTES + 8u * SUPER_MXFP4_BYTES, raw8);
}

template <unsigned R, unsigned W, bool ACTR>
__device__ __forceinline__ void dn3l_body(
        float          * __restrict__ partials,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        const int      * __restrict__ work_items,
        unsigned dbpe, unsigned xq_slot_stride, unsigned n_used, unsigned max_per_expert,
        unsigned chunk_size, unsigned n_rows, unsigned n_blocks_in,
        const int      * __restrict__ n_work_items_dev, unsigned n_items_host) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[W * Dn3Co<R>::NCH * 16u];
    if (n_blocks_in != DN3_NB || blockDim.x != 32u * W) return;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned wave = threadIdx.x >> 5;
    uint8_t* slice = lds + wave * (Dn3Co<R>::NCH * 16u);
    const unsigned rows_wg = W * R;
    const unsigned tiles = n_rows / rows_wg;                // launcher: n_rows % rows_wg == 0
    const unsigned wave_row = wave * R;
    const unsigned bl  = lane >> 2u;
    const unsigned sub = lane & 3u;
    const bool has8 = bl < R;
    const unsigned n_items = (n_work_items_dev != nullptr) ? (unsigned)max(0, *n_work_items_dev) : n_items_host;
    const unsigned total = n_items * tiles;
    // Memoised per work item (see dn3p_body).
    unsigned memo_item = 0xffffffffu;
    Dn3Tile memo{};
    auto resolve = [&](unsigned t) -> Dn3Tile {
        const unsigned item = t / tiles;
        if (item != memo_item) {
            const int packed_item = work_items[item];
            memo.e  = (unsigned)(packed_item >> 16) & 0xffffu;
            memo.ms = (unsigned)(packed_item) & 0xffffu;
            memo.me = min((unsigned)group_count[memo.e], memo.ms + chunk_size);
            memo_item = item;
        }
        Dn3Tile x = memo;
        x.row0 = (t - item * tiles) * rows_wg + wave_row;
        return x;
    };
    auto stripe = [&](const Dn3Tile& x) -> const uint8_t* {
        return w_base + (uint64_t)x.e * dbpe + (uint64_t)x.row0 * DN3_ROW_BYTES;
    };
    unsigned t = blockIdx.x;
    const unsigned item0 = t / tiles;              // < n_items_host: read before the count
    const int packed0 = work_items[item0];
    if (t >= total) return;
    Dn3Tile cur;
    cur.e  = (unsigned)(packed0 >> 16) & 0xffffu;
    cur.ms = (unsigned)(packed0) & 0xffffu;
    cur.row0 = (t - item0 * tiles) * rows_wg + wave_row;
    dn3_v4u co[Dn3Co<R>::NV];
    dn3_load_co<R>(stripe(cur), lane, co);
    cur.me = min((unsigned)group_count[cur.e], cur.ms + chunk_size);
    memo = cur;
    memo_item = item0;
    Dn3Act a0, a1;
    unsigned act_e = 0xffffffffu, act_ms = 0xffffffffu;
    while (true) {
        const unsigned tn = t + gridDim.x;
        const bool has_next = tn < total;
        Dn3Tile nxt = cur;
        if (has_next) nxt = resolve(tn);
        const bool pf_ok = has_next && nxt.me > nxt.ms;
        if (cur.me > cur.ms) {
            const bool two = cur.ms + 1u < cur.me;
            // Stage the current stripe first (the only loads in flight here are its own),
            // freeing its registers; then the activations (if this is a new work item);
            // then the next stripe into the same registers (branch-free: no / empty next
            // re-loads this one), all BEFORE the unpack + compute, so a stripe is always
            // in flight and the compute waits only for the activations.
            dn3_lds_put<R>(co, slice, lane);
            __builtin_amdgcn_sched_barrier(0);
            if (!ACTR || cur.e != act_e || cur.ms != act_ms) {
                dn3_load_act(xq_base, expert_members, cur.e, max_per_expert, cur.ms, n_used, xq_slot_stride, bl, sub, a0);
                dn3_load_act(xq_base, expert_members, cur.e, max_per_expert, two ? cur.ms + 1u : cur.ms, n_used, xq_slot_stride, bl, sub, a1);
                act_e = cur.e;
                act_ms = cur.ms;
            }
            __builtin_amdgcn_sched_barrier(0);
            dn3_load_co<R>(stripe(pf_ok ? nxt : cur), lane, co);
            __builtin_amdgcn_sched_barrier(0);
            Dn3Raw raw[R];
            Dn3Raw raw8;
            dn3_lds_get<R>(slice, bl, sub, has8, raw, raw8);
            int32_t w[R][16];
            float   sc[R][2];
            int32_t w8[16];
            float   sc8[2];
            #pragma unroll
            for (unsigned p = 0; p < R; p++) dn3_unpack(raw[p], w[p], sc[p]);
            dn3_unpack(raw8, w8, sc8);
            dn3_member<R>(w, sc, w8, sc8, a0, lane, bl, sub, partials, n_used, n_rows, cur.row0);
            if (two) dn3_member<R>(w, sc, w8, sc8, a1, lane, bl, sub, partials, n_used, n_rows, cur.row0);
            for (unsigned mi = cur.ms + 2u; mi < cur.me; mi++) {
                Dn3Act a;
                dn3_load_act(xq_base, expert_members, cur.e, max_per_expert, mi, n_used, xq_slot_stride, bl, sub, a);
                dn3_member<R>(w, sc, w8, sc8, a, lane, bl, sub, partials, n_used, n_rows, cur.row0);
            }
        } else if (pf_ok) {
            dn3_load_co<R>(stripe(nxt), lane, co);
        }
        if (!has_next) break;
        t = tn;
        cur = nxt;
    }
}

#define DN3L_KERNEL(NAME_, R_, W_, ACTR_)                                                      \
extern "C" __global__ __launch_bounds__(32 * W_) void mxfp4_matvec_par_by_expert_##NAME_(      \
        float* __restrict__ partials, const uint8_t* __restrict__ w_base,                      \
        const uint8_t* __restrict__ xq_base, const int* __restrict__ group_count,              \
        const int* __restrict__ expert_members, const int* __restrict__ work_items,            \
        unsigned dbpe, unsigned xq_slot_stride, unsigned n_used, unsigned max_per_expert,      \
        unsigned chunk_size, unsigned n_rows, unsigned n_blocks_in,                            \
        const int* __restrict__ n_work_items_dev, unsigned n_items_host) {                     \
    dn3l_body<R_, W_, ACTR_>(partials, w_base, xq_base, group_count, expert_members,           \
                             work_items, dbpe, xq_slot_stride, n_used, max_per_expert,         \
                             chunk_size, n_rows, n_blocks_in, n_work_items_dev, n_items_host);  \
}
DN3L_KERNEL(dn3l_r2w4a, 2, 4, true)
DN3L_KERNEL(dn3l_r2w4, 2, 4, false)
DN3L_KERNEL(dn3l_r4w4a, 4, 4, true)
DN3L_KERNEL(dn3l_r2w2a, 2, 2, true)
DN3L_KERNEL(dn3l_r2w8a, 2, 8, true)
DN3L_KERNEL(dn3l_r4w8a, 4, 8, true)

// WPE_: minimum waves per SIMD the register allocation must allow (amdgpu_waves_per_eu).
#define DN3P_KERNEL_W(NAME_, R_, PF_, MODE_, ACTR_, WPE_)                                      \
extern "C" __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(WPE_)))         \
void mxfp4_matvec_par_by_expert_##NAME_(                                                       \
        float* __restrict__ partials, const uint8_t* __restrict__ w_base,                      \
        const uint8_t* __restrict__ xq_base, const int* __restrict__ group_count,              \
        const int* __restrict__ expert_members, const int* __restrict__ work_items,            \
        unsigned dbpe, unsigned xq_slot_stride, unsigned n_used, unsigned max_per_expert,      \
        unsigned chunk_size, unsigned n_rows, unsigned n_blocks_in,                            \
        const int* __restrict__ n_work_items_dev, unsigned n_items_host) {                     \
    dn3p_body<R_, PF_, MODE_, ACTR_>(partials, w_base, xq_base, group_count, expert_members, work_items,     \
                       dbpe, xq_slot_stride, n_used, max_per_expert, chunk_size, n_rows,        \
                       n_blocks_in, n_work_items_dev, n_items_host);                           \
}
#define DN3P_KERNEL(NAME_, R_, PF_, MODE_, ACTR_) DN3P_KERNEL_W(NAME_, R_, PF_, MODE_, ACTR_, 1)
DN3P_KERNEL(dn3p_r2, 2, 0, 0, false)
DN3P_KERNEL(dn3p_r2pf, 2, 1, 0, false)
DN3P_KERNEL(dn3p_r4, 4, 0, 0, false)
DN3P_KERNEL(dn3p_r4pf, 4, 1, 0, false)
DN3P_KERNEL(dn3p_r4pfa, 4, 1, 0, true)
DN3P_KERNEL(dn3p_r2epf, 2, 2, 0, false)
DN3P_KERNEL(dn3p_r2epfa, 2, 2, 0, true)
DN3P_KERNEL(dn3q_r2epfa, 2, 2, 2, true)
DN3P_KERNEL(dn3q_r4pfa, 4, 1, 2, true)
// Column mode (dn3c): hardware WG scheduling over (tiles, Y), work items looped per WG
// with the next item's (= next expert's) stripe prefetched.
DN3P_KERNEL(dn3c_r2pf, 2, 1, 1, false)
DN3P_KERNEL(dn3c_r4, 4, 0, 1, false)
DN3P_KERNEL(dn3c_r4pf, 4, 1, 1, false)
