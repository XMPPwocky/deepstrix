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
