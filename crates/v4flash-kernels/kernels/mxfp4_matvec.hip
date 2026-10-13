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
// dn3 -- the DECODE by-expert MXFP4 down (`V41_MOE_DOWN_DN3`, default OFF):
// mxfp4_matvec_par_by_expert_dn3q_r2w4 (`=1`: persistent, dynamic wave-tile queue,
// LDS-staged coalesced stripes -- see "dn3q" below), the static persistent
// mxfp4_matvec_par_by_expert_dn3l_r2w4a (`=static`, and dn3q's fallback when the
// launcher runs out of queue slots), the 2-D-grid mxfp4_matvec_par_by_expert_dn3_r4
// (`=grid`) and the direct-load persistent dn3p_* kept for A/B. Same arguments (plus
// the host item count; dn3q also its queue slot), partials and work-item /
// device-count contract as dn2 (mxfp4_matvec_par_by_expert_smallb) and kwide2;
// BIT-IDENTICAL partials (tests/mxfp4_down_dn3.rs: every geometry == dn2 == kwide2
// over the whole sentinel-filled buffer, dn2 == a CPU replica of its own f32 order,
// and every geometry == that replica on n_rows 24 / 40, chunk 1 / 2 / 3 / 32, up to 8
// members per expert, pool-end slots, repeated launches per stream).
//
// ROOFLINE (gfx1151, cold weights: expert slots rotate over a 2.95 GB pool;
// tests/mxfp4_down_dn3.rs bench_dn3_decode, production bound grid + device count,
// HIP-event kernel time; median of 3 runs of the per-run p50 over 200 reps, arms in a
// fresh random order per rep, beside the live hub, 2026-10-13; A/A dn2 vs dn2 <= 0.5%).
// A decode down call must move D x 6,266,880 B of MXFP4 weights (5120 rows x 9
// super-blocks x 136 B; D distinct experts of 1-3 members) -- plus members x 2.6 KB
// of activations (L2) and members x 20 KB of partials, < 1%: a pure weight stream.
// The ceiling for exactly those bytes is the slot-streaming read probe
// (device_ceilings.hip ceil_read_slots, 32 KB chunks = best of 19K..512K; 64K is 2-4%
// slower): 241 GB/s incremental (94% of the 256 GB/s LPDDR5X peak) + ~7 us fixed
// (launch + event floor, ramp, drain); a 1 GiB contiguous stream reaches 216-219 GB/s.
//
//   us (p50)     b,D = 1,1   1,2   1,4   1,6   2,4   2,6   2,8  2,12   4,4
//   probe 32K          33.6  59.6 112.2 164.6 112.5 165.0 217.1 321.2 113.0
//   dn2                41.5  67.8 120.1 175.2 128.1 177.8 229.7 336.1 149.7
//   dn3l (static)      34.8  62.0 116.7 171.9 117.2 172.1 228.0 340.0 119.5
//   dn3q (default)     35.5  61.3 113.6 165.9 113.8 166.2 218.9 323.9 115.3
//   dn3q p90           37    63   117   172   117   171   227   333   119
//   dn3q loads-only    33.8  59.4 111.7 163.9 111.7 164.2 216.4 320.1 112.4
//   dn3q % probe        95%   97%   99%   99%   99%   99%   99%   99%   98%
//   dn3l % probe        97%   96%   96%   96%   96%   96%   95%   94%   95%
//   dn2 % probe         81%   88%   93%   94%   88%   93%   95%   96%   75%
//
//   us (p50)     b,D = 4,6   4,8  4,12  4,16   8,4   8,6   8,8  8,12  8,16
//   probe 32K         165.8 217.9 320.8 424.4 112.8 166.6 219.9 322.8 425.1
//   dn2               197.6 247.8 344.9 447.7 170.8 229.9 289.3 384.5 484.3
//   dn3l (static)     174.3 228.2 340.2 451.9 119.4 176.0 233.4 344.0 454.4
//   dn3q (default)    167.6 219.0 323.8 427.9 115.2 168.8 222.1 326.8 430.6
//   dn3q p90          173   225   334   440   119   173   228   336   445
//   dn3q loads-only   164.8 216.6 319.6 422.6 112.2 165.3 218.1 321.4 424.2
//   dn3q % probe        99%   99%   99%   99%   98%   99%   99%   99%   99%
//   dn3l % probe        95%   95%   94%   94%   94%   95%   94%   94%   94%
//   dn2 % probe         84%   88%   93%   95%   66%   72%   76%   84%   88%
//   (dn3_r4 `grid`: 83-98%, 83% at b = 8, D = 4. D = 1 swings 95-97% run to run.)
//
// WHERE dn2 LOSES (sub-staged): (1) the production grid is (320 row tiles, 6b) --
// grid.y is the work-item UPPER BOUND, the count lives on the device -- so a b = 8
// call over D = 4 experts launches 44 empty work-item columns AFTER the real ones,
// a tail of ~56K empty waves (each waits ~0.5 us on its count load; dn2's also stage
// the LUT + barrier): +60-80 us at b = 8; (2) 256 ds_load_u8 per 2-row wave for the
// nibble LUT, half of it for super-block 8 that only lanes 0-3 own; (3) a 3-load,
// half-line-per-instruction lane pattern.
//
// THE DESIGN (from the bytes; dn3l's, dn3q's additions in its own section):
//  * Persistent 1-D grid of four-wave work-groups walking the flattened (work item,
//    row tile) list up to the DEVICE count: no empty waves, launch cost independent
//    of the bound; the work-groups in flight stream one contiguous region (strided
//    beats contiguous per-WG ranges by 2-10%).
//  * Wave = 2 rows; its 2448 B stripe is read FULLY COALESCED (16 B per lane, 512 B
//    per instruction, 5 loads per lane), staged in a wave-private LDS slice (no
//    barriers: a wave-scope release / acquire fence pair -- dn3_wave_lds_sync, no
//    instruction on gfx11 -- orders the cross-lane hand-off) and read back in the
//    compute lane map -- dn3_load's lane pattern costs 2-4% vs coalesced
//    (ceil_read_slots_mx).
//  * The next tile's stripe is issued as soon as the current one is in LDS --
//    BEFORE its unpack and compute -- so a stripe is always in flight per wave;
//    every load is unconditional (clamped addresses) and the order is pinned with
//    sched_barriers so the compiler's vmcnt waits stay exact (loads return in
//    order: an activation load issued after the prefetch would wait for it).
//  * Activations of the first members are kept in registers across tiles of the same
//    work item; a work item's (e, members) are memoised the same way.
//  * Arithmetic per output = dn2's exactly: lane (bl = lane>>2, sub = lane&3)
//    owns super-block bl, MXFP4 blocks {2sub, 2sub+1}; f = sc0*s0 + sc1*s1;
//    pacc = 0 + yd*f; lanes 0-3 add yd8*f8 for super-block 8; dn2's shfl_down tree
//    (16, 8, 4, 2, 1). Only WHERE values are computed moved: super-block 8 off the
//    lanes that add it (each f8 moved to lane sub of its row, exact); the R rows
//    reduce by a TRANSPOSED butterfly (at offset o a lane with bit o set sums row B's
//    pair, else row A's -- the same operand pairs, IEEE + is commutative), so R
//    reductions cost ~R + 4 moves instead of 5R.
//  * Nibble -> int8 by v_perm (3 perms + 3 ALU per 4 values), no LUT.
//
// WHAT IS LEFT (dn3q, 1-2% = 1-5 us at D >= 4; ~2 us = 3-5% at D = 1-2): the
// loads-only twin dn3q_r2w4b (same queue, stream, LDS staging, activation loads; the
// unpack / dot / reduce replaced by an XOR fold) runs at 99-101% of the probe, so
// the stream is at its ceiling and what is left is the compute itself: (a) ~2 us per
// launch -- the first tile's activation wait and the LAST tile's unpack + dots +
// reduction after its stripe lands, with nothing left to stream behind it (the
// probe's last work-group just retires); (b) < 1% of the slope at 3 members per
// expert. Measured and NOT better: two stripes in flight per wave (-1..-2%: bytes in
// flight are not the limit), more or fewer waves (G 100..160 flat, 80 -2%), the
// members' chains interleaved in one block (+-0.1%), the first activations issued
// with the first stripe (-0.5 us), one activation set reused for all members (no
// gain at D = 4: activation loads are not the cost). Hiding (a) needs work after the
// last byte -- the partials reduce that follows -- i.e. a fused down + reduce, outside
// this kernel's contract.
//
// n_blocks_in is COMPILE-TIME 9 (kernels return without writing otherwise; the
// launcher refuses it); n_rows % (rows per WG) == 0 (launcher-checked).
// =============================================================================
#define DN3_NB 9u
#define DN3_ROW_BYTES (DN3_NB * SUPER_MXFP4_BYTES)   // 1224

// Sub-staging builds only (DEEPSTRIX_KERNEL_CFLAGS=-DDN3_ABLATE=1): the persistent
// kernels consume the loaded stripe + activations by an XOR fold instead of
// unpack + dot + reduce (wrong partials -- never a default build).
#ifndef DN3_ABLATE
#define DN3_ABLATE 0
#endif

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

// One member x the wave's R rows: dn2's per-lane chain, super-block 8 from lane
// 4r + sub by shuffle, the transposed butterfly, one store per row.
template <unsigned R>
__device__ __forceinline__ void dn3_member(const int32_t (&w)[R][16], const float (&sc)[R][2],
                                           const int32_t (&w8)[16], const float (&sc8)[2], const Dn3Act& a,
                                           unsigned lane, unsigned bl, unsigned sub,
                                           float* __restrict__ partials, unsigned n_used, unsigned n_rows,
                                           unsigned row0) {
    static_assert(R == 1u || R == 2u || R == 4u || R == 8u, "R must be 1, 2, 4 or 8");
    constexpr unsigned LOG_R = (R == 8u) ? 3u : (R == 4u) ? 2u : (R == 2u) ? 1u : 0u;
    constexpr unsigned GSZ = 32u / R;            // lanes per row after the transposed steps
    float v[R];
    #pragma unroll
    for (unsigned p = 0; p < R; p++) {
        const float f = dn3_f(w[p], sc[p], a.q);
        float pacc = 0.0f;
        pacc += a.yd * f;
        v[p] = pacc;
    }
    // Super-block 8: lane 4r + sub holds row r's f8; lane sub of row r adds it (dn2's it = 1).
    const float f8 = dn3_f(w8, sc8, a.q9);
    #pragma unroll
    for (unsigned p = 0; p < R; p++) {
        const float g = __shfl(f8, (int)(4u * p + sub), 32);
        if (bl == 0u) v[p] += a.yd9 * g;
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
        partials[((size_t)a.b * n_used + a.slot) * n_rows + row0 + rr] = v[0];
    }
}

// The wave's R-row stripe in the compute lane map. Lanes past the R rows (R < 8)
// re-load a valid row's super-block 8 (result unused): no divergent load, so the
// compiler's vmcnt waits stay exact.
template <unsigned R>
__device__ __forceinline__ void dn3_load_stripe(const uint8_t* w_r0, unsigned bl, unsigned sub, bool has8,
                                                Dn3Raw (&raw)[R], Dn3Raw& raw8) {
    #pragma unroll
    for (unsigned p = 0; p < R; p++) dn3_load(w_r0 + p * DN3_ROW_BYTES + bl * SUPER_MXFP4_BYTES, sub, raw[p]);
    dn3_load(w_r0 + (has8 ? bl : (bl & (R - 1u))) * DN3_ROW_BYTES + 8u * SUPER_MXFP4_BYTES, sub, raw8);
}

// The XOR fold standing in for unpack + compute in DN3_ABLATE builds.
template <unsigned R>
__device__ __forceinline__ void dn3_ablate_sink(const Dn3Raw (&raw)[R], const Dn3Raw& raw8, const Dn3Act& a0,
                                                const Dn3Act& a1, float* __restrict__ partials, unsigned row0) {
    uint32_t x = raw8.a.x ^ raw8.a.y ^ raw8.a.z ^ raw8.a.w ^ raw8.b.x ^ raw8.b.y ^ raw8.b.z ^ raw8.b.w ^ raw8.sc;
    #pragma unroll
    for (unsigned p = 0; p < R; p++)
        x ^= raw[p].a.x ^ raw[p].a.y ^ raw[p].a.z ^ raw[p].a.w ^ raw[p].b.x ^ raw[p].b.y ^ raw[p].b.z ^ raw[p].b.w ^ raw[p].sc;
    #pragma unroll
    for (unsigned k = 0; k < 4u; k++) x ^= a0.q[k].x ^ a0.q9[k].w ^ a1.q[k].y ^ a1.q9[k].z;
    if (x == 0x9e3779b9u) partials[row0] = (float)x;
}

// -----------------------------------------------------------------------------
// dn3_r4: the 2-D grid (n_rows / (4W), n_work_items) of dn2 with dn3's wave (4 rows,
// v_perm unpack, one super-block-8 pass, transposed reduction, next member's
// activations prefetched). `V41_MOE_DOWN_DN3=grid`: no persistence (hardware WG
// scheduling, least exposed to other queues) but dn2's empty-column tail at
// bound > D. Includes dn2's work-item loop (w += gridDim.y).
// -----------------------------------------------------------------------------
template <unsigned R>
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
    Dn3Raw raw8;
    dn3_load_stripe<R>(w_r0, bl, sub, has8, raw, raw8);
    int32_t w[R][16];
    float   sc[R][2];
    int32_t w8[16];
    float   sc8[2];
    #pragma unroll
    for (unsigned p = 0; p < R; p++) dn3_unpack(raw[p], w[p], sc[p]);
    dn3_unpack(raw8, w8, sc8);
    for (unsigned mi = member_start; mi < member_end; mi++) {
        // Unconditional (the last member re-loads itself, result unused): exact vmcnt waits.
        const bool more = mi + 1u < member_end;
        Dn3Act nxt;
        dn3_load_act(xq_base, expert_members, e, max_per_expert, more ? mi + 1u : mi, n_used, xq_slot_stride, bl, sub, nxt);
        dn3_member<R>(w, sc, w8, sc8, cur, lane, bl, sub, partials, n_used, n_rows, row0);
        if (more) cur = nxt;
    }
    }  // work-item loop
}

// Shared by the persistent bodies: the flattened (work item, tile) walk.
struct Dn3Tile { unsigned e, ms, me, row0; };

// -----------------------------------------------------------------------------
// dn3p: persistent, DIRECT loads in the compute lane map (no LDS). PF 1: the next
// stripe is issued after the current one is unpacked (same registers); PF 2: before
// it, into a second register set (the walk ping-pongs the two sets, so no copy waits
// on the prefetched data). ACTR: activations kept across tiles of one work item.
// -----------------------------------------------------------------------------
template <unsigned R, unsigned PF, bool ACTR>
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
    static_assert(PF == 1u || PF == 2u, "PF must be 1 or 2");
    if (n_blocks_in != DN3_NB) return;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned rows_wg = (blockDim.x >> 5) * R;
    const unsigned tiles = n_rows / rows_wg;                // launcher: n_rows % rows_wg == 0
    const unsigned wave_row = (threadIdx.x >> 5) * R;
    const unsigned bl  = lane >> 2u;
    const unsigned sub = lane & 3u;
    const bool has8 = bl < R;
    // dn2's item count: the device count when given (the grid is then sized by the
    // host bound only), else the launch's.
    const unsigned n_items = (n_work_items_dev != nullptr) ? (unsigned)max(0, *n_work_items_dev) : n_items_host;
    const unsigned total = n_items * tiles;
    // Consecutive tiles of a work-group are nearly always in the same work item (G <
    // tiles per item): reuse its (e, ms, me) instead of two dependent scalar loads
    // that the next tile's prefetch would wait on.
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
    // First tile: its work item is read BEFORE the device-count check (the grid never
    // exceeds the host bound, so item < n_items_host <= work_items.len()) so the two
    // loads overlap; its stripe is issued as soon as the expert is known -- a builder
    // item always has >= 1 member and the address is valid whatever group_count says.
    const unsigned item0 = t / tiles;
    const int packed0 = work_items[item0];
    if (t >= total) return;
    Dn3Tile cur;
    cur.e  = (unsigned)(packed0 >> 16) & 0xffffu;
    cur.ms = (unsigned)(packed0) & 0xffffu;
    cur.row0 = (t - item0 * tiles) * rows_wg + wave_row;
    Dn3Raw raw[R];
    Dn3Raw raw8;
    dn3_load_stripe<R>(stripe(cur), bl, sub, has8, raw, raw8);
    cur.me = min((unsigned)group_count[cur.e], cur.ms + chunk_size);
    memo = cur;
    memo_item = item0;
    Dn3Act a0, a1;
    unsigned act_e = 0xffffffffu, act_ms = 0xffffffffu;   // the item a0/a1 hold (ACTR)
    // One tile: `rc` holds its stripe (issued earlier), the next tile's goes to `rn`
    // (PF 1: rn aliases rc). Returns false after the last tile.
    auto step = [&](Dn3Raw (&rc)[R], Dn3Raw& rc8, Dn3Raw (&rn)[R], Dn3Raw& rn8) -> bool {
        const unsigned tn = t + gridDim.x;
        const bool has_next = tn < total;
        Dn3Tile nxt = cur;
        if (has_next) nxt = resolve(tn);
        const bool pf_ok = has_next && nxt.me > nxt.ms;
        if (cur.me > cur.ms) {
            const bool two = cur.ms + 1u < cur.me;
            // Activations of the first two members, issued before the prefetch (loads
            // return in order). Unconditional (the second re-loads the first when there
            // is one member): exact vmcnt waits.
            if (!ACTR || cur.e != act_e || cur.ms != act_ms) {
                dn3_load_act(xq_base, expert_members, cur.e, max_per_expert, cur.ms, n_used, xq_slot_stride, bl, sub, a0);
                dn3_load_act(xq_base, expert_members, cur.e, max_per_expert, two ? cur.ms + 1u : cur.ms, n_used, xq_slot_stride, bl, sub, a1);
                act_e = cur.e;
                act_ms = cur.ms;
            }
            // The machine scheduler would hoist the prefetch above the activation loads
            // (the compute would then wait for it): pin the order.
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (PF == 2u) {
                // Branch-free (no next / empty next: re-load this stripe, L2 hits).
                dn3_load_stripe<R>(stripe(pf_ok ? nxt : cur), bl, sub, has8, rn, rn8);
                __builtin_amdgcn_sched_barrier(0);
            }
#if DN3_ABLATE
            if constexpr (PF == 1u) dn3_load_stripe<R>(stripe(pf_ok ? nxt : cur), bl, sub, has8, rn, rn8);
            dn3_ablate_sink<R>(rc, rc8, a0, a1, partials, cur.row0);
#else
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
#endif
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
// dn3l: persistent with the wave's stripe loaded FULLY COALESCED (16 B per lane,
// 512 B per instruction) and staged through a wave-private LDS slice, then read
// back in the compute lane map -- the same bytes in the same lanes as dn3_load, so
// the arithmetic (and every bit) is dn3p's. The next stripe goes into the same
// registers right after the current one is in LDS (before its unpack + compute),
// so one register set is enough for an always-in-flight prefetch. R even.
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
        // Lanes past the stripe re-load its last chunk (written to the same LDS bytes
        // with the same value): no divergent load, exact vmcnt waits.
        const unsigned ch = min(32u * k + lane, Dn3Co<R>::NCH - 1u);
        v[k] = *(const dn3_v4u*)__builtin_assume_aligned(w_r0 + 16u * ch, 16);
    }
}

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

// Orders a wave's LDS accesses across its lanes (HIP's __syncwarp idiom): the release
// / acquire fences at WAVEFRONT scope make the cross-lane LDS hand-off well defined at
// the IR level (no instruction on gfx11: a wave's LDS operations execute in order),
// the wave barrier keeps the machine scheduler from moving LDS ops across it.
__device__ __forceinline__ void dn3_wave_lds_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
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
            dn3_wave_lds_sync();                      // the previous tile's LDS reads before this put
            dn3_lds_put<R>(co, slice, lane);
            dn3_wave_lds_sync();                      // this put before this tile's LDS reads
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
            if constexpr (DN3_ABLATE != 0) {
            dn3_ablate_sink<R>(raw, raw8, a0, a1, partials, cur.row0);
            } else {
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
            }
        } else if (pf_ok) {
            dn3_load_co<R>(stripe(nxt), lane, co);
        }
        if (!has_next) break;
        t = tn;
        cur = nxt;
    }
}

// __shfl_xor(x, O, 32) without LDS: DPP row_xmask (O < 16: lane i reads i ^ O within
// its row of 16) or v_permlanex16 (O = 16: lane i reads lane i of the other row). Exact
// moves -- the same operand reaches the same lane as ds_bpermute would deliver it.
template <unsigned O>
__device__ __forceinline__ float dn3_xor(float x) {
    static_assert(O >= 1u && O <= 16u && (O & (O - 1u)) == 0u, "xor offset: a power of two <= 16");
    const int xi = __float_as_int(x);
    if constexpr (O == 16u) {
        return __int_as_float(__builtin_amdgcn_permlanex16(xi, xi, 0x76543210u, 0xfedcba98u, false, false));
    } else {
        return __int_as_float(__builtin_amdgcn_mov_dpp(xi, 0x160 + (int)O, 0xF, 0xF, false));
    }
}
// One row's (super-block, sub) bytes from the LDS slice -> Dn3Raw (= dn3_lds_get's rd).
__device__ __forceinline__ void dn3_lds_rd(const uint8_t* w_super, unsigned sub, Dn3Raw& r) {
    __builtin_memcpy(&r.a, __builtin_assume_aligned(w_super + 32u * sub, 8), 16);
    __builtin_memcpy(&r.b, __builtin_assume_aligned(w_super + 32u * sub + 16u, 8), 16);
    const uint32_t sc4 = *(const uint32_t*)__builtin_assume_aligned(w_super + 128u + 4u * (sub >> 1), 4);
    r.sc = (sc4 >> (16u * (sub & 1u))) & 0xffffu;
}

// -----------------------------------------------------------------------------
// dn3q -- THE DEFAULT (`V41_MOE_DOWN_DN3=1`): dn3l's wave work (2-row stripe read
// fully coalesced, staged in a wave-private LDS slice, next stripe in flight during
// the compute) over a DYNAMIC wave-tile queue, with:
//  * QUEUE: wave tile u = 2 rows of one work item (item u / (n_rows / 2)). A wave's
//    first TWO tiles are static (u0 = its global wave index, then u0 + n_waves; the
//    first one's work item is read before the device count, so its stripe is one
//    dependent load in); every later one is a ticket from `ctr[0]` (tile = 2 n_waves +
//    ticket), one atomicAdd by lane 0 issued after the prefetch (its return never waits
//    on anything the tile needs) and read one iteration later -- the launch-start burst
//    (every wave grabs at once, one address) has a whole tile to drain: reading the
//    first ticket at once instead cost ~2 us per launch (D = 1: 92% -> 97% of the probe).
//    The static persistent walk (dn3l) loses 3-6% at D >= 8 to imbalance between its
//    fixed tile sets (loads-only twins: static 95-97% of the probe at G = 140, 97-98%
//    at G = 100; queued 99-101%).
//  * RE-ARM: a wave grabs once in every iteration that has a next tile, so the launch
//    hands out exactly max(0, total - 2 n_waves) work tickets + one terminal ticket per
//    wave that has a second tile (min(n_waves, total - n_waves)); the holder of the last
//    one stores ctr[0] = 0 (all other grabs of the launch have happened -- same
//    address). No done counter, no exit atomics; the only early returns (argument
//    guards, u0 >= total) are before any grab. `ctr` is the launcher's slot (mxfp4.rs
//    Dn3Queue: per stream, or per graph capture).
//  * Activations of the first 3 members of an item stay in registers across the wave's
//    tiles of that item (decode: <= 3 members per expert is the common case at b <= 8;
//    more load per tile, behind the prefetch); the members' packed (row, slot) are
//    read with the item (resolve), so a new item's activation loads issue at once.
//  * SUPER-BLOCK 8 QUARTERED: dn2 computes super-block 8 of a row on lanes 0-3 (sub =
//    blocks {2sub, 2sub+1}); dn3l did both rows in one pass on lanes 4r + sub, the other
//    24 lanes duplicating. Here every lane does a QUARTER of one (row, sub) pair: lane L
//    -> c = L & 7 (row p = c >> 2, sub = c & 3), k = L >> 3 (block 2sub + (k >> 1), raw
//    dwords {2h, 2h+1}, h = k & 1): 4 dot4 instead of 16, 2 raw dwords instead of 8, 4
//    activation dwords instead of 16 per member (-57 VGPRs: 8 waves / SIMD with 3
//    members held vs 6). The block sums are int32 (exact, any order): s_t = own +
//    lane ^ 8 (DPP), then lanes 0-15 (t = 0) take s_1 from lane ^ 16 (permlanex16) and
//    form f8 = (0 + sc0 * s0) + sc1 * s1 with dn3_f's expression -- the same f8 in lane
//    4p + sub as before (lanes 16-31 compute a value nobody reads).
//  * Every cross-lane move on DPP row_xmask / permlanex16 (dn3_xor) instead of a chain
//    of 7 ds_bpermute LDS round trips per member.
// Per output the same f, `pacc = 0; pacc += yd * f`, super-block-8 add and reduction
// tree as dn2: BIT-IDENTICAL (tests/mxfp4_down_dn3.rs).
// -----------------------------------------------------------------------------
struct Dn3ActQ { uint4 q[4]; uint4 q9; float yd, yd9; unsigned b, slot; };

__device__ __forceinline__ void dn3q_load_act(const uint8_t* __restrict__ xq_base, int packed, unsigned n_used,
                                              unsigned xq_slot_stride, unsigned bl, unsigned sub, unsigned lane,
                                              Dn3ActQ& a) {
    a.b    = (unsigned)(packed >> 16) & 0xffffu;
    a.slot = (unsigned)(packed) & 0xffffu;
    const uint8_t* y_m = xq_base + ((size_t)a.b * n_used + a.slot) * xq_slot_stride;
    const uint8_t* y0  = y_m + (size_t)bl * BLOCK_Q8_K_BYTES;
    const uint8_t* y8  = y_m + (size_t)8u * BLOCK_Q8_K_BYTES;
    #pragma unroll
    for (unsigned k = 0; k < 4u; k++) a.q[k] = dn3_ld16_a4(y0 + 4u + 64u * sub + 16u * k);
    // Quarter (c = lane & 7, k = lane >> 3): block 2 sub8 + t, Q8 bytes 8h..8h+7 (lo
    // nibbles' elements) and 16 + 8h .. (hi nibbles').
    const unsigned c = lane & 7u, kq = lane >> 3, t = kq >> 1, h = kq & 1u;
    const uint8_t* yq = y8 + 4u + 32u * (2u * (c & 3u) + t) + 8u * h;
    uint2 lo, hi;
    __builtin_memcpy(&lo, __builtin_assume_aligned(yq, 4), 8);
    __builtin_memcpy(&hi, __builtin_assume_aligned(yq + 16u, 4), 8);
    a.q9 = make_uint4(lo.x, lo.y, hi.x, hi.y);
    __builtin_memcpy(&a.yd, __builtin_assume_aligned(y0, 4), 4);
    __builtin_memcpy(&a.yd9, __builtin_assume_aligned(y8, 4), 4);
}

// The lane's super-block-8 quarter from the LDS slice: 2 raw nibble dwords of block
// 2 sub8 + t (dwords 2h, 2h+1) and the 2 scale bytes of blocks {2 sub8, 2 sub8 + 1}.
struct Dn3Raw8Q { uint32_t r0, r1, sc; };
__device__ __forceinline__ void dn3q_lds_rd8(const uint8_t* slice, unsigned lane, Dn3Raw8Q& r) {
    const unsigned c = lane & 7u, kq = lane >> 3, t = kq >> 1, h = kq & 1u;
    const unsigned p = c >> 2, sub8 = c & 3u;
    const uint8_t* w_super = slice + p * DN3_ROW_BYTES + 8u * SUPER_MXFP4_BYTES;
    uint2 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(w_super + 32u * sub8 + 16u * t + 8u * h, 8), 8);
    r.r0 = v.x;
    r.r1 = v.y;
    const uint32_t sc4 = *(const uint32_t*)__builtin_assume_aligned(w_super + 128u + 4u * (sub8 >> 1), 4);
    r.sc = (sc4 >> (16u * (sub8 & 1u))) & 0xffffu;
}

// w8[0..1] = lo nibbles of raw dwords (2h, 2h+1), w8[2..3] = their hi nibbles.
__device__ __forceinline__ void dn3q_unpack8(const Dn3Raw8Q& r, int32_t w8[4], float sc8[2]) {
    sc8[0] = e8m0_half_to_f32(r.sc & 0xFFu);
    sc8[1] = e8m0_half_to_f32(r.sc >> 8);
    w8[0] = (int32_t)dn3_e2m1x4(r.r0);
    w8[1] = (int32_t)dn3_e2m1x4(r.r1);
    w8[2] = (int32_t)dn3_e2m1x4(r.r0 >> 4);
    w8[3] = (int32_t)dn3_e2m1x4(r.r1 >> 4);
}

// dn3_member<2> with DPP moves and the quartered super-block 8 (same f8 bits in lanes 0-7).
__device__ __forceinline__ void dn3q_member(const int32_t (&w)[2][16], const float (&sc)[2][2],
                                            const int32_t (&w8)[4], const float (&sc8)[2], const Dn3ActQ& a,
                                            unsigned lane, unsigned bl,
                                            float* __restrict__ partials, unsigned n_used, unsigned n_rows,
                                            unsigned row0) {
    float v[2];
    #pragma unroll
    for (unsigned p = 0; p < 2u; p++) {
        const float f = dn3_f(w[p], sc[p], a.q);
        float pacc = 0.0f;
        pacc += a.yd * f;
        v[p] = pacc;
    }
    // Integer block sums of super-block 8 (exact in any order).
    int32_t s = 0;
    s = sudot4_par(w8[0], (int32_t)a.q9.x, s);
    s = sudot4_par(w8[1], (int32_t)a.q9.y, s);
    s = sudot4_par(w8[2], (int32_t)a.q9.z, s);
    s = sudot4_par(w8[3], (int32_t)a.q9.w, s);
    s = s + __builtin_amdgcn_mov_dpp(s, 0x160 + 8, 0xF, 0xF, false);                        // + lane ^ 8
    const int32_t s1 = __builtin_amdgcn_permlanex16(s, s, 0x76543210u, 0xfedcba98u, false, false);  // lane ^ 16
    // dn3_f's float expression on (s0, s1) (lanes 0-15: s = s0, s1 = s1).
    float f8 = 0.0f;
    f8 += sc8[0] * (float)s;
    f8 += sc8[1] * (float)s1;
    {
        const float g0 = f8;
        if (bl == 0u) v[0] += a.yd9 * g0;
        const float g1 = dn3_xor<4>(f8);
        if (bl == 0u) v[1] += a.yd9 * g1;
    }
    {
        const bool up = (lane & 16u) != 0u;
        const float A = v[0], B = v[1];
        const float send = up ? A : B;
        const float keep = up ? B : A;
        v[0] = keep + dn3_xor<16>(send);
    }
    v[0] = v[0] + dn3_xor<8>(v[0]);
    v[0] = v[0] + dn3_xor<4>(v[0]);
    v[0] = v[0] + dn3_xor<2>(v[0]);
    v[0] = v[0] + dn3_xor<1>(v[0]);
    if ((lane & 15u) == 0u) {
        const unsigned rr = lane >> 4;                  // bitreverse of a 1-bit group index
        partials[((size_t)a.b * n_used + a.slot) * n_rows + row0 + rr] = v[0];
    }
}

// A wave tile with its work item's (e, members) and the packed (row, slot) of the
// item's first NACT members (memoised per item: resolved one tile ahead, so a new
// item's activation loads never wait on a scalar round trip).
struct Dn3TileQ { unsigned e, ms, me, row0; int m0, m1, m2; };

template <unsigned W, unsigned NACT, unsigned ABL>
__device__ __forceinline__ void dn3q_body(
        float          * __restrict__ partials,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        const int      * __restrict__ work_items,
        unsigned dbpe, unsigned xq_slot_stride, unsigned n_used, unsigned max_per_expert,
        unsigned chunk_size, unsigned n_rows, unsigned n_blocks_in,
        const int      * __restrict__ n_work_items_dev, unsigned n_items_host,
        unsigned       * __restrict__ ctr) {
    constexpr unsigned R = 2u;
    static_assert(NACT == 2u || NACT == 3u, "NACT must be 2 or 3");
    __shared__ __attribute__((aligned(16))) uint8_t lds[W * Dn3Co<R>::NCH * 16u];
    if (n_blocks_in != DN3_NB || blockDim.x != 32u * W) return;   // grid-uniform, before any atomic
    const unsigned lane = threadIdx.x & 31u;
    const unsigned wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    uint8_t* slice = lds + wave * (Dn3Co<R>::NCH * 16u);
    const unsigned tiles = n_rows / R;                      // wave tiles per work item
    const unsigned bl  = lane >> 2u;
    const unsigned sub = lane & 3u;
    const unsigned n_waves = gridDim.x * W;
    // The first tile is static (u0 = the global wave index) and its work item is read
    // BEFORE the device count (the launcher's grid keeps u0 < tiles x the host bound,
    // so item0 < work_items.len()): the stripe is issued one dependent load in.
    const unsigned u0 = blockIdx.x * W + wave;
    const unsigned item0 = u0 / tiles;
    const int packed0 = work_items[item0];
    const unsigned n_items = (n_work_items_dev != nullptr) ? (unsigned)max(0, *n_work_items_dev) : n_items_host;
    const unsigned total = n_items * tiles;
    if (u0 >= total) return;                                // never grabs: the ticket count below excludes it
    // Tickets. A wave's first TWO tiles are static (u0, u0 + n_waves); the queue (ctr[0])
    // serves tile 2 n_waves + v for v = 0, 1, ... A wave grabs once in every iteration
    // that has a next tile and reads the ticket one iteration later, so the launch-start
    // burst (every wave grabs at once, same address) has a whole tile to drain.
    // Grabbing waves: those with a second tile, min(n_waves, total - n_waves); work
    // tickets: max(0, total - 2 n_waves); each grabbing wave's last grab is one terminal
    // ticket. The holder of the LAST ticket re-arms the queue (ctr[0] = 0): every other
    // grab of this launch has already happened.
    const unsigned grabbers = total > n_waves ? min(n_waves, total - n_waves) : 0u;
    const unsigned last_ticket = (total > 2u * n_waves ? total - 2u * n_waves : 0u) + grabbers - 1u;
    const size_t emax = max_per_expert;
    auto members = [&](Dn3TileQ& x) {
        // Clamped to the expert's row (valid memory); used only below me.
        const int* em = expert_members + (size_t)x.e * emax;
        x.m0 = em[min(x.ms, max_per_expert - 1u)];
        x.m1 = em[min(x.ms + 1u, max_per_expert - 1u)];
        if constexpr (NACT == 3u) x.m2 = em[min(x.ms + 2u, max_per_expert - 1u)];
    };
    unsigned memo_item = item0;
    Dn3TileQ memo;
    memo.e  = (unsigned)(packed0 >> 16) & 0xffffu;
    memo.ms = (unsigned)(packed0) & 0xffffu;
    memo.row0 = (u0 - item0 * tiles) * R;
    auto stripe = [&](const Dn3TileQ& x) -> const uint8_t* {
        return w_base + (uint64_t)x.e * dbpe + (uint64_t)x.row0 * DN3_ROW_BYTES;
    };
    auto grab = [&]() -> unsigned {
        unsigned v = 0;
        if (lane == 0u) v = atomicAdd(&ctr[0], 1u);
        return v;
    };
    // First stripe (needs e only: a builder item always has >= 1 member).
    dn3_v4u co[Dn3Co<R>::NV];
    dn3_load_co<R>(stripe(memo), lane, co);
    unsigned vn = 0;
    memo.me = min((unsigned)group_count[memo.e], memo.ms + chunk_size);
    members(memo);
    auto resolve = [&](unsigned u) -> Dn3TileQ {
        const unsigned item = u / tiles;
        if (item != memo_item) {
            const int packed_item = work_items[item];
            memo.e  = (unsigned)(packed_item >> 16) & 0xffffu;
            memo.ms = (unsigned)(packed_item) & 0xffffu;
            memo.me = min((unsigned)group_count[memo.e], memo.ms + chunk_size);
            members(memo);
            memo_item = item;
        }
        Dn3TileQ x = memo;
        x.row0 = (u - item * tiles) * R;
        return x;
    };
    Dn3TileQ cur = memo;
    Dn3ActQ a0, a1, a2;
    unsigned act_e = 0xffffffffu, act_ms = 0xffffffffu;
    unsigned cu = u0;
    bool first = true;
    while (true) {
        unsigned un;
        if (first) {
            un = cu + n_waves;                        // the static second tile
        } else {
            const unsigned v = __builtin_amdgcn_readfirstlane(vn);
            if (v == last_ticket) __hip_atomic_store(&ctr[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            un = 2u * n_waves + v;
        }
        const bool has_next = un < total;
        Dn3TileQ nxt = cur;
        if (has_next) nxt = resolve(un);
        const bool pf_ok = has_next && nxt.me > nxt.ms;
        if (cur.me > cur.ms) {
            const bool two = cur.ms + 1u < cur.me;
            const bool three = cur.ms + 2u < cur.me;
            // Stage this stripe (waits only for it), then the activations of a new item,
            // then the next stripe and the next ticket -- all before the compute, which
            // then waits only for the activations (loads return in order; the
            // sched_barriers pin that order).
            dn3_wave_lds_sync();                  // the previous tile's LDS reads before this put
            dn3_lds_put<R>(co, slice, lane);
            dn3_wave_lds_sync();                  // this put before this tile's LDS reads
            __builtin_amdgcn_sched_barrier(0);
            if (cur.e != act_e || cur.ms != act_ms) {
                // Unconditional (a missing member re-loads the first): exact vmcnt waits.
                dn3q_load_act(xq_base, cur.m0, n_used, xq_slot_stride, bl, sub, lane, a0);
                dn3q_load_act(xq_base, two ? cur.m1 : cur.m0, n_used, xq_slot_stride, bl, sub, lane, a1);
                if constexpr (NACT == 3u)
                    dn3q_load_act(xq_base, three ? cur.m2 : cur.m0, n_used, xq_slot_stride, bl, sub, lane, a2);
                act_e = cur.e;
                act_ms = cur.ms;
            }
            __builtin_amdgcn_sched_barrier(0);
            dn3_load_co<R>(stripe(pf_ok ? nxt : cur), lane, co);
            __builtin_amdgcn_sched_barrier(0);
            if (has_next) vn = grab();
            __builtin_amdgcn_sched_barrier(0);
            Dn3Raw raw[R];
            #pragma unroll
            for (unsigned p = 0; p < R; p++) dn3_lds_rd(slice + p * DN3_ROW_BYTES + bl * SUPER_MXFP4_BYTES, sub, raw[p]);
            Dn3Raw8Q raw8;
            dn3q_lds_rd8(slice, lane, raw8);
            if constexpr (ABL == 1u || DN3_ABLATE) {
                uint32_t x = raw8.r0 ^ raw8.r1 ^ raw8.sc ^ a0.q9.x ^ a1.q9.y;
                #pragma unroll
                for (unsigned p = 0; p < R; p++)
                    x ^= raw[p].a.x ^ raw[p].a.y ^ raw[p].a.z ^ raw[p].a.w ^ raw[p].b.x ^ raw[p].b.y ^ raw[p].b.z ^ raw[p].b.w ^ raw[p].sc;
                #pragma unroll
                for (unsigned k = 0; k < 4u; k++) x ^= a0.q[k].x ^ a1.q[k].y;
                if (x == 0x9e3779b9u) partials[cur.row0] = (float)x;
            } else {
                int32_t w[R][16];
                float   sc[R][2];
                int32_t w8[4];
                float   sc8[2];
                #pragma unroll
                for (unsigned p = 0; p < R; p++) dn3_unpack(raw[p], w[p], sc[p]);
                dn3q_unpack8(raw8, w8, sc8);
                dn3q_member(w, sc, w8, sc8, a0, lane, bl, partials, n_used, n_rows, cur.row0);
                if (two) dn3q_member(w, sc, w8, sc8, a1, lane, bl, partials, n_used, n_rows, cur.row0);
                if constexpr (NACT == 3u) {
                    if (three) dn3q_member(w, sc, w8, sc8, a2, lane, bl, partials, n_used, n_rows, cur.row0);
                }
                for (unsigned mi = cur.ms + NACT; mi < cur.me; mi++) {
                    Dn3ActQ am;
                    dn3q_load_act(xq_base, expert_members[(size_t)cur.e * emax + mi], n_used, xq_slot_stride, bl, sub, lane, am);
                    dn3q_member(w, sc, w8, sc8, am, lane, bl, partials, n_used, n_rows, cur.row0);
                }
            }
        } else {
            if (pf_ok) dn3_load_co<R>(stripe(nxt), lane, co);
            if (has_next) vn = grab();
        }
        if (!has_next) break;
        cur = nxt;
        cu = un;
        first = false;
    }
}

#define DN3_ARGS                                                                               \
        float* __restrict__ partials, const uint8_t* __restrict__ w_base,                      \
        const uint8_t* __restrict__ xq_base, const int* __restrict__ group_count,              \
        const int* __restrict__ expert_members, const int* __restrict__ work_items,            \
        unsigned dbpe, unsigned xq_slot_stride, unsigned n_used, unsigned max_per_expert,      \
        unsigned chunk_size, unsigned n_rows, unsigned n_blocks_in,                            \
        const int* __restrict__ n_work_items_dev
#define DN3_PASS partials, w_base, xq_base, group_count, expert_members, work_items, dbpe,      \
        xq_slot_stride, n_used, max_per_expert, chunk_size, n_rows, n_blocks_in, n_work_items_dev

extern "C" __global__ __launch_bounds__(256) void mxfp4_matvec_par_by_expert_dn3_r4(DN3_ARGS) {
    dn3_body<4>(DN3_PASS);
}
extern "C" __global__ __launch_bounds__(256) void mxfp4_matvec_par_by_expert_dn3p_r4pfa(DN3_ARGS, unsigned n_items_host) {
    dn3p_body<4, 1, true>(DN3_PASS, n_items_host);
}
extern "C" __global__ __launch_bounds__(256) void mxfp4_matvec_par_by_expert_dn3p_r2epfa(DN3_ARGS, unsigned n_items_host) {
    dn3p_body<2, 2, true>(DN3_PASS, n_items_host);
}
extern "C" __global__ __launch_bounds__(128) void mxfp4_matvec_par_by_expert_dn3l_r2w4a(DN3_ARGS, unsigned n_items_host) {
    dn3l_body<2, 4, true>(DN3_PASS, n_items_host);
}
extern "C" __global__ __launch_bounds__(128) void mxfp4_matvec_par_by_expert_dn3l_r2w4(DN3_ARGS, unsigned n_items_host) {
    dn3l_body<2, 4, false>(DN3_PASS, n_items_host);
}
extern "C" __global__ __launch_bounds__(128) void mxfp4_matvec_par_by_expert_dn3l_r4w4a(DN3_ARGS, unsigned n_items_host) {
    dn3l_body<4, 4, true>(DN3_PASS, n_items_host);
}
extern "C" __global__ __launch_bounds__(128) void mxfp4_matvec_par_by_expert_dn3q_r2w4(DN3_ARGS, unsigned n_items_host, unsigned* ctr) {
    dn3q_body<4, 3, 0>(DN3_PASS, n_items_host, ctr);
}
// Loads-only twin (bench reference: same stream, queue, LDS staging and activation
// loads; the unpack / dot / reduce replaced by an XOR fold). WRONG partials.
extern "C" __global__ __launch_bounds__(128) void mxfp4_matvec_par_by_expert_dn3q_r2w4b(DN3_ARGS, unsigned n_items_host, unsigned* ctr) {
    dn3q_body<4, 3, 1>(DN3_PASS, n_items_host, ctr);
}
