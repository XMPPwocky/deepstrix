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
        const uint8_t* blk = w_super + b32 * BLOCK_MXFP4_BYTES;
        const float scale = e8m0_half_to_f32(blk[0]);
        const uint8_t* qs = blk + 1;
        const int8_t* q8 = q8_base + b32 * 32u;

        int32_t sumi = 0;
        #pragma unroll
        for (unsigned int i = 0; i < 4u; i++) {         // 4 bytes → 1 lo + 1 hi dword
            // Assemble lo-nibble dword (elems 4i..4i+3) and hi-nibble dword
            // (elems 16+4i..16+4i+3).
            union { int32_t d; int8_t b[4]; } lo, hi;
            #pragma unroll
            for (unsigned int j = 0; j < 4u; j++) {
                const uint8_t q = qs[4u * i + j];
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
        const int e = (mode == 0u) ? (-dense - 1) : dense;
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
        unsigned int     n_blocks_in) {
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
                    const uint8_t* blk = w_super + (2u * sub + t) * BLOCK_MXFP4_BYTES;
                    sc[t] = e8m0_half_to_f32(blk[0]);
                    const uint8_t* qs = blk + 1;
                    // 4 lo dwords then 4 hi dwords (matches q8 layout order
                    // below: q[8t..8t+3] lo elems, q[8t+4..8t+7] hi elems).
                    #pragma unroll
                    for (unsigned int i = 0; i < 4u; i++) {
                        union { int32_t d; int8_t b[4]; } lo;
                        #pragma unroll
                        for (unsigned int j = 0; j < 4u; j++) {
                            lo.b[j] = s_lut[qs[4u * i + j] & 0x0F];
                        }
                        w[wi + i] = lo.d;
                    }
                    #pragma unroll
                    for (unsigned int i = 0; i < 4u; i++) {
                        union { int32_t d; int8_t b[4]; } hi;
                        #pragma unroll
                        for (unsigned int j = 0; j < 4u; j++) {
                            hi.b[j] = s_lut[qs[4u * i + j] >> 4];
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
