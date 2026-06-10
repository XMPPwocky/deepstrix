// q2_k_accumulate_matvec_par — parallel reimplementation of
// q2_k_accumulate_matvec (M14g).
//
// Original kernel uses 1 warp per row with each lane covering blocks
// `lane, lane + 32, ...`. For the routed-MoE down projection
// n_blocks_in = N_FF_EXP/256 = 8 — so only lanes 0..7 do work; 75% of
// the warp is idle.
//
// This variant maps **4 lanes per super-block** when n_blocks_in ≤ 8:
//   lane:    0  1  2  3   4  5  6  7  ...  28 29 30 31
//   block:   0  0  0  0   1  1  1  1  ...   7  7  7  7
//   sub:     0  1  2  3   0  1  2  3  ...   0  1  2  3
//
// Inner work split: each block's 16 dot_q2_16 calls (2 outer × 4 inner
// × 2 halves) are split across 4 lanes:
//   sub 0: k=0, j={0,1}  → 4 dot_q2_16
//   sub 1: k=0, j={2,3}  → 4 dot_q2_16
//   sub 2: k=1, j={0,1}  → 4 dot_q2_16
//   sub 3: k=1, j={2,3}  → 4 dot_q2_16
//
// The summs term (Σ_j bsums[j] * (sc[j]>>4), 16 mads) is also split
// 4 ways: each sub handles 4 of the 16 entries. The per-lane partial
// emits `dall * partial_isum - dmin * partial_summs`; summing all 4
// reproduces the original block_result formula exactly.
//
// For n_blocks_in > 8 the stride wraps with step 8, so each (block_lane)
// iterates ceil(n_blocks_in/8) times — generality preserved.
//
// Numerics: integer per-(k,j) contributions are identical to serial;
// the f32 reduction order differs (warp-tree across 32 partials instead
// of 8), so results match within f32-ULP.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q2_K_BYTES 84
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float f16_to_f32_par(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4_par(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// All call sites here read from 4-byte-aligned addresses: Q2_K blocks
// are 84 bytes (4-aligned) with q2 quants at block offset 16; Q8_K
// blocks are 292 bytes (4-aligned) with q8 quants at block offset 4.
// memcpy folds to a single aligned dword load — no scalar byte fallback.
__device__ __forceinline__ int32_t load_dword_par(const uint8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

__device__ __forceinline__ int32_t load_dword_i8_par(const int8_t* p) {
    return load_dword_par((const uint8_t*)p);
}

__device__ __forceinline__ int32_t dot_q2_16_par(
        const uint8_t* q2, const int8_t* q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (load_dword_par(q2 + i) >> shift) & 0x03030303;
        sum = sudot4_par(v, load_dword_i8_par(q8 + i), sum);
    }
    return sum;
}

// Half-block dot: 4 of 16 dot_q2_16 calls (2 of 4 j-iterations for one
// of 2 k-outers). Returns this lane's contribution to block_result:
//   sub=0 (k=0,j=0,1):  dall * partial_isum  - dmin * (summs[0..4])
//   sub=1 (k=0,j=2,3):  dall * partial_isum  - dmin * (summs[4..8])
//   sub=2 (k=1,j=0,1):  dall * partial_isum  - dmin * (summs[8..12])
//   sub=3 (k=1,j=2,3):  dall * partial_isum  - dmin * (summs[12..16])
__device__ __forceinline__ float dot_block_quarter(
        const uint8_t* x_blk, const uint8_t* y_blk, unsigned int sub) {
    const uint8_t* sc       = x_blk;
    const uint8_t* q2_base  = x_blk + 16;
    const uint16_t xd_bits  = *(const uint16_t*)(x_blk + 80);
    const uint16_t xdm_bits = *(const uint16_t*)(x_blk + 82);

    const float yd  = *(const float*)(y_blk + 0);
    const int8_t* q8_base = (const int8_t*)(y_blk + 4);
    const int16_t* bsums  = (const int16_t*)(y_blk + 260);

    const float dall = yd * f16_to_f32_par(xd_bits);
    const float dmin = yd * f16_to_f32_par(xdm_bits);

    const unsigned int k     = sub >> 1u;   // 0 or 1
    const unsigned int j_lo  = (sub & 1u) * 2u; // 0 or 2

    // is base for this (k, j_lo): 2 * (4*k + j_lo)
    const unsigned int is_base = 2u * (4u * k + j_lo);
    // q2 base for this k: q2_base + 32*k
    const uint8_t* q2 = q2_base + (uint64_t)k * 32u;
    // q8 base for this (k, j_lo): k*128 + j_lo*32
    const int8_t* q8 = q8_base + (uint64_t)k * 128u + (uint64_t)j_lo * 32u;
    // shift for this j_lo: 2 * j_lo
    int shift = (int)(2u * j_lo);

    int32_t partial_isum = 0;
    #pragma unroll
    for (int jj = 0; jj < 2; jj++) {
        const int32_t d_a = (int32_t)(sc[is_base + 2u * jj] & 0x0f);
        partial_isum += d_a * dot_q2_16_par(q2, q8, shift);
        const int32_t d_b = (int32_t)(sc[is_base + 2u * jj + 1u] & 0x0f);
        partial_isum += d_b * dot_q2_16_par(q2 + 16, q8 + 16, shift);
        shift += 2;
        q8 += 32;
    }

    // 4 of 16 summs entries; sub picks which 4.
    int32_t partial_summs = 0;
    const unsigned int summs_lo = sub * 4u;
    #pragma unroll
    for (unsigned int t = 0; t < 4u; t++) {
        const unsigned int idx = summs_lo + t;
        partial_summs += (int32_t)bsums[idx] * (int32_t)(sc[idx] >> 4);
    }

    return dall * (float)partial_isum - dmin * (float)partial_summs;
}

__device__ __forceinline__ float warp_sum_f32_par(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

extern "C" __global__ void q2_k_accumulate_matvec_par(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_expert,
        const uint8_t  * __restrict__ xq,
        unsigned int     n_rows,
        unsigned int     n_blocks_in,
        unsigned int     zero_init) {
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;        // 0..7
    const unsigned int sub        = lane & 3u;         // 0..3

    const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q2_K_BYTES;

    float acc = 0.0f;
    // Each lane group of 4 strides over blocks with step 8.
    for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
        const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_Q2_K_BYTES;
        const uint8_t* y_blk = xq       + (uint64_t)b * BLOCK_Q8_K_BYTES;
        acc += dot_block_quarter(w_blk, y_blk, sub);
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

// Batched variant (M14k.5): single launch loops over all N_EXPERT_USED
// slots internally per workgroup. Writes the summed result directly to
// out (no accumulate/zero_init dance). Eliminates 5 launches + 5
// kernel-boundary syncs per layer; weights and per-slot midq are read
// from base + (selected[slot] * dbpe) and (slot * xq_slot_stride)
// respectively.
extern "C" __global__ void q2_k_matvec_par_batched(
        float          * __restrict__ out,           // [n_rows]
        const uint8_t  * __restrict__ w_base,        // whole weight tensor
        const uint8_t  * __restrict__ xq_base,       // slot-major midq_cat
        const int      * __restrict__ selected,      // [n_used]
        unsigned int     dbpe,                       // weight bytes per expert
        unsigned int     xq_slot_stride,             // xq bytes per slot
        unsigned int     n_used,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
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
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q2_K_BYTES;
        for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
            const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_Q2_K_BYTES;
            const uint8_t* y_blk = xq_slot  + (uint64_t)b * BLOCK_Q8_K_BYTES;
            acc += dot_block_quarter(w_blk, y_blk, sub);
        }
    }
    acc = warp_sum_f32_par(acc);

    if (lane == 0) {
        out[row] = acc;
    }
}

// M50 Phase 3 v0: B-batched variant. grid.z = B; each WG processes ONE
// (row, batch) and loops over n_used slots internally. Per-batch xq
// midq[B, n_used, n_blocks * BLOCK_Q8_K_BYTES], selected[B, n_used],
// out[B, n_rows]. Weight tensor SHARED across batch.
extern "C" __global__ void q2_k_matvec_par_batched_BxN(
        float          * __restrict__ out,           // [B, n_rows]
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,       // [B, n_used*xq_slot_stride]
        const int      * __restrict__ selected,      // [B, n_used]
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const unsigned int b            = blockIdx.z;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    const uint8_t* xq_b = xq_base
                        + (size_t)b * (size_t)n_used * (size_t)xq_slot_stride;
    const int*     sel_b = selected + (size_t)b * (size_t)n_used;

    float acc = 0.0f;
    for (unsigned int s = 0; s < n_used; s++) {
        const int e = sel_b[s];
        const uint8_t* w_expert = w_base + (uint64_t)e * (uint64_t)dbpe;
        const uint8_t* xq_slot  = xq_b + (uint64_t)s * (uint64_t)xq_slot_stride;
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q2_K_BYTES;
        for (unsigned int bl = block_lane; bl < n_blocks_in; bl += 8u) {
            const uint8_t* w_blk = row_base + (uint64_t)bl * BLOCK_Q2_K_BYTES;
            const uint8_t* y_blk = xq_slot  + (uint64_t)bl * BLOCK_Q8_K_BYTES;
            acc += dot_block_quarter(w_blk, y_blk, sub);
        }
    }
    acc = warp_sum_f32_par(acc);

    if (lane == 0) {
        out[(size_t)b * (size_t)n_rows + (size_t)row] = acc;
    }
}

// M56 decode het-split: q2_k_matvec_par_batched with a resident-expert
// remap (see iq2 _hetsplit comment). mode 0 = iGPU (process remap[e] < 0,
// weights at e); mode 1 = dGPU (process remap[e] >= 0, weights at the dense
// remap[e] slot). `out` accumulates only this device's slots — the two
// devices' partials add at ffn_combine.
extern "C" __global__ void q2_k_matvec_par_batched_hetsplit(
        float          * __restrict__ out,           // [n_rows] (this device's partial)
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,       // slot-major midq_cat
        const int      * __restrict__ selected,      // [n_used]
        const int      * __restrict__ remap,         // [256]
        unsigned int     mode,
        unsigned int     dgpu_cap,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
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
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q2_K_BYTES;
        for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
            const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_Q2_K_BYTES;
            const uint8_t* y_blk = xq_slot  + (uint64_t)b * BLOCK_Q8_K_BYTES;
            acc += dot_block_quarter(w_blk, y_blk, sub);
        }
    }
    acc = warp_sum_f32_par(acc);

    if (lane == 0) {
        out[row] = acc;   // all-skipped slots → 0.0 (still written)
    }
}

// by_expert: invert the (B, expert) iteration of _batched_BxN.
//
// _batched_BxN has grid (n_rows/8, 1, B) and each WG loops over n_used
// experts for ITS batch slot, re-reading each expert's row-tile bytes
// every time. With B=512, n_used=6, ~128 distinct experts: each expert's
// row-tile is read ~24× redundantly across the dispatch. PMC confirms:
// L2CacheHit 4%, MemUnitBusy 99.98%, 94% of iGPU DRAM BW peak → the
// kernel is hard BW-bound on the redundant weight reads.
//
// This variant uses grid (n_rows/8, n_work_items) — one WG per (8-row
// tile, expert-chunk) — and loops over the chunk's (b, used_slot)
// members. Each expert's row-tile is read ONCE per WG (instead of K_avg
// times), so total weight bytes drop from B*n_used*tile to
// distinct(E)*tile.
//
// Writes per-(b, used_slot) PARTIAL sums to `partials[b*n_used + s][row]`
// — exactly one writer per slot, no atomic, deterministic. A separate
// reduce kernel (`q2_k_reduce_partials`) sums the n_used partials per
// (b, row) to produce `out[b][row]` matching the BXN kernel's semantics.
//
// `group_count` + `expert_members` + `work_items` are the same arrays
// the iq2 by-expert / staged kernels consume, built by the upstream
// moe_group_builder pre-pass (one builder run feeds both iq2 and q2k).
extern "C" __global__ void q2_k_matvec_par_by_expert(
        float          * __restrict__ partials,      // [B*n_used, n_rows]
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,       // [B, n_used*xq_slot_stride]
        const int      * __restrict__ group_count,   // [N_EXPERT]
        const int      * __restrict__ expert_members,// [N_EXPERT * max_per_expert]: (b<<16) | slot
        const int      * __restrict__ work_items,    // [n_work_items]:    (e<<16) | member_start
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     chunk_size,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const unsigned int w_idx        = blockIdx.y;
    if (row >= n_rows) return;

    const int packed_item = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    // Expert's row-tile bytes: read once per WG; the per-member loop below
    // hits these same n_blocks_in × 84 bytes repeatedly so they live in L1.
    const uint8_t* w_expert  = w_base + (uint64_t)e * (uint64_t)dbpe;
    const uint8_t* row_base  = w_expert + (uint64_t)row * (uint64_t)n_blocks_in * BLOCK_Q2_K_BYTES;

    for (unsigned int mi = member_start; mi < member_end; mi++) {
        const int packed = expert_members[(size_t)e * (size_t)max_per_expert + (size_t)mi];
        const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
        const unsigned int slot = (unsigned int)(packed       ) & 0xffffu;
        const uint8_t* xq_slot = xq_base
                               + (size_t)b * (size_t)n_used * (size_t)xq_slot_stride
                               + (size_t)slot * (size_t)xq_slot_stride;

        float acc = 0.0f;
        for (unsigned int bl = block_lane; bl < n_blocks_in; bl += 8u) {
            const uint8_t* w_blk = row_base + (uint64_t)bl * BLOCK_Q2_K_BYTES;
            const uint8_t* y_blk = xq_slot  + (uint64_t)bl * BLOCK_Q8_K_BYTES;
            acc += dot_block_quarter(w_blk, y_blk, sub);
        }
        acc = warp_sum_f32_par(acc);

        if (lane == 0) {
            // One writer per (b, slot, row) — no atomic, no race.
            partials[((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows
                     + (size_t)row] = acc;
        }
    }
}

// =============================================================================
// by_expert_kwide — M51 S2: unpack-once member loop for the down matvec.
//
// `by_expert` re-unpacks the lane's quarter-block weights for EVERY member:
// per (member, lane): 16 shift+and dwords + 4 scale imuls + 2 f16 converts
// + a warp_sum — ~48 VALU where the roofline counts 16 sudot4.
//
// This variant inverts the loops: per (block, lane) the 16 weight dwords are
// unpacked ONCE into registers with the 4-bit group scale FOLDED at unpack
// time (((q2>>shift)&0x03030303) * (sc&0xF) is byte-safe: 3·15 = 45 < 256,
// no cross-byte carries — the integer block sum is bit-identical), and the
// f16 d/dmin are converted once. Per (member, lane) only: 16 sudot4 +
// 4-entry bsums min-term + 2 cvt + 2 mul + 2 fma remain. Member partials
// accumulate in per-lane registers (pacc[32]) and warp-reduce ONCE per
// member at the very end instead of per (member, block).
//
// Geometry, arg list, partials semantics: identical to by_expert (drop-in;
// pair with q2_k_reduce_partials). chunk_size must be ≤ 32.
// =============================================================================

#define Q2K_KW_MAX_CHUNK 32u

extern "C" __global__ void q2_k_matvec_par_by_expert_kwide(
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
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const unsigned int w_idx        = blockIdx.y;
    if (row >= n_rows) return;

    const int packed_item = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;
    const unsigned int n_members = member_end - member_start;

    const unsigned int block_lane = lane >> 2u;        // 0..7
    const unsigned int sub        = lane & 3u;         // 0..3
    const unsigned int k_outer    = sub >> 1u;         // 0..1
    const unsigned int j_lo       = (sub & 1u) * 2u;   // 0 or 2
    const unsigned int is_base    = 2u * (4u * k_outer + j_lo);
    // q8 byte offset of this lane's 64-weight quarter within the member block.
    const unsigned int q8_off     = k_outer * 128u + j_lo * 32u;

    const uint8_t* w_expert  = w_base + (uint64_t)e * (uint64_t)dbpe;
    const uint8_t* row_base  = w_expert + (uint64_t)row * (uint64_t)n_blocks_in * BLOCK_Q2_K_BYTES;

    float pacc[Q2K_KW_MAX_CHUNK];
    #pragma unroll
    for (unsigned int m = 0; m < Q2K_KW_MAX_CHUNK; m++) pacc[m] = 0.0f;

    for (unsigned int bl = block_lane; bl < n_blocks_in; bl += 8u) {
        const uint8_t* w_blk = row_base + (uint64_t)bl * BLOCK_Q2_K_BYTES;
        const uint8_t* sc      = w_blk;
        const uint8_t* q2_base = w_blk + 16 + (uint64_t)k_outer * 32u;
        const float d_lane    = f16_to_f32_par(*(const uint16_t*)(w_blk + 80));
        const float dmin_lane = f16_to_f32_par(*(const uint16_t*)(w_blk + 82));

        // Unpack 64 weights once, group scale folded (byte-safe, see header).
        int32_t w[16];
        {
            unsigned int wi = 0;
            int shift = (int)(2u * j_lo);
            #pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                const int32_t d_a = (int32_t)(sc[is_base + 2u * jj] & 0x0f);
                const int32_t d_b = (int32_t)(sc[is_base + 2u * jj + 1u] & 0x0f);
                #pragma unroll
                for (uint32_t i = 0; i < 16; i += 4) {
                    w[wi++] = (((load_dword_par(q2_base + i) >> shift) & 0x03030303) * d_a);
                }
                #pragma unroll
                for (uint32_t i = 0; i < 16; i += 4) {
                    w[wi++] = (((load_dword_par(q2_base + 16 + i) >> shift) & 0x03030303) * d_b);
                }
                shift += 2;
            }
        }
        // 4-bit mins for this lane's 4 bsums entries.
        int32_t smin[4];
        #pragma unroll
        for (unsigned int t = 0; t < 4u; t++) {
            smin[t] = (int32_t)(sc[sub * 4u + t] >> 4);
        }

        #pragma unroll 2
        for (unsigned int m = 0; m < n_members; m++) {
            const int packed = expert_members[(size_t)e * (size_t)max_per_expert
                                            + (size_t)(member_start + m)];
            const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
            const unsigned int slot = (unsigned int)(packed       ) & 0xffffu;
            const uint8_t* y_blk = xq_base
                                 + (size_t)b * (size_t)n_used * (size_t)xq_slot_stride
                                 + (size_t)slot * (size_t)xq_slot_stride
                                 + (size_t)bl * (size_t)BLOCK_Q8_K_BYTES;
            const float yd = *(const float*)(y_blk + 0);
            const int8_t* q8 = (const int8_t*)(y_blk + 4) + q8_off;

            int32_t isum = 0;
            #pragma unroll
            for (unsigned int i = 0; i < 16u; i++) {
                isum = sudot4_par(w[i], load_dword_i8_par(q8 + 4u * i), isum);
            }
            const int16_t* bsums = (const int16_t*)(y_blk + 260);
            int32_t summs = 0;
            #pragma unroll
            for (unsigned int t = 0; t < 4u; t++) {
                summs += (int32_t)bsums[sub * 4u + t] * smin[t];
            }
            pacc[m] += (yd * d_lane) * (float)isum - (yd * dmin_lane) * (float)summs;
        }
    }

    for (unsigned int m = 0; m < n_members; m++) {
        const float acc = warp_sum_f32_par(pacc[m]);
        if (lane == 0) {
            const int packed = expert_members[(size_t)e * (size_t)max_per_expert
                                            + (size_t)(member_start + m)];
            const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
            const unsigned int slot = (unsigned int)(packed       ) & 0xffffu;
            partials[((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows
                     + (size_t)row] = acc;
        }
    }
}

// =============================================================================
// by_expert_kwide2 — M53: row-pair activation reuse on top of kwide.
//
// kwide is cache-BW-bound on cross-WG activation re-reads: every 8-row WG
// pulls its members' full midq (≈74 KB at chunk=32) from L2/MALL. Here each
// warp owns TWO adjacent rows and dots one loaded q8/bsums set against both
// rows' (separately unpacked, scale-folded) weights — 16 rows per WG, so
// per-token activation traffic HALVES. Weight traffic is unchanged (each
// row's tile still read once per work item).
//
// Register budget forces members to be processed in HALVES of 16
// (pacc[2 rows][16 members] = 32 VGPR + 2×16 weight dwords + q8 in flight).
// The second half re-reads q8 (the traffic win is 2× from rows, not 4×).
//
// Grid: (n_rows/16, n_work_items). Block 256 (8 warps × 2 rows = 16 rows).
// n_rows must be a multiple of 16. Same partials semantics as by_expert.
// =============================================================================

extern "C" __global__ void q2_k_matvec_par_by_expert_kwide2(
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

    const unsigned int block_lane = lane >> 2u;        // 0..7
    const unsigned int sub        = lane & 3u;         // 0..3
    const unsigned int k_outer    = sub >> 1u;
    const unsigned int j_lo       = (sub & 1u) * 2u;
    const unsigned int is_base    = 2u * (4u * k_outer + j_lo);
    const unsigned int q8_off     = k_outer * 128u + j_lo * 32u;

    const uint8_t* w_expert = w_base + (uint64_t)e * (uint64_t)dbpe;

    // Process members in halves of 16 to keep pacc at 2×16.
    for (unsigned int half_lo = 0; half_lo < n_members; half_lo += 16u) {
        const unsigned int half_n = min(16u, n_members - half_lo);

        float pacc0[16];
        float pacc1[16];
        #pragma unroll
        for (unsigned int m = 0; m < 16u; m++) { pacc0[m] = 0.0f; pacc1[m] = 0.0f; }

        for (unsigned int bl = block_lane; bl < n_blocks_in; bl += 8u) {
            // Unpack BOTH rows' quarters once, scales folded (see kwide).
            int32_t w0[16], w1[16];
            float d0, dmin0, d1, dmin1;
            int32_t smin0[4], smin1[4];
            #pragma unroll
            for (unsigned int rr = 0; rr < 2u; rr++) {
                const uint8_t* w_blk = w_expert
                    + (uint64_t)(row0 + rr) * (uint64_t)n_blocks_in * BLOCK_Q2_K_BYTES
                    + (uint64_t)bl * BLOCK_Q2_K_BYTES;
                const uint8_t* sc      = w_blk;
                const uint8_t* q2_base = w_blk + 16 + (uint64_t)k_outer * 32u;
                const float dd  = f16_to_f32_par(*(const uint16_t*)(w_blk + 80));
                const float dm  = f16_to_f32_par(*(const uint16_t*)(w_blk + 82));
                int32_t* w = rr ? w1 : w0;
                unsigned int wi = 0;
                int shift = (int)(2u * j_lo);
                #pragma unroll
                for (int jj = 0; jj < 2; jj++) {
                    const int32_t d_a = (int32_t)(sc[is_base + 2u * jj] & 0x0f);
                    const int32_t d_b = (int32_t)(sc[is_base + 2u * jj + 1u] & 0x0f);
                    #pragma unroll
                    for (uint32_t i = 0; i < 16; i += 4) {
                        w[wi++] = (((load_dword_par(q2_base + i) >> shift) & 0x03030303) * d_a);
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < 16; i += 4) {
                        w[wi++] = (((load_dword_par(q2_base + 16 + i) >> shift) & 0x03030303) * d_b);
                    }
                    shift += 2;
                }
                if (rr) { d1 = dd; dmin1 = dm; } else { d0 = dd; dmin0 = dm; }
                int32_t* sm = rr ? smin1 : smin0;
                #pragma unroll
                for (unsigned int t = 0; t < 4u; t++) {
                    sm[t] = (int32_t)(sc[sub * 4u + t] >> 4);
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
                const int8_t* q8 = (const int8_t*)(y_blk + 4) + q8_off;

                // q8 loaded once, dotted against both rows.
                int32_t q[16];
                #pragma unroll
                for (unsigned int i = 0; i < 16u; i++) {
                    q[i] = load_dword_i8_par(q8 + 4u * i);
                }
                int32_t isum0 = 0, isum1 = 0;
                #pragma unroll
                for (unsigned int i = 0; i < 16u; i++) {
                    isum0 = sudot4_par(w0[i], q[i], isum0);
                    isum1 = sudot4_par(w1[i], q[i], isum1);
                }
                const int16_t* bsums = (const int16_t*)(y_blk + 260);
                int32_t summs0 = 0, summs1 = 0;
                #pragma unroll
                for (unsigned int t = 0; t < 4u; t++) {
                    const int32_t bs = (int32_t)bsums[sub * 4u + t];
                    summs0 += bs * smin0[t];
                    summs1 += bs * smin1[t];
                }
                pacc0[m] += (yd * d0) * (float)isum0 - (yd * dmin0) * (float)summs0;
                pacc1[m] += (yd * d1) * (float)isum1 - (yd * dmin1) * (float)summs1;
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

// Sum partials across the n_used dimension, matching the BXN kernel's
// `acc = Σ_s dot(...)` semantics exactly. Each thread = one (b, row)
// output slot; reduces a tiny n_used (typically 6) values.
//
// Caller MUST pre-fill `partials` with zeros for any (b, slot) that the
// by_expert kernel didn't touch — i.e. before launching by_expert,
// memset partials to 0. (The by_expert kernel only writes the (b, slot)
// pairs that actually appear in expert_members; unwritten slots must be
// zero to leave `out` unchanged for that contribution.)
extern "C" __global__ void q2_k_reduce_partials(
        float       * __restrict__ out,        // [B, n_rows]
        const float * __restrict__ partials,   // [B*n_used, n_rows]
        unsigned int  n_used,
        unsigned int  n_rows) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = gridDim.x * blockDim.x;
    const unsigned int b   = tid / n_rows;
    const unsigned int row = tid % n_rows;
    if (b >= (total / n_rows)) return;
    float acc = 0.0f;
    for (unsigned int s = 0; s < n_used; s++) {
        acc += partials[((size_t)b * (size_t)n_used + (size_t)s) * (size_t)n_rows
                        + (size_t)row];
    }
    out[(size_t)b * (size_t)n_rows + (size_t)row] = acc;
}

// M61 prefill het-split reduce: sum ONLY the slots this device computed,
// so neither side needs a partials zero-fill (the M53 invariant survives
// the split). Slot ownership recomputes the resident-rank exactly like
// moe_group_builder_hetsplit: resident slots ranked in slot order, rank <
// cap → dGPU (mode 1), everything else → iGPU (mode 0).
extern "C" __global__ void q2_k_reduce_partials_hetsplit(
        float       * __restrict__ out,        // [B, n_rows]
        const float * __restrict__ partials,   // [B*n_used, n_rows]
        const int   * __restrict__ d_selected, // [B, n_used]
        const int   * __restrict__ remap,      // [N_EXPERT total] id → dense or -1
        unsigned int  mode,                    // 0 = iGPU misses, 1 = dGPU hits
        unsigned int  cap,
        unsigned int  n_used,
        unsigned int  n_rows) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = gridDim.x * blockDim.x;
    const unsigned int b   = tid / n_rows;
    const unsigned int row = tid % n_rows;
    if (b >= (total / n_rows)) return;
    const int* sel_b = d_selected + (size_t)b * (size_t)n_used;
    float acc = 0.0f;
    unsigned int rank = 0;
    for (unsigned int s = 0; s < n_used; s++) {
        const int e = sel_b[s];
        const bool resident   = (e >= 0) && (remap[e] >= 0);
        const bool dgpu_takes = resident && (rank < cap);
        if (resident) rank++;
        const bool take = (mode == 1) ? dgpu_takes : !dgpu_takes;
        if (take) {
            acc += partials[((size_t)b * (size_t)n_used + (size_t)s)
                            * (size_t)n_rows + (size_t)row];
        }
    }
    out[(size_t)b * (size_t)n_rows + (size_t)row] = acc;
}
