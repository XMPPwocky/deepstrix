// iq2_xxs_pair_matvec_par — parallel reimplementation of iq2_xxs_pair_matvec (M14d).
//
// Original kernel uses 1 warp per row, with each lane covering blocks
// `lane, lane + 32, ...`. For V4-Flash n_blocks = 4096/256 = 16, so only
// lanes 0..15 do work — 50% of every warp is idle, and each active lane
// runs an 8-iteration ib32 chain (8 sub-blocks of 32 weights each).
//
// This variant maps **2 lanes per super-block** when n_blocks ≤ 16:
//   lane: 0  1  2  3 ... 30 31
//   blk:  0  0  1  1 ... 15 15
//   half: lo hi lo hi ... lo hi
// All 32 lanes do work; each lane's inner chain shrinks to 4 ib32
// iterations. Memory access is *more* coalesced — lane 0 and lane 1 read
// adjacent halves of the same 66-byte IQ2_XXS block instead of two
// strided blocks.
//
// For n_blocks > 16 (in_dim > 4096, not used by V4-Flash) the lane→block
// step wraps with stride 16, so each (lane_pair) iterates n_blocks/16
// times — generality is preserved.
//
// Numerics: integer bsum order is identical (same per-sub-block grouping);
// the f32 reduction order differs (warp-tree sums 32 partials instead of
// 16), so results match the serial kernel within f32-ULP.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q8_K_BYTES 292
#define BLOCK_IQ2_XXS_BYTES 66
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

#include "iq2_xxs_tables.inc"

__device__ __forceinline__ float f16_to_f32_par(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4_par(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t load_i32_unaligned_par(const int8_t* p) {
    const uint8_t* u = (const uint8_t*)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
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

__device__ __forceinline__ int32_t iq2_dp4a_8_par(
        uint64_t grid_val, uint32_t sign_idx, const int8_t* q8, int32_t acc) {
    const uint32_t s = ksigns_iq2xs[sign_idx];
    const uint32_t signs = s * 0x01010101u;
    const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
    const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
    const int32_t g0 = cond_neg_bytes_par((int32_t)(uint32_t)grid_val, sm0);
    const int32_t g1 = cond_neg_bytes_par((int32_t)(uint32_t)(grid_val >> 32), sm1);
    acc = sudot4_par(g0, load_i32_unaligned_par(q8 + 0), acc);
    acc = sudot4_par(g1, load_i32_unaligned_par(q8 + 4), acc);
    return acc;
}

__device__ __forceinline__ int32_t dot_iq2_pair_16_par(
        uint8_t g0_idx, uint32_t s0_idx,
        uint8_t g1_idx, uint32_t s1_idx,
        const int8_t* q8) {
    int32_t sum = 0;
    sum = iq2_dp4a_8_par(iq2xxs_grid[g0_idx], s0_idx, q8 + 0, sum);
    sum = iq2_dp4a_8_par(iq2xxs_grid[g1_idx], s1_idx, q8 + 8, sum);
    return sum;
}

// Half-block dot product: 4 of 8 ib32 sub-blocks (selected by `ib32_lo` ∈ {0, 4}).
// Returns 0.125 * d * partial_bsum so caller can just sum f32s across lanes.
__device__ __forceinline__ float dot_block_half(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int ib32_lo) {
    const uint16_t wd = *(const uint16_t*)(w_blk + 0);
    const uint16_t* q2 = (const uint16_t*)(w_blk + 2) + (size_t)ib32_lo * 4u;
    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8 = (const int8_t*)(y_blk + 4) + (size_t)ib32_lo * 32u;
    const float d = f16_to_f32_par(wd) * yd;

    int32_t bsum = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        int32_t sumi = 0;
        sumi += dot_iq2_pair_16_par(
            (uint8_t)(aux0 & 0xffu),         (aux1 >>  0) & 127u,
            (uint8_t)((aux0 >>  8) & 0xffu), (aux1 >>  7) & 127u,
            q8 + 0);
        sumi += dot_iq2_pair_16_par(
            (uint8_t)((aux0 >> 16) & 0xffu), (aux1 >> 14) & 127u,
            (uint8_t)((aux0 >> 24) & 0xffu), (aux1 >> 21) & 127u,
            q8 + 16);
        bsum += sumi * (int32_t)ls;
        q8 += 32;
    }
    return 0.125f * d * (float)bsum;
}

__device__ __forceinline__ float warp_sum_f32_par(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// Paired matvec — par variant.
// gate[row] and up[row] for r in [0, n_rows). 1 warp per row, 2 lanes per
// super-block (stride 16 over blocks).
//
// Launch:
//   grid:  (n_rows / 8, 1, 1)
//   block: (256, 1, 1)  [8 rows × 32 lanes per warp]
extern "C" __global__ void iq2_xxs_pair_matvec_par(
        float        * __restrict__ gate,
        float        * __restrict__ up,
        const uint8_t * __restrict__ gate_w,
        const uint8_t * __restrict__ up_w,
        const uint8_t * __restrict__ xq,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 1u;        // 0..15
    const unsigned int ib32_lo    = (lane & 1u) * 4u;  // 0 or 4

    const uint8_t* gate_row_base = gate_w + (uint64_t)row * n_blocks * BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_row_base   = up_w   + (uint64_t)row * n_blocks * BLOCK_IQ2_XXS_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;

    // Each (lane_pair) sums blocks block_lane, block_lane+16, ...
    // For n_blocks ≤ 16 the loop runs once. For larger in_dim it strides.
    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* y   = xq            + (uint64_t)b * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half(w_g, y, ib32_lo);
        u_sum += dot_block_half(w_u, y, ib32_lo);
    }

    g_sum = warp_sum_f32_par(g_sum);
    u_sum = warp_sum_f32_par(u_sum);

    if (lane == 0) {
        gate[row] = g_sum;
        up[row]   = u_sum;
    }
}

// Fused variant: paired iq2 matvec → silu_clamp(g) * clamp(u) * expert_w[slot]
// → writes directly to mid[slot * n_rows + row], skipping the intermediate
// gate_cat / up_cat buffers and the standalone swiglu_clamp_weighted kernel.
// (M14e — eliminates ~192 KiB/layer of memory round-trip + one launch per
// layer.)
//
// Mirrors swiglu_clamp_weighted semantics:
//   g = min(g, clamp)
//   u = clamp(u, -clamp, clamp)
//   mid = silu(g) * u * expert_w[slot]
//
// `clamp <= 1e-6` disables clamping (matches stand-alone kernel).
//
// Launch geometry identical to iq2_xxs_pair_matvec_par:
//   grid:  (n_rows / 8, 1, 1)
//   block: (256, 1, 1)
extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu(
        float        * __restrict__ mid,         // [n_experts, n_rows]
        const uint8_t * __restrict__ gate_w,
        const uint8_t * __restrict__ up_w,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ expert_w,    // [n_experts]
        unsigned int    slot,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    const uint8_t* gate_row_base = gate_w + (uint64_t)row * n_blocks * BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_row_base   = up_w   + (uint64_t)row * n_blocks * BLOCK_IQ2_XXS_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;

    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* y   = xq            + (uint64_t)b * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half(w_g, y, ib32_lo);
        u_sum += dot_block_half(w_u, y, ib32_lo);
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

// LDS-cached helpers (M14k). Same algorithms as the _par helpers above,
// but read `ksigns_iq2xs` and `iq2xxs_grid` from shared memory instead
// of __constant__. The batched kernel cooperatively loads both tables
// (128 B + 2048 B = 2176 B of LDS) into shared memory once per
// workgroup, so the inner dequant loop's hundreds of lookups per row
// avoid constant-cache divergent-access tax.

__device__ __forceinline__ int32_t iq2_dp4a_8_lds(
        uint64_t grid_val, uint32_t sign_idx,
        const uint8_t* __restrict__ s_ksigns,
        const int8_t* q8, int32_t acc) {
    const uint32_t s = s_ksigns[sign_idx];
    const uint32_t signs = s * 0x01010101u;
    const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
    const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
    const int32_t g0 = cond_neg_bytes_par((int32_t)(uint32_t)grid_val, sm0);
    const int32_t g1 = cond_neg_bytes_par((int32_t)(uint32_t)(grid_val >> 32), sm1);
    acc = sudot4_par(g0, load_i32_unaligned_par(q8 + 0), acc);
    acc = sudot4_par(g1, load_i32_unaligned_par(q8 + 4), acc);
    return acc;
}

__device__ __forceinline__ int32_t dot_iq2_pair_16_lds(
        uint8_t g0_idx, uint32_t s0_idx,
        uint8_t g1_idx, uint32_t s1_idx,
        const uint8_t* __restrict__ s_ksigns,
        const uint64_t* __restrict__ s_grid,
        const int8_t* q8) {
    int32_t sum = 0;
    sum = iq2_dp4a_8_lds(s_grid[g0_idx], s0_idx, s_ksigns, q8 + 0, sum);
    sum = iq2_dp4a_8_lds(s_grid[g1_idx], s1_idx, s_ksigns, q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ float dot_block_half_lds(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int ib32_lo,
        const uint8_t* __restrict__ s_ksigns,
        const uint64_t* __restrict__ s_grid) {
    const uint16_t wd = *(const uint16_t*)(w_blk + 0);
    const uint16_t* q2 = (const uint16_t*)(w_blk + 2) + (size_t)ib32_lo * 4u;
    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8 = (const int8_t*)(y_blk + 4) + (size_t)ib32_lo * 32u;
    const float d = f16_to_f32_par(wd) * yd;

    int32_t bsum = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        int32_t sumi = 0;
        sumi += dot_iq2_pair_16_lds(
            (uint8_t)(aux0 & 0xffu),         (aux1 >>  0) & 127u,
            (uint8_t)((aux0 >>  8) & 0xffu), (aux1 >>  7) & 127u,
            s_ksigns, s_grid, q8 + 0);
        sumi += dot_iq2_pair_16_lds(
            (uint8_t)((aux0 >> 16) & 0xffu), (aux1 >> 14) & 127u,
            (uint8_t)((aux0 >> 24) & 0xffu), (aux1 >> 21) & 127u,
            s_ksigns, s_grid, q8 + 16);
        bsum += sumi * (int32_t)ls;
        q8 += 32;
    }
    return 0.125f * d * (float)bsum;
}

// Batched variant (M14j): single launch handles all N_EXPERT_USED slots.
// grid.y = slot; selected[slot] picks the expert index; gate_bpe/up_bpe
// give per-expert weight byte stride. Eliminates 5 launches + 5
// kernel-boundary syncs per layer (≈645μs/token of launch overhead +
// scheduler resets), while doing identical compute.
extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_batch(
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
    // M14k: cache iq2 tables (128 B ksigns + 2 KiB grid) in LDS so the
    // inner dequant loop's hundreds of lookups per row avoid the
    // constant-cache penalty for divergent indices.
    // M14k.3: also cache the Q8K activation (n_blocks × 292 B), which is
    // identical across all N_EXPERT_USED workgroup-Y slices. With grid.y
    // spawning 6 workgroups that each independently re-read xq from
    // global memory, LDS-staging once per workgroup keeps the inner
    // dot_q2_16 loads in scratchpad.
    __shared__ uint8_t  s_ksigns[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];  // BLOCK_Q8_K_BYTES * max n_blocks (16 for V4-Flash)
    const unsigned int tid = threadIdx.x;
    if (tid < 128) s_ksigns[tid] = ksigns_iq2xs[tid];
    if (tid < 256) s_grid[tid]   = iq2xxs_grid[tid];
    // Cooperatively load xq into LDS. n_blocks * 292 bytes; 256 threads
    // each grab ~18 bytes. Bound by sizeof(s_xq) at compile time.
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
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;

    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* y   = s_xq          + (size_t)b   * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half_lds(w_g, y, ib32_lo, s_ksigns, s_grid);
        u_sum += dot_block_half_lds(w_u, y, ib32_lo, s_ksigns, s_grid);
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
