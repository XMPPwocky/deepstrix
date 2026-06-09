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

// q8 + 0 and q8 + 4 are both 4-aligned (Q8_K blocks start at 0 mod 4,
// q8 lives at block offset 4). memcpy folds to a single dword load.
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
    acc = sudot4_par(g0, load_dword_par(q8 + 0), acc);
    acc = sudot4_par(g1, load_dword_par(q8 + 4), acc);
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

// M31: branchless byte-wise conditional negation. For each byte i,
// if sm.byte[i] == 0xff, negate pack.byte[i]; else keep.
// Two's complement: -x = ~x + 1. So `(pack ^ sm) + (sm & 0x01010101u)`
// XOR-flips the bytes selected by sm, then adds 1 to those bytes.
// Replaces 4 conditional branches per call in the inner loop.
__device__ __forceinline__ int32_t cond_neg_bytes_fast(int32_t pack, uint32_t sm) {
    return (int32_t)(((uint32_t)pack ^ sm) + (sm & 0x01010101u));
}

// M31: per-sign expanded mask pair. `pair` packs sm0 (low 32 bits) and
// sm1 (high 32 bits) — both byte masks of 0x00/0xff per byte. We
// precompute this table at WG init from `ksigns_iq2xs`, replacing two
// branch-heavy `expand_signs_par` calls per iq2_dp4a_8 invocation
// (~32 calls per lane in the inner loop).
__device__ __forceinline__ int32_t iq2_dp4a_8_lds(
        uint64_t grid_val, uint32_t sign_idx,
        const uint64_t* __restrict__ s_sign_pair,
        const int8_t* q8, int32_t acc) {
    const uint64_t pair = s_sign_pair[sign_idx];
    const uint32_t sm0 = (uint32_t)pair;
    const uint32_t sm1 = (uint32_t)(pair >> 32);
    const int32_t g0 = cond_neg_bytes_fast((int32_t)(uint32_t)grid_val, sm0);
    const int32_t g1 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid_val >> 32), sm1);
    acc = sudot4_par(g0, load_dword_par(q8 + 0), acc);
    acc = sudot4_par(g1, load_dword_par(q8 + 4), acc);
    return acc;
}

__device__ __forceinline__ int32_t dot_iq2_pair_16_lds(
        uint8_t g0_idx, uint32_t s0_idx,
        uint8_t g1_idx, uint32_t s1_idx,
        const uint64_t* __restrict__ s_sign_pair,
        const uint64_t* __restrict__ s_grid,
        const int8_t* q8) {
    int32_t sum = 0;
    sum = iq2_dp4a_8_lds(s_grid[g0_idx], s0_idx, s_sign_pair, q8 + 0, sum);
    sum = iq2_dp4a_8_lds(s_grid[g1_idx], s1_idx, s_sign_pair, q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ float dot_block_half_lds(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int ib32_lo,
        const uint64_t* __restrict__ s_sign_pair,
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
            s_sign_pair, s_grid, q8 + 0);
        sumi += dot_iq2_pair_16_lds(
            (uint8_t)((aux0 >> 16) & 0xffu), (aux1 >> 14) & 127u,
            (uint8_t)((aux0 >> 24) & 0xffu), (aux1 >> 21) & 127u,
            s_sign_pair, s_grid, q8 + 16);
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
    // M31: replaced s_ksigns[128] (1 B/entry) with s_sign_pair[128]
    // (8 B/entry, packed sm0|sm1<<32). Eliminates two expand_signs_par
    // calls per iq2_dp4a_8 invocation. WG init cost: 128 expands once
    // per launch (amortized; original did ~16K expands per WG in the
    // inner loop).
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];  // BLOCK_Q8_K_BYTES * max n_blocks (16 for V4-Flash)
    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];
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
        g_sum += dot_block_half_lds(w_g, y, ib32_lo, s_sign_pair, s_grid);
        u_sum += dot_block_half_lds(w_u, y, ib32_lo, s_sign_pair, s_grid);
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

// DECODE perf experiment (2026-05-27): block=512 → 16 rows/WG. Tested
// and REGRESSED 76→83 µs. The smwsum-style "more waves per WG" trick
// doesn't apply here because iq2 already has many WGs/CU (1536/40 = 38),
// so wave count was already saturated. Halving the WG count just costs
// scheduling flexibility. Kept around as a documented dead-end. To
// retry, swap launch_fused_swiglu_batch → launch_fused_swiglu_batch_b512.
extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_batch_b512(
        float        * __restrict__ mid,
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ expert_w,
        const int    * __restrict__ selected,
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];
    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];
    const unsigned int xq_bytes = n_blocks * 292u;
    for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) {
        s_xq[i] = xq[i];
    }
    __syncthreads();

    const unsigned int slot = blockIdx.y;
    const unsigned int row_in_block = tid / 32u;   // 0..15
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 16u + row_in_block;
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
        g_sum += dot_block_half_lds(w_g, y, ib32_lo, s_sign_pair, s_grid);
        u_sum += dot_block_half_lds(w_u, y, ib32_lo, s_sign_pair, s_grid);
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

// =============================================================================
// M55 decode: weight-STAGED variant of _fused_swiglu_batch.
//
// The B=1 decode kernel is weight-BW-bound, but its global weight loads are
// 2-byte u16s off the odd-offset iq2 layout (qs at block+2) — measured
// ~73% of achievable BW (139 µs/launch ≈ 187 GB/s). Here each WG coop-stages
// its 8 rows of one matrix (8 × 1056 B, row stride is 16-aligned) into LDS
// with b128 loads, then runs the existing dequant from LDS. Two passes
// (gate, then up) keep LDS at 1 KB sign + 2 KB grid + 4.7 KB xq + 8.4 KB
// weights ≈ 16.1 KB → full occupancy.
// =============================================================================
extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_batch_wstage(
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
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];
    __shared__ uint8_t  s_w[8 * 16 * BLOCK_IQ2_XXS_BYTES];  // 8 rows × 16 blocks × 66 B

    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];
    const unsigned int xq_bytes = n_blocks * 292u;
    for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) {
        s_xq[i] = xq[i];
    }

    const unsigned int slot = blockIdx.y;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool row_valid = (row < n_rows);

    const int e = selected[slot];
    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    const unsigned int row_bytes = n_blocks * BLOCK_IQ2_XXS_BYTES;   // 1056
    // Coop-stage source: the WG's 8 rows are contiguous (rows row0..row0+8
    // of one expert) — 8448 B. Row stride 1056 ≡ 0 (mod 16) and bpe is a
    // multiple of 16, so the base is b128-aligned.
    const unsigned int wg_bytes = 8u * row_bytes;
    const unsigned int wg_vec4  = wg_bytes / 16u;

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    #pragma unroll
    for (unsigned int pass = 0; pass < 2u; pass++) {
        const uint8_t* base = (pass ? up_w_base : gate_w_base)
                            + (uint64_t)e * (uint64_t)(pass ? up_bpe : gate_bpe)
                            + (uint64_t)blockIdx.x * 8u * (uint64_t)row_bytes;
        __syncthreads();   // pass 0: after LUT/xq stage; pass 1: protect s_w
        {
            const uint4* src = (const uint4*)base;
            uint4* dst = (uint4*)s_w;
            for (unsigned int i = tid; i < wg_vec4; i += blockDim.x) {
                dst[i] = src[i];
            }
        }
        __syncthreads();

        if (row_valid) {
            float acc = 0.0f;
            const uint8_t* w_row = s_w + (size_t)row_in_block * row_bytes;
            for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
                acc += dot_block_half_lds(
                    w_row + (size_t)b * BLOCK_IQ2_XXS_BYTES,
                    s_xq + (size_t)b * BLOCK_Q8_K_BYTES,
                    ib32_lo, s_sign_pair, s_grid);
            }
            if (pass) { u_sum = acc; } else { g_sum = acc; }
        }
    }

    if (!row_valid) return;
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

// M50 Phase 3 v0: batched variant of iq2_xxs_pair_matvec_fused_swiglu_batch.
// grid.z = B so each WG handles ONE (row_block, slot, batch) triple. Per-
// batch xq[B, n_blocks*292], selected[B, n_used], expert_w[B, n_used] and
// mid[B, n_used, n_rows] all have per-batch offsets. Gate/up weight
// tensors are SHARED across batch (expert weights live in iGPU global
// memory once; per-batch selection picks which expert to read).
extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_batch_BxN(
        float        * __restrict__ mid,           // [B, n_used, n_rows]
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,           // [B, n_blocks*292]
        const float  * __restrict__ expert_w,      // [B, n_used]
        const int    * __restrict__ selected,      // [B, n_used]
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        unsigned int    n_used,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];  // BLOCK_Q8_K_BYTES * max n_blocks
    const unsigned int tid = threadIdx.x;
    const unsigned int b   = blockIdx.z;

    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];

    const unsigned int xq_bytes = n_blocks * 292u;
    const uint8_t* xq_b = xq + (size_t)b * (size_t)xq_bytes;
    for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) {
        s_xq[i] = xq_b[i];
    }
    __syncthreads();

    const unsigned int slot         = blockIdx.y;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const int e = selected[(size_t)b * (size_t)n_used + (size_t)slot];

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

    for (unsigned int bl = block_lane; bl < n_blocks; bl += 16u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* y   = s_xq          + (size_t)bl   * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_half_lds(w_g, y, ib32_lo, s_sign_pair, s_grid);
        u_sum += dot_block_half_lds(w_u, y, ib32_lo, s_sign_pair, s_grid);
    }

    g_sum = warp_sum_f32_par(g_sum);
    u_sum = warp_sum_f32_par(u_sum);

    if (lane == 0) {
        const float ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
        if (clamp > 1.0e-6f) {
            if (g_sum > clamp)  g_sum = clamp;
            if (u_sum > clamp)  u_sum = clamp;
            if (u_sum < -clamp) u_sum = -clamp;
        }
        const float sig = 1.0f / (1.0f + expf(-g_sum));
        const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                             * (size_t)n_rows + (size_t)row;
        mid[out_off] = g_sum * sig * u_sum * ew;
    }
}

// M50 by-expert iq2 (Phase 7): grid (n_rows/8, n_expert). Each WG handles
// ONE (row_block, expert) pair, iterating over all (token, slot) members
// that picked this expert. Weight rows for this expert/row_block are
// read from global ONCE per first member; subsequent members hit L2.
//
// Expected win at B=64: ~6x reuse measured → ~6x weight-BW amortization.
// Wasted WGs on unselected experts (76% of 256 at B=64) exit immediately
// after the group_count[e]==0 check, so the waste is just launch overhead
// (~us per WG, total ~5ms per chunk vs ~1s chunk wall — negligible).
extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_by_expert(
        float        * __restrict__ mid,           // [B, n_used, n_rows]
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,           // [B, n_blocks * 292]
        const float  * __restrict__ expert_w,      // [B, n_used]
        const int    * __restrict__ group_count,   // [n_expert]
        const int    * __restrict__ expert_members,// [n_expert * max_per_expert]
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        unsigned int    n_used,
        unsigned int    max_per_expert,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return; // no token picked this expert this chunk

    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];

    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];
    __syncthreads();

    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;

    const unsigned int xq_bytes = n_blocks * 292u;
    for (int m = 0; m < n_members; m++) {
        // Load this member's (b, slot).
        const int packed = expert_members[(size_t)e * (size_t)max_per_expert + (size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);

        // Stage this batch's xq in LDS (overwrites previous member's).
        const uint8_t* xq_b = xq + (size_t)b * (size_t)xq_bytes;
        for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) {
            s_xq[i] = xq_b[i];
        }
        __syncthreads();

        float g_sum = 0.0f;
        float u_sum = 0.0f;
        for (unsigned int bl = block_lane; bl < n_blocks; bl += 16u) {
            const uint8_t* w_g = gate_row_base + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* w_u = up_row_base   + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* y   = s_xq          + (size_t)bl   * BLOCK_Q8_K_BYTES;
            g_sum += dot_block_half_lds(w_g, y, ib32_lo, s_sign_pair, s_grid);
            u_sum += dot_block_half_lds(w_u, y, ib32_lo, s_sign_pair, s_grid);
        }

        g_sum = warp_sum_f32_par(g_sum);
        u_sum = warp_sum_f32_par(u_sum);

        if (lane == 0) {
            const float ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
            if (clamp > 1.0e-6f) {
                if (g_sum > clamp)  g_sum = clamp;
                if (u_sum > clamp)  u_sum = clamp;
                if (u_sum < -clamp) u_sum = -clamp;
            }
            const float sig = 1.0f / (1.0f + expf(-g_sum));
            const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                                 * (size_t)n_rows + (size_t)row;
            mid[out_off] = g_sum * sig * u_sum * ew;
        }
        __syncthreads(); // before next member's xq overwrite
    }
}

// M50 by-expert iq2 — chunked variant (Phase 7.2). Solves the popular-
// expert tail latency that killed Phase 7 v0 at B≥192.
//
// Each WG handles ONE work item (= one chunk of up to CHUNK_SIZE members
// of one expert) for ONE row_block. Popular experts split into multiple
// chunks/WGs that run in parallel → bounded WG runtime regardless of
// group_count[e]. Combined with chunk-size matched to WMMA fragment
// (16×16×16), this is also the natural launch geometry for a future
// dequant-to-LDS + WMMA GEMM rewrite.
//
// Grid: (n_rows/8, n_work_items, 1). n_work_items determined at runtime
// by moe_work_items_builder (host syncs and reads it before launch).
// Block: 256 threads (8 warps × 32 lanes).
//
// work_items[i] packing: (expert_id << 16) | member_start.
// member_count = min(chunk_size, group_count[expert] - member_start),
// computed inside the kernel.
extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_chunked(
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
    if (member_end <= member_start) return; // shouldn't happen but safe

    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];

    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];
    __syncthreads();

    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;

    const unsigned int xq_bytes = n_blocks * 292u;
    for (unsigned int m = member_start; m < member_end; m++) {
        // Decode this member's (b, slot).
        const int packed = expert_members[(size_t)e * (size_t)max_per_expert + (size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);

        // Stage this batch's xq in LDS.
        const uint8_t* xq_b = xq + (size_t)b * (size_t)xq_bytes;
        for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) {
            s_xq[i] = xq_b[i];
        }
        __syncthreads();

        float g_sum = 0.0f;
        float u_sum = 0.0f;
        for (unsigned int bl = block_lane; bl < n_blocks; bl += 16u) {
            const uint8_t* w_g = gate_row_base + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* w_u = up_row_base   + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* y   = s_xq          + (size_t)bl   * BLOCK_Q8_K_BYTES;
            g_sum += dot_block_half_lds(w_g, y, ib32_lo, s_sign_pair, s_grid);
            u_sum += dot_block_half_lds(w_u, y, ib32_lo, s_sign_pair, s_grid);
        }

        g_sum = warp_sum_f32_par(g_sum);
        u_sum = warp_sum_f32_par(u_sum);

        if (lane == 0) {
            const float ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
            if (clamp > 1.0e-6f) {
                if (g_sum > clamp)  g_sum = clamp;
                if (u_sum > clamp)  u_sum = clamp;
                if (u_sum < -clamp) u_sum = -clamp;
            }
            const float sig = 1.0f / (1.0f + expf(-g_sum));
            const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                                 * (size_t)n_rows + (size_t)row;
            mid[out_off] = g_sum * sig * u_sum * ew;
        }
        __syncthreads(); // before next member's xq overwrite
    }
}

// Phase 7.2 diagnostic: same as iq2_xxs_pair_matvec_fused_swiglu_chunked
// but with the dot-product inner loop SKIPPED (writes 0.0 to mid).
// All other costs are still paid: LDS init, xq cooperative load, member
// loop, reduce shuffles, output write. Letting us isolate dot vs the rest.
extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_chunked_NODOT(
        float        * __restrict__ mid,
        const uint8_t * __restrict__ /*gate_w_base*/,
        const uint8_t * __restrict__ /*up_w_base*/,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ /*expert_w*/,
        const int    * __restrict__ group_count,
        const int    * __restrict__ expert_members,
        const int    * __restrict__ work_items,
        unsigned int    /*gate_bpe*/,
        unsigned int    /*up_bpe*/,
        unsigned int    n_used,
        unsigned int    max_per_expert,
        unsigned int    chunk_size,
        float           /*clamp*/,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    const unsigned int w_idx = blockIdx.y;
    const int packed_item = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;

    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];

    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];
    __syncthreads();

    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int xq_bytes = n_blocks * 292u;
    for (unsigned int m = member_start; m < member_end; m++) {
        const int packed = expert_members[(size_t)e * (size_t)max_per_expert + (size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);

        // Stage xq exactly like the real kernel.
        const uint8_t* xq_b = xq + (size_t)b * (size_t)xq_bytes;
        for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) {
            s_xq[i] = xq_b[i];
        }
        __syncthreads();

        // SKIP the dot loop. Touch s_xq[0] so it doesn't get DCE'd.
        float g_sum = s_xq[tid & 1023] * 0.001f;
        float u_sum = 0.0f;

        // Standard warp reduce (still pays this cost).
        g_sum = warp_sum_f32_par(g_sum);
        u_sum = warp_sum_f32_par(u_sum);

        if (lane == 0) {
            const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                                 * (size_t)n_rows + (size_t)row;
            mid[out_off] = g_sum + u_sum;
        }
        __syncthreads();
    }
}

// Phase 7.3 — same dispatch geometry as chunked, but stage the per-WG
// expert's iq2 weight bytes to LDS once before the member loop.
// Hypothesis: chunked-static didn't actually amortize weight reads via
// L2 (concurrent WGs touch other experts → eviction). By explicitly
// staging weights to LDS, the member loop reads from local scratchpad
// instead of DRAM → ~Nmembers× weight-BW reduction.
//
// LDS budget per WG (~24 KB):
//   s_sign_pair  1 KB
//   s_grid       2 KB
//   s_xq         4.7 KB
//   s_w_gate     8.4 KB (8 rows × 16 blocks × 66 bytes)
//   s_w_up       8.4 KB
// → ~3 WGs/CU LDS-limited (vs ~5 for chunked). Lower occupancy is a
// real concern but the BW savings (peak ~6× per the data) should win.
//
// We compile with conservative LDS_W_BYTES = 8*16*66 = 8448 (= max for
// V4-Flash's BLOCKS_Q8K_GATE_IN = 16). Smaller models with fewer blocks
// would waste LDS; not a concern here.

#define LDS_W_BYTES_PER_WG 8448u  // 8 rows × 16 blocks × 66 bytes

extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_chunked_lds(
        float        * __restrict__ mid,
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ expert_w,
        const int    * __restrict__ group_count,
        const int    * __restrict__ expert_members,
        const int    * __restrict__ work_items,
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

    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];
    __shared__ uint8_t  s_w_gate[LDS_W_BYTES_PER_WG];
    __shared__ uint8_t  s_w_up[LDS_W_BYTES_PER_WG];

    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];

    // Cooperatively stage this WG's 8 rows × n_blocks of gate + up
    // weights to LDS. Row offset for this WG = blockIdx.x * 8.
    const unsigned int row_offset = blockIdx.x * 8u;
    // Per-WG byte count actually used (smaller models could be < 8448).
    const unsigned int wg_bytes = 8u * n_blocks * BLOCK_IQ2_XXS_BYTES;

    // Source bases for this (expert, row_block). 8 rows are contiguous.
    const uint8_t* gate_src = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe
                            + (uint64_t)row_offset * (uint64_t)n_blocks
                            * (uint64_t)BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_src   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe
                            + (uint64_t)row_offset * (uint64_t)n_blocks
                            * (uint64_t)BLOCK_IQ2_XXS_BYTES;
    for (unsigned int i = tid; i < wg_bytes; i += blockDim.x) {
        s_w_gate[i] = gate_src[i];
        s_w_up[i]   = up_src[i];
    }
    __syncthreads();

    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    // LDS row bases for THIS warp's row.
    const uint8_t* lds_gate_row = s_w_gate + (size_t)row_in_block * (size_t)n_blocks
                                * (size_t)BLOCK_IQ2_XXS_BYTES;
    const uint8_t* lds_up_row   = s_w_up   + (size_t)row_in_block * (size_t)n_blocks
                                * (size_t)BLOCK_IQ2_XXS_BYTES;

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

        float g_sum = 0.0f;
        float u_sum = 0.0f;
        for (unsigned int bl = block_lane; bl < n_blocks; bl += 16u) {
            const uint8_t* w_g = lds_gate_row + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* w_u = lds_up_row   + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* y   = s_xq         + (size_t)bl   * BLOCK_Q8_K_BYTES;
            g_sum += dot_block_half_lds(w_g, y, ib32_lo, s_sign_pair, s_grid);
            u_sum += dot_block_half_lds(w_u, y, ib32_lo, s_sign_pair, s_grid);
        }

        g_sum = warp_sum_f32_par(g_sum);
        u_sum = warp_sum_f32_par(u_sum);

        if (lane == 0) {
            const float ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
            if (clamp > 1.0e-6f) {
                if (g_sum > clamp)  g_sum = clamp;
                if (u_sum > clamp)  u_sum = clamp;
                if (u_sum < -clamp) u_sum = -clamp;
            }
            const float sig = 1.0f / (1.0f + expf(-g_sum));
            const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                                 * (size_t)n_rows + (size_t)row;
            mid[out_off] = g_sum * sig * u_sum * ew;
        }
        __syncthreads();
    }
}

// Phase 7.4 — replace s_sign_pair LDS lookup with inline VALU bit math.
// rocprofv3 showed kernel is memory-unit-saturated (87% busy) while VALU
// is starving (7% busy) and LDS bank conflicts are 13.8%. The s_sign_pair
// table is 1 of 2 random-indexed LDS reads per dp4a_8 call; replacing
// it with inline bit math trades a high-latency LDS read for a few VALU
// ops, which is the favorable direction given the current utilization.
//
// Mathematical key: ksigns_iq2xs[i] = i ^ ((popcount(i) & 1) << 7)
// (verified: i is 7-bit, high bit = parity of i). So the entire
// pre-staged s_sign_pair[i] can be reconstructed from `i` directly
// without a memory access.
//
// Per dp4a_8 call we save 1 LDS read; per WG that's ~65K saved LDS
// reads (= ~half the data-dependent LDS reads). MemUnitBusy and
// LDSBankConflict should both drop noticeably.

__device__ __forceinline__ uint32_t ksigns_inline(uint32_t i) {
    // i is 7-bit (0..127), so the high bit is free for the parity stamp.
    return i ^ ((__popc(i) & 1u) << 7);
}

__device__ __forceinline__ int32_t iq2_dp4a_8_inline_signs(
        uint64_t grid_val, uint32_t sign_idx,
        const int8_t* q8, int32_t acc) {
    const uint32_t s = ksigns_inline(sign_idx);
    const uint32_t signs = s * 0x01010101u;
    const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
    const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
    const int32_t g0 = cond_neg_bytes_fast((int32_t)(uint32_t)grid_val, sm0);
    const int32_t g1 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid_val >> 32), sm1);
    acc = sudot4_par(g0, load_dword_par(q8 + 0), acc);
    acc = sudot4_par(g1, load_dword_par(q8 + 4), acc);
    return acc;
}

__device__ __forceinline__ int32_t dot_iq2_pair_16_inline_signs(
        uint8_t g0_idx, uint32_t s0_idx,
        uint8_t g1_idx, uint32_t s1_idx,
        const uint64_t* __restrict__ s_grid,
        const int8_t* q8) {
    int32_t sum = 0;
    sum = iq2_dp4a_8_inline_signs(s_grid[g0_idx], s0_idx, q8 + 0, sum);
    sum = iq2_dp4a_8_inline_signs(s_grid[g1_idx], s1_idx, q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ float dot_block_half_inline_signs(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int ib32_lo,
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
        sumi += dot_iq2_pair_16_inline_signs(
            (uint8_t)(aux0 & 0xffu),         (aux1 >>  0) & 127u,
            (uint8_t)((aux0 >>  8) & 0xffu), (aux1 >>  7) & 127u,
            s_grid, q8 + 0);
        sumi += dot_iq2_pair_16_inline_signs(
            (uint8_t)((aux0 >> 16) & 0xffu), (aux1 >> 14) & 127u,
            (uint8_t)((aux0 >> 24) & 0xffu), (aux1 >> 21) & 127u,
            s_grid, q8 + 16);
        bsum += sumi * (int32_t)ls;
        q8 += 32;
    }
    return 0.125f * d * (float)bsum;
}

// Chunked iq2 with inline-sign math (no s_sign_pair LDS).
extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_chunked_inline(
        float        * __restrict__ mid,
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ expert_w,
        const int    * __restrict__ group_count,
        const int    * __restrict__ expert_members,
        const int    * __restrict__ work_items,
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

    // NOTE: no s_sign_pair! Saves 1 KB of LDS per WG and skips its init.
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];

    const unsigned int tid = threadIdx.x;
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];

    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;

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

        float g_sum = 0.0f;
        float u_sum = 0.0f;
        for (unsigned int bl = block_lane; bl < n_blocks; bl += 16u) {
            const uint8_t* w_g = gate_row_base + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* w_u = up_row_base   + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* y   = s_xq          + (size_t)bl   * BLOCK_Q8_K_BYTES;
            g_sum += dot_block_half_inline_signs(w_g, y, ib32_lo, s_grid);
            u_sum += dot_block_half_inline_signs(w_u, y, ib32_lo, s_grid);
        }

        g_sum = warp_sum_f32_par(g_sum);
        u_sum = warp_sum_f32_par(u_sum);

        if (lane == 0) {
            const float ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
            if (clamp > 1.0e-6f) {
                if (g_sum > clamp)  g_sum = clamp;
                if (u_sum > clamp)  u_sum = clamp;
                if (u_sum < -clamp) u_sum = -clamp;
            }
            const float sig = 1.0f / (1.0f + expf(-g_sum));
            const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                                 * (size_t)n_rows + (size_t)row;
            mid[out_off] = g_sum * sig * u_sum * ew;
        }
        __syncthreads();
    }
}

// Phase 7.5 DIAGNOSTIC — same code structure as chunked + s_sign_pair
// LDS lookups, BUT forces all LDS indices to 0. All 32 lanes broadcast-
// read s_grid[0] and s_sign_pair[0] → no bank conflicts. Same LDS read
// COUNT as production, just conflict-free.
//
// Result will be numerically wrong (all weights look like grid[0]), but
// timing tells us how much of the wall is conflicts vs other factors.
// If wall drops a lot vs chunked → conflicts are the bottleneck.
// If wall is the same → conflicts aren't on the critical path.

__device__ __forceinline__ int32_t iq2_dp4a_8_zeroidx(
        uint64_t grid_val, uint32_t sign_idx,
        const uint64_t* __restrict__ s_sign_pair,
        const int8_t* q8, int32_t acc) {
    // Force sign_idx to 0 → all lanes read same bank.
    const uint64_t pair = s_sign_pair[0];
    const uint32_t sm0 = (uint32_t)pair;
    const uint32_t sm1 = (uint32_t)(pair >> 32);
    const int32_t g0 = cond_neg_bytes_fast((int32_t)(uint32_t)grid_val, sm0);
    const int32_t g1 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid_val >> 32), sm1);
    acc = sudot4_par(g0, load_dword_par(q8 + 0), acc);
    acc = sudot4_par(g1, load_dword_par(q8 + 4), acc);
    return acc;
}

__device__ __forceinline__ int32_t dot_iq2_pair_16_zeroidx(
        uint8_t g0_idx, uint32_t s0_idx,
        uint8_t g1_idx, uint32_t s1_idx,
        const uint64_t* __restrict__ s_sign_pair,
        const uint64_t* __restrict__ s_grid,
        const int8_t* q8) {
    int32_t sum = 0;
    // Force g_idx to 0 → broadcast access on s_grid.
    sum = iq2_dp4a_8_zeroidx(s_grid[0], s0_idx, s_sign_pair, q8 + 0, sum);
    sum = iq2_dp4a_8_zeroidx(s_grid[0], s1_idx, s_sign_pair, q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ float dot_block_half_zeroidx(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int ib32_lo,
        const uint64_t* __restrict__ s_sign_pair,
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
        sumi += dot_iq2_pair_16_zeroidx(
            (uint8_t)(aux0 & 0xffu),         (aux1 >>  0) & 127u,
            (uint8_t)((aux0 >>  8) & 0xffu), (aux1 >>  7) & 127u,
            s_sign_pair, s_grid, q8 + 0);
        sumi += dot_iq2_pair_16_zeroidx(
            (uint8_t)((aux0 >> 16) & 0xffu), (aux1 >> 14) & 127u,
            (uint8_t)((aux0 >> 24) & 0xffu), (aux1 >> 21) & 127u,
            s_sign_pair, s_grid, q8 + 16);
        bsum += sumi * (int32_t)ls;
        q8 += 32;
    }
    return 0.125f * d * (float)bsum;
}

extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_chunked_zeroidx(
        float        * __restrict__ mid,
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ expert_w,
        const int    * __restrict__ group_count,
        const int    * __restrict__ expert_members,
        const int    * __restrict__ work_items,
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

    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t  s_xq[16 * 292];

    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];
    __syncthreads();

    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;

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

        float g_sum = 0.0f;
        float u_sum = 0.0f;
        for (unsigned int bl = block_lane; bl < n_blocks; bl += 16u) {
            const uint8_t* w_g = gate_row_base + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* w_u = up_row_base   + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* y   = s_xq          + (size_t)bl   * BLOCK_Q8_K_BYTES;
            g_sum += dot_block_half_zeroidx(w_g, y, ib32_lo, s_sign_pair, s_grid);
            u_sum += dot_block_half_zeroidx(w_u, y, ib32_lo, s_sign_pair, s_grid);
        }

        g_sum = warp_sum_f32_par(g_sum);
        u_sum = warp_sum_f32_par(u_sum);

        if (lane == 0) {
            const float ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
            if (clamp > 1.0e-6f) {
                if (g_sum > clamp)  g_sum = clamp;
                if (u_sum > clamp)  u_sum = clamp;
                if (u_sum < -clamp) u_sum = -clamp;
            }
            const float sig = 1.0f / (1.0f + expf(-g_sum));
            const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                                 * (size_t)n_rows + (size_t)row;
            mid[out_off] = g_sum * sig * u_sum * ew;
        }
        __syncthreads();
    }
}

// Phase 7.5 — padded LDS tables for bank-conflict mitigation.
//
// Pads s_grid and s_sign_pair from 8-byte stride → 12-byte stride
// (8 useful bytes + 4 padding). Math (32-bank LDS):
//   8-byte stride:  gcd(2, 32) = 2 → 16 distinct bank classes
//                   → 32 balls in 16 bins → E[max load] ≈ 5.3
//   12-byte stride: gcd(3, 32) = 1 → 32 distinct bank classes
//                   → 32 balls in 32 bins → E[max load] ≈ 3.6
// Expected speedup on LDS-read step: ~5.3/3.6 = 1.47×.
//
// LDS overhead per WG:
//   s_grid:      2 KB → 3 KB   (+1 KB)
//   s_sign_pair: 1 KB → 1.5 KB (+0.5 KB)
// Total: ~9.5 KB per WG (was ~8 KB). Should not change occupancy tier.

__device__ __forceinline__ uint64_t lds_read_u64_padded12(const uint8_t* base, uint32_t idx) {
    return *(const uint64_t*)(base + idx * 12u);
}

__device__ __forceinline__ int32_t iq2_dp4a_8_padded(
        uint64_t grid_val, uint32_t sign_idx,
        const uint8_t* __restrict__ s_sign_pair_pad,
        const int8_t* q8, int32_t acc) {
    const uint64_t pair = lds_read_u64_padded12(s_sign_pair_pad, sign_idx);
    const uint32_t sm0 = (uint32_t)pair;
    const uint32_t sm1 = (uint32_t)(pair >> 32);
    const int32_t g0 = cond_neg_bytes_fast((int32_t)(uint32_t)grid_val, sm0);
    const int32_t g1 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid_val >> 32), sm1);
    acc = sudot4_par(g0, load_dword_par(q8 + 0), acc);
    acc = sudot4_par(g1, load_dword_par(q8 + 4), acc);
    return acc;
}

__device__ __forceinline__ int32_t dot_iq2_pair_16_padded(
        uint8_t g0_idx, uint32_t s0_idx,
        uint8_t g1_idx, uint32_t s1_idx,
        const uint8_t* __restrict__ s_sign_pair_pad,
        const uint8_t* __restrict__ s_grid_pad,
        const int8_t* q8) {
    int32_t sum = 0;
    const uint64_t g0_val = lds_read_u64_padded12(s_grid_pad, g0_idx);
    const uint64_t g1_val = lds_read_u64_padded12(s_grid_pad, g1_idx);
    sum = iq2_dp4a_8_padded(g0_val, s0_idx, s_sign_pair_pad, q8 + 0, sum);
    sum = iq2_dp4a_8_padded(g1_val, s1_idx, s_sign_pair_pad, q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ float dot_block_half_padded(
        const uint8_t* w_blk, const uint8_t* y_blk, unsigned int ib32_lo,
        const uint8_t* __restrict__ s_sign_pair_pad,
        const uint8_t* __restrict__ s_grid_pad) {
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
        sumi += dot_iq2_pair_16_padded(
            (uint8_t)(aux0 & 0xffu),         (aux1 >>  0) & 127u,
            (uint8_t)((aux0 >>  8) & 0xffu), (aux1 >>  7) & 127u,
            s_sign_pair_pad, s_grid_pad, q8 + 0);
        sumi += dot_iq2_pair_16_padded(
            (uint8_t)((aux0 >> 16) & 0xffu), (aux1 >> 14) & 127u,
            (uint8_t)((aux0 >> 24) & 0xffu), (aux1 >> 21) & 127u,
            s_sign_pair_pad, s_grid_pad, q8 + 16);
        bsum += sumi * (int32_t)ls;
        q8 += 32;
    }
    return 0.125f * d * (float)bsum;
}

extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_chunked_padded(
        float        * __restrict__ mid,
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ expert_w,
        const int    * __restrict__ group_count,
        const int    * __restrict__ expert_members,
        const int    * __restrict__ work_items,
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

    // Padded: 12-byte stride per entry. 256 × 12 = 3072 B, 128 × 12 = 1536 B.
    __shared__ uint8_t s_sign_pair_pad[128 * 12];
    __shared__ uint8_t s_grid_pad[256 * 12];
    __shared__ uint8_t s_xq[16 * 292];

    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        *(uint64_t*)(s_sign_pair_pad + tid * 12) =
            ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) {
        *(uint64_t*)(s_grid_pad + tid * 12) = iq2xxs_grid[tid];
    }
    __syncthreads();

    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 1u;
    const unsigned int ib32_lo    = (lane & 1u) * 4u;

    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;

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

        float g_sum = 0.0f;
        float u_sum = 0.0f;
        for (unsigned int bl = block_lane; bl < n_blocks; bl += 16u) {
            const uint8_t* w_g = gate_row_base + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* w_u = up_row_base   + (uint64_t)bl * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* y   = s_xq          + (size_t)bl   * BLOCK_Q8_K_BYTES;
            g_sum += dot_block_half_padded(w_g, y, ib32_lo, s_sign_pair_pad, s_grid_pad);
            u_sum += dot_block_half_padded(w_u, y, ib32_lo, s_sign_pair_pad, s_grid_pad);
        }

        g_sum = warp_sum_f32_par(g_sum);
        u_sum = warp_sum_f32_par(u_sum);

        if (lane == 0) {
            const float ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
            if (clamp > 1.0e-6f) {
                if (g_sum > clamp)  g_sum = clamp;
                if (u_sum > clamp)  u_sum = clamp;
                if (u_sum < -clamp) u_sum = -clamp;
            }
            const float sig = 1.0f / (1.0f + expf(-g_sum));
            const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)
                                 * (size_t)n_rows + (size_t)row;
            mid[out_off] = g_sum * sig * u_sum * ew;
        }
        __syncthreads();
    }
}

// M50 staged: dequant-amortization variant of chunked.
//
// Architectural change vs `iq2_xxs_pair_matvec_fused_swiglu_chunked`:
//   - OLD loop nesting: for member { for super_block { dequant + dot } }
//   - NEW loop nesting: for super_block { dequant ONCE, for member { dot } }
//
// The dequant phase reads s_grid / s_sign_pair (data-dependent indices, the
// source of the ~14% LDS bank conflicts diagnosed under rocprofv3). By
// pulling dequant outside the member loop and writing the dequantized int8
// weights into a conflict-free LDS tile, the bank-conflict cost is paid
// 1 / chunk_size as often per produced output. The dot loop becomes plain
// dp4a against staged int8 — no codebook indirection.
//
// LDS tile layout: s_gate_tile[8 rows][260 byte stride].
//   - 256 int8 weights per row + 4 byte pad → row stride = 65 dwords.
//   - Cross-row reads (8 lanes hitting 8 different rows, same dword offset)
//     get bank index (R*65 + K) mod 32 = (R + K) mod 32 → conflict-free.
//   - Within-row reads (32 lanes hitting 32 consecutive dwords of one row)
//     get bank index L mod 32 → conflict-free naturally.
//
// LDS budget per WG (~9.6 KiB total) leaves plenty of headroom. VGPR is
// expected to be the occupancy limiter (matches chunked baseline).
//
// Numerics: reduction order differs from chunked because per-member f32
// partial sums are accumulated across super-blocks BEFORE the warp reduce,
// rather than warp-reducing per (super-block, member). Expect ≤1e-3
// diff vs chunked (consistent with batched-vs-sequential float roundoff).

// Row stride must be 8-byte aligned so int2 (ds_read/write_b64) doesn't fall
// back to unaligned scalar loads. 256 int8 + 8 pad → 264 bytes = 66 dwords.
// Bank math: (66*row + dword_idx) mod 32 = (2*row + dword_idx) mod 32, which
// for the within-row stride-2 access still leaves a 2-way conflict (inherent
// to lane*8 stride), but 64-bit reads halve the LDS-issue instruction count
// to compensate.
#define IQ2_STAGED_TILE_STRIDE 264u
#define IQ2_STAGED_MAX_CHUNK   32u
#define IQ2_STAGED_MAX_ROWS    8u
// staged_v2 S0d full-chunk unroll. 1 = specialize n_members==32 with static
// accumulator indices (157 VGPR / 9 waves); 0 = generic loop (movrel, but
// lower register pressure). A/B'd 2026-06-09.
#ifndef IQ2_V2_UNROLL
#define IQ2_V2_UNROLL 4
#endif
// kwide member-loop unroll depth (pipelines the b128 q8 loads).
#ifndef IQ2_KW_UNROLL
#define IQ2_KW_UNROLL 4
#endif
// kwide full-chunk (n_members==32) static-unroll specialization.
#ifndef IQ2_KW_FULLUNROLL
#define IQ2_KW_FULLUNROLL 0
#endif

extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_chunked_staged(
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

    // === LDS layout ===
    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ int8_t   s_gate_tile[IQ2_STAGED_MAX_ROWS * IQ2_STAGED_TILE_STRIDE];
    __shared__ int8_t   s_up_tile  [IQ2_STAGED_MAX_ROWS * IQ2_STAGED_TILE_STRIDE];
    __shared__ int8_t   s_gate_ls  [IQ2_STAGED_MAX_ROWS * 8];
    __shared__ int8_t   s_up_ls    [IQ2_STAGED_MAX_ROWS * 8];
    // (wd_g / wd_u are kept in per-lane registers, not LDS — every lane in
    //  a row_valid warp already loaded them from the gate_block / up_block
    //  header. Avoids one LDS write + read + implicit sync per super-block.)
    // s_xq_blk_all holds ALL members' activations for the CURRENT super-block.
    // Staged in one cooperative pass per super-block → 1 sync/super-block instead of
    // 1 sync/(super-block, member). Costs 9344 B at chunk_size=32; trivial.
    __shared__ uint8_t  s_xq_blk_all[IQ2_STAGED_MAX_CHUNK * BLOCK_Q8_K_BYTES];
    // Per-member (b, slot) cache. Decoded once at WG init, reused everywhere.
    __shared__ int      s_member_packed[IQ2_STAGED_MAX_CHUNK];
    __shared__ float    s_partial_g[IQ2_STAGED_MAX_CHUNK * IQ2_STAGED_MAX_ROWS];
    __shared__ float    s_partial_u[IQ2_STAGED_MAX_CHUNK * IQ2_STAGED_MAX_ROWS];

    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];

    // (s_partial_g/u are not zero-initialized — the warp-reduce phase
    //  overwrites with `=` after accumulating into per-lane registers.)
    // Decode each member's (b, slot) once.
    for (unsigned int i = tid; i < n_members; i += blockDim.x) {
        s_member_packed[i] = expert_members[(size_t)e * (size_t)max_per_expert
                                          + (size_t)(member_start + i)];
    }
    __syncthreads();

    const unsigned int row_in_block = tid >> 5;            // 0..7 (warp index)
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool row_valid = (row < n_rows);

    // Lane assignment for dequant phase: lane L handles weight chunk
    // [L*8 .. L*8+7] in the warp's row. That maps to sub_idx = L/4,
    // chunk_in_sub = L%4 within the iq2_xxs sub-block layout.
    const unsigned int sub_idx      = lane >> 2;           // 0..7
    const unsigned int chunk_in_sub = lane & 3u;           // 0..3
    const unsigned int g_shift      = chunk_in_sub * 8u;   // codebook index bit offset in aux0
    const unsigned int s_shift      = chunk_in_sub * 7u;   // sign index bit offset in aux1

    const uint8_t* gate_e_base = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;
    const uint8_t* up_e_base   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;

    // Per-lane f32 accumulators, one per member. Accumulated across all
    // super-blocks; warp-reduced ONCE per member at the end (vs the prior
    // version which warp-reduced per (super-block, member) — that's 16×
    // more shfls and was the actual bottleneck).
    float lane_pacc_g[IQ2_STAGED_MAX_CHUNK];
    float lane_pacc_u[IQ2_STAGED_MAX_CHUNK];
    #pragma unroll
    for (unsigned int mi = 0; mi < IQ2_STAGED_MAX_CHUNK; mi++) {
        lane_pacc_g[mi] = 0.0f;
        lane_pacc_u[mi] = 0.0f;
    }

    // === Super-block loop ===
    for (unsigned int sb = 0; sb < n_blocks; sb++) {
        // Per-warp wd registers (lifted out of row_valid block so we can use
        // them after the sync to compute gd/ud directly — avoids one LDS
        // write+read+sync per super-block).
        uint16_t wd_g = 0, wd_u = 0;
        // --- Dequant phase: cooperative, once per (sb, WG) ---
        if (row_valid) {
            const uint8_t* gate_block = gate_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES
                                      + (uint64_t)sb  * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* up_block   = up_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES
                                      + (uint64_t)sb  * BLOCK_IQ2_XXS_BYTES;
            wd_g = *(const uint16_t*)(gate_block + 0);
            wd_u = *(const uint16_t*)(up_block   + 0);

            const uint8_t* sb_g = gate_block + 2 + sub_idx * 8;
            const uint8_t* sb_u = up_block   + 2 + sub_idx * 8;
            const uint32_t aux0_g = *(const uint32_t*)(sb_g + 0);
            const uint32_t aux1_g = *(const uint32_t*)(sb_g + 4);
            const uint32_t aux0_u = *(const uint32_t*)(sb_u + 0);
            const uint32_t aux1_u = *(const uint32_t*)(sb_u + 4);

            const uint8_t g_idx_g = (uint8_t)((aux0_g >> g_shift) & 0xffu);
            const uint32_t s_idx_g = (aux1_g >> s_shift) & 0x7fu;
            const uint8_t g_idx_u = (uint8_t)((aux0_u >> g_shift) & 0xffu);
            const uint32_t s_idx_u = (aux1_u >> s_shift) & 0x7fu;

            // Codebook lookup + sign apply — bank-conflict source, now paid once per (sb, WG).
            const uint64_t grid_g       = s_grid[g_idx_g];
            const uint64_t sign_pair_g  = s_sign_pair[s_idx_g];
            const uint32_t sm0_g        = (uint32_t)sign_pair_g;
            const uint32_t sm1_g        = (uint32_t)(sign_pair_g >> 32);
            const int32_t  g0           = cond_neg_bytes_fast((int32_t)(uint32_t)grid_g,         sm0_g);
            const int32_t  g1           = cond_neg_bytes_fast((int32_t)(uint32_t)(grid_g >> 32), sm1_g);

            const uint64_t grid_u       = s_grid[g_idx_u];
            const uint64_t sign_pair_u  = s_sign_pair[s_idx_u];
            const uint32_t sm0_u        = (uint32_t)sign_pair_u;
            const uint32_t sm1_u        = (uint32_t)(sign_pair_u >> 32);
            const int32_t  u0           = cond_neg_bytes_fast((int32_t)(uint32_t)grid_u,         sm0_u);
            const int32_t  u1           = cond_neg_bytes_fast((int32_t)(uint32_t)(grid_u >> 32), sm1_u);

            // Write 8 int8 weights as int2 → one ds_write_b64 per matrix.
            // Address: row_in_block * 264 (8-byte aligned) + lane * 8 (8-byte aligned).
            int8_t* tg = &s_gate_tile[row_in_block * IQ2_STAGED_TILE_STRIDE + lane * 8u];
            int8_t* tu = &s_up_tile  [row_in_block * IQ2_STAGED_TILE_STRIDE + lane * 8u];
            *(int2*)tg = int2{g0, g1};
            *(int2*)tu = int2{u0, u1};

            if (chunk_in_sub == 0u) {
                const uint32_t ls_g = 2u * (aux1_g >> 28) + 1u;
                const uint32_t ls_u = 2u * (aux1_u >> 28) + 1u;
                s_gate_ls[row_in_block * 8u + sub_idx] = (int8_t)ls_g;
                s_up_ls  [row_in_block * 8u + sub_idx] = (int8_t)ls_u;
            }
            // wd_g / wd_u are now per-lane registers (kept across the row_valid
            // scope via the outer declaration above). No LDS round-trip needed.
        }

        // Cooperative xq stage: ALL 256 threads pull n_members × 292 bytes for THIS
        // super-block into s_xq_blk_all. Per-member outer (no integer division),
        // dword-granular inner. Runs in parallel with the per-warp dequant above.
        {
            // 292 bytes = 73 dwords (292 / 4 == 73 exactly).
            const unsigned int n_dw = (unsigned int)BLOCK_Q8_K_BYTES / 4u;
            for (unsigned int mi = 0; mi < n_members; mi++) {
                const int packed     = s_member_packed[mi];
                const unsigned int b = (unsigned int)(packed >> 16);
                const uint32_t* xq_src_dw = (const uint32_t*)(
                    xq + (size_t)b * (size_t)n_blocks * (size_t)BLOCK_Q8_K_BYTES
                       + (size_t)sb * (size_t)BLOCK_Q8_K_BYTES);
                uint32_t* xq_dst_dw = (uint32_t*)&s_xq_blk_all[mi * BLOCK_Q8_K_BYTES];
                for (unsigned int i = tid; i < n_dw; i += blockDim.x) {
                    xq_dst_dw[i] = xq_src_dw[i];
                }
            }
        }
        __syncthreads();

        const float gd = row_valid ? (0.125f * f16_to_f32_par(wd_g)) : 0.0f;
        const float ud = row_valid ? (0.125f * f16_to_f32_par(wd_u)) : 0.0f;

        const int32_t ls_g_lane = row_valid ? (int32_t)s_gate_ls[row_in_block * 8u + sub_idx] : 0;
        const int32_t ls_u_lane = row_valid ? (int32_t)s_up_ls  [row_in_block * 8u + sub_idx] : 0;

        // Precompute weight dwords (constant across members). int2 = one
        // ds_read_b64 per matrix. Address is 8-byte aligned (264 byte stride).
        int32_t w_g0 = 0, w_g1 = 0, w_u0 = 0, w_u1 = 0;
        if (row_valid) {
            const int8_t* tg = &s_gate_tile[row_in_block * IQ2_STAGED_TILE_STRIDE + lane * 8u];
            const int8_t* tu = &s_up_tile  [row_in_block * IQ2_STAGED_TILE_STRIDE + lane * 8u];
            const int2 wg_pair = *(const int2*)tg;
            const int2 wu_pair = *(const int2*)tu;
            w_g0 = wg_pair.x; w_g1 = wg_pair.y;
            w_u0 = wu_pair.x; w_u1 = wu_pair.y;
        }

        // --- Dot phase: M members against the staged tile ---
        // (No sync inside — all members' xq is in s_xq_blk_all, dequanted weights
        // in s_*_tile. Each warp reads its own row's tile + per-member q8 slices.)
        if (row_valid) {
            for (unsigned int mi = 0; mi < n_members; mi++) {
                const uint8_t* xq_m = &s_xq_blk_all[mi * BLOCK_Q8_K_BYTES];
                const float yd = *(const float*)&xq_m[0];
                const int8_t* q8_base = (const int8_t*)&xq_m[4];
                // q8_base is at xq_m + 4 (4-byte aligned, not 8-byte) — but
                // (q8_base + lane*8) = xq_m + 4 + lane*8. xq_m is 292-byte
                // aligned (well, BLOCK_Q8_K_BYTES * mi from a 8-aligned LDS
                // base; 292 = 4 mod 8). To be safe, use int2 read of the
                // 8-byte chunk starting at lane*8 from a sub-region we know
                // is 8-aligned: shift to read xq_m + 8 + (lane*8 - 4).
                // Simpler: keep two int32 reads here — q8 is only 2 reads
                // per (sb,m), not the dominant LDS cost.
                const int32_t q8_0 = *(const int32_t*)(q8_base + lane * 8u + 0);
                const int32_t q8_1 = *(const int32_t*)(q8_base + lane * 8u + 4);

                int32_t sumi_g = 0;
                sumi_g = sudot4_par(w_g0, q8_0, sumi_g);
                sumi_g = sudot4_par(w_g1, q8_1, sumi_g);
                int32_t sumi_u = 0;
                sumi_u = sudot4_par(w_u0, q8_0, sumi_u);
                sumi_u = sudot4_par(w_u1, q8_1, sumi_u);

                // Fold per-sub-block ls (sumi ≤ 127×127×8 ≈ 1M, ls ≤ 31 → fits int32).
                // Accumulate per-lane; warp-reduce only once per member at the very end.
                lane_pacc_g[mi] += (float)(sumi_g * ls_g_lane) * yd * gd;
                lane_pacc_u[mi] += (float)(sumi_u * ls_u_lane) * yd * ud;
            }
        }
        __syncthreads();  // protect tile + s_xq_blk_all from next super-block's overwrite
    }

    // Warp-reduce each member's accumulator ONCE here (instead of n_blocks times
    // per member as in the prior version). Lane 0 stages into s_partial_g/u so
    // the final swiglu loop has uniform LDS-based access.
    if (row_valid) {
        for (unsigned int mi = 0; mi < n_members; mi++) {
            const float wg = warp_sum_f32_par(lane_pacc_g[mi]);
            const float wu = warp_sum_f32_par(lane_pacc_u[mi]);
            if (lane == 0u) {
                s_partial_g[mi * IQ2_STAGED_MAX_ROWS + row_in_block] = wg;
                s_partial_u[mi * IQ2_STAGED_MAX_ROWS + row_in_block] = wu;
            }
        }
    }

    // === Final swiglu + output, one per (member, row) ===
    if (row_valid && lane == 0u) {
        for (unsigned int mi = 0; mi < n_members; mi++) {
            const int packed          = s_member_packed[mi];
            const unsigned int b      = (unsigned int)(packed >> 16);
            const unsigned int slot   = (unsigned int)(packed & 0xFFFFu);

            float g_sum = s_partial_g[mi * IQ2_STAGED_MAX_ROWS + row_in_block];
            float u_sum = s_partial_u[mi * IQ2_STAGED_MAX_ROWS + row_in_block];

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

// =============================================================================
// staged_v2 — M51 S0: same geometry/semantics as _chunked_staged, minus the
// per-member issue overhead found in the 2026-06-09 ISA audit:
//
//   S0a  scale chain hoisted: gds = gd * (float)ls is member-invariant, so the
//        member tail is (float)sumi * (yd * gds) — no v_mul_lo_u32, no
//        separate per-matrix imul+cvt+mul+mul chain.
//   S0b  no LDS weight tile: the staged kernel's dot phase already hoists the
//        dequanted octets to registers; the s_*_tile/s_*_ls arrays only cost
//        8.5 KiB LDS (occupancy 12 waves/SIMD was LDS-bound). Each lane keeps
//        its octets + ls in registers across the barrier instead.
//   S0c  xq restaged as s_yd[32] + 8-aligned s_q8[32][256]: dot-phase q8 load
//        is one ds_load_b64 (was 2× b32 off the +4-misaligned 292-B layout),
//        and we stop copying the 32-byte Q2_K-only bsums tail.
//   S0d  full-chunk specialization: n_members == 32 takes a #pragma-unrolled
//        member loop with STATIC lane_pacc indices. The generic loop's dynamic
//        indices compile to v_movrels/movreld + m0 updates — 4 indexed register
//        moves per member, as expensive as the 4 dot4s themselves.
//
// LDS: 3072 LUT + 8192 q8 + 128 yd + 128 member + 2048 partials ≈ 13.6 KiB
// (was 18.9 KiB).
// =============================================================================

extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_chunked_staged_v2(
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

    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    // 8-byte aligned q8 (uint64_t base), 32 slices of 256 B. yd separate.
    __shared__ uint64_t s_q8_u64[IQ2_STAGED_MAX_CHUNK * 32];   // 32 × 256 B
    __shared__ float    s_yd[IQ2_STAGED_MAX_CHUNK];
    __shared__ int      s_member_packed[IQ2_STAGED_MAX_CHUNK];
    __shared__ float    s_partial_g[IQ2_STAGED_MAX_CHUNK * IQ2_STAGED_MAX_ROWS];
    __shared__ float    s_partial_u[IQ2_STAGED_MAX_CHUNK * IQ2_STAGED_MAX_ROWS];

    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];

    for (unsigned int i = tid; i < n_members; i += blockDim.x) {
        s_member_packed[i] = expert_members[(size_t)e * (size_t)max_per_expert
                                          + (size_t)(member_start + i)];
    }
    __syncthreads();

    const unsigned int row_in_block = tid >> 5;            // 0..7 (warp index)
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool row_valid = (row < n_rows);

    const unsigned int sub_idx      = lane >> 2;           // 0..7
    const unsigned int chunk_in_sub = lane & 3u;           // 0..3
    const unsigned int g_shift      = chunk_in_sub * 8u;
    const unsigned int s_shift      = chunk_in_sub * 7u;

    const uint8_t* gate_e_base = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;
    const uint8_t* up_e_base   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;

    float lane_pacc_g[IQ2_STAGED_MAX_CHUNK];
    float lane_pacc_u[IQ2_STAGED_MAX_CHUNK];
    #pragma unroll
    for (unsigned int mi = 0; mi < IQ2_STAGED_MAX_CHUNK; mi++) {
        lane_pacc_g[mi] = 0.0f;
        lane_pacc_u[mi] = 0.0f;
    }

    for (unsigned int sb = 0; sb < n_blocks; sb++) {
        // --- Dequant phase: octets + hoisted scales stay in REGISTERS ---
        int32_t w_g0 = 0, w_g1 = 0, w_u0 = 0, w_u1 = 0;
        float gds = 0.0f, uds = 0.0f;
        if (row_valid) {
            const uint8_t* gate_block = gate_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES
                                      + (uint64_t)sb  * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* up_block   = up_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES
                                      + (uint64_t)sb  * BLOCK_IQ2_XXS_BYTES;
            const uint16_t wd_g = *(const uint16_t*)(gate_block + 0);
            const uint16_t wd_u = *(const uint16_t*)(up_block   + 0);

            const uint8_t* sb_g = gate_block + 2 + sub_idx * 8;
            const uint8_t* sb_u = up_block   + 2 + sub_idx * 8;
            const uint32_t aux0_g = *(const uint32_t*)(sb_g + 0);
            const uint32_t aux1_g = *(const uint32_t*)(sb_g + 4);
            const uint32_t aux0_u = *(const uint32_t*)(sb_u + 0);
            const uint32_t aux1_u = *(const uint32_t*)(sb_u + 4);

            const uint8_t g_idx_g = (uint8_t)((aux0_g >> g_shift) & 0xffu);
            const uint32_t s_idx_g = (aux1_g >> s_shift) & 0x7fu;
            const uint8_t g_idx_u = (uint8_t)((aux0_u >> g_shift) & 0xffu);
            const uint32_t s_idx_u = (aux1_u >> s_shift) & 0x7fu;

            const uint64_t grid_g       = s_grid[g_idx_g];
            const uint64_t sign_pair_g  = s_sign_pair[s_idx_g];
            w_g0 = cond_neg_bytes_fast((int32_t)(uint32_t)grid_g,         (uint32_t)sign_pair_g);
            w_g1 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid_g >> 32), (uint32_t)(sign_pair_g >> 32));

            const uint64_t grid_u       = s_grid[g_idx_u];
            const uint64_t sign_pair_u  = s_sign_pair[s_idx_u];
            w_u0 = cond_neg_bytes_fast((int32_t)(uint32_t)grid_u,         (uint32_t)sign_pair_u);
            w_u1 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid_u >> 32), (uint32_t)(sign_pair_u >> 32));

            // S0a: every lane already holds aux1 — compute its own ls and fold
            // it into the member-invariant f32 scale here. (float)ls ≤ 31 is
            // exact; sumi ≤ 8·127·31 ≪ 2^24 so (float)sumi is exact too.
            const int32_t ls_g = (int32_t)(2u * (aux1_g >> 28) + 1u);
            const int32_t ls_u = (int32_t)(2u * (aux1_u >> 28) + 1u);
            gds = (0.125f * f16_to_f32_par(wd_g)) * (float)ls_g;
            uds = (0.125f * f16_to_f32_par(wd_u)) * (float)ls_u;
        }

        // --- Cooperative xq stage: 65 dwords per member (yd + 64 q8 dwords),
        //     q8 landing 8-byte aligned in s_q8_u64 ---
        {
            uint32_t* s_q8_dw = (uint32_t*)s_q8_u64;
            for (unsigned int mi = 0; mi < n_members; mi++) {
                const int packed     = s_member_packed[mi];
                const unsigned int b = (unsigned int)(packed >> 16);
                const uint32_t* xq_src_dw = (const uint32_t*)(
                    xq + (size_t)b * (size_t)n_blocks * (size_t)BLOCK_Q8_K_BYTES
                       + (size_t)sb * (size_t)BLOCK_Q8_K_BYTES);
                if (tid == 0u) s_yd[mi] = *(const float*)xq_src_dw;
                for (unsigned int i = tid; i < 64u; i += blockDim.x) {
                    s_q8_dw[mi * 64u + i] = xq_src_dw[1u + i];
                }
            }
        }
        __syncthreads();

        // --- Dot phase ---
        if (row_valid) {
            const uint64_t* q8_lane = &s_q8_u64[lane];   // member mi at +mi*32

            // S0d: full chunks take the unrolled loop (static lane_pacc
            // indices — no v_movrel); tail chunks take the generic loop.
            #define IQ2_V2_DOT_BODY(mi)                                          \
                {                                                                \
                    const uint64_t q8pair = q8_lane[(mi) * 32u];                 \
                    const int32_t q8_0 = (int32_t)(uint32_t)q8pair;              \
                    const int32_t q8_1 = (int32_t)(uint32_t)(q8pair >> 32);      \
                    int32_t sumi_g = 0;                                          \
                    sumi_g = sudot4_par(w_g0, q8_0, sumi_g);                     \
                    sumi_g = sudot4_par(w_g1, q8_1, sumi_g);                     \
                    int32_t sumi_u = 0;                                          \
                    sumi_u = sudot4_par(w_u0, q8_0, sumi_u);                     \
                    sumi_u = sudot4_par(w_u1, q8_1, sumi_u);                     \
                    const float yd = s_yd[(mi)];                                 \
                    lane_pacc_g[(mi)] += (float)sumi_g * (yd * gds);             \
                    lane_pacc_u[(mi)] += (float)sumi_u * (yd * uds);             \
                }
#if IQ2_V2_UNROLL == 1
            // Full static unroll: no movrel, but 157 VGPR → 9 waves/SIMD.
            if (n_members == IQ2_STAGED_MAX_CHUNK) {
                #pragma unroll
                for (unsigned int mi = 0; mi < IQ2_STAGED_MAX_CHUNK; mi++) {
                    IQ2_V2_DOT_BODY(mi);
                }
            } else {
                for (unsigned int mi = 0; mi < n_members; mi++) {
                    IQ2_V2_DOT_BODY(mi);
                }
            }
#elif IQ2_V2_UNROLL > 1
            // Partial unroll: movrel stays, but the compiler software-
            // pipelines the q8/yd LDS loads across body copies (the original
            // staged kernel compiled to unroll-4 + lgkmcnt(1) — that
            // pipelining, not the instruction count, dominated).
            #pragma unroll IQ2_V2_UNROLL
            for (unsigned int mi = 0; mi < n_members; mi++) {
                IQ2_V2_DOT_BODY(mi);
            }
#else
            // No unroll: every iteration stalls on lgkmcnt(0) twice. SLOW.
            for (unsigned int mi = 0; mi < n_members; mi++) {
                IQ2_V2_DOT_BODY(mi);
            }
#endif
            #undef IQ2_V2_DOT_BODY
        }
        __syncthreads();  // protect s_q8/s_yd from next super-block's overwrite
    }

    if (row_valid) {
        for (unsigned int mi = 0; mi < n_members; mi++) {
            const float wg = warp_sum_f32_par(lane_pacc_g[mi]);
            const float wu = warp_sum_f32_par(lane_pacc_u[mi]);
            if (lane == 0u) {
                s_partial_g[mi * IQ2_STAGED_MAX_ROWS + row_in_block] = wg;
                s_partial_u[mi * IQ2_STAGED_MAX_ROWS + row_in_block] = wu;
            }
        }
    }

    if (row_valid && lane == 0u) {
        for (unsigned int mi = 0; mi < n_members; mi++) {
            const int packed          = s_member_packed[mi];
            const unsigned int b      = (unsigned int)(packed >> 16);
            const unsigned int slot   = (unsigned int)(packed & 0xFFFFu);

            float g_sum = s_partial_g[mi * IQ2_STAGED_MAX_ROWS + row_in_block];
            float u_sum = s_partial_u[mi * IQ2_STAGED_MAX_ROWS + row_in_block];

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

// =============================================================================
// kwide — M51 S1: k-widened lanes. Same grid/WG geometry as _chunked_staged
// (8 warps × 1 row, grid (n_rows/8, n_work_items)); inside the warp, lanes
// split a PAIR of super-blocks along k instead of one:
//
//   sbh = lane>>4 ∈ {0,1}   which super-block of the pair
//   l15 = lane&15           which 16-weight half-sub-block within it
//   (sub_block = l15>>1, half = l15&1 — 2 lanes share each 32-weight
//    sub-block and its ls)
//
// Per (member, lane): ONE ds_load_b128 (16 B q8) feeds 8 v_dot4 (4 gate +
// 4 up) + cvt/mul/fma tail. The S0 experiments showed the staged kernel is
// LDS-round-trip-latency bound (not VALU-issue bound): this shape quarters
// the LDS round-trips per dot4 (b128:8dot4 vs b64+b32:4dot4) and halves the
// scale-chain ops per dot4, at UNCHANGED occupancy:
//   LDS = 1024 sign + 2048 grid + 16384 q8 + 256 yd + 128 member = 19840 B
//   → 6 WGs/WGP = 12 waves/SIMD (same as staged), no s_partial (swiglu is
//   folded into the post-reduce lane-0 loop).
//
// s_q8 is member-major [mi][sbh][16×uint4]: the wave's 32 lanes then read 32
// CONSECUTIVE uint4 (512 B) — bank-conflict-free. The [sbh][mi] layout would
// put the two half-waves 8 KiB apart → same banks → 2-way conflict on every
// phase of the b128.
//
// Numerics: integer sub-block sums are identical to staged (same octet
// grouping); f32 accumulation differs only in the member-invariant scale
// association ((float)sumi * (yd*gds) vs ((float)(sumi*ls)) * yd * gd) and
// the lane-partition of k — within the usual 5e-3 rel oracle tolerance.
// =============================================================================

extern "C" __global__
#if IQ2_KW_FULLUNROLL
// Cap VGPRs so the full-unroll's load pipelining can't blow occupancy
// (uncapped: 168 VGPR → 9 waves → +41%). 12 waves/SIMD needs ≤128 VGPR.
__launch_bounds__(256, 12)
#endif
void iq2_xxs_pair_matvec_fused_swiglu_kwide(
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

    __shared__ uint64_t s_sign_pair[128];
    __shared__ uint64_t s_grid[256];
    __shared__ uint4    s_q8v[IQ2_STAGED_MAX_CHUNK * 2 * 16]; // [mi][sbh][16]
    __shared__ float    s_yd[IQ2_STAGED_MAX_CHUNK * 2];       // [mi][sbh]
    __shared__ int      s_member_packed[IQ2_STAGED_MAX_CHUNK];

    const unsigned int tid = threadIdx.x;
    if (tid < 128) {
        const uint32_t s = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0 = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1 = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];

    for (unsigned int i = tid; i < n_members; i += blockDim.x) {
        s_member_packed[i] = expert_members[(size_t)e * (size_t)max_per_expert
                                          + (size_t)(member_start + i)];
    }
    __syncthreads();

    const unsigned int row_in_block = tid >> 5;            // 0..7 (warp index)
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool row_valid = (row < n_rows);

    const unsigned int sbh       = lane >> 4;              // 0..1: sb of pair
    const unsigned int l15       = lane & 15u;
    const unsigned int sub_block = l15 >> 1;               // 0..7
    const unsigned int half      = l15 & 1u;               // 0..1
    const unsigned int o0_shift8 = (2u * half) * 8u;       // first octet, aux0
    const unsigned int o0_shift7 = (2u * half) * 7u;       // first octet, aux1

    const uint8_t* gate_e_base = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;
    const uint8_t* up_e_base   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;

    float lane_pacc_g[IQ2_STAGED_MAX_CHUNK];
    float lane_pacc_u[IQ2_STAGED_MAX_CHUNK];
    #pragma unroll
    for (unsigned int mi = 0; mi < IQ2_STAGED_MAX_CHUNK; mi++) {
        lane_pacc_g[mi] = 0.0f;
        lane_pacc_u[mi] = 0.0f;
    }

    const unsigned int n_pairs = n_blocks >> 1;   // wrapper asserts n_blocks even
    for (unsigned int sbp = 0; sbp < n_pairs; sbp++) {
        const unsigned int sb_lane = 2u * sbp + sbh;       // this lane's sb

        // --- Dequant: this lane's 16 weights × 2 matrices → 8 dwords in regs ---
        int32_t wg0 = 0, wg1 = 0, wg2 = 0, wg3 = 0;
        int32_t wu0 = 0, wu1 = 0, wu2 = 0, wu3 = 0;
        float gds = 0.0f, uds = 0.0f;
        if (row_valid) {
            const uint8_t* gate_block = gate_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES
                                      + (uint64_t)sb_lane * BLOCK_IQ2_XXS_BYTES;
            const uint8_t* up_block   = up_e_base
                                      + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES
                                      + (uint64_t)sb_lane * BLOCK_IQ2_XXS_BYTES;
            const uint16_t wd_g = *(const uint16_t*)(gate_block + 0);
            const uint16_t wd_u = *(const uint16_t*)(up_block   + 0);

            const uint8_t* sb_g = gate_block + 2 + sub_block * 8;
            const uint8_t* sb_u = up_block   + 2 + sub_block * 8;
            const uint32_t aux0_g = *(const uint32_t*)(sb_g + 0);
            const uint32_t aux1_g = *(const uint32_t*)(sb_g + 4);
            const uint32_t aux0_u = *(const uint32_t*)(sb_u + 0);
            const uint32_t aux1_u = *(const uint32_t*)(sb_u + 4);

            // Two octets per matrix: o0 = 2*half, o1 = 2*half+1.
            {
                const uint8_t  gi0 = (uint8_t)((aux0_g >> o0_shift8) & 0xffu);
                const uint32_t si0 = (aux1_g >> o0_shift7) & 0x7fu;
                const uint8_t  gi1 = (uint8_t)((aux0_g >> (o0_shift8 + 8u)) & 0xffu);
                const uint32_t si1 = (aux1_g >> (o0_shift7 + 7u)) & 0x7fu;
                const uint64_t grid0 = s_grid[gi0];
                const uint64_t sp0   = s_sign_pair[si0];
                wg0 = cond_neg_bytes_fast((int32_t)(uint32_t)grid0,         (uint32_t)sp0);
                wg1 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid0 >> 32), (uint32_t)(sp0 >> 32));
                const uint64_t grid1 = s_grid[gi1];
                const uint64_t sp1   = s_sign_pair[si1];
                wg2 = cond_neg_bytes_fast((int32_t)(uint32_t)grid1,         (uint32_t)sp1);
                wg3 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid1 >> 32), (uint32_t)(sp1 >> 32));
            }
            {
                const uint8_t  gi0 = (uint8_t)((aux0_u >> o0_shift8) & 0xffu);
                const uint32_t si0 = (aux1_u >> o0_shift7) & 0x7fu;
                const uint8_t  gi1 = (uint8_t)((aux0_u >> (o0_shift8 + 8u)) & 0xffu);
                const uint32_t si1 = (aux1_u >> (o0_shift7 + 7u)) & 0x7fu;
                const uint64_t grid0 = s_grid[gi0];
                const uint64_t sp0   = s_sign_pair[si0];
                wu0 = cond_neg_bytes_fast((int32_t)(uint32_t)grid0,         (uint32_t)sp0);
                wu1 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid0 >> 32), (uint32_t)(sp0 >> 32));
                const uint64_t grid1 = s_grid[gi1];
                const uint64_t sp1   = s_sign_pair[si1];
                wu2 = cond_neg_bytes_fast((int32_t)(uint32_t)grid1,         (uint32_t)sp1);
                wu3 = cond_neg_bytes_fast((int32_t)(uint32_t)(grid1 >> 32), (uint32_t)(sp1 >> 32));
            }

            const int32_t ls_g = (int32_t)(2u * (aux1_g >> 28) + 1u);
            const int32_t ls_u = (int32_t)(2u * (aux1_u >> 28) + 1u);
            gds = (0.125f * f16_to_f32_par(wd_g)) * (float)ls_g;
            uds = (0.125f * f16_to_f32_par(wd_u)) * (float)ls_u;
        }

        // --- Cooperative xq stage (M53 rewrite): branch-free, all members
        //     concurrent. The old per-member loop compiled to a vmcnt(0)-
        //     serialized branchy copy (one exposed global-load latency per
        //     dword, members sequential, 130/256 threads active). Here
        //     thread t owns member t>>3, dword lanes (t&7)+8j of the
        //     member's 2×64 q8 dwords — 16 independent loads issue before
        //     any store waits, every thread busy. yd via the first
        //     2·n_members threads. 292-B source block = 73 dwords; dword 0
        //     is yd, 1..64 are q8. ---
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

            #define IQ2_KW_DOT_BODY(mi)                                          \
                {                                                                \
                    const uint4 q = q8_lane[(mi) * 32u];                         \
                    int32_t sumi_g = 0;                                          \
                    sumi_g = sudot4_par(wg0, (int32_t)q.x, sumi_g);              \
                    sumi_g = sudot4_par(wg1, (int32_t)q.y, sumi_g);              \
                    sumi_g = sudot4_par(wg2, (int32_t)q.z, sumi_g);              \
                    sumi_g = sudot4_par(wg3, (int32_t)q.w, sumi_g);              \
                    int32_t sumi_u = 0;                                          \
                    sumi_u = sudot4_par(wu0, (int32_t)q.x, sumi_u);              \
                    sumi_u = sudot4_par(wu1, (int32_t)q.y, sumi_u);              \
                    sumi_u = sudot4_par(wu2, (int32_t)q.z, sumi_u);              \
                    sumi_u = sudot4_par(wu3, (int32_t)q.w, sumi_u);              \
                    const float yd = yd_lane[(mi) * 2u];                         \
                    lane_pacc_g[(mi)] += (float)sumi_g * (yd * gds);             \
                    lane_pacc_u[(mi)] += (float)sumi_u * (yd * uds);             \
                }
#if IQ2_KW_FULLUNROLL
            // Full-chunk specialization: static pacc indices (no movrel, no
            // per-iter address math). kwide is VALU-issue-bound, so the
            // occupancy this costs may matter less than it did for staged.
            if (n_members == IQ2_STAGED_MAX_CHUNK) {
                #pragma unroll
                for (unsigned int mi = 0; mi < IQ2_STAGED_MAX_CHUNK; mi++) {
                    IQ2_KW_DOT_BODY(mi);
                }
            } else {
                #pragma unroll IQ2_KW_UNROLL
                for (unsigned int mi = 0; mi < n_members; mi++) {
                    IQ2_KW_DOT_BODY(mi);
                }
            }
#else
            #pragma unroll IQ2_KW_UNROLL
            for (unsigned int mi = 0; mi < n_members; mi++) {
                IQ2_KW_DOT_BODY(mi);
            }
#endif
            #undef IQ2_KW_DOT_BODY
        }
        __syncthreads();  // protect s_q8v/s_yd from next pair's overwrite
    }

    // --- Reduce + swiglu + output, fused (no s_partial round-trip) ---
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
// tile8_row32 — port of ejpir/ds4-hip's moe_gate_up_mid_expert_tile8_row32
// =============================================================================
//
// Different WG geometry vs `_chunked_staged`:
//   - chunked_staged: 8 rows × 32 lanes; lane L decodes ONE octet/sb to LDS,
//     32 lanes split one super-block 32 ways; warp_sum (32-wide) per row.
//   - tile8_row32:    32 rows × 8 lanes; lane L processes super-blocks
//     {L, L+8, L+16, ...} ENTIRELY in registers (full block8 dequant);
//     quarter_warp_sum (8-wide) per row.
//
// Trade: tile8 amortizes the 8-octet dequant work into ~xq_blocks/8 super-
// blocks per (row, lane) — no LDS round-trip for weights. LDS pressure
// drops from ~9.6 KiB to ~2.1 KiB (just the iq2 LUTs), which opens
// occupancy room. Per-row reduce drops to 3 shfls (8→1) instead of 5.
//
// Constraints:
//   - chunk_size MUST be ≤ 8  (caller asserts; see Iq2XxsPairMatvec::launch_fused_swiglu_tile8)
//   - n_rows MUST be a multiple of 32 (asserted in wrapper; 2048 is fine for V4-Flash N_FF_EXP)
//
// Numerics: bsum integer order matches dev_dot_iq2_xxs_q8_K_block_lut; f32
// reduction order differs vs chunked_staged (per-member acc across
// stride-8 super-blocks + 8-wide tree, vs warp-tree across 32 lanes). Same
// shape of accumulation order ejpir's reference uses, so behaviour matches
// the upstream fork.

#define IQ2_TILE8_MAX_MEMBERS 8u

// Quarter-warp 8-lane reduction. Mirrors ejpir's quarter_warp_sum_f32
// (rocm/ds4_rocm_moe.cuh:474). RDNA __shfl_down_sync wraps lanes within
// the warp; since each "row" (8 consecutive lanes within the wave) is
// already aligned to multiples of 8, the 4 shfl steps stay within the row.
__device__ __forceinline__ float quarter_warp_sum_f32(float v) {
    v += __shfl_down_sync(FULL_WARP_MASK, v, 4);
    v += __shfl_down_sync(FULL_WARP_MASK, v, 2);
    v += __shfl_down_sync(FULL_WARP_MASK, v, 1);
    return v;
}

// Block8 deq-LUT helper — direct port of ejpir's
// dev_dot_iq2_xxs_q8_K_block8_deq_lut. Reads one iq2_xxs super-block,
// unpacks 8 octets to register weights w[0..7] (4 grid lookups × 2 i32
// each), then dp4a's into n_members q8_K xq slices.
//
// `n_members` MUST be ≤ 8. `q8_ptrs[m]` may be NULL for unused slots
// (we still call dp4a; harmless because acc[m] is unused later).
__device__ __forceinline__ void block8_dot_lut(
        const uint8_t* iq2_block,                  // 66 bytes
        const int8_t* const q8_ptrs[8],            // [n_members] each pointing to 256 int8
        unsigned int n_members,
        float xd,                                  // f16_to_f32(iq2_block->d)
        const float yd[8],                         // per-member y->d
        const uint64_t* __restrict__ s_grid,
        const uint64_t* __restrict__ s_sign_pair,
        float acc[8]) {
    int32_t bsum[8] = {0,0,0,0,0,0,0,0};
    const uint16_t* q2 = (const uint16_t*)(iq2_block + 2);
    #pragma unroll
    for (int ib32 = 0; ib32 < 8; ib32++) {  // QK_K / 32 = 8 sub-blocks
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);

        int32_t w[8];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint8_t  g_idx = (uint8_t)((aux0 >> (k * 8))  & 0xffu);
            const uint32_t s_idx = (aux1 >> (k * 7)) & 0x7fu;
            const uint64_t g     = s_grid[g_idx];
            const uint64_t sp    = s_sign_pair[s_idx];
            const uint32_t sm0   = (uint32_t)sp;
            const uint32_t sm1   = (uint32_t)(sp >> 32);
            w[k * 2 + 0] = cond_neg_bytes_fast((int32_t)(uint32_t)g,         sm0);
            w[k * 2 + 1] = cond_neg_bytes_fast((int32_t)(uint32_t)(g >> 32), sm1);
        }

        #pragma unroll
        for (unsigned int p = 0; p < IQ2_TILE8_MAX_MEMBERS; p++) {
            if (p >= n_members) break;
            const int8_t* q = q8_ptrs[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = sudot4_par(w[0], *(const int32_t*)(q + 0),  sumi);
            sumi = sudot4_par(w[1], *(const int32_t*)(q + 4),  sumi);
            sumi = sudot4_par(w[2], *(const int32_t*)(q + 8),  sumi);
            sumi = sudot4_par(w[3], *(const int32_t*)(q + 12), sumi);
            sumi = sudot4_par(w[4], *(const int32_t*)(q + 16), sumi);
            sumi = sudot4_par(w[5], *(const int32_t*)(q + 20), sumi);
            sumi = sudot4_par(w[6], *(const int32_t*)(q + 24), sumi);
            sumi = sudot4_par(w[7], *(const int32_t*)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    #pragma unroll
    for (unsigned int p = 0; p < IQ2_TILE8_MAX_MEMBERS; p++) {
        if (p >= n_members) break;
        acc[p] += 0.125f * xd * yd[p] * (float)bsum[p];
    }
}

extern "C" __global__ void iq2_xxs_pair_matvec_fused_swiglu_tile8_row32(
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
        unsigned int    chunk_size,                // ≤ 8
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    const unsigned int w_idx = blockIdx.y;
    const int packed_item   = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;
    const unsigned int n_members = member_end - member_start;  // ≤ 8

    __shared__ uint64_t s_grid[256];
    __shared__ uint64_t s_sign_pair[128];

    const unsigned int tid = threadIdx.x;

    // Per-member (b, slot) decoded once. Kept in regs by replicating across all threads.
    int pkt[IQ2_TILE8_MAX_MEMBERS] = {0,0,0,0,0,0,0,0};
    #pragma unroll
    for (unsigned int p = 0; p < IQ2_TILE8_MAX_MEMBERS; p++) {
        if (p < n_members) {
            pkt[p] = expert_members[(size_t)e * (size_t)max_per_expert
                                  + (size_t)(member_start + p)];
        }
    }

    // Load LUTs.
    if (tid < 128) {
        const uint32_t s     = ksigns_iq2xs[tid];
        const uint32_t signs = s * 0x01010101u;
        const uint32_t sm0   = expand_signs_par(signs, 0x08040201u);
        const uint32_t sm1   = expand_signs_par(signs, 0x80402010u);
        s_sign_pair[tid] = ((uint64_t)sm1 << 32) | (uint64_t)sm0;
    }
    if (tid < 256) s_grid[tid] = iq2xxs_grid[tid];
    __syncthreads();

    const unsigned int lane         = tid & 7u;             // 0..7  (block-stride lane)
    const unsigned int row_in_block = tid >> 3;             // 0..31 (row within WG)
    const unsigned int row          = blockIdx.x * 32u + row_in_block;
    const bool row_valid = (row < n_rows);

    if (!row_valid) return;

    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * (uint64_t)n_blocks * BLOCK_IQ2_XXS_BYTES;

    // Per-member xq base pointers (start of super-block 0 — INCLUDES d header).
    const uint8_t* xq_base[IQ2_TILE8_MAX_MEMBERS];
    #pragma unroll
    for (unsigned int p = 0; p < IQ2_TILE8_MAX_MEMBERS; p++) {
        xq_base[p] = nullptr;
        if (p < n_members) {
            const unsigned int b = (unsigned int)(pkt[p] >> 16);
            xq_base[p] = xq + (size_t)b * (size_t)n_blocks * (size_t)BLOCK_Q8_K_BYTES;
        }
    }

    float gate_acc[IQ2_TILE8_MAX_MEMBERS] = {0,0,0,0,0,0,0,0};
    float up_acc  [IQ2_TILE8_MAX_MEMBERS] = {0,0,0,0,0,0,0,0};

    // Each lane (0..7) processes super-blocks {lane, lane+8, ...}. Per super-block,
    // each q8_K block has its OWN d scale at byte 0 — read it fresh per iteration
    // (caching once outside the loop was a bug: real xq has varying d per super-block).
    for (unsigned int b = lane; b < n_blocks; b += 8u) {
        const uint8_t* gate_blk = gate_row_base + (uint64_t)b * BLOCK_IQ2_XXS_BYTES;
        const uint8_t* up_blk   = up_row_base   + (uint64_t)b * BLOCK_IQ2_XXS_BYTES;
        const float gd = f16_to_f32_par(*(const uint16_t*)gate_blk);
        const float ud = f16_to_f32_par(*(const uint16_t*)up_blk);

        // Per-member q8 super-block start + its d. Same super-block index b
        // across members (we're matvec'ing the row against all member activations).
        const int8_t* q8_sb[IQ2_TILE8_MAX_MEMBERS];
        float         yd_sb[IQ2_TILE8_MAX_MEMBERS] = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (unsigned int p = 0; p < IQ2_TILE8_MAX_MEMBERS; p++) {
            if (p < n_members) {
                const uint8_t* xq_sb = xq_base[p] + (size_t)b * (size_t)BLOCK_Q8_K_BYTES;
                yd_sb[p] = *(const float*)xq_sb;
                q8_sb[p] = (const int8_t*)(xq_sb + 4);
            } else {
                q8_sb[p] = nullptr;
            }
        }

        block8_dot_lut(gate_blk, q8_sb, n_members, gd, yd_sb, s_grid, s_sign_pair, gate_acc);
        block8_dot_lut(up_blk,   q8_sb, n_members, ud, yd_sb, s_grid, s_sign_pair, up_acc);
    }

    // Quarter-warp reduce across the 8 lanes of each row.
    #pragma unroll
    for (unsigned int p = 0; p < IQ2_TILE8_MAX_MEMBERS; p++) {
        if (p >= n_members) break;
        gate_acc[p] = quarter_warp_sum_f32(gate_acc[p]);
        up_acc  [p] = quarter_warp_sum_f32(up_acc  [p]);
    }

    if (lane == 0u) {
        #pragma unroll
        for (unsigned int p = 0; p < IQ2_TILE8_MAX_MEMBERS; p++) {
            if (p >= n_members) break;
            const unsigned int b    = (unsigned int)(pkt[p] >> 16);
            const unsigned int slot = (unsigned int)(pkt[p] & 0xFFFFu);
            float g_sum = gate_acc[p];
            float u_sum = up_acc[p];
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
