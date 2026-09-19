// mxfp4_pair_matvec — gate+up fused-SwiGLU kernels for MXFP4 weights.
//
// DeepSeek-V4.1-Flash ships its routed experts natively as MXFP4 (llama.cpp
// block_mxfp4, 17 B per 32 elements: 1 B e8m0 scale + 16 B packed fp4
// nibbles; elems 0..15 = LOW nibbles of qs[0..16], elems 16..31 = HIGH
// nibbles; kvalues_mxfp4 are the E2M1 values doubled so the scale is
// 2^(e-128), see mxfp4_matvec.hip). V4-Flash only ever had MXFP4 *down*
// projections (blk.26/42 of the UD mixes), so gate/up need their own pair
// family. Contracts are identical to the iq3_s / iq2_s pair kernels
// (`mid[slot, row] = swiglu(gate·x, up·x) * expert_w[slot]`, Q8_K
// activations, 8 rows per 256-thread WG, 32 lanes per row, warp-reduced),
// so they drop into `het::dispatch::moe_gate_up_batch{,_hetsplit}`.
//
// Roofline (decode, one token, per layer): 6 experts × 2304 rows × 2 mats
// × 20 superblocks × 136 B = 75 MB of weight reads per layer at 5120-wide
// V4.1 — purely bandwidth-bound; ~0.4 ms/layer at the iGPU's ~200 GB/s, so
// the whole 40-layer gate/up leg is ~15 ms of a decode token. The kernel
// reads each weight byte exactly once; nibble decode is a 16-entry LDS LUT
// + sudot4 (8 sudot4 per 32 weights), the same VALU budget as the down
// kernel's quarter split.
//
// Lane split: 32 lanes per row = 16 superblock lanes × 2 halves. A half
// covers 4 of the superblock's 8 mxfp4 blocks (4 × (4 lo + 4 hi) = 32
// sudot4 per mat per superblock-half). Superblocks are strided by 16 across
// lanes; a 5120-wide row (20 superblocks) gives lanes 0..7 two superblocks
// and lanes 8..15 one — a 25% lane imbalance that the V4.1 width forces on
// every 16-lane pair kernel (V4-Flash's 4096 = 16 superblocks was exact).
//
// LDS: 16 B LUT + xq staging (MXFP4_PAIR_MAX_BLOCKS × 292 B = 9.3 KiB at
// 32 superblocks) — the iq3_s siblings staged only 16 superblocks (4096
// wide); V4.1 needs 20.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_MXFP4_BYTES 17
#define SUPER_MXFP4_BYTES (8 * BLOCK_MXFP4_BYTES)   // 136

// ---------------------------------------------------------------------------
// SUPER-BLOCK LAYOUT (v2, 2026-09-17). Within each 136-byte super-block:
//     bytes   0..128   eight 16-byte NIBBLE blocks, block k at k*16
//     bytes 128..136   eight E8M0 scale bytes, block k's at 128+k
// The old form interleaved them as 8 x [scale][16 nibbles], which put every
// block's nibbles at an ODD byte offset. Each nibble byte is consumed straight
// as a LUT index, so the compiler could not fuse them: 16 BYTE loads per block,
// 4x the memory instructions for the same bytes. That is what `MemUnitBusy=97%`
// at only 79% of achievable bandwidth was reporting -- a queue of tiny errands,
// not a bandwidth ceiling. (q8_0 learned this in M18: "load 32 int8s as 2 uint4s
// then dp4a"; mxfp4 arrived later and kept ggml's 17-byte block.)
//
// MEASURED, bench_mxfp4_moe_b1 (B=1, 75.2 MB/launch), 3+ runs:
//     old, byte loads                     ~410 us   183 GB/s   79% of achievable
//     new, uint2 (8 B) loads              ~360 us   209 GB/s   +14%
// Super-block STRIDE is unchanged at 136, so row addressing, bytes_per_expert,
// pool slots and snapshots are all untouched -- only the arrangement inside a
// super-block moves. 8-byte loads (not 16) because nb*17 is always a multiple
// of 8 but not of 16; going wider would need padding, which costs bytes on a
// bandwidth-bound kernel and was measured WORSE when misaligned.
//
// MEASURED NEGATIVE RESULTS (2026-09-19, bench_mxfp4_moe_b1 with a ROTATING
// 24-expert selection so the 32 MB last-level cache serves nothing; a fixed
// 3-slot selection reads 238 GB/s and is NOT a kernel measurement):
//     baseline (this kernel)          3 slots 189 GB/s   6 slots 201 GB/s
//     balanced lanes (each lane owns 5 MXFP4 blocks of gate + 5 of up,
//       no 16-lane/20-superblock imbalance)      189           195
//     row-SoA layout probe (every block 16-B aligned, ONE uint4 load;
//       timing-only, wrong results on v2 data)   190           202
// Neither the lane imbalance nor the 8-byte loads are the limiter; the kernel
// streams at ~93% of this iGPU's measured 214 GB/s. The ablation note's
// "226 GB/s with uint4" was taken with a fixed selection and is cache-assisted.
// Do not convert the layout to row-SoA for this kernel's sake.
#define MXFP4_NIB(w_super, b32)   ((w_super) + (b32) * 16u)
#define MXFP4_SCALE(w_super, b32) ((w_super)[128u + (b32)])

// 16 nibble bytes -> 4 dwords via two 8-byte loads.
__device__ __forceinline__ void mxfp4_load16(const uint8_t* qs, uint32_t* w) {
    uint2 a_, b_;
    __builtin_memcpy(&a_, qs, 8);
    __builtin_memcpy(&b_, qs + 8, 8);
    w[0] = a_.x; w[1] = a_.y; w[2] = b_.x; w[3] = b_.y;
}
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull
#ifndef MXFP4_PAIR_MAX_BLOCKS
#define MXFP4_PAIR_MAX_BLOCKS 32                    // Q8_K superblocks staged in LDS
#endif

__device__ __constant__ int8_t kvalues_mxfp4_pair[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

__device__ __forceinline__ int32_t sudot4_pair(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t load_dword_i8_pair(const int8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

// 2^(e-128) via bit construction (ggml_e8m0_to_fp32_half).
__device__ __forceinline__ float e8m0_half_to_f32_pair(uint32_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

__device__ __forceinline__ float warp_sum_f32_pair(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// This lane's half (4 mxfp4 blocks) of one 256-elem superblock:
// yd * Σ_b scale_b * sumi_b, sumi via the LDS nibble LUT + sudot4.
__device__ __forceinline__ float dot_super_half_mxfp4(
        const uint8_t* w_super, const uint8_t* y_blk, unsigned int half,
        const int8_t* s_lut) {
    const float yd = *(const float*)(y_blk + 0);
    const int8_t* q8_base = (const int8_t*)(y_blk + 4);
    float acc = 0.0f;
    #pragma unroll
    for (unsigned int t = 0; t < 4u; t++) {
        const unsigned int b32 = 4u * half + t;          // 0..7
        const float scale = e8m0_half_to_f32_pair(MXFP4_SCALE(w_super, b32));
        uint32_t ws_[4];
        mxfp4_load16(MXFP4_NIB(w_super, b32), ws_);
        const int8_t* q8 = q8_base + b32 * 32u;
        int32_t sumi = 0;
        #pragma unroll
        for (unsigned int i = 0; i < 4u; i++) {
            union { int32_t d; int8_t b[4]; } lo, hi;
            #pragma unroll
            for (unsigned int j = 0; j < 4u; j++) {
                const uint8_t q = (uint8_t)((ws_[i] >> (8u * j)) & 0xFFu);
                lo.b[j] = s_lut[q & 0x0F];
                hi.b[j] = s_lut[q >> 4];
            }
            sumi = sudot4_pair(lo.d, load_dword_i8_pair(q8 + 4u * i), sumi);
            sumi = sudot4_pair(hi.d, load_dword_i8_pair(q8 + 16u + 4u * i), sumi);
        }
        acc += scale * (float)sumi;
    }
    return yd * acc;
}

// Stage the LUT and the token's Q8_K activation into LDS; ends with a
// barrier (every thread of the WG must reach this).
#define MXFP4_PAIR_STAGE(s_lut, s_xq, xq, n_blocks)                              \
    {                                                                            \
        if (threadIdx.x < 16u) s_lut[threadIdx.x] = kvalues_mxfp4_pair[threadIdx.x]; \
        const unsigned int xq_bytes_ = (n_blocks) * BLOCK_Q8_K_BYTES;             \
        for (unsigned int i_ = threadIdx.x; i_ < xq_bytes_; i_ += blockDim.x) {  \
            s_xq[i_] = (xq)[i_];                                                 \
        }                                                                        \
        __syncthreads();                                                         \
    }

// One row of gate and up for expert `e` (this lane's share), reduced over
// the warp; lane 0 applies the clamp + SwiGLU + expert weight.
__device__ __forceinline__ void mxfp4_pair_row(
        float* __restrict__ mid_out,
        const uint8_t* gate_w_base, const uint8_t* up_w_base,
        const uint8_t* s_xq, const int8_t* s_lut,
        int e, unsigned int gate_bpe, unsigned int up_bpe,
        unsigned int row, unsigned int n_blocks, float clamp, float ew,
        unsigned int lane) {
    const unsigned int block_lane = lane >> 1u;
    const unsigned int half       = lane & 1u;
    const uint8_t* gate_row = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe
                            + (uint64_t)row * (uint64_t)n_blocks * SUPER_MXFP4_BYTES;
    const uint8_t* up_row   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe
                            + (uint64_t)row * (uint64_t)n_blocks * SUPER_MXFP4_BYTES;
    float g_sum = 0.0f;
    float u_sum = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks; b += 16u) {
        const uint8_t* y = s_xq + (size_t)b * BLOCK_Q8_K_BYTES;
        g_sum += dot_super_half_mxfp4(gate_row + (uint64_t)b * SUPER_MXFP4_BYTES, y, half, s_lut);
        u_sum += dot_super_half_mxfp4(up_row   + (uint64_t)b * SUPER_MXFP4_BYTES, y, half, s_lut);
    }
    g_sum = warp_sum_f32_pair(g_sum);
    u_sum = warp_sum_f32_pair(u_sum);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (g_sum > clamp)  g_sum = clamp;
            if (u_sum > clamp)  u_sum = clamp;
            if (u_sum < -clamp) u_sum = -clamp;
        }
        const float sig = 1.0f / (1.0f + expf(-g_sum));
        *mid_out = g_sum * sig * u_sum * ew;
    }
}

// Decode: batched over n_used slots, fused SwiGLU + expert weight.
// Contract identical to iq3_s_pair_matvec_fused_swiglu_batch.
extern "C" __global__ void mxfp4_pair_matvec_fused_swiglu_batch(
        float        * __restrict__ mid,           // [n_used, n_rows]
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,           // [n_blocks * 292]
        const float  * __restrict__ expert_w,      // [n_used]
        const int    * __restrict__ selected,      // [n_used]
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    __shared__ int8_t  s_lut[16];
    __shared__ uint8_t s_xq[MXFP4_PAIR_MAX_BLOCKS * BLOCK_Q8_K_BYTES];
    MXFP4_PAIR_STAGE(s_lut, s_xq, xq, n_blocks);
    const unsigned int tid  = threadIdx.x;
    const unsigned int slot = blockIdx.y;
    const unsigned int lane = tid & 31u;
    // Rows per workgroup derive from the launch (blockDim.x/32 warps, one
    // row each) instead of a hardcoded 8, so the geometry is a launch
    // parameter that can be swept. ds4/llama.cpp select warps- and
    // rows-per-block from a per-(type, batch, arch) table for exactly
    // this kernel class; ours was one constant chosen once.
    const unsigned int row  = blockIdx.x * (blockDim.x >> 5u) + (tid >> 5u);
    if (row >= n_rows) return;
    const int e = selected[slot];
    mxfp4_pair_row(mid + (uint64_t)slot * (uint64_t)n_rows + (uint64_t)row,
                   gate_w_base, up_w_base, s_xq, s_lut, e, gate_bpe, up_bpe,
                   row, n_blocks, clamp, expert_w[slot], lane);
}

// Decode het-split; contract identical to
// iq3_s_pair_matvec_fused_swiglu_batch_hetsplit: `remap[e]` holds the dGPU
// dense slot (>= 0) for hot experts or -(igpu_slot + 1) for cold ones;
// `mode` 0 = this launch runs on the iGPU and takes every slot the dGPU does
// not, 1 = dGPU side, taking at most `dgpu_cap` resident slots in slot
// order. Slots the launch does not own are zero-filled so the two devices'
// `mid` buffers sum.
extern "C" __global__ void mxfp4_pair_matvec_fused_swiglu_batch_hetsplit(
        float        * __restrict__ mid,
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,
        const float  * __restrict__ expert_w,
        const int    * __restrict__ selected,
        const int    * __restrict__ remap,         // [n_expert]
        unsigned int    mode,
        unsigned int    dgpu_cap,
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks) {
    __shared__ int8_t  s_lut[16];
    __shared__ uint8_t s_xq[MXFP4_PAIR_MAX_BLOCKS * BLOCK_Q8_K_BYTES];
    MXFP4_PAIR_STAGE(s_lut, s_xq, xq, n_blocks);
    const unsigned int tid  = threadIdx.x;
    const unsigned int slot = blockIdx.y;
    const unsigned int lane = tid & 31u;
    // Rows per workgroup derive from the launch (blockDim.x/32 warps, one
    // row each) instead of a hardcoded 8, so the geometry is a launch
    // parameter that can be swept. ds4/llama.cpp select warps- and
    // rows-per-block from a per-(type, batch, arch) table for exactly
    // this kernel class; ours was one constant chosen once.
    const unsigned int row  = blockIdx.x * (blockDim.x >> 5u) + (tid >> 5u);
    const int sel   = selected[slot];
    const int dense = remap[sel];
    unsigned int res_rank = 0;
    for (unsigned int t2 = 0; t2 < slot; t2++) {
        res_rank += (remap[selected[t2]] >= 0) ? 1u : 0u;
    }
    const bool dgpu_takes = (dense >= 0) && (res_rank < dgpu_cap);
    const bool ours = (mode == 0u) ? !dgpu_takes : dgpu_takes;
    if (row >= n_rows) return;
    if (!ours) {
        if (lane == 0) mid[(uint64_t)slot * (uint64_t)n_rows + (uint64_t)row] = 0.0f;
        return;
    }
    // iGPU side: a hot expert past the cap sits at its raw id in the full
    // buffer; a cold one at its packed slot. dGPU side: the dense slot.
    const int e = (mode == 0u) ? ((dense >= 0) ? sel : (-dense - 1)) : dense;
    mxfp4_pair_row(mid + (uint64_t)slot * (uint64_t)n_rows + (uint64_t)row,
                   gate_w_base, up_w_base, s_xq, s_lut, e, gate_bpe, up_bpe,
                   row, n_blocks, clamp, expert_w[slot], lane);
}

// =============================================================================
// Prefill: chunked by-expert (work-items interface — the same arrays as the
// iq3_s / iq2_s chunked and kwide kernels). One WG per (row-tile, work
// item); per member the token's Q8_K row is re-staged in LDS and the weights
// are re-read per member (the ae9bd61 trap) — kept as the PAIR_VARIANT=chunked
// fallback; kwide below is the default.
// =============================================================================
extern "C" __global__ void mxfp4_pair_matvec_fused_swiglu_chunked(
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

    __shared__ int8_t  s_lut[16];
    __shared__ uint8_t s_xq[MXFP4_PAIR_MAX_BLOCKS * BLOCK_Q8_K_BYTES];
    const unsigned int tid = threadIdx.x;
    if (tid < 16u) s_lut[tid] = kvalues_mxfp4_pair[tid];
    __syncthreads();

    const unsigned int lane = tid & 31u;
    // Rows per workgroup derive from the launch (blockDim.x/32 warps, one
    // row each) instead of a hardcoded 8, so the geometry is a launch
    // parameter that can be swept. ds4/llama.cpp select warps- and
    // rows-per-block from a per-(type, batch, arch) table for exactly
    // this kernel class; ours was one constant chosen once.
    const unsigned int row  = blockIdx.x * (blockDim.x >> 5u) + (tid >> 5u);
    const bool row_valid = row < n_rows;
    const unsigned int xq_bytes = n_blocks * BLOCK_Q8_K_BYTES;

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
            const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows + (size_t)row;
            mxfp4_pair_row(mid + out_off, gate_w_base, up_w_base, s_xq, s_lut, (int)e,
                           gate_bpe, up_bpe, row, n_blocks, clamp,
                           expert_w[(size_t)b * (size_t)n_used + (size_t)slot], lane);
        }
        __syncthreads(); // before the next member's xq overwrite
    }
}

// =============================================================================
// kwide — M51-structure port for MXFP4 gate/up, from
// iq3_s_pair_matvec_fused_swiglu_kwide. Only the per-lane unpack differs.
//
// Lane mapping: warp = row; lanes split a super-block PAIR — sbh = lane>>4
// picks the sb, l15 = lane&15 owns the 16 elements [16*l15, +16) of that
// 256-element super-block = mxfp4 block b32 = l15>>1, nibble half = l15&1
// (elements 0..15 of a block are the LOW nibbles of its 16 qs bytes, 16..31
// the HIGH nibbles). The lane's e8m0 scale covers exactly its 16 weights,
// so folding gds = 2^(e-128) per lane is exact. The staged s_q8v lane slice
// [16*l15, +16) of the block's 256 q8 bytes is the matching activation
// slice (staging code verbatim from the iq3_s kernel).
// =============================================================================
#ifndef MXFP4_KW_MAX_CHUNK
#define MXFP4_KW_MAX_CHUNK 32u
#endif
#ifndef MXFP4_KW_UNROLL
#define MXFP4_KW_UNROLL 4
#endif

// This lane's 16 weights of one mxfp4 block as 4 sudot4-ready dwords.
__device__ __forceinline__ void mxfp4_unpack16(
        const uint8_t* w_super, unsigned int b32, unsigned int half,
        const int8_t* s_lut,
        int32_t& w0, int32_t& w1, int32_t& w2, int32_t& w3, float& ds) {
    ds = e8m0_half_to_f32_pair(MXFP4_SCALE(w_super, b32));
    uint32_t qw_[4];
    mxfp4_load16(MXFP4_NIB(w_super, b32), qw_);
    const unsigned int sh = 4u * half;
    int32_t w[4];
    #pragma unroll
    for (unsigned int i = 0; i < 4u; i++) {
        union { int32_t d; int8_t b[4]; } u;
        #pragma unroll
        for (unsigned int j = 0; j < 4u; j++) {
            u.b[j] = s_lut[(((qw_[i] >> (8u * j)) & 0xFFu) >> sh) & 0x0Fu];
        }
        w[i] = u.d;
    }
    w0 = w[0]; w1 = w[1]; w2 = w[2]; w3 = w[3];
}

extern "C" __global__ void mxfp4_pair_matvec_fused_swiglu_kwide(
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

    __shared__ int8_t   s_lut[16];
    __shared__ uint4    s_q8v[MXFP4_KW_MAX_CHUNK * 2 * 16]; // [mi][sbh][16]
    __shared__ float    s_yd[MXFP4_KW_MAX_CHUNK * 2];       // [mi][sbh]
    __shared__ int      s_member_packed[MXFP4_KW_MAX_CHUNK];

    const unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i < n_members; i += blockDim.x) {
        s_member_packed[i] = expert_members[(size_t)e * (size_t)max_per_expert
                                          + (size_t)(member_start + i)];
    }
    if (tid < 16u) s_lut[tid] = kvalues_mxfp4_pair[tid];
    __syncthreads();

    const unsigned int lane = tid & 31u;
    // Rows per workgroup derive from the launch (blockDim.x/32 warps, one
    // row each) instead of a hardcoded 8, so the geometry is a launch
    // parameter that can be swept. ds4/llama.cpp select warps- and
    // rows-per-block from a per-(type, batch, arch) table for exactly
    // this kernel class; ours was one constant chosen once.
    const unsigned int row  = blockIdx.x * (blockDim.x >> 5u) + (tid >> 5u);
    const bool row_valid = (row < n_rows);

    const unsigned int sbh  = lane >> 4;               // 0..1: sb of the pair
    const unsigned int l15  = lane & 15u;
    const unsigned int b32  = l15 >> 1;                // mxfp4 block within the sb
    const unsigned int half = l15 & 1u;                // low / high nibbles

    const uint8_t* gate_e_base = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;
    const uint8_t* up_e_base   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;

    float lane_pacc_g[MXFP4_KW_MAX_CHUNK];
    float lane_pacc_u[MXFP4_KW_MAX_CHUNK];
    #pragma unroll
    for (unsigned int mi = 0; mi < MXFP4_KW_MAX_CHUNK; mi++) {
        lane_pacc_g[mi] = 0.0f;
        lane_pacc_u[mi] = 0.0f;
    }

    const unsigned int n_pairs = n_blocks >> 1;   // wrapper asserts n_blocks even
    for (unsigned int sbp = 0; sbp < n_pairs; sbp++) {
        const unsigned int sb_lane = 2u * sbp + sbh;

        // --- Dequant: this lane's 16 weights x 2 matrices -> 8 dwords ---
        int32_t wg0 = 0, wg1 = 0, wg2 = 0, wg3 = 0;
        int32_t wu0 = 0, wu1 = 0, wu2 = 0, wu3 = 0;
        float gds = 0.0f, uds = 0.0f;
        if (row_valid) {
            // Super-block base only; `mxfp4_unpack16` indexes block b32 inside it.
            const uint64_t off = (uint64_t)row * (uint64_t)n_blocks * SUPER_MXFP4_BYTES
                               + (uint64_t)sb_lane * SUPER_MXFP4_BYTES;
            mxfp4_unpack16(gate_e_base + off, b32, half, s_lut, wg0, wg1, wg2, wg3, gds);
            mxfp4_unpack16(up_e_base   + off, b32, half, s_lut, wu0, wu1, wu2, wu3, uds);
        }

        // --- Cooperative xq stage (M53 pattern, verbatim from iq3_s) ---
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

        // --- Dot phase: one b128 per (member, lane) feeds 8 sudot4 ---
        if (row_valid) {
            const uint4* q8_lane = &s_q8v[sbh * 16u + l15];   // member mi at +mi*32
            const float* yd_lane = &s_yd[sbh];                // member mi at +mi*2
            #pragma unroll MXFP4_KW_UNROLL
            for (unsigned int mi = 0; mi < n_members; mi++) {
                const uint4 q = q8_lane[mi * 32u];
                int32_t sumi_g = 0;
                sumi_g = sudot4_pair(wg0, (int32_t)q.x, sumi_g);
                sumi_g = sudot4_pair(wg1, (int32_t)q.y, sumi_g);
                sumi_g = sudot4_pair(wg2, (int32_t)q.z, sumi_g);
                sumi_g = sudot4_pair(wg3, (int32_t)q.w, sumi_g);
                int32_t sumi_u = 0;
                sumi_u = sudot4_pair(wu0, (int32_t)q.x, sumi_u);
                sumi_u = sudot4_pair(wu1, (int32_t)q.y, sumi_u);
                sumi_u = sudot4_pair(wu2, (int32_t)q.z, sumi_u);
                sumi_u = sudot4_pair(wu3, (int32_t)q.w, sumi_u);
                const float yd = yd_lane[mi * 2u];
                lane_pacc_g[mi] += (float)sumi_g * (yd * gds);
                lane_pacc_u[mi] += (float)sumi_u * (yd * uds);
            }
        }
        __syncthreads();  // protect s_q8v/s_yd from the next pair's overwrite
    }

    // --- Reduce + swiglu + output, fused ---
    if (row_valid) {
        for (unsigned int mi = 0; mi < n_members; mi++) {
            float g_sum = warp_sum_f32_pair(lane_pacc_g[mi]);
            float u_sum = warp_sum_f32_pair(lane_pacc_u[mi]);
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
