// A_moe_smallb/gu_smallb — small-b (b <= 8 rows) twin of mxfp4_pair_matvec_fused_swiglu_kwide.
//
// Same work-item interface, same lane -> data mapping (warp = row; lane sbh = lane>>4 picks the super-block of
// the pair, l15 = lane&15 owns 16 elements = mxfp4 block b32 = l15>>1, nibble half = l15&1), same per-lane
// accumulation order => bit-exact with kwide. What changes:
//   * no LDS staging of the members' Q8_K rows and no per-pair barriers: each lane reads its 16 q8 bytes per
//     (member, super-block) straight from global memory (the 8 warps of a WG read the same 5 KB per member, so
//     this is L0/L2 traffic);
//   * the member loop is unrolled over SB_MB (8) with wave-uniform predicates -> no dynamic VGPR indexing;
//   * the super-block-pair loop is fully unrolled (SB_NB = 20 blocks -> 10 pairs), so the compiler can put the
//     whole row's weight loads (340 B/lane) in flight instead of one pair's 34 B between two barriers.
// Members beyond SB_MB are processed in further passes that re-read the row's weights (only reachable if the
// dispatcher ever sends chunk_size > SB_MB to this kernel; at b <= 8 a group has <= b <= 8 members).
//
// Production symbol would be mxfp4_pair_matvec_fused_swiglu_smallb; exported here as cand_gate_up for the harness.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define SUPER_MXFP4_BYTES 136u
#define BLOCK_Q8_K_BYTES 292u
#ifndef SB_MB
#define SB_MB 8u
#endif
#ifndef SB_NB
#define SB_NB 20u
#endif
#ifndef SB_UNR
#define SB_UNR 10
#endif

__device__ __constant__ int8_t kvalues_mxfp4_sb[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ int32_t sudot4_sb(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
__device__ __forceinline__ float e8m0_half_sb(uint32_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}
__device__ __forceinline__ float warp_sum_sb(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, o);
    return v;
}

struct RawBlk { uint2 a, b; uint8_t sc; };
__device__ __forceinline__ void load_blk(const uint8_t* w_super, unsigned b32, RawBlk& r) {
    __builtin_memcpy(&r.a, w_super + b32 * 16u, 8);
    __builtin_memcpy(&r.b, w_super + b32 * 16u + 8u, 8);
    r.sc = w_super[128u + b32];
}
// Bit-identical to mxfp4_unpack16 in the in-tree kernel.
__device__ __forceinline__ void unpack_blk(const RawBlk& r, unsigned half, const int8_t* s_lut, int32_t w[4], float& ds) {
    ds = e8m0_half_sb(r.sc);
    const uint32_t qw[4] = {r.a.x, r.a.y, r.b.x, r.b.y};
    const unsigned sh = 4u * half;
    #pragma unroll
    for (unsigned i = 0; i < 4u; i++) {
        union { int32_t d; int8_t b[4]; } u;
        #pragma unroll
        for (unsigned j = 0; j < 4u; j++) u.b[j] = s_lut[(((qw[i] >> (8u * j)) & 0xFFu) >> sh) & 0x0Fu];
        w[i] = u.d;
    }
}

extern "C" __global__ void cand_gate_up(
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
        unsigned int    n_blocks,
        const int    * __restrict__ n_work_items_dev) {
    if (n_work_items_dev != nullptr && (int)blockIdx.y >= *n_work_items_dev) return;
    if (n_blocks != SB_NB) return;  // compile-time width; the wrapper must guarantee it
    const int packed_item = work_items[blockIdx.y];
    const unsigned e            = (unsigned)(packed_item >> 16) & 0xffffu;
    const unsigned member_start = (unsigned)(packed_item) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned member_end = min((unsigned)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;

    __shared__ int8_t s_lut[16];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = kvalues_mxfp4_sb[threadIdx.x];
    __syncthreads();

    const unsigned lane = threadIdx.x & 31u;
    const unsigned row  = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (row >= n_rows) return;  // wave-uniform
    const unsigned sbh  = lane >> 4;
    const unsigned l15  = lane & 15u;
    const unsigned b32  = l15 >> 1;
    const unsigned half = l15 & 1u;

    const uint8_t* g_row = gate_w_base + (uint64_t)e * gate_bpe + (uint64_t)row * SB_NB * SUPER_MXFP4_BYTES;
    const uint8_t* u_row = up_w_base   + (uint64_t)e * up_bpe   + (uint64_t)row * SB_NB * SUPER_MXFP4_BYTES;
    constexpr unsigned NP = SB_NB / 2u;

    for (unsigned mg = member_start; mg < member_end; mg += SB_MB) {
        const unsigned nm = min(SB_MB, member_end - mg);
        // member rows: pointer to the lane's super-block within the member's Q8_K row (block sbh of pair 0)
        const uint8_t* xq_m[SB_MB];
        int packed_m[SB_MB];
        #pragma unroll
        for (unsigned m = 0; m < SB_MB; m++) {
            packed_m[m] = 0;
            xq_m[m] = xq;
            if (m < nm) {
                packed_m[m] = expert_members[(size_t)e * max_per_expert + (size_t)(mg + m)];
                const unsigned b = (unsigned)(packed_m[m] >> 16);
                xq_m[m] = xq + (size_t)b * SB_NB * BLOCK_Q8_K_BYTES + (size_t)sbh * BLOCK_Q8_K_BYTES;
            }
        }
        float acc_g[SB_MB], acc_u[SB_MB];
        #pragma unroll
        for (unsigned m = 0; m < SB_MB; m++) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

        #pragma unroll SB_UNR
        for (unsigned sbp = 0; sbp < NP; sbp++) {
            const unsigned sb = 2u * sbp + sbh;
            RawBlk rg, ru;
            load_blk(g_row + sb * SUPER_MXFP4_BYTES, b32, rg);
            load_blk(u_row + sb * SUPER_MXFP4_BYTES, b32, ru);
            int32_t wg[4], wu[4];
            float gds, uds;
            unpack_blk(rg, half, s_lut, wg, gds);
            unpack_blk(ru, half, s_lut, wu, uds);
            #pragma unroll
            for (unsigned m = 0; m < SB_MB; m++) {
                if (m < nm) {
                    const uint8_t* blk = xq_m[m] + (size_t)(2u * sbp) * BLOCK_Q8_K_BYTES;
                    float yd;
                    __builtin_memcpy(&yd, blk, 4);
                    uint4 q;
                    __builtin_memcpy(&q, blk + 4u + 16u * l15, 16);
                    int32_t sg = 0, su = 0;
                    sg = sudot4_sb(wg[0], (int32_t)q.x, sg);
                    sg = sudot4_sb(wg[1], (int32_t)q.y, sg);
                    sg = sudot4_sb(wg[2], (int32_t)q.z, sg);
                    sg = sudot4_sb(wg[3], (int32_t)q.w, sg);
                    su = sudot4_sb(wu[0], (int32_t)q.x, su);
                    su = sudot4_sb(wu[1], (int32_t)q.y, su);
                    su = sudot4_sb(wu[2], (int32_t)q.z, su);
                    su = sudot4_sb(wu[3], (int32_t)q.w, su);
                    acc_g[m] += (float)sg * (yd * gds);
                    acc_u[m] += (float)su * (yd * uds);
                }
            }
        }
        #pragma unroll
        for (unsigned m = 0; m < SB_MB; m++) {
            if (m < nm) {
                float g_sum = warp_sum_sb(acc_g[m]);
                float u_sum = warp_sum_sb(acc_u[m]);
                if (lane == 0u) {
                    const unsigned b    = (unsigned)(packed_m[m] >> 16);
                    const unsigned slot = (unsigned)(packed_m[m] & 0xFFFF);
                    if (clamp > 1.0e-6f) {
                        if (g_sum > clamp)  g_sum = clamp;
                        if (u_sum > clamp)  u_sum = clamp;
                        if (u_sum < -clamp) u_sum = -clamp;
                    }
                    const float ew  = expert_w[(size_t)b * n_used + slot];
                    const float sig = 1.0f / (1.0f + expf(-g_sum));
                    mid[((size_t)b * n_used + slot) * n_rows + row] = g_sum * sig * u_sum * ew;
                }
            }
        }
    }
}
