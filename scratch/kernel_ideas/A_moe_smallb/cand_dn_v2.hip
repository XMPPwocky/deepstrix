// A_moe_smallb/dn_v2 — small-b twin of mxfp4_matvec_par_by_expert_kwide2, member-outer.
//
// Same interface, same lane -> data map (DN_ROWS rows per warp, block_lane = lane>>2 over the 9 super-blocks,
// sub = lane&3 owns mxfp4 blocks {2sub, 2sub+1}; lanes with block_lane == 0 also do super-block 8), same
// per-lane accumulation order => bit-exact with kwide2. What changes:
//   * the warp's WHOLE weight stripe (DN_ROWS rows x both super-block iterations) is loaded up front and kept
//     unpacked in registers: every weight byte the warp will ever need is in flight at once (kwide2 issues one
//     super-block's loads per bl iteration, inside a loop that also carries the member loop);
//   * the member loop is OUTER with scalar accumulators (no pacc[16] arrays, no dynamic indexing, no
//     per-member state across super-blocks): per member = 4 b128 q8 loads per iteration + 32 sudot4 +
//     DN_ROWS warp reductions + a lane-0 store. Members are independent so the compiler may overlap them.
//   * DN_ROWS = 2 keeps kwide2's grid (320 x n_wi); DN_ROWS = 1 is the down_hetsplit geometry (640 x n_wi).
// Production symbol would be mxfp4_matvec_par_by_expert_smallb; exported as cand_down.
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef SUPER_MXFP4_BYTES
#define SUPER_MXFP4_BYTES 136u
#endif
#ifndef BLOCK_Q8_K_BYTES
#define BLOCK_Q8_K_BYTES 292u
#endif
#ifndef DN_NB
#define DN_NB 9u   // n_blocks_in (2304 / 256)
#endif
#ifndef DN_ROWS
#define DN_ROWS 2u
#endif
#ifndef DN_MUNR
#define DN_MUNR 1
#endif

__device__ __constant__ int8_t kvalues_mxfp4_dn2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ int32_t sudot4_dn2(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
__device__ __forceinline__ float e8m0_half_dn2(uint32_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}
__device__ __forceinline__ float warp_sum_dn2(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, o);
    return v;
}
__device__ __forceinline__ int32_t ld_i8x4_dn2(const int8_t* p) { int32_t v; __builtin_memcpy(&v, p, 4); return v; }

struct RawSub { uint2 a0, b0, a1, b1; uint8_t sc0, sc1; };  // this lane's 2 blocks of one super-block
__device__ __forceinline__ void load_sub(const uint8_t* w_super, unsigned sub, RawSub& r) {
    const unsigned k0 = 2u * sub, k1 = 2u * sub + 1u;
    __builtin_memcpy(&r.a0, w_super + k0 * 16u, 8);
    __builtin_memcpy(&r.b0, w_super + k0 * 16u + 8u, 8);
    __builtin_memcpy(&r.a1, w_super + k1 * 16u, 8);
    __builtin_memcpy(&r.b1, w_super + k1 * 16u + 8u, 8);
    r.sc0 = w_super[128u + k0];
    r.sc1 = w_super[128u + k1];
}
// Bit-identical to kwide2's unpack: w[8t + i] = lo dwords, w[8t + 4 + i] = hi dwords.
__device__ __forceinline__ void unpack_sub(const RawSub& r, const int8_t* s_lut, int32_t w[16], float sc[2]) {
    sc[0] = e8m0_half_dn2(r.sc0);
    sc[1] = e8m0_half_dn2(r.sc1);
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

extern "C" __global__ void cand_down(
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
    if (n_blocks_in != DN_NB) return;
    __shared__ int8_t s_lut[16];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = kvalues_mxfp4_dn2[threadIdx.x];
    __syncthreads();

    const unsigned warp_in_block = threadIdx.x / 32u;
    const unsigned lane          = threadIdx.x & 31u;
    const unsigned row0          = blockIdx.x * (8u * DN_ROWS) + warp_in_block * DN_ROWS;
    if (row0 >= n_rows) return;

    const unsigned block_lane = lane >> 2u;   // 0..7
    const unsigned sub        = lane & 3u;
    const unsigned q8_off     = sub * 64u;
    const bool     second     = (block_lane + 8u) < DN_NB;   // block_lane 0 only

    // Work-item loop: with grid.y = the 6b upper bound this runs once (production semantics); the wrapper may
    // instead cap grid.y at a small constant so the (bound - n_wi) x 320 empty WGs are never launched.
    const unsigned n_items = (n_work_items_dev != nullptr) ? (unsigned)max(0, *n_work_items_dev) : gridDim.y;
    for (unsigned w_idx = blockIdx.y; w_idx < n_items; w_idx += gridDim.y) {
    const int packed_item = work_items[w_idx];
    const unsigned e            = (unsigned)(packed_item >> 16) & 0xffffu;
    const unsigned member_start = (unsigned)(packed_item) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned member_end = min((unsigned)group_n, member_start + chunk_size);
    if (member_end <= member_start) continue;

    const uint8_t* w_r0 = w_base + (uint64_t)e * dbpe + (uint64_t)row0 * DN_NB * SUPER_MXFP4_BYTES;

    // The warp's whole weight stripe: DN_ROWS rows x (1 or 2) super-blocks, all loads issued now.
    RawSub raw[DN_ROWS][2];
    #pragma unroll
    for (unsigned rr = 0; rr < DN_ROWS; rr++) {
        const uint8_t* w_r = w_r0 + (uint64_t)rr * DN_NB * SUPER_MXFP4_BYTES;
        load_sub(w_r + block_lane * SUPER_MXFP4_BYTES, sub, raw[rr][0]);
        // lanes without a second super-block re-load their first one (valid address, result unused)
        load_sub(w_r + (second ? block_lane + 8u : block_lane) * SUPER_MXFP4_BYTES, sub, raw[rr][1]);
    }
    int32_t w[DN_ROWS][2][16];
    float   sc[DN_ROWS][2][2];
    #pragma unroll
    for (unsigned rr = 0; rr < DN_ROWS; rr++) {
        unpack_sub(raw[rr][0], s_lut, w[rr][0], sc[rr][0]);
        unpack_sub(raw[rr][1], s_lut, w[rr][1], sc[rr][1]);
    }

    #pragma unroll DN_MUNR
    for (unsigned mi = member_start; mi < member_end; mi++) {
        const int packed = expert_members[(size_t)e * max_per_expert + (size_t)mi];
        const unsigned b    = (unsigned)(packed >> 16) & 0xffffu;
        const unsigned slot = (unsigned)(packed) & 0xffffu;
        const uint8_t* y_m = xq_base + ((size_t)b * n_used + slot) * xq_slot_stride;
        float pacc[DN_ROWS];
        #pragma unroll
        for (unsigned rr = 0; rr < DN_ROWS; rr++) pacc[rr] = 0.0f;
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
                    q[8u * t + i]      = ld_i8x4_dn2(q8b + 32u * t + 4u * i);
                    q[8u * t + 4u + i] = ld_i8x4_dn2(q8b + 32u * t + 16u + 4u * i);
                }
            }
            #pragma unroll
            for (unsigned rr = 0; rr < DN_ROWS; rr++) {
                float f = 0.0f;
                #pragma unroll
                for (unsigned t = 0; t < 2u; t++) {
                    int32_t s = 0;
                    #pragma unroll
                    for (unsigned i = 0; i < 8u; i++) s = sudot4_dn2(w[rr][it][8u * t + i], q[8u * t + i], s);
                    f += sc[rr][it][t] * (float)s;
                }
                pacc[rr] += yd * f;
            }
        }
        #pragma unroll
        for (unsigned rr = 0; rr < DN_ROWS; rr++) {
            const float a = warp_sum_dn2(pacc[rr]);
            if (lane == 0) partials[((size_t)b * n_used + slot) * n_rows + row0 + rr] = a;
        }
    }
    }  // work-item loop
}
