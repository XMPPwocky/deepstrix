// A_moe_smallb/dn_smallb — small-b (b <= 8) twin of mxfp4_matvec_par_by_expert_kwide2.
//
// Same interface, same lane -> data mapping (2 rows per warp, block_lane = lane>>2 over the 9 super-blocks,
// sub = lane&3 owns mxfp4 blocks {2sub, 2sub+1}), same per-lane accumulation order => bit-exact with kwide2.
// Changes: SB_MB (8) member accumulators instead of 16 (fewer VGPRs), the member loop unrolled with wave-uniform
// predicates (no dynamic indexing), and the two super-block iterations (bl = block_lane, block_lane + 8; the
// second only for block_lane == 0) written out so both iterations' weight loads can be hoisted and in flight.
// Members beyond SB_MB: further passes re-reading the rows' weights (kwide2 does the same in halves of 16).
//
// Production symbol would be mxfp4_matvec_par_by_expert_smallb; exported as cand_down for the harness.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define SUPER_MXFP4_BYTES 136u
#define BLOCK_Q8_K_BYTES 292u
#ifndef SB_MB
#define SB_MB 8u
#endif
#ifndef SB_NBD
#define SB_NBD 9u   // n_blocks_in (2304 / 256)
#endif

__device__ __constant__ int8_t kvalues_mxfp4_sbd[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ int32_t sudot4_sbd(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
__device__ __forceinline__ float e8m0_half_sbd(uint32_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}
__device__ __forceinline__ float warp_sum_sbd(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, o);
    return v;
}
__device__ __forceinline__ int32_t ld_i8x4(const int8_t* p) { int32_t v; __builtin_memcpy(&v, p, 4); return v; }

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
    sc[0] = e8m0_half_sbd(r.sc0);
    sc[1] = e8m0_half_sbd(r.sc1);
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
    if (n_work_items_dev != nullptr && (int)blockIdx.y >= *n_work_items_dev) return;
    if (n_blocks_in != SB_NBD) return;
    __shared__ int8_t s_lut[16];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = kvalues_mxfp4_sbd[threadIdx.x];
    __syncthreads();

    const unsigned warp_in_block = threadIdx.x / 32u;
    const unsigned lane          = threadIdx.x & 31u;
    const unsigned row0          = blockIdx.x * 16u + warp_in_block * 2u;
    if (row0 >= n_rows) return;

    const int packed_item = work_items[blockIdx.y];
    const unsigned e            = (unsigned)(packed_item >> 16) & 0xffffu;
    const unsigned member_start = (unsigned)(packed_item) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned member_end = min((unsigned)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;

    const unsigned block_lane = lane >> 2u;   // 0..7
    const unsigned sub        = lane & 3u;
    const unsigned q8_off     = sub * 64u;
    const bool     second     = (block_lane + 8u) < SB_NBD;   // lane 0..3 only (block_lane 0)

    const uint8_t* w_expert = w_base + (uint64_t)e * dbpe;
    const uint8_t* w_r0 = w_expert + (uint64_t)row0 * SB_NBD * SUPER_MXFP4_BYTES;
    const uint8_t* w_r1 = w_r0 + (uint64_t)SB_NBD * SUPER_MXFP4_BYTES;

    // Both iterations' weights for both rows, loaded up front.
    RawSub r0a, r1a, r0b, r1b;
    load_sub(w_r0 + block_lane * SUPER_MXFP4_BYTES, sub, r0a);
    load_sub(w_r1 + block_lane * SUPER_MXFP4_BYTES, sub, r1a);
    if (second) {
        load_sub(w_r0 + (block_lane + 8u) * SUPER_MXFP4_BYTES, sub, r0b);
        load_sub(w_r1 + (block_lane + 8u) * SUPER_MXFP4_BYTES, sub, r1b);
    }

    for (unsigned mg = member_start; mg < member_end; mg += SB_MB) {
        const unsigned nm = min(SB_MB, member_end - mg);
        const uint8_t* y_m[SB_MB];
        int packed_m[SB_MB];
        #pragma unroll
        for (unsigned m = 0; m < SB_MB; m++) {
            packed_m[m] = 0;
            y_m[m] = xq_base;
            if (m < nm) {
                packed_m[m] = expert_members[(size_t)e * max_per_expert + (size_t)(mg + m)];
                const unsigned b    = (unsigned)(packed_m[m] >> 16) & 0xffffu;
                const unsigned slot = (unsigned)(packed_m[m]) & 0xffffu;
                y_m[m] = xq_base + ((size_t)b * n_used + slot) * xq_slot_stride;
            }
        }
        float pacc0[SB_MB], pacc1[SB_MB];
        #pragma unroll
        for (unsigned m = 0; m < SB_MB; m++) { pacc0[m] = 0.0f; pacc1[m] = 0.0f; }

        #pragma unroll
        for (unsigned it = 0; it < 2u; it++) {
            if (it == 1u && !second) break;   // divergent only for lanes 4..31 at it==1
            const unsigned bl = block_lane + 8u * it;
            int32_t w0[16], w1[16];
            float sc0[2], sc1[2];
            unpack_sub(it ? r0b : r0a, s_lut, w0, sc0);
            unpack_sub(it ? r1b : r1a, s_lut, w1, sc1);
            #pragma unroll
            for (unsigned m = 0; m < SB_MB; m++) {
                if (m < nm) {
                    const uint8_t* y_blk = y_m[m] + (size_t)bl * BLOCK_Q8_K_BYTES;
                    float yd;
                    __builtin_memcpy(&yd, y_blk, 4);
                    const int8_t* q8b = (const int8_t*)(y_blk + 4) + q8_off;
                    int32_t q[16];
                    #pragma unroll
                    for (unsigned t = 0; t < 2u; t++) {
                        #pragma unroll
                        for (unsigned i = 0; i < 4u; i++) {
                            q[8u * t + i]      = ld_i8x4(q8b + 32u * t + 4u * i);
                            q[8u * t + 4u + i] = ld_i8x4(q8b + 32u * t + 16u + 4u * i);
                        }
                    }
                    float f0 = 0.0f, f1 = 0.0f;
                    #pragma unroll
                    for (unsigned t = 0; t < 2u; t++) {
                        int32_t s0 = 0, s1 = 0;
                        #pragma unroll
                        for (unsigned i = 0; i < 8u; i++) {
                            s0 = sudot4_sbd(w0[8u * t + i], q[8u * t + i], s0);
                            s1 = sudot4_sbd(w1[8u * t + i], q[8u * t + i], s1);
                        }
                        f0 += sc0[t] * (float)s0;
                        f1 += sc1[t] * (float)s1;
                    }
                    pacc0[m] += yd * f0;
                    pacc1[m] += yd * f1;
                }
            }
        }
        #pragma unroll
        for (unsigned m = 0; m < SB_MB; m++) {
            if (m < nm) {
                const float a0 = warp_sum_sbd(pacc0[m]);
                const float a1 = warp_sum_sbd(pacc1[m]);
                if (lane == 0) {
                    const unsigned b    = (unsigned)(packed_m[m] >> 16) & 0xffffu;
                    const unsigned slot = (unsigned)(packed_m[m]) & 0xffffu;
                    const size_t out_base = ((size_t)b * n_used + slot) * n_rows;
                    partials[out_base + row0]      = a0;
                    partials[out_base + row0 + 1u] = a1;
                }
            }
        }
    }
}
