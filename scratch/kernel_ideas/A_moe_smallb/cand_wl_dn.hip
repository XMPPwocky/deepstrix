// A_moe_smallb/wl_dn — the UNMODIFIED production kwide2 down kernel body (helpers copied verbatim under _wl
// names so it can share a code object with the pair-kernel include) with ONE change: the work-group loops over
// work items w = blockIdx.y, + gridDim.y, ... < n_work_items, so grid.y can be capped (see cand_wl_gu.hip).
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef SUPER_MXFP4_BYTES
#define SUPER_MXFP4_BYTES 136u
#endif
#ifndef BLOCK_Q8_K_BYTES
#define BLOCK_Q8_K_BYTES 292u
#endif
#define WL_NIB(w_super, b32)   ((w_super) + (b32) * 16u)
#define WL_SCALE(w_super, b32) ((w_super)[128u + (b32)])

__device__ __constant__ int8_t kvalues_mxfp4_wl[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
__device__ __forceinline__ void mxfp4_load16_wl(const uint8_t* qs, uint32_t* w) {
    uint2 a_, b_;
    __builtin_memcpy(&a_, qs, 8);
    __builtin_memcpy(&b_, qs + 8, 8);
    w[0] = a_.x; w[1] = a_.y; w[2] = b_.x; w[3] = b_.y;
}
__device__ __forceinline__ int32_t sudot4_wl(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
__device__ __forceinline__ int32_t load_dword_i8_wl(const int8_t* p) { int32_t v; __builtin_memcpy(&v, p, 4); return v; }
__device__ __forceinline__ float e8m0_half_to_f32_wl(uint32_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}
__device__ __forceinline__ float warp_sum_f32_wl(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, offset);
    return v;
}

extern "C" __global__ void cand_down(
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
    const unsigned int n_items = (n_work_items_dev != nullptr) ? (unsigned int)max(0, *n_work_items_dev) : gridDim.y;
    __shared__ int8_t s_lut[16];
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = kvalues_mxfp4_wl[threadIdx.x];
    __syncthreads();

    const unsigned int warp_in_block = threadIdx.x / 32u;
    const unsigned int lane          = threadIdx.x & 31u;
    const unsigned int row0          = blockIdx.x * 16u + warp_in_block * 2u;
    if (row0 >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;   // 0..7 (superblock)
    const unsigned int sub        = lane & 3u;    // 0..3 → mxfp4 blocks {2sub, 2sub+1}
    const unsigned int q8_off     = sub * 64u;

    for (unsigned int w_idx = blockIdx.y; w_idx < n_items; w_idx += gridDim.y) {
        const int packed_item = work_items[w_idx];
        const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
        const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
        const int group_n = group_count[e];
        const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
        if (member_end <= member_start) continue;
        const unsigned int n_members = member_end - member_start;

        const uint8_t* w_expert = w_base + (uint64_t)e * (uint64_t)dbpe;

        for (unsigned int half_lo = 0; half_lo < n_members; half_lo += 16u) {
            const unsigned int half_n = min(16u, n_members - half_lo);

            float pacc0[16];
            float pacc1[16];
            #pragma unroll
            for (unsigned int m = 0; m < 16u; m++) { pacc0[m] = 0.0f; pacc1[m] = 0.0f; }

            for (unsigned int bl = block_lane; bl < n_blocks_in; bl += 8u) {
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
                        sc[t] = e8m0_half_to_f32_wl(WL_SCALE(w_super, b32_));
                        uint32_t qw_[4];
                        mxfp4_load16_wl(WL_NIB(w_super, b32_), qw_);
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
                    int32_t q[16];
                    #pragma unroll
                    for (unsigned int t = 0; t < 2u; t++) {
                        #pragma unroll
                        for (unsigned int i = 0; i < 4u; i++) {
                            q[8u * t + i]      = load_dword_i8_wl(q8b + 32u * t + 4u * i);
                            q[8u * t + 4u + i] = load_dword_i8_wl(q8b + 32u * t + 16u + 4u * i);
                        }
                    }
                    float f0 = 0.0f, f1 = 0.0f;
                    #pragma unroll
                    for (unsigned int t = 0; t < 2u; t++) {
                        int32_t s0 = 0, s1 = 0;
                        #pragma unroll
                        for (unsigned int i = 0; i < 8u; i++) {
                            s0 = sudot4_wl(w0[8u * t + i], q[8u * t + i], s0);
                            s1 = sudot4_wl(w1[8u * t + i], q[8u * t + i], s1);
                        }
                        f0 += sc0[t] * (float)s0;
                        f1 += sc1[t] * (float)s1;
                    }
                    pacc0[m] += yd * f0;
                    pacc1[m] += yd * f1;
                }
            }

            for (unsigned int m = 0; m < half_n; m++) {
                const float a0 = warp_sum_f32_wl(pacc0[m]);
                const float a1 = warp_sum_f32_wl(pacc1[m]);
                if (lane == 0) {
                    const int packed = expert_members[(size_t)e * (size_t)max_per_expert
                                                    + (size_t)(member_start + half_lo + m)];
                    const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
                    const unsigned int slot = (unsigned int)(packed       ) & 0xffffu;
                    const size_t out_base = ((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows;
                    partials[out_base + row0]      = a0;
                    partials[out_base + row0 + 1u] = a1;
                }
            }
        }
    }
}
