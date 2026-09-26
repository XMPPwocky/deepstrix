// A_moe_smallb/wl_gu — the UNMODIFIED production kwide gate/up kernel body with ONE change: the work-group
// loops over work items w = blockIdx.y, blockIdx.y + gridDim.y, ... < n_work_items (device count, or gridDim.y
// when the grid is exact), so the wrapper can cap grid.y at a small constant instead of launching the 6b
// upper bound. With the hub idle, the (6b - n_wi) x 288 EMPTY work-groups of the bound grid cost 25 us at b=8
// for kwide and 48 us for kwide2 (results/parts_idle_run1_b8.txt). Bit-exact: every work item runs the same
// code on the same WG shape; only the WG that runs it changes. Launched with grid.y = min(bound, WL).
#include "mxfp4_pair_matvec.hip"

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
        const int    * __restrict__ n_work_items_dev) { // nullptr = grid.y is exact
    const unsigned int n_items = (n_work_items_dev != nullptr) ? (unsigned int)max(0, *n_work_items_dev) : gridDim.y;

    __shared__ int8_t   s_lut[16];
    __shared__ uint4    s_q8v[MXFP4_KW_MAX_CHUNK * 2 * 16]; // [mi][sbh][16]
    __shared__ float    s_yd[MXFP4_KW_MAX_CHUNK * 2];       // [mi][sbh]
    __shared__ int      s_member_packed[MXFP4_KW_MAX_CHUNK];

    const unsigned int tid = threadIdx.x;
    if (tid < 16u) s_lut[tid] = kvalues_mxfp4_pair[tid];
    const unsigned int lane = tid & 31u;
    const unsigned int row  = blockIdx.x * (blockDim.x >> 5u) + (tid >> 5u);
    const bool row_valid = (row < n_rows);
    const unsigned int sbh  = lane >> 4;
    const unsigned int l15  = lane & 15u;
    const unsigned int b32  = l15 >> 1;
    const unsigned int half = l15 & 1u;
    const unsigned int n_pairs = n_blocks >> 1;

    for (unsigned int w_idx = blockIdx.y; w_idx < n_items; w_idx += gridDim.y) {
        __syncthreads();   // previous item's epilogue reads s_member_packed; LUT store above
        const int packed_item = work_items[w_idx];
        const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
        const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
        const int group_n = group_count[e];
        const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);
        if (member_end <= member_start) continue;   // uniform
        const unsigned int n_members = member_end - member_start;

        for (unsigned int i = tid; i < n_members; i += blockDim.x) {
            s_member_packed[i] = expert_members[(size_t)e * (size_t)max_per_expert
                                              + (size_t)(member_start + i)];
        }
        __syncthreads();

        const uint8_t* gate_e_base = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;
        const uint8_t* up_e_base   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;

        float lane_pacc_g[MXFP4_KW_MAX_CHUNK];
        float lane_pacc_u[MXFP4_KW_MAX_CHUNK];
        #pragma unroll
        for (unsigned int mi = 0; mi < MXFP4_KW_MAX_CHUNK; mi++) {
            lane_pacc_g[mi] = 0.0f;
            lane_pacc_u[mi] = 0.0f;
        }

        for (unsigned int sbp = 0; sbp < n_pairs; sbp++) {
            const unsigned int sb_lane = 2u * sbp + sbh;
            int32_t wg0 = 0, wg1 = 0, wg2 = 0, wg3 = 0;
            int32_t wu0 = 0, wu1 = 0, wu2 = 0, wu3 = 0;
            float gds = 0.0f, uds = 0.0f;
            if (row_valid) {
                const uint64_t off = (uint64_t)row * (uint64_t)n_blocks * SUPER_MXFP4_BYTES
                                   + (uint64_t)sb_lane * SUPER_MXFP4_BYTES;
                mxfp4_unpack16(gate_e_base + off, b32, half, s_lut, wg0, wg1, wg2, wg3, gds);
                mxfp4_unpack16(up_e_base   + off, b32, half, s_lut, wu0, wu1, wu2, wu3, uds);
            }
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
                        const unsigned int ii = r + 8u * j;
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
            if (row_valid) {
                const uint4* q8_lane = &s_q8v[sbh * 16u + l15];
                const float* yd_lane = &s_yd[sbh];
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
            __syncthreads();
        }

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
}
