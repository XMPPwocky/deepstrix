// B_moe_prefill: ABLATION variants of the production kwide gate+up kernel.
// Includes the unmodified in-tree source so every helper is byte-identical; the
// kernel body below is a verbatim copy of mxfp4_pair_matvec_fused_swiglu_kwide
// with one phase removed per variant (timing-only kernels, results are wrong):
//   ABL=0 full     : copy of production (must time like the baseline)
//   ABL=1 nodot    : the 8 sudot4 per member replaced by one xor (keeps the LDS
//                    read + the accumulate; removes ~7 VALU per member-lane)
//   ABL=2 noload   : weight bytes synthesised from indices (no weight VMEM;
//                    keeps the LUT unpack VALU + LDS traffic)
//   ABL=3 nostage  : the cooperative xq stage skipped (no activation VMEM /
//                    LDS stores; the dot reads whatever is in LDS)
//   ABL=4 nolut    : LUT lookups replaced by a shift (no LDS reads in the
//                    unpack; keeps the weight loads and the dot)
//   ABL=5 noload_nodot : 2 + 1 together (what is left = staging + LDS + control)
// NOTE: ABL 3 must still write s_q8v once (sb-pair 0): LLVM folds loads of never-stored
// LDS to undef and then DCEs the whole weight path (first attempt read 391 GB/s, impossible).
#include "mxfp4_pair_matvec.hip"

#define KW_ABL_BODY(NAME, ABL)                                                              \
extern "C" __global__ void NAME(                                                             \
        float        * __restrict__ mid, const uint8_t * __restrict__ gate_w_base,           \
        const uint8_t * __restrict__ up_w_base, const uint8_t * __restrict__ xq,             \
        const float  * __restrict__ expert_w, const int * __restrict__ group_count,           \
        const int    * __restrict__ expert_members, const int * __restrict__ work_items,     \
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used,                     \
        unsigned int max_per_expert, unsigned int chunk_size, float clamp,                   \
        unsigned int n_rows, unsigned int n_blocks, const int * __restrict__ n_work_items_dev) { \
    if (n_work_items_dev != nullptr && (int)blockIdx.y >= *n_work_items_dev) return;         \
    const unsigned int w_idx = blockIdx.y;                                                   \
    const int packed_item = work_items[w_idx];                                               \
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;           \
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;          \
    const int group_n = group_count[e];                                                      \
    const unsigned int member_end = min((unsigned int)group_n, member_start + chunk_size);   \
    if (member_end <= member_start) return;                                                  \
    const unsigned int n_members = member_end - member_start;                                \
    __shared__ int8_t   s_lut[16];                                                           \
    __shared__ uint4    s_q8v[MXFP4_KW_MAX_CHUNK * 2 * 16];                                  \
    __shared__ float    s_yd[MXFP4_KW_MAX_CHUNK * 2];                                        \
    __shared__ int      s_member_packed[MXFP4_KW_MAX_CHUNK];                                 \
    const unsigned int tid = threadIdx.x;                                                    \
    for (unsigned int i = tid; i < n_members; i += blockDim.x) {                             \
        s_member_packed[i] = expert_members[(size_t)e * (size_t)max_per_expert               \
                                          + (size_t)(member_start + i)];                     \
    }                                                                                        \
    if (tid < 16u) s_lut[tid] = kvalues_mxfp4_pair[tid];                                     \
    __syncthreads();                                                                         \
    const unsigned int lane = tid & 31u;                                                     \
    const unsigned int row  = blockIdx.x * (blockDim.x >> 5u) + (tid >> 5u);                 \
    const bool row_valid = (row < n_rows);                                                   \
    const unsigned int sbh  = lane >> 4;                                                     \
    const unsigned int l15  = lane & 15u;                                                    \
    const unsigned int b32  = l15 >> 1;                                                      \
    const unsigned int half = l15 & 1u;                                                      \
    const uint8_t* gate_e_base = gate_w_base + (uint64_t)e * (uint64_t)gate_bpe;             \
    const uint8_t* up_e_base   = up_w_base   + (uint64_t)e * (uint64_t)up_bpe;               \
    float lane_pacc_g[MXFP4_KW_MAX_CHUNK];                                                   \
    float lane_pacc_u[MXFP4_KW_MAX_CHUNK];                                                   \
    _Pragma("unroll")                                                                        \
    for (unsigned int mi = 0; mi < MXFP4_KW_MAX_CHUNK; mi++) {                               \
        lane_pacc_g[mi] = 0.0f; lane_pacc_u[mi] = 0.0f;                                      \
    }                                                                                        \
    const unsigned int n_pairs = n_blocks >> 1;                                              \
    for (unsigned int sbp = 0; sbp < n_pairs; sbp++) {                                       \
        const unsigned int sb_lane = 2u * sbp + sbh;                                         \
        int32_t wg0 = 0, wg1 = 0, wg2 = 0, wg3 = 0;                                          \
        int32_t wu0 = 0, wu1 = 0, wu2 = 0, wu3 = 0;                                          \
        float gds = 0.0f, uds = 0.0f;                                                        \
        if (row_valid) {                                                                     \
            const uint64_t off = (uint64_t)row * (uint64_t)n_blocks * SUPER_MXFP4_BYTES      \
                               + (uint64_t)sb_lane * SUPER_MXFP4_BYTES;                      \
            KW_ABL_UNPACK(ABL, gate_e_base + off, wg0, wg1, wg2, wg3, gds);                  \
            KW_ABL_UNPACK(ABL, up_e_base   + off, wu0, wu1, wu2, wu3, uds);                  \
        }                                                                                    \
        if (ABL != 3 || sbp == 0u) {  /* ABL 3: stage once so LDS is defined (not undef->DCE) */ \
            uint32_t* s_q8_dw = (uint32_t*)s_q8v;                                            \
            const unsigned int src_blk0 = 2u * sbp;                                          \
            const unsigned int mi = tid >> 3;                                                \
            const unsigned int r  = tid & 7u;                                                \
            if (mi < n_members) {                                                            \
                const int packed     = s_member_packed[mi];                                  \
                const unsigned int b = (unsigned int)(packed >> 16);                         \
                const uint32_t* xq_src_dw = (const uint32_t*)(                               \
                    xq + (size_t)b * (size_t)n_blocks * (size_t)BLOCK_Q8_K_BYTES             \
                       + (size_t)src_blk0 * (size_t)BLOCK_Q8_K_BYTES);                       \
                _Pragma("unroll")                                                            \
                for (unsigned int j = 0; j < 8u; j++) {                                      \
                    const unsigned int ii = r + 8u * j;                                      \
                    s_q8_dw[(mi * 2u + 0u) * 64u + ii] = xq_src_dw[1u + ii];                 \
                }                                                                            \
                _Pragma("unroll")                                                            \
                for (unsigned int j = 0; j < 8u; j++) {                                      \
                    const unsigned int ii = r + 8u * j;                                      \
                    s_q8_dw[(mi * 2u + 1u) * 64u + ii] = xq_src_dw[74u + ii];                \
                }                                                                            \
            }                                                                                \
            if (tid < 2u * n_members) {                                                      \
                const unsigned int myd = tid >> 1;                                           \
                const unsigned int h   = tid & 1u;                                           \
                const int packed     = s_member_packed[myd];                                 \
                const unsigned int b = (unsigned int)(packed >> 16);                         \
                const uint32_t* xq_src_dw = (const uint32_t*)(                               \
                    xq + (size_t)b * (size_t)n_blocks * (size_t)BLOCK_Q8_K_BYTES             \
                       + (size_t)src_blk0 * (size_t)BLOCK_Q8_K_BYTES);                       \
                s_yd[myd * 2u + h] = __uint_as_float(xq_src_dw[h * 73u]);                    \
            }                                                                                \
        }                                                                                    \
        __syncthreads();                                                                     \
        if (row_valid) {                                                                     \
            const uint4* q8_lane = &s_q8v[sbh * 16u + l15];                                  \
            const float* yd_lane = &s_yd[sbh];                                               \
            _Pragma("unroll 4")                                                              \
            for (unsigned int mi = 0; mi < n_members; mi++) {                                \
                const uint4 q = q8_lane[mi * 32u];                                           \
                int32_t sumi_g = 0, sumi_u = 0;                                              \
                if (ABL == 1 || ABL == 5) {                                                  \
                    sumi_g = wg0 ^ (int32_t)q.x;                                             \
                    sumi_u = wu0 ^ (int32_t)q.y;                                             \
                } else {                                                                     \
                    sumi_g = sudot4_pair(wg0, (int32_t)q.x, sumi_g);                         \
                    sumi_g = sudot4_pair(wg1, (int32_t)q.y, sumi_g);                         \
                    sumi_g = sudot4_pair(wg2, (int32_t)q.z, sumi_g);                         \
                    sumi_g = sudot4_pair(wg3, (int32_t)q.w, sumi_g);                         \
                    sumi_u = sudot4_pair(wu0, (int32_t)q.x, sumi_u);                         \
                    sumi_u = sudot4_pair(wu1, (int32_t)q.y, sumi_u);                         \
                    sumi_u = sudot4_pair(wu2, (int32_t)q.z, sumi_u);                         \
                    sumi_u = sudot4_pair(wu3, (int32_t)q.w, sumi_u);                         \
                }                                                                            \
                const float yd = yd_lane[mi * 2u];                                           \
                lane_pacc_g[mi] += (float)sumi_g * (yd * gds);                               \
                lane_pacc_u[mi] += (float)sumi_u * (yd * uds);                               \
            }                                                                                \
        }                                                                                    \
        __syncthreads();                                                                     \
    }                                                                                        \
    if (row_valid) {                                                                         \
        for (unsigned int mi = 0; mi < n_members; mi++) {                                    \
            float g_sum = warp_sum_f32_pair(lane_pacc_g[mi]);                                \
            float u_sum = warp_sum_f32_pair(lane_pacc_u[mi]);                                \
            if (lane == 0u) {                                                                \
                const int packed          = s_member_packed[mi];                             \
                const unsigned int b      = (unsigned int)(packed >> 16);                    \
                const unsigned int slot   = (unsigned int)(packed & 0xFFFFu);                \
                if (clamp > 1.0e-6f) {                                                       \
                    if (g_sum > clamp)  g_sum = clamp;                                       \
                    if (u_sum > clamp)  u_sum = clamp;                                       \
                    if (u_sum < -clamp) u_sum = -clamp;                                      \
                }                                                                            \
                const float ew  = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];       \
                const float sig = 1.0f / (1.0f + expf(-g_sum));                              \
                const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot)           \
                                     * (size_t)n_rows + (size_t)row;                         \
                mid[out_off] = g_sum * sig * u_sum * ew;                                     \
            }                                                                                \
        }                                                                                    \
    }                                                                                        \
}

// Unpack selector. ABL 2: synthesise the 16 nibble bytes + scale from the address
// bits (no VMEM). ABL 4: nibble -> int8 by shifting instead of the LDS LUT.
__device__ __forceinline__ void kw_unpack_noload(uint64_t addr, unsigned int b32, unsigned int half,
        int32_t& w0, int32_t& w1, int32_t& w2, int32_t& w3, float& ds, const int8_t* s_lut) {
    ds = e8m0_half_to_f32_pair((uint32_t)((addr >> 7) & 0x7u) + 120u);
    uint32_t qw_[4];
    #pragma unroll
    for (unsigned int i = 0; i < 4u; i++) qw_[i] = (uint32_t)(addr * 2654435761u) ^ (i * 0x9E3779B9u) ^ (b32 << 3);
    const unsigned int sh = 4u * half;
    int32_t w[4];
    #pragma unroll
    for (unsigned int i = 0; i < 4u; i++) {
        union { int32_t d; int8_t b[4]; } u;
        #pragma unroll
        for (unsigned int j = 0; j < 4u; j++) u.b[j] = s_lut[(((qw_[i] >> (8u * j)) & 0xFFu) >> sh) & 0x0Fu];
        w[i] = u.d;
    }
    w0 = w[0]; w1 = w[1]; w2 = w[2]; w3 = w[3];
}
__device__ __forceinline__ void kw_unpack_nolut(const uint8_t* w_super, unsigned int b32, unsigned int half,
        int32_t& w0, int32_t& w1, int32_t& w2, int32_t& w3, float& ds) {
    ds = e8m0_half_to_f32_pair(MXFP4_SCALE(w_super, b32));
    uint32_t qw_[4];
    mxfp4_load16(MXFP4_NIB(w_super, b32), qw_);
    const unsigned int sh = 4u * half;
    // nibble -> byte by masking (wrong values, same VALU shape minus the LDS lookups)
    w0 = (int32_t)((qw_[0] >> sh) & 0x0F0F0F0Fu);
    w1 = (int32_t)((qw_[1] >> sh) & 0x0F0F0F0Fu);
    w2 = (int32_t)((qw_[2] >> sh) & 0x0F0F0F0Fu);
    w3 = (int32_t)((qw_[3] >> sh) & 0x0F0F0F0Fu);
}
#define KW_ABL_UNPACK(ABL, PTR, W0, W1, W2, W3, DS)                                            \
    do {                                                                                     \
        if (ABL == 2 || ABL == 5) kw_unpack_noload((uint64_t)(PTR), b32, half, W0, W1, W2, W3, DS, s_lut); \
        else if (ABL == 4) kw_unpack_nolut((PTR), b32, half, W0, W1, W2, W3, DS);            \
        else               mxfp4_unpack16((PTR), b32, half, s_lut, W0, W1, W2, W3, DS);      \
    } while (0)

KW_ABL_BODY(kw_abl_full, 0)
KW_ABL_BODY(kw_abl_nodot, 1)
KW_ABL_BODY(kw_abl_noload, 2)
KW_ABL_BODY(kw_abl_nostage, 3)
KW_ABL_BODY(kw_abl_nolut, 4)
KW_ABL_BODY(kw_abl_noload_nodot, 5)
