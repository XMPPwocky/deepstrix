// laguna_moe_tiled.hip — by-expert, bandwidth-amortized MoE GEMM for Laguna
// prefill. The BxN kernels (q4_k_*_bxn / q6_k_*_bxn) re-read each expert's
// weight tile once per (token, slot) that routed to it → weight-BW scales with
// B*top_k. These by-expert kernels instead launch grid.y = expert id: each WG
// streams expert e's weight row-tile ONCE (DRAM→L2) and dots it against ALL
// tokens that routed to e (its group members, built by moe_group_builder), so
// per-chunk weight traffic = active-expert bytes (the BW roofline).
//
// Laguna dequant (Q4_K nibble unpack, Q6_K 6-bit unpack) is CHEAP, so we do
// NOT register-hold the decoded weight; we re-read per member and let L2 serve
// the reuse (mirrors ds4's shipped iq2 `_by_expert` and q2k_down by-expert).
// The activation for the current member is staged in LDS once per member.
//
// Kernels:
//   q4_k_gate_up_swiglu_tiled — fused gate×up×swiglu×ew → mid[B,n_used,n_rows]
//   q6_k_down_tiled / q4_k_down_tiled — down projection, atomicAdd-accumulated
//     into out[B, n_rows] (each token gets top_k contributions from top_k
//     distinct expert WGs → atomic; caller MUST zero `out` before launch).
//
// Layouts match the BxN path exactly so parity is a drop-in swap:
//   xq (gate/up) : [B, n_blocks_in*292] token-major
//   mid          : [B, n_used, n_rows]  token-major, slot-major
//   xq_mid (down): [B, n_used, n_blk_mid*292] token/slot-major
//   selected/expert_w : [B, n_used]
//   expert_members[e*max_per_expert + i] = packed (b<<16 | slot)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q4_K_BYTES 144
#define BLOCK_Q6_K_BYTES 210
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

// LDS activation staging caps (worst-case for Laguna dims):
//   gate/up: n_blocks_in = HIDDEN/256 = 12 → 12*292 = 3504 B
//   down   : n_blk_mid   = FF_EXP/256 = 4  → 4*292  = 1168 B
#define LDS_XQ_BYTES 3504

__device__ __forceinline__ float f16_to_f32_t(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4_t(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t load_dword_t(const uint8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}
__device__ __forceinline__ int32_t load_dword_i8_t(const int8_t* p) {
    return load_dword_t((const uint8_t*)p);
}

// Vectorized DRAM->LDS staging copy. Replaces the byte-granular
// `for(i=tid;i<n;i+=blockDim) dst[i]=src[i]` (uncoalesced, ~n single-byte
// transactions) with a uint4 (16B) bulk copy + scalar tail. Requires dst/src
// 16-byte aligned at their base (shared arrays are __align__(16); per-member
// DRAM offsets are multiples of 16 for Laguna dims). Bit-identical bytes.
__device__ __forceinline__ void stage_lds_vec(
        uint8_t* __restrict__ dst, const uint8_t* __restrict__ src,
        unsigned int nbytes, unsigned int tid, unsigned int nthreads) {
    const unsigned int n16 = nbytes >> 4;
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    for (unsigned int i = tid; i < n16; i += nthreads) d4[i] = s4[i];
    const unsigned int base = n16 << 4;
    for (unsigned int i = base + tid; i < nbytes; i += nthreads) dst[i] = src[i];
}

__device__ __forceinline__ float warp_sum_f32_t(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// 16-lane segmented reduction: reduces within each aligned [0,16)/[16,32) half
// of the wavefront. Result lands in the segment's lane 0 (global lane 0 and 16).
// Offsets start at 8 so no add ever crosses the 16-lane segment boundary.
__device__ __forceinline__ float warp_sum16_f32_t(float v) {
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Q4_K quarter-block dot (copy of dot_block_quarter_q4k) — sub in [0,4)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void get_scale_min_q4k_t(
        int j, const uint8_t* scales, uint8_t* d_out, uint8_t* m_out) {
    if (j < 4) {
        *d_out = scales[j]     & 0x3F;
        *m_out = scales[j + 4] & 0x3F;
    } else {
        *d_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m_out = (scales[j + 4] >>  4)  | ((scales[j - 0] >> 6) << 4);
    }
}

__device__ __forceinline__ int32_t dot_q4_32_t(
        const uint8_t* qs, const int8_t* q8, bool high) {
    int32_t sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; i += 4) {
        const int32_t qs_dword = load_dword_t(qs + i);
        const int32_t v = high
            ? (int32_t)((((uint32_t)qs_dword) >> 4) & 0x0f0f0f0fu)
            : (qs_dword & 0x0f0f0f0f);
        sum = sudot4_t(v, load_dword_i8_t(q8 + i), sum);
    }
    return sum;
}

__device__ __forceinline__ float dot_block_quarter_q4k_t(
        const uint8_t* x_blk, const uint8_t* y_blk, unsigned int sub) {
    const uint16_t xd_bits   = *(const uint16_t*)(x_blk + 0);
    const uint16_t xdm_bits  = *(const uint16_t*)(x_blk + 2);
    const uint8_t* scales    = x_blk + 4;
    const uint8_t* qs_base   = x_blk + 16;

    const float    yd        = *(const float*)(y_blk + 0);
    const int8_t*  q8_base   = (const int8_t*)(y_blk + 4);
    const int16_t* bsums     = (const int16_t*)(y_blk + 260);

    const float d_super = yd * f16_to_f32_t(xd_bits);
    const float m_super = yd * f16_to_f32_t(xdm_bits);

    const int j_a = (int)(2u * sub);
    const int j_b = j_a + 1;
    uint8_t d_a, m_a, d_b, m_b;
    get_scale_min_q4k_t(j_a, scales, &d_a, &m_a);
    get_scale_min_q4k_t(j_b, scales, &d_b, &m_b);

    const uint8_t* qs_chunk = qs_base + (size_t)sub * 32u;
    const int8_t*  q8_chunk = q8_base + (size_t)sub * 64u;
    const int32_t bsum_a = (int32_t)bsums[4u*sub+0u] + (int32_t)bsums[4u*sub+1u];
    const int32_t bsum_b = (int32_t)bsums[4u*sub+2u] + (int32_t)bsums[4u*sub+3u];
    const int32_t dot_a = dot_q4_32_t(qs_chunk, q8_chunk,      false);
    const int32_t dot_b = dot_q4_32_t(qs_chunk, q8_chunk + 32, true);
    return d_super * ((float)d_a * (float)dot_a + (float)d_b * (float)dot_b)
         - m_super * ((float)m_a * (float)bsum_a + (float)m_b * (float)bsum_b);
}

// ---------------------------------------------------------------------------
// Q6_K quarter-block dot (copy of dot_block_quarter_q6k) — sub in [0,4)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dot_block_quarter_q6k_t(
        const uint8_t* x_blk, const uint8_t* y_blk, unsigned int sub) {
    const uint8_t* ql        = x_blk + 0;
    const uint8_t* qh        = x_blk + 128;
    const int8_t*  sc        = (const int8_t*)(x_blk + 192);
    const uint16_t d_bits    = *(const uint16_t*)(x_blk + 208);

    const float    yd        = *(const float*)(y_blk + 0);
    const int8_t*  q8_base   = (const int8_t*)(y_blk + 4);
    const float    d_all     = yd * f16_to_f32_t(d_bits);

    const unsigned int half  = sub >> 1u;
    const unsigned int upper = sub & 1u;
    const uint8_t* qlh = ql + (size_t)half * 64u;
    const uint8_t* qhh = qh + (size_t)half * 32u;
    const int8_t*  q8  = q8_base + (size_t)half * 128u + (size_t)upper * 64u;
    const int8_t*  sch = sc + (size_t)half * 8u + (size_t)upper * 4u;

    int32_t isum = 0;
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        int32_t dot = 0;
        #pragma unroll
        for (int q = 0; q < 16; q += 4) {
            const int r0 = g * 16 + q;
            uint32_t apack = 0u;
            #pragma unroll
            for (int t = 0; t < 4; t++) {
                const int r = r0 + t;
                const unsigned int shift = upper * 4u + ((r >= 32) ? 2u : 0u);
                const uint8_t nib = upper ? (uint8_t)(qlh[r] >> 4)
                                          : (uint8_t)(qlh[r] & 0x0Fu);
                const uint8_t hi2 = (uint8_t)((qhh[r & 31] >> shift) & 3u);
                const int8_t  av  = (int8_t)((int)(nib | (hi2 << 4)) - 32);
                apack |= ((uint32_t)(uint8_t)av) << (t * 8);
            }
            dot = sudot4_t((int32_t)apack, load_dword_i8_t(q8 + r0), dot);
        }
        isum += (int32_t)sch[g] * dot;
    }
    return d_all * (float)isum;
}

// ===========================================================================
// q4_k_gate_up_swiglu_tiled — by-expert fused gate×up×swiglu×ew.
//   grid:  (n_rows/8, n_expert, 1)   block: (256,1,1)
//   Weight rows for (expert e, row_block) read once (DRAM→L2), reused over all
//   group members. Activation for the current member staged in LDS.
// ===========================================================================
extern "C" __global__ void q4_k_gate_up_swiglu_tiled(
        float          * __restrict__ mid,           // [B, n_used, n_rows]
        const uint8_t  * __restrict__ gate_w_base,
        const uint8_t  * __restrict__ up_w_base,
        const uint8_t  * __restrict__ xq,            // [B, n_blocks_in*292]
        const float    * __restrict__ expert_w,      // [B, n_used]
        const int      * __restrict__ group_count,   // [n_expert]
        const int      * __restrict__ expert_members,// [n_expert * max_per_expert]
        unsigned int     gate_bpe,
        unsigned int     up_bpe,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        float            clamp,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;

    __shared__ uint8_t s_xq[LDS_XQ_BYTES];

    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool         row_ok       = (row < n_rows);

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;

    const unsigned int xq_bytes = n_blocks_in * BLOCK_Q8_K_BYTES;
    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e * (size_t)max_per_expert + (size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);

        const uint8_t* xq_b = xq + (size_t)b * (size_t)xq_bytes;
        for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) s_xq[i] = xq_b[i];
        __syncthreads();

        if (row_ok) {
            float g_sum = 0.0f, u_sum = 0.0f;
            for (unsigned int bb = block_lane; bb < n_blocks_in; bb += 8u) {
                const uint8_t* w_g = gate_row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
                const uint8_t* w_u = up_row_base   + (uint64_t)bb * BLOCK_Q4_K_BYTES;
                const uint8_t* y   = s_xq          + (size_t)bb   * BLOCK_Q8_K_BYTES;
                g_sum += dot_block_quarter_q4k_t(w_g, y, sub);
                u_sum += dot_block_quarter_q4k_t(w_u, y, sub);
            }
            g_sum = warp_sum_f32_t(g_sum);
            u_sum = warp_sum_f32_t(u_sum);
            if (lane == 0) {
                const float ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
                if (clamp > 1.0e-6f) {
                    if (g_sum > clamp)  g_sum = clamp;
                    if (u_sum > clamp)  u_sum = clamp;
                    if (u_sum < -clamp) u_sum = -clamp;
                }
                const float sig = 1.0f / (1.0f + expf(-g_sum));
                mid[((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows + (size_t)row]
                    = g_sum * sig * u_sum * ew;
            }
        }
        __syncthreads(); // before next member's xq overwrite
    }
}

// ===========================================================================
// down tiled — by-expert down projection, atomicAdd-accumulated into out.
//   grid:  (n_rows/8, n_expert, 1)   block: (256,1,1)
//   out MUST be pre-zeroed (each token accumulates top_k expert contributions
//   from top_k distinct WGs). The routing weight is already baked into `mid`
//   (hence into xq_mid) by the gate/up kernel, so down is weight-free-sum.
// ===========================================================================
#define DOWN_TILED_BODY(DOTFN)                                                        \
    const unsigned int e = blockIdx.y;                                                \
    __shared__ int n_members;                                                         \
    if (threadIdx.x == 0) n_members = group_count[e];                                 \
    __syncthreads();                                                                  \
    if (n_members == 0) return;                                                       \
    __shared__ uint8_t s_xq[LDS_XQ_BYTES];                                            \
    const unsigned int tid          = threadIdx.x;                                    \
    const unsigned int row_in_block = tid / 32u;                                      \
    const unsigned int lane         = tid & 31u;                                      \
    const unsigned int row          = blockIdx.x * 8u + row_in_block;                 \
    const bool         row_ok       = (row < n_rows);                                 \
    const unsigned int block_lane   = lane >> 2u;                                     \
    const unsigned int sub          = lane & 3u;                                      \
    const uint8_t* row_base = w_base + (uint64_t)e * (uint64_t)dbpe                    \
                            + (uint64_t)row * n_blocks_in * BLOCK_BYTES;              \
    for (int m = 0; m < n_members; m++) {                                             \
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];\
        const unsigned int b    = (unsigned int)(packed >> 16);                       \
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);                   \
        const uint8_t* xq_slot = xq_base                                              \
            + ((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)xq_slot_stride;   \
        for (unsigned int i = tid; i < xq_slot_stride; i += blockDim.x) s_xq[i]=xq_slot[i];\
        __syncthreads();                                                              \
        if (row_ok) {                                                                 \
            float acc = 0.0f;                                                         \
            for (unsigned int bb = block_lane; bb < n_blocks_in; bb += 8u) {          \
                const uint8_t* w_blk = row_base + (uint64_t)bb * BLOCK_BYTES;         \
                const uint8_t* y_blk = s_xq     + (size_t)bb   * BLOCK_Q8_K_BYTES;    \
                acc += DOTFN(w_blk, y_blk, sub);                                      \
            }                                                                         \
            acc = warp_sum_f32_t(acc);                                                \
            if (lane == 0) atomicAdd(&out[(size_t)b * (size_t)n_rows + (size_t)row], acc);\
        }                                                                             \
        __syncthreads();                                                             \
    }

extern "C" __global__ void q6_k_down_tiled(
        float          * __restrict__ out,           // [B, n_rows] — pre-zeroed
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,       // [B, n_used, xq_slot_stride]
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int BLOCK_BYTES = BLOCK_Q6_K_BYTES;
    DOWN_TILED_BODY(dot_block_quarter_q6k_t)
}

extern "C" __global__ void q4_k_down_tiled(
        float          * __restrict__ out,           // [B, n_rows] — pre-zeroed
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int BLOCK_BYTES = BLOCK_Q4_K_BYTES;
    DOWN_TILED_BODY(dot_block_quarter_q4k_t)
}

// ===========================================================================
// REGISTER-TILED variants. The naive tiled kernels above re-read + re-DECODE
// each expert's weight quarter once PER member (served from L2), so per-token
// time stays ~flat and the DRAM-read-once win never lands (measured ~1.4x, 7%
// of BW roofline at B=512). These variants decode each lane's owned weight
// quarter ONCE into registers, then stream all members through a bare dp4a —
// weight bytes touched (DRAM+L2) once per chunk, per-member cost = activation
// read + dp4a. This is the actual BW/amortization lever.
//   MAXB = ceil(max_n_blocks_in / 8): gate/up n_blocks_in=12 → 2; down =4 → 1.
// ===========================================================================
#define MAXB 2

// ---- Q4_K gate/up, register-held weight ----
extern "C" __global__ void q4_k_gate_up_swiglu_tiled_reg(
        float          * __restrict__ mid,
        const uint8_t  * __restrict__ gate_w_base,
        const uint8_t  * __restrict__ up_w_base,
        const uint8_t  * __restrict__ xq,
        const float    * __restrict__ expert_w,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     gate_bpe,
        unsigned int     up_bpe,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        float            clamp,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;

    __shared__ uint8_t s_xq[LDS_XQ_BYTES];

    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool         row_ok       = (row < n_rows);
    const unsigned int block_lane   = lane >> 2u;   // 0..7
    const unsigned int sub          = lane & 3u;    // 0..3

    const uint8_t* gate_row_base = gate_w_base + (uint64_t)e*gate_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
    const uint8_t* up_row_base   = up_w_base   + (uint64_t)e*up_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;

    // --- decode this lane's owned quarters ONCE into registers ---
    unsigned int bb_of[MAXB];
    bool  valid[MAXB];
    float g_xd[MAXB], g_xdm[MAXB], u_xd[MAXB], u_xdm[MAXB];
    int   g_da[MAXB], g_ma[MAXB], g_db[MAXB], g_mb[MAXB];
    int   u_da[MAXB], u_ma[MAXB], u_db[MAXB], u_mb[MAXB];
    uint32_t g_qs[MAXB][8];
    uint32_t u_qs[MAXB][8];
    #pragma unroll
    for (int k = 0; k < MAXB; k++) {
        const unsigned int bb = block_lane + (unsigned int)k * 8u;
        bb_of[k] = bb;
        valid[k] = row_ok && (bb < n_blocks_in);
        if (!valid[k]) continue;
        const uint8_t* wg = gate_row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        const uint8_t* wu = up_row_base   + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        g_xd[k]  = f16_to_f32_t(*(const uint16_t*)(wg + 0));
        g_xdm[k] = f16_to_f32_t(*(const uint16_t*)(wg + 2));
        u_xd[k]  = f16_to_f32_t(*(const uint16_t*)(wu + 0));
        u_xdm[k] = f16_to_f32_t(*(const uint16_t*)(wu + 2));
        uint8_t da,ma,db,mb;
        get_scale_min_q4k_t(2*sub,   wg+4, &da,&ma); get_scale_min_q4k_t(2*sub+1, wg+4, &db,&mb);
        g_da[k]=da; g_ma[k]=ma; g_db[k]=db; g_mb[k]=mb;
        get_scale_min_q4k_t(2*sub,   wu+4, &da,&ma); get_scale_min_q4k_t(2*sub+1, wu+4, &db,&mb);
        u_da[k]=da; u_ma[k]=ma; u_db[k]=db; u_mb[k]=mb;
        const uint8_t* gqs = wg + 16 + (size_t)sub*32u;
        const uint8_t* uqs = wu + 16 + (size_t)sub*32u;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            g_qs[k][i] = (uint32_t)load_dword_t(gqs + i*4);
            u_qs[k][i] = (uint32_t)load_dword_t(uqs + i*4);
        }
    }

    const unsigned int xq_bytes = n_blocks_in * BLOCK_Q8_K_BYTES;
    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* xq_b = xq + (size_t)b * (size_t)xq_bytes;
        for (unsigned int i = tid; i < xq_bytes; i += blockDim.x) s_xq[i] = xq_b[i];
        __syncthreads();

        if (row_ok) {
            float g_sum = 0.0f, u_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < MAXB; k++) {
                if (!valid[k]) continue;
                const uint8_t* y = s_xq + (size_t)bb_of[k] * BLOCK_Q8_K_BYTES;
                const float yd = *(const float*)(y + 0);
                const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
                const int16_t* bs = (const int16_t*)(y + 260);
                int gda=0, gdb=0, uda=0, udb=0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                    const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
                    gda = sudot4_t((int32_t)(g_qs[k][i] & 0x0f0f0f0fu), q8lo, gda);
                    gdb = sudot4_t((int32_t)((g_qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, gdb);
                    uda = sudot4_t((int32_t)(u_qs[k][i] & 0x0f0f0f0fu), q8lo, uda);
                    udb = sudot4_t((int32_t)((u_qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, udb);
                }
                const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
                const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
                g_sum += yd*g_xd[k]*((float)g_da[k]*gda + (float)g_db[k]*gdb)
                       - yd*g_xdm[k]*((float)g_ma[k]*bsa + (float)g_mb[k]*bsb);
                u_sum += yd*u_xd[k]*((float)u_da[k]*uda + (float)u_db[k]*udb)
                       - yd*u_xdm[k]*((float)u_ma[k]*bsa + (float)u_mb[k]*bsb);
            }
            g_sum = warp_sum_f32_t(g_sum);
            u_sum = warp_sum_f32_t(u_sum);
            if (lane == 0) {
                const float ew = expert_w[(size_t)b*(size_t)n_used+(size_t)slot];
                if (clamp > 1.0e-6f) {
                    if (g_sum > clamp)  g_sum = clamp;
                    if (u_sum > clamp)  u_sum = clamp;
                    if (u_sum < -clamp) u_sum = -clamp;
                }
                const float sig = 1.0f/(1.0f+expf(-g_sum));
                mid[((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)n_rows+(size_t)row]
                    = g_sum*sig*u_sum*ew;
            }
        }
        __syncthreads();
    }
}

// ---- Q6_K down, register-held weight (n_blocks_in=4 → 1 block/lane) ----
extern "C" __global__ void q6_k_down_tiled_reg(
        float          * __restrict__ out,           // [B, n_rows] — pre-zeroed
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;
    __shared__ uint8_t s_xq[LDS_XQ_BYTES];

    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool         row_ok       = (row < n_rows);
    const unsigned int block_lane   = lane >> 2u;
    const unsigned int sub          = lane & 3u;

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q6_K_BYTES;

    // decode this lane's owned quarters (MAXB) once -> apack[k][16], sch[k][4], d[k]
    unsigned int bb_of[MAXB];
    bool  valid[MAXB];
    uint32_t apack[MAXB][16];
    int   sch[MAXB][4];
    float dq[MAXB];
    #pragma unroll
    for (int k = 0; k < MAXB; k++) {
        const unsigned int bb = block_lane + (unsigned int)k*8u;
        bb_of[k] = bb;
        valid[k] = row_ok && (bb < n_blocks_in);
        if (!valid[k]) continue;
        const uint8_t* x_blk = row_base + (uint64_t)bb * BLOCK_Q6_K_BYTES;
        const uint8_t* ql = x_blk + 0;
        const uint8_t* qh = x_blk + 128;
        const int8_t*  sc = (const int8_t*)(x_blk + 192);
        dq[k] = f16_to_f32_t(*(const uint16_t*)(x_blk + 208));
        const unsigned int half  = sub >> 1u;
        const unsigned int upper = sub & 1u;
        const uint8_t* qlh = ql + (size_t)half*64u;
        const uint8_t* qhh = qh + (size_t)half*32u;
        const int8_t*  sch4 = sc + (size_t)half*8u + (size_t)upper*4u;
        #pragma unroll
        for (int g = 0; g < 4; g++) sch[k][g] = (int)sch4[g];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            #pragma unroll
            for (int qq = 0; qq < 16; qq += 4) {
                const int r0 = g*16 + qq;
                uint32_t ap = 0u;
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    const int r = r0 + t;
                    const unsigned int shift = upper*4u + ((r >= 32) ? 2u : 0u);
                    const uint8_t nib = upper ? (uint8_t)(qlh[r] >> 4) : (uint8_t)(qlh[r] & 0x0Fu);
                    const uint8_t hi2 = (uint8_t)((qhh[r & 31] >> shift) & 3u);
                    const int8_t  av  = (int8_t)((int)(nib | (hi2 << 4)) - 32);
                    ap |= ((uint32_t)(uint8_t)av) << (t*8);
                }
                apack[k][g*4 + qq/4] = ap;
            }
        }
    }

    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* xq_slot = xq_base
            + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride;
        for (unsigned int i = tid; i < xq_slot_stride; i += blockDim.x) s_xq[i]=xq_slot[i];
        __syncthreads();

        if (row_ok) {
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < MAXB; k++) {
                if (!valid[k]) continue;
                const uint8_t* y = s_xq + (size_t)bb_of[k]*BLOCK_Q8_K_BYTES;
                const float yd = *(const float*)(y + 0);
                const unsigned int half = sub >> 1u, upper = sub & 1u;
                const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)half*128u + (size_t)upper*64u;
                int isum = 0;
                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int dot = 0;
                    #pragma unroll
                    for (int qq = 0; qq < 16; qq += 4) {
                        dot = sudot4_t((int32_t)apack[k][g*4 + qq/4],
                                       load_dword_i8_t(q8 + g*16 + qq), dot);
                    }
                    isum += sch[k][g] * dot;
                }
                acc += yd * dq[k] * (float)isum;
            }
            acc = warp_sum_f32_t(acc);
            if (lane == 0) atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc);
        }
        __syncthreads();
    }
}

// ===========================================================================
// WIN #2 — FULL-WARP down kernels. The reg/non-reg down kernels above run
// n_blocks_in=4 → block_lane 0..3 active, block_lane 4..7 (16 of 32 lanes)
// IDLE. On RDNA the masked lanes still consume the VALU issue slot, so half
// the warp is wasted. These variants pack TWO output rows per 32-lane wave
// (lanes 0..15 = row_a, lanes 16..31 = row_b), each covering the full 4
// blocks × 4 subs = 16 useful lanes. 16 rows/WG (was 8): grid.x = n_rows/16.
// The staged activation is shared across all 16 rows (was 8) so per-row LDS
// traffic halves too. Reduction is a 16-lane segmented tree.
// ===========================================================================

// ---- Q6_K down, register-held weight, full-warp (2 rows/wave) ----
extern "C" __global__ void q6_k_down_tiled_reg_w32(
        float          * __restrict__ out,           // [B, n_rows] — pre-zeroed
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;
    __shared__ uint8_t s_xq[LDS_XQ_BYTES];

    const unsigned int tid   = threadIdx.x;
    const unsigned int wave  = tid >> 5;         // 0..7
    const unsigned int lane  = tid & 31u;
    const unsigned int half  = lane >> 4u;       // 0/1 -> which row of the pair
    const unsigned int local = lane & 15u;       // 0..15 within the row
    const unsigned int block_lane = local >> 2u; // 0..3 == block index (n_blocks_in=4)
    const unsigned int sub        = local & 3u;  // 0..3 quarter
    const unsigned int row = blockIdx.x * 16u + wave * 2u + half;
    const bool row_ok = (row < n_rows);
    const unsigned int bb = block_lane;          // one block per lane, no k-loop
    const bool valid = row_ok && (bb < n_blocks_in);

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q6_K_BYTES;

    // decode this lane's owned block/quarter ONCE
    uint32_t apack[16];
    int      sch[4];
    float    dq = 0.0f;
    if (valid) {
        const uint8_t* x_blk = row_base + (uint64_t)bb * BLOCK_Q6_K_BYTES;
        const uint8_t* ql = x_blk + 0;
        const uint8_t* qh = x_blk + 128;
        const int8_t*  sc = (const int8_t*)(x_blk + 192);
        dq = f16_to_f32_t(*(const uint16_t*)(x_blk + 208));
        const unsigned int hf = sub >> 1u, up = sub & 1u;
        const uint8_t* qlh = ql + (size_t)hf*64u;
        const uint8_t* qhh = qh + (size_t)hf*32u;
        const int8_t*  sch4 = sc + (size_t)hf*8u + (size_t)up*4u;
        #pragma unroll
        for (int g = 0; g < 4; g++) sch[g] = (int)sch4[g];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            #pragma unroll
            for (int qq = 0; qq < 16; qq += 4) {
                const int r0 = g*16 + qq;
                uint32_t ap = 0u;
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    const int r = r0 + t;
                    const unsigned int shift = up*4u + ((r >= 32) ? 2u : 0u);
                    const uint8_t nib = up ? (uint8_t)(qlh[r] >> 4) : (uint8_t)(qlh[r] & 0x0Fu);
                    const uint8_t hi2 = (uint8_t)((qhh[r & 31] >> shift) & 3u);
                    const int8_t  av  = (int8_t)((int)(nib | (hi2 << 4)) - 32);
                    ap |= ((uint32_t)(uint8_t)av) << (t*8);
                }
                apack[g*4 + qq/4] = ap;
            }
        }
    }

    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* xq_slot = xq_base
            + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride;
        for (unsigned int i = tid; i < xq_slot_stride; i += blockDim.x) s_xq[i]=xq_slot[i];
        __syncthreads();

        float acc = 0.0f;
        if (valid) {
            const uint8_t* y = s_xq + (size_t)bb*BLOCK_Q8_K_BYTES;
            const float yd = *(const float*)(y + 0);
            const unsigned int hf = sub >> 1u, up = sub & 1u;
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)hf*128u + (size_t)up*64u;
            int isum = 0;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                int dot = 0;
                #pragma unroll
                for (int qq = 0; qq < 16; qq += 4) {
                    dot = sudot4_t((int32_t)apack[g*4 + qq/4],
                                   load_dword_i8_t(q8 + g*16 + qq), dot);
                }
                isum += sch[g] * dot;
            }
            acc = yd * dq * (float)isum;
        }
        acc = warp_sum16_f32_t(acc);
        if (local == 0 && row_ok) atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc);
        __syncthreads();
    }
}

// ---- Q4_K down, re-decode per member (non-reg), full-warp (2 rows/wave) ----
extern "C" __global__ void q4_k_down_tiled_w32(
        float          * __restrict__ out,           // [B, n_rows] — pre-zeroed
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;
    __shared__ uint8_t s_xq[LDS_XQ_BYTES];

    const unsigned int tid   = threadIdx.x;
    const unsigned int wave  = tid >> 5;
    const unsigned int lane  = tid & 31u;
    const unsigned int half  = lane >> 4u;
    const unsigned int local = lane & 15u;
    const unsigned int block_lane = local >> 2u;
    const unsigned int sub        = local & 3u;
    const unsigned int row = blockIdx.x * 16u + wave * 2u + half;
    const bool row_ok = (row < n_rows);
    const unsigned int bb = block_lane;
    const bool valid = row_ok && (bb < n_blocks_in);

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;

    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* xq_slot = xq_base
            + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride;
        for (unsigned int i = tid; i < xq_slot_stride; i += blockDim.x) s_xq[i]=xq_slot[i];
        __syncthreads();

        float acc = 0.0f;
        if (valid) {
            const uint8_t* w_blk = row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
            const uint8_t* y_blk = s_xq     + (size_t)bb   * BLOCK_Q8_K_BYTES;
            acc = dot_block_quarter_q4k_t(w_blk, y_blk, sub);
        }
        acc = warp_sum16_f32_t(acc);
        if (local == 0 && row_ok) atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc);
        __syncthreads();
    }
}

// ===========================================================================
// WIN #3 diagnostic — _NODOT floor. Identical to q6_k_down_tiled_reg (the
// PRODUCTION reg down) but the dp4a MAC chain is stubbed: it still stages the
// activation in LDS per member, still __syncthreads, still reduces + atomics,
// but skips the sudot4 accumulation. Measures the LDS-stage + barrier floor
// (the ceiling any compute change on this layout can hit). NOT for production.
// ===========================================================================
extern "C" __global__ void q6_k_down_tiled_reg_nodot(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;
    __shared__ uint8_t s_xq[LDS_XQ_BYTES];
    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool         row_ok       = (row < n_rows);
    const unsigned int block_lane   = lane >> 2u;
    const unsigned int sub          = lane & 3u;
    const unsigned int bb = block_lane;
    const bool valid = row_ok && (bb < n_blocks_in);
    (void)w_base; (void)dbpe; (void)sub;

    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* xq_slot = xq_base
            + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride;
        for (unsigned int i = tid; i < xq_slot_stride; i += blockDim.x) s_xq[i]=xq_slot[i];
        __syncthreads();
        float acc = 0.0f;
        if (valid) {
            // read just the block header from LDS so the stage isn't DCE'd,
            // but skip the dp4a MAC chain.
            const uint8_t* y = s_xq + (size_t)bb*BLOCK_Q8_K_BYTES;
            acc = *(const float*)(y + 0);
        }
        acc = warp_sum_f32_t(acc);
        if (lane == 0 && row_ok) atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc);
        __syncthreads();
    }
}

// ===========================================================================
// WIN #4 — COLUMN-TILED reg kernels. The _reg / _reg_w32 kernels above stage
// ONE member's activation in LDS, __syncthreads, compute, __syncthreads — a
// barrier PER member. The _NODOT diagnostic proved that per-member LDS-stage +
// barrier is ~40% of the kernel. These variants stage NT_COL members'
// activations in LDS in one cooperative burst (ONE barrier), then stream all
// NT_COL columns through the register-held weight (weight regs reused across
// the whole tile). Barrier count drops n_members -> ceil(n_members/NT_COL),
// and the big coalesced multi-member load hides DRAM latency far better than
// the serialized one-member-at-a-time load. This is the ejpir "multi-column
// register tiling" lever — read weight once, reuse across many token columns.
// ===========================================================================
#ifndef NT_COL
#define NT_COL 8
#endif

// ---- Q4_K gate/up, register-held weight, NT_COL columns per barrier ----
extern "C" __global__ void q4_k_gate_up_swiglu_tiled_reg_col(
        float          * __restrict__ mid,
        const uint8_t  * __restrict__ gate_w_base,
        const uint8_t  * __restrict__ up_w_base,
        const uint8_t  * __restrict__ xq,
        const float    * __restrict__ expert_w,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     gate_bpe,
        unsigned int     up_bpe,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        float            clamp,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;

    __shared__ __align__(16) uint8_t s_xq[NT_COL * LDS_XQ_BYTES];

    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool         row_ok       = (row < n_rows);
    const unsigned int block_lane   = lane >> 2u;   // 0..7
    const unsigned int sub          = lane & 3u;    // 0..3

    const uint8_t* gate_row_base = gate_w_base + (uint64_t)e*gate_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
    const uint8_t* up_row_base   = up_w_base   + (uint64_t)e*up_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;

    // --- decode this lane's owned quarters ONCE into registers ---
    unsigned int bb_of[MAXB];
    bool  valid[MAXB];
    float g_xd[MAXB], g_xdm[MAXB], u_xd[MAXB], u_xdm[MAXB];
    int   g_da[MAXB], g_ma[MAXB], g_db[MAXB], g_mb[MAXB];
    int   u_da[MAXB], u_ma[MAXB], u_db[MAXB], u_mb[MAXB];
    uint32_t g_qs[MAXB][8];
    uint32_t u_qs[MAXB][8];
    #pragma unroll
    for (int k = 0; k < MAXB; k++) {
        const unsigned int bb = block_lane + (unsigned int)k * 8u;
        bb_of[k] = bb;
        valid[k] = row_ok && (bb < n_blocks_in);
        if (!valid[k]) continue;
        const uint8_t* wg = gate_row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        const uint8_t* wu = up_row_base   + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        g_xd[k]  = f16_to_f32_t(*(const uint16_t*)(wg + 0));
        g_xdm[k] = f16_to_f32_t(*(const uint16_t*)(wg + 2));
        u_xd[k]  = f16_to_f32_t(*(const uint16_t*)(wu + 0));
        u_xdm[k] = f16_to_f32_t(*(const uint16_t*)(wu + 2));
        uint8_t da,ma,db,mb;
        get_scale_min_q4k_t(2*sub,   wg+4, &da,&ma); get_scale_min_q4k_t(2*sub+1, wg+4, &db,&mb);
        g_da[k]=da; g_ma[k]=ma; g_db[k]=db; g_mb[k]=mb;
        get_scale_min_q4k_t(2*sub,   wu+4, &da,&ma); get_scale_min_q4k_t(2*sub+1, wu+4, &db,&mb);
        u_da[k]=da; u_ma[k]=ma; u_db[k]=db; u_mb[k]=mb;
        const uint8_t* gqs = wg + 16 + (size_t)sub*32u;
        const uint8_t* uqs = wu + 16 + (size_t)sub*32u;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            g_qs[k][i] = (uint32_t)load_dword_t(gqs + i*4);
            u_qs[k][i] = (uint32_t)load_dword_t(uqs + i*4);
        }
    }

    const unsigned int xq_bytes = n_blocks_in * BLOCK_Q8_K_BYTES;
    for (int m0 = 0; m0 < n_members; m0 += NT_COL) {
        const int nt = min((int)NT_COL, n_members - m0);
        // cooperative load of NT_COL members' activations (one big burst).
        for (int j = 0; j < nt; j++) {
            const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)(m0+j)];
            const unsigned int b = (unsigned int)(packed >> 16);
            const uint8_t* xq_b = xq + (size_t)b * (size_t)xq_bytes;
            uint8_t* dst = s_xq + (size_t)j * LDS_XQ_BYTES;
            stage_lds_vec(dst, xq_b, xq_bytes, tid, blockDim.x);
        }
        __syncthreads();

        if (row_ok) {
            #pragma unroll 1
            for (int j = 0; j < nt; j++) {
                const uint8_t* xj = s_xq + (size_t)j * LDS_XQ_BYTES;
                float g_sum = 0.0f, u_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < MAXB; k++) {
                    if (!valid[k]) continue;
                    const uint8_t* y = xj + (size_t)bb_of[k] * BLOCK_Q8_K_BYTES;
                    const float yd = *(const float*)(y + 0);
                    const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
                    const int16_t* bs = (const int16_t*)(y + 260);
                    int gda=0, gdb=0, uda=0, udb=0;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                        const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
                        gda = sudot4_t((int32_t)(g_qs[k][i] & 0x0f0f0f0fu), q8lo, gda);
                        gdb = sudot4_t((int32_t)((g_qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, gdb);
                        uda = sudot4_t((int32_t)(u_qs[k][i] & 0x0f0f0f0fu), q8lo, uda);
                        udb = sudot4_t((int32_t)((u_qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, udb);
                    }
                    const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
                    const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
                    g_sum += yd*g_xd[k]*((float)g_da[k]*gda + (float)g_db[k]*gdb)
                           - yd*g_xdm[k]*((float)g_ma[k]*bsa + (float)g_mb[k]*bsb);
                    u_sum += yd*u_xd[k]*((float)u_da[k]*uda + (float)u_db[k]*udb)
                           - yd*u_xdm[k]*((float)u_ma[k]*bsa + (float)u_mb[k]*bsb);
                }
                g_sum = warp_sum_f32_t(g_sum);
                u_sum = warp_sum_f32_t(u_sum);
                if (lane == 0) {
                    const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)(m0+j)];
                    const unsigned int b    = (unsigned int)(packed >> 16);
                    const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
                    const float ew = expert_w[(size_t)b*(size_t)n_used+(size_t)slot];
                    if (clamp > 1.0e-6f) {
                        if (g_sum > clamp)  g_sum = clamp;
                        if (u_sum > clamp)  u_sum = clamp;
                        if (u_sum < -clamp) u_sum = -clamp;
                    }
                    const float sig = 1.0f/(1.0f+expf(-g_sum));
                    mid[((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)n_rows+(size_t)row]
                        = g_sum*sig*u_sum*ew;
                }
            }
        }
        __syncthreads();
    }
}

// ---- Q6_K down, register-held weight, full-warp (2 rows/wave), NT_COL cols ----
extern "C" __global__ void q6_k_down_tiled_reg_w32_col(
        float          * __restrict__ out,           // [B, n_rows] — pre-zeroed
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;
    // down slot stride = n_blk_mid*292 = 4*292 = 1168 (tight LDS, not the
    // gate/up 3504 — frees ~18KB LDS at NT_COL=8 → higher occupancy).
    #define LDS_XQ_DOWN_BYTES 1168
    __shared__ __align__(16) uint8_t s_xq[NT_COL * LDS_XQ_DOWN_BYTES];

    const unsigned int tid   = threadIdx.x;
    const unsigned int wave  = tid >> 5;         // 0..7
    const unsigned int lane  = tid & 31u;
    const unsigned int half  = lane >> 4u;       // which row of the pair
    const unsigned int local = lane & 15u;       // 0..15 within the row
    const unsigned int block_lane = local >> 2u; // 0..3 == block index
    const unsigned int sub        = local & 3u;  // 0..3 quarter
    const unsigned int row = blockIdx.x * 16u + wave * 2u + half;
    const bool row_ok = (row < n_rows);
    const unsigned int bb = block_lane;
    const bool valid = row_ok && (bb < n_blocks_in);

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q6_K_BYTES;

    // decode this lane's owned block/quarter ONCE
    uint32_t apack[16];
    int      sch[4];
    float    dq = 0.0f;
    if (valid) {
        const uint8_t* x_blk = row_base + (uint64_t)bb * BLOCK_Q6_K_BYTES;
        const uint8_t* ql = x_blk + 0;
        const uint8_t* qh = x_blk + 128;
        const int8_t*  sc = (const int8_t*)(x_blk + 192);
        dq = f16_to_f32_t(*(const uint16_t*)(x_blk + 208));
        const unsigned int hf = sub >> 1u, up = sub & 1u;
        const uint8_t* qlh = ql + (size_t)hf*64u;
        const uint8_t* qhh = qh + (size_t)hf*32u;
        const int8_t*  sch4 = sc + (size_t)hf*8u + (size_t)up*4u;
        #pragma unroll
        for (int g = 0; g < 4; g++) sch[g] = (int)sch4[g];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            #pragma unroll
            for (int qq = 0; qq < 16; qq += 4) {
                const int r0 = g*16 + qq;
                uint32_t ap = 0u;
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    const int r = r0 + t;
                    const unsigned int shift = up*4u + ((r >= 32) ? 2u : 0u);
                    const uint8_t nib = up ? (uint8_t)(qlh[r] >> 4) : (uint8_t)(qlh[r] & 0x0Fu);
                    const uint8_t hi2 = (uint8_t)((qhh[r & 31] >> shift) & 3u);
                    const int8_t  av  = (int8_t)((int)(nib | (hi2 << 4)) - 32);
                    ap |= ((uint32_t)(uint8_t)av) << (t*8);
                }
                apack[g*4 + qq/4] = ap;
            }
        }
    }

    for (int m0 = 0; m0 < n_members; m0 += NT_COL) {
        const int nt = min((int)NT_COL, n_members - m0);
        for (int j = 0; j < nt; j++) {
            const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)(m0+j)];
            const unsigned int b    = (unsigned int)(packed >> 16);
            const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
            const uint8_t* xq_slot = xq_base
                + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride;
            uint8_t* dst = s_xq + (size_t)j * LDS_XQ_DOWN_BYTES;
            stage_lds_vec(dst, xq_slot, xq_slot_stride, tid, blockDim.x);
        }
        __syncthreads();

        #pragma unroll 1
        for (int j = 0; j < nt; j++) {
            float acc = 0.0f;
            if (valid) {
                const uint8_t* y = s_xq + (size_t)j*LDS_XQ_DOWN_BYTES + (size_t)bb*BLOCK_Q8_K_BYTES;
                const float yd = *(const float*)(y + 0);
                const unsigned int hf = sub >> 1u, up = sub & 1u;
                const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)hf*128u + (size_t)up*64u;
                int isum = 0;
                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int dot = 0;
                    #pragma unroll
                    for (int qq = 0; qq < 16; qq += 4) {
                        dot = sudot4_t((int32_t)apack[g*4 + qq/4],
                                       load_dword_i8_t(q8 + g*16 + qq), dot);
                    }
                    isum += sch[g] * dot;
                }
                acc = yd * dq * (float)isum;
            }
            acc = warp_sum16_f32_t(acc);
            if (local == 0 && row_ok) {
                const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)(m0+j)];
                const unsigned int b = (unsigned int)(packed >> 16);
                atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc);
            }
        }
        __syncthreads();
    }
}

// ---- Q4_K down, register-held weight, full-warp (2 rows/wave), NT_COL cols ----
// Q4_K analog of q6_k_down_tiled_reg_w32_col: the default q4_k_down_tiled_w32
// re-decodes the weight quarter per member (dot_block_quarter_q4k_t) AND takes
// a barrier per member. This variant decodes each lane's owned quarter ONCE
// into registers (weight-read-once) and stages NT_COL members' activations per
// barrier, streaming them through a bare dp4a — same WIN #4 lever the Q6_K down
// already ships. Layout/geometry identical to q6_k_down_tiled_reg_w32_col so it
// is a drop-in for the Q4_K-down experts. `out` MUST be zeroed. `n_rows` %16.
extern "C" __global__ void q4_k_down_tiled_reg_w32_col(
        float          * __restrict__ out,           // [B, n_rows] — pre-zeroed
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;
    __shared__ __align__(16) uint8_t s_xq[NT_COL * LDS_XQ_DOWN_BYTES];

    const unsigned int tid   = threadIdx.x;
    const unsigned int wave  = tid >> 5;         // 0..7
    const unsigned int lane  = tid & 31u;
    const unsigned int half  = lane >> 4u;       // which row of the pair
    const unsigned int local = lane & 15u;       // 0..15 within the row
    const unsigned int block_lane = local >> 2u; // 0..3 == block index
    const unsigned int sub        = local & 3u;  // 0..3 quarter
    const unsigned int row = blockIdx.x * 16u + wave * 2u + half;
    const bool row_ok = (row < n_rows);
    const unsigned int bb = block_lane;
    const bool valid = row_ok && (bb < n_blocks_in);

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;

    // decode this lane's owned block/quarter ONCE into registers
    uint32_t w_qs[8];
    float w_xd = 0.0f, w_xdm = 0.0f;
    int   w_da = 0, w_ma = 0, w_db = 0, w_mb = 0;
    if (valid) {
        const uint8_t* w_blk = row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        w_xd  = f16_to_f32_t(*(const uint16_t*)(w_blk + 0));
        w_xdm = f16_to_f32_t(*(const uint16_t*)(w_blk + 2));
        uint8_t da,ma,db,mb;
        get_scale_min_q4k_t(2*sub,   w_blk+4, &da,&ma);
        get_scale_min_q4k_t(2*sub+1, w_blk+4, &db,&mb);
        w_da=da; w_ma=ma; w_db=db; w_mb=mb;
        const uint8_t* qs = w_blk + 16 + (size_t)sub*32u;
        #pragma unroll
        for (int i = 0; i < 8; i++) w_qs[i] = (uint32_t)load_dword_t(qs + i*4);
    }

    for (int m0 = 0; m0 < n_members; m0 += NT_COL) {
        const int nt = min((int)NT_COL, n_members - m0);
        for (int j = 0; j < nt; j++) {
            const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)(m0+j)];
            const unsigned int b    = (unsigned int)(packed >> 16);
            const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
            const uint8_t* xq_slot = xq_base
                + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride;
            uint8_t* dst = s_xq + (size_t)j * LDS_XQ_DOWN_BYTES;
            stage_lds_vec(dst, xq_slot, xq_slot_stride, tid, blockDim.x);
        }
        __syncthreads();

        #pragma unroll 1
        for (int j = 0; j < nt; j++) {
            float acc = 0.0f;
            if (valid) {
                const uint8_t* y = s_xq + (size_t)j*LDS_XQ_DOWN_BYTES + (size_t)bb*BLOCK_Q8_K_BYTES;
                const float yd = *(const float*)(y + 0);
                const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
                const int16_t* bs = (const int16_t*)(y + 260);
                int da=0, db=0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                    const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
                    da = sudot4_t((int32_t)(w_qs[i] & 0x0f0f0f0fu), q8lo, da);
                    db = sudot4_t((int32_t)((w_qs[i] >> 4) & 0x0f0f0f0fu), q8hi, db);
                }
                const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
                const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
                acc = yd*w_xd*((float)w_da*da + (float)w_db*db)
                    - yd*w_xdm*((float)w_ma*bsa + (float)w_mb*bsb);
            }
            acc = warp_sum16_f32_t(acc);
            if (local == 0 && row_ok) {
                const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)(m0+j)];
                const unsigned int b = (unsigned int)(packed >> 16);
                atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc);
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// WIN #1 — DENSE weight-read-once dp4a GEMM. The dense-L0 FFN and the shared
// expert previously used q4_k/q6_k_dense_gemv_batched (grid.z=B): each batch
// element's WG re-reads + re-dequantizes the SINGLE shared weight matrix →
// weight-BW scales with B. These kernels apply ONE weight matrix to ALL B
// token-columns with the read-once register-tiled + column-staged structure of
// q4_k_gate_up_swiglu_tiled_reg_col (WIN #4), minus the expert index / gating:
//   out[b, i] = sum_k dequant(W[i,k]) * xq[b,k]
// Weight rows for this WG are decoded ONCE into registers; NT_COL token columns
// (Q8_K-quantized activations) are staged in LDS per barrier and streamed
// through a bare dp4a. Per-chunk weight traffic collapses from B× to 1×.
//   grid (n_rows/8, 1, 1)  block (256,1,1)  — 8 rows/WG, one warp/row.
//   DENSE_MAXB = ceil(max n_blk / 8): n_blk<=16 → 2 (covers HIDDEN=12 & mid=4).
// Activations are Q8_K [B, n_blk*292] token-major (same as the routed path's
// xq_hidden, which the shared gate/up can reuse verbatim).
// ===========================================================================
#define DENSE_MAXB 2
#define BLOCK_Q5_K_BYTES 176

// ---- Q4_K dense GEMM (n_blk<=12 → LDS stride LDS_XQ_BYTES) ----
extern "C" __global__ void q4_k_dense_gemm_dp4a(
        float          * __restrict__ out,   // [B, n_rows]
        const uint8_t  * __restrict__ w,     // [n_rows, n_blk*144]
        const uint8_t  * __restrict__ xq,    // [B, n_blk*292] Q8_K
        unsigned int     B,
        unsigned int     n_rows,
        unsigned int     n_blk) {
    __shared__ __align__(16) uint8_t s_xq[NT_COL * LDS_XQ_BYTES];

    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool         row_ok       = (row < n_rows);
    const unsigned int block_lane   = lane >> 2u;   // 0..7
    const unsigned int sub          = lane & 3u;    // 0..3

    const uint8_t* row_base = w + (uint64_t)row * n_blk * BLOCK_Q4_K_BYTES;

    // --- decode this lane's owned quarters ONCE into registers ---
    unsigned int bb_of[DENSE_MAXB];
    bool  valid[DENSE_MAXB];
    float xd[DENSE_MAXB], xdm[DENSE_MAXB];
    int   da[DENSE_MAXB], ma[DENSE_MAXB], db[DENSE_MAXB], mb[DENSE_MAXB];
    uint32_t qs[DENSE_MAXB][8];
    #pragma unroll
    for (int k = 0; k < DENSE_MAXB; k++) {
        const unsigned int bb = block_lane + (unsigned int)k * 8u;
        bb_of[k] = bb;
        valid[k] = row_ok && (bb < n_blk);
        if (!valid[k]) continue;
        const uint8_t* wg = row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        xd[k]  = f16_to_f32_t(*(const uint16_t*)(wg + 0));
        xdm[k] = f16_to_f32_t(*(const uint16_t*)(wg + 2));
        uint8_t d0,m0,d1,m1;
        get_scale_min_q4k_t(2*sub,   wg+4, &d0,&m0);
        get_scale_min_q4k_t(2*sub+1, wg+4, &d1,&m1);
        da[k]=d0; ma[k]=m0; db[k]=d1; mb[k]=m1;
        const uint8_t* q = wg + 16 + (size_t)sub*32u;
        #pragma unroll
        for (int i = 0; i < 8; i++) qs[k][i] = (uint32_t)load_dword_t(q + i*4);
    }

    const unsigned int xq_bytes = n_blk * BLOCK_Q8_K_BYTES;
    for (unsigned int b0 = 0; b0 < B; b0 += NT_COL) {
        const int nt = min((int)NT_COL, (int)(B - b0));
        for (int j = 0; j < nt; j++) {
            const uint8_t* xq_b = xq + (size_t)(b0 + j) * (size_t)xq_bytes;
            uint8_t* dst = s_xq + (size_t)j * LDS_XQ_BYTES;
            stage_lds_vec(dst, xq_b, xq_bytes, tid, blockDim.x);
        }
        __syncthreads();

        if (row_ok) {
            #pragma unroll 1
            for (int j = 0; j < nt; j++) {
                const uint8_t* xj = s_xq + (size_t)j * LDS_XQ_BYTES;
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < DENSE_MAXB; k++) {
                    if (!valid[k]) continue;
                    const uint8_t* y = xj + (size_t)bb_of[k] * BLOCK_Q8_K_BYTES;
                    const float yd = *(const float*)(y + 0);
                    const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
                    const int16_t* bs = (const int16_t*)(y + 260);
                    int lo=0, hi=0;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                        const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
                        lo = sudot4_t((int32_t)(qs[k][i] & 0x0f0f0f0fu), q8lo, lo);
                        hi = sudot4_t((int32_t)((qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, hi);
                    }
                    const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
                    const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
                    acc += yd*xd[k]*((float)da[k]*lo + (float)db[k]*hi)
                         - yd*xdm[k]*((float)ma[k]*bsa + (float)mb[k]*bsb);
                }
                acc = warp_sum_f32_t(acc);
                if (lane == 0) out[(size_t)(b0 + j) * (size_t)n_rows + (size_t)row] = acc;
            }
        }
        __syncthreads();
    }
}

// ---- Q6_K dense GEMM (down proj; n_blk<=4 → tight LDS stride 1168) ----
#define LDS_DENSE_Q6_BYTES 1168
extern "C" __global__ void q6_k_dense_gemm_dp4a(
        float          * __restrict__ out,   // [B, n_rows]
        const uint8_t  * __restrict__ w,     // [n_rows, n_blk*210]
        const uint8_t  * __restrict__ xq,    // [B, n_blk*292] Q8_K
        unsigned int     B,
        unsigned int     n_rows,
        unsigned int     n_blk) {
    __shared__ __align__(16) uint8_t s_xq[NT_COL * LDS_DENSE_Q6_BYTES];

    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    const bool         row_ok       = (row < n_rows);
    const unsigned int block_lane   = lane >> 2u;
    const unsigned int sub          = lane & 3u;

    const uint8_t* row_base = w + (uint64_t)row * n_blk * BLOCK_Q6_K_BYTES;

    // decode this lane's owned quarters (DENSE_MAXB) once
    unsigned int bb_of[DENSE_MAXB];
    bool  valid[DENSE_MAXB];
    uint32_t apack[DENSE_MAXB][16];
    int   sch[DENSE_MAXB][4];
    float dq[DENSE_MAXB];
    #pragma unroll
    for (int k = 0; k < DENSE_MAXB; k++) {
        const unsigned int bb = block_lane + (unsigned int)k*8u;
        bb_of[k] = bb;
        valid[k] = row_ok && (bb < n_blk);
        if (!valid[k]) continue;
        const uint8_t* x_blk = row_base + (uint64_t)bb * BLOCK_Q6_K_BYTES;
        const uint8_t* ql = x_blk + 0;
        const uint8_t* qh = x_blk + 128;
        const int8_t*  sc = (const int8_t*)(x_blk + 192);
        dq[k] = f16_to_f32_t(*(const uint16_t*)(x_blk + 208));
        const unsigned int half  = sub >> 1u;
        const unsigned int upper = sub & 1u;
        const uint8_t* qlh = ql + (size_t)half*64u;
        const uint8_t* qhh = qh + (size_t)half*32u;
        const int8_t*  sch4 = sc + (size_t)half*8u + (size_t)upper*4u;
        #pragma unroll
        for (int g = 0; g < 4; g++) sch[k][g] = (int)sch4[g];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            #pragma unroll
            for (int qq = 0; qq < 16; qq += 4) {
                const int r0 = g*16 + qq;
                uint32_t ap = 0u;
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    const int r = r0 + t;
                    const unsigned int shift = upper*4u + ((r >= 32) ? 2u : 0u);
                    const uint8_t nib = upper ? (uint8_t)(qlh[r] >> 4) : (uint8_t)(qlh[r] & 0x0Fu);
                    const uint8_t hi2 = (uint8_t)((qhh[r & 31] >> shift) & 3u);
                    const int8_t  av  = (int8_t)((int)(nib | (hi2 << 4)) - 32);
                    ap |= ((uint32_t)(uint8_t)av) << (t*8);
                }
                apack[k][g*4 + qq/4] = ap;
            }
        }
    }

    const unsigned int xq_bytes = n_blk * BLOCK_Q8_K_BYTES;
    for (unsigned int b0 = 0; b0 < B; b0 += NT_COL) {
        const int nt = min((int)NT_COL, (int)(B - b0));
        for (int j = 0; j < nt; j++) {
            const uint8_t* xq_b = xq + (size_t)(b0 + j) * (size_t)xq_bytes;
            uint8_t* dst = s_xq + (size_t)j * LDS_DENSE_Q6_BYTES;
            stage_lds_vec(dst, xq_b, xq_bytes, tid, blockDim.x);
        }
        __syncthreads();

        if (row_ok) {
            #pragma unroll 1
            for (int j = 0; j < nt; j++) {
                const uint8_t* xj = s_xq + (size_t)j * LDS_DENSE_Q6_BYTES;
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < DENSE_MAXB; k++) {
                    if (!valid[k]) continue;
                    const uint8_t* y = xj + (size_t)bb_of[k] * BLOCK_Q8_K_BYTES;
                    const float yd = *(const float*)(y + 0);
                    const unsigned int half = sub >> 1u, upper = sub & 1u;
                    const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)half*128u + (size_t)upper*64u;
                    int isum = 0;
                    #pragma unroll
                    for (int g = 0; g < 4; g++) {
                        int dot = 0;
                        #pragma unroll
                        for (int qq = 0; qq < 16; qq += 4) {
                            dot = sudot4_t((int32_t)apack[k][g*4 + qq/4],
                                           load_dword_i8_t(q8 + g*16 + qq), dot);
                        }
                        isum += sch[k][g] * dot;
                    }
                    acc += yd * dq[k] * (float)isum;
                }
                acc = warp_sum_f32_t(acc);
                if (lane == 0) out[(size_t)(b0 + j) * (size_t)n_rows + (size_t)row] = acc;
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// WIDE-ROW + NO-LDS dense GEMM (shared-expert / dense-L0). The _dp4a kernels
// above use 8 rows/WG and LDS-stage NT_COL columns, so each token activation is
// re-read n_rows/8 times AND round-trips DRAM->LDS->reg. These variants apply
// the two levers proven on the routed gate/up path: (1) 32 rows/WG (block=1024)
// so the activation amortizes over 4× more weight rows; (2) NO LDS staging —
// each of the 32 warps reads the token's Q8_K activation directly from global,
// L2 serving the intra-WG reuse. Bit-identical math to q{4,6}_k_dense_gemm_dp4a
// (same per-row accumulation order). grid.x = n_rows/32, block = 1024.
// ===========================================================================
#ifndef DENSE_ROWS_PER_WG
#define DENSE_ROWS_PER_WG 32
#endif

extern "C" __global__ void q4_k_dense_gemm_dp4a_r32_nolds(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w,
        const uint8_t  * __restrict__ xq,
        unsigned int     B,
        unsigned int     n_rows,
        unsigned int     n_blk) {
    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * DENSE_ROWS_PER_WG + row_in_block;
    if (row >= n_rows) return;
    const unsigned int block_lane   = lane >> 2u;
    const unsigned int sub          = lane & 3u;

    const uint8_t* row_base = w + (uint64_t)row * n_blk * BLOCK_Q4_K_BYTES;

    unsigned int bb_of[DENSE_MAXB];
    bool  valid[DENSE_MAXB];
    float xd[DENSE_MAXB], xdm[DENSE_MAXB];
    int   da[DENSE_MAXB], ma[DENSE_MAXB], db[DENSE_MAXB], mb[DENSE_MAXB];
    uint32_t qs[DENSE_MAXB][8];
    #pragma unroll
    for (int k = 0; k < DENSE_MAXB; k++) {
        const unsigned int bb = block_lane + (unsigned int)k * 8u;
        bb_of[k] = bb;
        valid[k] = (bb < n_blk);
        if (!valid[k]) continue;
        const uint8_t* wg = row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        xd[k]  = f16_to_f32_t(*(const uint16_t*)(wg + 0));
        xdm[k] = f16_to_f32_t(*(const uint16_t*)(wg + 2));
        uint8_t d0,m0,d1,m1;
        get_scale_min_q4k_t(2*sub,   wg+4, &d0,&m0);
        get_scale_min_q4k_t(2*sub+1, wg+4, &d1,&m1);
        da[k]=d0; ma[k]=m0; db[k]=d1; mb[k]=m1;
        const uint8_t* q = wg + 16 + (size_t)sub*32u;
        #pragma unroll
        for (int i = 0; i < 8; i++) qs[k][i] = (uint32_t)load_dword_t(q + i*4);
    }

    const unsigned int xq_bytes = n_blk * BLOCK_Q8_K_BYTES;
    for (unsigned int bcol = 0; bcol < B; bcol++) {
        const uint8_t* xj = xq + (size_t)bcol * (size_t)xq_bytes;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < DENSE_MAXB; k++) {
            if (!valid[k]) continue;
            const uint8_t* y = xj + (size_t)bb_of[k] * BLOCK_Q8_K_BYTES;
            const float yd = *(const float*)(y + 0);
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
            const int16_t* bs = (const int16_t*)(y + 260);
            int lo=0, hi=0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
                lo = sudot4_t((int32_t)(qs[k][i] & 0x0f0f0f0fu), q8lo, lo);
                hi = sudot4_t((int32_t)((qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, hi);
            }
            const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
            const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
            acc += yd*xd[k]*((float)da[k]*lo + (float)db[k]*hi)
                 - yd*xdm[k]*((float)ma[k]*bsa + (float)mb[k]*bsb);
        }
        acc = warp_sum_f32_t(acc);
        if (lane == 0) out[(size_t)bcol * (size_t)n_rows + (size_t)row] = acc;
    }
}

extern "C" __global__ void q6_k_dense_gemm_dp4a_r32_nolds(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w,
        const uint8_t  * __restrict__ xq,
        unsigned int     B,
        unsigned int     n_rows,
        unsigned int     n_blk) {
    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * DENSE_ROWS_PER_WG + row_in_block;
    if (row >= n_rows) return;
    const unsigned int block_lane   = lane >> 2u;
    const unsigned int sub          = lane & 3u;

    const uint8_t* row_base = w + (uint64_t)row * n_blk * BLOCK_Q6_K_BYTES;

    unsigned int bb_of[DENSE_MAXB];
    bool  valid[DENSE_MAXB];
    uint32_t apack[DENSE_MAXB][16];
    int   sch[DENSE_MAXB][4];
    float dq[DENSE_MAXB];
    #pragma unroll
    for (int k = 0; k < DENSE_MAXB; k++) {
        const unsigned int bb = block_lane + (unsigned int)k*8u;
        bb_of[k] = bb;
        valid[k] = (bb < n_blk);
        if (!valid[k]) continue;
        const uint8_t* x_blk = row_base + (uint64_t)bb * BLOCK_Q6_K_BYTES;
        const uint8_t* ql = x_blk + 0;
        const uint8_t* qh = x_blk + 128;
        const int8_t*  sc = (const int8_t*)(x_blk + 192);
        dq[k] = f16_to_f32_t(*(const uint16_t*)(x_blk + 208));
        const unsigned int half  = sub >> 1u;
        const unsigned int upper = sub & 1u;
        const uint8_t* qlh = ql + (size_t)half*64u;
        const uint8_t* qhh = qh + (size_t)half*32u;
        const int8_t*  sch4 = sc + (size_t)half*8u + (size_t)upper*4u;
        #pragma unroll
        for (int g = 0; g < 4; g++) sch[k][g] = (int)sch4[g];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            #pragma unroll
            for (int qq = 0; qq < 16; qq += 4) {
                const int r0 = g*16 + qq;
                uint32_t ap = 0u;
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    const int r = r0 + t;
                    const unsigned int shift = upper*4u + ((r >= 32) ? 2u : 0u);
                    const uint8_t nib = upper ? (uint8_t)(qlh[r] >> 4) : (uint8_t)(qlh[r] & 0x0Fu);
                    const uint8_t hi2 = (uint8_t)((qhh[r & 31] >> shift) & 3u);
                    const int8_t  av  = (int8_t)((int)(nib | (hi2 << 4)) - 32);
                    ap |= ((uint32_t)(uint8_t)av) << (t*8);
                }
                apack[k][g*4 + qq/4] = ap;
            }
        }
    }

    const unsigned int xq_bytes = n_blk * BLOCK_Q8_K_BYTES;
    for (unsigned int bcol = 0; bcol < B; bcol++) {
        const uint8_t* xj = xq + (size_t)bcol * (size_t)xq_bytes;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < DENSE_MAXB; k++) {
            if (!valid[k]) continue;
            const uint8_t* y = xj + (size_t)bb_of[k] * BLOCK_Q8_K_BYTES;
            const float yd = *(const float*)(y + 0);
            const unsigned int half = sub >> 1u, upper = sub & 1u;
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)half*128u + (size_t)upper*64u;
            int isum = 0;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                int dot = 0;
                #pragma unroll
                for (int qq = 0; qq < 16; qq += 4) {
                    dot = sudot4_t((int32_t)apack[k][g*4 + qq/4],
                                   load_dword_i8_t(q8 + g*16 + qq), dot);
                }
                isum += sch[k][g] * dot;
            }
            acc += yd * dq[k] * (float)isum;
        }
        acc = warp_sum_f32_t(acc);
        if (lane == 0) out[(size_t)bcol * (size_t)n_rows + (size_t)row] = acc;
    }
}

// ===========================================================================
// WIN #2 — WIDE-ROW gate/up. The _reg_col kernel above uses 8 rows/WG, so the
// LDS-staged member activation (NT_COL × 3504 B) is re-loaded from DRAM once
// per row-block → n_rows/8 = 128 times per expert. At B=512 the per-expert
// activation traffic (128 × ~20 members × 3504 B ≈ 9 MB) dwarfs the read-once
// weight (1.77 MB), so activation staging is the pole. This variant packs
// ROWS_PER_WG=32 rows per WG (32 warps, block=1024) so the identical staged
// activation amortizes over 4× more weight rows → activation DRAM traffic /4.
// Register-held weight per warp is unchanged (MAXB quarters); only the block
// widens. grid.x = n_rows/32. Math is byte-identical to _reg_col.
// ===========================================================================
#ifndef GU_ROWS_PER_WG
#define GU_ROWS_PER_WG 32
#endif
extern "C" __global__ void q4_k_gate_up_swiglu_tiled_reg_col_r32(
        float          * __restrict__ mid,
        const uint8_t  * __restrict__ gate_w_base,
        const uint8_t  * __restrict__ up_w_base,
        const uint8_t  * __restrict__ xq,
        const float    * __restrict__ expert_w,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     gate_bpe,
        unsigned int     up_bpe,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        float            clamp,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;

    __shared__ __align__(16) uint8_t s_xq[NT_COL * LDS_XQ_BYTES];

    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;             // 0..GU_ROWS_PER_WG-1
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * GU_ROWS_PER_WG + row_in_block;
    const bool         row_ok       = (row < n_rows);
    const unsigned int block_lane   = lane >> 2u;   // 0..7
    const unsigned int sub          = lane & 3u;    // 0..3

    const uint8_t* gate_row_base = gate_w_base + (uint64_t)e*gate_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
    const uint8_t* up_row_base   = up_w_base   + (uint64_t)e*up_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;

    unsigned int bb_of[MAXB];
    bool  valid[MAXB];
    float g_xd[MAXB], g_xdm[MAXB], u_xd[MAXB], u_xdm[MAXB];
    int   g_da[MAXB], g_ma[MAXB], g_db[MAXB], g_mb[MAXB];
    int   u_da[MAXB], u_ma[MAXB], u_db[MAXB], u_mb[MAXB];
    uint32_t g_qs[MAXB][8];
    uint32_t u_qs[MAXB][8];
    #pragma unroll
    for (int k = 0; k < MAXB; k++) {
        const unsigned int bb = block_lane + (unsigned int)k * 8u;
        bb_of[k] = bb;
        valid[k] = row_ok && (bb < n_blocks_in);
        if (!valid[k]) continue;
        const uint8_t* wg = gate_row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        const uint8_t* wu = up_row_base   + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        g_xd[k]  = f16_to_f32_t(*(const uint16_t*)(wg + 0));
        g_xdm[k] = f16_to_f32_t(*(const uint16_t*)(wg + 2));
        u_xd[k]  = f16_to_f32_t(*(const uint16_t*)(wu + 0));
        u_xdm[k] = f16_to_f32_t(*(const uint16_t*)(wu + 2));
        uint8_t da,ma,db,mb;
        get_scale_min_q4k_t(2*sub,   wg+4, &da,&ma); get_scale_min_q4k_t(2*sub+1, wg+4, &db,&mb);
        g_da[k]=da; g_ma[k]=ma; g_db[k]=db; g_mb[k]=mb;
        get_scale_min_q4k_t(2*sub,   wu+4, &da,&ma); get_scale_min_q4k_t(2*sub+1, wu+4, &db,&mb);
        u_da[k]=da; u_ma[k]=ma; u_db[k]=db; u_mb[k]=mb;
        const uint8_t* gqs = wg + 16 + (size_t)sub*32u;
        const uint8_t* uqs = wu + 16 + (size_t)sub*32u;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            g_qs[k][i] = (uint32_t)load_dword_t(gqs + i*4);
            u_qs[k][i] = (uint32_t)load_dword_t(uqs + i*4);
        }
    }

    const unsigned int xq_bytes = n_blocks_in * BLOCK_Q8_K_BYTES;
    for (int m0 = 0; m0 < n_members; m0 += NT_COL) {
        const int nt = min((int)NT_COL, n_members - m0);
        for (int j = 0; j < nt; j++) {
            const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)(m0+j)];
            const unsigned int b = (unsigned int)(packed >> 16);
            const uint8_t* xq_b = xq + (size_t)b * (size_t)xq_bytes;
            uint8_t* dst = s_xq + (size_t)j * LDS_XQ_BYTES;
            stage_lds_vec(dst, xq_b, xq_bytes, tid, blockDim.x);
        }
        __syncthreads();

        if (row_ok) {
            #pragma unroll 1
            for (int j = 0; j < nt; j++) {
                const uint8_t* xj = s_xq + (size_t)j * LDS_XQ_BYTES;
                float g_sum = 0.0f, u_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < MAXB; k++) {
                    if (!valid[k]) continue;
                    const uint8_t* y = xj + (size_t)bb_of[k] * BLOCK_Q8_K_BYTES;
                    const float yd = *(const float*)(y + 0);
                    const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
                    const int16_t* bs = (const int16_t*)(y + 260);
                    int gda=0, gdb=0, uda=0, udb=0;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                        const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
                        gda = sudot4_t((int32_t)(g_qs[k][i] & 0x0f0f0f0fu), q8lo, gda);
                        gdb = sudot4_t((int32_t)((g_qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, gdb);
                        uda = sudot4_t((int32_t)(u_qs[k][i] & 0x0f0f0f0fu), q8lo, uda);
                        udb = sudot4_t((int32_t)((u_qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, udb);
                    }
                    const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
                    const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
                    g_sum += yd*g_xd[k]*((float)g_da[k]*gda + (float)g_db[k]*gdb)
                           - yd*g_xdm[k]*((float)g_ma[k]*bsa + (float)g_mb[k]*bsb);
                    u_sum += yd*u_xd[k]*((float)u_da[k]*uda + (float)u_db[k]*udb)
                           - yd*u_xdm[k]*((float)u_ma[k]*bsa + (float)u_mb[k]*bsb);
                }
                g_sum = warp_sum_f32_t(g_sum);
                u_sum = warp_sum_f32_t(u_sum);
                if (lane == 0) {
                    const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)(m0+j)];
                    const unsigned int b    = (unsigned int)(packed >> 16);
                    const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
                    const float ew = expert_w[(size_t)b*(size_t)n_used+(size_t)slot];
                    if (clamp > 1.0e-6f) {
                        if (g_sum > clamp)  g_sum = clamp;
                        if (u_sum > clamp)  u_sum = clamp;
                        if (u_sum < -clamp) u_sum = -clamp;
                    }
                    const float sig = 1.0f/(1.0f+expf(-g_sum));
                    mid[((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)n_rows+(size_t)row]
                        = g_sum*sig*u_sum*ew;
                }
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// ABLATION variants of q4_k_gate_up_swiglu_tiled_reg_col_r32 (TIMING ONLY,
// parity intentionally broken). Same signature; body from the .inc, each
// stripping ONE component so rocprofv3 can attribute per-component ms.
// ===========================================================================
#define GATEUP_R32_SIG(NAME) \
extern "C" __global__ void NAME( \
        float          * __restrict__ mid, \
        const uint8_t  * __restrict__ gate_w_base, \
        const uint8_t  * __restrict__ up_w_base, \
        const uint8_t  * __restrict__ xq, \
        const float    * __restrict__ expert_w, \
        const int      * __restrict__ group_count, \
        const int      * __restrict__ expert_members, \
        unsigned int     gate_bpe, \
        unsigned int     up_bpe, \
        unsigned int     n_used, \
        unsigned int     max_per_expert, \
        float            clamp, \
        unsigned int     n_rows, \
        unsigned int     n_blocks_in)

// full copy (sanity: must ~= r32 timing)
GATEUP_R32_SIG(q4_k_gate_up_r32_abl_full)
#include "gateup_r32_ablation.inc"

// strip DRAM->LDS activation staging copy
#define ABL_STAGE 0
GATEUP_R32_SIG(q4_k_gate_up_r32_abl_nostage)
#include "gateup_r32_ablation.inc"
#undef ABL_STAGE

// strip q8 LDS activation read (keep staging + MAC-shaped work on dummy)
#define ABL_DOTLOAD 0
GATEUP_R32_SIG(q4_k_gate_up_r32_abl_noload)
#include "gateup_r32_ablation.inc"
#undef ABL_DOTLOAD

// strip the dp4a MACs (keep loads+stage+reduce+scale)
#define ABL_DOT 0
GATEUP_R32_SIG(q4_k_gate_up_r32_abl_nodot)
#include "gateup_r32_ablation.inc"
#undef ABL_DOT

// strip the 32-lane warp reduction
#define ABL_REDUCE 0
GATEUP_R32_SIG(q4_k_gate_up_r32_abl_noreduce)
#include "gateup_r32_ablation.inc"
#undef ABL_REDUCE

// strip the per-superblock scale/min FMA
#define ABL_SCALE 0
GATEUP_R32_SIG(q4_k_gate_up_r32_abl_noscale)
#include "gateup_r32_ablation.inc"
#undef ABL_SCALE

// floor: strip stage + load + dot + reduce (keeps weight preload + write)
#define ABL_STAGE 0
#define ABL_DOTLOAD 0
#define ABL_DOT 0
#define ABL_REDUCE 0
GATEUP_R32_SIG(q4_k_gate_up_r32_abl_floor)
#include "gateup_r32_ablation.inc"
#undef ABL_STAGE
#undef ABL_DOTLOAD
#undef ABL_DOT
#undef ABL_REDUCE

// ===========================================================================
// q4_k_gate_up_swiglu_tiled_reg_col_r32_nolds — same math/parity as the r32
// kernel but WITHOUT LDS activation staging. Each of the 32 warps (rows) reads
// the member's Q8_K activation directly from global; L2 serves the intra-WG
// reuse across the 32 warps. Removes the DRAM->LDS write + LDS read round-trip
// (the ablation's dominant "staging" cost) and all __syncthreads. Bit-parity
// with r32 (same accumulation order per row).
// ===========================================================================
extern "C" __global__ void q4_k_gate_up_swiglu_tiled_reg_col_r32_nolds(
        float          * __restrict__ mid,
        const uint8_t  * __restrict__ gate_w_base,
        const uint8_t  * __restrict__ up_w_base,
        const uint8_t  * __restrict__ xq,
        const float    * __restrict__ expert_w,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     gate_bpe,
        unsigned int     up_bpe,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        float            clamp,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    const int n_members = group_count[e];
    if (n_members == 0) return;

    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * GU_ROWS_PER_WG + row_in_block;
    if (row >= n_rows) return;
    const unsigned int block_lane   = lane >> 2u;
    const unsigned int sub          = lane & 3u;

    const uint8_t* gate_row_base = gate_w_base + (uint64_t)e*gate_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
    const uint8_t* up_row_base   = up_w_base   + (uint64_t)e*up_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;

    unsigned int bb_of[MAXB];
    bool  valid[MAXB];
    float g_xd[MAXB], g_xdm[MAXB], u_xd[MAXB], u_xdm[MAXB];
    int   g_da[MAXB], g_ma[MAXB], g_db[MAXB], g_mb[MAXB];
    int   u_da[MAXB], u_ma[MAXB], u_db[MAXB], u_mb[MAXB];
    uint32_t g_qs[MAXB][8];
    uint32_t u_qs[MAXB][8];
    #pragma unroll
    for (int k = 0; k < MAXB; k++) {
        const unsigned int bb = block_lane + (unsigned int)k * 8u;
        bb_of[k] = bb;
        valid[k] = (bb < n_blocks_in);
        if (!valid[k]) continue;
        const uint8_t* wg = gate_row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        const uint8_t* wu = up_row_base   + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        g_xd[k]  = f16_to_f32_t(*(const uint16_t*)(wg + 0));
        g_xdm[k] = f16_to_f32_t(*(const uint16_t*)(wg + 2));
        u_xd[k]  = f16_to_f32_t(*(const uint16_t*)(wu + 0));
        u_xdm[k] = f16_to_f32_t(*(const uint16_t*)(wu + 2));
        uint8_t da,ma,db,mb;
        get_scale_min_q4k_t(2*sub,   wg+4, &da,&ma); get_scale_min_q4k_t(2*sub+1, wg+4, &db,&mb);
        g_da[k]=da; g_ma[k]=ma; g_db[k]=db; g_mb[k]=mb;
        get_scale_min_q4k_t(2*sub,   wu+4, &da,&ma); get_scale_min_q4k_t(2*sub+1, wu+4, &db,&mb);
        u_da[k]=da; u_ma[k]=ma; u_db[k]=db; u_mb[k]=mb;
        const uint8_t* gqs = wg + 16 + (size_t)sub*32u;
        const uint8_t* uqs = wu + 16 + (size_t)sub*32u;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            g_qs[k][i] = (uint32_t)load_dword_t(gqs + i*4);
            u_qs[k][i] = (uint32_t)load_dword_t(uqs + i*4);
        }
    }

    const unsigned int xq_bytes = n_blocks_in * BLOCK_Q8_K_BYTES;
    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b = (unsigned int)(packed >> 16);
        const uint8_t* xj = xq + (size_t)b * (size_t)xq_bytes;
        float g_sum = 0.0f, u_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < MAXB; k++) {
            if (!valid[k]) continue;
            const uint8_t* y = xj + (size_t)bb_of[k] * BLOCK_Q8_K_BYTES;
            const float yd = *(const float*)(y + 0);
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
            const int16_t* bs = (const int16_t*)(y + 260);
            int gda=0, gdb=0, uda=0, udb=0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
                gda = sudot4_t((int32_t)(g_qs[k][i] & 0x0f0f0f0fu), q8lo, gda);
                gdb = sudot4_t((int32_t)((g_qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, gdb);
                uda = sudot4_t((int32_t)(u_qs[k][i] & 0x0f0f0f0fu), q8lo, uda);
                udb = sudot4_t((int32_t)((u_qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, udb);
            }
            const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
            const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
            g_sum += yd*g_xd[k]*((float)g_da[k]*gda + (float)g_db[k]*gdb)
                   - yd*g_xdm[k]*((float)g_ma[k]*bsa + (float)g_mb[k]*bsb);
            u_sum += yd*u_xd[k]*((float)u_da[k]*uda + (float)u_db[k]*udb)
                   - yd*u_xdm[k]*((float)u_ma[k]*bsa + (float)u_mb[k]*bsb);
        }
        g_sum = warp_sum_f32_t(g_sum);
        u_sum = warp_sum_f32_t(u_sum);
        if (lane == 0) {
            const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
            const float ew = expert_w[(size_t)b*(size_t)n_used+(size_t)slot];
            if (clamp > 1.0e-6f) {
                if (g_sum > clamp)  g_sum = clamp;
                if (u_sum > clamp)  u_sum = clamp;
                if (u_sum < -clamp) u_sum = -clamp;
            }
            const float sig = 1.0f/(1.0f+expf(-g_sum));
            mid[((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)n_rows+(size_t)row]
                = g_sum*sig*u_sum*ew;
        }
    }
}

// ===========================================================================
// No-LDS down variants: read the member's Q8_K mid activation directly from
// global (L2 serves intra-WG reuse across the 16 rows / 8 waves). Removes the
// DRAM->LDS staging round-trip + __syncthreads. Bit-parity with the _col
// kernels (same per-row accumulation order). `out` MUST be zeroed.
// ===========================================================================
extern "C" __global__ void q6_k_down_tiled_reg_w32_col_nolds(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    const int n_members = group_count[e];
    if (n_members == 0) return;

    const unsigned int tid   = threadIdx.x;
    const unsigned int wave  = tid >> 5;
    const unsigned int lane  = tid & 31u;
    const unsigned int half  = lane >> 4u;
    const unsigned int local = lane & 15u;
    const unsigned int block_lane = local >> 2u;
    const unsigned int sub        = local & 3u;
    const unsigned int row = blockIdx.x * 16u + wave * 2u + half;
    const bool row_ok = (row < n_rows);
    const unsigned int bb = block_lane;
    const bool valid = row_ok && (bb < n_blocks_in);

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q6_K_BYTES;
    uint32_t apack[16];
    int      sch[4];
    float    dq = 0.0f;
    if (valid) {
        const uint8_t* x_blk = row_base + (uint64_t)bb * BLOCK_Q6_K_BYTES;
        const uint8_t* ql = x_blk + 0;
        const uint8_t* qh = x_blk + 128;
        const int8_t*  sc = (const int8_t*)(x_blk + 192);
        dq = f16_to_f32_t(*(const uint16_t*)(x_blk + 208));
        const unsigned int hf = sub >> 1u, up = sub & 1u;
        const uint8_t* qlh = ql + (size_t)hf*64u;
        const uint8_t* qhh = qh + (size_t)hf*32u;
        const int8_t*  sch4 = sc + (size_t)hf*8u + (size_t)up*4u;
        #pragma unroll
        for (int g = 0; g < 4; g++) sch[g] = (int)sch4[g];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            #pragma unroll
            for (int qq = 0; qq < 16; qq += 4) {
                const int r0 = g*16 + qq;
                uint32_t ap = 0u;
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    const int r = r0 + t;
                    const unsigned int shift = up*4u + ((r >= 32) ? 2u : 0u);
                    const uint8_t nib = up ? (uint8_t)(qlh[r] >> 4) : (uint8_t)(qlh[r] & 0x0Fu);
                    const uint8_t hi2 = (uint8_t)((qhh[r & 31] >> shift) & 3u);
                    const int8_t  av  = (int8_t)((int)(nib | (hi2 << 4)) - 32);
                    ap |= ((uint32_t)(uint8_t)av) << (t*8);
                }
                apack[g*4 + qq/4] = ap;
            }
        }
    }

    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* y = xq_base
            + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride
            + (size_t)bb*BLOCK_Q8_K_BYTES;
        float acc = 0.0f;
        if (valid) {
            const float yd = *(const float*)(y + 0);
            const unsigned int hf = sub >> 1u, up = sub & 1u;
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)hf*128u + (size_t)up*64u;
            int isum = 0;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                int dot = 0;
                #pragma unroll
                for (int qq = 0; qq < 16; qq += 4) {
                    dot = sudot4_t((int32_t)apack[g*4 + qq/4],
                                   load_dword_i8_t(q8 + g*16 + qq), dot);
                }
                isum += sch[g] * dot;
            }
            acc = yd * dq * (float)isum;
        }
        acc = warp_sum16_f32_t(acc);
        if (local == 0 && row_ok) {
            atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc);
        }
    }
}

// ===========================================================================
// WIDE-ROW no-LDS down variants. Identical math to q{6,4}_k_down_tiled_reg_
// w32_col_nolds, but ROWS_PER_WG is derived from blockDim (= 2 rows/wave ×
// #waves) instead of hard-coded to 16. Launching with block=512 → 32 rows/WG,
// block=1024 → 64 rows/WG. More rows per WG amortizes the per-member Q8_K
// activation read (each WG re-fetches the member's 1168 B from global/L2) over
// more weight rows → activation traffic ÷ (rows/16). Bit-identical to the
// 16-row kernel at block=256 (same per-row accumulation order). `out` zeroed.
// ===========================================================================
extern "C" __global__ void q6_k_down_tiled_reg_w32_col_nolds_wide(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    const int n_members = group_count[e];
    if (n_members == 0) return;

    const unsigned int tid   = threadIdx.x;
    const unsigned int wave  = tid >> 5;
    const unsigned int lane  = tid & 31u;
    const unsigned int half  = lane >> 4u;
    const unsigned int local = lane & 15u;
    const unsigned int block_lane = local >> 2u;
    const unsigned int sub        = local & 3u;
    const unsigned int rows_per_wg = (blockDim.x >> 5) * 2u;
    const unsigned int row = blockIdx.x * rows_per_wg + wave * 2u + half;
    const bool row_ok = (row < n_rows);
    const unsigned int bb = block_lane;
    const bool valid = row_ok && (bb < n_blocks_in);

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q6_K_BYTES;
    uint32_t apack[16];
    int      sch[4];
    float    dq = 0.0f;
    if (valid) {
        const uint8_t* x_blk = row_base + (uint64_t)bb * BLOCK_Q6_K_BYTES;
        const uint8_t* ql = x_blk + 0;
        const uint8_t* qh = x_blk + 128;
        const int8_t*  sc = (const int8_t*)(x_blk + 192);
        dq = f16_to_f32_t(*(const uint16_t*)(x_blk + 208));
        const unsigned int hf = sub >> 1u, up = sub & 1u;
        const uint8_t* qlh = ql + (size_t)hf*64u;
        const uint8_t* qhh = qh + (size_t)hf*32u;
        const int8_t*  sch4 = sc + (size_t)hf*8u + (size_t)up*4u;
        #pragma unroll
        for (int g = 0; g < 4; g++) sch[g] = (int)sch4[g];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            #pragma unroll
            for (int qq = 0; qq < 16; qq += 4) {
                const int r0 = g*16 + qq;
                uint32_t ap = 0u;
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    const int r = r0 + t;
                    const unsigned int shift = up*4u + ((r >= 32) ? 2u : 0u);
                    const uint8_t nib = up ? (uint8_t)(qlh[r] >> 4) : (uint8_t)(qlh[r] & 0x0Fu);
                    const uint8_t hi2 = (uint8_t)((qhh[r & 31] >> shift) & 3u);
                    const int8_t  av  = (int8_t)((int)(nib | (hi2 << 4)) - 32);
                    ap |= ((uint32_t)(uint8_t)av) << (t*8);
                }
                apack[g*4 + qq/4] = ap;
            }
        }
    }

    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* y = xq_base
            + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride
            + (size_t)bb*BLOCK_Q8_K_BYTES;
        float acc = 0.0f;
        if (valid) {
            const float yd = *(const float*)(y + 0);
            const unsigned int hf = sub >> 1u, up = sub & 1u;
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)hf*128u + (size_t)up*64u;
            int isum = 0;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                int dot = 0;
                #pragma unroll
                for (int qq = 0; qq < 16; qq += 4) {
                    dot = sudot4_t((int32_t)apack[g*4 + qq/4],
                                   load_dword_i8_t(q8 + g*16 + qq), dot);
                }
                isum += sch[g] * dot;
            }
            acc = yd * dq * (float)isum;
        }
        acc = warp_sum16_f32_t(acc);
        if (local == 0 && row_ok) {
            atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc);
        }
    }
}

extern "C" __global__ void q4_k_down_tiled_reg_w32_col_nolds_wide(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    const int n_members = group_count[e];
    if (n_members == 0) return;

    const unsigned int tid   = threadIdx.x;
    const unsigned int wave  = tid >> 5;
    const unsigned int lane  = tid & 31u;
    const unsigned int half  = lane >> 4u;
    const unsigned int local = lane & 15u;
    const unsigned int block_lane = local >> 2u;
    const unsigned int sub        = local & 3u;
    const unsigned int rows_per_wg = (blockDim.x >> 5) * 2u;
    const unsigned int row = blockIdx.x * rows_per_wg + wave * 2u + half;
    const bool row_ok = (row < n_rows);
    const unsigned int bb = block_lane;
    const bool valid = row_ok && (bb < n_blocks_in);

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
    uint32_t w_qs[8];
    float w_xd = 0.0f, w_xdm = 0.0f;
    int   w_da = 0, w_ma = 0, w_db = 0, w_mb = 0;
    if (valid) {
        const uint8_t* w_blk = row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        w_xd  = f16_to_f32_t(*(const uint16_t*)(w_blk + 0));
        w_xdm = f16_to_f32_t(*(const uint16_t*)(w_blk + 2));
        uint8_t da,ma,db,mb;
        get_scale_min_q4k_t(2*sub,   w_blk+4, &da,&ma);
        get_scale_min_q4k_t(2*sub+1, w_blk+4, &db,&mb);
        w_da=da; w_ma=ma; w_db=db; w_mb=mb;
        const uint8_t* qs = w_blk + 16 + (size_t)sub*32u;
        #pragma unroll
        for (int i = 0; i < 8; i++) w_qs[i] = (uint32_t)load_dword_t(qs + i*4);
    }

    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* y = xq_base
            + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride
            + (size_t)bb*BLOCK_Q8_K_BYTES;
        float acc = 0.0f;
        if (valid) {
            const float yd = *(const float*)(y + 0);
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
            const int16_t* bs = (const int16_t*)(y + 260);
            int da=0, db=0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
                da = sudot4_t((int32_t)(w_qs[i] & 0x0f0f0f0fu), q8lo, da);
                db = sudot4_t((int32_t)((w_qs[i] >> 4) & 0x0f0f0f0fu), q8hi, db);
            }
            const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
            const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
            acc = yd*w_xd*((float)w_da*da + (float)w_db*db)
                - yd*w_xdm*((float)w_ma*bsa + (float)w_mb*bsb);
        }
        acc = warp_sum16_f32_t(acc);
        if (local == 0 && row_ok) {
            atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc);
        }
    }
}

extern "C" __global__ void q4_k_down_tiled_reg_w32_col_nolds(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    const int n_members = group_count[e];
    if (n_members == 0) return;

    const unsigned int tid   = threadIdx.x;
    const unsigned int wave  = tid >> 5;
    const unsigned int lane  = tid & 31u;
    const unsigned int half  = lane >> 4u;
    const unsigned int local = lane & 15u;
    const unsigned int block_lane = local >> 2u;
    const unsigned int sub        = local & 3u;
    const unsigned int row = blockIdx.x * 16u + wave * 2u + half;
    const bool row_ok = (row < n_rows);
    const unsigned int bb = block_lane;
    const bool valid = row_ok && (bb < n_blocks_in);

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
    uint32_t w_qs[8];
    float w_xd = 0.0f, w_xdm = 0.0f;
    int   w_da = 0, w_ma = 0, w_db = 0, w_mb = 0;
    if (valid) {
        const uint8_t* w_blk = row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        w_xd  = f16_to_f32_t(*(const uint16_t*)(w_blk + 0));
        w_xdm = f16_to_f32_t(*(const uint16_t*)(w_blk + 2));
        uint8_t da,ma,db,mb;
        get_scale_min_q4k_t(2*sub,   w_blk+4, &da,&ma);
        get_scale_min_q4k_t(2*sub+1, w_blk+4, &db,&mb);
        w_da=da; w_ma=ma; w_db=db; w_mb=mb;
        const uint8_t* qs = w_blk + 16 + (size_t)sub*32u;
        #pragma unroll
        for (int i = 0; i < 8; i++) w_qs[i] = (uint32_t)load_dword_t(qs + i*4);
    }

    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* y = xq_base
            + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride
            + (size_t)bb*BLOCK_Q8_K_BYTES;
        float acc = 0.0f;
        if (valid) {
            const float yd = *(const float*)(y + 0);
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
            const int16_t* bs = (const int16_t*)(y + 260);
            int da=0, db=0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
                da = sudot4_t((int32_t)(w_qs[i] & 0x0f0f0f0fu), q8lo, da);
                db = sudot4_t((int32_t)((w_qs[i] >> 4) & 0x0f0f0f0fu), q8hi, db);
            }
            const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
            const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
            acc = yd*w_xd*((float)w_da*da + (float)w_db*db)
                - yd*w_xdm*((float)w_ma*bsa + (float)w_mb*bsb);
        }
        acc = warp_sum16_f32_t(acc);
        if (local == 0 && row_ok) {
            atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc);
        }
    }
}

// ===========================================================================
// ABLATION variants of the no-LDS down kernels (TIMING ONLY, parity broken).
// ABL_DOT=0 stubs the dp4a MAC chain; ABL_REDUCE=0 stubs the 16-lane warp
// reduction; ABL_ATOMIC=0 stubs the global atomicAdd. Attributes per-member
// cost so rocprofv3 can show ms/component. Same signature as the production
// _nolds kernels. Select via LAGUNA_DOWN_ABL in down_reg_q{6,4}k_w32_col.
// ===========================================================================
#ifndef ABL_DOT
#define ABL_DOT 1
#endif
#ifndef ABL_REDUCE
#define ABL_REDUCE 1
#endif
#ifndef ABL_ATOMIC
#define ABL_ATOMIC 1
#endif

#define Q6_DOWN_NOLDS_ABL_BODY \
    const unsigned int e = blockIdx.y; \
    const int n_members = group_count[e]; \
    if (n_members == 0) return; \
    const unsigned int tid   = threadIdx.x; \
    const unsigned int wave  = tid >> 5; \
    const unsigned int lane  = tid & 31u; \
    const unsigned int half  = lane >> 4u; \
    const unsigned int local = lane & 15u; \
    const unsigned int block_lane = local >> 2u; \
    const unsigned int sub        = local & 3u; \
    const unsigned int row = blockIdx.x * 16u + wave * 2u + half; \
    const bool row_ok = (row < n_rows); \
    const unsigned int bb = block_lane; \
    const bool valid = row_ok && (bb < n_blocks_in); \
    const uint8_t* row_base = w_base + (uint64_t)e*dbpe \
                            + (uint64_t)row * n_blocks_in * BLOCK_Q6_K_BYTES; \
    uint32_t apack[16]; int sch[4]; float dq = 0.0f; \
    if (valid) { \
        const uint8_t* x_blk = row_base + (uint64_t)bb * BLOCK_Q6_K_BYTES; \
        const uint8_t* ql = x_blk + 0; const uint8_t* qh = x_blk + 128; \
        const int8_t*  sc = (const int8_t*)(x_blk + 192); \
        dq = f16_to_f32_t(*(const uint16_t*)(x_blk + 208)); \
        const unsigned int hf = sub >> 1u, up = sub & 1u; \
        const uint8_t* qlh = ql + (size_t)hf*64u; const uint8_t* qhh = qh + (size_t)hf*32u; \
        const int8_t*  sch4 = sc + (size_t)hf*8u + (size_t)up*4u; \
        _Pragma("unroll") for (int g = 0; g < 4; g++) sch[g] = (int)sch4[g]; \
        _Pragma("unroll") for (int g = 0; g < 4; g++) { \
            _Pragma("unroll") for (int qq = 0; qq < 16; qq += 4) { \
                const int r0 = g*16 + qq; uint32_t ap = 0u; \
                _Pragma("unroll") for (int t = 0; t < 4; t++) { \
                    const int r = r0 + t; \
                    const unsigned int shift = up*4u + ((r >= 32) ? 2u : 0u); \
                    const uint8_t nib = up ? (uint8_t)(qlh[r] >> 4) : (uint8_t)(qlh[r] & 0x0Fu); \
                    const uint8_t hi2 = (uint8_t)((qhh[r & 31] >> shift) & 3u); \
                    const int8_t  av  = (int8_t)((int)(nib | (hi2 << 4)) - 32); \
                    ap |= ((uint32_t)(uint8_t)av) << (t*8); \
                } apack[g*4 + qq/4] = ap; \
            } \
        } \
    } \
    for (int m = 0; m < n_members; m++) { \
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m]; \
        const unsigned int b    = (unsigned int)(packed >> 16); \
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu); \
        const uint8_t* y = xq_base + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride + (size_t)bb*BLOCK_Q8_K_BYTES; \
        float acc = 0.0f; \
        if (valid) { \
            const float yd = *(const float*)(y + 0); \
            const unsigned int hf = sub >> 1u, up = sub & 1u; \
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)hf*128u + (size_t)up*64u; \
            int isum = 0; \
            _Pragma("unroll") for (int g = 0; g < 4; g++) { \
                int dot = 0; \
                _Pragma("unroll") for (int qq = 0; qq < 16; qq += 4) { \
                    int32_t wv = ABL_DOT ? (int32_t)apack[g*4 + qq/4] : 0; \
                    dot = sudot4_t(wv, load_dword_i8_t(q8 + g*16 + qq), dot); \
                } isum += sch[g] * dot; \
            } acc = yd * dq * (float)isum; \
        } \
        if (ABL_REDUCE) acc = warp_sum16_f32_t(acc); \
        if (ABL_ATOMIC) { if (local == 0 && row_ok) atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc); } \
        else { if (local == 0 && row_ok && acc == -123456.0f) out[0] = acc; } \
    }

#define Q4_DOWN_NOLDS_ABL_BODY \
    const unsigned int e = blockIdx.y; \
    const int n_members = group_count[e]; \
    if (n_members == 0) return; \
    const unsigned int tid   = threadIdx.x; \
    const unsigned int wave  = tid >> 5; \
    const unsigned int lane  = tid & 31u; \
    const unsigned int half  = lane >> 4u; \
    const unsigned int local = lane & 15u; \
    const unsigned int block_lane = local >> 2u; \
    const unsigned int sub        = local & 3u; \
    const unsigned int row = blockIdx.x * 16u + wave * 2u + half; \
    const bool row_ok = (row < n_rows); \
    const unsigned int bb = block_lane; \
    const bool valid = row_ok && (bb < n_blocks_in); \
    const uint8_t* row_base = w_base + (uint64_t)e*dbpe \
                            + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES; \
    uint32_t w_qs[8]; float w_xd = 0.0f, w_xdm = 0.0f; \
    int w_da = 0, w_ma = 0, w_db = 0, w_mb = 0; \
    if (valid) { \
        const uint8_t* w_blk = row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES; \
        w_xd  = f16_to_f32_t(*(const uint16_t*)(w_blk + 0)); \
        w_xdm = f16_to_f32_t(*(const uint16_t*)(w_blk + 2)); \
        uint8_t da,ma,db,mb; \
        get_scale_min_q4k_t(2*sub, w_blk+4, &da,&ma); \
        get_scale_min_q4k_t(2*sub+1, w_blk+4, &db,&mb); \
        w_da=da; w_ma=ma; w_db=db; w_mb=mb; \
        const uint8_t* qs = w_blk + 16 + (size_t)sub*32u; \
        _Pragma("unroll") for (int i = 0; i < 8; i++) w_qs[i] = (uint32_t)load_dword_t(qs + i*4); \
    } \
    for (int m = 0; m < n_members; m++) { \
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m]; \
        const unsigned int b    = (unsigned int)(packed >> 16); \
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu); \
        const uint8_t* y = xq_base + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride + (size_t)bb*BLOCK_Q8_K_BYTES; \
        float acc = 0.0f; \
        if (valid) { \
            const float yd = *(const float*)(y + 0); \
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u; \
            const int16_t* bs = (const int16_t*)(y + 260); \
            int da=0, db=0; \
            _Pragma("unroll") for (int i = 0; i < 8; i++) { \
                const int32_t q8lo = load_dword_i8_t(q8 + i*4); \
                const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4); \
                int32_t wl = ABL_DOT ? (int32_t)(w_qs[i] & 0x0f0f0f0fu) : 0; \
                int32_t wh = ABL_DOT ? (int32_t)((w_qs[i] >> 4) & 0x0f0f0f0fu) : 0; \
                da = sudot4_t(wl, q8lo, da); db = sudot4_t(wh, q8hi, db); \
            } \
            const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u]; \
            const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u]; \
            acc = yd*w_xd*((float)w_da*da + (float)w_db*db) - yd*w_xdm*((float)w_ma*bsa + (float)w_mb*bsb); \
        } \
        if (ABL_REDUCE) acc = warp_sum16_f32_t(acc); \
        if (ABL_ATOMIC) { if (local == 0 && row_ok) atomicAdd(&out[(size_t)b*(size_t)n_rows+(size_t)row], acc); } \
        else { if (local == 0 && row_ok && acc == -123456.0f) out[0] = acc; } \
    }

#define DOWN_ABL_SIG(NAME) \
extern "C" __global__ void NAME( \
        float * __restrict__ out, const uint8_t * __restrict__ w_base, \
        const uint8_t * __restrict__ xq_base, const int * __restrict__ group_count, \
        const int * __restrict__ expert_members, unsigned int dbpe, \
        unsigned int xq_slot_stride, unsigned int n_used, unsigned int max_per_expert, \
        unsigned int n_rows, unsigned int n_blocks_in)

DOWN_ABL_SIG(q6_k_down_nolds_abl_full)     { Q6_DOWN_NOLDS_ABL_BODY }
DOWN_ABL_SIG(q4_k_down_nolds_abl_full)     { Q4_DOWN_NOLDS_ABL_BODY }
#undef ABL_DOT
#define ABL_DOT 0
DOWN_ABL_SIG(q6_k_down_nolds_abl_nodot)    { Q6_DOWN_NOLDS_ABL_BODY }
DOWN_ABL_SIG(q4_k_down_nolds_abl_nodot)    { Q4_DOWN_NOLDS_ABL_BODY }
#undef ABL_DOT
#define ABL_DOT 1
#undef ABL_REDUCE
#define ABL_REDUCE 0
DOWN_ABL_SIG(q6_k_down_nolds_abl_noreduce) { Q6_DOWN_NOLDS_ABL_BODY }
DOWN_ABL_SIG(q4_k_down_nolds_abl_noreduce) { Q4_DOWN_NOLDS_ABL_BODY }
#undef ABL_REDUCE
#define ABL_REDUCE 1
#undef ABL_ATOMIC
#define ABL_ATOMIC 0
DOWN_ABL_SIG(q6_k_down_nolds_abl_noatomic) { Q6_DOWN_NOLDS_ABL_BODY }
DOWN_ABL_SIG(q4_k_down_nolds_abl_noatomic) { Q4_DOWN_NOLDS_ABL_BODY }
#undef ABL_ATOMIC
#define ABL_ATOMIC 1

// ===========================================================================
// ATOMIC-FREE down: PARTIAL-STORE + slot-reduce. The _nolds down kernels above
// atomicAdd each member's (b,slot) contribution into the shared out[b,row]; the
// TOPK experts for a token contend on the same address → global-atomic
// throughput is ~23-35% of the down kernel (rocprofv3 noatomic ablation). Since
// (b,slot) → exactly one expert, each member owns a UNIQUE slice
// part[(b*n_used+slot)*n_rows + row], so it can PLAIN-STORE (no atomic, no
// contention). A cheap streaming reduce (moe_down_reduce_slots) then sums the
// n_used=TOPK slots into acc[b*n_rows+row]. Bit-identical dp4a; the only numeric
// change is the accumulation is a fixed-order slot sum instead of atomic order.
// `part` need NOT be pre-zeroed: every (b,slot,row) with row<n_rows is written
// (n_rows % 16 == 0 so no partial rows), and the reduce only reads slots < TOPK.
// ===========================================================================
extern "C" __global__ void q6_k_down_tiled_reg_w32_col_nolds_part(
        float          * __restrict__ part,          // [B, n_used, n_rows]
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    const int n_members = group_count[e];
    if (n_members == 0) return;

    const unsigned int tid   = threadIdx.x;
    const unsigned int wave  = tid >> 5;
    const unsigned int lane  = tid & 31u;
    const unsigned int half  = lane >> 4u;
    const unsigned int local = lane & 15u;
    const unsigned int block_lane = local >> 2u;
    const unsigned int sub        = local & 3u;
    const unsigned int rows_per_wg = (blockDim.x >> 5) * 2u;
    const unsigned int row = blockIdx.x * rows_per_wg + wave * 2u + half;
    const bool row_ok = (row < n_rows);
    const unsigned int bb = block_lane;
    const bool valid = row_ok && (bb < n_blocks_in);

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q6_K_BYTES;
    uint32_t apack[16];
    int      sch[4];
    float    dq = 0.0f;
    if (valid) {
        const uint8_t* x_blk = row_base + (uint64_t)bb * BLOCK_Q6_K_BYTES;
        const uint8_t* ql = x_blk + 0;
        const uint8_t* qh = x_blk + 128;
        const int8_t*  sc = (const int8_t*)(x_blk + 192);
        dq = f16_to_f32_t(*(const uint16_t*)(x_blk + 208));
        const unsigned int hf = sub >> 1u, up = sub & 1u;
        const uint8_t* qlh = ql + (size_t)hf*64u;
        const uint8_t* qhh = qh + (size_t)hf*32u;
        const int8_t*  sch4 = sc + (size_t)hf*8u + (size_t)up*4u;
        #pragma unroll
        for (int g = 0; g < 4; g++) sch[g] = (int)sch4[g];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            #pragma unroll
            for (int qq = 0; qq < 16; qq += 4) {
                const int r0 = g*16 + qq;
                uint32_t ap = 0u;
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    const int r = r0 + t;
                    const unsigned int shift = up*4u + ((r >= 32) ? 2u : 0u);
                    const uint8_t nib = up ? (uint8_t)(qlh[r] >> 4) : (uint8_t)(qlh[r] & 0x0Fu);
                    const uint8_t hi2 = (uint8_t)((qhh[r & 31] >> shift) & 3u);
                    const int8_t  av  = (int8_t)((int)(nib | (hi2 << 4)) - 32);
                    ap |= ((uint32_t)(uint8_t)av) << (t*8);
                }
                apack[g*4 + qq/4] = ap;
            }
        }
    }

    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* y = xq_base
            + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride
            + (size_t)bb*BLOCK_Q8_K_BYTES;
        float acc = 0.0f;
        if (valid) {
            const float yd = *(const float*)(y + 0);
            const unsigned int hf = sub >> 1u, up = sub & 1u;
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)hf*128u + (size_t)up*64u;
            int isum = 0;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                int dot = 0;
                #pragma unroll
                for (int qq = 0; qq < 16; qq += 4) {
                    dot = sudot4_t((int32_t)apack[g*4 + qq/4],
                                   load_dword_i8_t(q8 + g*16 + qq), dot);
                }
                isum += sch[g] * dot;
            }
            acc = yd * dq * (float)isum;
        }
        acc = warp_sum16_f32_t(acc);
        if (local == 0 && row_ok) {
            part[((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)n_rows + (size_t)row] = acc;
        }
    }
}

extern "C" __global__ void q4_k_down_tiled_reg_w32_col_nolds_part(
        float          * __restrict__ part,          // [B, n_used, n_rows]
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ group_count,
        const int      * __restrict__ expert_members,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     max_per_expert,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int e = blockIdx.y;
    const int n_members = group_count[e];
    if (n_members == 0) return;

    const unsigned int tid   = threadIdx.x;
    const unsigned int wave  = tid >> 5;
    const unsigned int lane  = tid & 31u;
    const unsigned int half  = lane >> 4u;
    const unsigned int local = lane & 15u;
    const unsigned int block_lane = local >> 2u;
    const unsigned int sub        = local & 3u;
    const unsigned int rows_per_wg = (blockDim.x >> 5) * 2u;
    const unsigned int row = blockIdx.x * rows_per_wg + wave * 2u + half;
    const bool row_ok = (row < n_rows);
    const unsigned int bb = block_lane;
    const bool valid = row_ok && (bb < n_blocks_in);

    const uint8_t* row_base = w_base + (uint64_t)e*dbpe
                            + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
    uint32_t w_qs[8];
    float w_xd = 0.0f, w_xdm = 0.0f;
    int   w_da = 0, w_ma = 0, w_db = 0, w_mb = 0;
    if (valid) {
        const uint8_t* w_blk = row_base + (uint64_t)bb * BLOCK_Q4_K_BYTES;
        w_xd  = f16_to_f32_t(*(const uint16_t*)(w_blk + 0));
        w_xdm = f16_to_f32_t(*(const uint16_t*)(w_blk + 2));
        uint8_t da,ma,db,mb;
        get_scale_min_q4k_t(2*sub,   w_blk+4, &da,&ma);
        get_scale_min_q4k_t(2*sub+1, w_blk+4, &db,&mb);
        w_da=da; w_ma=ma; w_db=db; w_mb=mb;
        const uint8_t* qs = w_blk + 16 + (size_t)sub*32u;
        #pragma unroll
        for (int i = 0; i < 8; i++) w_qs[i] = (uint32_t)load_dword_t(qs + i*4);
    }

    for (int m = 0; m < n_members; m++) {
        const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)m];
        const unsigned int b    = (unsigned int)(packed >> 16);
        const unsigned int slot = (unsigned int)(packed & 0xFFFFu);
        const uint8_t* y = xq_base
            + ((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)xq_slot_stride
            + (size_t)bb*BLOCK_Q8_K_BYTES;
        float acc = 0.0f;
        if (valid) {
            const float yd = *(const float*)(y + 0);
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
            const int16_t* bs = (const int16_t*)(y + 260);
            int da=0, db=0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
                da = sudot4_t((int32_t)(w_qs[i] & 0x0f0f0f0fu), q8lo, da);
                db = sudot4_t((int32_t)((w_qs[i] >> 4) & 0x0f0f0f0fu), q8hi, db);
            }
            const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
            const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
            acc = yd*w_xd*((float)w_da*da + (float)w_db*db)
                - yd*w_xdm*((float)w_ma*bsa + (float)w_mb*bsb);
        }
        acc = warp_sum16_f32_t(acc);
        if (local == 0 && row_ok) {
            part[((size_t)b*(size_t)n_used+(size_t)slot)*(size_t)n_rows + (size_t)row] = acc;
        }
    }
}

// Sum the n_used(=TOPK) slot-partials into acc[b*n_rows + row]. One thread per
// (b,row); grid = ceil(B*n_rows/256). part is [B, n_used, n_rows].
extern "C" __global__ void moe_down_reduce_slots(
        float          * __restrict__ acc,           // [B, n_rows]
        const float    * __restrict__ part,          // [B, n_used, n_rows]
        unsigned int     n_rows,
        unsigned int     n_used,
        unsigned int     total) {                     // = B * n_rows
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int b   = idx / n_rows;
    const unsigned int row = idx - b * n_rows;
    const float* base = part + (size_t)b * (size_t)n_used * (size_t)n_rows + (size_t)row;
    float s = 0.0f;
    for (unsigned int slot = 0; slot < n_used; slot++) {
        s += base[(size_t)slot * (size_t)n_rows];
    }
    acc[idx] = s;
}

// ---- Q5_K dense GEMM, wide register-tiled (unsloth UD het/ shexp+q_a) ----
// Clone of q4_k_dense_gemm_dp4a_r32_nolds with the 5th bit plane: value =
// nibble | (qh bit << 4) ∈ [0,32) packed bytewise straight into the dp4a
// operand. Per-lane registers: qs 2×8 + qh 2×8 dwords + scales.
// grid (n_rows/32, 1, 1), block (1024,1,1); n_blk <= 16.
extern "C" __global__ void q5_k_dense_gemm_dp4a_r32_nolds(
        float          * __restrict__ out,   // [B, n_rows]
        const uint8_t  * __restrict__ w,     // [n_rows, n_blk*176]
        const uint8_t  * __restrict__ xq,    // [B, n_blk*292] Q8_K
        unsigned int     B,
        unsigned int     n_rows,
        unsigned int     n_blk) {
    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * DENSE_ROWS_PER_WG + row_in_block;
    if (row >= n_rows) return;
    const unsigned int block_lane   = lane >> 2u;
    const unsigned int sub          = lane & 3u;   // 64-elem group g

    const uint8_t* row_base = w + (uint64_t)row * n_blk * BLOCK_Q5_K_BYTES;

    unsigned int bb_of[DENSE_MAXB];
    bool  valid[DENSE_MAXB];
    float xd[DENSE_MAXB], xdm[DENSE_MAXB];
    int   da[DENSE_MAXB], ma[DENSE_MAXB], db[DENSE_MAXB], mb[DENSE_MAXB];
    uint32_t vlo[DENSE_MAXB][8];
    uint32_t vhi[DENSE_MAXB][8];
    #pragma unroll
    for (int k = 0; k < DENSE_MAXB; k++) {
        const unsigned int bb = block_lane + (unsigned int)k * 8u;
        bb_of[k] = bb;
        valid[k] = (bb < n_blk);
        if (!valid[k]) continue;
        const uint8_t* wg = row_base + (uint64_t)bb * BLOCK_Q5_K_BYTES;
        xd[k]  = f16_to_f32_t(*(const uint16_t*)(wg + 0));
        xdm[k] = f16_to_f32_t(*(const uint16_t*)(wg + 2));
        uint8_t d0,m0,d1,m1;
        get_scale_min_q4k_t(2*sub,   wg+4, &d0,&m0);
        get_scale_min_q4k_t(2*sub+1, wg+4, &d1,&m1);
        da[k]=d0; ma[k]=m0; db[k]=d1; mb[k]=m1;
        const uint8_t* qh = wg + 16;
        const uint8_t* q  = wg + 48 + (size_t)sub*32u;
        const unsigned int s_lo = 2u*sub;
        const unsigned int s_hi = 2u*sub + 1u;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t qsd = (uint32_t)load_dword_t(q + i*4);
            const uint32_t qhd = (uint32_t)load_dword_t(qh + i*4);
            vlo[k][i] = (qsd & 0x0f0f0f0fu)
                      | (((qhd >> s_lo) & 0x01010101u) << 4);
            vhi[k][i] = ((qsd >> 4) & 0x0f0f0f0fu)
                      | (((qhd >> s_hi) & 0x01010101u) << 4);
        }
    }

    const unsigned int xq_bytes = n_blk * BLOCK_Q8_K_BYTES;
    for (unsigned int bcol = 0; bcol < B; bcol++) {
        const uint8_t* xj = xq + (size_t)bcol * (size_t)xq_bytes;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < DENSE_MAXB; k++) {
            if (!valid[k]) continue;
            const uint8_t* y = xj + (size_t)bb_of[k] * BLOCK_Q8_K_BYTES;
            const float yd = *(const float*)(y + 0);
            const int8_t* q8 = (const int8_t*)(y + 4) + (size_t)sub*64u;
            const int16_t* bs = (const int16_t*)(y + 260);
            int lo=0, hi=0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                lo = sudot4_t((int32_t)vlo[k][i], load_dword_i8_t(q8 + i*4), lo);
                hi = sudot4_t((int32_t)vhi[k][i], load_dword_i8_t(q8 + 32 + i*4), hi);
            }
            const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
            const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
            acc += yd*xd[k]*((float)da[k]*lo + (float)db[k]*hi)
                 - yd*xdm[k]*((float)ma[k]*bsa + (float)mb[k]*bsb);
        }
        acc = warp_sum_f32_t(acc);
        if (lane == 0) out[(size_t)bcol * (size_t)n_rows + (size_t)row] = acc;
    }
}
