// q4_k_matvec_par_batched — batched MoE matvec for Q4_K weights × Q8_K
// activations. Structural mirror of q2_k_matvec_par_batched.hip with the
// quant decode swapped for Q4_K's 4-bit scheme.
//
// Q4_K block layout (144 bytes per 256 elements):
//   offset  0: ggml_half d        (super-block scale for sub-block scales)
//   offset  2: ggml_half dmin     (super-block scale for sub-block mins)
//   offset  4: uint8 scales[12]   (packed 6-bit (d_sub, m_sub) for 8 sub-blocks)
//   offset 16: uint8 qs[128]      (4-bit packed weights; 2 weights per byte)
//
// The 8 sub-blocks decode (d_sub, m_sub) via the ggml `get_scale_min_k4` packing:
//   j < 4:  d_sub = scales[j]   & 0x3F;  m_sub = scales[j+4] & 0x3F
//   j >= 4: d_sub = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
//           m_sub = (scales[j+4] >>  4) | ((scales[j-0] >> 6) << 4)
//
// Per element: w = d_super * d_sub * q4 - dmin_super * m_sub. Dot of a
// 32-element sub-block with Q8_K activation `(yd * qs8)` factors to
//   block_result = d_super * sum(d_sub_k * dot(q4_k, qs8_k))
//                - dmin_super * sum(m_sub_k * bsum_k)
// where bsum_k = sum of the qs8 values in sub-block k (Q8_K stores these
// pre-computed as int16 bsums[16] in groups of 16; per sub-block of 32
// elements we sum bsums[2k] + bsums[2k+1]).
//
// Lane mapping (mirrors Q2_K par layout): 8 rows per workgroup, 1 warp
// per row, 32 lanes per warp. With n_blocks_in = N_FF_EXP/256 = 8 for
// V4-Flash, 4 lanes cover each Q4_K super-block (each lane handles a
// "quarter" = 64 elements = 2 sub-blocks).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q4_K_BYTES 144
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float f16_to_f32_q4k(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4_q4k(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t load_dword_q4k(const uint8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

__device__ __forceinline__ int32_t load_dword_i8_q4k(const int8_t* p) {
    return load_dword_q4k((const uint8_t*)p);
}

// Decode 6-bit (d_sub, m_sub) for sub-block j in [0, 8). Matches ggml's
// get_scale_min_k4 exactly.
__device__ __forceinline__ void get_scale_min_q4k(
        int j, const uint8_t* scales, uint8_t* d_out, uint8_t* m_out) {
    if (j < 4) {
        *d_out = scales[j]     & 0x3F;
        *m_out = scales[j + 4] & 0x3F;
    } else {
        *d_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m_out = (scales[j + 4] >>  4)  | ((scales[j - 0] >> 6) << 4);
    }
}

// Dot of 32 4-bit weights × 32 int8 activations.
//   qs points to the 32-byte chunk holding both halves of a 64-element block:
//     low nibbles  → sub-block 2k (first 32 elements)
//     high nibbles → sub-block 2k+1 (next 32 elements)
//   `high=false` selects the low-nibble half; `high=true` the high-nibble half.
// q8 points to the corresponding 32 int8 activations.
__device__ __forceinline__ int32_t dot_q4_32_q4k(
        const uint8_t* qs, const int8_t* q8, bool high) {
    int32_t sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; i += 4) {
        const int32_t qs_dword = load_dword_q4k(qs + i);
        // Each byte holds 2 nibbles. To pack 4 consecutive 4-bit weights
        // as 4 signed int8s in a uint32 (for sudot4), mask the low or
        // shift+mask the high nibbles. 0x0F0F0F0F broadcasts the mask
        // across all 4 bytes.
        const int32_t v = high
            ? (int32_t)((((uint32_t)qs_dword) >> 4) & 0x0f0f0f0fu)
            : (qs_dword & 0x0f0f0f0f);
        sum = sudot4_q4k(v, load_dword_i8_q4k(q8 + i), sum);
    }
    return sum;
}

// Quarter-block dot: one of 4 lanes per super-block. `sub` in [0, 4)
// selects which 64-element quarter (= 2 of 8 sub-blocks). Returns the
// lane's contribution to the super-block result; sum 4 lanes to recover.
__device__ __forceinline__ float dot_block_quarter_q4k(
        const uint8_t* x_blk, const uint8_t* y_blk, unsigned int sub) {
    const uint16_t xd_bits   = *(const uint16_t*)(x_blk + 0);
    const uint16_t xdm_bits  = *(const uint16_t*)(x_blk + 2);
    const uint8_t* scales    = x_blk + 4;
    const uint8_t* qs_base   = x_blk + 16;

    const float    yd        = *(const float*)(y_blk + 0);
    const int8_t*  q8_base   = (const int8_t*)(y_blk + 4);
    const int16_t* bsums     = (const int16_t*)(y_blk + 260);

    const float d_super = yd * f16_to_f32_q4k(xd_bits);
    const float m_super = yd * f16_to_f32_q4k(xdm_bits);

    // This lane handles sub-blocks (2*sub) and (2*sub+1).
    const int j_a = (int)(2u * sub);
    const int j_b = j_a + 1;

    uint8_t d_a, m_a, d_b, m_b;
    get_scale_min_q4k(j_a, scales, &d_a, &m_a);
    get_scale_min_q4k(j_b, scales, &d_b, &m_b);

    // qs: 64-element chunk shared by sub-blocks (j_a, j_b) starts at sub*32.
    const uint8_t* qs_chunk = qs_base + (size_t)sub * 32u;
    const int8_t*  q8_chunk = q8_base + (size_t)sub * 64u;

    // Q8_K bsums cover 16 elements each. Sub-block of 32 elements = 2 bsums.
    // Sub-block j_a (= 2*sub):     covers q8[64*sub..64*sub+31]     = bsums[4*sub  ] + bsums[4*sub+1]
    // Sub-block j_b (= 2*sub+1):   covers q8[64*sub+32..64*sub+63]  = bsums[4*sub+2] + bsums[4*sub+3]
    const int32_t bsum_a =
        (int32_t)bsums[4u * sub + 0u] + (int32_t)bsums[4u * sub + 1u];
    const int32_t bsum_b =
        (int32_t)bsums[4u * sub + 2u] + (int32_t)bsums[4u * sub + 3u];

    const int32_t dot_a = dot_q4_32_q4k(qs_chunk, q8_chunk,        false);
    const int32_t dot_b = dot_q4_32_q4k(qs_chunk, q8_chunk + 32,   true);

    return d_super * ((float)d_a * (float)dot_a + (float)d_b * (float)dot_b)
         - m_super * ((float)m_a * (float)bsum_a + (float)m_b * (float)bsum_b);
}

__device__ __forceinline__ float warp_sum_f32_q4k(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// q4_k_pair_matvec_fused_swiglu_batch — fused gate × up × swiglu × ew
// for routed-MoE gate+up step (the MTP equivalent of iq2's
// pair_matvec_fused_swiglu_batch). Per slot s in [0, n_used):
//   gate_s[r] = dot(W_gate[selected[s], r], xq)
//   up_s[r]   = dot(W_up[selected[s], r],   xq)
//   mid[s, r] = silu(clamp(gate_s)) * clamp(up_s) * expert_w[s]
//
// Layout:
//   grid:  (n_rows / 8, n_used, 1)
//   block: (256, 1, 1)
//   8 rows per workgroup, 32 lanes per warp, 4 lanes per super-block,
//   8 super-blocks per warp (for n_blocks_in = N_EMBD/256 = 16, each
//   lane iterates 2 times over its super-block stripe).
extern "C" __global__ void q4_k_pair_matvec_fused_swiglu_batch(
        float          * __restrict__ mid,           // [n_used, n_rows]
        const uint8_t  * __restrict__ gate_w_base,
        const uint8_t  * __restrict__ up_w_base,
        const uint8_t  * __restrict__ xq,            // [n_blocks_in * BLOCK_Q8_K_BYTES]
        const float    * __restrict__ expert_w,      // [n_used]
        const int      * __restrict__ selected,      // [n_used]
        unsigned int     gate_bpe,
        unsigned int     up_bpe,
        float            clamp,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int slot         = blockIdx.y;
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    const int e = selected[slot];
    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;

    for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_Q4_K_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_Q4_K_BYTES;
        const uint8_t* y   = xq            + (uint64_t)b * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_quarter_q4k(w_g, y, sub);
        u_sum += dot_block_quarter_q4k(w_u, y, sub);
    }

    g_sum = warp_sum_f32_q4k(g_sum);
    u_sum = warp_sum_f32_q4k(u_sum);

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

// q4_k_matvec_par_batched — direct port of q2_k_matvec_par_batched with
// the dot_block_quarter swap for Q4_K. Writes summed result over n_used
// experts to out[n_rows]; routing comes from selected[n_used] (expert
// indices) and per-slot midq comes from xq_base + s*xq_slot_stride.
//
// Launch:
//   grid:  (n_rows / 8, 1, 1)
//   block: (256, 1, 1)
//   shared_mem: 0
extern "C" __global__ void q4_k_matvec_par_batched(
        float          * __restrict__ out,           // [n_rows]
        const uint8_t  * __restrict__ w_base,        // whole weight tensor
        const uint8_t  * __restrict__ xq_base,       // slot-major midq_cat
        const int      * __restrict__ selected,      // [n_used]
        unsigned int     dbpe,                       // weight bytes per expert
        unsigned int     xq_slot_stride,             // xq bytes per slot
        unsigned int     n_used,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    // 4 lanes per super-block; 8 super-blocks per warp (for n_blocks_in=8).
    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    float acc = 0.0f;
    for (unsigned int s = 0; s < n_used; s++) {
        const int e = selected[s];
        const uint8_t* w_expert = w_base + (uint64_t)e * (uint64_t)dbpe;
        const uint8_t* xq_slot  = xq_base + (uint64_t)s * (uint64_t)xq_slot_stride;
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
        for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
            const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_Q4_K_BYTES;
            const uint8_t* y_blk = xq_slot  + (uint64_t)b * BLOCK_Q8_K_BYTES;
            acc += dot_block_quarter_q4k(w_blk, y_blk, sub);
        }
    }
    acc = warp_sum_f32_q4k(acc);

    if (lane == 0) {
        out[row] = acc;
    }
}

// ===========================================================================
// BATCHED-OVER-TOKENS (BxN) variants for prefill: process B prompt tokens in a
// single launch (grid.z = B). Each token carries its own top-`n_used` routing
// selection + activation, so per (token t, slot s) the expert weight is read
// independently — identical math to looping the B=1 kernels, but one dispatch.
// This kills the per-token dispatch/latency floor that made sequential prefill
// ~13 tok/s. (Weight-reuse-by-expert register tiling is a further lever.)
// ===========================================================================

// q4_k_pair_matvec_fused_swiglu_batch_bxn — B-token gate×up×swiglu×ew.
//   grid:  (n_rows / 8, n_used, B)   block: (256, 1, 1)
//   selected : [B, n_used]   expert_w : [B, n_used]
//   xq       : [B, n_blocks_in * 292]  (token-major)
//   mid      : [B, n_used, n_rows]      (token-major)
extern "C" __global__ void q4_k_pair_matvec_fused_swiglu_batch_bxn(
        float          * __restrict__ mid,
        const uint8_t  * __restrict__ gate_w_base,
        const uint8_t  * __restrict__ up_w_base,
        const uint8_t  * __restrict__ xq,            // [B, n_blocks_in*292]
        const float    * __restrict__ expert_w,      // [B, n_used]
        const int      * __restrict__ selected,      // [B, n_used]
        unsigned int     gate_bpe,
        unsigned int     up_bpe,
        float            clamp,
        unsigned int     n_rows,
        unsigned int     n_blocks_in,
        unsigned int     n_used) {
    const unsigned int slot         = blockIdx.y;
    const unsigned int t            = blockIdx.z;    // token in the chunk
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    const int e = selected[(uint64_t)t * n_used + slot];
    const uint8_t* xq_t = xq + (uint64_t)t * n_blocks_in * BLOCK_Q8_K_BYTES;
    const uint8_t* gate_row_base = gate_w_base
                                 + (uint64_t)e * (uint64_t)gate_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
    const uint8_t* up_row_base   = up_w_base
                                 + (uint64_t)e * (uint64_t)up_bpe
                                 + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;

    float g_sum = 0.0f;
    float u_sum = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
        const uint8_t* w_g = gate_row_base + (uint64_t)b * BLOCK_Q4_K_BYTES;
        const uint8_t* w_u = up_row_base   + (uint64_t)b * BLOCK_Q4_K_BYTES;
        const uint8_t* y   = xq_t          + (uint64_t)b * BLOCK_Q8_K_BYTES;
        g_sum += dot_block_quarter_q4k(w_g, y, sub);
        u_sum += dot_block_quarter_q4k(w_u, y, sub);
    }
    g_sum = warp_sum_f32_q4k(g_sum);
    u_sum = warp_sum_f32_q4k(u_sum);

    if (lane == 0) {
        const float ew = expert_w[(uint64_t)t * n_used + slot];
        if (clamp > 1.0e-6f) {
            if (g_sum > clamp)  g_sum = clamp;
            if (u_sum > clamp)  u_sum = clamp;
            if (u_sum < -clamp) u_sum = -clamp;
        }
        const float sig = 1.0f / (1.0f + expf(-g_sum));
        mid[((uint64_t)t * n_used + slot) * (uint64_t)n_rows + (uint64_t)row] =
            g_sum * sig * u_sum * ew;
    }
}

// q4_k_matvec_par_batched_bxn — B-token down projection (sum over n_used).
//   grid:  (n_rows / 8, 1, B)   block: (256, 1, 1)
//   selected : [B, n_used]
//   xq_base  : [B, n_used * xq_slot_stride]  (token-major, slot-major within)
//   out      : [B, n_rows]                    (token-major)
extern "C" __global__ void q4_k_matvec_par_batched_bxn(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_base,
        const uint8_t  * __restrict__ xq_base,
        const int      * __restrict__ selected,
        unsigned int     dbpe,
        unsigned int     xq_slot_stride,
        unsigned int     n_used,
        unsigned int     n_rows,
        unsigned int     n_blocks_in) {
    const unsigned int t            = blockIdx.z;
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    const int*     sel_t = selected + (uint64_t)t * n_used;
    const uint8_t* xq_t  = xq_base  + (uint64_t)t * n_used * (uint64_t)xq_slot_stride;

    float acc = 0.0f;
    for (unsigned int s = 0; s < n_used; s++) {
        const int e = sel_t[s];
        const uint8_t* w_expert = w_base + (uint64_t)e * (uint64_t)dbpe;
        const uint8_t* xq_slot  = xq_t   + (uint64_t)s * (uint64_t)xq_slot_stride;
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q4_K_BYTES;
        for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
            const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_Q4_K_BYTES;
            const uint8_t* y_blk = xq_slot  + (uint64_t)b * BLOCK_Q8_K_BYTES;
            acc += dot_block_quarter_q4k(w_blk, y_blk, sub);
        }
    }
    acc = warp_sum_f32_q4k(acc);
    if (lane == 0) {
        out[(uint64_t)t * n_rows + row] = acc;
    }
}
