// q6_k_accumulate_matvec_par — batched routed-MoE **down** projection for
// Q6_K-quantized expert weights × Q8_K activations. Structural mirror of
// q2_k_accumulate_matvec_par.hip / q4_k_matvec_par_batched.hip, with the
// quant decode swapped for Q6_K's 6-bit scheme.
//
// For the Laguna port the down-projection experts (`ffn_down_exps`,
// `ffn_down_shexp`) are Q6_K on most layers (Q4_K on a few — those reuse the
// existing q4_k_matvec_par_batched kernel; dispatch on the tensor's dtype).
//
// Q6_K block layout (210 bytes per 256 elements) — identical to ggml
// `block_q6_K`:
//   offset   0: uint8 ql[128]    (lower 4 bits of the 6-bit quants)
//   offset 128: uint8 qh[64]     (upper 2 bits)
//   offset 192: int8  scales[16] (per-16-element sub-block scales, SIGNED)
//   offset 208: ggml_half d      (super-block scale, f16) — 2 bytes
//
// Dequant (matches ggml dequantize_row_q6_K): each super-block is two
// 128-element halves. For half h (ql+=64h, qh+=32h, sc+=8h) and l in [0,32):
//   a[l +  0] = ((ql[l   ] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
//   a[l + 32] = ((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
//   a[l + 64] = ((ql[l   ] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32
//   a[l + 96] = ((ql[l+32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32
// Each 16-element run shares one signed scale sc[is]; the weight offset -32
// is baked directly into the signed int8 `a` value (Q6_K has no min/bias
// term, unlike Q4_K/Q2_K), so the Q8_K dot is a plain signed×signed dp4a.
//
// Lane mapping (mirrors the q4_k/q2_k par layout): 8 rows per workgroup, one
// warp (32 lanes) per row, 4 lanes per super-block. Each lane owns a
// "quarter" = 64 contiguous elements = 4 of the 16 sub-blocks. With
// n_blocks_in = N_FF_EXP/256 = 4 for Laguna, lanes stride blocks with step 8
// (block_lanes 0..3 cover the 4 super-blocks; 4..7 idle — same as q4_k for
// small n_blocks_in). Warp-tree reduce collapses the 32 partials.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K 256
#define BLOCK_Q6_K_BYTES 210
#define BLOCK_Q8_K_BYTES 292
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ float f16_to_f32_q6k(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = v;
    return __half2float(h);
}

__device__ __forceinline__ int32_t sudot4_q6k(int32_t a, int32_t b, int32_t c) {
    // both operands signed int8×4
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t load_dword_i8_q6k(const int8_t* p) {
    int32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

// Quarter-block dot: one of 4 lanes per super-block. `sub` in [0,4) selects
// which 64-element quarter (half = sub>>1, upper = sub&1 -> low/high nibble).
// Returns this lane's contribution to the super-block result:
//   d_all * Σ_g scale[g] * Σ_{r in g} a[r] * q8[r]
__device__ __forceinline__ float dot_block_quarter_q6k(
        const uint8_t* x_blk, const uint8_t* y_blk, unsigned int sub) {
    const uint8_t* ql        = x_blk + 0;
    const uint8_t* qh        = x_blk + 128;
    const int8_t*  sc        = (const int8_t*)(x_blk + 192);
    const uint16_t d_bits    = *(const uint16_t*)(x_blk + 208);

    const float    yd        = *(const float*)(y_blk + 0);
    const int8_t*  q8_base   = (const int8_t*)(y_blk + 4);
    const float    d_all     = yd * f16_to_f32_q6k(d_bits);

    const unsigned int half  = sub >> 1u;    // 0 or 1
    const unsigned int upper = sub & 1u;     // 0 = low nibble, 1 = high nibble
    const uint8_t* qlh = ql + (size_t)half * 64u;   // 64 bytes for this half
    const uint8_t* qhh = qh + (size_t)half * 32u;   // 32 bytes for this half
    const int8_t*  q8  = q8_base + (size_t)half * 128u + (size_t)upper * 64u;
    const int8_t*  sch = sc + (size_t)half * 8u + (size_t)upper * 4u; // 4 scales

    int32_t isum = 0;
    #pragma unroll
    for (int g = 0; g < 4; g++) {            // 4 sub-blocks of 16 elements
        int32_t dot = 0;
        #pragma unroll
        for (int q = 0; q < 16; q += 4) {    // 4 signed dp4a per sub-block
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
            dot = sudot4_q6k((int32_t)apack, load_dword_i8_q6k(q8 + r0), dot);
        }
        isum += (int32_t)sch[g] * dot;
    }
    return d_all * (float)isum;
}

__device__ __forceinline__ float warp_sum_f32_q6k(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

// Single-expert accumulate (mirror of q2_k_accumulate_matvec_par).
// zero_init=1 writes, =0 adds into out[row].
extern "C" __global__ void q6_k_accumulate_matvec_par(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w_expert,
        const uint8_t  * __restrict__ xq,
        unsigned int     n_rows,
        unsigned int     n_blocks_in,
        unsigned int     zero_init) {
    const unsigned int row_in_block = threadIdx.x / 32u;
    const unsigned int lane         = threadIdx.x & 31u;
    const unsigned int row          = blockIdx.x * 8u + row_in_block;
    if (row >= n_rows) return;

    const unsigned int block_lane = lane >> 2u;    // 0..7
    const unsigned int sub        = lane & 3u;     // 0..3

    const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q6_K_BYTES;

    float acc = 0.0f;
    for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
        const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_Q6_K_BYTES;
        const uint8_t* y_blk = xq       + (uint64_t)b * BLOCK_Q8_K_BYTES;
        acc += dot_block_quarter_q6k(w_blk, y_blk, sub);
    }
    acc = warp_sum_f32_q6k(acc);

    if (lane == 0) {
        if (zero_init) out[row] = acc;
        else           out[row] += acc;
    }
}

// Batched down-accumulate: single launch sums over all `n_used` selected
// experts internally per workgroup. Writes the summed result directly to
// out[n_rows]. Per-slot activations come from xq_base + s*xq_slot_stride;
// per-expert weights from w_base + selected[s]*dbpe. Direct port of
// q4_k_matvec_par_batched with the Q6_K quarter-block dot.
//
// Launch: grid (n_rows/8, 1, 1), block (256, 1, 1).
extern "C" __global__ void q6_k_matvec_par_batched(
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

    const unsigned int block_lane = lane >> 2u;
    const unsigned int sub        = lane & 3u;

    float acc = 0.0f;
    for (unsigned int s = 0; s < n_used; s++) {
        const int e = selected[s];
        if (e < 0) continue; // het-split sentinel (slot owned by the other device)
        const uint8_t* w_expert = w_base + (uint64_t)e * (uint64_t)dbpe;
        const uint8_t* xq_slot  = xq_base + (uint64_t)s * (uint64_t)xq_slot_stride;
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q6_K_BYTES;
        for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
            const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_Q6_K_BYTES;
            const uint8_t* y_blk = xq_slot  + (uint64_t)b * BLOCK_Q8_K_BYTES;
            acc += dot_block_quarter_q6k(w_blk, y_blk, sub);
        }
    }
    acc = warp_sum_f32_q6k(acc);

    if (lane == 0) {
        out[row] = acc;
    }
}

// q6_k_matvec_par_batched_bxn — B-token down projection (sum over n_used) for
// prefill. grid.z = B; each token has its own routing + activation slots.
//   grid:  (n_rows / 8, 1, B)   block: (256, 1, 1)
//   selected : [B, n_used]
//   xq_base  : [B, n_used * xq_slot_stride]  (token-major, slot-major within)
//   out      : [B, n_rows]                    (token-major)
extern "C" __global__ void q6_k_matvec_par_batched_bxn(
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
        const uint8_t* row_base = w_expert + (uint64_t)row * n_blocks_in * BLOCK_Q6_K_BYTES;
        for (unsigned int b = block_lane; b < n_blocks_in; b += 8u) {
            const uint8_t* w_blk = row_base + (uint64_t)b * BLOCK_Q6_K_BYTES;
            const uint8_t* y_blk = xq_slot  + (uint64_t)b * BLOCK_Q8_K_BYTES;
            acc += dot_block_quarter_q6k(w_blk, y_blk, sub);
        }
    }
    acc = warp_sum_f32_q6k(acc);
    if (lane == 0) {
        out[(uint64_t)t * n_rows + row] = acc;
    }
}
