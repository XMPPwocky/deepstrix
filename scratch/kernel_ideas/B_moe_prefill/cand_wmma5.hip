// B_moe_prefill candidate wmma_i8_mseq: cand_wmma.hip with MATRIX-SEQUENTIAL K loops.
//
// Same numerics / args / geometry as cand_wmma.hip. The gate+up kernel runs the whole K loop
// for the gate matrix, then again for the up matrix (activations are re-staged, twice the
// LDS staging + barriers, both cheap), so that at any moment each row has ONE weight stream
// in flight instead of two (the loads-only ablation of cand_wmma reached 180 GB/s with two
// streams per row vs 207 GB/s for the one-stream down kernel), and only 34 VGPRs of raw
// weights are live. With -DWM5_PIPE the freed registers hold a second buffer and the next
// super-block's loads are issued before the current staging.
#include "mxfp4_pair_matvec.hip"

#define WM_ROWS_PER_WG 128u
#define WM_MAX_CHUNK   32u
#define WM_X_STRIDE    272u      // bytes per staged member row (256 + 16 pad: conflict-free b128)

#ifdef WM_NT_LOAD   // knob: streaming (non-temporal) weight loads
__device__ __forceinline__ uint4 wm_ld16(const uint4* p) {
    uint4 r;
    r.x = __builtin_nontemporal_load(&p->x); r.y = __builtin_nontemporal_load(&p->y);
    r.z = __builtin_nontemporal_load(&p->z); r.w = __builtin_nontemporal_load(&p->w);
    return r;
}
#define WM_LOAD16(p) wm_ld16(p)
#else
#define WM_LOAD16(p) (*(p))
#endif
typedef int   wm_v4i __attribute__((ext_vector_type(4)));
typedef int   wm_v8i __attribute__((ext_vector_type(8)));
typedef float wm_v8f __attribute__((ext_vector_type(8)));

// 4 low nibbles of x (bytes 0..3) -> 4 unsigned bytes 12 + kvalue.  For the high
// nibbles call with x >> 4.
__device__ __forceinline__ uint32_t wm_unpack4(uint32_t x) {
    const uint32_t s   = x & 0x07070707u;
    const uint32_t mag = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, s);  // {0,1,2,3,4,6,8,12}[s]
    const uint32_t g   = x & 0x08080808u;
    const uint32_t sm  = (g << 5) - (g >> 3);          // 0xFF in every negative byte
    const uint32_t mn  = mag & sm;
    const uint32_t mp  = mag ^ mn;
    return 0x0C0C0C0Cu + mp - mn;                        // bytes in [0, 24], no carries
}
__device__ __forceinline__ void wm_unpack_block(const uint4 q, wm_v4i& lo, wm_v4i& hi) {
    lo[0] = (int)wm_unpack4(q.x); lo[1] = (int)wm_unpack4(q.y);
    lo[2] = (int)wm_unpack4(q.z); lo[3] = (int)wm_unpack4(q.w);
    hi[0] = (int)wm_unpack4(q.x >> 4); hi[1] = (int)wm_unpack4(q.y >> 4);
    hi[2] = (int)wm_unpack4(q.z >> 4); hi[3] = (int)wm_unpack4(q.w >> 4);
}
__device__ __forceinline__ wm_v8i wm_wmma(const wm_v4i a, const wm_v4i b, const wm_v8i c) {
#if defined(__gfx1151__) || defined(__gfx1150__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, false, b, c, false);
#else
    wm_v8i r = c; (void)a; (void)b; __builtin_trap(); return r;
#endif
}

struct WmSmem {
    __attribute__((aligned(16))) uint8_t x[WM_MAX_CHUNK * WM_X_STRIDE];  // [member][256 q8 + pad]
    __attribute__((aligned(16))) int     c0[8 * WM_MAX_CHUNK];           // [blk][tile][hi][i] = -12*sum32
    __attribute__((aligned(16))) float   yd[WM_MAX_CHUNK];               // [tile][hi][i]
    int   member_packed[WM_MAX_CHUNK];
    float ew[WM_MAX_CHUNK];
};

// Stage one super-block of every member (q8 bytes, -12*bsum per block, yd) into LDS.
// xq_row(b, slot) = base of that member's Q8_K row. 256 threads: member = tid>>3,
// block = tid&7 (32 bytes each).
template <int DOWN>
__device__ __forceinline__ void wm_stage(WmSmem& sm, const uint8_t* __restrict__ xq,
                                         size_t row_stride_b, size_t row_stride_slot,
                                         unsigned int sb, unsigned int n_members, unsigned int tid) {
    const unsigned int mi = tid >> 3, bb = tid & 7u;
    const unsigned int tile = mi >> 4, w = mi & 15u, hi = w & 1u, i = w >> 1;
    const unsigned int cidx = ((bb * 2u + tile) * 2u + hi) * 8u + i;
    const unsigned int yidx = (tile * 2u + hi) * 8u + i;
    uint4 q0 = make_uint4(0u, 0u, 0u, 0u), q1 = q0;
    int c0 = 0;
    float yd = 0.0f;
    if (mi < n_members) {
        const int packed = sm.member_packed[mi];
        const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
        const unsigned int slot = (unsigned int)(packed & 0xffffu);
        const uint8_t* src = xq + (size_t)b * row_stride_b + (DOWN ? (size_t)slot * row_stride_slot : 0)
                           + (size_t)sb * BLOCK_Q8_K_BYTES;
        const uint4* q8 = (const uint4*)(src + 4u + bb * 32u);
        q0 = q8[0]; q1 = q8[1];
        uint32_t bs;
        __builtin_memcpy(&bs, src + 260u + bb * 4u, 4);
        const int s32 = (int)(int16_t)(bs & 0xffffu) + (int)(int16_t)(bs >> 16);
        c0 = -12 * s32;
        if (bb == 0u) __builtin_memcpy(&yd, src, 4);
    }
    uint4* dst = (uint4*)(sm.x + mi * WM_X_STRIDE + bb * 32u);
    dst[0] = q0; dst[1] = q1;
    sm.c0[cidx] = c0;
    if (bb == 0u) sm.yd[yidx] = yd;
}

// Main loop over super-blocks for one wave (16 rows) and NT member tiles.
// MATS = 2: gate+up (acc[0] = gate, acc[1] = up); MATS = 1: down (acc[0]).
struct WmWBuf { uint4 q[8]; uint2 sc; };
__device__ __forceinline__ void wm_issue(WmWBuf& b, const uint8_t* __restrict__ w_row, unsigned int sb) {
    const uint4* p4 = (const uint4*)(w_row + (size_t)sb * SUPER_MXFP4_BYTES);
    #pragma unroll
    for (int i = 0; i < 8; i++) b.q[i] = p4[i];
    __builtin_memcpy(&b.sc, w_row + (size_t)sb * SUPER_MXFP4_BYTES + 128u, 8);
}
template <int NT>
__device__ __forceinline__ void wm_load_yd(wm_v8f (&ydv)[2], const WmSmem& sm, unsigned int hi) {
    #pragma unroll
    for (int t = 0; t < NT; t++) __builtin_memcpy(&ydv[t], &sm.yd[(t * 2u + hi) * 8u], 32);
}
// 8 blocks of one matrix super-block -> acc (NT tiles).
template <int NT>
__device__ __forceinline__ void wm_compute(wm_v8f (&acc)[2], const WmWBuf& wb, const wm_v8f (&ydv)[2],
                                           const WmSmem& sm, unsigned int l16, unsigned int hi) {
    #pragma unroll
    for (unsigned int b = 0; b < 8u; b++) {
        wm_v4i bw_lo, bw_hi;
        wm_unpack_block(wb.q[b], bw_lo, bw_hi);
        const uint32_t e8 = ((b < 4u) ? wb.sc.x : wb.sc.y) >> (8u * (b & 3u));
        const float sc = e8m0_half_to_f32_pair(e8 & 0xffu);
        #pragma unroll
        for (int t = 0; t < NT; t++) {
            const uint8_t* xa = sm.x + (t * 16u + l16) * WM_X_STRIDE + b * 32u;
            wm_v4i a_lo, a_hi;
            __builtin_memcpy(&a_lo, xa, 16);
            __builtin_memcpy(&a_hi, xa + 16u, 16);
            wm_v8i c;
            __builtin_memcpy(&c, &sm.c0[((b * 2u + t) * 2u + hi) * 8u], 32);
            c = wm_wmma(a_lo, bw_lo, c);
            c = wm_wmma(a_hi, bw_hi, c);
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[t][i] += (sc * ydv[t][i]) * (float)c[i];
        }
    }
}

// One matrix, all super-blocks. Rows are clamped by the caller (loads always in range).
template <int NT, int DOWN>
__device__ __forceinline__ void wm_kloop(wm_v8f (&acc)[2], WmSmem& sm, const uint8_t* __restrict__ w_row,
        const uint8_t* __restrict__ xq, size_t row_stride_b, size_t row_stride_slot,
        unsigned int n_members, unsigned int n_blocks, unsigned int tid, unsigned int l16, unsigned int hi) {
#ifdef WM5_PIPE
    WmWBuf b0, b1;
    wm_issue(b0, w_row, 0u);
    for (unsigned int sb = 0; sb < n_blocks; sb += 2u) {
        if (sb + 1u < n_blocks) wm_issue(b1, w_row, sb + 1u);
        wm_stage<DOWN>(sm, xq, row_stride_b, row_stride_slot, sb, n_members, tid);
        __syncthreads();
        {
            wm_v8f ydv[2];
            wm_load_yd<NT>(ydv, sm, hi);
            wm_compute<NT>(acc, b0, ydv, sm, l16, hi);
        }
        __syncthreads();
        if (sb + 1u < n_blocks) {
            if (sb + 2u < n_blocks) wm_issue(b0, w_row, sb + 2u);
            wm_stage<DOWN>(sm, xq, row_stride_b, row_stride_slot, sb + 1u, n_members, tid);
            __syncthreads();
            wm_v8f ydv[2];
            wm_load_yd<NT>(ydv, sm, hi);
            wm_compute<NT>(acc, b1, ydv, sm, l16, hi);
            __syncthreads();
        }
    }
#else
    for (unsigned int sb = 0; sb < n_blocks; sb++) {
        WmWBuf b0;
        wm_issue(b0, w_row, sb);
        wm_stage<DOWN>(sm, xq, row_stride_b, row_stride_slot, sb, n_members, tid);
        __syncthreads();
        wm_v8f ydv[2];
        wm_load_yd<NT>(ydv, sm, hi);
        wm_compute<NT>(acc, b0, ydv, sm, l16, hi);
        __syncthreads();
    }
#endif
}

template <int NT, int MATS, int DOWN>
__device__ __forceinline__ void wm_mainloop(
        wm_v8f (&acc)[2][2], WmSmem& sm,
        const uint8_t* __restrict__ w_row0, const uint8_t* __restrict__ w_row1,
        const uint8_t* __restrict__ xq, size_t row_stride_b, size_t row_stride_slot,
        unsigned int n_members, unsigned int n_blocks, bool row_valid,
        unsigned int tid, unsigned int l16, unsigned int hi) {
    (void)row_valid;
    wm_kloop<NT, DOWN>(acc[0], sm, w_row0, xq, row_stride_b, row_stride_slot, n_members, n_blocks, tid, l16, hi);
    if (MATS == 2)
        wm_kloop<NT, DOWN>(acc[1], sm, w_row1, xq, row_stride_b, row_stride_slot, n_members, n_blocks, tid, l16, hi);
}

template <int MATS, int DOWN>
__device__ __forceinline__ void wm_body(
        float          * __restrict__ out,
        const uint8_t  * __restrict__ w0_base, const uint8_t * __restrict__ w1_base,
        const uint8_t  * __restrict__ xq, const float * __restrict__ expert_w,
        const int      * __restrict__ group_count, const int * __restrict__ expert_members,
        const int      * __restrict__ work_items,
        unsigned int w0_bpe, unsigned int w1_bpe, unsigned int xq_slot_stride,
        unsigned int n_used, unsigned int max_per_expert, unsigned int chunk_size,
        float clamp, unsigned int n_rows, unsigned int n_blocks,
        const int * __restrict__ n_work_items_dev) {
    if (n_work_items_dev != nullptr && (int)blockIdx.y >= *n_work_items_dev) return;
    const unsigned int w_idx = blockIdx.y;
    const int packed_item = work_items[w_idx];
    const unsigned int e            = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item       ) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + min(chunk_size, WM_MAX_CHUNK));
    if (member_end <= member_start) return;
    const unsigned int n_members = member_end - member_start;
    const unsigned int n_tiles = (n_members + 15u) >> 4;

    __shared__ WmSmem sm;
    const unsigned int tid = threadIdx.x;
    if (tid < WM_MAX_CHUNK) {
        int packed = 0;
        float ew = 0.0f;
        if (tid < n_members) {
            packed = expert_members[(size_t)e * (size_t)max_per_expert + (size_t)(member_start + tid)];
            if (!DOWN) {
                const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
                const unsigned int slot = (unsigned int)(packed & 0xffffu);
                ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
            }
        }
        sm.member_packed[tid] = packed;
        sm.ew[tid] = ew;
    }
    __syncthreads();

    const unsigned int wave = tid >> 5, lane = tid & 31u, l16 = lane & 15u, hi = lane >> 4;
    const unsigned int m0 = blockIdx.x * WM_ROWS_PER_WG + wave * 16u;
    const unsigned int row_a = m0 + l16;
    const bool row_valid = row_a < n_rows;
    const unsigned int row_ld = row_valid ? row_a : (n_rows - 1u);   // clamp: loads always in range
    const uint8_t* w_row0 = w0_base + (uint64_t)e * (uint64_t)w0_bpe
                          + (uint64_t)row_ld * (uint64_t)n_blocks * SUPER_MXFP4_BYTES;
    const uint8_t* w_row1 = (MATS == 2)
        ? (w1_base + (uint64_t)e * (uint64_t)w1_bpe + (uint64_t)row_ld * (uint64_t)n_blocks * SUPER_MXFP4_BYTES)
        : w_row0;
    const size_t row_stride_b    = DOWN ? (size_t)n_used * (size_t)xq_slot_stride : (size_t)n_blocks * BLOCK_Q8_K_BYTES;
    const size_t row_stride_slot = (size_t)xq_slot_stride;

    wm_v8f acc[2][2];
    #pragma unroll
    for (int m = 0; m < 2; m++)
        #pragma unroll
        for (int t = 0; t < 2; t++)
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[m][t][i] = 0.0f;

    if (n_tiles == 2u) {
        wm_mainloop<2, MATS, DOWN>(acc, sm, w_row0, w_row1, xq, row_stride_b, row_stride_slot,
                                   n_members, n_blocks, row_valid, tid, l16, hi);
    } else {
        wm_mainloop<1, MATS, DOWN>(acc, sm, w_row0, w_row1, xq, row_stride_b, row_stride_slot,
                                   n_members, n_blocks, row_valid, tid, l16, hi);
    }

    // --- Epilogue: lane (l16, hi) elem i = member t*16 + 2i + hi, row m0 + l16.
    if (!row_valid) return;
    #pragma unroll
    for (int t = 0; t < 2; t++) {
        if ((unsigned int)t < n_tiles) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const unsigned int mi = (unsigned int)t * 16u + 2u * (unsigned int)i + hi;
                if (mi < n_members) {
                    const int packed        = sm.member_packed[mi];
                    const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
                    const unsigned int slot = (unsigned int)(packed & 0xffffu);
                    const size_t out_off = ((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows + (size_t)row_a;
                    if (MATS == 2) {
                        float g = acc[0][t][i], u = acc[1][t][i];
                        if (clamp > 1.0e-6f) {
                            if (g > clamp)  g = clamp;
                            if (u > clamp)  u = clamp;
                            if (u < -clamp) u = -clamp;
                        }
                        const float sig = 1.0f / (1.0f + expf(-g));
                        out[out_off] = g * sig * u * sm.ew[mi];
                    } else {
                        out[out_off] = acc[0][t][i];
                    }
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void mxfp4_pair_matvec_fused_swiglu_wmma(
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
    wm_body<2, 0>(mid, gate_w_base, up_w_base, xq, expert_w, group_count, expert_members, work_items,
                  gate_bpe, up_bpe, 0u, n_used, max_per_expert, chunk_size, clamp, n_rows, n_blocks,
                  n_work_items_dev);
}

extern "C" __global__ __launch_bounds__(256) void mxfp4_matvec_par_by_expert_wmma(
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
        const int      * __restrict__ n_work_items_dev) {
    wm_body<1, 1>(partials, w_base, w_base, xq_base, nullptr, group_count, expert_members, work_items,
                  dbpe, dbpe, xq_slot_stride, n_used, max_per_expert, chunk_size, 0.0f, n_rows, n_blocks_in,
                  n_work_items_dev);
}
