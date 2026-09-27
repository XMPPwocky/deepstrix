// mxfp4_moe_wmma — int8-WMMA arm of the MXFP4 routed-MoE kernels for LARGE batches
// (2026-09-26 sweep, B_moe_prefill/wmma_i8_gateup + wmma_i8_down; the candidate
// cand_wmma.hip copied VERBATIM below this header -- incl. the +-clamp, the E8M0
// scale handling, the -12*bsum offset trick and the timing-ablation #ifdefs; do not
// "clean it up" without re-running the review's cmp).
//
// Measured (reviewer, full 384-expert layer, cold weights, gfx1151): gate+up 1.56x at
// B=1024, 1.18x at B=512, ~1.0x at B<=128; down 2.7x at B=1024, 1.8x at B=512, 1.3x at
// B=128. NOT BIT-EXACT: identical int32 dot products, f32 re-association only
// (rel_rmse ~5e-7 gate+up / ~1.7e-7 down vs kwide/kwide2) -> covered by the combined
// golden (KLD) gate. Selected by het::dispatch (moe_gate_up_chunked_rows /
// moe_down_mxfp4) for rows >= 256 (V41_MOE_WMMA_GATEUP) / rows >= 128
// (V41_MOE_WMMA_DOWN); only the gfx1151 code object is ever loaded (the RDNA3 WMMA
// intrinsic; the gfx12 build traps). Requires Q8_K activations WITH bsums (every
// production q8_k_quantize writes them, incl. the xq rows shipped to box 2).
// Includes mxfp4_pair_matvec.hip for its helpers, so this code object also carries
// copies of the pair kernels (unused from here).
//
// ---- candidate header ----
// B_moe_prefill candidate: int8 WMMA arm for the MXFP4 prefill MoE kernels (gfx1151).
//
//   mxfp4_pair_matvec_fused_swiglu_wmma  — gate+up, same args as ..._kwide,
//                                          grid (ceil(n_rows/128), n_work_items) x 256
//   mxfp4_matvec_par_by_expert_wmma      — down, same args as ..._kwide2,
//                                          grid (ceil(n_rows/128), n_work_items) x 256
//
// Design (see NOTES.md, idea B_moe_prefill/wmma_i8):
//   * A operand = Q8_K activations (int8, signed): 16 members x 16 k per WMMA, staged in
//     LDS once per work-group and super-block (32 members x 256 B), shared by 8 waves.
//   * B operand = MXFP4 weights (uint8): 16 weight rows x 16 k. Lane l16 owns row
//     m0 + l16 and unpacks its OWN 32 nibbles per block with v_perm (no LDS LUT):
//     w' = 12 + kvalue  (unsigned, 0..24).  dot(w', a) = dot(w, a) + 12 * sum(a), so the
//     accumulator is initialised with -12 * sum_32(a) per (member, block) — taken from the
//     Q8_K bsums the production quantizer already writes (bsums[2b] + bsums[2b+1]) —
//     and the int32 result is the exact integer dot product of the production kernel.
//   * Per 32-block: 2 x v_wmma_i32_16x16x16_iu8 (lo nibbles k 0..15, hi nibbles k 16..31),
//     then acc[i] += (2^(e-128) * yd[member i]) * (float)C[i] (the block's E8M0 scale is per
//     weight row = per lane, one scalar; yd is per member = per element, 8 per lane).
//   * C-fragment layout (RDNA3 wave32, as the IQ2_S WMMA arm): lane l, element i holds
//     C[m = 2i + (l>>4)][n = l&15]  -> member 2i+hi of the tile, weight row m0 + l16.
//   * Members per work item <= 32 (chunk) -> 1 or 2 M-tiles (template). Rows past
//     n_members are zero-filled in LDS and skipped in the epilogue.
//   Numerics: identical integer sums; the f32 accumulation order differs from kwide
//   (per-block (scale*yd) x exact int32 32-element sum, accumulated over all blocks in one
//   f32 chain; kwide scales 16-element lane partials then a 32-lane tree). Not bit-exact ->
//   needs the fidelity gate.
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
template <int NT, int MATS, int DOWN>
__device__ __forceinline__ void wm_mainloop(
        wm_v8f (&acc)[2][2], WmSmem& sm,
        const uint8_t* __restrict__ w_row0, const uint8_t* __restrict__ w_row1,
        const uint8_t* __restrict__ xq, size_t row_stride_b, size_t row_stride_slot,
        unsigned int n_members, unsigned int n_blocks, bool row_valid,
        unsigned int tid, unsigned int l16, unsigned int hi) {
    for (unsigned int sb = 0; sb < n_blocks; sb++) {
        // --- this row's 136-byte super-block per matrix, issued before the staging barrier
        uint4 wq[2][8];
        uint2 wsc[2];
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            #pragma unroll
            for (int b = 0; b < 8; b++) wq[m][b] = make_uint4(0u, 0u, 0u, 0u);
            wsc[m] = make_uint2(0u, 0u);
        }
#ifdef WM_ABL_NOLOAD
        // ABLATION (timing only): weights synthesised from the address, no weight VMEM.
        if (row_valid) {
            #pragma unroll
            for (int m = 0; m < MATS; m++) {
                const uint32_t h = (uint32_t)((uint64_t)(m ? w_row1 : w_row0) >> 4) * 2654435761u + sb * 0x9E3779B9u;
                #pragma unroll
                for (int b = 0; b < 8; b++) wq[m][b] = make_uint4(h ^ b, h * 3u ^ b, h * 5u ^ b, h * 7u ^ b);
                wsc[m] = make_uint2(0x7C7B7A79u ^ h, 0x7C7B7A79u ^ (h >> 3));
            }
        }
#elif defined(WM_ABL_SPLITLOAD)
        // ABLATION (timing only, with WM_ABL_NOCOMPUTE): lanes 0-15 load blocks 0-3, lanes 16-31
        // blocks 4-7 of the same row -> no duplicate addresses across the two wave halves.
        if (row_valid) {
            const uint4* g4 = (const uint4*)(w_row0 + (size_t)sb * SUPER_MXFP4_BYTES) + hi * 4u;
            #pragma unroll
            for (int b = 0; b < 4; b++) wq[0][b] = g4[b];
            __builtin_memcpy(&wsc[0].x, w_row0 + (size_t)sb * SUPER_MXFP4_BYTES + 128u + hi * 4u, 4);
            if (MATS == 2) {
                const uint4* u4 = (const uint4*)(w_row1 + (size_t)sb * SUPER_MXFP4_BYTES) + hi * 4u;
                #pragma unroll
                for (int b = 0; b < 4; b++) wq[1][b] = u4[b];
                __builtin_memcpy(&wsc[1].x, w_row1 + (size_t)sb * SUPER_MXFP4_BYTES + 128u + hi * 4u, 4);
            }
        }
#else
        if (row_valid) {
            const uint4* g4 = (const uint4*)(w_row0 + (size_t)sb * SUPER_MXFP4_BYTES);
            #pragma unroll
            for (int b = 0; b < 8; b++) wq[0][b] = WM_LOAD16(g4 + b);
            __builtin_memcpy(&wsc[0], w_row0 + (size_t)sb * SUPER_MXFP4_BYTES + 128u, 8);
            if (MATS == 2) {
                const uint4* u4 = (const uint4*)(w_row1 + (size_t)sb * SUPER_MXFP4_BYTES);
                #pragma unroll
                for (int b = 0; b < 8; b++) wq[1][b] = WM_LOAD16(u4 + b);
                __builtin_memcpy(&wsc[1], w_row1 + (size_t)sb * SUPER_MXFP4_BYTES + 128u, 8);
            }
        }
#endif
        // --- stage activations (all 256 threads), then barrier
        wm_stage<DOWN>(sm, xq, row_stride_b, row_stride_slot, sb, n_members, tid);
        __syncthreads();

        // yd per member of this super-block (per tile: 8 members per lane)
        wm_v8f ydv[2];
        #pragma unroll
        for (int t = 0; t < NT; t++) __builtin_memcpy(&ydv[t], &sm.yd[(t * 2u + hi) * 8u], 32);

#ifdef WM_ABL_NOCOMPUTE
        // ABLATION (timing only): consume the loaded words with 2 xors per block, no unpack/WMMA.
        #pragma unroll
        for (unsigned int b = 0; b < 8u; b++) {
            #pragma unroll
            for (int m = 0; m < MATS; m++)
                acc[m][0][b & 7u] += (float)(int)(wq[m][b].x ^ wq[m][b].y ^ wq[m][b].z ^ wq[m][b].w ^ wsc[m].x);
        }
        if (false)
#endif
        #pragma unroll
        for (unsigned int b = 0; b < 8u; b++) {
            wm_v4i bw_lo[2], bw_hi[2];
            float sc[2];
            #pragma unroll
            for (int m = 0; m < MATS; m++) {
                wm_unpack_block(wq[m][b], bw_lo[m], bw_hi[m]);
#ifdef WM_PROBE_UPPER
                // PROBE: does the hardware read the B fragment from lanes 16..31 at all?
                if (hi) { bw_lo[m] = ~bw_lo[m]; bw_hi[m] = bw_hi[m] ^ 0x5A5A5A5A; }
#endif
                const uint32_t e8 = ((b < 4u) ? wsc[m].x : wsc[m].y) >> (8u * (b & 3u));
                sc[m] = e8m0_half_to_f32_pair(e8 & 0xffu);
            }
            #pragma unroll
            for (int t = 0; t < NT; t++) {
                const uint8_t* xa = sm.x + (t * 16u + l16) * WM_X_STRIDE + b * 32u;
                wm_v4i a_lo, a_hi;
                __builtin_memcpy(&a_lo, xa, 16);
                __builtin_memcpy(&a_hi, xa + 16u, 16);
                wm_v8i c0;
                __builtin_memcpy(&c0, &sm.c0[((b * 2u + t) * 2u + hi) * 8u], 32);
                #pragma unroll
                for (int m = 0; m < MATS; m++) {
                    wm_v8i c = wm_wmma(a_lo, bw_lo[m], c0);
                    c = wm_wmma(a_hi, bw_hi[m], c);
                    #pragma unroll
                    for (int i = 0; i < 8; i++) acc[m][t][i] += (sc[m] * ydv[t][i]) * (float)c[i];
                }
            }
        }
        __syncthreads();   // LDS reuse by the next super-block
    }
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
    const uint8_t* w_row0 = w0_base + (uint64_t)e * (uint64_t)w0_bpe
                          + (uint64_t)(row_valid ? row_a : 0u) * (uint64_t)n_blocks * SUPER_MXFP4_BYTES;
    const uint8_t* w_row1 = (MATS == 2)
        ? (w1_base + (uint64_t)e * (uint64_t)w1_bpe + (uint64_t)(row_valid ? row_a : 0u) * (uint64_t)n_blocks * SUPER_MXFP4_BYTES)
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
