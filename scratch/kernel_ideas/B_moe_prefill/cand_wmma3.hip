// B_moe_prefill candidate wmma_i8_split: gate+up with SPLIT-MATRIX waves + double-buffered
// weight loads; down as in cand_wmma.hip (non-pipelined, it is already at ~85% of roof).
//
// Same numerics as cand_wmma.hip (see its banner). Gate+up geometry change:
//   * WG = 256 threads = 8 waves covers 64 rows: waves 0-3 = GATE rows m0..m0+63 (16 each),
//     waves 4-7 = UP for the same rows. Launch grid (ceil(n_rows/64), n_work_items) x 256
//     (harness: KB_GU_RPW=64). One matrix per wave halves the live weight registers
//     (34 instead of 68), which buys a second buffer for the next super-block's loads
//     (issued before the current super-block's staging) at the SAME register footprint
//     as cand_wmma, i.e. pipelined loads with no spills.
//   * Epilogue: up waves park acc in LDS (reusing the staging buffer), barrier, gate waves
//     apply clamp + SwiGLU + expert weight and store. Activation staging / A-fragment reads
//     are shared by the 8 waves exactly as before.
#include "mxfp4_pair_matvec.hip"

#define WM_MAX_CHUNK   32u
#define WM_X_STRIDE    272u
#define WM3_ROWS_PER_WG 64u

typedef int   wm_v4i __attribute__((ext_vector_type(4)));
typedef int   wm_v8i __attribute__((ext_vector_type(8)));
typedef float wm_v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint32_t wm_unpack4(uint32_t x) {
    const uint32_t s   = x & 0x07070707u;
    const uint32_t mag = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, s);
    const uint32_t g   = x & 0x08080808u;
    const uint32_t sm  = (g << 5) - (g >> 3);
    const uint32_t mn  = mag & sm;
    const uint32_t mp  = mag ^ mn;
    return 0x0C0C0C0Cu + mp - mn;
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
    __attribute__((aligned(16))) uint8_t x[WM_MAX_CHUNK * WM_X_STRIDE];   // 8704 B; epilogue: float xchg[32][64]
    __attribute__((aligned(16))) int     c0[8 * WM_MAX_CHUNK];
    __attribute__((aligned(16))) float   yd[WM_MAX_CHUNK];
    int   member_packed[WM_MAX_CHUNK];
    float ew[WM_MAX_CHUNK];
};
struct WmWBuf { uint4 q[8]; uint2 sc; };

__device__ __forceinline__ void wm_issue(WmWBuf& b, const uint8_t* __restrict__ w_row, unsigned int sb) {
    const uint4* p4 = (const uint4*)(w_row + (size_t)sb * SUPER_MXFP4_BYTES);
    #pragma unroll
    for (int i = 0; i < 8; i++) b.q[i] = p4[i];
    __builtin_memcpy(&b.sc, w_row + (size_t)sb * SUPER_MXFP4_BYTES + 128u, 8);
}

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
template <int NT>
__device__ __forceinline__ void wm_load_yd(wm_v8f (&ydv)[2], const WmSmem& sm, unsigned int hi) {
    #pragma unroll
    for (int t = 0; t < NT; t++) __builtin_memcpy(&ydv[t], &sm.yd[(t * 2u + hi) * 8u], 32);
}

// One matrix per wave, two super-block buffers, loads one super-block ahead (issued before
// the staging of the current one). Used by the split gate+up kernel.
template <int NT, int DOWN>
__device__ __forceinline__ void wm_mainloop_pipe(
        wm_v8f (&acc)[2], WmSmem& sm, const uint8_t* __restrict__ w_row,
        const uint8_t* __restrict__ xq, size_t row_stride_b, size_t row_stride_slot,
        unsigned int n_members, unsigned int n_blocks,
        unsigned int tid, unsigned int l16, unsigned int hi) {
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
}

// Non-pipelined single-matrix loop (cand_wmma structure) for the down kernel.
template <int NT, int DOWN>
__device__ __forceinline__ void wm_mainloop_plain(
        wm_v8f (&acc)[2], WmSmem& sm, const uint8_t* __restrict__ w_row,
        const uint8_t* __restrict__ xq, size_t row_stride_b, size_t row_stride_slot,
        unsigned int n_members, unsigned int n_blocks,
        unsigned int tid, unsigned int l16, unsigned int hi) {
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
}

// Shared prologue: work item, members, LDS tables. Returns n_members (0 = nothing to do).
__device__ __forceinline__ unsigned int wm_prologue(WmSmem& sm, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        const float* __restrict__ expert_w, unsigned int n_used, unsigned int max_per_expert,
        unsigned int chunk_size, unsigned int& e, unsigned int tid) {
    const int packed_item = work_items[blockIdx.y];
    e = (unsigned int)(packed_item >> 16) & 0xffffu;
    const unsigned int member_start = (unsigned int)(packed_item) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned int member_end = min((unsigned int)group_n, member_start + min(chunk_size, WM_MAX_CHUNK));
    if (member_end <= member_start) return 0u;
    const unsigned int n_members = member_end - member_start;
    if (tid < WM_MAX_CHUNK) {
        int packed = 0;
        float ew = 0.0f;
        if (tid < n_members) {
            packed = expert_members[(size_t)e * (size_t)max_per_expert + (size_t)(member_start + tid)];
            if (expert_w != nullptr) {
                const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
                const unsigned int slot = (unsigned int)(packed & 0xffffu);
                ew = expert_w[(size_t)b * (size_t)n_used + (size_t)slot];
            }
        }
        sm.member_packed[tid] = packed;
        sm.ew[tid] = ew;
    }
    __syncthreads();
    return n_members;
}

extern "C" __global__ __launch_bounds__(256) void mxfp4_pair_matvec_fused_swiglu_wmma(
        float* __restrict__ mid, const uint8_t* __restrict__ gate_w_base, const uint8_t* __restrict__ up_w_base,
        const uint8_t* __restrict__ xq, const float* __restrict__ expert_w, const int* __restrict__ group_count,
        const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int gate_bpe, unsigned int up_bpe, unsigned int n_used, unsigned int max_per_expert,
        unsigned int chunk_size, float clamp, unsigned int n_rows, unsigned int n_blocks,
        const int* __restrict__ n_work_items_dev) {
    if (n_work_items_dev != nullptr && (int)blockIdx.y >= *n_work_items_dev) return;
    __shared__ WmSmem sm;
    const unsigned int tid = threadIdx.x;
    unsigned int e = 0;
    const unsigned int n_members = wm_prologue(sm, group_count, expert_members, work_items, expert_w,
                                               n_used, max_per_expert, chunk_size, e, tid);
    if (n_members == 0u) return;
    const unsigned int n_tiles = (n_members + 15u) >> 4;

    const unsigned int wave = tid >> 5, lane = tid & 31u, l16 = lane & 15u, hi = lane >> 4;
    const unsigned int mat  = wave >> 2;                 // 0 = gate, 1 = up
    const unsigned int rloc = (wave & 3u) * 16u + l16;   // row within the WG's 64
    const unsigned int row_a = blockIdx.x * WM3_ROWS_PER_WG + rloc;
    const bool row_valid = row_a < n_rows;
    const unsigned int row_ld = row_valid ? row_a : (n_rows - 1u);   // clamp: loads stay in range
    const uint8_t* w_row = (mat ? up_w_base : gate_w_base)
                         + (uint64_t)e * (uint64_t)(mat ? up_bpe : gate_bpe)
                         + (uint64_t)row_ld * (uint64_t)n_blocks * SUPER_MXFP4_BYTES;
    const size_t row_stride_b = (size_t)n_blocks * BLOCK_Q8_K_BYTES;

    wm_v8f acc[2];
    #pragma unroll
    for (int t = 0; t < 2; t++)
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[t][i] = 0.0f;

#ifdef WM3_NO_PIPE   // knob: split waves WITHOUT the load pipeline (isolates occupancy vs prefetch)
    if (n_tiles == 2u) wm_mainloop_plain<2, 0>(acc, sm, w_row, xq, row_stride_b, 0, n_members, n_blocks, tid, l16, hi);
    else               wm_mainloop_plain<1, 0>(acc, sm, w_row, xq, row_stride_b, 0, n_members, n_blocks, tid, l16, hi);
#else
    if (n_tiles == 2u) wm_mainloop_pipe<2, 0>(acc, sm, w_row, xq, row_stride_b, 0, n_members, n_blocks, tid, l16, hi);
    else               wm_mainloop_pipe<1, 0>(acc, sm, w_row, xq, row_stride_b, 0, n_members, n_blocks, tid, l16, hi);
#endif

    // --- Epilogue: up waves park acc in LDS as xchg[member][row64]; gate waves combine.
    float* xchg = (float*)sm.x;   // 32 x 64 floats = 8 KB <= 8704 B (last loop barrier protects it)
    if (mat == 1u) {
        #pragma unroll
        for (int t = 0; t < 2; t++) {
            if ((unsigned int)t < n_tiles) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const unsigned int mi = (unsigned int)t * 16u + 2u * (unsigned int)i + hi;
                    xchg[mi * 64u + rloc] = acc[t][i];
                }
            }
        }
    }
    __syncthreads();
    if (mat == 0u && row_valid) {
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
                        float g = acc[t][i], u = xchg[mi * 64u + rloc];
                        if (clamp > 1.0e-6f) {
                            if (g > clamp)  g = clamp;
                            if (u > clamp)  u = clamp;
                            if (u < -clamp) u = -clamp;
                        }
                        const float sig = 1.0f / (1.0f + expf(-g));
                        mid[((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows + (size_t)row_a] = g * sig * u * sm.ew[mi];
                    }
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void mxfp4_matvec_par_by_expert_wmma(
        float* __restrict__ partials, const uint8_t* __restrict__ w_base, const uint8_t* __restrict__ xq_base,
        const int* __restrict__ group_count, const int* __restrict__ expert_members, const int* __restrict__ work_items,
        unsigned int dbpe, unsigned int xq_slot_stride, unsigned int n_used, unsigned int max_per_expert,
        unsigned int chunk_size, unsigned int n_rows, unsigned int n_blocks_in,
        const int* __restrict__ n_work_items_dev) {
    if (n_work_items_dev != nullptr && (int)blockIdx.y >= *n_work_items_dev) return;
    __shared__ WmSmem sm;
    const unsigned int tid = threadIdx.x;
    unsigned int e = 0;
    const unsigned int n_members = wm_prologue(sm, group_count, expert_members, work_items, nullptr,
                                               n_used, max_per_expert, chunk_size, e, tid);
    if (n_members == 0u) return;
    const unsigned int n_tiles = (n_members + 15u) >> 4;
    const unsigned int wave = tid >> 5, lane = tid & 31u, l16 = lane & 15u, hi = lane >> 4;
    const unsigned int row_a = blockIdx.x * 128u + wave * 16u + l16;
    const bool row_valid = row_a < n_rows;
    const unsigned int row_ld = row_valid ? row_a : (n_rows - 1u);
    const uint8_t* w_row = w_base + (uint64_t)e * (uint64_t)dbpe + (uint64_t)row_ld * (uint64_t)n_blocks_in * SUPER_MXFP4_BYTES;
    const size_t row_stride_b = (size_t)n_used * (size_t)xq_slot_stride;

    wm_v8f acc[2];
    #pragma unroll
    for (int t = 0; t < 2; t++)
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[t][i] = 0.0f;
    if (n_tiles == 2u) wm_mainloop_plain<2, 1>(acc, sm, w_row, xq_base, row_stride_b, xq_slot_stride, n_members, n_blocks_in, tid, l16, hi);
    else               wm_mainloop_plain<1, 1>(acc, sm, w_row, xq_base, row_stride_b, xq_slot_stride, n_members, n_blocks_in, tid, l16, hi);
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
                    partials[((size_t)b * (size_t)n_used + (size_t)slot) * (size_t)n_rows + (size_t)row_a] = acc[t][i];
                }
            }
        }
    }
}
