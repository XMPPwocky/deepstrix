typedef _Float16 wm_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 wm_h2  __attribute__((ext_vector_type(2)));
typedef int   wm_v4i __attribute__((ext_vector_type(4)));
typedef int   wm_v8i __attribute__((ext_vector_type(8)));
typedef float wm_v8f __attribute__((ext_vector_type(8)));

// ---- f16 WMMA arm: weights f16 = kv * 2^(e8-128) (scale folded into the perm LUT), activations exact f16 q.
// kv = {0,1,2,3,4,6,8,12} * sign. Every e2m1 magnitude has f16 low byte 0, so a pattern is ONE hi byte
// (exp<<2 | mant?2:0) and *2^s is "+4*s" on that byte (verified: scratch/kernel_ideas/B_moe_prefill/f16lut_emul.py).
// Base LUT = kv * 2^-14 (all bytes >= 0, idx0 = 0); valid for e8 in [114, 137].
#define WM_LUT_LO0 0x0a080400u
#define WM_LUT_HI0 0x12100e0cu

__device__ __forceinline__ void wm_lut(uint32_t e8, uint32_t& lo, uint32_t& hi) {
    const int d = __builtin_amdgcn_sffbh(0) * 0 + ((int)e8 - 114);               // s + 14
    const uint32_t off = (uint32_t)min(max(d, 0), 23) * 4u;
    const uint32_t add = off * 0x01010101u;
    lo = (WM_LUT_LO0 + add) & 0xFFFFFF00u;                                          // idx0 stays 0
    hi = WM_LUT_HI0 + add;
}
// 4 nibbles (low nibble of each byte of x) -> 4 f16 as two u32 (w0,w1) (w2,w3)
__device__ __forceinline__ void wm_dq4(uint32_t x, uint32_t lo, uint32_t hi, uint32_t& a, uint32_t& b) {
    uint32_t hb = __builtin_amdgcn_perm(hi, lo, x & 0x07070707u);
    hb |= (x & 0x08080808u) << 4;
    a = __builtin_amdgcn_perm(0u, hb, 0x010C000Cu);
    b = __builtin_amdgcn_perm(0u, hb, 0x030C020Cu);
}
__device__ __forceinline__ wm_h16 wm_dq_frag(const uint4 q, uint32_t lo, uint32_t hi, bool hinib) {
    uint32_t x0 = q.x, x1 = q.y, x2 = q.z, x3 = q.w;
    if (hinib) { x0 >>= 4; x1 >>= 4; x2 >>= 4; x3 >>= 4; }
    uint32_t w[8];
    wm_dq4(x0, lo, hi, w[0], w[1]); wm_dq4(x1, lo, hi, w[2], w[3]);
    wm_dq4(x2, lo, hi, w[4], w[5]); wm_dq4(x3, lo, hi, w[6], w[7]);
    wm_h16 r;
    __builtin_memcpy(&r, w, 32);
    return r;
}
__device__ __forceinline__ wm_v8f wm_wmma(const wm_h16 a, const wm_h16 b, const wm_v8f c) {
#if defined(__gfx1151__) || defined(__gfx1150__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    wm_v8f r = c; (void)a; (void)b; __builtin_trap(); return r;
#endif
}

#define WM_X16_STRIDE 528u   // bytes per staged member row: 256 f16 + 16 pad (conflict-free b128)
struct WmSmem {
    __attribute__((aligned(16))) uint8_t x[WM_MAX_CHUNK * WM_X16_STRIDE];  // [member][256 f16 + pad]
    __attribute__((aligned(16))) float   yd[WM_MAX_CHUNK];                 // [tile][hi][i]
    int   member_packed[WM_MAX_CHUNK];
    float ew[WM_MAX_CHUNK];
};

// 4 int8 (one u32) -> 4 f16 (two u32), exact: (0x6400 | (q+128)) - 1152.0
__device__ __forceinline__ void wm_i8x4_to_h(uint32_t x, uint32_t& a, uint32_t& b) {
    const uint32_t u = x ^ 0x80808080u;                                   // q + 128 per byte
    const uint32_t pa = __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u);
    const uint32_t pb = __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u);
    wm_h2 ha, hb;
    __builtin_memcpy(&ha, &pa, 4); __builtin_memcpy(&hb, &pb, 4);
    const wm_h2 m = {(_Float16)-1152.0f, (_Float16)-1152.0f};
    ha = ha + m; hb = hb + m;
    __builtin_memcpy(&a, &ha, 4); __builtin_memcpy(&b, &hb, 4);
}

// Stage one super-block of every member (q8 -> exact f16, yd) into LDS. 256 threads: member = tid>>3
// (+32 per pass), block = tid&7 (32 q8 bytes -> 64 B of f16).
template <int DOWN>
__device__ __forceinline__ void wm_stage(WmSmem& sm, const uint8_t* __restrict__ xq,
                                         size_t row_stride_b, size_t row_stride_slot,
                                         unsigned int sb, unsigned int n_members, unsigned int tid) {
    const unsigned int bb = tid & 7u;
    for (unsigned int mi = tid >> 3; mi < WM_MAX_CHUNK; mi += 32u) {
        const unsigned int tile = mi >> 4, w = mi & 15u, hi = w & 1u, i = w >> 1;
        const unsigned int yidx = (tile * 2u + hi) * 8u + i;
        uint4 q0 = make_uint4(0u, 0u, 0u, 0u), q1 = q0;
        float yd = 0.0f;
        if (mi < n_members) {
            const int packed = sm.member_packed[mi];
            const unsigned int b    = (unsigned int)(packed >> 16) & 0xffffu;
            const unsigned int slot = (unsigned int)(packed & 0xffffu);
            const uint8_t* src = xq + (size_t)b * row_stride_b + (DOWN ? (size_t)slot * row_stride_slot : 0)
                               + (size_t)sb * BLOCK_Q8_K_BYTES;
            const uint4* q8 = (const uint4*)(src + 4u + bb * 32u);
#ifdef WMF_ABL_NOSTAGE   // ABLATION (timing only): no activation global loads
            q0 = make_uint4(sb ^ mi, sb, mi, bb); q1 = q0; (void)q8;
#else
            q0 = q8[0]; q1 = q8[1];
#endif
            if (bb == 0u) __builtin_memcpy(&yd, src, 4);
        }
        const uint32_t in[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        uint32_t out[16];
        #pragma unroll
        for (int j = 0; j < 8; j++) wm_i8x4_to_h(in[j], out[2 * j], out[2 * j + 1]);
        uint4* dst = (uint4*)(sm.x + mi * WM_X16_STRIDE + bb * 64u);
        dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
        dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
        dst[2] = make_uint4(out[8], out[9], out[10], out[11]);
        dst[3] = make_uint4(out[12], out[13], out[14], out[15]);
        if (bb == 0u) sm.yd[yidx] = yd;
    }
}

// Main loop over super-blocks for one wave (16 rows) and NT member tiles.
// MATS = 2: gate+up (acc[0] = gate, acc[1] = up); MATS = 1: down (acc[0]).
template <int NT, int MATS, int DOWN>
__device__ __forceinline__ void wm_mainloop(
        wm_v8f (&acc)[2][WM_NTMAX], WmSmem& sm,
        const uint8_t* __restrict__ w_row0, const uint8_t* __restrict__ w_row1,
        const uint8_t* __restrict__ xq, size_t row_stride_b, size_t row_stride_slot,
        unsigned int n_members, unsigned int n_blocks, bool row_valid,
        unsigned int tid, unsigned int l16, unsigned int hi) {
    for (unsigned int sb = 0; sb < n_blocks; sb++) {
        uint4 wq[2][8];
        uint2 wsc[2];
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            #pragma unroll
            for (int b = 0; b < 8; b++) wq[m][b] = make_uint4(0u, 0u, 0u, 0u);
            wsc[m] = make_uint2(0u, 0u);
        }
        if (row_valid) {
            const uint4* g4 = (const uint4*)(w_row0 + (size_t)sb * SUPER_MXFP4_BYTES);
#ifdef WMF_ABL_NOWLOAD   // ABLATION (timing only): weights synthesised, no weight VMEM
            { const uint32_t h = (uint32_t)((uint64_t)w_row0 >> 4) * 2654435761u + sb * 0x9E3779B9u; (void)g4;
              #pragma unroll
              for (int b = 0; b < 8; b++) wq[0][b] = make_uint4(h ^ b, h * 3u ^ b, h * 5u ^ b, h * 7u ^ b); }
#else
            #pragma unroll
            for (int b = 0; b < 8; b++) wq[0][b] = WM_LOAD16(g4 + b);
#endif
            __builtin_memcpy(&wsc[0], w_row0 + (size_t)sb * SUPER_MXFP4_BYTES + 128u, 8);
            if (MATS == 2) {
                const uint4* u4 = (const uint4*)(w_row1 + (size_t)sb * SUPER_MXFP4_BYTES);
#ifdef WMF_ABL_NOWLOAD
                { const uint32_t h = (uint32_t)((uint64_t)w_row1 >> 4) * 2654435761u + sb * 0x85EBCA6Bu; (void)u4;
                  #pragma unroll
                  for (int b = 0; b < 8; b++) wq[1][b] = make_uint4(h ^ b, h * 3u ^ b, h * 5u ^ b, h * 7u ^ b); }
#else
                #pragma unroll
                for (int b = 0; b < 8; b++) wq[1][b] = WM_LOAD16(u4 + b);
#endif
                __builtin_memcpy(&wsc[1], w_row1 + (size_t)sb * SUPER_MXFP4_BYTES + 128u, 8);
            }
        }
        wm_stage<DOWN>(sm, xq, row_stride_b, row_stride_slot, sb, n_members, tid);
        __syncthreads();

        wm_v8f ydv[WM_NTMAX];
        #pragma unroll
        for (int t = 0; t < NT; t++) __builtin_memcpy(&ydv[t], &sm.yd[(t * 2u + hi) * 8u], 32);

        wm_v8f tacc[2][WM_NTMAX];
        #pragma unroll
        for (int m = 0; m < MATS; m++)
            #pragma unroll
            for (int t = 0; t < NT; t++)
                #pragma unroll
                for (int i = 0; i < 8; i++) tacc[m][t][i] = 0.0f;

        #pragma unroll
        for (unsigned int b = 0; b < 8u; b++) {
            wm_h16 bw_lo[2], bw_hi[2];
            #pragma unroll
            for (int m = 0; m < MATS; m++) {
                const uint32_t e8 = (((b < 4u) ? wsc[m].x : wsc[m].y) >> (8u * (b & 3u))) & 0xffu;
                uint32_t lo, hi4;
                wm_lut(e8, lo, hi4);
                bw_lo[m] = wm_dq_frag(wq[m][b], lo, hi4, false);
                bw_hi[m] = wm_dq_frag(wq[m][b], lo, hi4, true);
            }
            #pragma unroll
            for (int t = 0; t < NT; t++) {
                const uint8_t* xa = sm.x + (t * 16u + l16) * WM_X16_STRIDE + b * 64u;
                wm_h16 a_lo, a_hi;
                __builtin_memcpy(&a_lo, xa, 32);
                __builtin_memcpy(&a_hi, xa + 32u, 32);
                #pragma unroll
                for (int m = 0; m < MATS; m++) {
                    tacc[m][t] = wm_wmma(a_lo, bw_lo[m], tacc[m][t]);
                    tacc[m][t] = wm_wmma(a_hi, bw_hi[m], tacc[m][t]);
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < MATS; m++)
            #pragma unroll
            for (int t = 0; t < NT; t++)
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[m][t][i] += ydv[t][i] * tacc[m][t][i];
        __syncthreads();   // LDS reuse by the next super-block
    }
}

