// gateup_r32_ablation.inc — parameterized body of
// q4_k_gate_up_swiglu_tiled_reg_col_r32 for component ablation (TIMING ONLY,
// parity intentionally broken). Controlled by macros (all default 1 = full):
//   ABL_STAGE   : DRAM->LDS activation copy
//   ABL_DOTLOAD : load q8 activation dwords from LDS (the LDS read round-trip)
//   ABL_DOT     : the dp4a (sudot4) MACs
//   ABL_REDUCE  : the 32-lane cross-lane warp reduction
//   ABL_SCALE   : the per-superblock d/m scale FMA accumulation
// The weight preload (registers) is always kept (it's the read-once BW).
#ifndef ABL_STAGE
#define ABL_STAGE 1
#endif
#ifndef ABL_DOTLOAD
#define ABL_DOTLOAD 1
#endif
#ifndef ABL_DOT
#define ABL_DOT 1
#endif
#ifndef ABL_REDUCE
#define ABL_REDUCE 1
#endif
#ifndef ABL_SCALE
#define ABL_SCALE 1
#endif
{
    const unsigned int e = blockIdx.y;
    __shared__ int n_members;
    if (threadIdx.x == 0) n_members = group_count[e];
    __syncthreads();
    if (n_members == 0) return;

    __shared__ __align__(16) uint8_t s_xq[NT_COL * LDS_XQ_BYTES];

    const unsigned int tid          = threadIdx.x;
    const unsigned int row_in_block = tid / 32u;
    const unsigned int lane         = tid & 31u;
    const unsigned int row          = blockIdx.x * GU_ROWS_PER_WG + row_in_block;
    const bool         row_ok       = (row < n_rows);
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
#if ABL_STAGE
        for (int j = 0; j < nt; j++) {
            const int packed = expert_members[(size_t)e*(size_t)max_per_expert+(size_t)(m0+j)];
            const unsigned int b = (unsigned int)(packed >> 16);
            const uint8_t* xq_b = xq + (size_t)b * (size_t)xq_bytes;
            uint8_t* dst = s_xq + (size_t)j * LDS_XQ_BYTES;
            stage_lds_vec(dst, xq_b, xq_bytes, tid, blockDim.x);
        }
#endif
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
#if ABL_DOTLOAD
                        const int32_t q8lo = load_dword_i8_t(q8 + i*4);
                        const int32_t q8hi = load_dword_i8_t(q8 + 32 + i*4);
#else
                        const int32_t q8lo = (int32_t)(i + 1);
                        const int32_t q8hi = (int32_t)(i + 2);
#endif
#if ABL_DOT
                        gda = sudot4_t((int32_t)(g_qs[k][i] & 0x0f0f0f0fu), q8lo, gda);
                        gdb = sudot4_t((int32_t)((g_qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, gdb);
                        uda = sudot4_t((int32_t)(u_qs[k][i] & 0x0f0f0f0fu), q8lo, uda);
                        udb = sudot4_t((int32_t)((u_qs[k][i] >> 4) & 0x0f0f0f0fu), q8hi, udb);
#else
                        gda += q8lo; gdb += q8hi; uda += q8lo; udb += q8hi;
#endif
                    }
#if ABL_SCALE
                    const int32_t bsa = (int32_t)bs[4u*sub+0u]+(int32_t)bs[4u*sub+1u];
                    const int32_t bsb = (int32_t)bs[4u*sub+2u]+(int32_t)bs[4u*sub+3u];
                    g_sum += yd*g_xd[k]*((float)g_da[k]*gda + (float)g_db[k]*gdb)
                           - yd*g_xdm[k]*((float)g_ma[k]*bsa + (float)g_mb[k]*bsb);
                    u_sum += yd*u_xd[k]*((float)u_da[k]*uda + (float)u_db[k]*udb)
                           - yd*u_xdm[k]*((float)u_ma[k]*bsa + (float)u_mb[k]*bsb);
#else
                    g_sum += yd * (float)(gda + gdb);
                    u_sum += yd * (float)(uda + udb);
#endif
                }
#if ABL_REDUCE
                g_sum = warp_sum_f32_t(g_sum);
                u_sum = warp_sum_f32_t(u_sum);
#endif
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
