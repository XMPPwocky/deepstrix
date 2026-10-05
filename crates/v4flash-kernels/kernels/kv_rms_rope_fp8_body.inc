// The body of kv_rms_rope_fp8 (rope_tail.hip), #included VERBATIM inside the direct kernel and inside the
// device function its `_ind` twin calls, so the direct kernel compiles from exactly the
// token stream it had before the twin existed (GRAPH_KEYS_DESIGN.md 2.3). Not standalone.
    __shared__ float sdata[256];
    const unsigned b = blockIdx.x, tid = threadIdx.x;
    const float* xb = x + (size_t)b * 512u;
    // rms_w_fast<2> (rms_norm.hip), verbatim up to the output expression
    float xv[2], wv[2], v[2];
    #pragma unroll
    for (unsigned j = 0; j < 2; ++j) xv[j] = xb[tid + 256u * j];
    #pragma unroll
    for (unsigned j = 0; j < 2; ++j) wv[j] = weight[tid + 256u * j];
    double partial = 0.0;
    #pragma unroll
    for (unsigned j = 0; j < 2; ++j) {
        const float xx = xv[j];
        partial += (double)xx * (double)xx;
    }
    sdata[tid] = (float)partial;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float mean_sq = sdata[0] / (float)512u;
    const float scale = rsqrtf(mean_sq + eps);
    #pragma unroll
    for (unsigned j = 0; j < 2; ++j) v[j] = xv[j] * scale * wv[j];
    // rope_tail_batched on elements 448..511 (waves 6-7 are fully active here)
    if (tid >= 192u) {
        const unsigned e = tid + 256u;
        v[1] = rf_rot_pair(v[1], e, pos_per_b[b], (e - 448u) >> 1, 1.0f, theta_scale, freq_scale, ext_factor,
                           mscale_eff, corr_low, corr_high);
    }
    // fp8_act_quant_inplace (fp4_kv_quant.hip) on every 32-block
    float* ob = out + (size_t)b * 512u;
    #pragma unroll
    for (unsigned j = 0; j < 2; ++j) {
        float a = fabsf(v[j]);
        a = fmaxf(a, __shfl_xor_sync(RF_FULL_WARP_MASK, a, 1));
        a = fmaxf(a, __shfl_xor_sync(RF_FULL_WARP_MASK, a, 2));
        a = fmaxf(a, __shfl_xor_sync(RF_FULL_WARP_MASK, a, 4));
        a = fmaxf(a, __shfl_xor_sync(RF_FULL_WARP_MASK, a, 8));
        a = fmaxf(a, __shfl_xor_sync(RF_FULL_WARP_MASK, a, 16));
        const float amax = fmaxf(a, 1.0e-4f);
        const float t = amax / 448.0f;
        int ex;
        const float m = frexpf(t, &ex);
        const int cexp = (m == 0.5f) ? ex - 1 : ex;
        const float sc = ldexpf(1.0f, cexp);
        float q = v[j] / sc;
        q = fminf(fmaxf(q, -448.0f), 448.0f);
        ob[tid + 256u * j] = dsv4_e4m3fn_dequant(q) * sc;
    }
