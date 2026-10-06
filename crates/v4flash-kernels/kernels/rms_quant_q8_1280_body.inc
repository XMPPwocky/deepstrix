// The body of rms_quant_q8_1280_batched (rms_norm.hip), #included VERBATIM inside the direct kernel and inside the
// device function its `_ind` twin calls, so the direct kernel compiles from exactly the
// token stream it had before the twin existed (GRAPH_KEYS_DESIGN.md 2.3). Not standalone.
    __shared__ float sdata[256];
    const unsigned int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, w = tid >> 5;
    const float* xb = x + (size_t)b * 1280u;
    float xv[5], wv[5];
    #pragma unroll
    for (unsigned j = 0; j < 5; ++j) xv[j] = xb[tid + 256u * j];
    #pragma unroll
    for (unsigned j = 0; j < 5; ++j) wv[j] = weight[tid + 256u * j];
    double partial = 0.0;
    #pragma unroll
    for (unsigned j = 0; j < 5; ++j) {
        const float v = xv[j];
        partial += (double)v * (double)v;
    }
    sdata[tid] = (float)partial;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float mean_sq = sdata[0] / (float)1280u;
    const float scale = rsqrtf(mean_sq + eps);
    float* ob = out + (size_t)b * 1280u;
    #pragma unroll
    for (unsigned j = 0; j < 5; ++j) {
        const float v = xv[j] * scale * wv[j];
        ob[tid + 256u * j] = v;
        const unsigned k = 8u * j + w;
        float amax = fabsf(v);
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, amax, off));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        if (lane == 0) xscale[(size_t)b * 40u + k] = d;
        int q = (int)lrintf(v * id);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        xq[(size_t)b * 1280u + 32u * k + lane] = (signed char)q;
    }
