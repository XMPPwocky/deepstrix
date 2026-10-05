// The body of q8_0_quantize_f32_wave (q8_0_matvec.hip), #included VERBATIM inside the direct kernel and inside the
// device function its `_ind` twins call, so the direct kernel compiles from exactly the
// token stream it had before the twins existed (GRAPH_KEYS_DESIGN.md 2.3). Not standalone.
    uint32_t b = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (b >= blocks) return;
    float v = x[(size_t)b * 32u + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULL_WARP_MASK, amax, off));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0) xscale[b] = d;
    int q = (int)lrintf(v * id);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    xq[(size_t)b * 32u + lane] = (int8_t)q;
