// The body of q8_0_grouped_gemv_bpack (q8_0_grouped_matvec.hip, the runtime-batch kernel), #included
// VERBATIM inside the direct kernel and inside the device function its `_ind` twin calls, so the
// direct kernel compiles from exactly the token stream it had before the twin existed
// (GRAPH_KEYS_DESIGN.md 2.3). Not standalone.
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;
    if (batch > GGEMV_BPACK_MAX) return;   // caller guards; belt and braces

    uint32_t g = row / rank;
    const unsigned char* wr = w + (size_t)row * (size_t)blocks_per_group * 34u;
    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)blocks_per_group * 2u);

    // Per-batch bases into the activation, which IS per-b (only w is shared).
    const uint32_t xq_stride = n_groups * group_dim;
    const uint32_t xs_stride = n_groups * blocks_per_group;

    float acc[GGEMV_BPACK_MAX];
    for (uint32_t b = 0; b < batch; ++b) acc[b] = 0.0f;

    for (uint32_t bl = lane; bl < blocks_per_group; bl += WARP_LANES) {
        const int8_t* qs = quants + (size_t)bl * 32u;
        uint4 av0, av1;
        __builtin_memcpy(&av0, qs, 16);
        __builtin_memcpy(&av1, qs + 16, 16);
        const float ws = __half2float(scales[bl]);

        for (uint32_t b = 0; b < batch; ++b) {
            const int8_t* xqb = xq + (size_t)b * xq_stride
                                   + (size_t)g * group_dim + (size_t)bl * 32u;
            uint4 bv0, bv1;
            __builtin_memcpy(&bv0, xqb, 16);
            __builtin_memcpy(&bv1, xqb + 16, 16);
            int32_t dot = 0;
            dot = gdp4a_sudot((int32_t)av0.x, (int32_t)bv0.x, dot);
            dot = gdp4a_sudot((int32_t)av0.y, (int32_t)bv0.y, dot);
            dot = gdp4a_sudot((int32_t)av0.z, (int32_t)bv0.z, dot);
            dot = gdp4a_sudot((int32_t)av0.w, (int32_t)bv0.w, dot);
            dot = gdp4a_sudot((int32_t)av1.x, (int32_t)bv1.x, dot);
            dot = gdp4a_sudot((int32_t)av1.y, (int32_t)bv1.y, dot);
            dot = gdp4a_sudot((int32_t)av1.z, (int32_t)bv1.z, dot);
            dot = gdp4a_sudot((int32_t)av1.w, (int32_t)bv1.w, dot);
            acc[b] += ws * xscale[(size_t)b * xs_stride
                                  + (size_t)g * blocks_per_group + bl] * (float)dot;
        }
    }

    for (uint32_t b = 0; b < batch; ++b) {
        float a = gwarp_sum_f32(acc[b]);
        if (lane == 0) out[(size_t)b * (size_t)out_dim + row] = a;
    }
