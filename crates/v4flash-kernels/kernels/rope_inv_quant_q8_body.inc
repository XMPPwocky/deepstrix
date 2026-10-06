// The body of rope_inv_quant_q8 (rope_tail.hip), #included VERBATIM inside the direct kernel and inside the
// device function its `_ind` twins call, so the direct kernel compiles from exactly the
// token stream it had before the twins existed (GRAPH_KEYS_DESIGN.md 2.3). Not standalone.
    const unsigned head = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, lane = tid & 31u, w = tid >> 5;
    if (head >= 64u) return;  // idle grid-pad column
    const size_t row = (size_t)b * 32768u;
    float* hb = heads + row + (size_t)head * 512u;
    #pragma unroll
    for (unsigned t = 0; t < 4; ++t) {
        const unsigned blk = 4u * w + t;          // 0..15 within the head
        const unsigned e = 32u * blk + lane;      // 0..511 within the head
        float v = hb[e];
        if (blk >= 14u) {                         // 448..511: the rope tail (wave-uniform branch)
            v = rf_rot_pair(v, e, pos_per_b[b], (e - 448u) >> 1, -1.0f, theta_scale, freq_scale, ext_factor,
                            mscale_eff, corr_low, corr_high);
            hb[e] = v;
        }
        const size_t k = (size_t)head * 16u + blk;
        rf_q8_block(v, lane, xq + row + 32u * k, xscale + (size_t)b * 1024u + k);
    }
