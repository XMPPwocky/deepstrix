// The body of rope_inv_quant_fp8 (rope_tail.hip; rope_inv_quant_q8_body.inc with the reference's
// rounding points and the fp8 block), #included VERBATIM inside the direct kernel and inside the
// device function its `_ind` twins call (GRAPH_KEYS_DESIGN.md 2.3). Not standalone.
// The 1017 reference's `o` (act_quant_fp8.inc aq_bf16): bf16(attention output) on every dim; on
// the rope tail the inverse rotation of those bf16 values (same rf_rot_pair arithmetic as the Q8_0
// path), rounded to bf16 again; act_quant of the bf16 values. `heads` keeps the f32 attention
// output on dims 0..447 (as on the Q8_0 path) and gets the bf16 rotated values on 448..511.
    const unsigned head = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, lane = tid & 31u, w = tid >> 5;
    if (head >= 64u) return;  // idle grid-pad column
    const size_t row = (size_t)b * 32768u;
    float* hb = heads + row + (size_t)head * 512u;
    #pragma unroll
    for (unsigned t = 0; t < 4; ++t) {
        const unsigned blk = 4u * w + t;          // 0..15 within the head
        const unsigned e = 32u * blk + lane;      // 0..511 within the head
        float v = aq_bf16(hb[e]);
        if (blk >= 14u) {                         // 448..511: the rope tail (wave-uniform branch)
            v = aq_bf16(rf_rot_pair(v, e, pos_per_b[b], (e - 448u) >> 1, -1.0f, theta_scale, freq_scale, ext_factor,
                                    mscale_eff, corr_low, corr_high));
            hb[e] = v;
        }
        const size_t k = (size_t)head * 16u + blk;
        rf_fp8_block(v, lane, xq + row + 32u * k, xscale + (size_t)b * 1024u + k);
    }
