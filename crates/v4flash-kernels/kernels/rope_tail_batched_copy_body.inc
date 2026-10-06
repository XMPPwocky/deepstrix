// The body of rope_tail_batched_copy (rope_tail.hip), #included VERBATIM inside the direct kernel and inside the
// device function its `_ind` twin calls, so the direct kernel compiles from exactly the
// token stream it had before the twin existed (GRAPH_KEYS_DESIGN.md 2.3). Not standalone.
    const unsigned head = blockIdx.x, b = blockIdx.z, tid = threadIdx.x;
    if (head >= n_head) return;
    const unsigned n_nope = head_dim - n_rot;
    const size_t base = (size_t)b * n_head * head_dim + (size_t)head * head_dim;
    const float4* s4 = reinterpret_cast<const float4*>(src + base);
    float4* d4 = reinterpret_cast<float4*>(dst + base);
    for (unsigned i = tid; i < n_nope / 4u; i += blockDim.x) d4[i] = s4[i];
    if (tid < 32u && 2u * tid < n_rot) {
        const unsigned lane = tid, i = 2u * lane;
        const float theta = rf_theta(pos_per_b[b], lane, theta_scale, freq_scale, ext_factor, corr_low, corr_high);
        const float sin_sign = (inverse != 0) ? -1.0f : 1.0f;
        const float c = cosf(theta) * mscale_eff;
        const float s = sin_sign * sinf(theta) * mscale_eff;
        const float x0 = src[base + n_nope + i + 0];
        const float x1 = src[base + n_nope + i + 1];
        dst[base + n_nope + i + 0] = x0 * c - x1 * s;
        dst[base + n_nope + i + 1] = x0 * s + x1 * c;
    }
