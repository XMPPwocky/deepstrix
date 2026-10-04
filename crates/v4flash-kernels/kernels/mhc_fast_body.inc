// The body of mhc_fast_batched (kernels/mhc_fast.hip), #included VERBATIM inside the direct
// kernel and inside `mhc_fast_body` (the device function its `_ind` twin calls), so the direct
// kernel compiles from exactly the token stream it had before the twin existed (instruction-
// identical; GRAPH_KEYS_DESIGN.md 2.3) and the twin shares one source. Not a standalone file.
    const uint32_t role = blockIdx.x;
    const uint32_t b = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (MF_WARP - 1u);
    const uint32_t warp = tid >> 5u;

    __shared__ float sq[MF_RMS_WGS][MF_BLOCK];
    __shared__ float inv_s;
    __shared__ float per_warp[MF_MAX_WARPS];
    __shared__ unsigned int is_last;

    if (role < n_mix) {
        // ------------------------------------------------ mix row `role`
        const float *xb = x + (size_t)b * (size_t)MF_K;
        const __half *wr = weight + (size_t)role * (size_t)MF_K;
        float xv[MF_PT];
        float wv[MF_PT];
        #pragma unroll
        for (uint32_t j = 0; j < MF_PT; j++) xv[j] = xb[tid + MF_BLOCK * j];
        #pragma unroll
        for (uint32_t j = 0; j < MF_PT; j++) wv[j] = __half2float(wr[tid + MF_BLOCK * j]);
        float acc = 0.0f;
        if (mode == 0u) {
            // f16_matvec_narrow_pre_scaled's dot over the RAW residual, in the
            // dot's order; the RMS WG computes the scalar, the tail applies it.
            #pragma unroll
            for (uint32_t j = 0; j < MF_PT; j++) acc += wv[j] * xv[j];
        } else {
            // rms_norm_no_weight_batched: one 256-wide tree over the row.
            double partial = 0.0;
            #pragma unroll
            for (uint32_t j = 0; j < MF_PT; j++) {
                const float v = xv[j];
                partial += (double)v * (double)v;
            }
            sq[0][tid] = (float)partial;
            __syncthreads();
            for (uint32_t s = MF_BLOCK / 2; s > 0; s >>= 1) {
                if (tid < s) sq[0][tid] += sq[0][tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                const float mean_sq = sq[0][0] / (float)k;
                inv_s = rsqrtf(mean_sq + rms_eps);
            }
        }
        __syncthreads();
        if (mode != 0u) {
            const float sc = inv_s;
            #pragma unroll
            for (uint32_t j = 0; j < MF_PT; j++) {
                const float f = xv[j] * sc;   // the value rms_nw stored in `flat`
                acc += wv[j] * f;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down(acc, offset, MF_WARP);
        }
        if (lane == 0) per_warp[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < MF_MAX_WARPS) ? per_warp[lane] : 0.0f;
            for (int offset = MF_MAX_WARPS / 2; offset > 0; offset >>= 1) {
                v += __shfl_down(v, offset, MF_WARP);
            }
            if (lane == 0) {
                // Mode 0: the RAW dot; the tail applies `* inv_s`.
                mix_out[(size_t)b * (size_t)MF_MIX + role] = v;
            }
        }
    } else if (mode == 0u && n_mix != 0u && role == n_mix + do_collapse) {
        // ------------------------------------------------ RMS WG (mode 0)
        // rms_norm_partial_sum_sq over 16 slices, as mhc_mix_batched computed
        // it: slice w's jj-th element for thread t is x[t + 256 (w TPS + jj)].
        const float *xb = x + (size_t)b * (size_t)MF_K;
        float xv[MF_PT];
        #pragma unroll
        for (uint32_t j = 0; j < MF_PT; j++) xv[j] = xb[tid + MF_BLOCK * j];
        #pragma unroll
        for (uint32_t w = 0; w < MF_RMS_WGS; w++) {
            double partial = 0.0;
            #pragma unroll
            for (uint32_t jj = 0; jj < MF_TPS; jj++) {
                const float v = xv[w * MF_TPS + jj];
                partial += (double)v * (double)v;
            }
            sq[w][tid] = (float)partial;
        }
        __syncthreads();
        for (uint32_t s = MF_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                for (uint32_t w = 0; w < MF_RMS_WGS; w++) sq[w][tid] += sq[w][tid + s];
            }
            __syncthreads();
        }
        // rms_norm_finalize_inv: one thread, partials summed in order.
        if (tid == 0) {
            float s = 0.0f;
            for (uint32_t w = 0; w < MF_RMS_WGS; w++) s += sq[w][0];
            const float mean_sq = s / (float)k;
            inv_rows[b] = rsqrtf(mean_sq + rms_eps);
        }
    } else if (do_collapse != 0u && role == n_mix) {
        // ------------------------------------------------ collapse
        // hc_weighted_sum_batched (weights = the carry) for this thread's
        // outputs d = tid + 256 j -- exactly the elements rms_norm_weighted's
        // thread tid owns -- then that kernel's passes on the values in
        // registers (the same f32 it would have re-read from `cur`).
        const float *cxb = cx + (size_t)b * (size_t)MF_K;
        const float *cwb = carry + (size_t)b * (size_t)MF_MIX;
        float xs[MF_HC][MF_PE];
        #pragma unroll
        for (uint32_t h = 0; h < MF_HC; h++) {
            #pragma unroll
            for (uint32_t j = 0; j < MF_PE; j++) xs[h][j] = cxb[(size_t)h * MF_NE + tid + MF_BLOCK * j];
        }
        float nw[MF_PE];
        #pragma unroll
        for (uint32_t j = 0; j < MF_PE; j++) nw[j] = norm_w[tid + MF_BLOCK * j];
        float cw[MF_HC];
        #pragma unroll
        for (uint32_t h = 0; h < MF_HC; h++) cw[h] = cwb[h];
        float cv[MF_PE];
        #pragma unroll
        for (uint32_t j = 0; j < MF_PE; j++) {
            float a = 0.0f;
            #pragma unroll
            for (uint32_t h = 0; h < MF_HC; h++) a += xs[h][j] * cw[h];
            cv[j] = a;
            cur_out[(size_t)b * MF_NE + tid + MF_BLOCK * j] = a;
        }
        double partial = 0.0;
        #pragma unroll
        for (uint32_t j = 0; j < MF_PE; j++) {
            const float v = cv[j];
            partial += (double)v * (double)v;
        }
        sq[0][tid] = (float)partial;
        __syncthreads();
        for (uint32_t s = MF_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sq[0][tid] += sq[0][tid + s];
            __syncthreads();
        }
        const float mean_sq = sq[0][0] / (float)ne;
        const float nscale = rsqrtf(mean_sq + rms_eps);
        #pragma unroll
        for (uint32_t j = 0; j < MF_PE; j++) {
            norm_out[(size_t)b * MF_NE + tid + MF_BLOCK * j] = cv[j] * nscale * nw[j];
        }
    } else {
        return;
    }
    if (n_mix == 0u) return;   // collapse only: no split, no counter

    // ---- the LAST WG of this batch row runs the Sinkhorn split
    const unsigned int n_part = n_mix + do_collapse + (mode == 0u ? 1u : 0u);
    if (tid == 0) {
        __threadfence();
        const unsigned int prev = atomicAdd(&counters[b], 1u);
        is_last = (prev == n_part - 1u) ? 1u : 0u;
    }
    __syncthreads();
    if (is_last == 0u) return;
    // One wave from here on (MF_WAVE_SYNC only, never a workgroup barrier
    // after the other waves exit -- see mhc_arena.hip).
    if (tid >= MF_WARP) return;
    __threadfence();
    if (tid == 0) counters[b] = 0u;
    mf_sinkhorn_tail(tid, mix_out + (size_t)b * (size_t)MF_MIX, mode == 0u ? inv_rows + b : nullptr,
                     split_out + (size_t)b * (size_t)MF_MIX,
                     write_carry ? carry + (size_t)b * (size_t)MF_MIX : nullptr,
                     scale, base, n_hc, iters, eps);
