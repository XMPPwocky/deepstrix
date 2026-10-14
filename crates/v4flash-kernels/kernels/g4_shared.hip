// g4_shared -- dGPU fusion group G4 (knob V41_DGPU_G4_SHARED, default off): the decode shared expert's
// input quantize + the late FFN mixes as ONE latency-bound launch (`g4_qmix`), every output
// BIT-IDENTICAL (tests/g4_shared_bitexact.rs):
//
//   production                                              G4
//   q8_0_quantize_f32_wave(ffn_input_norm) -> xq, xs        g4_qmix: QUANT workgroups (the wave quantize,
//   shared_gateup_swiglu_q8 / gate, up, swiglu, quant         8 blocks per WG, as before)
//   q8_0_gemv_bpack_tB<b> (down) -> ffn_shared                + the late mixes with the row RMS computed
//   mhc_fast_batched (mix NORMED + carry := split)              ONCE per row instead of in all 24 mix WGs
//                                                           the gate/up and down GEMVs: unchanged launches
//                                                           (they run at the byte roof: nothing to fold
//                                                           into them that does not slow them, measured)
//
// The GEMVs stay as they are. Two measured dead ends (built and measured 10-14, not kept): folding the quantize into the
// gate/up workgroups (each of 72 WGs re-quantizes the input: +10 us at b = 4) and running the mixes
// in the down GEMV's grid (their latency-bound loads queue behind the down's stream on the same CUs:
// +7 us at b = 4 vs the two launches).
//
// The mixes in mhc_fast mode 1 (normed) are latency-bound: every one of a row's 24 WGs computes the
// row's fp64 sum of squares (80 DFMAs per thread at 1/32 rate) and its 256-wide tree before its dot.
// Here ONE RMS workgroup per row does that (the same expressions, so the same inv_s) and releases it;
// the 24 mix workgroups prefetch their weight rows meanwhile, wait for it, then run the dot on the
// row read from L2 (f = x * inv_s, then fma(w, f, acc) in the same order).
//
// ROLES by TICKET (every WG draws one on entry; a WG only waits on LOWER tickets, held by WGs that are
// already resident -- forward progress needs no dispatch-order assumption; the G1 / G6 protocol):
//   t <  b             RMS of row z = t: x row, fp64 partials, tree, inv_s -> scratch, RELEASE ready[z]
//   t <  b + nq        QUANT workgroup t - b (blocks 8 (t - b) .. + 8 of the input, one wave each)
//   else               MIX row role r of row z ((t - b - nq) = 24 z + r): weights first, wait ready[z],
//                      the dot; the LAST of a row's 24 runs the Sinkhorn split (+ carry := split)
//   last of all        resets the ticket and the ready flags (self-resetting).
// Bounded polls (2 s): an expired poll sets the sticky S_ERR word instead of hanging the GPU.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef MHC_HC_DIM
#define MHC_HC_DIM 16384
#endif

#define G4_WARP 32u
#define G4_FULL_MASK 0xFFFFFFFFFFFFFFFFull
#define G4_K ((unsigned)MHC_HC_DIM)
#define G4_PT (G4_K / 256u)        // mix elements per thread (80)
#define G4_MIX 24u
#define G4_HC 4u
#define G4_MAXB 16u

#define S_TICKET 0u
#define S_FINISH 32u
#define S_READY(z) (64u + 32u * (z))
#define S_ERR (64u + 32u * G4_MAXB)
#define G4_SPIN_CAP 200000000ull

struct G4Args {
    // quantize (q8_0_quantize_f32_wave)
    int8_t* xq;
    float* xscale;
    const float* xin;          // [b, n_embd] ffn_input_norm
    // mixes (mode 1, no collapse)
    float* split_out;          // [b, 24]
    float* mix_out;            // [b, 24]
    unsigned* counters;        // [b], zero between launches
    const __half* weight;      // [24, K] hc_ffn_fn
    const float* x;            // [b, K] after_attn_hc
    const float* scale;        // [3]
    const float* base;         // [24]
    float* carry;              // [b, 24]
    float* inv_scr;            // [16] scratch: the row RMS scalars
    unsigned* sync;            // zero between launches (but S_ERR)
    unsigned long long* ts;    // debug phase timestamps or null: [ticket][8]
    unsigned b;
    unsigned qblocks;          // b * n_embd / 32
    unsigned nq;               // quantize WGs = ceil(qblocks / 8)
    unsigned write_carry;
    float rms_eps;
    unsigned n_hc;
    unsigned iters;
    float eps;
};

__device__ __forceinline__ unsigned long long g4_now() { return __builtin_readsteadycounter(); }
#define G4_TS(k) do { if (ts != nullptr) ts[k] = g4_now(); } while (0)

__device__ __forceinline__ unsigned g4_sload_dev(const unsigned* pv) {
    const unsigned long long pa = (unsigned long long)pv;
    const unsigned lo = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)pa);
    const unsigned hi = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)(pa >> 32));
    const unsigned* p = (const unsigned*)(((unsigned long long)hi << 32) | lo);
    unsigned r;
#if defined(__gfx1200__) || defined(__gfx1201__)
    asm volatile("s_dcache_inv\n\ts_wait_kmcnt 0x0\n\ts_load_b32 %0, %1, 0x0 scope:SCOPE_DEV\n\ts_wait_kmcnt 0x0" : "=s"(r) : "s"(p) : "memory");
#else
    asm volatile("s_dcache_inv\n\ts_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(r) : "s"(p) : "memory");
#endif
    return r;
}
__device__ __forceinline__ void g4_spoll(const unsigned* p, unsigned* err) {
    const unsigned long long t0 = g4_now();
    while (g4_sload_dev(p) == 0u) {
        if (g4_now() - t0 > G4_SPIN_CAP) {
            if ((threadIdx.x & 31u) == 0u) atomicOr(err, 1u);
            break;
        }
        __builtin_amdgcn_s_sleep(1);
    }
}

// Quad broadcast of lane (4 * (lane / 4) + K) -- DPP quad_perm [K, K, K, K].
template <int K>
__device__ __forceinline__ float g4_quad_bcast(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), K | (K << 2) | (K << 4) | (K << 6), 0xf, 0xf, false));
}
// Column gather: lane i (< 16) takes lane (i & 3) + 4 Y of its 16-lane row (v_permlane16_b32).
template <unsigned Y>
__device__ __forceinline__ float g4_col_get(float v) {
    constexpr unsigned lo = (0u + 4u * Y) | ((1u + 4u * Y) << 4) | ((2u + 4u * Y) << 8) | ((3u + 4u * Y) << 12)
                          | ((0u + 4u * Y) << 16) | ((1u + 4u * Y) << 20) | ((2u + 4u * Y) << 24) | ((3u + 4u * Y) << 28);
    return __int_as_float(__builtin_amdgcn_permlane16(__float_as_int(v), __float_as_int(v), lo, lo, false, false));
}

// mhc_fast.hip's mf_sinkhorn_tail with inv == nullptr (mode 1: the mix values are already scaled), in
// REGISTERS: the same expressions in the same order (lane i < 16 holds c[i]; a row's values sit in one
// DPP quad, a column's are gathered with v_permlane16, both in index order). Wave 0 only.
__device__ __forceinline__ static void g4_sinkhorn_regs(
        const unsigned int tid, float *mix_row, float *out_b, float *carry_b,
        const float *scale, const float *base, unsigned int n_hc, unsigned int iters, float eps) {
    float v = 0.0f;
    if (tid < G4_MIX) {
        volatile float *mv = mix_row;
        v = mv[tid];
    }
    const bool act = tid < 16u && n_hc == G4_HC;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    if (act) {
        if (tid < G4_HC) {
            const float z = v * pre_scale + base[tid];
            float sig;
            if (z >= 0.0f) sig = 1.0f / (1.0f + expf(-z));
            else           sig = expf(z) / (1.0f + expf(z));
            const float o = sig + eps;
            out_b[tid] = o;
            if (carry_b) carry_b[tid] = o;
        } else if (tid < 2 * G4_HC) {
            const unsigned int off = tid;
            const float z = v * post_scale + base[off];
            float sig;
            if (z >= 0.0f) sig = 1.0f / (1.0f + expf(-z));
            else           sig = expf(z) / (1.0f + expf(z));
            const float o = 2.0f * sig;
            out_b[off] = o;
            if (carry_b) carry_b[off] = o;
        }
    }
    const float mc = __shfl(v, (int)((tid + 2u * G4_HC) & 31u), 32);
    const unsigned int idx = tid & 15u;
    float c = mc * comb_scale + base[2u * G4_HC + idx];
    float m = -INFINITY;
    {
        const float r0 = g4_quad_bcast<0>(c), r1 = g4_quad_bcast<1>(c), r2 = g4_quad_bcast<2>(c), r3 = g4_quad_bcast<3>(c);
        if (r0 > m) m = r0;
        if (r1 > m) m = r1;
        if (r2 > m) m = r2;
        if (r3 > m) m = r3;
    }
    c = expf(c - m);
    auto row_sum = [&](float x) {
        const float r0 = g4_quad_bcast<0>(x), r1 = g4_quad_bcast<1>(x), r2 = g4_quad_bcast<2>(x), r3 = g4_quad_bcast<3>(x);
        float s = 0.0f;
        s += r0;
        s += r1;
        s += r2;
        s += r3;
        return s;
    };
    auto col_sum = [&](float x) {
        const float c0 = g4_col_get<0>(x), c1 = g4_col_get<1>(x), c2 = g4_col_get<2>(x), c3 = g4_col_get<3>(x);
        float s = 0.0f;
        s += c0;
        s += c1;
        s += c2;
        s += c3;
        return s;
    };
    c = c / row_sum(c) + eps;
    c = c / (col_sum(c) + eps);
    for (unsigned int iter = 1; iter < iters; iter++) {
        c = c / (row_sum(c) + eps);
        c = c / (col_sum(c) + eps);
    }
    if (act) {
        out_b[2u * G4_HC + idx] = c;
        if (carry_b) carry_b[2u * G4_HC + idx] = c;
    }
}

extern "C" __global__ __launch_bounds__(256) void g4_qmix(G4Args a) {
    __shared__ float sq[256];
    __shared__ float per_warp[8];
    __shared__ unsigned s_ticket;
    __shared__ unsigned is_last;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & (G4_WARP - 1u);
    const unsigned warp = tid >> 5u;
    const unsigned long long t_start = g4_now();
    if (tid == 0) s_ticket = atomicAdd(a.sync + S_TICKET, 1u);
    __syncthreads();
    const unsigned t = s_ticket;
    const unsigned b = a.b;
    unsigned long long* ts = (a.ts != nullptr && tid == 0) ? a.ts + (size_t)t * 8u : nullptr;
    if (ts != nullptr) { ts[0] = t_start; ts[1] = g4_now(); ts[5] = 0ull; }

    if (t < b) {
        // ------------------------------------------------ RMS of row z (rms_norm_no_weight_batched)
        const unsigned z = t;
        const float* xb = a.x + (size_t)z * G4_K;
        float xv[G4_PT];
#pragma unroll
        for (unsigned j = 0; j < G4_PT; j++) xv[j] = xb[tid + 256u * j];
        double partial = 0.0;
#pragma unroll
        for (unsigned j = 0; j < G4_PT; j++) {
            const double v = (double)xv[j];
            partial = __builtin_fma(v, v, partial);
        }
        sq[tid] = (float)partial;
        __syncthreads();
        for (unsigned s = 128; s > 0; s >>= 1) {
            if (tid < s) sq[tid] += sq[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            const float mean_sq = sq[0] / (float)G4_K;
            a.inv_scr[z] = rsqrtf(mean_sq + a.rms_eps);
            __threadfence();
            __hip_atomic_store(a.sync + S_READY(z), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        G4_TS(2);
    } else if (t < b + a.nq) {
        // ------------------------------------------------ QUANT (q8_0_quantize_f32_wave's body)
        const unsigned blk = (t - b) * 8u + warp;
        if (blk < a.qblocks) {
            float v = a.xin[(size_t)blk * 32u + lane];
            float amax = fabsf(v);
            for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(G4_FULL_MASK, amax, off));
            float d = amax / 127.0f;
            float id = d != 0.0f ? 1.0f / d : 0.0f;
            if (lane == 0) a.xscale[blk] = d;
            int q = (int)lrintf(v * id);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            a.xq[(size_t)blk * 32u + lane] = (int8_t)q;
        }
        G4_TS(2);
    } else {
        // ------------------------------------------------ MIX row role r of row z (mode 1)
        const unsigned qq = t - b - a.nq;
        const unsigned z = qq / G4_MIX;
        const unsigned role = qq - z * G4_MIX;
        // the weight row (packed f16 pairs) and the x row do not depend on the RMS: both in flight
        // before the wait
        const __half* wr = a.weight + (size_t)role * G4_K;
        const float* xb = a.x + (size_t)z * G4_K;
        __half2 wp[G4_PT / 2];
#pragma unroll
        for (unsigned j = 0; j < G4_PT / 2; j++) wp[j] = __halves2half2(wr[tid + 256u * (2u * j)], wr[tid + 256u * (2u * j + 1u)]);
        float xv[G4_PT];
#pragma unroll
        for (unsigned j = 0; j < G4_PT; j++) xv[j] = xb[tid + 256u * j];
        if (warp == 0) g4_spoll(a.sync + S_READY(z), a.sync + S_ERR);
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        G4_TS(2);
        const float sc = __hip_atomic_load(a.inv_scr + z, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        float acc = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < G4_PT; j++) {
            const float f = xv[j] * sc;   // the value rms_nw stored in `flat`
            const float w = (j & 1u) ? __high2float(wp[j / 2u]) : __low2float(wp[j / 2u]);
            acc = __builtin_fmaf(w, f, acc);
        }
        for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset, G4_WARP);
        if (lane == 0) per_warp[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < 8u) ? per_warp[lane] : 0.0f;
            for (int offset = 4; offset > 0; offset >>= 1) v += __shfl_down(v, offset, G4_WARP);
            if (lane == 0) a.mix_out[(size_t)z * G4_MIX + role] = v;
        }
        // the LAST of the row's 24 mix WGs: the Sinkhorn split
        if (tid == 0) {
            __threadfence();
            const unsigned prev = atomicAdd(&a.counters[z], 1u);
            is_last = (prev == G4_MIX - 1u) ? 1u : 0u;
        }
        __syncthreads();
        if (is_last != 0u && tid < G4_WARP) {
            __threadfence();
            if (tid == 0) a.counters[z] = 0u;
            g4_sinkhorn_regs(tid, a.mix_out + (size_t)z * G4_MIX, a.split_out + (size_t)z * G4_MIX,
                             a.write_carry ? a.carry + (size_t)z * G4_MIX : nullptr,
                             a.scale, a.base, a.n_hc, a.iters, a.eps);
            if (ts != nullptr) ts[5] = g4_now();
        }
        G4_TS(3);
    }
    // ---- finish: the last WG resets the sync words (not S_ERR)
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned prev = atomicAdd(a.sync + S_FINISH, 1u);
        if (ts != nullptr) { ts[4] = g4_now(); ts[6] = t < b ? 0u : (t < b + a.nq ? 1u : 2u); }
        if (prev == b * (1u + G4_MIX) + a.nq - 1u) {
            __threadfence();
            a.sync[S_TICKET] = 0u;
            for (unsigned zz = 0; zz < b; ++zz) a.sync[S_READY(zz)] = 0u;
            a.sync[S_FINISH] = 0u;
            __threadfence();
        }
    }
}
