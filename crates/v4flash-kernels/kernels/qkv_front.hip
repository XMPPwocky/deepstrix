// qkv_front — the decode attention FRONT (fusion group G2, 2026-10-13, `V41_DGPU_QKV_FRONT`):
// the chain
//
//   q8_0_quantize_f32_wave(attn_input_norm)                       (Q8_0 of the 5120-wide input)
//   q8_0_gemv_bpack_tB<b>  q_a  [1280 x 5120] -> qr
//   rms_quant_q8_1280_batched(qr)  -> qr_normed, qr_xq, qr_xscale
//   q8_0_gemv_bpack_tB<b>  q_b  [32768 x 1280] -> q
//   rope_tail_batched_copy(q)      -> q_normed
//   q8_0_gemv_bpack_tB<b>  kv   [512 x 5120] -> kv_raw
//   kv_rms_rope_fp8(kv_raw)        -> kv_normed
//   kv_cache_append_batched(kv_normed) -> the window KV cache (f16)
//
// (8 launches) as TWO launches, every output BIT-IDENTICAL (tests/qkv_front_bitexact.rs):
//
//   qkv_front_a_tB<b>  grid G x 256: each work-group quantizes the [b, 5120] input into LDS (the
//       wave kernel's expressions, one lane per 32-block; a 32-block's amax is a max, so the lane
//       split does not matter), then streams its q_a rows, takes a ticket, and streams its kv rows
//       (one warp per row, two rows in flight per warp, the tB GEMV's per-(row, b) lane striding,
//       expression and warp-sum tree). The work-group with the LAST ticket (no work-group ever
//       waits for another, so dispatch order cannot deadlock it) then runs rms_quant_q8_1280's
//       body over all b rows (same double partials, same 256-wide tree per row -- its last 5
//       strides in one wave's registers, same adds -- same output expression and quantize) and
//       resets the ticket.
//   qkv_front_b_tB<b>  grid 1 + 4096 x 256: work-group 0 runs kv_rms_rope_fp8's body per row and
//       stores the f16 cache row itself (kv_cache_append's `(_Float16)` cast of the same f32);
//       work-groups 1.. are the q_b tB GEMV, and the 8 work-groups of each head's rope tail
//       (rows 448..511) rotate their pairs (rope_tail_batched_copy's expressions) through LDS
//       before the store into q_normed. `q` and `kv_normed` (hand-offs that only existed because
//       the chain was sequential) are never written.
//
// Bytes: 9.75 MB (q_a + kv) + 44.56 MB (q_b) of Q8_0 weights; everything else is L2-resident.
//
// Code size matters here: every instruction of a kernel that runs once per lane-layer comes from a
// cold instruction cache, so the per-row loops stay rolled (a fully unrolled b = 8 prologue + tail
// was ~10k instructions and cost ~40 us). Measurement-only builds: -DQF_DBG_NO_PRO / _NO_GEMV /
// _NO_TAIL drop one part of launch A (tests/bench_qkv_front_chain.rs BENCH_SECTION=variant).
// A ONE-launch variant (q_b streamed by launch A's work-groups after an in-kernel publish) was
// built, bit-identical, and measured SLOWER (+15..+196 us; commit 271f512e, reverted): see
// src/qkv_front.rs.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "fp8_e4m3fn_common.inc"

#define QF_MASK 0xFFFFFFFFFFFFFFFFull
#define QF_K 5120u
#define QF_BLOCKS 160u          // QF_K / 32
#define QF_QA 1280u
#define QF_KV 512u
#define QF_PAIRS 896u           // (QF_QA + QF_KV) / 2
#define QF_QB_ROWS 32768u
#define QF_QB_K 1280u
#define QF_QB_BLOCKS 40u

__device__ __forceinline__ int32_t qf_dp4a(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// q8_0_matvec.hip dot8_regs, verbatim
__device__ __forceinline__ int32_t qf_dot8(const uint4& a0, const uint4& a1, const uint4& b0, const uint4& b1) {
    int32_t dot = 0;
    dot = qf_dp4a((int32_t)a0.x, (int32_t)b0.x, dot);
    dot = qf_dp4a((int32_t)a0.y, (int32_t)b0.y, dot);
    dot = qf_dp4a((int32_t)a0.z, (int32_t)b0.z, dot);
    dot = qf_dp4a((int32_t)a0.w, (int32_t)b0.w, dot);
    dot = qf_dp4a((int32_t)a1.x, (int32_t)b1.x, dot);
    dot = qf_dp4a((int32_t)a1.y, (int32_t)b1.y, dot);
    dot = qf_dp4a((int32_t)a1.z, (int32_t)b1.z, dot);
    dot = qf_dp4a((int32_t)a1.w, (int32_t)b1.w, dot);
    return dot;
}

// q8_0_matvec.hip warp_sum_f32, verbatim
__device__ __forceinline__ float qf_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset);
    return v;
}

// rope_tail.hip rf_theta, verbatim
__device__ __forceinline__ float qf_theta(int pos, unsigned lane, float theta_scale, float freq_scale,
                                          float ext_factor, float corr_low, float corr_high) {
    const float theta_extrap = (float)pos * powf(theta_scale, (float)lane);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float y = ((float)lane - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        const float ramp_mix = ramp * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    }
    return theta;
}

// rope_tail.hip rf_rot_pair, verbatim
__device__ __forceinline__ float qf_rot_pair(float v, unsigned e, int pos, unsigned l, float sin_sign,
                                             float theta_scale, float freq_scale, float ext_factor,
                                             float mscale_eff, float corr_low, float corr_high) {
    const float partner = __shfl_xor(v, 1);
    const float theta = qf_theta(pos, l, theta_scale, freq_scale, ext_factor, corr_low, corr_high);
    const float c = cosf(theta) * mscale_eff;
    const float s = sin_sign * sinf(theta) * mscale_eff;
    const float x0 = (e & 1u) ? partner : v;
    const float x1 = (e & 1u) ? v : partner;
    return (e & 1u) ? (x0 * s + x1 * c) : (x0 * c - x1 * s);
}

// ---------------------------------------------------------------------------------------------
// Launch A
// ---------------------------------------------------------------------------------------------
// LDS: xq as [b][half h][block][16 B] (a lane's two 16-B reads of block bl are contiguous across
// the warp: conflict-free ds_read_b128), then xscale [b][block].
// One weight row of q_a or kv in registers: lane's 5 blocks (bl = lane + 32 j).
struct QfRow {
    uint4 a0[5], a1[5];
    float ws[5];
};

__device__ __forceinline__ const unsigned char* qf_row_ptr(const unsigned char* w_qa, const unsigned char* w_kv,
                                                           unsigned r) {
    return r < QF_QA ? w_qa + (size_t)r * QF_BLOCKS * 34u : w_kv + (size_t)(r - QF_QA) * QF_BLOCKS * 34u;
}

__device__ __forceinline__ void qf_load_row(QfRow& w, const unsigned char* wr, unsigned lane) {
    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)QF_BLOCKS * 2u);
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const unsigned bl = lane + 32u * j;
        __builtin_memcpy(&w.a0[j], quants + (size_t)bl * 32u, 16);
        __builtin_memcpy(&w.a1[j], quants + (size_t)bl * 32u + 16, 16);
        w.ws[j] = __half2float(scales[bl]);
    }
}

// The tB GEMV's per-(row, b) accumulation (acc += ws * xs * dot over bl = lane, lane + 32, ...)
// and warp-sum, activations from LDS. Rolled over the 5 blocks with the weights rotating through
// slot 0: unrolled, the compiler hoists / interleaves all 5 x B activation reads and spills.
template <int B>
__device__ __forceinline__ void qf_row_dot(QfRow& w, const unsigned char* xq_l, const float* xs_l, unsigned lane,
                                           float* __restrict__ out, unsigned m, unsigned mrow) {
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;
#pragma unroll 1
    for (int j = 0; j < 5; ++j) {
        const unsigned bl = lane + 32u * j;
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const uint4 bv0 = *reinterpret_cast<const uint4*>(xq_l + ((size_t)(b * 2 + 0) * QF_BLOCKS + bl) * 16u);
            const uint4 bv1 = *reinterpret_cast<const uint4*>(xq_l + ((size_t)(b * 2 + 1) * QF_BLOCKS + bl) * 16u);
            const float xs = xs_l[b * QF_BLOCKS + bl];
            int32_t dot = qf_dot8(w.a0[0], w.a1[0], bv0, bv1);
            acc[b] += w.ws[0] * xs * (float)dot;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            w.a0[k] = w.a0[k + 1];
            w.a1[k] = w.a1[k + 1];
            w.ws[k] = w.ws[k + 1];
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        float a = qf_warp_sum(acc[b]);
        if (lane == 0) out[(size_t)b * m + mrow] = a;
    }
}

template <int B>
__device__ __forceinline__ void qf_front_a(
        float* __restrict__ qr, float* __restrict__ kv_raw,
        const unsigned char* __restrict__ w_qa, const unsigned char* __restrict__ w_kv,
        const float* __restrict__ x,
        float* __restrict__ qr_normed, signed char* __restrict__ qr_xq, float* __restrict__ qr_xscale,
        const float* __restrict__ q_a_norm, float eps, unsigned int* __restrict__ ticket) {
    extern __shared__ __attribute__((aligned(16))) unsigned char qf_lds[];
    unsigned char* xq_l = qf_lds;                                          // B * 5120 bytes
    float* xs_l = reinterpret_cast<float*>(qf_lds + (size_t)B * QF_K);    // B * 160 floats
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    constexpr unsigned R = QF_QA + QF_KV;
    const unsigned stride = gridDim.x * 8u;
    const unsigned p0 = blockIdx.x * 8u + warp;

    // ---- 1 + 2. Q8_0 of x into LDS as [b][half][block][16 B] (+ xscale [b][block]): ONE lane per
    // 32-block (8 float4 loads, amax by v_max over its 32 values: a max, so the order does not
    // change it), then q8_0_quantize_f32_wave's d = amax / 127, id, lrintf + clamp. No cross-lane
    // traffic: a butterfly per block (5 dependent shuffles) serialized ~6 us into the b = 8
    // prologue. The first pass's x loads go out, THEN the first weight row (DRAM latency under the
    // quantize), then the passes: loads retire in order, so x loads issued behind the weights
    // would wait for DRAM.
    QfRow wA, wB;
    auto quant_block = [&](unsigned gb, const float4 (&v)[8]) {
        float amax = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k)
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[k].x), fabsf(v[k].y)), fmaxf(fabsf(v[k].z), fabsf(v[k].w))));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        const unsigned rb = gb / QF_BLOCKS, bl = gb - rb * QF_BLOCKS;
        xs_l[rb * QF_BLOCKS + bl] = d;
        uint32_t pk[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int q0 = (int)lrintf(v[k].x * id); if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
            int q1 = (int)lrintf(v[k].y * id); if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
            int q2 = (int)lrintf(v[k].z * id); if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
            int q3 = (int)lrintf(v[k].w * id); if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;
            pk[k] = (uint32_t)(uint8_t)(int8_t)q0 | ((uint32_t)(uint8_t)(int8_t)q1 << 8) |
                    ((uint32_t)(uint8_t)(int8_t)q2 << 16) | ((uint32_t)(uint8_t)(int8_t)q3 << 24);
        }
        *reinterpret_cast<uint4*>(xq_l + ((size_t)(rb * 2u + 0u) * QF_BLOCKS + bl) * 16u) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(xq_l + ((size_t)(rb * 2u + 1u) * QF_BLOCKS + bl) * 16u) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
    };
#ifndef QF_DBG_NO_PRO
    float4 v0[8];
    const bool has0 = tid < (unsigned)B * QF_BLOCKS;
    if (has0) {
        const float4* src = reinterpret_cast<const float4*>(x + (size_t)tid * 32u);
#pragma unroll
        for (int k = 0; k < 8; ++k) v0[k] = src[k];
    }
#endif
#ifndef QF_DBG_NO_GEMV
    if (p0 < R) qf_load_row(wA, qf_row_ptr(w_qa, w_kv, p0), lane);
#endif
#ifndef QF_DBG_NO_PRO
    if (has0) quant_block(tid, v0);
#pragma unroll 1
    for (unsigned gb = tid + 256u; gb < (unsigned)B * QF_BLOCKS; gb += 256u) {
        const float4* src = reinterpret_cast<const float4*>(x + (size_t)gb * 32u);
        float4 v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = src[k];
        quant_block(gb, v);
    }
#endif
    __syncthreads();

    // ---- 3. GEMV, one warp per row, rows p0, p0 + stride, ...: every work-group's q_a rows first,
    // then a q ticket, then its kv rows. The last q ticket's work-group runs the q_a tail AFTER its
    // own kv rows (the GEMV's LDS is live until then), i.e. while the other work-groups still
    // stream kv (~29% of the bytes): the tail is off the launch's critical path. Two weight
    // buffers so the next row's loads are in flight while this one is reduced.
#ifndef QF_DBG_NO_GEMV
#pragma unroll 1
    for (unsigned p = p0; p < QF_QA; p += 2u * stride) {
        const unsigned pB = p + stride, pA2 = p + 2u * stride;
        if (pB < QF_QA) qf_load_row(wB, qf_row_ptr(w_qa, w_kv, pB), lane);
        qf_row_dot<B>(wA, xq_l, xs_l, lane, qr, QF_QA, p);
        if (pA2 < QF_QA) qf_load_row(wA, qf_row_ptr(w_qa, w_kv, pA2), lane);
        if (pB < QF_QA) qf_row_dot<B>(wB, xq_l, xs_l, lane, qr, QF_QA, pB);
    }
    // this warp's first kv row (p0 + k stride >= 1280): its loads go out before the ticket
    const unsigned kq = p0 >= QF_QA ? 0u : (QF_QA - p0 + stride - 1u) / stride;
    const unsigned pk = p0 + kq * stride;
    if (pk < R) qf_load_row(wA, qf_row_ptr(w_qa, w_kv, pk), lane);
#endif

    // ---- 4. q ticket: the last work-group to finish its q_a rows owns the q_a tail.
    __shared__ unsigned qf_last;
    __threadfence();
    __syncthreads();
    if (tid == 0) qf_last = (atomicAdd(ticket, 1u) == gridDim.x - 1u) ? 1u : 0u;
    __syncthreads();
    const bool last = qf_last != 0u;

#ifndef QF_DBG_NO_GEMV
#pragma unroll 1
    for (unsigned p = pk; p < R; p += 2u * stride) {
        const unsigned pB = p + stride, pA2 = p + 2u * stride;
        if (pB < R) qf_load_row(wB, qf_row_ptr(w_qa, w_kv, pB), lane);
        qf_row_dot<B>(wA, xq_l, xs_l, lane, kv_raw, QF_KV, p - QF_QA);
        if (pA2 < R) qf_load_row(wA, qf_row_ptr(w_qa, w_kv, pA2), lane);
        if (pB < R) qf_row_dot<B>(wB, xq_l, xs_l, lane, kv_raw, QF_KV, pB - QF_QA);
    }
#endif
    if (!last) return;
    __syncthreads();   // the GEMV's LDS is dead in this work-group
    __threadfence();
#ifdef QF_DBG_NO_TAIL
    if (tid == 0) *ticket = 0u;
    return;
#endif

    // rms_quant_q8_1280_body for all B rows: per row the same loads, double partials, 256-wide LDS
    // tree (stride 128 .. 1), mean / rsqrtf and output expression; the rows share the tree's 8
    // barriers. qr and q_a_norm are staged in LDS (the GEMV's LDS is dead) and the output +
    // quantize runs ONE lane per 32-block (amax by v_max: a max), not a 5-shuffle butterfly per
    // block in one work-group (b x 5 dependent chains).
    // vals / wl hold 32-blocks at a 36-float pitch: the per-block lane reads 8 x 16 B at a block
    // stride, and a 32-float pitch put 16 lanes on the same banks.
    float* vals = reinterpret_cast<float*>(qf_lds);   // B * 40 blocks * 36
    float* sdata = vals + B * 40 * 36;                 // B * 256
    float* wl = sdata + B * 256;                       // 40 * 36
#pragma unroll
    for (unsigned j = 0; j < 5; ++j) {
        const unsigned e = tid + 256u * j;
        wl[(e >> 5) * 36u + (e & 31u)] = q_a_norm[e];
    }
    {
        float xv[B][5];   // every load in flight at once (one L2 round trip, not B)
#pragma unroll
        for (unsigned b = 0; b < (unsigned)B; ++b)
#pragma unroll
            for (unsigned j = 0; j < 5; ++j) xv[b][j] = qr[(size_t)b * 1280u + tid + 256u * j];
#pragma unroll
        for (unsigned b = 0; b < (unsigned)B; ++b) {
            double partial = 0.0;
#pragma unroll
            for (unsigned j = 0; j < 5; ++j) {
                const float v = xv[b][j];
                partial += (double)v * (double)v;
            }
            sdata[b * 256u + tid] = (float)partial;
#pragma unroll
            for (unsigned j = 0; j < 5; ++j) {
                const unsigned e = tid + 256u * j;
                vals[(b * 40u + (e >> 5)) * 36u + (e & 31u)] = xv[b][j];
            }
        }
    }
    __syncthreads();
    for (unsigned int stride = 128u; stride >= 32u; stride >>= 1) {
        if (tid < stride) {
            float t[B];   // the B rows' adds are independent: issue every LDS read first
#pragma unroll
            for (unsigned b = 0; b < (unsigned)B; ++b) t[b] = sdata[b * 256u + tid] + sdata[b * 256u + tid + stride];
#pragma unroll
            for (unsigned b = 0; b < (unsigned)B; ++b) sdata[b * 256u + tid] = t[b];
        }
        __syncthreads();
    }
    // strides 16 .. 1 in wave 0's registers: lane t adds lane t + stride's value -- the tree's
    // `sdata[t] += sdata[t + stride]` for t < stride, the same adds in the same order (lanes
    // >= stride compute values nothing reads), without 5 barriers and LDS round trips
    if (tid < 32u) {
        float t[B];
#pragma unroll
        for (unsigned b = 0; b < (unsigned)B; ++b) t[b] = sdata[b * 256u + tid];
#pragma unroll
        for (unsigned int stride = 16u; stride > 0; stride >>= 1)
#pragma unroll
            for (unsigned b = 0; b < (unsigned)B; ++b) t[b] += __shfl_down(t[b], stride);
        if (tid == 0) {
#pragma unroll
            for (unsigned b = 0; b < (unsigned)B; ++b) sdata[b * 256u] = t[b];
        }
    }
    __syncthreads();
#pragma unroll 1
    for (unsigned g = tid; g < (unsigned)B * 40u; g += 256u) {
        const unsigned b = g / 40u, k = g - b * 40u;
        const float mean_sq = sdata[b * 256u] / (float)1280u;
        const float scale = rsqrtf(mean_sq + eps);
        const float4* xr = reinterpret_cast<const float4*>(vals + (b * 40u + k) * 36u);
        const float4* wr = reinterpret_cast<const float4*>(wl + k * 36u);
        float4* ob = reinterpret_cast<float4*>(qr_normed + (size_t)b * 1280u + 32u * k);
        float4 v[8];
        float amax = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float4 xx = xr[e], ww = wr[e];
            v[e].x = xx.x * scale * ww.x;
            v[e].y = xx.y * scale * ww.y;
            v[e].z = xx.z * scale * ww.z;
            v[e].w = xx.w * scale * ww.w;
            ob[e] = v[e];
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[e].x), fabsf(v[e].y)), fmaxf(fabsf(v[e].z), fabsf(v[e].w))));
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        qr_xscale[(size_t)b * 40u + k] = d;
        uint32_t pk[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            int q0 = (int)lrintf(v[e].x * id); if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
            int q1 = (int)lrintf(v[e].y * id); if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
            int q2 = (int)lrintf(v[e].z * id); if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
            int q3 = (int)lrintf(v[e].w * id); if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;
            pk[e] = (uint32_t)(uint8_t)(int8_t)q0 | ((uint32_t)(uint8_t)(int8_t)q1 << 8) |
                    ((uint32_t)(uint8_t)(int8_t)q2 << 16) | ((uint32_t)(uint8_t)(int8_t)q3 << 24);
        }
        uint4* xo = reinterpret_cast<uint4*>(qr_xq + (size_t)b * 1280u + 32u * k);
        xo[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        xo[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
    }
    if (tid == 0) *ticket = 0u;   // every other work-group has taken its ticket
}

#define QF_FRONT_A(NAME, B)                                                                          \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                          \
            float* __restrict__ qr, float* __restrict__ kv_raw, const unsigned char* __restrict__ w_qa, \
            const unsigned char* __restrict__ w_kv, const float* __restrict__ x,                     \
            float* __restrict__ qr_normed, signed char* __restrict__ qr_xq,                          \
            float* __restrict__ qr_xscale, const float* __restrict__ q_a_norm, float eps,            \
            unsigned int* __restrict__ ticket) {                                                     \
        qf_front_a<B>(qr, kv_raw, w_qa, w_kv, x, qr_normed, qr_xq, qr_xscale, q_a_norm, eps, ticket); \
    }
QF_FRONT_A(qkv_front_a_tB1, 1)
QF_FRONT_A(qkv_front_a_tB2, 2)
QF_FRONT_A(qkv_front_a_tB3, 3)
QF_FRONT_A(qkv_front_a_tB4, 4)
QF_FRONT_A(qkv_front_a_tB5, 5)
QF_FRONT_A(qkv_front_a_tB6, 6)
QF_FRONT_A(qkv_front_a_tB7, 7)
QF_FRONT_A(qkv_front_a_tB8, 8)

// ---------------------------------------------------------------------------------------------
// Launch B
// ---------------------------------------------------------------------------------------------
template <int B>
__device__ __forceinline__ void qf_front_b(
        float* __restrict__ q_normed, const unsigned char* __restrict__ w_qb,
        const int8_t* __restrict__ xq, const float* __restrict__ xscale, const int* __restrict__ pos_per_b,
        float theta_scale, float freq_scale, float ext_factor, float mscale_eff, float corr_low,
        float corr_high, int inverse,
        unsigned short* __restrict__ kv_cache, const float* __restrict__ kv_raw,
        const float* __restrict__ kv_norm, float eps, const int* __restrict__ slot_per, unsigned slot_base) {
    __shared__ float sdata[256];
    const unsigned tid = threadIdx.x;
    if (blockIdx.x == 0) {
        // kv_rms_rope_fp8_body per row (verbatim but for the row loop and the store), then
        // kv_cache_append_batched's `cache[slot * 512 + d] = (_Float16)kv_new[d]`.
        _Float16* cache = reinterpret_cast<_Float16*>(kv_cache);
        for (unsigned b = 0; b < (unsigned)B; ++b) {
            const float* xb = kv_raw + (size_t)b * 512u;
            float xv[2], wv[2], v[2];
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) xv[j] = xb[tid + 256u * j];
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) wv[j] = kv_norm[tid + 256u * j];
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
            __syncthreads();   // sdata[0] read by all before the next row's partials land
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) v[j] = xv[j] * scale * wv[j];
            if (tid >= 192u) {
                const unsigned e = tid + 256u;
                v[1] = qf_rot_pair(v[1], e, pos_per_b[b], (e - 448u) >> 1, 1.0f, theta_scale, freq_scale, ext_factor,
                                   mscale_eff, corr_low, corr_high);
            }
            const unsigned slot = slot_per ? (unsigned)slot_per[b] : slot_base + b;
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                float a = fabsf(v[j]);
                a = fmaxf(a, __shfl_xor(a, 1));
                a = fmaxf(a, __shfl_xor(a, 2));
                a = fmaxf(a, __shfl_xor(a, 4));
                a = fmaxf(a, __shfl_xor(a, 8));
                a = fmaxf(a, __shfl_xor(a, 16));
                const float amax = fmaxf(a, 1.0e-4f);
                const float t = amax / 448.0f;
                int ex;
                const float m = frexpf(t, &ex);
                const int cexp = (m == 0.5f) ? ex - 1 : ex;
                const float sc = ldexpf(1.0f, cexp);
                float q = v[j] / sc;
                q = fminf(fmaxf(q, -448.0f), 448.0f);
                float kvn = dsv4_e4m3fn_dequant(q) * sc;
                // materialize the f32 the chain stores: left to itself the compiler folds the
                // multiply + f16 cast into v_fma_mixlo_f16(sc, +-T, +0), and -0 + +0 = +0 drops
                // the sign of a zero code (the chain stores -0.0f, the append casts it to 0x8000)
                asm volatile("" : "+v"(kvn));
                cache[(unsigned long long)slot * 512u + tid + 256u * j] = (_Float16)kvn;
            }
        }
        return;
    }

    // q_b: q8_0_gemv_bpack_tb_body<B> (blocks = 40), one warp per row, 8 rows per work-group.
    const unsigned g = blockIdx.x - 1u;
    const unsigned row = g * 8u + (tid >> 5u);
    const unsigned lane = tid & 31u;
    const unsigned char* wr = w_qb + (size_t)row * QF_QB_BLOCKS * 34u;
    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)QF_QB_BLOCKS * 2u);
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;
    for (uint32_t bl = lane; bl < QF_QB_BLOCKS; bl += 32u) {
        uint4 av0, av1;
        __builtin_memcpy(&av0, quants + (size_t)bl * 32u, 16);
        __builtin_memcpy(&av1, quants + (size_t)bl * 32u + 16, 16);
        const float ws = __half2float(scales[bl]);
        uint4 bv0[B], bv1[B];
        float xs[B];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const int8_t* xqb = xq + (size_t)b * QF_QB_K + (size_t)bl * 32u;
            __builtin_memcpy(&bv0[b], xqb, 16);
            __builtin_memcpy(&bv1[b], xqb + 16, 16);
            xs[b] = xscale[(size_t)b * QF_QB_BLOCKS + bl];
        }
#pragma unroll
        for (int b = 0; b < B; ++b) {
            int32_t dot = qf_dot8(av0, av1, bv0[b], bv1[b]);
            acc[b] += ws * xs[b] * (float)dot;
        }
    }
    const unsigned hrow = (g * 8u) & 511u;          // the work-group's first row within its head
    if (hrow < 448u) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float a = qf_warp_sum(acc[b]);
            if (lane == 0) q_normed[(size_t)b * QF_QB_ROWS + row] = a;
        }
        return;
    }
    // rope tail rows: rope_tail_batched_copy's pair rotation, pair (2pp, 2pp + 1) = warps (2pp, 2pp + 1)
#pragma unroll
    for (int b = 0; b < B; ++b) {
        float a = qf_warp_sum(acc[b]);
        if (lane == 0) sdata[(tid >> 5u) * B + b] = a;
    }
    __syncthreads();
    if (tid < 4u * (unsigned)B) {
        const unsigned pp = tid & 3u, b = tid >> 2;
        const unsigned l = ((hrow - 448u) >> 1) + pp;   // pair index within the head's tail
        const float theta = qf_theta(pos_per_b[b], l, theta_scale, freq_scale, ext_factor, corr_low, corr_high);
        const float sin_sign = (inverse != 0) ? -1.0f : 1.0f;
        const float c = cosf(theta) * mscale_eff;
        const float s = sin_sign * sinf(theta) * mscale_eff;
        const float x0 = sdata[(2u * pp + 0u) * B + b];
        const float x1 = sdata[(2u * pp + 1u) * B + b];
        const size_t o = (size_t)b * QF_QB_ROWS + g * 8u + 2u * pp;
        q_normed[o + 0] = x0 * c - x1 * s;
        q_normed[o + 1] = x0 * s + x1 * c;
    }
}

#define QF_FRONT_B(NAME, B)                                                                          \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                          \
            float* __restrict__ q_normed, const unsigned char* __restrict__ w_qb,                    \
            const int8_t* __restrict__ xq, const float* __restrict__ xscale,                         \
            const int* __restrict__ pos_per_b, float theta_scale, float freq_scale, float ext_factor, \
            float mscale_eff, float corr_low, float corr_high, int inverse,                          \
            unsigned short* __restrict__ kv_cache, const float* __restrict__ kv_raw,                 \
            const float* __restrict__ kv_norm, float eps, const int* __restrict__ slot_per,          \
            unsigned slot_base) {                                                                    \
        qf_front_b<B>(q_normed, w_qb, xq, xscale, pos_per_b, theta_scale, freq_scale, ext_factor,    \
                      mscale_eff, corr_low, corr_high, inverse, kv_cache, kv_raw, kv_norm, eps,      \
                      slot_per, slot_base);                                                          \
    }
QF_FRONT_B(qkv_front_b_tB1, 1)
QF_FRONT_B(qkv_front_b_tB2, 2)
QF_FRONT_B(qkv_front_b_tB3, 3)
QF_FRONT_B(qkv_front_b_tB4, 4)
QF_FRONT_B(qkv_front_b_tB5, 5)
QF_FRONT_B(qkv_front_b_tB6, 6)
QF_FRONT_B(qkv_front_b_tB7, 7)
QF_FRONT_B(qkv_front_b_tB8, 8)
