// qe_attn_g4 -- causal GQA-4 / head_dim-128 prefill attention for the Qwen3-Embedding
// forward on gfx1201 (RX 9070 XT). docs/v41/EMBED_PHASE_DESIGN.md §19.
//
// ROOFLINE (Qwen3-Embedding-4B: 32 query heads over 8 KV heads, d 128, f16 in, f32 acc)
//   FLOPs: causal QK + PV = 512 FLOP per (query head, query, key <= query) pair
//     = 8192 * T^2 per layer at T tokens; one 16,384-token input: 2.20e12 per layer,
//     7.92e13 per 36-layer forward. One 1024-row sub-batch at depth 16K (queries
//     15360..16383): 1.63e7 pairs x 32 heads x 512 = 2.66e11 FLOP.
//   Bytes (that launch): K+V tiles 2 x 8 heads x 16384 x 128 x 2 B = 67 MB per sweep,
//     Q 8.4 MB, out 8.4 MB (f16, written straight into x16): ~84 MB (one sweep), ~151 MB
//     with the second LPT round re-reading K/V. 2.66e11 / 84e6 = 3170 FLOP/B against a
//     ridge of 194.6e12 / 640e9 = 304 FLOP/B: COMPUTE-bound by 10x (DRAM <= ~10 %).
//   Peak: 194.6 TFLOPS dense f16 = 512 FLOP/clk/SIMD32 x 128 SIMD32 x 2.97 GHz
//     -> 16 clk per v_wmma_f32_16x16x16_f16. At peak the 16K launch is 1.37 ms and the
//     36-layer 16K forward's attention 0.41 s.
//   THE BOUND: the issue port of each SIMD. WMMA is synchronous and issues on the
//     VALU port (RDNA3 ISA 7.9; in-repo RDNA3.5 measurement: WMMA and VALU costs are
//     exactly additive), so the exact f32 online softmax competes with the matrix work:
//     per wave per 32-key tile 32 WMMA x 16 clk = 512 clk vs (compiled steady path,
//     scripts/qe_attn_g4_g0.sh) 66 non-trans VALU + 16 v_exp_f32 issue slots ->
//     512 / 594 = 86.2 % issue ceiling (79.8 % if v_exp_f32 issues at quarter rate),
//     x the sustained WMMA-only rate (84 % measured on RDNA3.5; W0a re-measures
//     gfx1201) -> ~72 %. Losses below that: exact rescale ~1.5 %, exposed LDS latency
//     + barriers at 1 WG/CU 3-6 %, prologue/epilogue + LPT tails + 576 launches per
//     16K forward 2-3 %, mask tiles < 0.5 %.
//   EXPECTED: ~55 % of peak (45-65 %): ~2,490 us (2,100-3,040) for the depth-16K
//     1024-row launch (2.66e11 FLOP at 107 TFLOPS; the floor at 100 % is 1,368 us),
//     ~0.74 s (0.63-0.90) for the 16K forward's attention (fa2: 15.8 s).
//     If W0b shows VALU co-issuing under another wave's WMMA: ~65-70 %; if v_exp_f32
//     serializes at quarter rate: ~50 %.
//   WHY THE REST IS UNREACHABLE: the softmax work per score (one fma, one exp2, one
//     add, one RNE convert, half a pack, half a max) must issue on the same SIMD port
//     as the synchronous WMMA; RDNA4 has no async matrix pipe and no VMEM->LDS DMA
//     (global_load_lds_dwordx4 is rejected for gfx1201), so staging costs issue slots
//     too. Removing that work needs fp8 WMMA, f16 accumulation, f16/approximate exp
//     or RTZ packs -- all excluded by the fidelity bar (f16 in, f32 accumulate, exact
//     online softmax, P rounded once RNE).
//
// CELLS (one source; compile-time variants, never independent kernels):
//   default        qe_attn_g4_n16: 16 waves (512 threads), 64 query tokens x 4 heads,
//                  3-stage LDS ring (48 KB), split barrier signalled BEFORE compute
//                  (one tile of slack), 1 WG per CU; built with -mcumode (build.rs).
//   QA_CELL_NW8    qe_attn_g4_n8: 8 waves, 32 tokens, 2 stages (32 KB), signal after
//                  compute, 2 WGs per CU (qwen3_attn_g4n8.hip, -mcumode).
//   WGP arm        qwen3_attn_g4wgp.hip: this source without -mcumode (symbol _n16).
//   QA_LAZY_TAU    0 (default) = exact skip of the rescale, bit-identical to always
//                  rescaling; 8.0f only as a compile-time A/B, never the default.
//
// Lane maps (gfx12 wave32 v_wmma_f32_16x16x16_f16, verified in-repo):
//   A: lane L elem e -> (m = L&15,       k = (L>>4)*8+e)
//   B: lane L elem e -> (k = (L>>4)*8+e, n = L&15)
//   C: lane L elem e -> (m = (L>>4)*8+e, n = L&15)
// Orientation: keys on M. S^T = K . Q^T (A = K, B = Q^T held in VGPRs), so the C of
// S^T is, element for element, the B operand P^T of O^T = V^T . P^T: P never leaves
// registers. O^T accumulates in f32 VGPRs.
// Layout (per kv head h, per 32-key tile T of ABSOLUTE phase rows; 512 chunks x 16 B):
//   kc chunk (ks*2+kb)*32 + L : half e = K[32T+16kb+(L&15)][16ks+8(L>>4)+e]   ks 0..7, kb 0..1
//   vc chunk (kb*8+dt)*32 + L : half e = V[32T+16kb+8(L>>4)+e][16dt+(L&15)]   kb 0..1, dt 0..7
//   tile offset = ((size_t)h * n_tiles + T) * 512 chunks in each of kc, vc.
// Every operand fragment is 512 contiguous bytes: one lane-linear ds_load_b128 at an
// immediate offset, conflict-free by construction (an 8-lane pass covers 128 B = 32 banks).
#include <hip/hip_runtime.h>
#include "qwen3_embed_f16.inc"

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

#ifdef QA_CELL_NW8
#define QA_NW     8u
#define QA_STAGES 2u
#define QA_NAME   qe_attn_g4_n8
#else
#define QA_NW     16u
#define QA_STAGES 3u
#define QA_NAME   qe_attn_g4_n16
#endif
#ifndef QA_LAZY_TAU
#define QA_LAZY_TAU 0.0f          // 0 = exact skip (bit-identical to always-rescale)
#endif
#define QA_NT   (QA_NW * 32u)     // threads per WG
#define QA_BR   (QA_NW * 4u)      // query tokens per WG (each wave: 4 tokens x 4 heads)
#define QA_TCH  512u              // 16-B chunks per K (or V^T) tile
#define QA_CPT  (QA_TCH / QA_NT)  // chunks per thread per tile per operand: 1 (NW16) or 2 (NW8)

// LDS-only fences: the release waits for this wave's LDS ops (s_wait_dscnt 0) and
// nothing else, so the t+2 global loads stay in flight across the signal.
#define QA_SIGNAL() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); \
                         __builtin_amdgcn_s_barrier_signal(-1); } while (0)
#define QA_WAIT()   do { __builtin_amdgcn_s_barrier_wait(-1); \
                         __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); } while (0)

// The score max. fmaxf (llvm.maxnum) and even fmaximum_num make the compiler
// canonicalize every WMMA result first (one max(x, x) per input: ~7 extra issue slots
// per tile), because it cannot know a WMMA output is never a signalling NaN. The
// operands here are WMMA scores, -inf (masked) or a permlane copy of one: never NaN
// (edge tiles zero the operands), so v_max3_num_f32 / v_max_num_f32 on them are
// exactly max(). QA_MAX_PORTABLE selects the builtin form for an A/B.
#ifdef QA_MAX_PORTABLE
__device__ __forceinline__ float qa_max(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ float qa_max3(float a, float b, float c) { return fmaxf(fmaxf(a, b), c); }
#else
__device__ __forceinline__ float qa_max(float a, float b) {
    float r;
    asm("v_max_num_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
__device__ __forceinline__ float qa_max3(float a, float b, float c) {
    float r;
    asm("v_max3_num_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}
#endif

__device__ __forceinline__ float qa_xchg16(float v) {       // lane L <-> L^16
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v),
                                                       0x76543210u, 0xfedcba98u, false, false));
}

extern "C" __global__ __launch_bounds__(QA_NT) __attribute__((amdgpu_waves_per_eu(8, 8)))
void QA_NAME(
        _Float16       * __restrict__ out16,   // x16 at query row 0; pitch out_pitch halfs; head h at h*128
        unsigned int     out_pitch,
        float          * __restrict__ out32,   // tests only: [batch][n_head][128] f32; null in production
        const _Float16 * __restrict__ q,       // q16 at query row 0: [batch][n_head][128]
        const u32x4    * __restrict__ kc,      // whole buffer, fragment-native K tiles
        const u32x4    * __restrict__ vc,      // whole buffer, fragment-native V^T tiles
        unsigned int n_head,                   // == 4 * gridDim.x
        unsigned int kv_shift,                 // log2(kv_group / 4): kv head = blockIdx.x >> kv_shift
        unsigned int n_tiles,                  // tiles per kv head (ceil(phase_tokens / 32), the SIZING)
        unsigned int batch,                    // query rows in this launch (>= 1)
        unsigned int q_abs0,                   // phase row of query row 0
        unsigned int seg0,                     // phase row of the segment's first key (<= q_abs0)
        float c) {                             // RN(scale * log2(e)), > 0
#if defined(__gfx1200__) || defined(__gfx1201__)
    // Invariant: the launch matches this cell (a -D leaking from DEEPSTRIX_KERNEL_CFLAGS
    // cannot silently mismatch the Rust launch geometry).
    if (blockDim.x != QA_NT || 4u * gridDim.x != n_head) __builtin_trap();
    const unsigned int tid = threadIdx.x, lane = tid & 31u;
    // Wave-uniform by construction (wave32): in an SGPR, so every per-wave test
    // below (active, wv_t1, the mask tile) is a scalar branch, not an exec mask.
    const unsigned int w = __builtin_amdgcn_readfirstlane(tid >> 5);
    const unsigned int n = lane & 15u, hi = lane >> 4;
    const unsigned int hq = blockIdx.x;                           // fastest axis: query-head quad
    const unsigned int kvh = hq >> kv_shift;
    const unsigned int qb  = gridDim.y - 1u - blockIdx.y;         // LPT: longest q-tiles first
    const unsigned int g = n & 3u, tok = n >> 2;                  // column n = (token, head)
    const unsigned int rw0 = qb * QA_BR + w * 4u;                 // wave's first query row
    const unsigned int r_raw = rw0 + tok;
    const bool rv = r_raw < batch;
    const unsigned int r = rv ? r_raw : batch - 1u;               // padding lanes duplicate the last row
    const unsigned int h = hq * 4u + g;
    const unsigned int qa = q_abs0 + r;
    const unsigned int wg_last = min(qb * QA_BR + QA_BR, batch) - 1u;
    const unsigned int wg_qmax = q_abs0 + wg_last;
    const bool wave_active = rw0 < batch;
    const unsigned int wv_qmin = q_abs0 + rw0;
    const unsigned int wv_qmax = q_abs0 + min(rw0 + 3u, batch - 1u);
    const unsigned int t0 = seg0 >> 5, t1 = wg_qmax >> 5;        // WG-uniform inclusive tile range
    const unsigned int wv_t1 = wv_qmax >> 5;

    __shared__ u32x4 lds[QA_STAGES * 2u * QA_TCH];                // stage = [K 512 chunks | V^T 512 chunks]

    // Q^T as the B operand: lane L holds q[r][h][16ks + 8hi .. +8] for each ks.
    half8 qf[8];
    {
        const _Float16 *qp = q + ((size_t)r * n_head + h) * 128u + hi * 8u;
#pragma unroll
        for (unsigned int ks = 0; ks < 8u; ++ks) qf[ks] = *(const half8 *)(qp + ks * 16u);
    }
    float8 o[8];
#pragma unroll
    for (unsigned int dt = 0; dt < 8u; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, la = 0.0f, lb = 0.0f;

    // Per-head bases in SGPRs; a tile's chunk = base + one 32-bit byte offset shared by
    // K and V (saddr + voffset addressing: one VALU add per tile, no 64-bit pointer
    // arithmetic). n_tiles * 8 KB per head < 4 GB (Rust asserts the buffer sizes).
    const char *kh = (const char *)(kc + (size_t)kvh * n_tiles * QA_TCH);
    const char *vh = (const char *)(vc + (size_t)kvh * n_tiles * QA_TCH);
    u32x4 sk[QA_CPT], sv[QA_CPT];
    auto load_tile = [&](unsigned int t) {                       // lane-linear 16 B: 512 B per wave
        const unsigned int off = __builtin_amdgcn_readfirstlane(t) * (QA_TCH * 16u) + tid * 16u;
#pragma unroll
        for (unsigned int j = 0; j < QA_CPT; ++j) {
            sk[j] = *(const u32x4 *)(kh + (off + j * QA_NT * 16u));
            sv[j] = *(const u32x4 *)(vh + (off + j * QA_NT * 16u));
        }
    };
    auto store_tile = [&](unsigned int t, unsigned int stage) {
        const unsigned int kb0 = t * 32u;
        // Edge tile (<= 2 per WG): keys outside [seg0, wg_qmax] are legitimately
        // unwritten (another segment, a later sub-batch, the phase pad, NaN in the
        // tests); they are DEFINED as 0 in the operand so no NaN reaches a WMMA
        // (0 * NaN = NaN). Their scores are masked to -inf anyway.
        if (kb0 < seg0 || kb0 + 31u > wg_qmax) {
#pragma unroll
            for (unsigned int j = 0; j < QA_CPT; ++j) {
                const unsigned int cid = tid + j * QA_NT;
                const unsigned int kk = kb0 + ((cid >> 5) & 1u) * 16u + (cid & 15u);          // K chunk: one key
                if (kk - seg0 > wg_qmax - seg0) sk[j] = (u32x4){0, 0, 0, 0};
                const unsigned int vk = kb0 + (cid >> 8) * 16u + ((cid >> 4) & 1u) * 8u;      // V^T chunk: 8 keys
#pragma unroll
                for (unsigned int e2 = 0; e2 < 4u; ++e2) {
                    unsigned int x = sv[j][e2];
                    if (vk + 2u * e2 - seg0 > wg_qmax - seg0) x &= 0xffff0000u;
                    if (vk + 2u * e2 + 1u - seg0 > wg_qmax - seg0) x &= 0x0000ffffu;
                    sv[j][e2] = x;
                }
            }
        }
        u32x4 *st = lds + stage * (2u * QA_TCH);
#pragma unroll
        for (unsigned int j = 0; j < QA_CPT; ++j) { st[tid + j * QA_NT] = sk[j]; st[QA_TCH + tid + j * QA_NT] = sv[j]; }
    };

    load_tile(t0);
    store_tile(t0, 0u);
    if (t0 + 1u <= t1) load_tile(t0 + 1u);
    QA_SIGNAL();

    const unsigned int span = qa - seg0;                 // key valid iff (key - seg0) <=u span
    unsigned int stage = 0u;
    for (unsigned int t = t0; t <= t1; ++t) {
        const unsigned int nxt = (stage == QA_STAGES - 1u) ? 0u : stage + 1u;
        QA_WAIT();                                       // tile t is in `stage`; `nxt` is free
        if (t + 1u <= t1) store_tile(t + 1u, nxt);
        if (t + 2u <= t1) load_tile(t + 2u);             // in flight for all of compute(t)
#if QA_STAGES == 3u
        if (t < t1) QA_SIGNAL();                         // early: a whole compute(t) of slack
#endif
        if (wave_active && t <= wv_t1) {
            const u32x4 *L0 = lds + stage * (2u * QA_TCH) + lane;
            float8 s0 = {0, 0, 0, 0, 0, 0, 0, 0}, s1 = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (unsigned int ks = 0; ks < 8u; ++ks) {
                const half8 k0 = __builtin_bit_cast(half8, L0[(ks * 2u + 0u) * 32u]);
                const half8 k1 = __builtin_bit_cast(half8, L0[(ks * 2u + 1u) * 32u]);
                s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qf[ks], s0);
                s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qf[ks], s1);
            }
#ifdef QA_QK_SGB
            __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
#pragma unroll
            for (int i = 0; i < 12; ++i) {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);
#endif
            // s0[e] = score(key 32t + 8hi + e, column n), s1[e] = key + 16.
            const unsigned int kbase = t * 32u;
            if (kbase < seg0 || kbase + 31u > wv_qmin) {   // diagonal / segment-start tile (wave-uniform)
                const unsigned int b0 = kbase - seg0 + hi * 8u;
#pragma unroll
                for (unsigned int e = 0; e < 8u; ++e) {
                    if (!(b0 + e <= span)) s0[e] = -INFINITY;
                    if (!(b0 + 16u + e <= span)) s1[e] = -INFINITY;
                }
            }
            // Column max over the tile's 32 keys: 16 lane values by a max3 tree (8 ops),
            // then the L^16 half.
            const float m1 = qa_max3(s0[0], s0[1], s0[2]), m2 = qa_max3(s0[3], s0[4], s0[5]);
            const float m3 = qa_max3(s0[6], s0[7], s1[0]), m4 = qa_max3(s1[1], s1[2], s1[3]);
            const float m5 = qa_max3(s1[4], s1[5], s1[6]);
            float tm = qa_max(qa_max3(m1, m2, s1[7]), qa_max3(m3, m4, m5));
            tm = qa_max(tm, qa_xchg16(tm));
            const float tmc = tm * c;                        // c > 0: max commutes with the scale
            if (__builtin_amdgcn_ballot_w32(tmc > m + QA_LAZY_TAU)) {
                const float mn = fmaxf(m, tmc);
                // Lanes whose max did not move get exactly 1 (the skip is exact by
                // construction); tile t0: m = -inf -> exp2(-inf) = 0, O = l = 0 stay 0.
                const float corr = (mn == m) ? 1.0f : __builtin_amdgcn_exp2f(m - mn);
#pragma unroll
                for (unsigned int dt = 0; dt < 8u; ++dt) o[dt] *= corr;
                la *= corr; lb *= corr;
                m = mn;
            }
            const float negm = -m;
            half8 p0, p1;
#pragma unroll
            for (unsigned int e = 0; e < 8u; ++e) {
                // p = exp2(RN(s*c - m)): masked s = -inf -> fma = -inf -> p = +0.
                const float a = __builtin_amdgcn_exp2f(__builtin_fmaf(s0[e], c, negm));
                const float b = __builtin_amdgcn_exp2f(__builtin_fmaf(s1[e], c, negm));
                la += a; lb += b;                            // l from the unrounded f32 p
                p0[e] = (_Float16)a; p1[e] = (_Float16)b;     // RNE, in place: C(S^T) == B(P^T)
            }
            const u32x4 *V0 = L0 + QA_TCH;
#pragma unroll
            for (unsigned int dt = 0; dt < 8u; ++dt)
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    __builtin_bit_cast(half8, V0[dt * 32u]), p0, o[dt]);
#pragma unroll
            for (unsigned int dt = 0; dt < 8u; ++dt)
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    __builtin_bit_cast(half8, V0[(8u + dt) * 32u]), p1, o[dt]);
#ifndef QA_NO_PV_SGB
            // PV pipelining: 3 DS reads ahead, then {WMMA, DS read} x13, then 3 WMMAs.
            __builtin_amdgcn_sched_group_barrier(0x100, 3, 1);
#pragma unroll
            for (int i = 0; i < 13; ++i) {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
#endif
        }
#if QA_STAGES == 2u
        if (t < t1) QA_SIGNAL();                         // late: the next store overwrites this stage
#endif
        stage = nxt;
    }
    float lt = la + lb;
    lt += qa_xchg16(lt);
    // Invariant (assert, not a guard): key seg0 <= qa lies in tile t0, so every active
    // lane saw at least one valid key with p ~= 1. (Not lt >= 1: p_max = exp2 of a
    // rounding residual can be 0.99999994.)
    if (wave_active && !(lt > 0.0f && lt < INFINITY)) __builtin_trap();
    const float inv = 1.0f / lt;
    if (rv) {
        if (out32 != nullptr) {
            float *op = out32 + ((size_t)r * n_head + h) * 128u + hi * 8u;
#pragma unroll
            for (unsigned int dt = 0; dt < 8u; ++dt) {
                float8 ov;
#pragma unroll
                for (int e = 0; e < 8; ++e) ov[e] = o[dt][e] * inv;
                *(float8 *)(op + dt * 16u) = ov;
            }
        } else {
            // O^T C layout: lane L holds d = 16dt + 8hi + e of its column -> 16 contiguous B.
            _Float16 *op = out16 + (size_t)r * out_pitch + h * 128u + hi * 8u;
#pragma unroll
            for (unsigned int dt = 0; dt < 8u; ++dt) {
                half8 hv;
#pragma unroll
                for (int e = 0; e < 8; ++e) hv[e] = __builtin_bit_cast(_Float16, qe_f16(o[dt][e] * inv));
                *(half8 *)(op + dt * 16u) = hv;
            }
        }
    }
#else
    __builtin_trap();     // gfx1151: different WMMA builtin; the embed runs on the dGPU only
#endif
}
