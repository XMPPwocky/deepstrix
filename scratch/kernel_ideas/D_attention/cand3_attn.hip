// D_attention candidates, attempt 2b (gfx1201): register-lean flash-decoding split-K.
//   attention_dec_flash2_part1_k32/k64/k128 + attention_dec_flash2_combine
// Differences from cand2's flash_part1 (which spilled: 28-92 VGPRs to scratch, every V load
// serialised behind a s_wait_loadcnt 0x0 + scratch_store):
//   * q[16 heads][512] staged ONCE per WG as f16 in LDS (4 float4 loads per thread) -> the score
//     chain's A fragments come from LDS, no af[16] register array, no per-warp f32->f16 recast;
//   * V fragments (NT tiles x 16 u16 per lane) are issued right after the q loads and stay in
//     flight through the q-stage barrier and the score phase;
//   * score phase: warps 0..NT-1, GROUP K-chunk loads in flight (bf[GROUP] only).
// Numerics identical to cand2's flash kernels (scores bit-exact with production; softmax relative
// to the split max; f32 partials rescaled in the combine) -> NOT bit-exact with production.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define HD       512u
#define HT       16u
#define QSTRIDE  520u

#if defined(__gfx1200__) || defined(__gfx1201__)

template <int KPS, int GROUP>
__device__ __forceinline__ void flash2_part1_body(
        float* part_o, float* part_ml, const float* q, const _Float16* raw_kv,
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,
        const int* n_comp_per, const float* sinks, unsigned n_head, float kq_scale,
        unsigned comp_kv_batch_stride, const int* comp_base_per, unsigned max_splits,
        _Float16* q_lds, _Float16* w_lds, float* s_m, float* s_l) {
    constexpr int NT = KPS / 16;
    constexpr unsigned WS = (unsigned)KPS + 8u;
    const unsigned hg = blockIdx.x, s = blockIdx.y, b = blockIdx.z, tid = threadIdx.x;
    const unsigned h0 = hg * HT;
    if (h0 >= n_head) return;
    const unsigned n_raw = (unsigned)n_raw_per[b];
    const unsigned raw_off = (unsigned)n_raw_offset_per[b];
    const unsigned n_comp = (unsigned)n_comp_per[b];
    const unsigned n_total = n_raw + n_comp;
    const unsigned key0 = s * (unsigned)KPS;
    if (key0 >= n_total) return;                       // uniform per WG
    const unsigned nk = min((unsigned)KPS, n_total - key0);
    const size_t comp_kv_b_off = comp_base_per ? (size_t)(unsigned)comp_base_per[b] : (size_t)b * comp_kv_batch_stride;
    const unsigned wave = tid >> 5, lane = tid & 31u;
    const unsigned arow = lane & 15u, kh = (lane >> 4) * 8u;
    const unsigned d_a = wave * 16u + arow, d_b = (wave + 16u) * 16u + arow;

    auto key_row = [&](unsigned key) -> const _Float16* {   // key < n_total
        return (key < n_raw) ? raw_kv + (size_t)(key + raw_off) * HD
                             : comp_kv + (comp_kv_b_off + (key - n_raw)) * HD;
    };

    // ---- 1. q loads (f32, 4 float4 per thread)
    float4 qv[4];
    {
        const float* qb = q + ((size_t)b * n_head + h0) * HD;
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            const unsigned e = tid + (unsigned)it * 512u;
            const unsigned h = e >> 7, d4 = e & 127u;
            qv[it] = ((const float4*)(qb + h * HD))[d4];
        }
    }
    // ---- 2. V fragments for all NT tiles (in flight through the score phase)
    half8 va[NT], vb[NT];
    #pragma unroll
    for (int p = 0; p < NT; p++) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned kl = (unsigned)p * 16u + kh + (unsigned)e;
            const unsigned key = key0 + min(kl, nk - 1u);
            const _Float16* row = key_row(key);
            const _Float16 x = row[d_a], y = row[d_b];
            va[p][e] = (kl < nk) ? x : (_Float16)0;
            vb[p][e] = (kl < nk) ? y : (_Float16)0;
        }
    }
    // ---- 3. q -> LDS (f16, RN as production's per-warp cast)
    #pragma unroll
    for (int it = 0; it < 4; it++) {
        const unsigned e = tid + (unsigned)it * 512u;
        const unsigned h = e >> 7, d4 = e & 127u;
        _Float16* dst = q_lds + h * QSTRIDE + d4 * 4u;
        dst[0] = (_Float16)qv[it].x; dst[1] = (_Float16)qv[it].y; dst[2] = (_Float16)qv[it].z; dst[3] = (_Float16)qv[it].w;
    }
    __syncthreads();
    // ---- 4. phase S: warps 0..NT-1, production chain (A = f16 q, B = key row), GROUP chunks in flight
    if (wave < (unsigned)NT) {
        const unsigned col = lane & 15u;
        const unsigned kl = wave * 16u + col;
        const _Float16* brow = (kl < nk) ? key_row(key0 + kl) : nullptr;
        const _Float16* qrow = q_lds + col * QSTRIDE;
        float8 c = (float8){0,0,0,0,0,0,0,0};
        #pragma unroll 1
        for (unsigned g = 0; g < 32u; g += (unsigned)GROUP) {
            half8 bf[GROUP];
            #pragma unroll
            for (int u = 0; u < GROUP; u++) {
                const unsigned dim0 = (g + (unsigned)u) * 16u + kh;
                half8 tt = {0,0,0,0,0,0,0,0};
                if (brow) {
                    #pragma unroll
                    for (int e = 0; e < 8; e++) tt[e] = brow[dim0 + (unsigned)e];
                }
                bf[u] = tt;
            }
            #pragma unroll
            for (int u = 0; u < GROUP; u++) {
                const unsigned dim0 = (g + (unsigned)u) * 16u + kh;
                const half8 af = *(const half8*)(qrow + dim0);
                c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf[u], c);
            }
        }
        if (kl < nk) {
            #pragma unroll
            for (int e = 0; e < 8; e++) w_lds[(kh + (unsigned)e) * WS + kl] = (_Float16)(c[e] * kq_scale);
        }
    }
    __syncthreads();
    // ---- 5. local softmax: wave = head
    {
        constexpr int PER = KPS / 32;
        _Float16* wrow = w_lds + wave * WS;
        const float sink_h = sinks[h0 + wave];
        float v[PER];
        #pragma unroll
        for (int u = 0; u < PER; u++) {
            const unsigned rr = lane + (unsigned)u * 32u;
            v[u] = (rr < nk) ? (float)wrow[rr] : -INFINITY;
        }
        float m = sink_h;
        #pragma unroll
        for (int u = 0; u < PER; u++) if (v[u] > m) m = v[u];
        {
            float o;
            o = __shfl_xor(m, 16); if (o > m) m = o;
            o = __shfl_xor(m,  8); if (o > m) m = o;
            o = __shfl_xor(m,  4); if (o > m) m = o;
            o = __shfl_xor(m,  2); if (o > m) m = o;
            o = __shfl_xor(m,  1); if (o > m) m = o;
        }
        float l = 0.0f;
        #pragma unroll
        for (int u = 0; u < PER; u++) {
            const unsigned rr = lane + (unsigned)u * 32u;
            if (rr < nk) {
                const float w = expf(v[u] - m);
                wrow[rr] = (_Float16)w;
                l += w;
            }
        }
        l += __shfl_xor(l, 16);
        l += __shfl_xor(l,  8);
        l += __shfl_xor(l,  4);
        l += __shfl_xor(l,  2);
        l += __shfl_xor(l,  1);
        if (lane == 0) { s_m[wave] = m; s_l[wave] = l; }
    }
    __syncthreads();
    // ---- 6. W.V from registers
    float8 c_a = {0,0,0,0,0,0,0,0}, c_b = {0,0,0,0,0,0,0,0};
    const _Float16* wr = w_lds + arow * WS;
    #pragma unroll
    for (int p = 0; p < NT; p++) {
        if ((unsigned)p * 16u >= nk) break;
        half8 a;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned kl = (unsigned)p * 16u + kh + (unsigned)e;
            a[e] = (kl < nk) ? wr[kl] : (_Float16)0;
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, va[p], c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, vb[p], c_b);
    }
    const size_t pbase = ((size_t)b * max_splits + s) * n_head + h0;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned mh = (lane >> 4) * 8u + (unsigned)e;
        float* o = part_o + (pbase + mh) * HD;
        o[d_a] = c_a[e];
        o[d_b] = c_b[e];
    }
    if (wave == 0 && lane < HT) {
        part_ml[(pbase + lane) * 2u + 0u] = s_m[lane];
        part_ml[(pbase + lane) * 2u + 1u] = s_l[lane];
    }
}

#define DEC_FLASH2_PART1(NAME, KPS, GROUP)                                                     \
extern "C" __global__ __launch_bounds__(512) void NAME(                                        \
        float* part_o, float* part_ml, const float* q, const _Float16* raw_kv,                 \
        const _Float16* comp_kv, const int* n_raw_per, const int* n_raw_offset_per,            \
        const int* n_comp_per, const float* sinks, unsigned n_head, float kq_scale,            \
        unsigned comp_kv_batch_stride, const int* comp_base_per, unsigned max_splits) {        \
    __shared__ _Float16 q_lds[HT * QSTRIDE];                                                   \
    __shared__ _Float16 w_lds[HT * ((KPS) + 8u)];                                              \
    __shared__ float s_m[HT], s_l[HT];                                                         \
    flash2_part1_body<KPS, GROUP>(part_o, part_ml, q, raw_kv, comp_kv, n_raw_per,              \
                          n_raw_offset_per, n_comp_per, sinks, n_head, kq_scale,               \
                          comp_kv_batch_stride, comp_base_per, max_splits, q_lds, w_lds, s_m, s_l); \
}
DEC_FLASH2_PART1(attention_dec_flash2_part1_k32, 32, 16)
DEC_FLASH2_PART1(attention_dec_flash2_part1_k64, 64, 16)
DEC_FLASH2_PART1(attention_dec_flash2_part1_k128, 128, 16)
DEC_FLASH2_PART1(attention_dec_flash2_part1_k64g32, 64, 32)

extern "C" __global__ __launch_bounds__(512) void attention_dec_flash2_combine(
        float* out, const float* part_o, const float* part_ml, const float* sinks,
        const int* n_raw_per, const int* n_comp_per, unsigned n_head, unsigned kps, unsigned max_splits) {
    const unsigned h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
    const unsigned n_total = (unsigned)n_raw_per[b] + (unsigned)n_comp_per[b];
    const unsigned S = (n_total + kps - 1u) / kps;
    const size_t pbase = (size_t)b * max_splits * n_head + h;
    float M = sinks[h];
    for (unsigned s = 0; s < S; s++) M = fmaxf(M, part_ml[(pbase + (size_t)s * n_head) * 2u]);
    float L = expf(sinks[h] - M), o = 0.0f;
    for (unsigned s = 0; s < S; s++) {
        const size_t pi = pbase + (size_t)s * n_head;
        const float f = expf(part_ml[pi * 2u] - M);
        L += part_ml[pi * 2u + 1u] * f;
        o += part_o[pi * HD + d] * f;
    }
    out[((size_t)b * n_head + h) * HD + d] = o / L;
}

#endif  // gfx12
