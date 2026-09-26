// A_moe_smallb/gu_v2 — small-b (b <= GU_MB rows) twin of mxfp4_pair_matvec_fused_swiglu_kwide.
//
// Same work-item interface and grid, same lane -> data map as kwide (warp = row; sbh = lane>>4 picks the
// super-block of the pair, l15 = lane&15 owns mxfp4 block b32 = l15>>1, nibble half = l15&1), same per-lane
// accumulation order => bit-exact with kwide. What changes (all aimed at memory-level parallelism):
//   * the members' Q8_K rows are staged in LDS ONCE for the whole row (GU_MB x 20 blocks x (256 B qs + yd)),
//     so the pair loop has no barriers and no vmcnt(0) drain (kwide stages per pair between two barriers);
//   * the weight stream is explicitly software-pipelined: a static ring of GU_D pairs of raw weights is kept in
//     flight, with __builtin_amdgcn_sched_barrier pinning the prefetch distance (the compiler neither hoists
//     everything -> spills, nor serialises per pair);
//   * the member loop is straight-line over GU_MB (compile-time) members, no dynamic VGPR indexing (v_movrel);
//     members >= nm read a zeroed yd and are never written. GU_MB = 4 (b <= 4, LDS 20.8 KB, 3 WGs/CU) or 8
//     (b <= 8, LDS 41.6 KB, 1 WG/CU); the wrapper picks the instantiation by b. A group with more members than
//     GU_MB is handled by further passes that re-stream the row's weights (never at b <= GU_MB).
// Production symbol would be mxfp4_pair_matvec_fused_swiglu_smallb<MB>; exported as cand_gate_up.
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef SUPER_MXFP4_BYTES
#define SUPER_MXFP4_BYTES 136u
#endif
#ifndef BLOCK_Q8_K_BYTES
#define BLOCK_Q8_K_BYTES 292u
#endif
#ifndef GU_MB
#define GU_MB 8u
#endif
#ifndef GU_NB
#define GU_NB 20u          // n_blocks (5120 / 256); compile-time so the pair loop is static
#endif
#ifndef GU_D
#define GU_D 5             // prefetch depth in super-block pairs (must divide GU_NB/2)
#endif
#ifndef GU_SCHED_BARRIER
#define GU_SCHED_BARRIER 1
#endif
#ifndef GU_PIN
#define GU_PIN 0
#endif
#define GU_NP (GU_NB / 2u)

__device__ __constant__ int8_t kvalues_mxfp4_gu2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ int32_t sudot4_gu2(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
__device__ __forceinline__ float e8m0_half_gu2(uint32_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}
__device__ __forceinline__ float warp_sum_gu2(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, o);
    return v;
}

struct RawPair { uint2 ga, gb, ua, ub; uint8_t gs, us; };
// Pin a load to its program position: the address passes through an empty volatile asm, so the compiler cannot
// hoist the (otherwise invariant, freely movable) weight loads to the kernel entry -- which it did, then spilled
// the ~200 dwords it had in flight. Program order = prefetch order.
__device__ __forceinline__ const uint8_t* pin_gu2(const uint8_t* p) {
#if GU_PIN
    asm volatile("" : "+v"(p));
#endif
    return p;
}
__device__ __forceinline__ void load_pair(const uint8_t* g_sb, const uint8_t* u_sb, unsigned b32, RawPair& r) {
    g_sb = pin_gu2(g_sb);
    u_sb = pin_gu2(u_sb);
    __builtin_memcpy(&r.ga, g_sb + b32 * 16u, 8);
    __builtin_memcpy(&r.gb, g_sb + b32 * 16u + 8u, 8);
    r.gs = g_sb[128u + b32];
    __builtin_memcpy(&r.ua, u_sb + b32 * 16u, 8);
    __builtin_memcpy(&r.ub, u_sb + b32 * 16u + 8u, 8);
    r.us = u_sb[128u + b32];
}
// Bit-identical to mxfp4_unpack16 in the in-tree kernel.
__device__ __forceinline__ void unpack16_gu2(uint2 a, uint2 b, uint8_t sc, unsigned half, const int8_t* s_lut,
                                             int32_t w[4], float& ds) {
    ds = e8m0_half_gu2(sc);
    const uint32_t qw[4] = {a.x, a.y, b.x, b.y};
    const unsigned sh = 4u * half;
    #pragma unroll
    for (unsigned i = 0; i < 4u; i++) {
        union { int32_t d; int8_t b[4]; } u;
        #pragma unroll
        for (unsigned j = 0; j < 4u; j++) u.b[j] = s_lut[(((qw[i] >> (8u * j)) & 0xFFu) >> sh) & 0x0Fu];
        w[i] = u.d;
    }
}

extern "C" __global__ void cand_gate_up(
        float        * __restrict__ mid,           // [B, n_used, n_rows]
        const uint8_t * __restrict__ gate_w_base,
        const uint8_t * __restrict__ up_w_base,
        const uint8_t * __restrict__ xq,           // [B, n_blocks*292]
        const float  * __restrict__ expert_w,      // [B, n_used]
        const int    * __restrict__ group_count,   // [n_expert]
        const int    * __restrict__ expert_members,// [n_expert * max_per_expert]
        const int    * __restrict__ work_items,    // [n_work_items]
        unsigned int    gate_bpe,
        unsigned int    up_bpe,
        unsigned int    n_used,
        unsigned int    max_per_expert,
        unsigned int    chunk_size,
        float           clamp,
        unsigned int    n_rows,
        unsigned int    n_blocks,
        const int    * __restrict__ n_work_items_dev) {
    if (n_work_items_dev != nullptr && (int)blockIdx.y >= *n_work_items_dev) return;
    if (n_blocks != GU_NB) return;  // compile-time width; the wrapper must guarantee it
    const int packed_item = work_items[blockIdx.y];
    const unsigned e            = (unsigned)(packed_item >> 16) & 0xffffu;
    const unsigned member_start = (unsigned)(packed_item) & 0xffffu;
    const int group_n = group_count[e];
    const unsigned member_end = min((unsigned)group_n, member_start + chunk_size);
    if (member_end <= member_start) return;

    __shared__ int8_t s_lut[16];
    __shared__ __align__(16) uint4 s_q8v[GU_MB * GU_NB * 16];   // [m][blk][16 lanes] x 16 B = the 256 qs bytes
    __shared__ float s_yd[GU_MB * GU_NB];                       // [m][blk]
    __shared__ int   s_packed[GU_MB];

    const unsigned tid  = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned row  = blockIdx.x * (blockDim.x >> 5u) + (tid >> 5u);
    const bool row_valid = row < n_rows;   // wave-uniform; kept as a predicate on the loads/stores only
    const unsigned sbh  = lane >> 4;
    const unsigned l15  = lane & 15u;
    const unsigned b32  = l15 >> 1;
    const unsigned half = l15 & 1u;

    const uint8_t* g_row = gate_w_base + (uint64_t)e * gate_bpe + (uint64_t)(row_valid ? row : 0u) * GU_NB * SUPER_MXFP4_BYTES;
    const uint8_t* u_row = up_w_base   + (uint64_t)e * up_bpe   + (uint64_t)(row_valid ? row : 0u) * GU_NB * SUPER_MXFP4_BYTES;

    if (tid < 16u) s_lut[tid] = kvalues_mxfp4_gu2[tid];

    for (unsigned mg = member_start; mg < member_end; mg += GU_MB) {
        const unsigned nm = min(GU_MB, member_end - mg);
        // ---- stage all members' Q8_K rows once (qs as aligned uint4 lanes, yd separately)
        if (tid < GU_MB) s_packed[tid] = (tid < nm) ? expert_members[(size_t)e * max_per_expert + (size_t)(mg + tid)] : 0;
        __syncthreads();
        #pragma unroll
        for (unsigned m = 0; m < GU_MB; m++) {
            if (m < nm) {
                const unsigned b = (unsigned)(s_packed[m] >> 16);
                const uint32_t* src = (const uint32_t*)(xq + (size_t)b * GU_NB * BLOCK_Q8_K_BYTES);
                uint32_t* dst = (uint32_t*)&s_q8v[m * GU_NB * 16u];
                for (unsigned i = tid; i < GU_NB * 64u; i += blockDim.x) {
                    const unsigned blk = i >> 6, ii = i & 63u;
                    dst[i] = src[blk * 73u + 1u + ii];
                }
                if (tid < GU_NB) s_yd[m * GU_NB + tid] = __uint_as_float(src[tid * 73u]);
            } else {
                if (tid < GU_NB) s_yd[m * GU_NB + tid] = 0.0f;
            }
        }
        __syncthreads();

        float acc_g[GU_MB], acc_u[GU_MB];
        #pragma unroll
        for (unsigned m = 0; m < GU_MB; m++) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

        // ---- weight stream: ring of GU_D pairs in flight. ROLLED outer loop (unroll 1): a fully unrolled
        // straight-line body made the compiler hoist every load to the entry and spill ~200 dwords; the loop
        // back-edge plus the uniform prefetch branch keep program order = prefetch order. Each pair's raw
        // registers are reloaded (pair p + GU_D) right after that pair is consumed, so GU_D-1..GU_D pairs
        // (34-170 B/lane) are in flight while the members' dots run.
        RawPair ring[GU_D];
        #pragma unroll
        for (unsigned d = 0; d < GU_D; d++) {
            const unsigned sb = 2u * d + sbh;
            load_pair(g_row + sb * SUPER_MXFP4_BYTES, u_row + sb * SUPER_MXFP4_BYTES, b32, ring[d]);
        }
        #pragma unroll 1
        for (unsigned p0 = 0; p0 < GU_NP; p0 += GU_D) {
            #pragma unroll
            for (unsigned d = 0; d < GU_D; d++) {
                const unsigned p = p0 + d;
                const RawPair cur = ring[d];
                const unsigned pn = p + GU_D;
                if (pn < GU_NP) {   // wave-uniform
                    const unsigned sb = 2u * pn + sbh;
                    load_pair(g_row + sb * SUPER_MXFP4_BYTES, u_row + sb * SUPER_MXFP4_BYTES, b32, ring[d]);
                }
                int32_t wg[4], wu[4];
                float gds, uds;
                unpack16_gu2(cur.ga, cur.gb, cur.gs, half, s_lut, wg, gds);
                unpack16_gu2(cur.ua, cur.ub, cur.us, half, s_lut, wu, uds);
                const unsigned blk = 2u * p + sbh;
                #pragma unroll
                for (unsigned m = 0; m < GU_MB; m++) {
                    const uint4 q = s_q8v[(m * GU_NB + blk) * 16u + l15];
                    const float yd = s_yd[m * GU_NB + blk];
                    int32_t sg = 0, su = 0;
                    sg = sudot4_gu2(wg[0], (int32_t)q.x, sg);
                    sg = sudot4_gu2(wg[1], (int32_t)q.y, sg);
                    sg = sudot4_gu2(wg[2], (int32_t)q.z, sg);
                    sg = sudot4_gu2(wg[3], (int32_t)q.w, sg);
                    su = sudot4_gu2(wu[0], (int32_t)q.x, su);
                    su = sudot4_gu2(wu[1], (int32_t)q.y, su);
                    su = sudot4_gu2(wu[2], (int32_t)q.z, su);
                    su = sudot4_gu2(wu[3], (int32_t)q.w, su);
                    acc_g[m] += (float)sg * (yd * gds);
                    acc_u[m] += (float)su * (yd * uds);
                }
            }
        }
        // ---- reduce + swiglu + output (kwide order)
        #pragma unroll
        for (unsigned m = 0; m < GU_MB; m++) {
            float g_sum = warp_sum_gu2(acc_g[m]);
            float u_sum = warp_sum_gu2(acc_u[m]);
            if (lane == 0u && m < nm && row_valid) {
                const int packed    = s_packed[m];
                const unsigned b    = (unsigned)(packed >> 16);
                const unsigned slot = (unsigned)(packed & 0xFFFF);
                if (clamp > 1.0e-6f) {
                    if (g_sum > clamp)  g_sum = clamp;
                    if (u_sum > clamp)  u_sum = clamp;
                    if (u_sum < -clamp) u_sum = -clamp;
                }
                const float ew  = expert_w[(size_t)b * n_used + slot];
                const float sig = 1.0f / (1.0f + expf(-g_sum));
                mid[((size_t)b * n_used + slot) * n_rows + row] = g_sum * sig * u_sum * ew;
            }
        }
        __syncthreads();   // before a further pass overwrites the staging (only if nm > GU_MB)
    }
}
