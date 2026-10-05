// q8_0_grouped_gemv — variant of `q8_0_gemv_warp8` where each output row
// `idx` reads its group's input slice (`g = idx / rank`). Mirrors ds4's
// `matvec_q8_0_grouped_rows_decode_scratch` (ds4.c:3618).
//
// Used by V4 Flash for `attn_output_a`: maps `attn_heads_inv_rope`
// `[n_groups=8, group_dim=4096]` → `attn_out_low` `[n_groups=8, rank=1024]`.
//
// Block of 256 threads handles 8 output rows × 32 lanes per row (same
// shape as q8_0_matvec). Per-block input quantisation (q8_0_quantize_f32)
// works unchanged for the grouped case because Q8_0 blocks are 32-element
// chunks and group_dim is divisible by 32 — the per-block absmax of each
// chunk is identical whether we quantise per-group or as a flat 32k-element
// buffer. Same for xscale.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define WARP_LANES 32u
#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ int32_t gdp4a_sudot(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int32_t gdot_i8x32(const int8_t* a, const int8_t* b) {
    // M18: 2× uint4 loads per side, 8× dp4a. See q8_0_matvec.hip for rationale.
    int32_t dot = 0;
    uint4 av0, av1, bv0, bv1;
    __builtin_memcpy(&av0, a, 16);
    __builtin_memcpy(&av1, a + 16, 16);
    __builtin_memcpy(&bv0, b, 16);
    __builtin_memcpy(&bv1, b + 16, 16);
    dot = gdp4a_sudot((int32_t)av0.x, (int32_t)bv0.x, dot);
    dot = gdp4a_sudot((int32_t)av0.y, (int32_t)bv0.y, dot);
    dot = gdp4a_sudot((int32_t)av0.z, (int32_t)bv0.z, dot);
    dot = gdp4a_sudot((int32_t)av0.w, (int32_t)bv0.w, dot);
    dot = gdp4a_sudot((int32_t)av1.x, (int32_t)bv1.x, dot);
    dot = gdp4a_sudot((int32_t)av1.y, (int32_t)bv1.y, dot);
    dot = gdp4a_sudot((int32_t)av1.z, (int32_t)bv1.z, dot);
    dot = gdp4a_sudot((int32_t)av1.w, (int32_t)bv1.w, dot);
    return dot;
}

__device__ __forceinline__ float gwarp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
    }
    return v;
}

extern "C" __global__ void q8_0_grouped_gemv(
        float        * __restrict__ out,         // [n_groups * rank]
        const unsigned char * __restrict__ w,    // [n_groups * rank, blocks_per_group * 34]
        const int8_t * __restrict__ xq,          // [n_groups * group_dim]
        const float  * __restrict__ xscale,      // [n_groups * blocks_per_group]
        unsigned int   group_dim,                // K per group (e.g. 4096)
        unsigned int   rank,                     // rows per group (e.g. 1024)
        unsigned int   blocks_per_group,         // group_dim / 32 (e.g. 128)
        unsigned int   n_groups)                 // e.g. 8
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;

    uint32_t g = row / rank;
    const unsigned char* wr = w + (size_t)row * (size_t)blocks_per_group * 34u;
    const int8_t* xq_g = xq + (size_t)g * (size_t)group_dim;
    const float*  xs_g = xscale + (size_t)g * (size_t)blocks_per_group;

    // M18 repacked layout: [scales(2*blocks bytes) | quants(32*blocks bytes)] per row.
    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)blocks_per_group * 2u);
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks_per_group; b += WARP_LANES) {
        const int8_t* qs = quants + (size_t)b * 32u;     // 4-byte aligned
        const int8_t* xqb = xq_g + (size_t)b * 32u;
        int32_t dot = gdot_i8x32(qs, xqb);
        acc += __half2float(scales[b]) * xs_g[b] * (float)dot;
    }

    acc = gwarp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

// M40-P4.5: 2-wide pair variant. Same weight matrix, two input column-pairs.
// Each input pair has its OWN per-group xq/xscale (groups are sliced from
// the flat 2-token activation). Per row, per lane: load W block ONCE,
// compute TWO dot products (one per input column), multiply by two scales,
// accumulate into two scalars. Output to two separate buffers.
extern "C" __global__ void q8_0_grouped_gemv_pair(
        float        * __restrict__ out_a,
        float        * __restrict__ out_b,
        const unsigned char * __restrict__ w,
        const int8_t * __restrict__ xq_a,
        const int8_t * __restrict__ xq_b,
        const float  * __restrict__ xscale_a,
        const float  * __restrict__ xscale_b,
        unsigned int   group_dim,
        unsigned int   rank,
        unsigned int   blocks_per_group,
        unsigned int   n_groups)
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;

    uint32_t g = row / rank;
    const unsigned char* wr = w + (size_t)row * (size_t)blocks_per_group * 34u;
    const int8_t* xqg_a = xq_a + (size_t)g * (size_t)group_dim;
    const int8_t* xqg_b = xq_b + (size_t)g * (size_t)group_dim;
    const float*  xsg_a = xscale_a + (size_t)g * (size_t)blocks_per_group;
    const float*  xsg_b = xscale_b + (size_t)g * (size_t)blocks_per_group;

    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)blocks_per_group * 2u);
    float acc_a = 0.0f;
    float acc_b = 0.0f;
    for (uint32_t b = lane; b < blocks_per_group; b += WARP_LANES) {
        const int8_t* qs   = quants + (size_t)b * 32u;
        const int8_t* xqba = xqg_a + (size_t)b * 32u;
        const int8_t* xqbb = xqg_b + (size_t)b * 32u;
        int32_t dot_a = gdot_i8x32(qs, xqba);
        int32_t dot_b = gdot_i8x32(qs, xqbb);
        float ws = __half2float(scales[b]);
        acc_a += ws * xsg_a[b] * (float)dot_a;
        acc_b += ws * xsg_b[b] * (float)dot_b;
    }

    acc_a = gwarp_sum_f32(acc_a);
    acc_b = gwarp_sum_f32(acc_b);
    if (lane == 0) {
        out_a[row] = acc_a;
        out_b[row] = acc_b;
    }
}

// M50 Phase 2: batched grouped GEMV with grid.z = B. Each WG handles
// one (output_row, batch) pair. Per-batch inputs/outputs at offset
// `batch * (n_groups * group_dim)` for xq, `batch * (n_groups *
// blocks_per_group)` for xscale, and `batch * out_dim` for out.
// Weight is shared across batch.
extern "C" __global__ void q8_0_grouped_gemv_batched(
        float        * __restrict__ out,         // [B, n_groups * rank]
        const unsigned char * __restrict__ w,    // [n_groups * rank, blocks_per_group * 34]
        const int8_t * __restrict__ xq,          // [B, n_groups * group_dim]
        const float  * __restrict__ xscale,      // [B, n_groups * blocks_per_group]
        unsigned int   group_dim,
        unsigned int   rank,
        unsigned int   blocks_per_group,
        unsigned int   n_groups)
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t b = blockIdx.z;
    uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;

    uint32_t g = row / rank;
    const unsigned char* wr = w + (size_t)row * (size_t)blocks_per_group * 34u;
    const int8_t* xq_g = xq + (size_t)b * (size_t)(n_groups * group_dim)
                            + (size_t)g * (size_t)group_dim;
    const float*  xs_g = xscale + (size_t)b * (size_t)(n_groups * blocks_per_group)
                                + (size_t)g * (size_t)blocks_per_group;

    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)blocks_per_group * 2u);
    float acc = 0.0f;
    for (uint32_t bl = lane; bl < blocks_per_group; bl += WARP_LANES) {
        const int8_t* qs = quants + (size_t)bl * 32u;
        const int8_t* xqb = xq_g + (size_t)bl * 32u;
        int32_t dot = gdot_i8x32(qs, xqb);
        acc += __half2float(scales[bl]) * xs_g[bl] * (float)dot;
    }
    acc = gwarp_sum_f32(acc);
    if (lane == 0) {
        out[(size_t)b * (size_t)out_dim + row] = acc;
    }
}

#define GGEMV_BPACK_MAX 16u

// B-PACKED grouped GEMV. Twin of `q8_0_grouped_gemv_batched`, which launches
// grid.z = B so each batch element re-reads the ENTIRE weight matrix: the
// drafter's `attn_output_a` is 35.7 MB and was MEASURED at 6.05 ms/step
// against a 0.47 ms single-read roofline.
//
// Here each lane loads its weight block ONCE into registers and loops the
// batch inside. Bit-identical to the batched twin per (row, b): the int8 dot
// is exact integer arithmetic (identical dp4a chain), and the float
// accumulation `acc += ws * xs * dot` keeps the same order over `bl`.
extern "C" __global__ void q8_0_grouped_gemv_bpack(
        float        * __restrict__ out,         // [B, n_groups * rank]
        const unsigned char * __restrict__ w,    // [n_groups * rank, blocks_per_group * 34]
        const int8_t * __restrict__ xq,          // [B, n_groups * group_dim]
        const float  * __restrict__ xscale,      // [B, n_groups * blocks_per_group]
        unsigned int   group_dim,
        unsigned int   rank,
        unsigned int   blocks_per_group,
        unsigned int   n_groups,
        unsigned int   batch)                    // B <= GGEMV_BPACK_MAX
{
#include "q8_0_grouped_gemv_bpack_body.inc"
}

// q8_0_grouped_gemv_bpack_z16 — the q8_0_grouped_gemv_bpack body above with the
// batch split into chunks of 16 on grid.z = ceil(batch/16) (slice z offsets
// xq / xscale / out by 16 z rows): BIT-IDENTICAL per (row, b) to
// q8_0_grouped_gemv_batched, weight re-read ceil(b/16) times instead of b.
// 2026-09-26 sweep (C2_dense_prefill/replay_bpack_z16, cold, b = 64): wo_a 1210
// -> 535 us. Selected by Q8_0GroupedMatvec for 16 < b <= 64 on the dp4a arm
// (V41_GEMV_BPACK_Z16). grid (out_dim/8, 1, ceil(b/16)) x 256.
extern "C" __global__ void q8_0_grouped_gemv_bpack_z16(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, unsigned int group_dim, unsigned int rank,
    unsigned int blocks_per_group, unsigned int n_groups, unsigned int batch)
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;
    const uint32_t b0 = blockIdx.z * GGEMV_BPACK_MAX;
    if (b0 >= batch) return;
    const uint32_t nb = (batch - b0 < GGEMV_BPACK_MAX) ? (batch - b0) : GGEMV_BPACK_MAX;
    const uint32_t xq_stride = n_groups * group_dim;
    const uint32_t xs_stride = n_groups * blocks_per_group;
    xq += (size_t)b0 * xq_stride; xscale += (size_t)b0 * xs_stride; out += (size_t)b0 * out_dim;
    uint32_t g = row / rank;
    const unsigned char* wr = w + (size_t)row * (size_t)blocks_per_group * 34u;
    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)blocks_per_group * 2u);
    float acc[GGEMV_BPACK_MAX];
    for (uint32_t b = 0; b < nb; ++b) acc[b] = 0.0f;
    for (uint32_t bl = lane; bl < blocks_per_group; bl += WARP_LANES) {
        const int8_t* qs = quants + (size_t)bl * 32u;
        uint4 av0, av1;
        __builtin_memcpy(&av0, qs, 16);
        __builtin_memcpy(&av1, qs + 16, 16);
        const float ws = __half2float(scales[bl]);
        for (uint32_t b = 0; b < nb; ++b) {
            const int8_t* xqb = xq + (size_t)b * xq_stride + (size_t)g * group_dim + (size_t)bl * 32u;
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
            acc[b] += ws * xscale[(size_t)b * xs_stride + (size_t)g * blocks_per_group + bl] * (float)dot;
        }
    }
    for (uint32_t b = 0; b < nb; ++b) {
        float a = gwarp_sum_f32(acc[b]);
        if (lane == 0) out[(size_t)b * (size_t)out_dim + row] = a;
    }
}

// ---------------------------------------------------------------------------
// COMPILE-TIME-BATCH twins of q8_0_grouped_gemv_bpack: `q8_0_grouped_gemv_bpack_tB<B>`,
// B = 1..8. Same argument list (`batch` ignored; the wrapper picks the symbol), same grid
// ((out_dim/8) x 256), SAME arithmetic per (row, b) -> BIT-IDENTICAL to the runtime kernel
// (tests/q8_0_tb_bitexact.rs; sweep review incl. rank=1000 WGs crossing groups, G=3).
// Mechanism as in q8_0_matvec.hip (`q8_0_gemv_bpack_tB<B>`): the unrolled b-loop issues all
// activation loads with the weight loads, one wait per block, no v_movrel.
// 2026-09-26 sweep (C1_dense_decode/tB_compile_time_batch, wo_a 8 x (1024 x 4096), cold,
// graph): b=4 ratio 0.74 (reviewer 0.735), b=5 0.75, b=8 similar; b=1 tB1 is 2.5% SLOWER
// than the runtime kernel (76.8 -> 78.7 us, 6/6 runs) so the wrapper keeps the runtime
// kernel at b=1. `V41_GEMV_TB=0` (src/q8_0.rs) restores the runtime kernel at every batch.
__device__ __forceinline__ int32_t gdot8_regs(const uint4& a0, const uint4& a1,
                                              const uint4& b0, const uint4& b1) {
    int32_t dot = 0;
    dot = gdp4a_sudot((int32_t)a0.x, (int32_t)b0.x, dot);
    dot = gdp4a_sudot((int32_t)a0.y, (int32_t)b0.y, dot);
    dot = gdp4a_sudot((int32_t)a0.z, (int32_t)b0.z, dot);
    dot = gdp4a_sudot((int32_t)a0.w, (int32_t)b0.w, dot);
    dot = gdp4a_sudot((int32_t)a1.x, (int32_t)b1.x, dot);
    dot = gdp4a_sudot((int32_t)a1.y, (int32_t)b1.y, dot);
    dot = gdp4a_sudot((int32_t)a1.z, (int32_t)b1.z, dot);
    dot = gdp4a_sudot((int32_t)a1.w, (int32_t)b1.w, dot);
    return dot;
}

template <int B>
__device__ __forceinline__ void q8_0_grouped_gemv_bpack_tb_body(
    float* __restrict__ out, const unsigned char* __restrict__ w,
    const int8_t* __restrict__ xq, const float* __restrict__ xscale,
    unsigned group_dim, unsigned rank, unsigned bpg, unsigned n_groups)
{
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;
    uint32_t g = row / rank;
    const unsigned char* wr = w + (size_t)row * (size_t)bpg * 34u;
    const __half* scales = (const __half*)wr;
    const int8_t* quants = (const int8_t*)(wr + (size_t)bpg * 2u);
    const uint32_t xq_stride = n_groups * group_dim, xs_stride = n_groups * bpg;
    const int8_t* xq_g = xq + (size_t)g * group_dim;
    const float* xs_g = xscale + (size_t)g * bpg;
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;
    for (uint32_t bl = lane; bl < bpg; bl += WARP_LANES) {
        uint4 av0, av1;
        __builtin_memcpy(&av0, quants + (size_t)bl * 32u, 16);
        __builtin_memcpy(&av1, quants + (size_t)bl * 32u + 16, 16);
        const float ws = __half2float(scales[bl]);
        uint4 bv0[B], bv1[B];
        float xs[B];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const int8_t* xqb = xq_g + (size_t)b * xq_stride + (size_t)bl * 32u;
            __builtin_memcpy(&bv0[b], xqb, 16);
            __builtin_memcpy(&bv1[b], xqb + 16, 16);
            xs[b] = xs_g[(size_t)b * xs_stride + bl];
        }
#pragma unroll
        for (int b = 0; b < B; ++b) {
            int32_t dot = gdot8_regs(av0, av1, bv0[b], bv1[b]);
            acc[b] += ws * xs[b] * (float)dot;
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        float a = gwarp_sum_f32(acc[b]);
        if (lane == 0) out[(size_t)b * (size_t)out_dim + row] = a;
    }
}
#define Q8_0_GROUPED_GEMV_BPACK_TB(NAME, B)                                                \
    extern "C" __global__ void NAME(float* __restrict__ out,                               \
                                    const unsigned char* __restrict__ w,                   \
                                    const int8_t* __restrict__ xq,                         \
                                    const float* __restrict__ xscale,                      \
                                    unsigned group_dim, unsigned rank, unsigned bpg,       \
                                    unsigned n_groups, unsigned batch) {                   \
        (void)batch;                                                                       \
        q8_0_grouped_gemv_bpack_tb_body<B>(out, w, xq, xscale, group_dim, rank, bpg, n_groups); \
    }
Q8_0_GROUPED_GEMV_BPACK_TB(q8_0_grouped_gemv_bpack_tB1, 1)
Q8_0_GROUPED_GEMV_BPACK_TB(q8_0_grouped_gemv_bpack_tB2, 2)
Q8_0_GROUPED_GEMV_BPACK_TB(q8_0_grouped_gemv_bpack_tB3, 3)
Q8_0_GROUPED_GEMV_BPACK_TB(q8_0_grouped_gemv_bpack_tB4, 4)
Q8_0_GROUPED_GEMV_BPACK_TB(q8_0_grouped_gemv_bpack_tB5, 5)
Q8_0_GROUPED_GEMV_BPACK_TB(q8_0_grouped_gemv_bpack_tB6, 6)
Q8_0_GROUPED_GEMV_BPACK_TB(q8_0_grouped_gemv_bpack_tB7, 7)
Q8_0_GROUPED_GEMV_BPACK_TB(q8_0_grouped_gemv_bpack_tB8, 8)

// `_ind` twins (docs/v41/GRAPH_KEYS_DESIGN.md 2.3): operands 0..3 (out, w, xq, xscale) resolve
// through the arena context when their `ind_mask` bit is set; same body. Production twins only
// (no canary code, 2.8); `canary` / `canary_tag` are ignored.
#include "arena_ctx.inc"
#define Q8_0_GROUPED_IND_PARAMS                                                            \
    uint32_t ind_mask, const ArenaCtx* canary, uint32_t canary_tag, float* out,            \
    const unsigned char* w, const int8_t* xq, const float* xscale, unsigned group_dim,     \
    unsigned rank, unsigned bpg, unsigned n_groups, unsigned batch
#define Q8_0_GROUPED_IND_DEREF                                                             \
    ARENA_CTX_DEREF(ind_mask, 0, float*, out);                                             \
    ARENA_CTX_DEREF(ind_mask, 1, const unsigned char*, w);                                 \
    ARENA_CTX_DEREF(ind_mask, 2, const int8_t*, xq);                                       \
    ARENA_CTX_DEREF(ind_mask, 3, const float*, xscale);
#define Q8_0_GROUPED_IND_XOR (ARENA_CTX_U64(out) ^ ARENA_CTX_U64(w) ^ ARENA_CTX_U64(xq) ^ ARENA_CTX_U64(xscale))
#define Q8_0_GROUPED_GEMV_BPACK_TB_IND(NAME, B)                                            \
    extern "C" __global__ void NAME(Q8_0_GROUPED_IND_PARAMS) {                             \
        (void)batch;                                                                       \
        (void)canary;                                                                      \
        (void)canary_tag;                                                                  \
        Q8_0_GROUPED_IND_DEREF                                                             \
        q8_0_grouped_gemv_bpack_tb_body<B>(out, w, xq, xscale, group_dim, rank, bpg, n_groups); \
    }                                                                                      \
    extern "C" __global__ void NAME##_canary(Q8_0_GROUPED_IND_PARAMS) {                    \
        (void)batch;                                                                       \
        Q8_0_GROUPED_IND_DEREF                                                             \
        ARENA_CTX_CANARY_LOAD(canary, Q8_0_GROUPED_IND_XOR);                               \
        q8_0_grouped_gemv_bpack_tb_body<B>(out, w, xq, xscale, group_dim, rank, bpg, n_groups); \
        ARENA_CTX_CANARY_STORE(canary_tag);                                                \
    }
Q8_0_GROUPED_GEMV_BPACK_TB_IND(q8_0_grouped_gemv_bpack_tB2_ind, 2)
Q8_0_GROUPED_GEMV_BPACK_TB_IND(q8_0_grouped_gemv_bpack_tB3_ind, 3)
Q8_0_GROUPED_GEMV_BPACK_TB_IND(q8_0_grouped_gemv_bpack_tB4_ind, 4)
Q8_0_GROUPED_GEMV_BPACK_TB_IND(q8_0_grouped_gemv_bpack_tB5_ind, 5)
Q8_0_GROUPED_GEMV_BPACK_TB_IND(q8_0_grouped_gemv_bpack_tB6_ind, 6)
Q8_0_GROUPED_GEMV_BPACK_TB_IND(q8_0_grouped_gemv_bpack_tB7_ind, 7)
Q8_0_GROUPED_GEMV_BPACK_TB_IND(q8_0_grouped_gemv_bpack_tB8_ind, 8)

// The runtime-batch kernel's twin: production's b = 1 arm (`grouped_bpack_symbol`: tB1 measured
// slower) and every b under `V41_GEMV_TB=0`. Same body as a device function (inlined: the
// __restrict__ parameters become scoped-noalias metadata on the resolved pointers).
__device__ __forceinline__ static void q8_0_grouped_gemv_bpack_rt_body(
        float        * __restrict__ out,
        const unsigned char * __restrict__ w,
        const int8_t * __restrict__ xq,
        const float  * __restrict__ xscale,
        unsigned int   group_dim,
        unsigned int   rank,
        unsigned int   blocks_per_group,
        unsigned int   n_groups,
        unsigned int   batch) {
#include "q8_0_grouped_gemv_bpack_body.inc"
}
extern "C" __global__ void q8_0_grouped_gemv_bpack_ind(Q8_0_GROUPED_IND_PARAMS) {
    (void)canary;
    (void)canary_tag;
    Q8_0_GROUPED_IND_DEREF
    q8_0_grouped_gemv_bpack_rt_body(out, w, xq, xscale, group_dim, rank, bpg, n_groups, batch);
}
extern "C" __global__ void q8_0_grouped_gemv_bpack_ind_canary(Q8_0_GROUPED_IND_PARAMS) {
    Q8_0_GROUPED_IND_DEREF
    ARENA_CTX_CANARY_LOAD(canary, Q8_0_GROUPED_IND_XOR);
    q8_0_grouped_gemv_bpack_rt_body(out, w, xq, xscale, group_dim, rank, bpg, n_groups, batch);
    ARENA_CTX_CANARY_STORE(canary_tag);
}
