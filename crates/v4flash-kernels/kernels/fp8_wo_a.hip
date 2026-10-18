// fp8_wo_a -- `V41_ATTN_O_FP8`: the attention output projection's first stage (`wo_a`, block-
// diagonal over 8 groups: [8 x 1024 rows, 4096 dims]) as the 1017 reference computes it, FP8 x FP8:
//
//   o      = act_quant(o, 32, "ue8m0")      e4m3 codes + one power-of-two scale per 32 dims
//            (act_quant_fp8.inc; produced by the epilogue that used to emit the Q8_0 input:
//            rope_inv_quant_fp8, attn_dec_g3 / attn_dec_g3o, or act_quant_fp8_wave below)
//   wo_a   = the checkpoint's e4m3 weight + its e8m0 scale per 32 x 32 block, AS SHIPPED
//            (one device buffer: [8192 x 4096] codes, then the [256 x 128] scale bytes)
//   out[b][r] = sum_kb s_w[r/32][kb] * s_a[b][g, kb] * sum_{k in kb} w[r][k] * o[b][g, k]
//
// The reference dequantizes both operands to bf16 (exact: <= 4 significant bits times a power of
// two) and runs a bf16 einsum, f32 accumulation. Here every product is exact (v_dot4_f32_fp8_fp8 /
// v_wmma_f32_16x16x16_fp8_fp8 multiply the codes exactly) and the scales are powers of two, so the
// result is the reference's sum up to f32 summation order (tests/fp8_wo_a.rs bounds it).
//
// Kernels:
//   act_quant_fp8_wave       x f32 -> codes + scales, one wave per 32-block (q8_0_quantize_f32_wave's
//                            geometry): the wo_a input where no fused epilogue makes it.
//   q8_0_quantize_o8_wave    the DSpark drafter (iGPU, no FP8 dot): x -> act_quant rounding ->
//                            q8_0_quantize_f32_wave's Q8_0 of the rounded value, one launch.
//   fp8_grouped_gemv_tB<B>   decode: one wave per output row, B <= 8 rows of o per launch
//   fp8_grouped_gemv_rt      decode, any B: runtime batch loop, 16 rows per grid.z slice
//   fp8_grouped_gemm_wmma    prefill: 128 x 128 x 32 tiles of FP8 WMMA, scales applied per 32-block
// gfx12 only for the dot / WMMA kernels (gfx1151 has no FP8 dot: compiled stubs); the two quantize
// kernels build for both archs.

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "act_quant_fp8.inc"
#include "arena_ctx.inc"

#define FW_FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull

// ------------------------------------------------------------------------------------------------
// activation quantize
// ------------------------------------------------------------------------------------------------
// grid ceil(blocks / 8) [+ pad] x 256; lane i of wave w owns element i of block 8 * blockIdx.x + w.
__device__ __forceinline__ void act_quant_fp8_wave_body(uint8_t* __restrict__ xq, float* __restrict__ xscale,
                                                        const float* __restrict__ x, uint32_t blocks) {
    const uint32_t b = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (b >= blocks) return;
    const float v = x[(size_t)b * 32u + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FW_FULL_WARP_MASK, amax, off));
    const int e = aq_scale_exp(amax);
    if (lane == 0) xscale[b] = aq_pow2(e);
    xq[(size_t)b * 32u + lane] = (uint8_t)aq_e4m3_code(v, e);
}
extern "C" __global__ __launch_bounds__(256) void act_quant_fp8_wave(
        uint8_t* __restrict__ xq, float* __restrict__ xscale, const float* __restrict__ x, uint32_t blocks) {
    act_quant_fp8_wave_body(xq, xscale, x, blocks);
}

// The drafter's wo_a input: the reference's rounding, then q8_0_quantize_f32_wave's block math
// (d = amax / 127, id = 1 / d, lrintf, clamp) on the rounded values -- what
// `act_quant_inplace` followed by `quantize_input_batched` would write, in one launch.
extern "C" __global__ __launch_bounds__(256) void q8_0_quantize_o8_wave(
        int8_t* __restrict__ xq, float* __restrict__ xscale, const float* __restrict__ x, uint32_t blocks) {
    const uint32_t b = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (b >= blocks) return;
    const float v0 = x[(size_t)b * 32u + lane];
    float a0 = fabsf(v0);
    for (int off = 16; off > 0; off >>= 1) a0 = fmaxf(a0, __shfl_xor_sync(FW_FULL_WARP_MASK, a0, off));
    const int e = aq_scale_exp(a0);
    const float v = aq_e4m3_value(aq_e4m3_code(v0, e)) * aq_pow2(e);
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FW_FULL_WARP_MASK, amax, off));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0) xscale[b] = d;
    int q = (int)lrintf(v * id);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    xq[(size_t)b * 32u + lane] = (int8_t)q;
}

// Test hook (tests/fp8_wo_a.rs): the in-place rounding alone, x <- dec(code) * 2^e per 32-block,
// so the device arithmetic can be compared with `act_quant_inplace` bit for bit.
extern "C" __global__ __launch_bounds__(256) void act_quant_fp8_inplace_wave(float* __restrict__ x, uint32_t blocks) {
    const uint32_t b = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (b >= blocks) return;
    const float v = x[(size_t)b * 32u + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FW_FULL_WARP_MASK, amax, off));
    const int e = aq_scale_exp(amax);
    x[(size_t)b * 32u + lane] = aq_e4m3_value(aq_e4m3_code(v, e)) * aq_pow2(e);
}

#if defined(__gfx1200__) || defined(__gfx1201__)
#define FW_GFX12 1
#endif

// ------------------------------------------------------------------------------------------------
// decode: grouped GEMV, one wave per output row (q8_0_grouped_gemv_bpack_tb_body's geometry:
// grid (out_dim / 8) x 256, lanes stride the 32-blocks of the row's group)
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float fw_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(FW_FULL_WARP_MASK, v, offset);
    return v;
}

#ifdef FW_GFX12
// sum of the 32 exact products of one block: 8 x v_dot4_f32_fp8_fp8 chained from 0
__device__ __forceinline__ float fw_dot32(const uint4& w0, const uint4& w1, const uint4& a0, const uint4& a1) {
    float p = 0.0f;
    p = __builtin_amdgcn_dot4_f32_fp8_fp8((int)w0.x, (int)a0.x, p);
    p = __builtin_amdgcn_dot4_f32_fp8_fp8((int)w0.y, (int)a0.y, p);
    p = __builtin_amdgcn_dot4_f32_fp8_fp8((int)w0.z, (int)a0.z, p);
    p = __builtin_amdgcn_dot4_f32_fp8_fp8((int)w0.w, (int)a0.w, p);
    p = __builtin_amdgcn_dot4_f32_fp8_fp8((int)w1.x, (int)a1.x, p);
    p = __builtin_amdgcn_dot4_f32_fp8_fp8((int)w1.y, (int)a1.y, p);
    p = __builtin_amdgcn_dot4_f32_fp8_fp8((int)w1.z, (int)a1.z, p);
    p = __builtin_amdgcn_dot4_f32_fp8_fp8((int)w1.w, (int)a1.w, p);
    return p;
}

// B rows of o at compile time (all activation loads issued with the weight loads, as tB in
// q8_0_grouped_matvec.hip). `w`: codes [out_dim, group_dim] then scales [out_dim / 32, group_dim / 32].
template <int B>
__device__ __forceinline__ void fw_gemv_tb_body(float* __restrict__ out, const uint8_t* __restrict__ w,
                                                const uint8_t* __restrict__ xq, const float* __restrict__ xscale,
                                                unsigned group_dim, unsigned rank, unsigned n_groups) {
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;
    const uint32_t g = row / rank, bpg = group_dim >> 5;
    const uint8_t* wr = w + (size_t)row * group_dim;
    const uint8_t* ws = w + (size_t)out_dim * group_dim + (size_t)(row >> 5) * bpg;
    const uint32_t xq_stride = n_groups * group_dim, xs_stride = n_groups * bpg;
    const uint8_t* xq_g = xq + (size_t)g * group_dim;
    const float* xs_g = xscale + (size_t)g * bpg;
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;
    for (uint32_t bl = lane; bl < bpg; bl += 32u) {
        uint4 w0, w1;
        __builtin_memcpy(&w0, wr + (size_t)bl * 32u, 16);
        __builtin_memcpy(&w1, wr + (size_t)bl * 32u + 16, 16);
        const float sw = aq_e8m0_value(ws[bl]);
        uint4 a0[B], a1[B];
        float sa[B];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const uint8_t* xb = xq_g + (size_t)b * xq_stride + (size_t)bl * 32u;
            __builtin_memcpy(&a0[b], xb, 16);
            __builtin_memcpy(&a1[b], xb + 16, 16);
            sa[b] = xs_g[(size_t)b * xs_stride + bl];
        }
#pragma unroll
        for (int b = 0; b < B; ++b) acc[b] += (sw * sa[b]) * fw_dot32(w0, w1, a0[b], a1[b]);
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        const float a = fw_warp_sum(acc[b]);
        if (lane == 0) out[(size_t)b * out_dim + row] = a;
    }
}

#define FW_GEMV_MAX_B 16u
// any B: grid.z slices of 16 rows, the batch loop inside (the weight is read once per slice)
__device__ __forceinline__ void fw_gemv_rt_body(float* __restrict__ out, const uint8_t* __restrict__ w,
                                                const uint8_t* __restrict__ xq, const float* __restrict__ xscale,
                                                unsigned group_dim, unsigned rank, unsigned n_groups, unsigned batch) {
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t out_dim = n_groups * rank;
    if (row >= out_dim) return;
    const uint32_t b0 = blockIdx.z * FW_GEMV_MAX_B;
    if (b0 >= batch) return;
    const uint32_t nb = (batch - b0 < FW_GEMV_MAX_B) ? (batch - b0) : FW_GEMV_MAX_B;
    const uint32_t g = row / rank, bpg = group_dim >> 5;
    const uint32_t xq_stride = n_groups * group_dim, xs_stride = n_groups * bpg;
    xq += (size_t)b0 * xq_stride;
    xscale += (size_t)b0 * xs_stride;
    out += (size_t)b0 * out_dim;
    const uint8_t* wr = w + (size_t)row * group_dim;
    const uint8_t* ws = w + (size_t)out_dim * group_dim + (size_t)(row >> 5) * bpg;
    float acc[FW_GEMV_MAX_B];
    for (uint32_t b = 0; b < nb; ++b) acc[b] = 0.0f;
    for (uint32_t bl = lane; bl < bpg; bl += 32u) {
        uint4 w0, w1;
        __builtin_memcpy(&w0, wr + (size_t)bl * 32u, 16);
        __builtin_memcpy(&w1, wr + (size_t)bl * 32u + 16, 16);
        const float sw = aq_e8m0_value(ws[bl]);
        for (uint32_t b = 0; b < nb; ++b) {
            const uint8_t* xb = xq + (size_t)b * xq_stride + (size_t)g * group_dim + (size_t)bl * 32u;
            uint4 a0, a1;
            __builtin_memcpy(&a0, xb, 16);
            __builtin_memcpy(&a1, xb + 16, 16);
            const float sa = xscale[(size_t)b * xs_stride + (size_t)g * bpg + bl];
            acc[b] += (sw * sa) * fw_dot32(w0, w1, a0, a1);
        }
    }
    for (uint32_t b = 0; b < nb; ++b) {
        const float a = fw_warp_sum(acc[b]);
        if (lane == 0) out[(size_t)b * out_dim + row] = a;
    }
}
#endif  // FW_GFX12

#ifdef FW_GFX12
#define FW_BODY(...) __VA_ARGS__
#else
#define FW_BODY(...) (void)out; (void)w; (void)xq; (void)xscale; (void)group_dim; (void)rank; (void)n_groups;
#endif

#define FW_GEMV_TB(NAME, B)                                                                       \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                       \
            float* __restrict__ out, const uint8_t* __restrict__ w, const uint8_t* __restrict__ xq, \
            const float* __restrict__ xscale, unsigned group_dim, unsigned rank, unsigned n_groups, \
            unsigned batch) {                                                                     \
        (void)batch;                                                                              \
        FW_BODY(fw_gemv_tb_body<B>(out, w, xq, xscale, group_dim, rank, n_groups);)               \
    }
FW_GEMV_TB(fp8_grouped_gemv_tB1, 1)
FW_GEMV_TB(fp8_grouped_gemv_tB2, 2)
FW_GEMV_TB(fp8_grouped_gemv_tB3, 3)
FW_GEMV_TB(fp8_grouped_gemv_tB4, 4)
FW_GEMV_TB(fp8_grouped_gemv_tB5, 5)
FW_GEMV_TB(fp8_grouped_gemv_tB6, 6)
FW_GEMV_TB(fp8_grouped_gemv_tB7, 7)
FW_GEMV_TB(fp8_grouped_gemv_tB8, 8)

extern "C" __global__ __launch_bounds__(256) void fp8_grouped_gemv_rt(
        float* __restrict__ out, const uint8_t* __restrict__ w, const uint8_t* __restrict__ xq,
        const float* __restrict__ xscale, unsigned group_dim, unsigned rank, unsigned n_groups, unsigned batch) {
    FW_BODY(fw_gemv_rt_body(out, w, xq, xscale, group_dim, rank, n_groups, batch);)
    (void)batch;
}

// `_ind` twins (docs/v41/GRAPH_KEYS_DESIGN.md 2.3): operands 0..3 (out, w, xq, xscale) resolve
// through the arena context when their `ind_mask` bit is set; same bodies.
#define FW_IND_PARAMS                                                                             \
    uint32_t ind_mask, const ArenaCtx* canary, uint32_t canary_tag, float* out, const uint8_t* w, \
    const uint8_t* xq, const float* xscale, unsigned group_dim, unsigned rank, unsigned n_groups, \
    unsigned batch
#define FW_IND_DEREF                                                                              \
    ARENA_CTX_DEREF(ind_mask, 0, float*, out);                                                    \
    ARENA_CTX_DEREF(ind_mask, 1, const uint8_t*, w);                                              \
    ARENA_CTX_DEREF(ind_mask, 2, const uint8_t*, xq);                                             \
    ARENA_CTX_DEREF(ind_mask, 3, const float*, xscale);
#define FW_IND_XOR (ARENA_CTX_U64(out) ^ ARENA_CTX_U64(w) ^ ARENA_CTX_U64(xq) ^ ARENA_CTX_U64(xscale))
#define FW_GEMV_IND(NAME, CALL)                                                                   \
    extern "C" __global__ __launch_bounds__(256) void NAME(FW_IND_PARAMS) {                       \
        (void)canary; (void)canary_tag; (void)batch;                                              \
        FW_IND_DEREF                                                                              \
        FW_BODY(CALL;)                                                                            \
    }                                                                                             \
    extern "C" __global__ __launch_bounds__(256) void NAME##_canary(FW_IND_PARAMS) {              \
        (void)batch;                                                                              \
        FW_IND_DEREF                                                                              \
        ARENA_CTX_CANARY_LOAD(canary, FW_IND_XOR);                                                \
        FW_BODY(CALL;)                                                                            \
        ARENA_CTX_CANARY_STORE(canary_tag);                                                       \
    }
FW_GEMV_IND(fp8_grouped_gemv_tB1_ind, fw_gemv_tb_body<1>(out, w, xq, xscale, group_dim, rank, n_groups))
FW_GEMV_IND(fp8_grouped_gemv_tB2_ind, fw_gemv_tb_body<2>(out, w, xq, xscale, group_dim, rank, n_groups))
FW_GEMV_IND(fp8_grouped_gemv_tB3_ind, fw_gemv_tb_body<3>(out, w, xq, xscale, group_dim, rank, n_groups))
FW_GEMV_IND(fp8_grouped_gemv_tB4_ind, fw_gemv_tb_body<4>(out, w, xq, xscale, group_dim, rank, n_groups))
FW_GEMV_IND(fp8_grouped_gemv_tB5_ind, fw_gemv_tb_body<5>(out, w, xq, xscale, group_dim, rank, n_groups))
FW_GEMV_IND(fp8_grouped_gemv_tB6_ind, fw_gemv_tb_body<6>(out, w, xq, xscale, group_dim, rank, n_groups))
FW_GEMV_IND(fp8_grouped_gemv_tB7_ind, fw_gemv_tb_body<7>(out, w, xq, xscale, group_dim, rank, n_groups))
FW_GEMV_IND(fp8_grouped_gemv_tB8_ind, fw_gemv_tb_body<8>(out, w, xq, xscale, group_dim, rank, n_groups))
FW_GEMV_IND(fp8_grouped_gemv_rt_ind, fw_gemv_rt_body(out, w, xq, xscale, group_dim, rank, n_groups, batch))

// ------------------------------------------------------------------------------------------------
// prefill: grouped GEMM, FP8 WMMA. Tile 128 weight rows x 128 rows of o x 32 dims (one scale block)
// per k step, 8 waves as 4 (rows) x 2 (o rows): wave tile 32 x 64 = 2 x 4 WMMA tiles of 16 x 16.
// Per 32-block: tmp = WMMA(k 0..15) then WMMA(k 16..31) from zero, acc += tmp * (s_w * s_a): a wave's
// 32 weight rows are ONE scale block (s_w uniform per wave), a lane's C column is one row of o
// (s_a one scalar per lane and n tile). LDS double-buffered (2 x 12.5 KB), next tile's global loads
// in flight across the compute. Grid (ceil(batch / 128), rank / 128, n_groups) x 256.
//
// Fragment layout: lane l holds row (l & 15) of its 16-row A tile / column (l & 15) of its B tile and
// 8 consecutive k bytes at (l >> 4) * 8 -- the f16 WMMA's pattern (q8_0_matvec_wmma.hip f16x_core).
// The sum inside one WMMA runs over all 16 k, so ANY k -> (lane half, byte) assignment the hardware
// uses gives the same products as long as A and B use the same one (they do: both are loaded with
// this pattern), i.e. only the row / column mapping matters, and that is the f16 one.
// ------------------------------------------------------------------------------------------------
typedef int   fw_v2i __attribute__((ext_vector_type(2)));
typedef float fw_v8f __attribute__((ext_vector_type(8)));

#define FWG_BM 128u
#define FWG_BN 128u
#define FWG_LDS 48u   // bytes per staged row (32 + 16 pad: 16-B aligned stores, ds_read_b64 spread over banks)

__device__ __forceinline__ void fwg_sync_lds() {
#ifdef FW_GFX12
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}

extern "C" __global__ __launch_bounds__(256) void fp8_grouped_gemm_wmma(
        float* __restrict__ out,             // [batch, n_groups * M]
        const uint8_t* __restrict__ w,       // codes [n_groups * M, K] then scales [n_groups * M / 32, K / 32]
        const uint8_t* __restrict__ xq,      // [batch, n_groups * K] codes
        const float* __restrict__ xscale,    // [batch, n_groups * K / 32]
        unsigned K, unsigned M, unsigned n_groups, unsigned batch) {
#ifdef FW_GFX12
    __shared__ __attribute__((aligned(16))) uint8_t A_t[2][FWG_BM * FWG_LDS];
    __shared__ __attribute__((aligned(16))) uint8_t B_t[2][FWG_BN * FWG_LDS];
    __shared__ float sa_t[2][FWG_BN];
    __shared__ float sw_t[2][FWG_BM / 32u];

    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
    const uint32_t wave_m0 = (wave >> 1) * 32u;   // 4 x 32 weight rows
    const uint32_t wave_n0 = (wave & 1u) * 64u;   // 2 x 64 rows of o
    const uint32_t n0 = blockIdx.x * FWG_BN, m0 = blockIdx.y * FWG_BM, g = blockIdx.z;
    const uint32_t kbn = K >> 5, out_dim = n_groups * M;
    const uint8_t* wq = w + (size_t)(g * M + m0) * K;
    const uint8_t* ws = w + (size_t)out_dim * K + (size_t)((g * M + m0) >> 5) * kbn;
    // staging roles: thread t moves 16 bytes of row t >> 1 (half t & 1) of both tiles
    const uint32_t srow = tid >> 1, shalf = (tid & 1u) * 16u;
    const uint8_t* a_src = wq + (size_t)srow * K + shalf;
    // an o row past the batch reads the last valid row (never stored)
    const uint32_t tok = (n0 + srow < batch) ? n0 + srow : batch - 1u;
    const uint8_t* b_src = xq + (size_t)tok * (n_groups * K) + (size_t)g * K + shalf;
    const float* bs_src = xscale + (size_t)tok * (n_groups * kbn) + (size_t)g * kbn;

    uint4 ra, rb;
    float rs = 0.0f, rw = 0.0f;
    auto load = [&](uint32_t kb) {
        __builtin_memcpy(&ra, a_src + (size_t)kb * 32u, 16);
        __builtin_memcpy(&rb, b_src + (size_t)kb * 32u, 16);
        if ((tid & 1u) == 0u) rs = bs_src[kb];                        // one scale per o row
        if (tid < FWG_BM / 32u) rw = aq_e8m0_value(ws[(size_t)tid * kbn + kb]);
    };
    auto stage = [&](uint32_t buf) {
        *(uint4*)(&A_t[buf][srow * FWG_LDS + shalf]) = ra;
        *(uint4*)(&B_t[buf][srow * FWG_LDS + shalf]) = rb;
        if ((tid & 1u) == 0u) sa_t[buf][srow] = rs;
        if (tid < FWG_BM / 32u) sw_t[buf][tid] = rw;
    };

    fw_v8f acc[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) acc[mt][nt] = (fw_v8f){0, 0, 0, 0, 0, 0, 0, 0};

    load(0);
    stage(0);
    fwg_sync_lds();
    for (uint32_t kb = 0; kb < kbn; ++kb) {
        const uint32_t buf = kb & 1u;
        if (kb + 1u < kbn) load(kb + 1u);
        fw_v2i af[2][2], bf[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t kf = (uint32_t)h * 16u + (lane >> 4) * 8u;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                __builtin_memcpy(&af[h][mt], &A_t[buf][(wave_m0 + mt * 16u + (lane & 15u)) * FWG_LDS + kf], 8);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
                __builtin_memcpy(&bf[h][nt], &B_t[buf][(wave_n0 + nt * 16u + (lane & 15u)) * FWG_LDS + kf], 8);
        }
        const float sw = sw_t[buf][wave_m0 >> 5];
        float sa[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) sa[nt] = sa_t[buf][wave_n0 + nt * 16u + (lane & 15u)];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                fw_v8f t = (fw_v8f){0, 0, 0, 0, 0, 0, 0, 0};
                t = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[0][mt], bf[0][nt], t);
                t = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[1][mt], bf[1][nt], t);
                const float sc = sw * sa[nt];
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][nt][i] += t[i] * sc;
            }
        if (kb + 1u < kbn) stage(buf ^ 1u);
        fwg_sync_lds();
    }

#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const uint32_t n_idx = n0 + wave_n0 + nt * 16u + (lane & 15u);
        if (n_idx >= batch) continue;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const uint32_t m_idx = m0 + wave_m0 + mt * 16u + (lane >> 4) * 8u;
            float* dst = out + (size_t)n_idx * out_dim + (size_t)g * M + m_idx;
            const fw_v8f c = acc[mt][nt];
            const float4 c0 = {c[0], c[1], c[2], c[3]};
            const float4 c1 = {c[4], c[5], c[6], c[7]};
            __builtin_memcpy(dst, &c0, 16);
            __builtin_memcpy(dst + 4, &c1, 16);
        }
    }
#else
    (void)out; (void)w; (void)xq; (void)xscale; (void)K; (void)M; (void)n_groups; (void)batch;
#endif
}
