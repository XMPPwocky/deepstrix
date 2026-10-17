// Qwen3-Embedding forward, the small kernels around the GEMMs and attention
// (docs/v41/EMBED_PHASE_DESIGN.md §5.2). The projections run on the production
// Q8_0 x f16 WMMA GEMM (`q8_0_gemm_wmma_f16x`) and attention on
// `gqa_attn_prefill_flash_wmma_fa2` (or `qe_attn_g4`, §19); these kernels move
// activations between them. Every f32 -> f16 conversion is round-to-nearest
// (`__float2half_rn`, `qe_f16`).
//
// Row-major throughout ("pitch" = elements between consecutive rows), except the
// G4 attention's K / V tiles (`qe_k_frag_index`, `qe_v_frag`).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define QE_BLOCK 256u

// Largest Q8_0 row the in-place repack stages in LDS (blocks * 34 bytes):
// 480 blocks = K 15360 (Qwen3-Embedding-4B's widest row is K 9728, 10336 B).
#define QE_REPACK_MAX_BLOCKS 480u

// qe_f16: f32 -> f16, saturating to +-65504, NaN passthrough, RNE. Shared with the
// G4 attention kernel's epilogue (qwen3_attn_g4.hip), so its direct x16 store is
// bit-identical to qe_cast_f16 of the same f32 value.
#include "qwen3_embed_f16.inc"

// G4 attention layout (docs/v41/EMBED_PHASE_DESIGN.md §19.4): per kv head h and
// 32-key tile T of ABSOLUTE phase rows, 512 16-B chunks (4096 halfs) per buffer,
// tile offset ((size_t)h * n_tiles + T) * 4096 halfs:
//   kc chunk (ks*2+kb)*32 + L : half e = K[32T+16kb+(L&15)][16ks+8(L>>4)+e]
//   vc chunk (kb*8+dt)*32 + L : half e = V[32T+16kb+8(L>>4)+e][16dt+(L&15)]
// Host twins: `qwen3_embed::g4_k_frag_index` / `g4_v_frag_index`.
#define QE_G4_TILE_HALFS 4096u

// The kc index (in halfs) of K[R][i] of kv head kh (R = absolute phase row).
__device__ __forceinline__ size_t qe_k_frag_index(unsigned int kh, unsigned int R, unsigned int i, unsigned int n_tiles) {
    const unsigned int chunk = ((i >> 4) * 2u + ((R >> 4) & 1u)) * 32u + ((R & 15u) | (((i >> 3) & 1u) << 4));
    return ((size_t)kh * n_tiles + (R >> 5)) * QE_G4_TILE_HALFS + chunk * 8u + (i & 7u);
}

// In-place Q8_0 repack of `rows` rows of `blocks` blocks each, from the GGUF
// layout (per block: f16 scale, 32 x i8) to the split layout the WMMA GEMM
// reads (`gemm_f16x`; host twin `weights::repack_q8_0`): per row, all
// `blocks` scales, then all `blocks * 32` quants. Same bytes per row. One
// workgroup per row; the row is staged in LDS as u16 words (17 per block:
// the scale, then 16 words of quants), so rows must start 2-byte aligned.
extern "C" __global__ void qe_q8_0_repack_rows(
        unsigned short *w,
        unsigned int blocks) {
    __shared__ unsigned short s[QE_REPACK_MAX_BLOCKS * 17u];
    unsigned short *row = w + (size_t)blockIdx.x * blocks * 17u;
    const unsigned int words = blocks * 17u;
    for (unsigned int i = threadIdx.x; i < words; i += blockDim.x) s[i] = row[i];
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < words; i += blockDim.x) {
        if (i < blocks) {
            row[i] = s[i * 17u];
        } else {
            const unsigned int q = i - blocks;      // quant word index
            const unsigned int b = q / 16u;         // its block
            row[i] = s[b * 17u + 1u + (q % 16u)];
        }
    }
}

// Block-wide sum over QE_BLOCK threads (every thread gets the result).
__device__ __forceinline__ float qe_block_sum(float v, float *sdata) {
    const unsigned int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float r = sdata[0];
    __syncthreads();
    return r;
}

// out16[r, i] = f16(x[r, i] * rsqrt(mean_i(x[r, :]^2) + eps) * w[i]).
// Grid (rows), block QE_BLOCK. x is [rows, n] (pitch n).
extern "C" __global__ void qe_rmsnorm_f16(
        unsigned short *out16,
        unsigned int out_pitch,
        const float *x,
        const float *w,
        unsigned int n,
        float eps) {
    __shared__ float sdata[QE_BLOCK];
    const unsigned int r = blockIdx.x;
    const float *xr = x + (size_t)r * n;
    unsigned short *or_ = out16 + (size_t)r * out_pitch;
    double partial = 0.0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        partial += (double)v * (double)v;
    }
    const float ss = qe_block_sum((float)partial, sdata);
    const float scale = rsqrtf(ss / (float)n + eps);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        or_[i] = qe_f16(xr[i] * scale * w[i]);
    }
}

// Per (row, head) of the fused q|k|v GEMM output `qkv` [rows, (n_head +
// 2 n_kv_head) * head_dim]:
//   q heads: RMSNorm * q_norm, NeoX RoPE at pos[r]  -> q16[r, h, :]
//   k heads: RMSNorm * k_norm, NeoX RoPE at pos[r]  -> kc[row0 + r, h, :]
//   v heads: cast                                   -> vc[row0 + r, h, :]
// NeoX: element i < d/2 pairs with i + d/2, both rotated by
// pos * inv_freq[i], inv_freq[i] = 1 / theta^(2i/d) -- the HF numerics, in f32.
// Grid (rows, n_head + 2 n_kv_head), block = head_dim (<= QE_BLOCK, even,
// a power of two for the reduction).
//
// k_tiles > 0 (the G4 attention layout, head_dim 128): K goes to the
// fragment-native kc tiles (`qe_k_frag_index`, k_tiles tiles per kv head; 8
// consecutive threads write one 16-B run) and the grid is (rows, n_head +
// n_kv_head): V is written by `qe_v_frag`, so a V head here is a launch bug.
// Q is unchanged. k_tiles = 0: the row-major path, unchanged.
extern "C" __global__ void qe_qk_norm_rope(
        const float *qkv,
        unsigned short *q16,
        unsigned short *kc,
        unsigned short *vc,
        const float *q_norm,
        const float *k_norm,
        const unsigned int *pos,
        unsigned int row0,
        unsigned int n_head,
        unsigned int n_kv_head,
        unsigned int head_dim,
        float theta,
        float eps,
        unsigned int k_tiles) {
    __shared__ float sdata[QE_BLOCK];
    __shared__ float xs[QE_BLOCK];
    const unsigned int r = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int i = threadIdx.x;
    const unsigned int d = head_dim;
    const unsigned int half = d / 2u;
    const unsigned int width = (n_head + 2u * n_kv_head) * d;
    const float x = qkv[(size_t)r * width + (size_t)h * d + i];
    const size_t kv_width = (size_t)n_kv_head * d;
    if (h >= n_head + n_kv_head) {
        if (k_tiles != 0u) __builtin_trap();   // invariant: G4 mode launches no V heads
        const unsigned int vh = h - n_head - n_kv_head;
        vc[(size_t)(row0 + r) * kv_width + (size_t)vh * d + i] = qe_f16(x);
        return;
    }
    const bool is_q = h < n_head;
    const float ss = qe_block_sum(x * x, sdata);
    const float scale = rsqrtf(ss / (float)d + eps);
    const float wn = is_q ? q_norm[i] : k_norm[i];
    xs[i] = x * scale * wn;
    __syncthreads();
    const unsigned int j = i < half ? i : i - half;
    const float inv_freq = 1.0f / powf(theta, (float)(2u * j) / (float)d);
    const float a = (float)pos[r] * inv_freq;
    float s, c;
    sincosf(a, &s, &c);
    const float y = i < half ? xs[i] * c - xs[i + half] * s
                             : xs[i] * c + xs[i - half] * s;
    if (is_q) {
        q16[(size_t)r * n_head * d + (size_t)h * d + i] = qe_f16(y);
    } else {
        const unsigned int kh = h - n_head;
        if (k_tiles != 0u) {
            kc[qe_k_frag_index(kh, row0 + r, i, k_tiles)] = qe_f16(y);
        } else {
            kc[(size_t)(row0 + r) * kv_width + (size_t)kh * d + i] = qe_f16(y);
        }
    }
}

// G4 V producer: the f32 V columns of the fused qkv GEMM output (phase rows
// row0 .. row0 + rows) -> qe_f16 -> fragment-native vc tiles (layout above).
// Grid (tiles touched by [row0, row0 + rows), n_kv_head), block 256; head_dim 128.
// Tile T = (row0 >> 5) + blockIdx.x. Keys of T below row0 were written by the
// previous sub-batch of this layer (same stream): the straddled first tile is
// read-modify-written. Keys >= row0 + rows are written 0: the next sub-batch
// rewrites them, and the phase-end pad stays 0. Host twin: `g4_v_frag_host`.
extern "C" __global__ __launch_bounds__(256) void qe_v_frag(
        uint4       * __restrict__ vc,
        const float * __restrict__ qkv,       // [rows][(n_head + 2 n_kv_head) * 128] f32
        unsigned int n_head,
        unsigned int n_kv_head,
        unsigned int n_tiles,
        unsigned int row0,
        unsigned int rows) {
    __shared__ unsigned short T[32u * 136u];  // [key][d] f16 bits, +8 pad
    if (blockDim.x != 256u) __builtin_trap();
    const unsigned int tid = threadIdx.x, vh = blockIdx.y;
    const unsigned int tile = (row0 >> 5) + blockIdx.x;
    const unsigned int width = (n_head + 2u * n_kv_head) * 128u;
    const unsigned int rl = tid >> 3, d0 = (tid & 7u) * 16u;
    const unsigned int R = tile * 32u + rl;
    float x[16];
    if (R >= row0 && R < row0 + rows) {
        const float *src = qkv + (size_t)(R - row0) * width + (size_t)(n_head + n_kv_head + vh) * 128u + d0;
#pragma unroll
        for (int i = 0; i < 16; i += 4) *(float4 *)&x[i] = *(const float4 *)(src + i);
    } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) x[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) T[rl * 136u + d0 + (unsigned int)i] = qe_f16(x[i]);
    __syncthreads();
    uint4 *tb = vc + ((size_t)vh * n_tiles + tile) * 512u;
    const bool keep_old = tile * 32u < row0;  // block-uniform: the straddled first tile
#pragma unroll
    for (unsigned int i = 0; i < 2u; ++i) {
        const unsigned int cid = tid + i * 256u;
        const unsigned int kb = cid >> 8, dt = (cid >> 5) & 7u, L = cid & 31u;
        const unsigned int key0 = kb * 16u + (L >> 4) * 8u, d = dt * 16u + (L & 15u);
        unsigned int hv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) hv[e] = T[(key0 + (unsigned int)e) * 136u + d];
        if (keep_old) {
            const uint4 old = tb[cid];
            const unsigned int ow[4] = {old.x, old.y, old.z, old.w};
#pragma unroll
            for (int e = 0; e < 8; ++e)
                if (tile * 32u + key0 + (unsigned int)e < row0) hv[e] = (ow[e >> 1] >> ((e & 1) * 16)) & 0xffffu;
        }
        uint4 o;
        o.x = hv[0] | (hv[1] << 16);
        o.y = hv[2] | (hv[3] << 16);
        o.z = hv[4] | (hv[5] << 16);
        o.w = hv[6] | (hv[7] << 16);
        tb[cid] = o;
    }
}

// out16[r, i] = f16(x[r, i]), x [rows, n]. Grid (ceil(n / QE_BLOCK), rows).
extern "C" __global__ void qe_cast_f16(
        unsigned short *out16,
        unsigned int out_pitch,
        const float *x,
        unsigned int n) {
    const unsigned int r = blockIdx.y;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out16[(size_t)r * out_pitch + i] = qe_f16(x[(size_t)r * n + i]);
}

// out16[r, i] = f16(silu(g) * u), g = gu[r, i], u = gu[r, n_ff + i] (the
// fused gate|up GEMM output, pitch 2 n_ff). Grid (ceil(n_ff / QE_BLOCK), rows).
extern "C" __global__ void qe_swiglu_f16(
        unsigned short *out16,
        unsigned int out_pitch,
        const float *gu,
        unsigned int n_ff) {
    const unsigned int r = blockIdx.y;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ff) return;
    const float *row = gu + (size_t)r * 2u * n_ff;
    const float g = row[i];
    const float u = row[n_ff + i];
    out16[(size_t)r * out_pitch + i] = qe_f16(g / (1.0f + expf(-g)) * u);
}

// resid[i] += delta[i], i < n. Grid-stride.
extern "C" __global__ void qe_add(
        float *resid,
        const float *delta,
        unsigned int n) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        resid[i] += delta[i];
    }
}

// out[c, :] = src[idx[c], :], rows of n. Grid (ceil(n / QE_BLOCK), count).
extern "C" __global__ void qe_gather_rows(
        float *out,
        const float *src,
        const unsigned int *idx,
        unsigned int n) {
    const unsigned int c = blockIdx.y;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[(size_t)c * n + i] = src[(size_t)idx[c] * n + i];
}
