#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Q8_0 int8-WMMA GEMM:  out[B,M] = W[M,K] * Xq[B,K]^T  with per-32-block
// dequant scales folded into the f16 WMMA operands at load time.
//
//   W      : Q8_0 weights, row-major. Each of the M rows is
//            [scales: blocks * f16] [quants: blocks * 32 * int8]; the row
//            pitch is blocks*34 bytes. wscale[m,bl] is f16.
//   Xq     : activations pre-quantized to int8, [B, K] row-major.
//   xscale : per-(batch,block) f32 activation scale, [B, blocks].
//
//   out[n,m] = sum_k (qW[m,k]*wscale[m,k/32]) * (qX[n,k]*xscale[n,k/32])
//
// Folding both dequant scales into the f16 operands means the f32 WMMA
// accumulator can run across the WHOLE K with no per-block rescale: the
// scale of any (m,k)*(k,n) product is exactly wscale[m,bl]*xscale[n,bl],
// which is what Q8_0 dequant requires. (Within one 16-wide K-tile the
// block index k/32 is constant, since k0 is a multiple of 16, so each
// lane uses a single weight/activation scale for its 8 fragment elems.)
//
// Layout: one wave per 16-row M-tile; it loops the batch N-tiles *inside*
// each K-tile so the weight A-fragment is loaded ONCE per K-tile and
// reused across all batch columns (the whole point — weight read once
// instead of B times as in the grid.z=B matvec).
//
// gfx12 (RDNA4) wave32 16x16x16 f16 fragment layout (validated in
// wmma_wsum.hip via wmma_layout_probe):
//   A[m,k]: lane l holds a[e] = A[(l&15)][ (l>>4)*8 + e ]   e in 0..7
//   B[k,n]: lane l holds b[e] = B[ (l>>4)*8 + e ][(l&15)]   e in 0..7
//   C[m,n]: lane l holds c[e] = C[ e*2 + (l>>4) ][(l&15)]   e in 0..7
//
// Only ever runs on the dGPU (gfx1201); other arches get a scalar
// fallback so build.rs's per-arch compile (gfx1151) still succeeds.

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

// N-tiles processed per chunk. The accumulator array acc[] is unrolled over
// this many tiles and MUST be a compile-time index (a runtime index into the
// WMMA accumulator VGPRs wedges the wave), so this bounds per-chunk register
// pressure (8 * float8 = 64 VGPRs of accumulator). Batches wider than
// 8*16=128 columns are handled by looping this chunk at runtime (see nbase).
#define Q8WMMA_MAX_NTILES 8

// blockDim.x == 32 (one wave per M-tile); gridDim.x == ceil(M/16).
extern "C" __global__ void q8_0_gemm_wmma_i8(
    float* __restrict__ out,               // [B, M]
    const unsigned char* __restrict__ w,   // [M, blocks*34]
    const int8_t* __restrict__ xq,         // [B, K]
    const float* __restrict__ xscale,      // [B, blocks]
    uint32_t in_dim,                       // K
    uint32_t out_dim,                      // M
    uint32_t batch,                        // B
    uint32_t blocks)                       // K / 32
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t m0   = blockIdx.x * 16u;
    if (m0 >= out_dim) return;

    const uint32_t n_tiles = (batch + 15u) / 16u;
    const uint32_t arow = lane & 15u;        // A-row (m) and B-col (n) index
    const uint32_t ko   = (lane >> 4) * 8u;  // K half: lanes 0..15 -> 0, 16..31 -> 8

    const size_t rowbytes   = (size_t)blocks * 34u;
    const size_t scalebytes = (size_t)blocks * 2u;

    // This lane's weight row (for the A-operand).
    const uint32_t wm = m0 + arow;
    const int8_t* wq_row  = (const int8_t*)(w + (size_t)wm * rowbytes + scalebytes);
    const __half* wsc_row = (const __half*)(w + (size_t)wm * rowbytes);
    const bool wm_valid = (wm < out_dim);

    // Process batch N-tiles in chunks of Q8WMMA_MAX_NTILES so acc[] stays a
    // compile-time index. For B<=128 this outer loop runs once (identical to
    // the original single-pass kernel). For wider batches it runs again with
    // nbase advanced, re-reading the weight A-fragment once per chunk — still a
    // huge BW win over dp4a's per-batch-element weight reload (2 reads at B=256
    // vs 256).
    for (uint32_t nbase = 0; nbase < n_tiles; nbase += Q8WMMA_MAX_NTILES) {
        float8 acc[Q8WMMA_MAX_NTILES];
        #pragma unroll
        for (uint32_t t = 0; t < Q8WMMA_MAX_NTILES; ++t)
            acc[t] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

        for (uint32_t k0 = 0; k0 < in_dim; k0 += 16u) {
            const uint32_t bl = k0 >> 5;   // block index (constant within this K-tile)

            // A-fragment: dequantized weight row, scale baked in (loaded ONCE
            // per chunk; reused across all N-tiles in this chunk).
            const float wscl = wm_valid ? __half2float(wsc_row[bl]) : 0.0f;
            half8 a;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t g = k0 + ko + (uint32_t)e;
                const float q = (wm_valid && g < in_dim) ? (float)wq_row[g] : 0.0f;
                a[e] = (_Float16)(q * wscl);
            }

            // Constant-unrolled over Q8WMMA_MAX_NTILES (NOT n_tiles): the WMMA
            // builtin's accumulator operand must live in fixed VGPRs, so acc[t]
            // has to be a compile-time index. A runtime bound forces
            // indexed-register access the matrix unit can't satisfy and wedges
            // the wave — hence the unroll + runtime skip guard.
            #pragma unroll
            for (uint32_t t = 0; t < Q8WMMA_MAX_NTILES; ++t) {
                const uint32_t tile = nbase + t;
                if (tile >= n_tiles) continue;
                const uint32_t nn = tile * 16u + arow;
                const bool nn_valid = (nn < batch);
                const int8_t* xrow = xq + (size_t)nn * in_dim;
                const float xscl = nn_valid ? xscale[(size_t)nn * blocks + bl] : 0.0f;
                half8 b;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const uint32_t g = k0 + ko + (uint32_t)e;
                    const float q = (nn_valid && g < in_dim) ? (float)xrow[g] : 0.0f;
                    b[e] = (_Float16)(q * xscl);
                }
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[t]);
            }
        }

        // C[m,n] fragment (validated in wsum_wmma_f16 against a host f32 ref):
        //   lane l holds c[e] = C[ (l>>4)*8 + e ][ (l&15) ], e in 0..7.
        // -> C-row (m) = m0 + (lane>>4)*8 + e ; C-col (n) = tile*16 + (lane&15).
        #pragma unroll
        for (uint32_t t = 0; t < Q8WMMA_MAX_NTILES; ++t) {
            const uint32_t tile = nbase + t;
            if (tile >= n_tiles) continue;
            const uint32_t gn = tile * 16u + (lane & 15u);
            if (gn >= batch) continue;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t gm = m0 + (lane >> 4) * 8u + (uint32_t)e;
                if (gm < out_dim) out[(size_t)gn * out_dim + gm] = acc[t][e];
            }
        }
    }
#else
    // Scalar fallback (non-gfx12): correct but unoptimized. Never runs on
    // the iGPU for qb; present only so build.rs's gfx1151 compile succeeds.
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t m0   = blockIdx.x * 16u;
    const size_t rowbytes   = (size_t)blocks * 34u;
    const size_t scalebytes = (size_t)blocks * 2u;
    for (uint32_t idx = lane; idx < 16u * batch; idx += 32u) {
        const uint32_t m = idx % 16u;
        const uint32_t n = idx / 16u;
        const uint32_t mm = m0 + m;
        if (mm >= out_dim || n >= batch) continue;
        const int8_t* wq_row  = (const int8_t*)(w + (size_t)mm * rowbytes + scalebytes);
        const __half* wsc_row = (const __half*)(w + (size_t)mm * rowbytes);
        const int8_t* xrow    = xq + (size_t)n * in_dim;
        float acc = 0.0f;
        for (uint32_t bl = 0; bl < blocks; ++bl) {
            int32_t dot = 0;
            const uint32_t k0 = bl * 32u;
            for (uint32_t k = 0; k < 32u; ++k)
                dot += (int32_t)wq_row[k0 + k] * (int32_t)xrow[k0 + k];
            acc += __half2float(wsc_row[bl]) * xscale[(size_t)n * blocks + bl] * (float)dot;
        }
        out[(size_t)n * out_dim + mm] = acc;
    }
#endif
}
