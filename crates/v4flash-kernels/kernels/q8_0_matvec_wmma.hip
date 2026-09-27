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

// LDS-tiled Q8_0 GEMM with WMMA f16 (RDNA4 only).
//
// Target: kill the s_wait_loadcnt latency stall the simpler kernels hit
// (ATT confirmed ~83% of stall on q8_0_gemv_batched_warp8 is the per-lane
// global load throttle). Cooperative-load a BM×BK A tile + BK×BN B tile
// into LDS once per K-outer iter, then run the inner WMMA loop entirely
// from LDS — each weight byte amortizes across many compute ops.
//
// Tile parameters chosen for RDNA4 (gfx1201, 64 KB LDS/CU, ~3-4 waves/CU
// VGPR cap):
//   BM=64, BN=64, BK=32 (one Q8_0 block per K-outer iter)
//   WG = 4 warps × wave32 = 128 threads
//   M_WARPS=2, N_WARPS=2 → each warp owns 2×2 = 4 WMMA accumulators
//   LDS budget: 64×33 A f16 + 32×64 B f16 = ~8.2 KB → 7 WGs/CU LDS-wise
//
// Q8_0 layout (per row): [scales: blocks×f16] [quants: blocks×32×int8],
// row stride = blocks*34 bytes. K-outer aligned to BK=32 = exactly one
// Q8 block, so each row has ONE scale per K-outer iter.
//
// A_tile uses stride BK+1=33 (odd) to break the pow2-bank-conflict
// pattern that would hit when adjacent lanes read consecutive A rows
// for the WMMA A-fragment (16-lane stride 32 → all collide on the same
// banks).
extern "C" __global__ void q8_0_gemm_wmma_lds_tiled(
    float* __restrict__ out,               // [B, M]
    const unsigned char* __restrict__ w,   // [M, blocks*34]
    const int8_t* __restrict__ xq,         // [B, K]
    const float* __restrict__ xscale,      // [B, blocks]
    uint32_t in_dim,                       // K
    uint32_t out_dim,                      // M
    uint32_t batch,                        // N = B
    uint32_t blocks)                       // K / 32
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr uint32_t BM = 64u, BN = 64u, BK = 32u;
    constexpr uint32_t M_WARPS = 2u, N_WARPS = 2u;
    constexpr uint32_t A_STRIDE = BK + 1u;            // 33: odd, breaks pow2 bank pattern
    constexpr uint32_t M_TILES_PER_WARP = (BM / M_WARPS) / 16u;   // 2
    constexpr uint32_t N_TILES_PER_WARP = (BN / N_WARPS) / 16u;   // 2

    __shared__ _Float16 A_tile[BM * A_STRIDE];        // [m][k_padded]
    __shared__ _Float16 B_tile[BK * BN];              // [k][n]

    const uint32_t tid     = threadIdx.x;
    const uint32_t lane    = tid & 31u;
    const uint32_t wave_id = tid >> 5;                // 0..3
    const uint32_t wave_m  = wave_id / N_WARPS;       // 0..M_WARPS-1
    const uint32_t wave_n  = wave_id % N_WARPS;       // 0..N_WARPS-1
    const uint32_t wave_m0 = wave_m * (BM / M_WARPS); // 0 or 32
    const uint32_t wave_n0 = wave_n * (BN / N_WARPS); // 0 or 32

    const uint32_t m0 = blockIdx.x * BM;
    const uint32_t n0 = blockIdx.y * BN;
    if (m0 >= out_dim) return;

    const size_t rowbytes   = (size_t)blocks * 34u;
    const size_t scalebytes = (size_t)blocks * 2u;

    // C accumulators, one float8 per (M-tile, N-tile) WMMA output frag.
    float8 acc[M_TILES_PER_WARP][N_TILES_PER_WARP];
    #pragma unroll
    for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt)
            acc[mt][nt] = (float8){0,0,0,0,0,0,0,0};

    // === K-outer loop ===
    for (uint32_t k_outer = 0; k_outer < in_dim; k_outer += BK) {
        const uint32_t bl = k_outer >> 5;             // block index (scale lookup)

        // --- Cooperative load + dequant A_tile[BM][BK] ---
        // BM*BK = 2048 elements, 128 threads → 16 elements/thread.
        // tid = m_local * 2 + half (half = 0 or 1, picks low or high 16 cols).
        {
            const uint32_t m_local = tid >> 1;
            const uint32_t k_start = (tid & 1u) * 16u;
            const uint32_t m_global = m0 + m_local;
            _Float16 vals[16];
            if (m_global < out_dim) {
                const unsigned char* w_row = w + (size_t)m_global * rowbytes;
                const _Float16 wscl = ((const _Float16*)w_row)[bl];
                const int8_t* qrow = (const int8_t*)(w_row + scalebytes) + k_outer + k_start;
                #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    vals[e] = (_Float16)((float)qrow[e] * (float)wscl);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)0;
            }
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                A_tile[m_local * A_STRIDE + k_start + e] = vals[e];
            }
        }

        // --- Cooperative load + dequant B_tile[BK][BN] ---
        // Layout chosen so WMMA B-fragment access reads consecutive lanes
        // from different banks (lane l reads B_tile[(l>>4)*8 + e][l&15]).
        // Stored as B[k * BN + n] row-major in k.
        {
            const uint32_t n_local = tid >> 1;
            const uint32_t k_start = (tid & 1u) * 16u;
            const uint32_t n_global = n0 + n_local;
            _Float16 vals[16];
            if (n_global < batch) {
                const int8_t* x_row = xq + (size_t)n_global * in_dim;
                const float xscl = xscale[(size_t)n_global * blocks + bl];
                const int8_t* qsrc = x_row + k_outer + k_start;
                #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    vals[e] = (_Float16)((float)qsrc[e] * xscl);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)0;
            }
            // Write to LDS in [k][n] layout for B WMMA fragment shape.
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                B_tile[(k_start + e) * BN + n_local] = vals[e];
            }
        }

        __syncthreads();

        // --- Inner K loop: BK/16 = 2 WMMA K-tiles ---
        #pragma unroll
        for (uint32_t k_inner = 0; k_inner < BK; k_inner += 16u) {
            // Load A-frags for this warp's M-tiles (each 16×16).
            //   A[m,k]: lane l holds a[e] = A[(l&15)][(l>>4)*8 + e]
            half8 a_frags[M_TILES_PER_WARP];
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {
                const uint32_t m_in_tile = wave_m0 + mt * 16u + (lane & 15u);
                const uint32_t k_in_tile = k_inner + (lane >> 4) * 8u;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    a_frags[mt][e] = A_tile[m_in_tile * A_STRIDE + k_in_tile + e];
                }
            }

            // Load B-frags for this warp's N-tiles.
            //   B[k,n]: lane l holds b[e] = B[(l>>4)*8 + e][(l&15)]
            half8 b_frags[N_TILES_PER_WARP];
            #pragma unroll
            for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {
                const uint32_t k_in_tile_base = k_inner + (lane >> 4) * 8u;
                const uint32_t n_in_tile      = wave_n0 + nt * 16u + (lane & 15u);
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    b_frags[nt][e] = B_tile[(k_in_tile_base + e) * BN + n_in_tile];
                }
            }

            // M_TILES × N_TILES WMMAs against this warp's accumulators.
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {
                #pragma unroll
                for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a_frags[mt], b_frags[nt], acc[mt][nt]);
                }
            }
        }

        __syncthreads();
    }

    // --- Write C accumulators ---
    //   C[m,n]: lane l holds c[e] = C[(l>>4)*8 + e][l&15]
    //   -> m = m_tile_base + (lane>>4)*8 + e ; n = n_tile_base + (lane&15)
    #pragma unroll
    for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {
        #pragma unroll
        for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {
            const uint32_t n_idx = n0 + wave_n0 + nt * 16u + (lane & 15u);
            if (n_idx >= batch) continue;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t m_idx = m0 + wave_m0 + mt * 16u + (lane >> 4) * 8u + (uint32_t)e;
                if (m_idx < out_dim) {
                    out[(size_t)n_idx * out_dim + m_idx] = acc[mt][nt][e];
                }
            }
        }
    }
#else
    // Non-gfx12 fallback so build.rs's gfx1151 compile succeeds.
    // Won't run in production (iGPU never dispatches this).
    (void)out; (void)w; (void)xq; (void)xscale;
    (void)in_dim; (void)out_dim; (void)batch; (void)blocks;
#endif
}

// Grouped LDS-tiled Q8_0 GEMM (RDNA4 only). Same per-group inner kernel
// as q8_0_gemm_wmma_lds_tiled, but each WG additionally knows which group
// it's processing (gridDim.z) and applies per-group base offsets to the
// weight, xq, xscale, and out pointers. Used for V4 Flash's attention
// output_proj first stage (attn_output_a, 8 × 4096→1024 block-diagonal
// matmuls), where xq/out are interleaved per-batch as
//   xq:     [B, n_groups * group_dim]
//   xscale: [B, n_groups * blocks_per_group]
//   out:    [B, n_groups * rank]
// All three dimensions strided by n_groups within the batch dim.
extern "C" __global__ void q8_0_gemm_wmma_lds_tiled_grouped(
    float* __restrict__ out_base,          // [B, n_groups*rank]
    const unsigned char* __restrict__ w_base,  // [n_groups*rank, blocks*34]
    const int8_t* __restrict__ xq_base,    // [B, n_groups*group_dim]
    const float* __restrict__ xscale_base, // [B, n_groups*blocks]
    uint32_t group_dim,                    // K per group
    uint32_t rank,                         // M per group
    uint32_t n_groups,
    uint32_t batch,                        // N = B
    uint32_t blocks)                       // group_dim / 32
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr uint32_t BM = 64u, BN = 64u, BK = 32u;
    constexpr uint32_t M_WARPS = 2u, N_WARPS = 2u;
    constexpr uint32_t A_STRIDE = BK + 1u;
    constexpr uint32_t M_TILES_PER_WARP = (BM / M_WARPS) / 16u;
    constexpr uint32_t N_TILES_PER_WARP = (BN / N_WARPS) / 16u;

    __shared__ _Float16 A_tile[BM * A_STRIDE];
    __shared__ _Float16 B_tile[BK * BN];

    const uint32_t tid     = threadIdx.x;
    const uint32_t lane    = tid & 31u;
    const uint32_t wave_id = tid >> 5;
    const uint32_t wave_m  = wave_id / N_WARPS;
    const uint32_t wave_n  = wave_id % N_WARPS;
    const uint32_t wave_m0 = wave_m * (BM / M_WARPS);
    const uint32_t wave_n0 = wave_n * (BN / N_WARPS);

    const uint32_t m0    = blockIdx.x * BM;
    const uint32_t n0    = blockIdx.y * BN;
    const uint32_t group = blockIdx.z;
    if (m0 >= rank) return;

    const size_t rowbytes   = (size_t)blocks * 34u;
    const size_t scalebytes = (size_t)blocks * 2u;

    // Per-group base pointers.
    const unsigned char* w     = w_base + (size_t)group * (size_t)rank * rowbytes;
    const int8_t*  xq_grp      = xq_base + (size_t)group * (size_t)group_dim;
    const float*   xscale_grp  = xscale_base + (size_t)group * (size_t)blocks;
    float*         out         = out_base + (size_t)group * (size_t)rank;
    const size_t xq_b_stride     = (size_t)n_groups * (size_t)group_dim;
    const size_t xscale_b_stride = (size_t)n_groups * (size_t)blocks;
    const size_t out_b_stride    = (size_t)n_groups * (size_t)rank;

    float8 acc[M_TILES_PER_WARP][N_TILES_PER_WARP];
    #pragma unroll
    for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt)
            acc[mt][nt] = (float8){0,0,0,0,0,0,0,0};

    for (uint32_t k_outer = 0; k_outer < group_dim; k_outer += BK) {
        const uint32_t bl = k_outer >> 5;

        // A_tile load (same as non-grouped, using `w` base for this group)
        {
            const uint32_t m_local = tid >> 1;
            const uint32_t k_start = (tid & 1u) * 16u;
            const uint32_t m_global = m0 + m_local;
            _Float16 vals[16];
            if (m_global < rank) {
                const unsigned char* w_row = w + (size_t)m_global * rowbytes;
                const _Float16 wscl = ((const _Float16*)w_row)[bl];
                const int8_t* qrow = (const int8_t*)(w_row + scalebytes) + k_outer + k_start;
                #pragma unroll
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)((float)qrow[e] * (float)wscl);
            } else {
                #pragma unroll
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)0;
            }
            #pragma unroll
            for (int e = 0; e < 16; ++e)
                A_tile[m_local * A_STRIDE + k_start + e] = vals[e];
        }

        // B_tile load using per-batch strided pointers into the group's slice.
        {
            const uint32_t n_local = tid >> 1;
            const uint32_t k_start = (tid & 1u) * 16u;
            const uint32_t n_global = n0 + n_local;
            _Float16 vals[16];
            if (n_global < batch) {
                const int8_t* x_row = xq_grp + (size_t)n_global * xq_b_stride;
                const float xscl = xscale_grp[(size_t)n_global * xscale_b_stride + bl];
                const int8_t* qsrc = x_row + k_outer + k_start;
                #pragma unroll
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)((float)qsrc[e] * xscl);
            } else {
                #pragma unroll
                for (int e = 0; e < 16; ++e) vals[e] = (_Float16)0;
            }
            #pragma unroll
            for (int e = 0; e < 16; ++e)
                B_tile[(k_start + e) * BN + n_local] = vals[e];
        }

        __syncthreads();

        #pragma unroll
        for (uint32_t k_inner = 0; k_inner < BK; k_inner += 16u) {
            half8 a_frags[M_TILES_PER_WARP];
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {
                const uint32_t m_in_tile = wave_m0 + mt * 16u + (lane & 15u);
                const uint32_t k_in_tile = k_inner + (lane >> 4) * 8u;
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    a_frags[mt][e] = A_tile[m_in_tile * A_STRIDE + k_in_tile + e];
            }
            half8 b_frags[N_TILES_PER_WARP];
            #pragma unroll
            for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {
                const uint32_t k_in_tile_base = k_inner + (lane >> 4) * 8u;
                const uint32_t n_in_tile      = wave_n0 + nt * 16u + (lane & 15u);
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    b_frags[nt][e] = B_tile[(k_in_tile_base + e) * BN + n_in_tile];
            }
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {
                #pragma unroll
                for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a_frags[mt], b_frags[nt], acc[mt][nt]);
                }
            }
        }

        __syncthreads();
    }

    // Write to per-group, per-batch output region.
    #pragma unroll
    for (uint32_t mt = 0; mt < M_TILES_PER_WARP; ++mt) {
        #pragma unroll
        for (uint32_t nt = 0; nt < N_TILES_PER_WARP; ++nt) {
            const uint32_t n_idx = n0 + wave_n0 + nt * 16u + (lane & 15u);
            if (n_idx >= batch) continue;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t m_idx = m0 + wave_m0 + mt * 16u + (lane >> 4) * 8u + (uint32_t)e;
                if (m_idx < rank) {
                    out[(size_t)n_idx * out_b_stride + m_idx] = acc[mt][nt][e];
                }
            }
        }
    }
#else
    (void)out_base; (void)w_base; (void)xq_base; (void)xscale_base;
    (void)group_dim; (void)rank; (void)n_groups; (void)batch; (void)blocks;
#endif
}

// =============================================================================
// q8_0_gemm_wmma_f16x — Q8_0 weights x f16 activations, RDNA4 f16 WMMA
// (2026-09-08). Replaces q8_0_gemm_wmma_lds_tiled (11% of the 194.6-TFLOP
// matrix peak at B=1024: 64x64 tiles, 4 accumulators/wave, B tile k-major so
// each fragment was 8 scalar LDS reads, every element re-dequantized per
// k-step, load -> sync -> compute -> sync with no overlap).
//
//   - BM=128 (weight rows) x BN=128 (tokens) x BK=32 (one Q8_0 block);
//     256 threads = 8 waves as 2 (m) x 4 (n); wave tile 64x32 = 4x2 WMMA
//     accumulators (64 VGPRs). Per k-step: 4 A + 2 B fragment loads, 8 WMMAs.
//   - Both LDS tiles [row][k] with k contiguous, row stride 40 halves = 80 B
//     (20 dwords ≡ 4 banks mod 32): the gfx12 fragment (lane l = row l&15,
//     k (l>>4)*8..+8) is ONE ds_read_b128 per lane, conflict-free at the
//     4-cycle minimum for 32 lanes.
//   - Activations arrive as f16 [B, ldx] (cast once per layer by the caller;
//     no per-token dequant): the B tile is a b128 global->LDS copy.
//   - Weights: thread t stages 16 int8 of row t>>1 (one b128 load) and
//     dequantizes ONCE per k-outer via the perm trick (byte ^ 0x80 -> bias,
//     0x64|u = 1024+u exactly, pk_add -1152, pk_mul d): 7 VALU per 4 weights.
//   - Epilogue: each lane holds 8 consecutive m for one n -> two b128 stores
//     per (m-tile, n-tile) into out[n][m] (row-major [B, ldo]).
//   - grid.z = group: weights of group g at w + g*M*rowbytes, its input
//     columns at g*K within ldx, its output columns at g*M within ldo
//     (output_a's 8 groups; n_groups=1 = plain GEMM).
//   - Requires M % 128 == 0 (q_b 32768, kv 512, out_b 4096, out_a 1024 per
//     group) and K % 32 == 0; batch is guarded (zero-filled tokens, no store).
// LDS 20 KB/WG -> 3 WGs/CU.
// =============================================================================
typedef _Float16 half2v_q8 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ void q8x_magic4(uint32_t w, _Float16 scale, uint32_t& lo, uint32_t& hi) {
    const uint32_t u = w ^ 0x80808080u;                         // bytes = 128 + b (b signed)
    const uint32_t h01 = __builtin_amdgcn_perm(u, 0x64646464u, 0x00050004u);
    const uint32_t h23 = __builtin_amdgcn_perm(u, 0x64646464u, 0x00070006u);
    half2v_q8 a, b;
    __builtin_memcpy(&a, &h01, 4);
    __builtin_memcpy(&b, &h23, 4);
    const half2v_q8 bias = {(_Float16)-1152.0f, (_Float16)-1152.0f};
    a = (a + bias) * scale;
    b = (b + bias) * scale;
    __builtin_memcpy(&lo, &a, 4);
    __builtin_memcpy(&hi, &b, 4);
}

extern "C" __global__ void q8_0_gemm_wmma_f16x(
    float* __restrict__ out,               // [B, ldo], ldo = n_groups*M
    const unsigned char* __restrict__ w,   // [n_groups*M, blocks*34]
    const uint16_t* __restrict__ x16,      // [B, ldx] f16 bits (group g at cols g*K..)
    uint32_t K,                            // per group
    uint32_t M,                            // per group
    uint32_t n_groups,
    uint32_t batch,                        // N = B
    uint32_t blocks,                       // K / 32
    uint32_t ldx)                          // activation row pitch in halves (>= n_groups*K; pad
                                           // it off a power of two: out_a's 65536-B rows aliased L2 sets)
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr uint32_t BM = 128u, BN = 128u, BK = 32u;
    constexpr uint32_t LDS_STRIDE = BK + 8u;          // 40 halves = 80 B
    __shared__ __attribute__((aligned(16))) _Float16 A_tile[BM * LDS_STRIDE];
    __shared__ __attribute__((aligned(16))) _Float16 B_tile[BN * LDS_STRIDE];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;                   // 0..7
    const uint32_t wave_m0 = (wave >> 2) * 64u;       // 0 | 64
    const uint32_t wave_n0 = (wave & 3u) * 32u;       // 0,32,64,96
    // grid.x = N-blocks (fastest): the 4 N-blocks sharing one weight tile
    // run back to back, so the tile is fetched from DRAM once and hits L2
    // for the others (M-fastest dispatch re-read q_b's 35 MB four times).
    const uint32_t n0 = blockIdx.x * BN;
    const uint32_t m0 = blockIdx.y * BM;
    const uint32_t g  = blockIdx.z;
    if (m0 >= M) return;
    const uint32_t ldo = n_groups * M;
    const size_t rowbytes   = (size_t)blocks * 34u;
    const size_t scalebytes = (size_t)blocks * 2u;
    const unsigned char* w_g = w + (size_t)g * (size_t)M * rowbytes;

    // staging roles: thread t -> row/token t>>1, k half (t&1)*16
    const uint32_t st_row  = tid >> 1;
    const uint32_t st_koff = (tid & 1u) * 16u;
    const unsigned char* a_src_row = w_g + (size_t)(m0 + st_row) * rowbytes;   // m0+st_row < M guaranteed by M%128
    const bool b_valid = (n0 + st_row) < batch;
    const uint16_t* b_src_row = x16 + (size_t)(n0 + st_row) * (size_t)ldx + (size_t)g * K;

    float8 acc[4][2];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            acc[mt][nt] = (float8){0,0,0,0,0,0,0,0};

    // Register prefetch, one k-outer ahead: the raw A bytes + scale and the
    // B halves for k-outer i+1 are loaded right after the barrier that
    // publishes tile i, so their latency overlaps tile i's WMMAs.
    uint4 qa, vb0, vb1;
    uint16_t sbits;
    auto load_raw = [&](uint32_t ko) {
        __builtin_memcpy(&qa, a_src_row + scalebytes + ko + st_koff, 16);
        __builtin_memcpy(&sbits, a_src_row + 2u * (ko >> 5), 2);
        vb0 = make_uint4(0u, 0u, 0u, 0u); vb1 = vb0;
        if (b_valid) {
            __builtin_memcpy(&vb0, b_src_row + ko + st_koff, 16);
            __builtin_memcpy(&vb1, b_src_row + ko + st_koff + 8u, 16);
        }
    };
    load_raw(0u);
    for (uint32_t k_outer = 0; k_outer < K; k_outer += BK) {
        // --- stage A: 16 int8 -> 16 f16 (scale folded) -> 2 x b128 into LDS
        {
            _Float16 d;
            __builtin_memcpy(&d, &sbits, 2);
            uint32_t o[8];
            q8x_magic4(qa.x, d, o[0], o[1]);
            q8x_magic4(qa.y, d, o[2], o[3]);
            q8x_magic4(qa.z, d, o[4], o[5]);
            q8x_magic4(qa.w, d, o[6], o[7]);
            uint4* dst = (uint4*)(&A_tile[st_row * LDS_STRIDE + st_koff]);
            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
            dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
        }
        // --- stage B: straight copy
        {
            uint4* dst = (uint4*)(&B_tile[st_row * LDS_STRIDE + st_koff]);
            dst[0] = vb0;
            dst[1] = vb1;
        }
        __syncthreads();
        if (k_outer + BK < K) load_raw(k_outer + BK);   // in flight during the WMMAs below

        #pragma unroll
        for (uint32_t k_inner = 0; k_inner < BK; k_inner += 16u) {
            const uint32_t kf = k_inner + (lane >> 4) * 8u;
            half8 a_frags[4];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const uint32_t r = wave_m0 + mt * 16u + (lane & 15u);
                __builtin_memcpy(&a_frags[mt], &A_tile[r * LDS_STRIDE + kf], 16);
            }
            half8 b_frags[2];
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint32_t r = wave_n0 + nt * 16u + (lane & 15u);
                __builtin_memcpy(&b_frags[nt], &B_tile[r * LDS_STRIDE + kf], 16);
            }
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a_frags[mt], b_frags[nt], acc[mt][nt]);
        }
        __syncthreads();
    }

    // --- epilogue: C lane l elem e = C[m = (l>>4)*8 + e][n = l&15]; the 8 m
    //     are contiguous in out[n][.] -> two b128 stores.
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const uint32_t n_idx = n0 + wave_n0 + nt * 16u + (lane & 15u);
        if (n_idx >= batch) continue;
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const uint32_t m_idx = m0 + wave_m0 + mt * 16u + (lane >> 4) * 8u;
            float* dst = out + (size_t)n_idx * (size_t)ldo + (size_t)g * M + m_idx;
            const float8 c = acc[mt][nt];
            const float4 c0 = {c[0], c[1], c[2], c[3]};
            const float4 c1 = {c[4], c[5], c[6], c[7]};
            __builtin_memcpy(dst, &c0, 16);
            __builtin_memcpy(dst + 4, &c1, 16);
        }
    }
#else
    (void)out; (void)w; (void)x16; (void)K; (void)M; (void)n_groups; (void)batch; (void)blocks; (void)ldx;
#endif
}

// =============================================================================
// 2026-09-26 sweep (C2_dense_prefill): per-shape twins of q8_0_gemm_wmma_f16x /
// q8_0_gemm_wmma_lds_tiled built on ONE generic core (f16x_core, verbatim from
// scratch/kernel_ideas/C2_dense_prefill/cand.hip incl. its ablation template
// parameters, which default off). All BIT-EXACT vs the kernels they replace:
// same perm-trick weight dequant, same activation expression, same per-
// accumulator k order (k_outer ascending, two 16-wide inner steps), same WMMA;
// only the tile / wave geometry, barrier and prefetch change.
//   q8_0_gemm_wmma_i8x_db        Engram wkv (lds_tiled drop-in): 128x128 tile,
//                                int8 + xscale activations dequantised at stage
//                                as lds_tiled does ((f16)((float)q*xs)), double-
//                                buffered LDS (40 KB), 2-deep register prefetch.
//                                905 -> 484 us per 64-row chunk; 511 us per 128.
//   q8_0_gemm_wmma_f16x_db_bn64  kv / q_a: 128x64 tile (8 waves 4x2), DB LDS
//                                (30 KB -> 2 WGs/WGP), pf2. kv 106 -> 59 us, q_a
//                                125 -> 97 us at b=512 (q_a LOSES 4% at 1024).
//   q8_0_gemm_wmma_f16x_256x128  q_b / wo_a: 256x128 tile (wave tile 64x64, 216
//                                VGPR, 30 KB LDS), halves the activation-tile re-
//                                reads. q_b 458 -> 410 us, wo_a 367 -> 325 at 512.
// Selected in src/q8_0.rs (V41_ENGRAM_I8X, V41_F16X_DB_BN64, V41_F16X_256).
// Mechanisms worth keeping (NOTES.md): the B (activation) loads are branch-free
// (an out-of-range token reads the last valid row, zeroed by a select AFTER the
// barrier -- selecting right after the load made the compiler wait on it and
// turned the prefetch synchronous: "the load-select trap"); the barrier is
// LDS-only (sync_lds) so global prefetches stay in flight across it. The i8x
// kernel's speed was seen to be codegen-sensitive: re-measure this exact file.
// gfx12 only; other archs compile stubs (like the kernels above).
// =============================================================================
#if defined(__gfx1200__) || defined(__gfx1201__)
// LDS-only workgroup barrier: release/acquire fences restricted to the local address space so the
// compiler emits s_wait_dscnt 0 (not s_wait_loadcnt_dscnt 0) and no global_inv. Global loads issued
// before it stay in flight across the barrier. Valid here because the only data exchanged through
// the barrier is LDS.
__device__ __forceinline__ void sync_lds() {
#if defined(__gfx1200__) || defined(__gfx1201__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}

// ---------------------------------------------------------------------------------------------
// Generic core. BM x BN x 32 tile, WM x WN waves, wave tile (BM/WM) x (BN/WN), PF-deep register
// prefetch. A (weights) staged as 16-byte int8 slots (BM*2 slots), dequantised into LDS f16 [row][40].
// B (activations): F16 -> 16-byte f16 slots (BN*4); I8 -> 16-byte int8 slots (BN*2) + f32 xscale,
// dequantised as (f16)((float)q * xscale) exactly like q8_0_gemm_wmma_lds_tiled.
// ---------------------------------------------------------------------------------------------
// ABL (ablation, NOT correct, for pricing the parts): 0 = none; 1 = no global loads after k-outer 0
// (registers reused); 2 = no LDS traffic after k-outer 0 (fragments reused; the global loads are
// still consumed through a cheap xor so they are not dead); 3 = no WMMAs (fragments xor-folded).
// BRF: branch-free B loads (clamped row + select) vs execz-guarded loads (production style).
// DB: double-buffered LDS tiles; per k-outer: [frag ds_loads from buf cur] [stage next -> buf cur^1]
//     [reload regs] [WMMAs] [ONE barrier]. Register prefetch PF stages deep = PF iterations of
//     global latency hiding. LDS 2x.
template <int BM, int BN, int WM, int WN, int PF, bool BI8, bool LB, int ABL = 0, bool BRF = true, bool DB = false, bool AON = false, bool W0 = false>
__device__ __forceinline__ void f16x_core(
    float* __restrict__ out, const unsigned char* __restrict__ w,
    const uint16_t* __restrict__ x16, const int8_t* __restrict__ xq, const float* __restrict__ xscale,
    uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks, uint32_t ldx)
{
    constexpr int BK = 32;
    constexpr int LDS_STRIDE = BK + 8;               // 40 halves = 80 B
    constexpr int NT_ = WM * WN * 32;                // threads
    constexpr int MT = (BM / WM) / 16, NTL = (BN / WN) / 16;
    constexpr int A_SLOTS = BM * 2;                  // 16 B int8 slots per k-outer
    constexpr int A_PER = (A_SLOTS + NT_ - 1) / NT_;
    constexpr int B_SLOTS = BI8 ? BN * 2 : BN * 4;   // 16 B slots
    constexpr int B_PER = (B_SLOTS + NT_ - 1) / NT_;
    static_assert(A_SLOTS % NT_ == 0 || NT_ % A_SLOTS == 0, "A slots");
    static_assert(B_SLOTS % NT_ == 0 || NT_ % B_SLOTS == 0, "B slots");

    __shared__ __attribute__((aligned(16))) _Float16 A_tile[(DB ? 2 : 1) * BM * LDS_STRIDE];
    __shared__ __attribute__((aligned(16))) _Float16 B_tile[(DB ? 2 : 1) * BN * LDS_STRIDE];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t wave_m0 = (wave / WN) * (BM / WM);
    const uint32_t wave_n0 = (wave % WN) * (BN / WN);
    const uint32_t n0 = blockIdx.x * BN;
    const uint32_t m0 = blockIdx.y * BM;
    const uint32_t g  = blockIdx.z;
    if (m0 >= M) return;
    const uint32_t ldo = n_groups * M;
    const size_t rowbytes   = (size_t)blocks * 34u;
    const size_t scalebytes = (size_t)blocks * 2u;
    const unsigned char* w_g = w + (size_t)g * (size_t)M * rowbytes;

    // staging roles
    const unsigned char* a_src[A_PER]; uint32_t a_row[A_PER], a_koff[A_PER]; bool a_on[A_PER];
    #pragma unroll
    for (int i = 0; i < A_PER; ++i) {
        const uint32_t slot = tid + i * NT_;
        a_on[i] = (AON || A_SLOTS % NT_ != 0) ? (slot < A_SLOTS) : true;   // AON: runtime predicate (r1 style)
        a_row[i] = a_on[i] ? slot >> 1 : 0u;
        a_koff[i] = (slot & 1u) * 16u;
        a_src[i] = w_g + (size_t)(m0 + a_row[i]) * rowbytes;    // m0+row < M by M % BM == 0
    }
    const void* b_src[B_PER]; uint32_t b_row[B_PER], b_koff[B_PER]; bool b_on[B_PER];
    const float* bs_src[B_PER];
    #pragma unroll
    for (int i = 0; i < B_PER; ++i) {
        const uint32_t slot = tid + i * NT_;
        const bool in = (B_SLOTS % NT_ == 0) ? true : (slot < B_SLOTS);
        if (BI8) { b_row[i] = in ? slot >> 1 : 0u; b_koff[i] = (slot & 1u) * 16u; }
        else     { b_row[i] = in ? slot >> 2 : 0u; b_koff[i] = (slot & 3u) * 8u; }
        b_on[i] = in && (n0 + b_row[i]) < batch;
        // Branch-free loads: an out-of-range token reads the LAST valid row (always mapped) and the
        // register is zeroed by a select afterwards, so the waitcnt pass sees straight-line loads
        // and can count them (an execz-guarded load forced s_wait_loadcnt 0 at every stage).
        const uint32_t brow = (n0 + b_row[i] < batch) ? (n0 + b_row[i]) : (batch - 1u);
        if (BI8) {
            b_src[i] = xq + (size_t)brow * (size_t)ldx + (size_t)g * K;
            bs_src[i] = xscale + (size_t)brow * (size_t)(ldx / 32u) + (size_t)g * blocks;
        } else {
            b_src[i] = x16 + (size_t)brow * (size_t)ldx + (size_t)g * K;
            bs_src[i] = nullptr;
        }
    }

    float8 acc[MT][NTL];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NTL; ++nt)
            acc[mt][nt] = (float8){0,0,0,0,0,0,0,0};

    // prefetch registers, PF stages deep
    uint4 qa[PF][A_PER]; uint16_t sbits[PF][A_PER];
    uint4 vb[PF][B_PER]; float xs[PF][B_PER];
    auto load_raw = [&](int p, uint32_t ko) {
        #pragma unroll
        for (int i = 0; i < A_PER; ++i) if (a_on[i]) {
            __builtin_memcpy(&qa[p][i], a_src[i] + scalebytes + ko + a_koff[i], 16);
            __builtin_memcpy(&sbits[p][i], a_src[i] + 2u * (ko >> 5), 2);
        }
        #pragma unroll
        for (int i = 0; i < B_PER; ++i) {
            const uint32_t slot = tid + i * NT_;
            if ((B_SLOTS % NT_ != 0) && slot >= B_SLOTS) { vb[p][i] = make_uint4(0u, 0u, 0u, 0u); if (BI8) xs[p][i] = 0.f; continue; }
            if (!BRF) {   // production style: guarded load
                vb[p][i] = make_uint4(0u, 0u, 0u, 0u);
                if (BI8) xs[p][i] = 0.f;
                if (b_on[i]) {
                    if (BI8) { __builtin_memcpy(&vb[p][i], (const int8_t*)b_src[i] + ko + b_koff[i], 16); xs[p][i] = bs_src[i][ko >> 5]; }
                    else     { __builtin_memcpy(&vb[p][i], (const uint16_t*)b_src[i] + ko + b_koff[i], 16); }
                }
                continue;
            }
            uint4 t; float ts = 0.f;
            if (BI8) {
                __builtin_memcpy(&t, (const int8_t*)b_src[i] + ko + b_koff[i], 16);
                ts = bs_src[i][ko >> 5];
            } else {
                __builtin_memcpy(&t, (const uint16_t*)b_src[i] + ko + b_koff[i], 16);
            }
            vb[p][i] = t;                 // raw; masked at stage time (see patch6 note)
            if (BI8) xs[p][i] = ts;
        }
    };
    auto stage = [&](int p, uint32_t buf) {
        _Float16* At = A_tile + buf * (BM * LDS_STRIDE);
        _Float16* Bt = B_tile + buf * (BN * LDS_STRIDE);
        #pragma unroll
        for (int i = 0; i < A_PER; ++i) if (a_on[i]) {
            _Float16 d; __builtin_memcpy(&d, &sbits[p][i], 2);
            uint32_t o[8];
            q8x_magic4(qa[p][i].x, d, o[0], o[1]);
            q8x_magic4(qa[p][i].y, d, o[2], o[3]);
            q8x_magic4(qa[p][i].z, d, o[4], o[5]);
            q8x_magic4(qa[p][i].w, d, o[6], o[7]);
            uint4* dst = (uint4*)(&At[a_row[i] * LDS_STRIDE + a_koff[i]]);
            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
            dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
        }
        #pragma unroll
        for (int i = 0; i < B_PER; ++i) {
            const uint32_t slot = tid + i * NT_;
            if (slot >= B_SLOTS) continue;
            uint4 vraw = vb[p][i];
            if (BRF && !b_on[i]) vraw = make_uint4(0u, 0u, 0u, 0u);   // select AFTER the barrier
            if (BI8) {
                // (f16)((float)q * xscale), the lds_tiled expression, 16 values -> 2 x b128
                int8_t q[16]; __builtin_memcpy(q, &vraw, 16);
                const float sc = (BRF && !b_on[i]) ? 0.f : xs[p][i];
                _Float16 v[16];
                #pragma unroll
                for (int e = 0; e < 16; ++e) v[e] = (_Float16)((float)q[e] * sc);
                uint4* dst = (uint4*)(&Bt[b_row[i] * LDS_STRIDE + b_koff[i]]);
                __builtin_memcpy(&dst[0], &v[0], 16);
                __builtin_memcpy(&dst[1], &v[8], 16);
            } else {
                uint4* dst = (uint4*)(&Bt[b_row[i] * LDS_STRIDE + b_koff[i]]);
                dst[0] = vraw;
            }
        }
    };
    auto compute = [&]() {
        if (W0) asm volatile("s_wait_loadcnt 0x0" ::: "memory");
        #pragma unroll
        for (uint32_t k_inner = 0; k_inner < (uint32_t)BK; k_inner += 16u) {
            const uint32_t kf = k_inner + (lane >> 4) * 8u;
            half8 a_frags[MT];
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const uint32_t r = wave_m0 + mt * 16u + (lane & 15u);
                __builtin_memcpy(&a_frags[mt], &A_tile[r * LDS_STRIDE + kf], 16);
            }
            half8 b_frags[NTL];
            #pragma unroll
            for (int nt = 0; nt < NTL; ++nt) {
                const uint32_t r = wave_n0 + nt * 16u + (lane & 15u);
                __builtin_memcpy(&b_frags[nt], &B_tile[r * LDS_STRIDE + kf], 16);
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                #pragma unroll
                for (int nt = 0; nt < NTL; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frags[mt], b_frags[nt], acc[mt][nt]);
        }
    };
    auto sync = [&]() { if (LB) sync_lds(); else __syncthreads(); };

    if (DB) {
        #pragma unroll
        for (int p = 0; p < PF; ++p) if ((uint32_t)p * BK < K) load_raw(p, p * BK);
        stage(0, 0);
        if ((uint32_t)PF * BK < K) load_raw(0, PF * BK);
        sync();
        for (uint32_t k0 = 0; k0 < K; k0 += PF * BK) {
            #pragma unroll
            for (int p = 0; p < PF; ++p) {
                const uint32_t ko = k0 + p * BK;
                if (ko >= K) break;
                const uint32_t cur = (ko / BK) & 1u;
                const _Float16* At = A_tile + cur * (BM * LDS_STRIDE);
                const _Float16* Bt = B_tile + cur * (BN * LDS_STRIDE);
                half8 a_frags[2][MT], b_frags[2][NTL];
                #pragma unroll
                for (int ki = 0; ki < 2; ++ki) {
                    const uint32_t kf = ki * 16u + (lane >> 4) * 8u;
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) __builtin_memcpy(&a_frags[ki][mt], &At[(wave_m0 + mt * 16u + (lane & 15u)) * LDS_STRIDE + kf], 16);
                    #pragma unroll
                    for (int nt = 0; nt < NTL; ++nt) __builtin_memcpy(&b_frags[ki][nt], &Bt[(wave_n0 + nt * 16u + (lane & 15u)) * LDS_STRIDE + kf], 16);
                }
                const uint32_t kn = ko + BK;
                if (kn < K) {
                    stage((p + 1) % PF, cur ^ 1u);
                    if (kn + PF * BK < K) load_raw((p + 1) % PF, kn + PF * BK);
                }
                if (W0) asm volatile("s_wait_loadcnt 0x0" ::: "memory");
                #pragma unroll
                for (int ki = 0; ki < 2; ++ki)
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        #pragma unroll
                        for (int nt = 0; nt < NTL; ++nt)
                            acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frags[ki][mt], b_frags[ki][nt], acc[mt][nt]);
                sync();
            }
        }
    } else {
    uint32_t sink = 0u;   // ABL 2/3: keeps loads/fragments alive
    half8 keep_a[MT], keep_b[NTL];
    #pragma unroll
    for (int p = 0; p < PF; ++p) if ((uint32_t)p * BK < K) load_raw(p, p * BK);
    if (ABL == 2 || ABL == 3) {   // one real stage so the fragments are defined
        stage(0, 0); __syncthreads();
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) __builtin_memcpy(&keep_a[mt], &A_tile[(wave_m0 + mt * 16u + (lane & 15u)) * LDS_STRIDE + (lane >> 4) * 8u], 16);
        #pragma unroll
        for (int nt = 0; nt < NTL; ++nt) __builtin_memcpy(&keep_b[nt], &B_tile[(wave_n0 + nt * 16u + (lane & 15u)) * LDS_STRIDE + (lane >> 4) * 8u], 16);
        __syncthreads();
    }
    for (uint32_t k0 = 0; k0 < K; k0 += PF * BK) {
        #pragma unroll
        for (int p = 0; p < PF; ++p) {
            const uint32_t ko = k0 + p * BK;
            if (ko >= K) break;
            if (ABL == 0 || ABL == 1) { stage(p, 0); sync(); }
            if (ABL == 2) {   // consume the raw loads without LDS
                #pragma unroll
                for (int i = 0; i < A_PER; ++i) sink ^= qa[p][i].x ^ qa[p][i].w ^ (uint32_t)sbits[p][i];
                #pragma unroll
                for (int i = 0; i < B_PER; ++i) sink ^= vb[p][i].x ^ vb[p][i].w;
            }
            if (ABL == 3) {   // stage + sync, then fold the fragments into acc without WMMA
                stage(p, 0); sync();
                #pragma unroll
                for (uint32_t k_inner = 0; k_inner < (uint32_t)BK; k_inner += 16u) {
                    const uint32_t kf = k_inner + (lane >> 4) * 8u;
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) { half8 a; __builtin_memcpy(&a, &A_tile[(wave_m0 + mt * 16u + (lane & 15u)) * LDS_STRIDE + kf], 16); acc[mt][0][0] += (float)a[0]; }
                    #pragma unroll
                    for (int nt = 0; nt < NTL; ++nt) { half8 b; __builtin_memcpy(&b, &B_tile[(wave_n0 + nt * 16u + (lane & 15u)) * LDS_STRIDE + kf], 16); acc[0][nt][1] += (float)b[0]; }
                }
            }
            if (ABL != 1 || ko == 0) { if (ko + PF * BK < K) load_raw(p, ko + PF * BK); }
            if (ABL == 0 || ABL == 1) compute();
            if (ABL == 2) {
                #pragma unroll
                for (int k_inner = 0; k_inner < BK; k_inner += 16)
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        #pragma unroll
                        for (int nt = 0; nt < NTL; ++nt)
                            acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(keep_a[mt], keep_b[nt], acc[mt][nt]);
            }
            if (ABL == 0 || ABL == 1 || ABL == 3) sync();
        }
    }
    if (ABL == 2 && sink == 0xFFFFFFFFu) acc[0][0][0] += 1.f;   // practically never true; keeps sink live
    }

    #pragma unroll
    for (int nt = 0; nt < NTL; ++nt) {
        const uint32_t n_idx = n0 + wave_n0 + nt * 16u + (lane & 15u);
        if (n_idx >= batch) continue;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const uint32_t m_idx = m0 + wave_m0 + mt * 16u + (lane >> 4) * 8u;
            float* dst = out + (size_t)n_idx * (size_t)ldo + (size_t)g * M + m_idx;
            const float8 c = acc[mt][nt];
            const float4 c0 = {c[0], c[1], c[2], c[3]};
            const float4 c1 = {c[4], c[5], c[6], c[7]};
            __builtin_memcpy(dst, &c0, 16);
            __builtin_memcpy(dst + 4, &c1, 16);
        }
    }
}
#endif  // gfx12: sync_lds + f16x_core

// Engram wkv (drop-in for q8_0_gemm_wmma_lds_tiled's inputs): xq [B, K] int8 (ldx = K, plain
// rows), xscale [B, K/32] f32. Grid (ceil(batch/128), M/128, n_groups) x 256; needs M % 128 == 0
// and K % 256 == 0 (b128 loads of xq rows and weight rows at blocks*34 + ko).
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_i8x_db(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    f16x_core<128, 128, 2, 4, 2, true, true, 0, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K);
#else
    (void)out; (void)w; (void)xq; (void)xscale; (void)K; (void)M; (void)n_groups; (void)batch; (void)blocks;
#endif
}

// kv / q_a: f16x signature; grid (ceil(batch/64), M/128, n_groups) x 256.
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_f16x_db_bn64(
    float* __restrict__ out, const unsigned char* __restrict__ w, const uint16_t* __restrict__ x16,
    uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks, uint32_t ldx)
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    f16x_core<128, 64, 4, 2, 2, false, true, 0, true, true>(out, w, x16, nullptr, nullptr, K, M, n_groups, batch, blocks, ldx);
#else
    (void)out; (void)w; (void)x16; (void)K; (void)M; (void)n_groups; (void)batch; (void)blocks; (void)ldx;
#endif
}

// q_b / wo_a: f16x signature; grid (ceil(batch/128), M/256, n_groups) x 256; needs M % 256 == 0.
extern "C" __global__ void __launch_bounds__(256) q8_0_gemm_wmma_f16x_256x128(
    float* __restrict__ out, const unsigned char* __restrict__ w, const uint16_t* __restrict__ x16,
    uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks, uint32_t ldx)
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    f16x_core<256, 128, 4, 2, 1, false, true>(out, w, x16, nullptr, nullptr, K, M, n_groups, batch, blocks, ldx);
#else
    (void)out; (void)w; (void)x16; (void)K; (void)M; (void)n_groups; (void)batch; (void)blocks; (void)ldx;
#endif
}
