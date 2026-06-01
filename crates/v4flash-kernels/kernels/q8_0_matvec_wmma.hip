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
