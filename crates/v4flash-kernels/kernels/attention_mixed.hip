// attention_mixed — sink-aware causal softmax + weighted-sum over both the
// raw SWA cache AND the compressed-KV cache, with an optional per-comp-row
// allow mask. Mirrors ds4.c `layer_attention_mixed_one_decode_scratch`
// (ds4.c:6738). One workgroup per head, 256 threads per workgroup.
//
// When n_comp == 0 and mask is null, the math reduces exactly to
// `attention_swa` (M5) — that's how this kernel covers L=0,1 (pure SWA),
// L=3,5,…,41 (ratio==128 but n_comp==0 in our 57-tok prompt), and
// L=2,4,…,42 (ratio==4 with active comp rows + mask).
//
// Mask semantics: `mask[r] == 0` makes the r-th compressed row contribute
// `score = -INFINITY` to the softmax. `expf(-INFINITY - max_score)` is
// IEEE 754 zero, so the masked row drops out of both the denom and the
// weighted-sum axpy without any special-casing.
//
// ATTN_MIXED_MAX_KEYS — global-mem-bound, used by all attention kernels.
// 24576 covers 128 raw + 24448 comp ≈ 97K ctx at ratio=4. Scratch lives
// in DgpuScratch.attn_scores (caller-supplied global buffer).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// RDNA4 (gfx12) WMMA fragment types for the WMMA Phase-B variant below.
typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define ATTN_MIXED_MAX_KEYS  24576

// ---------------------------------------------------------------------------
// Split-kernel decode (perf diagnosis): same math as `attention_mixed`, but
// the three phases (dot-product score, softmax, weighted sum) become three
// distinct launches so each shows up with its own perfetto span.
//
// Scratch layout (per call, supplied by caller):
//   scores_g  [n_head][max_keys]   f32   (out of score, in of softmax,
//                                         softmax overwrites with weights)
//   inv_g     [n_head]             f32   (out of softmax, in of wsum)
//
// Mask is not supported here — the het decode path passes mask=None.
// ---------------------------------------------------------------------------

// Per-(head, row) WG kernel. Grid (n_head, n_total). Each WG is one
// wave32 that computes one dot product. The inner head_dim loop is
// hand-unrolled to 16 iterations (head_dim=512 / 32 lanes) so the
// compiler issues all 16 loads then waits ONCE — without the unroll,
// each load got its own `s_wait_loadcnt 0` stall (~245 cycles/wave),
// burning 3× the kernel wall on serialized memory waits. Confirmed
// via rocprofv3 ATT trace. ROWS_PER_WG kept as a parameter for
// experimentation but ROWS=1 (max parallelism) wins now that each
// wave runs fast.
#define ROWS_PER_WG 1

extern "C" __global__ void attention_mixed_score(
        float          * scores_g,       // [n_head, max_keys]
        const float    * q,              // [n_head, head_dim]
        const _Float16 * raw_kv,         // [n_raw, head_dim]   (f16)
        const _Float16 * comp_kv,        // [n_comp, head_dim]  (f16, may be null when n_comp==0)
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     n_raw,
        unsigned int     n_comp,
        unsigned int     max_keys,
        float            kq_scale) {
    const unsigned int h = blockIdx.x;
    const unsigned int r_base = blockIdx.y * ROWS_PER_WG;
    const unsigned int tid = threadIdx.x;
    if (h >= n_head) return;

    const unsigned int n_total = n_raw + n_comp;
    const float *qh = q + (size_t)h * head_dim;

    #pragma unroll
    for (unsigned int u = 0; u < ROWS_PER_WG; u++) {
        const unsigned int r = r_base + u;
        if (r >= n_total) break;

        const _Float16 *kv_row = (r < n_raw)
            ? raw_kv + (size_t)r * head_dim
            : comp_kv + (size_t)(r - n_raw) * head_dim;

        // Unrolled: with head_dim=512 and 32 lanes, each lane does 16
        // muls. Hardcoding lets the compiler pipeline all 16 loads
        // before waiting (was serialized 16× via per-load s_wait_loadcnt
        // = ~245 cycles each = ~3900 cycles wasted per wave per row).
        // f16 V → load as fp16 and let mul promote to f32 (kv_row[i] is
        // _Float16, qh[i] is float; the float operand promotes the f16).
        float p = 0.0f;
        #pragma unroll
        for (unsigned int u = 0; u < 16; u++) {
            const unsigned int i = tid + u * 32;
            p += qh[i] * (float)kv_row[i];
        }
        p += __shfl_down(p, 16);
        p += __shfl_down(p,  8);
        p += __shfl_down(p,  4);
        p += __shfl_down(p,  2);
        p += __shfl_down(p,  1);
        if (tid == 0) scores_g[(size_t)h * max_keys + r] = p * kq_scale;
    }
}

// Merged softmax + weighted-sum. One WG (512 threads) per head.
// Reads scores from global, does softmax in place on the same global
// buffer (writes weights back), then all 512 threads cooperate on the
// weighted sum reading weights from global.
//
// Global-mem sw means MAX_KEYS isn't capped by LDS (64 KB / WG) and
// can grow to support arbitrary context. Trade-off vs the old
// LDS-resident version: ~1-2 MB extra global RW per layer for weights
// (negligible vs the KV-cache reads which dominate at long context).
extern "C" __global__ void attention_mixed_softmax_wsum(
        float          * out,            // [n_head, head_dim]
        float          * scores_g,       // [n_head, max_keys] — in/out (overwritten with weights)
        const float    * sinks,          // [n_head]
        const _Float16 * raw_kv,         // [n_raw,  head_dim]  (f16)
        const _Float16 * comp_kv,        // [n_comp, head_dim]  (f16)
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     n_raw,
        unsigned int     n_comp,
        unsigned int     max_keys) {
    const unsigned int h = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (h >= n_head) return;

    __shared__ float s_inv;

    float *sw = scores_g + (size_t)h * max_keys;   // both score-in and weight-out
    float *oh = out + (size_t)h * head_dim;
    const float sink_h = sinks[h];
    const unsigned int n_total = n_raw + n_comp;

    // Phase A: softmax — wave 0 only. Reads sw[] (scores) from global,
    // rewrites in place with weights. Sink contributes to denom only.
    if (tid < 32) {
        float local_max = sink_h;
        for (unsigned int r = tid; r < n_total; r += 32) {
            float v = sw[r];
            if (v > local_max) local_max = v;
        }
        {
            float o;
            o = __shfl_xor(local_max, 16); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  8); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  4); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  2); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  1); if (o > local_max) local_max = o;
        }
        const float max_score = local_max;

        float local_sum = (tid == 0) ? expf(sink_h - max_score) : 0.0f;
        for (unsigned int r = tid; r < n_total; r += 32) {
            float w = expf(sw[r] - max_score);
            sw[r] = w;
            local_sum += w;
        }
        local_sum += __shfl_xor(local_sum, 16);
        local_sum += __shfl_xor(local_sum,  8);
        local_sum += __shfl_xor(local_sum,  4);
        local_sum += __shfl_xor(local_sum,  2);
        local_sum += __shfl_xor(local_sum,  1);
        if (tid == 0) s_inv = 1.0f / local_sum;
    }
    __syncthreads();

    // Phase B: weighted sum (all 512 threads). Two levers stack:
    //   block=512 → 16 waves/WG → 4 waves/SIMD on dGPU (was 2 with
    //     block=256), giving more concurrent waves to hide L2 latency.
    //   16-way r-loop ILP → compiler issues `s_clause 0xf` (16 loads),
    //     interleaves FMAs with per-load waits (`s_wait_loadcnt 0x0..0xf`).
    //     4-way was hitting `s_wait_loadcnt 0x2` = 75% of wall (only ~6
    //     loads in-flight). 8-way: 63%. 16-way: 44% stall. 32-way spills
    //     VGPRs and regresses.
    // ATT confirmed: per-wave irreducible 244c memory latency now mostly
    // hidden by wave parallelism + ILP. Net: 1.85 → 0.575 ms p50.
    // Weight reads from global sw are broadcast-cached across the 32
    // lanes of each wave; KV reads are coalesced across lanes within
    // each r-row, lane-strided across rows (one cache line per r).
    const float inv = s_inv;
    for (unsigned int d = tid; d < head_dim; d += block) {
        constexpr int U = 16;
        float s[U] = {0};

        unsigned int r = 0;
        for (; r + U <= n_raw; r += U) {
            #pragma unroll
            for (int k = 0; k < U; k++) {
                s[k] += sw[r + k] * (float)raw_kv[(size_t)(r + k) * head_dim + d];
            }
        }
        for (; r < n_raw; r++) {
            s[0] += sw[r] * (float)raw_kv[(size_t)r * head_dim + d];
        }

        float *cw = sw + n_raw;
        unsigned int c = 0;
        for (; c + U <= n_comp; c += U) {
            #pragma unroll
            for (int k = 0; k < U; k++) {
                s[k] += cw[c + k] * (float)comp_kv[(size_t)(c + k) * head_dim + d];
            }
        }
        for (; c < n_comp; c++) {
            s[0] += cw[c] * (float)comp_kv[(size_t)c * head_dim + d];
        }

        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < U; k++) acc += s[k];
        oh[d] = acc * inv;
    }
}

// Single-token (decode) variant with LDS-staged V. Same signature and
// semantics as attention_mixed_softmax_wsum — drops in without changing
// the call site. Each WG still handles one head, but Phase B
// cooperatively loads one K-tile of 16 V rows into LDS once per outer
// iter, then per-thread reads come from LDS instead of DRAM. Targets
// the long-ctx decode bottleneck (attn.smwsum p50=540 µs at ratio=4
// n_comp=16384 = ~10.8 ms/tok at 20 ratio=4 layers). The batched
// _ldsv_f16s wins (-23% over non-LDSV) came from V-staging + f16
// scores; the f16-scores half needs Phase A restructure (in-place
// f16/f32 mix), so we ship LDS-V on its own first.
//
// LDS budget: 16 rows × 512 d × 2 B = 16 KB per WG. At 64 KB/CU we
// could fit 4 WGs/CU but the grid is just 64 heads = 64 WGs over 64
// CUs = 1 WG/CU regardless. No occupancy hit.
extern "C" __global__ void attention_mixed_softmax_wsum_ldsv(
        float          * out,            // [n_head, head_dim]
        float          * scores_g,       // [n_head, max_keys] — in/out
        const float    * sinks,          // [n_head]
        const _Float16 * raw_kv,         // [n_raw,  head_dim]
        const _Float16 * comp_kv,        // [n_comp, head_dim]
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     n_raw,
        unsigned int     n_comp,
        unsigned int     max_keys) {
    const unsigned int h = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (h >= n_head) return;

    constexpr unsigned int LDSV_K = 16;          // K-tile size = ILP unroll
    constexpr unsigned int LDSV_HEAD_DIM = 512;  // hard-coded (matches MLA latent)
    __shared__ _Float16 V_tile[LDSV_K * LDSV_HEAD_DIM];   // 16 KB
    __shared__ float s_inv;

    float *sw = scores_g + (size_t)h * max_keys;
    float *oh = out + (size_t)h * head_dim;
    const float sink_h = sinks[h];
    const unsigned int n_total = n_raw + n_comp;

    // Phase A: softmax — unchanged from the non-LDSV variant. Wave 0
    // only, writes weights back into sw[] in place.
    if (tid < 32) {
        float local_max = sink_h;
        for (unsigned int r = tid; r < n_total; r += 32) {
            float v = sw[r];
            if (v > local_max) local_max = v;
        }
        {
            float o;
            o = __shfl_xor(local_max, 16); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  8); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  4); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  2); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  1); if (o > local_max) local_max = o;
        }
        const float max_score = local_max;

        float local_sum = (tid == 0) ? expf(sink_h - max_score) : 0.0f;
        for (unsigned int r = tid; r < n_total; r += 32) {
            float w = expf(sw[r] - max_score);
            sw[r] = w;
            local_sum += w;
        }
        local_sum += __shfl_xor(local_sum, 16);
        local_sum += __shfl_xor(local_sum,  8);
        local_sum += __shfl_xor(local_sum,  4);
        local_sum += __shfl_xor(local_sum,  2);
        local_sum += __shfl_xor(local_sum,  1);
        if (tid == 0) s_inv = 1.0f / local_sum;
    }
    __syncthreads();

    // Phase B: per-d weighted sum, with V cooperatively staged to LDS
    // once per K-tile. Each thread handles one d (block=512=head_dim).
    const float inv = s_inv;
    const unsigned int d = tid;
    if (d >= head_dim) return;

    // Cooperative load layout: 16 rows × 32 lane-slots × 16 f16 = 512
    // threads × 16 f16 per thread. tid maps to (row=tid/32, slot=tid%32);
    // each thread loads 16 contiguous f16 starting at row * head_dim +
    // slot * 16. Use *(int4*) for a 16-byte vec b128 load (head_dim=512,
    // row pitch 1024 B, slot offset 16 f16 = 32 B → 16-byte aligned).
    const unsigned int load_row  = tid >> 5;      // 0..15
    const unsigned int load_slot = tid & 31u;     // 0..31
    const unsigned int load_off  = load_slot * 16u;

    float s = 0.0f;

    // ---- raw_kv segment ----
    unsigned int r = 0;
    while (r + LDSV_K <= n_raw) {
        // Cooperative load 16 V rows starting at raw_kv[r..r+16].
        if (load_off < head_dim) {
            const _Float16 *src = raw_kv + (size_t)(r + load_row) * head_dim + load_off;
            int4 v = *(const int4*)src;
            *(int4*)(&V_tile[load_row * LDSV_HEAD_DIM + load_off]) = v;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int k = 0; k < LDSV_K; k++) {
            s += sw[r + k] * (float)V_tile[k * LDSV_HEAD_DIM + d];
        }
        __syncthreads();
        r += LDSV_K;
    }
    // Tail of raw_kv (n_raw % 16).
    for (; r < n_raw; r++) {
        s += sw[r] * (float)raw_kv[(size_t)r * head_dim + d];
    }

    // ---- comp_kv segment ----
    float *cw = sw + n_raw;
    unsigned int c = 0;
    while (c + LDSV_K <= n_comp) {
        if (load_off < head_dim) {
            const _Float16 *src = comp_kv + (size_t)(c + load_row) * head_dim + load_off;
            int4 v = *(const int4*)src;
            *(int4*)(&V_tile[load_row * LDSV_HEAD_DIM + load_off]) = v;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int k = 0; k < LDSV_K; k++) {
            s += cw[c + k] * (float)V_tile[k * LDSV_HEAD_DIM + d];
        }
        __syncthreads();
        c += LDSV_K;
    }
    for (; c < n_comp; c++) {
        s += cw[c] * (float)comp_kv[(size_t)c * head_dim + d];
    }

    oh[d] = s * inv;
}

// === Decode head-tiled + K-split smwsum (3 kernels) ====================
// Architectural restructure of the single-token smwsum to unlock MLA
// V-share at B=1. The baseline single-token kernel runs 1 WG per head,
// each reading the full V independently — V data is reused across heads
// only via L2 cache (small, partial reuse). The batched WMMA kernel
// shares V via LDS within a 16-head WG, but at B=1 it under-occupies the
// 64-CU dGPU (only 4 head-tile WGs).
//
// This pipeline restores V-share at B=1 by adding K-split:
//   Pass 1  attention_mixed_softmax_only
//     1 WG × 32 threads per head. Computes max/sum, writes weights in
//     place to scores buffer, writes inv[h] to a separate per-head buffer.
//   Pass 2  attention_mixed_wsum_b1_htiled_ksplit_ldsv
//     Grid (n_head/16=4, k_split, 1). Per WG: 16 heads × K_chunk keys ×
//     full head_dim=512. LDS-stages V (16 KB) once per K-tile of 16 keys,
//     each warp does WMMA against its 2 N-tile region. Writes partials
//     [k_split, h, d] to a global scratch buffer (no inv applied yet).
//   Pass 3  attention_mixed_reduce_partials
//     Per (h, d): sums k_split partials, multiplies by inv[h], writes out.
//
// With k_split=16, total grid = 4 × 16 = 64 WGs = 1 WG/CU on 64 CUs —
// matches baseline occupancy. Within each WG, V data is reused 16× across
// heads (the win the batched kernel can't deliver at B=1).
#define DEC_KS_HEAD_TILE 16
#define DEC_KS_N_TILE    16    // WMMA N-tile width
#define DEC_KS_K_TILE    16    // WMMA K-tile width

extern "C" __global__ void attention_mixed_softmax_only(
        float       * scores_g,       // [n_head, max_keys] — in-place
        const float * sinks,          // [n_head]
        float       * inv_per_head,   // [n_head] — output
        unsigned int  n_head,
        unsigned int  n_raw,
        unsigned int  n_comp,
        unsigned int  max_keys) {
    const unsigned int h = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (h >= n_head) return;
    if (tid >= 32) return;
    float *sw = scores_g + (size_t)h * max_keys;
    const float sink_h = sinks[h];
    const unsigned int n_total = n_raw + n_comp;

    // Max reduction.
    float local_max = sink_h;
    for (unsigned int r = tid; r < n_total; r += 32) {
        float v = sw[r];
        if (v > local_max) local_max = v;
    }
    {
        float o;
        o = __shfl_xor(local_max, 16); if (o > local_max) local_max = o;
        o = __shfl_xor(local_max,  8); if (o > local_max) local_max = o;
        o = __shfl_xor(local_max,  4); if (o > local_max) local_max = o;
        o = __shfl_xor(local_max,  2); if (o > local_max) local_max = o;
        o = __shfl_xor(local_max,  1); if (o > local_max) local_max = o;
    }
    const float max_score = local_max;

    // Sum (with sink in denom only) + write weights in place.
    float local_sum = (tid == 0) ? expf(sink_h - max_score) : 0.0f;
    for (unsigned int r = tid; r < n_total; r += 32) {
        float w = expf(sw[r] - max_score);
        sw[r] = w;
        local_sum += w;
    }
    local_sum += __shfl_xor(local_sum, 16);
    local_sum += __shfl_xor(local_sum,  8);
    local_sum += __shfl_xor(local_sum,  4);
    local_sum += __shfl_xor(local_sum,  2);
    local_sum += __shfl_xor(local_sum,  1);
    if (tid == 0) inv_per_head[h] = 1.0f / local_sum;
}

extern "C" __global__ void attention_mixed_wsum_b1_htiled_ksplit_ldsv(
        float          * partials,    // [k_split, n_head, head_dim] f32
        const float    * scores_g,    // [n_head, max_keys] — post-softmax weights (unscaled)
        const _Float16 * raw_kv,      // [n_raw, head_dim]
        const _Float16 * comp_kv,     // [n_comp, head_dim]
        unsigned int     n_raw,
        unsigned int     n_comp,
        unsigned int     n_head,
        unsigned int     head_dim,    // == 512
        unsigned int     max_keys,
        unsigned int     k_split) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr unsigned int HT = DEC_KS_HEAD_TILE;  // 16
    constexpr unsigned int KT = DEC_KS_K_TILE;     // 16
    constexpr unsigned int NT = DEC_KS_N_TILE;     // 16
    constexpr unsigned int HEAD_DIM = 512u;
    constexpr unsigned int N_TILES_TOTAL = HEAD_DIM / NT;   // 32
    constexpr unsigned int N_WARPS_PER_WG = 16u;            // 512 threads / 32 lanes
    constexpr unsigned int N_TILES_PER_WARP = N_TILES_TOTAL / N_WARPS_PER_WG;  // 2

    __shared__ _Float16 V_tile[KT * HEAD_DIM];               // 16 KB
    __shared__ float    W_tile[HT * KT];                     // 1 KB (16×16 f32)

    const unsigned int tid     = threadIdx.x;
    const unsigned int lane    = tid & 31u;
    const unsigned int warp    = tid >> 5;       // 0..15
    const unsigned int h_tile  = blockIdx.x;     // 0..n_head/HT - 1
    const unsigned int k_chunk = blockIdx.y;     // 0..k_split-1
    const unsigned int h_base  = h_tile * HT;

    // K-chunk range (rounded to multiples of KT inside each segment).
    const unsigned int n_total = n_raw + n_comp;
    const unsigned int chunk_size = (n_total + k_split - 1u) / k_split;
    const unsigned int k_lo = k_chunk * chunk_size;
    const unsigned int k_hi = min(k_lo + chunk_size, n_total);

    // Per-warp WMMA accumulators (N_TILES_PER_WARP per warp).
    float8 acc[N_TILES_PER_WARP];
    #pragma unroll
    for (unsigned int t = 0; t < N_TILES_PER_WARP; t++)
        acc[t] = (float8){0,0,0,0,0,0,0,0};

    // K-tile loop, stepping by KT=16. Edge tile handled by zero-padding
    // in cooperative loads.
    for (unsigned int k0 = k_lo; k0 < k_hi; k0 += KT) {
        // --- Cooperative load V tile (16 keys × 512 d) ---
        // 512 threads, 8192 f16 elements = 16 per thread. Use vec b128 =
        // 8 f16 per load, 2 loads per thread.
        // Thread tid handles V_tile[k_row][d_off] where k_row = tid/64,
        // d_off = (tid % 64) * 8.
        {
            const unsigned int k_row = tid >> 6;          // 0..7  (only half cover, see below)
            const unsigned int d_off = (tid & 63u) * 8u;  // 0,8,...,504
            // Each thread loads 2 rows (k_row and k_row+8) to cover all 16.
            #pragma unroll
            for (unsigned int kk = 0; kk < 2; kk++) {
                const unsigned int kk_row = k_row + kk * 8u;
                const unsigned int g_k = k0 + kk_row;
                _Float16 vals[8];
                bool valid = g_k < k_hi;
                if (valid) {
                    const _Float16 *src = (g_k < n_raw)
                        ? raw_kv + (size_t)g_k * head_dim
                        : comp_kv + (size_t)(g_k - n_raw) * head_dim;
                    *(int4*)vals = *(const int4*)(src + d_off);
                } else {
                    #pragma unroll
                    for (int e = 0; e < 8; e++) vals[e] = (_Float16)0;
                }
                *(int4*)(&V_tile[kk_row * HEAD_DIM + d_off]) =
                    *(int4*)vals;
            }
        }
        // --- Cooperative load W tile (16 heads × 16 keys) ---
        // 256 f32 elements over 512 threads → half cover. Thread t in [0, 256)
        // handles W_tile[t / 16][t % 16].
        if (tid < HT * KT) {
            const unsigned int h_local = tid / KT;
            const unsigned int k_local = tid % KT;
            const unsigned int g_k = k0 + k_local;
            const float w = (g_k < k_hi)
                ? scores_g[(size_t)(h_base + h_local) * max_keys + g_k]
                : 0.0f;
            W_tile[h_local * KT + k_local] = w;
        }
        __syncthreads();

        // --- Single WMMA M=16, K=16, N=16 per N-tile-per-warp ---
        // A frag = W_tile in f16 (cast on load): lane l holds
        //   a[e] = W[(l&15)][(l>>4)*8 + e]
        half8 a_frag;
        {
            const unsigned int m_in = lane & 15u;
            const unsigned int k_in_base = (lane >> 4) * 8u;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                a_frag[e] = (_Float16)W_tile[m_in * KT + k_in_base + (unsigned int)e];
            }
        }
        // B frag(s) = V_tile in f16: lane l holds
        //   b[e] = V[(l>>4)*8 + e][n_tile_base + (l&15)]
        // Per warp: N_TILES_PER_WARP n-tiles to cover.
        #pragma unroll
        for (unsigned int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            const unsigned int n_tile_idx = warp * N_TILES_PER_WARP + nt;
            const unsigned int n_base = n_tile_idx * NT;
            half8 b_frag;
            const unsigned int k_in_base = (lane >> 4) * 8u;
            const unsigned int n_in      = lane & 15u;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                b_frag[e] = V_tile[(k_in_base + (unsigned int)e) * HEAD_DIM + n_base + n_in];
            }
            acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                a_frag, b_frag, acc[nt]);
        }
        __syncthreads();
    }

    // --- Write partials to global ---
    // C frag layout: lane l holds c[e] = C[(l>>4)*8 + e][l & 15]
    //   → row h = h_base + (l>>4)*8 + e ; col d = n_base + (l & 15)
    float * out_chunk = partials + ((size_t)k_chunk * n_head + h_base) * head_dim;
    #pragma unroll
    for (unsigned int nt = 0; nt < N_TILES_PER_WARP; nt++) {
        const unsigned int n_tile_idx = warp * N_TILES_PER_WARP + nt;
        const unsigned int n_base = n_tile_idx * NT;
        const unsigned int n_idx = n_base + (lane & 15u);
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int h_off = (lane >> 4) * 8u + (unsigned int)e;
            out_chunk[(size_t)h_off * head_dim + n_idx] = acc[nt][e];
        }
    }
#else
    (void)partials; (void)scores_g; (void)raw_kv; (void)comp_kv;
    (void)n_raw; (void)n_comp; (void)n_head; (void)head_dim;
    (void)max_keys; (void)k_split;
#endif
}

extern "C" __global__ void attention_mixed_reduce_partials_apply_inv(
        float       * out,           // [n_head, head_dim]
        const float * partials,      // [k_split, n_head, head_dim]
        const float * inv_per_head,  // [n_head]
        unsigned int  n_head,
        unsigned int  head_dim,
        unsigned int  k_split) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = n_head * head_dim;
    if (gid >= total) return;
    const unsigned int h = gid / head_dim;
    const unsigned int d = gid % head_dim;
    const size_t hd_per_chunk = (size_t)n_head * head_dim;
    float acc = 0.0f;
    for (unsigned int kc = 0; kc < k_split; kc++) {
        acc += partials[(size_t)kc * hd_per_chunk + (size_t)h * head_dim + d];
    }
    out[(size_t)h * head_dim + d] = acc * inv_per_head[h];
}

#define SMWSUM_HEAD_TILE 16

// Head-tiled WMMA Phase-B smwsum. Each WG covers SMWSUM_HEAD_TILE(16)
// heads (so the V-latent is shared across heads via MLA). Phase A does
// f32 softmax in LDS as before; Phase B (the O(n_keys) weighted sum
// O[16 heads, 512] = W[16, n_keys] @ V[n_keys, 512]) runs as a RDNA4
// 16x16x16 f16 WMMA GEMM. W (post-softmax weights, f32 in scores_g) and
// V (raw_kv/comp_kv) is loaded as f16 directly — f32 accumulate, zero
// extra memory. The MLA shared-V reuse is
// automatic: V is the WMMA B-operand, reused across the 16-head m-tile.
//
// Layout: M=SMWSUM_HEAD_TILE(16)=1 m-tile, N=head_dim(512)=32 n-tiles, K=n_keys
// (raw then comp, each with a zero-padded partial-tile tail). 512 threads =
// 16 warps; warp w owns n-tiles {w, w+16} (2 independent accumulators / warp).
// RDNA4 wave32 frag layout (validated, see kernels/wmma_wsum.hip):
//   A[m,k]: a_frag[e] = W[h0+(lane&15)][kbase+(lane>>4)*8+e]
//   B[k,n]: b_frag[e] = V[kbase+(lane>>4)*8+e][nt*16+(lane&15)]
//   C[m,n]: c_frag[e] -> head h0+((lane>>4)*8+e), dim nt*16+(lane&15)
// Requires head_dim == 512 (the MLA latent width) and SMWSUM_HEAD_TILE == 16.
extern "C" __global__ void attention_mixed_softmax_wsum_batched_htiled_wmma(
        float          * out,            // [B, n_head, head_dim]
        float          * scores_g,       // [B, n_head, max_keys] in/out
        const float    * sinks,          // [n_head]
        const _Float16 * raw_kv,         // f16 V
        const _Float16 * comp_kv,
        const int      * n_raw_per,      // [B]
        const int      * n_raw_offset_per,// [B] starting raw-cache slot for this token
        const int      * n_comp_per,     // [B]
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys) {
    const unsigned int hg    = blockIdx.x;        // head group
    const unsigned int b     = blockIdx.y;
    const unsigned int tid   = threadIdx.x;
    const unsigned int h0    = hg * SMWSUM_HEAD_TILE;
    if (h0 >= n_head) return;

    unsigned int ht = SMWSUM_HEAD_TILE;
    if (h0 + ht > n_head) ht = n_head - h0;

    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int raw_off= (unsigned int)n_raw_offset_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    const unsigned int n_total = n_raw + n_comp;

    __shared__ float s_inv[SMWSUM_HEAD_TILE];

    // Phase A: softmax — wave w handles head h0+w (identical to the f32 kernel).
    const unsigned int wave = tid >> 5;
    const unsigned int lane = tid & 31u;
    if (wave < ht) {
        const unsigned int h = h0 + wave;
        float *sw = scores_g + ((size_t)b * n_head + h) * max_keys;
        const float sink_h = sinks[h];
        float local_max = sink_h;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float v = sw[rr];
            if (v > local_max) local_max = v;
        }
        {
            float o;
            o = __shfl_xor(local_max, 16); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  8); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  4); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  2); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  1); if (o > local_max) local_max = o;
        }
        const float max_score = local_max;
        float local_sum = (lane == 0) ? expf(sink_h - max_score) : 0.0f;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float w = expf(sw[rr] - max_score);
            sw[rr] = w;
            local_sum += w;
        }
        local_sum += __shfl_xor(local_sum, 16);
        local_sum += __shfl_xor(local_sum,  8);
        local_sum += __shfl_xor(local_sum,  4);
        local_sum += __shfl_xor(local_sum,  2);
        local_sum += __shfl_xor(local_sum,  1);
        if (lane == 0) s_inv[wave] = 1.0f / local_sum;
    }
    __syncthreads();

#if defined(__gfx1200__) || defined(__gfx1201__)
    // Phase B: WMMA GEMM. warp w owns n-tiles {w, w+16}.
    const unsigned int warp = wave;                 // 0..15
    const unsigned int arow = lane & 15u;           // A-row (head) / B-col (dim)
    const unsigned int kh   = (lane >> 4) * 8u;     // K-half offset
    const unsigned int nt_a = warp;
    const unsigned int nt_b = warp + 16u;
    const unsigned int d_a  = nt_a * 16u + arow;
    const unsigned int d_b  = nt_b * 16u + arow;

    const float *swrow = scores_g + ((size_t)b * n_head + h0 + arow) * max_keys;

    float8 c_a = {0,0,0,0,0,0,0,0};
    float8 c_b = {0,0,0,0,0,0,0,0};

    // --- raw section ---
    unsigned int r = 0;
    for (; r + 16u <= n_raw; r += 16u) {
        half8 a, ba, bb;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k = r + kh + (unsigned int)e;
            a[e]  = (_Float16)swrow[k];
            ba[e] = raw_kv[(size_t)(k + raw_off) * head_dim + d_a];
            bb[e] = raw_kv[(size_t)(k + raw_off) * head_dim + d_b];
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    }
    if (r < n_raw) {
        half8 a = {0,0,0,0,0,0,0,0}, ba = {0,0,0,0,0,0,0,0}, bb = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k = r + kh + (unsigned int)e;
            if (k < n_raw) {
                a[e]  = (_Float16)swrow[k];
                ba[e] = raw_kv[(size_t)(k + raw_off) * head_dim + d_a];
                bb[e] = raw_kv[(size_t)(k + raw_off) * head_dim + d_b];
            }
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    }

    // --- comp section ---
    if (comp_kv) {
        unsigned int c = 0;
        for (; c + 16u <= n_comp; c += 16u) {
            half8 a, ba, bb;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int cc = c + kh + (unsigned int)e;
                a[e]  = (_Float16)swrow[n_raw + cc];
                ba[e] = comp_kv[(size_t)cc * head_dim + d_a];
                bb[e] = comp_kv[(size_t)cc * head_dim + d_b];
            }
            c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
            c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
        }
        if (c < n_comp) {
            half8 a = {0,0,0,0,0,0,0,0}, ba = {0,0,0,0,0,0,0,0}, bb = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int cc = c + kh + (unsigned int)e;
                if (cc < n_comp) {
                    a[e]  = (_Float16)swrow[n_raw + cc];
                    ba[e] = comp_kv[(size_t)cc * head_dim + d_a];
                    bb[e] = comp_kv[(size_t)cc * head_dim + d_b];
                }
            }
            c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
            c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
        }
    }

    // store: c_frag[e] -> head h0+((lane>>4)*8+e), dim nt*16+(lane&15)
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned int mh = (lane >> 4) * 8u + (unsigned int)e;
        if (mh < ht) {
            const float scale = s_inv[mh];
            const size_t obase = ((size_t)b * n_head + h0 + mh) * head_dim;
            out[obase + d_a] = c_a[e] * scale;
            out[obase + d_b] = c_b[e] * scale;
        }
    }
#else
    (void)out; (void)raw_kv; (void)comp_kv; (void)head_dim; (void)lane;
#endif
}

// Phase 1 (score = Q·Kᵀ) via RDNA4 WMMA. O[heads, keys] = Q[heads, dim] @
// Kᵀ[dim, keys], reduction over head_dim=512 (32 K-chunks of 16). Each warp
// owns one 16-key tile and loops all (n_head/16) head-tiles, reusing the
// loaded K (B) fragment across head-tiles. f32→f16 at fragment-load (no f16
// KV cache). Grid (ceil(n_total_max/256), 1, batch), block 512 = 16 warps.
// Requires head_dim==512 and n_head a multiple of 16.
#define SCORE_WMMA_KEYS_PER_BLK 256u   // 16 warps × 16 keys
extern "C" __global__ void attention_mixed_score_batched_htiled_wmma(
        float          * scores_g,       // [B, n_head, max_keys]
        const float    * q,              // [B, n_head, head_dim]
        const _Float16 * raw_kv,         // [..., head_dim] f16 (oversized; per-token raw_off)
        const _Float16 * comp_kv,        // [n_comp, head_dim] f16 (may be null)
        const int      * n_raw_per,      // [B]
        const int      * n_raw_offset_per,// [B] starting raw-cache slot for this token
        const int      * n_comp_per,     // [B]
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys,
        float            kq_scale) {
    const unsigned int b    = blockIdx.z;
    const unsigned int tid  = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;

    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int raw_off= (unsigned int)n_raw_offset_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    const unsigned int n_total = n_raw + n_comp;

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int n0   = blockIdx.x * SCORE_WMMA_KEYS_PER_BLK + warp * 16u;
    const unsigned int col  = lane & 15u;            // key within tile / A-row(head)
    const unsigned int kh   = (lane >> 4) * 8u;      // K-half offset within chunk
    const unsigned int bkey = n0 + col;
    const unsigned int n_kc = head_dim / 16u;        // 32 K-chunks

    // Per-lane K-row pointer (the key this lane's B elements come from).
    const _Float16 *brow = nullptr;
    if (bkey < n_total) {
        brow = (bkey < n_raw)
            ? raw_kv + (size_t)(bkey + raw_off) * head_dim
            : comp_kv + (size_t)(bkey - n_raw) * head_dim;
    }

    const unsigned int n_ht = n_head / 16u;          // head-tiles
    float8 c[8];                                     // up to n_head/16 (=4 for 64)
    #pragma unroll
    for (unsigned int t = 0; t < 8; t++) c[t] = (float8){0,0,0,0,0,0,0,0};

    const float *qbase = q + ((size_t)b * n_head) * head_dim;
    for (unsigned int kc = 0; kc < n_kc; kc++) {
        const unsigned int dim0 = kc * 16u + kh;
        // B (K): b_frag[e] = K[key=col][dim=dim0+e]; zero-pad out-of-range keys.
        half8 bf = {0,0,0,0,0,0,0,0};
        if (brow) {
            #pragma unroll
            for (int e = 0; e < 8; e++) bf[e] = brow[dim0 + (unsigned int)e];
        }
        #pragma unroll
        for (unsigned int mt = 0; mt < 8; mt++) {
            if (mt >= n_ht) break;
            const float *qh = qbase + (size_t)(mt * 16u + col) * head_dim;
            half8 af;
            #pragma unroll
            for (int e = 0; e < 8; e++) af[e] = (_Float16)qh[dim0 + (unsigned int)e];
            c[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf, c[mt]);
        }
    }

    // store: c_frag[e] -> head mt*16+((lane>>4)*8+e), key n0+(lane&15).
    #pragma unroll
    for (unsigned int mt = 0; mt < 8; mt++) {
        if (mt >= n_ht) break;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int h = mt * 16u + (lane >> 4) * 8u + (unsigned int)e;
            if (h < n_head && bkey < n_total) {
                scores_g[((size_t)b * n_head + h) * max_keys + bkey] = c[mt][e] * kq_scale;
            }
        }
    }
#else
    (void)scores_g; (void)q; (void)raw_kv; (void)comp_kv; (void)n_head;
    (void)head_dim; (void)max_keys; (void)kq_scale; (void)warp; (void)lane;
    (void)n_total;
#endif
}

// === f16-scores experimental variants ====================================
// Same WMMA math as the f32-scores variants above, but the scores buffer
// itself is _Float16. Halves the largest single piece of DRAM traffic in
// the attention pair at long context (e.g. 1.04 GB → 520 MB at
// depth=32k, B=512). Phase A softmax math stays f32 (running max/sum are
// reductions and need it); only the storage type changes.
// `comp_allowed_bits` (nullable) carries the CSA indexer mask: when set,
// it's a bitpacked [B, max_keys_words] buffer where bit `c & 31` of word
// `b * max_keys_words + (c >> 5)` is 1 iff comp row c is allowed for
// batch token b. Raw-KV rows are never masked (the SWA window is always
// fully attended). Masked comp scores are stamped as f16 -INFINITY,
// which softmax converts to 0 weight downstream.
//
// `max_keys_words = ceil(max_keys / 32)` — uniform across batches so the
// indexing is just (b * max_keys_words + word). The caller is
// responsible for sizing the buffer.
//
// `comp_allowed_bits == nullptr` short-circuits to dense (no bit test
// emitted in the hot path). Bit-exact identical to the pre-mask kernel
// in this case.
extern "C" __global__ void attention_mixed_score_batched_htiled_wmma_f16s(
        _Float16       * scores_g,       // [B, n_head, max_keys]  (f16)
        const float    * q,
        const _Float16 * raw_kv,         // f16 V
        const _Float16 * comp_kv,
        const int      * n_raw_per,
        const int      * n_raw_offset_per,
        const int      * n_comp_per,
        const unsigned int * comp_allowed_bits, // nullable; [B, max_keys_words]
        unsigned int     max_keys_words,        // = ceil(max_keys/32)
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys,
        float            kq_scale) {
    const unsigned int b    = blockIdx.z;
    const unsigned int tid  = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int raw_off= (unsigned int)n_raw_offset_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    const unsigned int n_total = n_raw + n_comp;
#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int n0   = blockIdx.x * SCORE_WMMA_KEYS_PER_BLK + warp * 16u;
    const unsigned int col  = lane & 15u;
    const unsigned int kh   = (lane >> 4) * 8u;
    const unsigned int bkey = n0 + col;
    const unsigned int n_kc = head_dim / 16u;
    const _Float16 *brow = nullptr;
    if (bkey < n_total) {
        brow = (bkey < n_raw)
            ? raw_kv + (size_t)(bkey + raw_off) * head_dim
            : comp_kv + (size_t)(bkey - n_raw) * head_dim;
    }
    // CSA mask: per-(b, comp_row) bit test, hoisted out of the m-tile
    // loop since it depends only on (b, bkey) not on the head tile.
    bool key_masked = false;
    if (comp_allowed_bits != nullptr && bkey >= n_raw && bkey < n_total) {
        const unsigned int c = bkey - n_raw;
        const unsigned int word_idx = (size_t)b * max_keys_words + (c >> 5);
        key_masked = ((comp_allowed_bits[word_idx] >> (c & 31)) & 1u) == 0u;
    }
    const unsigned int n_ht = n_head / 16u;
    float8 c[8];
    #pragma unroll
    for (unsigned int t = 0; t < 8; t++) c[t] = (float8){0,0,0,0,0,0,0,0};
    const float *qbase = q + ((size_t)b * n_head) * head_dim;
    for (unsigned int kc = 0; kc < n_kc; kc++) {
        const unsigned int dim0 = kc * 16u + kh;
        half8 bf = {0,0,0,0,0,0,0,0};
        if (brow) {
            #pragma unroll
            for (int e = 0; e < 8; e++) bf[e] = brow[dim0 + (unsigned int)e];
        }
        #pragma unroll
        for (unsigned int mt = 0; mt < 8; mt++) {
            if (mt >= n_ht) break;
            const float *qh = qbase + (size_t)(mt * 16u + col) * head_dim;
            half8 af;
            #pragma unroll
            for (int e = 0; e < 8; e++) af[e] = (_Float16)qh[dim0 + (unsigned int)e];
            c[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf, c[mt]);
        }
    }
    const _Float16 NEG_INF_F16 = (_Float16)(-INFINITY);
    #pragma unroll
    for (unsigned int mt = 0; mt < 8; mt++) {
        if (mt >= n_ht) break;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int h = mt * 16u + (lane >> 4) * 8u + (unsigned int)e;
            if (h < n_head && bkey < n_total) {
                scores_g[((size_t)b * n_head + h) * max_keys + bkey] =
                    key_masked ? NEG_INF_F16 : (_Float16)(c[mt][e] * kq_scale);
            }
        }
    }
#else
    (void)scores_g; (void)q; (void)raw_kv; (void)comp_kv; (void)n_head;
    (void)head_dim; (void)max_keys; (void)kq_scale; (void)warp; (void)lane;
    (void)n_total; (void)comp_allowed_bits; (void)max_keys_words;
#endif
}

// B=1 scalar-arg variant of attention_mixed_score_batched_htiled_wmma
// (f32 scores). Same as the f16s sibling below but writes f32 — slots
// into the existing f32-reading attention_mixed_softmax_wsum without
// needing a paired f16-scores smwsum.
extern "C" __global__ void attention_mixed_score_b1_htiled_wmma(
        float          * scores_g,       // [n_head, max_keys]  (f32)
        const float    * q,              // [n_head, head_dim]
        const _Float16 * raw_kv,
        const _Float16 * comp_kv,
        unsigned int     n_raw,
        unsigned int     raw_off,
        unsigned int     n_comp,
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys,
        float            kq_scale) {
    const unsigned int tid  = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int n_total = n_raw + n_comp;
#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int n0   = blockIdx.x * SCORE_WMMA_KEYS_PER_BLK + warp * 16u;
    const unsigned int col  = lane & 15u;
    const unsigned int kh   = (lane >> 4) * 8u;
    const unsigned int bkey = n0 + col;
    const unsigned int n_kc = head_dim / 16u;
    const _Float16 *brow = nullptr;
    if (bkey < n_total) {
        brow = (bkey < n_raw)
            ? raw_kv + (size_t)(bkey + raw_off) * head_dim
            : comp_kv + (size_t)(bkey - n_raw) * head_dim;
    }
    const unsigned int n_ht = n_head / 16u;
    float8 c[8];
    #pragma unroll
    for (unsigned int t = 0; t < 8; t++) c[t] = (float8){0,0,0,0,0,0,0,0};
    const float *qbase = q;  // b=0
    for (unsigned int kc = 0; kc < n_kc; kc++) {
        const unsigned int dim0 = kc * 16u + kh;
        half8 bf = {0,0,0,0,0,0,0,0};
        if (brow) {
            #pragma unroll
            for (int e = 0; e < 8; e++) bf[e] = brow[dim0 + (unsigned int)e];
        }
        #pragma unroll
        for (unsigned int mt = 0; mt < 8; mt++) {
            if (mt >= n_ht) break;
            const float *qh = qbase + (size_t)(mt * 16u + col) * head_dim;
            half8 af;
            #pragma unroll
            for (int e = 0; e < 8; e++) af[e] = (_Float16)qh[dim0 + (unsigned int)e];
            c[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf, c[mt]);
        }
    }
    #pragma unroll
    for (unsigned int mt = 0; mt < 8; mt++) {
        if (mt >= n_ht) break;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int h = mt * 16u + (lane >> 4) * 8u + (unsigned int)e;
            if (h < n_head && bkey < n_total) {
                scores_g[(size_t)h * max_keys + bkey] = c[mt][e] * kq_scale;
            }
        }
    }
#else
    (void)scores_g; (void)q; (void)raw_kv; (void)comp_kv; (void)n_head;
    (void)head_dim; (void)max_keys; (void)kq_scale; (void)warp; (void)lane;
    (void)n_total; (void)raw_off;
#endif
}

// B=1 scalar-arg variant of attention_mixed_score_batched_htiled_wmma_f16s.
// Identical math; takes n_raw / n_raw_offset / n_comp as scalars instead of
// per-batch device buffers. Wired in the decode path where the buffer-
// indexed variant's per-layer copy_from_host (43 × 2 = 86 ints/token)
// erased the kernel's 6× speedup over launch_score with ~43 ms/token of
// host→device sync overhead. Score writes f16 into scores_g (caller must
// size the buffer accordingly; identical layout to the batched f16s kernel
// at batch=1). Grid: (ceil(n_total/256), 1, 1).
extern "C" __global__ void attention_mixed_score_b1_htiled_wmma_f16s(
        _Float16       * scores_g,       // [n_head, max_keys]  (f16)
        const float    * q,              // [n_head, head_dim]
        const _Float16 * raw_kv,
        const _Float16 * comp_kv,
        unsigned int     n_raw,
        unsigned int     raw_off,
        unsigned int     n_comp,
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys,
        float            kq_scale) {
    const unsigned int tid  = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int n_total = n_raw + n_comp;
#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int n0   = blockIdx.x * SCORE_WMMA_KEYS_PER_BLK + warp * 16u;
    const unsigned int col  = lane & 15u;
    const unsigned int kh   = (lane >> 4) * 8u;
    const unsigned int bkey = n0 + col;
    const unsigned int n_kc = head_dim / 16u;
    const _Float16 *brow = nullptr;
    if (bkey < n_total) {
        brow = (bkey < n_raw)
            ? raw_kv + (size_t)(bkey + raw_off) * head_dim
            : comp_kv + (size_t)(bkey - n_raw) * head_dim;
    }
    const unsigned int n_ht = n_head / 16u;
    float8 c[8];
    #pragma unroll
    for (unsigned int t = 0; t < 8; t++) c[t] = (float8){0,0,0,0,0,0,0,0};
    // b=0 so qbase = q.
    const float *qbase = q;
    for (unsigned int kc = 0; kc < n_kc; kc++) {
        const unsigned int dim0 = kc * 16u + kh;
        half8 bf = {0,0,0,0,0,0,0,0};
        if (brow) {
            #pragma unroll
            for (int e = 0; e < 8; e++) bf[e] = brow[dim0 + (unsigned int)e];
        }
        #pragma unroll
        for (unsigned int mt = 0; mt < 8; mt++) {
            if (mt >= n_ht) break;
            const float *qh = qbase + (size_t)(mt * 16u + col) * head_dim;
            half8 af;
            #pragma unroll
            for (int e = 0; e < 8; e++) af[e] = (_Float16)qh[dim0 + (unsigned int)e];
            c[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf, c[mt]);
        }
    }
    #pragma unroll
    for (unsigned int mt = 0; mt < 8; mt++) {
        if (mt >= n_ht) break;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int h = mt * 16u + (lane >> 4) * 8u + (unsigned int)e;
            if (h < n_head && bkey < n_total) {
                // b=0: row stride is just n_head.
                scores_g[(size_t)h * max_keys + bkey] =
                    (_Float16)(c[mt][e] * kq_scale);
            }
        }
    }
#else
    (void)scores_g; (void)q; (void)raw_kv; (void)comp_kv; (void)n_head;
    (void)head_dim; (void)max_keys; (void)kq_scale; (void)warp; (void)lane;
    (void)n_total; (void)raw_off;
#endif
}

// === LDS-V experimental variant ==========================================
// Same WMMA math as the f32-scores baseline, but each K-tile of 16 keys
// has its V latent rows cooperatively staged through LDS once per tile,
// then the per-warp WMMA B-fragment loads read from LDS instead of DRAM.
// Motivation: rocprofv3 ATT shows 82.8% of smwsum stall cycles on
// `s_wait_loadcnt` (waiting on V-row loads from DRAM); the WGs have low
// effective occupancy on DRAM-latency hiding. Per-tile cooperative load
// converts 16×head_dim DRAM reads into LDS-latency reads thereafter.
//
// LDS budget: 16 keys × head_dim (=512) × 2 bytes f16 = 16384 B = 16 KB
// per WG. RDNA4 CU has 64 KB LDS → still fits 2 WGs/CU + 32 KB headroom.
//
// Layout: v_lds[16 keys × head_dim] f16, no padding (head_dim==512 is
// already aligned). 2-way LDS bank conflict on per-fragment reads (16
// lanes packing into 8 banks at f16) is intrinsic to f16 packing —
// accepted as 2 cycles/read vs ~500 cycles DRAM latency.
extern "C" __global__ void attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv(
        float          * out,
        float          * scores_g,
        const float    * sinks,
        const _Float16 * raw_kv,         // f16 V
        const _Float16 * comp_kv,
        const int      * n_raw_per,
        const int      * n_raw_offset_per,
        const int      * n_comp_per,
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys) {
    const unsigned int hg    = blockIdx.x;
    const unsigned int b     = blockIdx.y;
    const unsigned int tid   = threadIdx.x;
    const unsigned int h0    = hg * SMWSUM_HEAD_TILE;
    if (h0 >= n_head) return;
    unsigned int ht = SMWSUM_HEAD_TILE;
    if (h0 + ht > n_head) ht = n_head - h0;
    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int raw_off= (unsigned int)n_raw_offset_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    const unsigned int n_total = n_raw + n_comp;

    __shared__ float    s_inv[SMWSUM_HEAD_TILE];
    __shared__ _Float16 v_lds[16 * 512];  // 16 keys × head_dim f16 = 16 KB

    // Phase A: softmax — identical to f32-scores baseline.
    const unsigned int wave = tid >> 5;
    const unsigned int lane = tid & 31u;
    if (wave < ht) {
        const unsigned int h = h0 + wave;
        float *sw = scores_g + ((size_t)b * n_head + h) * max_keys;
        const float sink_h = sinks[h];
        float local_max = sink_h;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float v = sw[rr];
            if (v > local_max) local_max = v;
        }
        {
            float o;
            o = __shfl_xor(local_max, 16); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  8); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  4); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  2); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  1); if (o > local_max) local_max = o;
        }
        const float max_score = local_max;
        float local_sum = (lane == 0) ? expf(sink_h - max_score) : 0.0f;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float w = expf(sw[rr] - max_score);
            sw[rr] = w;
            local_sum += w;
        }
        local_sum += __shfl_xor(local_sum, 16);
        local_sum += __shfl_xor(local_sum,  8);
        local_sum += __shfl_xor(local_sum,  4);
        local_sum += __shfl_xor(local_sum,  2);
        local_sum += __shfl_xor(local_sum,  1);
        if (lane == 0) s_inv[wave] = 1.0f / local_sum;
    }
    __syncthreads();

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int warp = wave;
    const unsigned int arow = lane & 15u;
    const unsigned int kh   = (lane >> 4) * 8u;
    const unsigned int nt_a = warp;
    const unsigned int nt_b = warp + 16u;
    const unsigned int d_a  = nt_a * 16u + arow;
    const unsigned int d_b  = nt_b * 16u + arow;
    const float *swrow = scores_g + ((size_t)b * n_head + h0 + arow) * max_keys;
    constexpr unsigned int LDS_STRIDE = 512u;  // head_dim, no padding

    float8 c_a = {0,0,0,0,0,0,0,0};
    float8 c_b = {0,0,0,0,0,0,0,0};

    // Cooperative-load helper: stage V[base..base+16) (raw or comp) into v_lds.
    // f16 V → b128 packs 8 fp16 per load. 16 keys × 512 d × 2 B = 16 KB =
    // 1024 b128 chunks / 512 threads = 2 b128 loads/thread/tile. Bitcopy
    // from DRAM (ulong2) to LDS (ulong2) — no precision loss.
    auto stage_v = [&](const _Float16 *kv_base, unsigned int base, unsigned int n_avail) {
        constexpr unsigned int N_VEC = 16u * (512u / 8u);   // 16 × 64 = 1024 b128
        constexpr unsigned int VEC_PER_KEY = 512u / 8u;     // 64 b128/key
        #pragma unroll
        for (unsigned int it = 0; it < N_VEC / 512u; it++) {
            const unsigned int s = tid + it * 512u;
            const unsigned int k_idx = s / VEC_PER_KEY;
            const unsigned int d_chunk = s % VEC_PER_KEY;
            ulong2 packed;
            if (k_idx < n_avail) {
                const ulong2 *src = (const ulong2 *)(kv_base + (size_t)(base + k_idx) * 512u + d_chunk * 8u);
                packed = *src;
            } else {
                packed.x = 0;
                packed.y = 0;
            }
            ulong2 *dst = (ulong2 *)(v_lds + k_idx * LDS_STRIDE + d_chunk * 8u);
            *dst = packed;
        }
    };

    // --- raw section ---
    unsigned int r = 0;
    for (; r + 16u <= n_raw; r += 16u) {
        __syncthreads();
        stage_v(raw_kv, r + raw_off, 16u);
        __syncthreads();
        half8 a, ba, bb;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k_idx = kh + (unsigned int)e;
            a[e]  = (_Float16)swrow[r + k_idx];
            ba[e] = v_lds[k_idx * LDS_STRIDE + d_a];
            bb[e] = v_lds[k_idx * LDS_STRIDE + d_b];
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    }
    if (r < n_raw) {
        __syncthreads();
        stage_v(raw_kv, r + raw_off, n_raw - r);
        __syncthreads();
        half8 a = {0,0,0,0,0,0,0,0}, ba = {0,0,0,0,0,0,0,0}, bb = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k_idx = kh + (unsigned int)e;
            const unsigned int k = r + k_idx;
            if (k < n_raw) {
                a[e]  = (_Float16)swrow[k];
                ba[e] = v_lds[k_idx * LDS_STRIDE + d_a];
                bb[e] = v_lds[k_idx * LDS_STRIDE + d_b];
            }
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    }

    // --- comp section ---
    if (comp_kv) {
        unsigned int c = 0;
        for (; c + 16u <= n_comp; c += 16u) {
            __syncthreads();
            stage_v(comp_kv, c, 16u);
            __syncthreads();
            half8 a, ba, bb;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int k_idx = kh + (unsigned int)e;
                a[e]  = (_Float16)swrow[n_raw + c + k_idx];
                ba[e] = v_lds[k_idx * LDS_STRIDE + d_a];
                bb[e] = v_lds[k_idx * LDS_STRIDE + d_b];
            }
            c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
            c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
        }
        if (c < n_comp) {
            __syncthreads();
            stage_v(comp_kv, c, n_comp - c);
            __syncthreads();
            half8 a = {0,0,0,0,0,0,0,0}, ba = {0,0,0,0,0,0,0,0}, bb = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int k_idx = kh + (unsigned int)e;
                const unsigned int cc = c + k_idx;
                if (cc < n_comp) {
                    a[e]  = (_Float16)swrow[n_raw + cc];
                    ba[e] = v_lds[k_idx * LDS_STRIDE + d_a];
                    bb[e] = v_lds[k_idx * LDS_STRIDE + d_b];
                }
            }
            c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
            c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
        }
    }

    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned int mh = (lane >> 4) * 8u + (unsigned int)e;
        if (mh < ht) {
            const float scale = s_inv[mh];
            const size_t obase = ((size_t)b * n_head + h0 + mh) * head_dim;
            out[obase + d_a] = c_a[e] * scale;
            out[obase + d_b] = c_b[e] * scale;
        }
    }
#else
    (void)out; (void)raw_kv; (void)comp_kv; (void)head_dim; (void)lane;
#endif
}

// === LDS-V double-buffered variant ========================================
// ATT on _ldsv shows the remaining bottleneck is the V-staging DRAM load
// (s_wait_loadcnt ~33% of stall) and the WG barriers around stage_v
// (s_barrier_wait ~20% of latency). Both come from serialising
// "barrier → load V → barrier → WMMA" per tile. Two ping-pong LDS V buffers
// let tile N+1's stage_v issue its global_load_b128 while the WMMA on tile N
// is in flight, and drops the per-iter barrier count from 2 to 1.
//
// LDS: 2 × 16 KB V buffers + s_inv = 32 KB out of 64 KB CU budget. Still
// fits 2 WGs/CU.
extern "C" __global__ void attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_db(
        float          * out,
        float          * scores_g,
        const float    * sinks,
        const _Float16 * raw_kv,         // f16 V
        const _Float16 * comp_kv,
        const int      * n_raw_per,
        const int      * n_raw_offset_per,
        const int      * n_comp_per,
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys) {
    const unsigned int hg    = blockIdx.x;
    const unsigned int b     = blockIdx.y;
    const unsigned int tid   = threadIdx.x;
    const unsigned int h0    = hg * SMWSUM_HEAD_TILE;
    if (h0 >= n_head) return;
    unsigned int ht = SMWSUM_HEAD_TILE;
    if (h0 + ht > n_head) ht = n_head - h0;
    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int raw_off= (unsigned int)n_raw_offset_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    const unsigned int n_total = n_raw + n_comp;

    __shared__ float    s_inv[SMWSUM_HEAD_TILE];
    __shared__ _Float16 v_lds[2][16 * 512];  // 2 × 16 keys × head_dim f16 = 32 KB

    // Phase A: softmax — identical to _ldsv baseline.
    const unsigned int wave = tid >> 5;
    const unsigned int lane = tid & 31u;
    if (wave < ht) {
        const unsigned int h = h0 + wave;
        float *sw = scores_g + ((size_t)b * n_head + h) * max_keys;
        const float sink_h = sinks[h];
        float local_max = sink_h;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float v = sw[rr];
            if (v > local_max) local_max = v;
        }
        {
            float o;
            o = __shfl_xor(local_max, 16); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  8); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  4); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  2); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  1); if (o > local_max) local_max = o;
        }
        const float max_score = local_max;
        float local_sum = (lane == 0) ? expf(sink_h - max_score) : 0.0f;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float w = expf(sw[rr] - max_score);
            sw[rr] = w;
            local_sum += w;
        }
        local_sum += __shfl_xor(local_sum, 16);
        local_sum += __shfl_xor(local_sum,  8);
        local_sum += __shfl_xor(local_sum,  4);
        local_sum += __shfl_xor(local_sum,  2);
        local_sum += __shfl_xor(local_sum,  1);
        if (lane == 0) s_inv[wave] = 1.0f / local_sum;
    }
    __syncthreads();

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int warp = wave;
    const unsigned int arow = lane & 15u;
    const unsigned int kh   = (lane >> 4) * 8u;
    const unsigned int nt_a = warp;
    const unsigned int nt_b = warp + 16u;
    const unsigned int d_a  = nt_a * 16u + arow;
    const unsigned int d_b  = nt_b * 16u + arow;
    const float *swrow = scores_g + ((size_t)b * n_head + h0 + arow) * max_keys;
    constexpr unsigned int LDS_STRIDE = 512u;

    float8 c_a = {0,0,0,0,0,0,0,0};
    float8 c_b = {0,0,0,0,0,0,0,0};

    const unsigned int n_raw_tiles  = (n_raw + 15u) / 16u;
    const unsigned int n_comp_tiles = comp_kv ? ((n_comp + 15u) / 16u) : 0u;
    const unsigned int total_tiles  = n_raw_tiles + n_comp_tiles;

    // Resolve a logical tile index into (kv_base, key offset within kv_base,
    // available keys in tile, score offset within scores_g row).
    auto tile_info = [&](unsigned int t,
                         const _Float16 *&kv_base,
                         unsigned int &kv_off,
                         unsigned int &n_avail,
                         unsigned int &score_off) {
        if (t < n_raw_tiles) {
            kv_base   = raw_kv;
            kv_off    = t * 16u + raw_off;
            unsigned int kstart = t * 16u;
            n_avail   = (kstart + 16u <= n_raw) ? 16u : (n_raw - kstart);
            score_off = kstart;
        } else {
            const unsigned int ct = t - n_raw_tiles;
            kv_base   = comp_kv;
            kv_off    = ct * 16u;
            unsigned int kstart = ct * 16u;
            n_avail   = (kstart + 16u <= n_comp) ? 16u : (n_comp - kstart);
            score_off = n_raw + kstart;
        }
    };

    // Cooperative load tile t's V into v_lds[buf]. f16 V: b128 packs 8 fp16.
    // No barriers inside — caller controls fencing.
    auto stage_v_buf = [&](unsigned int buf,
                           const _Float16 *kv_base,
                           unsigned int kv_off,
                           unsigned int n_avail) {
        constexpr unsigned int N_VEC = 16u * (512u / 8u);   // 1024 b128
        constexpr unsigned int VEC_PER_KEY = 512u / 8u;     // 64
        #pragma unroll
        for (unsigned int it = 0; it < N_VEC / 512u; it++) {
            const unsigned int s = tid + it * 512u;
            const unsigned int k_idx = s / VEC_PER_KEY;
            const unsigned int d_chunk = s % VEC_PER_KEY;
            ulong2 packed;
            if (k_idx < n_avail) {
                const ulong2 *src = (const ulong2 *)(kv_base + (size_t)(kv_off + k_idx) * 512u + d_chunk * 8u);
                packed = *src;
            } else {
                packed.x = 0;
                packed.y = 0;
            }
            ulong2 *dst = (ulong2 *)(v_lds[buf] + k_idx * LDS_STRIDE + d_chunk * 8u);
            *dst = packed;
        }
    };

    // WMMA on tile from v_lds[buf]. n_avail<16 masks unused keys to zero.
    auto wmma_tile = [&](unsigned int buf,
                         unsigned int score_off,
                         unsigned int n_avail) {
        half8 a = {0,0,0,0,0,0,0,0}, ba = {0,0,0,0,0,0,0,0}, bb = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k_idx = kh + (unsigned int)e;
            if (k_idx < n_avail) {
                a[e]  = (_Float16)swrow[score_off + k_idx];
                ba[e] = v_lds[buf][k_idx * LDS_STRIDE + d_a];
                bb[e] = v_lds[buf][k_idx * LDS_STRIDE + d_b];
            }
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    };

    if (total_tiles > 0) {
        // Prologue: stage tile 0 into buf 0.
        {
            const _Float16 *base0; unsigned int off0, navail0, sco0;
            tile_info(0, base0, off0, navail0, sco0);
            stage_v_buf(0, base0, off0, navail0);
        }
        __syncthreads();

        // Steady-state: stage next tile while WMMA consumes current.
        for (unsigned int t = 0; t + 1 < total_tiles; t++) {
            const unsigned int cur_buf  = t & 1u;
            const unsigned int next_buf = cur_buf ^ 1u;

            const _Float16 *baseN; unsigned int offN, navailN, scoN;
            tile_info(t + 1, baseN, offN, navailN, scoN);
            stage_v_buf(next_buf, baseN, offN, navailN);

            const _Float16 *baseC; unsigned int offC, navailC, scoC;
            tile_info(t, baseC, offC, navailC, scoC);
            wmma_tile(cur_buf, scoC, navailC);

            __syncthreads();
        }

        // Epilogue: consume final tile.
        {
            const unsigned int t = total_tiles - 1u;
            const unsigned int cur_buf = t & 1u;
            const _Float16 *baseF; unsigned int offF, navailF, scoF;
            tile_info(t, baseF, offF, navailF, scoF);
            wmma_tile(cur_buf, scoF, navailF);
        }
    }

    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned int mh = (lane >> 4) * 8u + (unsigned int)e;
        if (mh < ht) {
            const float scale = s_inv[mh];
            const size_t obase = ((size_t)b * n_head + h0 + mh) * head_dim;
            out[obase + d_a] = c_a[e] * scale;
            out[obase + d_b] = c_b[e] * scale;
        }
    }
#else
    (void)out; (void)raw_kv; (void)comp_kv; (void)head_dim; (void)lane;
#endif
}

// === Register-V double-buffered variant ===================================
// Eliminates LDS V entirely — each warp loads its OWN B-fragment slice of
// V directly into VGPRs, then double-buffers across tiles in registers.
// Per warp w, lane l, element e:
//   ba[e] = V[ k = (l>>4)*8 + e ][ d = w*16 + (l&15) ]        (first 256 d)
//   bb[e] = V[ k = (l>>4)*8 + e ][ d = (w+16)*16 + (l&15) ]    (next 256 d)
// This matches the WMMA wave32 B-fragment lane layout used by _ldsv.
//
// Why this dodges the _ldsv_db occupancy collapse:
//   _ldsv_db: 2 × 16 KB LDS V = 32 KB/WG → 1 WG/CU, 16 waves/CU
//   _regv_db: 64 B LDS (s_inv only) → wave-limited at 4 WGs/CU, 64 waves/CU
//                                  = 100% wave occupancy (vs 75% for _ldsv)
//   VGPR cost: 4 half8 (ba_cur, bb_cur, ba_next, bb_next) = 16 fp16 = 8 VGPR/lane
//
// What this kills:
//   - all per-tile __syncthreads (had been ~20% of latency in _ldsv)
//   - all ds_load_u16 / ds_store_b64 on V (had been ~16% of stall)
//   - some of the 66% s_wait_loadcnt via +33% wave parallelism for DRAM hiding
// What stays:
//   - same total DRAM traffic for V (cooperative LDS load and per-wave load
//     both touch the same cache lines; L2 catches the redundancy)
//   - s_inv still in LDS (64 B) — cross-warp share for output scaling
extern "C" __global__ void attention_mixed_softmax_wsum_batched_htiled_wmma_regv_db(
        float          * out,
        float          * scores_g,
        const float    * sinks,
        const _Float16 * raw_kv,         // f16 V
        const _Float16 * comp_kv,
        const int      * n_raw_per,
        const int      * n_raw_offset_per,
        const int      * n_comp_per,
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys) {
    const unsigned int hg    = blockIdx.x;
    const unsigned int b     = blockIdx.y;
    const unsigned int tid   = threadIdx.x;
    const unsigned int h0    = hg * SMWSUM_HEAD_TILE;
    if (h0 >= n_head) return;
    unsigned int ht = SMWSUM_HEAD_TILE;
    if (h0 + ht > n_head) ht = n_head - h0;
    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int raw_off= (unsigned int)n_raw_offset_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    const unsigned int n_total = n_raw + n_comp;

    __shared__ float s_inv[SMWSUM_HEAD_TILE];

    const unsigned int wave = tid >> 5;
    const unsigned int lane = tid & 31u;

    // Phase A: softmax — identical to _ldsv baseline.
    if (wave < ht) {
        const unsigned int h = h0 + wave;
        float *sw = scores_g + ((size_t)b * n_head + h) * max_keys;
        const float sink_h = sinks[h];
        float local_max = sink_h;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float v = sw[rr];
            if (v > local_max) local_max = v;
        }
        {
            float o;
            o = __shfl_xor(local_max, 16); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  8); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  4); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  2); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  1); if (o > local_max) local_max = o;
        }
        const float max_score = local_max;
        float local_sum = (lane == 0) ? expf(sink_h - max_score) : 0.0f;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float w = expf(sw[rr] - max_score);
            sw[rr] = w;
            local_sum += w;
        }
        local_sum += __shfl_xor(local_sum, 16);
        local_sum += __shfl_xor(local_sum,  8);
        local_sum += __shfl_xor(local_sum,  4);
        local_sum += __shfl_xor(local_sum,  2);
        local_sum += __shfl_xor(local_sum,  1);
        if (lane == 0) s_inv[wave] = 1.0f / local_sum;
    }
    __syncthreads();  // s_inv writes → cross-warp visible

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int warp = wave;
    const unsigned int arow = lane & 15u;
    const unsigned int kh   = (lane >> 4) * 8u;
    const unsigned int nt_a = warp;
    const unsigned int nt_b = warp + 16u;
    const unsigned int d_a  = nt_a * 16u + arow;
    const unsigned int d_b  = nt_b * 16u + arow;
    const float *swrow = scores_g + ((size_t)b * n_head + h0 + arow) * max_keys;

    float8 c_a = {0,0,0,0,0,0,0,0};
    float8 c_b = {0,0,0,0,0,0,0,0};

    const unsigned int n_raw_tiles  = (n_raw + 15u) / 16u;
    const unsigned int n_comp_tiles = comp_kv ? ((n_comp + 15u) / 16u) : 0u;
    const unsigned int total_tiles  = n_raw_tiles + n_comp_tiles;

    // Resolve a logical tile index into (kv_base, key offset, n_avail, score
    // offset). Same layout as _ldsv_db.
    auto tile_info = [&](unsigned int t,
                         const _Float16 *&kv_base,
                         unsigned int &kv_off,
                         unsigned int &n_avail,
                         unsigned int &score_off) {
        if (t < n_raw_tiles) {
            kv_base   = raw_kv;
            const unsigned int kstart = t * 16u;
            kv_off    = kstart + raw_off;
            n_avail   = (kstart + 16u <= n_raw) ? 16u : (n_raw - kstart);
            score_off = kstart;
        } else {
            const unsigned int ct = t - n_raw_tiles;
            const unsigned int kstart = ct * 16u;
            kv_base   = comp_kv;
            kv_off    = kstart;
            n_avail   = (kstart + 16u <= n_comp) ? 16u : (n_comp - kstart);
            score_off = n_raw + kstart;
        }
    };

    // Load this lane's slice of V[tile_t] into a half8 pair (ba, bb).
    // Per lane, 8 keys × 1 d-element each = 8 fp16. The 8 keys per lane are
    // not contiguous in memory (stride = head_dim) — compiler emits 8
    // global_load_u16 per lane per fragment (one per K row).
    auto load_v_tile = [&](unsigned int t, half8 &ba, half8 &bb) {
        const _Float16 *kv_base; unsigned int kv_off, n_avail, score_off;
        tile_info(t, kv_base, kv_off, n_avail, score_off);
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k_idx = kh + (unsigned int)e;
            if (k_idx < n_avail) {
                const _Float16 *row = kv_base + (size_t)(kv_off + k_idx) * 512u;
                ba[e] = row[d_a];
                bb[e] = row[d_b];
            } else {
                ba[e] = (_Float16)0.0f;
                bb[e] = (_Float16)0.0f;
            }
        }
    };

    // WMMA on V already in (ba, bb), reading scores for tile t.
    auto compute_tile = [&](unsigned int t, half8 ba, half8 bb) {
        const _Float16 *kv_base_unused; unsigned int kv_off_unused, n_avail, score_off;
        tile_info(t, kv_base_unused, kv_off_unused, n_avail, score_off);
        half8 a = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k_idx = kh + (unsigned int)e;
            if (k_idx < n_avail) {
                a[e] = (_Float16)swrow[score_off + k_idx];
            }
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    };

    if (total_tiles > 0) {
        // Prologue: prefetch tile 0 into ba_cur / bb_cur.
        half8 ba_cur = {0,0,0,0,0,0,0,0}, bb_cur = {0,0,0,0,0,0,0,0};
        half8 ba_next = {0,0,0,0,0,0,0,0}, bb_next = {0,0,0,0,0,0,0,0};
        load_v_tile(0, ba_cur, bb_cur);

        for (unsigned int t = 0; t + 1 < total_tiles; t++) {
            // Issue next-tile V loads. With DB in registers, the compiler is
            // free to interleave these global_load_b32 with the WMMA on _cur.
            load_v_tile(t + 1, ba_next, bb_next);
            compute_tile(t, ba_cur, bb_cur);
            ba_cur = ba_next;
            bb_cur = bb_next;
        }
        // Epilogue: final tile (already in ba_cur/bb_cur).
        compute_tile(total_tiles - 1u, ba_cur, bb_cur);
    }

    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned int mh = (lane >> 4) * 8u + (unsigned int)e;
        if (mh < ht) {
            const float scale = s_inv[mh];
            const size_t obase = ((size_t)b * n_head + h0 + mh) * head_dim;
            out[obase + d_a] = c_a[e] * scale;
            out[obase + d_b] = c_b[e] * scale;
        }
    }
#else
    (void)out; (void)raw_kv; (void)comp_kv; (void)head_dim; (void)lane;
#endif
}

extern "C" __global__ void attention_mixed_softmax_wsum_batched_htiled_wmma_f16s(
        float          * out,
        _Float16       * scores_g,       // [B, n_head, max_keys]  (f16, in/out)
        const float    * sinks,
        const _Float16 * raw_kv,         // f16 V
        const _Float16 * comp_kv,
        const int      * n_raw_per,
        const int      * n_raw_offset_per,
        const int      * n_comp_per,
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys) {
    const unsigned int hg    = blockIdx.x;
    const unsigned int b     = blockIdx.y;
    const unsigned int tid   = threadIdx.x;
    const unsigned int h0    = hg * SMWSUM_HEAD_TILE;
    if (h0 >= n_head) return;
    unsigned int ht = SMWSUM_HEAD_TILE;
    if (h0 + ht > n_head) ht = n_head - h0;
    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int raw_off= (unsigned int)n_raw_offset_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    const unsigned int n_total = n_raw + n_comp;
    __shared__ float s_inv[SMWSUM_HEAD_TILE];

    // Phase A: softmax — reads f16, computes in f32, writes back f16.
    const unsigned int wave = tid >> 5;
    const unsigned int lane = tid & 31u;
    if (wave < ht) {
        const unsigned int h = h0 + wave;
        _Float16 *sw = scores_g + ((size_t)b * n_head + h) * max_keys;
        const float sink_h = sinks[h];
        float local_max = sink_h;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float v = (float)sw[rr];
            if (v > local_max) local_max = v;
        }
        {
            float o;
            o = __shfl_xor(local_max, 16); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  8); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  4); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  2); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  1); if (o > local_max) local_max = o;
        }
        const float max_score = local_max;
        float local_sum = (lane == 0) ? expf(sink_h - max_score) : 0.0f;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float w = expf((float)sw[rr] - max_score);
            sw[rr] = (_Float16)w;
            local_sum += w;
        }
        local_sum += __shfl_xor(local_sum, 16);
        local_sum += __shfl_xor(local_sum,  8);
        local_sum += __shfl_xor(local_sum,  4);
        local_sum += __shfl_xor(local_sum,  2);
        local_sum += __shfl_xor(local_sum,  1);
        if (lane == 0) s_inv[wave] = 1.0f / local_sum;
    }
    __syncthreads();

#if defined(__gfx1200__) || defined(__gfx1201__)
    // Phase B: WMMA GEMM using f16 scores directly as A-fragment (no cast).
    const unsigned int warp = wave;
    const unsigned int arow = lane & 15u;
    const unsigned int kh   = (lane >> 4) * 8u;
    const unsigned int nt_a = warp;
    const unsigned int nt_b = warp + 16u;
    const unsigned int d_a  = nt_a * 16u + arow;
    const unsigned int d_b  = nt_b * 16u + arow;
    const _Float16 *swrow = scores_g + ((size_t)b * n_head + h0 + arow) * max_keys;

    float8 c_a = {0,0,0,0,0,0,0,0};
    float8 c_b = {0,0,0,0,0,0,0,0};

    unsigned int r = 0;
    for (; r + 16u <= n_raw; r += 16u) {
        half8 a, ba, bb;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k = r + kh + (unsigned int)e;
            a[e]  = swrow[k];
            ba[e] = raw_kv[(size_t)(k + raw_off) * head_dim + d_a];
            bb[e] = raw_kv[(size_t)(k + raw_off) * head_dim + d_b];
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    }
    if (r < n_raw) {
        half8 a = {0,0,0,0,0,0,0,0}, ba = {0,0,0,0,0,0,0,0}, bb = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k = r + kh + (unsigned int)e;
            if (k < n_raw) {
                a[e]  = swrow[k];
                ba[e] = raw_kv[(size_t)(k + raw_off) * head_dim + d_a];
                bb[e] = raw_kv[(size_t)(k + raw_off) * head_dim + d_b];
            }
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    }
    if (comp_kv) {
        unsigned int c = 0;
        for (; c + 16u <= n_comp; c += 16u) {
            half8 a, ba, bb;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int cc = c + kh + (unsigned int)e;
                a[e]  = swrow[n_raw + cc];
                ba[e] = comp_kv[(size_t)cc * head_dim + d_a];
                bb[e] = comp_kv[(size_t)cc * head_dim + d_b];
            }
            c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
            c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
        }
        if (c < n_comp) {
            half8 a = {0,0,0,0,0,0,0,0}, ba = {0,0,0,0,0,0,0,0}, bb = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int cc = c + kh + (unsigned int)e;
                if (cc < n_comp) {
                    a[e]  = swrow[n_raw + cc];
                    ba[e] = comp_kv[(size_t)cc * head_dim + d_a];
                    bb[e] = comp_kv[(size_t)cc * head_dim + d_b];
                }
            }
            c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
            c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
        }
    }
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned int mh = (lane >> 4) * 8u + (unsigned int)e;
        if (mh < ht) {
            const float scale = s_inv[mh];
            const size_t obase = ((size_t)b * n_head + h0 + mh) * head_dim;
            out[obase + d_a] = c_a[e] * scale;
            out[obase + d_b] = c_b[e] * scale;
        }
    }
#else
    (void)out; (void)raw_kv; (void)comp_kv; (void)head_dim; (void)lane;
#endif
}

// === Fused FlashAttention-style WMMA kernel ===============================
// Replaces (attention_mixed_score_batched_htiled_wmma +
// attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv) with a single
// streaming pass. Online softmax in f32 registers, never materializes
// scores to DRAM. Q stays in LDS for the whole token; V is staged to LDS
// per K-tile and reused by score and W·V phases.
//
// Grid: (n_head/16, B, 1). Block 512 = 16 warps.
// LDS per WG: q_lds 16 KB + v_lds 32 KB + scratch/s_lds 16 KB = 64 KB (max).
//
// Score phase uses split-K across all 16 warps (each owns 2 K-chunks), with
// cross-warp partial sums combined via LDS tree reduction (4 rounds,
// log2(16)=4). Prior implementation used `atomicAdd` to LDS; rocprofv3 ATT
// showed 26.8% stall on `ds_add_f32` due to RDNA4's LDS atomic unit
// serialising 16 contributors per cell.
extern "C" __global__ void attention_mixed_fused_wmma(
        float          * out,            // [B, n_head, head_dim]
        const float    * q,              // [B, n_head, head_dim]
        const _Float16 * raw_kv,         // f16 V
        const _Float16 * comp_kv,
        const float    * sinks,          // [n_head]
        const int      * n_raw_per,
        const int      * n_raw_offset_per,
        const int      * n_comp_per,
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys_unused,
        float            kq_scale) {
    (void)max_keys_unused; (void)head_dim;  // head_dim fixed at 512
    constexpr unsigned int HD = 512u;
    constexpr unsigned int HT = 16u;
    constexpr unsigned int BN = 32u;

    const unsigned int hg  = blockIdx.x;
    const unsigned int b   = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int h0  = hg * HT;
    if (h0 >= n_head) return;
    unsigned int ht = HT;
    if (h0 + ht > n_head) ht = n_head - h0;

    const unsigned int wave = tid >> 5;
    const unsigned int lane = tid & 31u;

    const unsigned int n_raw   = (unsigned int)n_raw_per[b];
    const unsigned int raw_off = (unsigned int)n_raw_offset_per[b];
    const unsigned int n_comp  = (unsigned int)n_comp_per[b];

    __shared__ _Float16 q_lds[HT * HD];          // 16 KB
    __shared__ _Float16 v_lds[BN * HD];          // 32 KB
    // Tree-reduction scratch — only the "writers" of each round store to LDS,
    // "readers" keep their partial sums in registers, so we only ever need
    // half the partials' worth of LDS (8 slots × 256 f32 = 8 KB at peak,
    // shrinking each round).
    __shared__ float    scratch_lds[8 * 256];     //  8 KB
    __shared__ float    s_lds[HT * BN];           //  2 KB
    __shared__ float    alpha_lds[HT];            // 64 B
    __shared__ float    inv_l_lds[HT];            // 64 B

    // === Step 3: cooperative Q-tile load ===
    {
        const float *q_base = q + (size_t)(b * n_head + h0) * HD;
        #pragma unroll
        for (unsigned int it = 0; it < 4; it++) {
            unsigned int s = tid + it * 512u;
            unsigned int k_idx   = s / (HD / 4u);
            unsigned int d_chunk = s % (HD / 4u);
            float4 vv = {0.0f, 0.0f, 0.0f, 0.0f};
            if (k_idx < ht) {
                vv = *(const float4 *)(q_base + (size_t)k_idx * HD + d_chunk * 4u);
            }
            _Float16 *dst = q_lds + k_idx * HD + d_chunk * 4u;
            dst[0] = (_Float16)vv.x; dst[1] = (_Float16)vv.y;
            dst[2] = (_Float16)vv.z; dst[3] = (_Float16)vv.w;
        }
    }
    __syncthreads();

    float m_i = -INFINITY, l_i = 0.0f;
    if (wave < ht) {
        m_i = sinks[h0 + wave];
        l_i = 1.0f;
    }

    float8 c_a = {0,0,0,0,0,0,0,0};
    float8 c_b = {0,0,0,0,0,0,0,0};

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int arow = lane & 15u;
    const unsigned int kh   = (lane >> 4) * 8u;
    const unsigned int d_a  = wave * 16u + arow;
    const unsigned int d_b  = (wave + 16u) * 16u + arow;

    // === Step 4-5: outer K-tile loop, raw then comp section ===
    for (int section = 0; section < 2; section++) {
        const _Float16 *kv_base = (section == 0) ? raw_kv : comp_kv;
        unsigned int n_section = (section == 0) ? n_raw : n_comp;
        unsigned int sec_off   = (section == 0) ? raw_off : 0u;
        if (section == 1 && !comp_kv) break;

        for (unsigned int r = 0; r < n_section; r += BN) {
            unsigned int tile_n = (n_section - r < BN) ? (n_section - r) : BN;

            // V cooperative load (32 keys × 512 dim f16 = 32 KB). f16 V:
            // b128 packs 8 fp16, so HD/8 = 64 chunks/key, 32 × 64 = 2048
            // b128 / 512 threads = 4 iters/thread (was 8 for f32 V).
            #pragma unroll
            for (unsigned int it = 0; it < 4; it++) {
                unsigned int s = tid + it * 512u;
                unsigned int k_idx   = s / (HD / 8u);
                unsigned int d_chunk = s % (HD / 8u);
                ulong2 packed;
                if (k_idx < tile_n) {
                    unsigned int slot = sec_off + r + k_idx;
                    packed = *(const ulong2 *)(kv_base + (size_t)slot * HD + d_chunk * 8u);
                } else {
                    packed.x = 0;
                    packed.y = 0;
                }
                ulong2 *dst = (ulong2 *)(v_lds + k_idx * HD + d_chunk * 8u);
                *dst = packed;
            }
            __syncthreads();

            // -- Score Q·Vᵀ → s_lds (split-K across all 16 warps, tree reduce) --
            // Each warp owns 2 of 32 head_dim K-tiles → 4 WMMAs/warp,
            // accumulating into per-warp cs0/cs1 (2 N-tiles per warp).
            // The 16 warps' partial sums are reduced via LDS tree reduction
            // (log2(16)=4 rounds, pairwise), NOT atomicAdd — RDNA4 ds_add_f32
            // serialises 16 contributors per cell on the LDS atomic unit.
            float8 cs0 = {0,0,0,0,0,0,0,0};
            float8 cs1 = {0,0,0,0,0,0,0,0};
            {
                unsigned int kc_base = wave * 2u;
                #pragma unroll
                for (unsigned int kk = 0; kk < 2u; kk++) {
                    unsigned int kc = kc_base + kk;
                    unsigned int dim0 = kc * 16u;
                    half8 a, b_n0, b_n1;
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        a[e]    = q_lds[arow * HD + dim0 + kh + (unsigned int)e];
                        b_n0[e] = v_lds[arow * HD + dim0 + kh + (unsigned int)e];
                        b_n1[e] = v_lds[(16u + arow) * HD + dim0 + kh + (unsigned int)e];
                    }
                    cs0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b_n0, cs0);
                    cs1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b_n1, cs1);
                }
            }

            // Tree reduction across 16 warps' cs0/cs1 partials.
            // Strategy: only "writer" warps store each round; "reader" warps
            // keep their running sum in registers. LDS peak = 8 KB (round 1).
            //
            // C-fragment lane mapping (gfx12 wave32, 16x16x16):
            //   cs[e] for lane l   <->   S[ h = (l>>4)*8 + e , n = l & 15 ]
            // i.e. lanes 0..15 cover heads 0..7, lanes 16..31 heads 8..15;
            // `e` indexes 8 heads within that half-bucket; `lane & 15` is the
            // key index within the 16-key N-tile.
            //
            // After the reduction below, wave 0's cs0/cs1 (in registers) hold
            // the full 16-way sum, distributed across its 32 lanes per the
            // mapping above. Wave 0 then writes the scaled+masked result
            // directly to s_lds (no extra LDS staging needed).

            // ---- Tree reduction for cs0 (N-tile 0) ----
            // Round 1: waves 8..15 -> slots 0..7. Waves 0..7 sum into registers.
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave >= 8u) {
                    scratch_lds[(wave - 8u) * 256u + (unsigned int)e * 32u + lane] = cs0[e];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave < 8u) {
                    cs0[e] += scratch_lds[wave * 256u + (unsigned int)e * 32u + lane];
                }
            }
            __syncthreads();
            // Round 2: waves 4..7 -> slots 0..3. Waves 0..3 read.
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave >= 4u && wave < 8u) {
                    scratch_lds[(wave - 4u) * 256u + (unsigned int)e * 32u + lane] = cs0[e];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave < 4u) {
                    cs0[e] += scratch_lds[wave * 256u + (unsigned int)e * 32u + lane];
                }
            }
            __syncthreads();
            // Round 3: waves 2..3 -> slots 0..1. Waves 0..1 read.
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave >= 2u && wave < 4u) {
                    scratch_lds[(wave - 2u) * 256u + (unsigned int)e * 32u + lane] = cs0[e];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave < 2u) {
                    cs0[e] += scratch_lds[wave * 256u + (unsigned int)e * 32u + lane];
                }
            }
            __syncthreads();
            // Round 4: wave 1 -> slot 0. Wave 0 reads.
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave == 1u) {
                    scratch_lds[(unsigned int)e * 32u + lane] = cs0[e];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave == 0u) {
                    cs0[e] += scratch_lds[(unsigned int)e * 32u + lane];
                }
            }
            __syncthreads();
            // Wave 0 has full S for N-tile 0. Write to s_lds (scale + mask).
            // lane l, e=0..7 holds S[h=(l>>4)*8+e, k_local=l&15].
            if (wave == 0u) {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    unsigned int h_idx   = (lane >> 4) * 8u + (unsigned int)e;
                    unsigned int k_local = lane & 15u;
                    if (h_idx < HT) {
                        float v = cs0[e];
                        v = (k_local < tile_n) ? (v * kq_scale) : -INFINITY;
                        s_lds[h_idx * BN + k_local] = v;
                    }
                }
            }
            // (no syncthreads here; we sync after cs1 is also written)

            // ---- Tree reduction for cs1 (N-tile 1) ----
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave >= 8u) {
                    scratch_lds[(wave - 8u) * 256u + (unsigned int)e * 32u + lane] = cs1[e];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave < 8u) {
                    cs1[e] += scratch_lds[wave * 256u + (unsigned int)e * 32u + lane];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave >= 4u && wave < 8u) {
                    scratch_lds[(wave - 4u) * 256u + (unsigned int)e * 32u + lane] = cs1[e];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave < 4u) {
                    cs1[e] += scratch_lds[wave * 256u + (unsigned int)e * 32u + lane];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave >= 2u && wave < 4u) {
                    scratch_lds[(wave - 2u) * 256u + (unsigned int)e * 32u + lane] = cs1[e];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave < 2u) {
                    cs1[e] += scratch_lds[wave * 256u + (unsigned int)e * 32u + lane];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave == 1u) {
                    scratch_lds[(unsigned int)e * 32u + lane] = cs1[e];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (wave == 0u) {
                    cs1[e] += scratch_lds[(unsigned int)e * 32u + lane];
                }
            }
            __syncthreads();
            if (wave == 0u) {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    unsigned int h_idx    = (lane >> 4) * 8u + (unsigned int)e;
                    unsigned int k_local  = lane & 15u;
                    unsigned int k_global = k_local + 16u;
                    if (h_idx < HT) {
                        float v = cs1[e];
                        v = (k_global < tile_n) ? (v * kq_scale) : -INFINITY;
                        s_lds[h_idx * BN + k_global] = v;
                    }
                }
            }
            __syncthreads();

            // -- Online softmax (per-wave) --
            if (wave < ht) {
                float my_max = -INFINITY;
                for (unsigned int kk = lane; kk < BN; kk += 32u) {
                    float v = s_lds[wave * BN + kk];
                    if (v > my_max) my_max = v;
                }
                {
                    float o;
                    o = __shfl_xor(my_max, 16); if (o > my_max) my_max = o;
                    o = __shfl_xor(my_max,  8); if (o > my_max) my_max = o;
                    o = __shfl_xor(my_max,  4); if (o > my_max) my_max = o;
                    o = __shfl_xor(my_max,  2); if (o > my_max) my_max = o;
                    o = __shfl_xor(my_max,  1); if (o > my_max) my_max = o;
                }
                float m_new = fmaxf(m_i, my_max);
                float alpha = expf(m_i - m_new);
                float my_sum = 0.0f;
                for (unsigned int kk = lane; kk < BN; kk += 32u) {
                    float w = expf(s_lds[wave * BN + kk] - m_new);
                    s_lds[wave * BN + kk] = w;
                    my_sum += w;
                }
                my_sum += __shfl_xor(my_sum, 16);
                my_sum += __shfl_xor(my_sum,  8);
                my_sum += __shfl_xor(my_sum,  4);
                my_sum += __shfl_xor(my_sum,  2);
                my_sum += __shfl_xor(my_sum,  1);
                m_i = m_new;
                l_i = l_i * alpha + my_sum;
                if (lane == 0) alpha_lds[wave] = alpha;
            }
            __syncthreads();

            // -- Rescale C-fragment by alpha[head] --
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int h_idx = kh + (unsigned int)e;
                float a_h = (h_idx < HT) ? alpha_lds[h_idx] : 0.0f;
                c_a[e] *= a_h;
                c_b[e] *= a_h;
            }

            // -- W·V WMMA: C += P[16, 32] @ V[32, 512] --
            #pragma unroll
            for (unsigned int kc = 0; kc < 2u; kc++) {
                unsigned int key_base = kc * 16u;
                half8 a, ba, bb;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    unsigned int k_local  = kh + (unsigned int)e;
                    unsigned int k_global = key_base + k_local;
                    a[e]  = (_Float16)s_lds[arow * BN + k_global];
                    ba[e] = v_lds[k_global * HD + d_a];
                    bb[e] = v_lds[k_global * HD + d_b];
                }
                c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
                c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
            }
            __syncthreads();
        }
    }

    // === Step 6: final scale + store ===
    if (wave < ht && lane == 0) {
        inv_l_lds[wave] = 1.0f / l_i;
    }
    __syncthreads();
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        unsigned int h_idx = kh + (unsigned int)e;
        if (h_idx < ht) {
            float scale = inv_l_lds[h_idx];
            const size_t obase = ((size_t)b * n_head + h0 + h_idx) * HD;
            out[obase + d_a] = c_a[e] * scale;
            out[obase + d_b] = c_b[e] * scale;
        }
    }
#else
    (void)out; (void)q; (void)raw_kv; (void)comp_kv; (void)sinks;
    (void)n_raw; (void)raw_off; (void)n_comp;
    (void)wave; (void)lane; (void)kq_scale;
#endif
}




// === LDS-V + f16 scores combined variant =================================
// Combines the LDS-V staging optimization (from `_ldsv`) with f16 scores
// (from `_f16s`). Pairs with `attention_mixed_score_batched_htiled_wmma_f16s`
// to halve the Phase A score-DRAM traffic — the 28%-of-stall remaining
// after f16 V.
extern "C" __global__ void attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s(
        float          * out,
        _Float16       * scores_g,       // [B, n_head, max_keys]  (f16, in/out)
        const float    * sinks,
        const _Float16 * raw_kv,
        const _Float16 * comp_kv,
        const int      * n_raw_per,
        const int      * n_raw_offset_per,
        const int      * n_comp_per,
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys) {
    const unsigned int hg    = blockIdx.x;
    const unsigned int b     = blockIdx.y;
    const unsigned int tid   = threadIdx.x;
    const unsigned int h0    = hg * SMWSUM_HEAD_TILE;
    if (h0 >= n_head) return;
    unsigned int ht = SMWSUM_HEAD_TILE;
    if (h0 + ht > n_head) ht = n_head - h0;
    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int raw_off= (unsigned int)n_raw_offset_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    const unsigned int n_total = n_raw + n_comp;

    __shared__ float    s_inv[SMWSUM_HEAD_TILE];
    __shared__ _Float16 v_lds[16 * 512];  // 16 KB

    // Phase A: softmax — reads f16 scores, computes in f32, writes f16 weights.
    const unsigned int wave = tid >> 5;
    const unsigned int lane = tid & 31u;
    if (wave < ht) {
        const unsigned int h = h0 + wave;
        _Float16 *sw = scores_g + ((size_t)b * n_head + h) * max_keys;
        const float sink_h = sinks[h];
        float local_max = sink_h;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float v = (float)sw[rr];
            if (v > local_max) local_max = v;
        }
        {
            float o;
            o = __shfl_xor(local_max, 16); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  8); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  4); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  2); if (o > local_max) local_max = o;
            o = __shfl_xor(local_max,  1); if (o > local_max) local_max = o;
        }
        const float max_score = local_max;
        float local_sum = (lane == 0) ? expf(sink_h - max_score) : 0.0f;
        for (unsigned int rr = lane; rr < n_total; rr += 32) {
            float w = expf((float)sw[rr] - max_score);
            sw[rr] = (_Float16)w;
            local_sum += w;
        }
        local_sum += __shfl_xor(local_sum, 16);
        local_sum += __shfl_xor(local_sum,  8);
        local_sum += __shfl_xor(local_sum,  4);
        local_sum += __shfl_xor(local_sum,  2);
        local_sum += __shfl_xor(local_sum,  1);
        if (lane == 0) s_inv[wave] = 1.0f / local_sum;
    }
    __syncthreads();

#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int warp = wave;
    const unsigned int arow = lane & 15u;
    const unsigned int kh   = (lane >> 4) * 8u;
    const unsigned int nt_a = warp;
    const unsigned int nt_b = warp + 16u;
    const unsigned int d_a  = nt_a * 16u + arow;
    const unsigned int d_b  = nt_b * 16u + arow;
    const _Float16 *swrow = scores_g + ((size_t)b * n_head + h0 + arow) * max_keys;
    constexpr unsigned int LDS_STRIDE = 512u;

    float8 c_a = {0,0,0,0,0,0,0,0};
    float8 c_b = {0,0,0,0,0,0,0,0};

    // Cooperative V stage (same as _ldsv).
    auto stage_v = [&](const _Float16 *kv_base, unsigned int base, unsigned int n_avail) {
        constexpr unsigned int N_VEC = 16u * (512u / 8u);   // 1024 b128
        constexpr unsigned int VEC_PER_KEY = 512u / 8u;     // 64
        #pragma unroll
        for (unsigned int it = 0; it < N_VEC / 512u; it++) {
            const unsigned int s = tid + it * 512u;
            const unsigned int k_idx = s / VEC_PER_KEY;
            const unsigned int d_chunk = s % VEC_PER_KEY;
            ulong2 packed;
            if (k_idx < n_avail) {
                const ulong2 *src = (const ulong2 *)(kv_base + (size_t)(base + k_idx) * 512u + d_chunk * 8u);
                packed = *src;
            } else {
                packed.x = 0;
                packed.y = 0;
            }
            ulong2 *dst = (ulong2 *)(v_lds + k_idx * LDS_STRIDE + d_chunk * 8u);
            *dst = packed;
        }
    };

    // --- raw section ---
    unsigned int r = 0;
    for (; r + 16u <= n_raw; r += 16u) {
        __syncthreads();
        stage_v(raw_kv, r + raw_off, 16u);
        __syncthreads();
        half8 a, ba, bb;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k_idx = kh + (unsigned int)e;
            a[e]  = swrow[r + k_idx];                       // f16 directly
            ba[e] = v_lds[k_idx * LDS_STRIDE + d_a];
            bb[e] = v_lds[k_idx * LDS_STRIDE + d_b];
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    }
    if (r < n_raw) {
        __syncthreads();
        stage_v(raw_kv, r + raw_off, n_raw - r);
        __syncthreads();
        half8 a = {0,0,0,0,0,0,0,0}, ba = {0,0,0,0,0,0,0,0}, bb = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int k_idx = kh + (unsigned int)e;
            const unsigned int k = r + k_idx;
            if (k < n_raw) {
                a[e]  = swrow[k];
                ba[e] = v_lds[k_idx * LDS_STRIDE + d_a];
                bb[e] = v_lds[k_idx * LDS_STRIDE + d_b];
            }
        }
        c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
        c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
    }

    // --- comp section ---
    if (comp_kv) {
        unsigned int c = 0;
        for (; c + 16u <= n_comp; c += 16u) {
            __syncthreads();
            stage_v(comp_kv, c, 16u);
            __syncthreads();
            half8 a, ba, bb;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int k_idx = kh + (unsigned int)e;
                a[e]  = swrow[n_raw + c + k_idx];
                ba[e] = v_lds[k_idx * LDS_STRIDE + d_a];
                bb[e] = v_lds[k_idx * LDS_STRIDE + d_b];
            }
            c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
            c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
        }
        if (c < n_comp) {
            __syncthreads();
            stage_v(comp_kv, c, n_comp - c);
            __syncthreads();
            half8 a = {0,0,0,0,0,0,0,0}, ba = {0,0,0,0,0,0,0,0}, bb = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int k_idx = kh + (unsigned int)e;
                const unsigned int cc = c + k_idx;
                if (cc < n_comp) {
                    a[e]  = swrow[n_raw + cc];
                    ba[e] = v_lds[k_idx * LDS_STRIDE + d_a];
                    bb[e] = v_lds[k_idx * LDS_STRIDE + d_b];
                }
            }
            c_a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, ba, c_a);
            c_b = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, c_b);
        }
    }

    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned int mh = (lane >> 4) * 8u + (unsigned int)e;
        if (mh < ht) {
            const float scale = s_inv[mh];
            const size_t obase = ((size_t)b * n_head + h0 + mh) * head_dim;
            out[obase + d_a] = c_a[e] * scale;
            out[obase + d_b] = c_b[e] * scale;
        }
    }
#else
    (void)out; (void)raw_kv; (void)comp_kv; (void)head_dim; (void)lane;
#endif
}
