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
// Two caps:
//   ATTN_PREFILL_LDS_MAX — LDS-bound, used by `attention_mixed_batched`
//     (prefill). 2304 covers 128 raw + 2176 comp ≈ 8.8K ctx at ratio=4.
//     Capped by 64 KB LDS / 2 arrays. Prefill chunks are typically
//     small (B_MAX=256) so context cap is the binding constraint, but
//     attention is O(B * n_total) anyway — a 64K-token prefill in one
//     chunk doesn't fit memory regardless.
//   ATTN_MIXED_MAX_KEYS — global-mem-bound, used by split decode
//     kernels (score / softmax_wsum). 16640 covers 128 raw + 16384
//     comp ≈ 65K ctx at ratio=4. Scratch lives in
//     DgpuScratch.attn_scores (caller-supplied global buffer).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// RDNA4 (gfx12) WMMA fragment types for the WMMA Phase-B variant below.
typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define ATTN_PREFILL_LDS_MAX 2304
#define ATTN_MIXED_MAX_KEYS  17664

// (Deleted 2026-05-27: the monolithic single-token `attention_mixed`
// kernel. Decode now uses the split kernels below — `attention_mixed_
// score` + `attention_mixed_softmax_wsum`. See git history if you need
// the original reference implementation.)

// M50 Phase 4: per-token causal mixed attention.
// Grid (n_head, B, 1). Per-token q[B,...], out[B,...], n_raw_per[B], n_comp_per[B].
// raw_kv shared (SWA), comp_kv shared (compressed KV). Mask not supported
// in batched mode (v4-flash prefill doesn't use it — n_comp_per gives the
// per-token causal prefix).
extern "C" __global__ void attention_mixed_batched(
        float          * out,
        const float    * q,
        const float    * raw_kv,
        const float    * comp_kv,
        const float    * sinks,
        const int      * n_raw_per,        // [B]
        const int      * n_comp_per,       // [B]
        unsigned int     n_head,
        unsigned int     head_dim,
        float            kq_scale) {
    const unsigned int h   = blockIdx.x;
    const unsigned int b   = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (h >= n_head) return;

    __shared__ float partial[256];
    __shared__ float scores[ATTN_PREFILL_LDS_MAX];
    __shared__ float weights[ATTN_PREFILL_LDS_MAX];
    __shared__ float inv_denom_s;

    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];

    const float *qh = q + ((size_t)b * (size_t)n_head + (size_t)h) * (size_t)head_dim;
    float       *oh = out + ((size_t)b * (size_t)n_head + (size_t)h) * (size_t)head_dim;
    const float sink_h = sinks[h];

    for (unsigned int r = 0; r < n_raw; r++) {
        const float *kv_row = raw_kv + (size_t)r * (size_t)head_dim;
        float p = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block) {
            p += qh[i] * kv_row[i];
        }
        partial[tid] = p;
        __syncthreads();
        for (unsigned int s = block / 2; s > 0; s >>= 1) {
            if (tid < s) partial[tid] += partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[r] = partial[0] * kq_scale;
        __syncthreads();
    }

    for (unsigned int r = 0; r < n_comp; r++) {
        const unsigned int idx = n_raw + r;
        const float *kv_row = comp_kv + (size_t)r * (size_t)head_dim;
        float p = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += block) {
            p += qh[i] * kv_row[i];
        }
        partial[tid] = p;
        __syncthreads();
        for (unsigned int s = block / 2; s > 0; s >>= 1) {
            if (tid < s) partial[tid] += partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[idx] = partial[0] * kq_scale;
        __syncthreads();
    }

    if (tid == 0) {
        const unsigned int n_total = n_raw + n_comp;
        float max_score = sink_h;
        for (unsigned int r = 0; r < n_total; r++) {
            if (scores[r] > max_score) max_score = scores[r];
        }
        float denom = expf(sink_h - max_score);
        for (unsigned int r = 0; r < n_total; r++) {
            float w = expf(scores[r] - max_score);
            weights[r] = w;
            denom += w;
        }
        inv_denom_s = 1.0f / denom;
    }
    __syncthreads();

    const float inv = inv_denom_s;
    for (unsigned int d = tid; d < head_dim; d += block) {
        float sum = 0.0f;
        for (unsigned int r = 0; r < n_raw; r++) {
            sum += weights[r] * raw_kv[(size_t)r * (size_t)head_dim + d];
        }
        for (unsigned int r = 0; r < n_comp; r++) {
            sum += weights[n_raw + r] * comp_kv[(size_t)r * (size_t)head_dim + d];
        }
        oh[d] = sum * inv;
    }
}

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
        const float    * raw_kv,         // [n_raw, head_dim]
        const float    * comp_kv,        // [n_comp, head_dim] (may be null when n_comp==0)
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

        const float *kv_row = (r < n_raw)
            ? raw_kv + (size_t)r * head_dim
            : comp_kv + (size_t)(r - n_raw) * head_dim;

        // Unrolled: with head_dim=512 and 32 lanes, each lane does 16
        // muls. Hardcoding lets the compiler pipeline all 16 loads
        // before waiting (was serialized 16× via per-load s_wait_loadcnt
        // = ~245 cycles each = ~3900 cycles wasted per wave per row).
        float p = 0.0f;
        #pragma unroll
        for (unsigned int u = 0; u < 16; u++) {
            const unsigned int i = tid + u * 32;
            p += qh[i] * kv_row[i];
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
        const float    * raw_kv,
        const float    * comp_kv,
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
                s[k] += sw[r + k] * raw_kv[(size_t)(r + k) * head_dim + d];
            }
        }
        for (; r < n_raw; r++) {
            s[0] += sw[r] * raw_kv[(size_t)r * head_dim + d];
        }

        float *cw = sw + n_raw;
        unsigned int c = 0;
        for (; c + U <= n_comp; c += U) {
            #pragma unroll
            for (int k = 0; k < U; k++) {
                s[k] += cw[c + k] * comp_kv[(size_t)(c + k) * head_dim + d];
            }
        }
        for (; c < n_comp; c++) {
            s[0] += cw[c] * comp_kv[(size_t)c * head_dim + d];
        }

        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < U; k++) acc += s[k];
        oh[d] = acc * inv;
    }
}

// ---------------------------------------------------------------------------
// BATCHED split kernels for PREFILL. Same math + optimizations as the decode
// split above, but each token b in the chunk carries its own causal prefix
// (n_raw_per[b], n_comp_per[b]) over the SHARED raw_kv / comp_kv buffers.
// q and out are [B, n_head, head_dim]; scores live in global
// [B, n_head, max_keys].
//
// Replaces `attention_mixed_batched` (monolithic) for long context: that
// kernel kept scores/weights in LDS arrays capped at ATTN_PREFILL_LDS_MAX
// (2304), which OVERFLOWS past ~9K tokens (n_comp = pos/ratio), and did a
// single-thread softmax + sequential per-row block reduction — the exact
// pathologies the decode split fixed.
// ---------------------------------------------------------------------------

// Phase 1, head-tiled. Each WG fixes (row r, token b) and a group of
// SCORE_HEAD_TILE heads. The KV row is loaded into registers ONCE and reused
// across every head in the group — KV is shared across heads in this MLA
// layout, so the un-tiled per-head kernel
// re-fetched the same 2KB row n_head times. Cuts KV global traffic by
// SCORE_HEAD_TILE×, lifting the score phase's ~0.5 MAC/float arithmetic
// intensity toward the q-bound. Grid (n_total_max, ceil(n_head/TILE), B).
#define SCORE_HEAD_TILE 8
extern "C" __global__ void attention_mixed_score_batched_htiled(
        float          * scores_g,           // [B, n_head, max_keys]
        const float    * q,                  // [B, n_head, head_dim]
        const float    * raw_kv,             // [..., head_dim]   (shared, oversized)
        const float    * comp_kv,            // [n_comp, head_dim]  (shared, may be null)
        const int      * n_raw_per,          // [B] causal count into raw cache
        const int      * n_raw_offset_per,   // [B] starting raw-cache slot for this token
        const int      * n_comp_per,         // [B]
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys,
        float            kq_scale) {
    const unsigned int r   = blockIdx.x;
    const unsigned int hg  = blockIdx.y;          // head group
    const unsigned int b   = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int h0  = hg * SCORE_HEAD_TILE;
    if (h0 >= n_head) return;

    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    if (r >= n_raw + n_comp) return;
    const unsigned int raw_off = (unsigned int)n_raw_offset_per[b];

    const float *kv_row = (r < n_raw)
        ? raw_kv + (size_t)(r + raw_off) * head_dim
        : comp_kv + (size_t)(r - n_raw) * head_dim;

    // head_dim=512 / 32 lanes = 16 elems/lane. Load the KV row ONCE.
    float kvr[16];
    #pragma unroll
    for (unsigned int u = 0; u < 16; u++) kvr[u] = kv_row[tid + u * 32];

    const float *qbase = q + ((size_t)b * n_head) * head_dim;
    #pragma unroll
    for (unsigned int hh = 0; hh < SCORE_HEAD_TILE; hh++) {
        const unsigned int h = h0 + hh;
        if (h >= n_head) break;
        const float *qh = qbase + (size_t)h * head_dim;
        float p = 0.0f;
        #pragma unroll
        for (unsigned int u = 0; u < 16; u++) {
            p += qh[tid + u * 32] * kvr[u];
        }
        p += __shfl_down(p, 16);
        p += __shfl_down(p,  8);
        p += __shfl_down(p,  4);
        p += __shfl_down(p,  2);
        p += __shfl_down(p,  1);
        if (tid == 0) scores_g[((size_t)b * n_head + h) * max_keys + r] = p * kq_scale;
    }
}

// Head-tiled merged softmax + weighted sum. Each WG covers SMWSUM_HEAD_TILE
// heads. Two wins over a hypothetical per-head kernel:
//   Phase A: wave w computes head (h0+w)'s softmax, so up to 16 heads run in
//            parallel instead of 1 (the per-head kernel left 15/16 waves idle).
//   Phase B: each thread loads each shared V-latent element kv[r][d] ONCE and
//            applies it across the whole head group (V latent is shared across
//            heads in MLA), cutting Phase-B KV traffic by SMWSUM_HEAD_TILE×.
// Grid (ceil(n_head/SMWSUM_HEAD_TILE), B), block 512.
#define SMWSUM_HEAD_TILE 16
#define SMWSUM_ROW_ILP   2
extern "C" __global__ void attention_mixed_softmax_wsum_batched_htiled(
        float          * out,                // [B, n_head, head_dim]
        float          * scores_g,           // [B, n_head, max_keys] in/out
        const float    * sinks,              // [n_head]
        const float    * raw_kv,             // oversized: cache holds [n_raw_before+b] rows during prefill
        const float    * comp_kv,
        const int      * n_raw_per,          // [B] causal count into raw cache
        const int      * n_raw_offset_per,   // [B] starting slot for this token's raw window
        const int      * n_comp_per,         // [B]
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     max_keys) {
    const unsigned int hg    = blockIdx.x;        // head group
    const unsigned int b     = blockIdx.y;
    const unsigned int tid   = threadIdx.x;
    const unsigned int block = blockDim.x;
    const unsigned int h0    = hg * SMWSUM_HEAD_TILE;
    if (h0 >= n_head) return;

    unsigned int ht = SMWSUM_HEAD_TILE;
    if (h0 + ht > n_head) ht = n_head - h0;

    const unsigned int n_raw  = (unsigned int)n_raw_per[b];
    const unsigned int n_comp = (unsigned int)n_comp_per[b];
    const unsigned int n_total = n_raw + n_comp;

    __shared__ float s_inv[SMWSUM_HEAD_TILE];

    // Phase A: softmax — wave w (lanes [w*32, w*32+32)) handles head h0+w.
    const unsigned int wave = tid >> 5;     // 0..15
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

    // Single weight base for the group + per-head inverse sums. Head hh's
    // weights live at swbase[hh*max_keys + ridx] (stride avoids an HT-wide
    // pointer array → no register blowup at large HEAD_TILE). Out-of-range
    // heads read head h0's lane (harmless); their outputs are dropped.
    const float *swbase = scores_g + ((size_t)b * n_head + h0) * max_keys;
    const size_t hstride = max_keys;
    float inv[SMWSUM_HEAD_TILE];
    #pragma unroll
    for (unsigned int hh = 0; hh < SMWSUM_HEAD_TILE; hh++)
        inv[hh] = (hh < ht) ? s_inv[hh] : 0.0f;

    // Phase B: each thread owns dim d; load each kv[r][d] once, reuse across
    // the head group. SMWSUM_ROW_ILP independent kv loads hide latency.
    const unsigned int raw_off = (unsigned int)n_raw_offset_per[b];
    for (unsigned int d = tid; d < head_dim; d += block) {
        constexpr unsigned int U = SMWSUM_ROW_ILP;
        float acc[SMWSUM_HEAD_TILE];
        #pragma unroll
        for (unsigned int hh = 0; hh < SMWSUM_HEAD_TILE; hh++) acc[hh] = 0.0f;

        unsigned int r = 0;
        for (; r + U <= n_raw; r += U) {
            float kvv[U];
            #pragma unroll
            for (unsigned int k = 0; k < U; k++)
                kvv[k] = raw_kv[(size_t)(r + k + raw_off) * head_dim + d];
            #pragma unroll
            for (unsigned int hh = 0; hh < SMWSUM_HEAD_TILE; hh++) {
                const float *swh = swbase + (size_t)hh * hstride;
                #pragma unroll
                for (unsigned int k = 0; k < U; k++)
                    acc[hh] += swh[r + k] * kvv[k];
            }
        }
        for (; r < n_raw; r++) {
            float kv = raw_kv[(size_t)(r + raw_off) * head_dim + d];
            #pragma unroll
            for (unsigned int hh = 0; hh < SMWSUM_HEAD_TILE; hh++)
                acc[hh] += swbase[(size_t)hh * hstride + r] * kv;
        }

        unsigned int c = 0;
        for (; c + U <= n_comp; c += U) {
            float kvv[U];
            #pragma unroll
            for (unsigned int k = 0; k < U; k++)
                kvv[k] = comp_kv[(size_t)(c + k) * head_dim + d];
            #pragma unroll
            for (unsigned int hh = 0; hh < SMWSUM_HEAD_TILE; hh++) {
                const float *swh = swbase + (size_t)hh * hstride + n_raw;
                #pragma unroll
                for (unsigned int k = 0; k < U; k++)
                    acc[hh] += swh[c + k] * kvv[k];
            }
        }
        for (; c < n_comp; c++) {
            float kv = comp_kv[(size_t)c * head_dim + d];
            #pragma unroll
            for (unsigned int hh = 0; hh < SMWSUM_HEAD_TILE; hh++)
                acc[hh] += swbase[(size_t)hh * hstride + n_raw + c] * kv;
        }

        #pragma unroll
        for (unsigned int hh = 0; hh < SMWSUM_HEAD_TILE; hh++) {
            if (hh < ht) {
                float *oh = out + ((size_t)b * n_head + (h0 + hh)) * head_dim;
                oh[d] = acc[hh] * inv[hh];
            }
        }
    }
}

// WMMA Phase-B variant. Identical output + identical Phase-A softmax to
// attention_mixed_softmax_wsum_batched_htiled, but Phase B (the O(n_keys)
// weighted sum O[16 heads, 512] = W[16, n_keys] @ V[n_keys, 512]) runs as a
// RDNA4 16x16x16 f16 WMMA GEMM. W (post-softmax weights, f32 in scores_g) and
// V (raw_kv/comp_kv, f32) are converted to f16 at fragment-load — f32
// accumulate, no f16 KV cache, zero extra memory. The MLA shared-V reuse is
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
        const float    * raw_kv,
        const float    * comp_kv,
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
            ba[e] = (_Float16)raw_kv[(size_t)(k + raw_off) * head_dim + d_a];
            bb[e] = (_Float16)raw_kv[(size_t)(k + raw_off) * head_dim + d_b];
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
                ba[e] = (_Float16)raw_kv[(size_t)(k + raw_off) * head_dim + d_a];
                bb[e] = (_Float16)raw_kv[(size_t)(k + raw_off) * head_dim + d_b];
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
                ba[e] = (_Float16)comp_kv[(size_t)cc * head_dim + d_a];
                bb[e] = (_Float16)comp_kv[(size_t)cc * head_dim + d_b];
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
                    ba[e] = (_Float16)comp_kv[(size_t)cc * head_dim + d_a];
                    bb[e] = (_Float16)comp_kv[(size_t)cc * head_dim + d_b];
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
        const float    * raw_kv,         // [..., head_dim]   (oversized; per-token raw_off)
        const float    * comp_kv,        // [n_comp, head_dim]  (may be null)
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
    const float *brow = nullptr;
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
            for (int e = 0; e < 8; e++) bf[e] = (_Float16)brow[dim0 + (unsigned int)e];
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
extern "C" __global__ void attention_mixed_score_batched_htiled_wmma_f16s(
        _Float16       * scores_g,       // [B, n_head, max_keys]  (f16)
        const float    * q,
        const float    * raw_kv,
        const float    * comp_kv,
        const int      * n_raw_per,
        const int      * n_raw_offset_per,
        const int      * n_comp_per,
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
    const float *brow = nullptr;
    if (bkey < n_total) {
        brow = (bkey < n_raw)
            ? raw_kv + (size_t)(bkey + raw_off) * head_dim
            : comp_kv + (size_t)(bkey - n_raw) * head_dim;
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
            for (int e = 0; e < 8; e++) bf[e] = (_Float16)brow[dim0 + (unsigned int)e];
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
                scores_g[((size_t)b * n_head + h) * max_keys + bkey] =
                    (_Float16)(c[mt][e] * kq_scale);
            }
        }
    }
#else
    (void)scores_g; (void)q; (void)raw_kv; (void)comp_kv; (void)n_head;
    (void)head_dim; (void)max_keys; (void)kq_scale; (void)warp; (void)lane;
    (void)n_total;
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
        const float    * raw_kv,
        const float    * comp_kv,
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
    // 16 keys × 512 dim = 8192 f32 → 512 threads × 16 elems/thread, vectorized
    // as 4 × float4 b128 loads/thread. f32→f16 cast at the LDS store.
    auto stage_v = [&](const float *kv_base, unsigned int base, unsigned int n_avail) {
        // Each thread covers 4 (k, d_chunk) pairs across the tile.
        // Mapping: thread tid → flat slot s = tid + 512*iter, where
        // s/(head_dim/4) = k_idx, s%(head_dim/4) = d_chunk (b128 offset).
        constexpr unsigned int N_VEC = 16u * (512u / 4u);   // 16 × 128 = 2048
        constexpr unsigned int VEC_PER_KEY = 512u / 4u;     // 128
        #pragma unroll
        for (unsigned int it = 0; it < N_VEC / 512u; it++) {
            const unsigned int s = tid + it * 512u;
            const unsigned int k_idx = s / VEC_PER_KEY;
            const unsigned int d_chunk = s % VEC_PER_KEY;
            float4 v = {0.0f, 0.0f, 0.0f, 0.0f};
            if (k_idx < n_avail) {
                const float4 *src = (const float4 *)(kv_base + (size_t)(base + k_idx) * 512u + d_chunk * 4u);
                v = *src;
            }
            _Float16 *dst = v_lds + k_idx * LDS_STRIDE + d_chunk * 4u;
            dst[0] = (_Float16)v.x;
            dst[1] = (_Float16)v.y;
            dst[2] = (_Float16)v.z;
            dst[3] = (_Float16)v.w;
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

extern "C" __global__ void attention_mixed_softmax_wsum_batched_htiled_wmma_f16s(
        float          * out,
        _Float16       * scores_g,       // [B, n_head, max_keys]  (f16, in/out)
        const float    * sinks,
        const float    * raw_kv,
        const float    * comp_kv,
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
            ba[e] = (_Float16)raw_kv[(size_t)(k + raw_off) * head_dim + d_a];
            bb[e] = (_Float16)raw_kv[(size_t)(k + raw_off) * head_dim + d_b];
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
                ba[e] = (_Float16)raw_kv[(size_t)(k + raw_off) * head_dim + d_a];
                bb[e] = (_Float16)raw_kv[(size_t)(k + raw_off) * head_dim + d_b];
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
                ba[e] = (_Float16)comp_kv[(size_t)cc * head_dim + d_a];
                bb[e] = (_Float16)comp_kv[(size_t)cc * head_dim + d_b];
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
                    ba[e] = (_Float16)comp_kv[(size_t)cc * head_dim + d_a];
                    bb[e] = (_Float16)comp_kv[(size_t)cc * head_dim + d_b];
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

