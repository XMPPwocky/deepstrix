// gqa_attention — correctness-first single-query GQA attention for the Laguna
// model port. Pure score -> stable softmax -> weighted-sum. NO WMMA, NO fancy
// tiling; one workgroup per query head, online (flash-style) softmax so there
// is NO n_kv cap (unlike attention_swa's ATTN_SWA_MAX_KV=128 LDS scores array).
//
// This is standard Grouped-Query Attention (GQA), NOT ds4's MLA:
//   head_dim = 128 (Laguna), n_kv_head = 8, n_head = 48 or 72 (runtime arg).
//   kv_group = n_head / n_kv_head (6 or 9). Query head h reads kv head
//   kv_head = h / kv_group.
//
// The caller has ALREADY applied qk-norm + RoPE to Q/K/V. This kernel does
// NOT do rope, qk-norm, the softplus gate, or o_proj — pure attention math.
//
// Per query head h (blockIdx.x), single query position attending to the whole
// causal KV history [0 .. n_kv) (history is assumed to already include self):
//   score[j] = dot(q[h], k[j, kv_head]) * scale        scale = 1/sqrt(head_dim)
//   softmax over j (numerically stable, running max)
//   out[h, d] = sum_j softmax[j] * v[j, kv_head, d]
//
// Numerical stability: online softmax keeps a running max m and denom l and
// rescales the value accumulator acc[d] by exp(m_old - m_new) each step — the
// standard flash-attention recurrence, done in f32. Equivalent to the
// subtract-the-global-max formulation but single-pass and cap-free.
//
// LDS budget per workgroup (head_dim <= GQA_HEAD_DIM_MAX = 256):
//   qs[256]      = 1024 B   (query, f32)
//   acc[256]     = 1024 B   (weighted-value accumulator, f32)
//   partial[256] = 1024 B   (dot-product tree reduction)
//   score/m/l scalars
//   ~3 KB total

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Max head_dim this kernel's static LDS supports. Laguna is 128; 256 leaves
// margin. The Rust binding validates head_dim <= this.
#define GQA_HEAD_DIM_MAX 256
// Workgroup size. 256 lanes cooperatively reduce the per-key dot product and
// stride over head_dim. Uses only portable HIP builtins (no arch intrinsics),
// so it compiles for gfx1201 AND gfx1151 unchanged.
#define GQA_BLOCK 256

extern "C" __global__ void gqa_attn_single_query(
        float          * out,       // [n_head, head_dim]              (f32)
        const _Float16 * q,         // [n_head, head_dim]              (f16)
        const _Float16 * k_cache,   // [n_kv, n_kv_head, head_dim]     (f16)
        const _Float16 * v_cache,   // [n_kv, n_kv_head, head_dim]     (f16)
        unsigned int     n_head,
        unsigned int     n_kv_head,
        unsigned int     head_dim,
        unsigned int     n_kv,
        float            scale,     // 1/sqrt(head_dim), host-precomputed
        unsigned int     k_base,    // absolute position of relative key 0 (SWA window start)
        unsigned int     kv_capacity) { // ring size: physical row = (k_base+j) % kv_capacity
    const unsigned int h     = blockIdx.x;
    const unsigned int tid   = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (h >= n_head) return;

    __shared__ float qs[GQA_HEAD_DIM_MAX];
    __shared__ float acc[GQA_HEAD_DIM_MAX];
    __shared__ float partial[GQA_BLOCK];
    __shared__ float score_s;   // broadcast of the current key's score
    __shared__ float m_s;       // running max
    __shared__ float l_s;       // running softmax denominator

    // GQA grouping: kv_group = n_head / n_kv_head (validated on the host).
    const unsigned int kv_group = n_head / n_kv_head;
    const unsigned int kv_head  = h / kv_group;

    float *oh = out + (unsigned long long)h * head_dim;

    // Load q[h] into LDS (f16 -> f32) and zero the value accumulator.
    const _Float16 *qh = q + (unsigned long long)h * head_dim;
    for (unsigned int d = tid; d < head_dim; d += block) {
        qs[d]  = (float)qh[d];
        acc[d] = 0.0f;
    }
    if (tid == 0) {
        m_s = -INFINITY;
        l_s = 0.0f;
    }
    __syncthreads();

    // Base offset of kv_head within a key/value row: [j, kv_head, :].
    const unsigned long long kvh_off = (unsigned long long)kv_head * head_dim;
    const unsigned long long row_stride = (unsigned long long)n_kv_head * head_dim;

    for (unsigned int j = 0; j < n_kv; j++) {
        // Ring-buffer physical row for SWA layers; identity for global (k_base=0,
        // kv_capacity=max_kv, (0+j)%max_kv == j).
        const unsigned long long jp = (unsigned long long)((k_base + j) % kv_capacity);
        const _Float16 *kj = k_cache + jp * row_stride + kvh_off;

        // Per-key dot product: each lane sums its strided dims, tree-reduce.
        float p = 0.0f;
        for (unsigned int d = tid; d < head_dim; d += block) {
            p += qs[d] * (float)kj[d];
        }
        partial[tid] = p;
        __syncthreads();
        for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0) score_s = partial[0] * scale;
        __syncthreads();

        // Online softmax update (all lanes read the OLD running max m_s).
        const float score = score_s;
        const float m_old = m_s;
        const float m_new = fmaxf(m_old, score);
        // m_old == -inf on the first key -> corr = exp(-inf) = 0 (not NaN,
        // since m_new is finite). p_j is always finite.
        const float corr  = expf(m_old - m_new);
        const float p_j   = expf(score - m_new);

        const _Float16 *vj = v_cache + jp * row_stride + kvh_off;
        for (unsigned int d = tid; d < head_dim; d += block) {
            acc[d] = acc[d] * corr + p_j * (float)vj[d];
        }
        __syncthreads();  // all lanes done reading m_s before thread 0 rewrites
        if (tid == 0) {
            l_s = l_s * corr + p_j;
            m_s = m_new;
        }
        __syncthreads();
    }

    // Normalize. n_kv == 0 (empty history) -> l_s == 0; emit zeros rather than
    // NaN so the caller gets a defined result.
    const float l = l_s;
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (unsigned int d = tid; d < head_dim; d += block) {
        oh[d] = acc[d] * inv;
    }
}

// gqa_attn_prefill — BATCHED single-query GQA for a whole B-token prompt chunk
// in ONE launch. Semantically identical to calling gqa_attn_single_query B
// times, but with all B*n_head workgroups dispatched at once so the dGPU is
// filled (the sequential single-token prefill was dispatch/latency bound at
// B=1). Each workgroup owns one (query position i, head h) pair.
//
// Causality: query row i is the token at ABSOLUTE position `q_offset + i`, so
// it attends to keys [0 .. q_offset + i] inclusive -> n_kv_i = q_offset+i+1.
// The K/V cache MUST already contain all keys for the whole chunk (the caller
// appends every chunk position before launching); this kernel only reads its
// own causal prefix, so later queries in the chunk don't leak into earlier ones.
//
// Layout (row-major, matches the append order):
//   q:       f16 [B, n_head, head_dim]
//   k_cache: f16 [n_kv_total, n_kv_head, head_dim]   (n_kv_total >= q_offset+B)
//   v_cache: f16 [n_kv_total, n_kv_head, head_dim]
//   out:     f32 [B, n_head, head_dim]
//   grid:  (n_head, B, 1)   block: (GQA_BLOCK, 1, 1)
extern "C" __global__ void gqa_attn_prefill(
        float          * out,       // [B, n_head, head_dim]           (f32)
        const _Float16 * q,         // [B, n_head, head_dim]           (f16)
        const _Float16 * k_cache,   // [n_kv_total, n_kv_head, head_dim](f16)
        const _Float16 * v_cache,   // [n_kv_total, n_kv_head, head_dim](f16)
        unsigned int     n_head,
        unsigned int     n_kv_head,
        unsigned int     head_dim,
        unsigned int     q_offset,  // absolute position of query row 0
        float            scale,
        unsigned int     swa_window, // 0 = full causal; else attend last `swa_window` keys
        unsigned int     kv_capacity) { // ring size: physical row = key_abs % kv_capacity
    const unsigned int h     = blockIdx.x;
    const unsigned int qi    = blockIdx.y;         // query row within the chunk
    const unsigned int tid   = threadIdx.x;
    const unsigned int block = blockDim.x;
    if (h >= n_head) return;

    // Causal key count for this query row (>= 1).
    const unsigned int n_kv = q_offset + qi + 1u;
    // Sliding-window lower bound: attend keys [j_start, n_kv). j_start=0 for full
    // layers (swa_window==0) — a NO-OP, so full-attention output is unchanged.
    const unsigned int j_start = (swa_window != 0u && n_kv > swa_window) ? (n_kv - swa_window) : 0u;

    __shared__ float qs[GQA_HEAD_DIM_MAX];
    __shared__ float acc[GQA_HEAD_DIM_MAX];
    __shared__ float partial[GQA_BLOCK];
    __shared__ float score_s;
    __shared__ float m_s;
    __shared__ float l_s;

    const unsigned int kv_group = n_head / n_kv_head;
    const unsigned int kv_head  = h / kv_group;

    // out[qi, h, :] and q[qi, h, :]
    float         *oh = out + ((unsigned long long)qi * n_head + h) * head_dim;
    const _Float16 *qh = q  + ((unsigned long long)qi * n_head + h) * head_dim;

    for (unsigned int d = tid; d < head_dim; d += block) {
        qs[d]  = (float)qh[d];
        acc[d] = 0.0f;
    }
    if (tid == 0) {
        m_s = -INFINITY;
        l_s = 0.0f;
    }
    __syncthreads();

    const unsigned long long kvh_off    = (unsigned long long)kv_head * head_dim;
    const unsigned long long row_stride = (unsigned long long)n_kv_head * head_dim;

    for (unsigned int j = j_start; j < n_kv; j++) {
        // Ring-buffer physical row (identity for global: kv_capacity=max_kv, j<max_kv).
        const unsigned long long jp = (unsigned long long)(j % kv_capacity);
        const _Float16 *kj = k_cache + jp * row_stride + kvh_off;

        float p = 0.0f;
        for (unsigned int d = tid; d < head_dim; d += block) {
            p += qs[d] * (float)kj[d];
        }
        partial[tid] = p;
        __syncthreads();
        for (unsigned int stride = block / 2; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0) score_s = partial[0] * scale;
        __syncthreads();

        const float score = score_s;
        const float m_old = m_s;
        const float m_new = fmaxf(m_old, score);
        const float corr  = expf(m_old - m_new);
        const float p_j   = expf(score - m_new);

        const _Float16 *vj = v_cache + jp * row_stride + kvh_off;
        for (unsigned int d = tid; d < head_dim; d += block) {
            acc[d] = acc[d] * corr + p_j * (float)vj[d];
        }
        __syncthreads();
        if (tid == 0) {
            l_s = l_s * corr + p_j;
            m_s = m_new;
        }
        __syncthreads();
    }

    const float l = l_s;
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (unsigned int d = tid; d < head_dim; d += block) {
        oh[d] = acc[d] * inv;
    }
}

// =============================================================================
// gqa_attn_prefill_flash — FLASH-TILED (block-tiled, K/V-reuse) GQA prefill.
//
// Replaces gqa_attn_prefill (grid=(n_head,B), one WG per single query, per-key
// __syncthreads, whole-K/V re-read from DRAM once PER QUERY -> ~B^2/2 DRAM
// reads and a barrier storm). This kernel tiles queries into blocks of FBR
// rows and keys into tiles of FBC, stages one K tile + V tile into LDS ONCE
// and reuses it across all FBR queries in the block (the whole point of flash
// attention), and does ONE barrier per key tile instead of one per key. DRAM
// K/V traffic drops from O(B^2) to O(B^2/FBR); the kernel goes from
// DRAM/barrier bound to (tiny) compute bound.
//
// Standard flash / online-softmax recurrence, all accumulation in f32:
//   per query row i keep running max m_i, denom l_i, unnormalised out O_i[d].
//   for each key tile:
//     S_ij = scale*dot(Q_i, K_j)             (causal-masked to -inf)
//     m'   = max(m_i, max_j S_ij)
//     P_ij = exp(S_ij - m')                  (masked -> 0)
//     corr = exp(m_i - m')
//     l_i  = l_i*corr + sum_j P_ij
//     O_i  = O_i*corr + sum_j P_ij*V_j
//   out_i = O_i / l_i
//
// GQA grouping: query head h reads kv_head = h/(n_head/n_kv_head); a whole
// FBR-row query block shares ONE kv_head's K/V tile (grid.x = head), so K/V is
// never reloaded per query head redundantly within a block.
//
// Causal masking is per-(i,j): query row i has absolute pos q_offset+q0+i and
// attends keys with global index <= that pos. We iterate key tiles
// [0 .. max_q_abs] so fully-masked tiles beyond the block are never launched.
//
// SCALAR/LDS-tiled flash (f16 K/V/Q staged in LDS, f32 math). Portable (no
// WMMA intrinsics) so it compiles for gfx1201 AND gfx1151, though it only ever
// launches on the dGPU (gfx1201). A WMMA upgrade (score + P.V as 16x16x16 f16
// tiles) is the noted next lever; the scalar version already removes the two
// dominant costs (redundant DRAM + per-key barrier).
//
// Tile shape (Laguna: head_dim=128, n_kv_head=8):
//   FBR=32 query rows, FBC=32 key cols, FD=128 max head_dim, FBLOCK=256.
//   LDS: Qs 8K + Ks 8K + Vs 8K (f16) + Sp 4K + Os 16K (f32) + per-row ~ 44.6 KB.
//
// Layout (row-major, matches gqa_attn_prefill and the KV-append order):
//   q:       f16 [B, n_head, head_dim]
//   k_cache: f16 [n_kv_total, n_kv_head, head_dim]
//   v_cache: f16 [n_kv_total, n_kv_head, head_dim]
//   out:     f32 [B, n_head, head_dim]
//   grid: (n_head, ceil(B/FBR), 1)   block: (FBLOCK, 1, 1)
// =============================================================================
#define FBR    32u
#define FBC    32u
#define FD     128u   // max head_dim this flash kernel's static LDS supports
#define FBLOCK 256u

extern "C" __global__ __launch_bounds__(FBLOCK)
void gqa_attn_prefill_flash(
        float          * __restrict__ out,         // [B, n_head, head_dim]
        const _Float16 * __restrict__ q,           // [B, n_head, head_dim]
        const _Float16 * __restrict__ k_cache,     // [n_kv_total, n_kv_head, head_dim]
        const _Float16 * __restrict__ v_cache,     // [n_kv_total, n_kv_head, head_dim]
        unsigned int     n_head,
        unsigned int     n_kv_head,
        unsigned int     head_dim,
        unsigned int     q_offset,                 // abs pos of query row 0
        unsigned int     batch,                    // B (number of query rows)
        unsigned int     n_kv_total,               // rows present in the cache
        float            scale,
        unsigned int     swa_window,               // 0 = full causal; else sliding window
        unsigned int     kv_capacity) {            // ring size: physical row = key_g % kv_capacity
    const unsigned int h   = blockIdx.x;
    const unsigned int qt  = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (h >= n_head) return;

    const unsigned int q_row_base = qt * FBR;      // first query row (within chunk)
    if (q_row_base >= batch) return;
    const unsigned int n_rows = (batch - q_row_base < FBR) ? (batch - q_row_base) : FBR;
    const unsigned int hd = head_dim;              // == FD for Laguna

    const unsigned int kv_group = n_head / n_kv_head;
    const unsigned int kv_head  = h / kv_group;
    const unsigned long long kvh_off    = (unsigned long long)kv_head * hd;
    const unsigned long long row_stride = (unsigned long long)n_kv_head * hd;

    __shared__ _Float16 Qs[FBR * FD];
    __shared__ _Float16 Ks[FBC * FD];
    __shared__ _Float16 Vs[FBC * FD];
    __shared__ float     Sp[FBR * FBC];
    __shared__ float     Os[FBR * FD];
    __shared__ float     m_s[FBR];
    __shared__ float     l_s[FBR];
    __shared__ float     corr_s[FBR];

    // --- Stage Q + zero the output accumulator ---
    for (unsigned int p = tid; p < FBR * FD; p += FBLOCK) {
        const unsigned int i = p / FD;
        const unsigned int d = p % FD;
        Os[p] = 0.0f;
        if (i < n_rows && d < hd) {
            Qs[p] = q[((unsigned long long)(q_row_base + i) * n_head + h) * hd + d];
        } else {
            Qs[p] = (_Float16)0.0f;
        }
    }
    if (tid < FBR) {
        m_s[tid]    = -INFINITY;
        l_s[tid]    = 0.0f;
        corr_s[tid] = 0.0f;
    }
    __syncthreads();

    // Absolute position of the LAST valid query row -> last key it attends.
    const unsigned int max_q_abs   = q_offset + q_row_base + n_rows - 1u;
    const unsigned int n_key_tiles = max_q_abs / FBC + 1u;
    // Sliding window: the block's SMALLEST query abs pos is row 0 (q_offset+q_row_base);
    // keys below its window lower bound are masked for EVERY row, so skip those tiles.
    const unsigned int min_q_abs = q_offset + q_row_base;
    const unsigned int win_lo    = (swa_window != 0u && min_q_abs + 1u > swa_window)
                                     ? (min_q_abs + 1u - swa_window) : 0u;
    const unsigned int kt_start  = win_lo / FBC;

    for (unsigned int kt = kt_start; kt < n_key_tiles; kt++) {
        const unsigned int key_base = kt * FBC;

        // --- Stage K/V tile into LDS once (reused across all FBR queries) ---
        for (unsigned int p = tid; p < FBC * FD; p += FBLOCK) {
            const unsigned int j = p / FD;
            const unsigned int d = p % FD;
            const unsigned int key_g = key_base + j;
            if (key_g < n_kv_total && d < hd) {
                // Ring-buffer physical row (identity for global: kv_capacity=max_kv,
                // key_g<n_kv_total<=max_kv). Only the row index wraps; kvh_off+d unchanged.
                const unsigned long long off = (unsigned long long)(key_g % kv_capacity) * row_stride + kvh_off + d;
                Ks[p] = k_cache[off];
                Vs[p] = v_cache[off];
            } else {
                Ks[p] = (_Float16)0.0f;
                Vs[p] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        // --- Score tile: Sp[i][j] = scale*dot(Q_i,K_j), causal-masked ---
        for (unsigned int pp = tid; pp < FBR * FBC; pp += FBLOCK) {
            const unsigned int i = pp / FBC;
            const unsigned int j = pp % FBC;
            float s = -INFINITY;
            if (i < n_rows) {
                const unsigned int q_abs   = q_offset + q_row_base + i;
                const unsigned int key_abs = key_base + j;
                const bool in_win = (swa_window == 0u) || (key_abs + swa_window > q_abs);
                if (key_abs <= q_abs && key_abs < n_kv_total && in_win) {
                    const _Float16 *qr = Qs + (unsigned long long)i * FD;
                    const _Float16 *kr = Ks + (unsigned long long)j * FD;
                    // 8-way ILP: break the D-deep dependent fma chain so the
                    // VALU isn't stalled on serial accumulate latency (this is
                    // the dominant loop; the chain latency, not LDS BW, bounds
                    // the low-occupancy WG).
                    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
                    unsigned int d = 0;
                    for (; d + 8u <= hd; d += 8u) {
                        a0 += (float)qr[d]     * (float)kr[d];
                        a1 += (float)qr[d + 1] * (float)kr[d + 1];
                        a2 += (float)qr[d + 2] * (float)kr[d + 2];
                        a3 += (float)qr[d + 3] * (float)kr[d + 3];
                        a4 += (float)qr[d + 4] * (float)kr[d + 4];
                        a5 += (float)qr[d + 5] * (float)kr[d + 5];
                        a6 += (float)qr[d + 6] * (float)kr[d + 6];
                        a7 += (float)qr[d + 7] * (float)kr[d + 7];
                    }
                    float acc = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
                    for (; d < hd; d++) acc += (float)qr[d] * (float)kr[d];
                    s = acc * scale;
                }
            }
            Sp[i * FBC + j] = s;
        }
        __syncthreads();

        // --- Per-row online-softmax stats (one thread per query row) ---
        if (tid < FBR && tid < n_rows) {
            const unsigned int i = tid;
            float tmax = -INFINITY;
            #pragma unroll
            for (unsigned int j = 0; j < FBC; j++) {
                const float v = Sp[i * FBC + j];
                if (v > tmax) tmax = v;
            }
            const float m_old = m_s[i];
            const float m_new = fmaxf(m_old, tmax);
            float corr;
            float rsum = 0.0f;
            if (m_new == -INFINITY) {
                // No valid keys seen yet (only before this row's diagonal
                // tile); leave running state untouched.
                corr = 1.0f;
                #pragma unroll
                for (unsigned int j = 0; j < FBC; j++) Sp[i * FBC + j] = 0.0f;
            } else {
                corr = expf(m_old - m_new);   // m_old=-inf -> 0 (m_new finite)
                #pragma unroll
                for (unsigned int j = 0; j < FBC; j++) {
                    const float sij = Sp[i * FBC + j];
                    const float p = (sij == -INFINITY) ? 0.0f : expf(sij - m_new);
                    Sp[i * FBC + j] = p;
                    rsum += p;
                }
            }
            l_s[i]    = l_s[i] * corr + rsum;
            m_s[i]    = m_new;
            corr_s[i] = corr;
        }
        __syncthreads();

        // --- Output update: O_i = O_i*corr + sum_j P_ij*V_j ---
        for (unsigned int p = tid; p < FBR * FD; p += FBLOCK) {
            const unsigned int i = p / FD;
            const unsigned int d = p % FD;
            if (i < n_rows && d < hd) {
                const float *pr = Sp + (unsigned long long)i * FBC;
                // 2-way ILP over the BC-deep contraction.
                float o0 = 0.0f, o1 = 0.0f;
                #pragma unroll
                for (unsigned int j = 0; j < FBC; j += 2u) {
                    o0 += pr[j]      * (float)Vs[j * FD + d];
                    o1 += pr[j + 1u] * (float)Vs[(j + 1u) * FD + d];
                }
                Os[p] = Os[p] * corr_s[i] + (o0 + o1);
            }
        }
        __syncthreads();
    }

    // --- Normalise + write out[q_row_base+i, h, :] ---
    for (unsigned int p = tid; p < FBR * FD; p += FBLOCK) {
        const unsigned int i = p / FD;
        const unsigned int d = p % FD;
        if (i < n_rows && d < hd) {
            const float l = l_s[i];
            const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
            out[((unsigned long long)(q_row_base + i) * n_head + h) * hd + d] = Os[p] * inv;
        }
    }
}

// =============================================================================
// gqa_attn_prefill_flash_wmma — WMMA (f16 16x16x16, f32 accumulate) FLASH prefill.
//
// Same math, layout, and output contract as gqa_attn_prefill_flash, but the two
// dominant matmuls run on the gfx1201 (RDNA4) f16 matrix core instead of the
// 8-way scalar-ILP dot. This is the dGPU-only lever: on gfx1201 the f16 WMMA
// path has a real ~194-TFLOP matrix engine (the iGPU gfx1151 does NOT — there
// f16-WMMA ≡ dp4a, a proven dead end — so the iGPU build gets a portable scalar
// fallback via the #else below and this kernel is never launched there).
//
// Tiling: FBR=32 query rows × FBC=32 key cols, head_dim contracted in 16-wide
// WMMA k-steps. Block = 128 threads = 4 wave32 waves.
//   SCORE  S[i][j] = scale·Σ_d Q[i][d]·K[j][d]   — C[m][n]=Σ_k A[m][k]B[k][n]
//          with A=Q (row i, contract d), B=Kᵀ (B[d][j]=K[j][d]). 2×2 wave grid
//          over the 32×32 score tile (wave→(rt,ct)); 8 k-steps of 16 (hd=128).
//   softmax (portable): one thread per query row, online max/denom, writes the
//          probabilities P as f16 into Ps (WMMA A-fragment needs f16).
//   AV     O[i][d] = Σ_j P[i][j]·V[j][d]           — A=P (contract j), B=V.
//          2×2 wave grid (wave→(rt row-tile, d-group of 4 16-col d-tiles));
//          2 k-steps of 16 (FBC=32). The running O accumulator lives in LDS
//          (Os) and is rescaled by corr each key tile (Os = Os·corr + P·V) — a
//          fresh WMMA partial per tile added to the rescaled accumulator, so the
//          proven LDS online-softmax recurrence is reused verbatim.
//
// WMMA fragment layout matches the validated f16_gemm_wmma_lds_tiled:
//   A-fragment: lane holds A[row=lane%16][k=8·(lane/16)+e], e∈0..8
//   B-fragment: lane holds B[k=8·(lane/16)+e][n=lane%16]
//   C-fragment: lane holds C[m=8·(lane/16)+e][n=lane%16]
//
// LDS: Qs/Ks/Vs (3×8K f16) + Ps (2K f16) + Sp (4K f32) + Os (16K f32) ≈ 46 KB.
// Env-gated (LAGUNA_ATTN_WMMA=1); gqa_attn_prefill_flash stays the default until
// this is proven faster AND parity-clean.
//   grid: (n_head, ceil(B/FBR), 1)   block: (128, 1, 1)
// =============================================================================
typedef _Float16 gqa_half8  __attribute__((ext_vector_type(8)));
typedef float     gqa_float8 __attribute__((ext_vector_type(8)));

#define WFBR   32u
#define WFBC   32u
#define WFD    128u   // max head_dim this WMMA flash kernel's static LDS supports
#define WBLOCK 128u

// ---- Head-grouped WMMA prefill (gqa_attn_prefill_flash_wmma_fa2_hg) tiling ----
// One WG owns ONE KV head-subgroup (HG_G query heads that share a KV head) and a
// HG_BR-row query tile, and streams each K/V key-tile into LDS ONCE, reusing it
// across all HG_G heads (kills the kv_group-fold redundant K/V load AND amortizes
// the per-key-tile barriers over HG_G heads). O accumulator lives in LDS (register
// O for HG_G heads would spill). Block is 256 (8 waves) so 1 WG/CU still fields 8
// waves/CU == the baseline's 2×4 — no wave-occupancy loss. Override via
// -DHG_G=/-DHG_BR= for A/B. Global Laguna: kv_group=6 → HG_G=3 → 2 subgroups.
#ifndef HG_G
#define HG_G   3u
#endif
#ifndef HG_BR
#define HG_BR  16u
#endif
#define HG_BC    32u
#define HG_BLOCK 256u

extern "C" __global__ __launch_bounds__(WBLOCK)
void gqa_attn_prefill_flash_wmma(
        float          * __restrict__ out,         // [B, n_head, head_dim]
        const _Float16 * __restrict__ q,           // [B, n_head, head_dim]
        const _Float16 * __restrict__ k_cache,     // [n_kv_total, n_kv_head, head_dim]
        const _Float16 * __restrict__ v_cache,     // [n_kv_total, n_kv_head, head_dim]
        unsigned int     n_head,
        unsigned int     n_kv_head,
        unsigned int     head_dim,
        unsigned int     q_offset,
        unsigned int     batch,
        unsigned int     n_kv_total,
        float            scale,
        unsigned int     swa_window,     // 0 = full causal; else sliding window
        unsigned int     kv_capacity) {  // ring size: physical row = key_g % kv_capacity
    const unsigned int h   = blockIdx.x;
    const unsigned int qt  = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (h >= n_head) return;

    const unsigned int q_row_base = qt * WFBR;
    if (q_row_base >= batch) return;
    const unsigned int n_rows = (batch - q_row_base < WFBR) ? (batch - q_row_base) : WFBR;
    const unsigned int hd = head_dim;              // == WFD for Laguna

    const unsigned int kv_group = n_head / n_kv_head;
    const unsigned int kv_head  = h / kv_group;
    const unsigned long long kvh_off    = (unsigned long long)kv_head * hd;
    const unsigned long long row_stride = (unsigned long long)n_kv_head * hd;

    __shared__ _Float16 Qs[WFBR * WFD];
    __shared__ _Float16 Ks[WFBC * WFD];
    __shared__ _Float16 Vs[WFBC * WFD];
    __shared__ _Float16 Ps[WFBR * WFBC];   // f16 probabilities (WMMA A-fragment)
    __shared__ float     Sp[WFBR * WFBC];  // f32 scores
    __shared__ float     Os[WFBR * WFD];   // f32 running output accumulator
    __shared__ float     m_s[WFBR];
    __shared__ float     l_s[WFBR];
    __shared__ float     corr_s[WFBR];

    // --- Stage Q (zero-padded), zero Os + Ps ---
    for (unsigned int p = tid; p < WFBR * WFD; p += WBLOCK) {
        const unsigned int i = p / WFD;
        const unsigned int d = p % WFD;
        Os[p] = 0.0f;
        if (i < n_rows && d < hd) {
            Qs[p] = q[((unsigned long long)(q_row_base + i) * n_head + h) * hd + d];
        } else {
            Qs[p] = (_Float16)0.0f;
        }
    }
    for (unsigned int p = tid; p < WFBR * WFBC; p += WBLOCK) Ps[p] = (_Float16)0.0f;
    if (tid < WFBR) {
        m_s[tid]    = -INFINITY;
        l_s[tid]    = 0.0f;
        corr_s[tid] = 0.0f;
    }
    __syncthreads();

    const unsigned int max_q_abs   = q_offset + q_row_base + n_rows - 1u;
    const unsigned int n_key_tiles = max_q_abs / WFBC + 1u;
    // Sliding window: skip key tiles entirely below the block's window lower bound.
    const unsigned int min_q_abs = q_offset + q_row_base;
    const unsigned int win_lo    = (swa_window != 0u && min_q_abs + 1u > swa_window)
                                     ? (min_q_abs + 1u - swa_window) : 0u;
    const unsigned int kt_start  = win_lo / WFBC;

    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;            // 0..3

    for (unsigned int kt = kt_start; kt < n_key_tiles; kt++) {
        const unsigned int key_base = kt * WFBC;

        // --- Stage K/V tile (zero-padded) ---
        for (unsigned int p = tid; p < WFBC * WFD; p += WBLOCK) {
            const unsigned int j = p / WFD;
            const unsigned int d = p % WFD;
            const unsigned int key_g = key_base + j;
            if (key_g < n_kv_total && d < hd) {
                // Ring-buffer physical row (identity for global: kv_capacity=max_kv,
                // key_g<n_kv_total<=max_kv). Only the row index wraps; kvh_off+d unchanged.
                const unsigned long long off = (unsigned long long)(key_g % kv_capacity) * row_stride + kvh_off + d;
                Ks[p] = k_cache[off];
                Vs[p] = v_cache[off];
            } else {
                Ks[p] = (_Float16)0.0f;
                Vs[p] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        // --- SCORE: Sp[i][j] = scale·dot(Q_i,K_j), causal-masked ---
#if defined(__gfx1200__) || defined(__gfx1201__)
        {
            const unsigned int rt = wave >> 1;     // row-tile 0..1
            const unsigned int ct = wave & 1u;     // col-tile 0..1
            gqa_float8 sacc = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (unsigned int ks = 0; ks < WFD; ks += 16u) {   // 8 k-steps
                gqa_half8 qf, kf;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned int row = rt * 16u + (lane & 15u);
                    const unsigned int d   = ks + (lane >> 4) * 8u + (unsigned int)e;
                    qf[e] = Qs[row * WFD + d];
                }
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned int krow = ct * 16u + (lane & 15u);
                    const unsigned int d    = ks + (lane >> 4) * 8u + (unsigned int)e;
                    kf[e] = Ks[krow * WFD + d];
                }
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qf, kf, sacc);
            }
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned int i = rt * 16u + (lane >> 4) * 8u + (unsigned int)e;
                const unsigned int j = ct * 16u + (lane & 15u);
                float s = -INFINITY;
                if (i < n_rows) {
                    const unsigned int q_abs   = q_offset + q_row_base + i;
                    const unsigned int key_abs = key_base + j;
                    const bool in_win = (swa_window == 0u) || (key_abs + swa_window > q_abs);
                    if (key_abs <= q_abs && key_abs < n_kv_total && in_win) s = sacc[e] * scale;
                }
                Sp[i * WFBC + j] = s;
            }
        }
#else
        for (unsigned int pp = tid; pp < WFBR * WFBC; pp += WBLOCK) {
            const unsigned int i = pp / WFBC;
            const unsigned int j = pp % WFBC;
            float s = -INFINITY;
            if (i < n_rows) {
                const unsigned int q_abs   = q_offset + q_row_base + i;
                const unsigned int key_abs = key_base + j;
                const bool in_win = (swa_window == 0u) || (key_abs + swa_window > q_abs);
                if (key_abs <= q_abs && key_abs < n_kv_total && in_win) {
                    const _Float16 *qr = Qs + (unsigned long long)i * WFD;
                    const _Float16 *kr = Ks + (unsigned long long)j * WFD;
                    float acc = 0.0f;
                    for (unsigned int d = 0; d < hd; d++) acc += (float)qr[d] * (float)kr[d];
                    s = acc * scale;
                }
            }
            Sp[i * WFBC + j] = s;
        }
#endif
        __syncthreads();

        // --- Online-softmax stats (one thread per query row) -> P into Ps (f16) ---
        if (tid < WFBR && tid < n_rows) {
            const unsigned int i = tid;
            float tmax = -INFINITY;
            #pragma unroll
            for (unsigned int j = 0; j < WFBC; j++) {
                const float v = Sp[i * WFBC + j];
                if (v > tmax) tmax = v;
            }
            const float m_old = m_s[i];
            const float m_new = fmaxf(m_old, tmax);
            float corr;
            float rsum = 0.0f;
            if (m_new == -INFINITY) {
                corr = 1.0f;
                #pragma unroll
                for (unsigned int j = 0; j < WFBC; j++) Ps[i * WFBC + j] = (_Float16)0.0f;
            } else {
                corr = expf(m_old - m_new);
                #pragma unroll
                for (unsigned int j = 0; j < WFBC; j++) {
                    const float sij = Sp[i * WFBC + j];
                    const float p = (sij == -INFINITY) ? 0.0f : expf(sij - m_new);
                    Ps[i * WFBC + j] = (_Float16)p;
                    rsum += p;
                }
            }
            l_s[i]    = l_s[i] * corr + rsum;
            m_s[i]    = m_new;
            corr_s[i] = corr;
        }
        __syncthreads();

        // --- AV: Os = Os·corr + P·V ---
#if defined(__gfx1200__) || defined(__gfx1201__)
        {
            const unsigned int rt   = wave >> 1;   // row-tile 0..1
            const unsigned int dgrp = wave & 1u;   // d-tile group 0..1 (cols 0..63 / 64..127)
            #pragma unroll
            for (unsigned int dti = 0; dti < 4u; ++dti) {
                const unsigned int dt = dgrp * 4u + dti;   // d-tile 0..7
                gqa_float8 oacc = {0,0,0,0,0,0,0,0};
                #pragma unroll
                for (unsigned int ks = 0; ks < WFBC; ks += 16u) {   // 2 k-steps
                    gqa_half8 pf, vf;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const unsigned int prow = rt * 16u + (lane & 15u);
                        const unsigned int jj   = ks + (lane >> 4) * 8u + (unsigned int)e;
                        pf[e] = Ps[prow * WFBC + jj];
                    }
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const unsigned int jj   = ks + (lane >> 4) * 8u + (unsigned int)e;
                        const unsigned int dcol = dt * 16u + (lane & 15u);
                        vf[e] = Vs[jj * WFD + dcol];
                    }
                    oacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pf, vf, oacc);
                }
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned int i = rt * 16u + (lane >> 4) * 8u + (unsigned int)e;
                    const unsigned int d = dt * 16u + (lane & 15u);
                    if (i < n_rows) Os[i * WFD + d] = Os[i * WFD + d] * corr_s[i] + oacc[e];
                }
            }
        }
#else
        for (unsigned int p = tid; p < WFBR * WFD; p += WBLOCK) {
            const unsigned int i = p / WFD;
            const unsigned int d = p % WFD;
            if (i < n_rows && d < hd) {
                const _Float16 *pr = Ps + (unsigned long long)i * WFBC;
                float o = 0.0f;
                #pragma unroll
                for (unsigned int j = 0; j < WFBC; j++) o += (float)pr[j] * (float)Vs[j * WFD + d];
                Os[p] = Os[p] * corr_s[i] + o;
            }
        }
#endif
        __syncthreads();
    }

    // --- Normalise + write out[q_row_base+i, h, :] ---
    for (unsigned int p = tid; p < WFBR * WFD; p += WBLOCK) {
        const unsigned int i = p / WFD;
        const unsigned int d = p % WFD;
        if (i < n_rows && d < hd) {
            const float l = l_s[i];
            const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
            out[((unsigned long long)(q_row_base + i) * n_head + h) * hd + d] = Os[p] * inv;
        }
    }
}

// =============================================================================
// gqa_attn_prefill_flash_wmma_fa2 — FlashAttention-2 REGISTER-RESIDENT-O WMMA
// prefill. Identical signature, output contract, online-softmax math, causal +
// sliding-window mask, and kt_start tile-skip as gqa_attn_prefill_flash_wmma —
// the ONLY difference is where the running O accumulator lives.
//
// The predecessor kept O in LDS (`Os[WFBR*WFD] = 16 KB`), which pushed total LDS
// to ~46 KB → only 1 WG/CU. At 1 WG/CU there is no occupancy to hide the 3
// __syncthreads per key tile, so as the key-tile count grows with context the
// barrier latency compounds → super-linear prefill falloff on the O(L²) global
// layers.
//
// FA2 fix: each wave owns a FIXED (row-tile rt, d-group dgrp) region of the
// 32×128 output for the WHOLE key loop, so the WMMA C-fragment can stay in
// REGISTERS across every key tile. Per wave that is 4 d-tiles × 8 f32/lane = 32
// f32/lane persistent (gqa_float8 oacc[4]). Each key tile:
//   1. rescale the register O by corr_s[row] (per-row, from LDS)
//   2. WMMA-accumulate P·V into it   →  O = O·corr + P·V   (same recurrence)
// O never touches LDS; it is written to global only in the final normalize.
//
// Dropping Os removes 16 KB → total LDS ≈ 30 KB → ≥2 WG/CU, restoring the
// latency hiding that flattens the L² curve.
//
// LDS: Qs/Ks/Vs (3×8K f16) + Ps (2K f16) + Sp (4K f32) + m/l/corr ≈ 30.4 KB.
//   grid: (n_head, ceil(B/WFBR), 1)   block: (128, 1, 1)
// =============================================================================
extern "C" __global__ __launch_bounds__(WBLOCK)
void gqa_attn_prefill_flash_wmma_fa2(
        float          * __restrict__ out,         // [B, n_head, head_dim]
        const _Float16 * __restrict__ q,           // [B, n_head, head_dim]
        const _Float16 * __restrict__ k_cache,     // [n_kv_total, n_kv_head, head_dim]
        const _Float16 * __restrict__ v_cache,     // [n_kv_total, n_kv_head, head_dim]
        unsigned int     n_head,
        unsigned int     n_kv_head,
        unsigned int     head_dim,
        unsigned int     q_offset,
        unsigned int     batch,
        unsigned int     n_kv_total,
        float            scale,
        unsigned int     swa_window,     // 0 = full causal; else sliding window
        unsigned int     kv_capacity,    // ring size: physical row = key_g % kv_capacity
        unsigned int     grid_mode) {    // 0=(head,qtile); 1=KV-first (Infinity-Cache locality)
    // grid_mode 1 remaps the block index so ALL query heads + query-tiles that
    // share one KV head are enumerated by CONSECUTIVE linear block ids (KV head =
    // blockIdx.y, the slow dimension). The HW dispatches blocks roughly in linear
    // order, so the ~128 concurrently-resident WGs span only ~2-3 KV heads instead
    // of all 8 → each KV head's K/V (16.8 MB @32K) stays 64 MB-MALL-resident across
    // its redundant reads. Pure remap: identical math, LDS, VGPR, occupancy.
    const unsigned int kvg = n_head / n_kv_head;
    unsigned int h, qt;
    if (grid_mode == 1u) {
        const unsigned int kvh = blockIdx.y;          // KV head (slow dim)
        h  = kvh * kvg + (blockIdx.x % kvg);          // query head within the group
        qt = blockIdx.x / kvg;                        // query-row tile
    } else {
        h  = blockIdx.x;
        qt = blockIdx.y;
    }
    const unsigned int tid = threadIdx.x;
    if (h >= n_head) return;

    const unsigned int q_row_base = qt * WFBR;
    if (q_row_base >= batch) return;
    const unsigned int n_rows = (batch - q_row_base < WFBR) ? (batch - q_row_base) : WFBR;
    const unsigned int hd = head_dim;              // == WFD for Laguna

    const unsigned int kv_group = n_head / n_kv_head;
    const unsigned int kv_head  = h / kv_group;
    const unsigned long long kvh_off    = (unsigned long long)kv_head * hd;
    const unsigned long long row_stride = (unsigned long long)n_kv_head * hd;

    __shared__ _Float16 Qs[WFBR * WFD];
    __shared__ _Float16 Ks[WFBC * WFD];
    __shared__ _Float16 Vs[WFBC * WFD];
    __shared__ _Float16 Ps[WFBR * WFBC];   // f16 probabilities (WMMA A-fragment)
    __shared__ float     Sp[WFBR * WFBC];  // f32 scores
    __shared__ float     m_s[WFBR];
    __shared__ float     l_s[WFBR];
    __shared__ float     corr_s[WFBR];
#if !(defined(__gfx1200__) || defined(__gfx1201__))
    // Portable (non-gfx120x) fallback keeps the running O in LDS — this kernel is
    // never launched on the iGPU, but the #else path must still compile/run.
    __shared__ float     Os[WFBR * WFD];
#endif

    // --- Stage Q (zero-padded), zero Ps ---
    for (unsigned int p = tid; p < WFBR * WFD; p += WBLOCK) {
        const unsigned int i = p / WFD;
        const unsigned int d = p % WFD;
#if !(defined(__gfx1200__) || defined(__gfx1201__))
        Os[p] = 0.0f;
#endif
        if (i < n_rows && d < hd) {
            Qs[p] = q[((unsigned long long)(q_row_base + i) * n_head + h) * hd + d];
        } else {
            Qs[p] = (_Float16)0.0f;
        }
    }
    for (unsigned int p = tid; p < WFBR * WFBC; p += WBLOCK) Ps[p] = (_Float16)0.0f;
    if (tid < WFBR) {
        m_s[tid]    = -INFINITY;
        l_s[tid]    = 0.0f;
        corr_s[tid] = 0.0f;
    }

    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;            // 0..3

#if defined(__gfx1200__) || defined(__gfx1201__)
    // FA2 register-resident O accumulator. Wave owns (rt = wave>>1, dgrp = wave&1);
    // dgrp selects the 4 d-tiles it accumulates. Persistent across the key loop.
    gqa_float8 oacc[4] = {{0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}};
#endif
    __syncthreads();

    const unsigned int max_q_abs   = q_offset + q_row_base + n_rows - 1u;
    const unsigned int n_key_tiles = max_q_abs / WFBC + 1u;
    const unsigned int min_q_abs = q_offset + q_row_base;
    const unsigned int win_lo    = (swa_window != 0u && min_q_abs + 1u > swa_window)
                                     ? (min_q_abs + 1u - swa_window) : 0u;
    const unsigned int kt_start  = win_lo / WFBC;

    for (unsigned int kt = kt_start; kt < n_key_tiles; kt++) {
        const unsigned int key_base = kt * WFBC;

        // --- Stage K/V tile (zero-padded) ---
        for (unsigned int p = tid; p < WFBC * WFD; p += WBLOCK) {
            const unsigned int j = p / WFD;
            const unsigned int d = p % WFD;
            const unsigned int key_g = key_base + j;
            if (key_g < n_kv_total && d < hd) {
                // Ring-buffer physical row (identity for global: kv_capacity=max_kv,
                // key_g<n_kv_total<=max_kv). Only the row index wraps; kvh_off+d unchanged.
                const unsigned long long off = (unsigned long long)(key_g % kv_capacity) * row_stride + kvh_off + d;
                Ks[p] = k_cache[off];
                Vs[p] = v_cache[off];
            } else {
                Ks[p] = (_Float16)0.0f;
                Vs[p] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        // --- SCORE: Sp[i][j] = scale·dot(Q_i,K_j), causal-masked ---
#if defined(__gfx1200__) || defined(__gfx1201__)
        {
            const unsigned int rt = wave >> 1;     // row-tile 0..1
            const unsigned int ct = wave & 1u;     // col-tile 0..1
            gqa_float8 sacc = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (unsigned int ks = 0; ks < WFD; ks += 16u) {   // 8 k-steps
                gqa_half8 qf, kf;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned int row = rt * 16u + (lane & 15u);
                    const unsigned int d   = ks + (lane >> 4) * 8u + (unsigned int)e;
                    qf[e] = Qs[row * WFD + d];
                }
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned int krow = ct * 16u + (lane & 15u);
                    const unsigned int d    = ks + (lane >> 4) * 8u + (unsigned int)e;
                    kf[e] = Ks[krow * WFD + d];
                }
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qf, kf, sacc);
            }
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned int i = rt * 16u + (lane >> 4) * 8u + (unsigned int)e;
                const unsigned int j = ct * 16u + (lane & 15u);
                float s = -INFINITY;
                if (i < n_rows) {
                    const unsigned int q_abs   = q_offset + q_row_base + i;
                    const unsigned int key_abs = key_base + j;
                    const bool in_win = (swa_window == 0u) || (key_abs + swa_window > q_abs);
                    if (key_abs <= q_abs && key_abs < n_kv_total && in_win) s = sacc[e] * scale;
                }
                Sp[i * WFBC + j] = s;
            }
        }
#else
        for (unsigned int pp = tid; pp < WFBR * WFBC; pp += WBLOCK) {
            const unsigned int i = pp / WFBC;
            const unsigned int j = pp % WFBC;
            float s = -INFINITY;
            if (i < n_rows) {
                const unsigned int q_abs   = q_offset + q_row_base + i;
                const unsigned int key_abs = key_base + j;
                const bool in_win = (swa_window == 0u) || (key_abs + swa_window > q_abs);
                if (key_abs <= q_abs && key_abs < n_kv_total && in_win) {
                    const _Float16 *qr = Qs + (unsigned long long)i * WFD;
                    const _Float16 *kr = Ks + (unsigned long long)j * WFD;
                    float acc = 0.0f;
                    for (unsigned int d = 0; d < hd; d++) acc += (float)qr[d] * (float)kr[d];
                    s = acc * scale;
                }
            }
            Sp[i * WFBC + j] = s;
        }
#endif
        __syncthreads();

        // --- Online-softmax stats (one thread per query row) -> P into Ps (f16) ---
        if (tid < WFBR && tid < n_rows) {
            const unsigned int i = tid;
            float tmax = -INFINITY;
            #pragma unroll
            for (unsigned int j = 0; j < WFBC; j++) {
                const float v = Sp[i * WFBC + j];
                if (v > tmax) tmax = v;
            }
            const float m_old = m_s[i];
            const float m_new = fmaxf(m_old, tmax);
            float corr;
            float rsum = 0.0f;
            if (m_new == -INFINITY) {
                corr = 1.0f;
                #pragma unroll
                for (unsigned int j = 0; j < WFBC; j++) Ps[i * WFBC + j] = (_Float16)0.0f;
            } else {
                corr = expf(m_old - m_new);
                #pragma unroll
                for (unsigned int j = 0; j < WFBC; j++) {
                    const float sij = Sp[i * WFBC + j];
                    const float p = (sij == -INFINITY) ? 0.0f : expf(sij - m_new);
                    Ps[i * WFBC + j] = (_Float16)p;
                    rsum += p;
                }
            }
            l_s[i]    = l_s[i] * corr + rsum;
            m_s[i]    = m_new;
            corr_s[i] = corr;
        }
        __syncthreads();

        // --- AV: O = O·corr + P·V, with O in REGISTERS (the FA2 lever) ---
#if defined(__gfx1200__) || defined(__gfx1201__)
        {
            const unsigned int rt   = wave >> 1;   // row-tile 0..1
            const unsigned int dgrp = wave & 1u;   // d-tile group 0..1 (cols 0..63 / 64..127)
            // Per-lane rescale factors for the 8 rows this lane's C-fragment owns
            // (row = rt*16 + (lane>>4)*8 + e). corr is the same for all 4 d-tiles.
            float cr[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned int i = rt * 16u + (lane >> 4) * 8u + (unsigned int)e;
                cr[e] = corr_s[i];
            }
            #pragma unroll
            for (unsigned int dti = 0; dti < 4u; ++dti) {
                const unsigned int dt = dgrp * 4u + dti;   // d-tile 0..7
                // 1) rescale the persistent register accumulator by corr (FA2)
                #pragma unroll
                for (int e = 0; e < 8; ++e) oacc[dti][e] *= cr[e];
                // 2) WMMA-accumulate P·V into it (2 k-steps of 16, FBC=32)
                #pragma unroll
                for (unsigned int ks = 0; ks < WFBC; ks += 16u) {
                    gqa_half8 pf, vf;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const unsigned int prow = rt * 16u + (lane & 15u);
                        const unsigned int jj   = ks + (lane >> 4) * 8u + (unsigned int)e;
                        pf[e] = Ps[prow * WFBC + jj];
                    }
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const unsigned int jj   = ks + (lane >> 4) * 8u + (unsigned int)e;
                        const unsigned int dcol = dt * 16u + (lane & 15u);
                        vf[e] = Vs[jj * WFD + dcol];
                    }
                    oacc[dti] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pf, vf, oacc[dti]);
                }
            }
        }
#else
        for (unsigned int p = tid; p < WFBR * WFD; p += WBLOCK) {
            const unsigned int i = p / WFD;
            const unsigned int d = p % WFD;
            if (i < n_rows && d < hd) {
                const _Float16 *pr = Ps + (unsigned long long)i * WFBC;
                float o = 0.0f;
                #pragma unroll
                for (unsigned int j = 0; j < WFBC; j++) o += (float)pr[j] * (float)Vs[j * WFD + d];
                Os[p] = Os[p] * corr_s[i] + o;
            }
        }
#endif
        __syncthreads();
    }

    // --- Normalise + write out[q_row_base+i, h, :] ---
#if defined(__gfx1200__) || defined(__gfx1201__)
    {
        const unsigned int rt   = wave >> 1;
        const unsigned int dgrp = wave & 1u;
        #pragma unroll
        for (unsigned int dti = 0; dti < 4u; ++dti) {
            const unsigned int dt = dgrp * 4u + dti;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned int i = rt * 16u + (lane >> 4) * 8u + (unsigned int)e;
                const unsigned int d = dt * 16u + (lane & 15u);
                if (i < n_rows) {
                    const float l = l_s[i];
                    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    out[((unsigned long long)(q_row_base + i) * n_head + h) * hd + d] = oacc[dti][e] * inv;
                }
            }
        }
    }
#else
    for (unsigned int p = tid; p < WFBR * WFD; p += WBLOCK) {
        const unsigned int i = p / WFD;
        const unsigned int d = p % WFD;
        if (i < n_rows && d < hd) {
            const float l = l_s[i];
            const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
            out[((unsigned long long)(q_row_base + i) * n_head + h) * hd + d] = Os[p] * inv;
        }
    }
#endif
}

// =============================================================================
// gqa_attn_prefill_flash_wmma_fa2_hg — HEAD-GROUPED WMMA FA prefill.
//
// The single-head fa2 launches one WG per QUERY head; the kv_group query heads
// that share a KV head each independently stream that KV head's K/V from DRAM and
// each pay their own per-key-tile __syncthreads. This kernel launches one WG per
// (KV head, sub-group of HG_G query heads, query-row tile). Per key tile it stages
// K/V ONCE into LDS and runs score+online-softmax+P·V for ALL HG_G heads against
// it — cutting the per-key-tile barrier-rounds (and redundant K/V loads) by HG_G.
//
// O accumulator is in LDS (HG_G heads × HG_BR × head_dim f32); register-O would
// spill for HG_G>1. Block = HG_BLOCK(256) = 8 waves so 1 WG/CU still fields 8
// waves/CU (== baseline 2 WG × 4 waves) — barrier *count* drops without losing
// wave occupancy. head_dim == WFD(128). Requires kv_group % HG_G == 0.
//   grid: (n_kv_head * (kv_group/HG_G), ceil(B/HG_BR), 1)   block: (256,1,1)
// =============================================================================
extern "C" __global__ __launch_bounds__(HG_BLOCK)
void gqa_attn_prefill_flash_wmma_fa2_hg(
        float          * __restrict__ out,         // [B, n_head, head_dim]
        const _Float16 * __restrict__ q,           // [B, n_head, head_dim]
        const _Float16 * __restrict__ k_cache,     // [n_kv_total, n_kv_head, head_dim]
        const _Float16 * __restrict__ v_cache,     // [n_kv_total, n_kv_head, head_dim]
        unsigned int     n_head,
        unsigned int     n_kv_head,
        unsigned int     head_dim,
        unsigned int     q_offset,
        unsigned int     batch,
        unsigned int     n_kv_total,
        float            scale,
        unsigned int     swa_window,     // 0 = full causal; else sliding window
        unsigned int     kv_capacity) {  // ring size: physical row = key_g % kv_capacity
#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wave = tid >> 5;            // 0..7

    const unsigned int kv_group   = n_head / n_kv_head;
    const unsigned int n_subgroup = kv_group / HG_G;
    const unsigned int kv_head    = blockIdx.x / n_subgroup;
    const unsigned int subgroup   = blockIdx.x % n_subgroup;
    const unsigned int qt         = blockIdx.y;
    if (kv_head >= n_kv_head) return;

    const unsigned int q_row_base = qt * HG_BR;
    if (q_row_base >= batch) return;
    const unsigned int n_rows = (batch - q_row_base < HG_BR) ? (batch - q_row_base) : HG_BR;
    const unsigned int hd = head_dim;              // == WFD
    // First query head of this sub-group; the HG_G heads are contiguous.
    const unsigned int h0 = kv_head * kv_group + subgroup * HG_G;

    const unsigned long long kvh_off    = (unsigned long long)kv_head * hd;
    const unsigned long long row_stride = (unsigned long long)n_kv_head * hd;

    __shared__ _Float16 Qs[HG_G * HG_BR * WFD];
    __shared__ _Float16 Ks[HG_BC * WFD];
    __shared__ _Float16 Vs[HG_BC * WFD];
    __shared__ _Float16 Ps[HG_G * HG_BR * HG_BC];
    __shared__ float    Sp[HG_G * HG_BR * HG_BC];
    __shared__ float    Os[HG_G * HG_BR * WFD];
    __shared__ float    m_s[HG_G * HG_BR];
    __shared__ float    l_s[HG_G * HG_BR];
    __shared__ float    corr_s[HG_G * HG_BR];

    // --- Stage all HG_G heads' Q (zero-padded), zero Os ---
    for (unsigned int p = tid; p < HG_G * HG_BR * WFD; p += HG_BLOCK) {
        const unsigned int g = p / (HG_BR * WFD);
        const unsigned int r = (p / WFD) % HG_BR;
        const unsigned int d = p % WFD;
        Os[p] = 0.0f;
        if (r < n_rows && d < hd) {
            Qs[p] = q[((unsigned long long)(q_row_base + r) * n_head + (h0 + g)) * hd + d];
        } else {
            Qs[p] = (_Float16)0.0f;
        }
    }
    for (unsigned int p = tid; p < HG_G * HG_BR * HG_BC; p += HG_BLOCK) Ps[p] = (_Float16)0.0f;
    if (tid < HG_G * HG_BR) { m_s[tid] = -INFINITY; l_s[tid] = 0.0f; corr_s[tid] = 0.0f; }
    __syncthreads();

    const unsigned int max_q_abs   = q_offset + q_row_base + n_rows - 1u;
    const unsigned int n_key_tiles = max_q_abs / HG_BC + 1u;
    const unsigned int min_q_abs   = q_offset + q_row_base;
    const unsigned int win_lo      = (swa_window != 0u && min_q_abs + 1u > swa_window)
                                       ? (min_q_abs + 1u - swa_window) : 0u;
    const unsigned int kt_start    = win_lo / HG_BC;

    for (unsigned int kt = kt_start; kt < n_key_tiles; kt++) {
        const unsigned int key_base = kt * HG_BC;

        // --- Stage K/V tile ONCE (shared across all HG_G heads) ---
        for (unsigned int p = tid; p < HG_BC * WFD; p += HG_BLOCK) {
            const unsigned int j = p / WFD;
            const unsigned int d = p % WFD;
            const unsigned int key_g = key_base + j;
            if (key_g < n_kv_total && d < hd) {
                const unsigned long long off = (unsigned long long)(key_g % kv_capacity) * row_stride + kvh_off + d;
                Ks[p] = k_cache[off];
                Vs[p] = v_cache[off];
            } else {
                Ks[p] = (_Float16)0.0f;
                Vs[p] = (_Float16)0.0f;
            }
        }
        __syncthreads();

        // --- SCORE for every head: Sp[g][i][j] = scale·dot(Qg_i,K_j). One wave per
        //     (head, col-tile); HG_BR=16 is a single 16-row WMMA tile. ---
        {
            const unsigned int g  = wave / 2u;     // 0..HG_G-1
            const unsigned int ct = wave & 1u;     // col-tile 0..1 (keys 0..15 / 16..31)
            if (g < HG_G) {
                const _Float16 *Qg = Qs + (unsigned long long)g * HG_BR * WFD;
                gqa_float8 sacc = {0,0,0,0,0,0,0,0};
                #pragma unroll
                for (unsigned int ks = 0; ks < WFD; ks += 16u) {   // 8 k-steps
                    gqa_half8 qf, kf;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const unsigned int row = (lane & 15u);
                        const unsigned int d   = ks + (lane >> 4) * 8u + (unsigned int)e;
                        qf[e] = Qg[row * WFD + d];
                    }
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const unsigned int krow = ct * 16u + (lane & 15u);
                        const unsigned int d    = ks + (lane >> 4) * 8u + (unsigned int)e;
                        kf[e] = Ks[krow * WFD + d];
                    }
                    sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qf, kf, sacc);
                }
                float *Spg = Sp + (unsigned long long)g * HG_BR * HG_BC;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned int i = (lane >> 4) * 8u + (unsigned int)e;   // row 0..15
                    const unsigned int j = ct * 16u + (lane & 15u);
                    float s = -INFINITY;
                    if (i < n_rows) {
                        const unsigned int q_abs   = q_offset + q_row_base + i;
                        const unsigned int key_abs = key_base + j;
                        const bool in_win = (swa_window == 0u) || (key_abs + swa_window > q_abs);
                        if (key_abs <= q_abs && key_abs < n_kv_total && in_win) s = sacc[e] * scale;
                    }
                    Spg[i * HG_BC + j] = s;
                }
            }
        }
        __syncthreads();

        // --- Online-softmax stats: one thread per (head, query row) ---
        if (tid < HG_G * HG_BR) {
            const unsigned int g = tid / HG_BR;
            const unsigned int i = tid % HG_BR;
            if (i < n_rows) {
                float *Spg = Sp + (unsigned long long)g * HG_BR * HG_BC;
                _Float16 *Psg = Ps + (unsigned long long)g * HG_BR * HG_BC;
                float tmax = -INFINITY;
                #pragma unroll
                for (unsigned int j = 0; j < HG_BC; j++) { const float v = Spg[i*HG_BC+j]; if (v > tmax) tmax = v; }
                const float m_old = m_s[g*HG_BR+i];
                const float m_new = fmaxf(m_old, tmax);
                float corr, rsum = 0.0f;
                if (m_new == -INFINITY) {
                    corr = 1.0f;
                    #pragma unroll
                    for (unsigned int j = 0; j < HG_BC; j++) Psg[i*HG_BC+j] = (_Float16)0.0f;
                } else {
                    corr = expf(m_old - m_new);
                    #pragma unroll
                    for (unsigned int j = 0; j < HG_BC; j++) {
                        const float sij = Spg[i*HG_BC+j];
                        const float p = (sij == -INFINITY) ? 0.0f : expf(sij - m_new);
                        Psg[i*HG_BC+j] = (_Float16)p;
                        rsum += p;
                    }
                }
                l_s[g*HG_BR+i]    = l_s[g*HG_BR+i] * corr + rsum;
                m_s[g*HG_BR+i]    = m_new;
                corr_s[g*HG_BR+i] = corr;
            }
        }
        __syncthreads();

        // --- AV: Os[g] = Os[g]·corr + P[g]·V. Wave = d-tile (0..7), loops heads.
        //     Os C-fragment loaded from LDS, rescaled, WMMA-accumulated, stored. ---
        {
            const unsigned int dt = wave;          // d-tile 0..7 (cols dt*16..dt*16+15)
            #pragma unroll
            for (unsigned int g = 0; g < HG_G; ++g) {
                float    *Osg = Os + (unsigned long long)g * HG_BR * WFD;
                _Float16 *Psg = Ps + (unsigned long long)g * HG_BR * HG_BC;
                const float *crg = corr_s + (unsigned long long)g * HG_BR;
                // load current O fragment + rescale by corr (per row)
                gqa_float8 oacc;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned int i = (lane >> 4) * 8u + (unsigned int)e;   // row 0..15
                    const unsigned int d = dt * 16u + (lane & 15u);
                    oacc[e] = Osg[i * WFD + d] * crg[i];
                }
                #pragma unroll
                for (unsigned int ks = 0; ks < HG_BC; ks += 16u) {   // 2 k-steps
                    gqa_half8 pf, vf;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const unsigned int prow = (lane & 15u);
                        const unsigned int jj   = ks + (lane >> 4) * 8u + (unsigned int)e;
                        pf[e] = Psg[prow * HG_BC + jj];
                    }
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const unsigned int jj   = ks + (lane >> 4) * 8u + (unsigned int)e;
                        const unsigned int dcol = dt * 16u + (lane & 15u);
                        vf[e] = Vs[jj * WFD + dcol];
                    }
                    oacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pf, vf, oacc);
                }
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned int i = (lane >> 4) * 8u + (unsigned int)e;
                    const unsigned int d = dt * 16u + (lane & 15u);
                    Osg[i * WFD + d] = oacc[e];
                }
            }
        }
        __syncthreads();
    }

    // --- Normalise + write out for all HG_G heads ---
    for (unsigned int p = tid; p < HG_G * HG_BR * WFD; p += HG_BLOCK) {
        const unsigned int g = p / (HG_BR * WFD);
        const unsigned int i = (p / WFD) % HG_BR;
        const unsigned int d = p % WFD;
        if (i < n_rows && d < hd) {
            const float l = l_s[g*HG_BR+i];
            const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
            out[((unsigned long long)(q_row_base + i) * n_head + (h0 + g)) * hd + d] = Os[p] * inv;
        }
    }
#endif
}

// =============================================================================
// SPLIT-KV DECODE ("flash decoding") — the fix for single-query decode at long
// context. The existing gqa_attn_prefill_flash at batch=1 launches only
// (n_head) workgroups, each of which marches SERIALLY through n_kv/32 key
// tiles — at 32K that's 1024 serial iterations per WG with only ~48-72 WGs on a
// 64-CU dGPU (catastrophic occupancy + latency). Measured 82.9% of decode wall
// and ~9x over the KV-read roofline.
//
// Split-KV partitions the causal history across `n_splits` workgroups per head
// (grid.z), each computing a PARTIAL online-softmax (running max m, denom l,
// unnormalised accumulator O[d]) over its own key slice. A tiny combine kernel
// then merges the partials with the standard flash rescale. This raises the WG
// count from n_head to n_head*n_splits (fills the GPU) and cuts each WG's serial
// key-tile chain by n_splits. Numerics are the identical flash recurrence, just
// reassociated across splits — bit-comparable to the reference within f32 order.
//
// Layout:
//   q:          f16 [n_head, head_dim]
//   k/v_cache:  f16 [n_kv, n_kv_head, head_dim]
//   out_partial:f32 [n_head, n_splits, head_dim]   (unnormalised acc)
//   m_partial:  f32 [n_head, n_splits]             (running max)
//   l_partial:  f32 [n_head, n_splits]             (softmax denom)
//   grid: (n_head, n_splits, 1)   block: (DEC_BLOCK == head_dim == 128)
// =============================================================================
#define DEC_FBC   32u    // keys staged per tile
#define DEC_FD    128u   // head_dim (Laguna) — also the block size
#define DEC_BLOCK 128u

extern "C" __global__ __launch_bounds__(DEC_BLOCK)
void gqa_attn_decode_partial(
        float          * __restrict__ out_partial,  // [n_head, n_splits, head_dim]
        float          * __restrict__ m_partial,    // [n_head, n_splits]
        float          * __restrict__ l_partial,    // [n_head, n_splits]
        const _Float16 * __restrict__ q,            // [n_head, head_dim]
        const _Float16 * __restrict__ k_cache,      // [n_kv, n_kv_head, head_dim]
        const _Float16 * __restrict__ v_cache,      // [n_kv, n_kv_head, head_dim]
        unsigned int     n_head,
        unsigned int     n_kv_head,
        unsigned int     head_dim,
        unsigned int     n_kv,
        unsigned int     n_splits,
        float            scale,
        unsigned int     k_base,        // absolute position of relative key 0 (SWA window start)
        unsigned int     kv_capacity) { // ring size: physical row = (k_base+key_g) % kv_capacity
    const unsigned int h   = blockIdx.x;
    const unsigned int s   = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (h >= n_head || s >= n_splits) return;
    const unsigned int hd = head_dim;  // == DEC_FD for Laguna

    const unsigned int kv_group = n_head / n_kv_head;
    const unsigned int kv_head  = h / kv_group;
    const unsigned long long kvh_off    = (unsigned long long)kv_head * hd;
    const unsigned long long row_stride = (unsigned long long)n_kv_head * hd;

    // This split's contiguous key range [k0, k1).
    const unsigned int per = (n_kv + n_splits - 1u) / n_splits;
    const unsigned int k0  = s * per;
    const unsigned int k1  = (k0 + per < n_kv) ? (k0 + per) : n_kv;

    __shared__ float    Qs[DEC_FD];
    __shared__ _Float16 Ks[DEC_FBC * DEC_FD];
    __shared__ _Float16 Vs[DEC_FBC * DEC_FD];
    __shared__ float    Sp[DEC_FBC];
    __shared__ float    m_s;
    __shared__ float    l_s;
    __shared__ float    corr_s;

    // Stage the query (one dim per thread; block == head_dim).
    if (tid < hd) Qs[tid] = (float)q[(unsigned long long)h * hd + tid];
    float accd = 0.0f;                 // thread `tid` owns output dim `tid`
    if (tid == 0) { m_s = -INFINITY; l_s = 0.0f; }
    __syncthreads();

    if (k0 < k1) {
        for (unsigned int kb = k0; kb < k1; kb += DEC_FBC) {
            // --- stage K/V tile into LDS (all threads, coalesced) ---
            for (unsigned int p = tid; p < DEC_FBC * hd; p += DEC_BLOCK) {
                const unsigned int j = p / hd;
                const unsigned int d = p % hd;
                const unsigned int key_g = kb + j;
                if (key_g < k1) {
                    // Ring-buffer physical row: absolute pos = k_base + key_g (key_g is
                    // relative to the window start). Identity for global (k_base=0,
                    // kv_capacity=max_kv). key_g/k0/k1 stay relative for the split math.
                    const unsigned long long off = (unsigned long long)((k_base + key_g) % kv_capacity) * row_stride + kvh_off + d;
                    Ks[p] = k_cache[off];
                    Vs[p] = v_cache[off];
                } else {
                    Ks[p] = (_Float16)0.0f;
                    Vs[p] = (_Float16)0.0f;
                }
            }
            __syncthreads();

            // --- scores: one thread per key (DEC_FBC <= DEC_BLOCK) ---
            const unsigned int n_keys = (kb + DEC_FBC < k1) ? DEC_FBC : (k1 - kb);
            if (tid < DEC_FBC) {
                if (tid < n_keys) {
                    const _Float16 *kr = Ks + (unsigned long long)tid * hd;
                    // 8-way ILP dot to break the dependent-fma latency chain.
                    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
                    unsigned int d = 0;
                    for (; d + 8u <= hd; d += 8u) {
                        a0 += Qs[d]     * (float)kr[d];
                        a1 += Qs[d + 1] * (float)kr[d + 1];
                        a2 += Qs[d + 2] * (float)kr[d + 2];
                        a3 += Qs[d + 3] * (float)kr[d + 3];
                        a4 += Qs[d + 4] * (float)kr[d + 4];
                        a5 += Qs[d + 5] * (float)kr[d + 5];
                        a6 += Qs[d + 6] * (float)kr[d + 6];
                        a7 += Qs[d + 7] * (float)kr[d + 7];
                    }
                    float acc = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
                    for (; d < hd; d++) acc += Qs[d] * (float)kr[d];
                    Sp[tid] = acc * scale;
                } else {
                    Sp[tid] = -INFINITY;
                }
            }
            __syncthreads();

            // --- online-softmax merge of this tile (thread 0, DEC_FBC scalars) ---
            if (tid == 0) {
                float tmax = -INFINITY;
                for (unsigned int j = 0; j < n_keys; j++) if (Sp[j] > tmax) tmax = Sp[j];
                const float m_old = m_s;
                const float m_new = fmaxf(m_old, tmax);
                const float corr  = (m_old == -INFINITY) ? 1.0f : expf(m_old - m_new);
                float rsum = 0.0f;
                for (unsigned int j = 0; j < DEC_FBC; j++) {
                    const float sij = Sp[j];
                    const float p = (sij == -INFINITY) ? 0.0f : expf(sij - m_new);
                    Sp[j] = p;
                    rsum += p;
                }
                l_s    = l_s * corr + rsum;
                m_s    = m_new;
                corr_s = corr;
            }
            __syncthreads();

            // --- output accumulate: acc[d] = acc[d]*corr + sum_j P_j * V_j[d] ---
            if (tid < hd) {
                const float corr = corr_s;
                float o = 0.0f;
                #pragma unroll
                for (unsigned int j = 0; j < DEC_FBC; j++) {
                    o += Sp[j] * (float)Vs[(unsigned long long)j * hd + tid];
                }
                accd = accd * corr + o;
            }
            __syncthreads();
        }
    }

    // --- write this split's UNNORMALISED partial + (m, l) ---
    const unsigned long long po = ((unsigned long long)h * n_splits + s) * hd;
    if (tid < hd) out_partial[po + tid] = accd;
    if (tid == 0) {
        m_partial[(unsigned long long)h * n_splits + s] = m_s;
        l_partial[(unsigned long long)h * n_splits + s] = l_s;
    }
}

// =============================================================================
// HEAD-GROUPED SPLIT-KV DECODE — the GQA-aware fix for gqa_attn_decode_partial.
//
// The per-head split kernel above launches one WG per (query_head, split). The
// 6 or 9 query heads that share a single KV head each re-read that KV head's
// K/V from DRAM independently (kv_group-fold redundant DRAM traffic), and its
// score phase uses only DEC_FBC(=32) of the 128 threads. Since long-context
// decode attention is bound by K/V DRAM bandwidth, that redundancy is the pole.
//
// This kernel launches one WG per (KV_head, split): grid (n_kv_head, n_splits).
// Each WG stages the K/V tile into LDS ONCE and computes ALL kv_group query
// heads that map to this KV head from it — killing the redundant DRAM reads
// (K/V read exactly once per KV head per split). The score phase uses all 128
// threads (4 lanes per key × 32 keys, a per-4-lane shuffle reduction), and the
// per-head running softmax state + unnormalised accumulator live in LDS so they
// persist across key tiles while the head loop stays INNERMOST (the K/V LDS
// tile is reused across the whole group).
//
// Output layout is IDENTICAL to gqa_attn_decode_partial (indexed by the derived
// query head `kh*kv_group + g`), so gqa_attn_decode_combine merges these
// partials unchanged. Numerically the identical online-softmax flash recurrence;
// only the per-key dot is reassociated (4-lane f32 sum vs the 8-way single-lane
// ILP), a ULP-level difference.
//
//   grid: (n_kv_head, n_splits, 1)   block: (DEC_BLOCK == head_dim == 128)
//   LDS (kv_group<=DEC_KVG_MAX): Qs+Os (2*12*128 f32=12KB) + Ks+Vs (2*32*128
//   f16=16KB) + Sp/m/l/corr ~ 28.4 KB → 2 WG/CU on a 64 KB CU.
// =============================================================================
#define DEC_KVG_MAX 12u   // max kv_group (Laguna SWA = 72/8 = 9)

extern "C" __global__ __launch_bounds__(DEC_BLOCK)
void gqa_attn_decode_partial_hg(
        float          * __restrict__ out_partial,  // [n_head, n_splits, head_dim]
        float          * __restrict__ m_partial,    // [n_head, n_splits]
        float          * __restrict__ l_partial,    // [n_head, n_splits]
        const _Float16 * __restrict__ q,            // [n_head, head_dim]
        const _Float16 * __restrict__ k_cache,      // [n_kv, n_kv_head, head_dim]
        const _Float16 * __restrict__ v_cache,      // [n_kv, n_kv_head, head_dim]
        unsigned int     n_head,
        unsigned int     n_kv_head,
        unsigned int     head_dim,
        unsigned int     n_kv,
        unsigned int     n_splits,
        float            scale,
        unsigned int     k_base,        // absolute position of relative key 0 (SWA window start)
        unsigned int     kv_capacity) { // ring size: physical row = (k_base+key_g) % kv_capacity
    const unsigned int kh  = blockIdx.x;   // KV head
    const unsigned int s   = blockIdx.y;   // split
    const unsigned int tid = threadIdx.x;
    if (kh >= n_kv_head || s >= n_splits) return;
    const unsigned int hd = head_dim;      // == DEC_FD for Laguna
    const unsigned int kv_group = n_head / n_kv_head;  // 6 or 9

    const unsigned long long kvh_off    = (unsigned long long)kh * hd;
    const unsigned long long row_stride = (unsigned long long)n_kv_head * hd;

    // This split's contiguous key range [k0, k1).
    const unsigned int per = (n_kv + n_splits - 1u) / n_splits;
    const unsigned int k0  = s * per;
    const unsigned int k1  = (k0 + per < n_kv) ? (k0 + per) : n_kv;

    __shared__ float    Qs[DEC_KVG_MAX * DEC_FD];
    __shared__ float    Os[DEC_KVG_MAX * DEC_FD];
    __shared__ _Float16 Ks[DEC_FBC * DEC_FD];
    __shared__ _Float16 Vs[DEC_FBC * DEC_FD];
    __shared__ float    Sp[DEC_KVG_MAX * DEC_FBC];   // per-head score tile
    __shared__ float    m_s[DEC_KVG_MAX];
    __shared__ float    l_s[DEC_KVG_MAX];
    __shared__ float    corr_s[DEC_KVG_MAX];

    // Stage all group heads' queries + zero the per-head output accumulators.
    if (tid < hd) {
        for (unsigned int g = 0; g < kv_group; g++) {
            const unsigned int qh_idx = kh * kv_group + g;
            Qs[g * hd + tid] = (float)q[(unsigned long long)qh_idx * hd + tid];
            Os[g * hd + tid] = 0.0f;
        }
    }
    if (tid < kv_group) { m_s[tid] = -INFINITY; l_s[tid] = 0.0f; }
    __syncthreads();

    if (k0 < k1) {
        // score-phase thread map: 4 lanes per key, 32 keys -> all 128 threads.
        const unsigned int key = tid >> 2;   // 0..31
        const unsigned int sub = tid & 3u;   // 0..3 (dim quarter)

        for (unsigned int kb = k0; kb < k1; kb += DEC_FBC) {
            // --- stage K/V tile into LDS ONCE (reused across the whole group) ---
            for (unsigned int p = tid; p < DEC_FBC * hd; p += DEC_BLOCK) {
                const unsigned int j = p / hd;
                const unsigned int d = p % hd;
                const unsigned int key_g = kb + j;
                if (key_g < k1) {
                    // Ring-buffer physical row: absolute pos = k_base + key_g (key_g is
                    // relative to the window start). Identity for global (k_base=0,
                    // kv_capacity=max_kv). key_g/k0/k1 stay relative for the split math.
                    const unsigned long long off = (unsigned long long)((k_base + key_g) % kv_capacity) * row_stride + kvh_off + d;
                    Ks[p] = k_cache[off];
                    Vs[p] = v_cache[off];
                } else {
                    Ks[p] = (_Float16)0.0f;
                    Vs[p] = (_Float16)0.0f;
                }
            }
            __syncthreads();
            const unsigned int n_keys = (kb + DEC_FBC < k1) ? DEC_FBC : (k1 - kb);

            // --- (1) scores for ALL group heads into Sp[g][key] (no barriers:
            //         every lane writes a distinct Sp slot). 4 lanes/key sum a
            //         32-dim quarter, then a 4-lane xor-shuffle gives the dot. ---
            for (unsigned int g = 0; g < kv_group; g++) {
                float part = 0.0f;
                if (key < n_keys) {
                    const _Float16 *kr = Ks + (unsigned long long)key * hd;
                    const float    *qg = Qs + (unsigned long long)g * hd;
                    const unsigned int d0 = sub * 32u;
                    #pragma unroll
                    for (unsigned int dd = 0; dd < 32u; dd++) {
                        const unsigned int d = d0 + dd;
                        part += qg[d] * (float)kr[d];
                    }
                }
                part += __shfl_xor(part, 1);
                part += __shfl_xor(part, 2);
                if (sub == 0) Sp[g * DEC_FBC + key] = (key < n_keys) ? part * scale : -INFINITY;
            }
            __syncthreads();

            // --- (2) online-softmax merge, ONE THREAD PER HEAD (parallel) ---
            if (tid < kv_group) {
                const unsigned int g = tid;
                float *sp = Sp + g * DEC_FBC;
                float tmax = -INFINITY;
                for (unsigned int j = 0; j < n_keys; j++) if (sp[j] > tmax) tmax = sp[j];
                const float m_old = m_s[g];
                const float m_new = fmaxf(m_old, tmax);
                const float corr  = (m_old == -INFINITY) ? 1.0f : expf(m_old - m_new);
                float rsum = 0.0f;
                for (unsigned int j = 0; j < DEC_FBC; j++) {
                    const float sij = sp[j];
                    const float p = (sij == -INFINITY) ? 0.0f : expf(sij - m_new);
                    sp[j] = p;
                    rsum += p;
                }
                l_s[g]    = l_s[g] * corr + rsum;
                m_s[g]    = m_new;
                corr_s[g] = corr;
            }
            __syncthreads();

            // --- (3) V-accumulate: thread `tid` owns output dim `tid`, loops
            //         the group heads (all 128 lanes busy, no inter-head sync) ---
            if (tid < hd) {
                for (unsigned int g = 0; g < kv_group; g++) {
                    const float *sp = Sp + g * DEC_FBC;
                    const float corr = corr_s[g];
                    float o = 0.0f;
                    #pragma unroll
                    for (unsigned int j = 0; j < DEC_FBC; j++) {
                        o += sp[j] * (float)Vs[(unsigned long long)j * hd + tid];
                    }
                    Os[g * hd + tid] = Os[g * hd + tid] * corr + o;
                }
            }
            __syncthreads();
        }
    }

    // --- write each group head's UNNORMALISED partial + (m, l) ---
    for (unsigned int g = 0; g < kv_group; g++) {
        const unsigned int qh_idx = kh * kv_group + g;
        const unsigned long long po = ((unsigned long long)qh_idx * n_splits + s) * hd;
        if (tid < hd) out_partial[po + tid] = Os[g * hd + tid];
        if (tid == 0) {
            m_partial[(unsigned long long)qh_idx * n_splits + s] = m_s[g];
            l_partial[(unsigned long long)qh_idx * n_splits + s] = l_s[g];
        }
    }
}

// gqa_attn_decode_combine — merge the n_splits partials for each head into the
// final normalised output. Standard flash rescale: global max M = max_s m_s;
// out[d] = (sum_s O_s[d] * exp(m_s - M)) / (sum_s l_s * exp(m_s - M)).
//   grid: (n_head, 1, 1)   block: (DEC_BLOCK == head_dim)
extern "C" __global__ __launch_bounds__(DEC_BLOCK)
void gqa_attn_decode_combine(
        float          * __restrict__ out,          // [n_head, head_dim]
        const float    * __restrict__ out_partial,  // [n_head, n_splits, head_dim]
        const float    * __restrict__ m_partial,    // [n_head, n_splits]
        const float    * __restrict__ l_partial,    // [n_head, n_splits]
        unsigned int     n_head,
        unsigned int     head_dim,
        unsigned int     n_splits) {
    const unsigned int h   = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (h >= n_head) return;
    const unsigned int hd = head_dim;
    if (tid >= hd) return;

    const unsigned long long mb = (unsigned long long)h * n_splits;

    // Global max across splits (ignore empty splits: l==0 -> m==-inf anyway).
    float M = -INFINITY;
    for (unsigned int s = 0; s < n_splits; s++) {
        const float m = m_partial[mb + s];
        if (m > M) M = m;
    }

    float acc = 0.0f;
    float L   = 0.0f;
    for (unsigned int s = 0; s < n_splits; s++) {
        const float m = m_partial[mb + s];
        const float l = l_partial[mb + s];
        if (l <= 0.0f || m == -INFINITY) continue;
        const float w = expf(m - M);
        L   += l * w;
        acc += out_partial[(mb + s) * hd + tid] * w;
    }
    const float inv = (L > 0.0f) ? (1.0f / L) : 0.0f;
    out[(unsigned long long)h * hd + tid] = acc * inv;
}
