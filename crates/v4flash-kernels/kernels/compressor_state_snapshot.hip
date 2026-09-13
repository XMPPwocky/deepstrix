// compressor_state_snapshot — copy the shared state_kv/state_score
// buffers to a per-boundary slot in scratch. Called at each boundary
// firing in the prefill compressor loop so that the post-boundary
// stages (pool, rms_w, rope, fp8, f16rt, append) can run as ONE
// batched pass over all boundaries instead of one launch chain per
// boundary.
//
// Total bytes copied per call:
//   ratio==4 main:    8 rows × 1024 cols ×  4B × 2 buffers =  64 KB
//   ratio==128 main:  128 × 512 × 4B × 2                    = 512 KB
//   ratio==4 indexer: 8 × 256 × 4B × 2                      =  16 KB
//
// Grid (ceil(elems/256), 1, 1), block 256. Vectorized 4 floats/thread.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void compressor_state_snapshot(
        float       * __restrict__ snap_kv,        // [snap_elems] dst
        float       * __restrict__ snap_score,     // [snap_elems] dst
        const float * __restrict__ state_kv,       // [snap_elems] src
        const float * __restrict__ state_score,    // [snap_elems] src
        unsigned int   snap_elems) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= snap_elems) return;
    snap_kv[i]    = state_kv[i];
    snap_score[i] = state_score[i];
}

// ---------------------------------------------------------------------
// compressor_snapshot_gather — batched, launch-free replacement for the
// serial `state_write → snapshot → shuffle` loop on the PREFILL path.
//
// The ring buffer + shuffle exist because DECODE sees one position at a
// time. In prefill every position of the chunk is already projected into
// kv_cur/sc_cur, so each boundary's snapshot is a pure gather:
//
//   snapshot[k] row r  <->  absolute position  p = L_k - (coff-1)*ratio + r
//     kv    = kv_cur[p - pos0]
//     score = sc_cur[p - pos0] + ape[p % ratio]
//
// which is bit-for-bit what state_write produced. Rows with p < pos0 (only
// the first boundary, and only when pos0 % ratio == 0 — the caller's
// precondition) are carried in from the previous chunk's state, where they
// already sit at row r with the APE term applied.
//
// Grid (ceil(width/256), rows, n_boundaries).
// ---------------------------------------------------------------------
extern "C" __global__ void compressor_snapshot_gather(
        float        * __restrict__ snap_kv,         // [n_bnd, rows, width]
        float        * __restrict__ snap_score,
        const float  * __restrict__ kv_cur,          // [B, width]
        const float  * __restrict__ sc_cur,          // [B, width]
        const float  * __restrict__ state_kv_in,     // [rows, width]
        const float  * __restrict__ state_score_in,  // [rows, width]
        const __half * __restrict__ ape,             // [ratio, width]
        const int    * __restrict__ group_start,     // [n_bnd] = L_k
        unsigned int   width,
        unsigned int   ratio,
        unsigned int   rows,                         // coff * ratio
        int            pos0,
        unsigned int   b) {
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;
    const unsigned int r = blockIdx.y;
    const unsigned int k = blockIdx.z;

    const int coff_m1 = (int)(rows / ratio) - 1;
    const int p = group_start[k] - coff_m1 * (int)ratio + (int)r;
    const unsigned long long dst = ((unsigned long long)k * rows + r) * width + j;

    if (p >= pos0) {
        const unsigned int i = (unsigned int)(p - pos0);
        if (i >= b) return;                       // defensive; caller guarantees
        const unsigned long long src = (unsigned long long)i * width + j;
        const unsigned int pm = (unsigned int)((unsigned int)p % ratio);
        snap_kv[dst]    = kv_cur[src];
        snap_score[dst] = sc_cur[src]
                        + __half2float(ape[(unsigned long long)pm * width + j]);
    } else {
        const unsigned long long src = (unsigned long long)r * width + j;
        snap_kv[dst]    = state_kv_in[src];
        snap_score[dst] = state_score_in[src];
    }
}
