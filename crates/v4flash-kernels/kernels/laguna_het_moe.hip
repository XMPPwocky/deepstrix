// Laguna-S-2.1 — heterogeneous (dual-GPU) DECODE MoE split helper.
//
// The routed top-K selection is computed on the dGPU (router lives there in the
// het-split path). This kernel partitions that selection into:
//   * a HOT subset whose experts are resident on the dGPU (run on dstream), and
//   * a COLD subset whose experts stay on the iGPU (run on istream),
// so the two devices compute independent partial MoE sums CONCURRENTLY. The
// partial sums are recombined by addition after the handoff.
//
// A per-layer `hot_map` gives, for each global expert id, its local slot in the
// dGPU-resident packed hot-expert buffer (0..K), or -1 if the expert is cold.
//
// Both partition outputs are written as full [n_used] arrays padded with the
// SENTINEL -1 in the "not mine" slots. The MoE matvec kernels
// (q4_k_pair_matvec_fused_swiglu_batch / q{4,6}_k_matvec_par_batched) skip any
// slot whose `selected[slot] < 0` (an additive, backward-compatible guard —
// existing callers only ever pass ids >= 0), so no dynamic launch-count or host
// readback is needed: both devices launch fixed n_used=TOPK grids and the
// sentinel slots early-out before touching any expert weights.
//
// Launch: grid (1,1,1), block (n_used,1,1) — trivially tiny (n_used = 10).

#include <hip/hip_runtime.h>

extern "C" __global__ void laguna_moe_hot_cold_split(
        const int   * __restrict__ sel,       // [n_used] global expert ids
        const float * __restrict__ ew,        // [n_used] routing weights
        const int   * __restrict__ hot_map,   // [n_expert] global -> local slot, or -1
        int         * __restrict__ hot_sel,   // [n_used] local slot (hot) else -1
        float       * __restrict__ hot_ew,    // [n_used]
        int         * __restrict__ cold_sel,  // [n_used] global id (cold) else -1
        float       * __restrict__ cold_ew,   // [n_used]
        unsigned int  n_used) {
    unsigned int s = threadIdx.x;
    if (s >= n_used) return;
    int e = sel[s];
    int loc = (e >= 0) ? hot_map[e] : -1;
    if (loc >= 0) {            // HOT -> dGPU
        hot_sel[s]  = loc;
        hot_ew[s]   = ew[s];
        cold_sel[s] = -1;
        cold_ew[s]  = 0.0f;
    } else {                   // COLD -> iGPU
        hot_sel[s]  = -1;
        hot_ew[s]   = 0.0f;
        cold_sel[s] = e;
        cold_ew[s]  = ew[s];
    }
}
