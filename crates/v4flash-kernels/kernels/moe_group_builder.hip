// M50 by-expert MoE — group-builder pre-pass kernel.
//
// Stats showed actual per-expert reuse at B=64 is ~6.6x (vs uniform-model
// 1.9x). The by-expert iq2 kernel reads each expert's weight rows ONCE
// per chunk, amortized over all (token, slot) pairs that picked it. To
// dispatch by expert, we need to invert the d_selected[B, n_used] map:
//
//   d_selected[b, slot] = expert_id    -->    per-expert (b, slot) lists
//
// This kernel does the inversion. Single launch per layer:
//
//   block(B * n_used) threads, grid(1, 1, 1)
//
// Each thread idx in [0, B*n_used) corresponds to one (b, slot) selection.
// Reads its expert e and atomicAdds into group_count[e] to claim a slot
// in expert_members[e * max_per_expert + slot]. After this kernel:
//
//   group_count[e]    = number of (b, slot) pairs that picked expert e
//   expert_members[e * max_per_expert + 0..group_count[e]] = packed
//                       (b<<8 | slot) — the members of expert e's group
//                       in INSERTION ORDER (atomicAdd ordering, not
//                       deterministic but doesn't matter for correctness).
//
// `max_per_expert` is sized to B so worst-case all B tokens picking the
// same expert in one slot fits. Caller must zero group_count[] before launch.
//
// Cost: 384 threads, 1 atomic per thread, < 5us at B=64.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void moe_group_builder(
        int       * __restrict__ group_count,      // [n_expert] — must be pre-zeroed
        int       * __restrict__ expert_members,   // [n_expert * max_per_expert]
        const int * __restrict__ d_selected,       // [B, n_used] — flat
        unsigned int batch,                        // B
        unsigned int n_used,
        unsigned int n_expert,
        unsigned int max_per_expert) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = batch * n_used;
    if (idx >= total) return;
    const unsigned int b = idx / n_used;
    const unsigned int slot = idx - b * n_used;
    const int e = d_selected[idx];
    if (e < 0 || (unsigned int)e >= n_expert) return;
    const int pos = atomicAdd(&group_count[e], 1);
    if ((unsigned int)pos >= max_per_expert) return; // overflow guard
    const int packed = (int)((b << 16) | slot);
    expert_members[(unsigned int)e * max_per_expert + (unsigned int)pos] = packed;
}
