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

// M61 prefill het-split: residency-aware group builder.
//
// remap[e] >= 0 marks a dGPU-resident ("hot") expert and gives its dense
// slot in the packed hot weight buffers. A token's resident slots are
// ranked in slot order; ranks < cap belong to the dGPU, the rest (and all
// non-resident slots) to the iGPU. Must agree bit-for-bit with the rank
// logic in q2_k_reduce_partials_hetsplit — both scan d_selected in slot
// order counting resident slots.
//
// M63: remap[e] < 0 additionally CARRIES the iGPU slot as -(igpu_slot+1)
// (not a bare -1). With the full 256-expert iGPU buffer igpu_slot == e, so
// the group id below is unchanged; with IGPU_DEDUP_HOT the iGPU holds only
// the 256-n_hot cold experts packed dense and igpu_slot is that packed slot.
//
//   mode 0 (iGPU): keep the complement (misses AND over-cap resident slots);
//                  group ids in the iGPU SLOT space (group arrays sized
//                  n_expert = the iGPU slot count). Over-cap resident slots
//                  use the raw id — only reachable without de-dup.
//   mode 1 (dGPU): keep the hits; group ids in DENSE remap space (group
//                  arrays sized n_hot).
extern "C" __global__ void moe_group_builder_hetsplit(
        int       * __restrict__ group_count,      // [n_expert] — must be pre-zeroed
        int       * __restrict__ expert_members,   // [n_expert * max_per_expert]
        const int * __restrict__ d_selected,       // [B, n_used] — flat
        const int * __restrict__ remap,            // [N_EXPERT total] id → dGPU dense slot, or -(iGPU slot + 1)
        unsigned int mode,                         // 0 = iGPU misses, 1 = dGPU hits
        unsigned int cap,                          // max resident slots/token on dGPU
        unsigned int batch,                        // B
        unsigned int n_used,
        unsigned int n_expert,                     // bound of the GROUP id space
        unsigned int max_per_expert) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = batch * n_used;
    if (idx >= total) return;
    const unsigned int b = idx / n_used;
    const unsigned int slot = idx - b * n_used;
    const int e = d_selected[idx];
    if (e < 0) return;
    const int dense = remap[e];
    // Resident rank = #resident slots before this one in the same token.
    unsigned int rank = 0;
    if (dense >= 0) {
        const int* sel_b = d_selected + (size_t)b * (size_t)n_used;
        for (unsigned int s = 0; s < slot; s++) {
            const int es = sel_b[s];
            if (es >= 0 && remap[es] >= 0) rank++;
        }
    }
    const bool dgpu_takes = (dense >= 0) && (rank < cap);
    int g;
    if (mode == 1) {
        if (!dgpu_takes) return;
        g = dense;
    } else {
        if (dgpu_takes) return;
        // Miss (dense < 0): the iGPU slot carried by remap. OVER-CAP resident
        // (dense >= 0, rank >= cap): the iGPU copy sits at the raw id `e`
        // (full 256-expert buffer). `-dense - 1` would be <= -1 here and the
        // bound check below would silently DROP the slot while
        // q2_k_reduce_partials_hetsplit (mode 0) still sums its never-written
        // partial row. Under IGPU_DEDUP_HOT the cap is pinned to
        // N_EXPERT_USED (dgpu_hot_cap), so over-cap cannot occur and a raw id
        // never reaches the packed buffer.
        g = (dense >= 0) ? e : (-dense - 1);
    }
    if ((unsigned int)g >= n_expert) return;
    const int pos = atomicAdd(&group_count[g], 1);
    if ((unsigned int)pos >= max_per_expert) return; // overflow guard
    expert_members[(unsigned int)g * max_per_expert + (unsigned int)pos]
        = (int)((b << 16) | slot);
}
