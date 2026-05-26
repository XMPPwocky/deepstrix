// M50 by-expert MoE — work-items builder (Phase 7.2).
//
// Reads group_count[n_expert] from the prior moe_group_builder pass.
// Emits a flat list of (expert_id, member_start) work items, one per
// chunk of CHUNK_SIZE members per active expert, via atomicAdd into
// n_work_items[0].
//
// Packing: work_items[i] = (expert_id << 16) | member_start.
//   expert_id fits in 16 bits (max 256 experts).
//   member_start fits in 16 bits (max B_MAX = 256 = 8 bits, fits).
//
// Launch: block(n_expert) threads, grid(1). Each thread t handles
// expert t. It computes its expert's chunk count = ceil(group_count[t]/CHUNK_SIZE)
// and reserves that many slots in work_items via atomicAdd. With n_expert=256
// threads and 256-element shared atomic counter, contention is bounded —
// each thread does at most ceil(B_MAX/CHUNK_SIZE) atomicAdds. At
// CHUNK_SIZE=16 and B_MAX=256 that's ≤16 atomics/thread × 256 threads,
// all coalescing onto n_work_items[0]. Total runtime: microseconds.
//
// `chunk_size` is passed as a kernel argument so the host can tune it
// without recompiling.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void moe_work_items_builder(
        int       * __restrict__ work_items,       // [max_items]
        int       * __restrict__ n_work_items,     // [1] — must be pre-zeroed
        const int * __restrict__ group_count,      // [n_expert]
        unsigned int n_expert,
        unsigned int chunk_size,
        unsigned int max_items) {
    const unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_expert) return;
    const int n = group_count[e];
    if (n <= 0) return;
    const unsigned int n_chunks = ((unsigned int)n + chunk_size - 1) / chunk_size;
    for (unsigned int c = 0; c < n_chunks; c++) {
        const int pos = atomicAdd(n_work_items, 1);
        if ((unsigned int)pos >= max_items) return; // overflow guard
        const unsigned int member_start = c * chunk_size;
        work_items[pos] = (int)((e << 16) | (member_start & 0xffffu));
    }
}
