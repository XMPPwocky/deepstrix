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

// M50 hybrid dispatch: split work items into two arrays AND sort by expert.
//
// Each thread (one per expert) computes its own chunks, then a block-wide
// prefix sum gives each thread its write offsets in both output arrays.
// Result: staged_wi and chunked_wi are sorted by expert (all of expert e's
// chunks consecutive, before expert e+1's). Adjacent WGs in the iq2 launch
// touch the SAME expert's weights → L2 reuse on weight reads.
//
// Why sorted matters: the staged kernel's PMC showed L2CacheHit dropped from
// 97% (chunked baseline) → 70% because each weight byte is read once per WG
// from DRAM/L2 without spatial-temporal locality across WGs. Sorting by
// expert restores locality — consecutive WGs reuse warmed L2 lines.
//
// Launch: one block of n_expert threads (so the in-block prefix sum works).
// Requires n_expert ≤ 1024.
extern "C" __global__ void moe_work_items_builder_split(
        int       * __restrict__ staged_wi,        // [max_items]
        int       * __restrict__ chunked_wi,       // [max_items]
        int       * __restrict__ n_staged,         // [1]
        int       * __restrict__ n_chunked,        // [1]
        const int * __restrict__ group_count,      // [n_expert]
        unsigned int n_expert,
        unsigned int chunk_size,
        unsigned int threshold,
        unsigned int max_items) {
    // n_expert ≤ 1024 (current model has 256); single block, one thread per
    // expert. Threads beyond n_expert sit idle (contribute 0 to prefix sums).
    __shared__ int s_n_staged_chunks[1024];
    __shared__ int s_n_chunked_chunks[1024];

    const unsigned int e = threadIdx.x;
    const bool active = e < n_expert;

    // Stage 1: per-expert chunk counts (staged vs chunked).
    int my_n_staged  = 0;
    int my_n_chunked = 0;
    int my_n_chunks  = 0;
    int my_total     = 0;
    if (active) {
        const int n = group_count[e];
        if (n > 0) {
            my_total    = n;
            my_n_chunks = (n + (int)chunk_size - 1) / (int)chunk_size;
            for (int c = 0; c < my_n_chunks; c++) {
                const int member_start = c * (int)chunk_size;
                const int remaining    = n - member_start;
                const int actual_size  = (remaining < (int)chunk_size) ? remaining : (int)chunk_size;
                if ((unsigned int)actual_size >= threshold) my_n_staged++;
                else                                         my_n_chunked++;
            }
        }
    }
    s_n_staged_chunks[e]  = my_n_staged;
    s_n_chunked_chunks[e] = my_n_chunked;
    __syncthreads();

    // Stage 2: in-place Hillis-Steele inclusive prefix sum over n_expert
    // elements. At most ceil(log2(n_expert)) iterations (≤10 for n_expert ≤ 1024).
    // We process both arrays in parallel.
    for (unsigned int offset = 1; offset < n_expert; offset <<= 1) {
        int add_s = (e >= offset) ? s_n_staged_chunks [e - offset] : 0;
        int add_c = (e >= offset) ? s_n_chunked_chunks[e - offset] : 0;
        __syncthreads();
        if (e < n_expert) {
            s_n_staged_chunks [e] += add_s;
            s_n_chunked_chunks[e] += add_c;
        }
        __syncthreads();
    }

    // Convert inclusive → exclusive prefix sum (subtract own count).
    const int staged_off  = active ? (s_n_staged_chunks [e] - my_n_staged ) : 0;
    const int chunked_off = active ? (s_n_chunked_chunks[e] - my_n_chunked) : 0;

    // Stage 3: write chunks to their pre-computed slots, preserving expert order.
    if (active && my_total > 0) {
        int si = 0, ci = 0;
        for (int c = 0; c < my_n_chunks; c++) {
            const int member_start = c * (int)chunk_size;
            const int remaining    = my_total - member_start;
            const int actual_size  = (remaining < (int)chunk_size) ? remaining : (int)chunk_size;
            const int packed = (int)((e << 16) | ((unsigned int)member_start & 0xffffu));
            if ((unsigned int)actual_size >= threshold) {
                const int pos = staged_off + si;
                if ((unsigned int)pos < max_items) staged_wi[pos] = packed;
                si++;
            } else {
                const int pos = chunked_off + ci;
                if ((unsigned int)pos < max_items) chunked_wi[pos] = packed;
                ci++;
            }
        }
    }

    // Thread (n_expert - 1) holds the final inclusive prefix-sum totals.
    if (e == n_expert - 1) {
        *n_staged  = s_n_staged_chunks [e];
        *n_chunked = s_n_chunked_chunks[e];
    }
}
