// Box 2's small-batch decode chain (`V41_B2_FAST_CHAIN`, 2026-09-27): two
// kernels that stand in for five commands of the batched by-expert chain on
// box 2's executor (`remote_experts::MoeExecutor`). Nothing on the hub uses
// them.
//
// 1. b2_moe_group_wi_builder -- ONE work-group replaces
//      memset(group_count) -> moe_group_builder_hetsplit
//      -> memset(n_work_items) -> moe_work_items_builder
//    for a pass of batch * n_used <= B2FB_MAX picks. It writes, in the SAME
//    formats the downstream kernels read (the kwide gate+up, the smallb /
//    kwide2 down):
//      group_count[g]                      for every group THIS pass touches
//      expert_members[g * max_per_expert + pos]   packed (b << 16) | slot
//      work_items[0 .. n)                  packed (g << 16) | member_start
//      n_work_items[0] = n
//    Untouched groups keep stale counts: every consumer reads group_count[e]
//    only for the `e` of a work item below the device-side count (kwide,
//    kwide2 and smallb all exit on blockIdx.y / w_idx >= *n_work_items first),
//    and every work item names a group written here.
//    Per pick the group id is EXACTLY moe_group_builder_hetsplit's (resident
//    rank, mode, cap, the `g >= n_expert` drop); group_count is the full
//    claim count and members past max_per_expert are dropped, as there.
//    Member order: (b, slot) ascending within a group (the old builder's
//    was atomicAdd order, i.e. unspecified). Order is immaterial downstream:
//    every consumer computes each member's (token, slot) output from that
//    member alone and writes it to its own (token, slot) row; work items are
//    independent work-groups writing disjoint rows.
//
// 2. b2_reduce_partials_zero -- q2_k_reduce_partials_hetsplit's sum,
//    bit for bit (same take rule, same slot order, same f32 adds), plus:
//    * zeroes every partial slot it consumes, and every slot holding a real
//      pick id (< zero_below), so `partials` is all-zero again after the
//      request and the next one needs no memset (the executor tracks the
//      rows that may be dirty after an old-chain pass or an error);
//    * optionally writes the f16 result (__float2half, RNE: exactly
//      f32_to_f16_cast) straight into a host-mapped pinned buffer. The event
//      the executor already waits on (recorded without
//      hipEventDisableSystemFence) makes it host-visible, like readback_pack:
//      no cast kernel, no stream sync, no blocking copy.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define B2FB_MAX 128u

extern "C" __global__ void __launch_bounds__(B2FB_MAX)
b2_moe_group_wi_builder(
        int       * __restrict__ group_count,      // [n_expert]; only touched groups written
        int       * __restrict__ expert_members,   // [n_expert * max_per_expert]
        int       * __restrict__ work_items,       // [max_items]
        int       * __restrict__ n_work_items,     // [1]; written (no pre-zero)
        const int * __restrict__ d_selected,       // [batch, n_used]
        const int * __restrict__ remap,            // id -> dGPU dense slot, or -(iGPU slot + 1)
        unsigned int mode,                         // 0 = iGPU misses, 1 = dGPU hits
        unsigned int cap,
        unsigned int batch,
        unsigned int n_used,
        unsigned int n_expert,                     // bound of the GROUP id space
        unsigned int max_per_expert,
        unsigned int chunk_size,
        unsigned int max_items) {
    __shared__ int s_g[B2FB_MAX];    // group of pick idx, -1 = not in this pass
    __shared__ int s_nch[B2FB_MAX];  // work items emitted by pick idx (its group's leader)
    const unsigned int idx = threadIdx.x;
    const unsigned int total = batch * n_used;   // host guarantees total <= blockDim.x
    int g = -1;
    unsigned int b = 0, slot = 0;
    if (idx < total) {
        b = idx / n_used;
        slot = idx - b * n_used;
        const int e = d_selected[idx];
        if (e >= 0) {
            const int dense = remap[e];
            unsigned int rank = 0;
            if (dense >= 0) {
                const int* sel_b = d_selected + (size_t)b * (size_t)n_used;
                for (unsigned int s = 0; s < slot; s++) {
                    const int es = sel_b[s];
                    if (es >= 0 && remap[es] >= 0) rank++;
                }
            }
            const bool dgpu_takes = (dense >= 0) && (rank < cap);
            int gg = -1;
            if (mode == 1) {
                if (dgpu_takes) gg = dense;
            } else if (!dgpu_takes) {
                gg = (dense >= 0) ? e : (-dense - 1);
            }
            if (gg >= 0 && (unsigned int)gg < n_expert) g = gg;
        }
    }
    s_g[idx] = g;
    __syncthreads();
    int nch = 0;
    if (g >= 0) {
        unsigned int pos = 0, cnt = 0;
        for (unsigned int j = 0; j < total; j++) {
            if (s_g[j] == g) {
                cnt++;
                if (j < idx) pos++;
            }
        }
        if (pos < max_per_expert) {
            expert_members[(unsigned int)g * max_per_expert + pos] = (int)((b << 16) | slot);
        }
        if (pos == 0) {
            group_count[g] = (int)cnt;
            nch = (int)((cnt + chunk_size - 1u) / chunk_size);
        }
    }
    s_nch[idx] = nch;
    __syncthreads();
    if (nch > 0) {
        unsigned int off = 0;
        for (unsigned int j = 0; j < idx; j++) off += (unsigned int)s_nch[j];
        for (unsigned int c = 0; c < (unsigned int)nch; c++) {
            const unsigned int p = off + c;
            if (p < max_items) {
                work_items[p] = (int)(((unsigned int)g << 16) | ((c * chunk_size) & 0xffffu));
            }
        }
    }
    if (idx == 0) {
        unsigned int tot = 0;
        for (unsigned int j = 0; j < total; j++) tot += (unsigned int)s_nch[j];
        *n_work_items = (int)tot;
    }
}

// 8 consecutive rows of one token per thread (n_rows % 8 == 0): 16-byte
// loads/stores, and the f16 row goes to host memory as 16-byte stores.
// Measured (kernel trace, 3 rows x 5120, gfx1151): one row per thread with
// 2-byte host stores 7.2 us; this 3.0 us (the old reduce alone was 2.5, plus
// the 1.2 us cast and the 4.1 us D2H it replaces). Each row's sum is still
// 0 + p[s0] + p[s1] + ... in slot order, exactly q2_k_reduce_partials_hetsplit's.
#define B2R_V 8u
extern "C" __global__ void b2_reduce_partials_zero(
        float          * __restrict__ out,        // [batch, n_rows]
        unsigned short * __restrict__ out16,      // [batch, n_rows] host-mapped, or nullptr
        float          * __restrict__ partials,   // [batch*n_used, n_rows]; consumed slots zeroed
        const int      * __restrict__ d_selected, // [batch, n_used]
        const int      * __restrict__ remap,
        unsigned int     mode,
        unsigned int     cap,
        unsigned int     n_used,
        unsigned int     n_rows,                  // multiple of 8
        unsigned int     batch,
        unsigned int     zero_below) {            // real pick ids are < zero_below
    const unsigned int per_b = n_rows / B2R_V;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch * per_b) return;
    const unsigned int b    = tid / per_b;
    const unsigned int row0 = (tid - b * per_b) * B2R_V;
    const int* sel_b = d_selected + (size_t)b * (size_t)n_used;
    float acc[B2R_V];
    #pragma unroll
    for (unsigned int k = 0; k < B2R_V; k++) acc[k] = 0.0f;
    unsigned int rank = 0;
    for (unsigned int s = 0; s < n_used; s++) {
        const int e = sel_b[s];
        const bool resident   = (e >= 0) && (remap[e] >= 0);
        const bool dgpu_takes = resident && (rank < cap);
        if (resident) rank++;
        const bool take = (mode == 1) ? dgpu_takes : !dgpu_takes;
        float4* p = (float4*)(partials + ((size_t)b * (size_t)n_used + (size_t)s) * (size_t)n_rows + (size_t)row0);
        if (take) {
            const float4 x = p[0];
            const float4 y = p[1];
            acc[0] += x.x; acc[1] += x.y; acc[2] += x.z; acc[3] += x.w;
            acc[4] += y.x; acc[5] += y.y; acc[6] += y.z; acc[7] += y.w;
        }
        if (take || (e >= 0 && (unsigned int)e < zero_below)) {
            const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            p[0] = z;
            p[1] = z;
        }
    }
    float4* o = (float4*)(out + (size_t)b * (size_t)n_rows + (size_t)row0);
    o[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    o[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    if (out16 != nullptr) {
        uint4 h;
        h.x = (unsigned)__half_as_ushort(__float2half(acc[0])) | ((unsigned)__half_as_ushort(__float2half(acc[1])) << 16);
        h.y = (unsigned)__half_as_ushort(__float2half(acc[2])) | ((unsigned)__half_as_ushort(__float2half(acc[3])) << 16);
        h.z = (unsigned)__half_as_ushort(__float2half(acc[4])) | ((unsigned)__half_as_ushort(__float2half(acc[5])) << 16);
        h.w = (unsigned)__half_as_ushort(__float2half(acc[6])) | ((unsigned)__half_as_ushort(__float2half(acc[7])) << 16);
        *(uint4*)(out16 + (size_t)b * (size_t)n_rows + (size_t)row0) = h;
    }
}
