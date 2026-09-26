// Candidates for indexer_gather_batched (gfx1201). Same signature as production:
//   (dst [B, top_k, head_dim] f16, comp_kv [n, head_dim] f16, selected [B, top_k], top_k, head_dim, comp_base_per)
// Production: grid (top_k, B, ceil(head_dim/256)) x 256, ONE f16 (2 B) per thread -> 2 x 4096 WGs
// per b=4 call, 32K waves each issuing a single 2-byte load: latency-bound (measured 64-90 GB/s
// of 640 for random cold rows). These variants move VEC bytes per thread (b128 / b64), put
// ROWS rows in one WG and give each thread IPT independent rows (IPT loads in flight before the
// first store). Pure copy: bit-exact by construction. Sentinel rows (selected < 0) are skipped
// exactly as production (destination row left untouched).
//
// Grid for a candidate: (ceil(top_k / ROWS), B), block = ROWS * (head_dim*2/VEC) / IPT.
// head_dim*2 must be a multiple of VEC (it is: 1024 B rows).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

template <unsigned VEC, unsigned ROWS, unsigned IPT>
__device__ __forceinline__ void gather_body(
        _Float16* __restrict__ dst_b, const _Float16* __restrict__ comp_kv,
        const int* __restrict__ selected_b, unsigned top_k, unsigned head_dim,
        const int* __restrict__ comp_base_per) {
    const unsigned row_bytes = head_dim * 2u;
    const unsigned tpr       = row_bytes / VEC;              // threads per row (64 at b128)
    const unsigned rows_per_pass = ROWS / IPT;               // rows covered by one block pass
    const unsigned tid = threadIdx.x;
    const unsigned b   = blockIdx.y;
    const unsigned r_in = tid / tpr, seg = tid % tpr;
    const unsigned row0 = blockIdx.x * ROWS;
    const size_t base = (size_t)(comp_base_per ? (unsigned)comp_base_per[b] : 0u);
    const int* sel = selected_b + (size_t)b * top_k;

    // branch-free: all IPT index loads, then all IPT data loads (a sentinel / out-of-range row
    // reads row `base` harmlessly and is not stored), then the stores.
    uint4 v[IPT]; unsigned long long v2[IPT];
    bool ok[IPT]; unsigned rr[IPT]; int src[IPT];
    #pragma unroll
    for (unsigned i = 0; i < IPT; i++) {
        rr[i] = row0 + r_in + i * rows_per_pass;
        src[i] = (rr[i] < top_k) ? sel[rr[i]] : -1;
    }
    #pragma unroll
    for (unsigned i = 0; i < IPT; i++) {
        ok[i] = src[i] >= 0;
        const unsigned s = ok[i] ? (unsigned)src[i] : 0u;
        const unsigned char* p = reinterpret_cast<const unsigned char*>(comp_kv + (base + s) * head_dim) + seg * VEC;
        if (VEC == 16) v[i] = *reinterpret_cast<const uint4*>(p);
        else           v2[i] = *reinterpret_cast<const unsigned long long*>(p);
    }
    #pragma unroll
    for (unsigned i = 0; i < IPT; i++) {
        if (!ok[i]) continue;
        unsigned char* d = reinterpret_cast<unsigned char*>(dst_b + ((size_t)b * top_k + rr[i]) * head_dim) + seg * VEC;
        if (VEC == 16) *reinterpret_cast<uint4*>(d) = v[i];
        else           *reinterpret_cast<unsigned long long*>(d) = v2[i];
    }
}

#define GATHER_KERNEL(NAME, VEC, ROWS, IPT)                                                            \
    extern "C" __global__ __launch_bounds__(ROWS * (1024u / VEC) / IPT)                                 \
    void NAME(_Float16* __restrict__ dst, const _Float16* __restrict__ comp_kv,                        \
              const int* __restrict__ selected, unsigned top_k, unsigned head_dim,                     \
              const int* __restrict__ comp_base_per) {                                                 \
        gather_body<VEC, ROWS, IPT>(dst, comp_kv, selected, top_k, head_dim, comp_base_per);           \
    }

// name            VEC ROWS IPT   block
GATHER_KERNEL(gather_u4_r1,       16,  1, 1)   //  64
GATHER_KERNEL(gather_u4_r2,       16,  2, 1)   // 128
GATHER_KERNEL(gather_u4_r4,       16,  4, 1)   // 256
GATHER_KERNEL(gather_u4_r8,       16,  8, 1)   // 512
GATHER_KERNEL(gather_u4_r16,      16, 16, 1)   // 1024
GATHER_KERNEL(gather_u2_r4,        8,  4, 1)   // 512
GATHER_KERNEL(gather_u4_r4_2ipt,  16,  8, 2)   // 256  (8 rows, 2 per thread)
GATHER_KERNEL(gather_u4_r16_4ipt, 16, 16, 4)   // 256  (16 rows, 4 per thread)
GATHER_KERNEL(gather_u4_r32_4ipt, 16, 32, 4)   // 512
GATHER_KERNEL(gather_u4_r32_8ipt, 16, 32, 8)   // 256
