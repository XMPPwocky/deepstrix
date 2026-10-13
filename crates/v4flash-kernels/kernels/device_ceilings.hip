// Device ceiling probes — the MEASURED roofline inputs for
// docs/v41/KERNEL_ROOFLINE.md. Each kernel isolates one resource with the
// access pattern named in its comment, so a ceiling is always quoted
// together with the pattern that achieved it.
//
//   ceil_read_v4     streaming read, 16 B/lane, 4 loads in flight per lane,
//                    grid-stride (the best case for a DRAM-bound kernel)
//   ceil_read_b32    same but 4 B/lane (what a scalar-load kernel gets)
//   ceil_read_rows   wave-per-row contiguous read (the gemv/matvec pattern:
//                    each wave streams its own row, 8 rows per WG)
//   ceil_write_v4    streaming write, 16 B/lane
//   ceil_copy_v4     read + write, 16 B/lane each way
//   dp4a_probe       v_dot4_i32_i8 issue throughput, 8 independent
//                    accumulators per lane (the Q8/MXFP4 matvec inner loop)
//
// The f32 FMA, packed-f16 FMA, f16 WMMA and IU8 WMMA peaks live in
// wmma_probe.hip and are driven by the same bench (bench_device_ceilings).
//
// All read probes fold the loaded bits into an XOR accumulator and store it
// only under a never-true predicate, so the compiler cannot drop the loads
// and the write traffic stays zero.

#include <hip/hip_runtime.h>

typedef unsigned int u32;
typedef unsigned long long u64;

extern "C" __global__ void ceil_read_v4(const uint4* __restrict__ in,
                                        u32* __restrict__ sink,
                                        u64 n_vec) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    u32 acc = 0;
    u64 i = tid;
    for (; i + 3 * stride < n_vec; i += 4 * stride) {
        const uint4 a = in[i];
        const uint4 b = in[i + stride];
        const uint4 c = in[i + 2 * stride];
        const uint4 d = in[i + 3 * stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
        acc ^= b.x ^ b.y ^ b.z ^ b.w;
        acc ^= c.x ^ c.y ^ c.z ^ c.w;
        acc ^= d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n_vec; i += stride) {
        const uint4 a = in[i];
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;
}

extern "C" __global__ void ceil_read_b32(const u32* __restrict__ in,
                                         u32* __restrict__ sink,
                                         u64 n) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    u32 acc = 0;
    u64 i = tid;
    for (; i + 3 * stride < n; i += 4 * stride) {
        acc ^= in[i] ^ in[i + stride] ^ in[i + 2 * stride] ^ in[i + 3 * stride];
    }
    for (; i < n; i += stride) acc ^= in[i];
    if (acc == 0x9e3779b9u) sink[0] = acc;
}

// Wave-per-row: 8 waves per WG, wave w of block b owns row (b*8 + w). Each
// wave streams its row with 32 lanes × 16 B = 512 B per step, 4 steps in
// flight. `row_vecs` = uint4 per row.
extern "C" __global__ void ceil_read_rows(const uint4* __restrict__ in,
                                          u32* __restrict__ sink,
                                          u64 row_vecs,
                                          u32 n_rows) {
    const u32 lane = threadIdx.x & 31u;
    const u32 wave = threadIdx.x >> 5u;
    const u32 row = blockIdx.x * 8u + wave;
    if (row >= n_rows) return;
    const uint4* p = in + (u64)row * row_vecs;
    u32 acc = 0;
    u64 i = lane;
    for (; i + 96 < row_vecs; i += 128) {
        const uint4 a = p[i];
        const uint4 b = p[i + 32];
        const uint4 c = p[i + 64];
        const uint4 d = p[i + 96];
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
        acc ^= b.x ^ b.y ^ b.z ^ b.w;
        acc ^= c.x ^ c.y ^ c.z ^ c.w;
        acc ^= d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < row_vecs; i += 32) {
        const uint4 a = p[i];
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;
}

extern "C" __global__ void ceil_write_v4(uint4* __restrict__ out, u64 n_vec, u32 v) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    const uint4 val = make_uint4(v, v + 1u, v + 2u, v + 3u);
    for (u64 i = tid; i < n_vec; i += stride) out[i] = val;
}

extern "C" __global__ void ceil_copy_v4(const uint4* __restrict__ in,
                                        uint4* __restrict__ out,
                                        u64 n_vec) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    u64 i = tid;
    for (; i + 3 * stride < n_vec; i += 4 * stride) {
        const uint4 a = in[i];
        const uint4 b = in[i + stride];
        const uint4 c = in[i + 2 * stride];
        const uint4 d = in[i + 3 * stride];
        out[i] = a;
        out[i + stride] = b;
        out[i + 2 * stride] = c;
        out[i + 3 * stride] = d;
    }
    for (; i < n_vec; i += stride) out[i] = in[i];
}

// dp4a issue throughput: 8 independent int32 accumulators per lane, each
// fed by v_dot4_i32_i8 on register-resident operands. Ops per iteration per
// lane = 8 dot4 × (4 mul + 4 add) = 64.
extern "C" __global__ void dp4a_probe(int* __restrict__ out,
                                      const int* __restrict__ a_in,
                                      const int* __restrict__ b_in,
                                      u32 n_iters) {
    const u32 gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = a_in[threadIdx.x & 7u];
    int b = b_in[threadIdx.x & 7u];
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
#pragma unroll 4
    for (u32 it = 0; it < n_iters; ++it) {
        c0 = __builtin_amdgcn_sudot4(true, a, true, b, c0, false);
        c1 = __builtin_amdgcn_sudot4(true, a, true, b ^ 1, c1, false);
        c2 = __builtin_amdgcn_sudot4(true, a, true, b ^ 2, c2, false);
        c3 = __builtin_amdgcn_sudot4(true, a, true, b ^ 3, c3, false);
        c4 = __builtin_amdgcn_sudot4(true, a ^ 1, true, b, c4, false);
        c5 = __builtin_amdgcn_sudot4(true, a ^ 2, true, b, c5, false);
        c6 = __builtin_amdgcn_sudot4(true, a ^ 3, true, b, c6, false);
        c7 = __builtin_amdgcn_sudot4(true, a ^ 4, true, b, c7, false);
        // Rotate operands so the compiler cannot hoist the dot out of the loop.
        a = a ^ c0;
        b = b ^ c4;
    }
    out[gtid] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
}

// Expert-slot streaming read: EXACTLY the weight bytes one decode MoE call
// must move (n_used experts of `slot_bytes` each, at arbitrary pool slots),
// with nothing else -- the achievable ceiling for that call shape, launch,
// ramp and tail included. Grid (ceil(slot_bytes / chunk_bytes), n_used) x 256:
// WG (x, y) streams bytes [x * chunk_bytes, (x + 1) * chunk_bytes) of slot
// slots[y], 16 B per lane, CEIL_SLOT_UNROLL loads in flight per lane before
// any is consumed (block-stride inside the chunk). chunk_bytes % 16 == 0 and
// slot_bytes % 16 == 0 (the MXFP4 down expert: 6,266,880 B).
#define CEIL_SLOT_UNROLL 8
extern "C" __global__ void ceil_read_slots(const unsigned char* __restrict__ base,
                                           const int* __restrict__ slots,
                                           u64 slot_bytes,
                                           u32 chunk_bytes,
                                           u32* __restrict__ sink) {
    const uint4* p = (const uint4*)(base + (u64)slots[blockIdx.y] * slot_bytes);
    const u64 v_lo = (u64)blockIdx.x * (chunk_bytes / 16u);
    u64 v_hi = v_lo + chunk_bytes / 16u;
    if (v_hi > slot_bytes / 16u) v_hi = slot_bytes / 16u;
    u32 acc = 0;
    u64 i = v_lo + threadIdx.x;
    const u32 st = blockDim.x;
    for (; i + (CEIL_SLOT_UNROLL - 1) * st < v_hi; i += CEIL_SLOT_UNROLL * st) {
        uint4 a[CEIL_SLOT_UNROLL];
#pragma unroll
        for (int k = 0; k < CEIL_SLOT_UNROLL; k++) a[k] = p[i + (u64)k * st];
#pragma unroll
        for (int k = 0; k < CEIL_SLOT_UNROLL; k++) acc ^= a[k].x ^ a[k].y ^ a[k].z ^ a[k].w;
    }
    for (; i < v_hi; i += st) {
        const uint4 a = p[i];
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;
}

// ceil_read_slots in the MXFP4 down kernels' LANE pattern: the chunk is walked in
// 1088 B groups of 8 super-blocks (136 B); lane (bl = lane >> 2, sub = lane & 3) of a
// wave reads its 2 x 16 B nibble blocks at bl * 136 + 32 * sub (+16) and the dword
// holding its 2 scale bytes at bl * 136 + 128 + 4 * (sub >> 1) -- 3 loads per lane
// per 1088 B instead of a coalesced 512 B per instruction. Same bytes as
// ceil_read_slots; chunk_bytes % 1088 == 0, slot_bytes % chunk_bytes == 0.
#define CEIL_MX_UNROLL 4
extern "C" __global__ void ceil_read_slots_mx(const unsigned char* __restrict__ base,
                                              const int* __restrict__ slots,
                                              u64 slot_bytes,
                                              u32 chunk_bytes,
                                              u32* __restrict__ sink) {
    const unsigned char* p = base + (u64)slots[blockIdx.y] * slot_bytes + (u64)blockIdx.x * chunk_bytes;
    const u32 lane = threadIdx.x & 31u, wave = threadIdx.x >> 5, n_waves = blockDim.x >> 5;
    const u32 bl = lane >> 2, sub = lane & 3u;
    const u32 groups = chunk_bytes / 1088u;
    u32 acc = 0;
    u32 g = wave;
    for (; g + (CEIL_MX_UNROLL - 1) * n_waves < groups; g += CEIL_MX_UNROLL * n_waves) {
        uint4 a[CEIL_MX_UNROLL], b[CEIL_MX_UNROLL];
        u32 s[CEIL_MX_UNROLL];
#pragma unroll
        for (int k = 0; k < CEIL_MX_UNROLL; k++) {
            const unsigned char* sb = p + (u64)(g + k * n_waves) * 1088u + bl * 136u;
            __builtin_memcpy(&a[k], __builtin_assume_aligned(sb + 32u * sub, 8), 16);
            __builtin_memcpy(&b[k], __builtin_assume_aligned(sb + 32u * sub + 16u, 8), 16);
            s[k] = *(const u32*)__builtin_assume_aligned(sb + 128u + 4u * (sub >> 1), 4);
        }
#pragma unroll
        for (int k = 0; k < CEIL_MX_UNROLL; k++) acc ^= a[k].x ^ a[k].y ^ a[k].z ^ a[k].w ^ b[k].x ^ b[k].y ^ b[k].z ^ b[k].w ^ s[k];
    }
    for (; g < groups; g += n_waves) {
        const unsigned char* sb = p + (u64)g * 1088u + bl * 136u;
        uint4 a, b;
        __builtin_memcpy(&a, __builtin_assume_aligned(sb + 32u * sub, 8), 16);
        __builtin_memcpy(&b, __builtin_assume_aligned(sb + 32u * sub + 16u, 8), 16);
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ *(const u32*)__builtin_assume_aligned(sb + 128u + 4u * (sub >> 1), 4);
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;
}
