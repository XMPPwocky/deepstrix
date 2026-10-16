// Cache-reuse probe kernels (tests/cache_reuse_bench.rs). NOT used by the engine.
//
// Questions they answer (docs in the test header):
//   Q1 (gfx1201, 64 MB Infinity Cache): can one 44-55 MB Q8_0 matrix stay resident in the IC
//      across a 64-200 MB stream of other weights if the loads carry temporal hints?
//   Q2 (gfx1151, 32 MB MALL): does the MALL cache iGPU reads of GTT / host-pinned memory at all?
//
// One kernel shape, instantiated per load policy: a ROW-STREAMING GEMV read. Each wave owns one
// row of `row_bytes` (a Q8_0 row: K = 5120 -> 160 blocks x 34 B = 5440 B), its lanes read the row
// with contiguous b128 loads (4 in flight per lane), and the wave reduces a wrapping u32 sum of
// every word it read into out[row] -- so no load is dead, and the host checks every byte was read
// (the fill pattern is a hash of the word index, recomputed on the host).
//
// Loads: __builtin_amdgcn_raw_buffer_load_b128(rsrc, voffset, soffset, aux). `aux` is the
// instruction's cache-policy immediate:
//   gfx12 (gfx1201): bits [2:0] = TH (0 RT, 1 NT, 2 HT, 3 LU, 4 NT_RT, 5 RT_NT, 6 NT_HT, 7 BYPASS),
//                    bits [4:3] = SCOPE (0 CU) -- we only pass 0..7, i.e. scope CU.
//   gfx11 (gfx1151): bit 0 = GLC, bit 1 = SLC, bit 2 = DLC (aux 4 = dlc, 6 = slc dlc).
// The descriptor's 4th dword `w3` is a kernel argument (host default 0x31004000 = 32-bit format,
// raw-buffer OOB select, resource level 1 -- the gfx11/gfx12 value composable_kernel uses), so a
// wrong default can be overridden at run time (CACHE_PROBE_W3) without a rebuild; the host checks
// the buffer path against the flat path's checksums before timing anything.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef unsigned int cp_u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint32_t cp_hsum(cp_u32x4 v) { return v.x + v.y + v.z + v.w; }

__device__ __forceinline__ uint32_t cp_wave_sum(uint32_t acc) {
    for (int o = warpSize / 2; o > 0; o >>= 1) {
        acc += (uint32_t)__shfl_xor((int)acc, o);
    }
    return acc;
}

template <int AUX>
__device__ __forceinline__ cp_u32x4 cp_ld(__amdgpu_buffer_rsrc_t rs, uint32_t voff, uint32_t soff) {
    return __builtin_amdgcn_raw_buffer_load_b128(rs, (int)voff, (int)soff, AUX);
}

// One wave per row; 4 b128 loads in flight per lane, then a tail.
template <int AUX>
__device__ __forceinline__ void cp_stream_rows_buf(const uint8_t* base, uint32_t row_bytes,
                                                   uint32_t rows, uint32_t w3,
                                                   uint32_t* __restrict__ out) {
    const uint32_t W = warpSize;
    const uint32_t lane = threadIdx.x % W;
    const uint32_t row = __builtin_amdgcn_readfirstlane(blockIdx.x * (blockDim.x / W) + threadIdx.x / W);
    if (row >= rows) return;
    // num_records = the whole buffer (rows * row_bytes < 2^31 by the host's sizing).
    __amdgpu_buffer_rsrc_t rs =
        __builtin_amdgcn_make_buffer_rsrc((void*)base, (short)0, (int)(rows * row_bytes), (int)w3);
    const uint32_t soff = __builtin_amdgcn_readfirstlane(row * row_bytes);
    const uint32_t nvec = row_bytes >> 4;
    uint32_t acc = 0;
    uint32_t v = lane;
    for (; v + 3 * W < nvec; v += 4 * W) {
        cp_u32x4 a = cp_ld<AUX>(rs, v * 16u, soff);
        cp_u32x4 b = cp_ld<AUX>(rs, (v + W) * 16u, soff);
        cp_u32x4 c = cp_ld<AUX>(rs, (v + 2 * W) * 16u, soff);
        cp_u32x4 d = cp_ld<AUX>(rs, (v + 3 * W) * 16u, soff);
        acc += cp_hsum(a) + cp_hsum(b) + cp_hsum(c) + cp_hsum(d);
    }
    for (; v < nvec; v += W) {
        acc += cp_hsum(cp_ld<AUX>(rs, v * 16u, soff));
    }
    acc = cp_wave_sum(acc);
    if (lane == 0) out[row] = acc;
}

// Same shape with plain (flat/global) loads: the reference the buffer path must match in both
// checksum and bandwidth.
__device__ __forceinline__ void cp_stream_rows_flat(const uint8_t* __restrict__ base, uint32_t row_bytes,
                                                    uint32_t rows, uint32_t* __restrict__ out) {
    const uint32_t W = warpSize;
    const uint32_t lane = threadIdx.x % W;
    const uint32_t row = __builtin_amdgcn_readfirstlane(blockIdx.x * (blockDim.x / W) + threadIdx.x / W);
    if (row >= rows) return;
    const cp_u32x4* __restrict__ p = (const cp_u32x4*)(base + (size_t)row * row_bytes);
    const uint32_t nvec = row_bytes >> 4;
    uint32_t acc = 0;
    uint32_t v = lane;
    for (; v + 3 * W < nvec; v += 4 * W) {
        cp_u32x4 a = p[v];
        cp_u32x4 b = p[v + W];
        cp_u32x4 c = p[v + 2 * W];
        cp_u32x4 d = p[v + 3 * W];
        acc += cp_hsum(a) + cp_hsum(b) + cp_hsum(c) + cp_hsum(d);
    }
    for (; v < nvec; v += W) acc += cp_hsum(p[v]);
    acc = cp_wave_sum(acc);
    if (lane == 0) out[row] = acc;
}

#define CP_STREAM_KERNEL(N)                                                                        \
    extern "C" __global__ void __launch_bounds__(256)                                              \
        cache_probe_stream_a##N(const uint8_t* base, uint32_t row_bytes, uint32_t rows,            \
                                uint32_t w3, uint32_t* __restrict__ out) {                         \
        cp_stream_rows_buf<N>(base, row_bytes, rows, w3, out);                                     \
    }

CP_STREAM_KERNEL(0)
CP_STREAM_KERNEL(1)
CP_STREAM_KERNEL(2)
CP_STREAM_KERNEL(3)
CP_STREAM_KERNEL(4)
CP_STREAM_KERNEL(5)
CP_STREAM_KERNEL(6)

extern "C" __global__ void __launch_bounds__(256)
    cache_probe_stream_flat(const uint8_t* base, uint32_t row_bytes, uint32_t rows, uint32_t w3,
                            uint32_t* __restrict__ out) {
    (void)w3;
    cp_stream_rows_flat(base, row_bytes, rows, out);
}

// Fill: word i of the buffer = cp_hash(i + word0, seed). The host recomputes the same hash.
__device__ __forceinline__ uint32_t cp_hash(uint32_t i, uint32_t seed) {
    uint32_t x = i * 0x9E3779B1u + seed;
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}

extern "C" __global__ void __launch_bounds__(256)
    cache_probe_fill(uint32_t* __restrict__ p, uint32_t n_words, uint32_t seed) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_words; i += gridDim.x * blockDim.x) {
        p[i] = cp_hash(i, seed);
    }
}
