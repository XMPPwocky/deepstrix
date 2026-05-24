// fp8_e4m3fn_quantize — in-place block-of-64 E4M3FN round-trip over the
// non-RoPE portion of a head_dim row. Mirrors ds4's
// `dsv4_fp8_kv_quantize_row_inplace_cpu` (ds4.c:1635) and its lookup-based
// `dsv4_e4m3fn_dequant_cpu` (ds4.c:1605) bit-for-bit.
//
// For V4 Flash: only the main attention's compressor invokes this
// (head_dim=512, n_rot=64, n_nope=448 → 7 blocks of 64 elements each).
// The indexer compressor (head_dim=128) skips this op.
//
// One workgroup per block. block.x = 64 threads. Grid.x = n_nope / 64.

#include <hip/hip_runtime.h>

// E4M3FN value table: 16 exponents × 8 mantissas = 128 indices (last
// unused). Generated to match dsv4_e4m3fn_value_cpu(i) for i ∈ [0, 127].
__constant__ float c_e4m3fn_values[128] = {
    // exp=0 (subnormal): mant * 1/512
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f,
    0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    // exp=1..15: (1 + mant/8) * exp_scale[exp]
    // exp_scale = {0, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 256}
    // exp=1: 1/64 .. 1/64 * 1.875
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f,
    0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    // exp=2: 1/32
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f,
    0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    // exp=3: 1/16
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f,
    0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    // exp=4: 1/8
    0.125f, 0.140625f, 0.15625f, 0.171875f,
    0.1875f, 0.203125f, 0.21875f, 0.234375f,
    // exp=5: 1/4
    0.25f, 0.28125f, 0.3125f, 0.34375f,
    0.375f, 0.40625f, 0.4375f, 0.46875f,
    // exp=6: 1/2
    0.5f, 0.5625f, 0.625f, 0.6875f,
    0.75f, 0.8125f, 0.875f, 0.9375f,
    // exp=7: 1
    1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f,
    // exp=8: 2
    2.0f, 2.25f, 2.5f, 2.75f,
    3.0f, 3.25f, 3.5f, 3.75f,
    // exp=9: 4
    4.0f, 4.5f, 5.0f, 5.5f,
    6.0f, 6.5f, 7.0f, 7.5f,
    // exp=10: 8
    8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f,
    // exp=11: 16
    16.0f, 18.0f, 20.0f, 22.0f,
    24.0f, 26.0f, 28.0f, 30.0f,
    // exp=12: 32
    32.0f, 36.0f, 40.0f, 44.0f,
    48.0f, 52.0f, 56.0f, 60.0f,
    // exp=13: 64
    64.0f, 72.0f, 80.0f, 88.0f,
    96.0f, 104.0f, 112.0f, 120.0f,
    // exp=14: 128
    128.0f, 144.0f, 160.0f, 176.0f,
    192.0f, 208.0f, 224.0f, 240.0f,
    // exp=15: 256
    256.0f, 288.0f, 320.0f, 352.0f,
    384.0f, 416.0f, 448.0f, /* unused: */ 0.0f,
};

__device__ __forceinline__ float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabsf(x);
    if (ax > 448.0f) ax = 448.0f;

    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (c_e4m3fn_values[mid] <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - c_e4m3fn_values[best]);
        const float next_diff = fabsf(ax - c_e4m3fn_values[best + 1]);
        const bool tie_pick_next =
            next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0;
        if (next_diff < best_diff || tie_pick_next) {
            best++;
        }
    }

    return sign * c_e4m3fn_values[best];
}

extern "C" __global__ void fp8_e4m3fn_quantize(
        float        * __restrict__ x,
        unsigned int   n_nope) {
    const unsigned int block_idx = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int off = block_idx * 64u;
    if (off >= n_nope) return;

    __shared__ float partial[64];
    __shared__ float scale_s;

    const float v = x[off + tid];
    const float av = fabsf(v);
    partial[tid] = av;
    __syncthreads();

    for (unsigned int stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = partial[tid + stride];
            if (other > partial[tid]) partial[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float amax = partial[0];
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        // scale = ldexpf(1, ceil(log2(amax/448)))
        const float l = log2f(amax / 448.0f);
        const int e = (int)ceilf(l);
        scale_s = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float scale = scale_s;

    float vs = v / scale;
    if (vs > 448.0f) vs = 448.0f;
    if (vs < -448.0f) vs = -448.0f;
    x[off + tid] = dsv4_e4m3fn_dequant(vs) * scale;
}
