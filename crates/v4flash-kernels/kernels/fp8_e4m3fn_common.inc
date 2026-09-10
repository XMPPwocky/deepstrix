// fp8_e4m3fn_common.inc — the E4M3FN value table, the nearest-code search
// and the block-exponent expression shared by every kernel that produces
// or consumes the compressed-KV rows. Included (not compiled) by
// fp8_e4m3fn.hip (the in-place round-trip quantiser, the raw-KV
// kv_post_fused tail) and comp_kv_fp8.hip (packed FP8 storage of the
// compressed KV: producer, gather/expand consumers).
//
// One implementation on purpose: the packed format is only bit-identical
// to the f16 cache it replaces if the code index, the exponent and the
// f32 -> f16 rounding are computed by exactly the same expressions. Do
// not "simplify" any of them:
//   * `e = ceil(log2f(amax / 448))` — device log2f is not correctly
//     rounded; a re-derived expression can flip e at exact powers of two
//     (see indexer_qat.hip:90-92).
//   * the nearest-code search and its tie-break mirror ds4's
//     `dsv4_e4m3fn_dequant_cpu` (ds4.c:1605) bit-for-bit.
//   * the f16 store is `__float2half_rn` on the f32 product `value * 2^e`
//     (what f16_roundtrip.hip does); the product is exact in f32 (a 4-bit
//     significand times a power of two), so the cast is the only rounding.

#pragma once

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

// Nearest E4M3FN magnitude index for |x| (x already scaled by 2^-e and
// clamped to [0, 448]). Returns an index in [0, 126]; index 127 is never
// produced (E4M3FN 0x7F is NaN and the table slot is unused).
__device__ __forceinline__ int e4m3fn_nearest_index(float ax) {
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
    return best;
}

// In-place round-trip quantise of one scaled value (the historical entry
// point). Sign convention: `x < 0` selects -1, so -0.0f and +0.0f both map
// to +0.0f, while a negative value that snaps to index 0 becomes -0.0f.
// The packed format reproduces exactly this (sign bit = `x < 0`).
__device__ __forceinline__ float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const int best = e4m3fn_nearest_index(fabsf(x));
    return sign * c_e4m3fn_values[best];
}

// Block exponent for a 64-element block with absolute max `amax`,
// VERBATIM from fp8_e4m3fn_quantize (floor 1e-4 included).
__device__ __forceinline__ int fp8_kv_block_exp(float amax) {
    if (amax < 1.0e-4f) amax = 1.0e-4f;
    // scale = ldexpf(1, ceil(log2(amax/448)))
    const float l = log2f(amax / 448.0f);
    return (int)ceilf(l);
}

// Arithmetic decode of a 7-bit E4M3FN magnitude index (no table lookup:
// the __constant__ table serialises on divergent indices, and the gather
// has 448 divergent indices per row). exp==0 is subnormal: mant/512.
// Asserted equal to c_e4m3fn_values[idx] for all idx in [0, 126] by
// tests/fp8_kv_format.rs (device kernel fp8_kv_table_check).
__device__ __forceinline__ float e4m3fn_decode_index(int idx) {
    const int ex = idx >> 3;
    const int mant = idx & 7;
    if (ex == 0) {
        return (float)mant * (1.0f / 512.0f);
    }
    return ldexpf((float)(8 + mant) * (1.0f / 8.0f), ex - 7);
}

// The stored f16 for one packed (code, e): exactly what the f16 cache held
// (fp8_e4m3fn_quantize -> f16_roundtrip -> comp_kv_append).
//
// The sign is applied to the f16 BITS, not to the f32 product. Reason: the
// compiler contracts `cvt_f16(a * b)` into a mixed-precision FMA with a
// +0.0 addend (v_fma_mix), and (-0.0 * 2^e) + 0.0 = +0.0 — the sign of a
// zero product is lost. The historical chain never contracted (the f32
// product was stored to memory by fp8_e4m3fn_quantize and converted by a
// separate kernel), so it stores 0x8000 for a negative value that snapped
// to code 0. Bitwise sign reproduces that regardless of contraction;
// for nonzero values round-to-nearest is symmetric so the bits agree.
__device__ __forceinline__ unsigned short fp8_kv_expand_half_bits(unsigned int code, int e) {
    const float mag = e4m3fn_decode_index((int)(code & 0x7Fu)) * ldexpf(1.0f, e);
    const unsigned short h = __half_as_ushort(__float2half_rn(mag));
    return (unsigned short)(h | ((code & 0x80u) << 8));
}
