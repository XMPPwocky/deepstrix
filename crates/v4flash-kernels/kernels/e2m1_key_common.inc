// e2m1_key_common.inc — the E2M1 (FP4) value table, the nearest-code search
// and the block-exponent expression shared by indexer_qat.hip (the QAT
// round trip, ds4-faithful) and index_kv_e2m1.hip (packed E2M1 storage of
// the indexer key cache: producer, expand). Included, not compiled.
//
// One implementation on purpose (see fp8_e4m3fn_common.inc for the same
// rule on the compressed KV): the packed format is bit-identical to the
// f16 key cache only if the code, the exponent and the f32 -> f16 rounding
// come from exactly these expressions.
//   * exponent: `t = amax / 6; frexpf(t) -> (m, e); cexp = (m == 0.5f) ? e - 1 : e`
//     with amax floored at 6 x 2^-126 — VERBATIM from indexer_qat (frexpf is
//     exact; device log2f is not correctly rounded).
//   * the f16 store is __float2half_rn on the f32 product `m * 2^e`, exact
//     in f32 (a 2-bit significand times a power of two); the sign goes on
//     the f16 BITS (the compiler contracts cvt(a*b) into v_fma_mix with a
//     +0 addend, which turns -0.0 into +0.0; the historical chain stored
//     the QAT'd -0.0 through memory as 0x8000).

#pragma once

#include <hip/hip_fp16.h>

__device__ static float e2m1fn_value(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

// Nearest E2M1 value with ties broken toward the even mantissa index
// (matches ds4's dsv4_e2m1fn_dequant_cpu exactly).
__device__ static float e2m1fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - e2m1fn_value(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - e2m1fn_value(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * e2m1fn_value(best);
}

// Block exponent for a 32-element block with absolute max `amax`,
// VERBATIM from indexer_qat (floor 6 x 2^-126 included).
__device__ __forceinline__ int e2m1_block_exp(float amax_in) {
    const float amax = fmaxf(amax_in, 7.052966104933725e-38f);
    // scale = 2^ceil(log2(amax / 6)). Computed exactly via frexpf instead
    // of log2f (device log2f is not correctly rounded; frexpf gives the
    // mathematically exact ceil for every finite input).
    const float t = amax / 6.0f;
    int e;
    const float m = frexpf(t, &e);      // t = m * 2^e, m in [0.5, 1)
    return (m == 0.5f) ? e - 1 : e;
}

// Nearest E2M1 magnitude index for |x| (ties to the even index), the
// search inside e2m1fn_dequant.
__device__ __forceinline__ int e2m1fn_nearest_index(float ax_in) {
    const float ax = fminf(ax_in, 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - e2m1fn_value(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - e2m1fn_value(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return best;
}

// The stored f16 for one packed nibble (sign<<3 | idx) at block exponent e:
// exactly what the f16 key cache held (indexer_qat -> f16_roundtrip ->
// comp_kv_append). Sign applied to the bits (see the header).
__device__ __forceinline__ unsigned short e2m1_key_expand_half_bits(unsigned int nib, int e) {
    const float mag = e2m1fn_value((int)(nib & 7u)) * ldexpf(1.0f, e);
    const unsigned short h = __half_as_ushort(__float2half_rn(mag));
    return (unsigned short)(h | ((nib & 8u) << 12));
}
