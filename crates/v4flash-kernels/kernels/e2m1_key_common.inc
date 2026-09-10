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

// Packed row geometry (index_kv_e2m1.hip is the authority; the score
// kernels include this file for the loads below).
#define E2M1_KEY_ROW_BYTES  80u
#define E2M1_KEY_OFF_EXP    64u

// The stored f16 for one packed nibble (sign<<3 | idx) at block scale 2^e
// (`scale` = ldexpf(1.0f, e) precomputed once per block): exactly what the
// f16 key cache held (indexer_qat -> f16_roundtrip -> comp_kv_append).
// Sign applied to the bits (see the header). ONE implementation: every
// consumer goes through this function.
__device__ __forceinline__ unsigned short e2m1_key_expand_half_bits_scaled(unsigned int nib, float scale) {
    const float mag = e2m1fn_value((int)(nib & 7u)) * scale;
    const unsigned short h = __half_as_ushort(__float2half_rn(mag));
    return (unsigned short)(h | ((nib & 8u) << 12));
}
__device__ __forceinline__ unsigned short e2m1_key_expand_half_bits(unsigned int nib, int e) {
    return e2m1_key_expand_half_bits_scaled(nib, ldexpf(1.0f, e));
}

// Eight consecutive nibbles (one u32, element 8j+i in nibble i) -> eight
// f16 packed as a uint4 (two per u32, low element in the low half).
//
// FAST PATH (e in [-13, 13], i.e. every result is a NORMAL f16, which is
// every real block: RMS-normed keys give e in roughly [-4, 4]): the f16 of
// an E2M1 magnitude m x 2^e has a ZERO low byte and a high byte of
// hb[m] + 4e, hb = {0, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46} (m = 0 stays
// 0), plus the sign in bit 7. So the whole expansion is byte permutes:
// v_perm_b32 picks hb[m] for four nibbles at once from a two-register
// table pre-biased by the block exponent (built at e = -13 so no byte
// underflows, then 4(e+13) added per byte: max 0x12 + 0x68 = 0x7A, no
// carry into bit 7), two ORs add the signs, four more perms interleave the
// high bytes with zero low bytes. ~12 VALU ops per 8 nibbles, no float
// math. Exactness: bit-identical to the general path by construction (the
// f16 value is representable, so round-to-nearest is the identity) and
// checked for every (nibble, position, e) by tests/e2m1_keys_format.rs.
// GENERAL PATH otherwise (subnormal / overflow results): the f32 product
// through __float2half_rn, sign on the bits.
__device__ __forceinline__ uint4 e2m1_key_expand8(unsigned int w, int e) {
    uint4 r;
    if (e >= -13 && e <= 13) {
        const unsigned int k = (unsigned int)(e + 13);
        const unsigned int tab_lo = 0x0A080400u + k * 0x04040400u;   // hb[0..3] at e (byte 0 = m=0 stays 0)
        const unsigned int tab_hi = 0x12100E0Cu + k * 0x04040404u;   // hb[4..7] at e
        const unsigned int sel_even = w & 0x07070707u;               // magnitudes of nibbles 0,2,4,6
        const unsigned int sel_odd  = (w >> 4) & 0x07070707u;        // magnitudes of nibbles 1,3,5,7
        const unsigned int hi_even = __builtin_amdgcn_perm(tab_hi, tab_lo, sel_even) | ((w & 0x08080808u) << 4);
        const unsigned int hi_odd  = __builtin_amdgcn_perm(tab_hi, tab_lo, sel_odd)  | (w & 0x80808080u);
        // out byte layout: [0]=0x00 [1]=hi_even.b[k] [2]=0x00 [3]=hi_odd.b[k]
        r.x = __builtin_amdgcn_perm(hi_odd, hi_even, 0x040C000Cu);
        r.y = __builtin_amdgcn_perm(hi_odd, hi_even, 0x050C010Cu);
        r.z = __builtin_amdgcn_perm(hi_odd, hi_even, 0x060C020Cu);
        r.w = __builtin_amdgcn_perm(hi_odd, hi_even, 0x070C030Cu);
        return r;
    }
    const float scale = ldexpf(1.0f, e);
    r.x = (unsigned int)e2m1_key_expand_half_bits_scaled((w      ) & 15u, scale)
        | ((unsigned int)e2m1_key_expand_half_bits_scaled((w >>  4) & 15u, scale) << 16);
    r.y = (unsigned int)e2m1_key_expand_half_bits_scaled((w >>  8) & 15u, scale)
        | ((unsigned int)e2m1_key_expand_half_bits_scaled((w >> 12) & 15u, scale) << 16);
    r.z = (unsigned int)e2m1_key_expand_half_bits_scaled((w >> 16) & 15u, scale)
        | ((unsigned int)e2m1_key_expand_half_bits_scaled((w >> 20) & 15u, scale) << 16);
    r.w = (unsigned int)e2m1_key_expand_half_bits_scaled((w >> 24) & 15u, scale)
        | ((unsigned int)e2m1_key_expand_half_bits_scaled((w >> 28) & 15u, scale) << 16);
    return r;
}

// The WMMA B-fragment for a packed key row: elements [k0, k0 + 8) (k0 a
// multiple of 8, so inside one 32-element block) as eight f16.
typedef _Float16 e2m1_half8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ e2m1_half8 e2m1_key_b_frag(const unsigned char * __restrict__ row, unsigned int k0) {
    const unsigned int w = *reinterpret_cast<const unsigned int *>(row + (k0 >> 1));
    const int e = (int)(signed char)row[E2M1_KEY_OFF_EXP + (k0 >> 5)];
    const uint4 v = e2m1_key_expand8(w, e);
    return __builtin_bit_cast(e2m1_half8, v);
}
