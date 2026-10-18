// act_quant_fp8.inc -- the DeepSeek-V4.1-Flash-1017 reference's FP8 activation quantization
// (inference/kernel.py `act_quant`, block 32, scale_fmt "ue8m0") as device functions, for the
// attention output `o` before `wo_a` (`V41_ATTN_O_FP8`). Included (not compiled) by every kernel
// whose epilogue produces wo_a's input: rope_tail.hip (rope_inv_quant_fp8, rope_tail[_batched]_bf16),
// fp8_wo_a.hip (act_quant_fp8_wave, q8_0_quantize_o8_wave), attn_dec_g3.hip, attn_dec_g3o.hip.
//
// Bit-exact with the CPU emulation `crate::act_quant_fp8` (src/act_quant_fp8.rs), which documents
// each step against kernel.py / model.py; the expressions here are those, one for one:
//   * aq_bf16: f32 -> bf16 round to nearest even, kept in an f32 register. The reference's `o` is
//     bf16 at both of its rounding points: sparse_attn stores the f32 accumulator as bf16
//     (kernel.py:386-387), and apply_rotary_emb(o[..., -64:], inverse) computes the rotation in f32
//     from those bf16 values and writes bf16 back (model.py:396-405). act_quant then reads the bf16
//     values (its amax AND its codes). Every caller rounds at the same two points: aq_bf16 on the
//     attention output, then (rope dims only) aq_bf16 of the inverse rotation of the bf16 values;
//   * aq_scale_exp: e = exp_field(max(amax, 1e-4) * f32(1/448)) - 127 + (mantissa != 0)
//     (kernel.py:76-78, fast_round_scale / fast_log2_ceil; the block scale is s = 2^e);
//   * aq_e4m3_code: x * 2^-e (exact: == x / s), clamp to +-448, round to nearest even onto the
//     e4m3fn grid WITH subnormals, sign kept (-0 -> 0x80) -- arithmetic, no table and no
//     hardware cvt (whose saturation / denormal modes are not pinned here): quantum 2^(E-3) for
//     |q| in [2^E, 2^(E+1)), 2^-9 below 2^-6; |q| * 2^(3-E) is exact, so rintf (v_rndne_f32) is
//     the only rounding;
//   * aq_e4m3_value: code -> f32 (exact), and code * 2^e is exact (<= 4 significant bits).
// Bytes: code = sign << 7 | exponent << 3 | mantissa (float8_e4m3fn); 0x7F / 0xFF never produced.

#pragma once

// Device assert: the invariant named at the call site is broken -> trap (the launch fails with
// the kernel's name instead of writing a wrong code). No correction path.
#define AQ_ASSERT(cond) do { if (!(cond)) __builtin_trap(); } while (0)

__device__ __forceinline__ float aq_pow2(int e) {   // 2^e, e in the f32 normal range
    return __uint_as_float((unsigned)(e + 127) << 23);
}

// f32 -> bf16 (round to nearest, ties to even), returned as the f32 it denotes. Finite inputs
// only (attention outputs are convex combinations of f16 V rows); |x| above bf16 max -> inf, which
// aq_scale_exp's assert then rejects.
__device__ __forceinline__ float aq_bf16(float x) {
    const unsigned u = __float_as_uint(x);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

__device__ __forceinline__ int aq_scale_exp(float amax) {
    // invariant: the block amax is finite (a NaN / inf here is a broken attention output; the
    // scale exponent of a finite amax is <= 120, the e8m0 / f32 normal range)
    AQ_ASSERT(amax <= 3.4028235e38f);
    const float a = fmaxf(amax, 1.0e-4f);
    const unsigned bits = __float_as_uint(a * (1.0f / 448.0f));
    return (int)((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
}

__device__ __forceinline__ unsigned aq_e4m3_code(float v, int e) {
    float q = v * aq_pow2(-e);
    q = fminf(fmaxf(q, -448.0f), 448.0f);
    const float a = fabsf(q);
    int eu = (int)((__float_as_uint(a) >> 23) & 0xFFu) - 127;
    eu = eu < -6 ? -6 : eu;
    const unsigned m = (unsigned)rintf(a * aq_pow2(3 - eu));   // 0..16
    unsigned code;
    if (eu == -6 && m < 8u) code = m;                           // subnormal (exponent field 0)
    else if (m == 16u) code = (unsigned)(eu + 8) << 3;          // rounded into the next binade
    else code = ((unsigned)(eu + 7) << 3) | (m - 8u);
    return code | ((__float_as_uint(q) >> 24) & 0x80u);
}

__device__ __forceinline__ float aq_e4m3_value(unsigned code) {
    const unsigned ex = (code >> 3) & 0xFu, m = code & 7u;
    const float mag = ex == 0u ? (float)m * (1.0f / 512.0f) : (float)(8u + m) * aq_pow2((int)ex - 10);
    return __uint_as_float(__float_as_uint(mag) | ((code & 0x80u) << 24));
}

// e8m0 byte -> f32 (2^(b - 127); 0x00 = 2^-127, an f32 subnormal). Weight scales of the
// checkpoint's fp8 wo_a; 0xFF (NaN) is rejected at load.
__device__ __forceinline__ float aq_e8m0_value(unsigned b) {
    return b == 0u ? __uint_as_float(0x00400000u) : __uint_as_float(b << 23);
}
