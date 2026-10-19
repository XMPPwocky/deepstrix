//! The DeepSeek-V4.1-Flash-1017 reference's FP8 activation quantization
//! (`inference/kernel.py` `act_quant`) on the CPU, bit for bit, and the
//! FP8 x FP8 `wo_a` product it feeds (`V41_ATTN_O_FP8`).
//!
//! 1017's `model.py` rounds the attention output `o` through FP8 right before
//! the block-diagonal `wo_a` einsum, in the main attention (`Attention.forward`,
//! model.py:786-787) and in the DSpark drafter block (`DSparkAttention.forward`,
//! model.py:1072-1073); both calls are new in 1017 (`diff` against the previous
//! checkpoint's model.py shows exactly these two lines added):
//!
//! ```text
//! o = o.view(bsz, seqlen, n_local_groups, -1)                  # [.., 8, 4096], bf16
//! act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)   # in place
//! ```
//!
//! with `fp8_block_size = 32`, `scale_fmt = "ue8m0"`, `scale_dtype =
//! torch.float8_e8m0fnu` (model.py:27-30, module globals; `ModelArgs` carries no
//! override). 32 divides the head dim (512), so a block is 32 consecutive dims
//! of ONE head: the engine's flat `heads` row `[64 x 512]` has exactly the
//! reference's blocks, at the same offsets as its Q8_0 blocks.
//!
//! Per 32-block, `act_quant_kernel(N, 32, BF16, FP8, FE8M0, round_scale = True,
//! inplace = True)` (kernel.py:41-95) computes, in f32 (`compute_dtype`):
//!
//! 1. `amax = max_j |x_j|` (kernel.py:74 `reduce_absmax`; exact: a max), then
//!    `amax = max(amax, 1e-4)` (kernel.py:76; `1e-4` is an f32 immediate);
//! 2. `s = fast_round_scale(amax, 1/448)` (kernel.py:78, 36-37) =
//!    `fast_pow2(fast_log2_ceil(amax * fp8_max_inv))`: the f32 product
//!    `t = amax * f32(1/448)`, `e = exponent_field(t) - 127 + (mantissa(t) != 0)`
//!    (kernel.py:22-27: ceil(log2 t) for a normal t; t >= 1e-4/448 is normal),
//!    `s = f32 with bits (e + 127) << 23` (kernel.py:30-33) = 2^e;
//! 3. `y_j = bf16(f32(e4m3(clamp(x_j / s, -448, 448))) * s)` (kernel.py:83-86):
//!    `x_j / s` is exact (s is a power of two and the quotient cannot underflow
//!    into a range the e4m3 cast distinguishes), the clamp never binds (s >=
//!    amax / 448, see `scale_exp`), the cast is round-to-nearest-even with
//!    subnormals (e4m3fn: bias 7, max 448, no infinities, 0x7F/0xFF NaN), and
//!    `code * s` is a <= 4-significant-bit value times a power of two: exact in
//!    f32 AND in bf16, so the bf16 store (`out_dtype = in_dtype`, kernel.py:52)
//!    does not round. Signs survive every step: -0.0 stays -0.0 and a negative
//!    value that rounds to zero becomes -0.0 (code 0x80).
//!
//! `S` (kernel.py:91, the e8m0 byte `e + 127`) is written but the in-place call
//! discards it; the FP8-native `wo_a` path keeps it (as the f32 `2^e`) instead
//! of re-deriving it.
//!
//! The reference's `o` is bf16 at this point, rounded at two points the engine
//! reproduces ([`bf16_rne`]; device `aq_bf16`): `sparse_attn` stores its f32
//! accumulator as bf16 (kernel.py:386-387, `o: T.Tensor[..., BF16]`), and
//! `apply_rotary_emb(o[..., -64:], freqs_cis, True)` (model.py:392-406) rotates the
//! bf16 values in f32 (`x.float()`, complex64 product) and writes bf16 back
//! (`y.copy_(x)`). act_quant then reads the bf16 `o` (its amax AND its codes).
//! Skipping the bf16 point is not a small change: f32 values just past an e4m3
//! midpoint that bf16 rounds ONTO it round the other way (ties to even), and an
//! f32 amax in (448 * 2^k, 449 * 2^k] that bf16 rounds to 448 * 2^k halves the
//! block scale (`tests::bf16_points_*`). The block functions below are exact on
//! their input; `*_bf16` round to bf16 first (the reference's act_quant input).
//! (The window-KV `act_quant` -- `kernels/fp4_kv_quant.hip`
//! `fp8_act_quant_inplace`, `kv_rms_rope_fp8` -- quantizes f32 without the bf16
//! points: production numerics, unchanged here.)
//!
//! After the rounding the reference multiplies by `wo_a` dequantized to bf16
//! (`convert.py` / `Linear` with `dtype = bf16`: `e4m3 * e8m0` per 32x32 block,
//! exact in bf16) in a bf16 einsum (f32 accumulation, bf16 output). With the
//! knob on, the dGPU keeps BOTH operands as e4m3 codes + power-of-two scales and
//! accumulates the exact products in f32 ([`fp8_grouped_matvec_ref`]), i.e.
//! the reference's sum up to summation order (and without its bf16 output
//! rounding, as everywhere in the engine).

/// `fp8_block_size` (model.py:27): activations per scale.
pub const ACT_QUANT_BLOCK: usize = 32;
/// `fp8_max` (kernel.py:45).
pub const E4M3_MAX: f32 = 448.0;
/// The amax floor (kernel.py:76 `T.max(amax, 1e-4)`).
pub const AMAX_FLOOR: f32 = 1.0e-4;
/// `fp8_max_inv` (kernel.py:46: `1 / fp8_max`, a Python double, an f32 immediate
/// in the kernel).
pub const E4M3_MAX_INV: f32 = (1.0f64 / 448.0) as f32;

/// The block scale's exponent `e` (`s = 2^e`) for a block whose `max |x|` is
/// `amax` (steps 1-2 of the module docs). `amax` must be finite and >= 0.
///
/// `e = ceil(log2(max(amax, 1e-4) / 448))` exactly: for every finite f32 amax,
/// the reference's `amax * f32(1/448)` and the correctly rounded `amax / 448`
/// round to the same side of every power of two (`tests::scale_matches_exact_ceil`
/// sweeps every bf16 and the f32 neighbourhoods of each boundary), so the
/// device kernels may use either.
pub fn scale_exp(amax: f32) -> i32 {
    assert!(amax.is_finite() && amax >= 0.0, "act_quant: block amax {amax} is not a finite magnitude");
    let a = amax.max(AMAX_FLOOR);
    let t = a * E4M3_MAX_INV;
    let bits = t.to_bits();
    let exp_field = ((bits >> 23) & 0xFF) as i32;
    // t >= 1e-4 / 448 > 2^-126: never zero or subnormal, so the bit trick is ceil(log2 t).
    assert!(exp_field != 0, "act_quant: amax * 1/448 = {t:e} is subnormal (floor not applied?)");
    exp_field - 127 + i32::from(bits & 0x7F_FFFF != 0)
}

/// `2^e` as an f32 (kernel.py:30-33 `fast_pow2`). `e` in the normal range.
pub fn pow2(e: i32) -> f32 {
    assert!((-126..=127).contains(&e), "act_quant: scale exponent {e} outside the f32 normal range");
    f32::from_bits(((e + 127) as u32) << 23)
}

/// The e8m0 byte the reference stores as the block scale (`float8_e8m0fnu`).
pub fn e8m0_byte(e: i32) -> u8 {
    assert!((-127..=127).contains(&e), "act_quant: scale exponent {e} has no e8m0 code");
    (e + 127) as u8
}

/// `float8_e8m0fnu` byte -> f32: 2^(b - 127); 0x00 = 2^-127 (an f32 subnormal).
/// 0xFF (NaN) never appears in a valid checkpoint scale (asserted).
pub fn e8m0_to_f32(b: u8) -> f32 {
    assert!(b != 0xFF, "e8m0 scale byte 0xFF is NaN");
    if b == 0 { f32::from_bits(0x0040_0000) } else { f32::from_bits((b as u32) << 23) }
}

/// e4m3fn code -> f32 (exact). 0x7F / 0xFF are NaN and never produced by
/// [`e4m3fn_encode`] (asserted).
pub fn e4m3fn_decode(code: u8) -> f32 {
    assert!(code & 0x7F != 0x7F, "e4m3fn code {code:#04x} is NaN");
    let e = ((code >> 3) & 0xF) as i32;
    let m = (code & 7) as f32;
    let mag = if e == 0 { m * (1.0 / 512.0) } else { (8.0 + m) * pow2(e - 10) };
    if code & 0x80 != 0 { -mag } else { mag }
}

/// f32 -> e4m3fn code, round to nearest, ties to even, subnormals kept: the
/// cast `T.Cast(FP8, clamp(x / s, -448, 448))` (kernel.py:85) and torch's
/// `.to(torch.float8_e4m3fn)`. `x` must lie in [-448, 448] (the clamp ran).
pub fn e4m3fn_encode(x: f32) -> u8 {
    assert!(x.is_finite() && x.abs() <= E4M3_MAX, "e4m3fn_encode: {x} outside the clamped range");
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    // The quantum: 2^-9 below 2^-6 (subnormals share the first binade's quantum),
    // else 2^(E - 3) for a in [2^E, 2^(E+1)). `a * 2^(3-E)` is exact, so
    // round_ties_even is the one rounding.
    let e_unb = (((a.to_bits() >> 23) & 0xFF) as i32 - 127).max(-6);
    let m = (a * pow2(3 - e_unb)).round_ties_even() as u32; // 0..=16
    let code = if e_unb == -6 && m < 8 {
        m // subnormal: exponent field 0, mantissa m
    } else if m == 16 {
        ((e_unb + 1 + 7) as u32) << 3 // rounded up into the next binade, mantissa 0
    } else {
        (((e_unb + 7) as u32) << 3) | (m - 8)
    };
    assert!(code <= 0x7E, "e4m3fn_encode: {x} encoded past 448 ({code:#04x})");
    sign | code as u8
}

/// One 32-block, the quantize form (`act_quant(x, 32, "ue8m0")`, inplace = False):
/// the e4m3 codes and the scale exponent `e` (`s = 2^e`).
pub fn act_quant_block(x: &[f32]) -> ([u8; ACT_QUANT_BLOCK], i32) {
    assert_eq!(x.len(), ACT_QUANT_BLOCK, "act_quant: a block is {ACT_QUANT_BLOCK} values");
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let e = scale_exp(amax);
    let inv = pow2(-e); // x / 2^e == x * 2^-e exactly (both one correctly rounded op)
    let mut codes = [0u8; ACT_QUANT_BLOCK];
    for (c, &v) in codes.iter_mut().zip(x) {
        let q = (v * inv).clamp(-E4M3_MAX, E4M3_MAX);
        *c = e4m3fn_encode(q);
    }
    (codes, e)
}

/// The in-place form (`act_quant(o, 32, "ue8m0", float8_e8m0fnu, True)`):
/// every 32-block of `x` replaced by `decode(code) * 2^e`. `x.len() % 32 == 0`.
pub fn act_quant_inplace(x: &mut [f32]) {
    assert!(x.len() % ACT_QUANT_BLOCK == 0, "act_quant: length {} not a multiple of {ACT_QUANT_BLOCK}", x.len());
    for blk in x.chunks_exact_mut(ACT_QUANT_BLOCK) {
        let (codes, e) = act_quant_block(blk);
        let s = pow2(e);
        for (v, &c) in blk.iter_mut().zip(&codes) {
            *v = e4m3fn_decode(c) * s;
        }
    }
}

/// [`act_quant_block`] over a whole row: codes into `codes`, per-block f32
/// scales `2^e` into `scales` (the layout the FP8 `wo_a` kernels read:
/// `heads_xq` bytes + `heads_xscale` f32).
pub fn act_quant_row(x: &[f32], codes: &mut [u8], scales: &mut [f32]) {
    assert!(x.len() % ACT_QUANT_BLOCK == 0 && codes.len() == x.len() && scales.len() == x.len() / ACT_QUANT_BLOCK,
        "act_quant_row: x {} / codes {} / scales {} do not tile", x.len(), codes.len(), scales.len());
    for ((xb, cb), s) in x.chunks_exact(ACT_QUANT_BLOCK).zip(codes.chunks_exact_mut(ACT_QUANT_BLOCK)).zip(scales.iter_mut()) {
        let (c, e) = act_quant_block(xb);
        cb.copy_from_slice(&c);
        *s = pow2(e);
    }
}

/// f32 -> bf16, round to nearest, ties to even (torch / tilelang's cast), as the f32 it denotes.
/// The device's `aq_bf16` (kernels/act_quant_fp8.inc), bit for bit: `(u + 0x7FFF + lsb) & !0xFFFF`.
pub fn bf16_rne(x: f32) -> f32 {
    assert!(x.is_finite(), "bf16_rne: {x} (the attention output is finite)");
    let u = x.to_bits();
    f32::from_bits(u.wrapping_add(0x7FFF + ((u >> 16) & 1)) & 0xFFFF_0000)
}

/// [`act_quant_row`] of the bf16 values of `x`: the reference's act_quant on its bf16 `o`
/// (`x` already holds the rope tail at its own rounding point; see the module docs).
pub fn act_quant_row_bf16(x: &[f32], codes: &mut [u8], scales: &mut [f32]) {
    let xb: Vec<f32> = x.iter().map(|&v| bf16_rne(v)).collect();
    act_quant_row(&xb, codes, scales)
}

/// [`act_quant_inplace`] of the bf16 values of `x`.
pub fn act_quant_inplace_bf16(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = bf16_rne(*v);
    }
    act_quant_inplace(x)
}

/// The device Q8_0 quantize of one 32-block (`q8_0_quantize_f32_wave`):
/// `d = amax / 127`, `id = 1 / d` (0 when d == 0), `q = clamp(lrintf(v * id))`.
/// The drafter's `V41_ATTN_O_FP8` path is `act_quant_inplace` then this.
pub fn q8_0_block_device(x: &[f32]) -> ([i8; 32], f32) {
    assert_eq!(x.len(), 32);
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut q = [0i8; 32];
    for (o, &v) in q.iter_mut().zip(x) {
        *o = ((v * id).round_ties_even() as i32).clamp(-128, 127) as i8;
    }
    (q, d)
}

/// The FP8 `wo_a` weight as the checkpoint ships it and the dGPU holds it:
/// `[out_dim, k]` e4m3 codes (row-major) followed by the `[out_dim / 32, k / 32]`
/// e8m0 block scales, both verbatim (one buffer; see `het::weights`).
pub fn fp8_weight_bytes(out_dim: usize, k: usize) -> usize {
    assert!(out_dim % 32 == 0 && k % 32 == 0);
    out_dim * k + (out_dim / 32) * (k / 32)
}

/// The exact FP8 x FP8 grouped product, the reference's `einsum("bsgd,grd->bsgr")`
/// on the dequantized operands before its bf16 output rounding: for output row
/// `r` of group `g = r / rank` and batch row `b`,
/// `out = sum_kb s_w[r/32][kb] * s_a[b][g, kb] * sum_{k in kb} dec(w[r][k]) * dec(a[b][g, k])`,
/// summed in f64 (every product is exact in f64; returns the sums and, per
/// output, `sum |term|` for the summation-order bound the GPU tests use).
#[allow(clippy::too_many_arguments)]
pub fn fp8_grouped_matvec_ref(
    w: &[u8],
    a_codes: &[u8],
    a_scales: &[f32],
    group_dim: usize,
    rank: usize,
    n_groups: usize,
    batch: usize,
) -> (Vec<f64>, Vec<f64>) {
    let out_dim = rank * n_groups;
    let kb_n = group_dim / 32;
    assert_eq!(w.len(), fp8_weight_bytes(out_dim, group_dim), "fp8 wo_a weight bytes");
    assert_eq!(a_codes.len(), batch * n_groups * group_dim, "fp8 activation codes");
    assert_eq!(a_scales.len(), batch * n_groups * kb_n, "fp8 activation scales");
    let (wq, ws) = w.split_at(out_dim * group_dim);
    let dec: Vec<f64> = (0..=255u16).map(|c| if c as u8 & 0x7F == 0x7F { f64::NAN } else { e4m3fn_decode(c as u8) as f64 }).collect();
    let mut out = vec![0f64; batch * out_dim];
    let mut mag = vec![0f64; batch * out_dim];
    for b in 0..batch {
        for r in 0..out_dim {
            let g = r / rank;
            let (mut s, mut m) = (0f64, 0f64);
            for kb in 0..kb_n {
                let sw = e8m0_to_f32(ws[(r / 32) * kb_n + kb]) as f64;
                let sa = a_scales[b * n_groups * kb_n + g * kb_n + kb] as f64;
                for k in kb * 32..kb * 32 + 32 {
                    let t = dec[wq[r * group_dim + k] as usize] * dec[a_codes[b * n_groups * group_dim + g * group_dim + k] as usize] * sw * sa;
                    s += t;
                    m += t.abs();
                }
            }
            out[b * out_dim + r] = s;
            mag[b * out_dim + r] = m;
        }
    }
    (out, mag)
}

/// The order-independent f32 summation bound for `n` terms (Higham 4.4):
/// any order of f32 additions of `n` exact terms errs by at most
/// `(n - 1) u / (1 - (n - 1) u) * sum |term|`, u = 2^-24. The FP8 products are
/// exact in f32 and the block scales are powers of two, so this bounds the GPU
/// `wo_a` against [`fp8_grouped_matvec_ref`] whatever its reduction order.
pub fn f32_sum_bound(n: usize, sum_abs: f64) -> f64 {
    let nu = (n.saturating_sub(1)) as f64 * f64::powi(2.0, -24);
    assert!(nu < 0.5, "f32_sum_bound: n = {n} too large for the bound");
    nu / (1.0 - nu) * sum_abs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference nearest e4m3fn magnitude by brute force: the closest of the
    /// 127 finite magnitudes, ties to the even code (the even mantissa).
    fn encode_brute(x: f32) -> u8 {
        let a = x.abs() as f64;
        let mut best = 0u8;
        let mut best_d = f64::INFINITY;
        for c in 0u8..=0x7E {
            let d = (e4m3fn_decode(c) as f64 - a).abs();
            if d < best_d || (d == best_d && c & 1 == 0) {
                best = c;
                best_d = d;
            }
        }
        if x.is_sign_negative() { best | 0x80 } else { best }
    }

    #[test]
    fn constants_match_the_reference() {
        assert_eq!(E4M3_MAX_INV, 1.0f32 / 448.0f32);
        assert_eq!(AMAX_FLOOR.to_bits(), (1e-4f64 as f32).to_bits());
    }

    #[test]
    fn decode_table_is_the_e4m3fn_spec() {
        assert_eq!(e4m3fn_decode(0x00), 0.0);
        assert!(e4m3fn_decode(0x80).is_sign_negative() && e4m3fn_decode(0x80) == 0.0);
        assert_eq!(e4m3fn_decode(0x01), 2f32.powi(-9)); // min subnormal
        assert_eq!(e4m3fn_decode(0x07), 7.0 * 2f32.powi(-9)); // max subnormal
        assert_eq!(e4m3fn_decode(0x08), 2f32.powi(-6)); // min normal
        assert_eq!(e4m3fn_decode(0x38), 1.0);
        assert_eq!(e4m3fn_decode(0x7E), 448.0);
        assert_eq!(e4m3fn_decode(0xFE), -448.0);
        // monotone over the finite codes
        for c in 1u8..=0x7E {
            assert!(e4m3fn_decode(c) > e4m3fn_decode(c - 1), "code {c:#x}");
        }
        // agrees with the checkpoint loader's decoder
        for c in 0u8..=255 {
            if c & 0x7F == 0x7F {
                continue;
            }
            assert_eq!(e4m3fn_decode(c).to_bits(), v4flash_core::hf_v41::e4m3_to_f32(c).to_bits(), "code {c:#x}");
        }
    }

    #[test]
    fn encode_round_trips_every_code() {
        for c in 0u8..=255 {
            if c & 0x7F == 0x7F {
                continue;
            }
            assert_eq!(e4m3fn_encode(e4m3fn_decode(c)), c, "code {c:#x}");
        }
    }

    #[test]
    fn encode_ties_to_even_at_every_midpoint() {
        for c in 0u8..0x7E {
            let lo = e4m3fn_decode(c) as f64;
            let hi = e4m3fn_decode(c + 1) as f64;
            let mid = ((lo + hi) / 2.0) as f32;
            assert_eq!(mid as f64, (lo + hi) / 2.0, "midpoint exact in f32");
            let even = if c & 1 == 0 { c } else { c + 1 };
            assert_eq!(e4m3fn_encode(mid), even, "tie between {c:#x} and {:#x}", c + 1);
            assert_eq!(e4m3fn_encode(-mid), even | 0x80);
            // one f32 ulp either side breaks the tie
            let below = f32::from_bits(mid.to_bits() - 1);
            let above = f32::from_bits(mid.to_bits() + 1);
            assert_eq!(e4m3fn_encode(below), c, "below the tie {c:#x}");
            assert_eq!(e4m3fn_encode(above), c + 1, "above the tie {c:#x}");
        }
    }

    #[test]
    fn encode_matches_brute_force() {
        // every f32 within 64 ulps of each code value and each midpoint, plus a
        // deterministic pseudo-random sweep of [0, 448]
        let mut probes: Vec<f32> = Vec::new();
        for c in 0u8..=0x7E {
            let v = e4m3fn_decode(c);
            let mid = if c < 0x7E { ((v as f64 + e4m3fn_decode(c + 1) as f64) / 2.0) as f32 } else { v };
            for base in [v, mid] {
                for d in -64i32..=64 {
                    let b = base.to_bits() as i64 + d as i64;
                    if b >= 0 {
                        let x = f32::from_bits(b as u32);
                        if x <= E4M3_MAX {
                            probes.push(x);
                        }
                    }
                }
            }
        }
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            probes.push((s >> 40) as f32 / (1u64 << 24) as f32 * E4M3_MAX);
        }
        // f32 subnormals and tiny normals all round to (signed) zero
        probes.extend([f32::from_bits(1), f32::MIN_POSITIVE, 2f32.powi(-10), 2f32.powi(-10) * 1.000_001]);
        for x in probes {
            assert_eq!(e4m3fn_encode(x), encode_brute(x), "x = {x:e} ({:#010x})", x.to_bits());
            assert_eq!(e4m3fn_encode(-x), encode_brute(-x), "x = -{x:e}");
        }
    }

    #[test]
    fn zero_and_negative_zero_keep_their_sign() {
        assert_eq!(e4m3fn_encode(0.0), 0x00);
        assert_eq!(e4m3fn_encode(-0.0), 0x80);
        // half the min subnormal ties to the even code 0, keeping the sign
        assert_eq!(e4m3fn_encode(2f32.powi(-10)), 0x00);
        assert_eq!(e4m3fn_encode(-2f32.powi(-10)), 0x80);
        let mut x = [0.0f32; 32];
        x[3] = -0.0;
        x[5] = 1.0;
        x[6] = -1.0e-30;
        act_quant_inplace(&mut x);
        assert!(x[3] == 0.0 && x[3].is_sign_negative(), "-0 stays -0");
        assert!(x[6] == 0.0 && x[6].is_sign_negative(), "a negative value that rounds to 0 becomes -0");
        assert!(x[0] == 0.0 && x[0].is_sign_positive());
        assert_eq!(x[5], 1.0);
    }

    /// `ceil(log2(max(amax, 1e-4) / 448))` in exact arithmetic (f64 division is
    /// exact enough: amax / 448 = amax / 7 / 64, and the comparison with the
    /// power of two is decided exactly by `amax` vs `448 * 2^e`).
    fn exact_ceil_exp(amax: f32) -> i32 {
        let a = amax.max(AMAX_FLOOR) as f64;
        // smallest e with 448 * 2^e >= a
        let mut e = (a / 448.0).log2().ceil() as i32;
        while 448.0 * 2f64.powi(e - 1) >= a {
            e -= 1;
        }
        while 448.0 * 2f64.powi(e) < a {
            e += 1;
        }
        e
    }

    #[test]
    fn scale_matches_exact_ceil() {
        // every finite positive bf16
        for hi in 0u32..0x7F80 {
            let a = f32::from_bits(hi << 16);
            assert_eq!(scale_exp(a), exact_ceil_exp(a), "bf16 amax {a:e}");
        }
        // the f32 neighbourhood of every boundary 448 * 2^k
        for k in -22..=119 {
            let b = (448.0f64 * 2f64.powi(k)) as f32;
            for d in -4096i64..=4096 {
                let a = f32::from_bits((b.to_bits() as i64 + d) as u32);
                if a.is_finite() {
                    assert_eq!(scale_exp(a), exact_ceil_exp(a), "f32 amax {a:e} (k {k}, d {d})");
                }
            }
        }
    }

    #[test]
    fn all_zero_block_takes_the_floor_scale() {
        let mut x = [0.0f32; 32];
        let (codes, e) = act_quant_block(&x);
        assert_eq!(e, -22, "2^ceil(log2(1e-4 / 448)) = 2^-22");
        assert!(codes.iter().all(|&c| c == 0));
        act_quant_inplace(&mut x);
        assert!(x.iter().all(|v| v.to_bits() == 0));
    }

    #[test]
    fn tiny_blocks_round_on_the_floor_scale_into_subnormals() {
        // amax < 1e-4: s = 2^-22 is fixed, so values land on the e4m3 subnormal grid
        // (quantum 2^-9 * 2^-22 = 2^-31) instead of being scaled up to 448.
        let mut x = [0.0f32; 32];
        let q = 2f32.powi(-31);
        x[0] = 3.0 * q; // exact subnormal code 3
        x[1] = 2.5 * q; // tie -> even code 2
        x[2] = 3.5 * q; // tie -> even code 4
        x[3] = 0.49 * q; // -> 0
        x[4] = -0.51 * q; // -> -code 1
        x[5] = 7.5 * q; // tie between max subnormal 7 and min normal 8 -> 8 (even)
        x[6] = 5.0e-5; // amax below the floor
        let (codes, e) = act_quant_block(&x);
        assert_eq!(e, -22);
        assert_eq!(&codes[..6], &[0x03, 0x02, 0x04, 0x00, 0x81, 0x08]);
        let mut y = x;
        act_quant_inplace(&mut y);
        assert_eq!(y[0], 3.0 * q);
        assert_eq!(y[1], 2.0 * q);
        assert_eq!(y[2], 4.0 * q);
        assert_eq!(y[4], -q);
        assert_eq!(y[5], 8.0 * q);
    }

    #[test]
    fn amax_at_and_above_the_448_boundaries() {
        for k in [-10i32, -1, 0, 1, 7] {
            let s = 2f32.powi(k);
            // amax exactly 448 * 2^k: scale 2^k, the max element is code 0x7E
            let mut x = [0.0f32; 32];
            x[0] = 448.0 * s;
            x[1] = -448.0 * s;
            x[2] = 1.0 * s;
            let (codes, e) = act_quant_block(&x);
            assert_eq!(e, k);
            assert_eq!((codes[0], codes[1], codes[2]), (0x7E, 0xFE, 0x38));
            // one ulp above: the scale doubles, 448 * 2^k -> 224 (code 0x76)
            x[0] = f32::from_bits((448.0 * s).to_bits() + 1);
            let (codes, e) = act_quant_block(&x);
            assert_eq!(e, k + 1);
            assert_eq!(codes[0], e4m3fn_encode(x[0] / pow2(k + 1)));
            assert_eq!(e4m3fn_decode(codes[1]), -224.0);
            // one ulp below: the scale stays 2^k and the max rounds up to 448
            x[0] = f32::from_bits((448.0 * s).to_bits() - 1);
            let (codes, e) = act_quant_block(&x);
            assert_eq!(e, k);
            assert_eq!(codes[0], 0x7E);
        }
    }

    #[test]
    fn mixed_magnitudes_lose_the_small_ones() {
        // amax 300: s = 1, so the e4m3 subnormal quantum is 2^-9 = 0.00195:
        // 0.0009 / 2^-9 = 0.46 -> 0; 0.001 -> 0.512 -> code 1; 0.003 -> 1.536 -> code 2
        let mut x = [0.0f32; 32];
        x[0] = 300.0;
        x[1] = 0.0009;
        x[2] = 0.003;
        x[3] = 17.3;
        x[4] = -250.0;
        x[5] = 0.001;
        let (codes, e) = act_quant_block(&x);
        assert_eq!(e, 0);
        assert_eq!(codes[1], 0x00);
        assert_eq!(codes[2], 0x02);
        assert_eq!(codes[5], 0x01);
        assert_eq!(e4m3fn_decode(codes[3]), 18.0); // 17.3 -> 16 + 2k grid: 16, 18
        assert_eq!(e4m3fn_decode(codes[4]), -256.0); // 250 between 240 and 256, closer to 256
        assert_eq!(e4m3fn_decode(codes[0]), 288.0); // 300 between 288 and 320
    }

    #[test]
    fn inplace_output_is_exact_in_bf16() {
        let mut s = 12345u64;
        let mut x = vec![0f32; 32 * 512];
        for (i, v) in x.iter_mut().enumerate() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
            let mag = 10f32.powi((i / 32 % 16) as i32 - 10);
            *v = u * mag;
        }
        act_quant_inplace(&mut x);
        for v in &x {
            assert_eq!(v.to_bits() & 0xFFFF, 0, "{v:e} is not a bf16 value");
        }
    }

    #[test]
    fn bf16_rne_rounds_to_nearest_even() {
        for &v in &[0.0f32, -0.0, 1.0, -3.5, 448.0, 65536.0, 2f32.powi(-100)] {
            assert_eq!(bf16_rne(v).to_bits(), v.to_bits(), "{v} is a bf16 value");
        }
        let one = 1.0f32;
        assert_eq!(bf16_rne(one + 2f32.powi(-8)), 1.0, "tie -> even (mantissa 0)");
        assert_eq!(bf16_rne(one + 3.0 * 2f32.powi(-8)), one + 2f32.powi(-6), "tie -> even (mantissa 2)");
        assert_eq!(bf16_rne(one + 2f32.powi(-8) + 2f32.powi(-20)), one + 2f32.powi(-7), "above the tie -> up");
        assert_eq!(bf16_rne(-(one + 2f32.powi(-8) + 2f32.powi(-20))), -(one + 2f32.powi(-7)));
        // a carry into the exponent
        assert_eq!(bf16_rne(2.0 - 2f32.powi(-10)), 2.0);
        // every output is a bf16 value, and bf16 values are fixed points
        let mut s = 99u64;
        for _ in 0..100_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = f32::from_bits((s as u32) & 0x7EFF_FFFF); // finite, below the bf16 overflow
            let r = bf16_rne(x);
            assert_eq!(r.to_bits() & 0xFFFF, 0);
            assert_eq!(bf16_rne(r), r);
            // half a bf16 ulp: relative 2^-8 for normals, 2^-134 absolute in the subnormal range
            assert!((r as f64 - x as f64).abs() <= ((x.abs() as f64) * 2f64.powi(-8)).max(2f64.powi(-134)), "{x:e} -> {r:e}");
        }
    }

    /// An f32 just past an e4m3 midpoint that bf16 rounds ONTO the midpoint: the reference (bf16
    /// first) ties to even, the f32 value would round the other way.
    #[test]
    fn bf16_points_midpoint_double_rounding() {
        let mut x = [0.0f32; 32];
        x[31] = 448.0; // pins s = 1
        x[0] = 1.0625 + 2f32.powi(-12); // 1.0625 = midpoint of 1.0 (0x38) and 1.125 (0x39)
        x[1] = -(1.0625 + 2f32.powi(-12));
        x[2] = 1.1875 - 2f32.powi(-12); // midpoint of 1.125 / 1.25 from below: bf16 -> 1.1875 -> even 1.25
        let (c_f32, e_f32) = act_quant_block(&x);
        let mut cb = [0u8; 32];
        let mut sb = [0f32; 1];
        act_quant_row_bf16(&x, &mut cb, &mut sb);
        assert_eq!(e_f32, 0);
        assert_eq!(sb[0], 1.0);
        assert_eq!((c_f32[0], c_f32[1], c_f32[2]), (0x39, 0xB9, 0x39), "f32: past / short of the midpoint");
        assert_eq!((cb[0], cb[1], cb[2]), (0x38, 0xB8, 0x3A), "bf16 first: onto the midpoint, ties to even");
    }

    /// An f32 amax in (448 * 2^k, 449 * 2^k] rounds to 448 * 2^k in bf16: the reference keeps the
    /// scale 2^k where the f32 amax would double it.
    #[test]
    fn bf16_points_amax_just_above_the_boundary() {
        for k in [-12i32, -3, 0, 2, 9] {
            let mut x = [0.0f32; 32];
            x[0] = 448.5 * 2f32.powi(k);
            x[1] = 1.0 * 2f32.powi(k);
            x[2] = f32::from_bits((448.0 * 2f32.powi(k)).to_bits() + 1);
            let (_, e_f32) = act_quant_block(&x);
            assert_eq!(e_f32, k + 1, "f32 amax above 448 * 2^{k} doubles the scale");
            let mut cb = [0u8; 32];
            let mut sb = [0f32; 1];
            act_quant_row_bf16(&x, &mut cb, &mut sb);
            assert_eq!(sb[0], pow2(k), "bf16 amax 448 * 2^{k}: scale 2^{k}");
            assert_eq!(cb[0], 0x7E, "448.5 * 2^{k} -> bf16 448 * 2^{k} -> code 448");
            assert_eq!(cb[1], 0x38);
            // 449.5 rounds to 450 in bf16: doubles there too
            x[0] = 449.5 * 2f32.powi(k);
            act_quant_row_bf16(&x, &mut cb, &mut sb);
            assert_eq!(sb[0], pow2(k + 1));
        }
    }

    #[test]
    fn q8_0_of_the_rounded_block() {
        let mut x = [0.0f32; 32];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 - 15.5) * 0.37;
        }
        act_quant_inplace(&mut x);
        let (q, d) = q8_0_block_device(&x);
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert_eq!(d, amax / 127.0);
        assert!(q.iter().any(|&v| v == 127 || v == -127));
    }

    #[test]
    fn grouped_ref_matches_a_dequantized_dot() {
        // small shape: 2 groups x 64 rows, group_dim 64
        let (gd, rank, ng, b) = (64usize, 64usize, 2usize, 3usize);
        let out_dim = rank * ng;
        let mut w = vec![0u8; fp8_weight_bytes(out_dim, gd)];
        let mut st = 7u64;
        let mut rnd = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for c in w[..out_dim * gd].iter_mut() {
            let v = (rnd() & 0xFF) as u8;
            *c = if v & 0x7F == 0x7F { 0x7E } else { v };
        }
        for s in w[out_dim * gd..].iter_mut() {
            *s = 120 + (rnd() % 8) as u8;
        }
        let x: Vec<f32> = (0..b * ng * gd).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01).collect();
        let mut codes = vec![0u8; x.len()];
        let mut sc = vec![0f32; x.len() / 32];
        act_quant_row(&x, &mut codes, &mut sc);
        let (out, mag) = fp8_grouped_matvec_ref(&w, &codes, &sc, gd, rank, ng, b);
        // independent: dequantize both operands to f64 and dot
        let kb_n = gd / 32;
        for bi in 0..b {
            for r in 0..out_dim {
                let g = r / rank;
                let mut s = 0f64;
                for k in 0..gd {
                    let wv = e4m3fn_decode(w[r * gd + k]) as f64 * e8m0_to_f32(w[out_dim * gd + (r / 32) * kb_n + k / 32]) as f64;
                    let i = bi * ng * gd + g * gd + k;
                    let av = e4m3fn_decode(codes[i]) as f64 * sc[i / 32] as f64;
                    s += wv * av;
                }
                let o = out[bi * out_dim + r];
                assert!((o - s).abs() <= 1e-12 * mag[bi * out_dim + r].max(1e-30), "row {r} b {bi}: {o} vs {s}");
            }
        }
    }

    /// f32 inputs (the engine's attention output) against torch: `x.to(bfloat16)` then the reference
    /// arithmetic (`gen_act_quant_golden.py`). [`act_quant_row_bf16`] must match bit for bit; plain
    /// [`act_quant_row`] on the f32 values must NOT (the fixture holds the double-rounding and the
    /// amax-boundary cases), which is why every device caller rounds to bf16 first.
    #[test]
    fn f32_inputs_match_torch_through_bf16() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/act_quant_fp8");
        let read = |n: &str| std::fs::read(format!("{dir}/{n}")).unwrap_or_else(|e| panic!("{dir}/{n}: {e}"));
        let xb = read("x_f32.bin");
        let codes = read("codes_bf16in_e4m3.bin");
        let scales = read("scales_bf16in_e8m0.bin");
        let n = xb.len() / 4;
        assert!(n > 0 && n % 32 == 0 && codes.len() == n && scales.len() == n / 32);
        let x: Vec<f32> = xb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let (mut c, mut s) = (vec![0u8; n], vec![0f32; n / 32]);
        act_quant_row_bf16(&x, &mut c, &mut s);
        assert_eq!(c, codes, "codes of the bf16 o");
        let sb: Vec<u8> = s.iter().map(|&v| e8m0_byte(scale_exp_of(v))).collect();
        assert_eq!(sb, scales, "scales of the bf16 o");
        let (mut c32, mut s32) = (vec![0u8; n], vec![0f32; n / 32]);
        act_quant_row(&x, &mut c32, &mut s32);
        let dc = c32.iter().zip(&c).filter(|(a, b)| a != b).count();
        let ds = s32.iter().zip(&s).filter(|(a, b)| a != b).count();
        eprintln!("skipping the bf16 point: {dc} of {n} codes, {ds} of {} scales differ", n / 32);
        assert!(dc > 0 && ds > 0, "the fixture must exercise both bf16-point cases");
    }

    fn scale_exp_of(s: f32) -> i32 {
        ((s.to_bits() >> 23) & 0xFF) as i32 - 127
    }

    /// Golden vectors from the reference kernel's arithmetic re-implemented in
    /// torch ops (`scripts/v41_oracle/gen_act_quant_golden.py`): bf16 inputs,
    /// the in-place bf16 output, the e4m3 codes and the e8m0 scale bytes.
    #[test]
    fn matches_torch_golden_vectors() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/act_quant_fp8");
        let read = |n: &str| std::fs::read(format!("{dir}/{n}")).unwrap_or_else(|e| panic!("{dir}/{n}: {e}"));
        let x_bf16 = read("x_bf16.bin");
        let y_bf16 = read("y_bf16.bin");
        let codes = read("codes_e4m3.bin");
        let scales = read("scales_e8m0.bin");
        let n = x_bf16.len() / 2;
        assert!(n > 0 && n % 32 == 0 && y_bf16.len() == 2 * n && codes.len() == n && scales.len() == n / 32);
        let bf = |b: &[u8], i: usize| f32::from_bits((u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32) << 16);
        let x: Vec<f32> = (0..n).map(|i| bf(&x_bf16, i)).collect();
        let mut y = x.clone();
        act_quant_inplace(&mut y);
        let mut bad = 0;
        for blk in 0..n / 32 {
            let (c, e) = act_quant_block(&x[blk * 32..blk * 32 + 32]);
            assert_eq!(e8m0_byte(e), scales[blk], "block {blk}: scale");
            for j in 0..32 {
                let i = blk * 32 + j;
                if c[j] != codes[i] || y[i].to_bits() != bf(&y_bf16, i).to_bits() {
                    if bad < 10 {
                        eprintln!("block {blk} elem {j}: x {:e} code {:#04x} vs {:#04x}, y {:e} vs {:e}", x[i], c[j], codes[i], y[i], bf(&y_bf16, i));
                    }
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "{bad} of {n} values differ from the torch golden vectors");
    }
}
