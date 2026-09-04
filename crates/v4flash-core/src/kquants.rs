//! Scalar CPU dequantization for the "dumper variant" GGUF rewriter
//! (`gguf-dequant-dense`), plus the F16 load-time conversion helpers.
//!
//! Two provenance contracts, both load-bearing:
//!
//! - The `dequant_row_*` functions are line-by-line ports of llama.cpp's
//!   `dequantize_row_q8_0/q4_K/q5_K/q6_K` (ggml/src/ggml-quants.c) so the
//!   F32 output is bit-identical to what a ggml f32 dequant computes.
//! - `f32_to_f16_bits` / `convert_to_f16` REPLICATE
//!   `v4flash_kernels::weight_contract` (v4flash-core cannot depend on
//!   v4flash-kernels — the dependency points the other way). Keeping the
//!   math byte-identical to the engine's load-time conversion minimizes
//!   oracle noise between the dumper variant and the engine. If you touch
//!   one copy, touch both.

use color_eyre::eyre::{self, eyre};

use crate::gguf::GgufType;

pub const QK_K: usize = 256;

/// f16 bits -> f32, exact (mirrors `v4flash_kernels::iq2_xxs_tables::f16_to_f32`).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal: normalise.
            let mut e: i32 = -14;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// f32 -> f16 bits, round-to-nearest-even, denormal + inf/nan correct.
/// REPLICA of `v4flash_kernels::weight_contract::f32_to_f16_bits`.
pub fn f32_to_f16_bits(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x007f_ffff;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half_mant = (m >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let mut result = sign | half_mant;
        if (m & round_bit) != 0 && ((m & (round_bit - 1)) != 0 || (half_mant & 1) != 0) {
            result += 1;
        }
        return result;
    }
    let half_mant = (mant >> 13) as u16;
    let mut result = sign | ((e as u16) << 10) | half_mant;
    if (mant & 0x0000_1000) != 0 && ((mant & 0x0000_0fff) != 0 || (half_mant & 1) != 0) {
        result += 1;
    }
    result
}

/// llama.cpp `get_scale_min_k4`: unpack the 6-bit scale/min pair `j`
/// (0..7) from a q4_K/q5_K 12-byte scales field.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

#[inline]
fn f16_at(src: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]))
}

/// Q8_0 block (34 B = f16 d + 32×i8) -> 32 f32. Port of `dequantize_row_q8_0`.
fn dequant_block_q8_0(blk: &[u8], out: &mut Vec<f32>) {
    let d = f16_at(blk, 0);
    for &q in &blk[2..34] {
        out.push(d * (q as i8) as f32);
    }
}

/// Q4_K super-block (144 B) -> 256 f32. Port of `dequantize_row_q4_K`.
/// Layout: d(2) dmin(2) scales(12) qs(128).
fn dequant_block_q4_k(blk: &[u8], out: &mut Vec<f32>) {
    let d = f16_at(blk, 0);
    let min = f16_at(blk, 2);
    let scales = &blk[4..16];
    let mut q = &blk[16..144];
    let mut is = 0;
    for _j in (0..QK_K).step_by(64) {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for l in 0..32 {
            out.push(d1 * (q[l] & 0xf) as f32 - m1);
        }
        for l in 0..32 {
            out.push(d2 * (q[l] >> 4) as f32 - m2);
        }
        q = &q[32..];
        is += 2;
    }
}

/// Q5_K super-block (176 B) -> 256 f32. Port of `dequantize_row_q5_K`.
/// Layout: d(2) dmin(2) scales(12) qh(32) qs(128).
fn dequant_block_q5_k(blk: &[u8], out: &mut Vec<f32>) {
    let d = f16_at(blk, 0);
    let min = f16_at(blk, 2);
    let scales = &blk[4..16];
    let qh = &blk[16..48];
    let mut ql = &blk[48..176];
    let mut is = 0;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for _j in (0..QK_K).step_by(64) {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            out.push(d1 * ((ql[l] & 0xf) + hi) as f32 - m1);
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            out.push(d2 * ((ql[l] >> 4) + hi) as f32 - m2);
        }
        ql = &ql[32..];
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Q6_K super-block (210 B) -> 256 f32. Port of `dequantize_row_q6_K`.
/// Layout: ql(128) qh(64) scales(16, i8) d(2).
fn dequant_block_q6_k(blk: &[u8], out: &mut Vec<f32>) {
    let d = f16_at(blk, 208);
    let mut ql = &blk[0..128];
    let mut qh = &blk[128..192];
    let mut sc = &blk[192..208];
    let base = out.len();
    out.resize(base + QK_K, 0.0);
    let y = &mut out[base..];
    let mut yo = 0;
    for _n in (0..QK_K).step_by(128) {
        for l in 0..32 {
            let is = l / 16;
            let q1 = (((ql[l] & 0xf) | (((qh[l] >> 0) & 3) << 4)) as i8 as i32 - 32) as f32;
            let q2 = (((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32) as f32;
            let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32) as f32;
            let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32) as f32;
            y[yo + l] = d * (sc[is] as i8) as f32 * q1;
            y[yo + l + 32] = d * (sc[is + 2] as i8) as f32 * q2;
            y[yo + l + 64] = d * (sc[is + 4] as i8) as f32 * q3;
            y[yo + l + 96] = d * (sc[is + 6] as i8) as f32 * q4;
        }
        yo += 128;
        ql = &ql[64..];
        qh = &qh[32..];
        sc = &sc[8..];
    }
}

/// Dequantize `src` (whole blocks of `dt`) to f32, appending to `out`.
/// Supported: Q8_0, Q4_K, Q5_K, Q6_K, IQ3_S, F16, BF16, F32 (widening copies).
pub fn dequant_to_f32(dt: GgufType, src: &[u8], out: &mut Vec<f32>) -> eyre::Result<()> {
    let (_, block_bytes) = dt
        .block_shape()
        .ok_or_else(|| eyre!("dequant_to_f32: unknown dtype {dt:?}"))?;
    let block_bytes = block_bytes as usize;
    if src.len() % block_bytes != 0 {
        return Err(eyre!(
            "dequant_to_f32: {} bytes is not whole {} blocks of {block_bytes} B",
            src.len(),
            dt.name()
        ));
    }
    match dt {
        GgufType::F32 => {
            for c in src.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        GgufType::F16 => {
            for c in src.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        GgufType::BF16 => {
            for c in src.chunks_exact(2) {
                out.push(f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16));
            }
        }
        GgufType::Q8_0 => {
            for blk in src.chunks_exact(block_bytes) {
                dequant_block_q8_0(blk, out);
            }
        }
        GgufType::Q4_K => {
            for blk in src.chunks_exact(block_bytes) {
                dequant_block_q4_k(blk, out);
            }
        }
        GgufType::Q5_K => {
            for blk in src.chunks_exact(block_bytes) {
                dequant_block_q5_k(blk, out);
            }
        }
        GgufType::Q6_K => {
            for blk in src.chunks_exact(block_bytes) {
                dequant_block_q6_k(blk, out);
            }
        }
        // Not reached by `gguf-dequant-dense` today (routed-expert tensors,
        // the only IQ3_S tensors in the unsloth UD-IQ3_XXS mix, are
        // `Action::Pass` there); here so every kernel-decoded quant has a
        // scalar CPU path for tests and future dumpers.
        GgufType::IQ3_S => {
            let base = out.len();
            out.resize(base + src.len() / block_bytes * QK_K, 0.0);
            crate::iq3_s_ref::dequant_row_iq3_s(src, &mut out[base..]);
        }
        other => return Err(eyre!("dequant_to_f32: unsupported dtype {other:?}")),
    }
    Ok(())
}

/// Convert an on-disk tensor payload of `dtype` to F16 little-endian bytes.
/// REPLICA of `v4flash_kernels::weight_contract::convert_to_f16` — same
/// source types (F16 passthrough, F32, BF16, Q8_0-dequant), same math, so
/// the dumper variant stores byte-for-byte what the engine computes at load.
pub fn convert_to_f16(dtype: GgufType, src: &[u8], n_elements: u64) -> eyre::Result<Vec<u8>> {
    let n = n_elements as usize;
    match dtype {
        GgufType::F16 => Ok(src.to_vec()),
        GgufType::F32 => {
            debug_assert_eq!(src.len(), n * 4);
            let mut out = Vec::with_capacity(n * 2);
            for c in src.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            Ok(out)
        }
        GgufType::BF16 => {
            debug_assert_eq!(src.len(), n * 2);
            let mut out = Vec::with_capacity(n * 2);
            for c in src.chunks_exact(2) {
                let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            Ok(out)
        }
        GgufType::Q8_0 => {
            // block: f16 scale + 32 × i8
            debug_assert_eq!(src.len(), n / 32 * 34);
            let mut out = Vec::with_capacity(n * 2);
            for blk in src.chunks_exact(34) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &q in &blk[2..34] {
                    let v = d * (q as i8) as f32;
                    out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
                }
            }
            Ok(out)
        }
        other => Err(eyre!("convert_to_f16: unsupported source dtype {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_exact() {
        // Every finite f16 must survive f16->f32->f16 exactly.
        for bits in 0..=0xffffu16 {
            let exp = (bits >> 10) & 0x1f;
            if exp == 0x1f {
                continue; // inf/nan
            }
            assert_eq!(f32_to_f16_bits(f16_to_f32(bits)), bits, "bits {bits:#06x}");
        }
    }

    /// Pins `f32_to_f16_bits` to the same values as
    /// `v4flash_kernels::weight_contract` (its unit tests assert the
    /// identical cases — the two copies must not drift).
    #[test]
    fn f32_to_f16_pinned_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-1.5), 0xbe00);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff); // f16 max
        assert_eq!(f32_to_f16_bits(1e6), 0x7c00); // overflow -> inf
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_f16_bits(6.1e-5), 0x03ff); // just below smallest normal
        assert_eq!(f32_to_f16_bits(5.9e-8), 0x0001); // denormal
        // Round-to-nearest-even at the halfway point:
        // 1.0009765625 = 1 + 2^-10 (exact f16), 1.00048828125 = 1 + 2^-11
        // rounds to even mantissa (1.0).
        assert_eq!(f32_to_f16_bits(1.00048828125), 0x3c00);
        assert_eq!(f32_to_f16_bits(1.0014648438), 0x3c02); // 1 + 3*2^-11 rounds up to even
    }

    #[test]
    fn bf16_to_f16_pinned() {
        // bf16 of 1.5 = 0x3FC0; convert [0x3FC0] -> f16 1.5 = 0x3E00.
        let src = 0x3fc0u16.to_le_bytes();
        let out = convert_to_f16(GgufType::BF16, &src, 1).unwrap();
        assert_eq!(u16::from_le_bytes([out[0], out[1]]), 0x3e00);
        // bf16 has fewer mantissa bits than f16 in the normal range:
        // 0x4049 (3.140625) widens exactly to f16 0x4248.
        let src = 0x4049u16.to_le_bytes();
        let out = convert_to_f16(GgufType::BF16, &src, 1).unwrap();
        assert_eq!(u16::from_le_bytes([out[0], out[1]]), 0x4248);
    }

    #[test]
    fn q8_0_dequant_to_f16_pinned() {
        // One block: scale 0.5, quants 0..31 (same case as weight_contract).
        let mut blk = vec![];
        blk.extend_from_slice(&f32_to_f16_bits(0.5).to_le_bytes());
        blk.extend(0..32u8);
        let out = convert_to_f16(GgufType::Q8_0, &blk, 32).unwrap();
        assert_eq!(out.len(), 64);
        let v5 = f16_to_f32(u16::from_le_bytes([out[10], out[11]]));
        assert_eq!(v5, 2.5); // 0.5 * 5
        let mut f32s = Vec::new();
        dequant_to_f32(GgufType::Q8_0, &blk, &mut f32s).unwrap();
        assert_eq!(f32s[5], 2.5);
        assert_eq!(f32s[31], 15.5);
    }


    /// Bit-for-bit pin of the K-quant dequants against llama.cpp's REAL
    /// scalar `dequantize_row_*` code. The expected u32 arrays are f32 bit
    /// patterns produced by a C harness compiled against
    /// llama.cpp/ggml/src/ggml-quants.c on identical synthetic blocks
    /// (same LCG, same seeds, d forced to f16 0x2e66, dmin to 0x2a66).
    #[test]
    fn dequant_matches_llamacpp_bit_for_bit() {
    const EXPECTED_Q8_0: [u32; 32] = [
        0x406cbe00,        0x4134c180,        0x41119080,        0x3ff32400,        0xbfb32800,        0xc0932a00,
        0x40532600,        0xbe4cc000,        0x4104c480,        0x3ffff000,        0xc117f680,        0xc107f780,
        0x413b2780,        0xc1132a00,        0x413cc100,        0x40dcbf00,        0xc0265c00,        0x3f7ff000,
        0xc1065e00,        0x406cbe00,        0xc0665800,        0x414b2680,        0x407ff000,        0x40d32600,
        0x40eff100,        0xc1332800,        0xbffff000,        0xc0bff400,        0x40665800,        0x3f4cc000,
        0xc114c380,        0xc124c280,
    ];
    const EXPECTED_Q4_K: [u32; 256] = [
        0x41f32400,        0x3f332800,        0x41f32400,        0x40c32700,        0x4219f660,        0x4137f480,
        0x420f2a40,        0x41632500,        0x40598c00,        0x4137f480,        0x419cc300,        0x42045e20,
        0x41c7f380,        0x420f2a40,        0x419cc300,        0x4219f660,        0x41872ac0,        0x41dd8bc0,
        0x3f332800,        0x42045e20,        0x42045e20,        0x42045e20,        0x40598c00,        0x41872ac0,
        0x41c7f380,        0x410cc400,        0x4137f480,        0x419cc300,        0x42045e20,        0x41dd8bc0,
        0x4219f660,        0x420f2a40,        0x41ad8ec0,        0x40ccc000,        0x41925d40,        0x4184c480,
        0x4184c480,        0x419ff600,        0x419ff600,        0x4184c480,        0x4137f480,        0x41532600,
        0xbeccc000,        0x419ff600,        0x40ccc000,        0x41532600,        0x41ad8ec0,        0x411cc300,
        0x4137f480,        0x41019180,        0x411cc300,        0x41019180,        0x416e5780,        0x416e5780,
        0x41532600,        0x41ad8ec0,        0x41ad8ec0,        0x4137f480,        0x411cc300,        0x40965d00,
        0x41925d40,        0xc0065e00,        0x411cc300,        0x40ccc000,        0x428d90c0,        0x406cbe00,
        0x4208c440,        0x42398e00,        0x42b22810,        0xc0199000,        0x42a5f5a0,        0x411cc300,
        0x411cc300,        0xc0199000,        0x4299c330,        0x41e0bec0,        0x406cbe00,        0x42398e00,
        0x42212920,        0x41e0bec0,        0x406cbe00,        0x4251f2e0,        0x41e0bec0,        0x411cc300,
        0x411cc300,        0x4299c330,        0x42212920,        0x4299c330,        0x417e5680,        0x42b22810,
        0x42212920,        0x41e0bec0,        0x428d90c0,        0x4299c330,        0xc0199000,        0x4208c440,
        0x4168be40,        0x40b18e80,        0x414bf340,        0x4182c4a0,        0x414bf340,        0x4182c4a0,
        0x4182c4a0,        0x41bc5aa0,        0x41adf520,        0xbfd32600,        0x40eb2480,        0x4182c4a0,
        0x4168be40,        0x4168be40,        0x3e199000,        0x41cac020,        0x40eb2480,        0x3ff98a00,
        0x406ff100,        0x3e199000,        0x40eb2480,        0x4182c4a0,        0x41adf520,        0x41adf520,
        0x41adf520,        0x3e199000,        0x412f2840,        0x4182c4a0,        0x40eb2480,        0x3ff98a00,
        0x41cac020,        0x406ff100,        0x41b65b00,        0x421929a0,        0x4157f280,        0x40865e00,
        0x41e7f180,        0x420cc400,        0x3f8cc400,        0x419d8fc0,        0x3f8cc400,        0x40865e00,
        0x42005e60,        0x4157f280,        0x40865e00,        0x40e98b00,        0x4157f280,        0x3f8cc400,
        0x4184c480,        0x41b65b00,        0x4157f280,        0x3f8cc400,        0x42258f40,        0x41e7f180,
        0x42258f40,        0x419d8fc0,        0x41265c00,        0x4157f280,        0x4157f280,        0x40865e00,
        0x40865e00,        0xbffff000,        0x40e98b00,        0x41e7f180,        0x416ff100,        0x41f32400,
        0xbeccc000,        0x416ff100,        0x416ff100,        0x41e18b80,        0x419b2980,        0x41e18b80,
        0x41065e00,        0x41e18b80,        0xbeccc000,        0x41be5a80,        0xbeccc000,        0x41acc200,
        0x42025e40,        0xbeccc000,        0x41acc200,        0x419b2980,        0x414cc000,        0x42025e40,
        0x407ff000,        0xbeccc000,        0x41cff300,        0x41f32400,        0x40c65a00,        0x41be5a80,
        0x419b2980,        0x41298f00,        0x41899100,        0x419b2980,        0x41065e00,        0x416ff100,
        0x40b98e00,        0x40532600,        0xbf332800,        0xbfd98c00,        0x40132a00,        0x40999000,
        0x40899100,        0xbfd98c00,        0xbfd98c00,        0xbfd98c00,        0xbe4cc000,        0xbe4cc000,
        0x3fa65c00,        0x40732400,        0x40a98f00,        0x3f4cc000,        0x40132a00,        0x3fa65c00,
        0x40b98e00,        0xbe4cc000,        0x3fe65800,        0xbfd98c00,        0x40332800,        0x40999000,
        0x40b98e00,        0x3f4cc000,        0x3e999000,        0x40a98f00,        0x3f4cc000,        0x40732400,
        0xbe4cc000,        0x40132a00,        0x41498d00,        0x41865e00,        0x41865e00,        0x41eb2480,
        0x41865e00,        0x4238c140,        0x41eb2480,        0x421729c0,        0x4227f580,        0x426b2480,
        0x41a7f580,        0x41a7f580,        0x40865e00,        0x40865e00,        0x4238c140,        0x41a7f580,
        0x42065e00,        0x00000000,        0x42065e00,        0x425a58c0,        0x41c98d00,        0x427bf040,
        0x40865e00,        0x00000000,        0x41a7f580,        0x4238c140,        0x425a58c0,        0x41498d00,
        0x41865e00,        0x42498d00,        0x4227f580,        0x41065e00,
    ];
    const EXPECTED_Q5_K: [u32; 256] = [
        0x414e5980,        0x416b2480,        0x40065e00,        0x3e999000,        0x408ff700,        0x4144c080,
        0x40532600,        0x40a32900,        0x40532600,        0xc0065e00,        0x416b2480,        0x40798a00,
        0x41318e80,        0x40065e00,        0x3fbff400,        0x4174bd80,        0x411e5c80,        0xbf665800,
        0x4183f7c0,        0x4174bd80,        0x402cc200,        0x414e5980,        0x3e999000,        0x41019180,
        0x417e5680,        0x414e5980,        0x41618b80,        0x3fbff400,        0x4114c380,        0x4144c080,
        0x410b2a80,        0x40798a00,        0x4240c0c0,        0x4147f380,        0x417e5680,        0x4217f680,
        0x4147f380,        0x40ecbe00,        0x42112a20,        0x407ff000,        0x418cc400,        0x421ec2e0,
        0x41632500,        0x41f98a00,        0x40132a00,        0x3f199000,        0x41d0bfc0,        0x41d0bfc0,
        0x41632500,        0x4239f460,        0x4240c0c0,        0x407ff000,        0xbf8cc400,        0x42039160,
        0x412cc200,        0x40ecbe00,        0x3f199000,        0x42332800,        0x417e5680,        0x41ebf140,
        0x4239f460,        0x41c32700,        0x41b58e40,        0xc0332800,        0x4235f4a0,        0x428e90b0,
        0x419d8fc0,        0x429fc2d0,        0x42698b00,        0x42f5bd70,        0x41765700,        0x41318e80,
        0x424726c0,        0x430377c8,        0x42698b00,        0x401ff600,        0x419d8fc0,        0x42fe5680,
        0x429fc2d0,        0x42a85be0,        0x42c22710,        0x42fe5680,        0x42fe5680,        0x4224c280,
        0x42698b00,        0x430377c8,        0x41e25840,        0x429fc2d0,        0x429729c0,        0x429fc2d0,
        0x42c22710,        0x428e90b0,        0x42e48b50,        0x40d98c00,        0x42f5bd70,        0x42fe5680,
        0x418b90e0,        0xbf598c00,        0x42d9d8c8,        0x40a7f580,        0x429cdc98,        0x4290aa28,
        0x42fe7018,        0x43179cec,        0x429cdc98,        0x42275bf0,        0x425825b0,        0x4290aa28,
        0x42cda658,        0x43055144,        0x431183b4,        0x42b54178,        0x4329e894,        0x41358e40,
        0x418b90e0,        0x43361b04,        0x431183b4,        0x431183b4,        0x42e60b38,        0x42d9d8c8,
        0x40a7f580,        0x41ed2460,        0x42a90f08,        0x425825b0,        0x428477b8,        0x430b6a7c,
        0x4290aa28,        0x42d9d8c8,        0x41d4bf80,        0x41b7f480,        0x415ff200,        0x41d4bf80,
        0x41bf2740,        0x40f65700,        0x4134c180,        0x4117f680,        0x3fb32800,        0x41099100,
        0x417cbd00,        0x41099100,        0x41432700,        0x418cc400,        0x416e5780,        0x41518c80,
        0x418cc400,        0x3fb32800,        0x41cd8cc0,        0x41dbf240,        0x40bcc100,        0x4193f6c0,
        0x41bf2740,        0xbeccc000,        0x40832b00,        0x4134c180,        0xbeccc000,        0x41dbf240,
        0xbeccc000,        0x41432700,        0x41265c00,        0x41d4bf80,        0x40298f00,        0x3f265c00,
        0x40298f00,        0x3fd32600,        0x41e524e0,        0x41dd2560,        0x414a59c0,        0x414a59c0,
        0x41c526e0,        0x3fd32600,        0x418d2a60,        0x41a528e0,        0x41d525e0,        0x41c526e0,
        0x41cd2660,        0x41e524e0,        0x41dd2560,        0x4094c380,        0x413a5ac0,        0xbeb32800,
        0x414a59c0,        0x41852ae0,        0x418d2a60,        0x411a5cc0,        0x41f523e0,        0x419529e0,
        0x411a5cc0,        0x40b4c180,        0x41cd2660,        0x413a5ac0,        0x41ad2860,        0x40f4bd80,
        0x424dbff0,        0x4295dd08,        0x423af450,        0x41945d20,        0x42e10b88,        0x4295dd08,
        0x422828b0,        0x42b20e78,        0x422828b0,        0x42608b90,        0x423af450,        0x428c7738,
        0x42608b90,        0x408e5d80,        0x42bb7448,        0x42029170,        0x43116a1c,        0x423af450,
        0x42ea7158,        0x43035164,        0x41945d20,        0x42c4da18,        0x4295dd08,        0x424dbff0,
        0x428c7738,        0x42ce3fe8,        0x428c7738,        0x42ea7158,        0x42d7a5b8,        0x41df8ba0,
        0x42fd3cf8,        0x41df8ba0,        0x41b58e40,        0x407ff000,        0x4147f380,        0x41de5880,
        0x407ff000,        0x4147f380,        0x41c32700,        0x41c32700,        0x42478d20,        0x3f199000,
        0x41f98a00,        0x41f98a00,        0x417e5680,        0x41632500,        0xc0332800,        0x422c5ba0,
        0x407ff000,        0x40132a00,        0x421ec2e0,        0x41b58e40,        0x42478d20,        0x41a7f580,
        0x40132a00,        0x418cc400,        0x42332800,        0xc0332800,        0x40ecbe00,        0x41ebf140,
        0x4147f380,        0x3f199000,        0x42112a20,        0x40b65b00,
    ];
    const EXPECTED_Q6_K: [u32; 256] = [
        0x42225c40,        0x41398e00,        0x432df520,        0x80000000,        0x43398e00,        0xc2b98e00,
        0x420b2a80,        0xc3055e10,        0x42e7f180,        0xc2225c40,        0x430b2a80,        0xc2b98e00,
        0xc18b2a80,        0x418b2a80,        0x420b2a80,        0x42398e00,        0x42a65c00,        0xc2932a00,
        0x414cc000,        0x429ff600,        0xc23ff400,        0x417ff000,        0x428cc400,        0xc2865e00,
        0xc0ccc000,        0xc1e65800,        0xc2865e00,        0x428cc400,        0xc2932a00,        0xc1999000,
        0x414cc000,        0x42332800,        0x42065e00,        0xc1acc200,        0xc20b2a80,        0xc0e65800,
        0xc0665800,        0xc0199000,        0xc20ff700,        0x420ff700,        0xc1a32900,        0x40bff400,
        0xc1bff400,        0xc1c98d00,        0xc0199000,        0x41c98d00,        0x40e65800,        0x40e65800,
        0x430ec3e0,        0x431c5ca0,        0x42598c00,        0xc3598c00,        0xc3232900,        0x00000000,
        0x4307f780,        0xc2959040,        0x42d98c00,        0xc2cbf340,        0x4287f780,        0xc0d98c00,
        0xc1d98c00,        0x43159040,        0xc287f780,        0xc2959040,        0xc21cc300,        0xc15ff200,
        0xc200c4c0,        0xc1f65700,        0xc1eb2480,        0x42225c40,        0x41be5a80,        0x41f65700,
        0xc1f65700,        0xc21cc300,        0x41498d00,        0xc1332800,        0x42065e00,        0xc15ff200,
        0x420bf740,        0x422d8ec0,        0xc317f680,        0xc327f580,        0x429ff600,        0xc337f480,
        0x42dff200,        0xc377f080,        0x4367f180,        0xc2cff300,        0xc2aff500,        0xc25ff200,
        0x435ff200,        0x434ff300,        0x4307f780,        0xc2cff300,        0xc307f780,        0xc35ff200,
        0x42b8f470,        0xc213f6c0,        0x42d68c30,        0x42318e80,        0x4213f6c0,        0x42c7c050,
        0x42c7c050,        0x42d68c30,        0xc1b18e80,        0xc293f6c0,        0x41cf2640,        0xc1b18e80,
        0xc26cbe00,        0xc26cbe00,        0x42cf2640,        0xc2d68c30,        0x4333c190,        0xc3402730,
        0x41465a00,        0xc3085de0,        0xc277f080,        0x4333c190,        0xc2f7f080,        0x41f7f080,
        0xc294c380,        0xc1465a00,        0x42d2bfa0,        0x42df2540,        0xc2a12920,        0xc3275bf0,
        0xc2d2bfa0,        0xc1c65a00,        0x42a9f560,        0x41a32900,        0x41be5a80,        0xc1be5a80,
        0xc2cbf340,        0x42159040,        0xc0598c00,        0x42b78e20,        0xc1d98c00,        0x42232900,
        0x426724c0,        0xc1be5a80,        0x428ec3e0,        0x42c526e0,        0x42b78e20,        0xc23e5a80,
        0xc36ff100,        0xc327f580,        0xc327f580,        0x42bff400,        0xc30ff700,        0xc36ff100,
        0xc25ff200,        0x437ff000,        0x428ff700,        0x431ff600,        0x437ff000,        0x41fff000,
        0xc27ff000,        0xc29ff600,        0x42eff100,        0x434ff300,        0xc2a32900,        0x41598c00,
        0xc2a9f560,        0xc2c526e0,        0x42d2bfa0,        0x42812b20,        0x42598c00,        0x42d2bfa0,
        0xc2d98c00,        0x42598c00,        0xc2959040,        0xc2598c00,        0x4287f780,        0xc2be5a80,
        0xc1f4bd80,        0x426724c0,        0x42c58d40,        0xc23b2780,        0x42bb2780,        0xc2798a00,
        0x42bb2780,        0xc1a65c00,        0xc3212920,        0xc2da58c0,        0xc1cff300,        0x42798a00,
        0xc2bb2780,        0xc0a65c00,        0x43072ac0,        0xc2872ac0,        0xc2cff300,        0x42f98a00,
        0xc08ff700,        0xc0dcbf00,        0x3fe65800,        0xc0199000,        0xc10b2a80,        0x41199000,
        0x40798a00,        0xc0865e00,        0xc0e65800,        0x40eff100,        0x40a32900,        0x40bff400,
        0xc0bff400,        0xc0865e00,        0x3e999000,        0x3f665800,        0x42ad8ec0,        0x40332800,
        0x41332800,        0x41f65700,        0x425ff200,        0xc26b2480,        0xc227f580,        0x415ff200,
        0xc1c98d00,        0xc2a7f580,        0x4254bf80,        0xc25ff200,        0xc21cc300,        0x41065e00,
        0x40332800,        0x42a7f580,        0x4331c1b0,        0xc24b2680,        0xc33e7418,        0xc3648b50,
        0xc1cb2680,        0x4357d8e8,        0xc38baa78,        0x427df020,        0xc14b2680,        0x43c4cd4c,
        0x42185ce0,        0xc2185ce0,        0x433e7418,        0x43855144,        0xc30baa78,        0xc1cb2680,
        0xc34b4018,        0x43408d90,        0x43a5cf3c,        0x43408d90,        0xc36b57b0,        0x42005e60,
        0x430b10e8,        0xc3408d90,        0xc3805e60,        0x42c08d90,        0xc3a5cf3c,        0x438b10e8,
        0xc36b57b0,        0x80000000,        0xc3805e60,        0xc36b57b0,
    ];
        struct Case {
            dt: GgufType,
            bytes: usize,
            seed: u32,
            d_off: usize,
            m_off: Option<usize>,
            want: &'static [u32],
        }
        let cases = [
            Case { dt: GgufType::Q8_0, bytes: 34, seed: 0x81, d_off: 0, m_off: None, want: &EXPECTED_Q8_0 },
            Case { dt: GgufType::Q4_K, bytes: 144, seed: 0x84, d_off: 0, m_off: Some(2), want: &EXPECTED_Q4_K },
            Case { dt: GgufType::Q5_K, bytes: 176, seed: 0x85, d_off: 0, m_off: Some(2), want: &EXPECTED_Q5_K },
            Case { dt: GgufType::Q6_K, bytes: 210, seed: 0x86, d_off: 208, m_off: None, want: &EXPECTED_Q6_K },
        ];
        for c in cases {
            let mut blk = lcg_bytes(c.bytes, c.seed);
            blk[c.d_off..c.d_off + 2].copy_from_slice(&0x2e66u16.to_le_bytes());
            if let Some(m) = c.m_off {
                blk[m..m + 2].copy_from_slice(&0x2a66u16.to_le_bytes());
            }
            let mut out = Vec::new();
            dequant_to_f32(c.dt, &blk, &mut out).unwrap();
            assert_eq!(out.len(), c.want.len(), "{:?}", c.dt);
            for (i, (&got, &want)) in out.iter().zip(c.want).enumerate() {
                assert_eq!(got.to_bits(), want, "{:?} elem {i}: got {got}", c.dt);
            }
        }
    }

    /// Deterministic byte pattern for block-content tests.
    fn lcg_bytes(n: usize, mut state: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push((state >> 24) as u8);
        }
        v
    }

    /// Q4_K: hand-check the first few values against the ggml formula.
    #[test]
    fn q4_k_dequant_formula() {
        let mut blk = vec![0u8; 144];
        blk[..2].copy_from_slice(&f32_to_f16_bits(0.25).to_le_bytes()); // d
        blk[2..4].copy_from_slice(&f32_to_f16_bits(0.125).to_le_bytes()); // dmin
        // scales[0]=17 -> sc0=17; scales[4]=9 -> m0=9
        blk[4] = 17;
        blk[8] = 9;
        // qs[0] = 0xB3 -> low nibble 3 (elem 0), high nibble 11 (elem 32)
        blk[16] = 0xb3;
        let mut out = Vec::new();
        dequant_to_f32(GgufType::Q4_K, &blk, &mut out).unwrap();
        assert_eq!(out.len(), 256);
        // y[0] = d*sc0*3 - dmin*m0 = 0.25*17*3 - 0.125*9*1... m applies to
        // every element of the sub-block: y = d1*q - m1.
        assert_eq!(out[0], 0.25 * 17.0 * 3.0 - 0.125 * 9.0);
        // Second sub-block (elems 32..63) uses scale pair j=1 (zeros here).
        assert_eq!(out[32], 0.0);
        // Elem 1: q=0 -> y = -m1
        assert_eq!(out[1], -(0.125 * 9.0));
    }

    #[test]
    fn q5_k_dequant_formula() {
        let mut blk = vec![0u8; 176];
        blk[..2].copy_from_slice(&f32_to_f16_bits(0.5).to_le_bytes()); // d
        blk[2..4].copy_from_slice(&f32_to_f16_bits(0.0).to_le_bytes()); // dmin
        blk[4] = 2; // sc0 = 2
        blk[16] = 0x01; // qh[0] bit0 -> +16 for elem 0
        blk[48] = 0x07; // ql[0] low nibble = 7
        let mut out = Vec::new();
        dequant_to_f32(GgufType::Q5_K, &blk, &mut out).unwrap();
        assert_eq!(out[0], 0.5 * 2.0 * (7.0 + 16.0));
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn q6_k_dequant_formula() {
        let mut blk = vec![0u8; 210];
        blk[208..210].copy_from_slice(&f32_to_f16_bits(0.5).to_le_bytes()); // d
        blk[0] = 0x2a; // ql[0]: low nibble 10 (elem 0), high nibble 2 (elem 64)
        blk[128] = 0x04; // qh[0]: bits2-3 = 1 -> elem 32 gets +16
        blk[192] = 3; // scales[0] = 3 (elems 0..15)
        blk[194] = 5; // scales[2] (elems 32..47)
        blk[196] = 7; // scales[4] (elems 64..79)
        let mut out = Vec::new();
        dequant_to_f32(GgufType::Q6_K, &blk, &mut out).unwrap();
        // q1 = (10 | 0) - 32 = -22, scale 3
        assert_eq!(out[0], 0.5 * 3.0 * -22.0);
        // q2 = (0 | 1<<4) - 32 = -16, scale 5
        assert_eq!(out[32], 0.5 * 5.0 * -16.0);
        // q3 = (2 | 0) - 32 = -30, scale 7
        assert_eq!(out[64], 0.5 * 7.0 * -30.0);
    }

    /// Structural invariants on random blocks: finite outputs, right count.
    #[test]
    fn random_blocks_dequant_shape() {
        for (dt, bytes) in [
            (GgufType::Q4_K, 144usize),
            (GgufType::Q5_K, 176),
            (GgufType::Q6_K, 210),
            (GgufType::Q8_0, 34),
        ] {
            let mut src = lcg_bytes(bytes * 3, 0xdead ^ bytes as u32);
            // Force sane f16 scales (avoid inf/nan) at each block's d/dmin.
            for b in 0..3 {
                let off = b * bytes;
                let (d_off, extra) = match dt {
                    GgufType::Q6_K => (off + 208, None),
                    GgufType::Q8_0 => (off, None),
                    _ => (off, Some(off + 2)),
                };
                src[d_off..d_off + 2]
                    .copy_from_slice(&f32_to_f16_bits(0.0625).to_le_bytes());
                if let Some(m_off) = extra {
                    src[m_off..m_off + 2]
                        .copy_from_slice(&f32_to_f16_bits(0.03125).to_le_bytes());
                }
            }
            let mut out = Vec::new();
            dequant_to_f32(dt, &src, &mut out).unwrap();
            let per_block = if dt == GgufType::Q8_0 { 32 } else { 256 };
            assert_eq!(out.len(), per_block * 3, "{dt:?}");
            assert!(out.iter().all(|v| v.is_finite()), "{dt:?}");
        }
    }
}
