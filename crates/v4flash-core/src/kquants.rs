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
/// Supported: Q8_0, Q4_K, Q5_K, Q6_K, F16, BF16, F32 (widening copies).
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
