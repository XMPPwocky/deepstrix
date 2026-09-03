//! IQ3_S CPU reference: codebook + row dequant + Q8_K dot.
//!
//! Provenance (load-bearing — this module is the truth the HIP
//! `iq3_s_pair_matvec` kernels are validated against):
//!
//! - [`IQ3S_GRID`] is `iq3s_grid` copied verbatim from llama.cpp
//!   `ggml/src/ggml-common.h` (512 × u32, four odd magnitudes 0x01..0x0f
//!   per entry).
//! - [`dequant_block_iq3_s`] is a line-by-line port of
//!   `dequantize_row_iq3_s` (`ggml/src/ggml-quants.c`): per 32-weight
//!   sub-block `ib32`, scale nibble `scales[ib32/2] >> 4*(ib32%2)`,
//!   `db = d * (1 + 2*nib)`; 8 grid entries per ib32 indexed by
//!   `qs[8*ib32 + 2l+{0,1}]` plus one high bit from `qh[ib32]`
//!   (bit `2l` -> grid1, bit `2l+1` -> grid2); signs are raw bytes
//!   `signs[4*ib32 + l]`, bit `j` (`kmask_iq2xs`) flips weight `j` of the
//!   8-weight group. There is NO fractional prefactor (iq2_s has 0.125,
//!   iq3_xxs 0.25).
//! - [`dot_iq3_s_q8_k`] mirrors `ggml_vec_dot_iq3_s_q8_K_generic`
//!   (`ggml/src/ggml-cpu/quants.c`): integer-domain
//!   `Σ_blk f16(d)·y.d · Σ_ib32 ls(ib32)·sumi(ib32)`, `ls = 2*nib + 1`.
//!
//! Both are pinned numerically to upstream's own code by
//! `crates/v4flash-kernels/tests/ref/iq3_s_gen.c` (see
//! `tests::matches_llama_cpp_reference`).
//!
//! Block layout (110 B = 3.4375 bpw):
//! `d`@0 (f16) | `qs`@2 (64) | `qh`@66 (8) | `signs`@74 (32) | `scales`@106 (4).

use crate::kquants::f16_to_f32;

/// Weights per super-block.
pub const QK_K: usize = 256;
/// Bytes per IQ3_S super-block.
pub const BLOCK_IQ3_S_BYTES: usize = 110;
/// Bytes per Q8_K super-block: f32 d | i8 qs[256] | i16 bsums[16].
pub const BLOCK_Q8_K_BYTES: usize = 292;

/// Byte offsets inside a 110-byte IQ3_S block.
pub const IQ3S_OFF_D: usize = 0;
pub const IQ3S_OFF_QS: usize = 2;
pub const IQ3S_OFF_QH: usize = 66;
pub const IQ3S_OFF_SIGNS: usize = 74;
pub const IQ3S_OFF_SCALES: usize = 106;

/// `kmask_iq2xs` from ggml-common.h: sign bit `j` flips weight `j`.
pub const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// `iq3s_grid` verbatim from llama.cpp ggml-common.h.
pub const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

/// Scale multiplier `ls = 1 + 2*nibble` for sub-block `ib32` (0..8).
#[inline]
pub fn iq3s_ls(blk: &[u8], ib32: usize) -> u32 {
    let sc = blk[IQ3S_OFF_SCALES + ib32 / 2];
    let nib = if ib32 % 2 == 0 { sc & 0xf } else { sc >> 4 };
    1 + 2 * nib as u32
}

/// Grid index (0..512) of 4-weight group `l4` (0..8) inside sub-block
/// `ib32`: 8 low bits from `qs`, high bit `l4` of `qh[ib32]`.
#[inline]
pub fn iq3s_grid_index(blk: &[u8], ib32: usize, l4: usize) -> usize {
    let lo = blk[IQ3S_OFF_QS + 8 * ib32 + l4] as usize;
    let hi = ((blk[IQ3S_OFF_QH + ib32] as usize) >> l4) & 1;
    lo | (hi << 8)
}

/// Dequantize one 110-byte IQ3_S super-block into `out[0..256]`
/// (ggml `dequantize_row_iq3_s` semantics, same float op order).
pub fn dequant_block_iq3_s(blk: &[u8], out: &mut [f32]) {
    assert!(blk.len() >= BLOCK_IQ3_S_BYTES);
    assert!(out.len() >= QK_K);
    let d = f16_to_f32(u16::from_le_bytes([blk[IQ3S_OFF_D], blk[IQ3S_OFF_D + 1]]));
    let signs = &blk[IQ3S_OFF_SIGNS..IQ3S_OFF_SIGNS + 32];
    for ib32 in 0..8 {
        // ggml: db = d * (1 + 2*nib) with the int product converted first.
        let db = d * iq3s_ls(blk, ib32) as f32;
        for l in 0..4 {
            let g1 = IQ3S_GRID[iq3s_grid_index(blk, ib32, 2 * l)].to_le_bytes();
            let g2 = IQ3S_GRID[iq3s_grid_index(blk, ib32, 2 * l + 1)].to_le_bytes();
            let s = signs[4 * ib32 + l];
            let y = &mut out[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
            for j in 0..4 {
                let f1 = if s & KMASK_IQ2XS[j] != 0 {
                    -1.0f32
                } else {
                    1.0
                };
                let f2 = if s & KMASK_IQ2XS[j + 4] != 0 {
                    -1.0f32
                } else {
                    1.0
                };
                y[j] = db * g1[j] as f32 * f1;
                y[j + 4] = db * g2[j] as f32 * f2;
            }
        }
    }
}

/// Dequantize whole IQ3_S blocks (`src.len() % 110 == 0`) into `out`
/// (`out.len() == src.len()/110 * 256`).
pub fn dequant_row_iq3_s(src: &[u8], out: &mut [f32]) {
    assert_eq!(src.len() % BLOCK_IQ3_S_BYTES, 0, "iq3_s: partial block");
    let nb = src.len() / BLOCK_IQ3_S_BYTES;
    assert_eq!(out.len(), nb * QK_K);
    for (b, blk) in src.chunks_exact(BLOCK_IQ3_S_BYTES).enumerate() {
        dequant_block_iq3_s(blk, &mut out[b * QK_K..(b + 1) * QK_K]);
    }
}

/// Scalar CPU reference mirroring llama.cpp's
/// `ggml_vec_dot_iq3_s_q8_K_generic`: per super-block
/// `d * y.d * Σ_ib32 ls * sumi` — no prefactor.
pub fn dot_iq3_s_q8_k(n_blocks: usize, w_bytes: &[u8], y_bytes: &[u8]) -> f32 {
    assert_eq!(w_bytes.len(), n_blocks * BLOCK_IQ3_S_BYTES);
    assert_eq!(y_bytes.len(), n_blocks * BLOCK_Q8_K_BYTES);

    let mut sumf = 0.0f32;
    for b in 0..n_blocks {
        let blk = &w_bytes[b * BLOCK_IQ3_S_BYTES..(b + 1) * BLOCK_IQ3_S_BYTES];
        let y_off = b * BLOCK_Q8_K_BYTES;
        let wd = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let yd = f32::from_le_bytes([
            y_bytes[y_off],
            y_bytes[y_off + 1],
            y_bytes[y_off + 2],
            y_bytes[y_off + 3],
        ]);
        let d = wd * yd;
        let q8 = &y_bytes[y_off + 4..y_off + 4 + QK_K];
        let signs = &blk[IQ3S_OFF_SIGNS..IQ3S_OFF_SIGNS + 32];

        let mut bsum: i32 = 0;
        for ib32 in 0..8 {
            let ls = iq3s_ls(blk, ib32) as i32;
            let mut sumi: i32 = 0;
            for l in 0..4 {
                let g1 = IQ3S_GRID[iq3s_grid_index(blk, ib32, 2 * l)].to_le_bytes();
                let g2 = IQ3S_GRID[iq3s_grid_index(blk, ib32, 2 * l + 1)].to_le_bytes();
                let s = signs[4 * ib32 + l];
                let q8b = ib32 * 32 + l * 8;
                for j in 0..4 {
                    let s1 = if s & KMASK_IQ2XS[j] != 0 { -1 } else { 1 };
                    let s2 = if s & KMASK_IQ2XS[j + 4] != 0 { -1 } else { 1 };
                    sumi += g1[j] as i32 * (q8[q8b + j] as i8) as i32 * s1;
                    sumi += g2[j] as i32 * (q8[q8b + j + 4] as i8) as i32 * s2;
                }
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum as f32;
    }
    sumf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufType;

    struct Lcg(u32);
    impl Lcg {
        fn next_u8(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
            (self.0 >> 16) as u8
        }
    }

    /// Same LCG stream / d table / y.d as `tests/ref/iq3_s_gen.c`.
    const D_TABLE: [u16; 4] = [0x2e66, 0x3266, 0x2a66, 0x3466];

    fn gen_case(seed: u32, nb: usize) -> (Vec<u8>, Vec<u8>) {
        let mut lcg = Lcg(seed.wrapping_mul(7919));
        let mut w = vec![0u8; nb * BLOCK_IQ3_S_BYTES];
        for b in w.iter_mut() {
            *b = lcg.next_u8();
        }
        for b in 0..nb {
            let dd = D_TABLE[(b + seed as usize) & 3];
            w[b * BLOCK_IQ3_S_BYTES] = (dd & 0xff) as u8;
            w[b * BLOCK_IQ3_S_BYTES + 1] = (dd >> 8) as u8;
        }
        let mut y = vec![0u8; nb * BLOCK_Q8_K_BYTES];
        for b in 0..nb {
            let yd = 0.05f32 * (b + 1) as f32;
            y[b * 292..b * 292 + 4].copy_from_slice(&yd.to_le_bytes());
            for j in 0..256 {
                y[b * 292 + 4 + j] = lcg.next_u8();
            }
        }
        (w, y)
    }

    #[test]
    fn block_shape_and_grid_invariants() {
        assert_eq!(GgufType::IQ3_S.block_shape(), Some((256, 110)));
        assert_eq!(GgufType::IQ3_S.size_of(4096 * 2048).unwrap(), 3_604_480);
        assert_eq!(IQ3S_GRID.len(), 512);
        for (i, &g) in IQ3S_GRID.iter().enumerate() {
            for b in g.to_le_bytes() {
                assert!(b & 1 == 1 && (1..=15).contains(&b), "grid[{i}]={g:#010x}");
            }
        }
        // Every grid row is distinct.
        let mut sorted = IQ3S_GRID.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 512);
        assert_eq!(IQ3S_GRID[0], 0x0101_0101);
        assert_eq!(IQ3S_GRID[511], 0x0f0f_0101);
    }

    /// dequant(w) · (y.d · q8) in f64 must agree with the integer-domain dot.
    #[test]
    fn dequant_then_dot_matches_dot() {
        for seed in 1..=8u32 {
            let nb = 6;
            let (w, y) = gen_case(seed, nb);
            let mut deq = vec![0f32; nb * QK_K];
            dequant_row_iq3_s(&w, &mut deq);
            let mut acc = 0f64;
            for b in 0..nb {
                let yd = f32::from_le_bytes(y[b * 292..b * 292 + 4].try_into().unwrap()) as f64;
                for j in 0..QK_K {
                    acc += deq[b * QK_K + j] as f64 * yd * (y[b * 292 + 4 + j] as i8) as f64;
                }
            }
            let got = dot_iq3_s_q8_k(nb, &w, &y) as f64;
            let rel = (got - acc).abs() / acc.abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "seed {seed}: dot {got:e} vs dequant-dot {acc:e} (rel {rel:e})"
            );
        }
    }

    /// Every super-block's |w| ≤ 15·31·d and the sub-block max hits
    /// exactly grid_max·ls·d — catches an off-by-one in the qh/scale decode.
    #[test]
    fn dequant_magnitudes_bounded() {
        let (w, _) = gen_case(5, 4);
        let mut deq = vec![0f32; 4 * QK_K];
        dequant_row_iq3_s(&w, &mut deq);
        for b in 0..4 {
            let blk = &w[b * 110..(b + 1) * 110];
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for ib32 in 0..8 {
                let ls = iq3s_ls(blk, ib32) as f32;
                let seg = &deq[b * 256 + ib32 * 32..b * 256 + ib32 * 32 + 32];
                let mut gmax = 0u8;
                for l4 in 0..8 {
                    for g in IQ3S_GRID[iq3s_grid_index(blk, ib32, l4)].to_le_bytes() {
                        gmax = gmax.max(g);
                    }
                }
                let amax = seg.iter().fold(0f32, |m, v| m.max(v.abs()));
                assert_eq!(amax, d * ls * gmax as f32);
                assert!(amax <= 15.0 * 31.0 * d + 1e-6);
            }
        }
    }

    /// Pinned to upstream: values printed by `tests/ref/iq3_s_gen.c`
    /// (verbatim ggml_vec_dot_iq3_s_q8_K_generic + dequantize_row_iq3_s on
    /// the identical LCG stream). Dot rel < 1e-6; dequant bit-exact.
    #[test]
    fn matches_llama_cpp_reference() {
        struct Want {
            seed: u32,
            dot: f32,
            sum: f64,
            asum: f64,
            y0: u32,
            y37: u32,
            y255: u32,
            y1023: u32,
        }
        const WANT: [Want; 3] = [
            Want {
                seed: 1,
                dot: -2.962029053e+03,
                sum: 8.091676025391e+02,
                asum: 1.522871520996e+04,
                y0: 0x40865e00,
                y37: 0xbf199000,
                y255: 0xc1365b00,
                y1023: 0x40cff300,
            },
            Want {
                seed: 2,
                dot: 9.427104492e+03,
                sum: -1.133337036133e+03,
                asum: 1.515329394531e+04,
                y0: 0xc163f1c0,
                y37: 0x406ff100,
                y255: 0xc0d7f280,
                y1023: 0x40865e00,
            },
            Want {
                seed: 3,
                dot: -5.965603027e+02,
                sum: 7.584012451172e+02,
                asum: 1.627424951172e+04,
                y0: 0x41605200,
                y37: 0x4145ee00,
                y255: 0xc1085a00,
                y1023: 0xbfc65a00,
            },
        ];
        let nb = 4;
        for want in WANT {
            let (w, y) = gen_case(want.seed, nb);
            let got = dot_iq3_s_q8_k(nb, &w, &y);
            let rel = (got - want.dot).abs() / want.dot.abs();
            assert!(
                rel < 1e-6,
                "seed {}: dot {got:e} vs llama.cpp {:e} (rel {rel:e})",
                want.seed,
                want.dot
            );

            let mut deq = vec![0f32; nb * QK_K];
            dequant_row_iq3_s(&w, &mut deq);
            assert_eq!(deq[0].to_bits(), want.y0, "seed {} y[0]", want.seed);
            assert_eq!(deq[37].to_bits(), want.y37, "seed {} y[37]", want.seed);
            assert_eq!(deq[255].to_bits(), want.y255, "seed {} y[255]", want.seed);
            assert_eq!(
                deq[1023].to_bits(),
                want.y1023,
                "seed {} y[1023]",
                want.seed
            );
            let (mut sum, mut asum) = (0f64, 0f64);
            for &v in &deq {
                sum += v as f64;
                asum += v.abs() as f64;
            }
            let rs = (sum - want.sum).abs() / want.sum.abs();
            let ra = (asum - want.asum).abs() / want.asum.abs();
            assert!(
                rs < 1e-12 && ra < 1e-12,
                "seed {}: sum {sum:e}/{:e} asum {asum:e}/{:e}",
                want.seed,
                want.sum,
                want.asum
            );
        }
    }

    /// Optional realism check on range-fetched blk.26 expert bytes:
    /// `DEEPSTRIX_IQ3S_BLOB=<file>` (whole 110-byte blocks). Prints
    /// min/max/mean/NaN stats with `--nocapture`; skipped when unset.
    #[test]
    fn real_expert_blob_sanity() {
        let Ok(path) = std::env::var("DEEPSTRIX_IQ3S_BLOB") else {
            eprintln!("DEEPSTRIX_IQ3S_BLOB unset; skipping");
            return;
        };
        let bytes = std::fs::read(&path).unwrap();
        let nb = bytes.len() / BLOCK_IQ3_S_BYTES;
        assert_eq!(bytes.len() % BLOCK_IQ3_S_BYTES, 0);
        let mut out = vec![0f32; nb * QK_K];
        dequant_row_iq3_s(&bytes, &mut out);
        let (mut mn, mut mx, mut sum, mut asum, mut nan) =
            (f32::INFINITY, f32::NEG_INFINITY, 0f64, 0f64, 0usize);
        for &v in &out {
            if v.is_nan() || v.is_infinite() {
                nan += 1;
                continue;
            }
            mn = mn.min(v);
            mx = mx.max(v);
            sum += v as f64;
            asum += v.abs() as f64;
        }
        let mut dmax = 0f32;
        for blk in bytes.chunks_exact(BLOCK_IQ3_S_BYTES) {
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            assert!(d.is_finite() && d >= 0.0, "block d={d}");
            dmax = dmax.max(d);
        }
        eprintln!(
            "{path}: blocks={nb} weights={} min={mn:e} max={mx:e} mean={:e} mean|w|={:e} d_max={dmax:e} nan/inf={nan}",
            out.len(),
            sum / out.len() as f64,
            asum / out.len() as f64
        );
        assert_eq!(nan, 0);
        assert!(mx.abs() <= 15.0 * 31.0 * dmax);
    }
}
