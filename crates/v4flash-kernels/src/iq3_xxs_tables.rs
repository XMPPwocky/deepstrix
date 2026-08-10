//! IQ3_XXS codebook + scalar CPU dot reference.
//!
//! Grid extracted verbatim from llama.cpp ggml-common.h; dot mirrors
//! ggml_vec_dot_iq3_xxs_q8_K_generic (ggml-cpu/quants.c). Shares
//! ksigns/kmask with [`crate::iq2_xxs_tables`].

pub const IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

use crate::iq2_xxs_tables::{f16_to_f32, KMASK_IQ2XS, KSIGNS_IQ2XS};

/// Block size of one IQ3_XXS super-block: f16 d + 64 grid-index bytes +
/// 8 packed u32 (4×7-bit sign indices + 4-bit scale each).
pub const BLOCK_IQ3_XXS_BYTES: usize = 98;

/// Scalar CPU reference mirroring llama.cpp's
/// `ggml_vec_dot_iq3_xxs_q8_K_generic`: per super-block
/// `0.25 * d * Σ_ib32 ls * sumi`, signs from `ksigns_iq2xs`.
pub fn cpu_dot_iq3_xxs_q8_k(n_blocks: usize, w_bytes: &[u8], y_bytes: &[u8]) -> f32 {
    assert_eq!(w_bytes.len(), n_blocks * BLOCK_IQ3_XXS_BYTES);
    assert_eq!(y_bytes.len(), n_blocks * 292);

    let mut sumf = 0.0f32;
    for b in 0..n_blocks {
        let w_off = b * BLOCK_IQ3_XXS_BYTES;
        let y_off = b * 292;

        let wd = f16_to_f32(u16::from_le_bytes([w_bytes[w_off], w_bytes[w_off + 1]]));
        let yd = f32::from_le_bytes([
            y_bytes[y_off],
            y_bytes[y_off + 1],
            y_bytes[y_off + 2],
            y_bytes[y_off + 3],
        ]);
        let d = wd * yd;

        let q3 = &w_bytes[w_off + 2..w_off + 2 + 64];
        let gas = &w_bytes[w_off + 2 + 64..w_off + 98];
        let q8 = &y_bytes[y_off + 4..y_off + 4 + 256];

        let mut bsum: i32 = 0;
        for ib32 in 0..8 {
            let aux32 = u32::from_le_bytes([
                gas[4 * ib32],
                gas[4 * ib32 + 1],
                gas[4 * ib32 + 2],
                gas[4 * ib32 + 3],
            ]);
            let ls = (2 * (aux32 >> 28) + 1) as i32;
            let mut sumi: i32 = 0;
            for l in 0..4 {
                let g1 = IQ3XXS_GRID[q3[ib32 * 8 + 2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[q3[ib32 * 8 + 2 * l + 1] as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                for j in 0..4 {
                    let s1 = if signs & KMASK_IQ2XS[j] != 0 { -1 } else { 1 };
                    let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1 } else { 1 };
                    let q8b = ib32 * 32 + l * 8;
                    sumi += g1[j] as i32 * (q8[q8b + j] as i8) as i32 * s1;
                    sumi += g2[j] as i32 * (q8[q8b + j + 4] as i8) as i32 * s2;
                }
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum as f32;
    }
    0.25f32 * sumf
}
