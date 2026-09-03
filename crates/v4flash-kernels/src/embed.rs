//! Host-side token-embedding row lookup, dtype-aware.
//!
//! token_embd stays host-resident (the 1.06 GB dGPU copy was removed in
//! M57); the server/CLI look rows up per token. antirez stores F16;
//! unsloth UD stores Q4_K (298 MB host — a 3.6× RAM saving); UD-Q2_K_XL
//! Q5_K, UD-IQ3_XXS Q6_K. Lifted here
//! from the duplicated `deepstrix-server/src/embed.rs` /
//! `chat.rs` helpers per their own todo-comment.

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::GgufType;

use crate::config::{HC_DIM, N_EMBD, N_HC};
use crate::iq2_xxs_tables::f16_to_f32;

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Dequantize one 144-byte Q4_K superblock into 256 f32 (ggml
/// `dequantize_row_q4_K` semantics).
pub fn dequant_q4k_superblock(blk: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
    let scales = &blk[4..16];
    let qs = &blk[16..144];
    for g in 0..4 {
        let (sc1, m1) = get_scale_min_k4(2 * g, scales);
        let (sc2, m2) = get_scale_min_k4(2 * g + 1, scales);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;
        for l in 0..32 {
            let q = qs[g * 32 + l];
            out[g * 64 + l] = d1 * (q & 0x0F) as f32 - min1;
            out[g * 64 + 32 + l] = d2 * (q >> 4) as f32 - min2;
        }
    }
}

/// Fill `out` (length `HC_DIM`) with token `token_id`'s embedding row,
/// broadcast across the `N_HC` hyper-connection channels.
/// Dequantize one 176-byte Q5_K superblock into 256 f32 (ggml
/// `dequantize_row_q5_K`). Same packed 6-bit (scale, min) scheme as Q4_K
/// plus a 5th bit plane; layout is pinned by q5_k_dense_matvec.hip:
///   0: f16 d | 2: f16 dmin | 4: scales[12] | 16: qh[32] | 48: qs[128]
pub fn dequant_q5k_superblock(blk: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
    let scales = &blk[4..16];
    let qh = &blk[16..48];
    let qs = &blk[48..176];
    for g in 0..4 {
        let (sc1, m1) = get_scale_min_k4(2 * g, scales);
        let (sc2, m2) = get_scale_min_k4(2 * g + 1, scales);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;
        let u1 = 1u8 << (2 * g);
        let u2 = 1u8 << (2 * g + 1);
        for l in 0..32 {
            let q = qs[g * 32 + l];
            let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            out[g * 64 + l] = d1 * ((q & 0x0F) as i32 + hi1) as f32 - min1;
            out[g * 64 + 32 + l] = d2 * ((q >> 4) as i32 + hi2) as f32 - min2;
        }
    }
}

/// Dequantize one 210-byte Q6_K superblock into 256 f32 (ggml
/// `dequantize_row_q6_K`). Layout (pinned by q6_k_dense_matvec.hip):
///   0: ql[128] | 128: qh[64] | 192: scales[16] (i8) | 208: f16 d
/// Each 128-weight half: q = (ql nibble | 2 qh bits << 4) - 32, four
/// 32-weight strips per half with their own i8 scale.
pub fn dequant_q6k_superblock(blk: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
    for n in 0..2 {
        let ql = &blk[n * 64..n * 64 + 64];
        let qh = &blk[128 + n * 32..128 + n * 32 + 32];
        let sc = &blk[192 + n * 8..192 + n * 8 + 8];
        let y = &mut out[n * 128..n * 128 + 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

pub fn embed_lookup(
    token_embd_bytes: &[u8],
    dtype: GgufType,
    token_id: i32,
    out: &mut [f32],
) -> eyre::Result<()> {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    assert_eq!(out.len(), HC_DIM as usize);
    assert_eq!(out.len(), n_embd * n_hc);
    match dtype {
        GgufType::F16 => {
            let row_off = (token_id as usize) * n_embd * 2;
            let row = token_embd_bytes
                .get(row_off..row_off + n_embd * 2)
                .ok_or_else(|| eyre!("embed_lookup: token {token_id} out of range"))?;
            for i in 0..n_embd {
                out[i] = f16_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
            }
        }
        GgufType::Q4_K => {
            let sb = n_embd / 256; // 16 superblocks per row
            let row_bytes = sb * 144;
            let row_off = (token_id as usize) * row_bytes;
            let row = token_embd_bytes
                .get(row_off..row_off + row_bytes)
                .ok_or_else(|| eyre!("embed_lookup: token {token_id} out of range"))?;
            for s in 0..sb {
                dequant_q4k_superblock(&row[s * 144..(s + 1) * 144], &mut out[s * 256..(s + 1) * 256]);
            }
        }
        GgufType::Q5_K => {
            let sb = n_embd / 256; // 16 superblocks per row
            let row_bytes = sb * 176;
            let row_off = (token_id as usize) * row_bytes;
            let row = token_embd_bytes
                .get(row_off..row_off + row_bytes)
                .ok_or_else(|| eyre!("embed_lookup: token {token_id} out of range"))?;
            for s in 0..sb {
                dequant_q5k_superblock(&row[s * 176..(s + 1) * 176], &mut out[s * 256..(s + 1) * 256]);
            }
        }
        GgufType::Q6_K => {
            let sb = n_embd / 256; // 16 superblocks per row
            let row_bytes = sb * 210;
            let row_off = (token_id as usize) * row_bytes;
            let row = token_embd_bytes
                .get(row_off..row_off + row_bytes)
                .ok_or_else(|| eyre!("embed_lookup: token {token_id} out of range"))?;
            for s in 0..sb {
                dequant_q6k_superblock(&row[s * 210..(s + 1) * 210], &mut out[s * 256..(s + 1) * 256]);
            }
        }
        other => return Err(eyre!("embed_lookup: unsupported token_embd dtype {other:?}")),
    }
    for h in 1..n_hc {
        let (head, tail) = out.split_at_mut(h * n_embd);
        let src = &head[0..n_embd];
        tail[0..n_embd].copy_from_slice(src);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u32);
    impl Lcg {
        fn next_u8(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
            (self.0 >> 16) as u8
        }
    }

    /// Hand-built Q6_K block: all-zero ql/qh except a few probe positions,
    /// checked against the closed-form `d * sc * (q - 32)`.
    #[test]
    fn q6k_superblock_formula() {
        let mut blk = [0u8; 210];
        blk[208..210].copy_from_slice(&crate::weight_contract::f32_to_f16_bits(0.5).to_le_bytes());
        for (i, s) in blk[192..208].iter_mut().enumerate() {
            *s = (i as i8 * 3 - 20) as u8; // signed scales -20..25
        }
        // half 0, l=5: low nibble 0xA, qh bits 0..1 = 0b11 -> q1 = 0x3A - 32 = 26
        blk[5] = 0x0A;
        blk[128 + 5] = 0b11;
        // half 1, l=17: ql[l+32] high nibble 0x7, qh bits 6..7 = 0b10 -> q4 = 0x27 - 32 = 7
        blk[64 + 17 + 32] = 0x70;
        blk[128 + 32 + 17] = 0b10 << 6;
        let mut out = [0f32; 256];
        dequant_q6k_superblock(&blk, &mut out);
        let sc = |i: usize| (blk[192 + i] as i8) as f32;
        assert_eq!(out[5], 0.5 * sc(0) * 26.0);
        // the other three strips at l=5 in half 0 read q=-32 (zero bits)
        assert_eq!(out[5 + 32], 0.5 * sc(2) * -32.0);
        assert_eq!(out[5 + 64], 0.5 * sc(4) * -32.0);
        assert_eq!(out[5 + 96], 0.5 * sc(6) * -32.0);
        // half 1, strip 4 (y[l+96]), l=17 -> is = 1 -> scale index 8 + 6 + 1
        assert_eq!(out[128 + 17 + 96], 0.5 * sc(8 + 6 + 1) * 7.0);
        assert_eq!(out[128 + 17], 0.5 * sc(8 + 1) * -32.0);
    }

    /// Random blocks: bit-identical to v4flash-core's independent port of
    /// `dequantize_row_q6_K` (`kquants::dequant_to_f32`, used by the
    /// dumper variant).
    #[test]
    fn q6k_superblock_matches_core_kquants() {
        let mut lcg = Lcg(0x6b);
        for _ in 0..16 {
            let mut blk = [0u8; 210];
            for b in blk.iter_mut() {
                *b = lcg.next_u8();
            }
            // sane d (~0.02); the random f16 could be inf/nan otherwise
            blk[208] = 0x1f;
            blk[209] = 0x25;
            let mut ours = [0f32; 256];
            dequant_q6k_superblock(&blk, &mut ours);
            let mut core = Vec::new();
            v4flash_core::kquants::dequant_to_f32(GgufType::Q6_K, &blk, &mut core).unwrap();
            assert_eq!(core.len(), 256);
            for i in 0..256 {
                assert_eq!(ours[i].to_bits(), core[i].to_bits(), "elem {i}");
            }
        }
    }

    /// Q6_K row lookup: 16 superblocks per row, broadcast across N_HC.
    #[test]
    fn embed_lookup_q6k_row_and_broadcast() {
        let n_embd = N_EMBD as usize;
        let sb = n_embd / 256;
        let row_bytes = sb * 210;
        let n_tok = 3;
        let mut lcg = Lcg(0x51);
        let mut table = vec![0u8; n_tok * row_bytes];
        for b in table.iter_mut() {
            *b = lcg.next_u8();
        }
        for blk in table.chunks_exact_mut(210) {
            blk[208] = 0x1f;
            blk[209] = 0x25;
        }
        let mut out = vec![0f32; HC_DIM as usize];
        embed_lookup(&table, GgufType::Q6_K, 2, &mut out).unwrap();
        let mut want = vec![0f32; n_embd];
        for s in 0..sb {
            dequant_q6k_superblock(
                &table[2 * row_bytes + s * 210..2 * row_bytes + (s + 1) * 210],
                &mut want[s * 256..(s + 1) * 256],
            );
        }
        assert_eq!(&out[..n_embd], &want[..]);
        for h in 1..N_HC as usize {
            assert_eq!(&out[h * n_embd..(h + 1) * n_embd], &want[..], "hc {h}");
        }
        assert!(embed_lookup(&table, GgufType::Q6_K, 3, &mut out).is_err());
    }
}
