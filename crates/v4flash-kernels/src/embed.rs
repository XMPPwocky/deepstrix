//! Host-side token-embedding row lookup, dtype-aware.
//!
//! token_embd stays host-resident (the 1.06 GB dGPU copy was removed in
//! M57); the server/CLI look rows up per token. antirez stores F16;
//! unsloth UD stores Q4_K (298 MB host — a 3.6× RAM saving). Lifted here
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
        other => return Err(eyre!("embed_lookup: unsupported token_embd dtype {other:?}")),
    }
    for h in 1..n_hc {
        let (head, tail) = out.split_at_mut(h * n_embd);
        let src = &head[0..n_embd];
        tail[0..n_embd].copy_from_slice(src);
    }
    Ok(())
}
