//! MXFP4 value table + scalar CPU dot reference vs Q8_K activations.
//!
//! Mirrors llama.cpp's kvalues_mxfp4 / e8m0 semantics; the engine's MoE
//! path uses Q8_K (block-256, f32 d) activations rather than the Q8_0 of
//! `ggml_vec_dot_mxfp4_q8_0`, so the reference here scales per 32-elem
//! mxfp4 block inside each 256-elem superblock.

pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Per-superblock (256 elems) byte count: 8 × 17.
///
/// LAYOUT v2 (struct-of-arrays *within* the super-block): the 136 bytes are
/// `[8 × 16 B nibbles][8 B scales]` — block `b`'s nibbles at `b*16`, its scale
/// at `128 + b`. v1 interleaved them as 8 × `[scale][16 B nibbles]`, which put
/// every nibble run at an odd byte offset and forced byte-at-a-time loads in
/// the matvecs. Total size and row stride are unchanged, so expert byte counts,
/// pool slots and snapshots are untouched — but the two layouts are NOT
/// interchangeable, and box 1 / box 2 must agree (see MXFP4_LAYOUT_VERSION).
pub const SUPER_MXFP4_BYTES: usize = 136;
/// Bytes per 32-elem block, as a *row* stride only (`nb * 17` per row). It is
/// no longer a stride *within* a super-block — use the v2 accessors instead.
pub const BLOCK_MXFP4_BYTES: usize = 17;
/// Bumped whenever the packed layout changes. Both boxes repack independently,
/// so a rolling deploy across a bump silently corrupts -- one side would decode
/// the other's bytes with the wrong layout and report nothing. The guard is the
/// remote-expert protocol version, which is bumped in step with this and is
/// checked on every frame; see `het::remote_experts::proto::VERSION`.
pub const MXFP4_LAYOUT_VERSION: u32 = 2;

/// Byte offset of block `b`'s 16 nibble bytes within a super-block.
#[inline]
pub const fn mxfp4_nib_off(b: usize) -> usize { b * 16 }
/// Byte offset of block `b`'s E8M0 scale within a super-block.
#[inline]
pub const fn mxfp4_scale_off(b: usize) -> usize { 128 + b }

/// ggml_e8m0_to_fp32_half: 2^(e-128) with denormal handling for e < 2.
pub fn e8m0_half_to_f32(e: u8) -> f32 {
    let bits: u32 = if e < 2 {
        0x0020_0000u32 << e
    } else {
        ((e as u32) - 1) << 23
    };
    f32::from_bits(bits)
}

/// CPU reference dot: `n_super` 256-elem superblocks of MXFP4 weights
/// against Q8_K activation blocks (292 B each: f32 d + 256×i8 + bsums).
pub fn cpu_dot_mxfp4_q8_k(n_super: usize, w_bytes: &[u8], y_bytes: &[u8]) -> f32 {
    assert_eq!(w_bytes.len(), n_super * SUPER_MXFP4_BYTES);
    assert_eq!(y_bytes.len(), n_super * 292);
    let mut sumf = 0.0f32;
    for s in 0..n_super {
        let w = &w_bytes[s * SUPER_MXFP4_BYTES..(s + 1) * SUPER_MXFP4_BYTES];
        let y = &y_bytes[s * 292..(s + 1) * 292];
        let yd = f32::from_le_bytes([y[0], y[1], y[2], y[3]]);
        let q8 = &y[4..4 + 256];
        let mut acc = 0.0f32;
        for b in 0..8 {
            let scale = e8m0_half_to_f32(w[mxfp4_scale_off(b)]);
            let qs = &w[mxfp4_nib_off(b)..mxfp4_nib_off(b) + 16];
            let mut sumi: i32 = 0;
            for j in 0..16 {
                let lo = KVALUES_MXFP4[(qs[j] & 0x0F) as usize] as i32;
                let hi = KVALUES_MXFP4[(qs[j] >> 4) as usize] as i32;
                sumi += lo * (q8[b * 32 + j] as i8) as i32;
                sumi += hi * (q8[b * 32 + 16 + j] as i8) as i32;
            }
            acc += scale * sumi as f32;
        }
        sumf += yd * acc;
    }
    sumf
}
