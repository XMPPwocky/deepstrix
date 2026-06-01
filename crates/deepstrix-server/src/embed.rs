//! Token-embedding lookup + GPT-2 byte decoder, duplicated from
//! `crates/deepstrix-cli/src/bin/chat.rs:265-339`.
//!
//! This is intentional: the chat REPL and the server share these
//! helpers verbatim, but we don't want a cross-crate dependency from
//! `deepstrix-server` → `deepstrix-cli`. If the helpers grow, lift them
//! into `v4flash-kernels` or a shared `deepstrix-common` crate at that
//! point.

use v4flash_kernels::config::{HC_DIM, N_EMBD, N_HC};

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let s: u32 = (sign as u32) << 31;
    let f32_bits: u32 = match exp {
        0 if mant == 0 => s,
        0 => {
            let mantissa = mant as f32 / 1024.0;
            let v = mantissa * (1.0 / (1u64 << 14) as f32);
            return if sign == 1 { -v } else { v };
        }
        0x1f => s | 0x7f800000 | ((mant as u32) << 13),
        _ => s | ((exp as u32 + 112) << 23) | ((mant as u32) << 13),
    };
    f32::from_bits(f32_bits)
}

/// Fill `out` (length `HC_DIM`) with the token-`token_id` row of the F16
/// embedding table, broadcast across all `N_HC` HC channels.
pub fn embed_lookup(token_embd_bytes: &[u8], token_id: i32, out: &mut [f32]) {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    assert_eq!(out.len(), HC_DIM as usize);
    assert_eq!(out.len(), n_embd * n_hc);
    let row_off = (token_id as usize) * n_embd * 2;
    for i in 0..n_embd {
        let b0 = token_embd_bytes[row_off + i * 2];
        let b1 = token_embd_bytes[row_off + i * 2 + 1];
        let bits = u16::from_le_bytes([b0, b1]);
        out[i] = f16_to_f32(bits);
    }
    for h in 1..n_hc {
        let (head, tail) = out.split_at_mut(h * n_embd);
        let src = &head[0..n_embd];
        let dst = &mut tail[0..n_embd];
        dst.copy_from_slice(src);
    }
}

/// Build the GPT-2 byte ↔ printable-char mapping table used to decode
/// token text from the tokenizer's encoded form to raw UTF-8 bytes.
/// Mirrors `chat.rs:302-339` / OpenAI tiktoken / HuggingFace tokenizers'
/// `bytes_to_unicode`.
pub fn build_gpt2_byte_decoder() -> std::collections::HashMap<char, u8> {
    let printable: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1u8..=0xACu8)
        .chain(0xAEu8..=0xFFu8)
        .collect();
    let mut bs: Vec<u8> = printable.clone();
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;
    for b in 0u8..=255 {
        if !printable.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut m = std::collections::HashMap::with_capacity(256);
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        if let Some(ch) = char::from_u32(c) {
            m.insert(ch, b);
        }
    }
    m
}

/// Reverse the GPT-2 byte encoding for a single token's text. The
/// tokenizer stores each byte 0..256 as a printable codepoint; this
/// turns those codepoints back into the original bytes.
pub fn gpt2_decode_token(token_bytes: &[u8], dec: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    let s = std::str::from_utf8(token_bytes).unwrap_or("");
    let mut out = Vec::with_capacity(token_bytes.len());
    for ch in s.chars() {
        if let Some(&b) = dec.get(&ch) {
            out.push(b);
        } else {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            out.extend_from_slice(s.as_bytes());
        }
    }
    out
}
