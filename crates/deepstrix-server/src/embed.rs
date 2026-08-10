//! Token-embedding lookup + GPT-2 byte decoder, duplicated from
//! `crates/deepstrix-cli/src/bin/chat.rs:265-339`.
//!
//! This is intentional: the chat REPL and the server share these
//! helpers verbatim, but we don't want a cross-crate dependency from
//! `deepstrix-server` → `deepstrix-cli`. If the helpers grow, lift them
//! into `v4flash-kernels` or a shared `deepstrix-common` crate at that
//! point.


/// Dtype-aware row lookup — delegates to the shared implementation.
pub fn embed_lookup(
    token_embd_bytes: &[u8],
    dtype: v4flash_core::gguf::GgufType,
    token_id: i32,
    out: &mut [f32],
) {
    v4flash_kernels::embed::embed_lookup(token_embd_bytes, dtype, token_id, out)
        .expect("embed_lookup: validated at load");
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
