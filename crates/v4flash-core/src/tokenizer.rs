//! V4-Flash-compatible BPE tokenizer.
//!
//! Ported from `external/ds4/ds4.c` lines 14258-14570 (byte encoding,
//! BPE merge loop, joyai-llm pre-tokenizer). The reference is C; the
//! Rust here mirrors the algorithm step-by-step without introducing a
//! regex dependency.
//!
//! V4 Flash uses three stages:
//!   1. **Pre-tokenize** input text into pieces using joyai-llm char-class
//!      rules. Pieces are byte sub-ranges of the input.
//!   2. **Byte-encode** each piece using the GPT-2 byte-to-codepoint table,
//!      mapping non-printable bytes into the U+0100..U+0173 range so BPE
//!      can operate on valid UTF-8 strings.
//!   3. **BPE merge loop**: split encoded piece into UTF-8 chars; repeatedly
//!      find adjacent pair with the lowest merge rank and merge; final
//!      symbols are looked up in `token_to_id`.

use std::collections::HashMap;

use color_eyre::eyre::{self, eyre};

use crate::gguf::{Gguf, GgufValue};

/// Loaded BPE vocab: tokens, merge ranks, special token IDs.
pub struct BpeVocab {
    /// id → owned token bytes (e.g. "hello" or the encoded form of " world")
    pub tokens: Vec<Vec<u8>>,
    /// token bytes → id
    token_to_id: HashMap<Vec<u8>, i32>,
    /// "<a> <b>" merge string → rank (lower rank = applied first)
    merge_rank: HashMap<Vec<u8>, i32>,
    /// Special tokens by canonical name from GGUF metadata.
    pub bos_id: Option<i32>,
    pub eos_id: Option<i32>,
    pub unknown_id: Option<i32>,
    pub padding_id: Option<i32>,
}

impl BpeVocab {
    /// Load vocab + merges from a parsed `Gguf`. Expects:
    ///   tokenizer.ggml.tokens   (string array)
    ///   tokenizer.ggml.merges   (string array, space-separated pairs)
    pub fn from_gguf(g: &Gguf) -> eyre::Result<Self> {
        let tokens_val = g
            .metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| eyre!("missing tokenizer.ggml.tokens"))?;
        let merges_val = g
            .metadata("tokenizer.ggml.merges")
            .ok_or_else(|| eyre!("missing tokenizer.ggml.merges"))?;

        let GgufValue::Array(tokens_arr) = tokens_val else {
            return Err(eyre!("tokenizer.ggml.tokens is not an array"));
        };
        let tokens_slice = tokens_arr
            .as_strings()
            .ok_or_else(|| eyre!("tokenizer.ggml.tokens is not a string array"))?;

        let mut tokens: Vec<Vec<u8>> = Vec::with_capacity(tokens_slice.len());
        let mut token_to_id: HashMap<Vec<u8>, i32> = HashMap::with_capacity(tokens_slice.len());
        for (i, s) in tokens_slice.iter().enumerate() {
            let bytes = s.as_bytes().to_vec();
            // Token strings are unique per spec; if not, last-write-wins
            // matches ds4's behavior (it also overwrites).
            token_to_id.insert(bytes.clone(), i as i32);
            tokens.push(bytes);
        }

        let GgufValue::Array(merges_arr) = merges_val else {
            return Err(eyre!("tokenizer.ggml.merges is not an array"));
        };
        let merges_slice = merges_arr
            .as_strings()
            .ok_or_else(|| eyre!("tokenizer.ggml.merges is not a string array"))?;
        let mut merge_rank: HashMap<Vec<u8>, i32> = HashMap::with_capacity(merges_slice.len());
        for (rank, m) in merges_slice.iter().enumerate() {
            // Merge keys are stored as raw bytes "left<space>right".
            merge_rank.insert(m.as_bytes().to_vec(), rank as i32);
        }

        let bos_id = g.metadata("tokenizer.ggml.bos_token_id").and_then(|v| v.as_u32()).map(|v| v as i32);
        let eos_id = g.metadata("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u32()).map(|v| v as i32);
        let unknown_id = g.metadata("tokenizer.ggml.unknown_token_id").and_then(|v| v.as_u32()).map(|v| v as i32);
        let padding_id = g.metadata("tokenizer.ggml.padding_token_id").and_then(|v| v.as_u32()).map(|v| v as i32);

        Ok(BpeVocab {
            tokens,
            token_to_id,
            merge_rank,
            bos_id,
            eos_id,
            unknown_id,
            padding_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Encode an input string into token IDs using the joyai pre-tokenizer
    /// + GPT-2 byte encoding + BPE merge.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut out = Vec::new();
        for piece in joyai_pre_tokenize(text.as_bytes()) {
            self.bpe_emit_piece(piece, &mut out);
        }
        out
    }

    fn bpe_emit_piece(&self, raw_piece: &[u8], out: &mut Vec<i32>) {
        // Step 1: byte-encode raw bytes into printable UTF-8.
        let encoded = byte_encode(raw_piece);

        // Step 2: split encoded into UTF-8 char "symbols".
        let mut sym: Vec<Vec<u8>> = Vec::new();
        let mut off = 0;
        while off < encoded.len() {
            let n = utf8_len_from_first_byte(encoded[off]);
            let end = (off + n).min(encoded.len());
            sym.push(encoded[off..end].to_vec());
            off = end;
        }

        // Step 3: greedy BPE merge loop. O(n²) per piece; pieces are
        // short (joyai pre-tok keeps them small) so this is fine.
        loop {
            let mut best_i: Option<usize> = None;
            let mut best_rank = i32::MAX;
            for i in 0..sym.len().saturating_sub(1) {
                if let Some(rank) = self.pair_rank(&sym[i], &sym[i + 1]) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }
            let Some(i) = best_i else { break };
            let mut merged = sym[i].clone();
            merged.extend_from_slice(&sym[i + 1]);
            sym[i] = merged;
            sym.remove(i + 1);
        }

        // Step 4: look up each final symbol; if not in vocab, fall back
        // to byte-by-byte lookup (matches ds4 lines 14376-14388).
        for piece in &sym {
            if let Some(&id) = self.token_to_id.get(piece) {
                out.push(id);
                continue;
            }
            for &b in piece {
                if let Some(&id) = self.token_to_id.get(&vec![b]) {
                    out.push(id);
                }
                // Note: if even single-byte lookup fails, ds4 drops the
                // byte silently. We mirror that here — TODO: warn?
            }
        }
    }

    fn pair_rank(&self, a: &[u8], b: &[u8]) -> Option<i32> {
        // Key format: "<a><space><b>" (raw bytes, not escaped)
        let mut key = Vec::with_capacity(a.len() + 1 + b.len());
        key.extend_from_slice(a);
        key.push(b' ');
        key.extend_from_slice(b);
        self.merge_rank.get(&key).copied()
    }

    pub fn token_text(&self, id: i32) -> Option<&[u8]> {
        let idx: usize = id.try_into().ok()?;
        self.tokens.get(idx).map(Vec::as_slice)
    }
}

// ---- byte encoding (GPT-2 byte → codepoint) ----

/// GPT-2 byte-to-codepoint: printable ASCII maps to itself, non-printable
/// bytes are remapped into U+0100..U+0173 so BPE can operate on valid
/// UTF-8 strings. Mirrors ds4's `gpt2_byte_to_codepoint`.
fn gpt2_byte_to_codepoint(b: u8) -> u32 {
    if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 {
        return b as u32;
    }
    let mut n = 0u32;
    for x in 0..=255u32 {
        if (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174 {
            continue;
        }
        if x == b as u32 {
            return 256 + n;
        }
        n += 1;
    }
    b as u32
}

/// Encode raw bytes into a printable-UTF-8 string per GPT-2 convention.
fn byte_encode(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() * 4);
    for &b in raw {
        utf8_put(&mut out, gpt2_byte_to_codepoint(b));
    }
    out
}

fn utf8_put(out: &mut Vec<u8>, cp: u32) {
    if cp <= 0x7f {
        out.push(cp as u8);
    } else if cp <= 0x7ff {
        out.push(0xc0 | (cp >> 6) as u8);
        out.push(0x80 | (cp & 0x3f) as u8);
    } else if cp <= 0xffff {
        out.push(0xe0 | (cp >> 12) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3f) as u8);
        out.push(0x80 | (cp & 0x3f) as u8);
    } else {
        out.push(0xf0 | (cp >> 18) as u8);
        out.push(0x80 | ((cp >> 12) & 0x3f) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3f) as u8);
        out.push(0x80 | (cp & 0x3f) as u8);
    }
}

fn utf8_len_from_first_byte(c: u8) -> usize {
    if c < 0x80 { 1 }
    else if c & 0xe0 == 0xc0 { 2 }
    else if c & 0xf0 == 0xe0 { 3 }
    else if c & 0xf8 == 0xf0 { 4 }
    else { 1 }
}

fn utf8_peek_one(s: &[u8], pos: usize) -> (u32, usize) {
    let c0 = s[pos];
    let mut n = utf8_len_from_first_byte(c0);
    if pos + n > s.len() {
        n = 1;
    }
    let cp = match n {
        1 => c0 as u32,
        2 => ((c0 as u32 & 0x1f) << 6) | (s[pos + 1] as u32 & 0x3f),
        3 => ((c0 as u32 & 0x0f) << 12)
            | ((s[pos + 1] as u32 & 0x3f) << 6)
            | (s[pos + 2] as u32 & 0x3f),
        _ => ((c0 as u32 & 0x07) << 18)
            | ((s[pos + 1] as u32 & 0x3f) << 12)
            | ((s[pos + 2] as u32 & 0x3f) << 6)
            | (s[pos + 3] as u32 & 0x3f),
    };
    (cp, pos + n)
}

fn next_utf8_char(s: &[u8], pos: usize) -> usize {
    let n = utf8_len_from_first_byte(s[pos]);
    (pos + n).min(s.len()).max(pos + 1)
}

// ---- joyai pre-tokenizer ----

fn ascii_alpha(c: u8) -> bool { (b'A'..=b'Z').contains(&c) || (b'a'..=b'z').contains(&c) }
fn ascii_digit(c: u8) -> bool { (b'0'..=b'9').contains(&c) }
fn ascii_space(c: u8) -> bool {
    c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == 0x0b || c == 0x0c
}
fn ascii_newline(c: u8) -> bool { c == b'\n' || c == b'\r' }

fn joyai_ascii_punct_symbol(c: u8) -> bool {
    (c >= b'!' && c <= b'/')
        || (c >= b':' && c <= b'@')
        || (c >= b'[' && c <= b'`')
        || (c >= b'{' && c <= b'~')
}

fn utf8_is_cjk_hira_kata(cp: u32) -> bool {
    (0x4e00..=0x9fa5).contains(&cp)
        || (0x3040..=0x309f).contains(&cp)
        || (0x30a0..=0x30ff).contains(&cp)
}

fn joyai_letter_like_at(s: &[u8], pos: usize) -> bool {
    let c = s[pos];
    if c < 128 {
        return ascii_alpha(c);
    }
    // Non-ASCII non-control bytes treated as letters (matches ds4 14452-14466).
    true
}

fn joyai_consume_letters(s: &[u8], mut pos: usize) -> usize {
    while pos < s.len() && joyai_letter_like_at(s, pos) {
        pos = next_utf8_char(s, pos);
    }
    pos
}

fn joyai_cjk_at(s: &[u8], pos: usize) -> bool {
    if s[pos] < 128 {
        return false;
    }
    let (cp, _) = utf8_peek_one(s, pos);
    utf8_is_cjk_hira_kata(cp)
}

/// joyai-llm pre-tokenizer. Returns an iterator-style Vec of byte
/// sub-ranges (each piece is a slice of `text`). Port of ds4's
/// `bpe_tokenize_text` line 14502 through 14569 (only the split, not
/// the bpe_emit step).
pub fn joyai_pre_tokenize(text: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let len = text.len();
    let mut pos = 0;
    while pos < len {
        let start = pos;
        let c = text[pos];

        if ascii_digit(c) {
            let mut n = 0;
            while pos < len && ascii_digit(text[pos]) && n < 3 {
                pos += 1;
                n += 1;
            }
        } else if joyai_cjk_at(text, pos) {
            loop {
                pos = next_utf8_char(text, pos);
                if pos >= len || !joyai_cjk_at(text, pos) {
                    break;
                }
            }
        } else if joyai_ascii_punct_symbol(c)
            && pos + 1 < len
            && ascii_alpha(text[pos + 1])
        {
            pos += 1;
            while pos < len && ascii_alpha(text[pos]) {
                pos += 1;
            }
        } else if joyai_letter_like_at(text, pos) {
            pos = joyai_consume_letters(text, pos);
        } else if !ascii_newline(c)
            && !joyai_ascii_punct_symbol(c)
            && pos + 1 < len
            && joyai_letter_like_at(text, pos + 1)
        {
            pos += 1;
            pos = joyai_consume_letters(text, pos);
        } else if c == b' '
            && pos + 1 < len
            && joyai_ascii_punct_symbol(text[pos + 1])
        {
            pos += 1;
            while pos < len && joyai_ascii_punct_symbol(text[pos]) {
                pos += 1;
            }
            while pos < len && ascii_newline(text[pos]) {
                pos += 1;
            }
        } else if joyai_ascii_punct_symbol(c) {
            while pos < len && joyai_ascii_punct_symbol(text[pos]) {
                pos += 1;
            }
            while pos < len && ascii_newline(text[pos]) {
                pos += 1;
            }
        } else if ascii_space(c) {
            let mut p = pos;
            let mut last_newline_end = 0usize;
            while p < len && ascii_space(text[p]) {
                let sc = text[p];
                p += 1;
                if ascii_newline(sc) {
                    last_newline_end = p;
                }
            }
            if last_newline_end != 0 {
                pos = last_newline_end;
            } else if p < len
                && p > pos + 1
                && (joyai_letter_like_at(text, p) || joyai_ascii_punct_symbol(text[p]))
            {
                // Leading single space joins the next word/punct run:
                // "    int" → "   " then " int", not "    " then "int".
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, pos);
        }

        if pos == start {
            pos = next_utf8_char(text, pos);
        }
        out.push(&text[start..pos]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_encode_printable_ascii_no_space_is_identity() {
        // GPT-2's "printable" range is 33-126 + 161-172 + 174+. Notably
        // EXCLUDES 32 (space), which gets remapped to U+0120.
        let s = b"hello";
        assert_eq!(byte_encode(s), s.to_vec());
        let s = b"abc!@#";
        assert_eq!(byte_encode(s), s.to_vec());
    }

    #[test]
    fn byte_encode_space_maps_to_u0120() {
        // GPT-2 space (byte 0x20) maps to codepoint 0x120 = "Ġ" (Latin
        // capital G with dot above)
        let s = b" ";
        let enc = byte_encode(s);
        // U+0120 is 0xC4 0xA0 in UTF-8
        assert_eq!(enc, vec![0xc4, 0xa0]);
    }

    #[test]
    fn byte_encode_newline_maps_to_u010a() {
        // GPT-2 newline (0x0a) → U+010A = "Ċ"
        let s = b"\n";
        let enc = byte_encode(s);
        // U+010A = 0xC4 0x8A
        assert_eq!(enc, vec![0xc4, 0x8a]);
    }

    #[test]
    fn byte_encode_all_256_round_trip_unique() {
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            let cp = gpt2_byte_to_codepoint(b);
            assert!(seen.insert(cp), "duplicate codepoint {cp:#x} at byte {b:#x}");
        }
    }

    #[test]
    fn joyai_digits_grouped_three() {
        // "12345" → "123" + "45"
        let pieces = joyai_pre_tokenize(b"12345");
        let pieces: Vec<&[u8]> = pieces.into_iter().collect();
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0], b"123");
        assert_eq!(pieces[1], b"45");
    }

    #[test]
    fn joyai_word_then_space_word() {
        // "hello world" → "hello" + " world"
        let pieces = joyai_pre_tokenize(b"hello world");
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0], b"hello");
        assert_eq!(pieces[1], b" world");
    }

    #[test]
    fn joyai_punct_then_alpha() {
        // ".foo" → ".foo" (punct+alpha rule)
        let pieces = joyai_pre_tokenize(b".foo");
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0], b".foo");
    }

    #[test]
    fn joyai_leading_spaces_pattern() {
        // "    int" → "   " (3 spaces) then " int" (1 space + word)
        let pieces = joyai_pre_tokenize(b"    int");
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0], b"   ");
        assert_eq!(pieces[1], b" int");
    }

    #[test]
    fn joyai_punct_run_keeps_trailing_newline() {
        // The comment in ds4 says ">;\n" stays together.
        let pieces = joyai_pre_tokenize(b">;\n");
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0], b">;\n");
    }

    #[test]
    fn bpe_with_synthetic_vocab() {
        // Build a tiny synthetic BpeVocab manually:
        //   tokens: ["a", "b", "c", "ab", "abc"]
        //   merges: ["a b", "ab c"]   (rank 0, rank 1)
        // Encoding "abc" should produce id=4 (the "abc" token)
        let tokens: Vec<Vec<u8>> = vec![
            b"a".to_vec(), b"b".to_vec(), b"c".to_vec(),
            b"ab".to_vec(), b"abc".to_vec(),
        ];
        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as i32);
        }
        let mut merge_rank = HashMap::new();
        merge_rank.insert(b"a b".to_vec(), 0);
        merge_rank.insert(b"ab c".to_vec(), 1);
        let vocab = BpeVocab {
            tokens, token_to_id, merge_rank,
            bos_id: None, eos_id: None, unknown_id: None, padding_id: None,
        };
        let mut out = Vec::new();
        vocab.bpe_emit_piece(b"abc", &mut out);
        assert_eq!(out, vec![4]);
    }
}
