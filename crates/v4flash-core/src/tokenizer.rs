//! V4-Flash-compatible BPE tokenizer.
//!
//! Byte encoding and the BPE merge loop are ported from `external/ds4/ds4.c`
//! lines 14258-14570. The joyai-llm pre-tokenizer is NOT ds4's ASCII-only
//! approximation but the exact split of the HF `tokenizer.json` regexes,
//! hand-coded over generated Unicode class tables (no regex dependency).
//!
//! V4 Flash uses three stages:
//!   1. **Pre-tokenize** input text into pieces using the joyai-llm
//!      (= deepseek-v3) regexes. Pieces are byte sub-ranges of the input.
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
    /// End-of-turn token (`tokenizer.ggml.eot_token_id`). For Laguna this
    /// is id 24 (`</assistant>`) and is the primary generation stop.
    pub eot_id: Option<i32>,
    /// Whether to prepend BOS on encode (`tokenizer.ggml.add_bos_token`).
    pub add_bos: bool,
    /// Pre-tokenizer name (`tokenizer.ggml.pre`), e.g. "laguna".
    pub pre: Option<String>,
    /// V4-Flash DSML control token (`｜DSML｜`). Resolved at load time
    /// via `lookup_token_id`, mirroring ds4.c:14952. None if the GGUF
    /// is not a V4-Flash vocab.
    pub dsml_id: Option<i32>,
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
        let dsml_id = token_to_id.get("｜DSML｜".as_bytes()).copied();
        let eot_id = g.metadata("tokenizer.ggml.eot_token_id").and_then(|v| v.as_u32()).map(|v| v as i32);
        // add_bos_token defaults to true when absent (matches llama.cpp for
        // Laguna, whose GGUF sets add_bos_token=true explicitly).
        let add_bos = g
            .metadata("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let pre = g
            .metadata("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(BpeVocab {
            tokens,
            token_to_id,
            merge_rank,
            bos_id,
            eos_id,
            unknown_id,
            padding_id,
            eot_id,
            add_bos,
            pre,
            dsml_id,
        })
    }

    /// Build a vocab from an explicit **sparse** token map plus an ordered
    /// merge list, without a GGUF. `tokens` is `(id, bytes)`; ids not listed
    /// stay empty and are never emitted. Merge rank is the position in
    /// `merges` — only the relative order matters to the merge loop, so a
    /// merge list with entries removed still tokenizes identically as long as
    /// every merge that could fire is kept.
    ///
    /// This exists for golden-vector tests that ship a trimmed vocab (a few
    /// thousand entries) instead of depending on a multi-GB model file. It is
    /// not used by any production path.
    /// Load from a Hugging Face `tokenizer.json` (byte-level BPE): `model.vocab`
    /// + `added_tokens` give the id table, `model.merges` the ranks (either
    /// `"a b"` strings or `[a, b]` pairs). Token strings are stored verbatim,
    /// exactly as `from_gguf` stores llama.cpp's `tokenizer.ggml.tokens` (for
    /// DeepSeek V4/V4.1 the two tables are identical, id for id). `pre` selects
    /// the pre-tokenizer as for GGUF (`"joyai-llm"` for DeepSeek V4/V4.1);
    /// `add_bos` follows the sibling `tokenizer_config.json` when present
    /// (V4.1's reference encoder puts the BOS text into the prompt itself).
    pub fn from_tokenizer_json(path: impl AsRef<std::path::Path>, pre: Option<String>) -> eyre::Result<Self> {
        let path = path.as_ref();
        let f = std::fs::File::open(path).map_err(|e| eyre!("open {}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_reader(std::io::BufReader::new(f)).map_err(|e| eyre!("parse {}: {e}", path.display()))?;
        let model = v.get("model").ok_or_else(|| eyre!("tokenizer.json: no model"))?;
        let vocab = model
            .get("vocab")
            .and_then(|x| x.as_object())
            .ok_or_else(|| eyre!("tokenizer.json: model.vocab missing"))?;
        let mut pairs: Vec<(i32, Vec<u8>)> = vocab
            .iter()
            .filter_map(|(tok, id)| id.as_i64().map(|i| (i as i32, tok.as_bytes().to_vec())))
            .collect();
        if let Some(added) = v.get("added_tokens").and_then(|x| x.as_array()) {
            for a in added {
                if let (Some(id), Some(tok)) = (a.get("id").and_then(|x| x.as_i64()), a.get("content").and_then(|x| x.as_str())) {
                    pairs.push((id as i32, tok.as_bytes().to_vec()));
                }
            }
        }
        let vocab_size = pairs.iter().map(|(id, _)| *id as usize + 1).max().unwrap_or(0);
        let merges: Vec<Vec<u8>> = model
            .get("merges")
            .and_then(|x| x.as_array())
            .ok_or_else(|| eyre!("tokenizer.json: model.merges missing"))?
            .iter()
            .filter_map(|m| match m {
                serde_json::Value::String(s) => Some(s.as_bytes().to_vec()),
                serde_json::Value::Array(ab) if ab.len() == 2 => Some(format!("{} {}", ab[0].as_str()?, ab[1].as_str()?).into_bytes()),
                _ => None,
            })
            .collect();
        let mut vocab_obj = Self::from_sparse_parts(vocab_size, pairs, merges, pre);
        // Sibling tokenizer_config.json: add_bos_token (V4.1: false).
        if let Some(dir) = path.parent() {
            if let Ok(cfg) = std::fs::read_to_string(dir.join("tokenizer_config.json")) {
                if let Ok(c) = serde_json::from_str::<serde_json::Value>(&cfg) {
                    if let Some(b) = c.get("add_bos_token").and_then(|x| x.as_bool()) {
                        vocab_obj.add_bos = b;
                    }
                }
            }
        }
        Ok(vocab_obj)
    }

    pub fn from_sparse_parts(
        vocab_size: usize,
        tokens_in: impl IntoIterator<Item = (i32, Vec<u8>)>,
        merges_in: impl IntoIterator<Item = Vec<u8>>,
        pre: Option<String>,
    ) -> Self {
        let mut tokens: Vec<Vec<u8>> = vec![Vec::new(); vocab_size];
        let mut token_to_id: HashMap<Vec<u8>, i32> = HashMap::new();
        for (id, bytes) in tokens_in {
            let Ok(idx) = usize::try_from(id) else { continue };
            if idx >= tokens.len() || bytes.is_empty() {
                continue;
            }
            token_to_id.insert(bytes.clone(), id);
            tokens[idx] = bytes;
        }
        let mut merge_rank: HashMap<Vec<u8>, i32> = HashMap::new();
        for (rank, m) in merges_in.into_iter().enumerate() {
            merge_rank.insert(m, rank as i32);
        }
        let lookup = |name: &str| token_to_id.get(name.as_bytes()).copied();
        BpeVocab {
            bos_id: lookup("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>"),
            eos_id: lookup("<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>"),
            unknown_id: None,
            padding_id: None,
            eot_id: None,
            add_bos: false,
            dsml_id: lookup("\u{ff5c}DSML\u{ff5c}"),
            pre,
            tokens,
            token_to_id,
            merge_rank,
        }
    }

    /// Look up a token id by its raw byte text. Used to resolve
    /// special-control tokens by name at load time (mirrors ds4's
    /// `vocab_lookup`). Returns None if the token is not in the vocab.
    pub fn lookup_token_id(&self, name: &str) -> Option<i32> {
        self.token_to_id.get(name.as_bytes()).copied()
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

    /// Encode using the Laguna pre-tokenizer (llama.cpp `LAGUNA` pre-type =
    /// newline pre-split + Qwen2-style GPT-2 regex, single-digit numbers),
    /// GPT-2 byte encoding + BPE merge. Prepends BOS (id 2) when `add_bos`.
    ///
    /// This is additive and does not touch the ds4/joyai `encode` path.
    pub fn encode_laguna(&self, text: &str) -> Vec<i32> {
        self.encode_laguna_opts(text, self.add_bos)
    }

    /// Laguna encode with explicit BOS control (server/CLI may already have
    /// prefixed BOS via the chat template).
    pub fn encode_laguna_opts(&self, text: &str, add_bos: bool) -> Vec<i32> {
        let mut out = Vec::new();
        if add_bos {
            if let Some(bos) = self.bos_id {
                out.push(bos);
            }
        }
        for (a, b) in laguna_pre_tokenize(text) {
            self.bpe_emit_piece(&text.as_bytes()[a..b], &mut out);
        }
        out
    }

    /// Encode with the Qwen2 pre-tokenizer (llama.cpp `QWEN2` pre-type: the
    /// Qwen2 GPT-2-style splitter over the whole text, no newline pre-split),
    /// GPT-2 byte encoding + BPE merge. Never prepends BOS: Qwen models do not
    /// use one (the embedding caller appends `<|endoftext|>` itself).
    ///
    /// Differences from the HF tokenizer: no NFC normalization, and special-
    /// token text in the input is byte-pair encoded as text, not split out.
    pub fn encode_qwen2(&self, text: &str) -> Vec<i32> {
        let mut out = Vec::new();
        for (a, b) in qwen2_pre_tokenize(text) {
            self.bpe_emit_piece(&text.as_bytes()[a..b], &mut out);
        }
        out
    }

    /// Dispatch on the GGUF `tokenizer.ggml.pre` value: "laguna" uses the
    /// Laguna path (with its `add_bos`), "qwen2" the Qwen2 path (no BOS),
    /// anything else keeps the legacy ds4/joyai `encode` (no implicit BOS,
    /// unchanged behavior).
    pub fn encode_auto(&self, text: &str) -> Vec<i32> {
        match self.pre.as_deref() {
            Some("laguna") => self.encode_laguna(text),
            Some("qwen2") => self.encode_qwen2(text),
            _ => self.encode(text),
        }
    }

    fn bpe_emit_piece(&self, raw_piece: &[u8], out: &mut Vec<i32>) {
        // Step 1: byte-encode raw bytes into printable UTF-8.
        let enc = byte_encode(raw_piece);
        let n = enc.len();

        // Step 2: one symbol per UTF-8 char of `enc`. Merges only join
        // neighbours, so a symbol is always a byte range of `enc`: kept as
        // `end[start]` (NONE = `start` begins no symbol) and `prev[start]`.
        const NONE: usize = usize::MAX;
        let mut end = vec![NONE; n];
        let mut prev = vec![NONE; n];
        let mut off = 0;
        let mut last = NONE;
        while off < n {
            let e = (off + utf8_len_from_first_byte(enc[off])).min(n);
            end[off] = e;
            prev[off] = last;
            last = off;
            off = e;
        }

        // Step 3: greedy BPE -- merge the lowest-rank adjacent pair, the
        // leftmost on a tie, until none has a rank -- off a min-heap of
        // candidate pairs `(rank, left, mid, right)`, each checked against
        // the current symbols when popped (boundaries only ever disappear,
        // so a pair whose two symbols still end where they did is the same
        // pair). Same merges in the same order as rescanning every pair
        // after every merge (`bpe_emit_piece_greedy`), but O(n log n): that
        // rescan was O(n^2) rank lookups per piece, ~2 s for one 4,000-char
        // run of a 3-byte symbol (a table border, a progress bar).
        let mut key = Vec::new();
        let mut heap = std::collections::BinaryHeap::new();
        let mut push = |heap: &mut std::collections::BinaryHeap<_>, l: usize, m: usize, r: usize| {
            if let Some(rank) = self.pair_rank_into(&mut key, &enc[l..m], &enc[m..r]) {
                heap.push(std::cmp::Reverse((rank, l, m, r)));
            }
        };
        let mut s = 0;
        while s < n && end[s] < n {
            let m = end[s];
            push(&mut heap, s, m, end[m]);
            s = m;
        }
        while let Some(std::cmp::Reverse((_, l, m, r))) = heap.pop() {
            if end[l] != m || end[m] != r {
                continue; // stale: one of the two symbols has merged since
            }
            end[l] = r;
            end[m] = NONE;
            if r < n {
                prev[r] = l;
                push(&mut heap, l, r, end[r]);
            }
            if prev[l] != NONE {
                push(&mut heap, prev[l], l, r);
            }
        }

        // Step 4: look up each final symbol; if not in vocab, fall back
        // to byte-by-byte lookup (matches ds4 lines 14376-14388).
        let mut s = 0;
        while s < n {
            let piece = &enc[s..end[s]];
            s = end[s];
            if let Some(&id) = self.token_to_id.get(piece) {
                out.push(id);
                continue;
            }
            for &b in piece {
                if let Some(&id) = self.token_to_id.get(&[b][..]) {
                    out.push(id);
                }
                // Note: if even single-byte lookup fails, ds4 drops the
                // byte silently. We mirror that here — TODO: warn?
            }
        }
    }

    /// The pre-2026-10-03 merge loop (rescan every pair after every merge),
    /// kept as the reference `bpe_emit_piece` must match id for id.
    #[cfg(test)]
    fn bpe_emit_piece_greedy(&self, raw_piece: &[u8], out: &mut Vec<i32>) {
        let encoded = byte_encode(raw_piece);
        let mut sym: Vec<Vec<u8>> = Vec::new();
        let mut off = 0;
        while off < encoded.len() {
            let n = utf8_len_from_first_byte(encoded[off]);
            let end = (off + n).min(encoded.len());
            sym.push(encoded[off..end].to_vec());
            off = end;
        }
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

    #[cfg(test)]
    fn pair_rank(&self, a: &[u8], b: &[u8]) -> Option<i32> {
        self.pair_rank_into(&mut Vec::new(), a, b)
    }

    /// Rank of merging `a` + `b`, building the key in `key` (reused across
    /// calls: no allocation per lookup).
    fn pair_rank_into(&self, key: &mut Vec<u8>, a: &[u8], b: &[u8]) -> Option<i32> {
        // Key format: "<a><space><b>" (raw bytes, not escaped)
        key.clear();
        key.extend_from_slice(a);
        key.push(b' ');
        key.extend_from_slice(b);
        self.merge_rank.get(key.as_slice()).copied()
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
pub(crate) fn byte_encode(raw: &[u8]) -> Vec<u8> {
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

// ---- joyai pre-tokenizer ----
//
// The reference is the `pre_tokenizer` of the DeepSeek V4 `tokenizer.json`
// (the V3 one, unchanged; read from the copy in the `deepseek-tokenizer` 0.3.0
// wheel, whose ids match the V4-Flash GGUF vocab; llama.cpp's
// LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM = DEEPSEEK3_LLM carries the same three
// regexes): a Sequence of three
// `Split { behavior: Isolated }`, each re-splitting every piece the previous
// one produced, then `ByteLevel { use_regex: false }` (no further split):
//
//   1. \p{N}{1,3}
//   2. [一-龥぀-ゟ゠-ヿ]+                    (U+4E00..=9FA5, U+3040..=30FF)
//   3. [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
//      |[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//      | ?[\p{P}\p{S}]+[\r\n]*
//      |\s*[\r\n]+|\s+(?!\S)|\s+
//
// Isolated keeps the matches AND the unmatched stretches between them as
// pieces, and each regex sees only its own piece: no match crosses a number or
// CJK boundary, and `(?!\S)` succeeds at a piece end. \p{N} and the CJK ranges
// are disjoint, and regex 3 has no match inside a pure \p{N} piece, so the
// passes reduce to: cut the text into maximal number / CJK / other runs, chunk
// number runs by 3, run regex 3 (leftmost-first alternation) on the rest.
//
// ds4.c's port, which this used to mirror, was ASCII-only: every byte >= 0x80
// counted as a letter, so "。\n", "”\n", "，世界" and "２０２４" split
// differently from the reference and produced non-canonical ids.
//
// Unicode classes come from `joyai_ucd` (generated by scripts/gen_joyai_ucd.py
// from the reference's own regex engine), not from `char::is_alphabetic` and
// friends, which are not \p{L} / \p{N} for every codepoint.

mod joyai_ucd;
use joyai_ucd::{L, M, N, PS, WS};

/// None of \p{L} \p{M} \p{N} \p{P} \p{S} \s (controls, format chars, private
/// use, unassigned).
const UC_OTHER: u8 = 0;

/// ASCII classes, so ASCII never touches the range table.
static JOYAI_ASCII: [u8; 128] = {
    let mut t = [UC_OTHER; 128];
    let mut b = 0;
    while b < 128 {
        let c = b as u8;
        t[b] = if c.is_ascii_alphabetic() {
            L
        } else if c.is_ascii_digit() {
            N
        } else if c.is_ascii_punctuation() {
            PS
        } else if matches!(c, b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | b' ') {
            WS
        } else {
            UC_OTHER
        };
        b += 1;
    }
    t
};

/// `joyai_ucd` class of `c`.
#[inline]
fn joyai_class(c: char) -> u8 {
    if c.is_ascii() {
        return JOYAI_ASCII[c as usize];
    }
    if ('\u{4e00}'..='\u{9fa5}').contains(&c) {
        return L; // common CJK, all Lo (pinned by a test)
    }
    let cp = c as u32;
    let r = &joyai_ucd::RANGES;
    match r.binary_search_by(|&(lo, hi, _)| {
        if hi < cp {
            std::cmp::Ordering::Less
        } else if lo > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => r[i].2,
        Err(_) => UC_OTHER,
    }
}

/// Regex 2's class `[一-龥぀-ゟ゠-ヿ]`.
fn joyai_cjk(c: char) -> bool {
    matches!(c, '\u{4e00}'..='\u{9fa5}' | '\u{3040}'..='\u{30ff}')
}

#[inline]
fn char_at(s: &str, pos: usize) -> char {
    let b = s.as_bytes()[pos];
    if b < 0x80 { b as char } else { s[pos..].chars().next().unwrap() }
}

/// joyai-llm pre-tokenizer: pieces are byte sub-ranges of `text`, in order,
/// covering it. Bytes that are not valid UTF-8 (never the case for `encode`,
/// which takes `&str`) become pieces of their own.
pub fn joyai_pre_tokenize(text: &[u8]) -> Vec<&[u8]> {
    let mut spans = Vec::new();
    let mut base = 0;
    for chunk in text.utf8_chunks() {
        let s = chunk.valid();
        joyai_split(s, base, &mut spans);
        base += s.len();
        if !chunk.invalid().is_empty() {
            spans.push((base, base + chunk.invalid().len()));
            base += chunk.invalid().len();
        }
    }
    spans.into_iter().map(|(a, b)| &text[a..b]).collect()
}

/// Run kinds of regexes 1 and 2 (a char is at most one: see the module note).
const RUN_OTHER: u8 = 0;
const RUN_NUM: u8 = 1;
const RUN_CJK: u8 = 2;

#[inline]
fn joyai_run_kind(c: char, class: u8) -> u8 {
    if class == N {
        RUN_NUM
    } else if joyai_cjk(c) {
        RUN_CJK
    } else {
        RUN_OTHER
    }
}

/// Regexes 1 and 2: cut `s` into number / CJK / other runs, number runs in
/// chunks of 3, the rest through regex 3. Spans are offset by `base`.
fn joyai_split(s: &str, base: usize, out: &mut Vec<(usize, usize)>) {
    let mut pos = 0;
    while pos < s.len() {
        let c = char_at(s, pos);
        let kind = joyai_run_kind(c, joyai_class(c));
        if kind == RUN_NUM {
            let start = pos;
            pos += c.len_utf8();
            for _ in 0..2 {
                match s.get(pos..).and_then(|r| r.chars().next()) {
                    Some(d) if joyai_class(d) == N => pos += d.len_utf8(),
                    _ => break,
                }
            }
            out.push((base + start, base + pos));
        } else {
            pos = joyai_regex3(s, pos, kind, base, out);
        }
    }
}

/// Regex 3 over the piece regexes 1-2 left at `pos`: the maximal run of chars
/// of run kind `kind` (found lazily, in the same scan). Returns the piece end.
/// Isolated: unmatched stretches are pieces too; only \p{C} and unassigned
/// codepoints not followed by a letter/mark end up there.
fn joyai_regex3(s: &str, mut pos: usize, kind: u8, base: usize, out: &mut Vec<(usize, usize)>) -> usize {
    // The char at `p` and its class; None past the piece end.
    let at = |p: usize| -> Option<(char, u8)> {
        if p >= s.len() {
            return None;
        }
        let c = char_at(s, p);
        let k = joyai_class(c);
        (joyai_run_kind(c, k) == kind).then_some((c, k))
    };
    fn skip(at: &impl Fn(usize) -> Option<(char, u8)>, mut p: usize, f: impl Fn(char, u8) -> bool) -> usize {
        while let Some((c, k)) = at(p) {
            if !f(c, k) {
                break;
            }
            p += c.len_utf8();
        }
        p
    }
    let mut gap: Option<usize> = None;
    while let Some((c0, k0)) = at(pos) {
        let p1 = pos + c0.len_utf8();
        let n1 = at(p1);
        let k1 = n1.map(|(_, k)| k);
        let m_end = if c0.is_ascii_punctuation() && n1.is_some_and(|(c, _)| c.is_ascii_alphabetic()) {
            // [!-/:-@[-`{-~][A-Za-z]+
            Some(skip(&at, p1, |c, _| c.is_ascii_alphabetic()))
        } else if (k0 == L || k0 == M) || (k0 != PS && c0 != '\r' && c0 != '\n' && matches!(k1, Some(L | M))) {
            // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ (c0 is either the optional
            // char or the first of the run; the run continues from p1 alike)
            Some(skip(&at, p1, |_, k| k == L || k == M))
        } else if k0 == PS || (c0 == ' ' && k1 == Some(PS)) {
            //  ?[\p{P}\p{S}]+[\r\n]*
            let p = skip(&at, p1, |_, k| k == PS);
            Some(skip(&at, p, |c, _| c == '\r' || c == '\n'))
        } else if k0 == WS {
            // \s*[\r\n]+ ends after the run's last CR/LF; else \s+(?!\S) leaves
            // the last blank for the next token unless the run ends the piece;
            // else \s+ (a single blank).
            let (mut p, mut last, mut nl_end) = (pos, pos, None);
            while let Some((c, WS)) = at(p) {
                last = p;
                p += c.len_utf8();
                if c == '\r' || c == '\n' {
                    nl_end = Some(p);
                }
            }
            Some(nl_end.unwrap_or(if at(p).is_some() && last > pos { last } else { p }))
        } else {
            None
        };
        match m_end {
            Some(e) => {
                if let Some(g) = gap.take() {
                    out.push((base + g, base + pos));
                }
                out.push((base + pos, base + e));
                pos = e;
            }
            None => {
                gap.get_or_insert(pos);
                pos = p1;
            }
        }
    }
    if let Some(g) = gap {
        out.push((base + g, base + pos));
    }
    pos
}

// ---- Laguna pre-tokenizer ----
//
// Port of llama.cpp `LLAMA_VOCAB_PRE_TYPE_LAGUNA` (poolside fork). The
// pre-type applies two regex expressions in sequence:
//
//   1. "[^\n]+|[\n]+"   -> newline pre-split (unicode_regex_split_custom_newlines)
//   2. "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//       |[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*
//       |\s*[\r\n]+|\s+(?!\S)|\s+"
//      -> the Qwen2 GPT-2-style splitter (unicode_regex_split_custom_qwen2)
//
// Both patterns route to hand-coded custom splitters in llama.cpp's
// unicode.cpp, so we port those splitters directly (no regex engine).
// Difference vs the llama3 pre-type: numbers are single codepoints (`\p{N}`,
// not `\p{N}{1,3}`); difference vs ds4/joyai: entirely — see the module docs.
//
// Unicode classes: `\p{L}` ~= char::is_alphabetic() minus numeric, `\p{N}`
// ~= char::is_numeric() (Nd|Nl|No), `\s` ~= char::is_whitespace(). The
// `flags.as_uint()` "has a category" guard maps to "codepoint in range".

/// Laguna pre-tokenize `text` into byte sub-ranges `[start, end)` of
/// `text.as_bytes()`. Operates on Unicode codepoints internally.
pub fn laguna_pre_tokenize(text: &str) -> Vec<(usize, usize)> {
    let cps: Vec<char> = text.chars().collect();
    // byte offset of each char index; boff[cps.len()] == text.len()
    let mut boff: Vec<usize> = Vec::with_capacity(cps.len() + 1);
    let mut b = 0usize;
    for c in &cps {
        boff.push(b);
        b += c.len_utf8();
    }
    boff.push(text.len());

    let n = cps.len();
    let mut bounds: Vec<usize> = Vec::new(); // absolute char-index token ends

    // Stage 1: split on newline runs vs non-newline runs.
    let mut seg_start = 0usize;
    while seg_start < n {
        let is_nl = cps[seg_start] == '\n';
        let mut seg_end = seg_start;
        while seg_end < n && (cps[seg_end] == '\n') == is_nl {
            seg_end += 1;
        }
        // Stage 2: Qwen2 split within [seg_start, seg_end).
        laguna_qwen2_split(&cps, seg_start, seg_end, &mut bounds);
        seg_start = seg_end;
    }

    let mut out = Vec::with_capacity(bounds.len());
    let mut prev = 0usize;
    for &e in &bounds {
        if e > prev {
            out.push((boff[prev], boff[e]));
        }
        prev = e;
    }
    out
}

/// Qwen2 pre-tokenize `text` (llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2`: the one
/// Qwen2 regex, `unicode_regex_split_custom_qwen2`, over the whole text) into
/// byte sub-ranges `[start, end)` of `text.as_bytes()`. Laguna's splitter
/// minus its newline pre-split, with the reference engine's exact Unicode classes.
pub fn qwen2_pre_tokenize(text: &str) -> Vec<(usize, usize)> {
    let cps: Vec<char> = text.chars().collect();
    let mut boff: Vec<usize> = Vec::with_capacity(cps.len() + 1);
    let mut b = 0usize;
    for c in &cps {
        boff.push(b);
        b += c.len_utf8();
    }
    boff.push(text.len());
    let mut bounds: Vec<usize> = Vec::new();
    qwen2_split_exact(&cps, 0, cps.len(), &mut bounds);
    let mut out = Vec::with_capacity(bounds.len());
    let mut prev = 0usize;
    for &e in &bounds {
        if e > prev {
            out.push((boff[prev], boff[e]));
        }
        prev = e;
    }
    out
}

/// Port of `unicode_regex_split_custom_qwen2` for a single segment
/// `[ini, end)` of `cps`. Pushes absolute char-index token ends onto
/// `bounds` (contiguous, covering the whole segment).
fn laguna_qwen2_split(cps: &[char], ini: usize, end: usize, bounds: &mut Vec<usize>) {
    // \p{N} ~ Nd|Nl|No; \p{L} ~ alphabetic minus the Nl overlap (Laguna's
    // approximation: it counts Other_Alphabetic combining marks as letters).
    qwen2_split_with(cps, ini, end, bounds, |c| c.is_alphabetic() && !c.is_numeric(), |c| c.is_numeric(), |c| c.is_whitespace());
}

/// The Qwen2 splitter with EXACT classes: the reference tokenizer engine's own
/// \p{L} / \p{N} / \s (`joyai_ucd`, generated from HF tokenizers' Oniguruma).
/// `char::is_alphabetic` also takes combining marks (Thai, Indic vowel signs)
/// as letters, which merges pieces the reference keeps apart.
fn qwen2_split_exact(cps: &[char], ini: usize, end: usize, bounds: &mut Vec<usize>) {
    qwen2_split_with(cps, ini, end, bounds, |c| joyai_class(c) == L, |c| joyai_class(c) == N, |c| joyai_class(c) == WS);
}

/// Port of `unicode_regex_split_custom_qwen2` over `[ini, end)` with the
/// given \p{L} / \p{N} / \s predicates.
fn qwen2_split_with(
    cps: &[char],
    ini: usize,
    end: usize,
    bounds: &mut Vec<usize>,
    letter: impl Fn(char) -> bool,
    number: impl Fn(char) -> bool,
    space: impl Fn(char) -> bool,
) {
    let cpt = |p: usize| -> Option<char> {
        if ini <= p && p < end { Some(cps[p]) } else { None }
    };
    let is_number = |p: usize| cpt(p).map_or(false, |c| number(c));
    let is_letter = |p: usize| cpt(p).map_or(false, |c| letter(c));
    let is_ws = |p: usize| cpt(p).map_or(false, |c| space(c));
    // flags.as_uint() != 0  <=>  codepoint present (in range).
    let has_flags = |p: usize| cpt(p).is_some();

    let mut pos = ini;
    while pos < end {
        let c = cps[pos];

        // (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
        if c == '\'' && pos + 1 < end {
            let n1 = cps[pos + 1].to_ascii_lowercase();
            if n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd' {
                pos += 2;
                bounds.push(pos);
                continue;
            }
            if pos + 2 < end {
                let n2 = cps[pos + 2].to_ascii_lowercase();
                if (n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l') {
                    pos += 3;
                    bounds.push(pos);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if !(c == '\r' || c == '\n' || is_number(pos)) && (is_letter(pos) || is_letter(pos + 1)) {
            pos += 1;
            while is_letter(pos) {
                pos += 1;
            }
            bounds.push(pos);
            continue;
        }

        // \p{N}   (single codepoint)
        if is_number(pos) {
            pos += 1;
            bounds.push(pos);
            continue;
        }

        // <space>?[^\s\p{L}\p{N}]+[\r\n]*
        let flags2 = if c == ' ' { pos + 1 } else { pos };
        let flags2_wln = is_ws(flags2) || is_letter(flags2) || is_number(flags2);
        if !flags2_wln && has_flags(pos) {
            if c == ' ' {
                pos += 1;
            }
            while has_flags(pos) && !(is_ws(pos) || is_letter(pos) || is_number(pos)) {
                pos += 1;
            }
            while matches!(cpt(pos), Some('\r') | Some('\n')) {
                pos += 1;
            }
            bounds.push(pos);
            continue;
        }

        // Whitespace runs: \s*[\r\n]+ | \s+(?!\S) | \s+
        let mut num_ws = 0usize;
        let mut last_end_rn = 0usize; // char index just past last \r or \n
        while is_ws(pos + num_ws) {
            let c2 = cps[pos + num_ws];
            if c2 == '\r' || c2 == '\n' {
                last_end_rn = pos + num_ws + 1;
            }
            num_ws += 1;
        }
        if last_end_rn > 0 {
            // \s*[\r\n]+
            pos = last_end_rn;
            bounds.push(pos);
            continue;
        }
        if num_ws > 1 && cpt(pos + num_ws).is_some() {
            // \s+(?!\S): leave one space for the following token
            pos += num_ws - 1;
            bounds.push(pos);
            continue;
        }
        if num_ws > 0 {
            // \s+
            pos += num_ws;
            bounds.push(pos);
            continue;
        }

        // no matches: emit single codepoint
        pos += 1;
        bounds.push(pos);
    }
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

    fn joyai_strs(s: &str) -> Vec<&str> {
        joyai_pre_tokenize(s.as_bytes()).into_iter().map(|p| std::str::from_utf8(p).unwrap()).collect()
    }

    #[test]
    fn joyai_non_ascii_follows_reference_regex() {
        // Pieces from HF tokenizers on the model's tokenizer.json (more, with
        // ids, in tests/joyai_pretok_goldens.rs). ds4's ASCII-only split gave
        // 你好|。|\n, .|”|\n, one piece for "，世界", one letter run for ２０２４.
        assert_eq!(joyai_strs("你好。\n"), ["你好", "。\n"]);
        assert_eq!(joyai_strs("Hello.”\n"), ["Hello", ".”\n"]);
        assert_eq!(joyai_strs("，世界"), ["，", "世界"]);
        assert_eq!(joyai_strs("２０２４年"), ["２０２", "４", "年"]);
        // Non-ASCII blank joins the next word like ' '; `(?!\S)` holds at a
        // piece end, so a blank run before CJK stays whole.
        assert_eq!(joyai_strs("a\u{a0}b  中"), ["a", "\u{a0}b", "  ", "中"]);
        // Unmatched controls/format chars form their own pieces.
        assert_eq!(joyai_strs("a\u{0}b\u{1}\u{2} c"), ["a", "\u{0}b", "\u{1}\u{2}", " c"]);
    }

    #[test]
    fn joyai_invalid_utf8_bytes_are_own_pieces() {
        assert_eq!(joyai_pre_tokenize(b"ab\xffcd"), [&b"ab"[..], b"\xff", b"cd"]);
    }

    #[test]
    fn joyai_ucd_table_is_sorted_and_agrees_with_fast_paths() {
        let r = &joyai_ucd::RANGES;
        assert!(r.iter().all(|&(lo, hi, k)| lo <= hi && (1..=5).contains(&k)));
        assert!(r.windows(2).all(|w| w[0].1 < w[1].0), "ranges sorted and disjoint");
        for cp in (0u32..0x80).chain(0x4e00..=0x9fa5) {
            let table = r.iter().find(|&&(lo, hi, _)| lo <= cp && cp <= hi).map_or(UC_OTHER, |x| x.2);
            assert_eq!(joyai_class(char::from_u32(cp).unwrap()), table, "U+{cp:04X}");
        }
        // The pass reduction in `joyai_split` relies on \p{N} and regex 2's
        // CJK ranges being disjoint.
        for c in ('\u{3040}'..='\u{30ff}').chain('\u{4e00}'..='\u{9fa5}') {
            assert_ne!(joyai_class(c), N, "{c:?}");
        }
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
            eot_id: None, add_bos: false, pre: None,
            dsml_id: None,
        };
        let mut out = Vec::new();
        vocab.bpe_emit_piece(b"abc", &mut out);
        assert_eq!(out, vec![4]);
    }

    /// The heap merge (`bpe_emit_piece`) emits exactly what the rescan-every-
    /// pair loop (`bpe_emit_piece_greedy`) does: overlapping candidates
    /// ("a a" in "aaa"), merges that create lower-rank pairs on both sides,
    /// a merge whose result is not in the vocab (byte fallback), and
    /// multi-byte symbols (é byte-encodes to two chars).
    #[test]
    fn bpe_heap_merge_matches_the_greedy_rescan() {
        let toks: [&[u8]; 14] = [b"a", b"b", b"c", b"aa", b"ab", b"ba", b"aaa", b"aab", b"abab", b"aaaa", b"bab", b"cab", "Ã".as_bytes(), "Ã©".as_bytes()];
        let merges: [&[u8]; 11] = [b"a b", b"a a", b"b a", b"aa a", b"a aa", b"ab ab", b"aa b", b"aa aa", b"b ab", b"c ab", "Ã ©".as_bytes()];
        let vocab = BpeVocab::from_sparse_parts(
            32,
            toks.iter().enumerate().map(|(i, t)| (i as i32, t.to_vec())).chain([(20, "©".as_bytes().to_vec())]),
            merges.iter().map(|m| m.to_vec()).chain([b"c c".to_vec()]), // "cc" is not a token: byte fallback
            None,
        );
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let alphabet: [&str; 4] = ["a", "b", "c", "é"];
        for i in 0..5000 {
            let len = 1 + (next() % if i < 4000 { 24 } else { 300 }) as usize;
            let s: String = (0..len).map(|_| alphabet[(next() % 4) as usize]).collect();
            let (mut fast, mut slow) = (Vec::new(), Vec::new());
            vocab.bpe_emit_piece(s.as_bytes(), &mut fast);
            vocab.bpe_emit_piece_greedy(s.as_bytes(), &mut slow);
            assert_eq!(fast, slow, "{s:?}");
        }
        for s in ["", "a", "aaaaaaaaa", "abababababab", "ccccc", "éééé", "aabaabaab"] {
            let (mut fast, mut slow) = (Vec::new(), Vec::new());
            vocab.bpe_emit_piece(s.as_bytes(), &mut fast);
            vocab.bpe_emit_piece_greedy(s.as_bytes(), &mut slow);
            assert_eq!(fast, slow, "{s:?}");
        }
    }

    /// Same check on the real V4.1 vocab, piece by piece over long runs and
    /// mixed text, plus the long-run cost. Needs the model's tokenizer.json
    /// (`TOKENIZER_JSON`).
    ///   TOKENIZER_JSON=.../tokenizer.json cargo test --release -p v4flash-core --lib -- --ignored --nocapture bpe_heap_merge_real_vocab
    #[test]
    #[ignore]
    fn bpe_heap_merge_real_vocab() {
        let tj = std::env::var("TOKENIZER_JSON").expect("TOKENIZER_JSON");
        let vocab = BpeVocab::from_tokenizer_json(&tj, Some("joyai-llm".to_string())).unwrap();
        let mut text = String::new();
        for unit in ["\u{2500}", "=", " ", "\n", "\u{7684}", "a", "\u{2588}", "\u{1F600}", "-", "é"] {
            for n in [1usize, 7, 64, 777, 4000] {
                text += &unit.repeat(n);
                text += " x ";
            }
        }
        text += "fn main() {\n    println!(\"héllo — 世界 🎉 ２０２４\");\n}\n\t\t  ┌──┬──┐ │a│b│ └──┴──┘ ...!!!??? ¿qué? naïve café ";
        let mut checked = 0usize;
        for piece in joyai_pre_tokenize(text.as_bytes()) {
            let (mut fast, mut slow) = (Vec::new(), Vec::new());
            vocab.bpe_emit_piece(piece, &mut fast);
            vocab.bpe_emit_piece_greedy(piece, &mut slow);
            assert_eq!(fast, slow, "piece {:?}", String::from_utf8_lossy(piece));
            checked += 1;
        }
        let t = std::time::Instant::now();
        let ids = vocab.encode(&"\u{2500}".repeat(4000));
        eprintln!("{checked} pieces identical; 4,000 x U+2500 now {:.2} ms ({} ids)", t.elapsed().as_secs_f64() * 1e3, ids.len());
    }

    fn qwen2_pieces(text: &str) -> Vec<&str> {
        qwen2_pre_tokenize(text).into_iter().map(|(a, b)| &text[a..b]).collect()
    }

    #[test]
    fn qwen2_words_digits_contractions() {
        assert_eq!(qwen2_pieces("Hello world"), ["Hello", " world"]);
        // \p{N}: one digit per piece.
        assert_eq!(qwen2_pieces("ab12"), ["ab", "1", "2"]);
        assert_eq!(qwen2_pieces("it's"), ["it", "'s"]);
        assert_eq!(qwen2_pieces("Query:{x}"), ["Query", ":{", "x", "}"]);
    }

    #[test]
    fn qwen2_has_no_newline_pre_split() {
        // `\s*[\r\n]+` spans the spaces before a newline. Laguna splits on
        // newline runs first, so the same text differs there.
        assert_eq!(qwen2_pieces("x  \n y"), ["x", "  \n", " y"]);
        let laguna: Vec<&str> = laguna_pre_tokenize("x  \n y").into_iter().map(|(a, b)| &"x  \n y"[a..b]).collect();
        assert_eq!(laguna, ["x", "  ", "\n", " y"]);
    }

    #[test]
    fn qwen2_pieces_cover_the_text() {
        for text in ["", "a", "Instruct: Given a query\nQuery:what is up?", "  lead\t\ttabs \r\n\r\nend  ", "日本語のテキスト 123", "emoji 🙂🙂 done"] {
            let pieces = qwen2_pre_tokenize(text);
            let mut prev = 0;
            for &(a, b) in &pieces {
                assert_eq!(a, prev, "gap in {text:?}");
                assert!(b > a);
                prev = b;
            }
            assert_eq!(prev, text.len(), "pieces of {text:?} do not reach the end");
        }
    }
}
