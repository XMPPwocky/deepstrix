//! Reference golden vectors for the joyai-llm pre-tokenizer and `encode`.
//!
//! `tests/data/joyai_pretok_goldens.json` is produced by
//! `scripts/gen_joyai_pretok_vectors.py`: HF `tokenizers` running the model's
//! own `tokenizer.json`, so both the pieces and the ids are the reference's, not
//! this crate's. The texts target what an ASCII-only split gets wrong (CJK and
//! full-width punctuation, typographic quotes, non-ASCII digits and blanks,
//! combining marks, emoji, controls). The file carries a trimmed vocab with the
//! model's real ids that is sufficient by construction for these texts.

use std::collections::BTreeMap;

use serde::Deserialize;
use v4flash_core::tokenizer::{joyai_pre_tokenize, BpeVocab};

const GOLDENS: &str = include_str!("data/joyai_pretok_goldens.json");

#[derive(Deserialize)]
struct Goldens {
    vocab_size: usize,
    cases: Vec<Case>,
    tokens: BTreeMap<String, String>,
    merges: Vec<String>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    pieces: Vec<String>,
    ids: Vec<i32>,
}

#[test]
fn joyai_matches_hf_reference() {
    let g: Goldens = serde_json::from_str(GOLDENS).expect("joyai_pretok_goldens.json parses");
    let vocab = BpeVocab::from_sparse_parts(
        g.vocab_size,
        g.tokens.into_iter().map(|(id, t)| (id.parse::<i32>().expect("numeric id"), t.into_bytes())),
        g.merges.into_iter().map(String::into_bytes),
        Some("joyai-llm".to_string()),
    );
    let mut bad = Vec::new();
    for c in &g.cases {
        let pieces: Vec<&str> = joyai_pre_tokenize(c.text.as_bytes())
            .into_iter()
            .map(|p| std::str::from_utf8(p).expect("pieces of valid UTF-8 are valid UTF-8"))
            .collect();
        if pieces != c.pieces {
            bad.push(format!("{:?}\n  pieces ours {:?}\n         ref  {:?}", c.text, pieces, c.pieces));
        }
        let ids = vocab.encode(&c.text);
        if ids != c.ids {
            bad.push(format!("{:?}\n  ids ours {:?}\n      ref  {:?}", c.text, ids, c.ids));
        }
    }
    assert!(bad.is_empty(), "{} mismatches vs the HF reference:\n{}", bad.len(), bad.join("\n"));
}
