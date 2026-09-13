//! `BpeVocab::from_tokenizer_json` (HF tokenizer.json) must be the same
//! tokenizer as `from_gguf` (llama.cpp's copy of it): same id table, merges,
//! special ids, and identical encodings — DeepSeek-V4.1's tokenizer is
//! id-for-id the V4-Flash one (checked with gguf-py: 0 differing ids).
//!   V41_TOKENIZER_JSON=~/.cache/deepstrix/models/dsv4.1f/tokenizer.json \
//!   V4F_GGUF=/persist/lumi/models/.../DeepSeek-V4-Flash-...-00001-of-00004.gguf \
//!   cargo test -p v4flash-core --release --test tokenizer_json_vs_gguf -- --nocapture
use v4flash_core::{gguf::Gguf, BpeVocab};

#[test]
fn tokenizer_json_matches_gguf() {
    let (Ok(tj), Ok(gg)) = (std::env::var("V41_TOKENIZER_JSON"), std::env::var("V4F_GGUF")) else {
        eprintln!("skip: V41_TOKENIZER_JSON / V4F_GGUF unset");
        return;
    };
    let g = Gguf::open(&gg).expect("gguf");
    let a = BpeVocab::from_gguf(&g).expect("from_gguf");
    // engine_worker.rs hardcodes this pre-tokenizer for the V4.1 HF load path; pin it here so the
    // test proves the literal is right, not merely that "same pre => same tables".
    assert_eq!(a.pre.as_deref(), Some("joyai-llm"), "V4.1/V4-Flash pre-tokenizer literal");
    let b = BpeVocab::from_tokenizer_json(&tj, a.pre.clone()).expect("from_tokenizer_json");
    assert_eq!(a.vocab_size(), b.vocab_size(), "vocab size");
    let mut diff = 0;
    for id in 0..a.vocab_size() as i32 {
        if a.token_text(id) != b.token_text(id) {
            diff += 1;
            if diff <= 5 {
                eprintln!("id {id}: gguf {:?} json {:?}", a.token_text(id).map(String::from_utf8_lossy), b.token_text(id).map(String::from_utf8_lossy));
            }
        }
    }
    assert_eq!(diff, 0, "token tables differ");
    assert_eq!((a.bos_id, a.eos_id, a.dsml_id), (b.bos_id, b.eos_id, b.dsml_id), "special ids");
    for name in ["<｜User｜>", "<｜Assistant｜>", "<｜System｜>", "<｜latest_reminder｜>", "<think>", "</think>"] {
        assert_eq!(a.lookup_token_id(name), b.lookup_token_id(name), "{name}");
    }
    let samples = [
        "The capital of France is",
        "  leading spaces and\ttabs\n\nnewlines 12345 3.14159 don't",
        "中文混合 English, emoji 🚀, code: fn main() { println!(\"hi\"); }",
        "<｜User｜>literal special text<｜Assistant｜>",
        "<tool_result>{\"a\": [1, 2, 3]}</tool_result>",
    ];
    for s in samples {
        let (ea, eb) = (a.encode(s), b.encode(s));
        assert_eq!(ea, eb, "encode differs for {s:?}");
    }
    eprintln!("OK: {} tokens, {} samples encode identically; add_bos gguf={} json={}", a.vocab_size(), samples.len(), a.add_bos, b.add_bos);
}
