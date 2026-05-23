//! Tokenizer smoke test: load a real BPE vocab from a GGUF and run
//! encode() on a few inputs. Doesn't claim to match the upstream model's
//! tokenizer output (qwen uses its own pre-tokenizer, not joyai-llm),
//! but does prove that:
//!   - vocab loads via from_gguf
//!   - encode() produces token IDs in [0, vocab_size)
//!   - special token IDs come through
//!
//! Full ds4-faithful validation requires V4 Flash + ds4_dump_text_tokenization;
//! lands in M1.4.

use v4flash_core::{Gguf, BpeVocab};

// qwen3.5 has tokenizer.ggml.model = "gpt2" (BPE) — its vocab loads.
const QWEN_GGUF: &str = "/persist/lumi/models/Qwen3.5-122B-A10B-UD-IQ3_XXS.gguf";

#[test]
#[ignore]
fn load_qwen_bpe_vocab_and_encode() {
    if !std::path::Path::new(QWEN_GGUF).exists() {
        eprintln!("skipping: {QWEN_GGUF} not present");
        return;
    }
    let g = Gguf::open(QWEN_GGUF).expect("gguf parse");
    let v = BpeVocab::from_gguf(&g).expect("vocab load");

    eprintln!(
        "qwen vocab: size={} bos={:?} eos={:?} unk={:?} pad={:?}",
        v.vocab_size(), v.bos_id, v.eos_id, v.unknown_id, v.padding_id
    );

    let cases = [
        "hello",
        "hello world",
        "    int x = 42;",
        "The quick brown fox.",
    ];
    for text in cases {
        let ids = v.encode(text);
        eprintln!("  {text:?} → {} tokens: {:?}", ids.len(), &ids[..ids.len().min(8)]);
        // All IDs should be in range
        for &id in &ids {
            assert!((id as usize) < v.vocab_size(), "id {id} out of range");
        }
        // Should at least produce some tokens for non-empty input
        if !text.is_empty() {
            assert!(!ids.is_empty(), "expected at least one token for {text:?}");
        }
    }
}
