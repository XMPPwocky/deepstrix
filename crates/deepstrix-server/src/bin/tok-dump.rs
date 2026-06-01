//! Tiny debug binary: enumerate special tokens around the DeepSeek
//! V4-Flash control range and report the `｜DSML｜` id.
//!
//! Usage:
//!   deepstrix-tok-dump <gguf>

use color_eyre::eyre::{self, eyre};
use v4flash_core::tokenizer::BpeVocab;
use v4flash_core::MappedGguf;

fn main() -> eyre::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| eyre!("usage: tok-dump <gguf>"))?;
    let gguf = MappedGguf::open(&path)?;
    let vocab = BpeVocab::from_gguf(gguf.gguf())?;
    println!("vocab_size={}", vocab.vocab_size());
    println!("bos_id={:?} eos_id={:?} dsml_id={:?}", vocab.bos_id, vocab.eos_id, vocab.dsml_id);
    println!();
    println!("Tokens 128800..128840:");
    for id in 128800..128840i32 {
        if let Some(bytes) = vocab.token_text(id) {
            let text = String::from_utf8_lossy(bytes);
            println!("  {id}: {:?}", text);
        }
    }
    println!();
    // Spot-check a few specific lookups.
    for name in [
        "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>",
        "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
        "<\u{ff5c}User\u{ff5c}>",
        "<\u{ff5c}Assistant\u{ff5c}>",
        "<think>",
        "</think>",
        "\u{ff5c}DSML\u{ff5c}",
    ] {
        let id = vocab.lookup_token_id(name);
        println!("  {:?} → id {:?}", name, id);
    }
    Ok(())
}
