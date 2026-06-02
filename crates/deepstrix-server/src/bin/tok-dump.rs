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

    // Verify encode_with_special_marker actually inlines TOK_DSML.
    // Run a small DSML-shaped fragment through it, then through plain
    // BPE, and print both token sequences alongside their decoded text.
    println!();
    println!("encode_with_special_marker spot-check:");
    let sample =
        "<\u{ff5c}DSML\u{ff5c}tool_calls>\n<\u{ff5c}DSML\u{ff5c}invoke name=\"bash\">";
    let dsml_id = vocab.dsml_id.unwrap_or(-1);
    let marker = "\u{ff5c}DSML\u{ff5c}";
    let mut out: Vec<i32> = Vec::new();
    let mut remaining = sample;
    while let Some(pos) = remaining.find(marker) {
        if pos > 0 {
            out.extend(vocab.encode(&remaining[..pos]));
        }
        out.push(dsml_id);
        remaining = &remaining[pos + marker.len()..];
    }
    if !remaining.is_empty() {
        out.extend(vocab.encode(remaining));
    }
    let plain: Vec<i32> = vocab.encode(sample);
    println!("  input:                {:?}", sample);
    println!("  with-marker tokens:   {:?}", out);
    println!("  plain encode tokens:  {:?}", plain);
    println!("  with-marker per-tok:");
    for &id in &out {
        let text = vocab
            .token_text(id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default();
        println!("    {:>6} {:?}{}", id, text, if id == dsml_id { "  ← TOK_DSML" } else { "" });
    }
    println!("  plain per-tok:");
    for &id in &plain {
        let text = vocab
            .token_text(id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default();
        println!("    {:>6} {:?}{}", id, text, if id == dsml_id { "  ← TOK_DSML" } else { "" });
    }
    Ok(())
}
