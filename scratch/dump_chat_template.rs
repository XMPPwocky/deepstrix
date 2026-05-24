// Dump the full chat_template from GGUF metadata.
use v4flash_core::{gguf::GgufValue, MappedGguf};

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_chat_template <gguf>");
    let gguf = MappedGguf::open(&path).expect("open");
    let v = gguf.gguf().metadata("tokenizer.chat_template").expect("missing chat_template");
    if let GgufValue::String(s) = v {
        println!("{}", s);
    } else {
        eprintln!("unexpected type for chat_template: {:?}", v);
    }

    // Also dump special-token strings at known IDs.
    let toks = gguf.gguf().metadata("tokenizer.ggml.tokens").expect("missing tokens");
    if let GgufValue::Array(arr) = toks {
        eprintln!("--- special tokens (0..16):");
        for i in 0..16 {
            if let Some(GgufValue::String(s)) = arr.get(i) {
                eprintln!("  [{i}] {s:?}");
            }
        }
        // Search for chat markers
        eprintln!("--- chat markers:");
        for (i, t) in arr.iter().enumerate() {
            if let GgufValue::String(s) = t {
                if s.contains("User") || s.contains("Assistant") || s.contains("begin") || s.contains("end")
                    || s.contains("system") {
                    eprintln!("  [{i}] {s:?}");
                }
            }
            if i > 500 { break; }
        }
    }
}
