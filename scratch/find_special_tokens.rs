// One-off: find IDs of V4-Flash chat special tokens.
use v4flash_core::gguf::{Gguf, GgufValue, GgufArray};

fn main() {
    let path = std::env::args().nth(1).expect("usage: find_special_tokens <gguf>");
    let g = Gguf::open(&path).expect("open");
    let toks = g.metadata("tokenizer.ggml.tokens").expect("tokens");
    let arr = match toks {
        GgufValue::Array(GgufArray::String(v)) => v,
        _ => panic!("tokens not string array"),
    };

    // Print IDs of known specials.
    let wanted = [
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
        "<｜end▁of▁file｜>",
        "<｜begin▁of▁file｜>",
    ];

    for w in wanted {
        let mut found = false;
        for (i, t) in arr.iter().enumerate() {
            if t == w {
                println!("{:>6}  {:?}", i, w);
                found = true;
                break;
            }
        }
        if !found {
            println!("{:>6}  {:?}  <NOT FOUND>", -1i32, w);
        }
    }
    let _ = path;
}
