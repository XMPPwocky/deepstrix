//! Laguna tokenizer oracle-parity test.
//!
//! Verifies `BpeVocab::encode_laguna` against reference token IDs produced by
//! the poolside llama.cpp fork (`llama-tokenize --ids`, CPU only) for the
//! Laguna-S-2.1 GGUF. Skips (passes) if the GGUF is not present.

use v4flash_core::gguf::Gguf;
use v4flash_core::tokenizer::BpeVocab;

const GGUF: &str = "/persist/lumi/models/laguna-s-2.1-int4/laguna-s-2.1-Q4_K_M.gguf";

/// (input, oracle IDs from llama-tokenize --ids -ngl 0)
fn cases() -> Vec<(&'static str, Vec<i32>)> {
    vec![
        ("The quick brown fox", vec![2, 785, 3454, 21438, 42850]),
        (
            "def foo(x):\n    return x + 1",
            vec![2, 1172, 12397, 1865, 1117, 268, 341, 658, 854, 585, 290, 86],
        ),
        (
            "Hello, world! (test) 12345 #hashtag",
            vec![2, 6352, 81, 3078, 70, 404, 2300, 78, 290, 86, 87, 88, 89, 90, 842, 6055, 49099],
        ),
        (
            "café résumé naïve 日本語 😀",
            vec![2, 48743, 2097, 18538, 2860, 2097, 12627, 42488, 602, 290, 76211, 53336, 31095, 292],
        ),
        ("   leading and trailing   ", vec![2, 328, 4165, 372, 20643, 341]),
        ("Don't we're I'll can't", vec![2, 13750, 861, 646, 2264, 397, 3091, 476, 861]),
    ]
}

#[test]
fn laguna_matches_oracle() {
    if !std::path::Path::new(GGUF).exists() {
        eprintln!("SKIP: {GGUF} not present");
        return;
    }
    let g = Gguf::open(GGUF).expect("open gguf");
    let vocab = BpeVocab::from_gguf(&g).expect("load vocab");

    assert_eq!(vocab.pre.as_deref(), Some("laguna"), "pre-tokenizer name");
    assert_eq!(vocab.bos_id, Some(2), "bos id");
    assert_eq!(vocab.eot_id, Some(24), "eot id");
    assert!(vocab.add_bos, "add_bos should be true");

    let mut fails = 0;
    for (text, oracle) in cases() {
        let got = vocab.encode_laguna(text);
        let ok = got == oracle;
        if !ok {
            fails += 1;
        }
        println!(
            "[{}] {:?}\n   oracle: {:?}\n      got: {:?}",
            if ok { "PASS" } else { "FAIL" },
            text,
            oracle,
            got
        );
    }
    assert_eq!(fails, 0, "{fails} case(s) diverged from oracle");
}
