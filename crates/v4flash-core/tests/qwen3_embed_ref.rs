//! Embed-phase gates E0 (tokenizer) and E1 (CPU oracle) against the HF
//! reference dump (docs/v41/EMBED_PHASE_DESIGN.md §10). CPU only; needs the
//! weights:
//!
//! ```text
//! QWEN3_EMBED_GGUF=/path/Qwen3-Embedding-4B-Q8_0.gguf \
//! QWEN3_EMBED_REF=~/.cache/deepstrix/goldens/qwen3emb/ref.json \
//!   nix develop -c cargo test --release -p v4flash-core --test qwen3_embed_ref -- --ignored --nocapture
//! ```
//!
//! The reference comes from `scripts/qwen3_embed/ref_embed.py`.
//! `QWEN3_EMBED_E1_MAX_TOKENS` (default 600) bounds which inputs E1 runs: the
//! oracle is f32 on the CPU, ~7.3 GFLOP per token.

use v4flash_core::qwen3_embed::{cosine, cpu_forward, Qwen3EmbedModel};
use v4flash_core::tokenizer::BpeVocab;
use v4flash_core::MappedGguf;

struct Case {
    kind: String,
    text: String,
    ids: Vec<u32>,
    embedding: Vec<f32>,
}

fn load() -> Option<(MappedGguf, Qwen3EmbedModel, BpeVocab, Vec<Case>, bool)> {
    let (Ok(gguf), Ok(refp)) = (std::env::var("QWEN3_EMBED_GGUF"), std::env::var("QWEN3_EMBED_REF")) else {
        eprintln!("QWEN3_EMBED_GGUF / QWEN3_EMBED_REF unset: skipped");
        return None;
    };
    let file = MappedGguf::open(&gguf).expect("open gguf");
    let model = Qwen3EmbedModel::from_gguf(&file).expect("model");
    let vocab = BpeVocab::from_gguf(file.gguf()).expect("vocab");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&refp).expect("read ref")).expect("ref json");
    let eos_appended = v["eos_appended"].as_bool().unwrap_or(false);
    let cases = v["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| Case {
            kind: c["kind"].as_str().unwrap_or("").to_string(),
            text: c["text"].as_str().expect("text").to_string(),
            ids: c["ids"].as_array().expect("ids").iter().map(|x| x.as_u64().expect("id") as u32).collect(),
            embedding: c["embedding"].as_array().expect("embedding").iter().map(|x| x.as_f64().expect("f") as f32).collect(),
        })
        .collect();
    Some((file, model, vocab, cases, eos_appended))
}

fn ours(vocab: &BpeVocab, model: &Qwen3EmbedModel, text: &str) -> Vec<u32> {
    vocab.encode_qwen2(text).into_iter().map(|t| t as u32).chain([model.eos_id]).collect()
}

/// The model card's own example (Qwen/Qwen3-Embedding-4B, transformers):
/// two instructed queries x two documents, scores
/// `[[0.7534, 0.1147], [0.0320, 0.6258]]` (vLLM: within 0.002). Needs only
/// `QWEN3_EMBED_GGUF`: tokenizer + CPU oracle on the real Q8_0 weights vs a
/// published number. Pre-registered: every score within 0.01 (Q8_0 budget).
#[test]
#[ignore]
fn model_card_scores() {
    let Ok(gguf) = std::env::var("QWEN3_EMBED_GGUF") else {
        eprintln!("QWEN3_EMBED_GGUF unset: skipped");
        return;
    };
    let file = MappedGguf::open(&gguf).expect("open gguf");
    let model = Qwen3EmbedModel::from_gguf(&file).expect("model");
    let vocab = BpeVocab::from_gguf(file.gguf()).expect("vocab");
    let task = "Given a web search query, retrieve relevant passages that answer the query";
    let texts = [
        format!("Instruct: {task}\nQuery:What is the capital of China?"),
        format!("Instruct: {task}\nQuery:Explain gravity"),
        "The capital of China is Beijing.".to_string(),
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.".to_string(),
    ];
    let inputs: Vec<Vec<u32>> = texts.iter().map(|t| ours(&vocab, &model, t)).collect();
    for (t, ids) in texts.iter().zip(&inputs) {
        eprintln!("{} tokens: {:?}", ids.len(), t.chars().take(40).collect::<String>());
    }
    let t0 = std::time::Instant::now();
    let e = cpu_forward(&model, &file, &inputs).expect("cpu forward");
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum::<f64>();
    let got = [[dot(&e[0], &e[2]), dot(&e[0], &e[3])], [dot(&e[1], &e[2]), dot(&e[1], &e[3])]];
    let want = [[0.7534, 0.1147], [0.0320, 0.6258]];
    eprintln!("scores {got:?} (model card {want:?}), {:.1} s", t0.elapsed().as_secs_f64());
    for i in 0..2 {
        for j in 0..2 {
            assert!((got[i][j] - want[i][j]).abs() <= 0.01, "score[{i}][{j}] = {:.4}, model card {:.4}", got[i][j], want[i][j]);
        }
    }
}

#[test]
#[ignore]
fn e0_tokenizer_matches_reference() {
    let Some((_f, model, vocab, cases, eos_appended)) = load() else { return };
    assert!(
        eos_appended,
        "the reference tokenizer does NOT append <|endoftext|>: the design's EOS assumption (§2) is wrong -- stop and revisit"
    );
    let mut bad = 0;
    for (i, c) in cases.iter().enumerate() {
        let got = ours(&vocab, &model, &c.text);
        if got != c.ids {
            bad += 1;
            let first = got.iter().zip(&c.ids).position(|(a, b)| a != b).unwrap_or(got.len().min(c.ids.len()));
            eprintln!(
                "E0 MISMATCH case {i} ({}): ours {} ids, ref {} ids, first difference at {first}: ours {:?} ref {:?}",
                c.kind,
                got.len(),
                c.ids.len(),
                &got[first.saturating_sub(2)..(first + 4).min(got.len())],
                &c.ids[first.saturating_sub(2)..(first + 4).min(c.ids.len())],
            );
        }
    }
    eprintln!("E0: {} / {} cases identical", cases.len() - bad, cases.len());
    assert_eq!(bad, 0, "E0: {bad} cases differ (each must be explained by design §8 and signed off)");
}

#[test]
#[ignore]
fn e1_cpu_oracle_matches_reference() {
    let Some((file, model, _vocab, cases, _)) = load() else { return };
    let max: usize = std::env::var("QWEN3_EMBED_E1_MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(600);
    let picked: Vec<&Case> = cases.iter().filter(|c| c.ids.len() <= max).collect();
    // The reference's own ids: E1 isolates the forward from the tokenizer.
    let inputs: Vec<Vec<u32>> = picked.iter().map(|c| c.ids.clone()).collect();
    let t0 = std::time::Instant::now();
    let got = cpu_forward(&model, &file, &inputs).expect("cpu forward");
    eprintln!("E1: {} inputs, {} tokens, {:.1} s", inputs.len(), inputs.iter().map(Vec::len).sum::<usize>(), t0.elapsed().as_secs_f64());
    let mut min = f64::INFINITY;
    for (c, g) in picked.iter().zip(&got) {
        let cs = cosine(g, &c.embedding);
        eprintln!("E1 {:12} {:5} tokens  cos {cs:.6}", c.kind, c.ids.len());
        min = min.min(cs);
    }
    eprintln!("E1: min cosine {min:.6} over {} inputs", picked.len());
    assert!(min >= 0.998, "E1: min cosine {min} < 0.998");
}
