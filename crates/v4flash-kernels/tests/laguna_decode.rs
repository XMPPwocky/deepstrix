//! Laguna-S-2.1 — REAL on-device decode. Promotes the correctness spike into a
//! runnable forward: tokenizes "The quick brown fox", prefills the prompt into a
//! per-layer KV cache, then greedily decodes 8 tokens ENTIRELY on the GPU
//! (norm/rope/router/gate/softmax all on device). Verifies the first generated
//! token == 22718 (" jumps"), the oracle's greedy continuation.
//!
//! Run (server stopped; GPU free):
//!   nix develop --command cargo test --release -p v4flash-kernels \
//!       --test laguna_decode -- --ignored --nocapture

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::Gguf;
use v4flash_core::tokenizer::BpeVocab;
use v4flash_hip::Device;
use v4flash_kernels::laguna::LagunaModel;

const GGUF_PATH: &str = "/persist/lumi/models/laguna-s-2.1-int4/laguna-s-2.1-Q4_K_M.gguf";
const N_GEN: usize = 8;
const FIRST_TOKEN: usize = 22718; // oracle greedy " jumps"

/// GPT-2 byte<->unicode table (for pretty-printing decoded tokens).
fn byte_decoder() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8))
        .collect()
}

fn decode_ids(vocab: &BpeVocab, ids: &[usize]) -> String {
    let dec = byte_decoder();
    let mut bytes = Vec::new();
    for &id in ids {
        if let Some(txt) = vocab.token_text(id as i32) {
            let s = String::from_utf8_lossy(txt);
            for ch in s.chars() {
                if let Some(&b) = dec.get(&ch) {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

#[test]
#[ignore = "drives the GPU + needs the 75GB Laguna GGUF; run explicitly"]
fn laguna_decode() -> eyre::Result<()> {
    let _ = v4flash_hip::install_panic_handler();

    if !std::path::Path::new(GGUF_PATH).exists() {
        eprintln!("SKIP: {GGUF_PATH} not present");
        return Ok(());
    }

    // ----- device -----
    // Experts are loaded RESIDENT (~59 GB), so we run on the iGPU (gfx1151),
    // whose unified memory (no_system_mem_limit) can hold them. The 16 GB
    // dGPU cannot. Everything (attention + non-expert + MoE) runs here; a
    // dGPU/iGPU compute split is the next perf step.
    let dev = Device::all()?
        .into_iter()
        .find(|d| {
            d.properties()
                .map(|p| p.gcn_arch_name.starts_with("gfx1151"))
                .unwrap_or(false)
        })
        .ok_or_else(|| eyre!("no gfx1151 (Strix iGPU) device"))?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    println!("device id={} arch={arch}", dev.id);

    // ----- tokenize -----
    let g = Gguf::open(GGUF_PATH)?;
    let vocab = BpeVocab::from_gguf(&g)?;
    let prompt = "The quick brown fox";
    let ids: Vec<usize> = vocab.encode_laguna(prompt).into_iter().map(|i| i as usize).collect();
    println!("prompt {prompt:?} -> {ids:?}");
    assert_eq!(ids, vec![2, 785, 3454, 21438, 42850], "tokenizer parity");

    // ----- load model (weights resident, experts streamed) -----
    let t_load = Instant::now();
    let mut model = LagunaModel::load(GGUF_PATH, dev.id, &arch, ids.len() + N_GEN + 4)?;
    println!("model loaded in {:.1}s", t_load.elapsed().as_secs_f32());
    println!("hparams: {:?}", model.hparams());

    // ----- prefill -----
    let t_pref = Instant::now();
    let (first_tok, first_logit) = model.prefill(&ids)?;
    let pref_ms = t_pref.elapsed().as_secs_f64() * 1e3;
    println!(
        "\nprefill ({} tok) in {:.1} ms -> next token {} (logit {:.4})",
        ids.len(),
        pref_ms,
        first_tok,
        first_logit
    );

    // ----- greedy decode -----
    let mut gen: Vec<usize> = vec![first_tok];
    let mut pos = ids.len();
    let mut decode_ms = Vec::new();
    for _ in 0..(N_GEN - 1) {
        let cur = *gen.last().unwrap();
        let t = Instant::now();
        let (next, _logit) = model.decode_step(cur, pos)?;
        decode_ms.push(t.elapsed().as_secs_f64() * 1e3);
        gen.push(next);
        pos += 1;
    }

    // ----- report -----
    println!("\n=== GENERATED {} tokens ===", gen.len());
    println!("ids: {gen:?}");
    println!("text: {:?}", decode_ids(&vocab, &gen));
    println!("full: {:?}", decode_ids(&vocab, &ids.iter().chain(gen.iter()).copied().collect::<Vec<_>>()));
    let avg = decode_ms.iter().sum::<f64>() / decode_ms.len().max(1) as f64;
    println!(
        "\ndecode latency: {:?} ms  (avg {:.1} ms/token, {:.2} tok/s)",
        decode_ms.iter().map(|m| (m * 10.0).round() / 10.0).collect::<Vec<_>>(),
        avg,
        1000.0 / avg
    );

    // ----- gate -----
    assert_eq!(first_tok, FIRST_TOKEN, "first generated token must be 22718 (\" jumps\")");
    println!("\n[OK] first generated token = {FIRST_TOKEN} (\" jumps\") — matches oracle greedy");
    Ok(())
}
