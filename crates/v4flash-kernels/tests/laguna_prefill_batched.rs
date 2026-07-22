//! Laguna-S-2.1 — BATCHED prefill parity + speedup driver.
//!
//! Wires the validated pieces (batched GQA attention, reg-tiled by-expert MoE,
//! moe_group_builder, batched projections/norms) into `LagunaHetModel::
//! prefill_batched`, and asserts it produces the SAME greedy token + logits as
//! the sequential single-query `prefill` path. Then it decodes a few tokens
//! (the decode path is untouched) and prints prefill tok/s batched vs
//! sequential.
//!
//! Parity contract: attention + routing are numerically identical between the
//! two paths; the MoE GEMM differs only by atomic-accumulation reorder (~1e-4),
//! which is greedy-stable. Oracle greedy token = 22718 (" jumps").
//!
//! Run (server stopped; GPU free):
//!   nix develop --command cargo test --release -p v4flash-kernels \
//!       --test laguna_prefill_batched -- --ignored --nocapture

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::Gguf;
use v4flash_core::tokenizer::BpeVocab;
use v4flash_hip::Device;
use v4flash_kernels::laguna_het::LagunaHetModel;

const GGUF_PATH: &str = "/persist/lumi/models/laguna-s-2.1-int4/laguna-s-2.1-Q4_K_M.gguf";
const N_GEN: usize = 8;
const FIRST_TOKEN: usize = 22718; // oracle greedy " jumps"

#[test]
#[ignore = "drives BOTH GPUs + needs the 75GB Laguna GGUF; run explicitly"]
fn laguna_prefill_batched() -> eyre::Result<()> {
    let _ = v4flash_hip::install_panic_handler();

    if !std::path::Path::new(GGUF_PATH).exists() {
        eprintln!("SKIP: {GGUF_PATH} not present");
        return Ok(());
    }

    // ----- devices -----
    let devs = Device::all()?;
    let dgpu = devs
        .iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .cloned()
        .ok_or_else(|| eyre!("no gfx1201 (dGPU) device"))?;
    let igpu = devs
        .iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1151")).unwrap_or(false))
        .cloned()
        .ok_or_else(|| eyre!("no gfx1151 (iGPU) device"))?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    println!("dGPU id={} arch={dgpu_arch}   iGPU id={} arch={igpu_arch}", dgpu.id, igpu.id);

    // ----- tokenize -----
    let g = Gguf::open(GGUF_PATH)?;
    let vocab = BpeVocab::from_gguf(&g)?;
    let prompt = "The quick brown fox";
    let ids: Vec<usize> = vocab.encode_laguna(prompt).into_iter().map(|i| i as usize).collect();
    println!("prompt {prompt:?} -> {ids:?}");
    assert_eq!(ids, vec![2, 785, 3454, 21438, 42850], "tokenizer parity");

    // ----- load (max_kv big enough for the long speedup prompt below) -----
    let long_len: usize = std::env::var("LAGUNA_LONG_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    #[allow(non_snake_case)]
    let LONG_LEN = long_len;
    let t_load = Instant::now();
    let mut model = LagunaHetModel::load(
        GGUF_PATH,
        dgpu.clone(),
        &dgpu_arch,
        igpu.clone(),
        &igpu_arch,
        LONG_LEN + N_GEN + 4,
    )?;
    println!("model loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    // Profiling gate: LAGUNA_PROF=1 runs ONE batched prefill of a LONG_LEN
    // prompt and returns — a clean per-kernel trace for rocprofv3.
    if std::env::var("LAGUNA_PROF").is_ok() {
        let long: Vec<usize> = (0..LONG_LEN).map(|i| ids[i % ids.len()]).collect();
        let _ = model.prefill_batched(&long)?; // warm
        let iters: usize = std::env::var("LAGUNA_PROF_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let t = Instant::now();
        let mut tok = 0;
        for _ in 0..iters {
            let (tk, _) = model.prefill_batched(&long)?;
            tok = tk;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!(
            "PROF batched prefill {LONG_LEN} tok -> {tok}   {ms:.1} ms  = {:.1} tok/s  (B_MAX per env)",
            LONG_LEN as f64 / (ms / 1e3)
        );
        return Ok(());
    }

    // ===================== SEQUENTIAL prefill (baseline) =====================
    // warm once (JIT / first-touch), then time.
    let (seq_tok, seq_logit) = model.prefill(&ids)?;
    let t = Instant::now();
    let (seq_tok2, seq_logit2) = model.prefill(&ids)?;
    let seq_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(seq_tok, seq_tok2);
    println!(
        "\nSEQUENTIAL prefill ({} tok) {:.1} ms -> tok {} (logit {:.5})  = {:.2} tok/s",
        ids.len(), seq_ms, seq_tok, seq_logit,
        ids.len() as f64 / (seq_ms / 1e3)
    );

    // ===================== BATCHED prefill =====================
    let (bat_tok, bat_logit) = model.prefill_batched(&ids)?; // warm
    let t = Instant::now();
    let (bat_tok2, bat_logit2) = model.prefill_batched(&ids)?;
    let bat_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(bat_tok, bat_tok2);
    println!(
        "BATCHED   prefill ({} tok) {:.1} ms -> tok {} (logit {:.5})  = {:.2} tok/s",
        ids.len(), bat_ms, bat_tok, bat_logit,
        ids.len() as f64 / (bat_ms / 1e3)
    );

    // ----- parity -----
    let rel = (bat_logit2 - seq_logit2).abs() / seq_logit2.abs().max(1e-6);
    println!(
        "\n=== PARITY ===\n  greedy: sequential={seq_tok}  batched={bat_tok}  match={}\n  \
         max-logit: seq={seq_logit2:.5} bat={bat_logit2:.5} rel={rel:.3e}\n  \
         speedup batched/sequential = {:.2}x",
        seq_tok == bat_tok,
        seq_ms / bat_ms,
    );
    assert_eq!(bat_tok, seq_tok, "batched greedy token must match sequential");
    assert_eq!(bat_tok, FIRST_TOKEN, "greedy token must be 22718 (\" jumps\")");
    assert!(rel < 5e-2, "batched max-logit rel err {rel:.3e} too high");
    println!("  [OK] batched prefill matches sequential + oracle");

    // ===================== SPEEDUP at a realistic batch (LONG_LEN tokens) =====================
    // The reg-tiled MoE only amortizes weight bandwidth at large B; a 5-token
    // prompt is far too small. Cycle the prompt ids to LONG_LEN and time both
    // paths (greedy token isn't meaningful for the synthetic repeat — we only
    // compare wall time + confirm both paths agree).
    let long: Vec<usize> = (0..LONG_LEN).map(|i| ids[i % ids.len()]).collect();
    let (lseq_tok, _) = model.prefill(&long)?; // warm
    let t = Instant::now();
    let (lseq_tok2, _) = model.prefill(&long)?;
    let lseq_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(lseq_tok, lseq_tok2);

    let (lbat_tok, _) = model.prefill_batched(&long)?; // warm
    let t = Instant::now();
    let (lbat_tok2, _) = model.prefill_batched(&long)?;
    let lbat_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(lbat_tok, lbat_tok2);
    println!(
        "\n=== SPEEDUP ({LONG_LEN} tok, B_MAX=256) ===\n  \
         SEQUENTIAL {:.1} ms = {:.1} tok/s\n  \
         BATCHED    {:.1} ms = {:.1} tok/s\n  \
         speedup = {:.2}x   (greedy agree: {})",
        lseq_ms, LONG_LEN as f64 / (lseq_ms / 1e3),
        lbat_ms, LONG_LEN as f64 / (lbat_ms / 1e3),
        lseq_ms / lbat_ms,
        lseq_tok == lbat_tok,
    );

    // ----- restore the real 5-token prefill state, then decode a few tokens
    //       (decode path unchanged) -----
    let (bat_tok, _) = model.prefill_batched(&ids)?;
    // -----
    let mut gen = vec![bat_tok];
    let mut pos = ids.len();
    for _ in 0..(N_GEN - 1) {
        let cur = *gen.last().unwrap();
        let (next, _l) = model.decode_step(cur, pos)?;
        gen.push(next);
        pos += 1;
    }
    println!("\ndecoded ids (after batched prefill): {gen:?}");

    Ok(())
}
