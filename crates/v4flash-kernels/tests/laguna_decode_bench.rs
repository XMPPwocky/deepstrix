//! Laguna-S-2.1 — REAL long-context DECODE benchmark (het / dual-GPU path).
//!
//! The other decode tests only ever reach ~12 KV positions, so no decode
//! kernel has ever been measured at real context. This bench:
//!   1. Prefills the pangram ("The quick brown fox") and asserts the greedy
//!      next token == 22718 (" jumps") — the PARITY GATE (real positions).
//!   2. Times `decode_step` at *jumped* KV positions to synthesize 4K / 32K
//!      (/ 96K) context. The KV *content* past the prompt is uninitialized
//!      garbage — irrelevant to timing / roofline, which is all we measure
//!      here (per-step compute is identical regardless of KV values).
//!
//! Reports steady-state ms/token: drops warmup + high outlier spikes (spikes
//! are noted; if periodic they're worth investigating — see stddev/max).
//!
//! Env:
//!   LAGUNA_BENCH_CTXS   comma list of contexts to time (default "4096,32768")
//!   LAGUNA_BENCH_ITERS  timed steps per context (default 24)
//!   LAGUNA_BENCH_WARMUP warmup steps dropped per context (default 4)
//!   LAGUNA_DECODE_ATTN_NAIVE=1  force naive attention (A/B the flash gate)
//!
//! Run (server stopped; GPUs free):
//!   nix develop --command cargo test --release -p v4flash-kernels \
//!       --test laguna_decode_bench -- --ignored --nocapture
//!
//! max_kv is sized to the largest requested context. NOTE: the het KV cache
//! lives on the 16 GB dGPU, so ~32K is the practical ceiling there (32K ≈
//! 6.3 GB KV + ~3.3 GB weights). 96K KV (~19 GB) overflows the dGPU; use the
//! single-device iGPU path for that (see report).

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::Gguf;
use v4flash_core::tokenizer::BpeVocab;
use v4flash_hip::Device;
use v4flash_kernels::laguna_het::LagunaHetModel;

const GGUF_PATH: &str = "/persist/lumi/models/laguna-s-2.1-int4/laguna-s-2.1-Q4_K_M.gguf";
const FIRST_TOKEN: usize = 22718; // oracle greedy " jumps"

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Robust steady-state summary: drop `warmup` leading samples, then report
/// median, trimmed mean (drop top/bottom 10%), min, max and the count of
/// "spikes" (> 1.5x median).
struct Stats {
    median: f64,
    trimmed_mean: f64,
    min: f64,
    max: f64,
    spikes: usize,
    n: usize,
}

fn summarize(samples: &[f64]) -> Stats {
    let mut s: Vec<f64> = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let median = if n == 0 { 0.0 } else { s[n / 2] };
    let trim = n / 10;
    let core = &s[trim..n.saturating_sub(trim).max(trim)];
    let trimmed_mean = if core.is_empty() { median } else { core.iter().sum::<f64>() / core.len() as f64 };
    let spikes = samples.iter().filter(|&&x| x > 1.5 * median).count();
    Stats {
        median,
        trimmed_mean,
        min: s.first().copied().unwrap_or(0.0),
        max: s.last().copied().unwrap_or(0.0),
        spikes,
        n,
    }
}

#[test]
#[ignore = "drives BOTH GPUs + needs the 75GB Laguna GGUF; run explicitly"]
fn laguna_decode_bench() -> eyre::Result<()> {
    let _ = v4flash_hip::install_panic_handler();

    if !std::path::Path::new(GGUF_PATH).exists() {
        eprintln!("SKIP: {GGUF_PATH} not present");
        return Ok(());
    }

    let ctxs: Vec<usize> = std::env::var("LAGUNA_BENCH_CTXS")
        .unwrap_or_else(|_| "4096,32768".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let iters = env_usize("LAGUNA_BENCH_ITERS", 24);
    let warmup = env_usize("LAGUNA_BENCH_WARMUP", 4);
    let max_ctx = ctxs.iter().copied().max().unwrap_or(4096);

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
        .ok_or_else(|| eyre!("no gfx1151 (Strix iGPU) device"))?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    println!("dGPU id={} arch={dgpu_arch}   iGPU id={} arch={igpu_arch}", dgpu.id, igpu.id);
    println!("ctxs={ctxs:?} iters={iters} warmup={warmup} max_ctx={max_ctx}");

    // ----- tokenize -----
    let g = Gguf::open(GGUF_PATH)?;
    let vocab = BpeVocab::from_gguf(&g)?;
    let prompt = "The quick brown fox";
    let ids: Vec<usize> = vocab.encode_laguna(prompt).into_iter().map(|i| i as usize).collect();
    println!("prompt {prompt:?} -> {ids:?}");
    assert_eq!(ids, vec![2, 785, 3454, 21438, 42850], "tokenizer parity");

    // ----- load: max_kv sized to the biggest ctx we'll jump to (+ margin) -----
    let max_kv = max_ctx + iters + warmup + 8;
    let t_load = Instant::now();
    let mut model =
        LagunaHetModel::load(GGUF_PATH, dgpu.clone(), &dgpu_arch, igpu.clone(), &igpu_arch, max_kv)?;
    println!("model loaded in {:.1}s (max_kv={max_kv})", t_load.elapsed().as_secs_f32());

    // ----- PARITY GATE: prefill pangram, greedy next token must be 22718 -----
    let (first_tok, first_logit) = model.prefill(&ids)?;
    println!("prefill -> next token {first_tok} (logit {first_logit:.4})");
    assert_eq!(first_tok, FIRST_TOKEN, "PARITY: first generated token must be 22718 (\" jumps\")");
    println!("[OK parity] first token = {FIRST_TOKEN}");

    // ----- timed decode at jumped KV positions per context -----
    // We reuse a fixed input token; only `pos` jumps. Attention reads pos+1
    // keys from the cache (garbage past the prompt — timing only).
    let feed_tok = first_tok;
    println!("\n=== DECODE TIMING (steady-state ms/token) ===");
    for &ctx in &ctxs {
        if ctx + 1 > max_kv {
            println!("ctx {ctx}: SKIP (exceeds max_kv {max_kv})");
            continue;
        }
        let mut samples = Vec::with_capacity(iters);
        // warmup + timed, positions jump forward from `ctx`.
        model.reset_diag();
        for i in 0..(warmup + iters) {
            let pos = ctx + i;
            let t = Instant::now();
            let (_next, _logit) = model.decode_step(feed_tok, pos)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if i >= warmup {
                samples.push(ms);
            }
            if i + 1 == warmup {
                model.reset_diag(); // drop warmup from the diag accumulation
            }
        }
        // LAGUNA_HET_DIAG=1: report the sequential dGPU-attn vs iGPU-MoE split
        // (per token, averaged over the timed steps). Only meaningful with diag.
        let (d_us, i_us) = model.diag_split();
        if d_us + i_us > 0 {
            let n = iters as f64;
            println!(
                "  [diag] dGPU-attn {:.2} ms/tok  iGPU-MoE {:.2} ms/tok  (sequential; overlap headroom ≈ min = {:.2} ms)",
                d_us as f64 / n / 1e3,
                i_us as f64 / n / 1e3,
                (d_us.min(i_us)) as f64 / n / 1e3,
            );
        }
        let st = summarize(&samples);
        println!(
            "ctx {:>6}: median {:6.2} ms  trimmed-mean {:6.2} ms  ({:6.2} tok/s)  [min {:.2} max {:.2} spikes {} n {}]",
            ctx,
            st.median,
            st.trimmed_mean,
            1000.0 / st.trimmed_mean,
            st.min,
            st.max,
            st.spikes,
            st.n,
        );
    }

    println!("\n[done]");
    Ok(())
}
