//! Capture a perfetto trace at LONG context — warm up to `WARMUP_POS`
//! positions without perfetto attached, then attach and capture
//! `TRACE_TOKENS` decode tokens. Used to diagnose decode slowdown as
//! context grows.
//!
//! Run two times to compare:
//! ```text
//! # short context baseline (no warmup, trace from pos 0):
//! HIP_VISIBLE_DEVICES=0,1 \
//!   WARMUP_POS=0 TRACE_TOKENS=10 \
//!   PERFETTO_DEVICE_OUT=/tmp/decode-short.pftrace \
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test perfetto_trace_long -- --ignored --nocapture
//!
//! # long context (warm up 1500 positions, then trace):
//! HIP_VISIBLE_DEVICES=0,1 \
//!   WARMUP_POS=1500 TRACE_TOKENS=10 \
//!   PERFETTO_DEVICE_OUT=/tmp/decode-long.pftrace \
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test perfetto_trace_long -- --ignored --nocapture
//! ```
//!
//! Compare span widths (especially `dgpu.attn_compute`, `dgpu.compressor`,
//! `dgpu.comp_kv_append`) at the two contexts. Anything that grows
//! linearly with WARMUP_POS is paying the n_comp scaling tax.

use std::fs;
use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};

use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::forward::{COMPRESS_RATIOS, HC_DIM, N_LAYER, SWA_WINDOW};
use v4flash_kernels::het::{
    DgpuScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine, IgpuScratch,
};
use v4flash_kernels::{ActivationDump, RopeParams};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
}

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201"))
}

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

#[test]
#[ignore]
fn perfetto_trace_long() -> eyre::Result<()> {
    install_panic_handler()?;

    let warmup_pos: i32 = std::env::var("WARMUP_POS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let trace_tokens: i32 = std::env::var("TRACE_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let out_path = std::env::var("PERFETTO_DEVICE_OUT")
        .unwrap_or_else(|_| format!("/tmp/decode-warmup{warmup_pos}.pftrace"));
    eprintln!("warmup_pos={warmup_pos}  trace_tokens={trace_tokens}");
    eprintln!("perfetto out: {out_path}");
    let _ = fs::remove_file(&out_path);

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;

    let dgpu = pick_dgpu()?;
    let igpu = pick_igpu()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;

    let rope_for_layer = |layer: i32| -> eyre::Result<RopeParams> {
        let entry = dump
            .weight("rope_params", layer)
            .ok_or_else(|| eyre!("missing weight:rope_params for L{layer}"))?;
        let floats = dump.read_f32(entry)?;
        let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
        RopeParams::from_dump_blob(&floats, n_ctx_orig)
    };

    eprintln!("loading het weights...");
    let weights = HetModelWeights::load_all(&gguf, dgpu, igpu, &rope_for_layer)?;

    let mut engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let n_positions = warmup_pos + trace_tokens;
    let mut state = HetModelState::alloc(dgpu, igpu, n_positions as u32)?;

    // Preload one input residual (the first prompt position). For pure
    // timing we don't care about values; the kernel structure is
    // pos-driven, not data-driven, and the router will pick whatever
    // experts the residual happens to score against — fine for tracing.
    let inp_entry = dump
        .tensor("layer_input_residual", 0, 0)
        .ok_or_else(|| eyre!("missing layer_input_residual L0 T0"))?;
    let input_hc = dump.read_f32(inp_entry)?;
    assert_eq!(input_hc.len(), HC_DIM as usize);

    // Phase 1: WARMUP. Two modes:
    //   FAKE_WARMUP=1 → skip real forwards, just bump the per-layer
    //     n_raw / n_comp counters to simulate position WARMUP_POS.
    //     Buffers stay zero-initialized; attention math runs over the
    //     right amount of data with the right shapes, just with zero
    //     values. Timing is representative (FMA/BW don't depend on
    //     values), correctness is not (output will be garbage).
    //   default → real forwards token-by-token (slow but valid output).
    let fake_warmup = std::env::var("FAKE_WARMUP").ok().as_deref() == Some("1");
    if fake_warmup && warmup_pos > 0 {
        // First do a short REAL warmup so all per-layer HIP graphs get
        // captured (~5 forwards is enough — each layer captures its
        // mhc_pre_attn / q_chain / output_proj / etc once and replays
        // forever after). Without this, the first timed token after
        // the fake bump spikes to hundreds of ms on graph capture.
        let graph_warm = 20.min(warmup_pos);
        eprintln!("FAKE_WARMUP=1: real-forwarding {graph_warm} tokens for graph capture...");
        for pos in 0..graph_warm {
            let token_id = if (pos as usize) < PROMPT_TOKENS.len() {
                PROMPT_TOKENS[pos as usize]
            } else { 0 };
            engine.forward_token(
                &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights,
                &input_hc, pos as u32, token_id,
            )?;
        }
        eprintln!("setting counters for pos={warmup_pos}");
        for layer in 0..N_LAYER as usize {
            let ls = &mut state.layers[layer];
            ls.n_raw = SWA_WINDOW.min(warmup_pos as u32);
            let ratio = COMPRESS_RATIOS[layer];
            if ratio > 0 {
                if let Some(cs) = ls.compressor.as_mut() {
                    cs.n_comp = (warmup_pos as u32) / ratio;
                }
                if let Some(ics) = ls.indexer_compressor.as_mut() {
                    ics.n_comp = (warmup_pos as u32) / ratio;
                }
            }
        }
        eprintln!(
            "fake warmup done: ratio=4 n_comp={}, ratio=128 n_comp={}",
            (warmup_pos as u32) / 4,
            (warmup_pos as u32) / 128
        );
    } else {
        eprintln!("warming up {warmup_pos} positions (real forwards, no trace)...");
        let warm_start = std::time::Instant::now();
        for pos in 0..warmup_pos {
            let token_id = if (pos as usize) < PROMPT_TOKENS.len() {
                PROMPT_TOKENS[pos as usize]
            } else {
                0
            };
            engine.forward_token(
                &mut dgpu_scratch,
                &mut igpu_scratch,
                &mut state,
                &weights,
                &input_hc,
                pos as u32,
                token_id,
            )?;
            if pos > 0 && pos % 100 == 0 {
                eprintln!(
                    "  warmup pos={pos} ({:.1} tok/s avg)",
                    (pos + 1) as f64 / warm_start.elapsed().as_secs_f64()
                );
            }
        }
        let warm_secs = warm_start.elapsed().as_secs_f64();
        if warmup_pos > 0 {
            eprintln!(
                "warmup done: {warmup_pos} tok in {warm_secs:.1}s = {:.2} tok/s avg",
                warmup_pos as f64 / warm_secs
            );
        }
    }

    // Phase 2: optionally attach perfetto, then time trace_tokens. When
    // SKIP_PERFETTO=1 the event-pool overhead is skipped — needed to get
    // realistic tok/s without the per-event timing tax inflating numbers.
    let skip_perfetto = std::env::var("SKIP_PERFETTO").ok().as_deref() == Some("1");
    if skip_perfetto {
        eprintln!("SKIP_PERFETTO=1: timing only, no trace output");
    } else {
        eprintln!("attaching perfetto + tracing {trace_tokens} tokens at pos={warmup_pos}+...");
        engine.attach_perfetto(&out_path)?;
    }
    let trace_start = std::time::Instant::now();
    let mut per_token_us: Vec<u64> = Vec::with_capacity(trace_tokens as usize);
    for i in 0..trace_tokens {
        let pos = warmup_pos + i;
        let t0 = std::time::Instant::now();
        engine.forward_token(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &weights,
            &input_hc,
            pos as u32,
            0,
        )?;
        per_token_us.push(t0.elapsed().as_micros() as u64);
    }
    let trace_secs = trace_start.elapsed().as_secs_f64();
    eprintln!(
        "{} done: {trace_tokens} tok in {trace_secs:.2}s = {:.2} tok/s",
        if skip_perfetto { "timed" } else { "trace" },
        trace_tokens as f64 / trace_secs
    );
    let mut sorted = per_token_us.clone();
    sorted.sort_unstable();
    let p = |q: usize| sorted[(sorted.len() * q / 100).min(sorted.len() - 1)] as f64 / 1000.0;
    eprintln!(
        "per-token ms: p0={:.2} p10={:.2} p50={:.2} p90={:.2} p99={:.2} p100={:.2}",
        p(0), p(10), p(50), p(90), p(99), p(100)
    );
    if !skip_perfetto {
        eprintln!("wrote {out_path}");
        eprintln!("open at https://ui.perfetto.dev");
    }
    Ok(())
}
