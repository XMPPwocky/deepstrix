//! Sustained decode-throughput benchmark for the het orchestrator.
//!
//! No tracing-perfetto layer, no per-stage DEBUG spans — just wall-clock
//! time for N tokens. Reports avg tok/s + per-token min/median/max.
//!
//! Run:
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 BENCH_TOKENS=100 \
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_decode -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};

use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{COMPRESS_RATIOS, HC_DIM, N_LAYER, SWA_WINDOW};
use v4flash_kernels::het::{
    DgpuScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine, IgpuScratch,
};
use v4flash_kernels::{oracle::ActivationDump, RopeParams};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
}

fn pick_dgpu_device() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 (9070 XT) device found"))
}

fn pick_igpu_device() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 (Strix iGPU) device found"))
}

#[test]
#[ignore]
fn bench_decode_het_parallel() -> eyre::Result<()> {
    install_panic_handler()?;

    let n_tokens: i32 = std::env::var("BENCH_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let warmup: i32 = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    // FAKE_POS=N stamps per-layer KV counters to simulate having decoded
    // N prior tokens (n_raw=min(N,W); cs.n_comp=N/ratio; ics.n_comp=N/ratio
    // for ratio==4 layers). Buffers stay zero-initialised — timing is
    // representative (FMA/BW don't depend on values) but output is
    // garbage. Use this for long-context decode benchmarks without
    // paying the hour-long cost of real prefill at depth.
    //
    // A short real warmup runs first to capture per-layer HIP graphs;
    // without it the first faked token spikes on graph capture.
    let fake_pos: u32 = std::env::var("FAKE_POS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    eprintln!("bench config: n_tokens={n_tokens}, warmup={warmup}, fake_pos={fake_pos}");

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;

    let dgpu = pick_dgpu_device()?;
    let igpu = pick_igpu_device()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    eprintln!("dGPU={} iGPU={}", dgpu_arch, igpu_arch);

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
    eprintln!("loaded.");

    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let n_positions = n_tokens + PROMPT_TOKENS.len() as i32 - 1;
    // Size n_kv_max to fit the faked depth too. comp_kv buffers are sized
    // from n_kv_max so they need to be large enough to hold the simulated
    // state's n_comp slots, even though we don't fill them.
    let n_kv_max = (n_positions as u32).max(fake_pos + n_tokens as u32 + 64);
    let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;

    // Use the dump's first-position embedding as input; we're not
    // sampling, just exercising the forward path. Each position reads
    // its own layer_input_residual from the dump if available, else
    // repeats the last.
    let max_inp_pos = dump.n_logit_rows as i32 + PROMPT_TOKENS.len() as i32 - 2;

    // Preload all per-position input residuals up front so the bench
    // loop measures only inference time, not first-touch disk I/O.
    // Without this, the first post-warmup token always spikes to
    // 300-400ms as the OS reads a fresh dump file.
    eprintln!("preloading {} input residuals...", n_positions);
    let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(n_positions as usize);
    for pos in 0..n_positions {
        let inp_pos = pos.min(max_inp_pos);
        let inp_entry = dump
            .tensor("layer_input_residual", 0, inp_pos)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{inp_pos}"))?;
        let input_hc = dump.read_f32(inp_entry)?;
        assert_eq!(input_hc.len(), HC_DIM as usize);
        inputs.push(input_hc);
    }
    eprintln!("preloaded.");

    // If FAKE_POS is set: short real warmup to capture HIP graphs, then
    // stamp per-layer counters to simulate having decoded `fake_pos`
    // tokens. After this, the bench loop measures decode at the faked
    // depth.
    let start_pos: i32 = if fake_pos > 0 {
        let graph_warm = 20i32.min(n_positions);
        eprintln!(
            "FAKE_POS={fake_pos}: real-forwarding {graph_warm} tokens to capture HIP graphs..."
        );
        for pos in 0..graph_warm {
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
                &inputs[pos as usize],
                pos as u32,
                token_id,
            )?;
        }
        eprintln!("stamping per-layer counters for fake depth {fake_pos}");
        for layer in 0..N_LAYER as usize {
            let ls = &mut state.layers[layer];
            ls.n_raw = SWA_WINDOW.min(fake_pos);
            ls.raw_off = 0; // graph_warm=20 < W, window starts at slot 0
            let ratio = COMPRESS_RATIOS[layer];
            if ratio > 0 {
                if let Some(cs) = ls.compressor.as_mut() {
                    cs.n_comp = fake_pos / ratio;
                }
                if let Some(ics) = ls.indexer_compressor.as_mut() {
                    ics.n_comp = fake_pos / ratio;
                }
            }
        }
        eprintln!(
            "fake counters set: ratio=4 n_comp={}, ratio=128 n_comp={}",
            fake_pos / 4,
            fake_pos / 128
        );
        // Continue at the faked position. The bench loop's positions go
        // [fake_pos, fake_pos+n_positions).
        fake_pos as i32
    } else {
        0
    };

    let mut token_us: Vec<u64> = Vec::with_capacity(n_positions as usize);
    let mut token_host_us: Vec<u64> = Vec::with_capacity(n_positions as usize);
    let mut token_sync_us: Vec<u64> = Vec::with_capacity(n_positions as usize);
    let bench_start = Instant::now();
    for i in 0..n_positions {
        let pos = start_pos + i;
        // For fake-pos runs, all bench tokens are post-prompt — use
        // input residual #0 cycled (it's just exercising kernels, output
        // is meaningless at faked depth anyway).
        let input_idx = (i as usize) % inputs.len();
        let token_id = if (i as usize) < PROMPT_TOKENS.len() && fake_pos == 0 {
            PROMPT_TOKENS[i as usize]
        } else {
            0
        };
        let input_hc = &inputs[input_idx];

        let t = Instant::now();
        engine.forward_token(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &weights,
            input_hc,
            pos as u32,
            token_id,
        )?;
        let dt = t.elapsed().as_micros() as u64;
        token_us.push(dt);
        use std::sync::atomic::Ordering;
        token_host_us.push(engine.last_host_us.load(Ordering::Relaxed));
        token_sync_us.push(engine.last_sync_us.load(Ordering::Relaxed));
    }
    let bench_wall = bench_start.elapsed();

    // Drop warmup tokens from the stats.
    let warm_us: Vec<u64> = token_us.iter().skip(warmup as usize).copied().collect();
    let warm_count = warm_us.len() as f64;
    let warm_total: u64 = warm_us.iter().sum();
    let mut sorted = warm_us.clone();
    sorted.sort_unstable();
    let median_us = sorted.get(sorted.len() / 2).copied().unwrap_or(0);
    let min_us = *sorted.first().unwrap_or(&0);
    let max_us = *sorted.last().unwrap_or(&0);
    let avg_us = warm_total as f64 / warm_count;
    let tps = 1_000_000.0 / avg_us;

    eprintln!(
        "BENCH (n={} tokens after {} warmup): wall={:.2}s avg={:.2}ms median={:.2}ms min={:.2}ms max={:.2}ms => {:.2} tok/s",
        warm_us.len(),
        warmup,
        bench_wall.as_secs_f64(),
        avg_us / 1000.0,
        median_us as f64 / 1000.0,
        min_us as f64 / 1000.0,
        max_us as f64 / 1000.0,
        tps,
    );

    eprintln!(
        "BENCH context: dGPU={} (gfx1201) + iGPU={} (gfx1151), V4-Flash {} layers, HetParallel mode",
        dgpu.id, igpu.id, v4flash_kernels::config::N_LAYER
    );

    // Per-token timing dump — correlate token speed against position
    // patterns (compressor boundaries, SWA crossover, etc.). Each row
    // shows: pos | µs | "B2" if any ratio-2 boundary fires, "B4" if
    // any ratio-4, "SWA" if past SWA_WINDOW.
    if std::env::var("BENCH_PER_TOKEN").is_ok() {
        const SWA_WINDOW: u32 = 128;
        eprintln!("BENCH per-token (after warmup):");
        eprintln!("    pos    total      host    sync   flags");
        for (i, &t) in token_us.iter().enumerate() {
            if (i as i32) < warmup { continue; }
            let pos = i as u32;
            let h = token_host_us[i];
            let s = token_sync_us[i];
            let b2 = (pos + 1) % 2 == 0;
            let b4 = (pos + 1) % 4 == 0;
            let swa = pos >= SWA_WINDOW;
            let flags = format!("{}{}",
                if b4 {"B4"} else if b2 {"B2"} else {"--"},
                if swa {" SWA"} else {""});
            eprintln!("  T{:>3} {:>7.2}  {:>6.2}  {:>6.2}  {}",
                pos, t as f64 / 1000.0, h as f64 / 1000.0, s as f64 / 1000.0, flags);
        }
    }

    // M27 summary: how is total wall split between host enqueue vs
    // device sync wait, on the slowest vs fastest tokens?
    let warm_host: Vec<u64> = token_host_us.iter().skip(warmup as usize).copied().collect();
    let warm_sync: Vec<u64> = token_sync_us.iter().skip(warmup as usize).copied().collect();
    let host_avg = warm_host.iter().sum::<u64>() as f64 / warm_host.len() as f64;
    let sync_avg = warm_sync.iter().sum::<u64>() as f64 / warm_sync.len() as f64;
    // sort warm_us with paired host/sync indexes so we can pull split for fastest/slowest tokens.
    let mut idx: Vec<usize> = (0..warm_us.len()).collect();
    idx.sort_by_key(|&i| warm_us[i]);
    let p10_i = idx[idx.len() / 10];
    let p50_i = idx[idx.len() / 2];
    let p90_i = idx[(idx.len() * 9) / 10];
    let p99_i = idx[(idx.len() * 99) / 100];
    eprintln!("BENCH host vs sync split:");
    eprintln!("  avg:  host {:>6.2} ms  sync {:>6.2} ms  total {:>6.2}",
        host_avg / 1000.0, sync_avg / 1000.0, (host_avg + sync_avg) / 1000.0);
    for (label, i) in [("p10 ", p10_i), ("p50 ", p50_i), ("p90 ", p90_i), ("p99 ", p99_i)] {
        eprintln!("  {}: host {:>6.2} ms  sync {:>6.2} ms  total {:>6.2}",
            label, warm_host[i] as f64 / 1000.0, warm_sync[i] as f64 / 1000.0,
            warm_us[i] as f64 / 1000.0);
    }

    // Per-decile latency. Reads from the same `sorted` Vec the
    // min/median/max came from — n is small enough that
    // sorted[idx_at_pct] is a perfectly good quantile estimate. p99
    // included because it's the load-bearing "occasional slow token"
    // metric (a single bad token at 3× median drags the avg far more
    // than the median moves).
    eprintln!("BENCH percentiles (ms):");
    let pcts: [(f64, &str); 12] = [
        (0.0, "p0  "),
        (10.0, "p10 "),
        (20.0, "p20 "),
        (30.0, "p30 "),
        (40.0, "p40 "),
        (50.0, "p50 "),
        (60.0, "p60 "),
        (70.0, "p70 "),
        (80.0, "p80 "),
        (90.0, "p90 "),
        (99.0, "p99 "),
        (100.0, "p100"),
    ];
    let n = sorted.len() as f64;
    for (p, label) in pcts {
        // p in [0,100]; index in [0, len-1].
        let idx = ((p / 100.0) * (n - 1.0)).round() as usize;
        let us = sorted[idx];
        let tps = 1_000_000.0 / us as f64;
        eprintln!(
            "  {label} {:>7.2} ms  ({:>5.2} tok/s)",
            us as f64 / 1000.0,
            tps
        );
    }

    // Print any post-warmup token >2x the median for fault diagnosis.
    let outlier_thresh = median_us.saturating_mul(2);
    let outliers: Vec<(usize, u64)> = token_us
        .iter()
        .enumerate()
        .filter(|(i, &t)| (*i as i32) >= warmup && t > outlier_thresh)
        .map(|(i, &t)| (i, t))
        .collect();
    if !outliers.is_empty() {
        eprintln!("BENCH outliers (>{:.2}ms):", outlier_thresh as f64 / 1000.0);
        for (i, t) in outliers.iter().take(20) {
            eprintln!(
                "  T{} (idx-after-warmup {}): {:.2}ms",
                i,
                i - warmup as usize,
                *t as f64 / 1000.0
            );
        }
        if outliers.len() > 20 {
            eprintln!("  ...{} more outliers truncated", outliers.len() - 20);
        }
    }

    Ok(())
}

