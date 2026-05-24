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
use v4flash_kernels::forward::HC_DIM;
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
    eprintln!("bench config: n_tokens={n_tokens}, warmup={warmup}");

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
    let mut state = HetModelState::alloc(dgpu, igpu, n_positions as u32)?;

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

    let mut token_us: Vec<u64> = Vec::with_capacity(n_positions as usize);
    let bench_start = Instant::now();
    for pos in 0..n_positions {
        let token_id = if (pos as usize) < PROMPT_TOKENS.len() {
            PROMPT_TOKENS[pos as usize]
        } else {
            0
        };
        let input_hc = &inputs[pos as usize];

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
        dgpu.id, igpu.id, v4flash_kernels::forward::N_LAYER
    );

    // M15 debug: print any post-warmup token >2x the median so we can
    // see whether outliers cluster (e.g. KV-cache reallocation) or are
    // random (e.g. dGPU power state / page faults).
    let outlier_thresh = median_us.saturating_mul(2);
    eprintln!("BENCH outliers (>{:.2}ms):", outlier_thresh as f64 / 1000.0);
    let mut n_outliers = 0;
    for (i, &t) in token_us.iter().enumerate() {
        if (i as i32) < warmup {
            continue;
        }
        if t > outlier_thresh {
            eprintln!("  T{} (idx-after-warmup {}): {:.2}ms", i, i - warmup as usize, t as f64 / 1000.0);
            n_outliers += 1;
            if n_outliers >= 20 {
                eprintln!("  ...truncated at 20 outliers");
                break;
            }
        }
    }

    Ok(())
}
