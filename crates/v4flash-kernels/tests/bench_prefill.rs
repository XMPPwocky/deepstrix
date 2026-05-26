//! M50 prefill bench: measure wall time for `forward_prompt_batch` at
//! various B. Reports per-token cost and effective tok/s.
//!
//! Phase 1 expectation: ~36 ms/token at all B (no actual batching —
//! kernels are looped sequentially, but with shared scratch + per-token
//! residual copies). Phase 2+ should see this drop as kernels get
//! batched. Reports also include per-token cost in single-token decode
//! for comparison (~36 ms = 27.95 tok/s).
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 BENCH_B=8 nix develop -c cargo test \
//!     --release -p v4flash-kernels --test bench_prefill bench_prefill \
//!     -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::het::{
    BatchScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine,
};
use v4flash_kernels::{ActivationDump, RopeParams};

const MAIN_MODEL_PATH: &str =
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
fn bench_prefill() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::forward::HC_DIM;

    let b: usize = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(PROMPT_TOKENS.len())
        .min(PROMPT_TOKENS.len());
    let n_warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let n_iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    eprintln!("bench_prefill: B={b}, warmup={n_warmup}, iters={n_iters}");

    let dump = ActivationDump::open(dump_dir())?;
    let main_gguf = MappedGguf::open(MAIN_MODEL_PATH)?;
    let dgpu = pick_dgpu()?;
    let igpu = pick_igpu()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;

    let rope_for_layer = |layer: i32| -> eyre::Result<RopeParams> {
        let entry = dump
            .weight("rope_params", layer)
            .ok_or_else(|| eyre!("missing rope_params L{layer}"))?;
        let floats = dump.read_f32(entry)?;
        let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
        RopeParams::from_dump_blob(&floats, n_ctx_orig)
    };

    eprintln!("loading main weights...");
    let main_weights = HetModelWeights::load_all(&main_gguf, dgpu, igpu, &rope_for_layer)?;
    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let mut batch_scratch = BatchScratch::alloc(dgpu, igpu)?;

    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(b);
    for i in 0..b {
        let entry = dump
            .tensor("layer_input_residual", 0, i as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{i}"))?;
        let hc = dump.read_f32(entry)?;
        assert_eq!(hc.len(), HC_DIM as usize);
        input_hcs.push(hc);
    }
    let tokens: Vec<i32> = PROMPT_TOKENS[..b].to_vec();

    // Warmup so first-call captures don't pollute timing.
    eprintln!("warmup × {n_warmup}");
    for _ in 0..n_warmup {
        let mut state = HetModelState::alloc(dgpu, igpu, b as u32 + 4)?;
        engine.forward_prompt_batch(
            &mut batch_scratch,
            &mut state,
            &main_weights,
            &input_hcs,
            &tokens,
            0,
        )?;
    }

    // Measure.
    let mut walls_ms: Vec<f64> = Vec::with_capacity(n_iters);
    for it in 0..n_iters {
        let mut state = HetModelState::alloc(dgpu, igpu, b as u32 + 4)?;
        let t0 = Instant::now();
        engine.forward_prompt_batch(
            &mut batch_scratch,
            &mut state,
            &main_weights,
            &input_hcs,
            &tokens,
            0,
        )?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        walls_ms.push(wall_ms);
        eprintln!(
            "  iter {it}: wall={:.2} ms  ({:.2} ms/tok = {:.2} tok/s)",
            wall_ms,
            wall_ms / b as f64,
            (b as f64 * 1000.0) / wall_ms
        );
    }

    walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = walls_ms[walls_ms.len() / 2];
    let min_ms = walls_ms[0];
    eprintln!("\n=== BENCH PREFILL B={b} ===");
    eprintln!(
        "best wall:   {:.2} ms ({:.2} ms/tok = {:.2} tok/s)",
        min_ms,
        min_ms / b as f64,
        (b as f64 * 1000.0) / min_ms
    );
    eprintln!(
        "median wall: {:.2} ms ({:.2} ms/tok = {:.2} tok/s)",
        median_ms,
        median_ms / b as f64,
        (b as f64 * 1000.0) / median_ms
    );
    eprintln!("reference: single-token decode = 35.78 ms/tok = 27.95 tok/s (master p50)");
    Ok(())
}
