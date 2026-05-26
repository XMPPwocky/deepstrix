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
use v4flash_kernels::forward::{N_EXPERT, N_EXPERT_USED, N_LAYER};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchIgpuScratch, BatchScratch, DgpuScratch, ExecMode, HetModelState,
    HetModelWeights, HeterogeneousEngine, PrefillStats,
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

/// M50 Phase 2 v2 bench: measure `forward_prompt_batch_v2` (real batched
/// kernels) at various B.
#[test]
#[ignore]
fn bench_prefill_v2() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::forward::HC_DIM;

    let b: usize = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(PROMPT_TOKENS.len())
        .min(64);
    let n_warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let n_iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    eprintln!("bench_prefill_v2: B={b}, warmup={n_warmup}, iters={n_iters}");

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
    let mut _bs = BatchScratch::alloc(dgpu, igpu)?;
    let mut bd = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi = BatchIgpuScratch::alloc(igpu)?;

    // Load real dump inputs for the first 7 positions; repeat thereafter
    // to support synthetic B>7 timing tests. (Repeated inputs make KV
    // cache contents bogus but kernel timings unchanged.)
    let n_real = PROMPT_TOKENS.len();
    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(b);
    let mut tokens: Vec<i32> = Vec::with_capacity(b);
    for i in 0..b {
        let src_i = i % n_real;
        let entry = dump
            .tensor("layer_input_residual", 0, src_i as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{src_i}"))?;
        let hc = dump.read_f32(entry)?;
        assert_eq!(hc.len(), HC_DIM as usize);
        input_hcs.push(hc);
        tokens.push(PROMPT_TOKENS[src_i]);
    }
    if b > n_real {
        eprintln!("(B>{n_real}: repeating real inputs cyclically — timing only, not correctness)");
    }

    eprintln!("warmup × {n_warmup}");
    for _ in 0..n_warmup {
        let mut state = HetModelState::alloc(dgpu, igpu, b as u32 + 4)?;
        engine.forward_prompt_batch_v2(
            &mut bd,
            &mut bi,
            &mut state,
            &main_weights,
            &input_hcs,
            &tokens,
            0,
            None,
        )?;
    }

    let mut walls_ms: Vec<f64> = Vec::with_capacity(n_iters);
    for it in 0..n_iters {
        let mut state = HetModelState::alloc(dgpu, igpu, b as u32 + 4)?;
        let t0 = Instant::now();
        engine.forward_prompt_batch_v2(
            &mut bd,
            &mut bi,
            &mut state,
            &main_weights,
            &input_hcs,
            &tokens,
            0,
            None,
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
    eprintln!("\n=== BENCH PREFILL V2 B={b} ===");
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
    eprintln!("reference: Phase 1 (looped) = 77 ms/tok @ B=7 (forfeits M30 graphs)");
    Ok(())
}

/// M50 Phase 6 bench: end-to-end chunked prefill on arbitrary-length T
/// via repeated dump inputs. Reports total wall + effective tok/s for the
/// full prefill (not per-chunk). T defaults to 200, last_only=true.
#[test]
#[ignore]
fn bench_prefill_chunked() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::forward::HC_DIM;

    let t: usize = std::env::var("BENCH_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let last_only: bool = std::env::var("BENCH_LAST_ONLY")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .map(|v| v != 0)
        .unwrap_or(true);
    let n_warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let n_iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    eprintln!(
        "bench_prefill_chunked: T={t}, last_only={last_only}, warmup={n_warmup}, iters={n_iters}"
    );

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
    let mut bd = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi = BatchIgpuScratch::alloc(igpu)?;
    let mut head_scratch = DgpuScratch::alloc(dgpu)?;

    let n_real = PROMPT_TOKENS.len();
    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(t);
    let mut tokens: Vec<i32> = Vec::with_capacity(t);
    for i in 0..t {
        let src_i = i % n_real;
        let entry = dump
            .tensor("layer_input_residual", 0, src_i as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{src_i}"))?;
        let hc = dump.read_f32(entry)?;
        assert_eq!(hc.len(), HC_DIM as usize);
        input_hcs.push(hc);
        tokens.push(PROMPT_TOKENS[src_i]);
    }
    if t > n_real {
        eprintln!("(T>{n_real}: repeating real inputs cyclically — timing only)");
    }

    eprintln!("warmup × {n_warmup}");
    for _ in 0..n_warmup {
        let mut state = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
        let _ = engine.forward_prefill(
            &mut bd,
            &mut bi,
            &mut head_scratch,
            &mut state,
            &main_weights,
            &input_hcs,
            &tokens,
            0,
            last_only,
            None,
        )?;
    }
    let mut walls_ms: Vec<f64> = Vec::with_capacity(n_iters);
    for it in 0..n_iters {
        let mut state = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
        let t0 = Instant::now();
        let _ = engine.forward_prefill(
            &mut bd,
            &mut bi,
            &mut head_scratch,
            &mut state,
            &main_weights,
            &input_hcs,
            &tokens,
            0,
            last_only,
            None,
        )?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        walls_ms.push(wall_ms);
        eprintln!(
            "  iter {it}: wall={:.2} ms  ({:.2} ms/tok = {:.2} tok/s)",
            wall_ms,
            wall_ms / t as f64,
            (t as f64 * 1000.0) / wall_ms
        );
    }
    walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = walls_ms[walls_ms.len() / 2];
    let min_ms = walls_ms[0];
    eprintln!("\n=== BENCH PREFILL CHUNKED T={t} last_only={last_only} ===");
    eprintln!(
        "best wall:   {:.2} ms ({:.2} ms/tok = {:.2} tok/s)",
        min_ms,
        min_ms / t as f64,
        (t as f64 * 1000.0) / min_ms
    );
    eprintln!(
        "median wall: {:.2} ms ({:.2} ms/tok = {:.2} tok/s)",
        median_ms,
        median_ms / t as f64,
        (t as f64 * 1000.0) / median_ms
    );
    Ok(())
}

/// M50 Phase 3 expert-stats run. Runs forward_prefill at T=BENCH_T (default
/// 128) with stats collection enabled and dumps the per-chunk reuse and
/// per-layer skew tables. Doesn't validate correctness — purely an
/// observation tool to inform whether by-expert MoE grouping would help.
#[test]
#[ignore]
fn bench_prefill_expert_stats() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::forward::HC_DIM;

    let t: usize = std::env::var("BENCH_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    eprintln!("expert-stats run: T={t}");

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
    let main_weights = HetModelWeights::load_all(&main_gguf, dgpu, igpu, &rope_for_layer)?;
    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let mut bd = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi = BatchIgpuScratch::alloc(igpu)?;
    let mut head_scratch = DgpuScratch::alloc(dgpu)?;

    let n_real = PROMPT_TOKENS.len();
    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(t);
    let mut tokens: Vec<i32> = Vec::with_capacity(t);
    for i in 0..t {
        let src_i = i % n_real;
        let entry = dump
            .tensor("layer_input_residual", 0, src_i as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{src_i}"))?;
        let hc = dump.read_f32(entry)?;
        assert_eq!(hc.len(), HC_DIM as usize);
        input_hcs.push(hc);
        tokens.push(PROMPT_TOKENS[src_i]);
    }

    let mut state = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
    let mut stats = PrefillStats::new(N_LAYER as u32, N_EXPERT_USED as u32, N_EXPERT);

    let _ = engine.forward_prefill(
        &mut bd,
        &mut bi,
        &mut head_scratch,
        &mut state,
        &main_weights,
        &input_hcs,
        &tokens,
        0,
        true,
        Some(&mut stats),
    )?;
    stats.print_summary();
    Ok(())
}
