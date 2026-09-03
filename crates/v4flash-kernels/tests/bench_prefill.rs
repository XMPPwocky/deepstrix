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
use v4flash_kernels::config::{COMPRESS_RATIOS, N_EXPERT, N_EXPERT_USED, N_LAYER, SWA_WINDOW};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, BatchScratch,
    DgpuScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine, PrefillStats,
    B_MAX,
};
use v4flash_kernels::{oracle::ActivationDump, RopeParams};

const MAIN_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;

fn dump_dir() -> PathBuf {
    std::env::var("DEEPSTRIX_DUMP_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("reference/v4flash-cpu-activations")
    })
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

/// Bench `forward_prompt_batch_v2` (real batched kernels) at various B.
#[test]
#[ignore]
fn bench_prefill_v2() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::config::HC_DIM;

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
    let main_gguf = MappedGguf::open(std::env::var("DEEPSTRIX_GGUF").unwrap_or_else(|_| MAIN_MODEL_PATH.to_string()))?;
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
    let mut sd = BatchDgpuShared::alloc(dgpu)?;
    let mut si = BatchIgpuShared::alloc(igpu)?;

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
            &mut sd,
            &mut si,
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
            &mut sd,
            &mut si,
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
    use v4flash_kernels::config::HC_DIM;

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
    let main_gguf = MappedGguf::open(std::env::var("DEEPSTRIX_GGUF").unwrap_or_else(|_| MAIN_MODEL_PATH.to_string()))?;
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
    let mut engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    // Defer perfetto attach until AFTER warmup so the trace contains only
    // the timed chunk(s), not the shallow graph-capture warmups (whose tiny
    // attention spans otherwise pollute the per-span aggregates).
    let perfetto_out = std::env::var("PERFETTO_OUT").ok();
    // PIPELINE_LANES=2 turns on the two-lane pipelined prefill driver
    // (two BatchScratch sets, lane A + lane B interleaved per layer).
    //
    // DEFAULT 2 — it must match production. The old default of 1 measured a
    // configuration nobody runs, and because single-lane leaves the dGPU
    // exposed instead of hidden under the iGPU MoE, it made dGPU-side
    // optimisations look 2-3x more impactful than they are. Set
    // PIPELINE_LANES=1 explicitly to A/B the serial path.
    let pipeline_lanes: u32 = std::env::var("PIPELINE_LANES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    // Lane sizing mirrors production: with two lanes each scratch set
    // holds at most ceil(B_MAX/2) rows of a chunk; single-lane
    // forward_prefill needs the full B_MAX. The shared set (one instance
    // for however many lanes) is sized at the same rows.
    let lane_rows = if pipeline_lanes >= 2 { B_MAX.div_ceil(2) } else { B_MAX };
    let mut bd = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut sd = BatchDgpuShared::alloc_rows(dgpu, lane_rows)?;
    let mut si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    let (mut bd_b, mut bi_b) = if pipeline_lanes >= 2 {
        eprintln!("PIPELINE_LANES=2: allocating second BatchScratch set ({lane_rows} rows/lane)");
        (
            Some(BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?),
            Some(BatchIgpuScratch::alloc_rows(igpu, lane_rows)?),
        )
    } else {
        (None, None)
    };
    let mut head_scratch = DgpuScratch::alloc(dgpu)?;

    let n_real = PROMPT_TOKENS.len();

    // FAKE_PREFILL_POS=N: measure the MARGINAL cost of prefilling one
    // B_MAX-token chunk at simulated context depth N, WITHOUT paying the
    // O(N²) real fill to reach depth N. Mirrors the decode FAKE_WARMUP
    // trick: stamp per-layer n_raw / n_comp counters so attention runs
    // over correctly-SIZED (but zero-valued) KV state. Values are garbage,
    // timing is representative — kernel cost depends on shapes/counts, not
    // values. Unlike the default path's `T / total_wall` (a blended
    // average over depths 0..T dominated by the quadratic tail), this
    // reports `B_MAX / chunk_wall` = the actual prefill throughput AT
    // depth N. Sweep N for a fast, clean scaling curve.
    let fake_depths: Vec<u32> = std::env::var("FAKE_PREFILL_POS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse::<u32>().ok())
                .filter(|&v| v > 0)
                .collect()
        })
        .unwrap_or_default();
    if !fake_depths.is_empty() {
        // FAKE_PREFILL_TOKENS: tokens pushed per timed iter. forward_prefill
        // chunks these into B_MAX-sized chunks internally and runs them
        // back-to-back, so >B_MAX gives a steady-state throughput number
        // (amortizing the per-chunk pipeline warmup/cooldown bubble) rather
        // than single-chunk latency. Default = B_MAX (one chunk).
        let chunk_b = std::env::var("FAKE_PREFILL_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(B_MAX);
        let max_depth = *fake_depths.iter().max().unwrap();
        eprintln!(
            "FAKE_PREFILL_POS sweep {fake_depths:?}: timing {chunk_b} tokens/iter per depth"
        );

        // One chunk of inputs (cycle the real dump residuals).
        let mut chunk_hcs: Vec<Vec<f32>> = Vec::with_capacity(chunk_b);
        let mut chunk_tokens: Vec<i32> = Vec::with_capacity(chunk_b);
        for i in 0..chunk_b {
            let src_i = i % n_real;
            let entry = dump
                .tensor("layer_input_residual", 0, src_i as i32)
                .ok_or_else(|| eyre!("missing layer_input_residual L0 T{src_i}"))?;
            let hc = dump.read_f32(entry)?;
            assert_eq!(hc.len(), HC_DIM as usize);
            chunk_hcs.push(hc);
            chunk_tokens.push(PROMPT_TOKENS[src_i]);
        }

        // State must hold the deepest position we touch: the real graph-
        // capture warmup spans `warm_span`, the timed chunk reaches
        // `fake_pos + chunk_b`. Size comp_kv/KV for the max of both.
        let graph_warm_chunks = 2u32;
        let warm_span = graph_warm_chunks * chunk_b as u32;
        let n_kv_max = max_depth.max(warm_span) + chunk_b as u32 + 8;
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;

        // Real warmup chunks (pos 0, B_MAX, …) so the per-layer HIP graphs
        // get captured before timing — else the first timed chunk spikes.
        eprintln!("  real-warming {graph_warm_chunks} chunks for graph capture...");
        for w in 0..graph_warm_chunks {
            if let (Some(bdb), Some(bib)) = (bd_b.as_mut(), bi_b.as_mut()) {
                let _ = engine.forward_prefill_pipelined(
                    &mut bd,
                    &mut bi,
                    bdb,
                    bib,
                    &mut sd,
                    &mut si,
                    &mut head_scratch,
                    &mut state,
                    &main_weights,
                    &chunk_hcs,
                    &chunk_tokens,
                    w * chunk_b as u32,
                    true,
                    None,
                    None,
                    None,
                )?;
            } else {
                let _ = engine.forward_prefill(
                    &mut bd,
                    &mut bi,
                    &mut sd,
                    &mut si,
                    &mut head_scratch,
                    &mut state,
                    &main_weights,
                    &chunk_hcs,
                    &chunk_tokens,
                    w * chunk_b as u32,
                    true,
                    None,
                )?;
            }
        }

        // Stamp per-layer counters to simulate `pos` prior tokens. Same
        // logic as the decode FAKE_WARMUP path in perfetto_trace_long.
        // The indexer compressor counter must be stamped too so the
        // mask-aware prefill path (CSA, ratio==4 layers, ratio fires at
        // the same boundaries as the main compressor) sees a realistic
        // n_index_comp. The underlying comp_kv buffers stay zeroed —
        // perf-shape is unaffected since the indexer's compute cost
        // depends on `n_index_comp` not the buffer contents.
        let set_fake = |state: &mut HetModelState, pos: u32| {
            for layer in 0..N_LAYER as usize {
                let ls = &mut state.layers[layer];
                ls.n_raw = SWA_WINDOW.min(pos);
                let ratio = COMPRESS_RATIOS[layer];
                if ratio > 0 {
                    if let Some(cs) = ls.compressor.as_mut() {
                        cs.n_comp = pos / ratio;
                    }
                    if let Some(ics) = ls.indexer_compressor.as_mut() {
                        ics.n_comp = pos / ratio;
                    }
                }
            }
        };

        // Attach perfetto now — after graph-capture warmup, before timing —
        // so the trace holds only the timed chunk(s) at the requested depth.
        if let Some(p) = &perfetto_out {
            eprintln!("perfetto: attaching after warmup, output → {p}");
            engine.attach_perfetto(p)?;
        }

        // QB_WMMA_AB=1: run the WHOLE sweep twice in-process (one model load)
        // — once with QB_WMMA off (dp4a qb), once on (int8-WMMA qb) — so the
        // A/B shares a thermal envelope (back-to-back methodology). The qb
        // kernel reads QB_WMMA fresh on every call, so flipping the env var
        // here switches the path with no rebuild.
        let ab = std::env::var_os("QB_WMMA_AB").is_some();
        let variants: Vec<(&str, Option<bool>)> = if ab {
            vec![("qb=dp4a", Some(false)), ("qb=wmma", Some(true))]
        } else {
            vec![("qb=current", None)]
        };

        // summary[variant] = Vec<(depth, min_ms, median_ms)>
        let mut summaries: Vec<(&str, Vec<(u32, f64, f64)>)> = Vec::new();
        for (vname, vflag) in &variants {
            match vflag {
                Some(true) => std::env::set_var("QB_WMMA", "1"),
                Some(false) => std::env::remove_var("QB_WMMA"),
                None => {}
            }
            eprintln!("\n--- variant {vname} ---");
            let mut summary: Vec<(u32, f64, f64)> = Vec::with_capacity(fake_depths.len());
            for &fake_pos in &fake_depths {
                let mut walls_ms: Vec<f64> = Vec::with_capacity(n_iters);
                for it in 0..n_iters {
                    // Reset depth before each timed chunk (forward_prefill advances
                    // the counters), so every iter measures the same depth.
                    set_fake(&mut state, fake_pos);
                    let t0 = Instant::now();
                    if let (Some(bdb), Some(bib)) = (bd_b.as_mut(), bi_b.as_mut()) {
                        let _ = engine.forward_prefill_pipelined(
                            &mut bd,
                            &mut bi,
                            bdb,
                            bib,
                            &mut sd,
                            &mut si,
                            &mut head_scratch,
                            &mut state,
                            &main_weights,
                            &chunk_hcs,
                            &chunk_tokens,
                            fake_pos,
                            true,
                            None,
                            None,
                            None,
                        )?;
                    } else {
                        let _ = engine.forward_prefill(
                            &mut bd,
                            &mut bi,
                            &mut sd,
                            &mut si,
                            &mut head_scratch,
                            &mut state,
                            &main_weights,
                            &chunk_hcs,
                            &chunk_tokens,
                            fake_pos,
                            true,
                            None,
                        )?;
                    }
                    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    walls_ms.push(wall_ms);
                    eprintln!(
                        "  [{vname}] depth {fake_pos} iter {it}: chunk wall={:.2} ms  ({:.3} ms/tok = {:.1} tok/s)",
                        wall_ms,
                        wall_ms / chunk_b as f64,
                        (chunk_b as f64 * 1000.0) / wall_ms
                    );
                }
                walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let min_ms = walls_ms[0];
                let median_ms = walls_ms[walls_ms.len() / 2];
                summary.push((fake_pos, min_ms, median_ms));
            }
            summaries.push((vname, summary));
        }

        eprintln!("\n=== BENCH FAKE PREFILL ({chunk_b} tokens/iter, B_MAX={B_MAX} chunks) ===");
        for (vname, summary) in &summaries {
            eprintln!("[{vname}]");
            eprintln!("depth   best ms/iter    ms/tok   tok/s   |  median ms/iter    ms/tok   tok/s");
            for (fake_pos, min_ms, median_ms) in summary {
                eprintln!(
                    "{:>6}  {:>11.2}  {:>7.3}  {:>6.1}  |  {:>13.2}  {:>7.3}  {:>6.1}",
                    fake_pos,
                    min_ms,
                    min_ms / chunk_b as f64,
                    (chunk_b as f64 * 1000.0) / min_ms,
                    median_ms,
                    median_ms / chunk_b as f64,
                    (chunk_b as f64 * 1000.0) / median_ms,
                );
            }
        }
        if summaries.len() == 2 {
            eprintln!("\n=== qb A/B (best ms/iter, dp4a -> wmma) ===");
            let (_, a) = &summaries[0];
            let (_, b) = &summaries[1];
            for ((d, a_min, _), (_, b_min, _)) in a.iter().zip(b.iter()) {
                eprintln!(
                    "  depth {d}: {:.2} ms -> {:.2} ms  ({:+.1}%, {:.1} -> {:.1} tok/s)",
                    a_min,
                    b_min,
                    (a_min - b_min) / a_min * 100.0,
                    (chunk_b as f64 * 1000.0) / a_min,
                    (chunk_b as f64 * 1000.0) / b_min,
                );
            }
        }
        engine.shutdown()?;
        return Ok(());
    }

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

    let mut run_once = |engine: &mut HeterogeneousEngine,
                    bd: &mut BatchDgpuScratch,
                    bi: &mut BatchIgpuScratch,
                    bd_b: Option<&mut BatchDgpuScratch>,
                    bi_b: Option<&mut BatchIgpuScratch>,
                    sd: &mut BatchDgpuShared,
                    si: &mut BatchIgpuShared,
                    state: &mut HetModelState|
     -> eyre::Result<()> {
        if let (Some(bdb), Some(bib)) = (bd_b, bi_b) {
            let _ = engine.forward_prefill_pipelined(
                bd,
                bi,
                bdb,
                bib,
                sd,
                si,
                &mut head_scratch,
                state,
                &main_weights,
                &input_hcs,
                &tokens,
                0,
                last_only,
                None,
                None,
                None,
            )?;
        } else {
            let _ = engine.forward_prefill(
                bd,
                bi,
                sd,
                si,
                &mut head_scratch,
                state,
                &main_weights,
                &input_hcs,
                &tokens,
                0,
                last_only,
                None,
            )?;
        }
        Ok(())
    };

    eprintln!("warmup × {n_warmup}");
    for _ in 0..n_warmup {
        let mut state = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
        run_once(
            &mut engine,
            &mut bd,
            &mut bi,
            bd_b.as_mut(),
            bi_b.as_mut(),
            &mut sd,
            &mut si,
            &mut state,
        )?;
    }
    // Attach perfetto after warmup so only the timed iters are traced.
    if let Some(p) = &perfetto_out {
        eprintln!("perfetto: attaching after warmup, output → {p}");
        engine.attach_perfetto(p)?;
    }
    let mut walls_ms: Vec<f64> = Vec::with_capacity(n_iters);
    for it in 0..n_iters {
        let mut state = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
        let t0 = Instant::now();
        run_once(
            &mut engine,
            &mut bd,
            &mut bi,
            bd_b.as_mut(),
            bi_b.as_mut(),
            &mut sd,
            &mut si,
            &mut state,
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
    engine.shutdown()?;
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
    use v4flash_kernels::config::HC_DIM;

    let t: usize = std::env::var("BENCH_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    eprintln!("expert-stats run: T={t}");

    let dump = ActivationDump::open(dump_dir())?;
    let main_gguf = MappedGguf::open(std::env::var("DEEPSTRIX_GGUF").unwrap_or_else(|_| MAIN_MODEL_PATH.to_string()))?;
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
    let mut sd = BatchDgpuShared::alloc(dgpu)?;
    let mut si = BatchIgpuShared::alloc(igpu)?;
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
        &mut sd,
        &mut si,
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
    // Dump representative layer's pick counts for offline plotting.
    let dump_layer: usize = std::env::var("DUMP_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let dump_path = std::env::var("DUMP_PATH")
        .unwrap_or_else(|_| format!("/tmp/expert_picks_T{t}_L{dump_layer}.json"));
    stats.dump_layer_picks(dump_layer, &dump_path)?;
    eprintln!("\nDumped layer {dump_layer} pick_counts to {dump_path}");
    Ok(())
}
