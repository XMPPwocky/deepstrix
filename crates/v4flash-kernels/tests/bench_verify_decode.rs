//! MTP verify-cost driver: measure C(B) = t(B=k)/t(B=1) for a batched
//! verify-k step on the DECODE path.
//!
//! ## What this settles
//! We measured decode is dGPU-bound (dGPU ~79% busy, iGPU ~71% idle). A
//! batched verify-k should keep the dGPU attention-weight wall ~flat
//! (weights read once/batch, applied to k rows) and let the iGPU absorb
//! k× cold MoE reads under its idle slack. So we expect C(k) ≪ k.
//!
//! ## Approach (and its honest biases)
//! The real B=1 decode path (`forward_token`) is graph-captured and hits
//! ~34 ms @4K. It is written for a SINGLE query row; batching it to B=k
//! would require a B=k rewrite of the whole decode forward (scalar-arg
//! attention, matvecs, …) — not tractable additively.
//!
//! Instead the verify-k step is run through `forward_prefill` with a
//! B=k chunk at FAKE context depth and `last_only=false` (per-position
//! logits). This IS kernel-shape-faithful to a verify-k step:
//!   * k query rows through attention against the shared KV window
//!     (prefill attention has each of the k consecutive positions attend
//!     causally to past KV + the earlier batch rows — exactly verify-k),
//!   * k KV appends, k routed tokens through MoE (grid.z=B kernels),
//!     RoPE for k consecutive positions, logits for EACH of the k rows.
//! The ONE infidelity: `forward_prefill` is NOT graph-captured, so its
//! wall carries a B-INDEPENDENT host launch-overhead L (same kernel COUNT
//! at every B, only grids grow). We remove it two ways and report both:
//!   (a) launch-subtracted:  t_dev(B) = wall_prefill(B) − L,
//!       L = wall_prefill(B=1) − anchor_decode(B=1).  C(B)=t_dev(B)/t_dev(1).
//!   (b) device-busy (perfetto): dGPU compute-busy ms per B via
//!       analyze_pftrace_gaps.py — busy excludes host gaps, i.e. it is the
//!       graph-replay-equivalent device time. C(B)=busy(B)/busy(1).
//! Both bracket the true graphed C(B). The graphed decode anchor is
//! reported alongside so the reader can see L directly.
//!
//! Attention kernel bias: prefill uses the batched WMMA attention; decode
//! B=1 uses a specialised scalar-arg WMMA score + K-split smwsum. Their
//! B=1 costs differ, so busy(B=1) is a PROXY for the graphed decode
//! attention. This is flagged in the report.
//!
//! ## Run
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 \
//! DGPU_HOT_EXPERTS=8 \
//! DGPU_HOT_EXPERTS_FILE=/home/claude-code/deepstrix/reference/decode_hot_experts.txt \
//! VERIFY_B=1,2,3,4 FAKE_POS=4096,98304 \
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_verify_decode -- --ignored --nocapture
//! ```
//! Optional: `PERFETTO_DEVICE_OUT=/tmp/verify` attaches the device-time
//! exporter for the B=1 and B=3 runs (one trace per B, suffixed `_b{B}.pftrace`)
//! and prints the analyze-script command. Don't trust traced wall/tok-s.

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};

use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{COMPRESS_RATIOS, HC_DIM, N_LAYER, SWA_WINDOW};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchIgpuScratch, DgpuScratch, ExecMode, HetModelState, HetModelWeights,
    HeterogeneousEngine, IgpuScratch,
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

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 (9070 XT) device found"))
}
fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 (Strix iGPU) device found"))
}

/// Stamp per-layer KV counters to simulate `pos` prior decoded tokens.
/// Same trick bench_decode / bench_prefill_chunked use: buffers stay
/// zeroed (garbage values) but attention runs over correctly-SIZED KV, so
/// kernel timing is representative. Applied to BOTH decode and prefill
/// state (identical `HetModelState`).
fn stamp_fake_depth(state: &mut HetModelState, pos: u32) {
    for layer in 0..N_LAYER as usize {
        let ls = &mut state.layers[layer];
        ls.n_raw = SWA_WINDOW.min(pos);
        ls.raw_off = 0;
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
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

#[test]
#[ignore]
fn bench_verify_decode() -> eyre::Result<()> {
    install_panic_handler()?;

    // VERIFY_B: comma list of batch sizes; C(B) baseline is the first entry.
    let bs_list: Vec<usize> = std::env::var("VERIFY_B")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .map(|b| b.clamp(1, 32))
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![1, 2, 3, 4]);
    let b_max = *bs_list.iter().max().unwrap();

    // FAKE_POS: comma list of context depths to sweep.
    let depths: Vec<u32> = std::env::var("FAKE_POS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<u32>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<u32>| !v.is_empty())
        .unwrap_or_else(|| vec![4096, 98304]);

    let n_warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let n_iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let perfetto_base = std::env::var("PERFETTO_DEVICE_OUT").ok();

    eprintln!(
        "bench_verify_decode: B={bs_list:?} depths={depths:?} warmup={n_warmup} iters={n_iters}"
    );
    eprintln!(
        "het-split: DGPU_HOT_EXPERTS={:?} FILE={:?}",
        std::env::var("DGPU_HOT_EXPERTS").ok(),
        std::env::var("DGPU_HOT_EXPERTS_FILE").ok(),
    );

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let dgpu = pick_dgpu()?;
    let igpu = pick_igpu()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    eprintln!("dGPU={dgpu_arch} iGPU={igpu_arch}");

    let rope_for_layer = |layer: i32| -> eyre::Result<RopeParams> {
        let entry = dump
            .weight("rope_params", layer)
            .ok_or_else(|| eyre!("missing rope_params L{layer}"))?;
        let floats = dump.read_f32(entry)?;
        let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
        RopeParams::from_dump_blob(&floats, n_ctx_orig)
    };

    eprintln!("loading het weights (slow, ~1-2 min)...");
    let weights = HetModelWeights::load_all(&gguf, dgpu, igpu, &rope_for_layer)?;
    eprintln!("loaded.");

    let mut engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;

    // Scratch: decode (B=1 anchor) + batch (verify-k). Alloc once, reuse.
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let mut bd = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi = BatchIgpuScratch::alloc(igpu)?;
    let mut head_scratch = DgpuScratch::alloc(dgpu)?;

    // Inputs: real dump residuals for the first 7 positions, cycled for B>7.
    // Size to at least n_real so the anchor warmup can cycle real inputs.
    let n_real = PROMPT_TOKENS.len();
    let n_inp = b_max.max(n_real);
    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(n_inp);
    let mut tokens: Vec<i32> = Vec::with_capacity(n_inp);
    for i in 0..n_inp {
        let src = i % n_real;
        let entry = dump
            .tensor("layer_input_residual", 0, src as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{src}"))?;
        let hc = dump.read_f32(entry)?;
        assert_eq!(hc.len(), HC_DIM as usize);
        input_hcs.push(hc);
        tokens.push(PROMPT_TOKENS[src]);
    }

    // ---- helpers ---------------------------------------------------------

    // Graph-captured B=1 decode anchor at depth `fake_pos`. Real-forwards a
    // few tokens to capture the per-layer HIP graphs, stamps depth, then
    // times `forward_token`. Returns median ms.
    let anchor_decode = |engine: &mut HeterogeneousEngine,
                             dgpu_scratch: &mut DgpuScratch,
                             igpu_scratch: &mut IgpuScratch,
                             fake_pos: u32|
     -> eyre::Result<f64> {
        let graph_warm = 12u32;
        let n_kv_max = fake_pos + graph_warm + n_iters as u32 + 64;
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        // Real warmup to capture graphs (pos 0..graph_warm).
        for pos in 0..graph_warm {
            let tid = if (pos as usize) < n_real {
                PROMPT_TOKENS[pos as usize]
            } else {
                0
            };
            engine.forward_token(
                dgpu_scratch,
                igpu_scratch,
                &mut state,
                &weights,
                &input_hcs[pos as usize % n_real],
                pos,
                tid,
            )?;
        }
        stamp_fake_depth(&mut state, fake_pos);
        // Timed tokens at faked depth.
        let mut ms: Vec<f64> = Vec::with_capacity(n_iters);
        for it in 0..(n_warmup + n_iters) {
            let pos = fake_pos + it as u32;
            let t = Instant::now();
            engine.forward_token(
                dgpu_scratch,
                igpu_scratch,
                &mut state,
                &weights,
                &input_hcs[0],
                pos,
                0,
            )?;
            let dt = t.elapsed().as_secs_f64() * 1000.0;
            if it >= n_warmup {
                ms.push(dt);
            }
        }
        Ok(median(&mut ms))
    };

    // One verify-k step via the batched prefill forward at faked depth.
    // last_only=false → per-position logits (a real verify needs them).
    // Returns median wall ms.
    let verify_step = |engine: &mut HeterogeneousEngine,
                           bd: &mut BatchDgpuScratch,
                           bi: &mut BatchIgpuScratch,
                           head_scratch: &mut DgpuScratch,
                           b: usize,
                           fake_pos: u32,
                           trace: Option<&str>|
     -> eyre::Result<f64> {
        let inp = &input_hcs[0..b];
        let toks = &tokens[0..b];
        // State must hold the faked depth + the k appended rows.
        let n_kv_max = fake_pos + b as u32 + 8;
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        // Warmup (also captures any per-layer prefill graphs / first-touch).
        for _ in 0..n_warmup {
            stamp_fake_depth(&mut state, fake_pos);
            let _ = engine.forward_prefill(
                bd,
                bi,
                head_scratch,
                &mut state,
                &weights,
                inp,
                toks,
                fake_pos,
                /*last_only=*/ false,
                None,
            )?;
        }
        if let Some(p) = trace {
            engine.attach_perfetto(p)?;
        }
        let mut ms: Vec<f64> = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            // forward_prefill advances counters; reset depth each iter.
            stamp_fake_depth(&mut state, fake_pos);
            let t = Instant::now();
            let _ = engine.forward_prefill(
                bd,
                bi,
                head_scratch,
                &mut state,
                &weights,
                inp,
                toks,
                fake_pos,
                /*last_only=*/ false,
                None,
            )?;
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(median(&mut ms))
    };

    // ---- measurement -----------------------------------------------------

    for &fake_pos in &depths {
        eprintln!("\n########## FAKE_POS = {fake_pos} ##########");

        // (1) graph-captured decode B=1 anchor.
        let anchor = anchor_decode(&mut engine, &mut dgpu_scratch, &mut igpu_scratch, fake_pos)?;
        eprintln!(
            ">>> ANCHOR graph-captured decode B=1: {anchor:.2} ms  ({:.2} tok/s)",
            1000.0 / anchor
        );
        if fake_pos <= 8192 && (anchor < 25.0 || anchor > 55.0) {
            eprintln!(
                "!!! WARNING: anchor {anchor:.1} ms is outside the ~34 ms decode band. \
                 If ~78 ms the driver is on the prefill path — C(B) below is suspect."
            );
        }

        // (2) verify-k via prefill batch at each B.
        let mut walls: Vec<(usize, f64)> = Vec::with_capacity(bs_list.len());
        for &b in &bs_list {
            let w = verify_step(
                &mut engine,
                &mut bd,
                &mut bi,
                &mut head_scratch,
                b,
                fake_pos,
                None,
            )?;
            eprintln!(
                "  verify B={b}: wall={w:.2} ms  ({:.2} ms/tok)",
                w / b as f64
            );
            walls.push((b, w));
        }

        // Launch overhead L = wall_prefill(B=1) − anchor. Requires B=1 in
        // the list; if absent, skip the subtracted column.
        let prefill_b1 = walls.iter().find(|(b, _)| *b == 1).map(|(_, w)| *w);
        let l = prefill_b1.map(|w| (w - anchor).max(0.0));
        if let Some(l) = l {
            eprintln!(
                "  launch overhead L = wall_prefill(B=1) {:.2} − anchor {anchor:.2} = {l:.2} ms",
                prefill_b1.unwrap()
            );
        }

        // C(B) table. Baseline: launch-subtracted device time at B0.
        let b0 = walls[0].0;
        let dev0_sub = l.map(|l| (walls[0].1 - l).max(0.001));
        eprintln!("\n=== C(B) @ depth {fake_pos} (baseline B={b0}) ===");
        eprintln!("   B | wall ms | C_wall | t_dev(sub) ms | C_dev(sub) | F(sub) | ms/tok(dev)");
        for &(b, w) in &walls {
            let c_wall = w / walls[0].1;
            let (tdev, c_dev, f) = match (l, dev0_sub) {
                (Some(l), Some(d0)) => {
                    let td = (w - l).max(0.001);
                    let c = td / d0;
                    let f = if b > b0 {
                        (b as f64 - c) / (b as f64 - 1.0)
                    } else {
                        f64::NAN
                    };
                    (td, c, f)
                }
                _ => (f64::NAN, f64::NAN, f64::NAN),
            };
            eprintln!(
                "  {b:>2} | {w:>7.2} | {c_wall:>6.3} | {tdev:>13.2} | {c_dev:>10.3} | {f:>6.3} | {:>7.2}",
                tdev / b as f64
            );
        }
        eprintln!(
            "  (C_dev anchored so C_dev(B0)=1; t_dev(B0) = anchor {anchor:.2} ms by construction)"
        );

        // (3) optional perfetto device split for B=1 and B=3.
        if let Some(base) = &perfetto_base {
            for &b in &[1usize, 3usize] {
                if !bs_list.contains(&b) {
                    continue;
                }
                let path = format!("{base}_d{fake_pos}_b{b}.pftrace");
                eprintln!("  [perfetto] verify B={b} depth {fake_pos} → {path}");
                // Fresh engine-less trace: attach inside verify_step for the timed iters.
                let _ = verify_step(
                    &mut engine,
                    &mut bd,
                    &mut bi,
                    &mut head_scratch,
                    b,
                    fake_pos,
                    Some(&path),
                )?;
                // Detach by dropping the exporter: re-attach on next call
                // overwrites; but we must flush. attach_perfetto owns a
                // fresh exporter per call, so the file is finalized when the
                // engine's perfetto lock is replaced or on shutdown. Print
                // the analyze command for the operator.
                eprintln!(
                    "    analyze: python3 ~/scripts/analyze_pftrace_gaps.py {path}"
                );
            }
        }
    }

    eprintln!("\n=== GO/NO-GO GUIDE ===");
    eprintln!("  net decode tok/s(MTP) = base_tps × acceptance / C(B=k+1)");
    eprintln!("  base_tps ≈ 1000 / anchor(short).  Use C_dev(sub) (graph-equivalent).");
    eprintln!("  Compare vs baseline non-MTP tok/s = base_tps. MTP wins iff acceptance > C.");

    engine.shutdown()?;
    Ok(())
}
