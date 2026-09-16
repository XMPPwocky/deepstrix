//! A/B oracle for the 2026-09-12 compressor gather fast path.
//!
//! `DEEPSTRIX_COMP_GATHER=1` (default) replaces the serial
//! `state_write → snapshot → shuffle` loop in prefill with one batched
//! gather. The claim is **bit-for-bit identity** with the serial loop, not
//! "close enough": the gather writes the same f32 values (same APE term,
//! same source rows), just without the ring buffer.
//!
//! This test exists because the standing prefill oracles run T=7, which
//! cannot reach the fast path (it needs `b % ratio == 0` with ratio ∈
//! {4,128}, so b ≥ 128). It runs the SAME prefill twice in ONE model load,
//! toggling the env between runs, and requires the logits to match exactly.
//!
//! Run:
//!   BENCH_T=512 HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test compressor_gather_ab -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::HC_DIM;
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, DgpuScratch, ExecMode,
    HetModelState, HetModelWeights, HeterogeneousEngine, B_MAX,
};
use v4flash_kernels::{oracle::ActivationDump, RopeParams};

const MAIN_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;

fn dump_dir() -> PathBuf {
    std::env::var("DEEPSTRIX_DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("reference/v4flash-cpu-activations")
        })
}

fn pick(want_igpu: bool) -> eyre::Result<Device> {
    for d in Device::all()? {
        let arch = d.properties()?.gcn_arch_name;
        let is_igpu = arch.starts_with("gfx1151");
        if is_igpu == want_igpu {
            return Ok(d);
        }
    }
    Err(eyre!("no {} found", if want_igpu { "iGPU" } else { "dGPU" }))
}

#[test]
#[ignore]
fn compressor_gather_matches_serial() -> eyre::Result<()> {
    install_panic_handler()?;
    let t: usize = std::env::var("BENCH_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    eprintln!("compressor gather A/B: T={t} (fast path needs b % ratio == 0, ratio ∈ {{4,128}})");

    let dump = ActivationDump::open(dump_dir())?;
    let main_gguf = MappedGguf::open(
        std::env::var("DEEPSTRIX_GGUF").unwrap_or_else(|_| MAIN_MODEL_PATH.to_string()),
    )?;
    let dgpu = pick(false)?;
    let igpu = pick(true)?;
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

    let lane_rows = B_MAX.div_ceil(2);
    let mut bd = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut sd = BatchDgpuShared::alloc_rows(dgpu, lane_rows)?;
    let mut si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    let mut bd_b = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_b = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
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

    let mut run = |engine: &mut HeterogeneousEngine,
                   bd: &mut BatchDgpuScratch,
                   bi: &mut BatchIgpuScratch,
                   bdb: &mut BatchDgpuScratch,
                   bib: &mut BatchIgpuScratch,
                   sd: &mut BatchDgpuShared,
                   si: &mut BatchIgpuShared|
     -> eyre::Result<Vec<f32>> {
        let mut state = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
        engine.forward_prefill_pipelined(
            bd,
            bi,
            bdb,
            bib,
            sd,
            si,
            &mut head_scratch,
            &mut state,
            &main_weights,
            &input_hcs,
            &tokens,
            0,
            true,
            None,
            None,
            None,
            None,
            None, // pager
            None, // engram_rows
        )
    };

    // Which optimisation to A/B. Both claim bit-identity.
    let var = std::env::var("COMP_AB_VAR")
        .unwrap_or_else(|_| "DEEPSTRIX_COMP_GATHER".to_string());
    eprintln!("A/B toggling {var}");
    // SAFETY: single-threaded test; the engine reads the var per call.
    std::env::set_var(&var, "1");
    let on = run(&mut engine, &mut bd, &mut bi, &mut bd_b, &mut bi_b, &mut sd, &mut si)?;
    std::env::set_var(&var, "0");
    let off = run(&mut engine, &mut bd, &mut bi, &mut bd_b, &mut bi_b, &mut sd, &mut si)?;
    std::env::set_var(&var, "1");

    assert_eq!(on.len(), off.len(), "logit vector length changed");
    let mut n_diff = 0usize;
    let mut max_abs = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in on.iter().zip(off.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
                at = i;
            }
        }
    }
    eprintln!(
        "logits: {} values, {} differing bitwise, max |Δ| = {:.3e} at i={}",
        on.len(),
        n_diff,
        max_abs,
        at
    );
    engine.shutdown()?;
    if n_diff != 0 {
        return Err(eyre!(
            "{var}=1 is NOT bit-identical to {var}=0: \
             {n_diff}/{} logits differ, max |Δ| = {max_abs:.3e} at i={at}",
            on.len()
        ));
    }
    eprintln!("PASS — {var}=1 is bit-identical to {var}=0 at T={t}");
    Ok(())
}

/// Accuracy oracle for the WMMA-GEMM compressor projection.
///
/// `DEEPSTRIX_COMP_GEMM=1` (default) replaces `matvec_pair_batched` (which
/// re-reads the whole weight matrix per batch row) with a batch-tiled WMMA
/// GEMM. This one is NOT bit-exact — WMMA casts the f32 activation to f16
/// and accumulates in a different order — so the bar is relative error on
/// the logits, not identity. Reference is the matvec path.
#[test]
#[ignore]
fn compressor_gemm_accuracy_vs_matvec() -> eyre::Result<()> {
    install_panic_handler()?;
    let t: usize = std::env::var("BENCH_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    // Max allowed max|Δ| as a fraction of the logit vector's scale (max|x|),
    // matching how the standing prefill oracles score divergence.
    let tol: f32 = std::env::var("COMP_GEMM_TOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5e-2);
    eprintln!("compressor GEMM accuracy: T={t}, tol={tol:.1e} of vector scale");

    let dump = ActivationDump::open(dump_dir())?;
    let main_gguf = MappedGguf::open(
        std::env::var("DEEPSTRIX_GGUF").unwrap_or_else(|_| MAIN_MODEL_PATH.to_string()),
    )?;
    let dgpu = pick(false)?;
    let igpu = pick(true)?;
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

    let lane_rows = B_MAX.div_ceil(2);
    let mut bd = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut sd = BatchDgpuShared::alloc_rows(dgpu, lane_rows)?;
    let mut si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    let mut bd_b = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_b = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
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

    let mut run = |engine: &mut HeterogeneousEngine,
                   bd: &mut BatchDgpuScratch,
                   bi: &mut BatchIgpuScratch,
                   bdb: &mut BatchDgpuScratch,
                   bib: &mut BatchIgpuScratch,
                   sd: &mut BatchDgpuShared,
                   si: &mut BatchIgpuShared|
     -> eyre::Result<Vec<f32>> {
        let mut state = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
        engine.forward_prefill_pipelined(
            bd, bi, bdb, bib, sd, si, &mut head_scratch, &mut state, &main_weights,
            &input_hcs, &tokens, 0, true, None, None, None, None,
            None,
            None,
        )
    };

    std::env::set_var("DEEPSTRIX_COMP_GEMM", "0");
    let reference = run(&mut engine, &mut bd, &mut bi, &mut bd_b, &mut bi_b, &mut sd, &mut si)?;
    std::env::set_var("DEEPSTRIX_COMP_GEMM", "1");
    let gemm = run(&mut engine, &mut bd, &mut bi, &mut bd_b, &mut bi_b, &mut sd, &mut si)?;

    assert_eq!(reference.len(), gemm.len());
    let scale = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut max_abs = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in reference.iter().zip(gemm.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            at = i;
        }
    }
    let scaled = if scale > 0.0 { max_abs / scale } else { 0.0 };
    // Argmax agreement is what actually matters for generated text.
    let am_ref = reference
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc })
        .0;
    let am_gemm = gemm
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc })
        .0;
    eprintln!(
        "logits: scale {scale:.3}, max |Δ| {max_abs:.4e} ({scaled:.3e} of scale) at i={at}; \
         argmax ref={am_ref} gemm={am_gemm} {}",
        if am_ref == am_gemm { "MATCH" } else { "DIFFER" }
    );
    engine.shutdown()?;
    if am_ref != am_gemm {
        return Err(eyre!("argmax changed: {am_ref} -> {am_gemm}"));
    }
    if scaled > tol {
        return Err(eyre!(
            "compressor GEMM diverges by {scaled:.3e} of vector scale (> {tol:.1e})"
        ));
    }
    eprintln!("PASS — GEMM within {tol:.1e} of the matvec reference, argmax identical");
    Ok(())
}
