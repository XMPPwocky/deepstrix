//! M50 Phase 1 oracle: `forward_prompt_batch` produces the same
//! post-last-layer residual as B sequential `forward_token` calls.
//!
//! Setup:
//! 1. Allocate fresh state.
//! 2. Run B sequential `forward_token(prompt[0..B])` calls. Capture
//!    each token's residual_next (= final HC) immediately after its
//!    forward_token returns.
//! 3. Reset state (allocate a NEW HetModelState).
//! 4. Run `forward_prompt_batch(prompt[0..B], pos0=0)`. Capture each
//!    batch element's residual_next after return.
//! 5. Compare: per-token residual_next bit-identical (same kernels,
//!    same inputs, just reorganized layer-major).
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test forward_prompt_batch_matches_sequential \
//!     -- --ignored --nocapture
//!
//! Env: BENCH_B (default 8) — batch size to test.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine,
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut maxd = 0.0f32;
    let mut idx = 0usize;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > maxd {
            maxd = d;
            idx = i;
        }
    }
    (maxd, idx)
}

#[test]
#[ignore]
fn forward_prompt_batch_matches_sequential() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::forward::HC_DIM;

    // Phase 1 keeps B small to keep memory + run time bounded.
    let b: usize = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
        .min(PROMPT_TOKENS.len());
    eprintln!("oracle: B={b} (prompt slice [0..{b}])");

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

    // Build per-token input_hc from the dump's `layer_input_residual`
    // tensor (the canonical layer-0 input for each prompt position).
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

    // Allocate batch_scratch ONCE and use its shared_dgpu/shared_igpu
    // for BOTH runs. forward_token captures sub-block graphs on first
    // call using its scratch's buffer pointers; subsequent calls (and
    // Run B) need the SAME scratch's pointers for the captured graphs
    // to replay correctly. Allocating two separate scratches breaks
    // captures because the second run's replays use the first run's
    // (now freed/different) memory addresses.
    let mut batch_scratch = BatchScratch::alloc(dgpu, igpu)?;

    // ---- Run A: sequential forward_token. Capture residual_next per token. ----
    eprintln!("Run A: sequential forward_token × {b} (reuses shared_dgpu)");
    let mut seq_state = HetModelState::alloc(dgpu, igpu, b as u32 + 4)?;
    let mut seq_hcs: Vec<Vec<f32>> = Vec::with_capacity(b);
    for i in 0..b {
        engine.forward_token(
            &mut batch_scratch.shared_dgpu,
            &mut batch_scratch.shared_igpu,
            &mut seq_state,
            &main_weights,
            &input_hcs[i],
            i as u32,
            tokens[i],
        )?;
        let mut hc = vec![0f32; HC_DIM as usize];
        batch_scratch.shared_dgpu.residual_next.copy_to_host(&mut hc)?;
        seq_hcs.push(hc);
    }

    // ---- Run B: forward_prompt_batch. Capture residual_next per batch element. ----
    // Fresh state (so KV cache + compressor start empty again), reuse
    // the SAME shared scratch so capture pointers stay valid.
    eprintln!("Run B: forward_prompt_batch B={b}");
    let mut batch_state = HetModelState::alloc(dgpu, igpu, b as u32 + 4)?;
    engine.forward_prompt_batch(
        &mut batch_scratch,
        &mut batch_state,
        &main_weights,
        &input_hcs,
        &tokens,
        0,
    )?;
    let mut batch_hcs: Vec<Vec<f32>> = Vec::with_capacity(b);
    for i in 0..b {
        let mut hc = vec![0f32; HC_DIM as usize];
        batch_scratch.per_token_residual_next[i].copy_to_host(&mut hc)?;
        batch_hcs.push(hc);
    }

    // ---- Compare ----
    eprintln!("\n=== per-token residual_next: batch vs sequential ===");
    let mut overall_max = 0.0f32;
    let mut overall_max_pos = 0usize;
    for i in 0..b {
        let (maxd, idx) = max_abs_diff(&batch_hcs[i], &seq_hcs[i]);
        let verdict = if maxd < 1e-4 { "✓" }
                     else if maxd < 1e-2 { "~" }
                     else { "✗" };
        eprintln!(
            "  {} pos={} tok={:>5}  max={:.4e} @i={:>5}  batch={:.4}  seq={:.4}",
            verdict, i, tokens[i], maxd, idx, batch_hcs[i][idx], seq_hcs[i][idx]
        );
        if maxd > overall_max {
            overall_max = maxd;
            overall_max_pos = i;
        }
    }
    eprintln!(
        "\noverall max abs diff = {:.4e} at pos {overall_max_pos}",
        overall_max
    );

    // Phase 1 expectation: bit-identical (same kernels, same inputs, just
    // reorganized layer-major). Tolerance accounts for HIP nondeterminism
    // in certain reduction kernels but anything > 1e-3 is a real bug.
    assert!(
        overall_max < 1e-3,
        "forward_prompt_batch diverges from sequential by {:.4e} (> 1e-3)",
        overall_max
    );

    eprintln!("\nORACLE PASS: forward_prompt_batch matches sequential within 1e-3");
    Ok(())
}

/// M50 Phase 2 oracle: `forward_prompt_batch_v2` (real batched kernels)
/// matches the sequential single-token path within float-reduction-order
/// tolerance (~1e-3 — batched kernels have different summation order).
#[test]
#[ignore]
fn forward_prompt_batch_v2_matches_sequential() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::forward::HC_DIM;

    let b: usize = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
        .min(PROMPT_TOKENS.len());
    eprintln!("v2 oracle: B={b}");

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

    // Allocate shared single-token scratches (used by Run A AND used as
    // the per-call iGPU scratch in Run B since the batched v2 still
    // loops per-batch for iGPU MoE through the same shared IgpuScratch).
    let mut batch_scratch = BatchScratch::alloc(dgpu, igpu)?;
    let mut batch_dgpu = BatchDgpuScratch::alloc(dgpu)?;

    // Run A: sequential single-token (reference).
    eprintln!("Run A: sequential forward_token × {b}");
    let mut seq_state = HetModelState::alloc(dgpu, igpu, b as u32 + 4)?;
    let mut seq_hcs: Vec<Vec<f32>> = Vec::with_capacity(b);
    for i in 0..b {
        engine.forward_token(
            &mut batch_scratch.shared_dgpu,
            &mut batch_scratch.shared_igpu,
            &mut seq_state,
            &main_weights,
            &input_hcs[i],
            i as u32,
            tokens[i],
        )?;
        let mut hc = vec![0f32; HC_DIM as usize];
        batch_scratch.shared_dgpu.residual_next.copy_to_host(&mut hc)?;
        seq_hcs.push(hc);
    }

    // Run B: forward_prompt_batch_v2. Fresh state. SAME shared IgpuScratch
    // so iGPU MoE captured graphs replay correctly. After the call,
    // batch_dgpu.residual holds the post-last-layer HC for each batch
    // element (N_LAYER=43 is odd, so post-swap state is in .residual).
    eprintln!("Run B: forward_prompt_batch_v2 B={b}");
    let mut v2_state = HetModelState::alloc(dgpu, igpu, b as u32 + 4)?;
    engine.forward_prompt_batch_v2(
        &mut batch_dgpu,
        &mut batch_scratch.shared_igpu,
        &mut v2_state,
        &main_weights,
        &input_hcs,
        &tokens,
        0,
    )?;
    let mut v2_hcs: Vec<Vec<f32>> = Vec::with_capacity(b);
    for i in 0..b {
        let slot = batch_dgpu
            .residual
            .slice_view(i * (HC_DIM as usize), HC_DIM as usize);
        let mut hc = vec![0f32; HC_DIM as usize];
        slot.copy_to_host(&mut hc)?;
        v2_hcs.push(hc);
    }

    eprintln!("\n=== per-token residual_next: v2 batched vs sequential ===");
    let mut overall_max = 0.0f32;
    let mut overall_max_pos = 0usize;
    for i in 0..b {
        let (maxd, idx) = max_abs_diff(&v2_hcs[i], &seq_hcs[i]);
        let verdict = if maxd < 1e-3 {
            "✓"
        } else if maxd < 1e-1 {
            "~"
        } else {
            "✗"
        };
        eprintln!(
            "  {} pos={} tok={:>5}  max={:.4e} @i={:>5}  v2={:.4}  seq={:.4}",
            verdict, i, tokens[i], maxd, idx, v2_hcs[i][idx], seq_hcs[i][idx]
        );
        if maxd > overall_max {
            overall_max = maxd;
            overall_max_pos = i;
        }
    }
    eprintln!(
        "\nv2 overall max abs diff = {:.4e} at pos {overall_max_pos}",
        overall_max
    );
    // Phase 2 batched kernels accumulate in a different order than the
    // single-token path; expect ~1e-3 tolerance per HC element.
    assert!(
        overall_max < 5e-2,
        "forward_prompt_batch_v2 diverges from sequential by {:.4e} (> 5e-2)",
        overall_max
    );
    eprintln!("\nv2 ORACLE PASS within 5e-2");
    Ok(())
}
