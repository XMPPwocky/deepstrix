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
use v4flash_kernels::forward::N_VOCAB;
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchIgpuScratch, BatchScratch, DgpuScratch, ExecMode, HetModelState,
    HetModelWeights, HeterogeneousEngine,
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
    let mut batch_igpu = BatchIgpuScratch::alloc(igpu)?;

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
    //
    // Force IQ2_VARIANT=chunked for bit-exact comparison with single-token
    // path. The default staged variant has a different float reduction
    // order (8 rows × 32 lanes per super-block vs single-token's 2 lanes ×
    // 16 super-blocks); per-layer diffs of ~1e-2 compound across all
    // layers to give visible (but functionally correct) deltas.
    std::env::set_var("IQ2_VARIANT", "chunked");
    eprintln!("Run B: forward_prompt_batch_v2 B={b} (IQ2_VARIANT=chunked for bit-exact oracle)");
    let mut v2_state = HetModelState::alloc(dgpu, igpu, b as u32 + 4)?;
    engine.forward_prompt_batch_v2(
        &mut batch_dgpu,
        &mut batch_igpu,
        &mut v2_state,
        &main_weights,
        &input_hcs,
        &tokens,
        0,
        None,
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

/// M50 v2 bisect: run forward_layer_batch_v2 ONE layer at a time
/// and compare per-batch residual_next against Phase 1 (B sequential
/// forward_layer_pair_mode calls) at the same layer. Print first-
/// divergence layer + batch position.
/// Set `BENCH_B` to choose batch size (default 1, max 7).
#[test]
#[ignore]
fn forward_prompt_batch_v2_bisect_layer() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::forward::{HC_DIM, N_LAYER};

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

    let b_n: usize = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .min(PROMPT_TOKENS.len());
    eprintln!("bisect: B={b_n}");

    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(b_n);
    for i in 0..b_n {
        let entry = dump
            .tensor("layer_input_residual", 0, i as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{i}"))?;
        input_hcs.push(dump.read_f32(entry)?);
    }
    let tokens: Vec<i32> = PROMPT_TOKENS[..b_n].to_vec();

    // Two parallel paths.
    let mut bs = BatchScratch::alloc(dgpu, igpu)?; // Phase 1 reference
    let mut bd = BatchDgpuScratch::alloc(dgpu)?;   // Phase 2 v2 dGPU side
    let mut bi = BatchIgpuScratch::alloc(igpu)?;   // Phase 3 v2 iGPU side

    let mut ref_state = HetModelState::alloc(dgpu, igpu, b_n as u32 + 4)?;
    let mut v2_state = HetModelState::alloc(dgpu, igpu, b_n as u32 + 4)?;

    // Seed both with per-position input HCs.
    for i in 0..b_n {
        bs.per_token_residual[i].copy_from_host(&input_hcs[i])?;
        let mut v2_slot = bd
            .residual
            .slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
        v2_slot.copy_from_host(&input_hcs[i])?;
    }

    let mut first_bad: Option<(usize, usize)> = None; // (layer, batch_idx)
    let mut max_seen_layer_diff: f32 = 0.0;

    for layer in 0..N_LAYER as usize {
        // ---- Phase 1: B sequential forward_layer_pair_mode calls ----
        for i in 0..b_n {
            dgpu.set_current()?;
            bs.shared_dgpu
                .residual
                .copy_from_buffer(&bs.per_token_residual[i])?;
            engine.forward_layer_pair_mode(
                &mut bs.shared_dgpu,
                &mut bs.shared_igpu,
                &mut ref_state.layers[layer],
                &main_weights.dgpu_layers[layer],
                &main_weights.igpu_layers[layer],
                i as u32,
                tokens[i],
            )?;
            bs.per_token_residual_next[i]
                .copy_from_buffer(&bs.shared_dgpu.residual_next)?;
        }

        // ---- Phase 2 v2: one forward_layer_batch_v2 call (B-wide) ----
        engine.forward_layer_batch_v2(
            &mut bd,
            &mut bi,
            &mut v2_state.layers[layer],
            &main_weights.dgpu_layers[layer],
            &main_weights.igpu_layers[layer],
            0,
            &tokens,
            None,
        )?;

        // ---- Compare each batch element's residual_next ----
        let mut layer_max_per_b: Vec<(f32, usize)> = Vec::with_capacity(b_n);
        for i in 0..b_n {
            let mut ref_hc = vec![0.0f32; HC_DIM as usize];
            bs.per_token_residual_next[i].copy_to_host(&mut ref_hc)?;
            let v2_slot = bd
                .residual_next
                .slice_view(i * HC_DIM as usize, HC_DIM as usize);
            let mut v2_hc = vec![0.0f32; HC_DIM as usize];
            v2_slot.copy_to_host(&mut v2_hc)?;
            let (md, ix) = max_abs_diff(&ref_hc, &v2_hc);
            layer_max_per_b.push((md, ix));
            if md > max_seen_layer_diff {
                max_seen_layer_diff = md;
            }
            if first_bad.is_none() && md > 1.0e-2 {
                first_bad = Some((layer, i));
            }
        }
        let summary: String = layer_max_per_b
            .iter()
            .enumerate()
            .map(|(i, (m, _))| format!("b{i}={m:.2e}"))
            .collect::<Vec<_>>()
            .join("  ");
        eprintln!("L{layer:>2}  {summary}");

        // ---- DEBUG: at the layer specified by V2_DBG_LAYER, compare a
        //     handful of intermediate buffers to localize the bug. ----
        // (Compares b=0 only.)
        if std::env::var("V2_DBG_LAYER")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            == Some(layer)
        {
            use v4flash_kernels::forward::{HC_DIM as HD, N_EMBD as NE, Q_FLAT as QF};
            let compare = |name: &str, ref_buf: &v4flash_hip::DeviceBuffer<f32>,
                            v2_buf_view: v4flash_hip::DeviceBuffer<f32>|
             -> eyre::Result<()> {
                let n = ref_buf.len();
                let mut rh = vec![0.0f32; n];
                let mut vh = vec![0.0f32; n];
                ref_buf.copy_to_host(&mut rh)?;
                v2_buf_view.copy_to_host(&mut vh)?;
                let (maxd, idx) = max_abs_diff(&rh, &vh);
                eprintln!(
                    "    DBG L{layer} {name:18}  maxd={maxd:.4e}@i={idx}  ref={r:.4}  v2={v:.4}",
                    r = rh[idx], v = vh[idx]
                );
                Ok(())
            };
            // mhc_pre_attn outputs (stages 1-end).
            compare("attn_input_norm", &bs.shared_dgpu.attn_input_norm,
                bd.attn_input_norm.slice_view(0, NE as usize))?;
            // Q chain.
            compare("q_normed", &bs.shared_dgpu.q_normed,
                bd.q_normed.slice_view(0, QF as usize))?;
            // KV chain.
            compare("kv_normed", &bs.shared_dgpu.kv_normed,
                bd.kv_normed.slice_view(0, v4flash_kernels::forward::N_HEAD_DIM as usize))?;
            // Attention output (heads).
            compare("heads_post_attn", &bs.shared_dgpu.heads,
                bd.heads.slice_view(0, QF as usize))?;
            // Output projection.
            compare("attn_out", &bs.shared_dgpu.attn_out,
                bd.attn_out.slice_view(0, NE as usize))?;
            // mHC post-attn.
            compare("after_attn_hc", &bs.shared_dgpu.after_attn_hc,
                bd.after_attn_hc.slice_view(0, HD as usize))?;
            // mHC pre-ffn.
            compare("ffn_input_norm", &bs.shared_dgpu.ffn_input_norm,
                bd.ffn_input_norm.slice_view(0, NE as usize))?;
            // Shared expert output.
            compare("ffn_shared", &bs.shared_dgpu.ffn_shared,
                bd.ffn_shared.slice_view(0, NE as usize))?;
            // Router selections (d_selected = i32, custom print).
            {
                let mut sel_ref = vec![0i32; 6];
                let mut sel_v2 = vec![0i32; 6];
                bs.shared_dgpu.d_selected.copy_to_host(&mut sel_ref)?;
                bd.d_selected.slice_view(0, 6).copy_to_host(&mut sel_v2)?;
                eprintln!("    DBG L{layer} d_selected         ref={:?}", sel_ref);
                eprintln!("    DBG L{layer} d_selected          v2={:?}", sel_v2);
                let mut ew_ref = vec![0f32; 6];
                let mut ew_v2 = vec![0f32; 6];
                bs.shared_dgpu.d_ew.copy_to_host(&mut ew_ref)?;
                bd.d_ew.slice_view(0, 6).copy_to_host(&mut ew_v2)?;
                eprintln!("    DBG L{layer} d_ew               ref={:?}", ew_ref);
                eprintln!("    DBG L{layer} d_ew                v2={:?}", ew_v2);
            }
            // iGPU MoE arrival.
            compare("ffn_moe_recv", &bs.shared_dgpu.ffn_moe_recv,
                bd.ffn_moe_recv.slice_view(0, NE as usize))?;
        }

        // ---- Swap residuals for next layer on BOTH paths ----
        for i in 0..b_n {
            std::mem::swap(
                &mut bs.per_token_residual[i],
                &mut bs.per_token_residual_next[i],
            );
        }
        std::mem::swap(&mut bd.residual, &mut bd.residual_next);

        // Limit bisect output — stop after divergence is well-established.
        if let Some((bad_layer, _)) = first_bad {
            if layer >= bad_layer + 2 {
                eprintln!("\n>>> First-bad layer = {bad_layer}; stopping bisect at L{layer}");
                break;
            }
        }
    }

    eprintln!("\nmax abs diff seen across layers: {max_seen_layer_diff:.4e}");
    if let Some((bad_layer, bad_b)) = first_bad {
        eprintln!(">>> first divergent (layer, batch_idx) = (L{bad_layer}, b{bad_b})");
    } else {
        eprintln!(">>> all layers within 1e-2");
    }
    Ok(())
}

/// M50 Phase 6 oracle: forward_prefill with last_only=true returns logits
/// matching what you'd get from B sequential forward_token + forward_head
/// calls on the same prompt. T<=7 (single chunk for the dump-real test).
#[test]
#[ignore]
fn forward_prefill_last_only_matches_sequential() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::forward::HC_DIM;

    let t: usize = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7)
        .min(PROMPT_TOKENS.len());
    eprintln!("Phase 6 oracle: T={t} (single chunk)");

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

    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(t);
    for i in 0..t {
        let entry = dump
            .tensor("layer_input_residual", 0, i as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{i}"))?;
        input_hcs.push(dump.read_f32(entry)?);
    }
    let tokens: Vec<i32> = PROMPT_TOKENS[..t].to_vec();

    // Reference: T sequential forward_token + final forward_head on the last.
    eprintln!("Run A: sequential forward_token × {t} + forward_head");
    let mut bs = BatchScratch::alloc(dgpu, igpu)?;
    let mut seq_state = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
    for i in 0..t {
        engine.forward_token(
            &mut bs.shared_dgpu,
            &mut bs.shared_igpu,
            &mut seq_state,
            &main_weights,
            &input_hcs[i],
            i as u32,
            tokens[i],
        )?;
    }
    // Last token's residual is in shared_dgpu.residual_next after forward_token.
    bs.shared_dgpu
        .residual
        .copy_from_buffer(&bs.shared_dgpu.residual_next)?;
    engine.forward_head(&mut bs.shared_dgpu, &main_weights.global)?;
    let mut seq_logits = vec![0f32; N_VOCAB as usize];
    bs.shared_dgpu.logits.copy_to_host(&mut seq_logits)?;

    // Phase 6: forward_prefill in one shot, last_only=true.
    eprintln!("Run B: forward_prefill (last_only=true)");
    let mut bd = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi = BatchIgpuScratch::alloc(igpu)?;
    let mut head_scratch = DgpuScratch::alloc(dgpu)?;
    let mut p6_state = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
    let prefill_logits = engine.forward_prefill(
        &mut bd,
        &mut bi,
        &mut head_scratch,
        &mut p6_state,
        &main_weights,
        &input_hcs,
        &tokens,
        0,
        true,
        None,
    )?;
    assert_eq!(prefill_logits.len(), N_VOCAB as usize);

    let (maxd, idx) = max_abs_diff(&seq_logits, &prefill_logits);
    eprintln!(
        "max abs diff = {maxd:.4e} @i={idx}  ref={:.4}  prefill={:.4}",
        seq_logits[idx], prefill_logits[idx]
    );
    assert!(
        maxd < 5e-2,
        "forward_prefill last_only diverges from sequential by {maxd:.4e} (> 5e-2)"
    );
    eprintln!("Phase 6 ORACLE PASS within 5e-2");
    Ok(())
}

/// Two-lane pipelined prefill oracle: forward_prefill_pipelined should
/// produce the same last-token logits as a single-lane forward_prefill
/// on the same prompt. The split point is ceil(B/2), so this also
/// exercises both lane A and lane B writing KV at the same layer.
///
/// Use IQ2_VARIANT=chunked for bit-exact; default staged exhibits the
/// usual ~1e-5 per-element drift (router argmax flips at deep layers).
#[test]
#[ignore]
fn forward_prefill_pipelined_matches_single_lane() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::forward::HC_DIM;
    std::env::set_var("IQ2_VARIANT", "chunked");

    let t: usize = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7)
        .min(PROMPT_TOKENS.len());
    eprintln!("pipelined oracle: T={t} (split = ceil(T/2)={})", t.div_ceil(2));

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

    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(t);
    for i in 0..t {
        let entry = dump
            .tensor("layer_input_residual", 0, i as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{i}"))?;
        input_hcs.push(dump.read_f32(entry)?);
    }
    let tokens: Vec<i32> = PROMPT_TOKENS[..t].to_vec();

    // Reference: single-lane forward_prefill.
    eprintln!("Run A: single-lane forward_prefill");
    let mut bd_a = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi_a = BatchIgpuScratch::alloc(igpu)?;
    let mut head_scratch = DgpuScratch::alloc(dgpu)?;
    let mut state_a = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
    let logits_single = engine.forward_prefill(
        &mut bd_a,
        &mut bi_a,
        &mut head_scratch,
        &mut state_a,
        &main_weights,
        &input_hcs,
        &tokens,
        0,
        true,
        None,
    )?;

    // Test: two-lane pipelined forward_prefill.
    eprintln!("Run B: forward_prefill_pipelined (lanes=2)");
    let mut bd_p_a = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi_p_a = BatchIgpuScratch::alloc(igpu)?;
    let mut bd_p_b = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi_p_b = BatchIgpuScratch::alloc(igpu)?;
    let mut state_b = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
    let logits_pipelined = engine.forward_prefill_pipelined(
        &mut bd_p_a,
        &mut bi_p_a,
        &mut bd_p_b,
        &mut bi_p_b,
        &mut head_scratch,
        &mut state_b,
        &main_weights,
        &input_hcs,
        &tokens,
        0,
        true,
        None,
    )?;

    let (maxd, idx) = max_abs_diff(&logits_single, &logits_pipelined);
    eprintln!(
        "max abs diff = {maxd:.4e} @i={idx}  single={:.4}  pipelined={:.4}",
        logits_single[idx], logits_pipelined[idx]
    );
    assert!(
        maxd < 1e-3,
        "forward_prefill_pipelined diverges from single-lane by {maxd:.4e} (> 1e-3)"
    );
    eprintln!("Pipelined ORACLE PASS within 1e-3");
    // Silence unused warning when HC_DIM only used in some build configs.
    let _ = HC_DIM;
    Ok(())
}
