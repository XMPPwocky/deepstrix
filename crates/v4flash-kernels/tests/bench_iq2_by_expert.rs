//! M50 Phase 7 micro-bench: isolate the iq2 MoE kernel wall under
//! by-token vs by-expert. Both paths consume the same inputs
//! (d_xq_q8k, d_selected, d_ew, expert weights) and produce d_mid_cat.
//!
//! Uses hipEvent-based timing so we get GPU-side kernel wall, not host
//! launch latency. Runs each kernel N iterations, takes the median.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device, Event};
use v4flash_kernels::forward::{
    BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, SWIGLU_CLAMP_EXP,
};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchIgpuScratch, BatchScratch, ExecMode, HetModelState, HetModelWeights,
    HeterogeneousEngine, PrefillStats,
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

/// Set up a realistic iq2 input state by running forward_prefill once.
/// After this, `bi` has populated d_xq_q8k / d_selected / d_ew / group_count
/// / expert_members for the LAST chunk processed. We then time the iq2
/// kernel in isolation under both call paths.
#[test]
#[ignore]
fn bench_iq2_by_token_vs_by_expert() -> eyre::Result<()> {
    install_panic_handler()?;
    use v4flash_kernels::forward::HC_DIM;

    let b: u32 = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    eprintln!("iq2 by-token vs by-expert: B={b}, iters={iters}");

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

    let mut bs = BatchScratch::alloc(dgpu, igpu)?;
    let mut bd = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi = BatchIgpuScratch::alloc(igpu)?;

    // Seed input_hcs (cyclically from dump) for B tokens.
    let n_real = PROMPT_TOKENS.len();
    let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(b as usize);
    let mut tokens: Vec<i32> = Vec::with_capacity(b as usize);
    for i in 0..b as usize {
        let src = i % n_real;
        let entry = dump
            .tensor("layer_input_residual", 0, src as i32)
            .ok_or_else(|| eyre!("missing"))?;
        let hc = dump.read_f32(entry)?;
        assert_eq!(hc.len(), HC_DIM as usize);
        input_hcs.push(hc);
        tokens.push(PROMPT_TOKENS[src]);
    }

    // Warm: run forward_prompt_batch_v2 once with stats. This populates bi
    // with realistic selections from layer 42 (the last layer processed).
    // Then bi.group_count + bi.expert_members are valid for layer 42's pick.
    let mut state = HetModelState::alloc(dgpu, igpu, b + 4)?;
    let mut stats = PrefillStats::new(43, N_EXPERT_USED as u32, N_EXPERT);
    eprintln!("warmup: forward_prompt_batch_v2 to populate bi.{{d_xq_q8k,d_selected,d_ew}}");
    engine.forward_prompt_batch_v2(
        &mut bd,
        &mut bi,
        &mut state,
        &main_weights,
        &input_hcs,
        &tokens,
        0,
        Some(&mut stats),
    )?;

    // Pick the last layer's weights as the kernel target (matches the
    // state we just left bi in).
    let layer = 42usize;
    let ilw = &main_weights.igpu_layers[layer];
    let gbpe = ilw.routed.gate_bytes_per_expert as u32;
    let ubpe = ilw.routed.up_bytes_per_expert as u32;

    let cs_n_used = N_EXPERT_USED as u32;

    // Force iGPU current.
    igpu.set_current()?;
    let ie = &engine.igpu;

    // Re-derive d_xq_q8k from the last chunk: we'd need ain in
    // bi.ffn_input_norm_recv to be the last chunk's last layer's input,
    // which is what forward_prompt_batch_v2 just produced. d_xq_q8k was
    // overwritten on every layer's q8k.launch — it's the LAST layer's xq.
    // d_selected was the LAST layer's picks. group_count/expert_members
    // were the LAST layer's groups. All consistent.

    // ---- Build group state freshly (in case anything got overwritten). ----
    let max_per_expert = bi.max_per_expert();
    bi.group_count.fill_zero()?;
    {
        let BatchIgpuScratch {
            group_count,
            expert_members,
            d_selected,
            ..
        } = &mut bi;
        ie.moe_group_builder.launch(
            &ie.compute,
            group_count,
            expert_members,
            d_selected,
            b,
            cs_n_used,
            N_EXPERT,
            max_per_expert,
        )?;
    }
    ie.compute.synchronize()?;

    // ---- Time the by-token kernel N iters ----
    let mut by_token_ms: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&ie.compute)?;
        ie.iq2.launch_fused_swiglu_batch_bxn(
            &ie.compute,
            &mut bi.d_mid_cat,
            &ilw.routed.gate.buffer,
            &ilw.routed.up.buffer,
            &bi.d_xq_q8k,
            &bi.d_ew,
            &bi.d_selected,
            gbpe,
            ubpe,
            cs_n_used,
            SWIGLU_CLAMP_EXP,
            N_FF_EXP,
            BLOCKS_Q8K_GATE_IN,
            b,
        )?;
        end.record(&ie.compute)?;
        ie.compute.synchronize()?;
        by_token_ms.push(Event::elapsed_ms(&start, &end)?);
    }

    // ---- Time the by-expert v0 kernel N iters ----
    let mut by_expert_ms: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&ie.compute)?;
        let BatchIgpuScratch {
            d_mid_cat,
            d_xq_q8k,
            d_ew,
            group_count,
            expert_members,
            ..
        } = &mut bi;
        ie.iq2.launch_fused_swiglu_by_expert(
            &ie.compute,
            d_mid_cat,
            &ilw.routed.gate.buffer,
            &ilw.routed.up.buffer,
            d_xq_q8k,
            d_ew,
            group_count,
            expert_members,
            gbpe,
            ubpe,
            cs_n_used,
            N_EXPERT,
            max_per_expert,
            SWIGLU_CLAMP_EXP,
            N_FF_EXP,
            BLOCKS_Q8K_GATE_IN,
        )?;
        end.record(&ie.compute)?;
        ie.compute.synchronize()?;
        by_expert_ms.push(Event::elapsed_ms(&start, &end)?);
    }

    // ---- Phase 7.2: Time the chunked by-expert kernel N iters ----
    // Need to (re)build the work_items pre-pass each call too, since the
    // chunked kernel reads it. Build once outside the timing loop —
    // group_count + work_items don't change across iters.
    let chunk_size: u32 = std::env::var("BENCH_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let CHUNK_SIZE = chunk_size; // for naming consistency below
    bi.n_work_items.fill_zero()?;
    {
        let BatchIgpuScratch {
            work_items,
            n_work_items,
            group_count,
            ..
        } = &mut bi;
        let max_items = work_items.len() as u32;
        ie.moe_group_builder.launch_work_items(
            &ie.compute,
            work_items,
            n_work_items,
            group_count,
            N_EXPERT,
            CHUNK_SIZE,
            max_items,
        )?;
    }
    ie.compute.synchronize()?;
    let mut n_wi_host = [0i32; 1];
    bi.n_work_items.copy_to_host(&mut n_wi_host)?;
    let n_work_items = n_wi_host[0] as u32;

    let mut chunked_ms: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&ie.compute)?;
        let BatchIgpuScratch {
            d_mid_cat,
            d_xq_q8k,
            d_ew,
            group_count,
            expert_members,
            work_items,
            ..
        } = &mut bi;
        ie.iq2.launch_fused_swiglu_chunked(
            &ie.compute,
            d_mid_cat,
            &ilw.routed.gate.buffer,
            &ilw.routed.up.buffer,
            d_xq_q8k,
            d_ew,
            group_count,
            expert_members,
            work_items,
            gbpe,
            ubpe,
            cs_n_used,
            max_per_expert,
            CHUNK_SIZE,
            SWIGLU_CLAMP_EXP,
            N_FF_EXP,
            BLOCKS_Q8K_GATE_IN,
            n_work_items,
        )?;
        end.record(&ie.compute)?;
        ie.compute.synchronize()?;
        chunked_ms.push(Event::elapsed_ms(&start, &end)?);
    }

    // ---- DIAGNOSTIC: chunked kernel with dot-product stubbed.
    // Same launch, same WG geometry, same LDS init + xq cooperative loads,
    // but no dot work. If wall ≈ same → dot is amortized into other costs
    // (BW, LDS, sync); if wall ≈ 0 → dot is the bottleneck.
    let mut nodot_ms: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&ie.compute)?;
        let BatchIgpuScratch {
            d_mid_cat,
            d_xq_q8k,
            d_ew,
            group_count,
            expert_members,
            work_items,
            ..
        } = &mut bi;
        ie.iq2.launch_fused_swiglu_chunked_nodot(
            &ie.compute,
            d_mid_cat,
            &ilw.routed.gate.buffer,
            &ilw.routed.up.buffer,
            d_xq_q8k,
            d_ew,
            group_count,
            expert_members,
            work_items,
            gbpe,
            ubpe,
            cs_n_used,
            max_per_expert,
            CHUNK_SIZE,
            SWIGLU_CLAMP_EXP,
            N_FF_EXP,
            BLOCKS_Q8K_GATE_IN,
            n_work_items,
        )?;
        end.record(&ie.compute)?;
        ie.compute.synchronize()?;
        nodot_ms.push(Event::elapsed_ms(&start, &end)?);
    }

    fn median(v: &mut [f32]) -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }
    fn pmin(v: &[f32]) -> f32 {
        v.iter().copied().fold(f32::INFINITY, f32::min)
    }
    let bt_min = pmin(&by_token_ms);
    let be_min = pmin(&by_expert_ms);
    let ch_min = pmin(&chunked_ms);
    let bt_med = median(&mut by_token_ms.clone());
    let be_med = median(&mut by_expert_ms.clone());
    let ch_med = median(&mut chunked_ms.clone());

    eprintln!("\n=== iq2 kernel wall (B={b}, {iters} iters, hipEvent-timed) ===");
    eprintln!("by-token   min={:.3} ms  median={:.3} ms", bt_min, bt_med);
    eprintln!("by-expert  min={:.3} ms  median={:.3} ms  (ratio min={:.3}x  med={:.3}x)",
              be_min, be_med, be_min / bt_min, be_med / bt_med);
    eprintln!("chunked-{chunk_size} min={:.3} ms  median={:.3} ms  (ratio min={:.3}x  med={:.3}x)  n_work_items={n_work_items}",
              ch_min, ch_med, ch_min / bt_min, ch_med / bt_med);
    let nd_min = pmin(&nodot_ms);
    let nd_med = median(&mut nodot_ms.clone());
    eprintln!("nodot-{chunk_size}   min={:.3} ms  median={:.3} ms  (= chunked - dot cost)",
              nd_min, nd_med);
    eprintln!("  dot-only cost:  min={:.3} ms  med={:.3} ms  ({:.0}% of chunked wall)",
              ch_min - nd_min, ch_med - nd_med, 100.0 * (ch_med - nd_med) / ch_med);

    // ---- Phase 7.3: LDS-staged weights variant ----
    let mut lds_ms: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&ie.compute)?;
        let BatchIgpuScratch {
            d_mid_cat,
            d_xq_q8k,
            d_ew,
            group_count,
            expert_members,
            work_items,
            ..
        } = &mut bi;
        ie.iq2.launch_fused_swiglu_chunked_lds(
            &ie.compute,
            d_mid_cat,
            &ilw.routed.gate.buffer,
            &ilw.routed.up.buffer,
            d_xq_q8k,
            d_ew,
            group_count,
            expert_members,
            work_items,
            gbpe,
            ubpe,
            cs_n_used,
            max_per_expert,
            CHUNK_SIZE,
            SWIGLU_CLAMP_EXP,
            N_FF_EXP,
            BLOCKS_Q8K_GATE_IN,
            n_work_items,
        )?;
        end.record(&ie.compute)?;
        ie.compute.synchronize()?;
        lds_ms.push(Event::elapsed_ms(&start, &end)?);
    }
    let lds_min = pmin(&lds_ms);
    let lds_med = median(&mut lds_ms.clone());
    eprintln!("lds-{chunk_size}     min={:.3} ms  median={:.3} ms  (ratio min={:.3}x  med={:.3}x)",
              lds_min, lds_med, lds_min / bt_min, lds_med / bt_med);

    // Print the active-expert count to give context for the ratio.
    let mut gc_host = vec![0i32; N_EXPERT as usize];
    bi.group_count.copy_to_host(&mut gc_host)?;
    let active = gc_host.iter().filter(|&&c| c > 0).count();
    let max_group = gc_host.iter().copied().max().unwrap_or(0);
    let total_picks: i32 = gc_host.iter().sum();
    eprintln!(
        "  context: L{layer} d_selected has {active} active experts ({:.1}%), max group size {max_group}, total picks {total_picks}",
        100.0 * active as f32 / N_EXPERT as f32
    );
    Ok(())
}
