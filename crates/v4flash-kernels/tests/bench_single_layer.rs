//! Micro-bench: drive `forward_layer_batch_v2` for a SINGLE layer in a
//! loop at a fake context depth. Pick the layer via LAYER_IDX, the
//! depth via FAKE_PREFILL_POS, and the trace output via PERFETTO_OUT.
//! All other layers' state is untouched — only the target layer's KV /
//! compressor counters get stamped.
//!
//! Why: lets us isolate one layer's per-call wall (e.g. compare a
//! ratio=4 layer's attention cost against a ratio=128 layer's at the
//! same depth) without paying the rest of the model on every iter.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 LAYER_IDX=2 FAKE_PREFILL_POS=32768 \
//!     PERFETTO_OUT=/tmp/layer2_32k.pftrace BENCH_ITERS=10 \
//!     nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_single_layer bench_single_layer \
//!     -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{COMPRESS_RATIOS, HC_DIM, N_LAYER, SWA_WINDOW};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, ExecMode,
    HetModelState, HetModelWeights, HeterogeneousEngine, B_MAX,
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

#[test]
#[ignore]
fn bench_single_layer() -> eyre::Result<()> {
    install_panic_handler()?;

    let layer_idx: usize = std::env::var("LAYER_IDX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    if layer_idx >= N_LAYER as usize {
        return Err(eyre!("LAYER_IDX {layer_idx} >= N_LAYER {N_LAYER}"));
    }
    let fake_pos: u32 = std::env::var("FAKE_PREFILL_POS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32768);
    let n_warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n_iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let b: usize = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(B_MAX);
    let ratio = COMPRESS_RATIOS[layer_idx];
    eprintln!(
        "bench_single_layer: L{layer_idx} (ratio={ratio}), \
         B={b}, fake_pos={fake_pos}, warmup={n_warmup}, iters={n_iters}"
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
    let mut bd = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi = BatchIgpuScratch::alloc(igpu)?;
    let mut sd = BatchDgpuShared::alloc(dgpu)?;
    let mut si = BatchIgpuShared::alloc(igpu)?;

    // State sized for the deepest position we touch on this layer.
    let n_kv_max = fake_pos + b as u32 + 8;
    let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;

    // Seed bd.residual once with cyclic real-dump residuals — values
    // are arbitrary for timing; the layer kernels don't short-circuit
    // on data.
    let n_real = PROMPT_TOKENS.len();
    {
        let cs_hc = HC_DIM as usize;
        for i in 0..b {
            let src_i = i % n_real;
            let entry = dump
                .tensor("layer_input_residual", 0, src_i as i32)
                .ok_or_else(|| eyre!("missing layer_input_residual L0 T{src_i}"))?;
            let hc = dump.read_f32(entry)?;
            assert_eq!(hc.len(), cs_hc);
            let mut slot = bd.residual.slice_view_mut(i * cs_hc, cs_hc);
            slot.copy_from_host(&hc)?;
        }
    }

    // pos_per_b = [fake_pos, fake_pos+1, ..., fake_pos+b-1] (fixed across iters)
    {
        let pos_host: Vec<i32> = (0..b).map(|i| (fake_pos + i as u32) as i32).collect();
        let mut pv = bd.pos_per_b.slice_view_mut(0, b);
        pv.copy_from_host(&pos_host)?;
    }

    let tokens: Vec<i32> = (0..b).map(|i| PROMPT_TOKENS[i % n_real]).collect();

    // Reset the target layer's depth counters to the fake-pos state.
    let stamp = |state: &mut HetModelState| {
        let ls = &mut state.layers[layer_idx];
        ls.n_raw = SWA_WINDOW.min(fake_pos);
        if ratio > 0 {
            if let Some(cs) = ls.compressor.as_mut() {
                cs.n_comp = fake_pos / ratio;
            }
        }
    };

    eprintln!("warmup x {n_warmup} (also captures HIP graph)");
    for _ in 0..n_warmup {
        stamp(&mut state);
        engine.forward_layer_batch_v2(
            &mut bd,
            &mut bi,
            &mut sd,
            &mut si,
            &mut state.layers[layer_idx],
            &main_weights.dgpu_layers[layer_idx],
            &main_weights.igpu_layers[layer_idx],
            fake_pos,
            &tokens,
            None,
        )?;
        engine.dgpu.compute.synchronize()?;
        engine.dgpu.xfer.synchronize()?;
        engine.igpu.compute.synchronize()?;
        engine.igpu.xfer.synchronize()?;
    }

    if let Some(p) = std::env::var("PERFETTO_OUT").ok() {
        eprintln!("perfetto: attaching after warmup → {p}");
        engine.attach_perfetto(&p)?;
    }

    let mut walls_ms: Vec<f64> = Vec::with_capacity(n_iters);
    for it in 0..n_iters {
        stamp(&mut state);
        engine.dgpu.events.reset();
        engine.igpu.events.reset();
        let t0 = Instant::now();
        engine.forward_layer_batch_v2(
            &mut bd,
            &mut bi,
            &mut sd,
            &mut si,
            &mut state.layers[layer_idx],
            &main_weights.dgpu_layers[layer_idx],
            &main_weights.igpu_layers[layer_idx],
            fake_pos,
            &tokens,
            None,
        )?;
        engine.dgpu.compute.synchronize()?;
        engine.dgpu.xfer.synchronize()?;
        engine.igpu.compute.synchronize()?;
        engine.igpu.xfer.synchronize()?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        walls_ms.push(wall_ms);
        engine.flush_perfetto()?;
        eprintln!(
            "  iter {it}: wall={:.2} ms  ({:.3} ms/tok = {:.1} tok/s)",
            wall_ms,
            wall_ms / b as f64,
            (b as f64 * 1000.0) / wall_ms
        );
    }

    walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_ms = walls_ms[0];
    let median_ms = walls_ms[walls_ms.len() / 2];
    eprintln!(
        "\n=== BENCH SINGLE LAYER L{layer_idx} (ratio={ratio}) @ depth {fake_pos} B={b} ==="
    );
    eprintln!(
        "best:    {:>7.2} ms ({:.3} ms/tok = {:.1} tok/s)",
        min_ms,
        min_ms / b as f64,
        (b as f64 * 1000.0) / min_ms
    );
    eprintln!(
        "median:  {:>7.2} ms ({:.3} ms/tok = {:.1} tok/s)",
        median_ms,
        median_ms / b as f64,
        (b as f64 * 1000.0) / median_ms
    );
    engine.shutdown()?;
    Ok(())
}
