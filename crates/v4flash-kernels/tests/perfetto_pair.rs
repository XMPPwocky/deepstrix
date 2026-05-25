//! Capture a perfetto trace of `forward_pair_interleaved` (M40 Phase 3).
//! After warming up the state with a few single-token forwards to fill
//! KV cache, runs several pair-forward iterations so you can see the
//! dGPU/iGPU substage interleaving and event-driven pipeline.
//!
//! Run:
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 \
//!   PERFETTO_OUT=/tmp/deepstrix-pair.pftrace \
//!   PERFETTO_DEVICE_OUT=/tmp/deepstrix-pair-device.pftrace \
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test perfetto_pair capture_perfetto_pair_trace -- --ignored --nocapture
//! ```
//! Open the .pftrace files at https://ui.perfetto.dev — drag & drop.
//! - host file: shows what the host (Rust) is doing per-stage (spans).
//! - device file: shows per-stream timing on dGPU/iGPU compute/xfer.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use color_eyre::eyre::{self, eyre};
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

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
const N_WARMUP: i32 = 8;
const N_PAIRS: i32 = 4;

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
fn capture_perfetto_pair_trace() -> eyre::Result<()> {
    install_panic_handler()?;

    let out_path = std::env::var("PERFETTO_OUT")
        .unwrap_or_else(|_| "/tmp/deepstrix-pair.pftrace".to_string());
    eprintln!("perfetto host-time trace: {out_path}");
    let _ = fs::remove_file(&out_path);
    let file = fs::File::create(&out_path)?;
    let writer = Arc::new(file);
    let perfetto_layer = PerfettoLayer::new(writer).with_debug_annotations(true);
    tracing_subscriber::registry()
        .with(EnvFilter::new(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "v4flash_kernels::het=debug".to_string()),
        ))
        .with(perfetto_layer)
        .try_init()
        .map_err(|e| eyre!("init tracing-perfetto subscriber: {e}"))?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let dgpu = pick_dgpu_device()?;
    let igpu = pick_igpu_device()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    eprintln!(
        "perfetto pair: dGPU id={} ({}) iGPU id={} ({})",
        dgpu.id, dgpu_arch, igpu.id, igpu_arch
    );

    let rope_for_layer = |layer: i32| -> eyre::Result<RopeParams> {
        let entry = dump
            .weight("rope_params", layer)
            .ok_or_else(|| eyre!("missing weight:rope_params for L{layer}"))?;
        let floats = dump.read_f32(entry)?;
        let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
        RopeParams::from_dump_blob(&floats, n_ctx_orig)
    };

    eprintln!("loading het weights (~minute)...");
    let weights = HetModelWeights::load_all(&gguf, dgpu, igpu, &rope_for_layer)?;
    eprintln!("loaded.");

    let mut engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let dev_out_path = std::env::var("PERFETTO_DEVICE_OUT")
        .unwrap_or_else(|_| "/tmp/deepstrix-pair-device.pftrace".to_string());
    let _ = fs::remove_file(&dev_out_path);
    eprintln!("perfetto device-time trace: {dev_out_path}");
    engine.attach_perfetto(&dev_out_path)?;

    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let total_positions = N_WARMUP + 2 * N_PAIRS + PROMPT_TOKENS.len() as i32 - 1;
    let mut state = HetModelState::alloc(dgpu, igpu, total_positions as u32)?;

    let max_inp_pos = dump.n_logit_rows as i32 + PROMPT_TOKENS.len() as i32 - 2;
    let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(total_positions as usize);
    for pos in 0..total_positions {
        let inp_pos = pos.min(max_inp_pos);
        let inp_entry = dump
            .tensor("layer_input_residual", 0, inp_pos)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{inp_pos}"))?;
        let input_hc = dump.read_f32(inp_entry)?;
        assert_eq!(input_hc.len(), HC_DIM as usize);
        inputs.push(input_hc);
    }

    // Warmup: single-token forwards to populate KV cache.
    eprintln!("warmup: {N_WARMUP} single-token forwards");
    for pos in 0..N_WARMUP {
        let token_id = if (pos as usize) < PROMPT_TOKENS.len() {
            PROMPT_TOKENS[pos as usize]
        } else {
            0
        };
        engine.forward_token(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &weights,
            &inputs[pos as usize],
            pos as u32,
            token_id,
        )?;
    }

    // Pair forwards traced.
    eprintln!("traced: {N_PAIRS} forward_pair_interleaved calls");
    for k in 0..N_PAIRS {
        let pos = N_WARMUP + 2 * k;
        let t0 = if (pos as usize) < PROMPT_TOKENS.len() {
            PROMPT_TOKENS[pos as usize]
        } else {
            0
        };
        let t1 = if ((pos + 1) as usize) < PROMPT_TOKENS.len() {
            PROMPT_TOKENS[(pos + 1) as usize]
        } else {
            0
        };
        let span = tracing::debug_span!("forward_pair_interleaved", pos, t0, t1).entered();
        engine.forward_pair_interleaved(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &weights,
            &inputs[pos as usize],
            &inputs[(pos + 1) as usize],
            pos as u32,
            t0,
            t1,
        )?;
        drop(span);
    }

    eprintln!("trace done");
    eprintln!("host file:   {out_path}");
    eprintln!("device file: {dev_out_path}");
    eprintln!("open at https://ui.perfetto.dev (drag & drop)");
    Ok(())
}
