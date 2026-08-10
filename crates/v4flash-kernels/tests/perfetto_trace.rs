//! Capture a perfetto trace of the het orchestrator running ~5 tokens
//! through the forward pass. Open the output in https://ui.perfetto.dev
//! to inspect the dGPU/iGPU pipeline timing.
//!
//! Run:
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 \
//!   PERFETTO_OUT=/tmp/deepstrix-5tok.pftrace \
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test perfetto_trace -- --ignored --nocapture
//! ```

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
use v4flash_kernels::config::{HC_DIM, N_LAYER};
use v4flash_kernels::het::{
    DgpuScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine, IgpuScratch,
};
use v4flash_kernels::{oracle::ActivationDump, RopeParams};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;
const N_TRACE_TOKENS: i32 = 5;

fn dump_dir() -> PathBuf {
    std::env::var("DEEPSTRIX_DUMP_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("reference/v4flash-cpu-activations")
    })
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
fn capture_perfetto_trace() -> eyre::Result<()> {
    install_panic_handler()?;

    let out_path = std::env::var("PERFETTO_OUT")
        .unwrap_or_else(|_| "/tmp/deepstrix-het.pftrace".to_string());
    eprintln!("perfetto trace destination: {out_path}");
    // Drop any prior trace at the same path so reruns start clean.
    let _ = fs::remove_file(&out_path);
    let file = fs::File::create(&out_path)?;
    let writer = Arc::new(file);

    // Capture everything emitted under `v4flash_kernels::het` at DEBUG+
    // (which is where the per-stage `debug_span!`s live). The PerfettoLayer
    // wraps spans as TYPE_SLICE_BEGIN/END.
    let perfetto_layer = PerfettoLayer::new(writer).with_debug_annotations(true);

    tracing_subscriber::registry()
        .with(EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| {
                "v4flash_kernels::het=debug".to_string()
            }),
        ))
        .with(perfetto_layer)
        .try_init()
        .map_err(|e| eyre!("init tracing-perfetto subscriber: {e}"))?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(std::env::var("DEEPSTRIX_GGUF").unwrap_or_else(|_| MODEL_PATH.to_string()))?;

    let dgpu = pick_dgpu_device()?;
    let igpu = pick_igpu_device()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    eprintln!(
        "perfetto: dGPU id={} ({}) iGPU id={} ({})",
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

    eprintln!("loading het weights — this is the slow part (~minute)...");
    let weights = HetModelWeights::load_all(&gguf, dgpu, igpu, &rope_for_layer)?;
    eprintln!("het weights loaded.");

    let mut engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    // Open a separate file for the device-time tracks; we don't try to
    // merge with the tracing-perfetto host-time file because both want
    // exclusive ownership of the underlying File handle (tracing-perfetto
    // takes an Arc<File> for itself).
    let dev_out_path = std::env::var("PERFETTO_DEVICE_OUT")
        .unwrap_or_else(|_| "/tmp/deepstrix-het-device.pftrace".to_string());
    let _ = fs::remove_file(&dev_out_path);
    eprintln!("device-time perfetto trace: {dev_out_path}");
    engine.attach_perfetto(&dev_out_path)?;
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let n_positions = N_TRACE_TOKENS + PROMPT_TOKENS.len() as i32 - 1; // prompt + decode
    let mut state = HetModelState::alloc(dgpu, igpu, n_positions as u32)?;

    eprintln!("forward pass: {n_positions} positions...");
    for pos in 0..n_positions {
        let token_id = if (pos as usize) < PROMPT_TOKENS.len() {
            PROMPT_TOKENS[pos as usize]
        } else {
            // After the prompt, just feed an arbitrary id — we're tracing,
            // not generating coherent output.
            0
        };
        let inp_entry = dump
            .tensor("layer_input_residual", 0, pos.min(PROMPT_TOKENS.len() as i32 - 1))
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{pos}"))?;
        let input_hc = dump.read_f32(inp_entry)?;
        assert_eq!(input_hc.len(), HC_DIM as usize);
        engine.forward_token(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &weights,
            &input_hc,
            pos as u32,
            token_id,
        )?;
    }

    eprintln!("forward done, {} layers/token", N_LAYER);
    eprintln!("wrote perfetto trace: {out_path}");
    eprintln!("open at https://ui.perfetto.dev — drag and drop the file");
    Ok(())
}
