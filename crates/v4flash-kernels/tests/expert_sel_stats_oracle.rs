//! M62 oracle: the device-side expert_sel_count histogram exactly equals
//! the host-side PrefillStats pick_counts collected on the same run, and
//! the prefill-token counter equals the prompt length.
//!
//! Run:
//!   DGPU_HOT_EXPERTS=8 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test expert_sel_stats_oracle \
//!     -- --ignored --nocapture --test-threads=1

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{N_EXPERT, N_EXPERT_USED, N_LAYER};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchIgpuScratch, DgpuScratch, ExecMode, HetModelState, HetModelWeights,
    HeterogeneousEngine, PrefillStats,
};
use v4flash_kernels::{oracle::ActivationDump, RopeParams};

const MAIN_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
}

fn pick(arch: &str) -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(arch) {
            return Ok(d);
        }
    }
    Err(eyre!("no {arch}"))
}

#[test]
#[ignore]
fn device_histogram_matches_prefill_stats() -> eyre::Result<()> {
    install_panic_handler()?;
    let t = PROMPT_TOKENS.len();
    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MAIN_MODEL_PATH)?;
    let dgpu = pick("gfx1201")?;
    let igpu = pick("gfx1151")?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    let rope = |layer: i32| -> eyre::Result<RopeParams> {
        let entry = dump
            .weight("rope_params", layer)
            .ok_or_else(|| eyre!("missing rope_params L{layer}"))?;
        let floats = dump.read_f32(entry)?;
        let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
        RopeParams::from_dump_blob(&floats, n_ctx_orig)
    };
    let weights = HetModelWeights::load_all(&gguf, dgpu, igpu, &rope)?;
    let engine = HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;

    let mut input_hcs = Vec::with_capacity(t);
    for i in 0..t {
        let entry = dump
            .tensor("layer_input_residual", 0, i as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{i}"))?;
        input_hcs.push(dump.read_f32(entry)?);
    }
    let tokens: Vec<i32> = PROMPT_TOKENS.to_vec();

    let mut bd = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi = BatchIgpuScratch::alloc(igpu)?;
    let mut head = DgpuScratch::alloc(dgpu)?;
    let mut state = HetModelState::alloc(dgpu, igpu, t as u32 + 4)?;
    let mut stats = PrefillStats::new(N_LAYER as u32, N_EXPERT_USED as u32, N_EXPERT);
    let _ = engine.forward_prefill(
        &mut bd,
        &mut bi,
        &mut head,
        &mut state,
        &weights,
        &input_hcs,
        &tokens,
        0,
        true,
        Some(&mut stats),
    )?;

    let ((prefill_counts, prefill_tokens), (_, decode_tokens)) = engine.harvest_sel_stats()?;
    assert_eq!(prefill_tokens, t as u64, "prefill token counter");
    assert_eq!(decode_tokens, 0, "decode bank must be untouched");

    let mut n_diff = 0usize;
    for l in 0..N_LAYER as usize {
        for e in 0..N_EXPERT as usize {
            let dev = prefill_counts[l * (N_EXPERT as usize) + e] as u64;
            let host = stats.layers[l].pick_counts[e] as u64;
            if dev != host {
                if n_diff < 8 {
                    eprintln!("L{l} e{e}: device={dev} host={host}");
                }
                n_diff += 1;
            }
        }
    }
    eprintln!("device vs host histogram: n_diff={n_diff} (tokens={prefill_tokens})");
    assert_eq!(n_diff, 0, "device histogram diverges from PrefillStats");

    // Second harvest must be empty (banks zeroed).
    let ((c2, t2), _) = engine.harvest_sel_stats()?;
    assert_eq!(t2, 0);
    assert!(c2.iter().all(|&c| c == 0), "bank not zeroed after harvest");
    eprintln!("M62 sel-stats ORACLE PASS");
    engine.shutdown()?;
    Ok(())
}
