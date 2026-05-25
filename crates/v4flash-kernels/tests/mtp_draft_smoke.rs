//! M40-P2.3 smoke + P2.5 mini-bench: run `forward_mtp_draft` end-to-end.
//! Confirms the 6-stage MTP pipeline executes without crashing and
//! produces logits. Measures wall time per draft to inform whether to
//! move MTP to iGPU for overlap (per the design doc's Phase 2 decision
//! point).
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test mtp_draft_smoke -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::het::{
    DgpuScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine, IgpuScratch,
    MtpWeights,
};
use v4flash_kernels::{ActivationDump, RopeParams};

const MAIN_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
const MTP_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf";

const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const PROMPT_LEN: i32 = PROMPT_TOKENS.len() as i32;
const ROPE_ORIG_CTX: u64 = 65536;

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

/// Read row `token_id` from the main model's `token_embd.weight` (F16
/// [N_VOCAB, N_EMBD]) and dequantize to F32. Used to supply the MTP
/// draft's last-token embedding (the proper path would be a device-
/// side embed kernel; for prototype, host-side is fine).
fn embed_token_host(gguf: &MappedGguf, token_id: i32, n_embd: u32) -> eyre::Result<Vec<f32>> {
    let t = gguf
        .gguf()
        .tensor("token_embd.weight")
        .ok_or_else(|| eyre!("token_embd.weight missing"))?;
    if t.dtype != GgufType::F16 {
        return Err(eyre!("token_embd dtype {:?} != F16", t.dtype));
    }
    // F16 row layout: [N_VOCAB, N_EMBD], each row N_EMBD * 2 bytes.
    let row_bytes = (n_embd as usize) * 2;
    let bytes = gguf.read_tensor(t)?;
    let off = (token_id as usize) * row_bytes;
    let row = &bytes[off..off + row_bytes];
    let mut out = vec![0f32; n_embd as usize];
    for i in 0..n_embd as usize {
        let bits = u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]);
        out[i] = v4flash_kernels::iq2_xxs_tables::f16_to_f32(bits);
    }
    Ok(out)
}

#[test]
#[ignore]
fn mtp_draft_smoke_and_bench() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let main_gguf = MappedGguf::open(MAIN_MODEL_PATH)?;
    let mtp_gguf = MappedGguf::open(MTP_MODEL_PATH)?;

    let dgpu = pick_dgpu_device()?;
    let igpu = pick_igpu_device()?;
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
    let rope = rope_for_layer(0)?;

    eprintln!("loading main weights (~9 GiB dGPU + ~52 GiB iGPU)...");
    let main_weights = HetModelWeights::load_all(&main_gguf, dgpu, igpu, &rope_for_layer)?;
    eprintln!("loading MTP weights (~120 MB dGPU + ~1.2 GiB iGPU)...");
    let mtp_weights = MtpWeights::load(&mtp_gguf, dgpu, igpu, rope)?;
    eprintln!("weights loaded.");

    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let n_positions = PROMPT_LEN + 5; // prompt + 5 decode tokens
    let mut state = HetModelState::alloc(dgpu, igpu, n_positions as u32)?;
    state.alloc_mtp(dgpu)?;

    use v4flash_kernels::forward::{HC_DIM, N_EMBD, N_VOCAB};

    // Run the main model forward through the prompt + 3 decode tokens to
    // populate KV cache. We'll then take the HC state at one of those
    // positions and use it as MTP's prev_hc.
    let warmup_tokens = (PROMPT_LEN + 3) as i32;
    eprintln!("warmup: {warmup_tokens} main-model forward_tokens...");
    let mut last_hc_dev: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, HC_DIM as usize)?;
    let mut last_token: i32 = 0;
    for pos in 0..warmup_tokens {
        let token_id = if (pos as usize) < PROMPT_TOKENS.len() {
            PROMPT_TOKENS[pos as usize]
        } else {
            0
        };
        let inp_entry = dump
            .tensor("layer_input_residual", 0, pos.min(PROMPT_LEN - 1))
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{pos}"))?;
        let input_hc = dump.read_f32(inp_entry)?;
        engine.forward_token(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &main_weights,
            &input_hc,
            pos as u32,
            token_id,
        )?;
        // After forward_token, residual holds the final-layer HC for THIS pos.
        if pos == warmup_tokens - 1 {
            // M15.1 epilogue swap was done; residual now reflects initial parity.
            // The "final HC" is in residual_next at end of forward_token, then
            // swapped into residual via the epilogue. Copy it out.
            last_hc_dev.copy_from_buffer(&dgpu_scratch.residual)?;
            last_token = token_id;
        }
    }
    eprintln!("warmup done. mtp_state.n_raw = {}", state.mtp.as_ref().unwrap().n_raw);

    // Compute embedding of last_token for MTP input.
    let last_embd = embed_token_host(&main_gguf, last_token, N_EMBD)?;

    // Smoke run: 1 MTP draft.
    eprintln!("running 1 MTP draft at pos={}, token={last_token}...", warmup_tokens);
    let t0 = Instant::now();
    engine.forward_mtp_draft(
        &mut dgpu_scratch,
        &mut igpu_scratch,
        &mut state,
        &main_weights.global,
        &mtp_weights,
        &last_hc_dev,
        &last_embd,
        warmup_tokens as u32,
        last_token,
    )?;
    let smoke_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("MTP draft (first call): {smoke_ms:.2} ms");

    let mut logits_host = vec![0f32; N_VOCAB as usize];
    dgpu_scratch.mtp_logits.copy_to_host(&mut logits_host)?;
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits_host.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    eprintln!("MTP draft top-1 token id = {best} (logit = {best_v:.3})");
    let logits_finite = logits_host.iter().filter(|v| v.is_finite()).count();
    eprintln!(
        "{}/{} logits finite ({:.3}% non-NaN)",
        logits_finite,
        N_VOCAB,
        100.0 * logits_finite as f64 / N_VOCAB as f64
    );
    assert!(
        logits_finite as u32 == N_VOCAB,
        "MTP logits contain NaN/inf"
    );

    // Mini-bench: 10 more calls, report median wall.
    eprintln!("mini-bench: 10 MTP draft calls back-to-back...");
    let mut times_us: Vec<u64> = Vec::with_capacity(10);
    for i in 0..10 {
        let t = Instant::now();
        engine.forward_mtp_draft(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &main_weights.global,
            &mtp_weights,
            &last_hc_dev,
            &last_embd,
            warmup_tokens as u32 + 1 + i as u32,
            last_token,
        )?;
        times_us.push(t.elapsed().as_micros() as u64);
    }
    times_us.sort_unstable();
    let p50 = times_us[5];
    let p90 = times_us[9];
    let min = times_us[0];
    eprintln!(
        "MTP draft bench: min={:.2} ms, p50={:.2} ms, p90={:.2} ms",
        min as f64 / 1000.0,
        p50 as f64 / 1000.0,
        p90 as f64 / 1000.0
    );

    // Phase 2 decision point per the plan:
    eprintln!();
    if p50 < 5_000 {
        eprintln!("(< 5 ms) → keep MTP on dGPU; serialize after target forward in spec_decode loop");
    } else {
        eprintln!("(>= 5 ms) → consider moving MTP to iGPU for overlap with target forward");
    }
    Ok(())
}
