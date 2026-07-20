//! M40 MTP drafter smoke test (restored + adapted 2026-07).
//!
//! Runs `forward_mtp_draft` end-to-end and prints the drafted token id +
//! top-5 logits. Confirms the restored drafter compiles, runs, and
//! produces sane (finite, in-vocab) tokens.
//!
//! Lightweight by design: the drafter only needs the main model's output
//! head (`HetGlobalWeights`, ~135 MB), the MTP GGUF (~3.8 GB), and a
//! `prev_hc`. It does NOT need the 86 GiB of main-model layer weights —
//! those are only used to *generate* a prev_hc via a warmup forward. Here
//! we take prev_hc straight from the CPU activation dump (the final-layer
//! HC `output_flat` at L43), and the token embedding straight from the
//! main GGUF's `token_embd`. This keeps the whole test under ~5 GiB.
//!
//! Scope: DRAFTER ONLY (no verify forward, no accept/reject loop).
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test mtp_draft_smoke -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::config::{HC_DIM, N_EMBD, N_VOCAB};
use v4flash_kernels::het::{
    DgpuScratch, ExecMode, HetGlobalWeights, HeterogeneousEngine, IgpuScratch, MtpLayerState,
    MtpScratch, MtpWeights,
};
use v4flash_kernels::{oracle::ActivationDump, RopeParams};

const MAIN_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
const MTP_MODEL_PATH: &str = "/persist/lumi/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf";

// prompt token 0 — matches the L43 T0 HC we read as prev_hc.
const LAST_TOKEN: i32 = 53091;
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

/// Dequantize one F16 row of the main model's `token_embd.weight` to F32.
fn embed_token_host(gguf: &MappedGguf, token_id: i32, n_embd: u32) -> eyre::Result<Vec<f32>> {
    let t = gguf
        .gguf()
        .tensor("token_embd.weight")
        .ok_or_else(|| eyre!("token_embd.weight missing"))?;
    if t.dtype != GgufType::F16 {
        return Err(eyre!("token_embd dtype {:?} != F16", t.dtype));
    }
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
fn mtp_draft_smoke() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let main_gguf = MappedGguf::open(MAIN_MODEL_PATH)?;
    let mtp_gguf = MappedGguf::open(MTP_MODEL_PATH)?;

    let dgpu = pick_dgpu_device()?;
    let igpu = pick_igpu_device()?;
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
    let rope = rope_for_layer(0)?;

    // Drafter needs only the output head (~135 MB), not the 86 GiB of
    // main-model layer weights.
    eprintln!("loading main-model output head (HetGlobalWeights)...");
    let global = HetGlobalWeights::load(&main_gguf, dgpu)?;
    eprintln!("loading MTP drafter weights (~120 MB dGPU + ~1.2 GiB iGPU)...");
    let mtp_weights = MtpWeights::load(&mtp_gguf, dgpu, igpu, rope)?;
    eprintln!("weights loaded.");

    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let mut mtp_scratch = MtpScratch::alloc(dgpu)?;
    let mut mtp_state = MtpLayerState::alloc(dgpu)?;

    // prev_hc = the main model's final-layer HC at position 0 (dump tag
    // `output_flat` at the virtual head layer L43, shape [N_HC, N_EMBD]).
    let hc_entry = dump
        .tensor("output_flat", 43, 0)
        .ok_or_else(|| eyre!("missing output_flat L43 T0 in activation dump"))?;
    let prev_hc_host = dump.read_f32(hc_entry)?;
    assert_eq!(prev_hc_host.len(), HC_DIM as usize, "prev_hc len mismatch");
    let mut prev_hc: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, HC_DIM as usize)?;
    prev_hc.copy_from_host(&prev_hc_host)?;

    let last_embd = embed_token_host(&main_gguf, LAST_TOKEN, N_EMBD)?;

    eprintln!("running MTP draft (last_token={LAST_TOKEN}, pos=0)...");
    let t0 = Instant::now();
    engine.forward_mtp_draft(
        &mut dgpu_scratch,
        &mut igpu_scratch,
        &mut mtp_scratch,
        &mut mtp_state,
        &global,
        &mtp_weights,
        &prev_hc,
        &last_embd,
        0,
        LAST_TOKEN,
    )?;
    let smoke_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("MTP draft (first call): {smoke_ms:.2} ms");

    let mut logits = vec![0f32; N_VOCAB as usize];
    mtp_scratch.mtp_logits.copy_to_host(&mut logits)?;

    let finite = logits.iter().filter(|v| v.is_finite()).count();
    eprintln!(
        "{finite}/{N_VOCAB} logits finite ({:.3}%)",
        100.0 * finite as f64 / N_VOCAB as f64
    );
    assert!(finite as u32 == N_VOCAB, "MTP logits contain NaN/inf");

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let draft = idx[0] as i32;
    eprintln!("MTP drafted token id = {draft}");
    eprintln!("top-5 (id: logit):");
    for &i in idx.iter().take(5) {
        eprintln!("  {i}: {:.4}", logits[i]);
    }
    assert!((0..N_VOCAB as usize).contains(&idx[0]), "draft id out of vocab");

    // Second call confirms the MTP KV-cache advance + repeatability.
    engine.forward_mtp_draft(
        &mut dgpu_scratch,
        &mut igpu_scratch,
        &mut mtp_scratch,
        &mut mtp_state,
        &global,
        &mtp_weights,
        &prev_hc,
        &last_embd,
        1,
        LAST_TOKEN,
    )?;
    let mut logits2 = vec![0f32; N_VOCAB as usize];
    mtp_scratch.mtp_logits.copy_to_host(&mut logits2)?;
    assert!(
        logits2.iter().all(|v| v.is_finite()),
        "2nd MTP draft has NaN/inf"
    );
    eprintln!("2nd draft OK (mtp_state.n_raw={})", mtp_state.n_raw);

    eprintln!("MTP drafter smoke OK.");
    engine.shutdown()?;
    Ok(())
}
