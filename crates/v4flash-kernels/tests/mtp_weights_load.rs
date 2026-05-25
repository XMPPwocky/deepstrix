//! M40-P2.1 smoke test: load every MTP tensor from the MTP GGUF and
//! verify it lands on the expected device with the expected dtype/shape.
//! Doesn't run any kernels — just checks the loader.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::het::MtpWeights;
use v4flash_kernels::{ActivationDump, RopeParams};

const MTP_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf";

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

#[test]
#[ignore]
fn mtp_weights_load_smoke() -> eyre::Result<()> {
    install_panic_handler()?;
    let dump = ActivationDump::open(dump_dir())?;
    let mtp_gguf = MappedGguf::open(MTP_MODEL_PATH)?;
    let dgpu = pick_dgpu_device()?;
    let igpu = pick_igpu_device()?;

    // Reuse the main-model layer 0 RoPE params for MTP — same context window.
    let rope_entry = dump
        .weight("rope_params", 0)
        .ok_or_else(|| eyre!("missing rope_params L0"))?;
    let rope_floats = dump.read_f32(rope_entry)?;
    let n_ctx_orig = if rope_floats[2] != 0.0 {
        ROPE_ORIG_CTX
    } else {
        0
    };
    let rope = RopeParams::from_dump_blob(&rope_floats, n_ctx_orig)?;

    eprintln!("loading MTP weights from {MTP_MODEL_PATH}...");
    let mtp = MtpWeights::load(&mtp_gguf, dgpu, igpu, rope)?;
    eprintln!("MTP weights loaded.");

    // Spot-check expected device residency + sizes.
    use v4flash_kernels::forward::{HC_DIM, HC_MIX_DIM, N_EMBD, N_HC, N_HEAD, N_HEAD_DIM, N_LORA_Q};

    // dGPU-resident things
    assert_eq!(mtp.enorm.len(), N_EMBD as usize);
    assert_eq!(mtp.hnorm.len(), N_EMBD as usize);
    assert_eq!(mtp.norm.len(), N_EMBD as usize);
    assert_eq!(mtp.attn_norm.len(), N_EMBD as usize);
    assert_eq!(mtp.ffn_norm.len(), N_EMBD as usize);
    assert_eq!(mtp.q_a_norm.len(), N_LORA_Q as usize);
    assert_eq!(mtp.kv_a_norm.len(), N_HEAD_DIM as usize);
    assert_eq!(mtp.attn_sinks.len(), N_HEAD as usize);
    assert_eq!(mtp.hc_attn_scale.len(), 3);
    assert_eq!(mtp.hc_attn_base.len(), HC_MIX_DIM as usize);
    assert_eq!(mtp.hc_ffn_scale.len(), 3);
    assert_eq!(mtp.hc_ffn_base.len(), HC_MIX_DIM as usize);
    assert_eq!(mtp.hc_head_scale.len(), 1);
    assert_eq!(mtp.hc_head_base.len(), N_HC as usize);

    // F16-converted hc projections (16384 × 24 halfs = 786432 bytes).
    assert_eq!(
        mtp.hc_attn_fn.buffer.byte_len(),
        (HC_DIM * HC_MIX_DIM) as usize * 2
    );
    assert_eq!(
        mtp.hc_ffn_fn.buffer.byte_len(),
        (HC_DIM * HC_MIX_DIM) as usize * 2
    );
    assert_eq!(
        mtp.hc_head_fn.buffer.byte_len(),
        (HC_DIM * N_HC) as usize * 2
    );

    // Router learned-bias on dGPU (256 experts).
    assert_eq!(mtp.router_bias_dev.len(), 256);

    // Q4_K experts on iGPU — just check device_id, not size (Q4_K bytes
    // are 144/256 elements, total ~1.2 GB for one MoE block).
    assert_eq!(mtp.routed.gate_exps.buffer.device_id(), igpu.id);
    assert_eq!(mtp.routed.up_exps.buffer.device_id(), igpu.id);
    assert_eq!(mtp.routed.down_exps.buffer.device_id(), igpu.id);
    // Everything else should be on dGPU.
    assert_eq!(mtp.attn_q_a.buffer.device_id(), dgpu.id);
    assert_eq!(mtp.attn_q_b.buffer.device_id(), dgpu.id);
    assert_eq!(mtp.ffn_gate_shexp.buffer.device_id(), dgpu.id);
    assert_eq!(mtp.hc_head_fn.buffer.device_id(), dgpu.id);

    eprintln!("all MtpWeights sanity checks passed.");
    Ok(())
}
