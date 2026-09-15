//! DSpark gate 2: the engine's drafter ENTRY matches the Python reference.
//!
//! `main_x = mtp.0.main_norm(mtp.0.main_proj(main_hidden))`, where main_hidden is
//! cat(mean-over-hc(residual after layers 36/37/38)).
//!
//! Compared by cosine similarity, not bit-equality: the engine quantises
//! `main_proj` to Q8_0 while the reference keeps f32, so ~8-bit weight noise is
//! expected and correct. A structural error (wrong layers, wrong concat order,
//! missing norm) destroys the direction and shows up immediately.
//!
//! Generate the reference first:
//!   nix-shell -p python3Packages.{torch,numpy,safetensors,pillow,transformers} --run \
//!     "python3 scripts/v41_oracle/dump_mtp_ref.py ~/.cache/deepstrix/v41/agentic/main --pos 200"
//! Then:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --features v41 --test mtp_entry_parity -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::N_EMBD;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::mtp::MtpState;
use v4flash_kernels::het::weights::MtpWeights;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
#[ignore = "needs the checkpoint, a GPU, and a generated reference dump"]
fn entry_matches_reference() {
    install_panic_handler().ok();
    let home = std::env::var("HOME").unwrap_or_default();
    let refdir = std::env::var("MTP_REF")
        .unwrap_or_else(|_| format!("{home}/.cache/deepstrix/v41/agentic/main/mtp_ref"));
    let main_hidden = read_f32(&format!("{refdir}/main_hidden.bin"));
    let want = read_f32(&format!("{refdir}/main_x.bin"));
    assert_eq!(want.len(), N_EMBD as usize, "reference main_x width");

    let model = std::env::var("V41_MODEL")
        .unwrap_or_else(|_| format!("{home}/.cache/deepstrix/models/dsv4.1f"));
    let hf = V41HfWeights::open(&model, None).expect("open checkpoint");
    let dev = pick_igpu().expect("igpu");
    let arch = dev.properties().expect("props").gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch).expect("engine");
    let w = MtpWeights::load(&hf, dev, 40).expect("load drafter");
    let mut st = MtpState::alloc(dev.id).expect("alloc state");

    st.inject_main_hidden(&main_hidden).expect("inject");
    st.entry(&e, &e.compute, &w).expect("entry");
    e.compute.synchronize().expect("sync");

    let mut got = vec![0f32; N_EMBD as usize];
    st.x.copy_to_host(&mut got).expect("readback");

    let dot: f64 = got.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb);
    let rel = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        .sqrt()
        / nb;

    println!("entry vs reference: cos {cos:.6}, rel L2 {rel:.4}, |engine| {na:.4}, |ref| {nb:.4}");
    println!("  first 6 engine: {:?}", &got[..6]);
    println!("  first 6 ref   : {:?}", &want[..6]);
    assert!(
        cos > 0.999,
        "cosine {cos:.6} — that is a STRUCTURAL mismatch, not Q8_0 noise \
         (wrong source layers, concat order, or a missing norm)"
    );
    assert!(rel < 0.05, "relative L2 {rel:.4} too large for Q8_0 weight noise");
}
