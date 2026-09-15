//! DSpark drafter loads onto the device with the right geometry.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --features v41 --test mtp_load -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::hf_v41::{MTP_N_EXPERT, MTP_STAGES};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::N_EMBD;
use v4flash_kernels::het::weights::MtpWeights;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

#[test]
#[ignore = "needs the V4.1 checkpoint and a GPU"]
fn drafter_loads() {
    install_panic_handler().ok();
    let model = std::env::var("V41_MODEL").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap_or_default())
    });
    let hf = V41HfWeights::open(&model, None).expect("open checkpoint");
    let dev = pick_igpu().expect("igpu");

    let t0 = std::time::Instant::now();
    let w = MtpWeights::load(&hf, dev, 40).expect("load drafter");
    let dt = t0.elapsed().as_secs_f64();

    assert_eq!(w.layers.len(), MTP_STAGES, "stage count");
    let mut total = 0usize;
    for (i, l) in w.layers.iter().enumerate() {
        assert_eq!(l.routed.n_slots, MTP_N_EXPERT as u32, "layer {i}: expert slots");
        // Every expert must be the same size, and the buffer an exact multiple.
        for (name, bpe, buf) in [
            ("gate", l.routed.gate_bytes_per_expert, l.routed.gate.buffer.len()),
            ("up", l.routed.up_bytes_per_expert, l.routed.up.buffer.len()),
            ("down", l.routed.down_bytes_per_expert, l.routed.down.buffer.len()),
        ] {
            assert!(bpe > 0, "layer {i} {name}: zero bytes/expert");
            assert_eq!(
                bpe * MTP_N_EXPERT,
                buf,
                "layer {i} {name}: {bpe} x {MTP_N_EXPERT} != buffer {buf}"
            );
            total += buf;
        }
        assert_eq!(l.attn_norm.len(), N_EMBD as usize, "layer {i}: attn_norm");
        assert_eq!(l.ffn_norm.len(), N_EMBD as usize, "layer {i}: ffn_norm");
        assert_eq!(l.exp_probs_b.len(), MTP_N_EXPERT, "layer {i}: router bias width");
        total += l.shared.gate.buffer.len() + l.shared.up.buffer.len() + l.shared.down.buffer.len();
        total += l.attn_q_a.buffer.len() + l.attn_q_b.buffer.len() + l.attn_kv.buffer.len();
        total += l.attn_output_a.buffer.len() + l.attn_output_b.buffer.len();
    }
    assert_eq!(w.main_norm.len(), N_EMBD as usize, "main_norm");
    assert_eq!(w.norm.len(), N_EMBD as usize, "exit norm");
    total += w.main_proj.buffer.len();

    println!(
        "drafter loaded in {dt:.1} s: {} layers, {MTP_N_EXPERT} experts each, {:.2} GB on device",
        w.layers.len(),
        total as f64 / 1e9
    );
    println!(
        "  bytes/expert: gate {} up {} down {}",
        w.layers[0].routed.gate_bytes_per_expert,
        w.layers[0].routed.up_bytes_per_expert,
        w.layers[0].routed.down_bytes_per_expert
    );
}
