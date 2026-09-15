//! Smoke test for the DSpark drafter's full three-layer pass.
//!
//! This is NOT a parity gate — `mtp_entry_parity` covers the entry, and the
//! real correctness gate for the rest is end-to-end acceptance rate against the
//! Python oracle's E = 4.94 at K=5. What this catches is the gross failure
//! modes of a freshly rewritten B=5 path: undersized buffers, kernel launch
//! failures, NaNs, and a stream that silently collapses to zeros or to a
//! constant.
//!
//!   cargo test -p v4flash-kernels --features v41 --test mtp_forward_smoke \
//!     -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{HC_DIM, N_EMBD, N_HC};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::mtp::{mtp_rope, MtpState, MTP_BLOCK};
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
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic stand-in for an embedding row, `HC_DIM` wide — the shape
/// `embed::embed_lookup` writes (it replicates the row across the N_HC copies).
/// Real rows come from the host-side `token_embd` lookup (M57); the drafter's
/// arithmetic does not care which values it gets, only their scale.
fn fake_row(seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..HC_DIM as usize)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.02
        })
        .collect()
}

fn stats(v: &[f32]) -> (f32, f32, f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let absmax = v.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    (mean, var.sqrt(), absmax)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn forward_runs_and_stays_finite() {
    install_panic_handler().ok();
    let home = std::env::var("HOME").unwrap_or_default();
    let refdir = std::env::var("MTP_REF")
        .unwrap_or_else(|_| format!("{home}/.cache/deepstrix/v41/agentic/main/mtp_ref"));
    let main_hidden = read_f32(&format!("{refdir}/main_hidden.bin"));

    let model = std::env::var("V41_MODEL")
        .unwrap_or_else(|_| format!("{home}/.cache/deepstrix/models/dsv4.1f"));
    let hf = V41HfWeights::open(&model, None).expect("open checkpoint");
    let dev = pick_igpu().expect("igpu");
    let arch = dev.properties().expect("props").gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch).expect("engine");
    let w = MtpWeights::load(&hf, dev, 40).expect("load drafter");
    let mut st = MtpState::alloc(dev.id).expect("alloc state");

    let rope = mtp_rope();

    let token_row = fake_row(7);
    let noise_row = fake_row(129);
    let hc = N_HC as usize * N_EMBD as usize;

    // Two positions: one below the window (ring still filling) and one above it
    // (ring wrapped and fully valid). They exercise different `n_valid`.
    let mut prev: Option<Vec<f32>> = None;
    for pos in [37u32, 900u32] {
        st.inject_main_hidden(&main_hidden).expect("inject");
        st.forward(&e, &e.compute, &w, &rope, pos, &token_row, &noise_row)
            .unwrap_or_else(|err| panic!("forward at pos {pos}: {err:?}"));
        e.compute.synchronize().expect("sync");

        let mut h = vec![0.0f32; MTP_BLOCK * hc];
        st.h.copy_to_host(&mut h).expect("read h");
        let (mean, std, absmax) = stats(&h);
        println!("pos {pos:4}: h mean {mean:+.6} std {std:.6} absmax {absmax:.4}");

        let mut ao = vec![0.0f32; MTP_BLOCK * N_EMBD as usize];
        st.attn_out.copy_to_host(&mut ao).expect("read attn_out");
        let (am, asd, aax) = stats(&ao);
        println!("           attn_out mean {am:+.6} std {asd:.6} absmax {aax:.4}");
        let mut fo = vec![0.0f32; MTP_BLOCK * N_EMBD as usize];
        st.ffn_out.copy_to_host(&mut fo).expect("read ffn_out");
        let (fm, fsd, fax) = stats(&fo);
        println!("           ffn_out  mean {fm:+.6} std {fsd:.6} absmax {fax:.4}");
        // Position dependence is the assertion that matters most here. Every
        // `*_batched_htiled_wmma*` attention kernel compiles its weighted sum
        // only for gfx1200/gfx1201, so on this iGPU one of them left `out`
        // untouched and the drafter still looked healthy — h was finite, varied
        // across the block, and identical at pos 37 and pos 900 because only
        // the MoE was contributing. Different positions must differ: they rope
        // at different angles and see a different `n_valid`.
        if let Some(p) = &prev {
            let d = p.iter().zip(&h).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!("           max|h(prev pos) - h(this pos)| = {d:.6}");
            assert!(d > 1e-3, "drafter output does not depend on position (max diff {d})");
        }
        prev = Some(h.clone());

        assert!(
            h.iter().all(|v| v.is_finite()),
            "pos {pos}: drafter produced non-finite values"
        );
        assert!(std > 1e-6, "pos {pos}: h collapsed to a constant (std {std})");
        assert!(absmax < 1e4, "pos {pos}: h blew up (absmax {absmax})");

        // The five block positions must not be identical: position 0 embeds the
        // real token and 1..4 the noise token, and attention is bidirectional
        // over the block, so every row still sees a different mix.
        let row0 = &h[..hc];
        let row3 = &h[3 * hc..4 * hc];
        let diff = row0
            .iter()
            .zip(row3)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("           max|row0 - row3| = {diff:.6}");
        assert!(diff > 1e-4, "pos {pos}: draft rows are identical (diff {diff})");
    }
}
