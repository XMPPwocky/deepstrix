//! Full indexer pipeline oracle — produces `comp_allowed` mask and
//! validates against ds4's `comp_allowed_mask` from the dump.
//!
//! For our 57-token M1 prompt, `n_comp ≤ 14 < INDEXER_TOP_K=512` always,
//! so ds4 (and our pipeline) takes the early-exit branch: mask is all-1s.
//! This test confirms that branch correctly produces all-1s of length
//! n_comp.
//!
//! For longer prompts (n_comp > 512) the pipeline would run:
//!   1. f16_matvec(indexer.attn_q_b × qr_norm) → indexer_q [64, 128]
//!   2. rope_tail forward (n_head=64, head_dim=128, n_rot=64, pos)
//!   3. f16_matvec(indexer.proj × attn_norm) → head_weights [64]
//!   4. Scale head_weights *= 1/sqrt(head_dim * n_head)
//!   5. indexer_score → scores [n_comp]
//!   6. Top-K = 512 greedy selection → bool mask
//!
//! That code path is unreachable in our prompt; full validation in M11.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::{oracle::ActivationDump, oracle::Dtype, INDEXER_TOP_K};

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
}

fn pick_device() -> eyre::Result<Device> {
    let devices = Device::all()?;
    for d in &devices {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre!("no HIP devices"))
}

/// Production-shaped pipeline: early-exit when n_comp ≤ top_k, otherwise
/// would run scoring + top-K (not exercised in this prompt).
fn indexer_run_mask(n_comp: u32) -> Vec<i32> {
    if n_comp == 0 {
        return Vec::new();
    }
    if n_comp <= INDEXER_TOP_K {
        return vec![1i32; n_comp as usize];
    }
    // Real scoring + top-K would happen here in M11+.
    unimplemented!("M7 pipeline only handles the early-permit branch (n_comp ≤ {INDEXER_TOP_K})");
}

#[test]
#[ignore]
fn indexer_pipeline_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let mut total_compared: usize = 0;
    let mut mismatches: usize = 0;
    let mut tested_layers = 0;

    // Walk ratio==4 layers (where ds4 emits comp_allowed_mask).
    for layer in (2..=42).step_by(2) {
        let mut n_comp: u32 = 0;
        let mut layer_compared = 0;
        for token in 0..n_tokens {
            // Track compressor pushes (every 4 tokens starting at T=3).
            if dump.tensor("comp_kv_row", layer, token).is_some() {
                n_comp += 1;
            }
            if n_comp == 0 {
                continue;
            }
            let mask_entry = match dump.tensor("comp_allowed_mask", layer, token) {
                Some(e) => e,
                None => continue,
            };
            if mask_entry.dtype != Dtype::I32 {
                return Err(eyre!(
                    "comp_allowed_mask at L{layer} T{token} dtype {:?}, expected I32",
                    mask_entry.dtype
                ));
            }
            let bytes = dump.read_bytes(mask_entry)?;
            if bytes.len() != (n_comp as usize) * 4 {
                return Err(eyre!(
                    "comp_allowed_mask L{layer} T{token}: {} bytes, expected {} (n_comp={n_comp})",
                    bytes.len(),
                    (n_comp as usize) * 4
                ));
            }
            let expected: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let got = indexer_run_mask(n_comp);

            assert_eq!(got.len(), expected.len(), "mask length mismatch at L{layer} T{token}");
            for (g, e) in got.iter().zip(expected.iter()) {
                total_compared += 1;
                layer_compared += 1;
                if g != e {
                    mismatches += 1;
                }
            }
        }
        if layer_compared > 0 {
            tested_layers += 1;
        }
    }

    eprintln!(
        "OVERALL: total compared = {total_compared} mask entries across {tested_layers} layers; mismatches = {mismatches}"
    );
    assert_eq!(mismatches, 0, "indexer pipeline mask had {mismatches} mismatches");
    assert!(total_compared > 0, "no comp_allowed_mask entries found — dump corrupt?");

    Ok(())
}
