//! Hash-gate router oracle (L0-L2). Per-token expert selection is a
//! direct lookup from `ffn_gate_tid2eid[token_id * 6 + slot]`; weights
//! are `probs[selected[i]] / sum × 1.5` where
//! `probs[i] = sqrt(softplus(matvec_f16(ffn_gate_inp, ffn_input_norm)))`.
//!
//! Mirrors ds4's `layer_hash_selected_experts` (ds4.c:5209) +
//! `layer_hash_router_weights_one` (ds4.c:5260).
//!
//! Validation: exact i32 match on `expert_selected`; f32 close-to-ULP
//! on `expert_weight_out`.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use color_eyre::eyre::WrapErr;
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{weights, oracle::ActivationDump, oracle::Dtype, F16Matvec};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_EMBD: u32 = 4096;
const N_EXPERT: u32 = 256;
const N_EXPERT_USED: u32 = 6;
const EXPERT_WEIGHT_SCALE: f32 = 1.5;
const HASH_LAYERS: &[i32] = &[0, 1, 2];
const WEIGHT_THRESHOLD: f32 = 5.0e-5;

const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];

/// Build the per-position token sequence from `logits.f32`. Position 0..6
/// is the prompt; positions 7..56 are argmax of the corresponding logit
/// rows (greedy decode, mirroring what the dumper fed back to ds4).
fn build_token_sequence(dump: &ActivationDump) -> eyre::Result<Vec<i32>> {
    let n_tokens = dump.n_logit_rows as usize + PROMPT_TOKENS.len() - 1;
    let vocab = dump.vocab_size;
    let logits_path = dump.root().join("logits.f32");
    let bytes = std::fs::read(&logits_path)?;
    let mut tokens = vec![0i32; n_tokens];
    for (i, &t) in PROMPT_TOKENS.iter().enumerate() {
        tokens[i] = t;
    }
    for row in 0..dump.n_logit_rows.saturating_sub(1) {
        let off = row * vocab * 4;
        let mut best_idx = 0i32;
        let mut best_val = f32::NEG_INFINITY;
        for j in 0..vocab {
            let b = &bytes[off + j * 4..off + (j + 1) * 4];
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if v > best_val {
                best_val = v;
                best_idx = j as i32;
            }
        }
        tokens[PROMPT_TOKENS.len() + row] = best_idx;
    }
    Ok(tokens)
}

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

fn softplus_stable(x: f32) -> f32 {
    // log1p(exp(x)) but stable for large/small x.
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

#[test]
#[ignore]
fn hash_gate_router_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let n_tokens = dump.n_logit_rows as i32;
    let token_seq = build_token_sequence(&dump)?;
    eprintln!(
        "token sequence ({} positions): first 12 = {:?}",
        token_seq.len(),
        &token_seq[..12.min(token_seq.len())]
    );

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let matvec = F16Matvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EXPERT as usize)?;
    let mut logits = vec![0f32; N_EXPERT as usize];

    let mut weight_max = 0f32;
    let mut weight_sum_abs = 0f64;
    let mut weight_count: usize = 0;
    let mut select_mismatches = 0usize;
    let mut total_select = 0usize;

    for &layer in HASH_LAYERS {
        // Load ffn_gate_inp (F16 [4096, 256]) and ffn_gate_tid2eid (I32 [6, vocab]).
        let gate_inp = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            device.id,
        )?;
        let tid2eid_tensor = gguf
            .gguf()
            .tensor(&format!("blk.{layer}.ffn_gate_tid2eid.weight"))
            .ok_or_else(|| eyre!("missing ffn_gate_tid2eid for L{layer}"))?;
        if tid2eid_tensor.dtype != GgufType::I32 {
            return Err(eyre!(
                "ffn_gate_tid2eid dtype {:?}, expected I32",
                tid2eid_tensor.dtype
            ));
        }
        let tid2eid_bytes = gguf
            .read_tensor(tid2eid_tensor).wrap_err("tid2eid bytes missing")?;
        // Shape: [n_used=6, vocab]; flat layout tid2eid[token * n_used + slot].
        let tid2eid: Vec<i32> = tid2eid_bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for token in 0..n_tokens {
            let token_id = token_seq[token as usize];

            // Hash-gate selection: direct table lookup.
            let mut selected = [0i32; N_EXPERT_USED as usize];
            for i in 0..N_EXPERT_USED as usize {
                selected[i] = tid2eid[(token_id as usize) * (N_EXPERT_USED as usize) + i];
            }

            // Compare to dumped expert_selected.
            let sel_entry = dump
                .tensor("expert_selected", layer, token)
                .ok_or_else(|| eyre!("missing expert_selected at L{layer} T{token}"))?;
            if sel_entry.dtype != Dtype::I32 {
                return Err(eyre!("expert_selected dtype != I32"));
            }
            let sel_bytes = dump.read_bytes(sel_entry)?;
            let expected_selected: Vec<i32> = sel_bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            for (s, e) in selected.iter().zip(expected_selected.iter()) {
                total_select += 1;
                if s != e {
                    select_mismatches += 1;
                }
            }

            // Router probs: matvec → softplus → sqrt.
            let x_entry = dump
                .tensor("ffn_input_norm", layer, token)
                .ok_or_else(|| eyre!("missing ffn_input_norm at L{layer} T{token}"))?;
            let x_host = dump.read_f32(x_entry)?;
            d_x.copy_from_host(&x_host)?;
            matvec.matvec(&stream, &mut d_logits, &gate_inp.buffer, &d_x, N_EXPERT, N_EMBD)?;
            stream.synchronize()?;
            d_logits.copy_to_host(&mut logits)?;

            let mut probs_sel = [0f32; N_EXPERT_USED as usize];
            let mut sum = 0f32;
            for i in 0..N_EXPERT_USED as usize {
                let p = softplus_stable(logits[selected[i] as usize]).sqrt();
                probs_sel[i] = p;
                sum += p;
            }
            if sum < 6.103515625e-5 {
                sum = 6.103515625e-5;
            }
            for i in 0..N_EXPERT_USED as usize {
                probs_sel[i] = probs_sel[i] / sum * EXPERT_WEIGHT_SCALE;
            }

            // Compare to dumped expert_weight_out.
            let w_entry = dump
                .tensor("expert_weight_out", layer, token)
                .ok_or_else(|| eyre!("missing expert_weight_out at L{layer} T{token}"))?;
            let expected_w = dump.read_f32(w_entry)?;
            for (g, e) in probs_sel.iter().zip(expected_w.iter()) {
                let d = (g - e).abs();
                if d > weight_max {
                    weight_max = d;
                }
                weight_sum_abs += d as f64;
                weight_count += 1;
            }
        }

        drop(gate_inp);
    }

    let weight_mean = weight_sum_abs / weight_count.max(1) as f64;
    eprintln!(
        "selection: {select_mismatches} / {total_select} mismatches; weights: max_abs={weight_max:.3e}, mean={weight_mean:.3e}, n={weight_count}",
    );

    assert_eq!(select_mismatches, 0, "hash-gate selection had mismatches");
    assert!(
        weight_max < WEIGHT_THRESHOLD,
        "weight max_abs_diff {weight_max:.3e} exceeds threshold {WEIGHT_THRESHOLD:.3e}"
    );

    Ok(())
}
