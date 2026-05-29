//! Learned router oracle (L3-L42). For each token:
//!   1. probs[i] = sqrt(softplus(matvec_f16(ffn_gate_inp, ffn_input_norm)[i]))
//!   2. selection[i] = probs[i] + bias[i]  (bias optional, F32 [256])
//!   3. selected = topk_desc(selection, 6)  — naive insertion sort
//!   4. weight[i] = probs[selected[i]] / sum × 1.5
//!
//! Mirrors ds4's `layer_topk_selected_experts_from_probs` (ds4.c:5307).

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
const N_EXPERT_USED: usize = 6;
const EXPERT_WEIGHT_SCALE: f32 = 1.5;
const WEIGHT_THRESHOLD: f32 = 5.0e-5;

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
    if x > 20.0 { x }
    else if x < -20.0 { x.exp() }
    else { (1.0f32 + x.exp()).ln() }
}

fn topk_desc(score: &[f32], k: usize) -> [i32; 6] {
    // Mirror ds4's naive insertion sort (ds4.c:5272).
    let mut idx = [-1i32; 6];
    for i in 0..score.len() {
        for j in 0..k {
            if idx[j] < 0 || score[i] > score[idx[j] as usize] {
                for m in (j + 1..k).rev() {
                    idx[m] = idx[m - 1];
                }
                idx[j] = i as i32;
                break;
            }
        }
    }
    idx
}

#[test]
#[ignore]
fn learned_router_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let n_tokens = dump.n_logit_rows as i32;

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

    for layer in 3..43 {
        let gate_inp = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            device.id,
        )?;

        // Optional bias tensor.
        let bias_opt: Option<Vec<f32>> = match gguf
            .gguf()
            .tensor(&format!("blk.{layer}.exp_probs_b.bias"))
        {
            Some(t) if t.dtype == GgufType::F32 => {
                let bytes = gguf.read_tensor(t).wrap_err("bias bytes")?;
                Some(
                    bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            }
            _ => None,
        };

        for token in 0..n_tokens {
            let x_entry = dump
                .tensor("ffn_input_norm", layer, token)
                .ok_or_else(|| eyre!("missing ffn_input_norm at L{layer} T{token}"))?;
            let x_host = dump.read_f32(x_entry)?;
            d_x.copy_from_host(&x_host)?;
            matvec.matvec(&stream, &mut d_logits, &gate_inp.buffer, &d_x, N_EXPERT, N_EMBD)?;
            stream.synchronize()?;
            d_logits.copy_to_host(&mut logits)?;

            // probs = sqrt(softplus(logits))
            let mut probs = vec![0f32; N_EXPERT as usize];
            for i in 0..N_EXPERT as usize {
                probs[i] = softplus_stable(logits[i]).sqrt();
            }

            // selection = probs + bias (if present)
            let mut selection = probs.clone();
            if let Some(ref b) = bias_opt {
                for i in 0..N_EXPERT as usize {
                    selection[i] += b[i];
                }
            }

            let selected = topk_desc(&selection, N_EXPERT_USED);

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

            // Weights: probs[selected] / sum * 1.5.
            let mut w = [0f32; N_EXPERT_USED];
            let mut sum = 0f32;
            for i in 0..N_EXPERT_USED {
                let p = probs[selected[i] as usize];
                w[i] = p;
                sum += p;
            }
            if sum < 6.103515625e-5 {
                sum = 6.103515625e-5;
            }
            for i in 0..N_EXPERT_USED {
                w[i] = w[i] / sum * EXPERT_WEIGHT_SCALE;
            }

            let w_entry = dump
                .tensor("expert_weight_out", layer, token)
                .ok_or_else(|| eyre!("missing expert_weight_out at L{layer} T{token}"))?;
            let expected_w = dump.read_f32(w_entry)?;
            for (g, e) in w.iter().zip(expected_w.iter()) {
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

    assert_eq!(select_mismatches, 0, "learned router selection had mismatches");
    assert!(
        weight_max < WEIGHT_THRESHOLD,
        "weight max_abs_diff {weight_max:.3e} exceeds threshold {WEIGHT_THRESHOLD:.3e}"
    );

    Ok(())
}
