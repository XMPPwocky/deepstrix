//! Head→logits oracle. Validates the complete M10+M3 tail:
//!   layer_output_residual[L=42, T] → output_embd → output_norm → logits
//! against the canonical `logits.f32` file (one row per token, vocab=129280
//! f32s each). Per row: top-1 argmax must match dumped tokens.json, and
//! max-abs-diff per design doc §10 threshold.
//!
//! This is the final-step validation that any V4-Flash orchestrator must
//! pass once it produces correct L=42 outputs. Composes:
//!   rms_norm_no_weight (n=16384) → F16Matvec(output_hc_fn[16384,4]) →
//!   HcSigmoidBias → HcWeightedSum → RmsNorm(weighted, output_norm) →
//!   Q8_0Matvec(output[4096, vocab])

use std::fs;
use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use color_eyre::eyre::WrapErr;
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    weights, oracle::ActivationDump, F16Matvec, HcSigmoidBias, HcWeightedSum, Q8_0Matvec, RmsNorm,
    RmsNormNoWeight,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";

const N_EMBD: u32 = 4096;
const N_HC: u32 = 4;
const HC_DIM: u32 = N_EMBD * N_HC;
const N_VOCAB: u32 = 129280;
const LAST_LAYER: i32 = 42;
const PROMPT_LEN_PREFILL: i32 = 7; // "DeepSeek-V4 Flash is" BPE-tokenised
const RMS_EPS: f32 = 1.0e-6;
// Per design doc §10 — argmax must match; max-abs follows M3's Q8_0
// vocab projection threshold (5e-3 baseline, looser end-to-end).
const THRESHOLD_LOGIT: f32 = 5.0e-2;

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

fn load_f32_weight(
    gguf: &MappedGguf,
    name: &str,
    device_id: i32,
    expected_len: usize,
) -> eyre::Result<DeviceBuffer<f32>> {
    let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("tensor {name} missing"))?;
    if t.dtype != GgufType::F32 {
        return Err(eyre!("tensor {name} dtype {:?} != F32", t.dtype));
    }
    let bytes = gguf.read_tensor(t).wrap_err("{name} bytes missing")?;
    if bytes.len() != expected_len * 4 {
        return Err(eyre!(
            "{name}: have {} bytes, expected {}",
            bytes.len(),
            expected_len * 4
        ));
    }
    let mut v = vec![0f32; expected_len];
    for (i, c) in bytes.chunks_exact(4).enumerate() {
        v[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(device_id, expected_len)?;
    buf.copy_from_host(&v)?;
    Ok(buf)
}

fn argmax_f32(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

#[test]
#[ignore]
fn head_to_logits_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let n_tokens = dump.n_logit_rows as i32;
    assert_eq!(dump.vocab_size, N_VOCAB as usize);

    // Read the canonical logits.f32 file: n_tokens × vocab × f32.
    let logits_bytes = fs::read(dump_dir().join("logits.f32"))?;
    let expected_len = (n_tokens as usize) * (N_VOCAB as usize) * 4;
    if logits_bytes.len() != expected_len {
        return Err(eyre!(
            "logits.f32 size: have {}, expected {}",
            logits_bytes.len(),
            expected_len
        ));
    }

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!(
        "using device {} ({arch}); n_tokens={n_tokens}, vocab={}",
        device.id, N_VOCAB
    );

    let rms_nw = RmsNormNoWeight::for_arch(&arch)?;
    let f16 = F16Matvec::for_arch(&arch)?;
    let sig = HcSigmoidBias::for_arch(&arch)?;
    let ws = HcWeightedSum::for_arch(&arch)?;
    let rms_w = RmsNorm::for_arch(&arch)?;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Head weights.
    let w_fn = weights::load_to_device(&gguf, "output_hc_fn.weight", device.id)?;
    let w_scale = load_f32_weight(&gguf, "output_hc_scale.weight", device.id, 1)?;
    let w_base = load_f32_weight(&gguf, "output_hc_base.weight", device.id, N_HC as usize)?;
    let w_norm = load_f32_weight(
        &gguf,
        "output_norm.weight",
        device.id,
        N_EMBD as usize,
    )?;
    let w_vocab = weights::load_to_device(&gguf, "output.weight", device.id)?;
    if w_vocab.dtype != GgufType::Q8_0 {
        return Err(eyre!("output.weight dtype {:?} != Q8_0", w_vocab.dtype));
    }

    let mut d_inp: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_DIM as usize)?;
    let mut d_flat: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_DIM as usize)?;
    let mut d_pre: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HC as usize)?;
    let mut d_w: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HC as usize)?;
    let mut d_embd: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_norm: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_xq: DeviceBuffer<i8> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_xscale: DeviceBuffer<f32> = DeviceBuffer::new(device.id, (N_EMBD / 32) as usize)?;
    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;
    let mut got_logits = vec![0f32; N_VOCAB as usize];

    let mut max_abs: f32 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut count: u64 = 0;
    let mut argmax_match = 0i32;

    for row in 0..n_tokens {
        // Logit row k maps to dump token position (prompt_len - 1 + k):
        // T6 = first generated logit, T7..T56 the rest. Same convention
        // as the M3 q8_0_matvec oracle.
        let token = (PROMPT_LEN_PREFILL - 1) + row;
        let inp_entry = dump
            .tensor("layer_output_residual", LAST_LAYER, token)
            .ok_or_else(|| eyre!("missing layer_output_residual L{LAST_LAYER} T{token}"))?;
        d_inp.copy_from_host(&dump.read_f32(inp_entry)?)?;

        // Head chain (M10).
        rms_nw.launch(&stream, &mut d_flat, &d_inp, 1, HC_DIM, RMS_EPS)?;
        f16.matvec(&stream, &mut d_pre, &w_fn.buffer, &d_flat, N_HC, HC_DIM)?;
        sig.launch(&stream, &mut d_w, &d_pre, &w_scale, &w_base, N_HC)?;
        ws.launch(&stream, &mut d_embd, &d_inp, &d_w, N_EMBD, N_HC)?;

        // Output RMS norm with learned weight.
        rms_w.launch_weighted(&stream, &mut d_norm, &d_embd, &w_norm, N_EMBD, RMS_EPS)?;

        // Q8_0 vocab projection: quantize input, matvec.
        q8.quantize_input(&stream, &mut d_xq, &mut d_xscale, &d_norm, N_EMBD)?;
        q8.matvec(
            &stream,
            &mut d_logits,
            &w_vocab.buffer,
            &d_xq,
            &d_xscale,
            N_VOCAB,
            N_EMBD,
        )?;
        stream.synchronize()?;
        d_logits.copy_to_host(&mut got_logits)?;

        let row_off = (row as usize) * (N_VOCAB as usize) * 4;
        let mut expected = vec![0f32; N_VOCAB as usize];
        for (i, c) in logits_bytes[row_off..row_off + (N_VOCAB as usize) * 4]
            .chunks_exact(4)
            .enumerate()
        {
            expected[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }

        let mut row_max: f32 = 0.0;
        for (g, e) in got_logits.iter().zip(expected.iter()) {
            let d = (g - e).abs();
            if d > row_max {
                row_max = d;
            }
            sum_abs += d as f64;
            count += 1;
        }
        if row_max > max_abs {
            max_abs = row_max;
        }

        let g_arg = argmax_f32(&got_logits);
        let e_arg = argmax_f32(&expected);
        if g_arg == e_arg {
            argmax_match += 1;
        } else {
            eprintln!(
                "row{row} (T{token}): argmax got {g_arg} ({:.4}) vs expected {e_arg} ({:.4})",
                got_logits[g_arg], expected[e_arg]
            );
        }
    }

    let mean_abs = if count > 0 { sum_abs / count as f64 } else { 0.0 };
    eprintln!(
        "head_to_logits: max_abs={:.3e}, mean_abs={:.3e}, argmax_match={}/{}",
        max_abs, mean_abs, argmax_match, n_tokens
    );

    // Argmax gate is the production correctness criterion (design doc §10).
    assert_eq!(
        argmax_match, n_tokens,
        "argmax mismatch on {}/{} rows",
        n_tokens - argmax_match,
        n_tokens
    );
    // FP threshold is a regression signal, not the gate.
    assert!(
        max_abs < THRESHOLD_LOGIT,
        "max_abs {:.3e} >= threshold {:.3e}",
        max_abs,
        THRESHOLD_LOGIT
    );
    Ok(())
}
