//! Q8_0 matvec oracle test — validates our HIP `q8_0_quantize_f32` +
//! `q8_0_gemv_warp8` against ds4 CPU output.
//!
//! Setup:
//! - input: `output_norm` at synthetic layer 43, token T = 6+k (where k is
//!   the logits.f32 row index 0..50). See PHASE1_REFERENCE.md for the
//!   logit-row ↔ output_norm-token mapping rationale.
//! - weight: `output.weight` from the GGUF (Q8_0, [4096, 129280])
//! - expected: the matching row of `logits.f32`
//!
//! Pass criterion:
//!  - `argmax(got) == argmax(expected)` for every row (the *real* gate)
//!  - `max_abs_diff < 5e-3` as a backup numeric sanity bound
//!
//! Threshold rationale: Q8_0 introduces ~0.4% per-weight noise via int8 + f16
//! scale; the 4096-dim dot product partially averages it. Logit magnitudes
//! span ~ -30..30 in practice; absolute diffs of a few millis are expected.
//! The argmax-match check is the production-relevant gate (if we don't pick
//! the same greedy token as ds4, we have a real bug).
//!
//! Run via:
//!   nix develop -c cargo test --release -p v4flash-kernels --test q8_0_matvec \
//!                 -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{weights, ActivationDump, Q8_0Matvec};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_EMBD: u32 = 4096; // V4 Flash hidden dim
const N_VOCAB: u32 = 129_280;
const PROMPT_LEN_PREFILL: i32 = 7; // M1 reference prompt: "DeepSeek-V4 Flash is"
const Q8_0_THRESHOLD: f32 = 5.0e-3;

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
}

fn pick_device() -> eyre::Result<Device> {
    let devices = Device::all()?;
    devices
        .iter()
        .find(|d| {
            d.properties()
                .map(|p| p.gcn_arch_name.starts_with("gfx1151"))
                .unwrap_or(false)
        })
        .copied()
        .or_else(|| devices.first().copied())
        .ok_or_else(|| eyre!("no HIP devices"))
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > x[best] {
            best = i;
        }
    }
    best
}

#[derive(Default)]
struct DiffStats {
    max_abs: f32,
    sum_abs: f64,
    count: usize,
}
impl DiffStats {
    fn update(&mut self, a: &[f32], b: &[f32]) {
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > self.max_abs {
                self.max_abs = d;
            }
            self.sum_abs += d as f64;
            self.count += 1;
        }
    }
    fn mean_abs(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_abs / self.count as f64
        }
    }
}

fn read_logits_rows(path: &std::path::Path, n_rows: usize, vocab: usize) -> eyre::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    let expected = n_rows * vocab * 4;
    if bytes.len() != expected {
        return Err(eyre!(
            "logits.f32 size: have {}, expected {} (n_rows={}, vocab={})",
            bytes.len(),
            expected,
            n_rows,
            vocab
        ));
    }
    let mut out = vec![0f32; n_rows * vocab];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

#[test]
#[ignore]
fn q8_0_matvec_output_projection_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    // Open dump + GGUF + device.
    let dump = ActivationDump::open(dump_dir())?;
    let n_logit_rows = dump.n_logit_rows; // 51 expected
    eprintln!(
        "dump loaded: n_tensors={}, n_logit_rows={}, vocab={}",
        dump.len(),
        n_logit_rows,
        dump.vocab_size,
    );

    let gguf = MappedGguf::open(MODEL_PATH)?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    // Reference logits as one flat f32 buffer.
    let logits_path = dump.root().join("logits.f32");
    let logits = read_logits_rows(&logits_path, n_logit_rows, N_VOCAB as usize)?;

    // Load Q8_0 output projection weight onto device (~537 MB).
    let weight = weights::load_to_device(&gguf, "output.weight", device.id)?;
    eprintln!(
        "loaded output.weight: dtype={:?}, shape={:?}, bytes={}",
        weight.dtype,
        weight.shape,
        weight.buffer.byte_len(),
    );
    assert_eq!(weight.shape, vec![N_EMBD as u64, N_VOCAB as u64]);

    let kernel = Q8_0Matvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_xq: DeviceBuffer<i8> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_xscale: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, (N_EMBD / 32) as usize)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;

    let mut stats = DiffStats::default();
    let mut argmax_matches = 0usize;
    let mut got = vec![0f32; N_VOCAB as usize];

    for row in 0..n_logit_rows {
        // Logit row k corresponds to output_norm at token position (prompt_len-1 + k)
        // i.e. T6, T7, ..., T56 for k=0..50.
        let token = (PROMPT_LEN_PREFILL - 1) + row as i32;
        let onorm_entry = dump
            .tensor("output_norm", 43, token)
            .ok_or_else(|| eyre!("missing output_norm at L43 T{token}"))?;
        let x_host = dump.read_f32(onorm_entry)?;
        assert_eq!(x_host.len(), N_EMBD as usize);

        d_x.copy_from_host(&x_host)?;
        kernel.quantize_input(&stream, &mut d_xq, &mut d_xscale, &d_x, N_EMBD)?;
        kernel.matvec(
            &stream,
            &mut d_out,
            &weight.buffer,
            &d_xq,
            &d_xscale,
            N_VOCAB,
            N_EMBD,
        )?;
        stream.synchronize()?;
        d_out.copy_to_host(&mut got)?;

        let expected =
            &logits[row * (N_VOCAB as usize)..(row + 1) * (N_VOCAB as usize)];

        stats.update(&got, expected);

        let got_argmax = argmax(&got);
        let exp_argmax = argmax(expected);
        if got_argmax == exp_argmax {
            argmax_matches += 1;
        } else {
            eprintln!(
                "  row {row}: argmax mismatch — got {} ({:.3}), expected {} ({:.3}) (our value @{}: {:.3})",
                got_argmax, got[got_argmax],
                exp_argmax, expected[exp_argmax],
                exp_argmax, got[exp_argmax],
            );
        }
    }

    eprintln!(
        "OVERALL: max_abs_diff={:.3e}, mean_abs_diff={:.3e}, argmax_match={}/{}",
        stats.max_abs,
        stats.mean_abs(),
        argmax_matches,
        n_logit_rows,
    );

    assert_eq!(
        argmax_matches, n_logit_rows,
        "argmax mismatched on at least one logit row — see logs above"
    );
    assert!(
        stats.max_abs < Q8_0_THRESHOLD,
        "max_abs_diff {:.3e} exceeds threshold {:.3e}",
        stats.max_abs,
        Q8_0_THRESHOLD
    );

    Ok(())
}
