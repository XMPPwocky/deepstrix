//! RMSNorm oracle test — validates our HIP `rms_norm_weighted` kernel
//! against the M2 activation dump produced by ds4-dump-activations.
//!
//! For each (layer, token) position in the dump, load the matching
//! input + weight + expected-output tensors, run our kernel on the
//! same inputs, and assert `max_abs_diff < THRESHOLD`.
//!
//! Validates both canonical RMSNorm sites in V4 Flash:
//!   1. `attn_cur`        → `attn_input_norm`   (weight: `attn_norm`)
//!   2. `ffn_cur`         → `ffn_input_norm`    (weight: `ffn_norm`)
//!
//! Marked `#[ignore]` because it requires the activation dump on disk
//! plus a working HIP device. Run via:
//!
//!   nix develop -c cargo test --release -p v4flash-kernels -- --ignored --nocapture
//!
//! Test is single-threaded — uses `--test-threads=1` implicitly via
//! per-test device initialization; concurrent device-buffer alloc on
//! the same device is fine, but pretty-print is cleaner serial.

use std::path::PathBuf;

use color_eyre::eyre;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{oracle::ActivationDump, RmsNorm};

/// Per-element RMSNorm matches ds4's CPU implementation to better than
/// this absolute tolerance. ds4's CPU rms_norm_weight accumulates in
/// double; our HIP kernel uses double per-thread partials with a single
/// f32 shared-mem tree reduction. Final scale is 1/sqrt(mean(x²)+eps),
/// then per-element f32 multiply. Expected drift is a few f32 ULPs.
const RMS_NORM_THRESHOLD: f32 = 1.0e-4;

/// V4 Flash dimensions, taken directly from the GGUF metadata
/// (deepseek4.embedding_length, deepseek4.attention.layer_norm_rms_epsilon).
const N_EMBD: u32 = 4096;
const N_LAYER: i32 = 43;
const RMS_EPS: f32 = 1.0e-6;

fn dump_dir() -> PathBuf {
    std::env::var("DEEPSTRIX_DUMP_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("reference/v4flash-cpu-activations")
    })
}

fn pick_device() -> eyre::Result<Device> {
    // Prefer the iGPU (gfx1151, device 1 on this box; see
    // memory/project-hardware-setup). Fall back to whatever's first.
    let devices = Device::all()?;
    for d in &devices {
        let arch = d.properties()?.gcn_arch_name;
        if arch.starts_with("gfx1151") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre::eyre!("no HIP devices"))
}

#[derive(Debug, Default, Clone, Copy)]
struct DiffStats {
    max_abs: f32,
    sum_abs: f64,
    count: usize,
}

impl DiffStats {
    fn update(&mut self, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
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

#[test]
#[ignore]
fn rms_norm_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    eprintln!(
        "loaded dump: {} tensors, prompt_len={}, n_logit_rows={}",
        dump.len(),
        dump.prompt_len,
        dump.n_logit_rows,
    );
    let n_tokens = dump.n_logit_rows as i32; // 51 = 7 prefill + 44 generated (or fewer if EOS)

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = RmsNorm::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // One buffer per role, reused across all (layer, token).
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_w: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;

    // Two RMSNorm sites in V4 Flash: attn and ffn. Both use n_embd=4096.
    let sites: &[(&str, &str, &str)] = &[
        ("attn_cur", "attn_input_norm", "attn_norm"),
        ("ffn_cur", "ffn_input_norm", "ffn_norm"),
    ];

    let mut overall_stats = DiffStats::default();
    let mut worst_layer = -1;
    let mut worst_token = -1;
    let mut worst_site = "";

    for &(in_tag, out_tag, w_tag) in sites {
        let mut per_site = DiffStats::default();

        for layer in 0..N_LAYER {
            // Weights are deduped — one per layer.
            let weight_entry = dump
                .weight(w_tag, layer)
                .ok_or_else(|| eyre::eyre!("missing weight {w_tag} for L{layer}"))?;
            let weight = dump.read_f32(weight_entry)?;
            assert_eq!(weight.len(), N_EMBD as usize);
            d_w.copy_from_host(&weight)?;

            for token in 0..n_tokens {
                let in_entry = match dump.tensor(in_tag, layer, token) {
                    Some(e) => e,
                    None => continue, // some token positions may be EOS-skipped
                };
                let out_entry = dump
                    .tensor(out_tag, layer, token)
                    .ok_or_else(|| eyre::eyre!("missing {out_tag} for L{layer} T{token}"))?;

                let x_host = dump.read_f32(in_entry)?;
                let expected = dump.read_f32(out_entry)?;
                assert_eq!(x_host.len(), N_EMBD as usize);
                assert_eq!(expected.len(), N_EMBD as usize);

                d_x.copy_from_host(&x_host)?;
                kernel.launch_weighted(&stream, &mut d_out, &d_x, &d_w, N_EMBD, RMS_EPS)?;
                stream.synchronize()?;

                let mut got = vec![0f32; N_EMBD as usize];
                d_out.copy_to_host(&mut got)?;

                let prev_max = per_site.max_abs;
                per_site.update(&got, &expected);
                overall_stats.update(&got, &expected);
                if per_site.max_abs > prev_max {
                    worst_layer = layer;
                    worst_token = token;
                    worst_site = in_tag;
                }
            }
        }

        eprintln!(
            "site {in_tag}→{out_tag}: max_abs_diff={:.3e}, mean_abs_diff={:.3e}, n_compared={}",
            per_site.max_abs,
            per_site.mean_abs(),
            per_site.count,
        );
    }

    eprintln!(
        "OVERALL: max_abs_diff={:.3e}, mean_abs_diff={:.3e}, worst at site={} L{} T{}",
        overall_stats.max_abs,
        overall_stats.mean_abs(),
        worst_site,
        worst_layer,
        worst_token,
    );

    assert!(
        overall_stats.max_abs < RMS_NORM_THRESHOLD,
        "max_abs_diff {:.3e} exceeds threshold {:.3e}",
        overall_stats.max_abs,
        RMS_NORM_THRESHOLD
    );

    Ok(())
}
