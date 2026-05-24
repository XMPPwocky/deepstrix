//! `rms_norm_no_weight` oracle test — validates the per-row no-weight
//! RMSNorm against ds4's `head_rms_norm_inplace` (the Q post-q_b
//! normalisation in the LoRA chain).
//!
//! Input  tag: `q_b_out`        (32768 f32 = 64 heads × 512 head_dim)
//! Output tag: `q_head_normed`  (32768 f32, in-place RMSNorm per head)
//!
//! Each of the 64 heads is RMS-normed independently; one workgroup
//! per head, 256 threads per workgroup. Pass: `max_abs_diff < 1e-4`.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!                              --test rms_norm_no_weight -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{ActivationDump, RmsNormNoWeight};

const N_HEAD: u32 = 64;
const N_HEAD_DIM: u32 = 512;
const N_FLAT: u32 = N_HEAD * N_HEAD_DIM; // 32768
const N_LAYER: i32 = 43;
const RMS_EPS: f32 = 1.0e-6;
const THRESHOLD: f32 = 1.0e-4;

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

#[test]
#[ignore]
fn rms_norm_no_weight_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let n_tokens = dump.n_logit_rows as i32;
    eprintln!(
        "dump: n_tensors={}, n_tokens={}, n_layers={}",
        dump.len(),
        n_tokens,
        N_LAYER,
    );

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = RmsNormNoWeight::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FLAT as usize)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_FLAT as usize)?;
    let mut got = vec![0f32; N_FLAT as usize];

    let mut stats = DiffStats::default();
    let mut worst_l = -1i32;
    let mut worst_t = -1i32;

    for layer in 0..N_LAYER {
        for token in 0..n_tokens {
            let in_entry = match dump.tensor("q_b_out", layer, token) {
                Some(e) => e,
                None => continue,
            };
            let out_entry = dump
                .tensor("q_head_normed", layer, token)
                .ok_or_else(|| eyre!("missing q_head_normed at L{layer} T{token}"))?;

            let x_host = dump.read_f32(in_entry)?;
            let expected = dump.read_f32(out_entry)?;
            assert_eq!(x_host.len(), N_FLAT as usize);
            assert_eq!(expected.len(), N_FLAT as usize);

            d_x.copy_from_host(&x_host)?;
            kernel.launch(&stream, &mut d_out, &d_x, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;

            let prev = stats.max_abs;
            stats.update(&got, &expected);
            if stats.max_abs > prev {
                worst_l = layer;
                worst_t = token;
            }
        }
    }

    eprintln!(
        "OVERALL: max_abs_diff={:.3e}, mean_abs_diff={:.3e}, n_compared={}, worst at L{} T{}",
        stats.max_abs,
        stats.mean_abs(),
        stats.count,
        worst_l,
        worst_t,
    );

    assert!(
        stats.max_abs < THRESHOLD,
        "max_abs_diff {:.3e} exceeds threshold {:.3e}",
        stats.max_abs,
        THRESHOLD
    );

    Ok(())
}
