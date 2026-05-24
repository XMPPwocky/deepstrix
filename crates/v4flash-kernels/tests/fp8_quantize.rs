//! FP8 E4M3FN quantize oracle — validates `Fp8E4m3fnQuantize` against
//! ds4's `dsv4_fp8_kv_quantize_row_inplace_cpu` (ds4.c:1635) using the
//! `comp_pre_fp8` (input) and `comp_post_fp8` (expected output) tags
//! captured by patch 0007.
//!
//! Only the main compressor invokes FP8 (head_dim == DS4_N_HEAD_DIM=512);
//! the indexer compressor (head_dim=128) skips this op. The kernel
//! quantises the first `n_nope = head_dim - n_rot = 448` elements per
//! row, in 64-element blocks; the RoPE'd tail (last 64) is untouched
//! and is bit-identical between comp_pre_fp8 and comp_post_fp8.
//!
//! Threshold 1e-5, mean<1e-7. The kernel mirrors ds4's lookup-based
//! encode/decode bit-for-bit; tie-breaking matches.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{ActivationDump, Fp8E4m3fnQuantize};

const HEAD_DIM: u32 = 512;
const N_ROT: u32 = 64;
const N_NOPE: u32 = HEAD_DIM - N_ROT; // 448
const THRESHOLD: f32 = 1.0e-5;

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
fn fp8_quantize_oracle_main_compressor() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = Fp8E4m3fnQuantize::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HEAD_DIM as usize)?;
    let mut got = vec![0f32; HEAD_DIM as usize];

    let mut stats = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    // Only ratio==4 layers produce comp_post_fp8 in our prompt
    // (ratio==128 never fires). For each boundary token, load pre, run, compare to post.
    for layer in (2..=42).step_by(2) {
        for token in 0..n_tokens {
            let pre_entry = match dump.tensor("comp_pre_fp8", layer, token) {
                Some(e) => e,
                None => continue,
            };
            let post_entry = dump
                .tensor("comp_post_fp8", layer, token)
                .ok_or_else(|| eyre!("missing comp_post_fp8 at L{layer} T{token}"))?;
            let pre = dump.read_f32(pre_entry)?;
            let post = dump.read_f32(post_entry)?;
            assert_eq!(pre.len(), HEAD_DIM as usize);
            assert_eq!(post.len(), HEAD_DIM as usize);

            d_x.copy_from_host(&pre)?;
            kernel.launch(&stream, &mut d_x, N_NOPE)?;
            stream.synchronize()?;
            d_x.copy_to_host(&mut got)?;

            let prev = stats.max_abs;
            stats.update(&got, &post);
            if stats.max_abs > prev {
                worst = (layer, token);
            }
        }
    }

    eprintln!(
        "OVERALL: max_abs_diff={:.3e}, mean_abs_diff={:.3e}, n={}, worst at L{} T{}",
        stats.max_abs,
        stats.mean_abs(),
        stats.count,
        worst.0,
        worst.1,
    );

    assert!(
        stats.max_abs < THRESHOLD,
        "max_abs_diff {:.3e} exceeds threshold {:.3e}",
        stats.max_abs,
        THRESHOLD
    );

    Ok(())
}
