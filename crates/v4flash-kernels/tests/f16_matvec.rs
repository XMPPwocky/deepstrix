//! F16 matvec oracle — validates `f16_matvec` against ds4's `matvec_any`
//! by computing `attn_compressor_kv × attn_input_norm` and comparing to
//! `comp_state_kv_row` (the per-token compressor state buffer write,
//! captured by patch 0007 right after the matvec).
//!
//! Covers both ratio==4 layers (compressor output width = 1024) and
//! ratio==128 layers (compressor output width = 512) — exercises the
//! kernel's variable-n_rows handling.
//!
//! Threshold 1e-3, mean<1e-5. F16 accumulation is in f32; the only
//! noise source is the F16 → F32 dequant per weight, which is bit-exact
//! to ds4's `__half2float`-equivalent.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{weights, ActivationDump, F16Matvec};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_EMBD: u32 = 4096;
const THRESHOLD: f32 = 1.0e-3;

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

fn layer_compress_ratio(il: i32) -> u32 {
    if il < 2 {
        0
    } else if (il & 1) == 0 {
        4
    } else {
        128
    }
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
fn f16_matvec_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = F16Matvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    // Output buffer sized for the largest case (ratio==4: n_rows=1024).
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, 1024)?;
    let mut got_buf = vec![0f32; 1024];

    let mut stats = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    for layer in 2..43 {
        let ratio = layer_compress_ratio(layer);
        let comp_width: u32 = if ratio == 4 { 2 * 512 } else { 512 }; // 1024 or 512

        let wkv = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.attn_compressor_kv.weight"),
            device.id,
        )?;
        assert_eq!(wkv.shape, vec![N_EMBD as u64, comp_width as u64]);

        for token in 0..n_tokens {
            let x_entry = dump
                .tensor("attn_input_norm", layer, token)
                .ok_or_else(|| eyre!("missing attn_input_norm at L{layer} T{token}"))?;
            let exp_entry = dump
                .tensor("comp_state_kv_row", layer, token)
                .ok_or_else(|| eyre!("missing comp_state_kv_row at L{layer} T{token}"))?;

            let x_host = dump.read_f32(x_entry)?;
            let expected = dump.read_f32(exp_entry)?;
            assert_eq!(x_host.len(), N_EMBD as usize);
            assert_eq!(expected.len(), comp_width as usize);

            d_x.copy_from_host(&x_host)?;
            kernel.matvec(&stream, &mut d_out, &wkv.buffer, &d_x, comp_width, N_EMBD)?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got_buf)?;
            let got = &got_buf[..comp_width as usize];

            let prev = stats.max_abs;
            stats.update(got, &expected);
            if stats.max_abs > prev {
                worst = (layer, token);
            }
        }

        drop(wkv);
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

/// M40-P4.5: f16_matvec_two_inputs vs two separate f16.matvec calls.
/// NOT bit-exact (reduction order differs), but should be tight enough
/// that argmax preservation holds at every output position. Tolerance
/// 1e-3 matches the threshold for f16_matvec_oracle vs ds4.
#[test]
#[ignore]
fn f16_matvec_two_inputs_matches_two_singles() -> eyre::Result<()> {
    install_panic_handler()?;
    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("f16 two-inputs oracle on device {} ({arch})", device.id);

    // Use an N_EMBD-wide weight that the wide kernel will pick (n_rows ≥ NARROW_ROWS_THRESHOLD).
    // attn_compressor_kv for ratio=4 layer is 1024 × 4096 — n_rows=1024 is plenty wide.
    let layer = 2i32;
    let comp_width = 1024u32;
    let wkv = weights::load_to_device(
        &gguf,
        &format!("blk.{}.attn_compressor_kv.weight", layer),
        device.id,
    )?;
    let kernel = F16Matvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_x_a: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_x_b: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_out_a: DeviceBuffer<f32> = DeviceBuffer::new(device.id, comp_width as usize)?;
    let mut d_out_b: DeviceBuffer<f32> = DeviceBuffer::new(device.id, comp_width as usize)?;
    let mut d_out_pair_a: DeviceBuffer<f32> = DeviceBuffer::new(device.id, comp_width as usize)?;
    let mut d_out_pair_b: DeviceBuffer<f32> = DeviceBuffer::new(device.id, comp_width as usize)?;

    let test_pairs = [(0i32, 1i32), (2, 3), (4, 5), (0, 6)];
    let mut max_diff_a = 0f32;
    let mut max_diff_b = 0f32;
    let mut got_a = vec![0f32; comp_width as usize];
    let mut got_b = vec![0f32; comp_width as usize];
    let mut got_pair_a = vec![0f32; comp_width as usize];
    let mut got_pair_b = vec![0f32; comp_width as usize];
    for &(ta, tb) in test_pairs.iter() {
        let xa_entry = dump
            .tensor("attn_input_norm", layer, ta)
            .ok_or_else(|| eyre!("missing attn_input_norm L{layer} T{ta}"))?;
        let xb_entry = dump
            .tensor("attn_input_norm", layer, tb)
            .ok_or_else(|| eyre!("missing attn_input_norm L{layer} T{tb}"))?;
        let xa = dump.read_f32(xa_entry)?;
        let xb = dump.read_f32(xb_entry)?;
        d_x_a.copy_from_host(&xa)?;
        d_x_b.copy_from_host(&xb)?;

        kernel.matvec(&stream, &mut d_out_a, &wkv.buffer, &d_x_a, comp_width, N_EMBD)?;
        kernel.matvec(&stream, &mut d_out_b, &wkv.buffer, &d_x_b, comp_width, N_EMBD)?;
        kernel.matvec_two_inputs(
            &stream,
            &mut d_out_pair_a,
            &mut d_out_pair_b,
            &wkv.buffer,
            &d_x_a,
            &d_x_b,
            comp_width,
            N_EMBD,
        )?;
        stream.synchronize()?;
        d_out_a.copy_to_host(&mut got_a)?;
        d_out_b.copy_to_host(&mut got_b)?;
        d_out_pair_a.copy_to_host(&mut got_pair_a)?;
        d_out_pair_b.copy_to_host(&mut got_pair_b)?;
        let mut a_max = 0f32;
        let mut b_max = 0f32;
        for i in 0..comp_width as usize {
            a_max = a_max.max((got_a[i] - got_pair_a[i]).abs());
            b_max = b_max.max((got_b[i] - got_pair_b[i]).abs());
        }
        eprintln!("pair ({ta},{tb}): max_diff a={:.3e} b={:.3e}", a_max, b_max);
        max_diff_a = max_diff_a.max(a_max);
        max_diff_b = max_diff_b.max(b_max);
    }
    eprintln!("F16 PAIR ORACLE: max_diff a={:.3e} b={:.3e}", max_diff_a, max_diff_b);
    // Reduction-order drift — bit-exact not expected. Tolerance 1e-3 (same
    // as f16_matvec_oracle).
    assert!(max_diff_a < 1e-3, "f16 two-inputs col a max diff {max_diff_a:.3e} > 1e-3");
    assert!(max_diff_b < 1e-3, "f16 two-inputs col b max diff {max_diff_b:.3e} > 1e-3");
    Ok(())
}
