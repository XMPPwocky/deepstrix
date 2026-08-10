//! Unit test for the fused per-head QK-RMSNorm kernel (`laguna_qk_rmsnorm`).
//! Compares the device kernel against a CPU per-head reference over random
//! data. No GGUF needed — just a GPU.
//!
//! Run:
//!   nix develop --command cargo test --release -p v4flash-kernels \
//!       --test laguna_qk_norm -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer, Stream};
use v4flash_kernels::laguna::LagunaOps;

const HEAD_DIM: usize = 128;
const EPS: f32 = 1e-6;

/// CPU per-head RMSNorm reference (double accum, matches the kernel).
fn cpu_qk_rmsnorm(input: &[f32], weight: &[f32], n_head: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_head * HEAD_DIM];
    for h in 0..n_head {
        let row = &input[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let mut ss = 0.0f64;
        for &v in row {
            ss += (v as f64) * (v as f64);
        }
        let mean_sq = (ss / HEAD_DIM as f64) as f32;
        let scale = 1.0f32 / (mean_sq + EPS).sqrt();
        for i in 0..HEAD_DIM {
            out[h * HEAD_DIM + i] = row[i] * scale * weight[i];
        }
    }
    out
}

#[test]
#[ignore = "drives the GPU; run explicitly"]
fn laguna_qk_rmsnorm_matches_cpu() -> eyre::Result<()> {
    let _ = v4flash_hip::install_panic_handler();

    let dev = Device::all()?
        .into_iter()
        .find(|d| {
            d.properties()
                .map(|p| p.gcn_arch_name.starts_with("gfx1151") || p.gcn_arch_name.starts_with("gfx1201"))
                .unwrap_or(false)
        })
        .ok_or_else(|| eyre!("no gfx1151/gfx1201 device"))?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    println!("device id={} arch={arch}", dev.id);

    let ops = LagunaOps::for_arch(&arch)?;
    let stream = Stream::new(dev.id)?;

    // Cover both attention widths used by Laguna (72 SWA, 8 KV heads).
    for &n_head in &[72usize, 48, 8, 1] {
        // Deterministic pseudo-random input + weight.
        let mut input = vec![0f32; n_head * HEAD_DIM];
        for (i, x) in input.iter_mut().enumerate() {
            let t = (i as f32) * 0.12345 + 0.3;
            *x = (t.sin() * 3.1 - 0.7) * (1.0 + 0.01 * (i % 7) as f32);
        }
        let mut weight = vec![0f32; HEAD_DIM];
        for (i, w) in weight.iter_mut().enumerate() {
            *w = 0.5 + 0.5 * ((i as f32) * 0.021).cos();
        }

        let mut d_in = DeviceBuffer::<f32>::new(dev.id, input.len())?;
        let mut d_w = DeviceBuffer::<f32>::new(dev.id, weight.len())?;
        let mut d_out = DeviceBuffer::<f32>::new(dev.id, input.len())?;
        d_in.copy_from_host(&input)?;
        d_w.copy_from_host(&weight)?;

        ops.qk_rmsnorm(&stream, &mut d_out, &d_in, &d_w, n_head as u32, HEAD_DIM as u32, EPS)?;
        stream.synchronize()?;

        let mut got = vec![0f32; input.len()];
        d_out.copy_to_host(&mut got)?;

        let want = cpu_qk_rmsnorm(&input, &weight, n_head);
        let mut max_abs = 0.0f32;
        for (a, b) in got.iter().zip(want.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        println!("n_head={n_head}: max_abs err = {max_abs:.3e}");
        assert!(max_abs < 1e-4, "n_head={n_head} max_abs {max_abs} too large");
    }

    println!("[OK] laguna_qk_rmsnorm matches CPU per-head reference");
    Ok(())
}
