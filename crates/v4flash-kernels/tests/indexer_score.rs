//! indexer_score kernel oracle. The `indexer_scores` dump tag from
//! patch 0007 doesn't fire on our 57-token M1 prompt — ds4 short-circuits
//! the indexer when `n_comp ≤ DS4_N_INDEXER_TOP_K (512)`, which is always
//! the case here (n_comp ≤ 14). So we can't validate against ds4 directly.
//!
//! Instead, this test uses **synthetic deterministic inputs** and
//! cross-validates the HIP kernel against a CPU reimplementation of the
//! exact same algorithm. It proves the kernel computes the documented
//! formula correctly; end-to-end validation against ds4 awaits M11
//! (long-prompt runs).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{IndexerScore, INDEXER_HEAD_DIM, INDEXER_N_HEAD};

const THRESHOLD: f32 = 1.0e-3;

fn pick_device() -> eyre::Result<Device> {
    let devices = Device::all()?;
    for d in &devices {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre!("no HIP devices"))
}

fn lcg_step(s: &mut u32) -> f32 {
    // Tiny deterministic PRNG for reproducible synthetic inputs.
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    let v = (*s >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
    v * 2.0 - 1.0                                    // [-1, 1)
}

fn cpu_indexer_score(
    q: &[f32],
    head_weights: &[f32],
    index_comp_kv: &[f32],
    n_comp: usize,
    n_head: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut scores = vec![0f32; n_comp];
    for c in 0..n_comp {
        let kv = &index_comp_kv[c * head_dim..(c + 1) * head_dim];
        let mut s = 0.0f32;
        for h in 0..n_head {
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += kv[i] * qh[i];
            }
            if dot < 0.0 {
                dot = 0.0;
            }
            s += dot * head_weights[h];
        }
        scores[c] = s;
    }
    scores
}

#[test]
#[ignore]
fn indexer_score_synthetic() -> eyre::Result<()> {
    install_panic_handler()?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = IndexerScore::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let n_head = INDEXER_N_HEAD as usize;
    let head_dim = INDEXER_HEAD_DIM as usize;

    // Test a few realistic n_comp values: a small one (similar to what
    // we'd see in our prompt) and a larger one (the regime where the
    // scoring path actually fires in production).
    for &n_comp in &[14usize, 128, 514] {
        let mut seed: u32 = 0xdeadbeef_u32.wrapping_add(n_comp as u32);
        let q_flat = n_head * head_dim;
        let mut q = vec![0f32; q_flat];
        let mut head_weights = vec![0f32; n_head];
        let mut comp_kv = vec![0f32; n_comp * head_dim];
        for v in &mut q {
            *v = lcg_step(&mut seed) * 0.5;
        }
        for v in &mut head_weights {
            *v = lcg_step(&mut seed) * 0.1;
        }
        for v in &mut comp_kv {
            *v = lcg_step(&mut seed) * 0.7;
        }

        let expected = cpu_indexer_score(&q, &head_weights, &comp_kv, n_comp, n_head, head_dim);

        let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(device.id, q_flat)?;
        let mut d_hw: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head)?;
        let mut d_kv: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_comp * head_dim)?;
        let mut d_scores: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_comp)?;
        d_q.copy_from_host(&q)?;
        d_hw.copy_from_host(&head_weights)?;
        d_kv.copy_from_host(&comp_kv)?;

        kernel.launch(
            &stream,
            &mut d_scores,
            &d_q,
            &d_hw,
            &d_kv,
            n_comp as u32,
            n_head as u32,
            head_dim as u32,
        )?;
        stream.synchronize()?;

        let mut got = vec![0f32; n_comp];
        d_scores.copy_to_host(&mut got)?;

        let mut max_abs = 0.0f32;
        for (a, e) in got.iter().zip(expected.iter()) {
            let d = (a - e).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!("n_comp={n_comp}: max_abs_diff={max_abs:.3e} (expected magnitude ~{:.3})",
                  expected.iter().map(|v| v.abs()).fold(0f32, f32::max));
        assert!(
            max_abs < THRESHOLD,
            "n_comp={n_comp}: max_abs_diff {max_abs:.3e} exceeds threshold {THRESHOLD:.3e}"
        );
    }

    Ok(())
}
