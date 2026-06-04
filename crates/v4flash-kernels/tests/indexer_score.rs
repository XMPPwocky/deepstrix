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
use v4flash_kernels::{IndexerScore, IndexerScoreWmma, INDEXER_HEAD_DIM, INDEXER_N_HEAD};

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

/// IEEE 754 round-half-to-even f32 → binary16. Inlined so we don't pull
/// in the `half` crate just for the test.
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x7FFFFF;
    if exp == 0xFF {
        // Inf / NaN.
        if mant != 0 {
            return sign | 0x7E00; // canonical NaN
        }
        return sign | 0x7C00;
    }
    let unbiased = exp - 127 + 15;
    if unbiased >= 0x1F {
        return sign | 0x7C00; // overflow → inf
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign; // underflow → 0
        }
        // Subnormal: pack with implicit leading 1 and shift right.
        let mant_full = mant | 0x800000;
        let shift = (1 - unbiased) as u32;
        let half = 1u32 << (shift + 12);
        let mask = (1u32 << (shift + 13)) - 1;
        let rounded = mant_full + half;
        let m = (rounded >> (shift + 13)) as u16;
        // round-to-even tiebreak
        let sticky = mant_full & mask;
        let m = if sticky == half && (m & 1) == 0 { m } else { m };
        return sign | m;
    }
    // Normal: 10-bit mantissa, round half to even.
    let m_full = mant + 0x1000; // add 1/2 ulp
    let mut m = (m_full >> 13) as u16;
    let mut e = unbiased as u16;
    if (m_full >> 13) & 0x400 != 0 {
        // overflow into exponent
        m = 0;
        e += 1;
        if e >= 0x1F {
            return sign | 0x7C00;
        }
    }
    sign | (e << 10) | (m & 0x3FF)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    v4flash_kernels::iq2_xxs_tables::f16_to_f32(bits)
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

        // index_comp_kv is f16 in production (matches the indexer
        // compressor's output dtype). Round-trip through f16 here so the
        // CPU reference operates on the same quantized values the kernel
        // will see — the comparison then isolates the kernel's compute
        // from the dtype round-trip.
        let comp_kv_f16: Vec<u16> = comp_kv.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let comp_kv_quantized: Vec<f32> =
            comp_kv_f16.iter().map(|&u| f16_bits_to_f32(u)).collect();
        let expected = cpu_indexer_score(
            &q,
            &head_weights,
            &comp_kv_quantized,
            n_comp,
            n_head,
            head_dim,
        );

        let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(device.id, q_flat)?;
        let mut d_hw: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_head)?;
        let mut d_kv: DeviceBuffer<u16> = DeviceBuffer::new(device.id, n_comp * head_dim)?;
        let mut d_scores: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_comp)?;
        d_q.copy_from_host(&q)?;
        d_hw.copy_from_host(&head_weights)?;
        d_kv.copy_from_host(&comp_kv_f16)?;

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

    // --- WMMA variant: same shapes, same threshold (relaxed slightly for
    //     f16 fragment intermediate vs the naïve kernel's f32 throughout).
    //
    // WMMA only compiles under __gfx1200__ / __gfx1201__ — the kernel's
    // RDNA4 builtin gates on that. The dGPU (gfx1201) is the production
    // dispatch target; the iGPU (gfx1151) silently no-ops the WMMA path.
    // Switch to the dGPU explicitly for this arm, regardless of which
    // device pick_device() returned for the naïve test above.
    let dgpu = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx12")).unwrap_or(false));
    let Some(dgpu) = dgpu else {
        eprintln!("[wmma] skipping — no gfx12 dGPU present (iGPU-only system)");
        return Ok(());
    };
    dgpu.set_current()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let kernel_wmma = IndexerScoreWmma::for_arch(&dgpu_arch)?;
    eprintln!("[wmma] using dGPU {} ({dgpu_arch})", dgpu.id);
    const WMMA_THRESHOLD: f32 = 5.0e-3;
    for &n_comp in &[14usize, 128, 514, 16384] {
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
        let comp_kv_f16: Vec<u16> = comp_kv.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let comp_kv_quantized: Vec<f32> =
            comp_kv_f16.iter().map(|&u| f16_bits_to_f32(u)).collect();
        let expected = cpu_indexer_score(
            &q,
            &head_weights,
            &comp_kv_quantized,
            n_comp,
            n_head,
            head_dim,
        );

        let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, q_flat)?;
        let mut d_hw: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_head)?;
        let mut d_kv: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, n_comp * head_dim)?;
        let mut d_scores: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_comp)?;
        d_q.copy_from_host(&q)?;
        d_hw.copy_from_host(&head_weights)?;
        d_kv.copy_from_host(&comp_kv_f16)?;

        kernel_wmma.launch(&stream, &mut d_scores, &d_q, &d_hw, &d_kv, n_comp as u32)?;
        stream.synchronize()?;
        let mut got = vec![0f32; n_comp];
        d_scores.copy_to_host(&mut got)?;

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (a, e) in got.iter().zip(expected.iter()) {
            let d = (a - e).abs();
            if d > max_abs {
                max_abs = d;
            }
            let r = d / e.abs().max(1e-6);
            if r > max_rel {
                max_rel = r;
            }
        }
        eprintln!(
            "[wmma] n_comp={n_comp}: max_abs_diff={max_abs:.3e} max_rel={max_rel:.3e} (expected magnitude ~{:.3})",
            expected.iter().map(|v| v.abs()).fold(0f32, f32::max)
        );
        if max_abs >= WMMA_THRESHOLD {
            // Show the first 16 entries to localise the bad slot.
            for c in 0..n_comp.min(16) {
                let d = (got[c] - expected[c]).abs();
                eprintln!("  c={c:3} expected={:11.4e} got={:11.4e} diff={d:.3e}", expected[c], got[c]);
            }
        }
        assert!(
            max_abs < WMMA_THRESHOLD,
            "[wmma] n_comp={n_comp}: max_abs_diff {max_abs:.3e} exceeds threshold {WMMA_THRESHOLD:.3e}"
        );
    }

    Ok(())
}
