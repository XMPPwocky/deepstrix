//! Sampler kernel validation.
//!
//! Three checks against synthetic N_VOCAB=129,280 logits:
//!   1. argmax_one bit-exactly matches CPU argmax (with tie-break by
//!      lowest index).
//!   2. softmax_sample_one with very low temperature (T = 0.001) collapses
//!      to argmax over 64 trials with varying u01 — quasi-argmax sanity.
//!   3. softmax_sample_one with T = 1.0 produces sample frequencies
//!      consistent with softmax(logits) on a small 8-bucket distribution
//!      stamped into the front of the vocab.
//!
//! Run with `cargo test --release --test sampler -- --ignored --nocapture`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{Sampler, SamplerRng, SAMPLER_N_WG};

const N_VOCAB: u32 = 129_280;

fn pick_device() -> eyre::Result<Device> {
    let devices = Device::all()?;
    // Prefer dGPU (gfx1201) since that's where the sampler runs in production.
    for d in &devices {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre!("no HIP devices"))
}

fn cpu_argmax(v: &[f32]) -> i32 {
    let mut best = 0i32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i as i32;
        }
    }
    best
}

#[test]
#[ignore]
fn sampler_argmax_matches_cpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let sampler = Sampler::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Generate logits with a unique maximum at a randomly-chosen index per trial.
    let mut rng = SamplerRng::new(0xBEEF);
    let mut logits = vec![0f32; N_VOCAB as usize];

    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;
    let mut d_out: DeviceBuffer<i32> = DeviceBuffer::new(device.id, 1)?;

    let mut got = [0i32; 1];
    let mut mismatches = 0usize;
    for trial in 0..32 {
        for x in logits.iter_mut() {
            // [-1.0, 1.0)
            *x = rng.next_f32() * 2.0 - 1.0;
        }
        // Stamp a clear winner.
        let winner = (rng.next_f32() * (N_VOCAB as f32)) as usize;
        logits[winner] = 10.0;
        let expected = cpu_argmax(&logits);

        d_logits.copy_from_host(&logits)?;
        sampler.launch_argmax(&stream, &mut d_out, &d_logits, N_VOCAB)?;
        stream.synchronize()?;
        d_out.copy_to_host(&mut got)?;
        if got[0] != expected {
            eprintln!(
                "trial {trial}: gpu={} cpu={} (winner@{})",
                got[0], expected, winner
            );
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "argmax must match CPU bit-for-bit");
    Ok(())
}

#[test]
#[ignore]
fn sampler_multinomial_low_T_collapses_to_argmax() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let sampler = Sampler::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut rng = SamplerRng::new(0xC0FFEE);
    let mut logits = vec![0f32; N_VOCAB as usize];
    for x in logits.iter_mut() {
        *x = rng.next_f32() * 2.0 - 1.0;
    }
    // Big margin so even with T=0.01 the next-best is exp(-100) away.
    let winner = 42_000usize;
    logits[winner] = 5.0;
    let expected = winner as i32;

    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;
    let mut d_partials_max: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_partials_z: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_u01: DeviceBuffer<f32> = DeviceBuffer::new(device.id, 1)?;
    let mut d_out: DeviceBuffer<i32> = DeviceBuffer::new(device.id, 1)?;

    d_logits.copy_from_host(&logits)?;

    let mut got = [0i32; 1];
    let mut mismatches = 0usize;
    let n_trials = 64;
    for trial in 0..n_trials {
        let u = rng.next_f32();
        d_u01.copy_from_host(&[u])?;
        sampler.launch_multinomial(
            &stream,
            &mut d_out,
            &d_logits,
            &mut d_partials_max,
            &mut d_partials_z,
            &d_u01,
            N_VOCAB,
            0.01,
            0.0,
        )?;
        stream.synchronize()?;
        d_out.copy_to_host(&mut got)?;
        if got[0] != expected {
            eprintln!("low-T trial {trial}: gpu={} expected={} (u={u})", got[0], expected, u = u);
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "low-T multinomial must collapse to argmax");
    Ok(())
}

#[test]
#[ignore]
fn sampler_multinomial_marginal_matches_softmax() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let sampler = Sampler::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // 8-bucket head distribution; rest of vocab is -INF-equivalent (-30).
    let bucket_logits: [f32; 8] = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5];
    let mut logits = vec![-30.0f32; N_VOCAB as usize];
    for (i, &v) in bucket_logits.iter().enumerate() {
        logits[i] = v;
    }
    // CPU softmax over the 8 buckets (rest are vanishingly small).
    let max_l = bucket_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f64> = bucket_logits.iter().map(|&v| ((v - max_l) as f64).exp()).collect();
    let z: f64 = exp.iter().sum();
    let expected_prob: Vec<f64> = exp.iter().map(|&e| e / z).collect();

    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;
    let mut d_partials_max: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_partials_z: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_u01: DeviceBuffer<f32> = DeviceBuffer::new(device.id, 1)?;
    let mut d_out: DeviceBuffer<i32> = DeviceBuffer::new(device.id, 1)?;
    d_logits.copy_from_host(&logits)?;

    let mut counts = [0u64; 8];
    let mut other = 0u64;
    let mut rng = SamplerRng::new(0xDA7A);
    let n_samples = 20_000u64;
    let mut got = [0i32; 1];
    for _ in 0..n_samples {
        let u = rng.next_f32();
        d_u01.copy_from_host(&[u])?;
        sampler.launch_multinomial(
            &stream,
            &mut d_out,
            &d_logits,
            &mut d_partials_max,
            &mut d_partials_z,
            &d_u01,
            N_VOCAB,
            1.0,
            0.0,
        )?;
        stream.synchronize()?;
        d_out.copy_to_host(&mut got)?;
        let i = got[0];
        if i >= 0 && (i as usize) < 8 {
            counts[i as usize] += 1;
        } else {
            other += 1;
        }
    }
    eprintln!("counts: {:?}, other: {}", counts, other);

    // Tolerance: 0.02 absolute on each bucket frequency. With n=20k the 1σ
    // for p=0.4 is ~0.0035, so 5σ ≈ 0.018 — 0.02 is safe.
    for i in 0..8 {
        let observed = counts[i] as f64 / n_samples as f64;
        let expected = expected_prob[i];
        let diff = (observed - expected).abs();
        eprintln!("  bucket {}: observed={:.4}, expected={:.4}, |Δ|={:.4}", i, observed, expected, diff);
        assert!(
            diff < 0.02,
            "bucket {i} marginal off by {diff:.4} (observed {observed:.4} vs expected {expected:.4})"
        );
    }
    // The remaining N_VOCAB-8 buckets share exp(-30 - 3) ≈ exp(-33) ≈ 4.7e-15
    // probability mass total; should never be sampled in 20k draws.
    assert_eq!(other, 0, "tail buckets must have negligible sampling mass");
    Ok(())
}

#[test]
#[ignore]
fn sampler_multinomial_min_p_renormalises() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let sampler = Sampler::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Same 8-bucket head as the marginal test. With min_p_rel = 0.1,
    // buckets 0..5 survive (rel probs 1.0, .61, .37, .22, .14) and 5..8
    // are pruned (.082, .050, .030). Marginals must renormalise over the
    // survivors; the pre-fix kernel left pruned mass in Z and routed it
    // to the argmax fallback, inflating bucket 0 by ~6.5 points.
    let min_p_rel = 0.1f32;
    let bucket_logits: [f32; 8] = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5];
    let mut logits = vec![-30.0f32; N_VOCAB as usize];
    for (i, &v) in bucket_logits.iter().enumerate() {
        logits[i] = v;
    }
    let max_l = bucket_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let rel: Vec<f64> = bucket_logits.iter().map(|&v| ((v - max_l) as f64).exp()).collect();
    let z: f64 = rel.iter().filter(|&&e| e >= min_p_rel as f64).sum();
    let expected_prob: Vec<f64> = rel
        .iter()
        .map(|&e| if e >= min_p_rel as f64 { e / z } else { 0.0 })
        .collect();

    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;
    let mut d_partials_max: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_partials_z: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_u01: DeviceBuffer<f32> = DeviceBuffer::new(device.id, 1)?;
    let mut d_out: DeviceBuffer<i32> = DeviceBuffer::new(device.id, 1)?;
    d_logits.copy_from_host(&logits)?;

    let mut counts = [0u64; 8];
    let mut other = 0u64;
    let mut rng = SamplerRng::new(0x5EED);
    let n_samples = 20_000u64;
    let mut got = [0i32; 1];
    for _ in 0..n_samples {
        let u = rng.next_f32();
        d_u01.copy_from_host(&[u])?;
        sampler.launch_multinomial(
            &stream,
            &mut d_out,
            &d_logits,
            &mut d_partials_max,
            &mut d_partials_z,
            &d_u01,
            N_VOCAB,
            1.0,
            min_p_rel,
        )?;
        stream.synchronize()?;
        d_out.copy_to_host(&mut got)?;
        let i = got[0];
        if i >= 0 && (i as usize) < 8 {
            counts[i as usize] += 1;
        } else {
            other += 1;
        }
    }
    eprintln!("counts: {:?}, other: {}", counts, other);

    for i in 0..8 {
        let observed = counts[i] as f64 / n_samples as f64;
        let expected = expected_prob[i];
        let diff = (observed - expected).abs();
        eprintln!("  bucket {}: observed={:.4}, expected={:.4}, |Δ|={:.4}", i, observed, expected, diff);
        assert!(
            diff < 0.02,
            "bucket {i} marginal off by {diff:.4} (observed {observed:.4} vs expected {expected:.4})"
        );
    }
    assert_eq!(other, 0, "pruned/tail buckets must never be sampled");
    Ok(())
}
