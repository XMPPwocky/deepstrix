//! Sampler kernel validation.
//!
//! Checks against synthetic N_VOCAB=129,280 logits:
//!   1. argmax_one bit-exactly matches CPU argmax (with tie-break by
//!      lowest index).
//!   2. softmax_sample_one with very low temperature (T = 0.001) collapses
//!      to argmax over 64 trials with varying u01 — quasi-argmax sanity.
//!   3. softmax_sample_one with T = 1.0 produces sample frequencies
//!      consistent with softmax(logits) on a small 8-bucket distribution
//!      stamped into the front of the vocab.
//!   4. the legacy (top_p = 1) chain obeys the inverse-CDF contract against
//!      an f64 host reference, and the top-p chain's sample frequencies match
//!      the nucleus renormalised over the cutoff it published.
//!
//! Run with `cargo test --release --test sampler -- --ignored --nocapture`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    Sampler, SamplerRng, SAMPLER_N_WG, SAMPLER_TOPP_LEVELS, SAMPLER_TOPP_LOG_RANGE,
    SAMPLER_TOPP_NBINS,
};

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

// ===========================================================================
// top-p (nucleus) sampling
// ===========================================================================

/// Unnormalised weights the kernels work in: `exp(x*inv_T - gmax)`, so the
/// most likely token weighs exactly 1.0.
fn cpu_weights(logits: &[f32], temperature: f64) -> Vec<f64> {
    let inv_t = 1.0 / temperature;
    let gmax = logits
        .iter()
        .map(|&x| x as f64 * inv_t)
        .fold(f64::NEG_INFINITY, f64::max);
    logits
        .iter()
        .map(|&x| ((x as f64 * inv_t) - gmax).exp())
        .collect()
}

/// A vocabulary-sized logit vector with a `head_n`-token head carrying the
/// interesting probability mass and a deep, noisy tail.
fn peaked_logits(rng: &mut SamplerRng, head_n: usize, head_span: f32, tail: f32) -> Vec<f32> {
    let mut v = vec![0f32; N_VOCAB as usize];
    for x in v.iter_mut() {
        *x = tail + rng.next_f32() * 0.5;
    }
    for i in 0..head_n {
        // Descending head: 0, -span/head_n, -2*span/head_n, ...
        v[i * 7 + 11] = -(head_span * i as f32) / head_n as f32;
    }
    v
}

struct ToppBufs {
    mass: DeviceBuffer<f32>,
    bracket: DeviceBuffer<f32>,
    thr: DeviceBuffer<f32>,
}

impl ToppBufs {
    fn new(device_id: i32) -> eyre::Result<Self> {
        Ok(Self {
            mass: DeviceBuffer::new(device_id, v4flash_kernels::sampler_topp_mass_len())?,
            bracket: DeviceBuffer::new(device_id, 2)?,
            thr: DeviceBuffer::new(device_id, 1)?,
        })
    }
}

/// `top_p = 1.0` must DELEGATE to `launch_multinomial` — same three launches,
/// same arguments — and an out-of-range `top_p` must be a hard error.
///
/// SCOPE: this only pins the wrapper's dispatch. Both arms end up executing
/// the identical function, so it cannot detect drift in the (modified)
/// `logits_expsum_partial` / `softmax_sample_one` kernels themselves; that is
/// what `sampler_legacy_chain_matches_cpu_inverse_cdf` below is for.
#[test]
#[ignore]
fn sampler_topp_one_delegates_to_legacy_path() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let sampler = Sampler::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut rng = SamplerRng::new(0x70_9D_1E);
    let logits = peaked_logits(&mut rng, 24, 9.0, -6.0);

    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;
    let mut d_max: DeviceBuffer<f32> = DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_z: DeviceBuffer<f32> = DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_u01: DeviceBuffer<f32> = DeviceBuffer::new(device.id, 1)?;
    let mut d_out: DeviceBuffer<i32> = DeviceBuffer::new(device.id, 1)?;
    let mut topp = ToppBufs::new(device.id)?;
    d_logits.copy_from_host(&logits)?;

    let mut legacy = [0i32; 1];
    let mut with_topp = [0i32; 1];
    for &min_p_rel in &[0.0f32, 0.1f32] {
        for trial in 0..256 {
            let u = rng.next_f32();
            d_u01.copy_from_host(&[u])?;
            sampler.launch_multinomial(
                &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z, &d_u01,
                N_VOCAB, 1.0, min_p_rel,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut legacy)?;

            d_u01.copy_from_host(&[u])?;
            sampler.launch_multinomial_topp(
                &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z,
                &mut topp.mass, &mut topp.bracket, &mut topp.thr, &d_u01,
                N_VOCAB, 1.0, min_p_rel, 1.0,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut with_topp)?;

            assert_eq!(
                legacy[0], with_topp[0],
                "top_p=1.0 must dispatch to the legacy path (min_p_rel={min_p_rel}, \
                 trial {trial}, u={u})"
            );
        }
    }
    // A value out of (0, 1] is a hard error, not a silent clamp, at the
    // kernel-wrapper level; the HTTP layer clamps before it gets here.
    let bad = sampler.launch_multinomial_topp(
        &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z,
        &mut topp.mass, &mut topp.bracket, &mut topp.thr, &d_u01,
        N_VOCAB, 1.0, 0.0, 0.0,
    );
    assert!(bad.is_err(), "top_p = 0 must be rejected");
    Ok(())
}

/// The device threshold search must land on the CPU reference nucleus:
/// the same cutoff (to the documented convergence bound), the same mass,
/// and no sampled token outside the set over many seeds.
///
/// The search returns the LOW end of the final bracket, so it never
/// over-truncates (`thr <= t*`) but may admit tokens whose weight sits in
/// the band `[thr, t*)` — at most a factor `exp(LOG_RANGE / NBINS^LEVELS)`
/// = 1.0000382 wide. For a peaked distribution that band is empty and the
/// device nucleus is *exactly* the CPU one; for a dense near-flat tail a
/// handful of near-tied tokens fall inside it, so the test asserts the real
/// guarantee (superset of the CPU nucleus, excess mass bounded by the band)
/// and reports how many draws land in it.
#[test]
#[ignore]
fn sampler_topp_nucleus_matches_cpu_reference() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let sampler = Sampler::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;
    let mut d_max: DeviceBuffer<f32> = DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_z: DeviceBuffer<f32> = DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_u01: DeviceBuffer<f32> = DeviceBuffer::new(device.id, 1)?;
    let mut d_out: DeviceBuffer<i32> = DeviceBuffer::new(device.id, 1)?;
    let mut topp = ToppBufs::new(device.id)?;

    // Worst-case bracket width after SAMPLER_TOPP_LEVELS refinements.
    let step = SAMPLER_TOPP_LOG_RANGE as f64
        / (SAMPLER_TOPP_NBINS as f64).powi(SAMPLER_TOPP_LEVELS as i32);
    let max_ratio = step.exp();

    let mut rng = SamplerRng::new(0xF0_0D_5E_ED);
    // (label, logits, temperature, top_p, min_p_rel, expect_exact_set)
    //
    // `expect_exact_set` = the cutoff lands in a sparse region, so the
    // convergence band is empty and the device nucleus must equal the CPU
    // nucleus token-for-token.
    let cases: Vec<(&str, Vec<f32>, f32, f32, f32, bool)> = vec![
        ("peaked-head", peaked_logits(&mut rng, 24, 9.0, -20.0), 1.0, 0.95, 0.0, true),
        ("peaked-head-tight", peaked_logits(&mut rng, 24, 9.0, -20.0), 1.0, 0.5, 0.0, true),
        ("cold-temperature", peaked_logits(&mut rng, 24, 9.0, -20.0), 0.7, 0.95, 0.0, true),
        ("dominant-token", {
            let mut v = peaked_logits(&mut rng, 8, 6.0, -20.0);
            v[777] = 12.0;
            v
        }, 1.0, 0.95, 0.0, true),
        // min-p strictly tighter than top-p: the composed cutoff is min_p
        // exactly, so the set is exact by construction.
        ("min-p-dominates", peaked_logits(&mut rng, 24, 9.0, -20.0), 1.0, 0.95, 0.25, true),
        // Dense tails: many tokens sit within a factor 1.0000382 of the
        // cutoff, so a threshold search cannot separate them.
        ("dense-tail", peaked_logits(&mut rng, 24, 9.0, -6.0), 1.0, 0.95, 0.0, false),
        ("wide-head", peaked_logits(&mut rng, 512, 14.0, -8.0), 1.0, 0.95, 0.0, false),
        ("near-flat", {
            let mut v = vec![0f32; N_VOCAB as usize];
            for x in v.iter_mut() {
                *x = rng.next_f32() * 2.0 - 1.0;
            }
            v
        }, 1.0, 0.95, 0.0, false),
    ];

    for (label, logits, temperature, top_p, min_p_rel, expect_exact) in cases {
        d_logits.copy_from_host(&logits)?;

        let w = cpu_weights(&logits, temperature as f64);
        let t_topp = v4flash_kernels::top_p_cutoff(&w, top_p as f64);
        let t_star = v4flash_kernels::top_p_min_p_threshold(&w, top_p as f64, min_p_rel as f64);
        let cpu_set: Vec<usize> = (0..w.len()).filter(|&i| w[i] >= t_star).collect();
        let z_full: f64 = w.iter().sum();
        let cpu_mass: f64 = cpu_set.iter().map(|&i| w[i]).sum();
        if min_p_rel as f64 <= t_topp {
            // top-p is the binding constraint: the reference must carry at
            // least top_p of the mass (min-p may legitimately cut below it).
            assert!(
                cpu_mass / z_full >= top_p as f64 - 1e-9,
                "{label}: CPU reference must keep >= top_p of the mass"
            );
        }

        // Run the search once and read back the published cutoff.
        d_u01.copy_from_host(&[0.5f32])?;
        sampler.launch_multinomial_topp(
            &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z,
            &mut topp.mass, &mut topp.bracket, &mut topp.thr, &d_u01,
            N_VOCAB, temperature, min_p_rel, top_p,
        )?;
        stream.synchronize()?;
        let mut thr_h = [0f32; 1];
        topp.thr.copy_to_host(&mut thr_h)?;
        let thr = thr_h[0] as f64;

        // Never over-truncates, and within one bracket width of exact.
        assert!(
            thr <= t_star * (1.0 + 1e-5),
            "{label}: device cutoff {thr:e} must not exceed the exact cutoff {t_star:e}"
        );
        assert!(
            t_star / thr <= max_ratio * (1.0 + 1e-5),
            "{label}: device cutoff {thr:e} off the exact cutoff {t_star:e} by {:.7}x, \
             bound is {max_ratio:.7}x",
            t_star / thr
        );

        let gpu_set: Vec<usize> = (0..w.len()).filter(|&i| w[i] >= thr).collect();
        let gpu_mass: f64 = gpu_set.iter().map(|&i| w[i]).sum();
        // Tokens admitted only because the search stopped one bracket short.
        let band: Vec<usize> = (0..w.len()).filter(|&i| w[i] >= thr && w[i] < t_star).collect();
        // The device cutoff should never sit above the exact one, so nothing
        // in the CPU nucleus may be missing. (The kernel shaves the published
        // cutoff by 1e-6 precisely so f32 rounding cannot flip this.)
        let missing: Vec<usize> = (0..w.len()).filter(|&i| w[i] >= t_star && w[i] < thr).collect();
        assert!(
            missing.is_empty(),
            "{label}: device nucleus dropped {} token(s) the CPU nucleus keeps \
             (cutoff {thr:e} vs {t_star:e})",
            missing.len()
        );
        assert_eq!(gpu_set.len(), cpu_set.len() + band.len());
        assert!(
            gpu_mass >= cpu_mass - 1e-9 * z_full,
            "{label}: device nucleus must be a superset of the CPU nucleus"
        );
        // The excess is bounded by |band| * t*, and must stay negligible.
        let excess = (gpu_mass - cpu_mass) / z_full;
        assert!(
            excess < 1e-4,
            "{label}: nucleus mass {gpu_mass} vs CPU {cpu_mass} (excess {excess:e} of Z, \
             {} band tokens)",
            band.len()
        );
        if expect_exact {
            assert!(
                band.is_empty(),
                "{label}: expected an empty convergence band, got {} tokens",
                band.len()
            );
            assert_eq!(gpu_set, cpu_set, "{label}: device nucleus must equal the CPU nucleus");
            // ... and then the mass matches to f32 precision.
            assert!(
                ((gpu_mass - cpu_mass) / cpu_mass).abs() < 1e-6,
                "{label}: nucleus mass {gpu_mass} vs CPU {cpu_mass}"
            );
        }

        // Now sample: every draw must land inside the nucleus.
        let mut got = [0i32; 1];
        let n_draws = 400;
        let mut in_band = 0usize;
        let mut distinct = std::collections::HashSet::new();
        for _ in 0..n_draws {
            let u = rng.next_f32();
            d_u01.copy_from_host(&[u])?;
            sampler.launch_multinomial_topp(
                &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z,
                &mut topp.mass, &mut topp.bracket, &mut topp.thr, &d_u01,
                N_VOCAB, temperature, min_p_rel, top_p,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;
            let id = got[0] as usize;
            assert!(
                w[id] >= thr * (1.0 - 1e-6),
                "{label}: sampled token {id} (w={:e}) is below the device cutoff {thr:e}",
                w[id]
            );
            if w[id] < t_star {
                in_band += 1;
            }
            distinct.insert(id);
        }
        if expect_exact {
            assert_eq!(
                in_band, 0,
                "{label}: no draw may fall outside the CPU reference nucleus"
            );
        }
        eprintln!(
            "{label}: top_p={top_p} min_p={min_p_rel} T={temperature} |cpu nucleus|={} \
             |band|={} cutoff={thr:e} (exact {t_star:e}) excess_mass={excess:e} \
             draws_in_band={in_band}/{n_draws} distinct={}",
            cpu_set.len(),
            band.len(),
            distinct.len()
        );
    }
    Ok(())
}

/// Per-token cost of the top-p threshold search, measured in the shape
/// production uses (`HeterogeneousEngine::sample_next`: host u01 upload,
/// the kernel chain, a stream sync, a 4-byte readback).
///
/// `cargo test --release --test sampler -- --ignored --nocapture bench_sampler_topp`
#[test]
#[ignore]
fn bench_sampler_topp_cost() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let sampler = Sampler::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut rng = SamplerRng::new(0xBE_11_CE);
    let logits = peaked_logits(&mut rng, 64, 10.0, -8.0);

    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;
    let mut d_max: DeviceBuffer<f32> = DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_z: DeviceBuffer<f32> = DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_u01: DeviceBuffer<f32> = DeviceBuffer::new(device.id, 1)?;
    let mut d_out: DeviceBuffer<i32> = DeviceBuffer::new(device.id, 1)?;
    let mut topp = ToppBufs::new(device.id)?;
    d_logits.copy_from_host(&logits)?;

    let iters = 2000usize;
    let warmup = 200usize;

    let mut run = |top_p: f32, n: usize, rng: &mut SamplerRng| -> eyre::Result<Vec<f64>> {
        let mut samples = Vec::with_capacity(n);
        let mut got = [0i32; 1];
        for _ in 0..n {
            let u = rng.next_f32();
            let t0 = std::time::Instant::now();
            d_u01.copy_from_host(&[u])?;
            if top_p >= 1.0 {
                sampler.launch_multinomial(
                    &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z, &d_u01,
                    N_VOCAB, 1.0, 0.0,
                )?;
            } else {
                sampler.launch_multinomial_topp(
                    &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z,
                    &mut topp.mass, &mut topp.bracket, &mut topp.thr, &d_u01,
                    N_VOCAB, 1.0, 0.0, top_p,
                )?;
            }
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(samples)
    };

    fn stats(mut v: Vec<f64>) -> (f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 2], v[0])
    }

    // Interleaved A/B so thermal / contention drift hits both arms equally.
    let _ = run(1.0, warmup, &mut rng)?;
    let _ = run(0.95, warmup, &mut rng)?;
    let mut off = Vec::new();
    let mut on = Vec::new();
    for _ in 0..10 {
        off.extend(run(1.0, iters / 10, &mut rng)?);
        on.extend(run(0.95, iters / 10, &mut rng)?);
    }
    // Second measurement: submit CHAIN chains back-to-back and sync once.
    // The per-token stream sync + 4-byte readback is paid identically with
    // and without top-p, and on a contended GPU it dominates the wall time;
    // batching isolates the marginal submit+execute cost of the extra
    // kernels, which is what "added per-token cost" actually means.
    const CHAIN: usize = 16;
    let mut run_batched = |top_p: f32, reps: usize| -> eyre::Result<f64> {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            for _ in 0..CHAIN {
                if top_p >= 1.0 {
                    sampler.launch_multinomial(
                        &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z, &d_u01,
                        N_VOCAB, 1.0, 0.0,
                    )?;
                } else {
                    sampler.launch_multinomial_topp(
                        &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z,
                        &mut topp.mass, &mut topp.bracket, &mut topp.thr, &d_u01,
                        N_VOCAB, 1.0, 0.0, top_p,
                    )?;
                }
            }
            stream.synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / CHAIN as f64;
            if us < best {
                best = us;
            }
        }
        Ok(best)
    };
    let _ = run_batched(1.0, 5)?;
    let _ = run_batched(0.95, 5)?;
    let b_off = run_batched(1.0, 300)?;
    let b_on = run_batched(0.95, 300)?;
    eprintln!(
        "sampler chain, sync amortised (us/token, best-of-40): top_p=1.0 {b_off:.1} |          top_p=0.95 {b_on:.1} | delta {:.1} = {:.4}% of a 38 ms token",
        b_on - b_off,
        (b_on - b_off) / 38_000.0 * 100.0
    );

    // Third measurement: HIP events bracketing ONE chain, min over many
    // reps. Event timestamps are GPU-side, so this excludes host launch
    // latency; the minimum picks the rep whose kernels were least
    // interleaved with the other process on the device, which is the
    // closest thing to an idle-GPU number available while the production
    // server is running.
    let ev_a = v4flash_hip::Event::new()?;
    let ev_b = v4flash_hip::Event::new()?;
    let mut run_evented = |top_p: f32, reps: usize| -> eyre::Result<f64> {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            ev_a.record(&stream)?;
            if top_p >= 1.0 {
                sampler.launch_multinomial(
                    &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z, &d_u01,
                    N_VOCAB, 1.0, 0.0,
                )?;
            } else {
                sampler.launch_multinomial_topp(
                    &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z,
                    &mut topp.mass, &mut topp.bracket, &mut topp.thr, &d_u01,
                    N_VOCAB, 1.0, 0.0, top_p,
                )?;
            }
            ev_b.record(&stream)?;
            ev_b.synchronize()?;
            let us = v4flash_hip::Event::elapsed_ms(&ev_a, &ev_b)? as f64 * 1e3;
            if us > 0.0 && us < best {
                best = us;
            }
        }
        Ok(best)
    };
    let _ = run_evented(1.0, 50)?;
    let _ = run_evented(0.95, 50)?;
    let e_off = run_evented(1.0, 2000)?;
    let e_on = run_evented(0.95, 2000)?;
    eprintln!(
        "sampler chain, GPU events (us/token, min-of-2000): top_p=1.0 {e_off:.1} |          top_p=0.95 {e_on:.1} | delta {:.1} = {:.4}% of a 38 ms token",
        e_on - e_off,
        (e_on - e_off) / 38_000.0 * 100.0
    );

    let (off_p50, off_p10) = stats(off);
    let (on_p50, on_p10) = stats(on);
    eprintln!(
        "sampler per-token, incl. sync (us): top_p=1.0 p50={off_p50:.1} min={off_p10:.1} | \
         top_p=0.95 p50={on_p50:.1} min={on_p10:.1} | delta p50={:.1} min={:.1}",
        on_p50 - off_p50,
        on_p10 - off_p10
    );
    // 38 ms/token decode budget.
    eprintln!(
        "added cost vs 38 ms/token decode: p50 {:.4}% min {:.4}%",
        (on_p50 - off_p50) / 38_000.0 * 100.0,
        (on_p10 - off_p10) / 38_000.0 * 100.0
    );
    Ok(())
}

// ===========================================================================
// golden checks on the kernels the top-p change actually touched
// ===========================================================================

/// Inverse-CDF golden check on the LEGACY (`top_p = 1.0`) chain.
///
/// `sampler_topp_one_delegates_to_legacy_path` only proves the wrapper
/// dispatches; it cannot see whether the *modified* `logits_expsum_partial`
/// and `softmax_sample_one` still implement the pre-change semantics, because
/// both of its arms run the same code. This one pins the semantics directly:
/// for the drawn `u`, the returned token's cumulative interval in an f64
/// host reference must contain `u * Z`, where `Z` is renormalised over the
/// min-p survivors and the walk runs in index order — exactly what the
/// kernels claim to do.
///
/// Any of the plausible regressions on this path (dropping the renorm,
/// reading the wrong `Z`, mis-handling the new `thr_dev == nullptr` branch,
/// falling into the `found < 0` fallback) moves `target` outside the returned
/// token's interval by far more than the tolerance.
#[test]
#[ignore]
fn sampler_legacy_chain_matches_cpu_inverse_cdf() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let sampler = Sampler::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;
    let mut d_max: DeviceBuffer<f32> = DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_z: DeviceBuffer<f32> = DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_u01: DeviceBuffer<f32> = DeviceBuffer::new(device.id, 1)?;
    let mut d_out: DeviceBuffer<i32> = DeviceBuffer::new(device.id, 1)?;

    let mut rng = SamplerRng::new(0xC0FFEE);
    // (label, logits, temperature, min_p_rel)
    let cases: Vec<(&str, Vec<f32>, f32, f32)> = vec![
        ("peaked", peaked_logits(&mut rng, 24, 9.0, -20.0), 1.0, 0.0),
        ("peaked-minp", peaked_logits(&mut rng, 24, 9.0, -20.0), 1.0, 0.1),
        ("dense-tail", peaked_logits(&mut rng, 24, 9.0, -6.0), 1.0, 0.0),
        ("cold", peaked_logits(&mut rng, 24, 9.0, -20.0), 0.7, 0.0),
        ("wide-head-minp", peaked_logits(&mut rng, 512, 14.0, -8.0), 1.0, 0.05),
    ];

    for (label, logits, temperature, min_p_rel) in cases {
        d_logits.copy_from_host(&logits)?;
        let w = cpu_weights(&logits, temperature as f64);
        let thr = min_p_rel as f64;
        // Exclusive-prefix CDF over the survivors, in index order.
        let mut cum = vec![0f64; w.len() + 1];
        for i in 0..w.len() {
            cum[i + 1] = cum[i] + if w[i] >= thr { w[i] } else { 0.0 };
        }
        let z = cum[w.len()];
        assert!(z > 0.0, "{label}: empty survivor set");
        // The device accumulates the inter-chunk prefix in f32 (256 chunk
        // sums), so allow a slack far below any semantic error.
        let tol = 1e-5 * z;

        let mut got = [0i32; 1];
        for trial in 0..512 {
            let u = rng.next_f32();
            d_u01.copy_from_host(&[u])?;
            sampler.launch_multinomial(
                &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z, &d_u01,
                N_VOCAB, temperature, min_p_rel,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;
            let id = got[0];
            assert!(id >= 0 && (id as usize) < w.len(), "{label}: token {id} out of range");
            let id = id as usize;
            assert!(
                w[id] >= thr,
                "{label}: trial {trial} drew pruned token {id} (w={:e} < min_p {thr:e})",
                w[id]
            );
            let target = u as f64 * z;
            assert!(
                target >= cum[id] - tol && target <= cum[id + 1] + tol,
                "{label}: trial {trial} u={u} target={target:e} outside token {id}'s \
                 CDF interval [{:e}, {:e}] (Z={z:e})",
                cum[id],
                cum[id + 1]
            );
        }
        eprintln!("{label}: 512 draws inside their CPU CDF interval (T={temperature}, min_p={min_p_rel})");
    }
    Ok(())
}

/// Marginal-frequency check for the TOP-P path — the sampling *distribution*,
/// not just nucleus membership.
///
/// `sampler_topp_nucleus_matches_cpu_reference` only asserts that every draw
/// lands inside the nucleus, which a badly-renormalised sampler still
/// satisfies: e.g. if step 4 (`logits_expsum_partial` re-run with `topp_thr`)
/// were skipped, `softmax_sample_one` would walk to `u * Z_full` while
/// summing only survivors, run off the end of the vocabulary for almost every
/// `u`, and return the `found < 0` fallback — the LAST surviving index, which
/// is still inside the nucleus. That collapses the sampler to near-greedy on
/// one arbitrary token and this test catches it immediately.
///
/// The reference set is derived from the cutoff the device actually published
/// (`topp_thr`), so a boundary that f32 can legitimately break either way
/// (see the dyadic case) does not make the test flaky, while the *shape* of
/// the distribution over whichever set was chosen is still pinned exactly.
#[test]
#[ignore]
fn sampler_topp_frequencies_match_renormalised_nucleus() -> eyre::Result<()> {
    install_panic_handler()?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let sampler = Sampler::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_logits: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_VOCAB as usize)?;
    let mut d_max: DeviceBuffer<f32> = DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_z: DeviceBuffer<f32> = DeviceBuffer::new(device.id, SAMPLER_N_WG as usize)?;
    let mut d_u01: DeviceBuffer<f32> = DeviceBuffer::new(device.id, 1)?;
    let mut d_out: DeviceBuffer<i32> = DeviceBuffer::new(device.id, 1)?;
    let mut topp = ToppBufs::new(device.id)?;

    /// Stamp `head` into the front of an otherwise-negligible vocabulary.
    fn head_logits(head: &[f32], tail: f32) -> Vec<f32> {
        let mut v = vec![tail; N_VOCAB as usize];
        for (i, &x) in head.iter().enumerate() {
            v[i] = x;
        }
        v
    }

    let ln2 = std::f64::consts::LN_2 as f32;
    // (label, head, tail, temperature, top_p, min_p_rel, expected nucleus size)
    //
    // `dyadic-exact-boundary` has weights 1, 1/2, 1/4, 1/8, 1/8 (Z = 2), so
    // top_p = 0.75 puts the cumulative EXACTLY on the 2-token boundary:
    // `>=` must keep 2 tokens. f32 can round that tie either way on-device,
    // so the assertion below accepts 2 or 5 and then checks the frequencies
    // against whichever set the published cutoff actually selected.
    let cases: Vec<(&str, Vec<f32>, f32, f32, f32, f32, &[usize])> = vec![
        // w = [1, .6065, .3679, .2231, .1353, .0821, .0498, .0302], Z = 2.4949.
        // top_p .6 -> target 1.497, cum 1 | 1.6065 -> 2 tokens.
        ("head8-truncating", vec![3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5], -30.0, 1.0, 0.6, 0.0, &[2]),
        // top_p .9 -> target 2.2454, cum reaches it at 2.3328 -> 5 tokens.
        ("head8-wide", vec![3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5], -30.0, 1.0, 0.9, 0.0, &[5]),
        // top_p .95 alone would keep 6; min_p .1 is tighter and binds -> 5.
        ("head8-topp-plus-minp", vec![3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5], -30.0, 1.0, 0.95, 0.1, &[5]),
        // T = .7 sharpens: w = [1, .4895, .2397, .1173, ...], Z = 1.9525,
        // target 1.562, cum 1 | 1.4895 | 1.7292 -> 3 tokens. (Also pins that
        // the nucleus is taken on the TEMPERED distribution: at T = 1 the
        // same top_p would keep 4.)
        ("head8-cold", vec![3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5], -30.0, 0.7, 0.8, 0.0, &[3]),
        // w = [1, 1/2, 1/4, 1/8, 1/8], Z = 2, target = 1.5 == cum after 2
        // tokens exactly. `>=` keeps 2; if f32 breaks the tie the other way
        // the next admissible set is 3 (cum 1.75).
        ("dyadic-exact-boundary", vec![0.0, -ln2, -2.0 * ln2, -3.0 * ln2, -3.0 * ln2], -60.0, 1.0, 0.75, 0.0, &[2, 3]),
    ];

    let n_samples = 20_000u64;
    let mut rng = SamplerRng::new(0x70_99_5A_11);
    for (label, head, tail, temperature, top_p, min_p_rel, expect_sizes) in cases {
        let logits = head_logits(&head, tail);
        d_logits.copy_from_host(&logits)?;
        let w = cpu_weights(&logits, temperature as f64);

        // One run to publish the cutoff the device chose.
        d_u01.copy_from_host(&[0.5f32])?;
        sampler.launch_multinomial_topp(
            &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z,
            &mut topp.mass, &mut topp.bracket, &mut topp.thr, &d_u01,
            N_VOCAB, temperature, min_p_rel, top_p,
        )?;
        stream.synchronize()?;
        let mut thr_h = [0f32; 1];
        topp.thr.copy_to_host(&mut thr_h)?;
        let thr = thr_h[0] as f64;

        let nucleus: Vec<usize> = (0..w.len()).filter(|&i| w[i] >= thr).collect();
        assert!(
            expect_sizes.contains(&nucleus.len()),
            "{label}: nucleus size {} not in {expect_sizes:?} (cutoff {thr:e})",
            nucleus.len()
        );
        assert!(
            nucleus.iter().all(|&i| i < head.len()),
            "{label}: nucleus leaked into the tail: {nucleus:?}"
        );
        // Renormalised target distribution over exactly that set.
        let z_nuc: f64 = nucleus.iter().map(|&i| w[i]).sum();
        let mut expected = vec![0f64; head.len()];
        for &i in &nucleus {
            expected[i] = w[i] / z_nuc;
        }

        let mut counts = vec![0u64; head.len()];
        let mut outside = 0u64;
        let mut got = [0i32; 1];
        for _ in 0..n_samples {
            let u = rng.next_f32();
            d_u01.copy_from_host(&[u])?;
            sampler.launch_multinomial_topp(
                &stream, &mut d_out, &d_logits, &mut d_max, &mut d_z,
                &mut topp.mass, &mut topp.bracket, &mut topp.thr, &d_u01,
                N_VOCAB, temperature, min_p_rel, top_p,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;
            let id = got[0];
            if id >= 0 && (id as usize) < head.len() && expected[id as usize] > 0.0 {
                counts[id as usize] += 1;
            } else {
                outside += 1;
            }
        }
        assert_eq!(
            outside, 0,
            "{label}: {outside}/{n_samples} draws fell outside the nucleus {nucleus:?}"
        );
        eprintln!(
            "{label}: top_p={top_p} min_p={min_p_rel} T={temperature} cutoff={thr:e} \
             |nucleus|={} counts={counts:?}",
            nucleus.len()
        );
        for i in 0..head.len() {
            let observed = counts[i] as f64 / n_samples as f64;
            let diff = (observed - expected[i]).abs();
            eprintln!(
                "  token {i}: observed={:.4} expected={:.4} |Δ|={:.4}",
                observed, expected[i], diff
            );
            assert!(
                diff < 0.02,
                "{label}: token {i} marginal off by {diff:.4} \
                 (observed {observed:.4} vs expected {:.4})",
                expected[i]
            );
        }
    }
    Ok(())
}
