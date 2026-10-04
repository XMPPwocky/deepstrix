//! Decode-head candidates (`vec_add.hip` `logits_nucleus_cands`, the device
//! half of `HeterogeneousEngine::head_cands`): per logit row its max, the
//! survivors' weight sum and every (logit, id) within the band of the max,
//! against a host computation -- on real V4.1 logits (the golden agentic
//! fixture's `logits_all.f32`, when present) and synthetic edge cases (ties,
//! NaN, -inf, a flat row past the cap, argmax params). The host half
//! (`TargetDist::from_cands`) is gated by `spec_sample` G-RS3.
//!
//! No model weights, ~10 MB of VRAM: runs on the dGPU beside a live hub.
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels \
//!   --release --features v41 --test head_cands -- --ignored --nocapture
//! ```

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::N_VOCAB;
use v4flash_kernels::het::scratch::{HEAD_BATCH_MAX, HEAD_CAND_BAND, HEAD_CAND_CAP, HEAD_CAND_STRIDE};
use v4flash_kernels::VecAddInplace;

fn pick(prefix: &str) -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(prefix) {
            return Ok(d);
        }
    }
    Err(eyre!("no {prefix} device"))
}

/// The host sampler's view of a row: max, survivor total (exact, f64 log-
/// weights), candidates (sorted by id), the tokens near the band edge (where
/// device FMA rounding of `x * inv_t - gmax` may decide differently; ~ulp of
/// `x * inv_t`, larger for large logits), and the log-weight error bound.
fn host_ref(row: &[f32], inv_t: f32, lo: f32, band: f32) -> (f32, f64, Vec<(u32, u32)>, Vec<u32>, f64) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let gmax = max * inv_t;
    let amax = row.iter().filter(|x| x.is_finite()).fold(0f32, |a, &x| a.max(x.abs()));
    let err = 1e-5 + 4.0 * (amax * inv_t) as f64 * f32::EPSILON as f64;
    let mut z = 0f64;
    let mut cands = Vec::new();
    let mut edge = Vec::new();
    for (i, &x) in row.iter().enumerate() {
        let l = x * inv_t - gmax;
        let l64 = x as f64 * inv_t as f64 - gmax as f64;
        if l >= lo {
            z += l64.exp();
        }
        if l >= -band {
            cands.push((x.to_bits(), i as u32));
        }
        if (l64 + band as f64).abs() < err {
            edge.push(i as u32);
        }
    }
    (max, z, cands, edge, err)
}

#[test]
#[ignore]
fn nucleus_cands_match_the_host() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick("gfx1201")?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let k = VecAddInplace::for_arch(&arch)?;
    let s = Stream::new(dev.id)?;
    let nv = N_VOCAB as usize;
    let rows_max = HEAD_BATCH_MAX;

    // Rows: real logits first (the fixture is ~1,000 rows of 129,280 f32).
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let fixture = format!("{}/.cache/deepstrix/goldens/agentic/logits_all.f32", std::env::var("HOME")?);
    if let Ok(bytes) = std::fs::read(&fixture) {
        let all: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        for r in (0..all.len() / nv).step_by(7).take(96) {
            rows.push(all[r * nv..(r + 1) * nv].to_vec());
        }
        println!("{} real rows from {fixture}", rows.len());
    } else {
        println!("no fixture at {fixture}: synthetic rows only");
    }
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = || {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    for case in 0..8 {
        let row: Vec<f32> = (0..nv)
            .map(|i| match case {
                0 => rnd() * 20.0 - 10.0,                              // broad
                1 => (rnd() * 8.0).floor(),                            // heavy ties
                2 => if i % 9973 == 0 { 15.0 } else { rnd() * 4.0 - 8.0 }, // tied peaks
                3 => if i % 5 == 0 { f32::NAN } else { rnd() * 6.0 },  // NaNs
                4 => if i % 3 == 0 { f32::NEG_INFINITY } else { rnd() }, // -inf
                5 => rnd() * 0.5,                                      // flat: past the cap
                6 => -1e4 + rnd(),                                     // huge negatives
                _ => if i == 4242 { 30.0 } else { rnd() * 2.0 },       // one spike
            })
            .collect();
        rows.push(row);
    }

    // Param sets: (inv_t, lo, band); the last is argmax.
    let floor_ln = 1e-10f32.ln();
    let psets: [(f32, f32, f32); 5] = [
        (1.0, floor_ln, HEAD_CAND_BAND),
        (1.0 / 0.7, floor_ln, HEAD_CAND_BAND),
        (1.0 / 0.3, floor_ln, HEAD_CAND_BAND),
        (1.0, 0.05f32.ln(), HEAD_CAND_BAND),
        (1.0, f32::INFINITY, 0.0),
    ];
    let mut logits = DeviceBuffer::<f32>::new(dev.id, rows_max * nv)?;
    let mut params = DeviceBuffer::<f32>::new(dev.id, rows_max * 3)?;
    let mut out = DeviceBuffer::<u32>::new(dev.id, rows_max * HEAD_CAND_STRIDE)?;
    let mut host_out = vec![0u32; rows_max * HEAD_CAND_STRIDE];
    let (mut checked, mut over_cap, mut edge_skips, mut worst_z) = (0usize, 0usize, 0usize, 0f64);
    let mut best_us = f64::MAX;
    for (pi, &(inv_t, lo, band)) in psets.iter().enumerate() {
        for chunk in rows.chunks(rows_max) {
            let n = chunk.len();
            let flat: Vec<f32> = chunk.iter().flatten().copied().collect();
            logits.slice_view_mut(0, n * nv).copy_from_host(&flat)?;
            let p: Vec<f32> = (0..n).flat_map(|_| [inv_t, lo, band]).collect();
            params.slice_view_mut(0, n * 3).copy_from_host(&p)?;
            s.synchronize()?;
            let t = std::time::Instant::now();
            k.launch_nucleus_cands(&s, &mut out, &logits, &params, n as u32, N_VOCAB, HEAD_CAND_CAP as u32)?;
            s.synchronize()?;
            if n == rows_max.min(8) || n >= 8 {
                best_us = best_us.min(t.elapsed().as_secs_f64() * 1e6 * 8.0 / n as f64);
            }
            out.slice_view(0, n * HEAD_CAND_STRIDE).copy_to_host(&mut host_out[..n * HEAD_CAND_STRIDE])?;
            for (r, row) in chunk.iter().enumerate() {
                let o = &host_out[r * HEAD_CAND_STRIDE..(r + 1) * HEAD_CAND_STRIDE];
                let (max, z, want, edge, err) = host_ref(row, inv_t, lo, band);
                let count = o[0] as usize;
                assert_eq!(o[1], max.to_bits(), "pset {pi} row {r}: max");
                // Count: exact up to edge tokens.
                assert!(count.abs_diff(want.len()) <= edge.len(), "pset {pi} row {r}: count {count} vs {} (edge {})", want.len(), edge.len());
                if lo.is_finite() {
                    let zd = f32::from_bits(o[2]) as f64;
                    // Relative error: f32 accumulation (~1e-6) plus the
                    // log-weight rounding bound (a relative error of each term).
                    let rel = (zd - z).abs() / z.max(1e-30);
                    worst_z = worst_z.max(rel);
                    assert!(rel < 1e-5 + err, "pset {pi} row {r}: z {zd} vs {z} (rel {rel:e}, bound {:e})", 1e-5 + err);
                }
                if count > HEAD_CAND_CAP {
                    over_cap += 1;
                    continue;
                }
                let mut got: Vec<(u32, u32)> = (0..count).map(|j| (o[4 + 2 * j], o[5 + 2 * j])).collect();
                got.sort_unstable_by_key(|c| c.1);
                if got != want {
                    // Only band-edge tokens may differ.
                    let g: std::collections::BTreeSet<_> = got.iter().copied().collect();
                    let w: std::collections::BTreeSet<_> = want.iter().copied().collect();
                    for d in g.symmetric_difference(&w) {
                        assert!(edge.contains(&d.1), "pset {pi} row {r}: candidate {d:?} differs and is not at the band edge");
                    }
                    edge_skips += 1;
                }
                checked += 1;
            }
        }
    }
    println!(
        "logits_nucleus_cands: {checked} rows exact (+{edge_skips} differing only at the band edge), {over_cap} past the cap, \
         worst z rel {worst_z:.2e}, ~{best_us:.0} us per 8 rows"
    );
    assert!(checked > 300);
    Ok(())
}
