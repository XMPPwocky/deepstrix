//! `router_topk` alternatives (`V41_ROUTER_ALTS`, docs/v41/BOX2_MISS_SUBSTITUTION.md).
//!
//! The router can also emit each token's next `n_alt` ranks. That must not move
//! the picks or the weights by a single bit, since those alternatives are
//! computed after the first `n_used`. This checks that, and that the
//! alternatives really are ranks 7..6+n_alt.
//!
//! Runs on the iGPU (gfx1151); skips when there is none.
#![cfg(feature = "v41")]

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{EXPERT_WEIGHT_SCALE, N_EXPERT, N_EXPERT_USED};
use v4flash_kernels::router_topk::{RouterTopk, ROUTER_MAX_ALT};

const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;
const B: usize = 257;

fn pick(prefix: &str) -> Option<Device> {
    Device::all().ok()?.into_iter().find(|d| {
        d.properties().map(|p| p.gcn_arch_name.starts_with(prefix)).unwrap_or(false)
    })
}

/// Deterministic xorshift, so a failure reproduces.
fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
    }
}

struct Out {
    sel: Vec<i32>,
    ew: Vec<f32>,
    alts: Vec<i32>,
}

#[allow(clippy::too_many_arguments)]
fn run(r: &RouterTopk, s: &Stream, dev: i32, logits: &DeviceBuffer<f32>, bias: &DeviceBuffer<f32>, n_alt: u32) -> eyre::Result<Out> {
    let nu = N_EXPERT_USED;
    let mut sel = DeviceBuffer::<i32>::new(dev, B * nu)?;
    let mut ew = DeviceBuffer::<f32>::new(dev, B * nu)?;
    let mut alts = DeviceBuffer::<i32>::new(dev, B * ROUTER_MAX_ALT as usize)?;
    alts.copy_from_host(&vec![-7; B * ROUTER_MAX_ALT as usize])?;
    r.launch_batched_alts(
        s, &mut sel, &mut ew, logits, Some(bias), N_EXPERT, nu as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS, B as u32,
        if n_alt > 0 { Some(&mut alts) } else { None }, n_alt,
    )?;
    s.synchronize()?;
    let mut o = Out { sel: vec![0; B * nu], ew: vec![0.0; B * nu], alts: vec![0; B * n_alt as usize] };
    sel.copy_to_host(&mut o.sel)?;
    ew.copy_to_host(&mut o.ew)?;
    if n_alt > 0 {
        alts.slice_view(0, o.alts.len()).copy_to_host(&mut o.alts)?;
    }
    Ok(o)
}

#[test]
fn alts_leave_picks_bit_identical_and_are_the_next_ranks() -> eyre::Result<()> {
    let Some(dev) = pick("gfx1151") else {
        eprintln!("no gfx1151 device; skipping");
        return Ok(());
    };
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let s = Stream::new(dev.id)?;
    let ne = N_EXPERT as usize;
    let nu = N_EXPERT_USED;

    let mut g = rng(0x9e3779b97f4a7c15);
    let logits_h: Vec<f32> = (0..B * ne).map(|_| g()).collect();
    // Small bias, like the checkpoint's: it reorders near-ties, not the bulk.
    let bias_h: Vec<f32> = (0..ne).map(|_| g() * 0.02).collect();
    let mut logits = DeviceBuffer::<f32>::new(dev.id, B * ne)?;
    logits.copy_from_host(&logits_h)?;
    let mut bias = DeviceBuffer::<f32>::new(dev.id, ne)?;
    bias.copy_from_host(&bias_h)?;

    let par = RouterTopk::for_arch(&arch)?;
    let base = run(&par, &s, dev.id, &logits, &bias, 0)?;
    for n_alt in 1..=ROUTER_MAX_ALT {
        let o = run(&par, &s, dev.id, &logits, &bias, n_alt)?;
        // 1. Picks and weights: bit-identical.
        assert_eq!(o.sel, base.sel, "n_alt={n_alt}: picks moved");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&o.ew), bits(&base.ew), "n_alt={n_alt}: weights moved");

        // 2. Alternatives are the next ranks by the host's f64 selection score.
        let na = n_alt as usize;
        for t in 0..B {
            let score = |e: usize| -> f64 {
                let x = logits_h[t * ne + e] as f64;
                let sp = if x > 20.0 { x } else if x < -20.0 { x.exp() } else { x.exp().ln_1p() };
                sp.sqrt() + bias_h[e] as f64
            };
            let row = &o.sel[t * nu..(t + 1) * nu];
            let alts = &o.alts[t * na..(t + 1) * na];
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|&a, &b| score(b).partial_cmp(&score(a)).unwrap().then(a.cmp(&b)));
            for (k, &a) in alts.iter().enumerate() {
                assert!((0..ne as i32).contains(&a), "t={t}: alt {a} out of range");
                assert!(!row.contains(&a), "t={t}: alt {a} duplicates a pick");
                assert!(!alts[..k].contains(&a), "t={t}: alt {a} repeated");
                let want = order[nu + k];
                // f32 vs f64 may reorder a genuine near-tie; nothing else.
                if a as usize != want {
                    let gap = (score(a as usize) - score(want)).abs();
                    assert!(gap < 1e-5, "t={t} rank {}: got {a}, want {want} (gap {gap:e})", nu + k + 1);
                }
            }
        }
    }

    // 3. The serial reference kernel agrees on picks and alternatives.
    let serial = RouterTopk::for_arch_serial(&arch)?;
    let mut sel = DeviceBuffer::<i32>::new(dev.id, nu)?;
    let mut ew = DeviceBuffer::<f32>::new(dev.id, nu)?;
    let one = logits.slice_view(0, ne);
    let full = run(&par, &s, dev.id, &logits, &bias, ROUTER_MAX_ALT)?;
    serial.launch(&s, &mut sel, &mut ew, &one, Some(&bias), N_EXPERT, nu as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS)?;
    s.synchronize()?;
    let mut sel_h = vec![0i32; nu];
    sel.copy_to_host(&mut sel_h)?;
    assert_eq!(sel_h, full.sel[..nu], "serial vs parallel picks (token 0)");
    Ok(())
}

/// The image-row path relaunches a sub-range with slice views; the
/// alternatives must land at that sub-range's offset.
#[test]
fn alts_slice_views_land_at_the_right_rows() -> eyre::Result<()> {
    let Some(dev) = pick("gfx1151") else {
        eprintln!("no gfx1151 device; skipping");
        return Ok(());
    };
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let s = Stream::new(dev.id)?;
    let (ne, nu, na) = (N_EXPERT as usize, N_EXPERT_USED, ROUTER_MAX_ALT as usize);
    let mut g = rng(7);
    let logits_h: Vec<f32> = (0..B * ne).map(|_| g()).collect();
    let mut logits = DeviceBuffer::<f32>::new(dev.id, B * ne)?;
    logits.copy_from_host(&logits_h)?;
    let mut bias = DeviceBuffer::<f32>::new(dev.id, ne)?;
    bias.copy_from_host(&vec![0.0; ne])?;
    let par = RouterTopk::for_arch(&arch)?;
    let full = run(&par, &s, dev.id, &logits, &bias, na as u32)?;

    let (r0, n) = (100usize, 37usize);
    let mut sel = DeviceBuffer::<i32>::new(dev.id, B * nu)?;
    let mut ew = DeviceBuffer::<f32>::new(dev.id, B * nu)?;
    let mut alts = DeviceBuffer::<i32>::new(dev.id, B * na)?;
    let lv = logits.slice_view(r0 * ne, n * ne);
    let mut sv = sel.slice_view_mut(r0 * nu, n * nu);
    let mut ev = ew.slice_view_mut(r0 * nu, n * nu);
    let mut av = alts.slice_view_mut(r0 * na, n * na);
    par.launch_batched_alts(&s, &mut sv, &mut ev, &lv, Some(&bias), N_EXPERT, nu as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS, n as u32, Some(&mut av), na as u32)?;
    s.synchronize()?;
    let mut got = vec![0i32; n * na];
    av.copy_to_host(&mut got)?;
    assert_eq!(got, full.alts[r0 * na..(r0 + n) * na], "sub-range alternatives");
    Ok(())
}
