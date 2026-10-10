//! The drafter's attention (`attention_mixed_score` + `attention_mixed_softmax_wsum`,
//! `MtpState::attn`) over `[ring ++ block]` must not care WHERE in the ring a row
//! sits: rotating the 128 ring rows (the block's 5 rows stay last) may change the
//! output only by f32 reassociation. Compared against an f64 CPU reference.
//!
//! No model weights (~22 MB of scratch on the iGPU): safe beside the live hub. Run:
//! ```text
//! CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels \
//!   --release --features v41 --test attn_mixed_ring_rotation -- --ignored --nocapture
//! ```
#![cfg(feature = "v41")]

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::attention::{AttentionMixed, ATTN_MIXED_MAX_KEYS};
use v4flash_kernels::config::{N_HEAD, N_HEAD_DIM};

const WINDOW: usize = 128;
const BLOCK: usize = 5;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

fn rows(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * scale
        })
        .collect()
}

/// f32 -> IEEE f16 bits, round to nearest even (values here are normal, |v| < 4).
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        return sign; // flush tiny values to zero (exactly representable below)
    }
    let mut h = ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

fn f16_to_f64(b: u16) -> f64 {
    let sign = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 10) & 0x1f) as i32;
    let man = (b & 0x3ff) as f64;
    if exp == 0 {
        return sign * man * 2f64.powi(-24);
    }
    sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15)
}

/// f64 reference: per head, softmax over the keys plus a sink (denominator only), weighted sum of V = K.
fn reference(q: &[f32], kv: &[u16], sinks: &[f32], n_kv: usize) -> Vec<f64> {
    let (nh, d) = (N_HEAD as usize, N_HEAD_DIM as usize);
    let k: Vec<f64> = kv.iter().map(|&b| f16_to_f64(b)).collect();
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0f64; nh * d];
    for h in 0..nh {
        let qh = &q[h * d..(h + 1) * d];
        let s: Vec<f64> = (0..n_kv).map(|r| (0..d).map(|i| qh[i] as f64 * k[r * d + i]).sum::<f64>() * scale).collect();
        let m = s.iter().cloned().fold(sinks[h] as f64, f64::max);
        let den: f64 = s.iter().map(|x| (x - m).exp()).sum::<f64>() + (sinks[h] as f64 - m).exp();
        for r in 0..n_kv {
            let w = (s[r] - m).exp() / den;
            for i in 0..d {
                out[h * d + i] += w * k[r * d + i];
            }
        }
    }
    out
}

#[test]
#[ignore]
fn ring_rotation_changes_attention_only_by_reassociation() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_igpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let attn = AttentionMixed::for_arch(&arch)?;
    let stream = Stream::new(dev.id)?;
    let (nh, d) = (N_HEAD as usize, N_HEAD_DIM as usize);
    let n_kv = WINDOW + BLOCK;
    // Peaked scores (scale 4 on q) like a trained head, and a moderate sink.
    let q = rows(1, nh * d, 4.0);
    let sinks = rows(2, nh, 2.0);
    let ring: Vec<u16> = rows(3, n_kv * d, 2.0).into_iter().map(f16_bits).collect();
    let mut q_d = DeviceBuffer::<f32>::new(dev.id, nh * d)?;
    q_d.copy_from_host(&q)?;
    let mut s_d = DeviceBuffer::<f32>::new(dev.id, nh)?;
    s_d.copy_from_host(&sinks)?;
    let mut scores = DeviceBuffer::<f32>::new(dev.id, nh * ATTN_MIXED_MAX_KEYS as usize)?;
    let mut out_d = DeviceBuffer::<f32>::new(dev.id, nh * d)?;
    let mut kv_d = DeviceBuffer::<u16>::new(dev.id, n_kv * d)?;

    let mut run = |kv: &[u16]| -> eyre::Result<Vec<f32>> {
        kv_d.copy_from_host(kv)?;
        attn.launch_score(&stream, &mut scores, &q_d, &kv_d, None, N_HEAD, N_HEAD_DIM, n_kv as u32, 0)?;
        attn.launch_softmax_wsum(&stream, &mut out_d, &mut scores, &s_d, &kv_d, None, N_HEAD, N_HEAD_DIM, n_kv as u32, 0)?;
        stream.synchronize()?;
        let mut o = vec![0f32; nh * d];
        out_d.copy_to_host(&mut o)?;
        Ok(o)
    };
    let rel = |a: &[f32], b: &[f64]| -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - y).powi(2)).sum();
        let den: f64 = b.iter().map(|y| y * y).sum();
        (num / den).sqrt()
    };
    let rel32 = |a: &[f32], b: &[f32]| -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
        (num / den).sqrt()
    };

    let base = run(&ring)?;
    let want = reference(&q, &ring, &sinks, n_kv);
    let base_err = rel(&base, &want);
    let mut worst = 0f64;
    let mut worst_ref = base_err;
    for shift in [1usize, 5, 28, 64, 96, 127] {
        // Rotate the ring part only; the block's rows stay at [WINDOW, WINDOW + BLOCK).
        let mut rot = Vec::with_capacity(ring.len());
        for r in 0..WINDOW {
            let src = (r + shift) % WINDOW;
            rot.extend_from_slice(&ring[src * d..(src + 1) * d]);
        }
        rot.extend_from_slice(&ring[WINDOW * d..]);
        let o = run(&rot)?;
        let vs_base = rel32(&o, &base);
        let vs_ref = rel(&o, &reference(&q, &rot, &sinks, n_kv));
        println!("shift {shift:3}: ||out_rot - out||/||out|| = {vs_base:.2e}; vs f64 reference {vs_ref:.2e}");
        worst = worst.max(vs_base);
        worst_ref = worst_ref.max(vs_ref);
    }
    println!("unrotated vs f64 reference {base_err:.2e}; worst rotation {worst:.2e}; worst vs reference {worst_ref:.2e}");
    if worst > 1e-5 || worst_ref > 1e-4 {
        return Err(eyre!("attention depends on ring slot order beyond reassociation ({worst:.2e} rotation, {worst_ref:.2e} vs reference)"));
    }
    Ok(())
}
