//! Is the decode-vs-batched attention divergence NUMERICS or a BUG? (2026-09-22)
//!
//! The fidelity bisect found decode and the batched/arena path bit-identical up
//! to attention's input (`q_normed`, `attn_cur`) at layer 0, then differing at
//! `heads` by a relative L2 of ~2e-4. The two paths use different kernel
//! families: the arena's `*_htiled_wmma_f16s_rows` pair (f16 SCORES) vs decode's
//! `score_b1_htiled_wmma -> softmax_only -> wsum_b1_htiled_ksplit_ldsv ->
//! reduce_partials_apply_inv` (f32 scores). This runs both on the SAME synthetic
//! inputs against a CPU fp64 reference of the same math:
//!
//!   score_r = (q . row_r) / sqrt(head_dim)   rows = raw[raw_off..raw_off+n_raw] ++ comp[..n_comp]
//!   w = softmax over {score_r} with the head's sink logit in the DENOMINATOR only
//!   out = sum_r w_r * row_r                  (MLA: K == V)
//!
//! Verdict rule: if BOTH kernels sit near the reference at their precision's
//! level (f16-score chain ~1e-4..1e-3, f32 chain ~1e-6) the divergence is
//! numerics. A bug (window/offset/mask/sink) shows as one path far off the
//! reference, typically varying with n_raw / raw_off / n_comp.
//!
//! Synthetic data, dGPU only, ~110 MB -- runs beside a live server:
//!   cargo test -p v4flash-kernels --features v41 --release --test attention_vs_reference -- --ignored --nocapture
#![cfg(feature = "v41")]

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::remote_experts::f32_to_f16_bits;
use v4flash_kernels::ATTN_MIXED_MAX_KEYS;

const N_HEAD: u32 = 64;
const HEAD_DIM: u32 = 512;
const K_SPLIT: u32 = 16;

struct Rng(u64);
impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
}

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 device"))
}
fn dev_f32(id: i32, v: &[f32]) -> DeviceBuffer<f32> {
    let mut b = DeviceBuffer::<f32>::new(id, v.len().max(1)).unwrap();
    b.copy_from_host(v).unwrap();
    b
}
fn dev_u16(id: i32, v: &[u16]) -> DeviceBuffer<u16> {
    let mut b = DeviceBuffer::<u16>::new(id, v.len().max(1)).unwrap();
    b.copy_from_host(v).unwrap();
    b
}
fn dev_i32(id: i32, v: &[i32]) -> DeviceBuffer<i32> {
    let mut b = DeviceBuffer::<i32>::new(id, v.len().max(1)).unwrap();
    b.copy_from_host(v).unwrap();
    b
}
fn host<T: Copy + Default>(b: &DeviceBuffer<T>) -> Vec<T> {
    let mut v = vec![T::default(); b.len()];
    b.copy_to_host(&mut v).unwrap();
    v
}
fn f16_to_f64(h: u16) -> f64 {
    half_to_f32(h) as f64
}
fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15)
    };
    if s == 1 { -v } else { v }
}
fn rel_l2(a: &[f32], r: &[f64]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(r) {
        d += (*x as f64 - y) * (*x as f64 - y);
        n += y * y;
    }
    (d / n.max(1e-300)).sqrt()
}

/// CPU fp64 reference for one sequence, all heads.
fn reference(q: &[f32], raw: &[u16], comp: &[u16], raw_off: usize, n_raw: usize, n_comp: usize, sinks: &[f32]) -> Vec<f64> {
    let (nh, hd) = (N_HEAD as usize, HEAD_DIM as usize);
    let scale = 1.0 / (hd as f64).sqrt();
    let row = |r: usize| -> &[u16] {
        if r < n_raw { &raw[(raw_off + r) * hd..(raw_off + r + 1) * hd] } else { &comp[(r - n_raw) * hd..(r - n_raw + 1) * hd] }
    };
    let n = n_raw + n_comp;
    let mut out = vec![0f64; nh * hd];
    for h in 0..nh {
        let qh = &q[h * hd..(h + 1) * hd];
        let sc: Vec<f64> = (0..n)
            .map(|r| row(r).iter().zip(qh).map(|(k, q)| f16_to_f64(*k) * *q as f64).sum::<f64>() * scale)
            .collect();
        let sink = sinks[h] as f64;
        let mx = sc.iter().cloned().fold(sink, f64::max);
        let den = (sink - mx).exp() + sc.iter().map(|s| (s - mx).exp()).sum::<f64>();
        for r in 0..n {
            let w = (sc[r] - mx).exp() / den;
            for (d, k) in row(r).iter().enumerate() {
                out[h * hd + d] += w * f16_to_f64(*k);
            }
        }
    }
    out
}

#[test]
#[ignore]
fn attention_kernels_vs_fp64_reference() -> eyre::Result<()> {
    install_panic_handler().ok();
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let s = &e.compute;
    let id = dev.id;
    let (nh, hd) = (N_HEAD as usize, HEAD_DIM as usize);

    // Cases span the window edge and offsets: (n_raw, raw_off, n_comp).
    // Layer 0 is pure SWA (n_comp 0); later layers add compressed rows.
    let cases: [(usize, usize, usize); 6] = [(5, 0, 0), (128, 0, 0), (128, 17, 0), (1, 0, 0), (64, 900, 300), (128, 0, 512)];
    let raw_rows = 1152usize;
    let mut worst = (0f64, 0f64);
    // Logit scale: q multiplier. Real attention logits reach ~10-20; the f16
    // score error grows with |logit|, so test a small and a realistic scale.
    for &qmul in &[1.0f32, 12.0] {
        eprintln!("--- q scale {qmul}");
        for (ci, &(n_raw, raw_off, n_comp)) in cases.iter().enumerate() {
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (ci as u64 + 1) * 0x1000_0001);
            let raw_h: Vec<u16> = (0..raw_rows * hd).map(|_| f32_to_f16_bits(rng.f32())).collect();
            let comp_h: Vec<u16> = (0..(n_comp.max(1)) * hd).map(|_| f32_to_f16_bits(rng.f32())).collect();
            let q_h: Vec<f32> = (0..nh * hd).map(|_| rng.f32() * qmul).collect();
            let sinks_h: Vec<f32> = (0..nh).map(|_| rng.f32() * 3.0).collect();
            let (raw, comp, q, sinks) = (dev_u16(id, &raw_h), dev_u16(id, &comp_h), dev_f32(id, &q_h), dev_f32(id, &sinks_h));
            let comp_opt = if n_comp > 0 { Some(&comp) } else { None };
            let refv = reference(&q_h, &raw_h, &comp_h, raw_off, n_raw, n_comp, &sinks_h);

            // (a) production arena kernels: f16 scores, per-row bases (one row, base 0)
            let stride = 1024u32.max(((n_raw + n_comp) as u32).div_ceil(256) * 256);
            let mut sc = DeviceBuffer::<f32>::new(id, nh * stride as usize)?;
            let mut out_a = DeviceBuffer::<f32>::new(id, nh * hd)?;
            sc.fill_zero()?;
            out_a.fill_zero()?;
            let nr = dev_i32(id, &[n_raw as i32]);
            let ro = dev_i32(id, &[raw_off as i32]);
            let nc = dev_i32(id, &[n_comp as i32]);
            let cb = dev_i32(id, &[0]);
            e.attn_mixed.launch_score_batched_htiled_wmma_f16s_rows(s, &mut sc, &q, &raw, comp_opt, &nr, &ro, &nc, None, N_HEAD, HEAD_DIM, (n_raw + n_comp) as u32, 1, 0, stride, Some(&cb))?;
            e.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows(s, &mut out_a, &mut sc, &sinks, &raw, comp_opt, &nr, &ro, &nc, N_HEAD, HEAD_DIM, 1, 0, stride, Some(&cb))?;

            // (b) decode chain: f32 scores
            let mut sc_b = DeviceBuffer::<f32>::new(id, nh * ATTN_MIXED_MAX_KEYS as usize)?;
            let mut inv = DeviceBuffer::<f32>::new(id, nh)?;
            let mut part = DeviceBuffer::<f32>::new(id, nh * K_SPLIT as usize * hd)?;
            let mut out_b = DeviceBuffer::<f32>::new(id, nh * hd)?;
            out_b.fill_zero()?;

            // Decode hands BOTH kernels the raw-window VIEW with raw_off 0.
            let raw_win = raw.slice_view(raw_off * hd, n_raw * hd);
            e.attn_mixed.launch_score_b1_htiled_wmma(s, &mut sc_b, &q, &raw_win, comp_opt, n_raw as u32, 0, n_comp as u32, N_HEAD, HEAD_DIM, (n_raw + n_comp) as u32)?;
            e.attn_mixed.launch_softmax_only(s, &mut sc_b, &sinks, &mut inv, N_HEAD, n_raw as u32, n_comp as u32)?;
            e.attn_mixed.launch_wsum_b1_htiled_ksplit_ldsv(s, &mut part, &sc_b, &raw_win, comp_opt, N_HEAD, HEAD_DIM, n_raw as u32, n_comp as u32, K_SPLIT)?;
            e.attn_mixed.launch_reduce_partials_apply_inv(s, &mut out_b, &part, &inv, N_HEAD, HEAD_DIM, K_SPLIT)?;
            s.synchronize()?;

            let (oa, ob) = (host(&out_a), host(&out_b));
            let (ea, eb) = (rel_l2(&oa, &refv), rel_l2(&ob, &refv));
            let dab = rel_l2(&oa, &ob.iter().map(|x| *x as f64).collect::<Vec<_>>());
            eprintln!(
                "n_raw {n_raw:4} raw_off {raw_off:4} n_comp {n_comp:4} | arena(f16s) vs fp64 {ea:9.2e} | decode(f32) vs fp64 {eb:9.2e} | arena vs decode {dab:9.2e}"
            );
            worst.0 = worst.0.max(ea);
            worst.1 = worst.1.max(eb);
        }
    }
    eprintln!("WORST: arena {:.2e}  decode {:.2e}", worst.0, worst.1);
    // A real bug (dropped/extra row, wrong offset, sink misuse) is >=1e-2 on
    // random data; f16 score rounding is ~1e-4..1e-3.
    if worst.0 > 5e-3 || worst.1 > 5e-3 {
        return Err(eyre!("a kernel is FAR from the fp64 reference: arena {:.2e}, decode {:.2e} -- likely a bug, not rounding", worst.0, worst.1));
    }
    Ok(())
}
