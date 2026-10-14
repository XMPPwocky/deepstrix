//! Focused bench of fusion group G4 (`V41_DGPU_G4_SHARED`): the production decode shared expert +
//! late FFN mixes [quantize -> fused gate/up/swiglu/quant (b <= 5; the 5-op chain at 6..8) -> down
//! GEMV -> mhc_fast mix NORMED] vs G4 [g4_qmix (quantize + mixes) -> the same GEMVs], the pieces
//! alone (`QuantMix` = production's quantize + mix launches, `QMix` = G4's one launch), plus the
//! floors that price it. Device wall of
//! the queued group (spin first, event pair around), COLD (MALL flushed by an 80 MB read: production
//! streams ~180 MB of weights per lane-layer) and warm, every arm once per round in a random order,
//! p10 / p50 / p90 and paired differences vs the chain; `chain#AA` = the A/A spread.
//!
//! Floors: `read` = one cold streaming read of exactly the group's weight bytes (gate + up + down =
//! 37.6 MB, + the 0.98 MB mix weight), `empty` = one empty launch (event-pair overhead).
//!
//!   HIP_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 \
//!   DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 G4_ROUNDS=150 G4_B=1,2,4,8 \
//!   cargo test --release --features v41 -p v4flash-kernels --test bench_g4_shared -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_FF_SHARED, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS, SWIGLU_CLAMP_EXP};
use v4flash_kernels::device_ceilings::DeviceCeilings;
use v4flash_kernels::g4_shared::{G4Mix, G4Quant};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::{FastMix, MIX_NORMED};
use v4flash_kernels::wmma_probe::WmmaProbe;

const BMAX: usize = 8;
const FLUSH: usize = 80 << 20;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn sym(&mut self, a: f32) -> f32 {
        ((self.next() & 0xFFFFFF) as f32 / 16777216.0 * 2.0 - 1.0) * a
    }
}
fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len())?;
    d.copy_from_host(host)?;
    Ok(d)
}
fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        return f64::NAN;
    }
    s[((s.len() - 1) as f64 * p).round() as usize]
}
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e <= 0 {
        return sign;
    }
    sign | (((e as u32) << 10) | (m >> 13)) as u16
}
fn q8_weight(id: i32, rng: &mut Lcg, rows: usize, k: usize) -> eyre::Result<DeviceBuffer<u8>> {
    let blocks = k / 32;
    let mut h = vec![0u8; rows * blocks * 34];
    for r in 0..rows {
        let row = &mut h[r * blocks * 34..(r + 1) * blocks * 34];
        for bl in 0..blocks {
            row[2 * bl..2 * bl + 2].copy_from_slice(&f16_bits(0.002 + 0.01 * (rng.next() % 100) as f32 / 100.0).to_le_bytes());
        }
        for q in row[blocks * 2..].iter_mut() {
            *q = (rng.next() & 0xff) as u8;
        }
    }
    upload(id, &h)
}

struct Bufs {
    wg: DeviceBuffer<u8>,
    wu: DeviceBuffer<u8>,
    wd: DeviceBuffer<u8>,
    hc: DeviceBuffer<u8>,
    scale: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    ffn_in: DeviceBuffer<f32>,
    after_attn: DeviceBuffer<f32>,
    split: DeviceBuffer<f32>,
    carry: DeviceBuffer<f32>,
    ffn_shared: DeviceBuffer<f32>,
    xq: DeviceBuffer<i8>,
    xs: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    mid: DeviceBuffer<f32>,
    mid_xq: DeviceBuffer<i8>,
    mid_xs: DeviceBuffer<f32>,
    mix: DeviceBuffer<f32>,
    counters: DeviceBuffer<u32>,
    inv_rows: DeviceBuffer<f32>,
    region: DeviceBuffer<u8>,
    sink: DeviceBuffer<u32>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arm {
    Chain,
    ChainAA,
    G4,
    Quant,
    Fused,
    QMix,
    Down,
    Mix,
    QuantMix,
    Read,
    Empty,
}

fn run(e: &DeviceEngine, ceil: &DeviceCeilings, s: &Stream, bf: &mut Bufs, arm: Arm, b: u32) -> eyre::Result<()> {
    let fused = b <= 5;
    let quant = |bf: &mut Bufs| e.q8.quantize_input_batched(s, &mut bf.xq, &mut bf.xs, &bf.ffn_in, N_EMBD, b);
    let first = |bf: &mut Bufs| -> eyre::Result<()> {
        if fused {
            e.shared_fused.launch(s, &mut bf.mid_xq, &mut bf.mid_xs, &bf.wg, &bf.wu, &bf.xq, &bf.xs, N_EMBD, N_FF_SHARED, b, SWIGLU_CLAMP_EXP)
        } else {
            e.q8.matvec_batched(s, &mut bf.gate, &bf.wg, &bf.xq, &bf.xs, N_FF_SHARED, N_EMBD, b)?;
            e.q8.matvec_batched(s, &mut bf.up, &bf.wu, &bf.xq, &bf.xs, N_FF_SHARED, N_EMBD, b)?;
            e.swiglu.launch_clamped(s, &mut bf.mid, &bf.gate, &bf.up, b * N_FF_SHARED, SWIGLU_CLAMP_EXP)?;
            e.q8.quantize_input_batched(s, &mut bf.mid_xq, &mut bf.mid_xs, &bf.mid, N_FF_SHARED, b)
        }
    };
    let down = |bf: &mut Bufs| e.q8.matvec_batched(s, &mut bf.ffn_shared, &bf.wd, &bf.mid_xq, &bf.mid_xs, N_EMBD, N_FF_SHARED, b);
    let mix = |bf: &mut Bufs| {
        e.mhc_arena.launch_fast(
            s,
            Some(FastMix {
                weight: &bf.hc,
                x: &bf.after_attn,
                scale: &bf.scale,
                base: &bf.base,
                mode: MIX_NORMED,
                split_out: &mut bf.split,
                mix_out: &mut bf.mix,
                counters: &mut bf.counters,
                inv_rows: &mut bf.inv_rows,
            }),
            None,
            &mut bf.carry,
            true,
            HC_DIM,
            RMS_EPS,
            SINKHORN_ITERS,
            SINKHORN_EPS,
            b,
        )
    };
    let qmix = |bf: &mut Bufs| {
        e.g4.launch_qmix(
            s,
            G4Quant { xq: &mut bf.xq, xscale: &mut bf.xs, x: &bf.ffn_in, n: N_EMBD },
            G4Mix {
                weight: &bf.hc,
                x: &bf.after_attn,
                scale: &bf.scale,
                base: &bf.base,
                split_out: &mut bf.split,
                mix_out: &mut bf.mix,
                counters: &mut bf.counters,
                carry: &mut bf.carry,
                write_carry: true,
                rms_eps: RMS_EPS,
                sinkhorn_iters: SINKHORN_ITERS,
                sinkhorn_eps: SINKHORN_EPS,
            },
            b,
        )
    };
    match arm {
        Arm::Chain | Arm::ChainAA => {
            quant(bf)?;
            first(bf)?;
            down(bf)?;
            mix(bf)
        }
        Arm::G4 => {
            qmix(bf)?;
            first(bf)?;
            down(bf)
        }
        Arm::Quant => quant(bf),
        Arm::Fused => first(bf),
        Arm::QMix => qmix(bf),
        Arm::Down => down(bf),
        Arm::Mix => mix(bf),
        Arm::QuantMix => {
            quant(bf)?;
            mix(bf)
        }
        Arm::Read => {
            // the group's weight bytes, cold, in one streaming read each (gate+up, down, hc)
            ceil.read_v4(s, &bf.wg, &mut bf.sink, 64 * 8, 256)?;
            ceil.read_v4(s, &bf.wu, &mut bf.sink, 64 * 8, 256)?;
            ceil.read_v4(s, &bf.wd, &mut bf.sink, 64 * 8, 256)
        }
        Arm::Empty => e.vec_scale.launch(s, &mut bf.inv_rows, 1.0, 1),
    }
}

#[test]
#[ignore]
fn bench_g4_shared() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201 visible"))?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let id = e.device.id;
    let s = Stream::new(id)?;
    let (ea, eb) = (Event::new()?, Event::new()?);
    let wp = WmmaProbe::for_arch(&arch)?;
    let ceil = DeviceCeilings::for_arch(&arch)?;
    let mut rng = Lcg(0x6b4_5ead);
    let (ne, nf, k, m) = (N_EMBD as usize, N_FF_SHARED as usize, HC_DIM as usize, HC_MIX_DIM as usize);
    let carry0: Vec<f32> = (0..BMAX * m).map(|i| if i % m < 4 { 0.25 } else { 0.1 }).collect();
    let mut bf = Bufs {
        wg: q8_weight(id, &mut rng, nf, ne)?,
        wu: q8_weight(id, &mut rng, nf, ne)?,
        wd: q8_weight(id, &mut rng, ne, nf)?,
        hc: upload(id, &(0..m * k).flat_map(|_| f16_bits(rng.sym(0.05)).to_le_bytes()).collect::<Vec<u8>>())?,
        scale: upload(id, &[0.7f32, 0.45, 1.3])?,
        base: upload(id, &(0..m).map(|_| rng.sym(0.6)).collect::<Vec<f32>>())?,
        ffn_in: upload(id, &(0..BMAX * ne).map(|_| rng.sym(2.0)).collect::<Vec<f32>>())?,
        after_attn: upload(id, &(0..BMAX * k).map(|_| rng.sym(1.0)).collect::<Vec<f32>>())?,
        split: upload(id, &carry0)?,
        carry: upload(id, &carry0)?,
        ffn_shared: upload(id, &vec![0f32; BMAX * ne])?,
        xq: upload(id, &vec![0i8; BMAX * ne])?,
        xs: upload(id, &vec![0f32; BMAX * ne / 32])?,
        gate: upload(id, &vec![0f32; BMAX * nf])?,
        up: upload(id, &vec![0f32; BMAX * nf])?,
        mid: upload(id, &vec![0f32; BMAX * nf])?,
        mid_xq: upload(id, &vec![0i8; BMAX * nf])?,
        mid_xs: upload(id, &vec![0f32; BMAX * nf / 32])?,
        mix: upload(id, &vec![0f32; BMAX * m])?,
        counters: upload(id, &vec![0u32; BMAX])?,
        inv_rows: upload(id, &vec![0f32; BMAX])?,
        region: DeviceBuffer::<u8>::new(id, FLUSH)?,
        sink: upload(id, &vec![0u32; 1 << 16])?,
    };
    let (mut spin_out, spin_a, spin_b) = (DeviceBuffer::<f32>::new(id, 32)?, upload(id, &[1.0f32])?, upload(id, &[0.999f32])?);
    let rounds: usize = std::env::var("G4_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
    let bs: Vec<u32> = std::env::var("G4_B").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or(vec![1, 2, 4, 8]);
    let mut spin_iters = 2000u32;
    {
        let mut ts = Vec::new();
        for _ in 0..20 {
            ea.record(&s)?;
            wp.launch_fma_f32(&s, &mut spin_out, &spin_a, &spin_b, spin_iters, 1, 32)?;
            eb.record(&s)?;
            s.synchronize()?;
            ts.push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
        }
        spin_iters = ((2000.0 * 40.0 / pct(&ts, 0.5).max(1.0)) as u32).clamp(100, 2_000_000);
    }
    let arms = [Arm::Chain, Arm::ChainAA, Arm::G4, Arm::Quant, Arm::Fused, Arm::QMix, Arm::Down, Arm::Mix, Arm::QuantMix, Arm::Read, Arm::Empty];
    eprintln!("bench_g4_shared: rounds {rounds}, b {bs:?}, kernargs HIP_FORCE_DEV_KERNARG={:?}", std::env::var("HIP_FORCE_DEV_KERNARG").ok());
    for &b in &bs {
        for cold in [true, false] {
            let mut t: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
            let mut order: Vec<usize> = (0..arms.len()).collect();
            for _ in 0..rounds {
                for i in (1..order.len()).rev() {
                    let j = (rng.next() as usize) % (i + 1);
                    order.swap(i, j);
                }
                for &i in &order {
                    if cold {
                        ceil.read_v4(&s, &bf.region, &mut bf.sink, 64 * 8, 256)?;
                    }
                    wp.launch_fma_f32(&s, &mut spin_out, &spin_a, &spin_b, spin_iters, 1, 32)?;
                    ea.record(&s)?;
                    run(&e, &ceil, &s, &mut bf, arms[i], b)?;
                    eb.record(&s)?;
                    s.synchronize()?;
                    t[i].push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            eprintln!("b={b} {}: arm p10 / p50 / p90 us | paired (arm - chain) p10 / p50 / p90", if cold { "COLD" } else { "warm" });
            for (i, a) in arms.iter().enumerate() {
                let d: Vec<f64> = t[i].iter().zip(&t[0]).map(|(x, y)| x - y).collect();
                eprintln!(
                    "  {:<8} {:7.1} {:7.1} {:7.1} | {:+7.1} {:+7.1} {:+7.1}",
                    format!("{a:?}"), pct(&t[i], 0.1), pct(&t[i], 0.5), pct(&t[i], 0.9), pct(&d, 0.1), pct(&d, 0.5), pct(&d, 0.9)
                );
            }
            let aa: Vec<f64> = t[1].iter().zip(&t[0]).map(|(x, y)| (x - y).abs()).collect();
            eprintln!("  A/A |d| p50 {:.2} p90 {:.2}", pct(&aa, 0.5), pct(&aa, 0.9));
        }
        if std::env::var_os("V41_DGPU_G4_TS").is_some() {
            // in-kernel phases of g4_qmix (cold), us from the first WG's entry, p50 of the per-launch max
            let mut ph: Vec<[f64; 7]> = Vec::new();
            for _ in 0..rounds.min(60) {
                ceil.read_v4(&s, &bf.region, &mut bf.sink, 64 * 8, 256)?;
                wp.launch_fma_f32(&s, &mut spin_out, &spin_a, &spin_b, spin_iters, 1, 32)?;
                run(&e, &ceil, &s, &mut bf, Arm::QMix, b)?;
                s.synchronize()?;
                if let Some(v) = e.g4.ts_dump()? {
                    let nw = (b * 25 + v4flash_kernels::g4_shared::G4Shared::quant_wgs(b, N_EMBD)) as usize;
                    let t0 = (0..nw).map(|w| v[w * 8]).min().unwrap_or(0) as f64;
                    let us = |x: u64| (x as f64 - t0) / 100.0;
                    let mx = |r: u64, k: usize| (0..nw).filter(|&w| v[w * 8 + 6] == r).map(|w| us(v[w * 8 + k])).fold(0.0, f64::max);
                    let tail = (0..nw).filter(|&w| v[w * 8 + 5] != 0).map(|w| us(v[w * 8 + 5])).fold(0.0, f64::max);
                    let last_entry = (0..nw).map(|w| us(v[w * 8])).fold(0.0, f64::max);
                    let fin = (0..nw).map(|w| us(v[w * 8 + 4])).fold(0.0, f64::max);
                    ph.push([last_entry, mx(0, 2), mx(1, 2), mx(2, 2), mx(2, 3), tail, fin]);
                }
            }
            if !ph.is_empty() {
                let col = |k: usize| pct(&ph.iter().map(|r| r[k]).collect::<Vec<_>>(), 0.5);
                eprintln!(
                    "  QMix phases b={b}: last entry {:.2} | RMS released {:.2} | quant done {:.2} | mix wait over {:.2} done {:.2} | tail end {:.2} | finish {:.2}",
                    col(0), col(1), col(2), col(3), col(4), col(5), col(6)
                );
            }
        }
    }
    Ok(())
}
