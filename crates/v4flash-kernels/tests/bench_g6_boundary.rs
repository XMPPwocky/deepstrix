//! Focused bench of fusion group G6 (the decode layer boundary, `V41_DGPU_G6_BOUNDARY`): the
//! production chain [box 2's partial pinned H2D -> hc_post_from_split_batched_add2 -> mhc_fast_batched
//! pre-attn] vs ONE `g6_boundary` launch, plus the floors that price it. Device wall of the queued
//! group (a spin kernel first, so the host is ahead; event pair around the group), cold (MALL flushed
//! by an 80 MB read first: the per-layer hc weights are cold in production) and warm, every arm once
//! per round in a RANDOM order, many rounds, p10 / p50 / p90 and PAIRED differences against the chain;
//! `chain#AA` is the chain timed twice (the A/A spread).
//!
//! Floors (G6_PROBE_HSACO = scratch/g46/probe.hsaco, built by scratch/g46/probe.sh): `zc` = the
//! lead's zero-copy read of b rows of the partial alone (system-scope loads, one WG per row);
//! `vram` = the lead's VRAM loads + stores alone; `empty` = one empty launch.
//! G6_TS=1 (+ V41_DGPU_G6_TS=1): in-kernel phase timestamps of the G6 launch (100 MHz counter).
//!
//!   HIP_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 \
//!   DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 G6_ROUNDS=150 G6_B=1,2,4,8,16 \
//!   cargo test --release --features v41 -p v4flash-kernels --test bench_g6_boundary -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, launch_kernel, Device, DeviceBuffer, Event, LaunchConfig, Module, PinnedBuffer, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_HC, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
use v4flash_kernels::device_ceilings::DeviceCeilings;
use v4flash_kernels::g6_boundary::{G6Combine, G6Mhc, G6Remote, G6_MAX_B};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::{FastCollapse, FastMix, MIX_PRE_SCALED};
use v4flash_kernels::wmma_probe::WmmaProbe;

const BMAX: usize = G6_MAX_B as usize;
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

struct Bufs {
    moe: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    remote_dev: DeviceBuffer<f32>,
    remote_pin: PinnedBuffer<f32>,
    after_attn: DeviceBuffer<f32>,
    split: DeviceBuffer<f32>,
    carry: DeviceBuffer<f32>,
    resid: DeviceBuffer<f32>,
    cur: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    mix: DeviceBuffer<f32>,
    counters: DeviceBuffer<u32>,
    inv_rows: DeviceBuffer<f32>,
    w: DeviceBuffer<u8>,
    scale: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    norm_w: DeviceBuffer<f32>,
    sink: DeviceBuffer<f32>,
}

fn bufs(id: i32, rng: &mut Lcg) -> eyre::Result<Bufs> {
    let (ne, k, m) = (N_EMBD as usize, HC_DIM as usize, HC_MIX_DIM as usize);
    let mut remote_pin = PinnedBuffer::<f32>::new(BMAX * ne)?;
    for x in remote_pin.as_mut_slice().iter_mut() {
        *x = rng.sym(0.1);
    }
    let split: Vec<f32> = (0..BMAX * m).map(|i| if i % m < 8 { 0.5 } else { 0.25 }).collect();
    let wh: Vec<u8> = (0..m * k).flat_map(|_| f16_bits(rng.sym(0.05)).to_le_bytes()).collect();
    Ok(Bufs {
        moe: upload(id, &(0..BMAX * ne).map(|_| rng.sym(0.1)).collect::<Vec<_>>())?,
        shared: upload(id, &(0..BMAX * ne).map(|_| rng.sym(0.1)).collect::<Vec<_>>())?,
        remote_dev: upload(id, &vec![0f32; BMAX * ne])?,
        remote_pin,
        after_attn: upload(id, &(0..BMAX * k).map(|_| rng.sym(1.0)).collect::<Vec<_>>())?,
        split: upload(id, &split)?,
        carry: upload(id, &split)?,
        resid: upload(id, &vec![0f32; BMAX * k])?,
        cur: upload(id, &vec![0f32; BMAX * ne])?,
        norm: upload(id, &vec![0f32; BMAX * ne])?,
        mix: upload(id, &vec![0f32; BMAX * m])?,
        counters: upload(id, &vec![0u32; BMAX])?,
        inv_rows: upload(id, &vec![0f32; BMAX])?,
        w: upload(id, &wh)?,
        scale: upload(id, &[0.7f32, 0.45, 1.3])?,
        base: upload(id, &(0..m).map(|_| rng.sym(0.6)).collect::<Vec<f32>>())?,
        norm_w: upload(id, &(0..ne).map(|_| 0.5 + rng.sym(0.4)).collect::<Vec<f32>>())?,
        sink: upload(id, &vec![0f32; 64])?,
    })
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arm {
    Chain,
    ChainAA,
    G6,
    ChainNr,
    G6Nr,
    G6Dev,
    H2d,
    Add2,
    Mhc,
    Zc,
    ZcPlain,
    Vram,
    Empty,
}

fn run(e: &DeviceEngine, s: &Stream, bf: &mut Bufs, probe: Option<&Module>, arm: Arm, b: u32) -> eyre::Result<()> {
    let n = b as usize * N_EMBD as usize;
    let h2d = |bf: &mut Bufs| -> eyre::Result<()> {
        let src = &bf.remote_pin.as_slice()[..n];
        bf.remote_dev.slice_view_mut(0, n).copy_from_host_async(src, s)
    };
    let add2 = |bf: &mut Bufs| e.hc_post.launch_from_split_batched_add2(s, &mut bf.resid, &bf.moe, &bf.shared, &bf.remote_dev, &bf.after_attn, &bf.split, N_HC, N_EMBD, N_HC, b);
    let add = |bf: &mut Bufs| e.hc_post.launch_from_split_batched_add(s, &mut bf.resid, &bf.moe, &bf.shared, &bf.after_attn, &bf.split, N_HC, N_EMBD, N_HC, b);
    let mhc = |bf: &mut Bufs| {
        e.mhc_arena.launch_fast(
            s,
            Some(FastMix {
                weight: &bf.w,
                x: &bf.resid,
                scale: &bf.scale,
                base: &bf.base,
                mode: MIX_PRE_SCALED,
                split_out: &mut bf.split,
                mix_out: &mut bf.mix,
                counters: &mut bf.counters,
                inv_rows: &mut bf.inv_rows,
            }),
            Some(FastCollapse { x: &bf.resid, cur_out: &mut bf.cur, norm_out: &mut bf.norm, norm_w: &bf.norm_w }),
            &mut bf.carry,
            true,
            HC_DIM,
            RMS_EPS,
            SINKHORN_ITERS,
            SINKHORN_EPS,
            b,
        )
    };
    let g6 = |bf: &mut Bufs, mode: u32| {
        let remote = match mode {
            1 => Some(G6Remote::Pinned(&bf.remote_pin, 0usize)),
            2 => Some(G6Remote::Device(&bf.remote_dev)),
            _ => None,
        };
        e.g6.launch(
            s,
            G6Combine { moe: &bf.moe, shared: &bf.shared, remote, after_attn: &bf.after_attn, resid_out: &mut bf.resid },
            G6Mhc {
                weight: &bf.w,
                scale: &bf.scale,
                base: &bf.base,
                split: &mut bf.split,
                mix_out: &mut bf.mix,
                counters: &mut bf.counters,
                inv_rows: &mut bf.inv_rows,
                carry: &mut bf.carry,
                write_carry: true,
                cur_out: &mut bf.cur,
                norm_out: &mut bf.norm,
                norm_w: &bf.norm_w,
                rms_eps: RMS_EPS,
                sinkhorn_iters: SINKHORN_ITERS,
                sinkhorn_eps: SINKHORN_EPS,
            },
            b,
        )
    };
    match arm {
        Arm::Chain | Arm::ChainAA => {
            h2d(bf)?;
            add2(bf)?;
            mhc(bf)
        }
        Arm::ChainNr => {
            add(bf)?;
            mhc(bf)
        }
        Arm::G6 => g6(bf, 1),
        Arm::G6Nr => g6(bf, 0),
        Arm::G6Dev => g6(bf, 2),
        Arm::H2d => h2d(bf),
        Arm::Add2 => add2(bf),
        Arm::Mhc => mhc(bf),
        Arm::Zc | Arm::ZcPlain | Arm::Vram | Arm::Empty => {
            let m = probe.ok_or_else(|| eyre!("no probe module"))?;
            let sink = bf.sink.raw();
            match arm {
                Arm::Zc | Arm::ZcPlain => {
                    let f = m.get_function("zc_rows")?;
                    let p = bf.remote_pin.device_ptr();
                    let sys = u32::from(arm == Arm::Zc);
                    launch_kernel!(f, LaunchConfig { grid: (b, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, s, [p, sink, N_EMBD, sys])
                }
                Arm::Vram => {
                    let f = m.get_function("vram_rows")?;
                    launch_kernel!(f, LaunchConfig { grid: (b, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }, s, [
                        bf.moe.raw(), bf.shared.raw(), bf.after_attn.raw(), bf.resid.raw(), N_EMBD
                    ])
                }
                _ => {
                    let f = m.get_function("empty_k")?;
                    launch_kernel!(f, LaunchConfig { grid: (1, 1, 1), block: (64, 1, 1), shared_mem_bytes: 0 }, s, [sink])
                }
            }
        }
    }
}

#[test]
#[ignore]
fn bench_g6_boundary() -> eyre::Result<()> {
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
    let mut rng = Lcg(0x6b0d_a7e5);
    let mut bf = bufs(id, &mut rng)?;
    let flush_region = DeviceBuffer::<u8>::new(id, FLUSH)?;
    let mut flush_sink = upload(id, &vec![0u32; 1 << 16])?;
    let (mut spin_out, spin_a, spin_b) = (DeviceBuffer::<f32>::new(id, 32)?, upload(id, &[1.0f32])?, upload(id, &[0.999f32])?);
    let probe = match std::env::var("G6_PROBE_HSACO") {
        Ok(p) if !p.is_empty() => Some(Module::load_data(&std::fs::read(&p)?)?),
        _ => None,
    };
    let rounds: usize = std::env::var("G6_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
    let bs: Vec<u32> = std::env::var("G6_B").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or(vec![1, 2, 4, 8, 16]);
    // spin calibration: ~40 us
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
    let mut arms = vec![Arm::Chain, Arm::ChainAA, Arm::G6, Arm::ChainNr, Arm::G6Nr, Arm::G6Dev, Arm::H2d, Arm::Add2, Arm::Mhc];
    if probe.is_some() {
        arms.extend([Arm::Zc, Arm::ZcPlain, Arm::Vram, Arm::Empty]);
    }
    eprintln!("bench_g6_boundary: rounds {rounds}, b {bs:?}, arms {arms:?}, kernargs HIP_FORCE_DEV_KERNARG={:?}", std::env::var("HIP_FORCE_DEV_KERNARG").ok());
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
                        ceil.read_v4(&s, &flush_region, &mut flush_sink, 64 * 8, 256)?;
                    }
                    wp.launch_fma_f32(&s, &mut spin_out, &spin_a, &spin_b, spin_iters, 1, 32)?;
                    ea.record(&s)?;
                    run(&e, &s, &mut bf, probe.as_ref(), arms[i], b)?;
                    eb.record(&s)?;
                    s.synchronize()?;
                    t[i].push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            let base = 0usize; // Chain
            eprintln!("b={b} {}: arm p10 / p50 / p90 us | paired (arm - chain) p10 / p50 / p90", if cold { "COLD" } else { "warm" });
            for (i, a) in arms.iter().enumerate() {
                let d: Vec<f64> = t[i].iter().zip(&t[base]).map(|(x, y)| x - y).collect();
                eprintln!(
                    "  {:<8} {:7.1} {:7.1} {:7.1} | {:+7.1} {:+7.1} {:+7.1}",
                    format!("{a:?}"), pct(&t[i], 0.1), pct(&t[i], 0.5), pct(&t[i], 0.9), pct(&d, 0.1), pct(&d, 0.5), pct(&d, 0.9)
                );
            }
            let aa: Vec<f64> = t[1].iter().zip(&t[0]).map(|(x, y)| (x - y).abs()).collect();
            eprintln!("  A/A |d| p50 {:.2} p90 {:.2}", pct(&aa, 0.5), pct(&aa, 0.9));
        }
        if std::env::var_os("G6_TS").is_some() {
            // in-kernel phases (relative to the first WG's entry), cold, median over rounds
            let mut ph: Vec<[f64; 12]> = Vec::new();
            let ts_mode: u32 = std::env::var("G6_TS_MODE").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
            for _ in 0..rounds.min(60) {
                ceil.read_v4(&s, &flush_region, &mut flush_sink, 64 * 8, 256)?;
                wp.launch_fma_f32(&s, &mut spin_out, &spin_a, &spin_b, spin_iters, 1, 32)?;
                run(&e, &s, &mut bf, None, match ts_mode { 0 => Arm::G6Nr, 1 => Arm::G6, _ => Arm::G6Dev }, b)?;
                s.synchronize()?;
                if let Some(v) = e.g6.ts_dump()? {
                    let nw = (b * v4flash_kernels::g6_boundary::g6_wgs_per_row(v4flash_kernels::g6_boundary::g6_mr())) as usize;
                    let t0 = (0..nw).map(|w| v[w * 16]).min().unwrap_or(0) as f64;
                    let us = |x: u64| (x as f64 - t0) / 100.0;
                    let role = |w: usize, r: u64| v[w * 16 + 6] == r;
                    let mx = |r: u64, k: usize| (0..nw).filter(|&w| role(w, r)).map(|w| us(v[w * 16 + k])).fold(0.0, f64::max);
                    let lead_flag = mx(0, 2);
                    let lead_done = mx(0, 3);
                    let last_entry = (0..nw).map(|w| us(v[w * 16])).fold(0.0, f64::max);
                    let fin = (0..nw).map(|w| us(v[w * 16 + 4])).fold(0.0, f64::max);
                    let tail = (0..nw).filter(|&w| v[w * 16 + 7] != 0).map(|w| us(v[w * 16 + 7])).fold(0.0, f64::max);
                    let tail0 = (0..nw).filter(|&w| v[w * 16 + 7] != 0).map(|w| us(v[w * 16 + 10])).fold(0.0, f64::max);
                    ph.push([last_entry, mx(0, 11), lead_flag, lead_done, mx(2, 2), mx(2, 8), mx(2, 9), mx(2, 3), tail0, tail, fin, 0.0]);
                }
            }
            if !ph.is_empty() {
                let col = |k: usize| pct(&ph.iter().map(|r| r[k]).collect::<Vec<_>>(), 0.5);
                eprintln!(
                    "  G6 phases b={b} mode {ts_mode} (us from the first WG, p50 of the per-launch max): last entry {:.2} | lead stored {:.2} flag {:.2} done {:.2} | mix wait {:.2} dots {:.2} bar {:.2} done {:.2} | tail start {:.2} end {:.2} | finish {:.2}",
                    col(0), col(1), col(2), col(3), col(4), col(5), col(6), col(7), col(8), col(9), col(10)
                );
            }
        }
    }
    if e.g6.error_word()? != 0 {
        return Err(eyre!("g6 error word set"));
    }
    Ok(())
}
