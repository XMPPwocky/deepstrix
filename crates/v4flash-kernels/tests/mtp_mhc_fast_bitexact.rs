//! (2) `V41_MTP_ROUTER_BATCHED`: the drafter's per-row router (5 x `f16.matvec` 128 x 5120 + 5 x
//! `router_topk.launch`, 128 experts, top 3, bias) against `f16.matvec_batched` +
//! `router_topk.launch_batched`: logits, picks and weights bit-identical, on both GPUs.
//!
//! (1) `V41_MTP_MHC_FAST`: the DSpark drafter's mHC sub-block chain (`MtpState::hc_mixes` +
//! `hc_pre_and_carry` + the attn/ffn `rms_w` norm) against `MhcArena::launch_fast` in mode
//! `MIX_NORMED`, at the drafter's batch (B = 5), on BOTH GPUs (the drafter runs on the iGPU).
//! Every output -- split, mix, the rewritten carry, cur, normed -- must be bit-identical.
//! Synthetic data (no model load, < 20 MB), runs beside a live hub; also prints paired timings.
//!
//!   cargo test --release --features v41 -p v4flash-kernels --test mtp_mhc_fast_bitexact \
//!     -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_HC, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::{FastCollapse, FastMix, MIX_NORMED};

const B: u32 = 5;

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

fn download<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}

fn diff(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

struct Bufs {
    split: DeviceBuffer<f32>,
    mix: DeviceBuffer<f32>,
    carry: DeviceBuffer<f32>,
    cur: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    flat: DeviceBuffer<f32>,
    counters: DeviceBuffer<u32>,
    inv: DeviceBuffer<f32>,
}

impl Bufs {
    fn new(id: i32, carry0: &[f32]) -> eyre::Result<Self> {
        let (b, m, ne, hd) = (B as usize, HC_MIX_DIM as usize, N_EMBD as usize, HC_DIM as usize);
        Ok(Self {
            split: upload(id, &vec![f32::NAN; b * m])?,
            mix: upload(id, &vec![f32::NAN; b * m])?,
            carry: upload(id, carry0)?,
            cur: upload(id, &vec![f32::NAN; b * ne])?,
            normed: upload(id, &vec![f32::NAN; b * ne])?,
            flat: upload(id, &vec![f32::NAN; b * hd])?,
            counters: upload(id, &vec![0u32; b])?,
            inv: upload(id, &vec![f32::NAN; b])?,
        })
    }
}

/// The drafter's chain, call for call (mtp.rs `hc_mixes` + `hc_pre_and_carry` + `rms_w`).
#[allow(clippy::too_many_arguments)]
fn chain(e: &DeviceEngine, s: &Stream, o: &mut Bufs, h: &DeviceBuffer<f32>, w: &DeviceBuffer<u8>, scale: &DeviceBuffer<f32>, base: &DeviceBuffer<f32>, nw: &DeviceBuffer<f32>) -> eyre::Result<()> {
    e.rms_nw.launch_batched(s, &mut o.flat, h, 1, HC_DIM, RMS_EPS, B)?;
    for j in 0..B as usize {
        let fj = o.flat.slice_view(j * HC_DIM as usize, HC_DIM as usize);
        let mut mj = o.mix.slice_view_mut(j * HC_MIX_DIM as usize, HC_MIX_DIM as usize);
        e.f16.matvec(s, &mut mj, w, &fj, HC_MIX_DIM, HC_DIM)?;
    }
    e.hc_sinkhorn.launch_batched(s, &mut o.split, &o.mix, scale, base, N_HC, SINKHORN_ITERS, SINKHORN_EPS, B)?;
    e.hc_weighted.launch_batched(s, &mut o.cur, h, &o.carry, N_EMBD, N_HC, HC_MIX_DIM, B)?;
    o.carry.copy_from_buffer_async(&o.split, s)?;
    e.rms_w.launch_weighted_batched(s, &mut o.normed, &o.cur, nw, N_EMBD, RMS_EPS, B)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fast(e: &DeviceEngine, s: &Stream, o: &mut Bufs, h: &DeviceBuffer<f32>, w: &DeviceBuffer<u8>, scale: &DeviceBuffer<f32>, base: &DeviceBuffer<f32>, nw: &DeviceBuffer<f32>) -> eyre::Result<()> {
    e.mhc_arena.launch_fast(
        s,
        Some(FastMix {
            weight: w,
            x: h,
            scale,
            base,
            mode: MIX_NORMED,
            split_out: &mut o.split,
            mix_out: &mut o.mix,
            counters: &mut o.counters,
            inv_rows: &mut o.inv,
        }),
        Some(FastCollapse { x: h, cur_out: &mut o.cur, norm_out: &mut o.normed, norm_w: nw }),
        &mut o.carry,
        true,
        HC_DIM,
        RMS_EPS,
        SINKHORN_ITERS,
        SINKHORN_EPS,
        B,
    )
}

#[test]
#[ignore]
fn drafter_mhc_fast_matches_chain() -> eyre::Result<()> {
    install_panic_handler()?;
    for want in ["gfx1151", "gfx1201"] {
        let Some(dev) = Device::all()?.into_iter().find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with(want)).unwrap_or(false)) else {
            return Err(eyre!("no {want}"));
        };
        let arch = dev.properties()?.gcn_arch_name;
        dev.set_current()?;
        let id = dev.id;
        let e = DeviceEngine::for_arch(dev, &arch)?;
        let s = Stream::new(id)?;
        let (m, ne, hd) = (HC_MIX_DIM as usize, N_EMBD as usize, HC_DIM as usize);
        let mut rng = Lcg(0xD5_9A4C ^ id as u64);
        let mut cases = 0;
        for seed in 0..6u32 {
            // Residual scale varies per row and seed (the drafter's h grows across layers).
            let amp = [0.05f32, 1.0, 30.0][seed as usize % 3];
            let h_host: Vec<f32> = (0..B as usize * hd).map(|i| rng.sym(amp * (1.0 + (i / hd) as f32))).collect();
            let w_host: Vec<u8> = (0..m * hd)
                .flat_map(|_| {
                    let r = rng.next();
                    let h = ((r >> 31) << 15) as u16 | ((8 + (r >> 10) % 10) << 10) as u16 | (r & 0x3FF) as u16;
                    h.to_le_bytes()
                })
                .collect();
            let scale = upload(id, &[rng.sym(1.0).abs() + 0.5, rng.sym(1.0).abs() + 0.5, rng.sym(1.0).abs() + 0.5])?;
            let base = upload(id, &(0..m).map(|_| rng.sym(1.0)).collect::<Vec<f32>>())?;
            let nw = upload(id, &(0..ne).map(|_| 0.5 + rng.sym(0.5).abs()).collect::<Vec<f32>>())?;
            let carry0: Vec<f32> = (0..B as usize * m).map(|_| rng.sym(1.0).abs()).collect();
            let h = upload(id, &h_host)?;
            let w = upload(id, &w_host)?;
            let mut a = Bufs::new(id, &carry0)?;
            let mut f = Bufs::new(id, &carry0)?;
            chain(&e, &s, &mut a, &h, &w, &scale, &base, &nw)?;
            fast(&e, &s, &mut f, &h, &w, &scale, &base, &nw)?;
            s.synchronize()?;
            for (name, x, y) in [
                ("split", &a.split, &f.split),
                ("mix", &a.mix, &f.mix),
                ("carry", &a.carry, &f.carry),
                ("cur", &a.cur, &f.cur),
                ("normed", &a.normed, &f.normed),
            ] {
                let d = diff(&download(x)?, &download(y)?);
                assert_eq!(d, 0, "{arch}: {name} differs in {d} values (seed {seed})");
            }
            assert_eq!(download(&f.counters)?, vec![0u32; B as usize], "{arch}: counters not left zero");
            cases += 1;
            if seed == 0 {
                // Paired timing: 4 alternating rounds of 300 reps each (first round warm-up).
                let (ea, eb) = (Event::new()?, Event::new()?);
                let (mut tc, mut tf) = (0f64, 0f64);
                for round in 0..4 {
                    for arm in [round % 2, 1 - round % 2] {
                        ea.record(&s)?;
                        for _ in 0..300 {
                            if arm == 0 {
                                chain(&e, &s, &mut a, &h, &w, &scale, &base, &nw)?;
                            } else {
                                fast(&e, &s, &mut f, &h, &w, &scale, &base, &nw)?;
                            }
                        }
                        eb.record(&s)?;
                        s.synchronize()?;
                        let us = Event::elapsed_ms(&ea, &eb)? as f64 * 1e3 / 300.0;
                        if round > 0 {
                            if arm == 0 { tc += us } else { tf += us }
                        }
                    }
                }
                eprintln!("{arch}: one drafter mHC sub-block, device time: chain {:.1} us, fast {:.1} us", tc / 3.0, tf / 3.0);
            }
        }
        eprintln!("{arch}: PASS -- {cases} seeds, split/mix/carry/cur/normed bit-identical, counters left zero");
    }
    Ok(())
}

#[test]
#[ignore]
fn drafter_router_batched_matches_rows() -> eyre::Result<()> {
    install_panic_handler()?;
    const NE: u32 = 128;
    const TOPK: u32 = 3;
    const SCALE: f32 = 1.5;
    const EPS: f32 = 6.103515625e-5;
    for want in ["gfx1151", "gfx1201"] {
        let Some(dev) = Device::all()?.into_iter().find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with(want)).unwrap_or(false)) else {
            return Err(eyre!("no {want}"));
        };
        let arch = dev.properties()?.gcn_arch_name;
        dev.set_current()?;
        let id = dev.id;
        let e = DeviceEngine::for_arch(dev, &arch)?;
        let s = Stream::new(id)?;
        let (ne, bu) = (N_EMBD as usize, B as usize);
        let mut rng = Lcg(0x5_0A7E ^ id as u64);
        let mut cases = 0;
        for seed in 0..8u32 {
            let w_host: Vec<u8> = (0..NE as usize * ne)
                .flat_map(|_| {
                    let r = rng.next();
                    let h = ((r >> 31) << 15) as u16 | ((6 + (r >> 10) % 12) << 10) as u16 | (r & 0x3FF) as u16;
                    h.to_le_bytes()
                })
                .collect();
            let w = upload(id, &w_host)?;
            // Seeds 4..8 make near-ties: rows built from a few repeated values.
            let x_host: Vec<f32> = (0..bu * ne)
                .map(|i| if seed >= 4 { [0.25f32, -0.5, 1.0][(i * 7 + seed as usize) % 3] } else { rng.sym(2.0) })
                .collect();
            let x = upload(id, &x_host)?;
            let bias = upload(id, &(0..NE as usize).map(|_| rng.sym(0.05)).collect::<Vec<f32>>())?;
            let mut la = upload(id, &vec![f32::NAN; bu * NE as usize])?;
            let mut lf = upload(id, &vec![f32::INFINITY; bu * NE as usize])?;
            let mut sa = upload(id, &vec![-1i32; bu * TOPK as usize])?;
            let mut sf = upload(id, &vec![-2i32; bu * TOPK as usize])?;
            let mut wa = upload(id, &vec![f32::NAN; bu * TOPK as usize])?;
            let mut wf = upload(id, &vec![f32::INFINITY; bu * TOPK as usize])?;
            for j in 0..bu {
                let xj = x.slice_view(j * ne, ne);
                let mut lj = la.slice_view_mut(j * NE as usize, NE as usize);
                e.f16.matvec(&s, &mut lj, &w, &xj, NE, N_EMBD)?;
            }
            for j in 0..bu {
                let lg = la.slice_view(j * NE as usize, NE as usize);
                let mut sel = sa.slice_view_mut(j * TOPK as usize, TOPK as usize);
                let mut ew = wa.slice_view_mut(j * TOPK as usize, TOPK as usize);
                e.router_topk.launch(&s, &mut sel, &mut ew, &lg, Some(&bias), NE, TOPK, SCALE, EPS)?;
            }
            e.f16.matvec_batched(&s, &mut lf, &w, &x, NE, N_EMBD, B)?;
            e.router_topk.launch_batched(&s, &mut sf, &mut wf, &lf, Some(&bias), NE, TOPK, SCALE, EPS, B)?;
            s.synchronize()?;
            assert_eq!(diff(&download(&la)?, &download(&lf)?), 0, "{arch}: router logits differ (seed {seed})");
            assert_eq!(download(&sa)?, download(&sf)?, "{arch}: picks differ (seed {seed})");
            assert_eq!(diff(&download(&wa)?, &download(&wf)?), 0, "{arch}: expert weights differ (seed {seed})");
            cases += 1;
            if seed == 0 {
                let (ea, eb) = (Event::new()?, Event::new()?);
                let (mut tr, mut tb) = (0f64, 0f64);
                for round in 0..4 {
                    for arm in [round % 2, 1 - round % 2] {
                        ea.record(&s)?;
                        for _ in 0..300 {
                            if arm == 0 {
                                for j in 0..bu {
                                    let xj = x.slice_view(j * ne, ne);
                                    let mut lj = la.slice_view_mut(j * NE as usize, NE as usize);
                                    e.f16.matvec(&s, &mut lj, &w, &xj, NE, N_EMBD)?;
                                }
                                for j in 0..bu {
                                    let lg = la.slice_view(j * NE as usize, NE as usize);
                                    let mut sel = sa.slice_view_mut(j * TOPK as usize, TOPK as usize);
                                    let mut ew = wa.slice_view_mut(j * TOPK as usize, TOPK as usize);
                                    e.router_topk.launch(&s, &mut sel, &mut ew, &lg, Some(&bias), NE, TOPK, SCALE, EPS)?;
                                }
                            } else {
                                e.f16.matvec_batched(&s, &mut lf, &w, &x, NE, N_EMBD, B)?;
                                e.router_topk.launch_batched(&s, &mut sf, &mut wf, &lf, Some(&bias), NE, TOPK, SCALE, EPS, B)?;
                            }
                        }
                        eb.record(&s)?;
                        s.synchronize()?;
                        let us = Event::elapsed_ms(&ea, &eb)? as f64 * 1e3 / 300.0;
                        if round > 0 {
                            if arm == 0 { tr += us } else { tb += us }
                        }
                    }
                }
                eprintln!("{arch}: one drafter router (matvec + top-k), device time: per-row {:.1} us, batched {:.1} us", tr / 3.0, tb / 3.0);
            }
        }
        eprintln!("{arch}: PASS -- {cases} seeds (4 with near-ties), router logits / picks / weights bit-identical");
    }
    Ok(())
}
