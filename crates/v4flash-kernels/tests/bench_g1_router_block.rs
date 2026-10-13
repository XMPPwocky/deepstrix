//! dGPU fusion group G1 bench (`V41_DGPU_G1_FUSED`): the decode router block -- production's nine
//! queue operations vs ONE `G1RouterBlock` launch -- in one process, beside the live hub, A/B
//! interleaved in a random order every round, with an A/A copy of the production arm.
//!
//! Items per batch size b (BENCH_B, default 1,2,4,8,16):
//!   cur / cur2   production chain, queued behind a ~50 us spin (device-paced): collapse, router_x2,
//!                prior H2D, top-k (prior), look-ahead top-k, q8k, rb_pack, selected_ready (no-timing
//!                event, as production), expert_sel_count. Wall = timing-event pair around it. cur2 =
//!                the A/A copy.
//!   g1[zb,pf]    the fused launch + selected_ready, for each kernel variant (BENCH_G1_VARIANTS).
//!   *_route      the same, cut at selected_ready (the end event takes its place): the router's
//!                critical path.
//!   host_*       host-observed completion, idle stream: enqueue the group, spin on
//!                selected_ready.query() (production) or on the pinned completion word (`host_g1_flag`,
//!                no event) -- enqueue start to the host seeing the pack complete.
//! Every device item runs COLD (MALL flushed: a 96 MB read first; production streams ~180 MB of
//! weights per lane-layer, so the router's 7.9 MB is never cached) and WARM (BENCH_WARM=1).
//!
//! Floors measured in the same process: dGPU streaming read BW (cold, 96 MB), and an empty launch.
//!
//!   HIP_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 \
//!   DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 cargo test --release --features v41 -p v4flash-kernels \
//!     --test bench_g1_router_block -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, PinnedBuffer, Stream};
use v4flash_kernels::config::{EXPERT_WEIGHT_SCALE, HC_DIM, N_EMBD, N_EXPERT, N_EXPERT_USED, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
use v4flash_kernels::device_ceilings::DeviceCeilings;
use v4flash_kernels::g1_router_block::{G1Collapse, G1Io, G1Look, G1PackOffsets, G1Prior};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::FastCollapse;
use v4flash_kernels::readback_pack::{PackPlan, PackSeg};
use v4flash_kernels::router_topk::RouterEx;
use v4flash_kernels::wmma_probe::WmmaProbe;

const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;
const MB: usize = 1 << 20;
const FLUSH: usize = 96 * MB;
const BMAX: usize = 16;
const NE: usize = N_EMBD as usize;
const NX: usize = N_EXPERT as usize;
const NU: usize = N_EXPERT_USED;
const XQ_ROW: usize = 20 * 292;
const LAYER: u32 = 17;

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

fn up<T: Copy>(id: i32, h: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
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

struct Bufs {
    w0: DeviceBuffer<u8>,
    w1: DeviceBuffer<u8>,
    bias0: DeviceBuffer<f32>,
    bias1: DeviceBuffer<f32>,
    norm_w: DeviceBuffer<f32>,
    x: DeviceBuffer<f32>,
    x_src: DeviceBuffer<f32>,
    carry: DeviceBuffer<f32>,
    cur: DeviceBuffer<f32>,
    ffn_in: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    logits2: DeviceBuffer<f32>,
    sel: DeviceBuffer<i32>,
    ew: DeviceBuffer<f32>,
    orig: DeviceBuffer<i32>,
    range: DeviceBuffer<f32>,
    look_sel: DeviceBuffer<i32>,
    look_ew: DeviceBuffer<f32>,
    d_prior: DeviceBuffer<f32>,
    prior_pin: PinnedBuffer<f32>,
    xq: DeviceBuffer<u8>,
    pack: PinnedBuffer<u32>,
    bank: DeviceBuffer<u32>,
    done: PinnedBuffer<u32>,
    sel_ev: Event,
}

impl Bufs {
    fn new(id: i32, rng: &mut Lcg) -> eyre::Result<Self> {
        let mut wmk = |rng: &mut Lcg| -> eyre::Result<DeviceBuffer<u8>> {
            let w: Vec<u8> = (0..NX * NE).flat_map(|_| {
                let v = rng.sym(0.03);
                // f16 bits of small values via the f32 bit pattern (good enough for a bench)
                let h = half_from_f32(v);
                h.to_le_bytes()
            }).collect();
            up(id, &w)
        };
        let w0 = wmk(rng)?;
        let w1 = wmk(rng)?;
        let mut prior_pin = PinnedBuffer::<f32>::new(64 * NX)?;
        for (i, v) in prior_pin.as_mut_slice().iter_mut().enumerate() {
            *v = if i % 3 == 0 { 0.02 } else { 0.0 };
        }
        Ok(Self {
            w0,
            w1,
            bias0: up(id, &(0..NX).map(|_| rng.sym(0.05)).collect::<Vec<_>>())?,
            bias1: up(id, &(0..NX).map(|_| rng.sym(0.05)).collect::<Vec<_>>())?,
            norm_w: up(id, &(0..NE).map(|_| 0.5 + rng.sym(0.4).abs()).collect::<Vec<_>>())?,
            x: up(id, &(0..BMAX * 4 * NE).map(|_| rng.sym(1.0)).collect::<Vec<_>>())?,
            x_src: up(id, &(0..BMAX * 4 * NE).map(|_| rng.sym(1.0)).collect::<Vec<_>>())?,
            carry: up(id, &(0..BMAX * 24).map(|_| 0.1 + rng.sym(0.5).abs()).collect::<Vec<_>>())?,
            cur: DeviceBuffer::new(id, BMAX * NE)?,
            ffn_in: DeviceBuffer::new(id, BMAX * NE)?,
            logits: DeviceBuffer::new(id, BMAX * NX)?,
            logits2: DeviceBuffer::new(id, BMAX * NX)?,
            sel: DeviceBuffer::new(id, BMAX * NU)?,
            ew: DeviceBuffer::new(id, BMAX * NU)?,
            orig: DeviceBuffer::new(id, BMAX * NU)?,
            range: DeviceBuffer::new(id, BMAX)?,
            look_sel: DeviceBuffer::new(id, BMAX * NU)?,
            look_ew: DeviceBuffer::new(id, BMAX * NU)?,
            d_prior: DeviceBuffer::new(id, NX)?,
            prior_pin,
            xq: DeviceBuffer::new(id, BMAX * XQ_ROW)?,
            pack: PinnedBuffer::new(BMAX * (6 * NU + 9) + BMAX * XQ_ROW / 4)?,
            bank: up(id, &vec![0u32; 64 * NX])?,
            done: PinnedBuffer::new(16)?,
            sel_ev: Event::new_no_timing()?,
        })
    }
}

fn half_from_f32(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | ((x >> 13) & 0x3ff) as u16
}

/// Production's chain (cut at selected_ready when `!tail`; `ev` = record selected_ready).
fn cur_group(e: &DeviceEngine, s: &Stream, bf: &mut Bufs, b: u32, ev: bool, tail: bool) -> eyre::Result<()> {
    let bu = b as usize;
    e.mhc_arena.launch_fast(
        s,
        None,
        Some(FastCollapse { x: &bf.x, cur_out: &mut bf.cur, norm_out: &mut bf.ffn_in, norm_w: &bf.norm_w }),
        &mut bf.carry,
        false,
        HC_DIM,
        RMS_EPS,
        SINKHORN_ITERS,
        SINKHORN_EPS,
        b,
    )?;
    if !e.f16.matvec_batched_router_x2(s, &mut bf.logits, &mut bf.logits2, &bf.w0, &bf.w1, &bf.ffn_in, N_EXPERT, N_EMBD, b)? {
        return Err(eyre!("router_x2 declined b = {b}"));
    }
    let slot = (LAYER as usize) * NX;
    bf.d_prior.copy_from_host_async(&bf.prior_pin.as_slice()[slot..slot + NX], s)?;
    e.router_topk.launch_batched_ex(
        s,
        &mut bf.sel,
        &mut bf.ew,
        &bf.logits,
        Some(&bf.bias0),
        N_EXPERT,
        NU as u32,
        EXPERT_WEIGHT_SCALE,
        ROUTER_WEIGHT_EPS,
        b,
        RouterEx {
            alts: None,
            n_alt: 0,
            alt_w: None,
            prior: Some(&bf.d_prior),
            n_protect: 2,
            prior_dry: false,
            orig_sel: Some(&mut bf.orig),
            range_out: Some(&mut bf.range),
        },
    )?;
    e.router_topk.launch_batched(s, &mut bf.look_sel, &mut bf.look_ew, &bf.logits2, Some(&bf.bias1), N_EXPERT, NU as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS, b)?;
    e.q8k.launch(s, &mut bf.xq, &bf.ffn_in, 20 * b)?;
    {
        let n_sel = bu * NU;
        let mut plan = PackPlan::default();
        plan.push(PackSeg::words(&bf.sel, n_sel)?);
        plan.push(PackSeg::words(&bf.look_sel, n_sel)?);
        plan.push(PackSeg::words(&bf.look_ew, n_sel)?);
        plan.push(PackSeg::words(&bf.orig, n_sel)?);
        plan.push(PackSeg::words(&bf.range, bu)?);
        plan.push(PackSeg::words(&bf.ew, n_sel)?);
        plan.push(PackSeg::bytes(&bf.xq, bu * XQ_ROW)?);
        // SAFETY: the pack is a separate allocation from the plan's device buffers.
        let pack: &mut PinnedBuffer<u32> = unsafe { &mut *(&mut bf.pack as *mut PinnedBuffer<u32>) };
        e.rb_pack.launch(s, pack, plan.segs())?;
    }
    if ev {
        bf.sel_ev.record(s)?;
    }
    if tail {
        e.expert_sel_count.launch(s, &mut bf.bank, &bf.orig, LAYER, N_EXPERT, b * NU as u32)?;
    }
    Ok(())
}

/// Rewrite after_attn_hc's live rows with a KERNEL (x *= 1.0, values unchanged), as production's
/// hc_post writes them right before the group: the collapse then reads them from L2. (An SDMA copy
/// would leave them in DRAM: a pessimistic collapse.)
fn touch_x(e: &DeviceEngine, s: &Stream, bf: &mut Bufs, b: u32) -> eyre::Result<()> {
    let _ = &bf.x_src;
    e.vec_scale.launch(s, &mut bf.x, 1.0, b * 4 * NE as u32)?;
    // and the carry (production: written by the previous sub-block's mix-late launch)
    e.vec_scale.launch(s, &mut bf.carry, 1.0, b * 24)
}

fn offsets(b: usize) -> G1PackOffsets {
    let n = (b * NU) as u32;
    G1PackOffsets { sel: 0, look: n, look_ew: 2 * n, orig: 3 * n, range: 4 * n, ew: 4 * n + b as u32, xq: 5 * n + b as u32 }
}

fn g1_group(e: &DeviceEngine, s: &Stream, bf: &mut Bufs, b: u32, v: (u32, u32, u32), ev: bool, done: Option<u32>) -> eyre::Result<()> {
    let pack_ptr: *mut PinnedBuffer<u32> = &mut bf.pack;
    let done_ptr: *const PinnedBuffer<u32> = &bf.done;
    let io = G1Io {
        collapse: G1Collapse { x: &bf.x, carry: &bf.carry, norm_w: &bf.norm_w, cur_out: &mut bf.cur, norm_out: &mut bf.ffn_in, rms_eps: RMS_EPS },
        w0: &bf.w0,
        bias0: Some(&bf.bias0),
        logits0: &mut bf.logits,
        look: Some(G1Look { w: &bf.w1, bias: Some(&bf.bias1), logits: &mut bf.logits2, sel: &mut bf.look_sel, ew: &mut bf.look_ew }),
        prior: Some(G1Prior { host: &bf.prior_pin, host_off: LAYER as usize * NX, dev: &mut bf.d_prior, n_protect: 2, dry: false }),
        sel: &mut bf.sel,
        ew: &mut bf.ew,
        orig_sel: Some(&mut bf.orig),
        range: Some(&mut bf.range),
        xq: Some(&mut bf.xq),
        // SAFETY: separate allocations.
        pack: Some((unsafe { &mut *pack_ptr }, offsets(b as usize))),
        sel_bank: Some((&mut bf.bank, LAYER * N_EXPERT)),
        done: done.map(|v| (unsafe { &*done_ptr }, 0usize, v)),
        n_used: NU as u32,
        scale: EXPERT_WEIGHT_SCALE,
        weight_eps: ROUTER_WEIGHT_EPS,
        b,
    };
    e.g1.launch_variant(s, io, v)?;
    if ev {
        bf.sel_ev.record(s)?;
    }
    Ok(())
}

struct Rig {
    s: Stream,
    ea: Event,
    eb: Event,
    probe: WmmaProbe,
    ceil: DeviceCeilings,
    sink: DeviceBuffer<u32>,
    spin_out: DeviceBuffer<f32>,
    spin_a: DeviceBuffer<f32>,
    spin_b: DeviceBuffer<f32>,
    spin_iters: u32,
    flush: DeviceBuffer<u8>,
}

impl Rig {
    fn spin(&mut self) -> eyre::Result<()> {
        let st = unsafe { &*(&self.s as *const Stream) };
        self.probe.launch_fma_f32(st, &mut self.spin_out, &self.spin_a, &self.spin_b, self.spin_iters, 1, 32)
    }
    /// Clock ramp: ~`us` of a one-wave spin (the dGPU idles at 41 MHz sclk / 96 MHz mclk under auto
    /// DVFS; production decode keeps it busy, so every burst starts by waking the clocks).
    fn ramp(&mut self, us: f64) -> eyre::Result<()> {
        let st = unsafe { &*(&self.s as *const Stream) };
        let n = (us / 50.0).ceil() as usize;
        for _ in 0..n {
            self.spin()?;
        }
        st.synchronize()
    }
    fn spin_time(&mut self, iters: u32) -> eyre::Result<f64> {
        let st = unsafe { &*(&self.s as *const Stream) };
        self.ea.record(st)?;
        self.probe.launch_fma_f32(st, &mut self.spin_out, &self.spin_a, &self.spin_b, iters, 1, 32)?;
        self.eb.record(st)?;
        st.synchronize()?;
        Ok(Event::elapsed_ms(&self.ea, &self.eb)? as f64 * 1e3)
    }
    fn do_flush(&mut self) -> eyre::Result<()> {
        let st = unsafe { &*(&self.s as *const Stream) };
        self.ceil.read_v4(st, &self.flush, &mut self.sink, 64 * 8, 256)
    }
    /// [flush] f(pre) spin evA f(measured) evB -> device us. `f(st, false)` = the untimed prelude
    /// (production's hc_post just WROTE after_attn_hc: the bench rewrites it after the flush, so the
    /// collapse reads it from L2 as in production).
    fn time_q(&mut self, cold: bool, f: &mut dyn FnMut(&Stream, bool) -> eyre::Result<()>) -> eyre::Result<f64> {
        let st = unsafe { &*(&self.s as *const Stream) };
        if cold {
            self.do_flush()?;
        }
        f(st, false)?;
        self.spin()?;
        self.ea.record(st)?;
        f(st, true)?;
        self.eb.record(st)?;
        st.synchronize()?;
        Ok(Event::elapsed_ms(&self.ea, &self.eb)? as f64 * 1e3)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Item {
    Cur { aa: bool, route: bool },
    G1 { v: (u32, u32, u32), route: bool },
    HostCur,
    HostG1Ev { v: (u32, u32, u32) },
    HostG1Flag { v: (u32, u32, u32) },
}

fn name(it: &Item) -> String {
    match it {
        Item::Cur { aa, route } => format!("cur{}{}", if *aa { "2" } else { "" }, if *route { "_route" } else { "" }),
        Item::G1 { v, route } => format!("g1{v:?}{}", if *route { "_route" } else { "" }),
        Item::HostCur => "host_cur_ev".into(),
        Item::HostG1Ev { v } => format!("host_g1{v:?}_ev"),
        Item::HostG1Flag { v } => format!("host_g1{v:?}_flag"),
    }
}

fn env_list(name: &str, def: &[u32]) -> Vec<u32> {
    std::env::var(name).ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_else(|| def.to_vec())
}

#[test]
#[ignore]
fn bench_g1_router_block() -> eyre::Result<()> {
    install_panic_handler();
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201 visible"))?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let mut rng = Lcg(0xb1_9e11);
    let mut bf = Bufs::new(id, &mut rng)?;
    let mut rig = Rig {
        s: Stream::new(id)?,
        ea: Event::new()?,
        eb: Event::new()?,
        probe: WmmaProbe::for_arch(&arch)?,
        ceil: DeviceCeilings::for_arch(&arch)?,
        sink: DeviceBuffer::new(id, 1 << 16)?,
        spin_out: DeviceBuffer::new(id, 32)?,
        spin_a: up(id, &[1.0f32])?,
        spin_b: up(id, &[0.999f32])?,
        spin_iters: 2000,
        flush: DeviceBuffer::new(id, FLUSH)?,
    };
    // spin calibration: ~50 us
    for _ in 0..3 {
        rig.spin_time(2000)?;
    }
    let t = pct(&(0..15).map(|_| rig.spin_time(2000)).collect::<eyre::Result<Vec<_>>>()?, 0.5);
    rig.spin_iters = ((2000.0 * 50.0 / t.max(1.0)) as u32).clamp(100, 2_000_000);
    let rounds: usize = std::env::var("BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    // burst shape: BENCH_RAMP_US of clock ramp before each round, BENCH_GAP_MS of pause after it
    let ramp_us: f64 = std::env::var("BENCH_RAMP_US").ok().and_then(|v| v.parse().ok()).unwrap_or(3000.0);
    let gap_ms: u64 = std::env::var("BENCH_GAP_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let bs = env_list("BENCH_B", &[1, 2, 4, 8, 16]);
    let warm = std::env::var("BENCH_WARM").as_deref() == Ok("1");
    let vlist: Vec<(u32, u32, u32)> = std::env::var("BENCH_G1_VARIANTS")
        .ok()
        .map(|v| {
            v.split(';')
                .filter_map(|p| {
                    let mut it = p.split(',').map(|x| x.trim().parse::<u32>().ok());
                    Some((it.next()??, it.next()??, it.next().flatten().unwrap_or(0)))
                })
                .collect()
        })
        .unwrap_or_default();

    // ---- floors: streaming read BW (cold) and one empty-ish launch (the pack kernel at 1 word)
    {
        let mut bw = Vec::new();
        for _ in 0..20 {
            let st = unsafe { &*(&rig.s as *const Stream) };
            rig.spin()?;
            rig.ea.record(st)?;
            rig.do_flush()?;
            rig.eb.record(st)?;
            st.synchronize()?;
            let us = Event::elapsed_ms(&rig.ea, &rig.eb)? as f64 * 1e3;
            bw.push(FLUSH as f64 / us / 1e3);
            std::thread::sleep(std::time::Duration::from_millis(3));
        }
        eprintln!("floor: cold streaming read {} MB: p50 {:.0} GB/s (p10 {:.0}, p90 {:.0})", FLUSH / MB, pct(&bw, 0.5), pct(&bw, 0.1), pct(&bw, 0.9));
    }

    // ---- in-kernel phase trace (V41_DGPU_G1_TS=1): where the fused launch spends its time.
    if std::env::var_os("V41_DGPU_G1_TS").is_some() {
        for &b in &bs {
            let mut vs = vec![v4flash_kernels::g1_router_block::g1_variant_for(b)];
            for v in &vlist {
                if !vs.contains(v) && v4flash_kernels::g1_router_block::g1_variant_ok(*v, b) {
                    vs.push(*v);
                }
            }
            for &v in &vs {
                let (zb, pf) = (v.0, v.1);
                let total = v4flash_kernels::g1_router_block::G1RouterBlock::grid_for(b, true, true, v) as usize;
                let nq = 1usize; // the prior's WG (family r: always present)
                let n_rwg = if v4flash_kernels::g1_router_block::g1_is_r(v) { 384 / v4flash_kernels::g1_router_block::g1r_nw(v) as usize } else { 48usize };
                let mut acc: Vec<Vec<f64>> = vec![Vec::new(); 24];
                // family r: router logits time by how many WGs share the router WG's WGP (HW_ID1 in slot 14)
                let mut by_occ: std::collections::BTreeMap<usize, Vec<f64>> = Default::default();
                // family r (CPP 8): per-phase completion time across router WGs (p50 / max per launch)
                let mut ph_p50: Vec<Vec<f64>> = vec![Vec::new(); 10];
                let mut ph_max: Vec<Vec<f64>> = vec![Vec::new(); 10];
                let mut slow_roles: std::collections::BTreeMap<String, usize> = Default::default();
                for _ in 0..rounds.min(60) {
                    rig.ramp(ramp_us)?;
                    rig.time_q(!warm, &mut |st, m| if m { g1_group(&e, st, &mut bf, b, v, false, None) } else { touch_x(&e, st, &mut bf, b) })?;
                    let ts = e.g1.ts_dump()?.ok_or_else(|| eyre!("no ts buffer"))?;
                    let t = |k: usize, s: usize| ts[k * 16 + s];
                    let t0 = (0..total).map(|k| t(k, 0)).min().unwrap_or(0);
                    let us = |v: u64| (v.saturating_sub(t0)) as f64 / 100.0; // 100 MHz
                    let coll: Vec<usize> = (0..b as usize).collect();
                    let rout: Vec<usize> = (b as usize + nq..total).collect();
                    let fslot = 2;
                    let q8w: Vec<usize> = (b as usize..b as usize + nq).collect();
                    let mx = |ks: &[usize], s: usize| ks.iter().map(|&k| us(t(k, s))).fold(0f64, f64::max);
                    let md = |ks: &[usize], s: usize| pct(&ks.iter().map(|&k| us(t(k, s))).collect::<Vec<_>>(), 0.5);
                    let tails: Vec<usize> = rout.iter().copied().filter(|&k| t(k, 5) >= t0 && t(k, 6) > t(k, 5) && t(k, 5) != 0).collect();
                    let vals = [
                        mx(&(0..total).collect::<Vec<_>>(), 0), // last WG start
                        md(&rout, 1),                            // router ticket p50
                        mx(&coll, fslot),                        // collapse flag (last row)
                        mx(&q8w, 2).max(mx(&coll, 3)),           // q8k / prior end
                        md(&rout, 2),                            // router prefetch issued p50
                        md(&rout, 3),                            // router flag seen p50
                        mx(&rout, 3),                            // router flag seen max
                        md(&rout, 4),                            // logits p50
                        mx(&rout, 4),                            // logits max
                        tails.iter().map(|&k| us(t(k, 5))).fold(f64::INFINITY, f64::min), // first tail start
                        tails.iter().map(|&k| us(t(k, 6))).fold(0f64, f64::max),           // last tail end
                        mx(&(0..total).collect::<Vec<_>>(), 7),  // last finish
                        tails.len() as f64,
                        n_rwg as f64,
                        pct(&tails.iter().map(|&k| us(t(k, 10)) - us(t(k, 5))).collect::<Vec<_>>(), 0.5), // tail: probs
                        pct(&tails.iter().map(|&k| us(t(k, 11)) - us(t(k, 10))).collect::<Vec<_>>(), 0.5), // tail: passes
                        pct(&tails.iter().map(|&k| us(t(k, 12)) - us(t(k, 11))).collect::<Vec<_>>(), 0.5), // tail: weights+emit
                        pct(&tails.iter().map(|&k| us(t(k, 8)) - us(t(k, 6))).collect::<Vec<_>>(), 0.5),   // tail again (warm I$)
                        us(t(0, 1)),  // collapse 0: ticket
                        us(t(0, 8)),  // collapse 0: cur computed (loads consumed)
                        us(t(0, 9)),  // collapse 0: rms reduced
                        us(t(0, 2)),  // collapse 0: flag
                        mx(&coll, 3),                // collapse: q8k done (max)
                        pct(&tails.iter().map(|&k| us(t(k, 13)) - us(t(k, 5))).collect::<Vec<_>>(), 0.5), // family r: tail loads
                    ];
                    for (i, v) in vals.iter().enumerate() {
                        acc[i].push(*v);
                    }
                    if v4flash_kernels::g1_router_block::g1_is_r(v) && v.2 / 1_000_000 % 100 == 8 {
                        for ph in 0..10 {
                            let vals: Vec<f64> = rout.iter().take(2 * n_rwg).map(|&k| us(ts[(1024 + k) * 16 + ph])).collect();
                            ph_p50[ph].push(pct(&vals, 0.5));
                            ph_max[ph].push(pct(&vals, 1.0));
                        }
                        // the 4 slowest router WGs at phase 2: which roles share their WGP?
                        let wgp = |k: usize| (t(k, 14) >> 10) & 0x7ff;
                        let mut rs: Vec<usize> = rout.iter().take(2 * n_rwg).copied().collect();
                        rs.sort_by(|&x, &y| ts[(1024 + y) * 16 + 2].cmp(&ts[(1024 + x) * 16 + 2]));
                        for &k in rs.iter().take(4) {
                            let mut co: Vec<String> = Vec::new();
                            for j in 0..total {
                                if j != k && wgp(j) == wgp(k) {
                                    let role = if j < b as usize { "collapse" } else if j < b as usize + nq { "prior" } else if j < b as usize + nq + 2 * n_rwg { "router" } else { "tail" };
                                    co.push(role.to_string());
                                }
                            }
                            co.sort();
                            *slow_roles.entry(co.join("+")).or_default() += 1;
                        }
                    }
                    if v4flash_kernels::g1_router_block::g1_is_r(v) {
                        let wgp = |k: usize| (t(k, 14) >> 10) & 0x7ff;
                        let mut occ: std::collections::HashMap<u64, usize> = Default::default();
                        let mut occr: std::collections::HashMap<u64, usize> = Default::default();
                        for k in 0..total {
                            *occ.entry(wgp(k)).or_default() += 1;
                        }
                        for &k in rout.iter().take(2 * n_rwg) {
                            *occr.entry(wgp(k)).or_default() += 1;
                        }
                        for &k in rout.iter().take(2 * n_rwg) {
                            by_occ.entry(10 * occ[&wgp(k)] + occr[&wgp(k)]).or_default().push(us(t(k, 4)));
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(gap_ms));
                }
                let names = ["last WG start", "router ticket p50", "collapse flag (max)", "q8k/prior end (max)", "router prefetch issued p50",
                    "router flag seen p50", "router flag seen max", "router logits p50", "router logits max", "tail start (first)", "tail end (last)",
                    "last finish", "tails", "router WGs/mat", "tail row0: probs", "tail row0: passes", "tail row0: emit", "tail again (dbg 32)", "collapse0 ticket", "collapse0 cur done", "collapse0 rms done", "collapse0 flag",
                    "collapse q8k done max", "r: tail scores loaded"];
                eprintln!("\n== TRACE b = {b} {v:?} ({}; us from the first WG's start, p50 over launches)", if warm { "WARM" } else { "COLD" });
                for (i, n) in names.iter().enumerate() {
                    eprintln!("  {:<28} {:>8.2}   (p10 {:>7.2}, p90 {:>7.2})", n, pct(&acc[i], 0.5), pct(&acc[i], 0.1), pct(&acc[i], 0.9));
                }
                for (k, n) in &slow_roles {
                    eprintln!("  slowest-4 router WGs share their WGP with [{k}]: {n}");
                }
                if !ph_p50[0].is_empty() {
                    let mut line = String::from("  phase done (p50 WG / last WG):");
                    for ph in 0..10 {
                        line += &format!(" {:.1}/{:.1}", pct(&ph_p50[ph], 0.5), pct(&ph_max[ph], 0.5));
                    }
                    eprintln!("{line}");
                }
                for (o, v) in &by_occ {
                    eprintln!("  router WGs on a WGP holding {} WGs ({} routers): {:>5} samples, logits p50 {:>6.2} p90 {:>6.2} max {:>6.2}", o / 10, o % 10, v.len(), pct(v, 0.5), pct(v, 0.9), pct(v, 1.0));
                }
            }
        }
        return Ok(());
    }

    for &b in &bs {
        let bu = b as usize;
        let dv = v4flash_kernels::g1_router_block::g1_variant_for(b);
        let mut variants = vec![dv];
        for v in &vlist {
            if !variants.contains(v) && v4flash_kernels::g1_router_block::g1_variant_ok(*v, b) {
                variants.push(*v);
            }
        }
        let mut items: Vec<Item> = vec![
            Item::Cur { aa: false, route: false },
            Item::Cur { aa: true, route: false },
            Item::Cur { aa: false, route: true },
            Item::HostCur,
        ];
        for &v in &variants {
            items.push(Item::G1 { v, route: false });
            items.push(Item::G1 { v, route: true });
        }
        items.push(Item::HostG1Ev { v: dv });
        items.push(Item::HostG1Flag { v: dv });
        let mut res: Vec<Vec<f64>> = vec![Vec::new(); items.len()];
        let mut done_v = 1u32;
        // warmup
        for _ in 0..5 {
            rig.time_q(true, &mut |st, m| if m { cur_group(&e, st, &mut bf, b, true, true) } else { touch_x(&e, st, &mut bf, b) })?;
            rig.time_q(true, &mut |st, m| if m { g1_group(&e, st, &mut bf, b, dv, true, None) } else { touch_x(&e, st, &mut bf, b) })?;
        }
        for _round in 0..rounds {
            // one BURST per round: clock ramp, every item back to back, then a pause (low duty)
            rig.ramp(ramp_us)?;
            let mut order: Vec<usize> = (0..items.len()).collect();
            for i in (1..order.len()).rev() {
                let j = (rng.next() as usize) % (i + 1);
                order.swap(i, j);
            }
            for &k in &order {
                let it = items[k].clone();
                let us = match it {
                    Item::Cur { route, .. } => rig.time_q(!warm, &mut |st, m| if m { cur_group(&e, st, &mut bf, b, !route, !route) } else { touch_x(&e, st, &mut bf, b) })?,
                    Item::G1 { v, route } => rig.time_q(!warm, &mut |st, m| if m { g1_group(&e, st, &mut bf, b, v, !route, None) } else { touch_x(&e, st, &mut bf, b) })?,
                    Item::HostCur | Item::HostG1Ev { .. } | Item::HostG1Flag { .. } => {
                        let st = unsafe { &*(&rig.s as *const Stream) };
                        if !warm {
                            rig.do_flush()?;
                        }
                        touch_x(&e, st, &mut bf, b)?;
                        st.synchronize()?;
                        let t0 = std::time::Instant::now();
                        match it {
                            Item::HostCur => {
                                cur_group(&e, st, &mut bf, b, true, true)?;
                                while !bf.sel_ev.query()? {}
                            }
                            Item::HostG1Ev { v } => {
                                g1_group(&e, st, &mut bf, b, v, true, None)?;
                                while !bf.sel_ev.query()? {}
                            }
                            Item::HostG1Flag { v } => {
                                done_v = done_v.wrapping_add(1).max(1);
                                g1_group(&e, st, &mut bf, b, v, false, Some(done_v))?;
                                let p = bf.done.device_ptr() as *const u32;
                                // SAFETY: pinned, host-coherent word written by the kernel's last WG.
                                while unsafe { std::ptr::read_volatile(p) } != done_v {
                                    std::hint::spin_loop();
                                }
                            }
                            _ => unreachable!(),
                        }
                        let us = t0.elapsed().as_secs_f64() * 1e6;
                        st.synchronize()?;
                        us
                    }
                };
                res[k].push(us);
            }
            std::thread::sleep(std::time::Duration::from_millis(gap_ms));
        }
        eprintln!("\n== b = {b} ({} rounds, {}; g1 default variant {:?}; grid {} WGs)", rounds, if warm { "WARM" } else { "COLD" }, dv,
            v4flash_kernels::g1_router_block::G1RouterBlock::grid(b, true, true));
        eprintln!("  {:<24} {:>8} {:>8} {:>8}   paired vs cur: p50  |d| p50", "item", "p10", "p50", "p90");
        let base = &res[0];
        for (k, it) in items.iter().enumerate() {
            let v = &res[k];
            let paired: Vec<f64> = v.iter().zip(base).map(|(a, c)| a - c).collect();
            let absd: Vec<f64> = paired.iter().map(|d| d.abs()).collect();
            eprintln!("  {:<24} {:>8.1} {:>8.1} {:>8.1}   {:>+9.1} {:>7.1}", name(it), pct(v, 0.1), pct(v, 0.5), pct(v, 0.9), pct(&paired, 0.5), pct(&absd, 0.5));
        }
        let _ = bu;
    }
    Ok(())
}
