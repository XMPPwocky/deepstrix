//! Chain-wall bench of the decode attention FRONT (fusion group G2, `V41_DGPU_QKV_FRONT`): the
//! production per-lane-layer chain (8 launches: quantize, q_a, rms_quant, q_b, rope_copy, kv,
//! kv_rms_rope_fp8, kv_append -- forward_prefill.rs q_chain + kv_chain + window append at decode
//! rows, dp4a arm, V41_DEC_FUSE) vs `QkvFront` (2 launches), replayed with real shapes on
//! synthetic Q8_0 weights, in ONE process beside the live hub, interleaved in random order per
//! round. Device time between two events behind a sleep kernel (the host has enqueued the whole
//! chain before the device reaches the first event, so the wall is device time incl. every
//! inter-kernel gap; production's host enqueue is ~1.6-1.9 us per launch and stays ahead).
//!
//! Sections (BENCH_SECTION=floors|chain|grid|all, default all):
//!   floors : streaming BW of the production GEMVs (q_b 44.6 MB, q_a + kv 9.75 MB, cold) and the
//!            dependent empty-launch slope (16 vs 1 `slack_probe_spin(0)` on one stream).
//!   chain  : arms cur / cur2 (A/A) / new, plus the parts: qb (production q_b alone), new_a,
//!            new_b, qakv (production q_a + kv GEMVs alone); cold (MALL flushed by an 80 MB read,
//!            then the activation row re-touched so it is L2-resident as in production) and
//!            warm; b = BENCH_B (default 1,2,4,8). Paired per-round differences new - cur and
//!            cur2 - cur (the A/A spread).
//!   grid   : launch A's work-group count sweep (cold), b = 4 and 8.
//!
//! VRAM ~140 MB (weights 54 MB + 80 MB flush range + scratch). Run (dGPU only, production kernarg
//! mode): target-fuse/ct.sh run bench_qkv_front_chain   [BENCH_ROUNDS=150 BENCH_SECTION=chain]
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::device_ceilings::DeviceCeilings;
use v4flash_kernels::wmma_probe::WmmaProbe;
use v4flash_kernels::q8_0::Q8_0Matvec;
use v4flash_kernels::qkv_front::{grid_a, grid_one, QkvFront};
use v4flash_kernels::{KvCacheAppend, RmsNorm, RopeParams, RopeTail};

const MB: usize = 1 << 20;
const FLUSH: usize = 80 * MB;
const EPS: f32 = 1e-6;
const W_QA: usize = 1280 * 160 * 34;
const W_KV: usize = 512 * 160 * 34;
const W_QB: usize = 32768 * 40 * 34;

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
fn q8_weight(rng: &mut Lcg, rows: usize, k: usize) -> Vec<u8> {
    let blocks = k / 32;
    let mut out = Vec::with_capacity(rows * blocks * 34);
    for _ in 0..rows {
        for _ in 0..blocks {
            // f16 0.002..0.012, random sign
            let h: u16 = 0x1818 + (rng.next() % 0x0a00) as u16 | if rng.next() % 2 == 0 { 0x8000 } else { 0 };
            out.extend_from_slice(&h.to_le_bytes());
        }
        for _ in 0..blocks * 32 {
            out.push((rng.next() & 0xff) as u8);
        }
    }
    out
}
fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * p).round() as usize]
}

struct Bufs {
    x: DeviceBuffer<f32>,
    xq: DeviceBuffer<i8>,
    xs: DeviceBuffer<f32>,
    xq_dummy: DeviceBuffer<i8>,
    xs_dummy: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    qr_normed: DeviceBuffer<f32>,
    qr_xq: DeviceBuffer<i8>,
    qr_xs: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    kv_raw: DeviceBuffer<f32>,
    kv_normed: DeviceBuffer<f32>,
    cache: DeviceBuffer<u16>,
    pos: DeviceBuffer<i32>,
    slots: DeviceBuffer<i32>,
    ticket: DeviceBuffer<u32>,
    epoch: u32,
    qa_norm: DeviceBuffer<f32>,
    kv_norm: DeviceBuffer<f32>,
}

struct Rig {
    s: Stream,
    ea: Event,
    eb: Event,
    q8: Q8_0Matvec,
    rms: RmsNorm,
    rope: RopeTail,
    app: KvCacheAppend,
    qf: QkvFront,
    ceil: DeviceCeilings,
    sink: DeviceBuffer<u32>,
    flush: DeviceBuffer<u8>,
    w: DeviceBuffer<u8>, // [q_a | kv | q_b]
    spin_ticks: u64,
    p: RopeParams,
    probe: WmmaProbe,
    spin_out: DeviceBuffer<f32>,
    spin_a: DeviceBuffer<f32>,
    spin_b: DeviceBuffer<f32>,
    /// `BENCH_SPIN=sleep`: the s_sleep kernel (GPU idle ahead of the sample) instead of the FMA
    /// spinner (one busy wave, as the map bench; production's dGPU is never idle mid-chain).
    sleep_spin: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    Cur,
    Cur2,
    New,
    Qb,
    NewA,
    NewB,
    QaKv,
    One,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Cur => "cur",
            Arm::Cur2 => "cur2",
            Arm::New => "new",
            Arm::Qb => "qb",
            Arm::NewA => "new_a",
            Arm::NewB => "new_b",
            Arm::QaKv => "qakv",
            Arm::One => "one",
        }
    }
    fn launches(self) -> u32 {
        match self {
            Arm::Cur | Arm::Cur2 => 8,
            Arm::New | Arm::QaKv => 2,
            Arm::Qb | Arm::NewA | Arm::NewB | Arm::One => 1,
        }
    }
}

impl Rig {
    fn spin(&self, n: u64) -> eyre::Result<()> {
        if self.sleep_spin {
            return self.q8.slack_probe_spin(&self.s, n);
        }
        let mut out = self.spin_out.slice_view(0, self.spin_out.len());
        self.probe.launch_fma_f32(&self.s, &mut out, &self.spin_a, &self.spin_b, n as u32, 1, 32)
    }
    fn wqa(&self) -> DeviceBuffer<u8> {
        self.w.slice_view(0, W_QA)
    }
    fn wkv(&self) -> DeviceBuffer<u8> {
        self.w.slice_view(W_QA, W_KV)
    }
    fn wqb(&self) -> DeviceBuffer<u8> {
        self.w.slice_view(W_QA + W_KV, W_QB)
    }
    fn run(&self, arm: Arm, d: &mut Bufs, b: u32, grid: u32) -> eyre::Result<()> {
        let s = &self.s;
        let (wqa, wkv, wqb) = (self.wqa(), self.wkv(), self.wqb());
        match arm {
            Arm::Cur | Arm::Cur2 => {
                self.q8.quantize_input_batched(s, &mut d.xq, &mut d.xs, &d.x, 5120, b)?;
                self.q8.matvec_batched(s, &mut d.qr, &wqa, &d.xq, &d.xs, 1280, 5120, b)?;
                self.rms.launch_weighted_quant_q8_1280(s, &mut d.qr_normed, &mut d.qr_xq, &mut d.qr_xs, &d.qr, &d.qa_norm, EPS, b)?;
                self.q8.matvec_batched(s, &mut d.q, &wqb, &d.qr_xq, &d.qr_xs, 32768, 1280, b)?;
                self.rope.launch_forward_batched_copy(s, &mut d.q_normed, &d.q, &d.pos, 64, 512, 64, b, &self.p)?;
                self.q8.matvec_batched(s, &mut d.kv_raw, &wkv, &d.xq, &d.xs, 512, 5120, b)?;
                self.rope.launch_kv_rms_rope_fp8(s, &mut d.kv_normed, &d.kv_raw, &d.kv_norm, EPS, &d.pos, 64, b, &self.p)?;
                self.app.launch_batched_rows(s, &mut d.cache, &d.kv_normed, 0, 512, b, Some(&d.slots))
            }
            Arm::New => {
                self.qf.launch_a(s, &mut d.qr, &mut d.kv_raw, &wqa, &wkv, &d.x, &mut d.qr_normed, &mut d.qr_xq, &mut d.qr_xs,
                    &d.qa_norm, EPS, &mut d.ticket, b, grid)?;
                self.qf.launch_b(s, &mut d.q_normed, &wqb, &d.qr_xq, &d.qr_xs, &d.pos, &self.p, &mut d.cache, &d.kv_raw,
                    &d.kv_norm, EPS, Some(&d.slots), 0, b)
            }
            Arm::NewA => self.qf.launch_a(s, &mut d.qr, &mut d.kv_raw, &wqa, &wkv, &d.x, &mut d.qr_normed, &mut d.qr_xq,
                &mut d.qr_xs, &d.qa_norm, EPS, &mut d.ticket, b, grid),
            Arm::NewB => self.qf.launch_b(s, &mut d.q_normed, &wqb, &d.qr_xq, &d.qr_xs, &d.pos, &self.p, &mut d.cache,
                &d.kv_raw, &d.kv_norm, EPS, Some(&d.slots), 0, b),
            Arm::Qb => self.q8.matvec_batched(s, &mut d.q, &wqb, &d.qr_xq, &d.qr_xs, 32768, 1280, b),
            Arm::QaKv => {
                self.q8.matvec_batched(s, &mut d.qr, &wqa, &d.xq, &d.xs, 1280, 5120, b)?;
                self.q8.matvec_batched(s, &mut d.kv_raw, &wkv, &d.xq, &d.xs, 512, 5120, b)
            }
            Arm::One => {
                d.epoch = d.epoch.wrapping_add(1).max(1);
                self.qf.launch_one(s, &mut d.qr, &mut d.kv_raw, &wqa, &wkv, &d.x, &mut d.qr_normed, &mut d.qr_xq,
                    &mut d.qr_xs, &d.qa_norm, EPS, &mut d.ticket, d.epoch, &mut d.q_normed, &wqb, &d.pos, &self.p,
                    &mut d.cache, &d.kv_norm, Some(&d.slots), 0, b, grid_one(b))
            }
        }
    }
    /// [flush + re-touch x] [sleep kernel] evA <f> evB -> device us.
    fn sample(&self, cold: bool, d: &mut Bufs, b: u32, f: &mut dyn FnMut(&Self, &mut Bufs) -> eyre::Result<()>) -> eyre::Result<f64> {
        let s = &self.s;
        if cold {
            let mut sink = self.sink.slice_view(0, self.sink.len());
            self.ceil.read_v4(s, &self.flush, &mut sink, 64 * 8, 256)?;
            // the previous kernel just wrote attn_input_norm: L2-resident in production
            self.q8.quantize_input_batched(s, &mut d.xq_dummy, &mut d.xs_dummy, &d.x, 5120, b)?;
        }
        self.spin(self.spin_ticks)?;
        self.ea.record(s)?;
        f(self, d)?;
        self.eb.record(s)?;
        s.synchronize()?;
        Ok(Event::elapsed_ms(&self.ea, &self.eb)? as f64 * 1e3)
    }
}

fn bufs(id: i32, rng: &mut Lcg) -> eyre::Result<Bufs> {
    let bm = 8usize;
    Ok(Bufs {
        x: upload(id, &(0..bm * 5120).map(|_| rng.sym(2.0)).collect::<Vec<_>>())?,
        xq: upload(id, &vec![0i8; bm * 5120])?,
        xs: upload(id, &vec![0f32; bm * 160])?,
        xq_dummy: upload(id, &vec![0i8; bm * 5120])?,
        xs_dummy: upload(id, &vec![0f32; bm * 160])?,
        qr: upload(id, &vec![0f32; bm * 1280])?,
        qr_normed: upload(id, &vec![0f32; bm * 1280])?,
        qr_xq: upload(id, &vec![0i8; bm * 1280])?,
        qr_xs: upload(id, &vec![0f32; bm * 40])?,
        q: upload(id, &vec![0f32; bm * 32768])?,
        q_normed: upload(id, &vec![0f32; bm * 32768])?,
        kv_raw: upload(id, &vec![0f32; bm * 512])?,
        kv_normed: upload(id, &vec![0f32; bm * 512])?,
        cache: upload(id, &vec![0u16; 256 * 512])?,
        pos: upload(id, &(0..bm as i32).map(|i| 4000 + 17 * i).collect::<Vec<_>>())?,
        slots: upload(id, &(0..bm as i32).map(|i| 3 + 29 * i).collect::<Vec<_>>())?,
        ticket: upload(id, &[0u32; 4])?,
        epoch: 0,
        qa_norm: upload(id, &(0..1280).map(|_| 0.5 + rng.sym(0.3)).collect::<Vec<_>>())?,
        kv_norm: upload(id, &(0..512).map(|_| 0.5 + rng.sym(0.3)).collect::<Vec<_>>())?,
    })
}

fn env_list(name: &str, default: &[u32]) -> Vec<u32> {
    std::env::var(name)
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

#[test]
#[ignore]
fn bench_qkv_front_chain() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let section = std::env::var("BENCH_SECTION").unwrap_or_else(|_| "all".into());
    let rounds: usize = std::env::var("BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
    let bs = env_list("BENCH_B", &[1, 2, 4, 8]);
    let mut rng = Lcg(0xB00C_F00D);
    let mut wh = q8_weight(&mut rng, 1280, 5120);
    wh.extend(q8_weight(&mut rng, 512, 5120));
    wh.extend(q8_weight(&mut rng, 32768, 1280));
    let mut rig = Rig {
        s: Stream::new(id)?,
        ea: Event::new()?,
        eb: Event::new()?,
        q8: Q8_0Matvec::for_arch(&arch)?,
        rms: RmsNorm::for_arch(&arch)?,
        rope: RopeTail::for_arch(&arch)?,
        app: KvCacheAppend::for_arch(&arch)?,
        qf: QkvFront::for_arch(&arch)?,
        ceil: DeviceCeilings::for_arch(&arch)?,
        sink: upload(id, &vec![0u32; 1 << 16])?,
        flush: upload(id, &vec![1u8; FLUSH])?,
        w: upload(id, &wh)?,
        spin_ticks: 0,
        p: RopeParams { freq_base: 160000.0, freq_scale: 0.0625, ext_factor: 1.0, attn_factor: 1.0,
                        beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65536 },
        probe: WmmaProbe::for_arch(&arch)?,
        spin_out: upload(id, &vec![0f32; 64])?,
        spin_a: upload(id, &[1.0f32])?,
        spin_b: upload(id, &[0.999f32])?,
        sleep_spin: std::env::var("BENCH_SPIN").as_deref() == Ok("sleep"),
    };
    drop(wh);
    let mut d = bufs(id, &mut rng)?;
    // calibrate the spinner: ~60 us of queue ahead of the first event
    let mut t = Vec::new();
    for _ in 0..9 {
        rig.ea.record(&rig.s)?;
        rig.spin(2000)?;
        rig.eb.record(&rig.s)?;
        rig.s.synchronize()?;
        t.push(Event::elapsed_ms(&rig.ea, &rig.eb)? as f64 * 1e3);
    }
    let us_per_2000 = pct(&t, 0.5);
    rig.spin_ticks = (2000.0 * 60.0 / us_per_2000.max(1.0)) as u64;
    eprintln!("spinner ({}): 2000 = {us_per_2000:.1} us -> {}", if rig.sleep_spin { "sleep" } else { "fma" }, rig.spin_ticks);
    let pause = || std::thread::sleep(std::time::Duration::from_millis(3));

    if section == "floors" || section == "all" {
        let mut qb = Vec::new();
        let mut qakv = Vec::new();
        let mut l1 = Vec::new();
        let mut l16 = Vec::new();
        let mut rd = Vec::new();
        let mut rd_a = Vec::new();
        for _ in 0..rounds.min(80) {
            qb.push(rig.sample(true, &mut d, 4, &mut |r, d| r.run(Arm::Qb, d, 4, 64))?);
            qakv.push(rig.sample(true, &mut d, 4, &mut |r, d| r.run(Arm::QaKv, d, 4, 64))?);
            l1.push(rig.sample(false, &mut d, 4, &mut |r, _| r.q8.slack_probe_spin(&r.s, 0))?);
            l16.push(rig.sample(false, &mut d, 4, &mut |r, _| {
                for _ in 0..16 {
                    r.q8.slack_probe_spin(&r.s, 0)?;
                }
                Ok(())
            })?);
            // raw streaming read of the 54 MB weight range, cold
            rd.push(rig.sample(true, &mut d, 4, &mut |r, _| {
                let mut sink = r.sink.slice_view(0, r.sink.len());
                r.ceil.read_v4(&r.s, &r.w, &mut sink, 64 * 8, 256)
            })?);
            rd_a.push(rig.sample(true, &mut d, 4, &mut |r, _| {
                let mut sink = r.sink.slice_view(0, r.sink.len());
                let wa = r.w.slice_view(0, W_QA + W_KV);
                r.ceil.read_v4(&r.s, &wa, &mut sink, 64 * 8, 256)
            })?);
            pause();
        }
        let bw = |bytes: usize, us: f64| bytes as f64 / us / 1e3;
        eprintln!("FLOOR read_v4 q_a+kv range (9.75 MB) cold p10 {:.1} p50 {:.1} us = {:.0} GB/s",
            pct(&rd_a, 0.1), pct(&rd_a, 0.5), bw(W_QA + W_KV, pct(&rd_a, 0.5)));
        let (qb50, qakv50, rd50) = (pct(&qb, 0.5), pct(&qakv, 0.5), pct(&rd, 0.5));
        let (qb10, qakv10, rd10) = (pct(&qb, 0.1), pct(&qakv, 0.1), pct(&rd, 0.1));
        eprintln!("FLOOR bw: q_b tB4 cold p50 {qb50:.1} us = {:.0} GB/s (p10 {qb10:.1} = {:.0}); q_a+kv tB4 p50 {qakv50:.1} us = {:.0} GB/s (p10 {:.0}); read_v4 54 MB p50 {rd50:.1} = {:.0} GB/s (p10 {:.0})",
            bw(W_QB, qb50), bw(W_QB, qb10), bw(W_QA + W_KV, qakv50), bw(W_QA + W_KV, qakv10), bw(W_QA + W_KV + W_QB, rd50), bw(W_QA + W_KV + W_QB, rd10));
        let (a, b16) = (pct(&l1, 0.5), pct(&l16, 0.5));
        eprintln!("FLOOR launch: 1 empty {a:.1} us, 16 dependent {b16:.1} us -> {:.2} us per extra launch (p10 {:.2})",
            (b16 - a) / 15.0, (pct(&l16, 0.1) - pct(&l1, 0.1)) / 15.0);
    }

    if section == "probe" {
        // streaming-read model: t(bytes) for read_v4 over the head of the weight range, cold + warm
        let sizes_mb = [1.0f64, 2.79, 6.96, 9.75, 20.0, 44.56];
        for cold in [true, false] {
            let mut res: Vec<Vec<f64>> = vec![Vec::new(); sizes_mb.len() + 2];
            for _ in 0..rounds.min(60) {
                for (k, &mb) in sizes_mb.iter().enumerate() {
                    let n = ((mb * 1e6) as usize / 16) * 16;
                    res[k].push(rig.sample(cold, &mut d, 4, &mut |r, _| {
                        let mut sink = r.sink.slice_view(0, r.sink.len());
                        let v = r.w.slice_view(0, n);
                        r.ceil.read_v4(&r.s, &v, &mut sink, 64 * 8, 256)
                    })?);
                }
                let k = sizes_mb.len();
                res[k].push(rig.sample(cold, &mut d, 4, &mut |r, d| {
                    let wqa = r.wqa();
                    r.q8.matvec_batched(&r.s, &mut d.qr, &wqa, &d.xq, &d.xs, 1280, 5120, 4)
                })?);
                res[k + 1].push(rig.sample(cold, &mut d, 4, &mut |r, d| {
                    let wkv = r.wkv();
                    r.q8.matvec_batched(&r.s, &mut d.kv_raw, &wkv, &d.xq, &d.xs, 512, 5120, 4)
                })?);
                pause();
            }
            let tag = if cold { "cold" } else { "warm" };
            for (k, &mb) in sizes_mb.iter().enumerate() {
                eprintln!("PROBE {tag} read_v4 {mb:6.2} MB p10 {:6.1} p50 {:6.1} us ({:.0} GB/s at p50)", pct(&res[k], 0.1), pct(&res[k], 0.5), mb * 1e3 / pct(&res[k], 0.5));
            }
            let k = sizes_mb.len();
            eprintln!("PROBE {tag} q_a tB4 alone p50 {:.1} us; kv tB4 alone p50 {:.1} us", pct(&res[k], 0.5), pct(&res[k + 1], 0.5));
        }
    }

    if section == "variant" {
        // launch A from other builds of kernels/qkv_front.hip (BENCH_VARIANTS=a.hsaco,b.hsaco), cold
        let paths: Vec<String> = std::env::var("BENCH_VARIANTS").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
        let mods: Vec<QkvFront> = paths.iter().map(|p| QkvFront::from_image(&std::fs::read(p).expect("variant hsaco"))).collect::<eyre::Result<_>>()?;
        for &b in &bs {
            let g = grid_a(b);
            let mut res: Vec<Vec<f64>> = vec![Vec::new(); mods.len() + 1];
            for _ in 0..rounds.min(60) {
                for k in 0..=mods.len() {
                    res[k].push(rig.sample(true, &mut d, b, &mut |r, d| {
                        let qf = if k == 0 { &r.qf } else { &mods[k - 1] };
                        let (wqa, wkv) = (r.wqa(), r.wkv());
                        qf.launch_a(&r.s, &mut d.qr, &mut d.kv_raw, &wqa, &wkv, &d.x, &mut d.qr_normed, &mut d.qr_xq,
                            &mut d.qr_xs, &d.qa_norm, EPS, &mut d.ticket, b, g)
                    })?);
                }
                pause();
            }
            for k in 0..=mods.len() {
                let name = if k == 0 { "built-in" } else { paths[k - 1].as_str() };
                eprintln!("VARIANT b={b} G={g} {name}: p10 {:.1} p50 {:.1} p90 {:.1} us", pct(&res[k], 0.1), pct(&res[k], 0.5), pct(&res[k], 0.9));
            }
        }
    }

    if section == "chain" || section == "all" {
        let arms = [Arm::Cur, Arm::Cur2, Arm::New, Arm::Qb, Arm::NewA, Arm::NewB, Arm::QaKv, Arm::One];
        for &b in &bs {
            let g = grid_a(b);
            for cold in [true, false] {
                let mut res: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
                let mut order: Vec<usize> = (0..arms.len()).collect();
                for _ in 0..rounds {
                    for i in (1..order.len()).rev() {
                        let j = rng.next() as usize % (i + 1);
                        order.swap(i, j);
                    }
                    for &k in &order {
                        let arm = arms[k];
                        res[k].push(rig.sample(cold, &mut d, b, &mut |r, d| r.run(arm, d, b, g))?);
                    }
                    pause();
                }
                let tag = if cold { "cold" } else { "warm" };
                for (k, arm) in arms.iter().enumerate() {
                    eprintln!("CHAIN b={b} {tag} {:6} launches {} p10 {:7.1} p50 {:7.1} p90 {:7.1} us", arm.name(), arm.launches(),
                        pct(&res[k], 0.1), pct(&res[k], 0.5), pct(&res[k], 0.9));
                }
                let pd = |x: usize, y: usize| -> Vec<f64> { res[x].iter().zip(&res[y]).map(|(a, b)| a - b).collect() };
                let aa = pd(1, 0);
                let nd = pd(2, 0);
                let od = pd(7, 0);
                let on = pd(7, 2);
                let aa_abs: Vec<f64> = aa.iter().map(|v| v.abs()).collect();
                eprintln!("PAIRED b={b} {tag}: new-cur p10 {:.1} p50 {:.1} p90 {:.1} | A/A cur2-cur p50 {:.1} |d| p50 {:.1} p90 {:.1} | G2 part: cur-qb {:.1} new-qb {:.1}",
                    pct(&nd, 0.1), pct(&nd, 0.5), pct(&nd, 0.9), pct(&aa, 0.5), pct(&aa_abs, 0.5), pct(&aa_abs, 0.9),
                    pct(&res[0], 0.5) - pct(&res[3], 0.5), pct(&res[2], 0.5) - pct(&res[3], 0.5));
                eprintln!("PAIRED b={b} {tag}: one-cur p10 {:.1} p50 {:.1} p90 {:.1} | one-new p10 {:.1} p50 {:.1} p90 {:.1} | G2 part one-qb {:.1}",
                    pct(&od, 0.1), pct(&od, 0.5), pct(&od, 0.9), pct(&on, 0.1), pct(&on, 0.5), pct(&on, 0.9),
                    pct(&res[7], 0.5) - pct(&res[3], 0.5));
            }
        }
    }

    if section == "grid" || section == "all" {
        let grids = env_list("BENCH_GRIDS", &[32, 48, 56, 64, 80, 96, 128, 160, 224]);
        for &b in &[4u32, 8] {
            let mut res: Vec<Vec<f64>> = vec![Vec::new(); grids.len()];
            for _ in 0..rounds.min(100) {
                for (k, &g) in grids.iter().enumerate() {
                    res[k].push(rig.sample(true, &mut d, b, &mut |r, d| r.run(Arm::NewA, d, b, g))?);
                }
                pause();
            }
            for (k, &g) in grids.iter().enumerate() {
                eprintln!("GRID b={b} new_a G={g:3} cold p10 {:6.1} p50 {:6.1} p90 {:6.1} us", pct(&res[k], 0.1), pct(&res[k], 0.5), pct(&res[k], 0.9));
            }
        }
    }
    Ok(())
}
