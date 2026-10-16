//! Cache-reuse microbenchmarks: do the last-level caches hold OUR weights?
//!
//! Q1 (dGPU, gfx1201, 64 MB Infinity Cache, ~645 GB/s GDDR6). In decode the two lanes stream the
//! SAME per-layer attention weights ~600 us apart (q chain ~55 MB, output projection ~80 MB per
//! layer, Q8_0), with ~120-200 MB of other weights in between. Can ONE 44 / 55 MB matrix stay in
//! the IC across that stream if the loads carry temporal hints, so lane 2 reads it from the cache?
//! Measured per (M size, M load policy, S size, S load policy):
//!   [flush: 512 MB HT pass + 512 MB RT pass] -> M cold -> S MB of other weights -> M re-read
//! and, as the best case, [flush] -> M cold -> M warm (immediately). The HT flush pass is there
//! because an RT stream may not evict HT lines (then "cold" would not be cold).
//!
//! How to read it:
//!   Q1-A: cold/warm for RT on 44/55 MB must be clearly > 1 (the IC can hold M at all); the
//!         160 MB control must be ~1. HT cold ~= RT cold (else HT lines survived the flush).
//!   Q1-B: per row, `xRT` = RT-cold / re-read and `captured` = share of the best-case (Q1-A RT
//!         warm) gain the re-read keeps. aM=RT rows are the no-hint baseline (expect ~0% at
//!         S >= 64 MB). KEEP PURSUING cross-lane IC reuse if some (aM, aS) keeps captured >~ 60%
//!         at S = 128-200 MB -- ideally with aS = RT (only M's kernel changes), else with aS = NT
//!         (every other weight read must carry NT). DROP IT if every row stays < ~20-25%.
//!   Q1-C: the production GEMV's 2nd read, same sequences (the no-hint status quo).
//!   Q2-A: `dev` (the expert pool's hipMalloc) cold/warm at 16/24 MB clearly > 1 while the 64 MB
//!         control is ~1 => the MALL caches our iGPU weights; ~1 everywhere (beyond what L2 alone
//!         gives) => it does not. Same reading per host kind; aux dlc on M should kill the gain
//!         if dlc = MALL NOALLOC.
//!   Q2-B: whether a dlc / slc+dlc stream between the two reads protects M's MALL lines.
//!
//! Q2 (iGPU, gfx1151, 32 MB MALL). Does the MALL serve iGPU reads of our weights at all? The
//! expert pool is `hipMalloc` on the iGPU (`het/expert_pager.rs`), which on this kernel lands in
//! GTT anyway (apu_prefer_gtt, docs/IGPU_VRAM_POOL_PLAN.md); the miss staging the repack reads in
//! place is `hipHostMalloc(NON_COHERENT)`. Both are measured, plus coherent `hipHostMalloc(0)`:
//! cold vs immediate warm re-read (16 / 24 MB, 64 MB control > MALL), then the warm re-read after a
//! 32 / 64 MB stream read with dlc / slc dlc hints.
//!
//! Kernels: `kernels/cache_probe.hip` -- a row-streaming GEMV read (one wave per 5440 B Q8_0 row,
//! b128 loads, 4 in flight per lane, wrapping u32 sum per row; the host checks every row's sum
//! against the hash fill), instantiated per cache-policy immediate (`aux` 0..6) plus a flat
//! `global_load_b128` reference. Q1 also times the PRODUCTION Q8_0 GEMV (`q8_0_gemv_warp8`, no
//! hints) for the same sequences.
//!
//! Timing: hipEvents around single launches queued back to back on one stream, 2 warm-up reps,
//! then `CACHE_PROBE_REPS` (default 31) reps; median / min reported. A hard internal time limit:
//! each rep must finish within 2 s, the whole test soft-stops at 200 s and a watchdog kills the
//! process at 230 s. Memory: < 0.8 GB on the dGPU, < 0.5 GB device + 0.2 GB pinned on the iGPU.
//!
//! RUN ONLY WITH THE SERVER DOWN (both GPUs exclusive):
//!   CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels --features v41 \
//!     --release --test cache_reuse_bench -- --ignored --nocapture --test-threads=1
//! Env: CACHE_PROBE_REPS=N, CACHE_PROBE_W3=0x... (buffer descriptor dword 3 override),
//!      CACHE_PROBE_ONLY=q1|q2.

use std::ffi::c_void;
use std::time::{Duration, Instant};

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{
    install_panic_handler, launch_kernel, Device, DeviceBuffer, Event, LaunchConfig, Module,
    PinnedBuffer, Stream, HIP_HOST_MALLOC_NON_COHERENT,
};
use v4flash_kernels::Q8_0Matvec;

const CP_GFX1201: &[u8] = include_bytes!(env!("KERNEL_CACHE_PROBE_GFX1201"));
const CP_GFX1151: &[u8] = include_bytes!(env!("KERNEL_CACHE_PROBE_GFX1151"));

/// One Q8_0 row at K = 5120: 160 blocks x 34 B.
const K: u32 = 5120;
const ROW_BYTES: u32 = 5440;
const WORDS_PER_ROW: u32 = ROW_BYTES / 4;
const MB: usize = 1 << 20;
/// gfx11 / gfx12 raw-buffer descriptor dword 3 (32-bit format, OOB select 3, resource level 1).
const W3_DEFAULT: u32 = 0x3100_4000;

const REP_LIMIT: Duration = Duration::from_secs(2);
const SOFT_LIMIT: Duration = Duration::from_secs(200);
const HARD_LIMIT: Duration = Duration::from_secs(230);
const WARMUP: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Ld {
    Flat,
    Aux(u32),
}

fn ld_name(arch: &str, ld: Ld) -> String {
    match ld {
        Ld::Flat => "flat".into(),
        Ld::Aux(a) if arch.starts_with("gfx12") => {
            ["RT", "NT", "HT", "LU", "NT_RT", "RT_NT", "NT_HT", "BYPASS"][a as usize].into()
        }
        Ld::Aux(0) => "dflt".into(),
        Ld::Aux(a) => {
            let mut v = Vec::new();
            if a & 1 != 0 { v.push("glc"); }
            if a & 2 != 0 { v.push("slc"); }
            if a & 4 != 0 { v.push("dlc"); }
            v.join("+")
        }
    }
}

/// The fill pattern; identical to `cp_hash` in cache_probe.hip.
fn cp_hash(i: u32, seed: u32) -> u32 {
    let mut x = i.wrapping_mul(0x9E37_79B1).wrapping_add(seed);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    x
}

/// Per-row wrapping u32 sums of a buffer filled with `cp_hash(word, seed)`.
fn expected_row_sums(rows: u32, seed: u32) -> Vec<u32> {
    (0..rows)
        .map(|r| {
            let w0 = r * WORDS_PER_ROW;
            (w0..w0 + WORDS_PER_ROW).fold(0u32, |a, w| a.wrapping_add(cp_hash(w, seed)))
        })
        .collect()
}

fn rows_for(bytes: usize) -> u32 {
    (bytes / ROW_BYTES as usize) as u32
}

fn pick(prefix: &str) -> eyre::Result<Option<Device>> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(prefix) {
            return Ok(Some(d));
        }
    }
    Ok(None)
}

fn wait(ev: &Event, what: &str) -> eyre::Result<()> {
    let t0 = Instant::now();
    while !ev.query()? {
        if t0.elapsed() > REP_LIMIT {
            eprintln!(
                "cache_reuse_bench: {what} did not finish within {REP_LIMIT:?} -- GPU busy or hung; \
                 aborting the process"
            );
            std::process::exit(3);
        }
        std::thread::sleep(Duration::from_micros(50));
    }
    Ok(())
}

struct Probe {
    m: Module,
    w3: u32,
    rows_per_wg: u32,
}

/// The production GEMV, for the Q1 "real consumer" rows.
struct Q8Rig {
    k: Q8_0Matvec,
    xq: DeviceBuffer<i8>,
    xs: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
}

enum Op<'a> {
    Read { base: *mut c_void, rows: u32, ld: Ld, out: &'a DeviceBuffer<u32> },
    Q8 { w: &'a DeviceBuffer<u8>, rows: u32 },
}

fn rd(base: *mut c_void, rows: u32, ld: Ld, out: &DeviceBuffer<u32>) -> Op<'_> {
    Op::Read { base, rows, ld, out }
}

struct Bench<'a> {
    s: Stream,
    evs: Vec<Event>,
    probe: Probe,
    q8: Option<Q8Rig>,
    t_start: Instant,
    failures: &'a mut Vec<String>,
}

#[derive(Clone, Copy)]
struct St {
    med: f64,
    min: f64,
}

fn stats(mut v: Vec<f64>) -> St {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    St { med: v[v.len() / 2], min: v[0] }
}

fn gbs(bytes: usize, us: f64) -> f64 {
    bytes as f64 / (us * 1e-6) / 1e9
}

impl Bench<'_> {
    fn over_soft_limit(&self) -> bool {
        self.t_start.elapsed() > SOFT_LIMIT
    }

    fn launch(&self, op: &Op) -> eyre::Result<()> {
        match *op {
            Op::Read { base, rows, ld, out } => {
                assert!(out.len() >= rows as usize, "out buffer too short for {rows} rows");
                let name = match ld {
                    Ld::Flat => "cache_probe_stream_flat".to_string(),
                    Ld::Aux(a) => format!("cache_probe_stream_a{a}"),
                };
                let f = self.probe.m.get_function(&name)?;
                let cfg = LaunchConfig::simple(rows.div_ceil(self.probe.rows_per_wg), 256);
                launch_kernel!(f, cfg, &self.s, [base, ROW_BYTES, rows, self.probe.w3, out.raw()])
            }
            Op::Q8 { w, rows } => {
                let q = self.q8.as_ref().expect("Q8 op without a Q8 rig");
                let mut o = q.out.slice_view(0, rows as usize);
                q.k.matvec(&self.s, &mut o, w, &q.xq, &q.xs, rows, K)
            }
        }
    }

    /// One rep: ops back to back, an event between each; returns per-op device us.
    fn rep(&self, ops: &[Op], what: &str) -> eyre::Result<Vec<f64>> {
        assert!(ops.len() < self.evs.len());
        self.evs[0].record(&self.s)?;
        for (i, op) in ops.iter().enumerate() {
            self.launch(op)?;
            self.evs[i + 1].record(&self.s)?;
        }
        wait(&self.evs[ops.len()], what)?;
        (0..ops.len())
            .map(|i| Ok(Event::elapsed_ms(&self.evs[i], &self.evs[i + 1])? as f64 * 1e3))
            .collect()
    }

    /// WARMUP + reps; per-op stats.
    fn measure(&self, ops: &[Op], reps: usize, what: &str) -> eyre::Result<Vec<St>> {
        let mut samples = vec![Vec::with_capacity(reps); ops.len()];
        for r in 0..WARMUP + reps {
            let t = self.rep(ops, what)?;
            if r >= WARMUP {
                for (i, x) in t.into_iter().enumerate() {
                    samples[i].push(x);
                }
            }
        }
        Ok(samples.into_iter().map(stats).collect())
    }

    /// Every row sum the probe wrote for this buffer must match the fill (proves every byte was
    /// loaded through the path under test). Recorded, not fatal, so one bad path does not cost
    /// the window the rest of the table; the test fails at the end.
    fn check(&mut self, out: &DeviceBuffer<u32>, expect: &[u32], what: &str) -> eyre::Result<bool> {
        let mut got = vec![0u32; expect.len()];
        out.slice_view(0, expect.len()).copy_to_host(&mut got)?;
        let bad = got.iter().zip(expect).filter(|(a, b)| a != b).count();
        if bad > 0 {
            let msg = format!("{what}: {bad}/{} row checksums WRONG", expect.len());
            eprintln!("  !! {msg}");
            self.failures.push(msg);
        }
        Ok(bad == 0)
    }

    fn fill(&self, base: *mut c_void, bytes: usize, seed: u32) -> eyre::Result<()> {
        let n_words = u32::try_from(bytes / 4).expect("fill > 16 GB");
        let f = self.probe.m.get_function("cache_probe_fill")?;
        launch_kernel!(f, LaunchConfig::simple(4096, 256), &self.s, [base, n_words, seed])?;
        self.s.synchronize()
    }
}

fn load_probe(arch: &str, wave: i32) -> eyre::Result<Probe> {
    let image = if arch.starts_with("gfx1201") {
        CP_GFX1201
    } else if arch.starts_with("gfx1151") {
        CP_GFX1151
    } else {
        return Err(eyre!("no cache_probe image for {arch}"));
    };
    let w3 = match std::env::var("CACHE_PROBE_W3") {
        Ok(s) => u32::from_str_radix(s.trim_start_matches("0x"), 16)
            .map_err(|e| eyre!("CACHE_PROBE_W3={s}: {e}"))?,
        Err(_) => W3_DEFAULT,
    };
    assert!(wave == 32 || wave == 64, "warp size {wave}");
    Ok(Probe { m: Module::load_data(image)?, w3, rows_per_wg: 256 / wave as u32 })
}

fn header(dev: &Device, what: &str) -> eyre::Result<String> {
    let p = dev.properties()?;
    eprintln!(
        "\n=== {what}: HIP dev {} '{}' {} | {} CUs | L2 {} MB | wave{} | integrated={}",
        dev.id,
        p.name,
        p.gcn_arch_name,
        p.multi_processor_count,
        p.l2_cache_size as usize / MB,
        p.warp_size,
        p.integrated
    );
    let arch = p.gcn_arch_name.clone();
    let map: Vec<String> =
        (0..7).map(|a| format!("{a}={}", ld_name(&arch, Ld::Aux(a)))).collect();
    eprintln!("    aux -> hint: {}  (flat = global_load_b128)", map.join(" "));
    Ok(arch)
}

fn reps() -> usize {
    std::env::var("CACHE_PROBE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(31)
}

fn only(q: &str) -> bool {
    std::env::var("CACHE_PROBE_ONLY").map(|v| v == q).unwrap_or(true)
}

// ------------------------------------------------------------------------------------------------
// Q1: dGPU Infinity Cache, cross-lane reuse of one matrix.
// ------------------------------------------------------------------------------------------------
fn q1(dev: &Device, t_start: Instant, failures: &mut Vec<String>) -> eyre::Result<()> {
    dev.set_current()?;
    let arch = header(dev, "Q1 dGPU Infinity Cache")?;
    let p = dev.properties()?;
    let id = dev.id;
    let n = reps();
    let probe = load_probe(&arch, p.warp_size)?;
    eprintln!("    buffer descriptor dword3 = {:#010x}", probe.w3);

    let flush_bytes = 512 * MB;
    let flush_rows = rows_for(flush_bytes);
    let other = DeviceBuffer::<u8>::new(id, flush_rows as usize * ROW_BYTES as usize)?;
    let out_s = DeviceBuffer::<u32>::new(id, flush_rows as usize)?;
    let out_m = DeviceBuffer::<u32>::new(id, rows_for(160 * MB) as usize)?;
    let mut xq = DeviceBuffer::<i8>::new(id, K as usize)?;
    xq.copy_from_host(&(0..K).map(|i| (i % 7) as i8 - 3).collect::<Vec<_>>())?;
    let mut xs = DeviceBuffer::<f32>::new(id, (K / 32) as usize)?;
    xs.copy_from_host(&vec![0.01f32; (K / 32) as usize])?;
    let q8 = Q8Rig {
        k: Q8_0Matvec::for_arch(&arch)?,
        xq,
        xs,
        out: DeviceBuffer::<f32>::new(id, rows_for(55 * MB) as usize)?,
    };
    let mut b = Bench {
        s: Stream::new(id)?,
        evs: (0..6).map(|_| Event::new()).collect::<eyre::Result<_>>()?,
        probe,
        q8: Some(q8),
        t_start,
        failures,
    };
    let other_seed = 0x5151;
    b.fill(other.raw(), other.byte_len(), other_seed)?;
    let other_sums = expected_row_sums(flush_rows, other_seed);
    // The per-rep FLUSH is two passes over the 512 MB other buffer: HT first (evicts M's lines
    // whatever hint they were loaded with -- an RT stream may not evict HT lines, which would make
    // the next rep's "cold" read warm), then RT (leaves the cache in the ordinary state).
    let flush = || [rd(other.raw(), flush_rows, Ld::Aux(2), &out_s), rd(other.raw(), flush_rows, Ld::Aux(0), &out_s)];
    const NF: usize = 2; // ops of the flush; timed ops start at st[NF]

    // Clock ramp + the flush path's own check.
    for _ in 0..20 {
        for op in flush() {
            b.launch(&op)?;
        }
    }
    b.s.synchronize()?;
    b.check(&out_s, &other_sums, "Q1 flush")?;
    let fl = b.measure(&flush(), 9, "Q1 flush ref")?;
    eprintln!(
        "    DRAM stream ref (512 MB): HT {:.0} us = {:.0} GB/s, then RT {:.0} us = {:.0} GB/s",
        fl[0].med,
        gbs(flush_bytes, fl[0].med),
        fl[1].med,
        gbs(flush_bytes, fl[1].med)
    );

    // (M MB, load) -> (cold us, warm us) medians from Q1-A.
    let mut a_of = std::collections::HashMap::new();
    // HT / NT_HT last, so a sticky line they might leave cannot flatter the other rows.
    let all_ld: Vec<Ld> = [Ld::Flat, Ld::Aux(0), Ld::Aux(1), Ld::Aux(3), Ld::Aux(4), Ld::Aux(5), Ld::Aux(2), Ld::Aux(6)].into();

    // ---- A. cold vs immediate warm, every load path (also validates every aux kernel) ----
    eprintln!("\n  [Q1-A] [flush HT+RT 512 MB] -> M cold -> M warm   (median / min us, GB/s at median; reps {n})");
    eprintln!("   M MB  load   |   cold us        GB/s |   warm us        GB/s | cold/warm");
    for (mb, lds) in [(44usize, all_ld.clone()), (55, all_ld.clone()), (160, vec![Ld::Flat, Ld::Aux(0), Ld::Aux(2)])] {
        let rows = rows_for(mb * MB);
        let bytes = rows as usize * ROW_BYTES as usize;
        let m = DeviceBuffer::<u8>::new(id, bytes)?;
        let seed = 0x1000 + mb as u32;
        b.fill(m.raw(), m.byte_len(), seed)?;
        let sums = expected_row_sums(rows, seed);
        for ld in lds {
            if b.over_soft_limit() {
                eprintln!("  SOFT TIME LIMIT: skipping the rest of Q1");
                return Ok(());
            }
            let mut ops: Vec<Op> = flush().into();
            ops.push(rd(m.raw(), rows, ld, &out_m));
            ops.push(rd(m.raw(), rows, ld, &out_m));
            let st = b.measure(&ops, n, "Q1-A")?;
            let ok = b.check(&out_m, &sums, &format!("Q1-A M {mb} MB {}", ld_name(&arch, ld)))?;
            let (c, w) = (st[NF], st[NF + 1]);
            eprintln!(
                "  {mb:>4}  {:<6} | {:>7.1}/{:>7.1} {:>6.0} | {:>7.1}/{:>7.1} {:>6.0} | x{:.2}{}",
                ld_name(&arch, ld),
                c.med, c.min, gbs(bytes, c.med),
                w.med, w.min, gbs(bytes, w.med),
                c.med / w.med,
                if ok { "" } else { "  CHECKSUM BAD" }
            );
            a_of.insert((mb, ld), (c.med, w.med));
        }
        drop(m);
    }

    // ---- B. M re-read after S MB of other weights, hinted ----
    eprintln!(
        "\n  [Q1-B] [flush] -> M(aM) cold -> S MB other(aS) -> M(aR) re-read\n         \
         xRT = RT cold[Q1-A] / reread;  captured = (RT cold - reread) / (RT cold - RT warm)[Q1-A]"
    );
    eprintln!("   M MB aM    aR    | S MB aS     |   cold us  GB/s |   S us   GB/s |  reread us(min)    GB/s |  xRT  | captured");
    let mut combos: Vec<(u32, u32, u32)> = Vec::new(); // (aM, aR, aS)
    for am in [0u32, 2, 6] {
        for as_ in [0u32, 1, 5] {
            combos.push((am, am, as_));
        }
    }
    for am in [2u32, 6] {
        for as_ in [1u32, 5] {
            combos.push((am, 3, as_)); // lane 2 reads with LU (last use)
        }
    }
    for mb in [44usize, 55] {
        let rows = rows_for(mb * MB);
        let bytes = rows as usize * ROW_BYTES as usize;
        let m = DeviceBuffer::<u8>::new(id, bytes)?;
        let seed = 0x2000 + mb as u32;
        b.fill(m.raw(), m.byte_len(), seed)?;
        let sums = expected_row_sums(rows, seed);
        let (rt_cold, rt_warm) = a_of.get(&(mb, Ld::Aux(0))).copied().unwrap_or((f64::NAN, f64::NAN));
        for &(am, ar, as_) in &combos {
            for smb in [64usize, 128, 200] {
                if b.over_soft_limit() {
                    eprintln!("  SOFT TIME LIMIT: skipping the rest of Q1");
                    return Ok(());
                }
                let srows = rows_for(smb * MB);
                let mut ops: Vec<Op> = flush().into();
                ops.push(rd(m.raw(), rows, Ld::Aux(am), &out_m));
                ops.push(rd(other.raw(), srows, Ld::Aux(as_), &out_s));
                ops.push(rd(m.raw(), rows, Ld::Aux(ar), &out_m));
                let st = b.measure(&ops, n, "Q1-B")?;
                let ok = b.check(&out_m, &sums, &format!("Q1-B M {mb} aM {am} aR {ar}"))?
                    & b.check(&out_s, &other_sums[..srows as usize], &format!("Q1-B S {smb} aS {as_}"))?;
                let (c, sv, r) = (st[NF], st[NF + 1], st[NF + 2]);
                eprintln!(
                    "  {mb:>4} {:<6}{:<6}| {smb:>4} {:<6} | {:>7.1} {:>5.0} | {:>6.1} {:>5.0} | {:>7.1}({:>6.1}) {:>6.0} | x{:.2} | {:>5.0}%{}",
                    ld_name(&arch, Ld::Aux(am)),
                    ld_name(&arch, Ld::Aux(ar)),
                    ld_name(&arch, Ld::Aux(as_)),
                    c.med, gbs(bytes, c.med),
                    sv.med, gbs(srows as usize * ROW_BYTES as usize, sv.med),
                    r.med, r.min, gbs(bytes, r.med),
                    rt_cold / r.med,
                    100.0 * (rt_cold - r.med) / (rt_cold - rt_warm),
                    if ok { "" } else { "  CHECKSUM BAD" }
                );
            }
        }
        drop(m);
    }

    // ---- C. the production Q8_0 GEMV (no hints) as the consumer ----
    eprintln!("\n  [Q1-C] production q8_0_gemv_warp8 (no hints): [flush] -> GEMV(M) -> [S(aS)] -> GEMV(M)");
    eprintln!("   M MB  S MB aS     |  GEMV cold us  GB/s |  GEMV 2nd us(min)   GB/s | cold/2nd");
    for mb in [44usize, 55] {
        let rows = rows_for(mb * MB);
        let bytes = rows as usize * ROW_BYTES as usize;
        let m = DeviceBuffer::<u8>::new(id, bytes)?;
        b.fill(m.raw(), m.byte_len(), 0x3000 + mb as u32)?;
        let mut cases: Vec<Option<(usize, u32)>> = vec![None];
        for smb in [64usize, 128, 200] {
            for as_ in [0u32, 1, 5] {
                cases.push(Some((smb, as_)));
            }
        }
        for c in cases {
            if b.over_soft_limit() {
                eprintln!("  SOFT TIME LIMIT: skipping the rest of Q1");
                return Ok(());
            }
            let mut ops: Vec<Op> = flush().into();
            ops.push(Op::Q8 { w: &m, rows });
            if let Some((smb, as_)) = c {
                ops.push(rd(other.raw(), rows_for(smb * MB), Ld::Aux(as_), &out_s));
            }
            ops.push(Op::Q8 { w: &m, rows });
            let st = b.measure(&ops, n, "Q1-C")?;
            let (first, last) = (st[NF], st[ops.len() - 1]);
            let label = match c {
                None => "   -  (warm)  ".to_string(),
                Some((smb, as_)) => format!("{smb:>4} {:<6}   ", ld_name(&arch, Ld::Aux(as_))),
            };
            eprintln!(
                "  {mb:>4} {label}| {:>8.1} {:>5.0}     | {:>8.1}({:>6.1}) {:>5.0}   | x{:.2}",
                first.med, gbs(bytes, first.med),
                last.med, last.min, gbs(bytes, last.med),
                first.med / last.med
            );
        }
        drop(m);
    }
    drop(other);
    Ok(())
}

// ------------------------------------------------------------------------------------------------
// Q2: iGPU MALL vs device / GTT / host-pinned memory.
// ------------------------------------------------------------------------------------------------
enum MBuf {
    Dev(DeviceBuffer<u8>),
    Host(PinnedBuffer<u8>),
}

impl MBuf {
    fn ptr(&self) -> *mut c_void {
        match self {
            MBuf::Dev(d) => d.raw(),
            MBuf::Host(h) => h.device_ptr(),
        }
    }
}

/// Host fill (the engine's pread-into-staging analogue), then evict every line from the CPU
/// caches so a coherent (snooped) GPU read cannot be served from the CPU L3 instead of memory.
fn host_fill(h: &mut PinnedBuffer<u8>, seed: u32) {
    let s = h.as_mut_slice();
    for (i, c) in s.chunks_exact_mut(4).enumerate() {
        c.copy_from_slice(&cp_hash(i as u32, seed).to_le_bytes());
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_clflush, _mm_mfence};
        _mm_mfence();
        for off in (0..s.len()).step_by(64) {
            _mm_clflush(s.as_ptr().add(off));
        }
        _mm_mfence();
    }
}

fn q2(dev: &Device, t_start: Instant, failures: &mut Vec<String>) -> eyre::Result<()> {
    dev.set_current()?;
    let arch = header(dev, "Q2 iGPU MALL")?;
    let p = dev.properties()?;
    let id = dev.id;
    let n = reps();
    let probe = load_probe(&arch, p.warp_size)?;
    eprintln!("    buffer descriptor dword3 = {:#010x}", probe.w3);
    eprintln!(
        "    kinds: dev = hipMalloc on the iGPU (the expert pool's path; GTT-backed on this kernel), \
         hostNC = hipHostMalloc(NON_COHERENT) (miss staging), hostC = hipHostMalloc(0) (coherent)"
    );

    let flush_bytes = 256 * MB;
    let flush_rows = rows_for(flush_bytes);
    let other = DeviceBuffer::<u8>::new(id, flush_rows as usize * ROW_BYTES as usize)?;
    let out_s = DeviceBuffer::<u32>::new(id, flush_rows as usize)?;
    let out_m = DeviceBuffer::<u32>::new(id, rows_for(64 * MB) as usize)?;
    let mut b = Bench {
        s: Stream::new(id)?,
        evs: (0..6).map(|_| Event::new()).collect::<eyre::Result<_>>()?,
        probe,
        q8: None,
        t_start,
        failures,
    };
    let other_seed = 0x6161;
    b.fill(other.raw(), other.byte_len(), other_seed)?;
    let other_sums = expected_row_sums(flush_rows, other_seed);
    let flush = || Op::Read { base: other.raw(), rows: flush_rows, ld: Ld::Aux(0), out: &out_s };
    for _ in 0..40 {
        b.launch(&flush())?;
    }
    b.s.synchronize()?;
    b.check(&out_s, &other_sums, "Q2 flush")?;
    let fl = b.measure(&[flush()], 9, "Q2 flush ref")?[0];
    eprintln!(
        "    DRAM stream ref (256 MB dev, dflt): {:.0} us = {:.0} GB/s (min {:.0} us)",
        fl.med,
        gbs(flush_bytes, fl.med),
        fl.min
    );

    let make = |kind: &str, bytes: usize, seed: u32, b: &Bench| -> eyre::Result<MBuf> {
        Ok(match kind {
            "dev" => {
                let d = DeviceBuffer::<u8>::new(id, bytes)?;
                b.fill(d.raw(), bytes, seed)?;
                MBuf::Dev(d)
            }
            "hostNC" | "hostC" => {
                let flags = if kind == "hostNC" { HIP_HOST_MALLOC_NON_COHERENT } else { 0 };
                let mut h = PinnedBuffer::<u8>::new_with_flags(bytes, flags)?;
                host_fill(&mut h, seed);
                MBuf::Host(h)
            }
            _ => unreachable!(),
        })
    };
    let kinds = ["dev", "hostNC", "hostC"];

    // ---- A. cold vs immediate warm; aux 4 (dlc) on M itself is the negative control ----
    eprintln!("\n  [Q2-A] [flush 256 MB] -> M cold -> M warm   (median / min us, GB/s at median; reps {n})");
    eprintln!("   kind    M MB  load     |   cold us        GB/s |   warm us        GB/s | cold/warm");
    let mut warm_of = std::collections::HashMap::new();
    for kind in kinds {
        for mb in [16usize, 24, 64] {
            let rows = rows_for(mb * MB);
            let bytes = rows as usize * ROW_BYTES as usize;
            let seed = 0x4000 + mb as u32;
            let m = make(kind, bytes, seed, &b)?;
            let sums = expected_row_sums(rows, seed);
            for ld in [Ld::Flat, Ld::Aux(0), Ld::Aux(4), Ld::Aux(6)] {
                if b.over_soft_limit() {
                    eprintln!("  SOFT TIME LIMIT: skipping the rest of Q2");
                    return Ok(());
                }
                let mr = || Op::Read { base: m.ptr(), rows, ld, out: &out_m };
                let st = b.measure(&[flush(), mr(), mr()], n, "Q2-A")?;
                let ok = b.check(&out_m, &sums, &format!("Q2-A {kind} {mb} MB {}", ld_name(&arch, ld)))?;
                eprintln!(
                    "   {kind:<7}{mb:>4}  {:<9}| {:>7.1}/{:>7.1} {:>6.0} | {:>7.1}/{:>7.1} {:>6.0} | x{:.2}{}",
                    ld_name(&arch, ld),
                    st[1].med, st[1].min, gbs(bytes, st[1].med),
                    st[2].med, st[2].min, gbs(bytes, st[2].med),
                    st[1].med / st[2].med,
                    if ok { "" } else { "  CHECKSUM BAD" }
                );
                warm_of.insert((kind, mb, ld), st[2].med);
            }
        }
    }

    // ---- B. M (dflt) re-read after S MB of other (dev) data with MALL hints ----
    eprintln!(
        "\n  [Q2-B] [flush] -> M(dflt) cold -> S MB other dev (aS) -> M(dflt) re-read   \
         captured = (cold - reread) / (cold - warm[Q2-A, dflt])"
    );
    eprintln!("   kind    M MB | S MB aS       |   cold us  GB/s |   S us   GB/s |  reread us(min)    GB/s | cold/re | captured");
    for kind in kinds {
        for mb in [16usize, 24] {
            let rows = rows_for(mb * MB);
            let bytes = rows as usize * ROW_BYTES as usize;
            let seed = 0x5000 + mb as u32;
            let m = make(kind, bytes, seed, &b)?;
            let sums = expected_row_sums(rows, seed);
            for smb in [32usize, 64] {
                for as_ in [0u32, 4, 6] {
                    if b.over_soft_limit() {
                        eprintln!("  SOFT TIME LIMIT: skipping the rest of Q2");
                        return Ok(());
                    }
                    let srows = rows_for(smb * MB);
                    let ops = [
                        flush(),
                        Op::Read { base: m.ptr(), rows, ld: Ld::Aux(0), out: &out_m },
                        Op::Read { base: other.raw(), rows: srows, ld: Ld::Aux(as_), out: &out_s },
                        Op::Read { base: m.ptr(), rows, ld: Ld::Aux(0), out: &out_m },
                    ];
                    let st = b.measure(&ops, n, "Q2-B")?;
                    let ok = b.check(&out_m, &sums, &format!("Q2-B {kind} {mb} MB"))?
                        & b.check(&out_s, &other_sums[..srows as usize], &format!("Q2-B S {smb} aS {as_}"))?;
                    let cap = warm_of
                        .get(&(kind, mb, Ld::Aux(0)))
                        .map(|w| format!("{:>5.0}%", 100.0 * (st[1].med - st[3].med) / (st[1].med - w)))
                        .unwrap_or_else(|| "   n/a".into());
                    eprintln!(
                        "   {kind:<7}{mb:>4} | {smb:>4} {:<9}| {:>7.1} {:>5.0} | {:>6.1} {:>5.0} | {:>7.1}({:>6.1}) {:>6.0} | x{:.2}   | {cap}{}",
                        ld_name(&arch, Ld::Aux(as_)),
                        st[1].med, gbs(bytes, st[1].med),
                        st[2].med, gbs(srows as usize * ROW_BYTES as usize, st[2].med),
                        st[3].med, st[3].min, gbs(bytes, st[3].med),
                        st[1].med / st[3].med,
                        if ok { "" } else { "  CHECKSUM BAD" }
                    );
                }
            }
        }
    }
    drop(other);
    Ok(())
}

#[test]
#[ignore]
fn cache_reuse_bench() -> eyre::Result<()> {
    install_panic_handler()?;
    let t_start = Instant::now();
    // The window can never hang on this test: a watchdog ends the process at HARD_LIMIT.
    std::thread::spawn(|| {
        std::thread::sleep(HARD_LIMIT);
        eprintln!("cache_reuse_bench: HARD TIME LIMIT {HARD_LIMIT:?} hit -- exiting the process");
        std::process::exit(4);
    });
    let mut failures = Vec::new();
    if only("q1") {
        match pick("gfx1201")? {
            Some(d) => q1(&d, t_start, &mut failures)?,
            None => eprintln!("Q1 SKIPPED: no gfx1201 visible (HIP_VISIBLE_DEVICES?)"),
        }
    }
    if only("q2") {
        match pick("gfx1151")? {
            Some(d) => q2(&d, t_start, &mut failures)?,
            None => eprintln!("Q2 SKIPPED: no gfx1151 visible (HIP_VISIBLE_DEVICES?)"),
        }
    }
    eprintln!("\ncache_reuse_bench: total {:.1} s", t_start.elapsed().as_secs_f64());
    if !failures.is_empty() {
        return Err(eyre!("{} checksum failures:\n  {}", failures.len(), failures.join("\n  ")));
    }
    Ok(())
}
