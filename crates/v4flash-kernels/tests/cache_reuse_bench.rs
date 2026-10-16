//! Cache-reuse microbenchmarks: do the last-level caches hold OUR weights?
//!
//! Q1 (dGPU, gfx1201, 64 MB Infinity Cache, ~645 GB/s GDDR6). In decode the two lanes stream the
//! SAME per-layer attention weights ~600 us apart (q chain ~55 MB, output projection ~80 MB per
//! layer, Q8_0), with ~120-200 MB of other weights in between. Can ONE 44 / 55 MB matrix stay in
//! the IC across that stream if the loads carry temporal hints, so lane 2 reads it from the cache?
//! Measured per (M size, M load policy, S size, S load policy):
//!   [flush: 512 MB passes HT, LU, RT] -> M cold -> S MB of other weights -> M re-read
//! and, as the best case, [flush] -> M cold -> M warm (immediately). HT evicts M whatever its hint,
//! LU marks those lines evictable, RT restores the ordinary state (sanity row: vs a PRISTINE RT-only-flush baseline taken before any hinted load).
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
//!   Q2 placement: sysfs mem_info_gtt_used / mem_info_vram_used deltas say GTT vs VRAM carve-out.
//!   A flush checksum failure ABORTS that question (wrong descriptor => junk rows). The log ends
//!   with a SUMMARY block (best captured at S 128/200 + hint pair, Q2 ratios + placement).
//!
//! Q2 (iGPU, gfx1151, 32 MB MALL). Does the MALL serve iGPU reads of our weights at all? The
//! expert pool is `hipMalloc` on the iGPU (`het/expert_pager.rs`), which on this kernel lands in
//! GTT anyway (apu_prefer_gtt, docs/IGPU_VRAM_POOL_PLAN.md); the miss staging the repack reads in
//! place is `hipHostMalloc(NON_COHERENT)`. Both are measured, plus coherent `hipHostMalloc(COHERENT)`:
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
    DeviceProperties, PinnedBuffer, Stream, HIP_HOST_MALLOC_COHERENT, HIP_HOST_MALLOC_NON_COHERENT,
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

/// Verify each flush pass on its own (output zeroed first, so a pass that writes nothing cannot
/// inherit the previous pass's correct sums). A wrong buffer descriptor makes raw buffer loads
/// return 0 at no cost: the flush would flush nothing and every later row would be junk, so the
/// caller ABORTS the question on `false`.
fn check_flush(
    b: &mut Bench,
    other: &DeviceBuffer<u8>,
    rows: u32,
    out_s: &mut DeviceBuffer<u32>,
    sums: &[u32],
    lds: &[Ld],
    what: &str,
) -> eyre::Result<bool> {
    for &ld in lds {
        out_s.fill_zero()?;
        b.launch(&rd(other.raw(), rows, ld, out_s))?;
        b.s.synchronize()?;
        if !b.check(out_s, sums, &format!("{what} pass aux {ld:?}"))? {
            eprintln!(
                "  {what} ABORTED: the flush's buffer loads ({ld:?}) returned wrong data -- buffer \
                 descriptor dword3 wrong? set CACHE_PROBE_W3=0x... (current {:#010x}); every row \
                 of this question would be junk",
                b.probe.w3
            );
            return Ok(false);
        }
    }
    Ok(true)
}

/// Valid Q8_0 blocks (f16 scale 1.0 + 32 small int8 quants) for the production GEMV rows.
fn q8_0_bytes(rows: u32) -> Vec<u8> {
    let blocks = rows as usize * (ROW_BYTES as usize / 34);
    let mut v = Vec::with_capacity(blocks * 34);
    for blk in 0..blocks {
        v.extend_from_slice(&[0x00, 0x3C]); // f16 1.0
        for j in 0..32 {
            v.push((((j * 7 + blk) % 15) as i8 - 7) as u8);
        }
    }
    v
}

fn q1(dev: &Device, t_start: Instant, failures: &mut Vec<String>, summary: &mut Vec<String>) -> eyre::Result<()> {
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
    let mut out_s = DeviceBuffer::<u32>::new(id, flush_rows as usize)?;
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
        evs: (0..8).map(|_| Event::new()).collect::<eyre::Result<_>>()?,
        probe,
        q8: Some(q8),
        t_start,
        failures,
    };
    let other_seed = 0x5151;
    b.fill(other.raw(), other.byte_len(), other_seed)?;
    let other_sums = expected_row_sums(flush_rows, other_seed);
    // The per-rep FLUSH: three passes over the 512 MB other buffer. HT evicts M's lines whatever
    // hint they were loaded with; LU (last use) then marks those HT lines evictable, so the RT
    // pass replaces them and leaves the cache in the ordinary RT state. The sanity row in Q1-A
    // compares this against the PRISTINE baseline below.
    const FLUSH_LDS: [Ld; 3] = [Ld::Aux(2), Ld::Aux(3), Ld::Aux(0)];
    const NF: usize = 3; // ops of the flush; timed ops start at st[NF]

    // PRISTINE baseline, FIRST: 44 MB RT cold/warm after an RT-only flush, before any HT / LU /
    // NT_HT load has run on this device (if hinted lines were sticky, a baseline taken later would
    // be polluted the same way as the rows it is meant to check). Only RT loads and the fill's
    // stores run before it.
    if !check_flush(&mut b, &other, flush_rows, &mut out_s, &other_sums, &[Ld::Aux(0)], "Q1")? {
        summary.push(format!("Q1 ABORTED: flush checksums wrong (dword3 {:#010x}; try CACHE_PROBE_W3)", b.probe.w3));
        return Ok(());
    }
    let pristine = {
        let rows = rows_for(44 * MB);
        let m = DeviceBuffer::<u8>::new(id, rows as usize * ROW_BYTES as usize)?;
        let seed = 0x0F44;
        b.fill(m.raw(), m.byte_len(), seed)?;
        for _ in 0..30 {
            b.launch(&rd(other.raw(), flush_rows, Ld::Aux(0), &out_s))?; // RT-only clock ramp
        }
        b.s.synchronize()?;
        let ops = [
            rd(other.raw(), flush_rows, Ld::Aux(0), &out_s),
            rd(m.raw(), rows, Ld::Aux(0), &out_m),
            rd(m.raw(), rows, Ld::Aux(0), &out_m),
        ];
        let r = b.measure(&ops, n, "Q1 pristine")?;
        let ok = b.check(&out_m, &expected_row_sums(rows, seed), "Q1 pristine 44 MB RT")?;
        eprintln!(
            "    PRISTINE 44 MB RT (RT-only flush, before any hinted load): cold {:.1} us, warm {:.1} us, x{:.2}{}",
            r[1].med,
            r[2].med,
            r[1].med / r[2].med,
            if ok { "" } else { "  CHECKSUM BAD" }
        );
        (r[1].med, r[2].med)
    };

    // Now the hinted flush passes themselves.
    if !check_flush(&mut b, &other, flush_rows, &mut out_s, &other_sums, &FLUSH_LDS, "Q1")? {
        summary.push(format!("Q1 ABORTED: flush checksums wrong (dword3 {:#010x}; try CACHE_PROBE_W3)", b.probe.w3));
        return Ok(());
    }
    let out_s = out_s; // read-only from here on
    let flush = || FLUSH_LDS.map(|ld| rd(other.raw(), flush_rows, ld, &out_s));

    // Clock ramp.
    for _ in 0..15 {
        for op in flush() {
            b.launch(&op)?;
        }
    }
    b.s.synchronize()?;
    let fl = b.measure(&flush(), 9, "Q1 flush ref")?;
    eprintln!(
        "    DRAM stream ref (512 MB passes): HT {:.0} GB/s, LU {:.0} GB/s, RT {:.0} GB/s",
        gbs(flush_bytes, fl[0].med),
        gbs(flush_bytes, fl[1].med),
        gbs(flush_bytes, fl[2].med)
    );

    // (M MB, load) -> (cold us, warm us) medians from Q1-A.
    let mut a_of = std::collections::HashMap::new();
    // HT / NT_HT last, so a sticky line they might leave cannot flatter the other rows.
    let all_ld: Vec<Ld> = [Ld::Flat, Ld::Aux(0), Ld::Aux(1), Ld::Aux(3), Ld::Aux(4), Ld::Aux(5), Ld::Aux(2), Ld::Aux(6)].into();
    let few: Vec<Ld> = vec![Ld::Flat, Ld::Aux(0), Ld::Aux(2)];

    // ---- A. cold vs immediate warm, every load path (also validates every aux kernel) ----
    eprintln!(
        "\n  [Q1-A] [flush HT+LU+RT 512 MB] -> M cold -> M warm   (median / min us, GB/s at median; reps {n})\n         \
         16 MB: > L2, < IC (L2 vs IC split); 160 MB: > IC (control)"
    );
    eprintln!("   M MB  load   |   cold us        GB/s |   warm us        GB/s | cold/warm");
    for (mb, lds) in [(16usize, few.clone()), (44, all_ld.clone()), (55, all_ld.clone()), (160, few.clone())] {
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

            // Sanity: the 3-pass flush (after HT / LU / NT_HT loads have run) must give the
            // PRISTINE RT-only numbers measured before any hinted load touched the dGPU.
            if mb == 44 && ld == Ld::Aux(0) {
                let (pc, pw) = pristine;
                let off = ((pc - c.med) / pc).abs().max(((pw - w.med) / pw).abs());
                let verdict = if off <= 0.05 {
                    "match".to_string()
                } else {
                    format!(
                        "MISMATCH ({:.0}% > 5%): hinted lines change the cache state (sticky HT?) -- Q1 rows are suspect",
                        100.0 * off
                    )
                };
                eprintln!(
                    "  sanity 44 MB RT: HT+LU+RT flush cold/warm {:.1}/{:.1} us vs PRISTINE RT-only {pc:.1}/{pw:.1} us -> {verdict}",
                    c.med, w.med
                );
                summary.push(format!(
                    "Q1 flush sanity (44 MB RT cold/warm, 3-pass flush {:.1}/{:.1} vs pristine RT-only {pc:.1}/{pw:.1} us): {verdict}",
                    c.med, w.med
                ));
            }
        }
        drop(m);
    }
    if let (Some(a), Some(c)) = (a_of.get(&(44, Ld::Aux(0))), a_of.get(&(160, Ld::Aux(0)))) {
        summary.push(format!(
            "Q1 RT cold/warm: 44 MB x{:.2}, 160 MB control x{:.2}{}",
            a.0 / a.1,
            c.0 / c.1,
            a_of.get(&(16, Ld::Aux(0))).map(|s| format!(", 16 MB x{:.2}", s.0 / s.1)).unwrap_or_default()
        ));
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
    // Best captured per S (over hinted combos), and the no-hint baseline, for the summary.
    let mut best: std::collections::BTreeMap<usize, (f64, String)> = Default::default();
    let mut base: std::collections::BTreeMap<usize, f64> = Default::default();
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
                let cap = 100.0 * (rt_cold - r.med) / (rt_cold - rt_warm);
                eprintln!(
                    "  {mb:>4} {:<6}{:<6}| {smb:>4} {:<6} | {:>7.1} {:>5.0} | {:>6.1} {:>5.0} | {:>7.1}({:>6.1}) {:>6.0} | x{:.2} | {cap:>5.0}%{}",
                    ld_name(&arch, Ld::Aux(am)),
                    ld_name(&arch, Ld::Aux(ar)),
                    ld_name(&arch, Ld::Aux(as_)),
                    c.med, gbs(bytes, c.med),
                    sv.med, gbs(srows as usize * ROW_BYTES as usize, sv.med),
                    r.med, r.min, gbs(bytes, r.med),
                    rt_cold / r.med,
                    if ok { "" } else { "  CHECKSUM BAD" }
                );
                if ok && (am, ar, as_) == (0, 0, 0) {
                    let e = base.entry(smb).or_insert(f64::NEG_INFINITY);
                    *e = e.max(cap);
                }
                if ok && am != 0 && best.get(&smb).map(|x| cap > x.0).unwrap_or(true) {
                    let pair = format!(
                        "M {mb} MB aM {} aR {} aS {}",
                        ld_name(&arch, Ld::Aux(am)),
                        ld_name(&arch, Ld::Aux(ar)),
                        ld_name(&arch, Ld::Aux(as_))
                    );
                    best.insert(smb, (cap, pair));
                }
            }
        }
        drop(m);
    }
    for smb in [128usize, 200] {
        if let Some((cap, pair)) = best.get(&smb) {
            summary.push(format!(
                "Q1 best captured at S {smb} MB: {cap:.0}% ({pair}); no-hint RT/RT baseline {:.0}%",
                base.get(&smb).copied().unwrap_or(f64::NAN)
            ));
        }
    }

    // ---- C. the production Q8_0 GEMV (no hints) as the consumer ----
    eprintln!("\n  [Q1-C] production q8_0_gemv_warp8 (no hints, valid Q8_0 blocks): [flush] -> GEMV(M) -> [S(aS)] -> GEMV(M)");
    eprintln!("   M MB  S MB aS     |  GEMV cold us  GB/s |  GEMV 2nd us(min)   GB/s | cold/2nd");
    for mb in [44usize, 55] {
        let rows = rows_for(mb * MB);
        let bytes = rows as usize * ROW_BYTES as usize;
        let mut m = DeviceBuffer::<u8>::new(id, bytes)?;
        m.copy_from_host(&q8_0_bytes(rows))?;
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

/// The DRM sysfs device dir (`/sys/class/drm/cardN/device`) of this HIP device, by PCI address.
fn drm_dir(p: &DeviceProperties) -> Option<std::path::PathBuf> {
    for e in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dir = e.path().join("device");
        let Ok(real) = std::fs::canonicalize(&dir) else { continue };
        let addr = real.file_name()?.to_string_lossy().into_owned(); // "0000:c5:00.0"
        let f: Vec<&str> = addr.split([':', '.']).collect();
        if f.len() != 4 {
            continue;
        }
        let hex = |s: &str| i64::from_str_radix(s, 16).ok();
        if hex(f[0]) == Some(p.pci_domain_id as i64)
            && hex(f[1]) == Some(p.pci_bus_id as i64)
            && hex(f[2]) == Some(p.pci_device_id as i64)
        {
            return Some(dir);
        }
    }
    None
}

/// (mem_info_gtt_used, mem_info_vram_used) bytes, when readable.
fn mem_used(dir: Option<&std::path::Path>) -> (Option<i64>, Option<i64>) {
    let rd = |f: &str| -> Option<i64> {
        std::fs::read_to_string(dir?.join(f)).ok()?.trim().parse().ok()
    };
    (rd("mem_info_gtt_used"), rd("mem_info_vram_used"))
}

/// Where an allocation of `bytes` landed, from the sysfs deltas around it.
fn placement(before: (Option<i64>, Option<i64>), after: (Option<i64>, Option<i64>), bytes: usize) -> String {
    let d = |a: Option<i64>, b: Option<i64>| a.zip(b).map(|(a, b)| b - a);
    let (g, v) = (d(before.0, after.0), d(before.1, after.1));
    let mb = |x: Option<i64>| x.map(|x| format!("{:+.1} MB", x as f64 / MB as f64)).unwrap_or_else(|| "n/a".into());
    let big = |x: Option<i64>| x.map(|x| x as f64 >= 0.9 * bytes as f64).unwrap_or(false);
    let verdict = if big(g) && !big(v) {
        "GTT"
    } else if big(v) && !big(g) {
        "VRAM carve-out"
    } else {
        "unclear"
    };
    format!("{verdict} (gtt_used {}, vram_used {})", mb(g), mb(v))
}

fn q2(dev: &Device, t_start: Instant, failures: &mut Vec<String>, summary: &mut Vec<String>) -> eyre::Result<()> {
    dev.set_current()?;
    let arch = header(dev, "Q2 iGPU MALL")?;
    let p = dev.properties()?;
    let id = dev.id;
    let n = reps();
    let probe = load_probe(&arch, p.warp_size)?;
    eprintln!("    buffer descriptor dword3 = {:#010x}", probe.w3);
    eprintln!(
        "    kinds: dev = hipMalloc on the iGPU (the expert pool's path), \
         hostNC = hipHostMalloc(NON_COHERENT) (miss staging), hostC = hipHostMalloc(COHERENT)"
    );
    let drm = drm_dir(&p);
    eprintln!(
        "    sysfs: {} (placement = mem_info_gtt_used / mem_info_vram_used deltas around alloc + fill)",
        drm.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| "NOT FOUND (placement n/a)".into())
    );

    let flush_bytes = 256 * MB;
    let flush_rows = rows_for(flush_bytes);
    let mu0 = mem_used(drm.as_deref());
    let other = DeviceBuffer::<u8>::new(id, flush_rows as usize * ROW_BYTES as usize)?;
    let mut out_s = DeviceBuffer::<u32>::new(id, flush_rows as usize)?;
    let out_m = DeviceBuffer::<u32>::new(id, rows_for(64 * MB) as usize)?;
    let mut b = Bench {
        s: Stream::new(id)?,
        evs: (0..8).map(|_| Event::new()).collect::<eyre::Result<_>>()?,
        probe,
        q8: None,
        t_start,
        failures,
    };
    let other_seed = 0x6161;
    b.fill(other.raw(), other.byte_len(), other_seed)?;
    let other_place = placement(mu0, mem_used(drm.as_deref()), other.byte_len());
    eprintln!("    256 MB dev (hipMalloc) flush buffer lands in: {other_place}");
    let other_sums = expected_row_sums(flush_rows, other_seed);
    if !check_flush(&mut b, &other, flush_rows, &mut out_s, &other_sums, &[Ld::Aux(0)], "Q2")? {
        summary.push(format!("Q2 ABORTED: flush checksums wrong (dword3 {:#010x}; try CACHE_PROBE_W3)", b.probe.w3));
        return Ok(());
    }
    let out_s = out_s;
    let flush = || rd(other.raw(), flush_rows, Ld::Aux(0), &out_s);
    for _ in 0..40 {
        b.launch(&flush())?;
    }
    b.s.synchronize()?;
    let fl = b.measure(&[flush()], 9, "Q2 flush ref")?[0];
    eprintln!(
        "    DRAM stream ref (256 MB dev, dflt): {:.0} us = {:.0} GB/s (min {:.0} us)",
        fl.med,
        gbs(flush_bytes, fl.med),
        fl.min
    );

    let make = |kind: &str, bytes: usize, seed: u32, b: &Bench| -> eyre::Result<(MBuf, String)> {
        let before = mem_used(drm.as_deref());
        let m = match kind {
            "dev" => {
                let d = DeviceBuffer::<u8>::new(id, bytes)?;
                b.fill(d.raw(), bytes, seed)?;
                MBuf::Dev(d)
            }
            "hostNC" | "hostC" => {
                let flags = if kind == "hostNC" { HIP_HOST_MALLOC_NON_COHERENT } else { HIP_HOST_MALLOC_COHERENT };
                let mut h = PinnedBuffer::<u8>::new_with_flags(bytes, flags)?;
                host_fill(&mut h, seed);
                MBuf::Host(h)
            }
            _ => unreachable!(),
        };
        Ok((m, placement(before, mem_used(drm.as_deref()), bytes)))
    };
    let kinds = ["dev", "hostNC", "hostC"];

    // ---- A. cold vs immediate warm; aux 4 (dlc) on M itself is the negative control ----
    eprintln!("\n  [Q2-A] [flush 256 MB] -> M cold -> M warm   (median / min us, GB/s at median; reps {n})");
    eprintln!("   kind    M MB  load     |   cold us        GB/s |   warm us        GB/s | cold/warm");
    let mut a_of = std::collections::HashMap::new();
    let mut dev_place = String::from("n/a");
    for kind in kinds {
        for mb in [16usize, 24, 64] {
            let rows = rows_for(mb * MB);
            let bytes = rows as usize * ROW_BYTES as usize;
            let seed = 0x4000 + mb as u32;
            let (m, place) = make(kind, bytes, seed, &b)?;
            eprintln!("   {kind:<7}{mb:>4}  placement: {place}");
            if kind == "dev" && mb == 24 {
                dev_place = place;
            }
            let sums = expected_row_sums(rows, seed);
            for ld in [Ld::Flat, Ld::Aux(0), Ld::Aux(4), Ld::Aux(6)] {
                if b.over_soft_limit() {
                    eprintln!("  SOFT TIME LIMIT: skipping the rest of Q2");
                    return Ok(());
                }
                let mr = || rd(m.ptr(), rows, ld, &out_m);
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
                a_of.insert((kind, mb, ld), (st[1].med, st[2].med));
            }
        }
    }
    let ratio = |kind: &str, mb: usize, ld: Ld| {
        a_of.get(&(kind, mb, ld)).map(|(c, w)| format!("x{:.2}", c / w)).unwrap_or_else(|| "n/a".into())
    };
    summary.push(format!(
        "Q2 dev (hipMalloc) lands in: {dev_place}; 256 MB flush buffer: {other_place}"
    ));
    for kind in kinds {
        summary.push(format!(
            "Q2 {kind:<6} cold/warm dflt: 16 MB {}, 24 MB {}, 64 MB control {} | dlc on M: 16 MB {}",
            ratio(kind, 16, Ld::Aux(0)),
            ratio(kind, 24, Ld::Aux(0)),
            ratio(kind, 64, Ld::Aux(0)),
            ratio(kind, 16, Ld::Aux(4))
        ));
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
            let (m, _) = make(kind, bytes, seed, &b)?;
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
                        rd(m.ptr(), rows, Ld::Aux(0), &out_m),
                        rd(other.raw(), srows, Ld::Aux(as_), &out_s),
                        rd(m.ptr(), rows, Ld::Aux(0), &out_m),
                    ];
                    let st = b.measure(&ops, n, "Q2-B")?;
                    let ok = b.check(&out_m, &sums, &format!("Q2-B {kind} {mb} MB"))?
                        & b.check(&out_s, &other_sums[..srows as usize], &format!("Q2-B S {smb} aS {as_}"))?;
                    let cap = a_of
                        .get(&(kind, mb, Ld::Aux(0)))
                        .map(|(_, w)| format!("{:>5.0}%", 100.0 * (st[1].med - st[3].med) / (st[1].med - w)))
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
    let mut summary = Vec::new();
    if only("q1") {
        match pick("gfx1201")? {
            Some(d) => q1(&d, t_start, &mut failures, &mut summary)?,
            None => summary.push("Q1 SKIPPED: no gfx1201 visible (HIP_VISIBLE_DEVICES?)".into()),
        }
    }
    if only("q2") {
        match pick("gfx1151")? {
            Some(d) => q2(&d, t_start, &mut failures, &mut summary)?,
            None => summary.push("Q2 SKIPPED: no gfx1151 visible (HIP_VISIBLE_DEVICES?)".into()),
        }
    }
    eprintln!("\n=== SUMMARY ({:.1} s, {} checksum failures)", t_start.elapsed().as_secs_f64(), failures.len());
    for l in &summary {
        eprintln!("  {l}");
    }
    if !failures.is_empty() {
        return Err(eyre!("{} checksum failures:\n  {}", failures.len(), failures.join("\n  ")));
    }
    Ok(())
}
