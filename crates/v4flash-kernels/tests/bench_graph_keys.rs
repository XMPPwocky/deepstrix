//! Step 0 of docs/v41/GRAPH_KEYS_DESIGN.md (2.0): the measurements that gate building
//! (stage, rows)-keyed arena graphs. dGPU only; the hub must be DOWN (it holds the dGPU).
//!
//!  1. The context write, AMORTIZED (queue pre-filled behind a spin, N small enough for the AQL
//!     ring): N x [nop; write; nop] for write = none / H2D from pinned / `arena_ctx_store` by
//!     value / `hipStreamWriteValue32`; host enqueue time (mean, max) and GPU time per write.
//!  2. 80 `hipGraphLaunch` round-robin over K executables (K = 1 back to back, 2, 4 as
//!     diagnostics; K = 8, production's pattern, vs 80 all distinct as 11 alternating pairs), 1-
//!     and 8-node graphs, behind a 50 ms spin. BLOCKS = the host enqueue time grows with the spin
//!     (5 vs 50 ms).
//!  3. COHERENCE: `arena_ctx_store` then a captured `_ind` graph reading the slot (operands on
//!     every 64-B line of the entry), alternating entries, 10k rounds; every output and every
//!     canary record (the seq and the operand pointers each launch resolved) checked.
//!  4. `_ind` twins vs direct, bit-exact and timed with the canary compiled in (null):
//!     `q8_0_gemv_bpack_tB{1,4,8}` on the q_b shape and `mhc_fast_batched` (13 operands,
//!     pre_attn case) at b = 1, 4, 8, 11 alternating pairs queued behind a spin. (VGPR / SGPR counts come from the code objects, offline:
//!     ~/scratch-ms/graph_keys_regs.py.)
//!  5. The mechanism: ONE captured `_ind` graph replayed after storing entry A, then B,
//!     reproduces the direct kernel on A, then on B; and a capture of N launches has N nodes
//!     (design 2.5's vetted-count check assumes one node per launch).
//!  The go / no-go: the exactness items, no drained queue (a drained run is re-run up to 3 times),
//!  no BLOCKS, and the per-step budget at each operating point (b = 1 / 4 / 8 per lane, two lanes,
//!  ms.step 60 / 121.6 / 196.8 ms, GK_STEP_MS_B{1,4,8}) -- writes x write cost + graph launches x
//!  the K = 8 relaunch delta (1- and 8-node) + twin launches x the twin deltas at that b, medians
//!  and ~96.7% upper bounds, vs 1% of that step: GO / MARGINAL (the live A/B decides) / NO-GO.
//!
//! HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release \
//!   --features v41 -p v4flash-kernels --test bench_graph_keys -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use std::time::Instant;
use v4flash_hip::{sys, Device, DeviceBuffer, Event, GraphExec, PinnedBuffer, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_EMBD, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
use v4flash_kernels::het::arena_ctx::{ArenaCtx, ArenaCtxKernels, CanaryLog, Ind, ARENA_CTX_WORDS};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::{FastCollapse, FastMix, MIX_PRE_SCALED};
use v4flash_kernels::MhcArena;

fn pick_dgpu() -> eyre::Result<(Device, String)> {
    for id in 0..4 {
        let d = Device::new(id);
        if d.set_current().is_err() {
            continue;
        }
        let Ok(p) = d.properties() else { continue };
        if p.gcn_arch_name.starts_with("gfx120") {
            return Ok((d, p.gcn_arch_name));
        }
    }
    Err(eyre!("no gfx120x device"))
}

fn lcg_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 56) as u8
        })
        .collect()
}

fn lcg_f32(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    lcg_bytes(seed, 2 * n).chunks(2).map(|c| lo + (hi - lo) * (c[0] as f32 * 256.0 + c[1] as f32) / 65536.0).collect()
}

fn up<T: Copy>(id: i32, h: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}

fn down(d: &DeviceBuffer<f32>, n: usize) -> eyre::Result<Vec<u32>> {
    let mut h = vec![0f32; n];
    d.slice_view(0, n).copy_to_host(&mut h)?;
    Ok(h.iter().map(|v| v.to_bits()).collect())
}

/// A Q8_0 weight of `rows` x `k` with sane f16 scales (bytes otherwise random).
fn q8_weight(dev: i32, rows: usize, k: usize, seed: u64) -> eyre::Result<DeviceBuffer<u8>> {
    let blocks = k / 32;
    let mut host = lcg_bytes(seed, rows * blocks * 34);
    for r in 0..rows {
        let base = r * blocks * 34;
        for b in 0..blocks {
            // f16 scale in [~0.001, ~0.03]: exponent 0x14..0x1c, random mantissa.
            let s = 0x1400u16 | ((host[base + 2 * b] as u16 & 0x07) << 8) | host[base + 2 * b + 1] as u16;
            host[base + 2 * b..base + 2 * b + 2].copy_from_slice(&s.to_le_bytes());
        }
    }
    up(dev, &host)
}

fn capture(s: &Stream, body: &mut dyn FnMut(&Stream) -> eyre::Result<()>) -> eyre::Result<(usize, GraphExec)> {
    s.begin_capture(sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let r = body(s);
    let g = s.end_capture()?;
    r?;
    Ok((g.nodes()?.len(), g.instantiate()?))
}

/// us per launch over the rounds of one arm.
#[derive(Clone, Copy, Debug)]
struct Stat {
    med: f64,
    min: f64,
    max: f64,
}

impl Stat {
    fn of(v: &[f64]) -> Self {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        Stat { med: v[v.len() / 2], min: v[0], max: v[v.len() - 1] }
    }
}

impl std::fmt::Display for Stat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A spread over 8% between runs flags clock-state changes (run 2: ~64 / ~74 us).
        let flag = if self.max > 1.08 * self.min { " SPREAD" } else { "" };
        write!(f, "{:7.2} us [{:.2}..{:.2}]{flag}", self.med, self.min, self.max)
    }
}

/// Paired differences b_i - a_i (us) of alternating runs: the median (charged when positive,
/// resolved or not), a ~96.7% one-sided upper bound on it (the k-th smallest difference, k the
/// smallest with P(Bin(n, 1/2) <= k - 1) >= 0.967: the 9th of 11, the 16th of 21), the range, and
/// the sign count as a label.
#[derive(Clone, Copy, Debug)]
struct Delta {
    med: f64,
    ub: f64,
    min: f64,
    max: f64,
    pos: usize,
    n: usize,
}

impl Delta {
    fn of(a: &[f64], b: &[f64]) -> Self {
        let mut d: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
        d.sort_by(f64::total_cmp);
        let n = d.len();
        // Binomial(n, 1/2) CDF, smallest k with CDF(k - 1) >= 0.967.
        let mut cdf = 0.0;
        let mut c = 1.0f64; // C(n, i)
        let mut k = n;
        for i in 0..n {
            cdf += c / 2f64.powi(n as i32);
            if cdf >= 0.967 {
                k = i + 1;
                break;
            }
            c = c * (n - i) as f64 / (i + 1) as f64;
        }
        Delta {
            med: d[n / 2],
            ub: d[k.min(n) - 1],
            min: d[0],
            max: d[n - 1],
            pos: d.iter().filter(|v| **v > 0.0).count(),
            n,
        }
    }

    /// The budget's charge: the median, never negative.
    fn charge(&self) -> f64 {
        self.med.max(0.0)
    }

    /// The upper-bound charge.
    fn charge_ub(&self) -> f64 {
        self.ub.max(0.0)
    }
}

impl std::fmt::Display for Delta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:+.2} us (ub {:+.2}) [{:+.2}..{:+.2}], {}/{} pairs > 0",
            self.med, self.ub, self.min, self.max, self.pos, self.n
        )
    }
}

/// Pairs of runs of `reps` launches of `a` and of `b` (the order alternating pair to pair). Each
/// run is queued behind `prefill` (a short spin covering the enqueue) and `warm` launches of `a`
/// (real work: the clocks are up and the caches warm alike for both arms when the events start),
/// so the events time the GPU back to back, not the host. A pair with a run whose queue drained
/// before the host finished enqueueing (an OS stall) is discarded and re-run, up to 3 times:
/// (a, b, b - a paired, pairs still drained after the retries, retries).
fn time_ab(
    s: &Stream,
    reps: usize,
    pairs: usize,
    prefill: &dyn Fn(&Stream) -> eyre::Result<()>,
    a: &mut dyn FnMut() -> eyre::Result<()>,
    b: &mut dyn FnMut() -> eyre::Result<()>,
) -> eyre::Result<(Stat, Stat, Delta, usize, usize)> {
    let warm = (reps / 5).max(1);
    a()?;
    b()?;
    // (us per launch, drained) of one run of `b` (else `a`)
    let mut once = |use_b: bool| -> eyre::Result<(f64, bool)> {
        s.synchronize()?;
        let (e0, e1) = (Event::new()?, Event::new()?);
        prefill(s)?;
        for _ in 0..warm {
            a()?;
        }
        e0.record(s)?;
        for _ in 0..reps {
            if use_b {
                b()?
            } else {
                a()?
            }
        }
        let drained = e0.query()?;
        e1.record(s)?;
        e1.synchronize()?;
        Ok((Event::elapsed_ms(&e0, &e1)? as f64 * 1e3 / reps as f64, drained))
    };
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    let (mut drained, mut retries) = (0usize, 0usize);
    for i in 0..pairs {
        for attempt in 0..4 {
            let ((xa, da), (xb, db)) = if i % 2 == 0 {
                let ra = once(false)?;
                (ra, once(true)?)
            } else {
                let rb = once(true)?;
                (once(false)?, rb)
            };
            if !(da || db) || attempt == 3 {
                drained += usize::from(da || db);
                ta.push(xa);
                tb.push(xb);
                break;
            }
            retries += 1;
        }
    }
    Ok((Stat::of(&ta), Stat::of(&tb), Delta::of(&ta, &tb), drained, retries))
}

/// One mhc_fast output set (the pre_attn case writes all of it).
struct MhcSet {
    split: DeviceBuffer<f32>,
    mix: DeviceBuffer<f32>,
    cnt: DeviceBuffer<u32>,
    inv: DeviceBuffer<f32>,
    carry: DeviceBuffer<f32>,
    cur: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
}

impl MhcSet {
    fn new(id: i32, bmax: usize, carry0: &[f32]) -> eyre::Result<Self> {
        let (m, ne) = (HC_MIX_DIM as usize, N_EMBD as usize);
        let z = |n: usize| -> eyre::Result<DeviceBuffer<f32>> {
            let mut d = DeviceBuffer::new(id, n)?;
            d.fill_zero()?;
            Ok(d)
        };
        let mut cnt = DeviceBuffer::<u32>::new(id, bmax)?;
        cnt.fill_zero()?;
        Ok(Self {
            split: z(bmax * m)?,
            mix: z(bmax * m)?,
            cnt,
            inv: z(bmax)?,
            carry: up(id, carry0)?,
            cur: z(bmax * ne)?,
            norm: z(bmax * ne)?,
        })
    }

    fn reset(&mut self, carry0: &[f32]) -> eyre::Result<()> {
        for d in [&mut self.split, &mut self.mix, &mut self.inv, &mut self.cur, &mut self.norm] {
            d.fill_zero()?;
        }
        self.carry.copy_from_host(carry0)
    }

    fn dump(&self, b: usize) -> eyre::Result<Vec<Vec<u32>>> {
        let (m, ne) = (HC_MIX_DIM as usize, N_EMBD as usize);
        Ok(vec![
            down(&self.split, b * m)?,
            down(&self.mix, b * m)?,
            down(&self.inv, b)?,
            down(&self.carry, b * m)?,
            down(&self.cur, b * ne)?,
            down(&self.norm, b * ne)?,
        ])
    }
}

#[test]
#[ignore]
fn graph_keys_step0() -> eyre::Result<()> {
    color_eyre::install().ok();
    let (dev, arch) = pick_dgpu()?;
    dev.set_current()?;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let ctxk = ArenaCtxKernels::for_arch(&arch)?;
    let arena = MhcArena::for_arch(&arch)?;
    let s = Stream::new(dev.id)?;
    let mut sink = DeviceBuffer::<u32>::new(dev.id, 64)?;
    let mut slot = DeviceBuffer::<u64>::new(dev.id, ARENA_CTX_WORDS)?;
    let mut pinned = PinnedBuffer::<u64>::new(ARENA_CTX_WORDS)?;
    let mut entry = ArenaCtx::default();
    for (i, p) in entry.p.iter_mut().enumerate() {
        *p = 0x1000 + i as u64;
    }
    // Every 2.0 bar, by item: bit-exactness / coherence / node count (`exact`), no drained queue,
    // no BLOCKS, and ONE per-step budget (end of the run). The last line says STEP0: GO or
    // STEP0: NO-GO (items).
    let mut exact = true;
    let mut nogo: Vec<String> = Vec::new();
    let mut bar = |ok: bool, item: String| {
        if !ok {
            nogo.push(item);
        }
    };

    // The spin's clock (wall_clock64, constant rate): ticks per ms.
    let tpm = {
        e.q8.slack_probe_spin(&s, 1000)?;
        s.synchronize()?;
        let (e0, e1) = (Event::new()?, Event::new()?);
        e0.record(&s)?;
        e.q8.slack_probe_spin(&s, 2_000_000)?;
        e1.record(&s)?;
        e1.synchronize()?;
        2_000_000.0 / Event::elapsed_ms(&e0, &e1)? as f64
    };
    let spin_ms = |ms: f64| (ms * tpm) as u64;
    println!("wall_clock64: {:.0} ticks/ms", tpm);

    // ---- 1. the context write, amortized behind a spin ---------------------------------
    // N x 3 packets stays well inside the AQL ring (ROC_AQL_QUEUE_SIZE, 4096 by default) and the
    // kernarg pool: a full queue would block the host until the spin ends, timing the host.
    // The spins only cover the enqueue (drained = fail): a long idle spin lets the dGPU clock down
    // before the timed work (run 2: per-launch times bimodal at ~64 / ~74 us behind 50 ms spins).
    println!("== 1. context write between two launches, queue pre-filled behind a 5 ms spin (per write, median of 3; 'none' = bare launches)");
    let n = 400usize;
    let variants = ["none", "h2d", "ctx_store", "write_value32"];
    let mut host_v = vec![Vec::new(); variants.len()];
    let mut gpu_v = vec![Vec::new(); variants.len()];
    let mut iter_max = vec![0f64; variants.len()];
    let mut retries1 = 0usize;
    for _ in 0..3 {
        for (vi, variant) in variants.iter().enumerate() {
            for attempt in 0..4 {
                s.synchronize()?;
                let (e0, e1) = (Event::new()?, Event::new()?);
                e.q8.slack_probe_spin(&s, spin_ms(5.0))?;
                e0.record(&s)?;
                let t = Instant::now();
                for i in 0..n {
                    let ti = Instant::now();
                    ctxk.nop(&s, &mut sink)?;
                    match *variant {
                        "h2d" => {
                            pinned.as_mut_slice()[ARENA_CTX_WORDS - 2] = i as u64;
                            slot.copy_from_host_async(pinned.as_slice(), &s)?;
                        }
                        "ctx_store" => {
                            entry.seq = i as u64;
                            ctxk.store(&s, &entry, &mut slot)?;
                        }
                        "write_value32" => unsafe { s.write_value32(slot.raw() as *mut u32, i as u32)? },
                        _ => {}
                    }
                    ctxk.nop(&s, &mut sink)?;
                    iter_max[vi] = iter_max[vi].max(ti.elapsed().as_secs_f64() * 1e6);
                }
                let host_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
                // The spin still running when the host finished = the GPU ran the loop back to back;
                // a drained run (an OS stall) is re-run, up to 3 times.
                let drained = e0.query()?;
                e1.record(&s)?;
                e1.synchronize()?;
                if drained && attempt < 3 {
                    retries1 += 1;
                    continue;
                }
                bar(!drained, format!("1:{variant}:drained"));
                host_v[vi].push(host_us);
                gpu_v[vi].push(Event::elapsed_ms(&e0, &e1)? as f64 * 1e3 / n as f64);
                break;
            }
        }
    }
    if retries1 > 0 {
        println!("  ({retries1} drained runs re-run)");
    }
    let med = |v: &Vec<f64>| Stat::of(v).med;
    let (base_h, base_g) = (med(&host_v[0]), med(&gpu_v[0]));
    let mut write_cost = (0.0, 0.0); // ctx_store: (GPU, host) us per write
    for (vi, variant) in variants.iter().enumerate() {
        let (h, g) = (med(&host_v[vi]), med(&gpu_v[vi]));
        if *variant == "ctx_store" {
            write_cost = (g - base_g, h - base_h);
        }
        println!(
            "  {variant:>14}: host {h:7.2} us/iter (+{:6.2}, max {:.1}), GPU {:7.2} us/iter [{:.2}..{:.2}] (+{:6.2})",
            h - base_h,
            iter_max[vi],
            g,
            Stat::of(&gpu_v[vi]).min,
            Stat::of(&gpu_v[vi]).max,
            g - base_g,
        );
    }

    // ---- 2. re-launching executables that are still in flight ----------------------------
    // Step 0 run 1 (2026-10-04): 80 BACK-TO-BACK launches of one 8-node executable cost +7..13% GPU
    // time per launch vs 80 distinct executables. Production replays a stage graph only after the
    // lane's other 7 stages (plus direct launches), so its pattern is round-robin over K >= 8: the
    // budget charges K = 8 vs all-distinct (today's legacy graphs), measured as alternating pairs.
    // K = 1 / 2 / 4 are diagnostics.
    println!("== 2. 80 launches round-robin over K executables behind a 1 ms spin + 10 warm-up graphs (GPU us/launch)");
    let (rows, k) = (1280usize, 5120usize);
    let w = q8_weight(dev.id, rows, k, 1)?;
    let mut xq = up(dev.id, &lcg_bytes(2, 8 * k).iter().map(|&b| b as i8).collect::<Vec<_>>())?;
    let mut xs = up(dev.id, &vec![0.01f32; 8 * (k / 32)])?;
    let mut out = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
    // (host us per launch, GPU us per launch, queue drained before the host finished). `execs`
    // holds 90: the timed launches use [0, kk), the warm-up (real work, so the clocks are up when
    // the events start) [80, 90).
    let run = |execs: &[GraphExec], kk: usize, spin: f64| -> eyre::Result<(f64, f64, bool)> {
        s.synchronize()?;
        let (e0, e1) = (Event::new()?, Event::new()?);
        e.q8.slack_probe_spin(&s, spin_ms(spin))?;
        for x in &execs[80..90] {
            x.launch(&s)?;
        }
        e0.record(&s)?;
        let t = Instant::now();
        for i in 0..80 {
            execs[i % kk].launch(&s)?;
        }
        let host_us = t.elapsed().as_secs_f64() * 1e6 / 80.0;
        let drained = e0.query()?;
        e1.record(&s)?;
        e1.synchronize()?;
        Ok((host_us, Event::elapsed_ms(&e0, &e1)? as f64 * 1e3 / 80.0, drained))
    };
    let mut relaunch = Vec::new(); // (nodes, GPU delta, host delta) per launch, K = 8 vs K = 80
    for nodes in [1u32, 8] {
        let mut body = |s: &Stream| -> eyre::Result<()> {
            for b in 1..=nodes {
                e.q8.matvec_bpack(s, &mut out, &w, &xq, &xs, rows as u32, k as u32, b)?;
            }
            Ok(())
        };
        let execs: Vec<GraphExec> =
            (0..90).map(|_| capture(&s, &mut body).map(|c| c.1)).collect::<eyre::Result<_>>()?;
        let mut any_drained = false;
        let mut line = format!("  {nodes} node(s), diagnostics (median of 3):");
        for kk in [1usize, 2, 4] {
            let mut g = Vec::new();
            for _ in 0..3 {
                let (_, gpu, _) = run(&execs, kk, 1.0)?; // diagnostics: no bar
                g.push(gpu);
            }
            line += &format!("  K={kk} {}", Stat::of(&g));
        }
        println!("{line}");
        // K = 8 vs K = 80, 21 alternating pairs.
        let (mut g8, mut g80, mut h8, mut h80) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut retries2 = 0usize;
        for i in 0..21 {
            // A pair with a drained run (an OS stall) is re-run, up to 3 times.
            for attempt in 0..4 {
                let order = if i % 2 == 0 { [8usize, 80] } else { [80, 8] };
                let r0 = run(&execs, order[0], 1.0)?;
                let r1 = run(&execs, order[1], 1.0)?;
                if (r0.2 || r1.2) && attempt < 3 {
                    retries2 += 1;
                    continue;
                }
                any_drained |= r0.2 || r1.2;
                let (r8, r80) = if order[0] == 8 { (r0, r1) } else { (r1, r0) };
                g8.push(r8.1);
                h8.push(r8.0);
                g80.push(r80.1);
                h80.push(r80.0);
                break;
            }
        }
        if retries2 > 0 {
            println!("  ({retries2} drained pairs re-run)");
        }
        let (dg, dh) = (Delta::of(&g80, &g8), Delta::of(&h80, &h8));
        println!("  {nodes} node(s): K=8 {} vs K=80 {}; GPU K=8 - K=80 {dg}", Stat::of(&g8), Stat::of(&g80));
        println!(
            "  {nodes} node(s): host per launch K=8 {} vs K=80 {}; host K=8 - K=80 {dh}",
            Stat::of(&h8),
            Stat::of(&h80)
        );
        if any_drained {
            println!("  [a run DRAINED: host-bound]");
        }
        bar(!any_drained, format!("2:{nodes}n:drained"));
        relaunch.push((nodes, dg, dh));
        if nodes == 8 {
            // BLOCKS: the host enqueue time of 80 launches must not grow with the spin ahead.
            for (label, kk) in [("K=1", 1usize), ("K=80", 80)] {
                let (h5, _, _) = run(&execs, kk, 5.0)?;
                let (h50, _, _) = run(&execs, kk, 50.0)?;
                let blocks = (h50 - h5) * 80.0 / 1e3 > 22.5; // half the 45 ms the spin grew by
                bar(!blocks, format!("2:{label}:BLOCKS"));
                println!(
                    "  {label}: host {:.3} ms behind 5 ms, {:.3} ms behind 50 ms -> {}",
                    h5 * 80.0 / 1e3,
                    h50 * 80.0 / 1e3,
                    if blocks { "BLOCKS" } else { "does not block" }
                );
            }
        }
    }

    // ---- 3. coherence: store then a graph that reads the slot, 10k rounds ----------------
    println!("== 3. coherence: arena_ctx_store then a captured _ind gemv (canary on), alternating entries");
    {
        let (rows_c, k_c, rounds) = (64usize, 256usize, 10_000usize);
        let w0 = q8_weight(dev.id, rows_c, k_c, 7)?;
        let w1 = q8_weight(dev.id, rows_c, k_c, 8)?;
        let xq_c = up(dev.id, &lcg_bytes(9, k_c).iter().map(|&b| b as i8).collect::<Vec<_>>())?;
        let xs_c = up(dev.id, &vec![0.01f32; k_c / 32])?;
        let mut tmp = DeviceBuffer::<f32>::new(dev.id, rows_c)?;
        let mut refs = Vec::new();
        for wj in [&w0, &w1] {
            e.q8.matvec_bpack(&s, &mut tmp, wj, &xq_c, &xs_c, rows_c as u32, k_c as u32, 1)?;
            s.synchronize()?;
            refs.push(down(&tmp, rows_c)?);
        }
        let outs = DeviceBuffer::<f32>::new(dev.id, rounds * rows_c)?;
        let log = CanaryLog::new(dev.id, rounds as u32)?;
        // Slots 0 / 9 / 18 / 27 + seq / log: every 64-B line of the entry.
        const SL: [usize; 4] = [0, 9, 18, 27];
        let ind =
            Ind::new(slot.raw() as u64).with(0, SL[0]).with(1, SL[1]).with(2, SL[2]).with(3, SL[3]).with_canary(7);
        let (_, exec) = capture(&s, &mut |s| {
            e.q8.matvec_bpack_ind(s, ind, &mut tmp, &w0, &xq_c, &xs_c, rows_c as u32, k_c as u32, 1)
        })?;
        e.q8.slack_probe_spin(&s, spin_ms(20.0))?;
        for i in 0..rounds {
            let mut ent = ArenaCtx::default();
            ent.p[SL[0]] = outs.raw() as u64 + (i * rows_c * 4) as u64;
            ent.p[SL[1]] = [&w0, &w1][i % 2].raw() as u64;
            ent.p[SL[2]] = xq_c.raw() as u64;
            ent.p[SL[3]] = xs_c.raw() as u64;
            ent.seq = i as u64;
            ent.log = log.buf.raw() as u64;
            ctxk.store(&s, &ent, &mut slot)?;
            exec.launch(&s)?;
        }
        s.synchronize()?;
        let host = down(&outs, rounds * rows_c)?;
        let bad = (0..rounds).filter(|&i| host[i * rows_c..(i + 1) * rows_c] != refs[i % 2][..]).count();
        let (cursor, recs) = log.read()?;
        let bad_seq = recs.iter().enumerate().filter(|&(i, r)| r.seq != i as u64 || r.tag != 7).count();
        // What each launch resolved: out (round i's row block), w (alternating), xq, xs.
        let want_xor = |i: usize| {
            (outs.raw() as u64 + (i * rows_c * 4) as u64)
                ^ [&w0, &w1][i % 2].raw() as u64
                ^ xq_c.raw() as u64
                ^ xs_c.raw() as u64
        };
        let bad_xor = recs.iter().enumerate().filter(|&(i, r)| r.ptr_xor != want_xor(i)).count();
        let ok = bad == 0 && cursor as usize == rounds && bad_seq == 0 && bad_xor == 0;
        exact &= ok;
        bar(ok, "3:coherence".into());
        println!(
            "  {rounds} rounds: {bad} outputs from a stale / wrong entry; canary {cursor} records, {bad_seq} with a wrong seq, {bad_xor} with wrong resolved pointers"
        );
    }

    // ---- 4a. gemv `_ind` twin vs direct, b = 1, 4, 8 -------------------------------------
    println!("== 4a. q8_0_gemv_bpack_tB{{b}}_ind vs direct (q_b shape 32768 x 1280), canary compiled in, null");
    let (rows, k) = (32768usize, 1280usize);
    let w = q8_weight(dev.id, rows, k, 3)?;
    xq = up(dev.id, &lcg_bytes(4, 8 * k).iter().map(|&b| b as i8).collect::<Vec<_>>())?;
    xs = up(dev.id, &(0..8 * (k / 32)).map(|i| 0.002 + (i % 7) as f32 * 0.001).collect::<Vec<_>>())?;
    let mut out_d = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
    let mut out_i = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
    let prefill = |s: &Stream| e.q8.slack_probe_spin(s, spin_ms(1.5));
    let mut gemv_d: Vec<Delta> = Vec::new(); // per b: _ind - direct, us per launch
    for b in [1u32, 4, 8] {
        out_i.fill_zero()?;
        let mut ent = ArenaCtx::default();
        ent.p[0] = out_i.raw() as u64;
        ent.p[1] = w.raw() as u64;
        ent.p[2] = xq.raw() as u64;
        ent.p[3] = xs.raw() as u64;
        ctxk.store(&s, &ent, &mut slot)?;
        let ind = Ind::new(slot.raw() as u64).with(0, 0).with(1, 1).with(2, 2).with(3, 3);
        let mut out_x = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
        // The twin's real operands come from the slot (out_i); out_x is only size-checked.
        let (sd_, si_, dt, drained, retries) = time_ab(
            &s,
            200,
            21,
            &prefill,
            &mut || e.q8.matvec_bpack(&s, &mut out_d, &w, &xq, &xs, rows as u32, k as u32, b),
            &mut || e.q8.matvec_bpack_ind(&s, ind, &mut out_x, &w, &xq, &xs, rows as u32, k as u32, b),
        )?;
        bar(drained == 0, format!("4a:b{b}:drained"));
        s.synchronize()?;
        let n = b as usize * rows;
        let diff = down(&out_d, n)?.iter().zip(down(&out_i, n)?).filter(|(a, c)| **a != *c).count();
        exact &= diff == 0;
        bar(diff == 0, format!("4a:b{b}:bitexact"));
        gemv_d.push(dt);
        println!(
            "  b={b}: direct {sd_}, _ind {si_} ({:+.1}%); _ind - direct {dt}; differing outputs {diff}{}{}",
            (si_.med / sd_.med - 1.0) * 100.0,
            if retries > 0 { format!("; {retries} pairs re-run") } else { String::new() },
            if drained > 0 { format!("; {drained} pairs still DRAINED, FAILS") } else { String::new() }
        );
    }

    // ---- 4b. mhc_fast_batched `_ind` twin vs direct, b = 1, 4, 8 -------------------------
    println!("== 4b. mhc_fast_batched_ind vs direct (pre_attn: mix pre-scaled + collapse + write_carry), 13 operands indirect");
    let mut mhc_d: Vec<Delta> = Vec::new();
    {
        let (hcd, m, ne, bmax) = (HC_DIM as usize, HC_MIX_DIM as usize, N_EMBD as usize, 8usize);
        // f16 weights in a sane range: sign, exponent 10..14, random mantissa.
        let w_bits: Vec<u8> = lcg_bytes(11, 2 * m * hcd)
            .chunks(2)
            .flat_map(|c| {
                ((((c[0] >> 7) as u16) << 15)
                    | ((10 + (c[0] as u16 % 5)) << 10)
                    | (((c[0] as u16) << 8 | c[1] as u16) & 0x3ff))
                    .to_le_bytes()
            })
            .collect();
        let mw = up(dev.id, &w_bits)?;
        let mx = up(dev.id, &lcg_f32(12, bmax * hcd, -2.0, 2.0))?;
        let msc = up(dev.id, &[0.8f32, 1.3, 0.6])?;
        let mbase = up(dev.id, &lcg_f32(13, m, -1.0, 1.0))?;
        let mnw = up(dev.id, &lcg_f32(14, ne, 0.5, 1.5))?;
        let carry0: Vec<f32> = lcg_f32(15, bmax * m, 0.0, 1.0)
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % m < 4 { 0.1 + 0.5 * v } else { v - 0.5 })
            .collect();
        let mut sd = MhcSet::new(dev.id, bmax, &carry0)?;
        let mut si = MhcSet::new(dev.id, bmax, &carry0)?;
        let mut ent = ArenaCtx::default();
        let ptrs = [
            si.split.raw() as u64,
            si.mix.raw() as u64,
            si.cnt.raw() as u64,
            si.inv.raw() as u64,
            mw.raw() as u64,
            mx.raw() as u64,
            msc.raw() as u64,
            mbase.raw() as u64,
            si.carry.raw() as u64,
            mx.raw() as u64,
            si.cur.raw() as u64,
            si.norm.raw() as u64,
            mnw.raw() as u64,
        ];
        ent.p[..13].copy_from_slice(&ptrs);
        ctxk.store(&s, &ent, &mut slot)?;
        let ind = (0..13).fold(Ind::new(slot.raw() as u64), |ind, i| ind.with(i, i));
        // `set` = the buffers the call is handed: the direct kernel writes them, the twin only
        // size-checks them (its operands come from the slot, pointing at `si`).
        let go = |set: &mut MhcSet, ind: Option<Ind>, b: u32| -> eyre::Result<()> {
            let mix = FastMix {
                weight: &mw,
                x: &mx,
                scale: &msc,
                base: &mbase,
                mode: MIX_PRE_SCALED,
                split_out: &mut set.split,
                mix_out: &mut set.mix,
                counters: &mut set.cnt,
                inv_rows: &mut set.inv,
            };
            let col = FastCollapse { x: &mx, cur_out: &mut set.cur, norm_out: &mut set.norm, norm_w: &mnw };
            match ind {
                None => arena.launch_fast(
                    &s,
                    Some(mix),
                    Some(col),
                    &mut set.carry,
                    true,
                    HC_DIM,
                    RMS_EPS,
                    SINKHORN_ITERS,
                    SINKHORN_EPS,
                    b,
                ),
                Some(ind) => arena.launch_fast_ind(
                    &s,
                    ind,
                    Some(mix),
                    Some(col),
                    &mut set.carry,
                    true,
                    HC_DIM,
                    RMS_EPS,
                    SINKHORN_ITERS,
                    SINKHORN_EPS,
                    b,
                ),
            }
        };
        for b in [1u32, 4, 8] {
            sd.reset(&carry0)?;
            si.reset(&carry0)?;
            go(&mut sd, None, b)?;
            s.synchronize()?;
            let ref_out = sd.dump(b as usize)?;
            // The twin is handed `sd` (a decoy): a missed dereference would leave `si` zero.
            go(&mut sd, Some(ind), b)?;
            s.synchronize()?;
            let got = si.dump(b as usize)?;
            let diff: usize =
                ref_out.iter().zip(&got).map(|(a, c)| a.iter().zip(c).filter(|(x, y)| x != y).count()).sum();
            let mut decoy = MhcSet::new(dev.id, bmax, &carry0)?;
            let (sd_, si_, dt, drained, retries) =
                time_ab(&s, 200, 21, &prefill, &mut || go(&mut decoy, None, b), &mut || go(&mut sd, Some(ind), b))?;
            exact &= diff == 0;
            bar(diff == 0, format!("4b:b{b}:bitexact"));
            bar(drained == 0, format!("4b:b{b}:drained"));
            mhc_d.push(dt);
            println!(
                "  b={b}: direct {sd_}, _ind {si_} ({:+.1}%); _ind - direct {dt}; differing outputs {diff} (split, mix, inv, carry, cur, norm){}{}",
                (si_.med / sd_.med - 1.0) * 100.0,
                if retries > 0 { format!("; {retries} pairs re-run") } else { String::new() },
                if drained > 0 { format!("; {drained} pairs still DRAINED, FAILS") } else { String::new() }
            );
        }
    }

    // ---- 5. one captured _ind graph, two context entries; node count ---------------------
    println!("== 5. one _ind graph replayed with context A, then B; nodes per captured launch");
    let w_b = q8_weight(dev.id, rows, k, 5)?;
    let b = 4u32;
    let ind = Ind::new(slot.raw() as u64).with(0, 0).with(1, 1).with(2, 2).with(3, 3);
    let (_, exec) =
        capture(&s, &mut |s| e.q8.matvec_bpack_ind(s, ind, &mut out_d, &w, &xq, &xs, rows as u32, k as u32, b))?;
    for (label, weight) in [("A", &w), ("B", &w_b)] {
        let mut ent = ArenaCtx::default();
        ent.p[0] = out_i.raw() as u64;
        ent.p[1] = weight.raw() as u64;
        ent.p[2] = xq.raw() as u64;
        ent.p[3] = xs.raw() as u64;
        ctxk.store(&s, &ent, &mut slot)?;
        exec.launch(&s)?;
        e.q8.matvec_bpack(&s, &mut out_d, weight, &xq, &xs, rows as u32, k as u32, b)?;
        s.synchronize()?;
        let n = b as usize * rows;
        let diff = down(&out_d, n)?.iter().zip(down(&out_i, n)?).filter(|(a, c)| **a != *c).count();
        exact &= diff == 0;
        bar(diff == 0, format!("5:context{label}"));
        println!("  context {label}: differing outputs vs direct on {label}: {diff}");
    }
    let launches = 4;
    let (nodes, _) = capture(&s, &mut |s| {
        ctxk.nop(s, &mut sink)?;
        e.q8.matvec_bpack_ind(s, ind, &mut out_d, &w, &xq, &xs, rows as u32, k as u32, b)?;
        e.q8.matvec_bpack(s, &mut out_d, &w, &xq, &xs, rows as u32, k as u32, 8)?;
        e.q8.matvec_bpack_ind(s, ind, &mut out_d, &w, &xq, &xs, rows as u32, k as u32, 1)
    })?;
    exact &= nodes == launches;
    bar(nodes == launches, "5:nodes".into());
    println!("  {launches} captured launches -> {nodes} graph nodes");

    // ---- the 2.0 per-step budget, per operating point ---------------------------------------
    // Everything the design adds per step against 1% of that step, at each operating point -- the
    // twin deltas grow with b, the write / launch counts do not, the step grows with rows (review
    // of 752a3c3: pairing b=8's delta with b=1's step overstated run 2 about threefold). Points,
    // two lanes (the binding case: 80 lane-layers): b = 1 / lane at 60 ms (2 rows, live 10-04),
    // b = 4 at 121.6 ms (8 rows, MULTI_LADDER_TWO), b = 8 at 196.8 ms (16 rows: 159.2 at 12 + 4 x
    // 9.4); GK_STEP_MS_B{1,4,8} override. Counts (review of 0122a52): 1 write per lane-layer
    // (presubmit off; 2 is a what-if); 8 stage graphs per lane-layer, 4 of them 1-node, charged at
    // the 1- / 8-node relaunch deltas; per lane-layer 8 gemv-like twins (q_b-shape delta:
    // conservative), 3 mhc twins, 7 small twins (a fixed prologue cost: charged at the larger of the
    // gemv / mhc deltas), each at the point's own b. Medians are charged when positive; the
    // upper-bound total uses each delta's ~96.7% upper bound. GPU + host is an upper bound (only
    // one is the pole at a time).
    let lane_layers = 2.0 * 40.0;
    let writes = lane_layers;
    let (gemv_per, mhc_per, small_per) = (8.0, 3.0, 7.0);
    let env_ms = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let points = [
        (1u32, env_ms("GK_STEP_MS_B1", 60.0)),
        (4, env_ms("GK_STEP_MS_B4", 121.6)),
        (8, env_ms("GK_STEP_MS_B8", 196.8)),
    ];
    let bs = [1u32, 4, 8];
    // (GPU us, host us) per step at batch `b`; `ub` = upper bounds.
    let cost = |b: u32, ub: bool| -> (f64, f64) {
        let ch = |d: &Delta| if ub { d.charge_ub() } else { d.charge() };
        let mut gpu = writes * write_cost.0.max(0.0);
        let mut host = writes * write_cost.1.max(0.0);
        for (_, dg, dh) in &relaunch {
            let launches = 4.0 * lane_layers; // 4 stage graphs per lane-layer of each size class
            gpu += launches * ch(dg);
            host += launches * ch(dh);
        }
        let i = bs.iter().position(|&x| x == b).expect("measured b");
        let (g, m) = (ch(&gemv_d[i]), ch(&mhc_d[i]));
        gpu += lane_layers * (gemv_per * g + mhc_per * m + small_per * g.max(m));
        (gpu, host)
    };
    println!("== per-step budget, 2 lanes x 40 layers, at each operating point:");
    println!(
        "  context writes   {writes:.0} x ({:.2} GPU + {:.2} host) us = {:.0} us  (what-if presubmit, 160 writes: {:.0} us)",
        write_cost.0,
        write_cost.1,
        writes * (write_cost.0 + write_cost.1).max(0.0),
        2.0 * writes * (write_cost.0 + write_cost.1).max(0.0)
    );
    for (nodes, dg, dh) in &relaunch {
        println!(
            "  relaunch {nodes}-node {:.0} x ({:.2} GPU + {:.2} host) us, ub ({:.2} + {:.2})",
            4.0 * lane_layers,
            dg.charge(),
            dh.charge(),
            dg.charge_ub(),
            dh.charge_ub()
        );
    }
    let mut verdicts = Vec::new();
    for &(b, step_ms) in &points {
        let i = bs.iter().position(|&x| x == b).expect("measured b");
        let budget = 0.01 * step_ms * 1e3;
        let pct = |us: f64| us / (step_ms * 1e3) * 100.0;
        let (g_med, h_med) = cost(b, false);
        let (g_ub, h_ub) = cost(b, true);
        let (med, ub) = (g_med + h_med, g_ub + h_ub);
        let (gm, mm) = (gemv_d[i].charge(), mhc_d[i].charge());
        // GO: medians and upper bounds fit. MARGINAL (the live A/B's <= 1% ms.step bar decides): only
        // the medians fit, or GPU + host does not while each timeline alone does. NO-GO otherwise.
        let v = if med <= budget && ub <= budget {
            "GO"
        } else if med <= budget || (g_med <= budget && h_med <= budget) {
            "MARGINAL"
        } else {
            "NO-GO"
        };
        println!(
            "  b={b} at {step_ms:.1} ms (1% = {budget:.0} us): twins {lane_layers:.0} x ({gemv_per:.0} x {gm:.2} + {mhc_per:.0} x {mm:.2} + {small_per:.0} x {:.2}) us; total {med:.0} us = {:.2}% (GPU {:.2}%, host {:.2}%), upper bounds {:.2}% -> {v}",
            gm.max(mm),
            pct(med),
            pct(g_med),
            pct(h_med),
            pct(ub)
        );
        verdicts.push((b, v, pct(med), pct(ub)));
    }
    let budget_verdict = if verdicts.iter().any(|v| v.1 == "NO-GO") {
        "NO-GO"
    } else if verdicts.iter().all(|v| v.1 == "GO") {
        "GO"
    } else {
        "MARGINAL"
    };
    let summary =
        verdicts.iter().map(|(b, v, m, u)| format!("b{b} {v} {m:.2}%/{u:.2}%")).collect::<Vec<_>>().join(", ");
    println!("  budget: {budget_verdict} ({summary})");
    if !nogo.is_empty() {
        println!("STEP0: NO-GO ({})", nogo.join(", "));
    } else if budget_verdict == "GO" {
        println!("STEP0: GO");
    } else {
        println!("STEP0: {budget_verdict} (budget per point, medians/upper bounds: {summary})");
    }
    if !exact {
        return Err(eyre!("graph_keys step 0: a bit-exactness, coherence or node-count check failed"));
    }
    Ok(())
}

/// Host only: the upper bound is the k-th smallest paired difference (9th of 11, 16th of 21:
/// P(Bin(21, 1/2) <= 14) = 0.961 < 0.967 <= P(<= 15) = 0.987).
#[test]
fn delta_upper_bound_order_statistic() {
    for (n, k) in [(11usize, 9usize), (21, 16)] {
        let a = vec![0.0; n];
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let d = Delta::of(&a, &b);
        assert_eq!(d.ub, k as f64, "n={n}");
        assert_eq!(d.med, (n / 2 + 1) as f64);
        assert_eq!(d.pos, n);
    }
}

/// Paired host enqueue cost per launch (us) of `a` vs `b`: `n` launches per run, each run
/// behind `spin` (so the queue never fills), the order alternating pair to pair.
fn host_ab(
    s: &Stream,
    n: usize,
    pairs: usize,
    spin: &dyn Fn(&Stream) -> eyre::Result<()>,
    a: &mut dyn FnMut() -> eyre::Result<()>,
    b: &mut dyn FnMut() -> eyre::Result<()>,
) -> eyre::Result<(Stat, Stat, Delta)> {
    a()?;
    b()?;
    let mut run = |use_b: bool| -> eyre::Result<f64> {
        s.synchronize()?;
        spin(s)?;
        let t = Instant::now();
        for _ in 0..n {
            if use_b {
                b()?
            } else {
                a()?
            }
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        s.synchronize()?;
        Ok(us)
    };
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    for i in 0..pairs {
        if i % 2 == 0 {
            ta.push(run(false)?);
            tb.push(run(true)?);
        } else {
            tb.push(run(true)?);
            ta.push(run(false)?);
        }
    }
    Ok((Stat::of(&ta), Stat::of(&tb), Delta::of(&ta, &tb)))
}

/// Step 0b of docs/v41/GRAPH_KEYS_DESIGN.md (2.11, rev 4.1): dGPU only, hub DOWN.
///  1. Every twin at its call site's REAL shape, b = 1 / 4 / 8, bit-exact, 21 paired runs: the
///     bpack gemv at q_a / q_b / kv / wo_b / shared down, the grouped wo_a, and the small twins
///     `rms_quant_q8_1280_batched`, `kv_rms_rope_fp8`, `rope_tail_batched_copy` (rope arguments
///     from context slots).
///  2. The carrier (`mhc_fast_batched_ctx`) vs the direct `mhc_fast_batched`: bit-exact, the slot
///     holding the entry afterwards, 1000 carrier -> `_ind_canary` reader rounds, paired GPU and
///     host cost.
///  3. R1: a direct `launch_fast` vs a replay of its captured 1-node graph, paired GPU and host;
///     the uncached env reads a stage body makes today (diagnostic).
///  4. Relaunch, K = 8 vs 80 (8-node graphs), 21 pairs.
///  The per-point budget (2.11): carrier + R1 + relaunch + the twins of a lane-layer at their
///  measured deltas, vs 1% of ms.step at b = 1 / 4 / 8. Last line STEP0B: GO / MARGINAL / NO-GO.
///
/// HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release \
///   --features v41 -p v4flash-kernels --test bench_graph_keys graph_keys_step0b -- --ignored --nocapture
#[test]
#[ignore]
fn graph_keys_step0b() -> eyre::Result<()> {
    use std::collections::BTreeMap;
    use v4flash_kernels::{RopeParams, RopeTail};
    color_eyre::install().ok();
    let (dev, arch) = pick_dgpu()?;
    dev.set_current()?;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let ctxk = ArenaCtxKernels::for_arch(&arch)?;
    let arena = MhcArena::for_arch(&arch)?;
    let s = Stream::new(dev.id)?;
    let mut slot = DeviceBuffer::<u64>::new(dev.id, ARENA_CTX_WORDS)?;
    let ctx = slot.raw() as u64;
    let mut exact = true;
    let mut nogo: Vec<String> = Vec::new();
    let mut bar = |ok: bool, item: String| {
        if !ok {
            nogo.push(item);
        }
    };
    let tpm = {
        e.q8.slack_probe_spin(&s, 1000)?;
        s.synchronize()?;
        let (e0, e1) = (Event::new()?, Event::new()?);
        e0.record(&s)?;
        e.q8.slack_probe_spin(&s, 2_000_000)?;
        e1.record(&s)?;
        e1.synchronize()?;
        2_000_000.0 / Event::elapsed_ms(&e0, &e1)? as f64
    };
    let spin_ms = |ms: f64| (ms * tpm) as u64;
    let prefill = |s: &Stream| e.q8.slack_probe_spin(s, spin_ms(1.5));
    let bs = [1u32, 4, 8];
    // (family, b) -> _ind - direct
    let mut twin: BTreeMap<(String, u32), Delta> = BTreeMap::new();
    let report = |name: &str, b: u32, sd_: Stat, si_: Stat, dt: Delta, diff: usize, retries: usize, drained: usize| {
        println!(
            "  {name:>12} b={b}: direct {sd_}, _ind {si_} ({:+.1}%); _ind - direct {dt}; differing outputs {diff}{}{}",
            (si_.med / sd_.med - 1.0) * 100.0,
            if retries > 0 { format!("; {retries} pairs re-run") } else { String::new() },
            if drained > 0 { format!("; {drained} pairs still DRAINED, FAILS") } else { String::new() }
        );
    };

    // ---- 1a. the bpack gemv twin at every call site's shape --------------------------------
    println!("== 1. twins at the real shapes (21 pairs; 1.5 ms spin + warm-up)");
    for (name, rows, k) in [
        ("q_a", 1280usize, 5120usize),
        ("q_b", 32768, 1280),
        ("kv", 512, 5120),
        ("wo_b", 5120, 8192),
        ("shared_down", 5120, 2304),
    ] {
        let w = q8_weight(dev.id, rows, k, 31)?;
        let xq = up(dev.id, &lcg_bytes(32, 8 * k).iter().map(|&b| b as i8).collect::<Vec<_>>())?;
        let xs = up(dev.id, &(0..8 * (k / 32)).map(|i| 0.002 + (i % 7) as f32 * 0.001).collect::<Vec<_>>())?;
        let mut out_d = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
        let mut out_i = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
        let mut out_x = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
        let mut ent = ArenaCtx::default();
        ent.p[..4].copy_from_slice(&[out_i.raw() as u64, w.raw() as u64, xq.raw() as u64, xs.raw() as u64]);
        ctxk.store(&s, &ent, &mut slot)?;
        let ind = Ind::new(ctx).with(0, 0).with(1, 1).with(2, 2).with(3, 3);
        for b in bs {
            out_i.fill_zero()?;
            let (sd_, si_, dt, drained, retries) = time_ab(
                &s,
                200,
                21,
                &prefill,
                &mut || e.q8.matvec_bpack(&s, &mut out_d, &w, &xq, &xs, rows as u32, k as u32, b),
                &mut || e.q8.matvec_bpack_ind(&s, ind, &mut out_x, &w, &xq, &xs, rows as u32, k as u32, b),
            )?;
            s.synchronize()?;
            let n = b as usize * rows;
            let diff = down(&out_d, n)?.iter().zip(down(&out_i, n)?).filter(|(a, c)| **a != *c).count();
            exact &= diff == 0;
            bar(diff == 0, format!("1:{name}:b{b}:bitexact"));
            bar(drained == 0, format!("1:{name}:b{b}:drained"));
            report(name, b, sd_, si_, dt, diff, retries, drained);
            twin.insert((name.to_string(), b), dt);
        }
    }

    // ---- 1b. the grouped wo_a twin (8 groups x 1024 rows x 4096) ------------------------------
    {
        let (group_dim, rank, n_groups) = (4096usize, 1024usize, 8usize);
        let out_dim = rank * n_groups;
        let w = q8_weight(dev.id, out_dim, group_dim, 33)?;
        let kin = n_groups * group_dim;
        let xq = up(dev.id, &lcg_bytes(34, 8 * kin).iter().map(|&b| b as i8).collect::<Vec<_>>())?;
        let xs = up(dev.id, &(0..8 * (kin / 32)).map(|i| 0.002 + (i % 5) as f32 * 0.001).collect::<Vec<_>>())?;
        let mut out_d = DeviceBuffer::<f32>::new(dev.id, 8 * out_dim)?;
        let mut out_i = DeviceBuffer::<f32>::new(dev.id, 8 * out_dim)?;
        let mut out_x = DeviceBuffer::<f32>::new(dev.id, 8 * out_dim)?;
        let mut ent = ArenaCtx::default();
        ent.p[..4].copy_from_slice(&[out_i.raw() as u64, w.raw() as u64, xq.raw() as u64, xs.raw() as u64]);
        ctxk.store(&s, &ent, &mut slot)?;
        let ind = Ind::new(ctx).with(0, 0).with(1, 1).with(2, 2).with(3, 3);
        let g = &e.q8_grouped;
        for b in bs {
            out_i.fill_zero()?;
            let (gd, rk, ng) = (group_dim as u32, rank as u32, n_groups as u32);
            let (sd_, si_, dt, drained, retries) = time_ab(
                &s,
                200,
                21,
                &prefill,
                &mut || g.matvec_grouped_bpack(&s, &mut out_d, &w, &xq, &xs, gd, rk, ng, b),
                &mut || g.matvec_grouped_bpack_ind(&s, ind, &mut out_x, &w, &xq, &xs, gd, rk, ng, b),
            )?;
            s.synchronize()?;
            // b = 1: the runtime-batch kernel's twin (production's arm); 2..8: the tB twins.
            let got = &out_i;
            let n = b as usize * out_dim;
            let diff = down(&out_d, n)?.iter().zip(down(got, n)?).filter(|(a, c)| **a != *c).count();
            exact &= diff == 0;
            bar(diff == 0, format!("1:wo_a:b{b}:bitexact"));
            bar(drained == 0, format!("1:wo_a:b{b}:drained"));
            report("wo_a", b, sd_, si_, dt, diff, retries, drained);
            twin.insert(("wo_a".to_string(), b), dt);
        }
    }

    // ---- 1c. small twins ----------------------------------------------------------------------
    let rope = RopeParams {
        freq_base: 10000.0,
        freq_scale: 0.025,
        ext_factor: 1.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        n_ctx_orig: 65536,
    };
    let rope_words = RopeTail::arena_ctx_rope_words(&rope, 64);
    const RS: [usize; 3] = [20, 21, 22];
    let pos = up(dev.id, &[17i32, 4096, 65000, 3, 123456, 9, 31337, 77])?;
    {
        // rms_quant_q8_1280_batched
        let x = up(dev.id, &lcg_f32(41, 8 * 1280, -3.0, 3.0))?;
        let wt = up(dev.id, &lcg_f32(42, 1280, 0.5, 1.5))?;
        let mk_f = |n: usize| DeviceBuffer::<f32>::new(dev.id, n);
        let (mut od, mut oi, mut ox) = (mk_f(8 * 1280)?, mk_f(8 * 1280)?, mk_f(8 * 1280)?);
        let (mut sdd, mut sdi, mut sdx) = (mk_f(8 * 40)?, mk_f(8 * 40)?, mk_f(8 * 40)?);
        let mk_i = |n: usize| DeviceBuffer::<i8>::new(dev.id, n);
        let (mut qd, mut qi, mut qx) = (mk_i(8 * 1280)?, mk_i(8 * 1280)?, mk_i(8 * 1280)?);
        let mut ent = ArenaCtx::default();
        ent.p[..5].copy_from_slice(&[
            oi.raw() as u64,
            qi.raw() as u64,
            sdi.raw() as u64,
            x.raw() as u64,
            wt.raw() as u64,
        ]);
        ctxk.store(&s, &ent, &mut slot)?;
        let ind = (0..5).fold(Ind::new(ctx), |ind, i| ind.with(i, i));
        for b in bs {
            oi.fill_zero()?;
            sdi.fill_zero()?;
            qi.fill_zero()?;
            let (sd_, si_, dt, drained, retries) = time_ab(
                &s,
                200,
                21,
                &prefill,
                &mut || e.rms_w.launch_weighted_quant_q8_1280(&s, &mut od, &mut qd, &mut sdd, &x, &wt, RMS_EPS, b),
                &mut || {
                    e.rms_w.launch_weighted_quant_q8_1280_ind(&s, ind, &mut ox, &mut qx, &mut sdx, &x, &wt, RMS_EPS, b)
                },
            )?;
            s.synchronize()?;
            let n = b as usize * 1280;
            let mut diff = down(&od, n)?.iter().zip(down(&oi, n)?).filter(|(a, c)| **a != *c).count();
            diff += down(&sdd, b as usize * 40)?
                .iter()
                .zip(down(&sdi, b as usize * 40)?)
                .filter(|(a, c)| **a != *c)
                .count();
            let (mut hq, mut hqi) = (vec![0i8; 8 * 1280], vec![0i8; 8 * 1280]);
            qd.copy_to_host(&mut hq)?;
            qi.copy_to_host(&mut hqi)?;
            diff += hq[..n].iter().zip(&hqi[..n]).filter(|(a, c)| a != c).count();
            exact &= diff == 0;
            bar(diff == 0, format!("1:rms_quant:b{b}:bitexact"));
            bar(drained == 0, format!("1:rms_quant:b{b}:drained"));
            report("rms_quant", b, sd_, si_, dt, diff, retries, drained);
            twin.insert(("rms_quant".to_string(), b), dt);
        }
    }
    {
        // kv_rms_rope_fp8
        let x = up(dev.id, &lcg_f32(43, 8 * 512, -3.0, 3.0))?;
        let wt = up(dev.id, &lcg_f32(44, 512, 0.5, 1.5))?;
        let (mut od, mut oi, mut ox) = (
            DeviceBuffer::<f32>::new(dev.id, 8 * 512)?,
            DeviceBuffer::<f32>::new(dev.id, 8 * 512)?,
            DeviceBuffer::<f32>::new(dev.id, 8 * 512)?,
        );
        let mut ent = ArenaCtx::default();
        ent.p[..4].copy_from_slice(&[oi.raw() as u64, x.raw() as u64, wt.raw() as u64, pos.raw() as u64]);
        for (j, sl) in RS.iter().enumerate() {
            ent.p[*sl] = rope_words[j];
        }
        ctxk.store(&s, &ent, &mut slot)?;
        let ind = (0..4).fold(Ind::new(ctx), |ind, i| ind.with(i, i));
        for b in bs {
            oi.fill_zero()?;
            let (sd_, si_, dt, drained, retries) = time_ab(
                &s,
                200,
                21,
                &prefill,
                &mut || e.rope.launch_kv_rms_rope_fp8(&s, &mut od, &x, &wt, RMS_EPS, &pos, 64, b, &rope),
                &mut || e.rope.launch_kv_rms_rope_fp8_ind(&s, ind, RS, &mut ox, &x, &wt, RMS_EPS, &pos, 64, b),
            )?;
            s.synchronize()?;
            let n = b as usize * 512;
            let diff = down(&od, n)?.iter().zip(down(&oi, n)?).filter(|(a, c)| **a != *c).count();
            exact &= diff == 0;
            bar(diff == 0, format!("1:kv_rms_rope:b{b}:bitexact"));
            bar(drained == 0, format!("1:kv_rms_rope:b{b}:drained"));
            report("kv_rms_rope", b, sd_, si_, dt, diff, retries, drained);
            twin.insert(("kv_rms_rope".to_string(), b), dt);
        }
    }
    {
        // rope_tail_batched_copy (q: 64 heads x 512, n_rot 64)
        let n_all = 8 * 64 * 512;
        let src = up(dev.id, &lcg_f32(45, n_all, -2.0, 2.0))?;
        let (mut dd, mut di, mut dx) = (
            DeviceBuffer::<f32>::new(dev.id, n_all)?,
            DeviceBuffer::<f32>::new(dev.id, n_all)?,
            DeviceBuffer::<f32>::new(dev.id, n_all)?,
        );
        let mut ent = ArenaCtx::default();
        ent.p[..3].copy_from_slice(&[di.raw() as u64, src.raw() as u64, pos.raw() as u64]);
        for (j, sl) in RS.iter().enumerate() {
            ent.p[*sl] = rope_words[j];
        }
        ctxk.store(&s, &ent, &mut slot)?;
        let ind = (0..3).fold(Ind::new(ctx), |ind, i| ind.with(i, i));
        for b in bs {
            di.fill_zero()?;
            let (sd_, si_, dt, drained, retries) = time_ab(
                &s,
                200,
                21,
                &prefill,
                &mut || e.rope.launch_forward_batched_copy(&s, &mut dd, &src, &pos, 64, 512, 64, b, &rope),
                &mut || e.rope.launch_forward_batched_copy_ind(&s, ind, RS, &mut dx, &src, &pos, 64, 512, 64, b),
            )?;
            s.synchronize()?;
            let n = b as usize * 64 * 512;
            let diff = down(&dd, n)?.iter().zip(down(&di, n)?).filter(|(a, c)| **a != *c).count();
            exact &= diff == 0;
            bar(diff == 0, format!("1:rope_copy:b{b}:bitexact"));
            bar(drained == 0, format!("1:rope_copy:b{b}:drained"));
            report("rope_copy", b, sd_, si_, dt, diff, retries, drained);
            twin.insert(("rope_copy".to_string(), b), dt);
        }
    }

    // ---- 2. the carrier --------------------------------------------------------------------------
    println!("== 2. carrier mhc_fast_batched_ctx (pre_attn) vs direct mhc_fast_batched");
    let (hcd, m, ne, bmax) = (HC_DIM as usize, HC_MIX_DIM as usize, N_EMBD as usize, 8usize);
    let w_bits: Vec<u8> = lcg_bytes(11, 2 * m * hcd)
        .chunks(2)
        .flat_map(|c| {
            ((((c[0] >> 7) as u16) << 15)
                | ((10 + (c[0] as u16 % 5)) << 10)
                | (((c[0] as u16) << 8 | c[1] as u16) & 0x3ff))
                .to_le_bytes()
        })
        .collect();
    let mw = up(dev.id, &w_bits)?;
    let mx = up(dev.id, &lcg_f32(12, bmax * hcd, -2.0, 2.0))?;
    let msc = up(dev.id, &[0.8f32, 1.3, 0.6])?;
    let mbase = up(dev.id, &lcg_f32(13, m, -1.0, 1.0))?;
    let mnw = up(dev.id, &lcg_f32(14, ne, 0.5, 1.5))?;
    let carry0: Vec<f32> = lcg_f32(15, bmax * m, 0.0, 1.0)
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % m < 4 { 0.1 + 0.5 * v } else { v - 0.5 })
        .collect();
    let (mut sd, mut sc) = (MhcSet::new(dev.id, bmax, &carry0)?, MhcSet::new(dev.id, bmax, &carry0)?);
    // None = the direct kernel; Some(entry) = the carrier writing `entry`.
    let go = |set: &mut MhcSet, carrier: Option<&ArenaCtx>, slot: &mut DeviceBuffer<u64>, b: u32| -> eyre::Result<()> {
        let mix = FastMix {
            weight: &mw,
            x: &mx,
            scale: &msc,
            base: &mbase,
            mode: MIX_PRE_SCALED,
            split_out: &mut set.split,
            mix_out: &mut set.mix,
            counters: &mut set.cnt,
            inv_rows: &mut set.inv,
        };
        let col = FastCollapse { x: &mx, cur_out: &mut set.cur, norm_out: &mut set.norm, norm_w: &mnw };
        match carrier {
            None => arena.launch_fast(
                &s,
                Some(mix),
                Some(col),
                &mut set.carry,
                true,
                HC_DIM,
                RMS_EPS,
                SINKHORN_ITERS,
                SINKHORN_EPS,
                b,
            ),
            Some(ent) => arena.launch_fast_carrier(
                &s,
                ent,
                slot.raw() as u64,
                Some(mix),
                Some(col),
                &mut set.carry,
                true,
                HC_DIM,
                RMS_EPS,
                SINKHORN_ITERS,
                SINKHORN_EPS,
                b,
            ),
        }
    };
    let mut probe = ArenaCtx::default();
    for (i, p) in probe.p.iter_mut().enumerate() {
        *p = 0xC0DE_0000 + 16 * i as u64;
    }
    probe.seq = 0x5EED;
    let mut carrier_d = BTreeMap::new();
    for b in bs {
        sd.reset(&carry0)?;
        sc.reset(&carry0)?;
        slot.fill_zero()?;
        go(&mut sd, None, &mut slot, b)?;
        go(&mut sc, Some(&probe), &mut slot, b)?;
        s.synchronize()?;
        let (rd, rc) = (sd.dump(b as usize)?, sc.dump(b as usize)?);
        let diff: usize = rd.iter().zip(&rc).map(|(a, c)| a.iter().zip(c).filter(|(x, y)| x != y).count()).sum();
        let mut hs = vec![0u64; ARENA_CTX_WORDS];
        slot.copy_to_host(&mut hs)?;
        let slot_ok = hs[..32] == probe.p[..] && hs[32] == probe.seq && hs[33] == probe.log;
        exact &= diff == 0 && slot_ok;
        bar(diff == 0, format!("2:carrier:b{b}:bitexact"));
        bar(slot_ok, format!("2:carrier:b{b}:slot"));
        let (mut d1, mut d2) = (MhcSet::new(dev.id, bmax, &carry0)?, MhcSet::new(dev.id, bmax, &carry0)?);
        let mut slot2 = DeviceBuffer::<u64>::new(dev.id, ARENA_CTX_WORDS)?;
        let mut unused = DeviceBuffer::<u64>::new(dev.id, ARENA_CTX_WORDS)?; // the direct arm writes no slot
        let (sd_, si_, dt, drained, retries) =
            time_ab(&s, 200, 21, &prefill, &mut || go(&mut d1, None, &mut unused, b), &mut || {
                go(&mut d2, Some(&probe), &mut slot2, b)
            })?;
        bar(drained == 0, format!("2:carrier:b{b}:drained"));
        let (hd, hc, dh) = host_ab(
            &s,
            100,
            21,
            &|s: &Stream| e.q8.slack_probe_spin(s, spin_ms(5.0)),
            &mut || go(&mut d1, None, &mut unused, b),
            &mut || go(&mut d2, Some(&probe), &mut slot2, b),
        )?;
        println!(
            "  b={b}: GPU direct {sd_}, carrier {si_}; carrier - direct {dt}{}; host direct {hd}, carrier {hc}, delta {dh}; outputs differing {diff}, slot {}",
            if retries > 0 { format!(" ({retries} pairs re-run)") } else { String::new() },
            if slot_ok { "holds the entry" } else { "WRONG" }
        );
        carrier_d.insert(b, (dt, dh));
    }
    // Carrier -> reader coherence: the carrier writes round i's entry, a captured `_ind_canary` gemv
    // graph reads it (outputs and canary records checked).
    {
        let (rows_c, k_c, rounds) = (64usize, 256usize, 1000usize);
        let w0 = q8_weight(dev.id, rows_c, k_c, 7)?;
        let w1 = q8_weight(dev.id, rows_c, k_c, 8)?;
        let xq_c = up(dev.id, &lcg_bytes(9, k_c).iter().map(|&b| b as i8).collect::<Vec<_>>())?;
        let xs_c = up(dev.id, &vec![0.01f32; k_c / 32])?;
        let mut tmp = DeviceBuffer::<f32>::new(dev.id, rows_c)?;
        let mut refs = Vec::new();
        for wj in [&w0, &w1] {
            e.q8.matvec_bpack(&s, &mut tmp, wj, &xq_c, &xs_c, rows_c as u32, k_c as u32, 1)?;
            s.synchronize()?;
            refs.push(down(&tmp, rows_c)?);
        }
        let outs = DeviceBuffer::<f32>::new(dev.id, rounds * rows_c)?;
        let log = CanaryLog::new(dev.id, rounds as u32)?;
        const SL: [usize; 4] = [0, 9, 18, 27];
        let ind = Ind::new(ctx).with(0, SL[0]).with(1, SL[1]).with(2, SL[2]).with(3, SL[3]).with_canary(7);
        let (_, exec) = capture(&s, &mut |s| {
            e.q8.matvec_bpack_ind(s, ind, &mut tmp, &w0, &xq_c, &xs_c, rows_c as u32, k_c as u32, 1)
        })?;
        let mut cset = MhcSet::new(dev.id, bmax, &carry0)?;
        for i in 0..rounds {
            let mut ent = ArenaCtx::default();
            ent.p[SL[0]] = outs.raw() as u64 + (i * rows_c * 4) as u64;
            ent.p[SL[1]] = [&w0, &w1][i % 2].raw() as u64;
            ent.p[SL[2]] = xq_c.raw() as u64;
            ent.p[SL[3]] = xs_c.raw() as u64;
            ent.seq = i as u64;
            ent.log = log.buf.raw() as u64;
            go(&mut cset, Some(&ent), &mut slot, 1)?;
            exec.launch(&s)?;
        }
        s.synchronize()?;
        let host = down(&outs, rounds * rows_c)?;
        let bad = (0..rounds).filter(|&i| host[i * rows_c..(i + 1) * rows_c] != refs[i % 2][..]).count();
        let (cursor, recs) = log.read()?;
        let want_xor = |i: usize| {
            (outs.raw() as u64 + (i * rows_c * 4) as u64)
                ^ [&w0, &w1][i % 2].raw() as u64
                ^ xq_c.raw() as u64
                ^ xs_c.raw() as u64
        };
        let bad_rec = recs
            .iter()
            .enumerate()
            .filter(|&(i, r)| r.seq != i as u64 || r.tag != 7 || r.ptr_xor != want_xor(i))
            .count();
        let ok = bad == 0 && cursor as usize == rounds && bad_rec == 0;
        exact &= ok;
        bar(ok, "2:carrier:coherence".into());
        println!("  carrier -> reader, {rounds} rounds: {bad} stale outputs; canary {cursor} records, {bad_rec} wrong");
    }

    // ---- 3. R1: direct launch vs a 1-node graph replay -----------------------------------------
    println!("== 3. R1: direct launch_fast vs replay of its captured 1-node graph (b = 4, pre_attn)");
    let r1 = {
        let b = 4u32;
        let mut d1 = MhcSet::new(dev.id, bmax, &carry0)?;
        let mut d2 = MhcSet::new(dev.id, bmax, &carry0)?;
        let mut slot2 = DeviceBuffer::<u64>::new(dev.id, ARENA_CTX_WORDS)?;
        let (nodes, g1) = capture(&s, &mut |_s| go(&mut d2, None, &mut slot2, b))?;
        let (gg, gd, dg, drained, _) =
            time_ab(&s, 200, 21, &prefill, &mut || g1.launch(&s), &mut || go(&mut d1, None, &mut slot2, b))?;
        bar(drained == 0, "3:r1:drained".into());
        let (hg, hd, dh) = host_ab(
            &s,
            100,
            21,
            &|s: &Stream| e.q8.slack_probe_spin(s, spin_ms(5.0)),
            &mut || g1.launch(&s),
            &mut || go(&mut d1, None, &mut slot2, b),
        )?;
        println!("  graph ({nodes} node) GPU {gg}, direct {gd}; direct - graph {dg}");
        println!("  host per launch: graph {hg}, direct {hd}; direct - graph {dh}");
        // Diagnostic: the env reads the four stage bodies make per call today (become LazyLock).
        let t = Instant::now();
        let mut sink = 0usize;
        for _ in 0..1000 {
            for k in ["V41_MHC_PRE_SCALED", "V41_MHC_NARROW", "V41_ROUTER_WMMA", "V41_MHC_PRE_SCALED"] {
                sink += std::env::var(k).map(|v| v.len()).unwrap_or(0);
            }
        }
        println!(
            "  4 uncached env reads: {:.2} us per lane-layer (removed by LazyLock) [{sink}]",
            t.elapsed().as_secs_f64() * 1e6 / 1000.0
        );
        (dg, dh)
    };

    // ---- 4. relaunch, K = 8 vs 80, 8-node graphs --------------------------------------------------
    println!("== 4. relaunch: 80 launches round-robin over K = 8 vs 80 8-node executables, 21 pairs");
    let relaunch = {
        let (rows, k) = (1280usize, 5120usize);
        let w = q8_weight(dev.id, rows, k, 1)?;
        let xq = up(dev.id, &lcg_bytes(2, 8 * k).iter().map(|&b| b as i8).collect::<Vec<_>>())?;
        let xs = up(dev.id, &vec![0.01f32; 8 * (k / 32)])?;
        let mut out = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
        let mut body = |s: &Stream| -> eyre::Result<()> {
            for b in 1..=8u32 {
                e.q8.matvec_bpack(s, &mut out, &w, &xq, &xs, rows as u32, k as u32, b)?;
            }
            Ok(())
        };
        let execs: Vec<GraphExec> =
            (0..90).map(|_| capture(&s, &mut body).map(|c| c.1)).collect::<eyre::Result<_>>()?;
        let run = |kk: usize| -> eyre::Result<(f64, f64, bool)> {
            s.synchronize()?;
            let (e0, e1) = (Event::new()?, Event::new()?);
            e.q8.slack_probe_spin(&s, spin_ms(1.0))?;
            for x in &execs[80..90] {
                x.launch(&s)?;
            }
            e0.record(&s)?;
            let t = Instant::now();
            for i in 0..80 {
                execs[i % kk].launch(&s)?;
            }
            let host_us = t.elapsed().as_secs_f64() * 1e6 / 80.0;
            let drained = e0.query()?;
            e1.record(&s)?;
            e1.synchronize()?;
            Ok((host_us, Event::elapsed_ms(&e0, &e1)? as f64 * 1e3 / 80.0, drained))
        };
        let (mut g8, mut g80, mut h8, mut h80) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut any_drained = false;
        for i in 0..21 {
            for attempt in 0..4 {
                let order = if i % 2 == 0 { [8usize, 80] } else { [80, 8] };
                let r0 = run(order[0])?;
                let r1_ = run(order[1])?;
                if (r0.2 || r1_.2) && attempt < 3 {
                    continue;
                }
                any_drained |= r0.2 || r1_.2;
                let (r8, r80) = if order[0] == 8 { (r0, r1_) } else { (r1_, r0) };
                g8.push(r8.1);
                h8.push(r8.0);
                g80.push(r80.1);
                h80.push(r80.0);
                break;
            }
        }
        bar(!any_drained, "4:relaunch:drained".into());
        let (dg, dh) = (Delta::of(&g80, &g8), Delta::of(&h80, &h8));
        println!("  K=8 {} vs K=80 {}; GPU {dg}; host {dh}", Stat::of(&g8), Stat::of(&g80));
        (dg, dh)
    };

    // ---- the per-point budget (2.11, rev 4.1) ----------------------------------------------------
    // Per lane-layer: the carrier's delta (instead of a standalone write); R1's direct - graph delta
    // x 4 single-launch stages; 4 multi-node graph launches at the relaunch delta; the twins at their
    // measured deltas -- q_a, q_b, kv, wo_b, shared_down, wo_a measured, the shared gate/up charged
    // at the largest gemv delta (twice at b >= 6: two launches), rms_quant / kv_rms_rope / rope_copy
    // measured and the 2 unmeasured small families (rope_inv_quant_q8, the shared input quantize)
    // at the largest small delta. 80 lane-layers (2 lanes x 40 layers) at b = 1 / 4 / 8
    // per lane, ms.step 60 / 121.6 / 196.8 ms (GK_STEP_MS_B{1,4,8}).
    let lane_layers = 80.0;
    let env_ms = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let points = [
        (1u32, env_ms("GK_STEP_MS_B1", 60.0)),
        (4, env_ms("GK_STEP_MS_B4", 121.6)),
        (8, env_ms("GK_STEP_MS_B8", 196.8)),
    ];
    let ch = |d: &Delta, ub: bool| if ub { d.charge_ub() } else { d.charge() };
    let gemv_sites = ["q_a", "q_b", "kv", "wo_b", "shared_down"];
    let small = ["rms_quant", "kv_rms_rope", "rope_copy"];
    // (GPU, host) us per step at b
    let cost = |b: u32, ub: bool| -> (f64, f64) {
        let t = |name: &str| ch(&twin[&(name.to_string(), b)], ub);
        let gemv_max = gemv_sites.iter().map(|n| t(n)).fold(0.0, f64::max);
        let small_max = small.iter().map(|n| t(n)).fold(0.0, f64::max);
        // shared gate / up: one fused twin at b <= 5, two gemv twins at b = 6..8 (charged at the
        // largest gemv delta); the unmeasured small families: rope_inv_quant_q8 and the shared
        // input quantize (2, at the largest small delta).
        let gateup = if b <= 5 { 1.0 } else { 2.0 };
        let twins = gemv_sites.iter().map(|n| t(n)).sum::<f64>()
            + gateup * gemv_max
            + t("wo_a")
            + small.iter().map(|n| t(n)).sum::<f64>()
            + 2.0 * small_max;
        let (cg, chh) = &carrier_d[&b];
        let gpu = lane_layers * (ch(cg, ub) + 4.0 * ch(&r1.0, ub) + 4.0 * ch(&relaunch.0, ub) + twins);
        let host = lane_layers * (ch(chh, ub) + 4.0 * ch(&r1.1, ub) + 4.0 * ch(&relaunch.1, ub));
        (gpu, host)
    };
    println!("== per-step budget (rev 4.1), 2 lanes x 40 layers:");
    let mut verdicts = Vec::new();
    for &(b, step_ms) in &points {
        let budget = 0.01 * step_ms * 1e3;
        let pct = |us: f64| us / (step_ms * 1e3) * 100.0;
        let (g_med, h_med) = cost(b, false);
        let (g_ub, h_ub) = cost(b, true);
        let (med, ub) = (g_med + h_med, g_ub + h_ub);
        let v = if med <= budget && ub <= budget {
            "GO"
        } else if med <= budget || (g_med <= budget && h_med <= budget) {
            "MARGINAL"
        } else {
            "NO-GO"
        };
        println!(
            "  b={b} at {step_ms:.1} ms (1% = {budget:.0} us): total {med:.0} us = {:.2}% (GPU {:.2}%, host {:.2}%), upper bounds {:.2}% -> {v}",
            pct(med),
            pct(g_med),
            pct(h_med),
            pct(ub)
        );
        verdicts.push((b, v, pct(med), pct(ub)));
    }
    let budget_verdict = if verdicts.iter().any(|v| v.1 == "NO-GO") {
        "NO-GO"
    } else if verdicts.iter().all(|v| v.1 == "GO") {
        "GO"
    } else {
        "MARGINAL"
    };
    let summary =
        verdicts.iter().map(|(b, v, m, u)| format!("b{b} {v} {m:.2}%/{u:.2}%")).collect::<Vec<_>>().join(", ");
    if !nogo.is_empty() {
        println!("STEP0B: NO-GO ({})", nogo.join(", "));
    } else if budget_verdict == "GO" {
        println!("STEP0B: GO ({summary})");
    } else {
        println!("STEP0B: {budget_verdict} (budget per point, medians/upper bounds: {summary})");
    }
    if !exact {
        return Err(eyre!("graph_keys step 0b: a bit-exactness, slot or coherence check failed"));
    }
    Ok(())
}
