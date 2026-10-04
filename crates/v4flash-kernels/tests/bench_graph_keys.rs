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
//!  The go / no-go: the exactness items, no drained queue, no BLOCKS, and ONE per-step budget --
//!  writes x write cost + graph launches x the K = 8 relaunch delta (1- and 8-node) + twin launches
//!  x the twin deltas, medians and ~96.7% upper bounds, vs 1% of ms.step p50 (GK_STEP_MS, default
//!  60 = 2 rows / 2 lanes live): GO / MARGINAL (the live A/B decides) / NO-GO.
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
        write!(f, "{:7.2} us [{:.2}..{:.2}]", self.med, self.min, self.max)
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
        Delta { med: d[n / 2], ub: d[k.min(n) - 1], min: d[0], max: d[n - 1], pos: d.iter().filter(|v| **v > 0.0).count(), n }
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

/// Pairs of runs of `reps` launches of `a` and of `b` (the order alternating pair to pair), each
/// run queued behind `prefill` (a spin long enough to cover the enqueue, so the events time the
/// GPU back to back, not the host): (a, b, b - a paired, runs whose spin ended before the host
/// finished enqueueing).
fn time_ab(
    s: &Stream,
    reps: usize,
    pairs: usize,
    prefill: &dyn Fn(&Stream) -> eyre::Result<()>,
    a: &mut dyn FnMut() -> eyre::Result<()>,
    b: &mut dyn FnMut() -> eyre::Result<()>,
) -> eyre::Result<(Stat, Stat, Delta, usize)> {
    let mut drained = 0usize;
    let mut once = |f: &mut dyn FnMut() -> eyre::Result<()>| -> eyre::Result<f64> {
        s.synchronize()?;
        let (e0, e1) = (Event::new()?, Event::new()?);
        prefill(s)?;
        e0.record(s)?;
        for _ in 0..reps {
            f()?;
        }
        drained += usize::from(e0.query()?);
        e1.record(s)?;
        e1.synchronize()?;
        Ok(Event::elapsed_ms(&e0, &e1)? as f64 * 1e3 / reps as f64)
    };
    let mut ta = Vec::new();
    let mut tb = Vec::new();
    a()?;
    b()?;
    for i in 0..pairs {
        if i % 2 == 0 {
            ta.push(once(&mut *a)?);
            tb.push(once(&mut *b)?);
        } else {
            tb.push(once(&mut *b)?);
            ta.push(once(&mut *a)?);
        }
    }
    Ok((Stat::of(&ta), Stat::of(&tb), Delta::of(&ta, &tb), drained))
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
        Ok(Self { split: z(bmax * m)?, mix: z(bmax * m)?, cnt, inv: z(bmax)?, carry: up(id, carry0)?, cur: z(bmax * ne)?, norm: z(bmax * ne)? })
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
    for _ in 0..3 {
        for (vi, variant) in variants.iter().enumerate() {
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
            host_v[vi].push(t.elapsed().as_secs_f64() * 1e6 / n as f64);
            // The spin still running when the host finished = the GPU ran the loop back to back.
            let drained = e0.query()?;
            bar(!drained, format!("1:{variant}:drained"));
            e1.record(&s)?;
            e1.synchronize()?;
            gpu_v[vi].push(Event::elapsed_ms(&e0, &e1)? as f64 * 1e3 / n as f64);
        }
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
    println!("== 2. 80 launches round-robin over K executables behind a 3 ms spin (GPU us/launch)");
    let (rows, k) = (1280usize, 5120usize);
    let w = q8_weight(dev.id, rows, k, 1)?;
    let mut xq = up(dev.id, &lcg_bytes(2, 8 * k).iter().map(|&b| b as i8).collect::<Vec<_>>())?;
    let mut xs = up(dev.id, &vec![0.01f32; 8 * (k / 32)])?;
    let mut out = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
    // (host us per launch, GPU us per launch, queue drained before the host finished)
    let run = |execs: &[GraphExec], kk: usize, spin: f64| -> eyre::Result<(f64, f64, bool)> {
        s.synchronize()?;
        let (e0, e1) = (Event::new()?, Event::new()?);
        e.q8.slack_probe_spin(&s, spin_ms(spin))?;
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
        let execs: Vec<GraphExec> = (0..80).map(|_| capture(&s, &mut body).map(|c| c.1)).collect::<eyre::Result<_>>()?;
        let mut any_drained = false;
        let mut line = format!("  {nodes} node(s), diagnostics (median of 3):");
        for kk in [1usize, 2, 4] {
            let mut g = Vec::new();
            for _ in 0..3 {
                let (_, gpu, drained) = run(&execs, kk, 3.0)?;
                any_drained |= drained;
                g.push(gpu);
            }
            line += &format!("  K={kk} {}", Stat::of(&g));
        }
        println!("{line}");
        // K = 8 vs K = 80, 21 alternating pairs.
        let (mut g8, mut g80, mut h8, mut h80) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..21 {
            for kk in if i % 2 == 0 { [8usize, 80] } else { [80, 8] } {
                let (h, g, drained) = run(&execs, kk, 3.0)?;
                any_drained |= drained;
                if kk == 8 {
                    g8.push(g);
                    h8.push(h);
                } else {
                    g80.push(g);
                    h80.push(h);
                }
            }
        }
        let (dg, dh) = (Delta::of(&g80, &g8), Delta::of(&h80, &h8));
        println!("  {nodes} node(s): K=8 {} vs K=80 {}; GPU K=8 - K=80 {dg}", Stat::of(&g8), Stat::of(&g80));
        println!("  {nodes} node(s): host per launch K=8 {} vs K=80 {}; host K=8 - K=80 {dh}", Stat::of(&h8), Stat::of(&h80));
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
        let ind = Ind::new(slot.raw() as u64).with(0, SL[0]).with(1, SL[1]).with(2, SL[2]).with(3, SL[3]).with_canary(7);
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
            (outs.raw() as u64 + (i * rows_c * 4) as u64) ^ [&w0, &w1][i % 2].raw() as u64 ^ xq_c.raw() as u64 ^ xs_c.raw() as u64
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
    let prefill = |s: &Stream| e.q8.slack_probe_spin(s, spin_ms(3.0));
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
        let (sd_, si_, dt, drained) = time_ab(
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
            "  b={b}: direct {sd_}, _ind {si_} ({:+.1}%); _ind - direct {dt}; differing outputs {diff}{}",
            (si_.med / sd_.med - 1.0) * 100.0,
            if drained > 0 { format!("; {drained} runs DRAINED (host-bound), FAILS") } else { String::new() }
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
            .flat_map(|c| ((((c[0] >> 7) as u16) << 15) | ((10 + (c[0] as u16 % 5)) << 10) | (((c[0] as u16) << 8 | c[1] as u16) & 0x3ff)).to_le_bytes())
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
            si.split.raw() as u64, si.mix.raw() as u64, si.cnt.raw() as u64, si.inv.raw() as u64,
            mw.raw() as u64, mx.raw() as u64, msc.raw() as u64, mbase.raw() as u64,
            si.carry.raw() as u64, mx.raw() as u64, si.cur.raw() as u64, si.norm.raw() as u64, mnw.raw() as u64,
        ];
        ent.p[..13].copy_from_slice(&ptrs);
        ctxk.store(&s, &ent, &mut slot)?;
        let ind = (0..13).fold(Ind::new(slot.raw() as u64), |ind, i| ind.with(i, i));
        // `set` = the buffers the call is handed: the direct kernel writes them, the twin only
        // size-checks them (its operands come from the slot, pointing at `si`).
        let go = |set: &mut MhcSet, ind: Option<Ind>, b: u32| -> eyre::Result<()> {
            let mix = FastMix {
                weight: &mw, x: &mx, scale: &msc, base: &mbase, mode: MIX_PRE_SCALED,
                split_out: &mut set.split, mix_out: &mut set.mix, counters: &mut set.cnt, inv_rows: &mut set.inv,
            };
            let col = FastCollapse { x: &mx, cur_out: &mut set.cur, norm_out: &mut set.norm, norm_w: &mnw };
            match ind {
                None => arena.launch_fast(&s, Some(mix), Some(col), &mut set.carry, true, HC_DIM, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b),
                Some(ind) => arena.launch_fast_ind(&s, ind, Some(mix), Some(col), &mut set.carry, true, HC_DIM, RMS_EPS, SINKHORN_ITERS, SINKHORN_EPS, b),
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
            let diff: usize = ref_out.iter().zip(&got).map(|(a, c)| a.iter().zip(c).filter(|(x, y)| x != y).count()).sum();
            let mut decoy = MhcSet::new(dev.id, bmax, &carry0)?;
            let (sd_, si_, dt, drained) =
                time_ab(&s, 200, 21, &prefill, &mut || go(&mut decoy, None, b), &mut || go(&mut sd, Some(ind), b))?;
            exact &= diff == 0;
            bar(diff == 0, format!("4b:b{b}:bitexact"));
            bar(drained == 0, format!("4b:b{b}:drained"));
            mhc_d.push(dt);
            println!(
                "  b={b}: direct {sd_}, _ind {si_} ({:+.1}%); _ind - direct {dt}; differing outputs {diff} (split, mix, inv, carry, cur, norm){}",
                (si_.med / sd_.med - 1.0) * 100.0,
                if drained > 0 { format!("; {drained} runs DRAINED (host-bound), FAILS") } else { String::new() }
            );
        }
    }

    // ---- 5. one captured _ind graph, two context entries; node count ---------------------
    println!("== 5. one _ind graph replayed with context A, then B; nodes per captured launch");
    let w_b = q8_weight(dev.id, rows, k, 5)?;
    let b = 4u32;
    let ind = Ind::new(slot.raw() as u64).with(0, 0).with(1, 1).with(2, 2).with(3, 3);
    let (_, exec) = capture(&s, &mut |s| e.q8.matvec_bpack_ind(s, ind, &mut out_d, &w, &xq, &xs, rows as u32, k as u32, b))?;
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

    // ---- the 2.0 per-step budget ----------------------------------------------------------
    // Everything the design adds per step against 1% of `ms.step` p50. The worst case is 2 rows /
    // 2 lanes (~60 ms live, 2026-10-04; 3 lanes need >= 6 rows): write and launch counts grow with
    // lanes x layers, the step with rows. GK_STEP_MS overrides. Counts (review of 0122a52):
    // 1 write per lane-layer (presubmit is off in production; 2 is a what-if line); 8 stage graphs
    // per lane-layer, 4 of them 1-node (mhc_pre_attn, mhc_pre_ffn, router, mix_ffn_late) charged at
    // the 1-node relaunch delta, 4 at the 8-node one; per lane-layer 8 gemv-like twins (q_b-shape
    // delta: conservative), 3 mhc twins, 7 small twins (a fixed prologue cost: charged at the larger
    // of the gemv / mhc deltas). Medians are charged when positive; the upper-bound total uses each
    // delta's ~96.7% upper bound. GPU + host is an upper bound (only one is the pole at a time).
    let step_ms: f64 = std::env::var("GK_STEP_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(60.0);
    let lane_layers = 2.0 * 40.0;
    let writes = lane_layers;
    let (gemv_per, mhc_per, small_per) = (8.0, 3.0, 7.0);
    let worst = |v: &[Delta], ub: bool| v.iter().map(|d| if ub { d.charge_ub() } else { d.charge() }).fold(0.0, f64::max);
    // (GPU us, host us) per step; `ub` = upper bounds.
    let cost = |ub: bool| -> (f64, f64) {
        let mut gpu = writes * write_cost.0.max(0.0);
        let mut host = writes * write_cost.1.max(0.0);
        for (_, dg, dh) in &relaunch {
            let launches = 4.0 * lane_layers; // 4 stage graphs per lane-layer of each size class
            gpu += launches * if ub { dg.charge_ub() } else { dg.charge() };
            host += launches * if ub { dh.charge_ub() } else { dh.charge() };
        }
        let (g, m) = (worst(&gemv_d, ub), worst(&mhc_d, ub));
        gpu += lane_layers * (gemv_per * g + mhc_per * m + small_per * g.max(m));
        (gpu, host)
    };
    let budget = 0.01 * step_ms * 1e3;
    let pct = |us: f64| us / (step_ms * 1e3) * 100.0;
    let (g_med, h_med) = cost(false);
    let (g_ub, h_ub) = cost(true);
    println!("== per-step budget at ms.step {step_ms:.0} ms (1% = {budget:.0} us; 2 lanes x 40 layers):");
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
    println!(
        "  twins            {lane_layers:.0} x ({gemv_per:.0} gemv x {:.2} + {mhc_per:.0} mhc x {:.2} + {small_per:.0} small x {:.2}) us; ub gemv {:.2} mhc {:.2}",
        worst(&gemv_d, false),
        worst(&mhc_d, false),
        worst(&gemv_d, false).max(worst(&mhc_d, false)),
        worst(&gemv_d, true),
        worst(&mhc_d, true)
    );
    let (med, ub) = (g_med + h_med, g_ub + h_ub);
    println!(
        "  total (medians)  {med:6.0} us = {:.2}%  (GPU-only {:.2}%, host-only {:.2}%)",
        pct(med),
        pct(g_med),
        pct(h_med)
    );
    println!("  total (upper bd) {ub:6.0} us = {:.2}%  (GPU-only {:.2}%, host-only {:.2}%)", pct(ub), pct(g_ub), pct(h_ub));
    // GO: the medians and the upper bounds fit. MARGINAL (the live A/B's <= 1% ms.step bar
    // decides): the medians fit but the upper bounds do not, or the GPU + host sum does not fit
    // while each timeline alone does. NO-GO otherwise.
    let budget_verdict = if med <= budget && ub <= budget {
        "GO"
    } else if med <= budget || (g_med <= budget && h_med <= budget) {
        "MARGINAL"
    } else {
        "NO-GO"
    };
    println!("  budget: {budget_verdict}");
    if !nogo.is_empty() {
        println!("STEP0: NO-GO ({})", nogo.join(", "));
    } else if budget_verdict == "GO" {
        println!("STEP0: GO");
    } else {
        println!("STEP0: {budget_verdict} (budget: medians {:.2}%, upper bounds {:.2}%)", pct(med), pct(ub));
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
