//! Step 0 of docs/v41/GRAPH_KEYS_DESIGN.md (2.0): the measurements that gate building
//! (stage, rows)-keyed arena graphs. dGPU only; the hub must be DOWN (it holds the dGPU).
//!
//!  1. The context write, AMORTIZED (queue pre-filled behind a spin): N x [nop; write; nop]
//!     for write = none / H2D from pinned / `arena_ctx_store` by value /
//!     `hipStreamWriteValue32`; host enqueue time and GPU time per write.
//!  2. 80 `hipGraphLaunch` of ONE executable vs 80 distinct executables behind a 5 ms and a
//!     50 ms spin. BLOCKS = the host enqueue time grows with the spin; SERIALIZES = GPU time
//!     per launch of the one executable > 10% above the distinct executables'.
//!  3. COHERENCE: `arena_ctx_store` then a captured `_ind` graph reading the slot, alternating
//!     entries, 10k rounds; every output and every canary record (the seq each launch read)
//!     checked.
//!  4. `_ind` twins vs direct, bit-exact and timed with the canary compiled in (null):
//!     `q8_0_gemv_bpack_tB{1,4,8}` on the q_b shape and `mhc_fast_batched` (13 operands,
//!     pre_attn case) at b = 1, 4, 8. Bar: within max(2%, 0.3 us). (VGPR / SGPR counts come
//!     from the code objects, offline.)
//!  5. The mechanism: ONE captured `_ind` graph replayed after storing entry A, then B,
//!     reproduces the direct kernel on A, then on B; and a capture of N launches has N nodes
//!     (design 2.5's vetted-count check assumes one node per launch).
//!
//! HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release \
//!   --features v41 -p v4flash-kernels --test bench_graph_keys -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use std::time::Instant;
use v4flash_hip::{launch_kernel, sys, Device, DeviceBuffer, Event, GraphExec, LaunchConfig, Module, PinnedBuffer, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_HC, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS};
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

/// Median over `rounds` alternating rounds of `reps` launches each: (us per launch of a, of b).
fn time_ab(
    s: &Stream,
    reps: usize,
    rounds: usize,
    a: &mut dyn FnMut() -> eyre::Result<()>,
    b: &mut dyn FnMut() -> eyre::Result<()>,
) -> eyre::Result<(f64, f64)> {
    let once = |f: &mut dyn FnMut() -> eyre::Result<()>| -> eyre::Result<f64> {
        s.synchronize()?;
        let (e0, e1) = (Event::new()?, Event::new()?);
        e0.record(s)?;
        for _ in 0..reps {
            f()?;
        }
        e1.record(s)?;
        e1.synchronize()?;
        Ok(Event::elapsed_ms(&e0, &e1)? as f64 * 1e3 / reps as f64)
    };
    let mut ta = Vec::new();
    let mut tb = Vec::new();
    a()?;
    b()?;
    for _ in 0..rounds {
        ta.push(once(&mut *a)?);
        tb.push(once(&mut *b)?);
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    Ok((med(&mut ta), med(&mut tb)))
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
    let mut exact = true; // every bit-exactness / coherence / node-count check
    let mut gate = true; // the 2.0 go / no-go bars this bench can decide alone

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
    println!("== 1. context write between two launches, queue pre-filled behind a 300 ms spin (per write; 'none' = bare launches)");
    let n = 3000usize;
    let mut base = (0.0, 0.0);
    for variant in ["none", "h2d", "ctx_store", "write_value32"] {
        s.synchronize()?;
        let (e0, e1) = (Event::new()?, Event::new()?);
        e.q8.slack_probe_spin(&s, spin_ms(300.0))?;
        e0.record(&s)?;
        let t = Instant::now();
        for i in 0..n {
            ctxk.nop(&s, &mut sink)?;
            match variant {
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
        }
        let host_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        // The spin still running when the host finished = the GPU ran the loop back to back.
        let drained = e0.query()?;
        e1.record(&s)?;
        e1.synchronize()?;
        let gpu_us = Event::elapsed_ms(&e0, &e1)? as f64 * 1e3 / n as f64;
        if variant == "none" {
            base = (host_us, gpu_us);
        }
        println!(
            "  {variant:>14}: host {host_us:7.2} us/iter (+{:6.2}), GPU {gpu_us:7.2} us/iter (+{:6.2}){}; at 160 writes/step: +{:.0} us host, +{:.0} us GPU",
            host_us - base.0,
            gpu_us - base.1,
            if drained { "  [queue DRAINED before the host finished: GPU figure is host-bound]" } else { "" },
            160.0 * (host_us - base.0),
            160.0 * (gpu_us - base.1),
        );
    }

    // ---- 2. one executable re-launched 80x vs 80 executables ---------------------------
    println!("== 2. 80 graph launches queued behind a 5 ms / 50 ms spin");
    let (rows, k) = (1280usize, 5120usize);
    let w = q8_weight(dev.id, rows, k, 1)?;
    let mut xq = up(dev.id, &lcg_bytes(2, 8 * k).iter().map(|&b| b as i8).collect::<Vec<_>>())?;
    let mut xs = up(dev.id, &vec![0.01f32; 8 * (k / 32)])?;
    let mut out = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
    let mut gemv = |s: &Stream| e.q8.matvec_bpack(s, &mut out, &w, &xq, &xs, rows as u32, k as u32, 1);
    let one = capture(&s, &mut gemv)?.1;
    let many: Vec<GraphExec> = (0..80).map(|_| capture(&s, &mut gemv).map(|c| c.1)).collect::<eyre::Result<_>>()?;
    let mut res = std::collections::HashMap::new();
    for (label, reuse) in [("same exec x80", true), ("80 distinct execs", false)] {
        for spin in [5.0f64, 50.0] {
            s.synchronize()?;
            let (e0, e1) = (Event::new()?, Event::new()?);
            e.q8.slack_probe_spin(&s, spin_ms(spin))?;
            e0.record(&s)?;
            let t = Instant::now();
            for i in 0..80 {
                if reuse { one.launch(&s)? } else { many[i].launch(&s)? }
            }
            let host_ms = t.elapsed().as_secs_f64() * 1e3;
            e1.record(&s)?;
            e1.synchronize()?;
            let gpu_us = Event::elapsed_ms(&e0, &e1)? as f64 * 1e3 / 80.0;
            res.insert((reuse, spin as u32), (host_ms, gpu_us));
            println!("  {label:>18}, {spin:>2} ms spin: host {host_ms:7.3} ms for 80 launches, GPU {gpu_us:7.2} us/launch");
        }
    }
    for (label, reuse) in [("same exec", true), ("distinct execs", false)] {
        let grow = res[&(reuse, 50)].0 - res[&(reuse, 5)].0;
        let blocks = grow > 22.5; // half the 45 ms the spin grew by
        gate &= !blocks;
        println!("  {label}: host time grew {grow:+.3} ms with the spin -> {}", if blocks { "BLOCKS" } else { "does not block" });
    }
    let (same, distinct) = (res[&(true, 50)].1, res[&(false, 50)].1);
    let serializes = same > 1.10 * distinct;
    gate &= !serializes;
    println!("  GPU per launch same {same:.2} vs distinct {distinct:.2} us -> {}", if serializes { "SERIALIZES" } else { "does not serialize" });

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
        let ind = Ind::new(slot.raw() as u64).with(0, 0).with(1, 1).with(2, 2).with(3, 3).with_canary(7);
        let (_, exec) = capture(&s, &mut |s| {
            e.q8.matvec_bpack_ind(s, ind, &mut tmp, &w0, &xq_c, &xs_c, rows_c as u32, k_c as u32, 1)
        })?;
        e.q8.slack_probe_spin(&s, spin_ms(20.0))?;
        for i in 0..rounds {
            let mut ent = ArenaCtx::default();
            ent.p[0] = outs.raw() as u64 + (i * rows_c * 4) as u64;
            ent.p[1] = [&w0, &w1][i % 2].raw() as u64;
            ent.p[2] = xq_c.raw() as u64;
            ent.p[3] = xs_c.raw() as u64;
            ent.seq = i as u64;
            ent.log = log.buf.raw() as u64;
            ctxk.store(&s, &ent, &mut slot)?;
            exec.launch(&s)?;
        }
        s.synchronize()?;
        let host = down(&outs, rounds * rows_c)?;
        let bad = (0..rounds).filter(|&i| host[i * rows_c..(i + 1) * rows_c] != refs[i % 2][..]).count();
        let (cursor, recs) = log.read()?;
        let bad_seq = recs.iter().enumerate().filter(|&(i, &(seq, tag))| seq != i as u64 || tag != 7).count();
        exact &= bad == 0 && cursor as usize == rounds && bad_seq == 0;
        println!("  {rounds} rounds: {bad} outputs from a stale / wrong entry; canary {cursor} records, {bad_seq} out of order");
    }

    // ---- 4a. gemv `_ind` twin vs direct, b = 1, 4, 8 -------------------------------------
    println!("== 4a. q8_0_gemv_bpack_tB{{b}}_ind vs direct (q_b shape 32768 x 1280), canary compiled in, null");
    let (rows, k) = (32768usize, 1280usize);
    let w = q8_weight(dev.id, rows, k, 3)?;
    xq = up(dev.id, &lcg_bytes(4, 8 * k).iter().map(|&b| b as i8).collect::<Vec<_>>())?;
    xs = up(dev.id, &(0..8 * (k / 32)).map(|i| 0.002 + (i % 7) as f32 * 0.001).collect::<Vec<_>>())?;
    let mut out_d = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
    let out_i = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
    let twin_ok = |td: f64, ti: f64| ti - td <= f64::max(0.02 * td, 0.3);
    for b in [1u32, 4, 8] {
        let mut ent = ArenaCtx::default();
        ent.p[0] = out_i.raw() as u64;
        ent.p[1] = w.raw() as u64;
        ent.p[2] = xq.raw() as u64;
        ent.p[3] = xs.raw() as u64;
        ctxk.store(&s, &ent, &mut slot)?;
        let ind = Ind::new(slot.raw() as u64).with(0, 0).with(1, 1).with(2, 2).with(3, 3);
        let mut out_x = DeviceBuffer::<f32>::new(dev.id, 8 * rows)?;
        // The twin's real operands come from the slot (out_i); out_x is only size-checked.
        let (td, ti) = time_ab(
            &s,
            200,
            5,
            &mut || e.q8.matvec_bpack(&s, &mut out_d, &w, &xq, &xs, rows as u32, k as u32, b),
            &mut || e.q8.matvec_bpack_ind(&s, ind, &mut out_x, &w, &xq, &xs, rows as u32, k as u32, b),
        )?;
        s.synchronize()?;
        let n = b as usize * rows;
        let diff = down(&out_d, n)?.iter().zip(down(&out_i, n)?).filter(|(a, c)| **a != *c).count();
        let pass = twin_ok(td, ti);
        exact &= diff == 0;
        gate &= pass;
        println!(
            "  b={b}: direct {td:7.2} us, _ind {ti:7.2} us ({:+.1}%, {}); differing outputs {diff}",
            (ti / td - 1.0) * 100.0,
            if pass { "within bar" } else { "OVER BAR" }
        );
    }

    // ---- 4b. mhc_fast_batched `_ind` twin vs direct, b = 1, 4, 8 -------------------------
    println!("== 4b. mhc_fast_batched_ind vs direct (pre_attn: mix pre-scaled + collapse + write_carry), 13 operands indirect");
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
        let old_mhc = match std::env::var("GK_OLD_MHC") {
            Ok(p) => Some(Module::load_data(&std::fs::read(&p)?)?),
            Err(_) => None,
        };
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
            let (td, ti) = time_ab(&s, 200, 5, &mut || go(&mut decoy, None, b), &mut || go(&mut sd, Some(ind), b))?;
            let pass = twin_ok(td, ti);
            exact &= diff == 0;
            gate &= pass;
            println!(
                "  b={b}: direct {td:7.2} us, _ind {ti:7.2} us ({:+.1}%, {}); differing outputs {diff} (split, mix, inv, carry, cur, norm)",
                (ti / td - 1.0) * 100.0,
                if pass { "within bar" } else { "OVER BAR" }
            );
            // Moving the body into a __device__ function rescheduled the DIRECT kernel (same
            // arithmetic ops, offline disassembly): GK_OLD_MHC = the pre-refactor code object
            // (hipcc flags as build.rs) times and checks the old direct kernel against the new.
            if let Some(old) = old_mhc.as_ref() {
                let f = old.get_function("mhc_fast_batched")?;
                let rms_wg = 1u32;
                let cfg = LaunchConfig { grid: (HC_MIX_DIM + 1 + rms_wg, 1, b), block: (256, 1, 1), shared_mem_bytes: 0 };
                let go_old = |set: &mut MhcSet| -> eyre::Result<()> {
                    launch_kernel!(f, cfg, &s, [
                        set.split.raw(), set.mix.raw(), set.cnt.raw(), set.inv.raw(), mw.raw(), mx.raw(), msc.raw(), mbase.raw(),
                        set.carry.raw(), mx.raw(), set.cur.raw(), set.norm.raw(), mnw.raw(),
                        HC_DIM, HC_DIM / N_HC, HC_MIX_DIM, 1u32, 1u32, MIX_PRE_SCALED, RMS_EPS, N_HC, SINKHORN_ITERS, SINKHORN_EPS
                    ])
                };
                si.reset(&carry0)?;
                go_old(&mut si)?;
                s.synchronize()?;
                let got = si.dump(b as usize)?;
                let diff: usize = ref_out.iter().zip(&got).map(|(a, c)| a.iter().zip(c).filter(|(x, y)| x != y).count()).sum();
                let mut decoy2 = MhcSet::new(dev.id, bmax, &carry0)?;
                let (tn, to) = time_ab(&s, 200, 5, &mut || go(&mut decoy, None, b), &mut || go_old(&mut decoy2))?;
                exact &= diff == 0;
                println!("        direct new {tn:7.2} us vs pre-refactor {to:7.2} us ({:+.1}%); differing outputs {diff}", (tn / to - 1.0) * 100.0);
            }
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
    println!("  {launches} captured launches -> {nodes} graph nodes");

    println!(
        "== bit-exact / coherence / nodes: {}; BLOCKS / SERIALIZES / twin bars: {} (write cost vs 1% of ms.step p50: compare section 1 by hand)",
        if exact { "PASS" } else { "FAIL" },
        if gate { "PASS" } else { "FAIL" }
    );
    if !exact {
        return Err(eyre!("graph_keys step 0: a bit-exactness, coherence or node-count check failed"));
    }
    Ok(())
}
