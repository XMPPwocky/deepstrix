//! Dual-bus (dGPU GDDR6 ‖ iGPU LPDDR5X) split-matvec micro-bench.
//!
//! Settles ONE question for decode direction B1: at B=1, can splitting a
//! bandwidth-bound Q8_0 projection matvec across BOTH memory buses in
//! parallel beat running it dGPU-only, once you pay the cross-device
//! peer-copy latency?
//!
//! Representative projection = output_proj (`attn_output_b`, Q8_0):
//!   M = N_EMBD = 4096 rows, K = OUT_LOW = 8192  →  weight ≈ 34 MiB.
//!
//! Baseline:  full weight on dGPU, matvec all M rows on dGPU.
//! Split(f):  iGPU holds the last f·M rows (its own weight copy, pre-staged);
//!            dGPU holds the first (1−f)·M rows. Per token:
//!              1. push xq + xscale dGPU→iGPU (peer, on dGPU xfer stream),
//!              2. dGPU matvec (1−f) rows  ‖  iGPU matvec f rows (separate
//!                 streams/devices, event-ordered on the push),
//!              3. pull iGPU partial iGPU→dGPU (peer, on iGPU xfer stream),
//!              4. concat on dGPU (disjoint rows → no reduce needed).
//!            Wall = host critical path incl. both peer copies + concurrency.
//!
//! Quantize is done ONCE outside the timed loop (same cost either way; in
//! production the activation is already on the dGPU), so the delta measured
//! is purely matvec vs (split-matvec + peer round-trip).
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test bench_dualbus_matvec \
//!     bench_dualbus_matvec -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use std::time::Instant;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{N_EMBD, OUT_LOW};
use v4flash_kernels::q8_0::{Q8_0Matvec, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};

fn pick(arch_prefix: &str) -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(arch_prefix) {
            return Ok(d);
        }
    }
    Err(eyre!("no device with arch {arch_prefix}"))
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

#[test]
#[ignore]
fn bench_dualbus_matvec() -> eyre::Result<()> {
    install_panic_handler()?;

    let iters: usize = std::env::var("BENCH_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(400);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(50);

    // output_proj dims.
    let m: u32 = N_EMBD; // 4096 rows
    let k: u32 = OUT_LOW; // 8192 contraction
    let blocks = k / Q8_0_BLOCK_ELEMS; // 256
    let bpe = (blocks as usize) * (Q8_0_BLOCK_BYTES as usize); // bytes per row
    let w_full_bytes = (m as usize) * bpe;
    eprintln!(
        "output_proj matvec: M={m} K={k} blocks={blocks} row_bytes={bpe} \
         weight={:.2} MiB  iters={iters} warmup={warmup}",
        w_full_bytes as f64 / (1024.0 * 1024.0)
    );

    let dgpu = pick("gfx1201")?;
    let igpu = pick("gfx1151")?;
    eprintln!(
        "dGPU id={} ({}), iGPU id={} ({})",
        dgpu.id, dgpu.properties()?.gcn_arch_name,
        igpu.id, igpu.properties()?.gcn_arch_name
    );

    // Peer access both directions.
    dgpu.set_current()?;
    if !dgpu.can_access_peer(igpu)? { return Err(eyre!("dGPU cannot peer iGPU")); }
    let _ = dgpu.enable_peer_access(igpu);
    igpu.set_current()?;
    if !igpu.can_access_peer(dgpu)? { return Err(eyre!("iGPU cannot peer dGPU")); }
    let _ = igpu.enable_peer_access(dgpu);

    let q8_d = { dgpu.set_current()?; Q8_0Matvec::for_arch(&dgpu.properties()?.gcn_arch_name)? };
    let q8_i = { igpu.set_current()?; Q8_0Matvec::for_arch(&igpu.properties()?.gcn_arch_name)? };

    // ---- dGPU-resident buffers ----
    dgpu.set_current()?;
    let dgpu_compute = Stream::new(dgpu.id)?;
    let dgpu_xfer = Stream::new(dgpu.id)?;

    let mut w_full: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, w_full_bytes)?;
    w_full.fill_zero()?;

    // Activation on dGPU, quantized once.
    let x_host: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect();
    let mut x: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, k as usize)?;
    x.copy_from_host(&x_host)?;
    let mut xq: DeviceBuffer<i8> = DeviceBuffer::new(dgpu.id, k as usize)?;
    let mut xscale: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, blocks as usize)?;
    q8_d.quantize_input(&dgpu_compute, &mut xq, &mut xscale, &x, k)?;
    dgpu_compute.synchronize()?;

    let mut out_full: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, m as usize)?;
    out_full.fill_zero()?;

    // ================= BASELINE: dGPU-only full matvec =================
    let bench_dgpu_only = |q8: &Q8_0Matvec,
                           s: &Stream,
                           out: &mut DeviceBuffer<f32>,
                           w: &DeviceBuffer<u8>|
     -> eyre::Result<(f64, f64)> {
        for _ in 0..warmup { q8.matvec(s, out, w, &xq, &xscale, m, k)?; }
        s.synchronize()?;
        // Event-timed pure GPU compute.
        let mut ev_ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let a = Event::new()?; let b = Event::new()?;
            a.record(s)?;
            q8.matvec(s, out, w, &xq, &xscale, m, k)?;
            b.record(s)?;
            s.synchronize()?;
            ev_ms.push(Event::elapsed_ms(&a, &b)? as f64);
        }
        // Host-wall (issue + sync), comparable to the split's host-wall.
        let mut host_ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            q8.matvec(s, out, w, &xq, &xscale, m, k)?;
            s.synchronize()?;
            host_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok((median(&mut ev_ms), median(&mut host_ms)))
    };
    let (base_ev, base_host) = bench_dgpu_only(&q8_d, &dgpu_compute, &mut out_full, &w_full)?;
    let base_bw = w_full_bytes as f64 / 1e9 / (base_ev / 1e3);
    eprintln!(
        "\n[BASELINE dGPU-only]  event={:.4} ms ({:.0} GB/s)   host-wall={:.4} ms",
        base_ev, base_bw, base_host
    );

    // ================= PEER ROUND-TRIP LATENCY + BW SWEEP =================
    // Activation-sized round trip (what the split actually pays): push xq
    // (k bytes) + xscale (blocks*4 bytes) dGPU→iGPU, pull an f32 partial back.
    igpu.set_current()?;
    let igpu_compute = Stream::new(igpu.id)?;
    let igpu_xfer = Stream::new(igpu.id)?;
    let mut xq_i: DeviceBuffer<i8> = DeviceBuffer::new(igpu.id, k as usize)?;
    let mut xscale_i: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, blocks as usize)?;
    xq_i.fill_zero()?; xscale_i.fill_zero()?;

    // Round-trip: push activation (dGPU→iGPU) then pull it back (iGPU→dGPU),
    // each synchronized — the true serialized latency of the two hops.
    dgpu.set_current()?;
    let mut back: DeviceBuffer<i8> = DeviceBuffer::new(dgpu.id, k as usize)?;
    let act_bytes = k as usize + blocks as usize * 4;
    {
        for _ in 0..warmup {
            xq.copy_to_peer_async(&mut xq_i, &dgpu_xfer)?; dgpu_xfer.synchronize()?;
            xq_i.copy_to_peer_async(&mut back, &igpu_xfer)?; igpu_xfer.synchronize()?;
        }
        let mut push_ms: Vec<f64> = Vec::with_capacity(iters);
        let mut pull_ms: Vec<f64> = Vec::with_capacity(iters);
        let mut rt_ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            xq.copy_to_peer_async(&mut xq_i, &dgpu_xfer)?;
            xscale.copy_to_peer_async(&mut xscale_i, &dgpu_xfer)?;
            dgpu_xfer.synchronize()?;
            let t1 = Instant::now();
            xq_i.copy_to_peer_async(&mut back, &igpu_xfer)?;
            igpu_xfer.synchronize()?;
            let t2 = Instant::now();
            push_ms.push((t1 - t0).as_secs_f64() * 1e3);
            pull_ms.push((t2 - t1).as_secs_f64() * 1e3);
            rt_ms.push((t2 - t0).as_secs_f64() * 1e3);
        }
        let p = median(&mut push_ms); let pl = median(&mut pull_ms); let rt = median(&mut rt_ms);
        eprintln!(
            "\n[PEER round-trip @ activation size]  push({} B)={:.4} ms  pull({} B)={:.4} ms  \
             ROUND-TRIP={:.4} ms",
            act_bytes, p, k, pl, rt
        );
    }

    // Per-bus effective BW sweep (dGPU→iGPU) across sizes.
    eprintln!("\n[per-bus effective BW: dGPU→iGPU peer copy]");
    for &sz in &[4096usize, 65536, 1 << 20, 8 << 20, 32 << 20] {
        dgpu.set_current()?;
        let src: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, sz)?;
        igpu.set_current()?;
        let mut dst: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, sz)?;
        dgpu.set_current()?;
        for _ in 0..warmup.max(20) {
            src.copy_to_peer_async(&mut dst, &dgpu_xfer)?; dgpu_xfer.synchronize()?;
        }
        let mut ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            src.copy_to_peer_async(&mut dst, &dgpu_xfer)?;
            dgpu_xfer.synchronize()?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let md = median(&mut ms);
        eprintln!(
            "  {:>9} B: {:.4} ms  ({:.1} GB/s)",
            sz, md, sz as f64 / 1e9 / (md / 1e3)
        );
    }

    // ================= SPLIT SWEEP =================
    eprintln!("\n[SPLIT: dGPU (1−f) rows ‖ iGPU f rows]");
    eprintln!("  f     dgpu_rows  igpu_rows  split-wall(ms)  vs-base   dgpu-mv(ms)  igpu-mv(ms)");
    let mut best: Option<(f64, f64)> = None; // (f, wall)
    for &f in &[0.20f64, 0.30, 0.40, 0.50] {
        let igpu_rows = ((f * m as f64).round() as u32).max(8);
        let dgpu_rows = m - igpu_rows;
        let w_dgpu_bytes = dgpu_rows as usize * bpe;
        let w_igpu_bytes = igpu_rows as usize * bpe;

        // dGPU holds first dgpu_rows rows.
        dgpu.set_current()?;
        let mut w_dgpu: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, w_dgpu_bytes)?;
        w_dgpu.fill_zero()?;
        let mut out_d: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, dgpu_rows as usize)?;
        out_d.fill_zero()?;
        // Destination for the iGPU partial, on the dGPU.
        let mut out_i_on_d: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, igpu_rows as usize)?;
        out_i_on_d.fill_zero()?;

        // iGPU holds last igpu_rows rows.
        igpu.set_current()?;
        let mut w_igpu: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, w_igpu_bytes)?;
        w_igpu.fill_zero()?;
        let mut out_i: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, igpu_rows as usize)?;
        out_i.fill_zero()?;

        // Isolated per-device matvec times (event) for the crossover math.
        let mut dmv: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            dgpu.set_current()?;
            let a = Event::new()?; let b = Event::new()?;
            a.record(&dgpu_compute)?;
            q8_d.matvec(&dgpu_compute, &mut out_d, &w_dgpu, &xq, &xscale, dgpu_rows, k)?;
            b.record(&dgpu_compute)?;
            dgpu_compute.synchronize()?;
            dmv.push(Event::elapsed_ms(&a, &b)? as f64);
        }
        let mut imv: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            igpu.set_current()?;
            let a = Event::new()?; let b = Event::new()?;
            a.record(&igpu_compute)?;
            q8_i.matvec(&igpu_compute, &mut out_i, &w_igpu, &xq_i, &xscale_i, igpu_rows, k)?;
            b.record(&igpu_compute)?;
            igpu_compute.synchronize()?;
            imv.push(Event::elapsed_ms(&a, &b)? as f64);
        }
        let dmv_md = median(&mut dmv);
        let imv_md = median(&mut imv);

        // Full split critical path, host-timed. Create each event while its
        // recording device is current.
        dgpu.set_current()?;
        let ev_push = Event::new_no_timing()?;
        igpu.set_current()?;
        let ev_igpu = Event::new_no_timing()?;
        let run_split = |ev_push: &Event, ev_igpu: &Event,
                         w_dgpu: &DeviceBuffer<u8>, out_d: &mut DeviceBuffer<f32>,
                         w_igpu: &DeviceBuffer<u8>, out_i: &mut DeviceBuffer<f32>,
                         out_i_on_d: &mut DeviceBuffer<f32>,
                         xq_i: &mut DeviceBuffer<i8>, xscale_i: &mut DeviceBuffer<f32>|
         -> eyre::Result<()> {
            // 1. push activation dGPU→iGPU on dGPU xfer stream; signal.
            dgpu.set_current()?;
            xq.copy_to_peer_async(xq_i, &dgpu_xfer)?;
            xscale.copy_to_peer_async(xscale_i, &dgpu_xfer)?;
            ev_push.record(&dgpu_xfer)?;
            // 2a. dGPU matvec (independent of the push) — concurrent.
            q8_d.matvec(&dgpu_compute, out_d, w_dgpu, &xq, &xscale, dgpu_rows, k)?;
            // 2b. iGPU matvec — waits on the push, then computes; signal done.
            igpu.set_current()?;
            igpu_compute.wait_event(ev_push)?;
            q8_i.matvec(&igpu_compute, out_i, w_igpu, xq_i, xscale_i, igpu_rows, k)?;
            ev_igpu.record(&igpu_compute)?;
            // 3. pull iGPU partial iGPU→dGPU on iGPU xfer stream (src device).
            igpu_xfer.wait_event(ev_igpu)?;
            out_i.copy_to_peer_async(out_i_on_d, &igpu_xfer)?;
            Ok(())
        };

        for _ in 0..warmup {
            run_split(&ev_push, &ev_igpu, &w_dgpu, &mut out_d, &w_igpu,
                      &mut out_i, &mut out_i_on_d, &mut xq_i, &mut xscale_i)?;
            dgpu_compute.synchronize()?; dgpu_xfer.synchronize()?;
            igpu_compute.synchronize()?; igpu_xfer.synchronize()?;
        }
        let mut wall: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            run_split(&ev_push, &ev_igpu, &w_dgpu, &mut out_d, &w_igpu,
                      &mut out_i, &mut out_i_on_d, &mut xq_i, &mut xscale_i)?;
            // Critical path ends when both dGPU matvec AND the pulled iGPU
            // partial have landed on the dGPU.
            dgpu_compute.synchronize()?;
            igpu_xfer.synchronize()?;
            // Drain the remaining streams so the next iter starts clean.
            dgpu_xfer.synchronize()?;
            igpu_compute.synchronize()?;
            wall.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let wmd = median(&mut wall);
        let speedup = base_host / wmd;
        eprintln!(
            "  {:.2}   {:>8}   {:>8}   {:>12.4}   {:.3}x   {:>10.4}   {:>10.4}",
            f, dgpu_rows, igpu_rows, wmd, speedup, dmv_md, imv_md
        );
        if best.map_or(true, |(_, w)| wmd < w) { best = Some((f, wmd)); }
    }

    let (bf, bw) = best.unwrap();
    eprintln!(
        "\n==== VERDICT ====\n\
         dGPU-only host-wall : {:.4} ms\n\
         best split          : {:.4} ms  @ f={:.2}\n\
         split beats dGPU?   : {}  ({:+.1}%)",
        base_host, bw, bf,
        if bw < base_host { "YES" } else { "NO" },
        (base_host - bw) / base_host * 100.0
    );
    Ok(())
}
