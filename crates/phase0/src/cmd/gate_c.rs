//! Gate C — peer access, events, peer-direct bandwidth + sync.
//!
//! This gate decides whether the ~40 us host-bounce floor we measured in
//! pingpong is a fundamental cost or whether the peer-direct +
//! `hipMemcpyPeerAsync` + event-based sync path delivers materially
//! lower latency (as the design doc estimates).
//!
//! Measurements:
//!  1. `hipDeviceCanAccessPeer` matrix for both directions.
//!  2. Enable peer access on each direction; record what error (if any).
//!  3. Bandwidth sweep via `hipMemcpyPeerAsync` at 1 MB / 16 MB / 64 MB /
//!     256 MB. Time using `hipEvent` start/end so we get GPU-attributed
//!     time, not host-side.
//!  4. Empty-kernel cross-device event RTT: launch noop on src stream,
//!     record event, wait on dst stream, launch noop on dst stream,
//!     record + wait back on src. Sync. Time host-side.
//!  5. Cache coherency: fill pattern on src via kernel, record event,
//!     wait + peer-copy on dst, sync, host-validate the bytes survived.

use std::ffi::c_void;
use std::time::Instant;

use color_eyre::eyre;
use serde::Serialize;
use v4flash_hip::{
    Device, DeviceBuffer, Event, LaunchConfig, Module, Stream,
};

use crate::results;

const FILL_GFX1201: &[u8] = include_bytes!(env!("KERNEL_FILL_PATTERN_GFX1201"));
const FILL_GFX1151: &[u8] = include_bytes!(env!("KERNEL_FILL_PATTERN_GFX1151"));

const SIZES_BYTES: &[usize] = &[
    1 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
];

#[derive(Serialize)]
pub struct GateCReport {
    pub gate: &'static str,
    pub timestamp: u64,
    pub peer_matrix: Vec<PeerCell>,
    pub bandwidth: Vec<BandwidthSample>,
    pub event_rtt: Vec<EventRttSample>,
    pub coherency: Vec<CoherencySample>,
    pub decision: Decision,
}

#[derive(Serialize)]
pub struct PeerCell {
    pub src_device: i32,
    pub dst_device: i32,
    pub can_access: bool,
    pub enable_error: Option<String>,
}

#[derive(Serialize)]
pub struct BandwidthSample {
    pub src_device: i32,
    pub dst_device: i32,
    pub size_bytes: usize,
    pub iterations: u32,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub stddev_ms: f64,
    pub mean_gb_per_s: f64,
    pub p50_gb_per_s: f64,
    pub ok: bool,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct EventRttSample {
    pub src_device: i32,
    pub dst_device: i32,
    pub variant: String,
    pub iterations: u32,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub stddev_us: f64,
}

#[derive(Serialize)]
pub struct CoherencySample {
    pub src_device: i32,
    pub dst_device: i32,
    pub ok: bool,
    pub mismatch_count: usize,
    pub first_mismatch_index: Option<usize>,
    pub first_mismatch_got: Option<u32>,
    pub first_mismatch_expected: Option<u32>,
    pub src_local_check_ok: Option<bool>,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct Decision {
    pub direct_peer_available: bool,
    pub recommendation: String,
    pub rationale: String,
}

pub fn run(iterations: u32) -> eyre::Result<()> {
    let devices = Device::all()?;
    if devices.len() < 2 {
        println!("gate-c needs at least 2 devices, found {}", devices.len());
        return Ok(());
    }
    let archs: Vec<String> = devices
        .iter()
        .map(|d| d.properties().map(|p| p.gcn_arch_name).unwrap_or_default())
        .collect();

    // --- 1 + 2. Peer access matrix
    println!("== peer access matrix ==");
    let mut peer_matrix = Vec::new();
    for src in &devices {
        for dst in &devices {
            if src.id == dst.id {
                continue;
            }
            let can = src.can_access_peer(*dst)?;
            let mut enable_error = None;
            if can {
                src.set_current()?;
                if let Err(e) = src.enable_peer_access(*dst) {
                    enable_error = Some(format!("{e:#}"));
                }
            }
            println!(
                "    {} ({}) -> {} ({}): can_access={}, enable_error={:?}",
                src.id, archs[src.id as usize], dst.id, archs[dst.id as usize], can,
                enable_error.as_deref().unwrap_or("ok"),
            );
            peer_matrix.push(PeerCell {
                src_device: src.id,
                dst_device: dst.id,
                can_access: can,
                enable_error,
            });
        }
    }

    let peer_available =
        peer_matrix.iter().any(|c| c.can_access && c.enable_error.is_none());

    // --- 3. Bandwidth sweep
    println!("\n== peer bandwidth sweep ==");
    let mut bandwidth = Vec::new();
    if peer_available {
        for cell in &peer_matrix {
            if !cell.can_access || cell.enable_error.is_some() {
                continue;
            }
            let src = Device::new(cell.src_device);
            let dst = Device::new(cell.dst_device);
            for &size in SIZES_BYTES {
                let n = size / 4;
                let timed_iter = bw_iters_for_size(size);
                let warmup = 5;
                match measure_peer_bandwidth(src, dst, n, warmup, timed_iter) {
                    Ok(bs) => {
                        println!(
                            "    {} -> {}, {:>9} B ({}x): mean {:.3} ms ({:.2} GB/s), p50 {:.3} ms ({:.2} GB/s), stddev {:.3} ms, min {:.3} / max {:.3}",
                            src.id, dst.id, size, timed_iter, bs.mean_ms, bs.mean_gb_per_s,
                            bs.p50_ms, bs.p50_gb_per_s, bs.stddev_ms, bs.min_ms, bs.max_ms,
                        );
                        bandwidth.push(BandwidthSample {
                            src_device: src.id,
                            dst_device: dst.id,
                            size_bytes: size,
                            iterations: timed_iter,
                            mean_ms: bs.mean_ms,
                            p50_ms: bs.p50_ms,
                            p99_ms: bs.p99_ms,
                            min_ms: bs.min_ms,
                            max_ms: bs.max_ms,
                            stddev_ms: bs.stddev_ms,
                            mean_gb_per_s: bs.mean_gb_per_s,
                            p50_gb_per_s: bs.p50_gb_per_s,
                            ok: true,
                            error: None,
                        });
                    }
                    Err(e) => {
                        println!(
                            "    {} -> {}, {:>9} B: FAILED ({e:#})",
                            src.id, dst.id, size
                        );
                        bandwidth.push(BandwidthSample {
                            src_device: src.id,
                            dst_device: dst.id,
                            size_bytes: size,
                            iterations: 0,
                            mean_ms: 0.0,
                            p50_ms: 0.0,
                            p99_ms: 0.0,
                            min_ms: 0.0,
                            max_ms: 0.0,
                            stddev_ms: 0.0,
                            mean_gb_per_s: 0.0,
                            p50_gb_per_s: 0.0,
                            ok: false,
                            error: Some(format!("{e:#}")),
                        });
                    }
                }
            }
        }
    }

    // --- 4. Cross-device event RTT — two variants:
    //   noop_event:   launches a noop kernel on each side around the event chain
    //                 (Gate C original; matches what most apps do).
    //   pure_event:   chains record/wait/record/wait with reused events, no kernels.
    //                 Isolates the cost of HSA signal cross-device propagation.
    println!("\n== cross-device event RTT ==");
    let mut event_rtt = Vec::new();
    if peer_available {
        for src in &devices {
            for dst in &devices {
                if src.id == dst.id {
                    continue;
                }
                let src_image = pick_fill(&archs[src.id as usize]);
                let dst_image = pick_fill(&archs[dst.id as usize]);

                // amortized_pipeline: most production-relevant. Submits a
                // batch of B event-record + wait_event pairs, syncs once,
                // divides by B. Removes the host-poll-per-iteration cost
                // that dominates RTT measurements but isn't paid in
                // steady-state kernel-to-kernel chaining.
                let pipe_warmup = 5;
                let batch_size = 200;
                let n_batches = 20;
                match measure_pipelined_sync(*src, *dst, batch_size, n_batches, pipe_warmup) {
                    Ok(s) => {
                        println!(
                            "    pipelined    {} -> {} ({}x{}): mean {:.2} us, p50 {:.2}, p99 {:.2}, stddev {:.2}",
                            src.id, dst.id, n_batches, batch_size,
                            s.mean, s.p50, s.p99, s.stddev,
                        );
                        event_rtt.push(EventRttSample {
                            src_device: src.id,
                            dst_device: dst.id,
                            variant: format!("pipelined_b{batch_size}"),
                            iterations: n_batches * batch_size,
                            mean_us: s.mean, p50_us: s.p50, p99_us: s.p99,
                            min_us: s.min, max_us: s.max, stddev_us: s.stddev,
                        });
                    }
                    Err(e) => println!("    pipelined {} -> {}: FAILED ({e:#})", src.id, dst.id),
                }

                // pure_event: no kernels, events created once.
                let warmup = 50;
                match measure_event_rtt_pure(*src, *dst, warmup, iterations) {
                    Ok(s) => {
                        println!(
                            "    pure_event   {} -> {}: mean {:.1} us, p50 {:.1}, p99 {:.1}, stddev {:.1}, min {:.1} / max {:.1}",
                            src.id, dst.id, s.mean, s.p50, s.p99, s.stddev, s.min, s.max,
                        );
                        event_rtt.push(EventRttSample {
                            src_device: src.id,
                            dst_device: dst.id,
                            variant: "pure_event".into(),
                            iterations,
                            mean_us: s.mean, p50_us: s.p50, p99_us: s.p99,
                            min_us: s.min, max_us: s.max, stddev_us: s.stddev,
                        });
                    }
                    Err(e) => println!("    pure_event {} -> {}: FAILED ({e:#})", src.id, dst.id),
                }

                if let (Some(si), Some(di)) = (src_image, dst_image) {
                    match measure_event_rtt_noop(*src, *dst, si, di, warmup, iterations) {
                        Ok(s) => {
                            println!(
                                "    noop_event   {} -> {}: mean {:.1} us, p50 {:.1}, p99 {:.1}, stddev {:.1}, min {:.1} / max {:.1}",
                                src.id, dst.id, s.mean, s.p50, s.p99, s.stddev, s.min, s.max,
                            );
                            event_rtt.push(EventRttSample {
                                src_device: src.id,
                                dst_device: dst.id,
                                variant: "noop_event".into(),
                                iterations,
                                mean_us: s.mean, p50_us: s.p50, p99_us: s.p99,
                                min_us: s.min, max_us: s.max, stddev_us: s.stddev,
                            });
                        }
                        Err(e) => {
                            println!("    noop_event {} -> {}: FAILED ({e:#})", src.id, dst.id)
                        }
                    }
                }
            }
        }
    }

    // --- 5. Cache coherency
    println!("\n== coherency (fill on src, peer-copy to dst, validate) ==");
    let mut coherency = Vec::new();
    if peer_available {
        let n = 1 << 14; // 16k u32 = 64 KiB
        for src in &devices {
            for dst in &devices {
                if src.id == dst.id {
                    continue;
                }
                let src_image = pick_fill(&archs[src.id as usize]);
                if src_image.is_none() {
                    continue;
                }
                match coherency_check(*src, *dst, src_image.unwrap(), n) {
                    Ok(r) => {
                        println!(
                            "    {} -> {}: ok={}, mismatches={}/{}, src_local_ok={}, first_idx={:?}, got=0x{:08x}, expected=0x{:08x}",
                            src.id, dst.id, r.ok, r.mismatch_count, n,
                            r.src_local_check_ok, r.first_mismatch_index,
                            r.first_mismatch_got.unwrap_or(0),
                            r.first_mismatch_expected.unwrap_or(0),
                        );
                        coherency.push(CoherencySample {
                            src_device: src.id,
                            dst_device: dst.id,
                            ok: r.ok,
                            mismatch_count: r.mismatch_count,
                            first_mismatch_index: r.first_mismatch_index,
                            first_mismatch_got: r.first_mismatch_got,
                            first_mismatch_expected: r.first_mismatch_expected,
                            src_local_check_ok: Some(r.src_local_check_ok),
                            error: None,
                        });
                    }
                    Err(e) => {
                        println!("    {} -> {}: FAILED ({e:#})", src.id, dst.id);
                        coherency.push(CoherencySample {
                            src_device: src.id,
                            dst_device: dst.id,
                            ok: false,
                            mismatch_count: 0,
                            first_mismatch_index: None,
                            first_mismatch_got: None,
                            first_mismatch_expected: None,
                            src_local_check_ok: None,
                            error: Some(format!("{e:#}")),
                        });
                    }
                }
            }
        }
    }

    let decision = decide(peer_available, &event_rtt);
    println!("\n== decision ==");
    println!("direct_peer_available: {}", decision.direct_peer_available);
    println!("recommendation:        {}", decision.recommendation);
    println!("rationale:             {}", decision.rationale);

    let report = GateCReport {
        gate: "gate_c",
        timestamp: results::now_unix(),
        peer_matrix,
        bandwidth,
        event_rtt,
        coherency,
        decision,
    };
    let path = results::write("gate_c", &report)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn pick_fill(arch: &str) -> Option<&'static [u8]> {
    if arch.starts_with("gfx1201") {
        Some(FILL_GFX1201)
    } else if arch.starts_with("gfx1151") {
        Some(FILL_GFX1151)
    } else {
        None
    }
}

struct BwStats {
    mean_ms: f64,
    p50_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
    stddev_ms: f64,
    mean_gb_per_s: f64,
    p50_gb_per_s: f64,
}

/// Per-transfer GPU-event-timed bandwidth. We allocate one event-pair per
/// timed iteration so we measure individual transfers and can compute
/// percentiles + stddev (instead of just averaging the batch time).
fn measure_peer_bandwidth(
    src: Device,
    dst: Device,
    n_u32: usize,
    warmup: u32,
    timed: u32,
) -> eyre::Result<BwStats> {
    src.set_current()?;
    let src_buf: DeviceBuffer<u32> = DeviceBuffer::new(src.id, n_u32)?;
    let src_stream = Stream::new(src.id)?;

    dst.set_current()?;
    let mut dst_buf: DeviceBuffer<u32> = DeviceBuffer::new(dst.id, n_u32)?;

    src.set_current()?;
    for _ in 0..warmup {
        src_buf.copy_to_peer_async(&mut dst_buf, &src_stream)?;
    }
    src_stream.synchronize()?;

    // Pre-create event pairs. hipEventCreate is non-trivial; doing it
    // outside the timed window keeps the measurement focused on the
    // copy itself.
    let mut events: Vec<(Event, Event)> = Vec::with_capacity(timed as usize);
    for _ in 0..timed {
        events.push((Event::new()?, Event::new()?));
    }

    src.set_current()?;
    for (start, end) in &events {
        start.record(&src_stream)?;
        src_buf.copy_to_peer_async(&mut dst_buf, &src_stream)?;
        end.record(&src_stream)?;
    }
    src_stream.synchronize()?;

    let mut samples = Vec::with_capacity(timed as usize);
    for (start, end) in &events {
        samples.push(Event::elapsed_ms(start, end)? as f64);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[samples.len() / 2];
    let p99 = samples[(samples.len() * 99) / 100];
    let min = *samples.first().unwrap();
    let max = *samples.last().unwrap();
    let variance =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    let stddev = variance.sqrt();

    let bytes = (n_u32 * 4) as f64;
    Ok(BwStats {
        mean_ms: mean,
        p50_ms: p50,
        p99_ms: p99,
        min_ms: min,
        max_ms: max,
        stddev_ms: stddev,
        mean_gb_per_s: bytes / (mean * 1e-3) / 1e9,
        p50_gb_per_s: bytes / (p50 * 1e-3) / 1e9,
    })
}

/// Choose iteration count adaptively: small transfers tolerate many iters
/// (cheap), large ones need fewer to keep total runtime bounded.
fn bw_iters_for_size(size_bytes: usize) -> u32 {
    if size_bytes <= 1 << 20 { 200 }       // 1 MiB → 200 iters
    else if size_bytes <= 16 << 20 { 100 } // 16 MiB → 100 iters
    else if size_bytes <= 64 << 20 { 50 }  // 64 MiB → 50 iters
    else { 30 }                             // 256 MiB → 30 iters
}

struct RttStats {
    mean: f64,
    p50: f64,
    p99: f64,
    min: f64,
    max: f64,
    stddev: f64,
}

/// Amortized per-sync cost via pipelining. Submits `batch_size` event-record
/// + wait_event pairs back-to-back with no per-iteration host sync. Syncs
/// once at the end of each batch, divides batch time by batch_size to get
/// per-sync cost.
///
/// This is the steady-state cost number: in production we don't sync per
/// transfer — we chain kernels and let the GPU keep working. The host
/// `Instant::now()` measurement and the per-iter `synchronize()` overhead
/// in `measure_event_rtt_pure` aren't paid in real workloads.
fn measure_pipelined_sync(
    src: Device,
    dst: Device,
    batch_size: u32,
    n_batches: u32,
    warmup_batches: u32,
) -> eyre::Result<RttStats> {
    src.set_current()?;
    let src_stream = Stream::new(src.id)?;
    let events: Vec<Event> = (0..batch_size)
        .map(|_| Event::new_no_timing())
        .collect::<eyre::Result<_>>()?;

    dst.set_current()?;
    let dst_stream = Stream::new(dst.id)?;

    let run_batch = |events: &[Event]| -> eyre::Result<()> {
        for ev in events {
            src.set_current()?;
            ev.record(&src_stream)?;
            dst.set_current()?;
            dst_stream.wait_event(ev)?;
        }
        // The single final sync — that's what we save vs per-iter RTT.
        dst_stream.synchronize()?;
        src_stream.synchronize()?;
        Ok(())
    };

    for _ in 0..warmup_batches {
        run_batch(&events)?;
    }

    let mut per_sync = Vec::with_capacity(n_batches as usize);
    for _ in 0..n_batches {
        let t0 = Instant::now();
        run_batch(&events)?;
        let total_us = t0.elapsed().as_nanos() as f64 / 1000.0;
        per_sync.push(total_us / batch_size as f64);
    }

    Ok(rtt_stats(&mut per_sync))
}

/// Pure event-chain RTT: no kernels, events created once and reused.
/// Decomposes "cross-device sync floor" into just the event/wait
/// machinery (HSA signal cross-device propagation + queue submission).
fn measure_event_rtt_pure(
    src: Device,
    dst: Device,
    warmup: u32,
    timed: u32,
) -> eyre::Result<RttStats> {
    src.set_current()?;
    let src_stream = Stream::new(src.id)?;
    let ev_src = Event::new_no_timing()?;

    dst.set_current()?;
    let dst_stream = Stream::new(dst.id)?;
    let ev_dst = Event::new_no_timing()?;

    let do_one = |src: Device, dst: Device, ss: &Stream, ds: &Stream| -> eyre::Result<()> {
        src.set_current()?;
        ev_src.record(ss)?;
        dst.set_current()?;
        ds.wait_event(&ev_src)?;
        ev_dst.record(ds)?;
        src.set_current()?;
        ss.wait_event(&ev_dst)?;
        ss.synchronize()?;
        Ok(())
    };

    for _ in 0..warmup {
        do_one(src, dst, &src_stream, &dst_stream)?;
    }

    let mut samples = Vec::with_capacity(timed as usize);
    for _ in 0..timed {
        let t0 = Instant::now();
        do_one(src, dst, &src_stream, &dst_stream)?;
        samples.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }

    Ok(rtt_stats(&mut samples))
}

/// Noop kernel + event chain. Useful reference for how much the kernel
/// launches add to the event-only chain.
fn measure_event_rtt_noop(
    src: Device,
    dst: Device,
    src_image: &[u8],
    dst_image: &[u8],
    warmup: u32,
    timed: u32,
) -> eyre::Result<RttStats> {
    src.set_current()?;
    let src_mod = Module::load_data(src_image)?;
    let src_fn = src_mod.get_function("noop_kernel")?;
    let src_stream = Stream::new(src.id)?;
    let src_scratch: DeviceBuffer<u32> = DeviceBuffer::new(src.id, 1)?;

    dst.set_current()?;
    let dst_mod = Module::load_data(dst_image)?;
    let dst_fn = dst_mod.get_function("noop_kernel")?;
    let dst_stream = Stream::new(dst.id)?;
    let dst_scratch: DeviceBuffer<u32> = DeviceBuffer::new(dst.id, 1)?;

    let mut src_ptr = src_scratch.raw();
    let mut src_args: [*mut c_void; 1] = [&mut src_ptr as *mut _ as *mut c_void];
    let mut dst_ptr = dst_scratch.raw();
    let mut dst_args: [*mut c_void; 1] = [&mut dst_ptr as *mut _ as *mut c_void];

    // Reuse events across iterations. Events are bound to the *current
    // device* at creation, so we must set_current(src) before ev_src and
    // set_current(dst) before ev_dst — otherwise the event lives on the
    // wrong device and hipEventRecord returns hipErrorInvalidHandle.
    src.set_current()?;
    let ev_src = Event::new_no_timing()?;
    dst.set_current()?;
    let ev_dst = Event::new_no_timing()?;

    let do_one = |src: Device, dst: Device, ss: &Stream, ds: &Stream,
                  s_args: &mut [*mut c_void], d_args: &mut [*mut c_void]| -> eyre::Result<()> {
        src.set_current()?;
        unsafe { src_fn.launch_raw(LaunchConfig::simple(1, 1), ss, s_args)? };
        ev_src.record(ss)?;
        dst.set_current()?;
        ds.wait_event(&ev_src)?;
        unsafe { dst_fn.launch_raw(LaunchConfig::simple(1, 1), ds, d_args)? };
        ev_dst.record(ds)?;
        src.set_current()?;
        ss.wait_event(&ev_dst)?;
        ss.synchronize()?;
        Ok(())
    };

    for _ in 0..warmup {
        do_one(src, dst, &src_stream, &dst_stream, &mut src_args, &mut dst_args)?;
    }

    let mut samples = Vec::with_capacity(timed as usize);
    for _ in 0..timed {
        let t0 = Instant::now();
        do_one(src, dst, &src_stream, &dst_stream, &mut src_args, &mut dst_args)?;
        samples.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }

    Ok(rtt_stats(&mut samples))
}

fn rtt_stats(samples: &mut [f64]) -> RttStats {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[samples.len() / 2];
    let p99 = samples[(samples.len() * 99) / 100];
    let min = *samples.first().unwrap();
    let max = *samples.last().unwrap();
    let variance =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    RttStats { mean, p50, p99, min, max, stddev: variance.sqrt() }
}

struct CoherencyResult {
    ok: bool,
    mismatch_count: usize,
    first_mismatch_index: Option<usize>,
    first_mismatch_got: Option<u32>,
    first_mismatch_expected: Option<u32>,
    /// Did the buffer look right when read back on the SRC device itself
    /// (without involving dst)? If false, fill is broken; if true, the
    /// peer-copy or coherency is broken.
    src_local_check_ok: bool,
}

/// Fill on src with a deterministic kernel, then:
///   (a) validate by reading src buffer back to host on src device.
///   (b) event-sync + peer-copy to dst.
///   (c) read dst buffer to host, validate.
/// This separates "fill kernel works" from "peer-copy preserves bytes."
fn coherency_check(
    src: Device,
    dst: Device,
    src_image: &[u8],
    n_u32: usize,
) -> eyre::Result<CoherencyResult> {
    const SEED: u32 = 0xCAFE_BABE;
    const MULT: u32 = 2654435761;

    src.set_current()?;
    let src_mod = Module::load_data(src_image)?;
    let fill = src_mod.get_function("fill_pattern")?;
    let src_stream = Stream::new(src.id)?;
    let src_buf: DeviceBuffer<u32> = DeviceBuffer::new(src.id, n_u32)?;

    dst.set_current()?;
    let mut dst_buf: DeviceBuffer<u32> = DeviceBuffer::new(dst.id, n_u32)?;
    // dst stream is not needed: copy is queued on src_stream (see below).

    // Launch fill on src.
    src.set_current()?;
    let mut src_ptr = src_buf.raw();
    let n_arg: u32 = n_u32 as u32;
    let seed_arg: u32 = SEED;
    let mut args: [*mut c_void; 3] = [
        &mut src_ptr as *mut _ as *mut c_void,
        &seed_arg as *const _ as *mut c_void,
        &n_arg as *const _ as *mut c_void,
    ];
    let block: u32 = 256;
    let grid: u32 = (n_u32 as u32).div_ceil(block);
    unsafe {
        fill.launch_raw(
            v4flash_hip::LaunchConfig {
                grid: (grid, 1, 1),
                block: (block, 1, 1),
                shared_mem_bytes: 0,
            },
            &src_stream,
            &mut args,
        )?
    };
    let fill_done = Event::new_no_timing()?;
    fill_done.record(&src_stream)?;
    src_stream.synchronize()?;

    // (a) Read src buffer back to host ON SRC device. This bypasses peer
    //     entirely — if this is wrong, the fill kernel itself is broken.
    let mut src_local = vec![0u32; n_u32];
    src_buf.copy_to_host(&mut src_local)?;
    let src_local_ok = (0..n_u32).all(|i| {
        src_local[i] == (SEED ^ (i as u32)).wrapping_mul(MULT)
    });

    // (b) + (c) Peer-copy via SRC stream. The earlier dst_stream version
    // returned zeros for the dGPU→iGPU direction; src_stream fixes it.
    // Hypothesis: dst_stream's wait_event released before the dGPU's L2
    // had flushed to VRAM (HIP's default event-record fence scope appears
    // to be agent, not system, on RDNA 4). Queuing the copy on src_stream
    // serializes the copy after the kernel on the same agent, so dGPU's
    // DMA engine sees coherent L2.
    src.set_current()?;
    src_buf.copy_to_peer_async(&mut dst_buf, &src_stream)?;
    src_stream.synchronize()?;

    let mut host = vec![0u32; n_u32];
    dst_buf.copy_to_host(&mut host)?;

    let mut mismatches = 0;
    let mut first: Option<(usize, u32, u32)> = None;
    for (i, &v) in host.iter().enumerate() {
        let expected = (SEED ^ (i as u32)).wrapping_mul(MULT);
        if v != expected {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, v, expected));
            }
        }
    }

    Ok(CoherencyResult {
        ok: mismatches == 0,
        mismatch_count: mismatches,
        first_mismatch_index: first.map(|(i, _, _)| i),
        first_mismatch_got: first.map(|(_, g, _)| g),
        first_mismatch_expected: first.map(|(_, _, e)| e),
        src_local_check_ok: src_local_ok,
    })
}

fn decide(peer_available: bool, rtt: &[EventRttSample]) -> Decision {
    if !peer_available {
        return Decision {
            direct_peer_available: false,
            recommendation: "host-bounce required".into(),
            rationale: "hipDeviceCanAccessPeer returned false or peer access enable failed \
                       on both directions. Fall back to host-pinned bounce buffer per the \
                       design doc's §5.2 fallback path."
                .into(),
        };
    }
    let best_rtt = rtt
        .iter()
        .map(|s| s.p50_us)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(f64::INFINITY);

    let (recommendation, rationale) = if best_rtt < 30.0 {
        (
            "peer-direct + events (matches doc's 10-30 us estimate)".into(),
            format!(
                "Best cross-device event RTT p50 = {:.1} us. Within design doc's \
                 estimated 10-30 us range; the peer-direct path is the recommended \
                 transport for cross-device transfers and event sync.",
                best_rtt,
            ),
        )
    } else if best_rtt < 80.0 {
        (
            "peer-direct usable but slower than doc estimate".into(),
            format!(
                "Best cross-device event RTT p50 = {:.1} us — slower than doc's \
                 10-30 us estimate but still better than host-bounce floor (~40 us). \
                 Use peer-direct; may need to revisit per-layer sync count or batch \
                 transfers.",
                best_rtt,
            ),
        )
    } else {
        (
            "peer-direct available but not faster — investigate".into(),
            format!(
                "Cross-device event RTT p50 = {:.1} us, comparable to or worse than \
                 host-bounce. Peer access mechanism may be falling back internally; \
                 check rocm-bandwidth-test output and topology.",
                best_rtt,
            ),
        )
    };
    Decision {
        direct_peer_available: true,
        recommendation,
        rationale,
    }
}
