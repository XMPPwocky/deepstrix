//! `phase0 pingpong` — characterize the cross-device sync floor by
//! decomposing it into four orthogonal measurements:
//!
//! 1. **Empty kernel launch** — single-launch latency with zero DMA.
//!    Isolates the GPU command-submission floor (per-call cost of getting
//!    *anything* onto the device).
//! 2. **Same-device DtoH+HtoD** — two synchronous memcpys on one device,
//!    no peer or device-switch involved. Isolates "two memcpys" overhead
//!    from "cross-device" overhead.
//! 3. **Pinned vs unpinned host scratch** — `Vec<u32>` (pageable) vs
//!    `PinnedBuffer<u32>` (hipHostMalloc). Quantifies the unpinned
//!    double-staging penalty.
//! 4. **Payload sweep** — 64 B / 4 KiB / 64 KiB / 1 MiB. Finds where the
//!    per-call floor stops dominating and bandwidth starts.
//!
//! All in one report so we can compare. Async/peer-direct is Gate C.

use std::ffi::c_void;
use std::time::Instant;

use color_eyre::eyre;
use serde::Serialize;
use v4flash_hip::{
    Device, DeviceBuffer, Event, LaunchConfig, Module, PinnedBuffer, Stream,
};

use crate::results;

const HELLO_GFX1201: &[u8] = include_bytes!(env!("KERNEL_HELLO_GFX1201"));
const HELLO_GFX1151: &[u8] = include_bytes!(env!("KERNEL_HELLO_GFX1151"));

const PAYLOAD_SIZES: &[usize] = &[64, 4096, 65_536, 1_048_576];

#[derive(Serialize)]
pub struct PingpongReport {
    pub gate: &'static str,
    pub timestamp: u64,
    pub iterations: u32,
    pub tests: Vec<TestResult>,
}

#[derive(Serialize, Debug)]
pub struct TestResult {
    pub name: String,
    pub src_device: i32,
    pub src_arch: String,
    pub dst_device: Option<i32>,
    pub dst_arch: Option<String>,
    pub payload_bytes: Option<usize>,
    pub pinned: Option<bool>,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub correctness_ok: bool,
}

pub fn run(iterations: u32) -> eyre::Result<()> {
    let devices = Device::all()?;
    let device_archs: Vec<String> = devices
        .iter()
        .map(|d| d.properties().map(|p| p.gcn_arch_name).unwrap_or_default())
        .collect();

    let mut tests = Vec::new();

    // 1. Empty kernel launch latency (single launch + event sync).
    for dev in &devices {
        let arch = &device_archs[dev.id as usize];
        if let Some(image) = pick_hello(arch) {
            println!("\n== empty kernel launch, device {} ({}) ==", dev.id, arch);
            match empty_kernel_launch(*dev, image, iterations) {
                Ok((mean, p50, p99, min, max)) => {
                    print_stats(mean, p50, p99, min, max);
                    tests.push(TestResult {
                        name: "empty_kernel_launch".into(),
                        src_device: dev.id,
                        src_arch: arch.clone(),
                        dst_device: None,
                        dst_arch: None,
                        payload_bytes: None,
                        pinned: None,
                        mean_us: mean,
                        p50_us: p50,
                        p99_us: p99,
                        min_us: min,
                        max_us: max,
                        correctness_ok: true,
                    });
                }
                Err(e) => println!("    FAILED: {e:#}"),
            }
        }
    }

    // 2 + 3. Same-device DtoH+HtoD, pinned and unpinned, at 4 KiB.
    for dev in &devices {
        let arch = &device_archs[dev.id as usize];
        for pinned in [false, true] {
            let label = if pinned { "pinned" } else { "unpinned" };
            println!(
                "\n== samedev DtoH+HtoD, device {} ({}), {}, {} B ==",
                dev.id, arch, label, 4096
            );
            match samedev_double_memcpy(*dev, 4096, pinned, iterations) {
                Ok((mean, p50, p99, min, max, ok)) => {
                    print_stats(mean, p50, p99, min, max);
                    tests.push(TestResult {
                        name: format!("samedev_double_memcpy_{label}"),
                        src_device: dev.id,
                        src_arch: arch.clone(),
                        dst_device: None,
                        dst_arch: None,
                        payload_bytes: Some(4096),
                        pinned: Some(pinned),
                        mean_us: mean,
                        p50_us: p50,
                        p99_us: p99,
                        min_us: min,
                        max_us: max,
                        correctness_ok: ok,
                    });
                }
                Err(e) => println!("    FAILED: {e:#}"),
            }
        }
    }

    // 4. Cross-device host-bounce, payload sweep, pinned and unpinned.
    for src in &devices {
        for dst in &devices {
            if src.id == dst.id {
                continue;
            }
            let src_arch = &device_archs[src.id as usize];
            let dst_arch = &device_archs[dst.id as usize];

            for &payload in PAYLOAD_SIZES {
                for pinned in [false, true] {
                    let label = if pinned { "pinned" } else { "unpinned" };
                    println!(
                        "\n== host-bounce {} ({}) -> {} ({}), {}, {} B ==",
                        src.id, src_arch, dst.id, dst_arch, label, payload
                    );
                    match host_bounce(*src, *dst, payload, pinned, iterations) {
                        Ok((mean, p50, p99, min, max, ok)) => {
                            print_stats(mean, p50, p99, min, max);
                            let gb_per_s = (payload as f64) / (mean * 1e-6) / 1e9;
                            println!(
                                "    effective throughput: {:.2} GB/s (single direction)",
                                gb_per_s
                            );
                            tests.push(TestResult {
                                name: format!("host_bounce_{label}"),
                                src_device: src.id,
                                src_arch: src_arch.clone(),
                                dst_device: Some(dst.id),
                                dst_arch: Some(dst_arch.clone()),
                                payload_bytes: Some(payload),
                                pinned: Some(pinned),
                                mean_us: mean,
                                p50_us: p50,
                                p99_us: p99,
                                min_us: min,
                                max_us: max,
                                correctness_ok: ok,
                            });
                        }
                        Err(e) => println!("    FAILED: {e:#}"),
                    }
                }
            }
        }
    }

    let report = PingpongReport {
        gate: "pingpong",
        timestamp: results::now_unix(),
        iterations,
        tests,
    };
    let path = results::write("pingpong", &report)?;
    println!("\nwrote {}", path.display());
    Ok(())
}

fn pick_hello(arch: &str) -> Option<&'static [u8]> {
    if arch.starts_with("gfx1201") {
        Some(HELLO_GFX1201)
    } else if arch.starts_with("gfx1151") {
        Some(HELLO_GFX1151)
    } else {
        None
    }
}

fn print_stats(mean: f64, p50: f64, p99: f64, min: f64, max: f64) {
    println!(
        "    mean {:.1} us, p50 {:.1} us, p99 {:.1} us (min {:.1} / max {:.1})",
        mean, p50, p99, min, max
    );
}

/// Time a single empty hello-kernel launch on its own stream, with an event
/// recorded after the launch and synchronized. This is the minimum "queue
/// something on the GPU and wait for it" round-trip.
fn empty_kernel_launch(
    dev: Device,
    image: &[u8],
    iterations: u32,
) -> eyre::Result<(f64, f64, f64, f64, f64)> {
    dev.set_current()?;
    let module = Module::load_data(image)?;
    let function = module.get_function("hello")?;
    let stream = Stream::new(dev.id)?;
    let mut buf: DeviceBuffer<i32> = DeviceBuffer::new(dev.id, 1)?;

    let mut raw_ptr = buf.raw();
    let mut args: [*mut c_void; 1] = [&mut raw_ptr as *mut _ as *mut c_void];

    // Warm: one untimed launch.
    unsafe { function.launch_raw(LaunchConfig::simple(1, 1), &stream, &mut args)? };
    stream.synchronize()?;

    let mut samples = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        buf.fill_zero()?; // not timed: just to make each iteration uniform
        let t0 = Instant::now();
        unsafe { function.launch_raw(LaunchConfig::simple(1, 1), &stream, &mut args)? };
        let done = Event::new_no_timing()?;
        done.record(&stream)?;
        done.synchronize()?;
        samples.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }

    Ok(stats(&mut samples))
}

/// Two synchronous memcpys on one device: DtoH then HtoD via a host scratch
/// buffer. With `pinned=true`, the scratch is hipHostMalloc'd.
fn samedev_double_memcpy(
    dev: Device,
    payload_bytes: usize,
    pinned: bool,
    iterations: u32,
) -> eyre::Result<(f64, f64, f64, f64, f64, bool)> {
    dev.set_current()?;
    let n = payload_bytes / 4;

    let pattern: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761)).collect();
    let mut src: DeviceBuffer<u32> = DeviceBuffer::new(dev.id, n)?;
    src.copy_from_host(&pattern)?;
    let mut dst: DeviceBuffer<u32> = DeviceBuffer::new(dev.id, n)?;
    dst.fill_zero()?;

    // Two scratch storages; only one used per run, but we want both to
    // satisfy the borrow checker without conditional types.
    let mut unpinned = vec![0u32; n];
    let mut pinned_buf: Option<PinnedBuffer<u32>> =
        if pinned { Some(PinnedBuffer::new(n)?) } else { None };

    // Warm.
    do_samedev(&src, &mut dst, &mut unpinned, pinned_buf.as_mut())?;

    let mut samples = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let t0 = Instant::now();
        do_samedev(&src, &mut dst, &mut unpinned, pinned_buf.as_mut())?;
        samples.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }

    // Validate.
    let mut readback = vec![0u32; n];
    dst.copy_to_host(&mut readback)?;
    let correct = readback == pattern;

    let (mean, p50, p99, min, max) = stats(&mut samples);
    Ok((mean, p50, p99, min, max, correct))
}

fn do_samedev(
    src: &DeviceBuffer<u32>,
    dst: &mut DeviceBuffer<u32>,
    unpinned: &mut [u32],
    mut pinned: Option<&mut PinnedBuffer<u32>>,
) -> eyre::Result<()> {
    match pinned.as_mut() {
        Some(p) => {
            src.copy_to_host(p.as_mut_slice())?;
            dst.copy_from_host(p.as_slice())?;
        }
        None => {
            src.copy_to_host(unpinned)?;
            dst.copy_from_host(unpinned)?;
        }
    }
    Ok(())
}

/// Cross-device host-bounce: DtoH on src, HtoD on dst, sync per memcpy.
fn host_bounce(
    src: Device,
    dst: Device,
    payload_bytes: usize,
    pinned: bool,
    iterations: u32,
) -> eyre::Result<(f64, f64, f64, f64, f64, bool)> {
    let n = payload_bytes / 4;
    let pattern: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761)).collect();

    src.set_current()?;
    let mut src_buf: DeviceBuffer<u32> = DeviceBuffer::new(src.id, n)?;
    src_buf.copy_from_host(&pattern)?;

    dst.set_current()?;
    let mut dst_buf: DeviceBuffer<u32> = DeviceBuffer::new(dst.id, n)?;
    dst_buf.fill_zero()?;

    let mut unpinned = vec![0u32; n];
    let mut pinned_buf: Option<PinnedBuffer<u32>> =
        if pinned { Some(PinnedBuffer::new(n)?) } else { None };

    // Warm.
    do_bounce(src, &src_buf, dst, &mut dst_buf, &mut unpinned, pinned_buf.as_mut())?;

    let mut samples = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let t0 = Instant::now();
        do_bounce(src, &src_buf, dst, &mut dst_buf, &mut unpinned, pinned_buf.as_mut())?;
        samples.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }

    // Validate via one final read from dst.
    dst.set_current()?;
    let mut readback = vec![0u32; n];
    dst_buf.copy_to_host(&mut readback)?;
    let correct = readback == pattern;

    let (mean, p50, p99, min, max) = stats(&mut samples);
    Ok((mean, p50, p99, min, max, correct))
}

fn do_bounce(
    src: Device,
    src_buf: &DeviceBuffer<u32>,
    dst: Device,
    dst_buf: &mut DeviceBuffer<u32>,
    unpinned: &mut [u32],
    mut pinned: Option<&mut PinnedBuffer<u32>>,
) -> eyre::Result<()> {
    src.set_current()?;
    match pinned.as_mut() {
        Some(p) => src_buf.copy_to_host(p.as_mut_slice())?,
        None => src_buf.copy_to_host(unpinned)?,
    }
    dst.set_current()?;
    match pinned.as_mut() {
        Some(p) => dst_buf.copy_from_host(p.as_slice())?,
        None => dst_buf.copy_from_host(unpinned)?,
    }
    Ok(())
}

/// (mean, p50, p99, min, max) in microseconds; sorts in place.
fn stats(samples: &mut [f64]) -> (f64, f64, f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[samples.len() / 2];
    let p99 = samples[(samples.len() * 99) / 100];
    let min = *samples.first().unwrap();
    let max = *samples.last().unwrap();
    (mean, p50, p99, min, max)
}
