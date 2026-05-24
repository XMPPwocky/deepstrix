//! Smoke test for the new HIP-graph wrappers (M14.1).
//!
//! Captures a sequence of vec_add kernel launches into a graph,
//! instantiates it, launches the graph many times, and compares wall
//! time vs the equivalent direct-launch loop. The whole point of HIP
//! graphs is to fold N launches into one — if the wrappers work and
//! the underlying runtime cooperates, the graph form should be
//! noticeably faster for small kernels (where launch overhead
//! dominates) when N > 1.

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, sys, Device, DeviceBuffer, Stream};
use v4flash_kernels::ffn::VecAddInplace;

const N_ELEMS: u32 = 4096;
const N_LAUNCHES_PER_ITER: u32 = 16; // kernels per "layer" in the test
const ITERS: u32 = 500; // how many "layers" we run

fn pick_device(want_integrated: bool) -> eyre::Result<Device> {
    Device::all()?
        .into_iter()
        .find(|d| {
            d.properties()
                .map(|p| p.integrated == want_integrated)
                .unwrap_or(false)
        })
        .ok_or_else(|| eyre!("no matching HIP device (want_integrated={})", want_integrated))
}

fn pick_device_dgpu() -> eyre::Result<Device> {
    pick_device(false)
}

fn pick_device_igpu() -> eyre::Result<Device> {
    pick_device(true)
}

fn run_capture_bench(label: &str, device: Device) -> eyre::Result<()> {
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("=== {label}: device={} ({}) ===", device.id, arch);

    let stream = Stream::new(device.id)?;
    let vec_add = VecAddInplace::for_arch(&arch)?;

    let mut a: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_ELEMS as usize)?;
    let mut b: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_ELEMS as usize)?;
    let zeros = vec![0f32; N_ELEMS as usize];
    let ones = vec![1f32; N_ELEMS as usize];
    a.copy_from_host(&zeros)?;
    b.copy_from_host(&ones)?;

    a.copy_from_host(&zeros)?;
    stream.synchronize()?;
    let t_direct = Instant::now();
    for _ in 0..ITERS {
        for _ in 0..N_LAUNCHES_PER_ITER {
            vec_add.launch(&stream, &mut a, &b, N_ELEMS)?;
        }
    }
    stream.synchronize()?;
    let direct_ms = t_direct.elapsed().as_secs_f64() * 1000.0;

    a.copy_from_host(&zeros)?;
    stream.synchronize()?;
    stream.begin_capture(sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    for _ in 0..N_LAUNCHES_PER_ITER {
        vec_add.launch(&stream, &mut a, &b, N_ELEMS)?;
    }
    let graph = stream.end_capture()?;
    let exec = graph.instantiate()?;
    stream.synchronize()?;

    let t_graph = Instant::now();
    for _ in 0..ITERS {
        exec.launch(&stream)?;
    }
    stream.synchronize()?;
    let graph_ms = t_graph.elapsed().as_secs_f64() * 1000.0;

    let total_launches = (ITERS * N_LAUNCHES_PER_ITER) as f64;
    let direct_us_per = direct_ms * 1000.0 / total_launches;
    let graph_us_per = graph_ms * 1000.0 / (ITERS as f64);
    eprintln!(
        "  DIRECT: {:.2}ms total, {:.2}μs/launch",
        direct_ms, direct_us_per
    );
    eprintln!(
        "  GRAPH : {:.2}ms total, {:.2}μs/iter (= {:.2}μs/contained-launch)",
        graph_ms,
        graph_us_per,
        graph_us_per / N_LAUNCHES_PER_ITER as f64
    );
    let saved_per_iter = (direct_ms - graph_ms) * 1000.0 / ITERS as f64;
    eprintln!(
        "  speedup {:.2}x ({:.2}μs saved per iter)",
        direct_ms / graph_ms,
        saved_per_iter
    );
    Ok(())
}

#[test]
#[ignore]
fn hip_graph_capture_and_launch() -> eyre::Result<()> {
    install_panic_handler()?;
    run_capture_bench("dGPU", pick_device_dgpu()?)?;
    eprintln!();
    run_capture_bench("iGPU", pick_device_igpu()?)?;
    Ok(())
}
