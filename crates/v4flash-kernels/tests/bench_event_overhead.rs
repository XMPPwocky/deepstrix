//! Cost of the `EventPool::stage` pattern (a timing-enabled hipEventRecord pair
//! around a kernel), which `V41_MS_PROFILE=1` pays ~1300x per 8-row decode step
//! (1020 dGPU pairs + 280 iGPU pairs, ms.stage `calls` / steps, 2026-09-21).
//! HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release \
//!   --features v41 -p v4flash-kernels --test bench_event_overhead -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::het::engine::DeviceEngine;

fn run(dev: Device, arch: &str) -> eyre::Result<()> {
    dev.set_current()?;
    let e = DeviceEngine::for_arch(dev, arch)?;
    let stream = Stream::new(dev.id)?;
    let n = 5120u32; let b = 1u32;
    let x = DeviceBuffer::<f32>::new(dev.id, (b * n) as usize)?;
    let mut out16 = DeviceBuffer::<u16>::new(dev.id, (b * (n + 64)) as usize)?;
    let pitch = n + 64;
    let iters = 2000usize;
    let mut evs = Vec::with_capacity(2 * iters);
    for _ in 0..2 * iters { evs.push(Event::new()?); }
    for pass in 0..2 {
        // (a) kernels only
        stream.synchronize()?;
        let t = std::time::Instant::now();
        for _ in 0..iters { e.q8k.launch_cast_f16_2d(&stream, &mut out16, &x, b, n, pitch)?; }
        let enq_a = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        stream.synchronize()?;
        let wall_a = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        // (b) kernels with a timing event pair around each (the k.* stage pattern)
        stream.synchronize()?;
        let t = std::time::Instant::now();
        for i in 0..iters {
            evs[2 * i].record(&stream)?;
            e.q8k.launch_cast_f16_2d(&stream, &mut out16, &x, b, n, pitch)?;
            evs[2 * i + 1].record(&stream)?;
        }
        let enq_b = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        stream.synchronize()?;
        let wall_b = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        // (c) harvest: hipEventElapsedTime per pair (what ms.stage does after the step)
        let t = std::time::Instant::now();
        let mut acc = 0f32;
        for i in 0..iters { acc += Event::elapsed_ms(&evs[2 * i], &evs[2 * i + 1])?; }
        let harvest = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        // (d) records only, no kernels
        stream.synchronize()?;
        let t = std::time::Instant::now();
        for i in 0..iters { evs[2 * i].record(&stream)?; evs[2 * i + 1].record(&stream)?; }
        let enq_d = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        stream.synchronize()?;
        let wall_d = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        if pass == 1 {
            println!("{arch}: kernel only          host {enq_a:6.2} us  wall {wall_a:6.2} us per kernel");
            println!("{arch}: kernel + event pair  host {enq_b:6.2} us  wall {wall_b:6.2} us per kernel  (pair adds host {:+.2} / device {:+.2} us)", enq_b - enq_a, wall_b - wall_a);
            println!("{arch}: event pair only      host {enq_d:6.2} us  wall {wall_d:6.2} us per pair");
            println!("{arch}: elapsed_ms harvest   {harvest:6.2} us per pair (mean stage {:.3} ms)", acc / iters as f32);
        }
    }
    Ok(())
}

#[test]
#[ignore]
fn bench_event_overhead() -> eyre::Result<()> {
    color_eyre::install().ok();
    for id in 0..4 {
        let d = Device::new(id);
        if d.set_current().is_err() { continue; }
        let Ok(p) = d.properties() else { continue };
        if p.gcn_arch_name.starts_with("gfx120") || p.gcn_arch_name.starts_with("gfx1151") {
            run(d, &p.gcn_arch_name)?;
        }
    }
    Ok(())
}
