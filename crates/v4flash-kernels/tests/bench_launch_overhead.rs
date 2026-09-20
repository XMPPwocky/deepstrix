//! Host-side cost of a kernel launch on the dGPU: function lookup + launch,
//! serial (host wall per launch with the queue kept shallow) — the per-launch
//! floor the batched driver pays ~2000 times per step.
//! HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release \
//!   --features v41 -p v4flash-kernels --test bench_launch_overhead -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer, Stream};
use v4flash_kernels::het::engine::DeviceEngine;

#[test]
#[ignore]
fn bench_launch_overhead() -> eyre::Result<()> {
    color_eyre::install().ok();
    let mut dev = Device::new(0); let mut arch = String::new();
    for id in 0..4 { let d = Device::new(id); if d.set_current().is_err() { continue; } let Ok(p) = d.properties() else { continue }; if p.gcn_arch_name.starts_with("gfx120") { dev = d; arch = p.gcn_arch_name; break; } }
    dev.set_current()?;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let stream = Stream::new(dev.id)?;
    let n = 5120u32; let b = 1u32;
    let x = DeviceBuffer::<f32>::new(dev.id, (b * n) as usize)?;
    let mut out16 = DeviceBuffer::<u16>::new(dev.id, (b * (n + 64)) as usize)?;
    let pitch = n + 64;
    let iters = 2000;
    // (a) function lookup only
    let t = std::time::Instant::now();
    for _ in 0..iters { let _ = e.q8.module().get_function("q8_0_gemv_warp8")?; }
    println!("get_function: {:.2} us/call", t.elapsed().as_secs_f64() * 1e6 / iters as f64);
    // (b) launches back-to-back (host enqueue cost; queue absorbs)
    stream.synchronize()?;
    let t = std::time::Instant::now();
    for _ in 0..iters { e.q8k.launch_cast_f16_2d(&stream, &mut out16, &x, b, n, pitch)?; }
    let enq = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    stream.synchronize()?;
    let total = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("tiny kernel: host enqueue {enq:.2} us/launch, wall incl. drain {total:.2} us/launch");
    // (c) launch + sync each (the pattern around host readbacks)
    let t = std::time::Instant::now();
    for _ in 0..200 { e.q8k.launch_cast_f16_2d(&stream, &mut out16, &x, b, n, pitch)?; stream.synchronize()?; }
    println!("tiny kernel + synchronize: {:.2} us/round trip", t.elapsed().as_secs_f64() * 1e6 / 200.0);
    Ok(())
}
