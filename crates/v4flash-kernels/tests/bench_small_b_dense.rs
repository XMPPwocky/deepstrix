//! Small-B dense projections on the dGPU: WMMA f16x GEMM (what the batched
//! driver runs at every B) vs the B-packed Q8 dp4a GEMV (decode's numerics,
//! weight read once for all B) vs B serial decode GEMVs, at the V4.1 shapes,
//! B = 1..8. Event-timed min of BENCH_ITERS. Tiny VRAM: runs next to a live server.
//! HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release \
//!   --features v41 -p v4flash-kernels --test bench_small_b_dense -- --ignored --nocapture
use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::het::batch_scratch::f16_pitch;
use v4flash_kernels::het::engine::DeviceEngine;

fn time_min(stream: &Stream, iters: usize, mut f: impl FnMut(&Stream) -> eyre::Result<()>) -> eyre::Result<f64> {
    for _ in 0..3 { f(stream)?; }
    stream.synchronize()?;
    let mut best = f64::MAX;
    for _ in 0..iters {
        let s = Event::new()?; let e = Event::new()?;
        s.record(stream)?; f(stream)?; e.record(stream)?; stream.synchronize()?;
        best = best.min(Event::elapsed_ms(&s, &e)? as f64 * 1000.0);
    }
    Ok(best)
}

#[test]
#[ignore]
fn bench_small_b_dense() -> eyre::Result<()> {
    color_eyre::install().ok();
    // Pick the dGPU by arch: the WMMA kernel is compiled for gfx1200/1201 only
    // and is an empty body elsewhere (a 4.6 us "kernel" on the iGPU).
    let mut dev = Device::new(0);
    let mut arch = String::new();
    for id in 0..4 {
        let d = Device::new(id);
        if d.set_current().is_err() { continue; }
        let Ok(p) = d.properties() else { continue };
        if p.gcn_arch_name.starts_with("gfx120") { dev = d; arch = p.gcn_arch_name; break; }
    }
    if arch.is_empty() { return Err(eyre::eyre!("no gfx120x device visible")); }
    dev.set_current()?;
    println!("device {} {arch}", dev.id);
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let stream = Stream::new(dev.id)?;
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    // (name, n_rows (M), k)
    let shapes = [("q_a 1280x5120", 1280u32, 5120u32), ("kv 512x5120", 512, 5120), ("shared gate 2304x5120", 2304, 5120),
                  ("shared down 5120x2304", 5120, 2304), ("out_proj 5120x8192", 5120, 8192), ("q_b 32768x1280", 32768, 1280)];
    // Ring of 4 weight copies so the 64 MB infinity cache does not serve a 7 MB matrix.
    println!("{:<24} {:>2} | {:>9} {:>9} {:>9} | {:>7} {:>7}", "shape", "B", "wmma_us", "bpack_us", "serialB_us", "wmmaGBs", "bpackGBs");
    for (name, m, k) in shapes {
        let blocks = k / 32;
        let wbytes = (m as usize) * (blocks as usize) * 34;
        let copies = (64usize * 1024 * 1024 / wbytes).clamp(2, 12);
        let mut ws: Vec<DeviceBuffer<u8>> = Vec::new();
        for c in 0..copies {
            let host: Vec<u8> = (0..wbytes).map(|i| ((i * 31 + c * 7) % 251) as u8).collect();
            let mut w = DeviceBuffer::<u8>::new(dev.id, wbytes)?; w.copy_from_host(&host)?; ws.push(w);
        }
        for b in [1u32, 2, 4, 8] {
            let x: Vec<f32> = (0..(b * k) as usize).map(|i| ((i % 97) as f32 - 48.0) / 50.0).collect();
            let mut xd = DeviceBuffer::<f32>::new(dev.id, x.len())?; xd.copy_from_host(&x)?;
            let mut xq = DeviceBuffer::<i8>::new(dev.id, (b * k) as usize)?;
            let mut xs = DeviceBuffer::<f32>::new(dev.id, (b * blocks) as usize)?;
            e.q8.quantize_input_batched(&stream, &mut xq, &mut xs, &xd, k, b)?;
            let pitch = f16_pitch(k);
            let mut x16 = DeviceBuffer::<u16>::new(dev.id, (b * pitch) as usize)?;
            e.q8k.launch_cast_f16_2d(&stream, &mut x16, &xd, b, k, pitch)?;
            let mut out = DeviceBuffer::<f32>::new(dev.id, (b * m) as usize)?;
            let mut i = 0usize;
            let wmma = if m % 128 == 0 {
                time_min(&stream, iters, |s| { i = (i + 1) % copies; e.q8_wmma.gemm_f16x(s, &mut out, &ws[i], &x16, k, m, 1, b, pitch) })?
            } else { f64::NAN };
            let bpack = time_min(&stream, iters, |s| { i = (i + 1) % copies; e.q8.matvec_bpack(s, &mut out, &ws[i], &xq, &xs, m, k, b) })?;
            let serial = time_min(&stream, iters, |s| {
                i = (i + 1) % copies;
                for r in 0..b {
                    let mut o = out.slice_view_mut((r * m) as usize, m as usize);
                    e.q8.matvec(s, &mut o, &ws[i], &xq.slice_view((r * k) as usize, k as usize), &xs.slice_view((r * blocks) as usize, blocks as usize), m, k)?;
                }
                Ok(())
            })?;
            let gbs = |us: f64| wbytes as f64 / us / 1e3;
            println!("{:<24} {:>2} | {:>9.1} {:>9.1} {:>9.1} | {:>7.0} {:>7.0}", name, b, wmma, bpack, serial, gbs(wmma), gbs(bpack));
        }
    }
    Ok(())
}
