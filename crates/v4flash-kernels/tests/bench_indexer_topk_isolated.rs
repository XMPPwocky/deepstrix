//! Isolated IndexerTopk timing harness. Sibling to bench_indexer_score_isolated.
//! Run after profiling IndexerScore to see how much of the indexer regression
//! is attributable to the top-K kernel vs the score kernel.

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{IndexerTopk, IndexerTopkBitonic, INDEXER_TOP_K};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 device"))
}

fn lcg(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    let v = (*s >> 8) as f32 / (1u32 << 24) as f32;
    v * 2.0 - 1.0
}

#[test]
#[ignore]
fn bench_indexer_topk_isolated() -> eyre::Result<()> {
    install_panic_handler()?;

    let n_comp: u32 = std::env::var("BENCH_N_COMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16384);
    let top_k: u32 = std::env::var("BENCH_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(INDEXER_TOP_K);
    let iters: u32 = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let warmup: u32 = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let kernel = IndexerTopk::for_arch(&arch)?;

    eprintln!("isolated IndexerTopk: n_comp={n_comp} top_k={top_k} iters={iters} warmup={warmup}");

    let mut seed: u32 = 0xdeadbeef;
    let scores: Vec<f32> = (0..n_comp).map(|_| lcg(&mut seed)).collect();
    let mut d_scores: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_comp as usize)?;
    d_scores.copy_from_host(&scores)?;
    let mut d_selected: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, top_k as usize)?;
    let mut d_bits: DeviceBuffer<u32> =
        DeviceBuffer::new(dgpu.id, ((n_comp + 31) / 32) as usize)?;

    let launch = |stream: &Stream,
                  d_selected: &mut DeviceBuffer<i32>,
                  d_bits: &mut DeviceBuffer<u32>|
     -> eyre::Result<()> { kernel.launch(stream, d_selected, d_bits, &d_scores, n_comp, top_k) };

    for _ in 0..warmup {
        launch(&stream, &mut d_selected, &mut d_bits)?;
    }
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..iters {
        launch(&stream, &mut d_selected, &mut d_bits)?;
    }
    stream.synchronize()?;
    let elapsed = t0.elapsed();
    let per_call_us = elapsed.as_micros() as f64 / iters as f64;

    eprintln!(
        "BENCH IndexerTopk (greedy)  n_comp={n_comp} K={top_k}: {iters} iters in {:.2}ms = {:.2}us/call",
        elapsed.as_secs_f64() * 1000.0,
        per_call_us
    );

    // --- Bitonic variant ---
    let kernel_b = IndexerTopkBitonic::for_arch(&arch)?;
    let max_chunks = (n_comp + 4095) / 4096;
    let mut d_scratch: DeviceBuffer<u32> =
        DeviceBuffer::new(dgpu.id, (max_chunks * top_k).max(1) as usize)?;
    let launch_b = |stream: &Stream,
                    d_selected: &mut DeviceBuffer<i32>,
                    d_bits: &mut DeviceBuffer<u32>,
                    d_scratch: &mut DeviceBuffer<u32>|
     -> eyre::Result<()> {
        kernel_b.launch(stream, d_selected, d_bits, d_scratch, &d_scores, n_comp, top_k)
    };
    for _ in 0..warmup {
        launch_b(&stream, &mut d_selected, &mut d_bits, &mut d_scratch)?;
    }
    stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        launch_b(&stream, &mut d_selected, &mut d_bits, &mut d_scratch)?;
    }
    stream.synchronize()?;
    let elapsed_b = t0.elapsed();
    let per_call_us_b = elapsed_b.as_micros() as f64 / iters as f64;
    eprintln!(
        "BENCH IndexerTopk (bitonic) n_comp={n_comp} K={top_k}: {iters} iters in {:.2}ms = {:.2}us/call  ({:.1}× speedup)",
        elapsed_b.as_secs_f64() * 1000.0,
        per_call_us_b,
        per_call_us / per_call_us_b,
    );

    Ok(())
}
