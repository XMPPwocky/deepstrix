//! Isolated IndexerScore profile harness. One model load, N warmup +
//! N timed dispatches at the production decode shape (n_comp=16384,
//! n_head=64, head_dim=128). The 64K-decode bench showed IndexerScore
//! is the dominant added cost; this harness is for narrowing in on
//! exactly which instruction stalls dominate via rocprofv3 ATT.
//!
//! Usage:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_indexer_score_isolated -- --ignored --nocapture
//!
//! ATT (one traced dispatch):
//!   rocprofv3 --att --att-library-path \
//!     /nix/store/.../librocprof-trace-decoder.so \
//!     --att-gpu-index 0 --att-consecutive-kernels 1 \
//!     --att-buffer-size 1073741824 -d /tmp/att-ix -o ix \
//!     -- $(cargo test --release --no-run -p v4flash-kernels \
//!           --test bench_indexer_score_isolated 2>&1 | \
//!           grep -oP 'target/release/deps/bench_indexer_score[^ ]+') \
//!     --ignored --nocapture

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{IndexerScore, INDEXER_HEAD_DIM, INDEXER_N_HEAD};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 device"))
}

fn lcg(s: &mut u32) -> u16 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    (*s >> 16) as u16
}

#[test]
#[ignore]
fn bench_indexer_score_isolated() -> eyre::Result<()> {
    install_panic_handler()?;

    let n_comp: u32 = std::env::var("BENCH_N_COMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16384);
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
    let kernel = IndexerScore::for_arch(&arch)?;

    let n_head = INDEXER_N_HEAD as usize;
    let head_dim = INDEXER_HEAD_DIM as usize;

    eprintln!("isolated IndexerScore: n_comp={n_comp} n_head={n_head} head_dim={head_dim} iters={iters} warmup={warmup}");

    // q[n_head, head_dim] f32
    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_head * head_dim)?;
    d_q.copy_from_host(&vec![0.1f32; n_head * head_dim])?;

    // head_weights[n_head] f32
    let mut d_hw: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_head)?;
    d_hw.copy_from_host(&vec![0.05f32; n_head])?;

    // index_comp_kv[n_comp, head_dim] f16-stored
    let mut seed: u32 = 0xdeadbeef;
    let kv_host: Vec<u16> = (0..(n_comp as usize) * head_dim).map(|_| lcg(&mut seed)).collect();
    let mut d_kv: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, kv_host.len())?;
    d_kv.copy_from_host(&kv_host)?;

    // scores[n_comp] f32
    let mut d_scores: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_comp as usize)?;
    d_scores.copy_from_host(&vec![0f32; n_comp as usize])?;

    let launch = |stream: &Stream, d_scores: &mut DeviceBuffer<f32>| -> eyre::Result<()> {
        kernel.launch(
            stream,
            d_scores,
            &d_q,
            &d_hw,
            &d_kv,
            n_comp,
            n_head as u32,
            head_dim as u32,
        )
    };

    // Warmup.
    for _ in 0..warmup {
        launch(&stream, &mut d_scores)?;
    }
    stream.synchronize()?;

    // Timed.
    let t0 = Instant::now();
    for _ in 0..iters {
        launch(&stream, &mut d_scores)?;
    }
    stream.synchronize()?;
    let elapsed = t0.elapsed();
    let per_call_us = elapsed.as_micros() as f64 / iters as f64;

    eprintln!(
        "BENCH IndexerScore n_comp={n_comp}: {iters} iters in {:.2}ms = {:.2}us/call",
        elapsed.as_secs_f64() * 1000.0,
        per_call_us
    );

    Ok(())
}
