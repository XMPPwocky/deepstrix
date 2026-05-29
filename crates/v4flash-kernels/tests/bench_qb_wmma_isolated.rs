//! Isolated qb microbench: dp4a `matvec_batched` vs int8-WMMA `gemm` at the
//! real prefill qb shape (M=Q_FLAT=32768, K=N_LORA_Q=1024, B=64). No model
//! load. Both kernels are timed back-to-back in the SAME process so thermal
//! drift can't confound the A/B (see bench A/B methodology).
//!
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test bench_qb_wmma_isolated -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::q8_0::{Q8_0Matvec, Q8_0MatvecWmma, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 dGPU"))
}

fn percentile(xs_sorted: &[f32], p: f32) -> f32 {
    if xs_sorted.is_empty() {
        return 0.0;
    }
    let k = ((xs_sorted.len() - 1) as f32 * p / 100.0).round() as usize;
    xs_sorted[k.min(xs_sorted.len() - 1)]
}

fn stats(walls: &mut [f32], label: &str) -> f32 {
    walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = walls[0];
    let mean = walls.iter().sum::<f32>() / walls.len() as f32;
    let p50 = percentile(walls, 50.0);
    let p99 = percentile(walls, 99.0);
    eprintln!(
        "{label}: min={min:.4} mean={mean:.4} p50={p50:.4} p99={p99:.4} (ms)"
    );
    p50
}

#[test]
#[ignore]
fn bench_qb_wmma_isolated() -> eyre::Result<()> {
    install_panic_handler()?;

    let iters: usize = std::env::var("BENCH_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let batch: u32 = std::env::var("BENCH_B")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(64);

    // qb shape.
    let n_rows: u32 = 32768; // Q_FLAT
    let k: u32 = 1024; // N_LORA_Q
    let blocks = k / Q8_0_BLOCK_ELEMS;

    let dgpu = pick_dgpu()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    eprintln!("device: {} ({arch})  M={n_rows} K={k} B={batch}", dgpu.id);
    dgpu.set_current()?;

    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let wmma = Q8_0MatvecWmma::for_arch(&arch)?;
    let stream = Stream::new(dgpu.id)?;

    // Weight bytes (content irrelevant for timing — just valid layout).
    let bb = Q8_0_BLOCK_BYTES as usize;
    let w_len = (n_rows as usize) * (blocks as usize) * bb;
    let mut w_dev: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, w_len)?;
    w_dev.copy_from_host(&vec![1u8; w_len])?;

    let xq_len = (batch as usize) * (k as usize);
    let mut xq_dev: DeviceBuffer<i8> = DeviceBuffer::new(dgpu.id, xq_len)?;
    xq_dev.copy_from_host(&vec![1i8; xq_len])?;
    let xs_len = (batch as usize) * (blocks as usize);
    let mut xs_dev: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, xs_len)?;
    xs_dev.copy_from_host(&vec![1.0f32; xs_len])?;

    let out_len = (batch as usize) * (n_rows as usize);
    let mut out_dp4a: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, out_len)?;
    let mut out_wmma: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, out_len)?;

    let time = |stream: &Stream,
                f: &mut dyn FnMut(&Stream) -> eyre::Result<()>|
     -> eyre::Result<f32> {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(stream)?;
        f(stream)?;
        end.record(stream)?;
        stream.synchronize()?;
        Ok(Event::elapsed_ms(&start, &end)?)
    };

    // Warmup both.
    for _ in 0..warmup {
        q8.matvec_batched(&stream, &mut out_dp4a, &w_dev, &xq_dev, &xs_dev, n_rows, k, batch)?;
        wmma.gemm(&stream, &mut out_wmma, &w_dev, &xq_dev, &xs_dev, n_rows, k, batch)?;
    }
    stream.synchronize()?;

    // Interleave the A/B per iteration so both see the same thermal envelope.
    let mut dp4a_ms = Vec::with_capacity(iters);
    let mut wmma_ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        dp4a_ms.push(time(&stream, &mut |s| {
            q8.matvec_batched(s, &mut out_dp4a, &w_dev, &xq_dev, &xs_dev, n_rows, k, batch)
        })?);
        wmma_ms.push(time(&stream, &mut |s| {
            wmma.gemm(s, &mut out_wmma, &w_dev, &xq_dev, &xs_dev, n_rows, k, batch)
        })?);
    }

    eprintln!();
    let dp4a_p50 = stats(&mut dp4a_ms, "dp4a matvec_batched");
    let wmma_p50 = stats(&mut wmma_ms, "wmma gemm         ");
    eprintln!(
        "\nqb p50 speedup (dp4a / wmma) = {:.2}x  ({dp4a_p50:.4} ms -> {wmma_p50:.4} ms)",
        dp4a_p50 / wmma_p50
    );
    Ok(())
}
