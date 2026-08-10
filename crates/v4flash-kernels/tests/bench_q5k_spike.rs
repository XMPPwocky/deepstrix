//! Phase-1 spike: Q5_K dense vs the Q8_0 incumbents on shexp shapes (dGPU —
//! shared expert + q_a live in DgpuLayerWeights).
//!
//! Decode: q5_k_dense_gemv vs q8_0 dp4a matvec at [K=4096 → 2048].
//!   Byte ratio 5.5/8.5 = 0.647 → if both BW-bound, q5k should WIN.
//! Prefill: q5_k_dense_gemv_batched at B=512 vs q8_0_gemm_wmma_lds_tiled —
//!   quantifies the known matvec-batched weight-re-read loss
//!   (forward_prefill.rs:857-859) to size the Phase-2 GEMM work.
//!
//! Run: nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_q5k_spike -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::q5_k_dense::{Q5_KDenseMatvec, Q5_K_DENSE_BLOCK_BYTES};
use v4flash_kernels::q8_0::{Q8_0Matvec, Q8_0MatvecWmma};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 dGPU found"))
}

fn time_it(
    stream: &Stream,
    warmup: usize,
    iters: usize,
    mut f: impl FnMut() -> eyre::Result<()>,
) -> eyre::Result<f32> {
    for _ in 0..warmup {
        f()?;
    }
    stream.synchronize()?;
    let mut walls: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = Event::new()?;
        let e = Event::new()?;
        s.record(stream)?;
        f()?;
        e.record(stream)?;
        stream.synchronize()?;
        walls.push(Event::elapsed_ms(&s, &e)?);
    }
    walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(walls[walls.len() / 2])
}

#[test]
#[ignore]
fn bench_q5k_vs_q8() -> eyre::Result<()> {
    install_panic_handler()?;
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let warmup: usize = 10;
    let b_big: u32 = std::env::var("BENCH_B").ok().and_then(|s| s.parse().ok()).unwrap_or(512);

    let k = 4096u32;
    let n_rows = 2048u32;
    let n_super = k / 256;
    let blocks32 = k / 32;

    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let q5 = Q5_KDenseMatvec::for_arch(&arch)?;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let q8w = Q8_0MatvecWmma::for_arch(&arch)?;

    let w5_bytes = (n_rows * n_super) as usize * Q5_K_DENSE_BLOCK_BYTES as usize;
    let w8_bytes = (n_rows * blocks32) as usize * 34;
    eprintln!("=== Q5_K vs Q8_0 dense spike (dGPU, [K={k} → {n_rows}]) ===");
    eprintln!(
        "weight bytes: q5k {} KiB, q8 {} KiB (ratio {:.3})",
        w5_bytes / 1024,
        w8_bytes / 1024,
        w5_bytes as f64 / w8_bytes as f64
    );

    let mut w5: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, w5_bytes)?;
    let mut w8: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, w8_bytes)?;
    w5.fill_zero()?;
    w8.fill_zero()?;

    // ---- decode (B=1) ----
    let mut x: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, k as usize)?;
    x.fill_zero()?;
    let mut xq: DeviceBuffer<i8> = DeviceBuffer::new(dgpu.id, k as usize)?;
    let mut xscale: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, blocks32 as usize)?;
    xq.fill_zero()?;
    xscale.fill_zero()?;
    let mut out: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_rows as usize)?;

    let t_q8 = time_it(&stream, warmup, iters, || {
        q8.matvec(&stream, &mut out, &w8, &xq, &xscale, n_rows, k)
    })?;
    let t_q5 = time_it(&stream, warmup, iters, || {
        q5.matvec(&stream, &mut out, &w5, &x, n_rows, k)
    })?;
    eprintln!(
        "decode:  q8_0 dp4a matvec {t_q8:.4} ms   q5_k gemv {t_q5:.4} ms   ratio {:.3} (byte ratio 0.647)",
        t_q5 / t_q8
    );

    // ---- prefill (B=b_big) ----
    let mut xb: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, (b_big * k) as usize)?;
    xb.fill_zero()?;
    let mut xqb: DeviceBuffer<i8> = DeviceBuffer::new(dgpu.id, (b_big * k) as usize)?;
    let mut xscaleb: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, (b_big * blocks32) as usize)?;
    xqb.fill_zero()?;
    xscaleb.fill_zero()?;
    let mut outb: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, (b_big * n_rows) as usize)?;

    let t_gemm = time_it(&stream, warmup, iters, || {
        q8w.gemm_lds_tiled(&stream, &mut outb, &w8, &xqb, &xscaleb, n_rows, k, b_big)
    })?;
    let t_q5b = time_it(&stream, warmup, iters, || {
        q5.matvec_batched(&stream, &mut outb, &w5, &xb, n_rows, k, b_big)
    })?;
    let t_q8b = time_it(&stream, warmup, iters, || {
        q8.matvec_batched(&stream, &mut outb, &w8, &xqb, &xscaleb, n_rows, k, b_big)
    })?;
    eprintln!(
        "prefill B={b_big}: q8 WMMA GEMM {t_gemm:.3} ms   q8 matvec_batched {t_q8b:.3} ms   q5_k matvec_batched {t_q5b:.3} ms"
    );
    eprintln!(
        "PREFILL VERDICT: q5k batched is {:.1}× the Q8 GEMM — {}",
        t_q5b / t_gemm,
        if t_q5b / t_gemm > 1.5 {
            "GEMM port required (as planned, Phase 2.5)"
        } else {
            "batched may suffice — re-check in e2e traces"
        }
    );
    Ok(())
}
