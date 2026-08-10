//! Isolated timings for the dense K-quant decode matvecs vs the Q8_0
//! incumbent on their real shapes. Sizes the dp4a-formulation work: the
//! correctness-first scalar-f32 gemv kernels (q4/q5/q6 dense) are
//! suspected of eating the unsloth mix's decode BW win.
//!
//! Shapes: head [K=4096 → 129280] (Q4_K vs Q8_0), shexp gate [4096→2048]
//! (Q5_K), shexp down [2048→4096] (Q6_K), q_a [4096→1024] (Q5_K).
//!
//! Run: nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_kquant_dense_isolated -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::q4_k_dense::Q4_KDenseMatvec;
use v4flash_kernels::q5_k_dense::Q5_KDenseMatvec;
use v4flash_kernels::q6_k_dense::Q6_KDenseMatvec;
use v4flash_kernels::dense_gemm::DenseGemmDp4a;
use v4flash_kernels::q8_0::Q8_0Matvec;

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201"))
}

fn median(stream: &Stream, iters: usize, mut f: impl FnMut() -> eyre::Result<()>) -> eyre::Result<f32> {
    for _ in 0..10 {
        f()?;
    }
    stream.synchronize()?;
    let mut w = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = Event::new()?;
        let e = Event::new()?;
        s.record(stream)?;
        f()?;
        e.record(stream)?;
        stream.synchronize()?;
        w.push(Event::elapsed_ms(&s, &e)?);
    }
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(w[w.len() / 2])
}

#[test]
#[ignore]
fn bench_kquant_dense() -> eyre::Result<()> {
    install_panic_handler()?;
    let d = pick_dgpu()?;
    d.set_current()?;
    let arch = d.properties()?.gcn_arch_name;
    let stream = Stream::new(d.id)?;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let q4 = Q4_KDenseMatvec::for_arch(&arch)?;
    let q5 = Q5_KDenseMatvec::for_arch(&arch)?;
    let q6 = Q6_KDenseMatvec::for_arch(&arch)?;

    // (label, rows, k, kernel-tag, block_bytes)
    let cases: &[(&str, u32, u32, char, usize)] = &[
        ("head Q4_K   [4096->129280]", 129280, 4096, '4', 144),
        ("head Q8_0   [4096->129280]", 129280, 4096, '8', 34),
        ("shexp Q5_K  [4096->2048]", 2048, 4096, '5', 176),
        ("shexp Q8_0  [4096->2048]", 2048, 4096, '8', 34),
        ("shdown Q6_K [2048->4096]", 4096, 2048, '6', 210),
        ("shdown Q8_0 [2048->4096]", 4096, 2048, '8', 34),
        ("q_a Q5_K    [4096->1024]", 1024, 4096, '5', 176),
        ("q_a Q8_0    [4096->1024]", 1024, 4096, '8', 34),
    ];
    let gemm = DenseGemmDp4a::for_arch(&arch)?;
    for &(label, rows, k, tag, bb) in cases {
        let wbytes = if tag == '8' {
            (rows as usize) * (k as usize / 32) * 34
        } else {
            (rows as usize) * (k as usize / 256) * bb
        };
        let mut w: DeviceBuffer<u8> = DeviceBuffer::new(d.id, wbytes)?;
        w.fill_zero()?;
        let mut x: DeviceBuffer<f32> = DeviceBuffer::new(d.id, k as usize)?;
        x.fill_zero()?;
        let mut xq: DeviceBuffer<i8> = DeviceBuffer::new(d.id, k as usize)?;
        let mut xs: DeviceBuffer<f32> = DeviceBuffer::new(d.id, (k / 32) as usize)?;
        xq.fill_zero()?;
        xs.fill_zero()?;
        let mut out: DeviceBuffer<f32> = DeviceBuffer::new(d.id, rows as usize)?;
        let ms = median(&stream, 50, || match tag {
            '8' => q8.matvec(&stream, &mut out, &w, &xq, &xs, rows, k),
            '4' => q4.matvec(&stream, &mut out, &w, &x, rows, k),
            '5' => q5.matvec(&stream, &mut out, &w, &x, rows, k),
            '6' => q6.matvec(&stream, &mut out, &w, &x, rows, k),
            _ => unreachable!(),
        })?;
        eprintln!(
            "{label}: {ms:.4} ms   ({:.0} GB/s weights)",
            wbytes as f64 / 1e9 / (ms as f64 / 1e3)
        );
        // dp4a GEMM at B=1 (Q8_K activation) — candidate decode path for
        // the compute-bound scalar gemvs. Same weight bytes.
        if tag == '5' || tag == '6' || tag == '4' {
            let dt = match tag {
                '4' => v4flash_core::gguf::GgufType::Q4_K,
                '5' => v4flash_core::gguf::GgufType::Q5_K,
                _ => v4flash_core::gguf::GgufType::Q6_K,
            };
            let mut xq8k: DeviceBuffer<u8> = DeviceBuffer::new(d.id, (k as usize / 256) * 292)?;
            xq8k.fill_zero()?;
            let mut outg: DeviceBuffer<f32> = DeviceBuffer::new(d.id, rows as usize)?;
            let ms_g = median(&stream, 50, || {
                gemm.gemm(&stream, dt, &mut outg, &w, &xq8k, 1, rows, k / 256)
            })?;
            eprintln!(
                "  └ gemm@B=1: {ms_g:.4} ms   ({:.0} GB/s weights)",
                wbytes as f64 / 1e9 / (ms_g as f64 / 1e3)
            );
        }
    }
    Ok(())
}
