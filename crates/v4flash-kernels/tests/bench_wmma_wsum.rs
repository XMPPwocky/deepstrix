//! De-risk prototype for converting attention Phase B (weighted sum over V)
//! to WMMA. Validates a RDNA4 16x16x16 f16 WMMA GEMM against a host f32
//! reference, then A/Bs it against the tuned f32 Phase-B kernel at the shapes
//! that matter (M=64 heads, N=512 head_dim, K=n_keys at 8K and 64K).
//!
//! Run: `cargo test --release -p v4flash-kernels --test bench_wmma_wsum -- --ignored --nocapture`

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::wmma_wsum::WmmaWsum;

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 dGPU"))
}

/// Minimal round-to-nearest f32 -> f16 bit conversion. Test inputs are in
/// [-1,1], so no inf/subnormal edge cases — good enough for a correctness
/// check at f16 tolerance.
fn f32_to_f16_bits(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (x & 0x007f_ffff) as i32;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (14 - exp);
        return sign | (m as u16);
    } else if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let m = (mant >> 13) as u16;
    let round = ((mant >> 12) & 1) as u16;
    (sign | ((exp as u16) << 10) | m) + round
}

// Cheap deterministic PRNG so the test is reproducible without a dep.
struct Lcg(u64);
impl Lcg {
    fn next_f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // [-1,1)
    }
}

fn host_ref(w: &[f32], v: &[f32], inv: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += w[mi * k + ki] * v[ki * n + ni];
            }
            out[mi * n + ni] = acc * inv[mi];
        }
    }
    out
}

#[test]
#[ignore]
fn wmma_layout_probe() -> eyre::Result<()> {
    install_panic_handler();
    let dev = pick_dgpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let wsum = WmmaWsum::for_arch(&arch)?;
    let stream = Stream::new(dev.id)?;

    let mut draw: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, 32 * 8)?;
    wsum.launch_layout_probe(&stream, &mut draw)?;
    stream.synchronize()?;
    let mut raw = vec![0.0f32; 32 * 8];
    draw.copy_to_host(&mut raw)?;

    eprintln!("=== C accumulator layout: (lane,e) -> (m,n) where C[m][n]=m*100+n ===");
    for lane in 0..32usize {
        let mut row = format!("lane {lane:2}: ");
        for e in 0..8usize {
            let v = raw[lane * 8 + e];
            let iv = v.round() as i32;
            let (m, n) = (iv / 100, iv % 100);
            row.push_str(&format!("e{e}=({m:2},{n:2}) "));
        }
        eprintln!("{row}");
    }
    Ok(())
}

#[test]
#[ignore]
fn wmma_wsum_correctness() -> eyre::Result<()> {
    install_panic_handler();
    let dev = pick_dgpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let wsum = WmmaWsum::for_arch(&arch)?;
    let stream = Stream::new(dev.id)?;

    // Small, host-verifiable. K=40 exercises the partial-tile tail.
    let (m, n, k) = (32usize, 64usize, 40usize);
    let mut rng = Lcg(0x1234_5678);
    let w: Vec<f32> = (0..m * k).map(|_| rng.next_f().abs()).collect(); // weights >= 0
    let v: Vec<f32> = (0..k * n).map(|_| rng.next_f()).collect();
    let inv: Vec<f32> = (0..m).map(|_| 0.5 + 0.5 * rng.next_f().abs()).collect();

    let ref_out = host_ref(&w, &v, &inv, m, n, k);

    let w16: Vec<u16> = w.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let v16: Vec<u16> = v.iter().map(|&x| f32_to_f16_bits(x)).collect();

    let mut dw: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, w16.len())?;
    let mut dv: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, v16.len())?;
    let mut dinv: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, inv.len())?;
    let mut dout: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, m * n)?;
    dw.copy_from_host(&w16)?;
    dv.copy_from_host(&v16)?;
    dinv.copy_from_host(&inv)?;

    wsum.launch_wmma(&stream, &mut dout, &dw, &dv, &dinv, m as u32, n as u32, k as u32)?;
    stream.synchronize()?;

    let mut got = vec![0.0f32; m * n];
    dout.copy_to_host(&mut got)?;

    // Relative L2 norm is the standard GEMM accuracy metric — robust to the
    // pointwise rel-error blowup on near-zero outputs (w>=0 · v∈[-1,1] cancel).
    let mut max_abs = 0.0f32;
    let mut err_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for i in 0..m * n {
        let abs = (got[i] - ref_out[i]).abs();
        max_abs = max_abs.max(abs);
        err_sq += (abs as f64) * (abs as f64);
        ref_sq += (ref_out[i] as f64) * (ref_out[i] as f64);
    }
    let rel_l2 = (err_sq / ref_sq).sqrt();
    eprintln!("WMMA wsum correctness (M={m} N={n} K={k}): rel_L2={rel_l2:.4e} max_abs={max_abs:.4e}");
    // f16 inputs vs f32 host ref: rel_L2 ~ few×1e-3 expected. A wrong layout is
    // O(1) — this cleanly separates the two.
    if rel_l2 > 1e-2 {
        return Err(eyre!(
            "WMMA wsum diverges from host f32 ref: rel_L2={rel_l2:.4e} (layout likely wrong)"
        ));
    }
    eprintln!("PASS: WMMA fragment layout validated (rel_L2={rel_l2:.4e}, f16-level accuracy).");
    Ok(())
}

#[test]
#[ignore]
fn wmma_wsum_bench() -> eyre::Result<()> {
    install_panic_handler();
    let dev = pick_dgpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let wsum = WmmaWsum::for_arch(&arch)?;
    let stream = Stream::new(dev.id)?;

    let (m, n) = (64usize, 512usize);
    let depths = [8192usize, 65536usize];
    let n_runs = 7;

    eprintln!("\n=== WMMA wsum vs tuned f32 Phase-B (M={m} heads, N={n} dim) ===");
    eprintln!("  depth(K)   wmma ms   f32 ms   speedup   wmma_eff_TFLOPs");

    for &k in &depths {
        let mut rng = Lcg(0xABCD_0001 ^ k as u64);
        let w: Vec<f32> = (0..m * k).map(|_| rng.next_f().abs()).collect();
        let v: Vec<f32> = (0..k * n).map(|_| rng.next_f()).collect();
        let inv: Vec<f32> = vec![1.0; m];
        let w16: Vec<u16> = w.iter().map(|&x| f32_to_f16_bits(x)).collect();
        let v16: Vec<u16> = v.iter().map(|&x| f32_to_f16_bits(x)).collect();

        let mut dw16: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, w16.len())?;
        let mut dv16: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, v16.len())?;
        let mut dw32: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, w.len())?;
        let mut dv32: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, v.len())?;
        let mut dinv: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, inv.len())?;
        let mut dout: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, m * n)?;
        dw16.copy_from_host(&w16)?;
        dv16.copy_from_host(&v16)?;
        dw32.copy_from_host(&w)?;
        dv32.copy_from_host(&v)?;
        dinv.copy_from_host(&inv)?;

        // warm up both
        wsum.launch_wmma(&stream, &mut dout, &dw16, &dv16, &dinv, m as u32, n as u32, k as u32)?;
        wsum.launch_f32_ref(&stream, &mut dout, &dw32, &dv32, &dinv, m as u32, n as u32, k as u32)?;
        stream.synchronize()?;

        let mut time = |which: u8| -> eyre::Result<f32> {
            let mut best = f32::INFINITY;
            for _ in 0..n_runs {
                let s = Event::new()?;
                let e = Event::new()?;
                s.record(&stream)?;
                if which == 0 {
                    wsum.launch_wmma(&stream, &mut dout, &dw16, &dv16, &dinv, m as u32, n as u32, k as u32)?;
                } else {
                    wsum.launch_f32_ref(&stream, &mut dout, &dw32, &dv32, &dinv, m as u32, n as u32, k as u32)?;
                }
                e.record(&stream)?;
                stream.synchronize()?;
                best = best.min(Event::elapsed_ms(&s, &e)?);
            }
            Ok(best)
        };

        let wmma_ms = time(0)?;
        let f32_ms = time(1)?;
        let macs = (m * n * k) as f64;
        let wmma_tflops = (2.0 * macs / 1.0e12) / (wmma_ms as f64 / 1000.0);
        eprintln!(
            "  {k:>8}   {wmma_ms:7.3}  {f32_ms:7.3}   {:6.2}×   {wmma_tflops:7.2}",
            f32_ms / wmma_ms
        );
    }
    Ok(())
}
