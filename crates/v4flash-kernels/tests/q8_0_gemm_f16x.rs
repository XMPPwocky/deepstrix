//! Oracle + bench for `q8_0_gemm_wmma_f16x` (Q8_0 x f16, 128x128 RDNA4 WMMA).
//! Oracle: CPU f32 (dequantized Q8_0 weights . f16-rounded activations) on
//! the four production shapes at a small batch; also runs the old
//! lds_tiled kernel on a Q8_0 quantization of the same activations for
//! scale. Bench (`--ignored bench_q8_0_gemm_f16x`): B=BENCH_B (512) on all
//! four shapes, old vs new, with TFLOPS and % of the 194.6-TFLOP f16 peak.
//! dGPU only.
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::q8_0::{Q8_0MatvecWmma, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
use v4flash_kernels::weight_contract::f32_to_f16_bits;
use v4flash_kernels::iq2_xxs_tables::f16_to_f32;

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") { return Ok(d); }
    }
    Err(eyre!("no gfx1201 device"))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 32) as u32 }
    fn unit(&mut self) -> f32 { (self.next() as f32) / (u32::MAX as f32) }
}

/// (name, K per group, M per group, n_groups)
const SHAPES: [(&str, u32, u32, u32); 4] = [
    ("q_b   (32768x1024)", 1024, 32768, 1),
    ("kv    (512x4096)", 4096, 512, 1),
    ("out_a (8 x 1024x4096)", 4096, 1024, 8),
    ("out_b (4096x8192)", 8192, 4096, 1),
];

fn make_weight(rng: &mut Lcg, rows: u32, k: u32) -> Vec<u8> {
    let blocks = (k / Q8_0_BLOCK_ELEMS) as usize;
    let rb = blocks * Q8_0_BLOCK_BYTES as usize;
    let mut w = vec![0u8; rows as usize * rb];
    for r in 0..rows as usize {
        for b in 0..blocks {
            let s = 0.004 + 0.02 * rng.unit();
            w[r * rb + 2 * b..r * rb + 2 * b + 2].copy_from_slice(&f32_to_f16_bits(s).to_le_bytes());
        }
        let q = r * rb + 2 * blocks;
        for j in 0..blocks * 32 { w[q + j] = ((rng.next() % 255) as i32 - 127) as i8 as u8; }
    }
    w
}

/// Quantize one activation row to Q8_0 (per-32 scale, symmetric).
fn q8_0_quant_row(x: &[f32], xq: &mut [i8], xs: &mut [f32]) {
    for (b, blk) in x.chunks(32).enumerate() {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        xs[b] = d;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (j, &v) in blk.iter().enumerate() { xq[b * 32 + j] = (v * id).round().clamp(-127.0, 127.0) as i8; }
    }
}

fn cpu_ref(w: &[u8], k: u32, m: u32, g: u32, x: &[f32], batch: usize) -> Vec<f32> {
    let blocks = (k / 32) as usize;
    let rb = blocks * 34;
    let ldx = (g * k) as usize;
    let ldo = (g * m) as usize;
    let mut out = vec![0f32; batch * ldo];
    let mut wrow = vec![0f32; k as usize];
    for gi in 0..g as usize {
        for r in 0..m as usize {
            let base = (gi * m as usize + r) * rb;
            for b in 0..blocks {
                let d = f16_to_f32(u16::from_le_bytes([w[base + 2 * b], w[base + 2 * b + 1]]));
                for j in 0..32 { wrow[b * 32 + j] = (w[base + 2 * blocks + b * 32 + j] as i8) as f32 * d; }
            }
            for n in 0..batch {
                let xr = &x[n * ldx + gi * k as usize..n * ldx + (gi + 1) * k as usize];
                out[n * ldo + gi * m as usize + r] = wrow.iter().zip(xr).map(|(a, b)| a * b).sum();
            }
        }
    }
    out
}

#[test]
#[ignore]
fn q8_0_gemm_f16x_matches_cpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let q8 = Q8_0MatvecWmma::for_arch(&arch)?;
    let mut rng = Lcg(0x2026_0908_0808);
    let batch: usize = 40;   // exercises the partial 128-tile guard
    let mut failed = Vec::new();
    for &(name, k, m, g) in &SHAPES {
        // keep the CPU reference tractable: q_b uses 4096 rows of its 32768
        let m_eff = if m > 4096 { 4096 } else { m };
        let w_h = make_weight(&mut rng, m_eff * g, k);
        let ldx = (g * k) as usize;
        let x_f32: Vec<f32> = (0..batch * ldx).map(|_| (rng.unit() - 0.5) * 4.0).collect();
        let x16_h: Vec<u16> = x_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let x_ref: Vec<f32> = x16_h.iter().map(|&h| f16_to_f32(h)).collect();
        let want = cpu_ref(&w_h, k, m_eff, g, &x_ref, batch);
        let mut w_d: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, w_h.len())?; w_d.copy_from_host(&w_h)?;
        let mut x16_d: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, x16_h.len())?; x16_d.copy_from_host(&x16_h)?;
        let ldo = (g * m_eff) as usize;
        let mut out_d: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, batch * ldo)?; out_d.fill_zero()?;
        q8.gemm_f16x(&stream, &mut out_d, &w_d, &x16_d, k, m_eff, g, batch as u32, ldx as u32)?;
        stream.synchronize()?;
        let mut got = vec![0f32; batch * ldo]; out_d.copy_to_host(&mut got)?;
        let (mut md, mut mr, mut sd, mut sr) = (0f32, 0f32, 0f64, 0f64);
        for (a, b) in got.iter().zip(&want) { md = md.max((a - b).abs()); mr = mr.max(b.abs()); sd += ((a - b) as f64).powi(2); sr += (*b as f64).powi(2); }
        let rel = md / mr.max(1e-30); let rms = (sd / sr.max(1e-30)).sqrt();
        // old kernel on Q8_0-quantized activations, for scale (plain GEMM shapes only)
        let mut old_note = String::new();
        if g == 1 {
            let blocks = (k / 32) as usize;
            let mut xq_h = vec![0i8; batch * k as usize]; let mut xs_h = vec![0f32; batch * blocks];
            for n in 0..batch { q8_0_quant_row(&x_ref[n * k as usize..(n + 1) * k as usize], &mut xq_h[n * k as usize..(n + 1) * k as usize], &mut xs_h[n * blocks..(n + 1) * blocks]); }
            let mut xq_d: DeviceBuffer<i8> = DeviceBuffer::new(dgpu.id, xq_h.len())?; xq_d.copy_from_host(&xq_h)?;
            let mut xs_d: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, xs_h.len())?; xs_d.copy_from_host(&xs_h)?;
            let mut o2: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, batch * ldo)?; o2.fill_zero()?;
            q8.gemm_lds_tiled(&stream, &mut o2, &w_d, &xq_d, &xs_d, m_eff, k, batch as u32)?;
            stream.synchronize()?;
            let mut g2 = vec![0f32; batch * ldo]; o2.copy_to_host(&mut g2)?;
            let (mut sd2, mut sr2) = (0f64, 0f64);
            for (a, b) in g2.iter().zip(&want) { sd2 += ((a - b) as f64).powi(2); sr2 += (*b as f64).powi(2); }
            old_note = format!("  | lds_tiled(Q8_0 acts) rms_rel={:.2e}", (sd2 / sr2.max(1e-30)).sqrt());
        }
        eprintln!("{name:26} f16x vs CPU f32: max_rel={rel:.2e} rms_rel={rms:.2e}{old_note}");
        if !(rel < 5e-3 && rms < 1e-3) { failed.push(name); }
    }
    if !failed.is_empty() { return Err(eyre!("q8_0_gemm_f16x oracle failed: {failed:?}")); }
    Ok(())
}

#[test]
#[ignore]
fn bench_q8_0_gemm_f16x() -> eyre::Result<()> {
    install_panic_handler()?;
    let batch: u32 = std::env::var("BENCH_B").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let q8 = Q8_0MatvecWmma::for_arch(&arch)?;
    let mut rng = Lcg(0xBEEF_2026_0908);
    const PEAK_TFLOPS: f64 = 194.6;
    let mut tot_old = 0f64; let mut tot_new = 0f64;
    for &(name, k, m, g) in &SHAPES {
        let w_h = make_weight(&mut rng, m * g, k);
        let mut w_d: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, w_h.len())?; w_d.copy_from_host(&w_h)?;
        let ldx = (g * k) as usize; let ldo = (g * m) as usize;
        // BENCH_PAD=1: pad the activation row pitch by 64 halves (128 B) so
        // rows do not sit at a power-of-two stride.
        let pad: usize = if std::env::var("BENCH_PAD").map(|v| v == "1").unwrap_or(false) { 64 } else { 0 };
        let pitch = ldx + pad;
        let mut x16_d: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, batch as usize * pitch)?;
        x16_d.copy_from_host(&(0..batch as usize * pitch).map(|_| f32_to_f16_bits((rng.unit() - 0.5) * 2.0)).collect::<Vec<_>>())?;
        let blocks = (k / 32) as usize;
        let mut xq_d: DeviceBuffer<i8> = DeviceBuffer::new(dgpu.id, batch as usize * ldx)?; xq_d.fill_zero()?;
        let mut xs_d: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, batch as usize * blocks * g as usize)?; xs_d.fill_zero()?;
        let mut out_d: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, batch as usize * ldo)?;
        let flops = 2.0 * batch as f64 * (g * m) as f64 * k as f64;
        let mut time = |which: u32| -> eyre::Result<f32> {
            let run = |out_d: &mut DeviceBuffer<f32>| -> eyre::Result<()> {
                if which == 0 {
                    if g == 1 { q8.gemm_lds_tiled(&stream, out_d, &w_d, &xq_d, &xs_d, m, k, batch) }
                    else { q8.gemm_lds_tiled_grouped(&stream, out_d, &w_d, &xq_d, &xs_d, k, m, g, batch) }
                } else {
                    q8.gemm_f16x(&stream, out_d, &w_d, &x16_d, k, m, g, batch, pitch as u32)
                }
            };
            run(&mut out_d)?; stream.synchronize()?;
            let mut ws: Vec<f32> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let s = Event::new()?; let e = Event::new()?;
                s.record(&stream)?; run(&mut out_d)?; e.record(&stream)?; stream.synchronize()?;
                ws.push(Event::elapsed_ms(&s, &e)?);
            }
            ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(ws[ws.len() / 2])
        };
        let t_old = time(0)?; let t_new = time(1)?;
        tot_old += t_old as f64; tot_new += t_new as f64;
        let tf = |ms: f32| flops / (ms as f64 * 1e-3) / 1e12;
        eprintln!("{name:26} B={batch}: lds_tiled {t_old:7.3} ms ({:5.1} TF, {:4.1}% peak) | f16x {t_new:7.3} ms ({:5.1} TF, {:4.1}% peak) | {:.2}x",
            tf(t_old), 100.0 * tf(t_old) / PEAK_TFLOPS, tf(t_new), 100.0 * tf(t_new) / PEAK_TFLOPS, t_old / t_new);
    }
    eprintln!("per-layer total (4 GEMMs, B={batch}): lds_tiled {tot_old:.3} ms | f16x {tot_new:.3} ms | {:.2}x  -> x43 layers x2 lanes: {:.0} vs {:.0} ms per 1024-token chunk",
        tot_old / tot_new, tot_old * 86.0, tot_new * 86.0);
    Ok(())
}
