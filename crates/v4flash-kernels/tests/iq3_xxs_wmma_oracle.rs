//! Oracle for `iq3_xxs_matvec_par_by_expert_wmma` (f16 WMMA down projection)
//! against a CPU f32 dequant-and-dot reference on the SAME f16-rounded mid
//! activations; the Q8_K kwide2 kernel is run on a Q8_K quantization of the
//! same mid for scale. Layout: 3 experts with 23, 32 and 7 members, chunk 32,
//! B=40 tokens x 6 slots, n_rows=4096, K=2048.
//!
//! Run (iGPU): `cargo test --release -p v4flash-kernels --test iq3_xxs_wmma_oracle -- --ignored --nocapture`

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, N_EMBD, N_EXPERT, N_EXPERT_USED};
use v4flash_kernels::iq2_xxs_tables::{f16_to_f32, KSIGNS_IQ2XS};
use v4flash_kernels::iq3_xxs::{Iq3XxsMatvec, BLOCK_IQ3_XXS_BYTES};
use v4flash_kernels::iq3_xxs_tables::IQ3XXS_GRID;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;
use v4flash_kernels::weight_contract::f32_to_f16_bits;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") { return Ok(d); }
    }
    Err(eyre!("no gfx1151 device"))
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 { (self.next() & 0xff) as u8 }
    fn unit(&mut self) -> f32 { (self.next() as f32) / (u32::MAX as f32) }
}

/// Dequantize one IQ3_XXS row (n_blocks super-blocks) to f32.
fn dequant_row_iq3_xxs(n_blocks: usize, w: &[u8], out: &mut [f32]) {
    for bi in 0..n_blocks {
        let blk = &w[bi * BLOCK_IQ3_XXS_BYTES..(bi + 1) * BLOCK_IQ3_XXS_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let q3 = &blk[2..66];
        for ib32 in 0..8 {
            let o = 66 + 4 * ib32;
            let aux = u32::from_le_bytes([blk[o], blk[o + 1], blk[o + 2], blk[o + 3]]);
            let ls = (2 * (aux >> 28) + 1) as f32;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[q3[ib32 * 8 + 2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[q3[ib32 * 8 + 2 * l + 1] as usize].to_le_bytes();
                let base = bi * 256 + ib32 * 32 + l * 8;
                for j in 0..4 {
                    let m1 = g1[j] as f32 * ls * 0.25 * d;
                    let m2 = g2[j] as f32 * ls * 0.25 * d;
                    out[base + j] = if (signs >> j) & 1 == 1 { -m1 } else { m1 };
                    out[base + 4 + j] = if (signs >> (j + 4)) & 1 == 1 { -m2 } else { m2 };
                }
            }
        }
    }
}

fn quantize_q8_k(x: &[f32], n_blocks: usize, out: &mut [u8]) {
    for bi in 0..n_blocks {
        let xs = &x[bi * 256..(bi + 1) * 256];
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let o = bi * BLOCK_Q8_K_BYTES;
        out[o..o + 4].copy_from_slice(&d.to_le_bytes());
        let mut bsums = [0i16; 16];
        for k in 0..256 {
            let q = (xs[k] * id).round().clamp(-127.0, 127.0) as i8;
            out[o + 4 + k] = q as u8;
            bsums[k / 16] += q as i16;
        }
        for (j, s) in bsums.iter().enumerate() {
            out[o + 260 + 2 * j..o + 262 + 2 * j].copy_from_slice(&s.to_le_bytes());
        }
    }
}

fn report(name: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    let (mut max_diff, mut max_ref, mut sd, mut sr, mut nan) = (0f32, 0f32, 0f64, 0f64, 0usize);
    for (g, w) in got.iter().zip(want) {
        if !g.is_finite() { nan += 1; continue; }
        max_diff = max_diff.max((g - w).abs());
        max_ref = max_ref.max(w.abs());
        sd += ((g - w) as f64).powi(2);
        sr += (*w as f64).powi(2);
    }
    let rel = max_diff / max_ref.max(1e-30);
    let rms = (sd / sr.max(1e-30)).sqrt() as f32;
    eprintln!("{name}: n={} max|ref|={max_ref:.4} max_diff={max_diff:.6} rel={rel:.2e} rms_rel={rms:.2e} non_finite={nan}", got.len());
    if nan > 0 { return (f32::INFINITY, f32::INFINITY); }
    (rel, rms)
}

#[test]
#[ignore]
fn iq3_xxs_wmma_matches_cpu_f32() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq3 = Iq3XxsMatvec::for_arch(&arch)?;

    let n_used = N_EXPERT_USED as usize;
    let n_rows = N_EMBD as usize;           // 4096
    let nb = BLOCKS_Q8K_DOWN_IN as usize;   // 8 → K = 2048
    let k_dim = nb * 256;
    let dbpe = n_rows * nb * BLOCK_IQ3_XXS_BYTES;
    let mut rng = Lcg::new(0x2026_0908_1A3);

    let experts: [(usize, usize); 3] = [(9, 23), (128, 32), (200, 7)];
    let f16_scales: [u16; 4] = [0x2400, 0x2800, 0x2c00, 0x3000];
    let mut w_h = vec![0u8; (N_EXPERT as usize) * dbpe];
    for &(e, _) in &experts {
        for r in 0..n_rows {
            for bi in 0..nb {
                let o = e * dbpe + (r * nb + bi) * BLOCK_IQ3_XXS_BYTES;
                w_h[o..o + 2].copy_from_slice(&f16_scales[(rng.next() & 3) as usize].to_le_bytes());
                for i in 2..BLOCK_IQ3_XXS_BYTES { w_h[o + i] = rng.next_byte(); }
            }
        }
    }

    // mid activations per (b, slot): f32 in [-2,2]/64 (swiglu outputs are
    // O(0.01-1)), rounded to f16 once = THE input for kernel and reference.
    let b: usize = 40;
    let n_mem = b * n_used;
    let mut x_f32 = vec![0f32; n_mem * k_dim];
    for v in x_f32.iter_mut() { *v = (rng.unit() - 0.5) * 4.0 / 64.0; }
    let x16_h: Vec<u16> = x_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let x_ref: Vec<f32> = x16_h.iter().map(|&h| f16_to_f32(h)).collect();
    let xq_stride = nb * BLOCK_Q8_K_BYTES;
    let mut xq_h = vec![0u8; n_mem * xq_stride];
    for m in 0..n_mem {
        quantize_q8_k(&x_ref[m * k_dim..(m + 1) * k_dim], nb, &mut xq_h[m * xq_stride..(m + 1) * xq_stride]);
    }

    let max_per_expert = b;
    let chunk: u32 = 32;
    let mut gc_h = vec![0i32; N_EXPERT as usize];
    let mut em_h = vec![0i32; (N_EXPERT as usize) * max_per_expert];
    let mut wi_h: Vec<i32> = Vec::new();
    let mut touched = Vec::new();
    let mut np = 0usize;
    for &(e, n) in &experts {
        gc_h[e] = n as i32;
        let mut start = 0i32;
        while (start as usize) < n { wi_h.push(((e as i32) << 16) | start); start += chunk as i32; }
        for i in 0..n {
            let bi = np % b;
            let sl = (np / b) % n_used;
            em_h[e * max_per_expert + i] = ((bi as i32) << 16) | (sl as i32);
            touched.push((e, bi, sl));
            np += 1;
        }
    }

    // reference
    let mut wrow = vec![0f32; k_dim];
    let mut want = Vec::with_capacity(touched.len() * n_rows);
    for &(e, bi, sl) in &touched {
        let m = bi * n_used + sl;
        let x = &x_ref[m * k_dim..(m + 1) * k_dim];
        for row in 0..n_rows {
            let go = e * dbpe + row * nb * BLOCK_IQ3_XXS_BYTES;
            dequant_row_iq3_xxs(nb, &w_h[go..go + nb * BLOCK_IQ3_XXS_BYTES], &mut wrow);
            want.push(wrow.iter().zip(x).map(|(w, x)| w * x).sum::<f32>());
        }
    }

    let mut w_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, w_h.len())?;
    w_d.copy_from_host(&w_h)?;
    let mut x16_d: DeviceBuffer<u16> = DeviceBuffer::new(igpu.id, x16_h.len())?;
    x16_d.copy_from_host(&x16_h)?;
    let mut xq_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq_h.len())?;
    xq_d.copy_from_host(&xq_h)?;
    let mut gc_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, gc_h.len())?;
    gc_d.copy_from_host(&gc_h)?;
    let mut em_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, em_h.len())?;
    em_d.copy_from_host(&em_h)?;
    let mut wi_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, wi_h.len())?;
    wi_d.copy_from_host(&wi_h)?;
    let mut part_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_mem * n_rows)?;
    let mut part_h = vec![0f32; n_mem * n_rows];
    let gather = |p: &[f32]| -> Vec<f32> {
        let mut g = Vec::with_capacity(touched.len() * n_rows);
        for &(_, bi, sl) in &touched { let m = bi * n_used + sl; g.extend_from_slice(&p[m * n_rows..(m + 1) * n_rows]); }
        g
    };

    part_d.fill_zero()?;
    iq3.launch_by_expert_kwide2(&stream, &mut part_d, &w_d, &xq_d, &gc_d, &em_d, &wi_d, wi_h.len() as u32,
        dbpe as u32, xq_stride as u32, n_used as u32, max_per_expert as u32, chunk, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    part_d.copy_to_host(&mut part_h)?;
    let (_, rms_kwide) = report("kwide2 (Q8_K mid) vs f32 ref", &gather(&part_h), &want);

    part_d.fill_zero()?;
    iq3.launch_by_expert_wmma(&stream, &mut part_d, &w_d, &x16_d, &gc_d, &em_d, &wi_d, wi_h.len() as u32,
        dbpe as u32, k_dim as u32, n_used as u32, max_per_expert as u32, chunk, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    part_d.copy_to_host(&mut part_h)?;
    let (rel_wmma, rms_wmma) = report("wmma (f16 mid) vs f32 ref", &gather(&part_h), &want);

    let touched_set: std::collections::HashSet<usize> = touched.iter().map(|&(_, bi, sl)| bi * n_used + sl).collect();
    let mut leaks = 0usize;
    for m in 0..n_mem {
        if touched_set.contains(&m) { continue; }
        if part_h[m * n_rows..(m + 1) * n_rows].iter().any(|&v| v != 0.0) { leaks += 1; }
    }
    eprintln!("padded-column leaks: {leaks}");

    let mut bad = Vec::new();
    if !(rel_wmma < 1e-2 && rms_wmma < 2e-3) { bad.push(format!("wmma rel={rel_wmma:.2e} rms={rms_wmma:.2e}")); }
    if !(rms_kwide < 1e-1) { bad.push(format!("kwide2 rms={rms_kwide:.2e}")); }
    if leaks > 0 { bad.push(format!("{leaks} padded columns written")); }
    if !bad.is_empty() { return Err(eyre!("iq3_xxs wmma oracle failed: {}", bad.join(", "))); }
    Ok(())
}
