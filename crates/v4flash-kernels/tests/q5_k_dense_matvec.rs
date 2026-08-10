//! Dense Q5_K matvec correctness: GPU `q5_k_dense_gemv{,_batched}` vs a CPU
//! decode of the SAME random block bytes (any byte pattern is a valid Q5_K
//! block, so no quantizer needed — mirror of the iq3_xxs oracle approach).
//!
//! Run: nix develop -c cargo test --release -p v4flash-kernels \
//!     --test q5_k_dense_matvec -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::iq2_xxs_tables::f16_to_f32;
use v4flash_kernels::q5_k_dense::{Q5_KDenseMatvec, Q5_K_DENSE_BLOCK_BYTES};

const QK_K: usize = 256;
const BB: usize = Q5_K_DENSE_BLOCK_BYTES as usize; // 176

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 { (self.next() & 0xff) as u8 }
    fn next_f32(&mut self) -> f32 { (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0 }
}

const F16_SCALES: [u16; 4] = [0x2400, 0x2800, 0x2c00, 0x3000];

fn get_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// CPU decode of one Q5_K block into 256 f32 (ggml dequantize_row_q5_K).
fn dequant_q5k(blk: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
    let scales = &blk[4..16];
    let qh = &blk[16..48];
    let qs = &blk[48..176];
    for g in 0..4 {
        let (sc1, m1) = get_scale_min(2 * g, scales);
        let (sc2, m2) = get_scale_min(2 * g + 1, scales);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;
        let u1 = 1u8 << (2 * g);
        let u2 = 1u8 << (2 * g + 1);
        for l in 0..32 {
            let q = qs[g * 32 + l];
            out[g * 64 + l] =
                d1 * ((q & 0x0F) + if qh[l] & u1 != 0 { 16 } else { 0 }) as f32 - min1;
            out[g * 64 + 32 + l] =
                d2 * ((q >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 }) as f32 - min2;
        }
    }
}

fn run_on(device: Device) -> eyre::Result<()> {
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let stream = Stream::new(device.id)?;
    let q5 = Q5_KDenseMatvec::for_arch(&arch)?;

    let k = 4096usize;
    let n_rows = 2048usize;
    let n_super = k / QK_K;
    let batch = 3usize;

    let mut rng = Lcg::new(0x51c0de);
    let mut w = vec![0u8; n_rows * n_super * BB];
    for r in 0..n_rows {
        for b in 0..n_super {
            let o = (r * n_super + b) * BB;
            let dd = F16_SCALES[(rng.next() & 3) as usize].to_le_bytes();
            let dm = F16_SCALES[(rng.next() & 3) as usize].to_le_bytes();
            w[o..o + 2].copy_from_slice(&dd);
            w[o + 2..o + 4].copy_from_slice(&dm);
            for i in 4..BB {
                w[o + i] = rng.next_byte();
            }
        }
    }
    let x: Vec<f32> = (0..batch * k).map(|_| rng.next_f32()).collect();

    // CPU reference.
    let mut want = vec![0f32; batch * n_rows];
    let mut deq = vec![0f32; QK_K];
    for r in 0..n_rows {
        for b in 0..n_super {
            dequant_q5k(&w[(r * n_super + b) * BB..(r * n_super + b + 1) * BB], &mut deq);
            for bt in 0..batch {
                let xs = &x[bt * k + b * QK_K..bt * k + (b + 1) * QK_K];
                let mut s = 0f32;
                for i in 0..QK_K {
                    s += deq[i] * xs[i];
                }
                want[bt * n_rows + r] += s;
            }
        }
    }

    let mut w_d: DeviceBuffer<u8> = DeviceBuffer::new(device.id, w.len())?;
    w_d.copy_from_host(&w)?;
    let mut x_d: DeviceBuffer<f32> = DeviceBuffer::new(device.id, x.len())?;
    x_d.copy_from_host(&x)?;

    // Single-token path.
    let mut out_d: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_rows)?;
    let x0: Vec<f32> = x[..k].to_vec();
    let mut x0_d: DeviceBuffer<f32> = DeviceBuffer::new(device.id, k)?;
    x0_d.copy_from_host(&x0)?;
    q5.matvec(&stream, &mut out_d, &w_d, &x0_d, n_rows as u32, k as u32)?;
    stream.synchronize()?;
    let mut got = vec![0f32; n_rows];
    out_d.copy_to_host(&mut got)?;
    check(&format!("{arch} gemv"), &got, &want[..n_rows])?;

    // Batched path.
    let mut outb_d: DeviceBuffer<f32> = DeviceBuffer::new(device.id, batch * n_rows)?;
    q5.matvec_batched(&stream, &mut outb_d, &w_d, &x_d, n_rows as u32, k as u32, batch as u32)?;
    stream.synchronize()?;
    let mut gotb = vec![0f32; batch * n_rows];
    outb_d.copy_to_host(&mut gotb)?;
    check(&format!("{arch} gemv_batched"), &gotb, &want)?;
    Ok(())
}

fn check(name: &str, got: &[f32], want: &[f32]) -> eyre::Result<()> {
    let mut max_diff = 0f32;
    let mut max_ref = 0f32;
    for (g, w) in got.iter().zip(want) {
        max_diff = max_diff.max((g - w).abs());
        max_ref = max_ref.max(w.abs());
    }
    let rel = max_diff / max_ref.max(1e-30);
    eprintln!("q5_k {name}: max|ref|={max_ref:.3} max_diff={max_diff:.5} rel={rel:.2e}");
    if rel >= 1e-4 {
        return Err(eyre!("q5_k {name} diverges: rel={rel}"));
    }
    Ok(())
}

#[test]
#[ignore]
fn q5_k_dense_matches_cpu_both_gpus() -> eyre::Result<()> {
    install_panic_handler()?;
    let mut ran = 0;
    for d in Device::all()? {
        let arch = d.properties()?.gcn_arch_name;
        if arch.starts_with("gfx1201") || arch.starts_with("gfx1151") {
            run_on(d)?;
            ran += 1;
        }
    }
    if ran == 0 {
        return Err(eyre!("no supported GPU found"));
    }
    Ok(())
}
