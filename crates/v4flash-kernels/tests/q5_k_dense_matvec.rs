//! Dense Q5_K matvec correctness: GPU `q5_k_dense_gemv{,_batched}` vs a CPU
//! decode of the SAME random block bytes (any byte pattern is a valid Q5_K
//! block, so no quantizer needed — mirror of the iq3_xxs oracle approach).
//!
//! Run: nix develop -c cargo test --release -p v4flash-kernels \
//!     --test q5_k_dense_matvec -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::dense_gemm::DenseGemmDp4a;
use v4flash_kernels::iq2_xxs_tables::f16_to_f32;
use v4flash_kernels::weight_contract::f32_to_f16_bits;
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
    let batch = 96usize; // >=64 so the WMMA GEMM path is covered too

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

    // dp4a GEMM (prefill path): Q8_K activations. Host-quantize x to Q8_K
    // and build its own CPU reference (integer dot × scales) so the GEMM's
    // math is checked against the same numbers it actually computes.
    let gemm = DenseGemmDp4a::for_arch(&arch)?;
    let n_blk = k / 256;
    let mut xq8k = vec![0u8; batch * n_blk * 292];
    let mut want_g = vec![0f32; batch * n_rows];
    let mut deq = vec![0f32; QK_K];
    let mut xq_int = vec![0i8; batch * k];
    let mut xd = vec![0f32; batch * n_blk];
    for bt in 0..batch {
        for blk in 0..n_blk {
            let xs = &x[bt * k + blk * 256..bt * k + (blk + 1) * 256];
            let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let o = (bt * n_blk + blk) * 292;
            xq8k[o..o + 4].copy_from_slice(&d.to_le_bytes());
            let mut bsums = [0i16; 16];
            for i in 0..256 {
                let q = (xs[i] * id).round().clamp(-127.0, 127.0) as i8;
                xq_int[bt * k + blk * 256 + i] = q;
                xq8k[o + 4 + i] = q as u8;
                bsums[i / 16] += q as i16;
            }
            for (j, sv) in bsums.iter().enumerate() {
                xq8k[o + 260 + 2 * j..o + 262 + 2 * j].copy_from_slice(&sv.to_le_bytes());
            }
            xd[bt * n_blk + blk] = d;
        }
    }
    for r in 0..n_rows {
        for blk in 0..n_blk {
            dequant_q5k(&w[(r * n_blk + blk) * BB..(r * n_blk + blk + 1) * BB], &mut deq);
            for bt in 0..batch {
                let d = xd[bt * n_blk + blk];
                let mut s = 0f32;
                for i in 0..QK_K {
                    s += deq[i] * d * xq_int[bt * k + blk * 256 + i] as f32;
                }
                want_g[bt * n_rows + r] += s;
            }
        }
    }
    // f16-operand reference (what the WMMA kernel actually computes).
    let mut want_h = vec![0f32; batch * n_rows];
    for r in 0..n_rows {
        for blk in 0..n_blk {
            dequant_q5k(&w[(r * n_blk + blk) * BB..(r * n_blk + blk + 1) * BB], &mut deq);
            for bt in 0..batch {
                let d = xd[bt * n_blk + blk];
                let mut s = 0f32;
                for i in 0..QK_K {
                    let a = f16_to_f32(f32_to_f16_bits(deq[i]));
                    let b = f16_to_f32(f32_to_f16_bits(d * xq_int[bt * k + blk * 256 + i] as f32));
                    s += a * b;
                }
                want_h[bt * n_rows + r] += s;
            }
        }
    }
    let mut xq8k_d: DeviceBuffer<u8> = DeviceBuffer::new(device.id, xq8k.len())?;
    xq8k_d.copy_from_host(&xq8k)?;
    let mut outg_d: DeviceBuffer<f32> = DeviceBuffer::new(device.id, batch * n_rows)?;
    gemm.gemm(&stream, v4flash_core::gguf::GgufType::Q5_K, &mut outg_d, &w_d, &xq8k_d,
        batch as u32, n_rows as u32, n_blk as u32)?;
    stream.synchronize()?;
    let mut gotg = vec![0f32; batch * n_rows];
    outg_d.copy_to_host(&mut gotg)?;
    // gfx1201 routes B>=64 to the WMMA kernel (f16 operands, f32
    // accumulate — same precision class as the shipped Q8_0 WMMA GEMM);
    // gfx1151 has no WMMA module and stays on the exact-integer dp4a.
    if arch.starts_with("gfx1201") {
        // Loose vs exact f32: quantifies the f16-operand rounding budget.
        check_tol(&format!("{arch} gemm wmma (vs exact f32)"), &gotg, &want_g, 1e-3)?;
        // Tight vs the f16-operand reference: the kernel's own math.
        check(&format!("{arch} gemm wmma (vs f16-operand ref)"), &gotg, &want_h)?;
    } else {
        check(&format!("{arch} gemm dp4a (vs exact f32)"), &gotg, &want_g)?;
    }
    // dp4a path: B<64 routes there by the dispatcher's own rule (its
    // integer math is exact, so it holds the tight tolerance).
    let b_small = 32usize;
    let mut outd_d: DeviceBuffer<f32> = DeviceBuffer::new(device.id, b_small * n_rows)?;
    gemm.gemm(&stream, v4flash_core::gguf::GgufType::Q5_K, &mut outd_d, &w_d, &xq8k_d,
        b_small as u32, n_rows as u32, n_blk as u32)?;
    stream.synchronize()?;
    let mut gotd = vec![0f32; b_small * n_rows];
    outd_d.copy_to_host(&mut gotd)?;
    check(&format!("{arch} gemm_dp4a (B={b_small})"), &gotd, &want_g[..b_small * n_rows])?;
    Ok(())
}

fn check(name: &str, got: &[f32], want: &[f32]) -> eyre::Result<()> {
    check_tol(name, got, want, 1e-4)
}

fn check_tol(name: &str, got: &[f32], want: &[f32], tol: f32) -> eyre::Result<()> {
    let mut max_diff = 0f32;
    let mut max_ref = 0f32;
    for (g, w) in got.iter().zip(want) {
        max_diff = max_diff.max((g - w).abs());
        max_ref = max_ref.max(w.abs());
    }
    let rel = max_diff / max_ref.max(1e-30);
    eprintln!("q5_k {name}: max|ref|={max_ref:.3} max_diff={max_diff:.5} rel={rel:.2e}");
    if rel >= tol {
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
