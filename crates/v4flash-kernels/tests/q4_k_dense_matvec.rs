//! Dense Q4_K matvec unit test — validates the HIP `q4_k_dense_gemv` and
//! `q4_k_dense_gemv_batched` kernels against an independent CPU Q4_K decode.
//!
//! Strategy (self-contained, no GGUF / no activation dump):
//!   1. Build a known random f32 weight matrix `[n_rows, K]`.
//!   2. Quantize each 256-element superblock to the standard llama.cpp/ggml
//!      `block_q4_K` byte layout (144 B/superblock) on the CPU.
//!   3. CPU-decode those exact bytes back to f32 and dot with a random `x`
//!      — this is the reference.
//!   4. Run the GPU dense matvec over the same bytes + `x`, compare.
//!
//! Because the CPU reference and the GPU kernel decode the *same* Q4_K
//! bytes with the *same* dequant formula, the reconstructed per-weight
//! values are bit-identical; the only source of divergence is f32
//! accumulation order (warp-tree reduce on GPU vs. sequential CPU sum).
//! That yields ~1e-6..1e-4 relative error in practice, so a 1e-2 relative
//! bound is a comfortable, defensible ceiling that also tolerates the
//! inherent Q4_K lossiness of the round-trip. The bound is intentionally
//! generous — a real kernel decode bug (wrong nibble/scale/min packing)
//! blows past it by orders of magnitude.
//!
//! NOTE: this test drives the GPU. It is `#[ignore]`-gated and must be run
//! explicitly (and only when the production server is not using the GPUs):
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!       --test q4_k_dense_matvec -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::iq2_xxs_tables::f16_to_f32;
use v4flash_kernels::{Q4_KDenseMatvec, Q4_K_DENSE_BLOCK_BYTES};

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 144;

// ---------------------------------------------------------------------------
// f16 helpers
// ---------------------------------------------------------------------------

/// Round-to-nearest f32 -> IEEE-754 half bits. Inputs in this test are
/// modest positive magnitudes (no inf/subnormal edge cases), and the exact
/// stored bits are decoded identically on both CPU and GPU, so this only
/// needs to produce a *valid* half — its rounding precision does not affect
/// the GPU-vs-CPU comparison.
fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x007f_ffff;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 0xff {
        // inf / nan
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    } else if e <= 0 {
        if e < -10 {
            return sign; // underflow -> signed zero
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half_mant = (m >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let mut result = sign | half_mant;
        if (m & round_bit) != 0 && ((m & (round_bit - 1)) != 0 || (half_mant & 1) != 0) {
            result += 1;
        }
        return result;
    }
    let half_mant = (mant >> 13) as u16;
    let mut result = sign | ((e as u16) << 10) | half_mant;
    if (mant & 0x0000_1000) != 0 && ((mant & 0x0000_0fff) != 0 || (half_mant & 1) != 0) {
        result += 1;
    }
    result
}

// ---------------------------------------------------------------------------
// Q4_K quantize / dequant (matches ggml block_q4_K + get_scale_min_k4)
// ---------------------------------------------------------------------------

/// `get_scale_min_k4` — decode the 6-bit (d_sub, m_sub) for sub-block j.
fn get_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Quantize 256 f32 weights into one 144-byte Q4_K superblock.
fn quantize_superblock(w: &[f32]) -> [u8; BLOCK_BYTES] {
    assert_eq!(w.len(), QK_K);
    // Per sub-block scale (>=0) and effective min offset mn = -lmin (>=0).
    let mut sc = [0f32; 8];
    let mut mn = [0f32; 8];
    for (j, chunk) in w.chunks_exact(32).enumerate() {
        let mut lo = 0.0f32;
        let mut hi = 0.0f32;
        for &v in chunk {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        // Force lmin<=0<=lmax so the reconstruction offset (subtracted) is
        // representable with a non-negative m code.
        sc[j] = (hi - lo) / 15.0;
        mn[j] = -lo; // >= 0
    }

    let dmax = sc.iter().cloned().fold(0.0f32, f32::max);
    let mmax = mn.iter().cloned().fold(0.0f32, f32::max);
    let d = dmax / 63.0;
    let dmin = mmax / 63.0;

    let mut d_code = [0u8; 8];
    let mut m_code = [0u8; 8];
    for j in 0..8 {
        d_code[j] = if d > 0.0 {
            (sc[j] / d).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
        m_code[j] = if dmin > 0.0 {
            (mn[j] / dmin).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
    }

    // 4-bit quant codes per element.
    let mut q = [0u8; QK_K];
    for j in 0..8 {
        let scale = sc[j];
        let lmin = -mn[j];
        for l in 0..32 {
            let i = j * 32 + l;
            q[i] = if scale > 0.0 {
                (((w[i] - lmin) / scale).round().clamp(0.0, 15.0)) as u8
            } else {
                0
            };
        }
    }

    // Assemble the 144 bytes.
    let mut blk = [0u8; BLOCK_BYTES];
    blk[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    blk[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());

    // Pack scales[12] (inverse of get_scale_min_k4).
    let scales = &mut blk[4..16];
    for j in 0..4 {
        scales[j] |= d_code[j] & 0x3F;
        scales[j + 4] |= m_code[j] & 0x3F;
    }
    for j in 4..8 {
        scales[j + 4] = (d_code[j] & 0x0F) | ((m_code[j] & 0x0F) << 4);
        scales[j - 4] |= (d_code[j] >> 4) << 6;
        scales[j] |= (m_code[j] >> 4) << 6;
    }

    // Pack qs[128]: for group g in 0..4, byte 32g+l holds element (64g+l) in
    // the low nibble and element (64g+32+l) in the high nibble.
    let qs = &mut blk[16..144];
    for g in 0..4 {
        for l in 0..32 {
            let lo = q[64 * g + l] & 0x0F;
            let hi = q[64 * g + 32 + l] & 0x0F;
            qs[32 * g + l] = lo | (hi << 4);
        }
    }
    blk
}

/// CPU decode of one 144-byte Q4_K superblock into 256 f32 values — the
/// bit-for-bit twin of the kernel's `dot_super_q4kd_f32` dequant.
fn dequant_superblock(blk: &[u8], out: &mut [f32]) {
    assert_eq!(blk.len(), BLOCK_BYTES);
    assert_eq!(out.len(), QK_K);
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
    let scales = &blk[4..16];
    let qs = &blk[16..144];
    for g in 0..4 {
        let (sc1, m1) = get_scale_min(2 * g, scales);
        let (sc2, m2) = get_scale_min(2 * g + 1, scales);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;
        for l in 0..32 {
            let byte = qs[32 * g + l];
            out[64 * g + l] = d1 * (byte & 0x0F) as f32 - min1;
            out[64 * g + 32 + l] = d2 * (byte >> 4) as f32 - min2;
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic pseudo-random data (no external rng dep)
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        // xorshift* — plenty for test data.
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let u = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as u32; // 24 bits
        (u as f32 / (1u32 << 24) as f32) * 2.0 - 1.0 // [-1, 1)
    }
}

fn pick_device() -> eyre::Result<Device> {
    let devices = Device::all()?;
    for d in &devices {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre!("no HIP devices"))
}

/// Quantize a whole f32 weight matrix `[n_rows, K]` (row-major) to Q4_K and
/// return (weight_bytes, cpu_dequant_weights). Both are row-major.
fn quantize_matrix(w: &[f32], n_rows: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % QK_K, 0);
    let n_super = k / QK_K;
    let mut bytes = vec![0u8; n_rows * n_super * BLOCK_BYTES];
    let mut deq = vec![0f32; n_rows * k];
    let mut tmp = vec![0f32; QK_K];
    for r in 0..n_rows {
        for s in 0..n_super {
            let src = &w[r * k + s * QK_K..r * k + s * QK_K + QK_K];
            let blk = quantize_superblock(src);
            let off = (r * n_super + s) * BLOCK_BYTES;
            bytes[off..off + BLOCK_BYTES].copy_from_slice(&blk);
            dequant_superblock(&blk, &mut tmp);
            deq[r * k + s * QK_K..r * k + s * QK_K + QK_K].copy_from_slice(&tmp);
        }
    }
    (bytes, deq)
}

#[test]
#[ignore]
fn q4_k_dense_matvec_correctness() -> eyre::Result<()> {
    install_panic_handler()?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("q4_k dense matvec: using device {} ({arch})", device.id);

    let kernel = Q4_KDenseMatvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Shapes: n_rows deliberately NOT a multiple of 8 (exercises the row>=
    // out_dim guard), K a multiple of 256 (3 superblocks).
    let n_rows: usize = 37;
    let k: usize = 768;
    let n_super = k / QK_K;
    assert_eq!(Q4_K_DENSE_BLOCK_BYTES as usize, BLOCK_BYTES);

    // Build weight matrix + quantize.
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let mut w = vec![0f32; n_rows * k];
    for v in w.iter_mut() {
        *v = rng.next_f32() * 0.5; // modest magnitudes
    }
    let (w_bytes, w_deq) = quantize_matrix(&w, n_rows, k);
    assert_eq!(w_bytes.len(), n_rows * n_super * BLOCK_BYTES);

    // Upload weight once (shared by both cases).
    let mut d_w: DeviceBuffer<u8> = DeviceBuffer::new(device.id, w_bytes.len())?;
    d_w.copy_from_host(&w_bytes)?;

    // ----- Non-batched case -----
    let mut x = vec![0f32; k];
    for v in x.iter_mut() {
        *v = rng.next_f32();
    }
    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, k)?;
    d_x.copy_from_host(&x)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_rows)?;

    kernel.matvec(&stream, &mut d_out, &d_w, &d_x, n_rows as u32, k as u32)?;
    stream.synchronize()?;
    let mut got = vec![0f32; n_rows];
    d_out.copy_to_host(&mut got)?;

    let mut max_rel = 0f32;
    let mut max_abs = 0f32;
    for r in 0..n_rows {
        let mut expect = 0f32;
        for c in 0..k {
            expect += w_deq[r * k + c] * x[c];
        }
        let d = (got[r] - expect).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / expect.abs().max(1e-6));
    }
    eprintln!(
        "non-batched: n_rows={n_rows}, K={k}, max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
    );
    const REL_TOL: f32 = 1.0e-2;
    assert!(
        max_rel < REL_TOL,
        "non-batched max_rel {max_rel:.3e} >= tol {REL_TOL:.3e}"
    );

    // ----- Batched case (B=3) -----
    let batch: usize = 3;
    let mut xb = vec![0f32; batch * k];
    for v in xb.iter_mut() {
        *v = rng.next_f32();
    }
    let mut d_xb: DeviceBuffer<f32> = DeviceBuffer::new(device.id, batch * k)?;
    d_xb.copy_from_host(&xb)?;
    let mut d_outb: DeviceBuffer<f32> = DeviceBuffer::new(device.id, batch * n_rows)?;

    kernel.matvec_batched(
        &stream,
        &mut d_outb,
        &d_w,
        &d_xb,
        n_rows as u32,
        k as u32,
        batch as u32,
    )?;
    stream.synchronize()?;
    let mut gotb = vec![0f32; batch * n_rows];
    d_outb.copy_to_host(&mut gotb)?;

    let mut max_rel_b = 0f32;
    let mut max_abs_b = 0f32;
    for b in 0..batch {
        for r in 0..n_rows {
            let mut expect = 0f32;
            for c in 0..k {
                expect += w_deq[r * k + c] * xb[b * k + c];
            }
            let d = (gotb[b * n_rows + r] - expect).abs();
            max_abs_b = max_abs_b.max(d);
            max_rel_b = max_rel_b.max(d / expect.abs().max(1e-6));
        }
    }
    eprintln!(
        "batched B={batch}: max_abs={max_abs_b:.3e}, max_rel={max_rel_b:.3e}"
    );
    assert!(
        max_rel_b < REL_TOL,
        "batched max_rel {max_rel_b:.3e} >= tol {REL_TOL:.3e}"
    );

    Ok(())
}
