//! Dense Q6_K matvec unit test — validates the HIP `q6_k_dense_gemv` and
//! `q6_k_dense_gemv_batched` kernels against an independent CPU Q6_K decode.
//!
//! Strategy (self-contained, no GGUF / no activation dump):
//!   1. Build a known random f32 weight matrix `[n_rows, K]`.
//!   2. Quantize each 256-element superblock to the standard llama.cpp/ggml
//!      `block_q6_K` byte layout (210 B/superblock) on the CPU.
//!   3. CPU-decode those exact bytes back to f32 and dot with a random `x`
//!      — this is the reference.
//!   4. Run the GPU dense matvec over the same bytes + `x`, compare.
//!
//! Because the CPU reference and the GPU kernel decode the *same* Q6_K
//! bytes with the *same* dequant formula, the reconstructed per-weight
//! values are bit-identical; the only source of divergence is f32
//! accumulation order (warp-tree reduce on GPU vs. sequential CPU sum).
//! That yields ~1e-6..1e-4 relative error in practice, so a 1e-2 relative
//! bound is a comfortable, defensible ceiling that also tolerates the
//! inherent Q6_K lossiness of the round-trip. A real kernel decode bug
//! (wrong nibble/high-bit/scale packing) blows past it by orders of
//! magnitude.
//!
//! NOTE: this test drives the GPU. It is `#[ignore]`-gated and must be run
//! explicitly (and only when the production server is not using the GPUs):
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!       --test q6_k_dense_matvec -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::iq2_xxs_tables::f16_to_f32;
use v4flash_kernels::{Q6_KDenseMatvec, Q6_K_DENSE_BLOCK_BYTES};

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 210;

// ---------------------------------------------------------------------------
// f16 helper (round-to-nearest f32 -> IEEE-754 half bits)
// ---------------------------------------------------------------------------

fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x007f_ffff;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 0xff {
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
// Q6_K quantize / dequant (matches ggml block_q6_K + dequantize_row_q6_K)
// ---------------------------------------------------------------------------

/// Quantize 256 f32 weights into one 210-byte Q6_K superblock.
///
/// Layout: ql[128] | qh[64] | scales[16] (int8) | d (f16). Each int8 scale
/// covers a contiguous 16-element sub-block; element m uses scale[m/16].
/// Correctness of the round-trip (not RD-optimality) is all this needs.
fn quantize_superblock(w: &[f32]) -> [u8; BLOCK_BYTES] {
    assert_eq!(w.len(), QK_K);

    // Per 16-element sub-block float scale = amax / 32 (q in [-32, 31]).
    let mut sub_scale = [0f32; 16];
    for (j, chunk) in w.chunks_exact(16).enumerate() {
        let amax = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        sub_scale[j] = amax / 32.0;
    }

    // Super-block scale d encodes the sub-scales as int8 in [0, 127].
    let max_scale = sub_scale.iter().cloned().fold(0.0f32, f32::max);
    let d = max_scale / 127.0;

    let mut scales = [0i8; 16];
    for j in 0..16 {
        scales[j] = if d > 0.0 {
            (sub_scale[j] / d).round().clamp(0.0, 127.0) as i8
        } else {
            0
        };
    }

    // Signed 6-bit quant codes in [-32, 31], stored unsigned (+32) in [0, 63].
    let mut qu = [0u8; QK_K];
    for m in 0..QK_K {
        let eff = d * scales[m / 16] as f32; // effective per-element scale
        let q = if eff > 0.0 {
            (w[m] / eff).round().clamp(-32.0, 31.0) as i32
        } else {
            0
        };
        qu[m] = (q + 32) as u8; // 0..63
    }

    // Assemble the 210 bytes.
    let mut blk = [0u8; BLOCK_BYTES];
    {
        let (ql, rest) = blk.split_at_mut(128);
        let (qh, rest2) = rest.split_at_mut(64);
        let (sc, dbytes) = rest2.split_at_mut(16);

        // Pack ql[128] / qh[64] as the inverse of the decode.
        for h in 0..2 {
            let by = h * 128; // base element index for this 128-element half
            for l in 0..32 {
                let e1 = qu[by + l] as u16;
                let e2 = qu[by + l + 32] as u16;
                let e3 = qu[by + l + 64] as u16;
                let e4 = qu[by + l + 96] as u16;
                ql[h * 64 + l] = ((e1 & 0x0F) | ((e3 & 0x0F) << 4)) as u8;
                ql[h * 64 + l + 32] = ((e2 & 0x0F) | ((e4 & 0x0F) << 4)) as u8;
                qh[h * 32 + l] = (((e1 >> 4) & 3)
                    | (((e2 >> 4) & 3) << 2)
                    | (((e3 >> 4) & 3) << 4)
                    | (((e4 >> 4) & 3) << 6)) as u8;
            }
        }

        for j in 0..16 {
            sc[j] = scales[j] as u8;
        }
        dbytes.copy_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    blk
}

/// CPU decode of one 210-byte Q6_K superblock into 256 f32 values — the
/// bit-for-bit twin of the kernel's `dot_super_q6kd_f32` dequant.
fn dequant_superblock(blk: &[u8], out: &mut [f32]) {
    assert_eq!(blk.len(), BLOCK_BYTES);
    assert_eq!(out.len(), QK_K);
    let ql = &blk[0..128];
    let qh = &blk[128..192];
    let sc = &blk[192..208]; // int8, read as i8 below
    let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));

    for h in 0..2 {
        let base_y = h * 128;
        for l in 0..32 {
            let is = l >> 4; // l / 16
            let lo = ql[h * 64 + l];
            let hi = ql[h * 64 + l + 32];
            let hb = qh[h * 32 + l];
            let q1 = (((lo & 0x0F) | (((hb >> 0) & 3) << 4)) as i32) - 32;
            let q2 = (((hi & 0x0F) | (((hb >> 2) & 3) << 4)) as i32) - 32;
            let q3 = (((lo >> 4) | (((hb >> 4) & 3) << 4)) as i32) - 32;
            let q4 = (((hi >> 4) | (((hb >> 6) & 3) << 4)) as i32) - 32;
            let s0 = sc[h * 8 + is + 0] as i8 as f32;
            let s2 = sc[h * 8 + is + 2] as i8 as f32;
            let s4 = sc[h * 8 + is + 4] as i8 as f32;
            let s6 = sc[h * 8 + is + 6] as i8 as f32;
            out[base_y + l + 0] = d * s0 * q1 as f32;
            out[base_y + l + 32] = d * s2 * q2 as f32;
            out[base_y + l + 64] = d * s4 * q3 as f32;
            out[base_y + l + 96] = d * s6 * q4 as f32;
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic pseudo-random data (no external rng dep)
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
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

/// Quantize a whole f32 weight matrix `[n_rows, K]` (row-major) to Q6_K and
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
fn q6_k_dense_matvec_correctness() -> eyre::Result<()> {
    install_panic_handler()?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("q6_k dense matvec: using device {} ({arch})", device.id);

    let kernel = Q6_KDenseMatvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Shapes: n_rows deliberately NOT a multiple of 8 (exercises the row>=
    // out_dim guard), K a multiple of 256 (3 superblocks).
    let n_rows: usize = 37;
    let k: usize = 768;
    let n_super = k / QK_K;
    assert_eq!(Q6_K_DENSE_BLOCK_BYTES as usize, BLOCK_BYTES);

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
