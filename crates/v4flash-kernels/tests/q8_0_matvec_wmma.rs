//! WMMA Q8_0 GEMM correctness: `Q8_0MatvecWmma::gemm` vs the proven dp4a
//! `Q8_0Matvec::matvec_batched` on qb-shaped data. The WMMA path folds both
//! dequant scales into f16 operands, so it is NOT bit-identical to the
//! int32-exact dp4a path — we check relative error + cosine similarity
//! (KLdiv-style "close enough", per project guidance), not abs-diff.

use color_eyre::eyre;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::q8_0::{Q8_0Matvec, Q8_0MatvecWmma, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Device::all()?
        .into_iter()
        .next()
        .ok_or_else(|| color_eyre::eyre::eyre!("no HIP devices"))
}

/// Minimal f32→f16 (truncating mantissa). Test scales are small positive
/// normals, so the simple path suffices; and since BOTH kernels read these
/// exact same f16 bytes, the encoder's rounding is irrelevant to the
/// dp4a-vs-WMMA comparison.
fn f16_bits(x: f32) -> [u8; 2] {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = ((b >> 13) & 0x3ff) as u16;
    let h = if exp <= 0 {
        sign
    } else if exp >= 0x1f {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | mant
    };
    h.to_le_bytes()
}

/// M18 repacked Q8_0 row: [scales(blocks*f16)] [quants(blocks*32*i8)], pitch
/// blocks*34. Per-block scale and quants both vary by row/block so the
/// scale-fold path is genuinely exercised.
fn make_weight(n_rows: u32, k: u32) -> Vec<u8> {
    let blocks = k / Q8_0_BLOCK_ELEMS;
    let bb = Q8_0_BLOCK_BYTES as usize;
    let row_bytes = (blocks as usize) * bb;
    let mut bytes = vec![0u8; (n_rows as usize) * row_bytes];
    for r in 0..n_rows {
        let row_off = (r as usize) * row_bytes;
        for b in 0..blocks as usize {
            // Realistic Q8_0 scale magnitude ~0.01-0.05, varying.
            let s = 0.012 + 0.03 * (((r as usize * 7 + b * 3) % 11) as f32 / 11.0);
            let sb = f16_bits(s);
            bytes[row_off + b * 2] = sb[0];
            bytes[row_off + b * 2 + 1] = sb[1];
        }
        let q_off = row_off + (blocks as usize) * 2;
        for b in 0..blocks as usize {
            for j in 0..32 {
                let v = (((r as i32 * 13 + b as i32 * 7 + j as i32) % 64) - 32) as i8;
                bytes[q_off + b * 32 + j] = v as u8;
            }
        }
    }
    bytes
}

fn make_xq(k: u32, seed: i32) -> (Vec<i8>, Vec<f32>) {
    let blocks = (k / Q8_0_BLOCK_ELEMS) as usize;
    let xq: Vec<i8> = (0..k as usize)
        .map(|i| (((i as i32 + seed * 5) % 32) - 16) as i8)
        .collect();
    let xscale: Vec<f32> = (0..blocks)
        .map(|b| 0.008 + 0.02 * (((b + seed as usize) % 7) as f32 / 7.0))
        .collect();
    (xq, xscale)
}

#[test]
#[ignore]
fn q8_0_wmma_matches_dp4a() -> eyre::Result<()> {
    install_panic_handler()?;

    let dgpu = pick_dgpu()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    eprintln!("device: {} ({arch})", dgpu.id);
    dgpu.set_current()?;

    // qb shape: M=32768, K=1024 (32 blocks), B=64. Also non-tile-aligned M/B
    // to exercise boundary masks, and B>128 (multiple chunk-loop passes) plus
    // a non-aligned wide B to exercise the chunk boundary.
    for &(n_rows, k, batch) in &[
        (200u32, 1024u32, 50u32),
        (32768u32, 1024u32, 64u32),
        (32768u32, 1024u32, 256u32),
        (512u32, 1024u32, 200u32),
    ] {
        let blocks = k / Q8_0_BLOCK_ELEMS;
        eprintln!("\n=== M={n_rows} K={k} B={batch} (blocks={blocks}) ===");

        let q8 = Q8_0Matvec::for_arch(&arch)?;
        let wmma = Q8_0MatvecWmma::for_arch(&arch)?;
        let stream = v4flash_hip::Stream::new(dgpu.id)?;

        let w_bytes = make_weight(n_rows, k);
        let mut w_dev: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, w_bytes.len())?;
        w_dev.copy_from_host(&w_bytes)?;

        let mut xq_host = vec![0i8; (batch as usize) * (k as usize)];
        let mut xscale_host = vec![0f32; (batch as usize) * (blocks as usize)];
        for b in 0..batch {
            let (xq_b, xs_b) = make_xq(k, b as i32);
            let off_q = (b as usize) * (k as usize);
            let off_s = (b as usize) * (blocks as usize);
            xq_host[off_q..off_q + k as usize].copy_from_slice(&xq_b);
            xscale_host[off_s..off_s + blocks as usize].copy_from_slice(&xs_b);
        }
        let mut xq_dev: DeviceBuffer<i8> = DeviceBuffer::new(dgpu.id, xq_host.len())?;
        xq_dev.copy_from_host(&xq_host)?;
        let mut xscale_dev: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, xscale_host.len())?;
        xscale_dev.copy_from_host(&xscale_host)?;

        let out_len = (batch as usize) * (n_rows as usize);

        // Reference: dp4a batched (int32-exact dot, f32 scales).
        let mut out_ref: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, out_len)?;
        eprintln!("  [dp4a] launch…");
        q8.matvec_batched(&stream, &mut out_ref, &w_dev, &xq_dev, &xscale_dev, n_rows, k, batch)?;
        stream.synchronize()?;
        eprintln!("  [dp4a] done");
        let mut ref_host = vec![0f32; out_len];
        out_ref.copy_to_host(&mut ref_host)?;

        // WMMA path.
        let mut out_wmma: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, out_len)?;
        eprintln!("  [wmma] launch…");
        wmma.gemm(&stream, &mut out_wmma, &w_dev, &xq_dev, &xscale_dev, n_rows, k, batch)?;
        stream.synchronize()?;
        eprintln!("  [wmma] done");
        let mut wmma_host = vec![0f32; out_len];
        out_wmma.copy_to_host(&mut wmma_host)?;

        // Stats: relative error + cosine similarity. Note dp4a layout is
        // [B, n_rows]; WMMA writes out[gn*out_dim+gm] = [B, M] too.
        let mut dot = 0.0f64;
        let mut nr = 0.0f64;
        let mut nw = 0.0f64;
        let mut max_rel = 0.0f32;
        let mut sum_rel = 0.0f64;
        let mut nonfinite = 0usize;
        for i in 0..out_len {
            let r = ref_host[i];
            let w = wmma_host[i];
            if !w.is_finite() {
                nonfinite += 1;
                continue;
            }
            dot += (r as f64) * (w as f64);
            nr += (r as f64) * (r as f64);
            nw += (w as f64) * (w as f64);
            let denom = r.abs().max(1e-3);
            let rel = (r - w).abs() / denom;
            sum_rel += rel as f64;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        let cos = dot / (nr.sqrt() * nw.sqrt());
        let mean_rel = sum_rel / (out_len as f64);
        eprintln!(
            "cosine={cos:.6}  mean_rel={mean_rel:.4e}  max_rel={max_rel:.4e}  nonfinite={nonfinite}"
        );

        assert_eq!(nonfinite, 0, "WMMA produced non-finite outputs");
        assert!(cos > 0.9999, "cosine similarity too low: {cos:.6}");
        assert!(mean_rel < 1e-2, "mean relative error too high: {mean_rel:.4e}");
    }

    eprintln!("\nPASS: WMMA Q8_0 GEMM matches dp4a within f16 tolerance");
    Ok(())
}
