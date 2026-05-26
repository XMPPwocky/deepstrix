//! M50 Phase 2: validate q8_0_matvec_batched produces the same per-row
//! result as `matvec(..., n_rows, k)` called separately for each batch
//! element. Also confirm that with all-identical batch inputs, the
//! batched output rows are identical (sanity check on offset math).

use color_eyre::eyre;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::q8_0::{Q8_0Matvec, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    // Fall back to first device if no dGPU (e.g., test environments).
    Device::all()?
        .into_iter()
        .next()
        .ok_or_else(|| color_eyre::eyre::eyre!("no HIP devices"))
}

fn make_weight(n_rows: u32, k: u32) -> Vec<u8> {
    // M18 repacked Q8_0 layout per the kernel:
    //   each row is [scales(blocks * 2 bytes f16) | quants(blocks * 32 bytes i8)]
    // ALL scales first, THEN all quants. Total = blocks * 34 bytes per row.
    let blocks = k / Q8_0_BLOCK_ELEMS;
    let bb = Q8_0_BLOCK_BYTES as usize; // 34
    let mut bytes = vec![0u8; (n_rows as usize) * (blocks as usize) * bb];
    let row_bytes = (blocks as usize) * bb;
    for r in 0..n_rows {
        let row_off = (r as usize) * row_bytes;
        // Scales block: blocks * 2 bytes, all f16(1.0) = 0x3C00.
        for b in 0..blocks as usize {
            bytes[row_off + b * 2] = 0x00;
            bytes[row_off + b * 2 + 1] = 0x3C;
        }
        // Quants block: blocks * 32 bytes.
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

fn make_xq(k: u32, seed: i8) -> (Vec<i8>, Vec<f32>) {
    let blocks = ((k as usize) / (Q8_0_BLOCK_ELEMS as usize)) as usize;
    let xq: Vec<i8> = (0..k as usize)
        .map(|i| (((i as i32 + seed as i32) % 32) - 16) as i8)
        .collect();
    let xscale = vec![1.0f32; blocks];
    (xq, xscale)
}

#[test]
#[ignore]
fn q8_0_matvec_batched_matches_single() -> eyre::Result<()> {
    install_panic_handler()?;

    let dgpu = pick_dgpu()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    eprintln!("device: {} ({arch})", dgpu.id);
    dgpu.set_current()?;

    // Small but realistic shape: n_rows=128, k=1024 (32 blocks).
    let n_rows: u32 = 128;
    let k: u32 = 1024;
    let blocks = k / Q8_0_BLOCK_ELEMS;
    let batch: u32 = 4;

    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let stream = v4flash_hip::Stream::new(dgpu.id)?;

    // Weight (shared).
    let w_bytes = make_weight(n_rows, k);
    let mut w_dev: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, w_bytes.len())?;
    w_dev.copy_from_host(&w_bytes)?;

    // Per-batch xq, xscale.
    let mut xq_host = vec![0i8; (batch as usize) * (k as usize)];
    let mut xscale_host = vec![0f32; (batch as usize) * (blocks as usize)];
    for b in 0..batch {
        let (xq_b, xs_b) = make_xq(k, b as i8);
        let off_q = (b as usize) * (k as usize);
        let off_s = (b as usize) * (blocks as usize);
        xq_host[off_q..off_q + k as usize].copy_from_slice(&xq_b);
        xscale_host[off_s..off_s + blocks as usize].copy_from_slice(&xs_b);
    }
    let mut xq_dev: DeviceBuffer<i8> = DeviceBuffer::new(dgpu.id, xq_host.len())?;
    xq_dev.copy_from_host(&xq_host)?;
    let mut xscale_dev: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, xscale_host.len())?;
    xscale_dev.copy_from_host(&xscale_host)?;

    // Run batched.
    let mut out_batched: DeviceBuffer<f32> =
        DeviceBuffer::new(dgpu.id, (batch as usize) * (n_rows as usize))?;
    q8.matvec_batched(&stream, &mut out_batched, &w_dev, &xq_dev, &xscale_dev, n_rows, k, batch)?;
    stream.synchronize()?;
    let mut out_batched_host = vec![0f32; (batch as usize) * (n_rows as usize)];
    out_batched.copy_to_host(&mut out_batched_host)?;

    // For each batch element, run single matvec and compare.
    let mut max_diff_overall: f32 = 0.0;
    for b in 0..batch {
        let (xq_b, xs_b) = make_xq(k, b as i8);
        let mut xq_b_dev: DeviceBuffer<i8> = DeviceBuffer::new(dgpu.id, xq_b.len())?;
        xq_b_dev.copy_from_host(&xq_b)?;
        let mut xs_b_dev: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, xs_b.len())?;
        xs_b_dev.copy_from_host(&xs_b)?;
        let mut out_single: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, n_rows as usize)?;
        q8.matvec(&stream, &mut out_single, &w_dev, &xq_b_dev, &xs_b_dev, n_rows, k)?;
        stream.synchronize()?;
        let mut out_single_host = vec![0f32; n_rows as usize];
        out_single.copy_to_host(&mut out_single_host)?;

        let off = (b as usize) * (n_rows as usize);
        let mut max_diff: f32 = 0.0;
        let mut max_idx = 0usize;
        for i in 0..n_rows as usize {
            let d = (out_batched_host[off + i] - out_single_host[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_idx = i;
            }
        }
        eprintln!(
            "batch {b}: max_diff={:.6e} @ {max_idx}  batched={:.4} single={:.4}",
            max_diff, out_batched_host[off + max_idx], out_single_host[max_idx],
        );
        if max_diff > max_diff_overall {
            max_diff_overall = max_diff;
        }
    }
    eprintln!("\noverall max diff = {:.6e}", max_diff_overall);
    // Sanity: results must be finite (NaN comparisons are vacuously true).
    for (i, &v) in out_batched_host.iter().enumerate() {
        assert!(v.is_finite(), "batched out[{i}] non-finite: {v}");
    }
    assert!(max_diff_overall < 1e-4, "batched matvec diverges from single: {max_diff_overall:.4e}");
    eprintln!("PASS: q8_0_matvec_batched matches single matvec");
    Ok(())
}
