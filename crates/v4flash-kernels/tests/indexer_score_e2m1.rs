//! Packed-E2M1 indexer score kernels versus their f16 originals: bit-identical
//! scores (`docs/E2M1_INDEXER_KEYS_2026-09.md`, step 2 gate), plus an
//! isolated timing of each pair.
//!
//! Keys are produced by the REAL chain on both sides (indexer_qat on random
//! rows -> f16 cache via f16_roundtrip + comp_kv_append; -> packed cache via
//! index_kv_append_e2m1), so the f16 rows are exactly what production holds.
//! Per-token n_idx varies (tails of -inf), n_idx_stride > n_idx_max.
//!
//! WMMA kernels run on gfx12 only (the dGPU); the naive kernel runs on both
//! GPUs (it is the gfx1151 oracle reference).
//!
//! `cargo test -p v4flash-kernels --release --test indexer_score_e2m1 -- --nocapture`

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::index_kv_e2m1::{E2M1_KEY_DIM, E2M1_KEY_ROW_BYTES};
use v4flash_kernels::weight_contract::f32_to_f16_bits;
use v4flash_kernels::{CompKvAppend, F16Roundtrip, IndexKvE2m1, IndexerQat, IndexerScore, IndexerScoreWmma};

const N_HEAD: usize = 64;
const DIM: usize = E2M1_KEY_DIM;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn diff_report(label: &str, a: &[f32], b: &[f32]) -> eyre::Result<()> {
    let mut bad = 0usize;
    let mut first = None;
    for i in 0..a.len() {
        if a[i].to_bits() != b[i].to_bits() {
            bad += 1;
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    if bad != 0 {
        let i = first.unwrap();
        return Err(eyre!("{label}: {bad} / {} scores differ (first at {i}: f16 {} packed {})", a.len(), a[i], b[i]));
    }
    println!("  {label}: {} scores bit-identical", a.len());
    Ok(())
}

fn time<F: FnMut() -> eyre::Result<()>>(stream: &Stream, iters: u32, mut f: F) -> eyre::Result<f64> {
    for _ in 0..3 {
        f()?;
    }
    stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / iters as f64)
}

#[test]
fn indexer_score_e2m1_matches_f16() -> eyre::Result<()> {
    install_panic_handler()?;
    let n_idx_max: u32 = std::env::var("E2M1_N_IDX").ok().and_then(|s| s.parse().ok()).unwrap_or(12000);
    let batch: u32 = std::env::var("E2M1_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let iters: u32 = std::env::var("E2M1_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let n_idx_stride = n_idx_max + 1000;
    let mut rng = Rng(0x0DDB_A11_5EED_0001);

    for d in Device::all()? {
        d.set_current()?;
        let arch = d.properties()?.gcn_arch_name.clone();
        let is_gfx12 = arch.starts_with("gfx12");
        println!("device {} ({arch}) n_idx_max={n_idx_max} batch={batch}", d.id);
        let stream = Stream::new(d.id)?;
        let qat = IndexerQat::for_arch(&arch)?;
        let f16rt = F16Roundtrip::for_arch(&arch)?;
        let append = CompKvAppend::for_arch(&arch)?;
        let packer = IndexKvE2m1::for_arch(&arch)?;
        let naive = IndexerScore::for_arch(&arch)?;

        // Keys through the real chain: random rows (mixed magnitudes) -> QAT.
        let n_rows = n_idx_max as usize;
        let mut rows = vec![0f32; n_rows * DIM];
        for r in 0..n_rows {
            let mag = 10f32.powf(rng.unit() * 3.0 - 1.5);
            for v in rows[r * DIM..(r + 1) * DIM].iter_mut() {
                *v = (rng.unit() * 2.0 - 1.0) * mag;
            }
        }
        let mut x = DeviceBuffer::<f32>::new(d.id, rows.len())?;
        x.copy_from_host(&rows)?;
        qat.launch(&stream, &mut x, n_rows as u32)?;
        let mut packed = DeviceBuffer::<u8>::new(d.id, n_rows * E2M1_KEY_ROW_BYTES)?;
        let mut kv16 = DeviceBuffer::<u16>::new(d.id, rows.len())?;
        // The batched producers index rows on grid.y (<= 65535; production
        // chunks are <= 256 rows), so upload in slices.
        const SLICE: usize = 16384;
        let mut r0 = 0usize;
        while r0 < n_rows {
            let n = (n_rows - r0).min(SLICE);
            let xs = x.slice_view(r0 * DIM, n * DIM);
            packer.launch_append_batched(&stream, &mut packed, &xs, r0 as u32, n as u32)?;
            r0 += n;
        }
        f16rt.launch(&stream, &mut x, rows.len() as u32)?;
        r0 = 0;
        while r0 < n_rows {
            let n = (n_rows - r0).min(SLICE);
            let xs = x.slice_view(r0 * DIM, n * DIM);
            append.launch_batched(&stream, &mut kv16, &xs, r0 as u32, DIM as u32, n as u32)?;
            r0 += n;
        }
        stream.synchronize()?;

        // Q / head weights per token; q16 (f16 cast) for the gemm kernel.
        let b = batch as usize;
        let q_host: Vec<f32> = (0..b * N_HEAD * DIM).map(|_| (rng.unit() * 2.0 - 1.0) * 0.5).collect();
        let hw_host: Vec<f32> = (0..b * N_HEAD).map(|_| rng.unit() * 0.2).collect();
        let q16_host: Vec<u16> = q_host.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let mut q = DeviceBuffer::<f32>::new(d.id, q_host.len())?;
        q.copy_from_host(&q_host)?;
        let mut q16 = DeviceBuffer::<u16>::new(d.id, q16_host.len())?;
        q16.copy_from_host(&q16_host)?;
        let mut hw = DeviceBuffer::<f32>::new(d.id, hw_host.len())?;
        hw.copy_from_host(&hw_host)?;
        // Per-token n_idx: a spread around n_idx_max (some below, none above).
        let n_idx_host: Vec<u32> = (0..b).map(|i| (n_idx_max - (i as u32 * 37) % 3000).max(600)).collect();
        let mut n_idx = DeviceBuffer::<u32>::new(d.id, b)?;
        n_idx.copy_from_host(&n_idx_host)?;

        let poison = vec![-12345.0f32; b * n_idx_stride as usize];
        let mut s_a = DeviceBuffer::<f32>::new(d.id, poison.len())?;
        let mut s_b = DeviceBuffer::<f32>::new(d.id, poison.len())?;
        let mut ha = vec![0f32; poison.len()];
        let mut hb = vec![0f32; poison.len()];

        // Naive (both GPUs): token 0, all rows.
        s_a.copy_from_host(&poison)?;
        s_b.copy_from_host(&poison)?;
        let q0 = q.slice_view(0, N_HEAD * DIM);
        let hw0 = hw.slice_view(0, N_HEAD);
        naive.launch(&stream, &mut s_a, &q0, &hw0, &kv16, n_idx_max, N_HEAD as u32, DIM as u32)?;
        naive.launch_e2m1(&stream, &mut s_b, &q0, &hw0, &packed, n_idx_max, N_HEAD as u32, DIM as u32)?;
        stream.synchronize()?;
        s_a.copy_to_host(&mut ha)?;
        s_b.copy_to_host(&mut hb)?;
        diff_report("naive", &ha[..n_idx_max as usize], &hb[..n_idx_max as usize])?;

        if !is_gfx12 {
            continue;
        }
        let wmma = IndexerScoreWmma::for_arch(&arch)?;

        // mw (B=1 decode).
        s_a.copy_from_host(&poison)?;
        s_b.copy_from_host(&poison)?;
        wmma.launch_mw(&stream, &mut s_a, &q0, &hw0, &kv16, n_idx_max)?;
        wmma.launch_mw_e2m1(&stream, &mut s_b, &q0, &hw0, &packed, n_idx_max)?;
        stream.synchronize()?;
        s_a.copy_to_host(&mut ha)?;
        s_b.copy_to_host(&mut hb)?;
        diff_report("mw (B=1)", &ha[..n_idx_max as usize], &hb[..n_idx_max as usize])?;
        let t_f16 = time(&stream, iters, || wmma.launch_mw(&stream, &mut s_a, &q0, &hw0, &kv16, n_idx_max))?;
        let t_e2m1 = time(&stream, iters, || wmma.launch_mw_e2m1(&stream, &mut s_b, &q0, &hw0, &packed, n_idx_max))?;
        println!("    mw B=1 n={n_idx_max}: f16 {t_f16:.1} us, e2m1 {t_e2m1:.1} us ({:+.1}%)", (t_e2m1 - t_f16) / t_f16 * 100.0);

        // batched mw.
        s_a.copy_from_host(&poison)?;
        s_b.copy_from_host(&poison)?;
        wmma.launch_batched_mw(&stream, &mut s_a, &q, &hw, &kv16, &n_idx, n_idx_max, n_idx_stride, batch)?;
        wmma.launch_batched_mw_e2m1(&stream, &mut s_b, &q, &hw, &packed, &n_idx, n_idx_max, n_idx_stride, batch)?;
        stream.synchronize()?;
        s_a.copy_to_host(&mut ha)?;
        s_b.copy_to_host(&mut hb)?;
        diff_report("batched_mw (incl. -inf tails)", &ha, &hb)?;
        let t_f16 = time(&stream, iters, || wmma.launch_batched_mw(&stream, &mut s_a, &q, &hw, &kv16, &n_idx, n_idx_max, n_idx_stride, batch))?;
        let t_e2m1 = time(&stream, iters, || wmma.launch_batched_mw_e2m1(&stream, &mut s_b, &q, &hw, &packed, &n_idx, n_idx_max, n_idx_stride, batch))?;
        println!("    batched_mw B={batch}: f16 {t_f16:.1} us, e2m1 {t_e2m1:.1} us ({:+.1}%)", (t_e2m1 - t_f16) / t_f16 * 100.0);

        // gemm (the prefill production path).
        s_a.copy_from_host(&poison)?;
        s_b.copy_from_host(&poison)?;
        wmma.launch_batched_gemm(&stream, &mut s_a, &q16, &hw, &kv16, &n_idx, n_idx_max, n_idx_stride, batch, 0)?;
        wmma.launch_batched_gemm_e2m1(&stream, &mut s_b, &q16, &hw, &packed, &n_idx, n_idx_max, n_idx_stride, batch, 0)?;
        stream.synchronize()?;
        s_a.copy_to_host(&mut ha)?;
        s_b.copy_to_host(&mut hb)?;
        diff_report("gemm (incl. -inf tails)", &ha, &hb)?;
        let t_f16 = time(&stream, iters, || wmma.launch_batched_gemm(&stream, &mut s_a, &q16, &hw, &kv16, &n_idx, n_idx_max, n_idx_stride, batch, 0))?;
        let t_e2m1 = time(&stream, iters, || wmma.launch_batched_gemm_e2m1(&stream, &mut s_b, &q16, &hw, &packed, &n_idx, n_idx_max, n_idx_stride, batch, 0))?;
        println!("    gemm B={batch}: f16 {t_f16:.1} us, e2m1 {t_e2m1:.1} us ({:+.1}%)", (t_e2m1 - t_f16) / t_f16 * 100.0);
    }
    Ok(())
}
