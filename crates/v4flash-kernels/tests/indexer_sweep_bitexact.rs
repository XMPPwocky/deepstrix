//! Bit-exactness of the 2026-09-26 E_indexer sweep kernels against the kernels
//! they replace, on synthetic data (no model load; < 90 MB of device memory
//! per test, so it runs beside a live hub):
//!
//!   1. `s2_w8n8_pf_hw` vs `indexer_score_wmma_batched_mw_e2m1` (scores incl.
//!      the -inf tail stamps), real packed E2M1 keys, per-row n_comp incl. 0 /
//!      1 / partial tiles, keys_base_per null and per-row, negative head
//!      weights, plus `launch_batched_mw_e2m1_rows` (`V41_IDX_SCORE_QREG`).
//!   2. `topk_select_v3_u8` vs `indexer_topk_select_batched_ilp` (selected[]
//!      and done[]): ragged n, float4-chunk edges, n = 0, all-equal (tie
//!      append), -inf masked, and a forced give-up row (done = 0), plus the
//!      `launch_batched_sel` wrapper on the small path (`V41_IDX_TOPK_HYBRID`).
//!   3. `gather_u4_r1` vs `indexer_gather_batched` (the whole destination incl.
//!      untouched sentinel rows): b = 1..8 and 16, top_k 512 / 333 / 1,
//!      comp_base_per null and per-row, plus `launch_batched_rows`
//!      (`V41_IDX_GATHER_B128`, b >= 4).
//!   4. `candidate_threshold_ilp` vs `candidate_threshold` on random / tie-heavy
//!      / all-equal / half -inf block scores, per-row n, plus `launch_build`
//!      (`V41_CAND_THRESH_ILP`, b <= 8).
//!
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test
//! indexer_sweep_bitexact -- --ignored --test-threads=1 --nocapture`. With every
//! knob = 0 the wrapper comparisons pit the OLD path against the old symbols.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, launch_kernel, sys, Device, DeviceBuffer, LaunchConfig, Stream};
use v4flash_kernels::candidate_blocks::CandidateBlocks;
use v4flash_kernels::config::{CANDIDATE_BLOCK_SIZE, CANDIDATE_TOPK_BLOCKS, N_INDEXER_HEAD, N_INDEXER_HEAD_DIM};
use v4flash_kernels::index_kv_e2m1::{IndexKvE2m1, E2M1_KEY_ROW_BYTES};
use v4flash_kernels::indexer::{IndexerGather, IndexerScoreWmma, IndexerTopkBitonic};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Device::all()?.into_iter().next().ok_or_else(|| eyre!("no HIP devices"))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFFFF) as f32 / 16777216.0
    }
    fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len().max(1))?;
    if !host.is_empty() {
        d.copy_from_host(host)?;
    }
    Ok(d)
}

fn download<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}

const SENTINEL: f32 = f32::from_bits(0x7FC0_BEEF);

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ------------------------------------------------------------------ 1. score

#[test]
#[ignore]
fn score_qreg_matches_mw_e2m1() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    if !arch.starts_with("gfx1201") {
        eprintln!("SKIP: WMMA indexer score is gfx12-only (device is {arch})");
        return Ok(());
    }
    dev.set_current()?;
    let id = dev.id;
    let stream = Stream::new(id)?;
    let isw = IndexerScoreWmma::for_arch(&arch)?;
    let packer = IndexKvE2m1::for_arch(&arch)?;
    let mut rng = Lcg(0x5EED_E001);
    let dim = N_INDEXER_HEAD_DIM as usize;
    let nh = N_INDEXER_HEAD as usize;

    // Real packed keys through the production packer: 70000 rows.
    const ROWS: usize = 70_000;
    let packed = {
        let rows: Vec<f32> = (0..ROWS * dim).map(|_| rng.sym(2.0)).collect();
        let d_rows = upload(id, &rows)?;
        let mut packed: DeviceBuffer<u8> = DeviceBuffer::new(id, ROWS * E2M1_KEY_ROW_BYTES)?;
        // The packer indexes rows on grid.y (<= 65535): append in slices.
        const SLICE: usize = 16384;
        let mut r0 = 0usize;
        while r0 < ROWS {
            let n = (ROWS - r0).min(SLICE);
            let xs = d_rows.slice_view(r0 * dim, n * dim);
            packer.launch_append_batched(&stream, &mut packed, &xs, r0 as u32, n as u32)?;
            r0 += n;
        }
        stream.synchronize()?;
        packed
    };

    // (n_per, keys_base_per or none, stride)
    let cases: Vec<(Vec<u32>, Option<Vec<u32>>, u32)> = vec![
        (vec![65536], None, 65636),
        (vec![65536], Some(vec![4000]), 65536),
        (vec![65536, 4113, 1, 0], Some(vec![0, 1000, 5, 60000]), 65540),
        (vec![17, 1025, 30000], None, 30000),
        (vec![1000, 4097, 20000, 16, 1023, 12345, 2, 16384], Some(vec![69000, 5, 40000, 0, 100, 3, 69990, 50000]), 20480),
        (vec![100003 - 40000, 1, 4096], Some(vec![7, 69000, 30000]), 60003 + 1),
    ];
    let mut n_cases = 0usize;
    for (n_per, bases, stride) in &cases {
        let b = n_per.len() as u32;
        let n_max = *n_per.iter().max().unwrap();
        let q_h: Vec<f32> = (0..b as usize * nh * dim).map(|_| rng.sym(1.5)).collect();
        let hw_h: Vec<f32> = (0..b as usize * nh).map(|_| rng.sym(1.0)).collect();
        let q = upload(id, &q_h)?;
        let hw = upload(id, &hw_h)?;
        let d_n = upload(id, n_per)?;
        let d_base = bases.as_ref().map(|v| upload(id, v)).transpose()?;
        let base_ptr: sys::hipDeviceptr_t = d_base.as_ref().map_or(std::ptr::null_mut(), |d| d.raw());
        let n_out = (b * stride) as usize;
        let cfg = LaunchConfig { grid: (n_max.div_ceil(1024), b, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        let s_old = upload(id, &vec![SENTINEL; n_out])?;
        let f = isw.module().get_function("indexer_score_wmma_batched_mw_e2m1")?;
        launch_kernel!(f, cfg, &stream, [s_old.raw(), q.raw(), hw.raw(), packed.raw(), d_n.raw(), *stride, base_ptr])?;
        let s_new = upload(id, &vec![SENTINEL; n_out])?;
        let f = isw.module().get_function("s2_w8n8_pf_hw")?;
        launch_kernel!(f, cfg, &stream, [s_new.raw(), q.raw(), hw.raw(), packed.raw(), d_n.raw(), *stride, base_ptr])?;
        let mut s_wr = upload(id, &vec![SENTINEL; n_out])?;
        isw.launch_batched_mw_e2m1_rows(&stream, &mut s_wr, &q, &hw, &packed, &d_n, n_max, *stride, b, d_base.as_ref())?;
        stream.synchronize()?;
        let (o, n, w) = (bits(&download(&s_old)?), bits(&download(&s_new)?), bits(&download(&s_wr)?));
        let finite = download(&s_old)?.iter().take(n_per[0] as usize).filter(|v| v.is_finite() && **v != 0.0).count();
        let d_n_ = o.iter().zip(&n).filter(|(a, b)| a != b).count();
        let d_w = o.iter().zip(&w).filter(|(a, b)| a != b).count();
        eprintln!("score b={b} n={n_per:?} stride={stride} base={}: twin bit_diff={d_n_}, wrapper bit_diff={d_w} ({finite} nonzero finite scores in row 0)",
            bases.is_some());
        assert!(n_per[0] < 16 || finite > 0, "degenerate reference");
        assert_eq!(d_n_, 0, "s2_w8n8_pf_hw not bit-exact: b={b} n={n_per:?}");
        assert_eq!(d_w, 0, "launch_batched_mw_e2m1_rows not bit-exact: b={b} n={n_per:?}");
        n_cases += 1;
    }
    eprintln!("PASS: {n_cases} score cases bit-exact (incl. -inf stamps and untouched tails)");
    Ok(())
}

// ------------------------------------------------------------- 2. topk select

#[test]
#[ignore]
fn topk_hybrid_matches_select_ilp() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let stream = Stream::new(id)?;
    let tk = IndexerTopkBitonic::for_arch(&arch)?;
    let mut rng = Lcg(0x5EED_E002);
    const TOP_K: u32 = 512;
    // Row lengths (per case, one row each at b = 1..4).
    let cases: Vec<Vec<u32>> = vec![
        vec![1], vec![511], vec![4096], vec![4097], vec![32767, 32771], vec![36863, 36865, 8191, 8193],
        vec![65536, 0, 100003, 4095], vec![131072], vec![235000, 234999], vec![300001], vec![235000, 65536, 131072, 16385],
    ];
    let mut n_cases = 0usize;
    // class 0 random, 1 tie-heavy (0.25 quanta), 2 -inf masked (3/4), 3 all-equal, 4 give-up trap.
    for class in 0..5 {
        for n_per in &cases {
            if class == 4 && n_per.iter().all(|&n| n < 65536) {
                continue;
            }
            let b = n_per.len() as u32;
            let stride = (n_per.iter().max().unwrap().div_ceil(4) * 4).max(4);
            let mut s_h = vec![-3.4028235e38f32; (b * stride) as usize];
            for (bi, &n) in n_per.iter().enumerate() {
                let row = &mut s_h[bi * stride as usize..bi * stride as usize + n as usize];
                match class {
                    0 => row.iter_mut().for_each(|v| *v = rng.sym(10.0)),
                    1 => row.iter_mut().for_each(|v| *v = (rng.below(40) as f32) * 0.25),
                    2 => row.iter_mut().for_each(|v| *v = if rng.below(4) != 0 { -3.4028235e38 } else { rng.sym(10.0) }),
                    3 => row.iter_mut().for_each(|v| *v = 0.5),
                    _ => {
                        // >4096 equal maxima at positions the 2048-key sample never reads
                        // (src = j*n/2048): every threshold attempt counts > CAP -> give up.
                        row.iter_mut().for_each(|v| *v = 0.0);
                        let mut sampled = vec![false; n as usize];
                        for j in 0..2048u64 {
                            let src = (j * n as u64 / 2048) as usize;
                            if src < n as usize { sampled[src] = true; }
                        }
                        let mut put = 0;
                        for p in 0..n as usize {
                            if put >= 5000 { break; }
                            if !sampled[p] {
                                row[p] = 1.0;
                                put += 1;
                            }
                        }
                    }
                }
            }
            let scores = upload(id, &s_h)?;
            let d_n = upload(id, n_per)?;
            let cfg = LaunchConfig { grid: (b, 1, 1), block: (1024, 1, 1), shared_mem_bytes: 0 };
            let run = |sym: &str| -> eyre::Result<(Vec<i32>, Vec<u32>)> {
                let sel = upload(id, &vec![-7i32; (b * TOP_K) as usize])?;
                let done = upload(id, &vec![7u32; b as usize])?;
                let f = tk.module().get_function(sym)?;
                launch_kernel!(f, cfg, &stream, [sel.raw(), done.raw(), scores.raw(), d_n.raw(), stride, TOP_K])?;
                stream.synchronize()?;
                Ok((download(&sel)?, download(&done)?))
            };
            let (sel_o, done_o) = run("indexer_topk_select_batched_ilp")?;
            let (sel_n, done_n) = run("topk_select_v3_u8")?;
            let tag = format!("class={class} n={n_per:?}");
            assert!(done_o.iter().all(|&d| d <= 1), "old kernel left done unwritten: {tag}");
            if class == 4 {
                assert!(done_o.iter().zip(n_per).all(|(&d, &n)| n < 65536 || d == 0), "give-up trap did not trigger: {tag}");
            }
            assert_eq!(done_o, done_n, "done[] differs: {tag}");
            assert_eq!(sel_o, sel_n, "selected[] differs: {tag}");
            eprintln!("select {tag}: identical (done={done_o:?})");
            n_cases += 1;
        }
    }
    // Wrapper on the small path (n_idx_max <= 4096 returns right after the select).
    for &n in &[1u32, 700, 4096] {
        let b = 3u32;
        let stride = 4096u32;
        let s_h: Vec<f32> = (0..b * stride).map(|_| rng.sym(5.0)).collect();
        let scores = upload(id, &s_h)?;
        let d_n = upload(id, &vec![n; b as usize])?;
        let cfg = LaunchConfig { grid: (b, 1, 1), block: (1024, 1, 1), shared_mem_bytes: 0 };
        let sel_o = upload(id, &vec![-7i32; (b * TOP_K) as usize])?;
        let done_o = upload(id, &vec![7u32; b as usize])?;
        let f = tk.module().get_function("indexer_topk_select_batched_ilp")?;
        launch_kernel!(f, cfg, &stream, [sel_o.raw(), done_o.raw(), scores.raw(), d_n.raw(), stride, TOP_K])?;
        let mut sel_w = upload(id, &vec![-7i32; (b * TOP_K) as usize])?;
        let mut done_w = upload(id, &vec![7u32; b as usize])?;
        let mut scratch: DeviceBuffer<u32> = DeviceBuffer::new(id, 4096)?;
        tk.launch_batched_sel(&stream, &mut sel_w, None, &mut scratch, &scores, &d_n, n, stride, 0, TOP_K, b, Some(&mut done_w), true)?;
        stream.synchronize()?;
        assert_eq!(download(&sel_o)?, download(&sel_w)?, "launch_batched_sel selected differs at n={n}");
        assert_eq!(download(&done_o)?, download(&done_w)?, "launch_batched_sel done differs at n={n}");
        n_cases += 1;
    }
    eprintln!("PASS: {n_cases} select cases identical (selected + done)");
    Ok(())
}

// ------------------------------------------------------------------ 3. gather

#[test]
#[ignore]
fn gather_b128_matches_indexer_gather() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let stream = Stream::new(id)?;
    let g = IndexerGather::for_arch(&arch)?;
    let mut rng = Lcg(0x5EED_E003);
    const HD: u32 = 512;
    const STORE_ROWS: u32 = 16384;
    // Arbitrary 16-bit patterns (NaN / Inf / denormal encodings included): a pure copy.
    let store_h: Vec<u16> = (0..STORE_ROWS * HD).map(|_| rng.next() as u16).collect();
    let store = upload(id, &store_h)?;
    let mut n_cases = 0usize;
    for &(b, top_k, per_row) in &[
        (1u32, 512u32, false), (2, 512, true), (3, 512, false), (4, 512, true), (4, 512, false), (5, 333, true),
        (7, 1, false), (8, 512, true), (16, 512, false), (16, 333, true),
    ] {
        let bases: Vec<i32> = (0..b).map(|_| rng.below(8000) as i32).collect();
        let span = if per_row { 8000 } else { STORE_ROWS };
        let mut sel_h = vec![-1i32; (b * top_k) as usize];
        for bi in 0..b as usize {
            // Row 1 is all-sentinel; others have scattered and trailing sentinels.
            if bi == 1 { continue; }
            let valid = top_k - top_k / 5;
            for i in 0..valid as usize {
                sel_h[bi * top_k as usize + i] = if rng.below(10) == 0 { -1 } else { rng.below(span) as i32 };
            }
        }
        let sel = upload(id, &sel_h)?;
        let d_base = upload(id, &bases)?;
        let base_ptr: sys::hipDeviceptr_t = if per_row { d_base.raw() } else { std::ptr::null_mut() };
        let n_out = (b * top_k * HD) as usize;
        let fill = vec![0xBEEFu16; n_out];
        let dst_o = upload(id, &fill)?;
        let f = g.module().get_function("indexer_gather_batched")?;
        let cfg = LaunchConfig { grid: (top_k, b, HD.div_ceil(256)), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &stream, [dst_o.raw(), store.raw(), sel.raw(), top_k, HD, base_ptr])?;
        let dst_n = upload(id, &fill)?;
        let f = g.module().get_function("gather_u4_r1")?;
        let cfg = LaunchConfig { grid: (top_k, b, 1), block: (HD * 2 / 16, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &stream, [dst_n.raw(), store.raw(), sel.raw(), top_k, HD, base_ptr])?;
        let mut dst_w = upload(id, &fill)?;
        g.launch_batched_rows(&stream, &mut dst_w, &store, &sel, top_k, HD, b, if per_row { Some(&d_base) } else { None })?;
        stream.synchronize()?;
        let (o, n, w) = (download(&dst_o)?, download(&dst_n)?, download(&dst_w)?);
        let untouched = o.iter().filter(|&&x| x == 0xBEEF).count();
        let d_n = o.iter().zip(&n).filter(|(a, b)| a != b).count();
        let d_w = o.iter().zip(&w).filter(|(a, b)| a != b).count();
        eprintln!("gather b={b} top_k={top_k} per_row={per_row}: b128 diff={d_n}, wrapper diff={d_w} ({untouched} sentinel-row halves untouched)");
        assert!(untouched > 0, "test must include sentinel rows");
        assert_eq!(d_n, 0, "gather_u4_r1 not bit-exact at b={b} top_k={top_k}");
        assert_eq!(d_w, 0, "launch_batched_rows not bit-exact at b={b} top_k={top_k}");
        n_cases += 1;
    }
    eprintln!("PASS: {n_cases} gather cases bit-exact (whole destination)");
    Ok(())
}

// ------------------------------------------------------- 4. candidate threshold

#[test]
#[ignore]
fn cand_threshold_ilp_matches_candidate_threshold() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let stream = Stream::new(id)?;
    let cb = CandidateBlocks::for_arch(&arch)?;
    let mut rng = Lcg(0x5EED_E004);
    let cbs = CANDIDATE_BLOCK_SIZE;
    const MAX_N: u32 = 369_152;
    let nb_stride = MAX_N.div_ceil(cbs);
    let mut n_cases = 0usize;
    for class in 0..4 {
        for n_per in [vec![16385u32], vec![65536], vec![100003, 16392, 235000, 17000], vec![368640], vec![20000, 30000, 50000, 131072, 16384, 1, 0, 235000]] {
            let b = n_per.len() as u32;
            let mut bs_h = vec![0f32; (b * nb_stride) as usize];
            for (bi, &n) in n_per.iter().enumerate() {
                let nb = n.div_ceil(cbs) as usize;
                let row = &mut bs_h[bi * nb_stride as usize..bi * nb_stride as usize + nb];
                match class {
                    0 => row.iter_mut().for_each(|v| *v = rng.sym(8.0)),
                    1 => row.iter_mut().for_each(|v| *v = (rng.below(9) as f32) * 0.5),
                    2 => row.iter_mut().for_each(|v| *v = 1.25),
                    _ => row.iter_mut().for_each(|v| *v = if rng.below(2) == 0 { f32::NEG_INFINITY } else { rng.sym(1e30) }),
                }
                if nb > 0 { row[nb - 1] = f32::INFINITY; } // the pinned newest block
            }
            let bs = upload(id, &bs_h)?;
            let d_n = upload(id, &n_per)?;
            let run = |sym: &str, block: u32| -> eyre::Result<Vec<u32>> {
                let thr = upload(id, &vec![0xDEAD_BEEFu32; b as usize])?;
                let f = cb.module().get_function(sym)?;
                let cfg = LaunchConfig { grid: (b, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 };
                launch_kernel!(f, cfg, &stream, [thr.raw(), bs.raw(), d_n.raw(), nb_stride, cbs, CANDIDATE_TOPK_BLOCKS])?;
                stream.synchronize()?;
                download(&thr)
            };
            let o = run("candidate_threshold", 256)?;
            let n = run("candidate_threshold_ilp", 1024)?;
            assert!(o.iter().all(|&t| t != 0xDEAD_BEEF), "old kernel left thresholds unwritten");
            assert_eq!(o, n, "candidate_threshold_ilp differs: class={class} n={n_per:?}");
            n_cases += 1;
        }
    }
    // Wrapper: launch_build (block max + threshold) vs the old threshold kernel on the
    // block scores launch_build itself produced, at decode batches (ILP) and b = 16 (old).
    for &b in &[1u32, 4, 8, 16] {
        let stride = 131_072u32;
        let n_per: Vec<u32> = (0..b).map(|i| 20_000 + rng.below(stride - 20_000) - (i % 3)).collect();
        let s_h: Vec<f32> = (0..b * stride).map(|_| rng.sym(4.0)).collect();
        let scores = upload(id, &s_h)?;
        let d_n = upload(id, &n_per)?;
        let nb = stride.div_ceil(cbs);
        let mut block_score: DeviceBuffer<f32> = DeviceBuffer::new(id, (b * nb) as usize)?;
        let mut thr_w = upload(id, &vec![0xDEAD_BEEFu32; b as usize])?;
        cb.launch_build(&stream, &scores, &mut block_score, &mut thr_w, d_n.raw(), stride, nb, *n_per.iter().max().unwrap(), b)?;
        let thr_o = upload(id, &vec![0xDEAD_BEEFu32; b as usize])?;
        let f = cb.module().get_function("candidate_threshold")?;
        let cfg = LaunchConfig { grid: (b, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &stream, [thr_o.raw(), block_score.raw(), d_n.raw(), nb, cbs, CANDIDATE_TOPK_BLOCKS])?;
        stream.synchronize()?;
        assert_eq!(download(&thr_o)?, download(&thr_w)?, "launch_build threshold differs at b={b}");
        n_cases += 1;
    }
    eprintln!("PASS: {n_cases} threshold cases bit-identical");
    Ok(())
}
