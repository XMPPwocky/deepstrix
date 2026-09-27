//! The 2026-09-27 round-2 crossover-scan gate changes (item b):
//!
//!   1. `f16x_tile_qb_wo_a`: the 256x128 tile only from `F16X_256_MIN_ROWS` (192)
//!      rows (it lost x1.27 / x1.18 on q_b at 65 / 128 rows).
//!   2. `bpack_z16_for`: z16 at 16 < b <= 64 only when n_rows >= 2048 or b >= 48
//!      (it lost x1.72 on kv at b = 17, x1.14 on q_a).
//!   3. `GATHER_B128_MIN_B` = 3 (was 4), the b128 launch with the `V41_GRID_PAD`
//!      idle column; `CAND_THRESH_ILP_MAX_B` = 32 (was 8).
//!
//! Selector asserts (knob-aware, CPU only) plus GPU bit-exactness of the paths the
//! new gates newly route to, on synthetic data (< 40 MB of device memory): the
//! padded gather wrapper at b = 3 and the ILP threshold via `launch_build` at
//! b = 24 / 32, each against the production kernel. The existing sweep tests
//! (indexer_sweep_bitexact, q8_0_sweep_c2_bitexact) cover the kernels themselves.
//!
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test
//! round2_gates -- --ignored --test-threads=1 --nocapture`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, launch_kernel, Device, DeviceBuffer, LaunchConfig, Stream};
use v4flash_kernels::candidate_blocks::{cand_thresh_ilp_for, CandidateBlocks, CAND_THRESH_ILP_MAX_B};
use v4flash_kernels::config::{CANDIDATE_BLOCK_SIZE, CANDIDATE_TOPK_BLOCKS};
use v4flash_kernels::indexer::{IndexerGather, GATHER_B128_MIN_B};
use v4flash_kernels::q8_0::{bpack_z16_for, f16x_tile_qb_wo_a, F16xTile, F16X_256_MIN_ROWS};

fn knob_on(name: &str) -> bool {
    std::env::var(name).as_deref() != Ok("0")
}

#[test]
fn selectors_follow_the_round2_gates() {
    let t256 = knob_on("V41_F16X_256");
    for b in [65u32, 100, 128, 191] {
        assert_eq!(f16x_tile_qb_wo_a(32768, b), F16xTile::Base, "q_b at {b} rows must keep the base tile");
        assert_eq!(f16x_tile_qb_wo_a(1024, b), F16xTile::Base, "wo_a at {b} rows must keep the base tile");
    }
    for b in [F16X_256_MIN_ROWS, 256, 512, 1024] {
        let want = if t256 { F16xTile::T256x128 } else { F16xTile::Base };
        assert_eq!(f16x_tile_qb_wo_a(32768, b), want);
        assert_eq!(f16x_tile_qb_wo_a(1024, b), want);
    }
    let z16 = knob_on("V41_GEMV_BPACK_Z16") && knob_on("V41_GEMV_BPACK");
    for b in 1..=80u32 {
        for n_rows in [512u32, 1280, 2304, 5120, 8192, 32768] {
            let want = z16 && b > 16 && b <= 64 && (n_rows >= 2048 || b >= 48);
            assert_eq!(bpack_z16_for(b, n_rows), want, "bpack_z16_for(b={b}, n_rows={n_rows})");
        }
    }
    let ilp = knob_on("V41_CAND_THRESH_ILP");
    assert_eq!(CAND_THRESH_ILP_MAX_B, 32);
    for b in 1..=128u32 {
        assert_eq!(cand_thresh_ilp_for(b), ilp && b <= 32, "cand_thresh_ilp_for({b})");
    }
    assert_eq!(GATHER_B128_MIN_B, 3);
}

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
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n.max(1)
    }
    fn sym(&mut self, a: f32) -> f32 {
        ((self.next() & 0xFFFFFF) as f32 / 16777216.0 * 2.0 - 1.0) * a
    }
}

fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len())?;
    d.copy_from_host(host)?;
    Ok(d)
}

fn download<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}

#[test]
#[ignore]
fn newly_gated_paths_are_bit_exact() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_B002);

    // Gather at b = 2, 3, 4 (3 newly on gather_u4_r1, all with the pad column):
    // wrapper vs indexer_gather_batched, whole destination incl. sentinel rows.
    let g = IndexerGather::for_arch(&arch)?;
    const HD: u32 = 512;
    const TOP_K: u32 = 512;
    const STORE_ROWS: u32 = 8192;
    let store_h: Vec<u16> = (0..STORE_ROWS * HD).map(|_| rng.next() as u16).collect();
    let store = upload(id, &store_h)?;
    for (b, per_row) in [(2u32, false), (3, false), (3, true), (4, true)] {
        let bases: Vec<i32> = (0..b).map(|_| rng.below(4000) as i32).collect();
        let span = if per_row { 4000 } else { STORE_ROWS };
        let mut sel_h = vec![-1i32; (b * TOP_K) as usize];
        for bi in 0..b as usize {
            for i in 0..(TOP_K - 37) as usize {
                sel_h[bi * TOP_K as usize + i] = if rng.below(9) == 0 { -1 } else { rng.below(span) as i32 };
            }
        }
        let sel = upload(id, &sel_h)?;
        let d_base = upload(id, &bases)?;
        let n = (b * TOP_K * HD) as usize;
        let dst_o = upload(id, &vec![0xBEEFu16; n])?;
        let mut dst_w = upload(id, &vec![0xBEEFu16; n])?;
        let f = g.module().get_function("indexer_gather_batched")?;
        let cfg = LaunchConfig { grid: (TOP_K, b, HD.div_ceil(256)), block: (256, 1, 1), shared_mem_bytes: 0 };
        let base_ptr = if per_row { d_base.raw() as *const i32 } else { std::ptr::null() };
        launch_kernel!(f, cfg, &stream, [dst_o.raw(), store.raw(), sel.raw(), TOP_K, HD, base_ptr])?;
        g.launch_batched_rows(&stream, &mut dst_w, &store, &sel, TOP_K, HD, b, if per_row { Some(&d_base) } else { None })?;
        stream.synchronize()?;
        let (o, w) = (download(&dst_o)?, download(&dst_w)?);
        let d = o.iter().zip(&w).filter(|(x, y)| x != y).count();
        eprintln!("gather b={b} per_row={per_row}: wrapper vs indexer_gather_batched diff={d}");
        assert_eq!(d, 0, "gather wrapper differs at b={b}");
    }

    // Threshold via launch_build at b = 12, 24, 32 (newly ILP) vs candidate_threshold.
    let cb = CandidateBlocks::for_arch(&arch)?;
    let cbs = CANDIDATE_BLOCK_SIZE;
    for &b in &[12u32, 24, 32] {
        let stride = 32_768u32;
        let n_per: Vec<u32> = (0..b).map(|i| 4_000 + rng.below(stride - 4_000) - (i % 3)).collect();
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
        let (o, w) = (download(&thr_o)?, download(&thr_w)?);
        assert!(o.iter().all(|&t| t != 0xDEAD_BEEF), "old threshold kernel left rows unwritten");
        eprintln!("threshold b={b}: launch_build vs candidate_threshold equal={}", o == w);
        assert_eq!(o, w, "launch_build threshold differs at b={b}");
    }
    eprintln!("PASS: round-2 gated paths bit-exact");
    Ok(())
}
