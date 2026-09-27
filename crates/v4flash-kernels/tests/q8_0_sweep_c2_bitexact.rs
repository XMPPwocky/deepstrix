//! The 2026-09-26 C2_dense_prefill sweep kernels vs the kernels they replace, on
//! synthetic data (no model load; < 110 MB of device memory per test):
//!
//!   1. `q8_0_gemv_bpack_z16` / `q8_0_grouped_gemv_bpack_z16` vs the grid.z = b
//!      kernels at 16 < b <= 64 (q_b, kv, wo_b, a row-tail shape, grouped wo_a),
//!      plus the `matvec_batched` / `matvec_grouped_batched` wrappers
//!      (`V41_GEMV_BPACK_Z16`). BIT-EXACT.
//!   2. `q8_0_gemm_wmma_i8x_db` vs `q8_0_gemm_wmma_lds_tiled` (Engram shape with
//!      M cut to 2560 rows for memory) at b = 1..129 incl. tails, and the
//!      `engram_pass_rows` rule: every row runs the same kernel arm with
//!      128-row passes as with 64-row passes (`V41_ENGRAM_I8X`,
//!      `V41_ENGRAM_CHUNK128`). BIT-EXACT.
//!   3. `q8_0_gemm_wmma_f16x_db_bn64` / `_256x128` vs `q8_0_gemm_wmma_f16x` at
//!      kv / q_a / q_b (M cut to 4096) / grouped wo_a shapes incl. tails, and the
//!      per-site selectors (`V41_F16X_DB_BN64`, `V41_F16X_256`). BIT-EXACT.
//!   4. `V41_REPLAY_F16X` numerics: the f16x GEMM on f16-cast activations vs
//!      the dp4a GEMV on Q8-quantised activations at a replay-sized b = 64 --
//!      NOT bit-exact by design; reports rel_rmse (ledger: 2.9e-4).
//!
//! gfx1201 (the WMMA kernels are gfx12-only). Run: `cargo test --release
//! --features v41 -p v4flash-kernels --test q8_0_sweep_c2_bitexact -- --ignored
//! --test-threads=1 --nocapture`.

use color_eyre::eyre;
use v4flash_hip::{install_panic_handler, launch_kernel, Device, DeviceBuffer, LaunchConfig, Stream};
use v4flash_kernels::het::forward_prefill::engram_pass_rows;
use v4flash_kernels::q8_0::{
    engram_i8x_for, f16x_tile_kv, f16x_tile_q_a, f16x_tile_qb_wo_a, F16xTile, Q8_0GroupedMatvec, Q8_0Matvec,
    Q8_0MatvecWmma, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use v4flash_kernels::q8_k::Q8KQuantize;

fn pick_dgpu() -> eyre::Result<Option<Device>> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(Some(d));
        }
    }
    Ok(None)
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
    fn i8(&mut self) -> i8 {
        (self.next() & 0xFF) as u8 as i8
    }
    /// A finite f16 bit pattern, |v| in ~[2^-9, 2^0), random sign.
    fn f16_bits(&mut self) -> u16 {
        let exp = 6 + (self.next() % 9) as u16;
        let man = (self.next() & 0x3FF) as u16;
        (((self.next() & 1) as u16) << 15) | (exp << 10) | man
    }
}

/// Repacked Q8_0 rows: [scales (blocks*2 B f16) | quants (blocks*32 B)], f16
/// scales in +-(0.0156..1.0), full-range int8.
fn make_q8_rows(n_rows: u32, k: u32, rng: &mut Lcg) -> Vec<u8> {
    let blocks = (k / Q8_0_BLOCK_ELEMS) as usize;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES as usize;
    let mut bytes = vec![0u8; n_rows as usize * row_bytes];
    for r in 0..n_rows as usize {
        let off = r * row_bytes;
        for b in 0..blocks {
            let bits: u16 = (0x2C00 + (rng.next() & 0x0FFF)) as u16 | (((rng.next() & 1) as u16) << 15);
            bytes[off + b * 2] = (bits & 0xFF) as u8;
            bytes[off + b * 2 + 1] = (bits >> 8) as u8;
        }
        let q = off + blocks * 2;
        for i in 0..blocks * 32 {
            bytes[q + i] = rng.i8() as u8;
        }
    }
    bytes
}

fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len().max(1))?;
    if !host.is_empty() {
        d.copy_from_host(host)?;
    }
    Ok(d)
}

fn download(d: &DeviceBuffer<f32>) -> eyre::Result<Vec<f32>> {
    let mut v = vec![0f32; d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}

const SENTINEL: f32 = f32::from_bits(0x7FC0_BEEF);

fn sentinel_buf(id: i32, n: usize) -> eyre::Result<DeviceBuffer<f32>> {
    upload(id, &vec![SENTINEL; n])
}

fn bit_diff(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn unwritten(a: &[f32]) -> usize {
    a.iter().filter(|x| x.to_bits() == SENTINEL.to_bits()).count()
}

fn acts(k: u32, b: u32, rng: &mut Lcg) -> (Vec<i8>, Vec<f32>) {
    let xq: Vec<i8> = (0..(k * b) as usize).map(|_| rng.i8()).collect();
    let xs: Vec<f32> = (0..(k / 32 * b) as usize).map(|_| 0.001 + rng.unit() * 0.02).collect();
    (xq, xs)
}

// ------------------------------------------------------------------ 1. z16

#[test]
#[ignore]
fn bpack_z16_matches_grid_z() -> eyre::Result<()> {
    install_panic_handler()?;
    let Some(dev) = pick_dgpu()? else { eprintln!("SKIP: no gfx1201"); return Ok(()); };
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let q8g = Q8_0GroupedMatvec::for_arch(&arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_C201);
    let mut cases = 0usize;
    // (M, K, batches): q_b, kv, wo_b, a row / lane-loop tail shape.
    let shapes: [(u32, u32, &[u32]); 4] = [
        (32768, 1280, &[17, 33, 49, 63, 64]),
        (512, 5120, &[17, 64]),
        (5120, 8192, &[31, 64]),
        (1001, 2304, &[17, 40, 64]),
    ];
    for &(m, k, bs) in &shapes {
        let blocks = k / 32;
        let w = upload(id, &make_q8_rows(m, k, &mut rng))?;
        for &b in bs {
            let (xq_h, xs_h) = acts(k, b, &mut rng);
            let (xq, xs) = (upload(id, &xq_h)?, upload(id, &xs_h)?);
            let n_out = (b * m) as usize;
            let o_ref = sentinel_buf(id, n_out)?;
            let f = q8.module().get_function("q8_0_gemv_batched_warp8")?;
            let cfg = LaunchConfig { grid: (m.div_ceil(8), 1, b), block: (256, 1, 1), shared_mem_bytes: 0 };
            launch_kernel!(f, cfg, &s, [o_ref.raw(), w.raw(), xq.raw(), xs.raw(), k, m, blocks])?;
            let o_z = sentinel_buf(id, n_out)?;
            let f = q8.module().get_function("q8_0_gemv_bpack_z16")?;
            let cfg = LaunchConfig { grid: (m.div_ceil(8), 1, b.div_ceil(16)), block: (256, 1, 1), shared_mem_bytes: 0 };
            launch_kernel!(f, cfg, &s, [o_z.raw(), w.raw(), xq.raw(), xs.raw(), k, m, blocks, b])?;
            let mut o_w = sentinel_buf(id, n_out)?;
            q8.matvec_batched(&s, &mut o_w, &w, &xq, &xs, m, k, b)?;
            s.synchronize()?;
            let (r, z, wr) = (download(&o_ref)?, download(&o_z)?, download(&o_w)?);
            assert_eq!(unwritten(&r), 0);
            let (dz, dw) = (bit_diff(&r, &z), bit_diff(&r, &wr));
            eprintln!("gemv {m}x{k} b={b}: z16 bit_diff={dz}, matvec_batched bit_diff={dw}");
            assert_eq!(dz, 0, "q8_0_gemv_bpack_z16 not bit-exact at {m}x{k} b={b}");
            assert_eq!(dw, 0, "matvec_batched not bit-exact at {m}x{k} b={b}");
            cases += 1;
        }
    }
    // Grouped: wo_a 8 x (1024 x 4096).
    let (g, gd, rank) = (8u32, 4096u32, 1024u32);
    let bpg = gd / 32;
    let out_dim = g * rank;
    let w = upload(id, &make_q8_rows(out_dim, gd, &mut rng))?;
    for &b in &[17u32, 49, 64] {
        let (xq_h, xs_h) = acts(g * gd, b, &mut rng);
        let (xq, xs) = (upload(id, &xq_h)?, upload(id, &xs_h)?);
        let n_out = (b * out_dim) as usize;
        let o_ref = sentinel_buf(id, n_out)?;
        let f = q8g.module().get_function("q8_0_grouped_gemv_batched")?;
        let cfg = LaunchConfig { grid: (out_dim.div_ceil(8), 1, b), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &s, [o_ref.raw(), w.raw(), xq.raw(), xs.raw(), gd, rank, bpg, g])?;
        let o_z = sentinel_buf(id, n_out)?;
        let f = q8g.module().get_function("q8_0_grouped_gemv_bpack_z16")?;
        let cfg = LaunchConfig { grid: (out_dim.div_ceil(8), 1, b.div_ceil(16)), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &s, [o_z.raw(), w.raw(), xq.raw(), xs.raw(), gd, rank, bpg, g, b])?;
        let mut o_w = sentinel_buf(id, n_out)?;
        q8g.matvec_grouped_batched(&s, &mut o_w, &w, &xq, &xs, gd, rank, g, b)?;
        s.synchronize()?;
        let (r, z, wr) = (download(&o_ref)?, download(&o_z)?, download(&o_w)?);
        assert_eq!(unwritten(&r), 0);
        let (dz, dw) = (bit_diff(&r, &z), bit_diff(&r, &wr));
        eprintln!("grouped wo_a b={b}: z16 bit_diff={dz}, matvec_grouped_batched bit_diff={dw}");
        assert_eq!(dz, 0, "q8_0_grouped_gemv_bpack_z16 not bit-exact at b={b}");
        assert_eq!(dw, 0, "matvec_grouped_batched not bit-exact at b={b}");
        cases += 1;
    }
    eprintln!("PASS: {cases} z16 cases bit-exact");
    Ok(())
}

// ----------------------------------------------------------- 2. Engram i8x_db

#[test]
#[ignore]
fn engram_i8x_db_matches_lds_tiled() -> eyre::Result<()> {
    install_panic_handler()?;
    // Pass rule first (host only): with 128-row passes every row takes the same
    // kernel arm as with 64-row passes.
    for b in 1..=700usize {
        let arms = |chunk: usize| -> Vec<bool> {
            let mut v = Vec::with_capacity(b);
            let mut c0 = 0;
            while c0 < b {
                let n = engram_pass_rows(b - c0, chunk);
                assert!(n >= 1 && n <= chunk);
                let gemv = v4flash_kernels::het::dispatch::small_b_dense_dp4a(n as u32);
                v.extend(std::iter::repeat(gemv).take(n));
                c0 += n;
            }
            v
        };
        assert_eq!(arms(64), arms(128), "Engram pass arms differ at b={b}");
    }
    eprintln!("pass rule: rows take identical arms with 64- and 128-row passes for b = 1..700");

    let Some(dev) = pick_dgpu()? else { eprintln!("SKIP: no gfx1201"); return Ok(()); };
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let wm = Q8_0MatvecWmma::for_arch(&arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_C202);
    // Engram K = 6144; M cut from 25600 to 2560 (same %128 tiling) for memory.
    let (m, k) = (2560u32, 6144u32);
    assert!(engram_i8x_for(25600, 6144) || std::env::var("V41_ENGRAM_I8X").as_deref() == Ok("0"));
    assert!(!engram_i8x_for(25600, 6144 - 32), "K % 256 != 0 must keep lds_tiled");
    let blocks = k / 32;
    let w = upload(id, &make_q8_rows(m, k, &mut rng))?;
    let (xq_h, xs_h) = acts(k, 256, &mut rng);
    let (xq, xs) = (upload(id, &xq_h)?, upload(id, &xs_h)?);
    let mut cases = 0usize;
    for &b in &[1u32, 9, 17, 63, 64, 65, 100, 127, 128, 129, 200, 256] {
        let n_out = (b * m) as usize;
        let o_ref = sentinel_buf(id, n_out)?;
        let f = wm.module().get_function("q8_0_gemm_wmma_lds_tiled")?;
        let cfg = LaunchConfig { grid: (m.div_ceil(64), b.div_ceil(64), 1), block: (128, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &s, [o_ref.raw(), w.raw(), xq.raw(), xs.raw(), k, m, b, blocks])?;
        let mut o_new = sentinel_buf(id, n_out)?;
        wm.gemm_i8x_db(&s, &mut o_new, &w, &xq, &xs, m, k, b)?;
        s.synchronize()?;
        let (r, n) = (download(&o_ref)?, download(&o_new)?);
        assert_eq!(unwritten(&r), 0, "lds_tiled left outputs unwritten at b={b}");
        assert!(r.iter().all(|v| v.is_finite()));
        let d = bit_diff(&r, &n);
        eprintln!("engram {m}x{k} b={b}: i8x_db bit_diff={d}");
        assert_eq!(d, 0, "q8_0_gemm_wmma_i8x_db not bit-exact at b={b}");
        cases += 1;
    }
    eprintln!("PASS: {cases} i8x_db cases bit-exact");
    Ok(())
}

// ------------------------------------------------------------ 3. f16x tiles

#[test]
#[ignore]
fn f16x_tiles_match_base() -> eyre::Result<()> {
    install_panic_handler()?;
    // Selectors (defaults ON unless the knobs are 0).
    let bn64 = std::env::var("V41_F16X_DB_BN64").as_deref() != Ok("0");
    let t256 = std::env::var("V41_F16X_256").as_deref() != Ok("0");
    let on = |c: bool, t: F16xTile| if c { t } else { F16xTile::Base };
    assert_eq!(f16x_tile_kv(512), on(bn64, F16xTile::DbBn64));
    assert_eq!(f16x_tile_kv(64), F16xTile::Base);
    assert_eq!(f16x_tile_q_a(1280, 512), on(bn64, F16xTile::DbBn64));
    assert_eq!(f16x_tile_q_a(1280, 1024), F16xTile::Base, "q_a loses at 1024 rows");
    assert_eq!(f16x_tile_q_a(2304, 512), F16xTile::Base, "shared expert keeps the base tile");
    assert_eq!(f16x_tile_qb_wo_a(32768, 512), on(t256, F16xTile::T256x128));
    assert_eq!(f16x_tile_qb_wo_a(1024, 1024), on(t256, F16xTile::T256x128));
    assert_eq!(f16x_tile_qb_wo_a(32768, 64), F16xTile::Base, "replay-sized passes keep the base tile");

    let Some(dev) = pick_dgpu()? else { eprintln!("SKIP: no gfx1201"); return Ok(()); };
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let wm = Q8_0MatvecWmma::for_arch(&arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_C203);
    let mut cases = 0usize;
    // (name, M, K, groups, tile, batches)
    let shapes: [(&str, u32, u32, u32, F16xTile, &[u32]); 5] = [
        ("kv", 512, 5120, 1, F16xTile::DbBn64, &[1, 3, 17, 65, 100, 129, 255, 256, 500, 512, 513]),
        ("q_a", 1280, 5120, 1, F16xTile::DbBn64, &[9, 63, 64, 65, 256, 511, 512]),
        ("q_b/4096", 4096, 1280, 1, F16xTile::T256x128, &[1, 7, 65, 129, 257, 512, 513]),
        ("wo_a", 1024, 4096, 8, F16xTile::T256x128, &[65, 129, 257]),
        ("wo_a/bn64", 1024, 4096, 8, F16xTile::DbBn64, &[65, 129]),
    ];
    for &(name, m, k, g, tile, bs) in &shapes {
        let w = upload(id, &make_q8_rows(g * m, k, &mut rng))?;
        let pitch = g * k + 64;
        let bmax = *bs.iter().max().unwrap();
        let x_h: Vec<u16> = (0..(bmax * pitch) as usize).map(|_| rng.f16_bits()).collect();
        let x16 = upload(id, &x_h)?;
        for &b in bs {
            let n_out = (b * g * m) as usize;
            let mut o_base = sentinel_buf(id, n_out)?;
            wm.gemm_f16x_tile(F16xTile::Base, &s, &mut o_base, &w, &x16, k, m, g, b, pitch)?;
            let mut o_new = sentinel_buf(id, n_out)?;
            wm.gemm_f16x_tile(tile, &s, &mut o_new, &w, &x16, k, m, g, b, pitch)?;
            s.synchronize()?;
            let (r, n) = (download(&o_base)?, download(&o_new)?);
            assert_eq!(unwritten(&r), 0, "{name}: base left outputs unwritten at b={b}");
            assert!(r.iter().all(|v| v.is_finite()));
            let d = bit_diff(&r, &n);
            eprintln!("f16x {name} {m}x{k}x{g} b={b} {tile:?}: bit_diff={d}");
            assert_eq!(d, 0, "{tile:?} not bit-exact vs base at {name} b={b}");
            cases += 1;
        }
    }
    // 256x128 refuses M % 256 != 0.
    let w = upload(id, &make_q8_rows(640, 1280, &mut rng))?;
    let x16 = upload(id, &vec![0u16; (1344 * 65) as usize])?;
    let mut o = sentinel_buf(id, 65 * 640)?;
    assert!(wm.gemm_f16x_tile(F16xTile::T256x128, &s, &mut o, &w, &x16, 1280, 640, 1, 65, 1344).is_err());
    eprintln!("PASS: {cases} f16x tile cases bit-exact");
    Ok(())
}

// ------------------------------------------------- 4. replay f16x numerics

#[test]
#[ignore]
fn replay_f16x_numerics_vs_dp4a() -> eyre::Result<()> {
    install_panic_handler()?;
    let Some(dev) = pick_dgpu()? else { eprintln!("SKIP: no gfx1201"); return Ok(()); };
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let wm = Q8_0MatvecWmma::for_arch(&arch)?;
    let q8k = Q8KQuantize::for_arch(&arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_C204);
    let (m, k, b) = (4096u32, 1280u32, 64u32);
    let w = upload(id, &make_q8_rows(m, k, &mut rng))?;
    // RMS-normed-like activations.
    let x_h: Vec<f32> = (0..(b * k) as usize).map(|_| (rng.unit() * 2.0 - 1.0) * 1.5).collect();
    let x = upload(id, &x_h)?;
    let mut xq: DeviceBuffer<i8> = DeviceBuffer::new(id, (b * k) as usize)?;
    let mut xs: DeviceBuffer<f32> = DeviceBuffer::new(id, (b * k / 32) as usize)?;
    q8.quantize_input_batched(&s, &mut xq, &mut xs, &x, k, b)?;
    let pitch = k + 64;
    let mut x16: DeviceBuffer<u16> = DeviceBuffer::new(id, (b * pitch) as usize)?;
    q8k.launch_cast_f16_2d(&s, &mut x16, &x, b, k, pitch)?;
    let mut o_dp4a = sentinel_buf(id, (b * m) as usize)?;
    q8.matvec_batched(&s, &mut o_dp4a, &w, &xq, &xs, m, k, b)?;
    let mut o_f16x = sentinel_buf(id, (b * m) as usize)?;
    wm.gemm_f16x(&s, &mut o_f16x, &w, &x16, k, m, 1, b, pitch)?;
    s.synchronize()?;
    let (a, f) = (download(&o_dp4a)?, download(&o_f16x)?);
    let (mut se, mut sw, mut max_abs) = (0f64, 0f64, 0f32);
    for (&p, &q) in a.iter().zip(&f) {
        let d = (p - q) as f64;
        se += d * d;
        sw += (p as f64) * (p as f64);
        max_abs = max_abs.max((p - q).abs());
    }
    let rel = (se / sw).sqrt();
    eprintln!("replay f16x vs dp4a at {m}x{k} b={b}: rel_rmse={rel:.3e} max_abs={max_abs:.3e} differing={} of {} (NOT bit-exact by design; ledger 2.9e-4)",
        bit_diff(&a, &f), a.len());
    assert_eq!(unwritten(&a) + unwritten(&f), 0);
    assert!(rel < 1e-2, "f16x vs dp4a rel_rmse {rel:.3e} is not a rounding-level difference");
    Ok(())
}
