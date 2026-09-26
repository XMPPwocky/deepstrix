//! Bit-exactness of the 2026-09-26 C1_dense_decode sweep kernels against the
//! kernels they replace, on synthetic data (no model load; < 100 MB of device
//! memory per test, so it runs beside a live hub):
//!
//!   1. `q8_0_gemv_bpack_tB<b>` (b = 1..10) vs `q8_0_gemv_bpack_warp8`, incl.
//!      row tails (n_rows % 8 != 0), K = 32 (one block per row) and a lane-loop
//!      tail (K = 2304), plus the `matvec_batched` wrapper's default pick.
//!   2. `q8_0_grouped_gemv_bpack_tB<b>` (b = 1..8) vs `q8_0_grouped_gemv_bpack`
//!      at the production wo_a shape and at rank = 1000 (WG rows crossing groups).
//!   3. `q8_0_quantize_f32_wave` and the `+1` grid pad vs `q8_0_quantize_f32`
//!      (xq bytes and xscale bits), incl. blocks % 8 != 0 and adversarial blocks.
//!   4. `shared_gateup_swiglu_q8_tB<b>_r1` (b = 1..10) vs the 4-launch chain
//!      gate/up/swiglu/quantize (mid_xq, mid_xscale) and the down output after
//!      it, with saturating (clamp hit) and non-saturating activations.
//!
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test
//! q8_0_tb_bitexact -- --ignored --test-threads=1 --nocapture`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, launch_kernel, Device, DeviceBuffer, LaunchConfig, Stream};
use v4flash_kernels::q8_0::{
    Q8_0GroupedMatvec, Q8_0Matvec, SharedExpertFused, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use v4flash_kernels::Swiglu;

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
    fn i8(&mut self) -> i8 {
        (self.next() & 0xFF) as u8 as i8
    }
}

/// M18 repacked Q8_0 rows: [scales(blocks*2 B f16) | quants(blocks*32 B)].
/// Random f16 scales in +-(0.0156..1.0), random full-range int8 quants; every
/// 97th row is all -128 (int8 extreme) with the largest scale.
fn make_q8_rows(n_rows: u32, k: u32, rng: &mut Lcg) -> Vec<u8> {
    let blocks = (k / Q8_0_BLOCK_ELEMS) as usize;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES as usize;
    let mut bytes = vec![0u8; n_rows as usize * row_bytes];
    for r in 0..n_rows as usize {
        let off = r * row_bytes;
        let extreme = r % 97 == 96;
        for b in 0..blocks {
            let bits: u16 = if extreme {
                0x3C00
            } else {
                (0x2C00 + (rng.next() & 0x0FFF)) as u16 | (((rng.next() & 1) as u16) << 15)
            };
            bytes[off + b * 2] = (bits & 0xFF) as u8;
            bytes[off + b * 2 + 1] = (bits >> 8) as u8;
        }
        let q = off + blocks * 2;
        for i in 0..blocks * 32 {
            bytes[q + i] = if extreme { 0x80 } else { rng.i8() as u8 };
        }
    }
    bytes
}

/// `[b, k]` int8 activations (full range) + `[b, k/32]` scales in (0, scale].
fn make_acts(k: u32, b: u32, scale: f32, rng: &mut Lcg) -> (Vec<i8>, Vec<f32>) {
    let xq: Vec<i8> = (0..(k * b) as usize).map(|_| rng.i8()).collect();
    let xs: Vec<f32> = (0..(k / 32 * b) as usize).map(|_| (rng.unit() + 1e-3) * scale).collect();
    (xq, xs)
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

const SENTINEL: f32 = f32::from_bits(0x7FC0_BEEF);

fn sentinel_buf(id: i32, n: usize) -> eyre::Result<DeviceBuffer<f32>> {
    upload(id, &vec![SENTINEL; n])
}

/// Count of bit-different f32 lanes; a sentinel left in place counts too.
fn bit_diff(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn unwritten(a: &[f32]) -> usize {
    a.iter().filter(|x| x.to_bits() == SENTINEL.to_bits()).count()
}

#[test]
#[ignore]
fn gemv_bpack_tb_matches_runtime_kernel() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_0001);

    // (n_rows, k): q_a; kv; row tail + lane-loop tail; row tail; K = 32; single row.
    let shapes: [(u32, u32); 6] = [(1280, 5120), (512, 5120), (1001, 2304), (523, 1280), (33, 32), (1, 1280)];
    let mut cases = 0usize;
    for &(n_rows, k) in &shapes {
        let blocks = k / 32;
        let w = upload(id, &make_q8_rows(n_rows, k, &mut rng))?;
        for b in 1..=10u32 {
            let (xq_h, xs_h) = make_acts(k, b, 1.5, &mut rng);
            let xq = upload(id, &xq_h)?;
            let xs = upload(id, &xs_h)?;
            let n_out = (b * n_rows) as usize;
            let cfg = LaunchConfig {
                grid: (n_rows.div_ceil(8), 1, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_ref = sentinel_buf(id, n_out)?;
            let f = q8.module().get_function("q8_0_gemv_bpack_warp8")?;
            launch_kernel!(f, cfg, &stream, [out_ref.raw(), w.raw(), xq.raw(), xs.raw(), k, n_rows, blocks, b])?;
            let out_tb = sentinel_buf(id, n_out)?;
            let sym = format!("q8_0_gemv_bpack_tB{b}");
            let f = q8.module().get_function(&sym)?;
            launch_kernel!(f, cfg, &stream, [out_tb.raw(), w.raw(), xq.raw(), xs.raw(), k, n_rows, blocks, b])?;
            // The wrapper's own pick (V41_GEMV_TB default ON -> the twin).
            let mut out_wr = sentinel_buf(id, n_out)?;
            q8.matvec_batched(&stream, &mut out_wr, &w, &xq, &xs, n_rows, k, b)?;
            stream.synchronize()?;
            let (r, t, wr) = (download(&out_ref)?, download(&out_tb)?, download(&out_wr)?);
            assert_eq!(unwritten(&r), 0, "runtime kernel left outputs unwritten at {n_rows}x{k} b={b}");
            assert!(r.iter().all(|v| v.is_finite()), "non-finite reference at {n_rows}x{k} b={b}");
            let d_tb = bit_diff(&r, &t);
            let d_wr = bit_diff(&r, &wr);
            eprintln!("gemv {n_rows}x{k} b={b}: {sym} bit_diff={d_tb}, wrapper bit_diff={d_wr}");
            assert_eq!(d_tb, 0, "{sym} is not bit-exact vs q8_0_gemv_bpack_warp8 at {n_rows}x{k} b={b}");
            assert_eq!(d_wr, 0, "matvec_batched wrapper is not bit-exact at {n_rows}x{k} b={b}");
            cases += 1;
        }
    }
    eprintln!("PASS: {cases} (shape, b) cases bit-exact");
    Ok(())
}

#[test]
#[ignore]
fn grouped_bpack_tb_matches_runtime_kernel() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let q8g = Q8_0GroupedMatvec::for_arch(&arch)?;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_0002);

    // (n_groups, group_dim, rank): production wo_a; rank not % 8 so WG rows cross groups.
    let shapes: [(u32, u32, u32); 2] = [(8, 4096, 1024), (3, 2048, 1000)];
    let mut cases = 0usize;
    for &(g, gd, rank) in &shapes {
        let bpg = gd / 32;
        let out_dim = g * rank;
        let w = upload(id, &make_q8_rows(out_dim, gd, &mut rng))?;
        for b in 1..=8u32 {
            let (xq_h, xs_h) = make_acts(g * gd, b, 1.5, &mut rng);
            let xq = upload(id, &xq_h)?;
            let xs = upload(id, &xs_h)?;
            let n_out = (b * out_dim) as usize;
            let cfg = LaunchConfig {
                grid: (out_dim.div_ceil(8), 1, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_ref = sentinel_buf(id, n_out)?;
            let f = q8g.module().get_function("q8_0_grouped_gemv_bpack")?;
            launch_kernel!(f, cfg, &stream, [out_ref.raw(), w.raw(), xq.raw(), xs.raw(), gd, rank, bpg, g, b])?;
            let out_tb = sentinel_buf(id, n_out)?;
            let sym = format!("q8_0_grouped_gemv_bpack_tB{b}");
            let f = q8g.module().get_function(&sym)?;
            launch_kernel!(f, cfg, &stream, [out_tb.raw(), w.raw(), xq.raw(), xs.raw(), gd, rank, bpg, g, b])?;
            let mut out_wr = sentinel_buf(id, n_out)?;
            q8g.matvec_grouped_batched(&stream, &mut out_wr, &w, &xq, &xs, gd, rank, g, b)?;
            stream.synchronize()?;
            let (r, t, wr) = (download(&out_ref)?, download(&out_tb)?, download(&out_wr)?);
            assert_eq!(unwritten(&r), 0, "runtime grouped kernel left outputs unwritten");
            assert!(r.iter().all(|v| v.is_finite()));
            let d_tb = bit_diff(&r, &t);
            let d_wr = bit_diff(&r, &wr);
            eprintln!("grouped {g}x({rank}x{gd}) b={b}: {sym} bit_diff={d_tb}, wrapper bit_diff={d_wr}");
            assert_eq!(d_tb, 0, "{sym} is not bit-exact vs q8_0_grouped_gemv_bpack");
            assert_eq!(d_wr, 0, "matvec_grouped_batched wrapper is not bit-exact at b={b}");
            cases += 1;
        }
    }
    eprintln!("PASS: {cases} grouped cases bit-exact");
    Ok(())
}

/// Random activations in [-3, 3] with adversarial leading blocks: all-zero
/// (d = 0 path), one 1e30 spike (clamp), 1e-30 values, exact .5 ties at
/// d = 1, +-127 / 126.5 rounding boundary, and denormals.
fn make_quant_input(blocks: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut x: Vec<f32> = (0..blocks * 32).map(|_| rng.unit() * 6.0 - 3.0).collect();
    let special: [fn(usize) -> f32; 6] = [
        |_| 0.0,
        |i| if i == 5 { 1e30 } else { 0.5 },
        |i| 1e-30 * (i as f32 + 1.0),
        |i| if i == 0 { 127.0 } else { (i as f32) + 0.5 },
        |i| if i % 2 == 0 { 127.0 } else { -126.5 },
        |i| f32::from_bits(1 + i as u32),
    ];
    for (blk, f) in special.iter().enumerate() {
        if blk < blocks {
            for i in 0..32 {
                x[blk * 32 + i] = f(i);
            }
        }
    }
    x
}

#[test]
#[ignore]
fn quantize_wave_and_grid_pad_match_serial_kernel() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_0003);

    // WG tails (blocks % 8 != 0), production K=5120 b=4 (640), the slow
    // power-of-two grids (2048, 4096) and K=32768 b=3 (3072).
    for &blocks in &[1usize, 7, 9, 41, 160, 640, 2047, 2048, 3072, 4096] {
        let x = upload(id, &make_quant_input(blocks, &mut rng))?;
        let nb = blocks as u32;
        let run = |sym: &str, grid: u32, block: u32| -> eyre::Result<(Vec<i8>, Vec<f32>)> {
            let xq: DeviceBuffer<i8> = upload(id, &vec![0x55i8; blocks * 32])?;
            let xs = sentinel_buf(id, blocks)?;
            let f = q8.module().get_function(sym)?;
            let cfg = LaunchConfig { grid: (grid, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 };
            launch_kernel!(f, cfg, &stream, [xq.raw(), xs.raw(), x.raw(), nb])?;
            stream.synchronize()?;
            Ok((download(&xq)?, download(&xs)?))
        };
        let (rq, rs) = run("q8_0_quantize_f32", nb, 32)?;
        assert_eq!(unwritten(&rs), 0);
        let variants: [(&str, u32, u32); 3] = [
            ("q8_0_quantize_f32", nb + 1, 32),
            ("q8_0_quantize_f32_wave", nb.div_ceil(8), 256),
            ("q8_0_quantize_f32_wave", nb.div_ceil(8) + 1, 256),
        ];
        for (sym, grid, block) in variants {
            let (cq, cs) = run(sym, grid, block)?;
            let dq = rq.iter().zip(&cq).filter(|(a, b)| a != b).count();
            let ds = bit_diff(&rs, &cs);
            eprintln!("quantize blocks={blocks} {sym} grid={grid}: xq int8_diff={dq} xscale bit_diff={ds}");
            assert_eq!(dq, 0, "{sym} grid={grid}: xq differs at blocks={blocks}");
            assert_eq!(ds, 0, "{sym} grid={grid}: xscale differs at blocks={blocks}");
        }
        // The wrapper (wave + pad by default) over the same elements as one batch row.
        let mut xq: DeviceBuffer<i8> = upload(id, &vec![0x55i8; blocks * 32])?;
        let mut xs = sentinel_buf(id, blocks)?;
        q8.quantize_input_batched(&stream, &mut xq, &mut xs, &x, nb * 32, 1)?;
        stream.synchronize()?;
        let (cq, cs) = (download(&xq)?, download(&xs)?);
        assert_eq!(rq.iter().zip(&cq).filter(|(a, b)| a != b).count(), 0, "wrapper xq differs at blocks={blocks}");
        assert_eq!(bit_diff(&rs, &cs), 0, "wrapper xscale differs at blocks={blocks}");
    }
    eprintln!("PASS: quantize wave / grid pad bit-exact");
    Ok(())
}

#[test]
#[ignore]
fn shared_fused_matches_four_launch_chain() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let swiglu = Swiglu::for_arch(&arch)?;
    let fused = SharedExpertFused::for_arch(&arch)?;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_0004);

    // Production shared-expert shape: gate/up [2304, 5120], down [5120, 2304].
    let (k, n_ff) = (5120u32, 2304u32);
    let clamp = 10.0f32;
    let blocks = k / 32;
    let wg = upload(id, &make_q8_rows(n_ff, k, &mut rng))?;
    let wu = upload(id, &make_q8_rows(n_ff, k, &mut rng))?;
    let wd = upload(id, &make_q8_rows(k, n_ff, &mut rng))?;
    let f_bp = q8.module().get_function("q8_0_gemv_bpack_warp8")?;
    let f_q = q8.module().get_function("q8_0_quantize_f32")?;
    let cfg_gu = LaunchConfig { grid: (n_ff / 8, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
    let cfg_d = LaunchConfig { grid: (k / 8, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };

    // xscale magnitude: 1.0 saturates (|gate| ~ 1e5, every element clamped),
    // 1e-4 hits the clamp lightly (|gate| ~ 15), 2e-5 stays under it (|gate|
    // ~ 3): the smooth swiglu path must be exercised, not only the clamp
    // (the sweep reviewer's C1_XS_SCALE point). Asserted below.
    for &xs_scale in &[1.0f32, 1e-4, 2e-5] {
        for b in 1..=10u32 {
            let (xq_h, xs_h) = make_acts(k, b, xs_scale, &mut rng);
            let xq = upload(id, &xq_h)?;
            let xs = upload(id, &xs_h)?;
            let n_mid = (b * n_ff) as usize;
            let n_blk = (b * n_ff / 32) as usize;
            // Reference chain: gate, up (runtime bpack), swiglu, serial quantize.
            let gate = sentinel_buf(id, n_mid)?;
            let up = sentinel_buf(id, n_mid)?;
            let mut mid = sentinel_buf(id, n_mid)?;
            launch_kernel!(f_bp, cfg_gu, &stream, [gate.raw(), wg.raw(), xq.raw(), xs.raw(), k, n_ff, blocks, b])?;
            launch_kernel!(f_bp, cfg_gu, &stream, [up.raw(), wu.raw(), xq.raw(), xs.raw(), k, n_ff, blocks, b])?;
            swiglu.launch_clamped(&stream, &mut mid, &gate, &up, b * n_ff, clamp)?;
            let mid_xq_r: DeviceBuffer<i8> = upload(id, &vec![0x55i8; n_mid])?;
            let mid_xs_r = sentinel_buf(id, n_blk)?;
            let nb = n_blk as u32;
            let cfg_q = LaunchConfig { grid: (nb, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
            launch_kernel!(f_q, cfg_q, &stream, [mid_xq_r.raw(), mid_xs_r.raw(), mid.raw(), nb])?;
            // Fused candidate through the production wrapper.
            let mut mid_xq_c: DeviceBuffer<i8> = upload(id, &vec![0x55i8; n_mid])?;
            let mut mid_xs_c = sentinel_buf(id, n_blk)?;
            fused.launch(&stream, &mut mid_xq_c, &mut mid_xs_c, &wg, &wu, &xq, &xs, k, n_ff, b, clamp)?;
            // Down GEMV on both mids.
            let out_r = sentinel_buf(id, (b * k) as usize)?;
            let out_c = sentinel_buf(id, (b * k) as usize)?;
            let bd = n_ff / 32;
            launch_kernel!(f_bp, cfg_d, &stream, [out_r.raw(), wd.raw(), mid_xq_r.raw(), mid_xs_r.raw(), n_ff, k, bd, b])?;
            launch_kernel!(f_bp, cfg_d, &stream, [out_c.raw(), wd.raw(), mid_xq_c.raw(), mid_xs_c.raw(), n_ff, k, bd, b])?;
            stream.synchronize()?;

            let gate_h = download(&gate)?;
            let up_h = download(&up)?;
            let sat = gate_h.iter().zip(&up_h).filter(|(g, u)| **g > clamp || u.abs() > clamp).count();
            if xs_scale < 5e-5 {
                assert!(sat * 100 < n_mid, "smallest regime still saturates: {sat}/{n_mid} (lower xs_scale)");
            }
            if xs_scale > 0.5 {
                assert!(sat * 2 > n_mid, "largest regime does not saturate: {sat}/{n_mid}");
            }
            let (rq, cq) = (download(&mid_xq_r)?, download(&mid_xq_c)?);
            let (rs, cs) = (download(&mid_xs_r)?, download(&mid_xs_c)?);
            let (ro, co) = (download(&out_r)?, download(&out_c)?);
            assert_eq!(unwritten(&rs), 0);
            assert_eq!(unwritten(&cs), 0, "fused kernel left mid_xscale unwritten at b={b}");
            let dq = rq.iter().zip(&cq).filter(|(a, b)| a != b).count();
            let ds = bit_diff(&rs, &cs);
            let dout = bit_diff(&ro, &co);
            eprintln!(
                "shared b={b} xs_scale={xs_scale}: clamp hits {sat}/{n_mid}; mid_xq int8_diff={dq} mid_xscale bit_diff={ds} down bit_diff={dout}"
            );
            assert_eq!(dq, 0, "fused mid_xq differs at b={b} xs_scale={xs_scale}");
            assert_eq!(ds, 0, "fused mid_xscale differs at b={b} xs_scale={xs_scale}");
            assert_eq!(dout, 0, "down output differs at b={b} xs_scale={xs_scale}");
        }
    }
    eprintln!("PASS: fused shared chain bit-exact b=1..10 x 3 activation regimes");
    Ok(())
}
