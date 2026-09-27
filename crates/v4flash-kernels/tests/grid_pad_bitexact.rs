//! Bit-exactness of the `V41_GRID_PAD` launches (2026-09-27 round 2, item a):
//! each padded wrapper launch (one extra, idle work-group in grid.x) against a
//! raw launch of the same kernel with the exact production grid, on synthetic
//! data (no model load, < 40 MB of device memory, so it runs beside a live hub):
//!
//!   1. `Fp4KvQuant::launch_fp8_window` (`fp8_act_quant_inplace`, width 512)
//!      at n_rows 1..512 incl. the slow 128/256/512 grids and odd tails.
//!   2. `IndexerQat::launch_fp4` (`indexer_fp4`) at 32*b rows, b = 1..128, + tails.
//!   3. `RopeTail::launch_{forward,inverse}_batched` (`rope_tail_batched`) at
//!      the q (64 x 512) and idx-q (32 x 128) shapes, b = 1..64.
//!   4. `Q8KQuantize::launch_cast_f16_2d` (`f32_to_f16_cast_2d`) at the replay
//!      heads / low shapes and tails, incl. an out_pitch > cols destination whose
//!      pitch padding must stay untouched.
//!
//! Every output is pre-filled with a sentinel, so an element the padded launch
//! failed to write (or wrote twice differently) shows. Passes with the knob on
//! (default) and off (`V41_GRID_PAD=0`: then both sides use the exact grid).
//!
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test
//! grid_pad_bitexact -- --ignored --test-threads=1 --nocapture`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, launch_kernel, Device, DeviceBuffer, LaunchConfig, Stream};
use v4flash_kernels::fp4_kv::Fp4KvQuant;
use v4flash_kernels::{IndexerQat, Q8KQuantize, RopeParams, RopeTail};

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
    /// Uniform in [-a, a), with an occasional exact zero and a rare large spike.
    fn val(&mut self, a: f32) -> f32 {
        match self.next() % 257 {
            0 => 0.0,
            1 => 300.0 * a,
            _ => (self.unit() * 2.0 - 1.0) * a,
        }
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

fn f32_bit_diff(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn pad_on() -> bool {
    std::env::var("V41_GRID_PAD").as_deref() != Ok("0")
}

/// In-place kernels: the wrapper and the raw exact-grid launch each run on
/// their own copy of `host` (+ one trailing sentinel row that must survive).
#[test]
#[ignore]
fn fp8_window_and_indexer_fp4_pad_bit_exact() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x6A1D_0001);
    eprintln!("arch {arch}, V41_GRID_PAD {}", if pad_on() { "on" } else { "off" });

    // fp8_act_quant_inplace, width 512 (window KV), one sentinel row past n_rows.
    let fp8 = Fp4KvQuant::for_arch(&arch)?;
    const W: u32 = 512;
    for &n_rows in &[1u32, 2, 4, 7, 64, 127, 128, 129, 255, 256, 257, 511, 512] {
        let n = ((n_rows + 1) * W) as usize;
        let mut host: Vec<f32> = (0..n).map(|_| rng.val(4.0)).collect();
        for v in &mut host[(n_rows * W) as usize..] {
            *v = f32::from_bits(0x7FC0_BEEF);
        }
        let mut a = upload(id, &host)?;
        let b = upload(id, &host)?;
        fp8.launch_fp8_window(&stream, &mut a, n_rows, W)?;
        let f = fp8.module().get_function("fp8_act_quant_inplace")?;
        let cfg = LaunchConfig { grid: (n_rows, 1, 1), block: (W, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &stream, [b.raw(), n_rows, W])?;
        stream.synchronize()?;
        let (ha, hb) = (download(&a)?, download(&b)?);
        let d = f32_bit_diff(&ha, &hb);
        let tail = f32_bit_diff(&ha[(n_rows * W) as usize..], &host[(n_rows * W) as usize..]);
        eprintln!("fp8_act_quant_inplace n_rows={n_rows}: bit_diff={d} sentinel_row_changed={tail}");
        assert_eq!(d, 0, "fp8 window n_rows={n_rows}: padded launch differs");
        assert_eq!(tail, 0, "fp8 window n_rows={n_rows}: the idle WG touched row n_rows");
    }

    // indexer_fp4: 32 heads x b rows of 128.
    let qat = IndexerQat::for_arch(&arch)?;
    for &rows in &[32u32, 128, 256, 512, 1024, 2048, 4096, 33, 511, 2049] {
        let n = ((rows + 1) * 128) as usize;
        let mut host: Vec<f32> = (0..n).map(|_| rng.val(2.0)).collect();
        for v in &mut host[(rows * 128) as usize..] {
            *v = f32::from_bits(0x7FC0_BEEF);
        }
        let mut a = upload(id, &host)?;
        let b = upload(id, &host)?;
        qat.launch_fp4(&stream, &mut a, rows)?;
        let f = qat.module().get_function("indexer_fp4")?;
        let cfg = LaunchConfig { grid: (rows, 1, 1), block: (128, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &stream, [b.raw(), rows])?;
        stream.synchronize()?;
        let (ha, hb) = (download(&a)?, download(&b)?);
        let d = f32_bit_diff(&ha, &hb);
        let tail = f32_bit_diff(&ha[(rows * 128) as usize..], &host[(rows * 128) as usize..]);
        eprintln!("indexer_fp4 n_rows={rows}: bit_diff={d} sentinel_row_changed={tail}");
        assert_eq!(d, 0, "indexer_fp4 n_rows={rows}: padded launch differs");
        assert_eq!(tail, 0, "indexer_fp4 n_rows={rows}: the idle WG touched row n_rows");
    }
    eprintln!("PASS: fp8 window / indexer_fp4 grid pad bit-exact");
    Ok(())
}

#[test]
#[ignore]
fn rope_batched_pad_bit_exact() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let stream = Stream::new(id)?;
    let rope = RopeTail::for_arch(&arch)?;
    let mut rng = Lcg(0x6A1D_0002);
    // ext_factor = 0: the wrapper's derived args are then plain functions of
    // freq_base / attn_factor, which the raw launch below reproduces exactly.
    let params = RopeParams {
        freq_base: 10000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        n_ctx_orig: 0,
    };
    const N_ROT: u32 = 64;
    let theta_scale = params.freq_base.powf(-2.0 / N_ROT as f32);
    for &(n_head, head_dim) in &[(64u32, 512u32), (32, 128), (1, 512)] {
        for &b in &[1u32, 2, 4, 5, 16, 32, 33, 64] {
            for inverse in [false, true] {
                let n = (b * n_head * head_dim) as usize;
                let host: Vec<f32> = (0..n).map(|_| rng.val(3.0)).collect();
                let pos: Vec<i32> = (0..b).map(|i| (1000 + 37 * i) as i32).collect();
                let pos_d = upload(id, &pos)?;
                let mut a = upload(id, &host)?;
                let bb = upload(id, &host)?;
                if inverse {
                    rope.launch_inverse_batched(&stream, &mut a, &pos_d, n_head, head_dim, N_ROT, b, &params)?;
                } else {
                    rope.launch_forward_batched(&stream, &mut a, &pos_d, n_head, head_dim, N_ROT, b, &params)?;
                }
                let f = rope.module().get_function("rope_tail_batched")?;
                let cfg = LaunchConfig { grid: (n_head, 1, b), block: (N_ROT / 2, 1, 1), shared_mem_bytes: 0 };
                let inv_i: i32 = if inverse { 1 } else { 0 };
                let (zero, one) = (0.0f32, 1.0f32);
                launch_kernel!(f, cfg, &stream, [
                    bb.raw(), pos_d.raw(), n_head, head_dim, N_ROT,
                    theta_scale, one, zero, one, zero, zero, inv_i
                ])?;
                stream.synchronize()?;
                let d = f32_bit_diff(&download(&a)?, &download(&bb)?);
                eprintln!("rope_tail_batched n_head={n_head} head_dim={head_dim} b={b} inverse={inverse}: bit_diff={d}");
                assert_eq!(d, 0, "rope n_head={n_head} b={b} inverse={inverse}: padded launch differs");
            }
        }
    }
    eprintln!("PASS: rope_tail_batched grid pad bit-exact");
    Ok(())
}

#[test]
#[ignore]
fn cast_f16_2d_pad_bit_exact() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let stream = Stream::new(id)?;
    let q8k = Q8KQuantize::for_arch(&arch)?;
    let mut rng = Lcg(0x6A1D_0003);
    // (rows, cols, out_pitch): replay heads (16 x 32768) and low (64 x 8192) =
    // the slow 256-WG grids, decode / prefill neighbours, a partial last WG
    // (3 x 8200), and a pitched destination whose padding must stay untouched.
    let shapes: &[(u32, u32, u32)] = &[
        (16, 32768, 32768), (64, 8192, 8192), (4, 32768, 32768), (1, 5120, 5120),
        (512, 1280, 1280), (3, 8200, 8200), (32, 8192, 8192), (5, 1280, 1344),
    ];
    for &(rows, cols, pitch) in shapes {
        let x: Vec<f32> = (0..(rows * cols) as usize).map(|_| rng.val(8.0)).collect();
        let xd = upload(id, &x)?;
        let sentinel = vec![0xBEEFu16; ((rows + 1) * pitch) as usize];
        let mut a = upload(id, &sentinel)?;
        let b = upload(id, &sentinel)?;
        q8k.launch_cast_f16_2d(&stream, &mut a, &xd, rows, cols, pitch)?;
        let f = q8k.module().get_function("f32_to_f16_cast_2d")?;
        let threads = rows * (cols / 8);
        let cfg = LaunchConfig { grid: (threads.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &stream, [b.raw(), xd.raw(), rows, cols, pitch])?;
        stream.synchronize()?;
        let (ha, hb) = (download(&a)?, download(&b)?);
        let d = ha.iter().zip(&hb).filter(|(p, q)| p != q).count();
        let untouched = (0..rows as usize)
            .flat_map(|r| (cols as usize..pitch as usize).map(move |c| r * pitch as usize + c))
            .chain((rows * pitch) as usize..((rows + 1) * pitch) as usize)
            .filter(|&i| ha[i] != 0xBEEF)
            .count();
        let unwritten = (0..rows as usize)
            .flat_map(|r| (0..cols as usize).map(move |c| r * pitch as usize + c))
            .filter(|&i| ha[i] == 0xBEEF && hb[i] == 0xBEEF)
            .count();
        eprintln!("cast_f16_2d rows={rows} cols={cols} pitch={pitch}: diff={d} pitch/tail touched={untouched} both-unwritten={unwritten}");
        assert_eq!(d, 0, "cast rows={rows} cols={cols}: padded launch differs");
        assert_eq!(untouched, 0, "cast rows={rows} cols={cols}: wrote outside the rows x cols region");
    }
    eprintln!("PASS: f32_to_f16_cast_2d grid pad bit-exact");
    Ok(())
}
