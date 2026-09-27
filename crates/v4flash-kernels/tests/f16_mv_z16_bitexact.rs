//! Bit-exactness of `f16_matvec_batched_z16_n<NB>` (2026-09-27 round 2, item c)
//! against the production `f16_matvec_batched` (grid.z = batch), on synthetic
//! data (no model load, < 60 MB of device memory, runs beside a live hub):
//! every NB symbol explicitly (grid.z = ceil(b / NB), partial last slices incl.)
//! and the `F16Matvec::matvec_batched_z16` wrapper (NB pick + `V41_GRID_PAD`
//! idle WG), at the three wired shapes -- indexer q 4096 x 1280, idx proj
//! 32 x 5120, ratio-1 compressor 512 x 5120 -- plus the router 384 x 5120, a row
//! tail (1000 x 1024), a plain-loop k (41 x 5152, k % 256 != 0) and a tiny k.
//! Outputs pre-filled with a NaN sentinel: an unwritten or stray write shows.
//! Passes with the knobs on (default) and `V41_F16_MV_Z16=0` / `V41_GRID_PAD=0`.
//!
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test
//! f16_mv_z16_bitexact -- --ignored --test-threads=1 --nocapture`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, launch_kernel, Device, DeviceBuffer, LaunchConfig, Stream};
use v4flash_kernels::F16Matvec;

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
    /// A finite f16 bit pattern, |v| in ~[2^-9, 2^0), random sign.
    fn f16_bits(&mut self) -> u16 {
        let exp = 6 + (self.next() % 9) as u16;
        let man = (self.next() & 0x3FF) as u16;
        (((self.next() & 1) as u16) << 15) | (exp << 10) | man
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

const SENTINEL: f32 = f32::from_bits(0x7FC0_BEEF);

fn sentinel_buf(id: i32, n: usize) -> eyre::Result<DeviceBuffer<f32>> {
    upload(id, &vec![SENTINEL; n])
}

fn bit_diff(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn unwritten(a: &[f32]) -> usize {
    a.iter().filter(|x| x.to_bits() == SENTINEL.to_bits()).count()
}

fn f16_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

#[test]
fn z16_site_gates() {
    use v4flash_kernels::f16::{z16_comp_ratio1_for, Z16_PROJ_MAX_B, Z16_ROUTER_MIN_B};
    assert_eq!(Z16_PROJ_MAX_B, 4, "idx proj: z16 loses x1.36 at 8 rows");
    assert_eq!(Z16_ROUTER_MIN_B, 48, "router: z16 loses to _h20 at <= 32 rows");
    for b in 1..=64u32 {
        assert_eq!(z16_comp_ratio1_for(b), !(3..=4).contains(&b), "compressor gate at b={b}");
    }
}

#[test]
#[ignore]
fn z16_matches_grid_z_batched_matvec() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let f16 = F16Matvec::for_arch(&arch)?;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_2C16);
    let shapes: [(u32, u32); 7] = [(4096, 1280), (32, 5120), (512, 5120), (384, 5120), (1000, 1024), (41, 5152), (16, 256)];
    let mut cases = 0usize;
    for &(n_rows, k) in &shapes {
        let w_bits: Vec<u16> = (0..(n_rows * k)).map(|_| rng.f16_bits()).collect();
        let w = upload(id, &f16_bytes(&w_bits))?;
        let bs: &[u32] = if (n_rows, k) == (4096, 1280) || (n_rows, k) == (512, 5120) {
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 33, 64]
        } else {
            &[1, 2, 3, 5, 8, 16, 17, 64]
        };
        for &b in bs {
            let x_h: Vec<f32> = (0..(b * k)).map(|_| rng.sym(3.0)).collect();
            let x = upload(id, &x_h)?;
            let n_out = (b * n_rows) as usize;
            let out_ref = sentinel_buf(id, n_out)?;
            let f = f16.wide_module().get_function("f16_matvec_batched")?;
            let cfg = LaunchConfig { grid: (n_rows.div_ceil(8), 1, b), block: (256, 1, 1), shared_mem_bytes: 0 };
            launch_kernel!(f, cfg, &stream, [out_ref.raw(), w.raw(), x.raw(), k, n_rows])?;
            stream.synchronize()?;
            let r = download(&out_ref)?;
            assert_eq!(unwritten(&r), 0, "f16_matvec_batched left outputs unwritten at {n_rows}x{k} b={b}");
            for nb in [1u32, 2, 4, 8, 16] {
                for pad in [0u32, 1] {
                    let out = sentinel_buf(id, n_out)?;
                    let f = f16.wide_module().get_function(&format!("f16_matvec_batched_z16_n{nb}"))?;
                    let cfg = LaunchConfig { grid: (n_rows.div_ceil(8) + pad, 1, b.div_ceil(nb)), block: (256, 1, 1), shared_mem_bytes: 0 };
                    launch_kernel!(f, cfg, &stream, [out.raw(), w.raw(), x.raw(), k, n_rows, b])?;
                    stream.synchronize()?;
                    let d = bit_diff(&r, &download(&out)?);
                    assert_eq!(d, 0, "z16_n{nb} pad={pad} differs at {n_rows}x{k} b={b} ({d} lanes)");
                }
            }
            let mut out_w = sentinel_buf(id, n_out)?;
            f16.matvec_batched_z16(&stream, &mut out_w, &w, &x, n_rows, k, b)?;
            let mut out_r = sentinel_buf(id, n_out)?;
            f16.matvec_batched_router(&stream, &mut out_r, &w, &x, n_rows, k, b)?;
            stream.synchronize()?;
            let d = bit_diff(&r, &download(&out_w)?);
            let dr = bit_diff(&r, &download(&out_r)?);
            eprintln!("mv {n_rows}x{k} b={b}: z16 n1..n16 x pad 0/1 bit-exact, wrapper bit_diff={d}, router wrapper bit_diff={dr}");
            assert_eq!(d, 0, "matvec_batched_z16 differs at {n_rows}x{k} b={b}");
            assert_eq!(dr, 0, "matvec_batched_router differs at {n_rows}x{k} b={b}");
            cases += 1;
        }
    }
    eprintln!("PASS: {cases} (shape, b) cases, every NB symbol + wrapper bit-exact");
    Ok(())
}
