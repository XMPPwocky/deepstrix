//! Bit-exactness of the 2026-09-27 round-2 decode launch-count fusions
//! (`V41_DEC_FUSE`, item d): each fused wrapper against the chain of production
//! wrappers it replaces, on synthetic data (no model load, < 30 MB of device
//! memory, runs beside a live hub), b = 1..16, plain and YaRN rope parameters:
//!
//!   1. `RopeTail::launch_kv_rms_rope_fp8` vs `RmsNorm::launch_weighted_batched(512)`
//!      -> `RopeTail::launch_forward_batched(1 head)` -> `Fp4KvQuant::launch_fp8_window`.
//!   2. `RmsNorm::launch_weighted_quant_q8_1280` vs `launch_weighted_batched(1280)`
//!      -> `Q8_0Matvec::quantize_input_batched(1280)` (qr_normed, xq, xscale).
//!   3. `RopeTail::launch_forward_batched_copy` vs a copy + `launch_forward_batched`.
//!   4. `RopeTail::launch_inverse_quant_q8` vs `launch_inverse_batched` ->
//!      `quantize_input_batched(32768)` (heads, xq, xscale).
//!   5. `HcPost::launch_from_split_batched_add` vs `VecAddInplace::launch` ->
//!      `HcPost::launch_from_split_batched`.
//!
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test
//! decode_fusion_bitexact -- --ignored --test-threads=1 --nocapture`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::fp4_kv::Fp4KvQuant;
use v4flash_kernels::q8_0::Q8_0Matvec;
use v4flash_kernels::{HcPost, RmsNorm, RopeParams, RopeTail, VecAddInplace};

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
    fn sym(&mut self, a: f32) -> f32 {
        ((self.next() & 0xFFFFFF) as f32 / 16777216.0 * 2.0 - 1.0) * a
    }
    /// Mostly uniform, with exact zeros, a huge spike and a tiny row segment.
    fn val(&mut self, a: f32) -> f32 {
        match self.next() % 331 {
            0 => 0.0,
            1 => 2000.0 * a,
            2 => 1e-7 * a,
            _ => self.sym(a),
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

fn f32_diff(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn rope_params() -> [(&'static str, RopeParams); 2] {
    [
        ("plain", RopeParams { freq_base: 10000.0, freq_scale: 1.0, ext_factor: 0.0, attn_factor: 1.0,
                               beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 0 }),
        ("yarn", RopeParams { freq_base: 160000.0, freq_scale: 0.0625, ext_factor: 1.0, attn_factor: 1.0,
                              beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65536 }),
    ]
}

#[test]
#[ignore]
fn decode_fusions_match_their_chains() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let stream = Stream::new(id)?;
    let rms = RmsNorm::for_arch(&arch)?;
    let rope = RopeTail::for_arch(&arch)?;
    let fp4 = Fp4KvQuant::for_arch(&arch)?;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let hcp = HcPost::for_arch(&arch)?;
    let vadd = VecAddInplace::for_arch(&arch)?;
    let mut rng = Lcg(0x5EED_D00D);
    const BMAX: usize = 16;
    let eps = 1e-6f32;
    let w512 = upload(id, &(0..512).map(|_| 0.2 + rng.sym(1.0).abs()).collect::<Vec<f32>>())?;
    let w1280 = upload(id, &(0..1280).map(|_| 0.2 + rng.sym(1.0).abs()).collect::<Vec<f32>>())?;
    let mut cases = 0usize;
    for b in 1..=BMAX as u32 {
        let pos_h: Vec<i32> = (0..b).map(|i| (1 + 70_001 * (i as i64 + 1) % 300_000) as i32).collect();
        let pos = upload(id, &pos_h)?;
        for (pname, params) in rope_params() {
            // 1. kv chain
            let kv_raw_h: Vec<f32> = (0..b * 512).map(|_| rng.val(3.0)).collect();
            let kv_raw = upload(id, &kv_raw_h)?;
            let mut a: DeviceBuffer<f32> = DeviceBuffer::new(id, BMAX * 512)?;
            let mut f = upload(id, &vec![f32::NAN; BMAX * 512])?;
            rms.launch_weighted_batched(&stream, &mut a, &kv_raw, &w512, 512, eps, b)?;
            rope.launch_forward_batched(&stream, &mut a, &pos, 1, 512, 64, b, &params)?;
            fp4.launch_fp8_window(&stream, &mut a, b, 512)?;
            rope.launch_kv_rms_rope_fp8(&stream, &mut f, &kv_raw, &w512, eps, &pos, 64, b, &params)?;
            stream.synchronize()?;
            let n = (b * 512) as usize;
            let d = f32_diff(&download(&a)?[..n], &download(&f)?[..n]);
            assert_eq!(d, 0, "kv_rms_rope_fp8 differs at b={b} {pname}");

            // 3. q copy + rope (64 heads)
            let q_h: Vec<f32> = (0..b * 32768).map(|_| rng.val(2.0)).collect();
            let q = upload(id, &q_h)?;
            let mut qa = upload(id, &q_h)?;
            let mut qf = upload(id, &vec![f32::NAN; b as usize * 32768])?;
            rope.launch_forward_batched(&stream, &mut qa, &pos, 64, 512, 64, b, &params)?;
            assert!(RopeTail::copy_ok(&q, &qf, 512, 64));
            rope.launch_forward_batched_copy(&stream, &mut qf, &q, &pos, 64, 512, 64, b, &params)?;
            stream.synchronize()?;
            let d = f32_diff(&download(&qa)?, &download(&qf)?);
            assert_eq!(d, 0, "rope_tail_batched_copy differs at b={b} {pname}");

            // 4. inverse rope + heads quantize
            let h_h: Vec<f32> = (0..b * 32768).map(|_| rng.val(2.0)).collect();
            let mut ha = upload(id, &h_h)?;
            let mut hf = upload(id, &h_h)?;
            let mut xqa: DeviceBuffer<i8> = upload(id, &vec![0x55i8; b as usize * 32768])?;
            let mut xsa = upload(id, &vec![f32::NAN; b as usize * 1024])?;
            let mut xqf: DeviceBuffer<i8> = upload(id, &vec![0x33i8; b as usize * 32768])?;
            let mut xsf = upload(id, &vec![f32::INFINITY; b as usize * 1024])?;
            rope.launch_inverse_batched(&stream, &mut ha, &pos, 64, 512, 64, b, &params)?;
            q8.quantize_input_batched(&stream, &mut xqa, &mut xsa, &ha, 32768, b)?;
            rope.launch_inverse_quant_q8(&stream, &mut hf, &mut xqf, &mut xsf, &pos, 64, 512, 64, b, &params)?;
            stream.synchronize()?;
            assert_eq!(f32_diff(&download(&ha)?, &download(&hf)?), 0, "rope_inv_quant heads differ at b={b} {pname}");
            assert_eq!(download(&xqa)?, download(&xqf)?, "rope_inv_quant xq differs at b={b} {pname}");
            assert_eq!(f32_diff(&download(&xsa)?, &download(&xsf)?), 0, "rope_inv_quant xscale differs at b={b} {pname}");
            cases += 1;
        }

        // 2. q_a norm + quantize
        let qr_h: Vec<f32> = (0..b * 1280).map(|_| rng.val(2.0)).collect();
        let qr = upload(id, &qr_h)?;
        let mut na = upload(id, &vec![f32::NAN; b as usize * 1280])?;
        let mut nf = upload(id, &vec![f32::INFINITY; b as usize * 1280])?;
        let mut xqa: DeviceBuffer<i8> = upload(id, &vec![0x55i8; b as usize * 1280])?;
        let mut xsa = upload(id, &vec![f32::NAN; b as usize * 40])?;
        let mut xqf: DeviceBuffer<i8> = upload(id, &vec![0x33i8; b as usize * 1280])?;
        let mut xsf = upload(id, &vec![f32::INFINITY; b as usize * 40])?;
        rms.launch_weighted_batched(&stream, &mut na, &qr, &w1280, 1280, eps, b)?;
        q8.quantize_input_batched(&stream, &mut xqa, &mut xsa, &na, 1280, b)?;
        rms.launch_weighted_quant_q8_1280(&stream, &mut nf, &mut xqf, &mut xsf, &qr, &w1280, eps, b)?;
        stream.synchronize()?;
        assert_eq!(f32_diff(&download(&na)?, &download(&nf)?), 0, "rms_quant qr_normed differs at b={b}");
        assert_eq!(download(&xqa)?, download(&xqf)?, "rms_quant xq differs at b={b}");
        assert_eq!(f32_diff(&download(&xsa)?, &download(&xsf)?), 0, "rms_quant xscale differs at b={b}");

        // 5. remote add + hc_post (n_hc 4, n_embd 5120, split stride 24 + 4 + 16)
        let (ne, nh, nw) = (5120u32, 4u32, 24u32);
        let ss = (nw + nh + nh * nh) as usize;
        let bo_h: Vec<f32> = (0..b * ne).map(|_| rng.val(1.0)).collect();
        let rem_h: Vec<f32> = (0..b * ne).map(|_| rng.val(1.0)).collect();
        let rh = upload(id, &(0..b * nh * ne).map(|_| rng.val(1.0)).collect::<Vec<f32>>())?;
        let sp = upload(id, &(0..b as usize * ss).map(|_| rng.sym(1.5)).collect::<Vec<f32>>())?;
        let rem = upload(id, &rem_h)?;
        let mut bo_a = upload(id, &bo_h)?;
        let bo_f = upload(id, &bo_h)?;
        let mut oa = upload(id, &vec![f32::NAN; (b * nh * ne) as usize])?;
        let mut of = upload(id, &vec![f32::INFINITY; (b * nh * ne) as usize])?;
        vadd.launch(&stream, &mut bo_a, &rem, b * ne)?;
        hcp.launch_from_split_batched(&stream, &mut oa, &bo_a, &rh, &sp, nw, ne, nh, b)?;
        hcp.launch_from_split_batched_add(&stream, &mut of, &bo_f, &rem, &rh, &sp, nw, ne, nh, b)?;
        stream.synchronize()?;
        assert_eq!(f32_diff(&download(&oa)?, &download(&of)?), 0, "hc_post_add differs at b={b}");
        eprintln!("b={b}: kv / q copy / heads (plain + yarn), q_a rms_quant, hc_post_add bit-exact");
    }
    eprintln!("PASS: {cases} rope-param cases + 16 rms_quant / hc_post_add batches bit-exact");
    Ok(())
}
