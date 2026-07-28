//! Isolated fp8-KV A/B: quantize random K/V to e4m3fn+scale on-device, run the
//! fp8 decode kernel, compare to the f16 decode kernel on the same values.
use color_eyre::eyre::Result;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::gqa_attention::{decode_kv_splits_hg, GqaAttention};
use v4flash_kernels::iq2_xxs_tables::f16_to_f32;
use v4flash_kernels::laguna::LagunaOps;

fn pick_dgpu() -> Result<Device> {
    Ok(Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .expect("no gfx1201"))
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27; self.0 = x;
        let u = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as u32;
        (u as f32 / (1u32 << 24) as f32) * 2.0 - 1.0 // [-1,1)
    }
}
fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x007f_ffff;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 0xff { return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 }; }
    let e = exp - 127 + 15;
    if e >= 0x1f { return sign | 0x7c00; }
    if e <= 0 {
        if e < -10 { return sign; }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half_mant = (m >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let mut result = sign | half_mant;
        if (m & round_bit) != 0 && ((m & (round_bit - 1)) != 0 || (half_mant & 1) != 0) { result += 1; }
        return result;
    }
    let half_mant = (mant >> 13) as u16;
    let mut result = sign | ((e as u16) << 10) | half_mant;
    if (mant & 0x0000_1000) != 0 && ((mant & 0x0000_0fff) != 0 || (half_mant & 1) != 0) { result += 1; }
    result
}
fn round_f16(v: f32) -> (u16, f32) {
    let b = f32_to_f16(v);
    (b, f16_to_f32(b))
}

#[test]
#[ignore]
fn fp8_kv_decode_ab() -> Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let gqa = GqaAttention::for_arch(&arch)?;
    let ops = LagunaOps::for_arch(&arch)?;
    let st = Stream::new(dev.id)?;

    let n_kv_head = 8usize;
    let head_dim = 128usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for &(n_head, n_kv) in &[(48usize, 34usize), (48, 512), (48, 4096), (72, 600)] {
        let mut rng = Lcg(0x1234 ^ (n_kv as u64));
        let mut q_bits = vec![0u16; n_head * head_dim];
        for b in q_bits.iter_mut() { *b = round_f16(rng.f()).0; }
        let kv_len = n_kv * n_kv_head * head_dim;
        // f32 K/V, then f16-rounded bits for the f16 path.
        let mut k_f = vec![0f32; kv_len];
        let mut v_f = vec![0f32; kv_len];
        let mut k_bits = vec![0u16; kv_len];
        let mut v_bits = vec![0u16; kv_len];
        for i in 0..kv_len {
            let (kb, kv) = round_f16(rng.f()); k_bits[i] = kb; k_f[i] = kv;
            let (vb, vv) = round_f16(rng.f()); v_bits[i] = vb; v_f[i] = vv;
        }

        let mut d_q: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, q_bits.len())?;
        d_q.copy_from_host(&q_bits)?;
        let mut d_k16: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, kv_len)?; d_k16.copy_from_host(&k_bits)?;
        let mut d_v16: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, kv_len)?; d_v16.copy_from_host(&v_bits)?;

        // fp8: upload f32 K/V, quantize on-device.
        let mut d_kf: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, kv_len)?; d_kf.copy_from_host(&k_f)?;
        let mut d_vf: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, kv_len)?; d_vf.copy_from_host(&v_f)?;
        let rows = (n_kv * n_kv_head) as u32;
        let mut d_k8: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, kv_len)?;
        let mut d_v8: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, kv_len)?;
        let mut d_ks: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, n_kv * n_kv_head)?;
        let mut d_vs: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, n_kv * n_kv_head)?;
        ops.quantize_fp8_kv(&st, &mut d_k8, &mut d_ks, &d_kf, rows, head_dim as u32)?;
        ops.quantize_fp8_kv(&st, &mut d_v8, &mut d_vs, &d_vf, rows, head_dim as u32)?;

        let n_splits = decode_kv_splits_hg(n_kv as u32);
        let mut d_out16: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, n_head * head_dim)?;
        let mut d_out8: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, n_head * head_dim)?;
        let mk = |n: usize| -> Result<DeviceBuffer<f32>> { Ok(DeviceBuffer::new(dev.id, n)?) };
        let (mut op, mut mp, mut lp) = (mk(n_head*n_splits as usize*head_dim)?, mk(n_head*n_splits as usize)?, mk(n_head*n_splits as usize)?);

        gqa.single_query_splitkv_hg(&st, &mut d_out16, &mut op, &mut mp, &mut lp,
            &d_q, &d_k16, &d_v16, n_head as u32, n_kv_head as u32, head_dim as u32,
            n_kv as u32, n_splits, scale, 0, n_kv as u32)?;
        gqa.single_query_splitkv_hg_fp8(&st, &mut d_out8, &mut op, &mut mp, &mut lp,
            &d_q, &d_k8, &d_v8, &d_ks, &d_vs, n_head as u32, n_kv_head as u32, head_dim as u32,
            n_kv as u32, n_splits, scale, 0, n_kv as u32)?;
        st.synchronize()?;

        let mut o16 = vec![0f32; n_head*head_dim];
        let mut o8 = vec![0f32; n_head*head_dim];
        d_out16.copy_to_host(&mut o16)?; d_out8.copy_to_host(&mut o8)?;
        // scales sanity
        let mut ks_h = vec![0f32; n_kv*n_kv_head]; d_ks.copy_to_host(&mut ks_h)?;
        let n_bad_scale = ks_h.iter().filter(|s| !s.is_finite() || **s <= 0.0).count();

        let mut max_abs = 0f32; let mut max_rel = 0f32; let mut n_nan = 0;
        for i in 0..o16.len() {
            if !o8[i].is_finite() { n_nan += 1; continue; }
            let a = (o16[i]-o8[i]).abs();
            max_abs = max_abs.max(a);
            let denom = o16[i].abs().max(1e-3);
            max_rel = max_rel.max(a/denom);
        }
        eprintln!("n_head={n_head} n_kv={n_kv} n_splits={n_splits}: max_abs={max_abs:.4} max_rel={max_rel:.4} n_nan={n_nan} bad_scale={n_bad_scale} ks0={:.5}", ks_h[0]);
        assert_eq!(n_nan, 0, "fp8 output has NaN/inf");
        assert!(max_abs < 0.05, "fp8 vs f16 decode diverged: max_abs={max_abs}");
    }
    Ok(())
}

// Isolated decode-kernel TIMING: f16 vs fp8 at long ctx. The KV DRAM bytes halve,
// so this is the direct read-bandwidth signal ATT flagged (s_wait_loadcnt).
#[test]
#[ignore]
fn fp8_kv_decode_timing() -> Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let gqa = GqaAttention::for_arch(&arch)?;
    let ops = LagunaOps::for_arch(&arch)?;
    let st = Stream::new(dev.id)?;
    let (n_head, n_kv_head, head_dim) = (48usize, 8usize, 128usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    const ITERS: usize = 60;
    const WARM: usize = 8;
    // Contexts to time; override with e.g. FP8_AB_NKV=32768,196608
    let nkv_list: Vec<usize> = std::env::var("FP8_AB_NKV")
        .unwrap_or_else(|_| "32768,65536,100000,196608".to_string())
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("bad FP8_AB_NKV"))
        .collect();
    for n_kv in nkv_list {
        let mut rng = Lcg(0xabc ^ n_kv as u64);
        let mut q_bits = vec![0u16; n_head * head_dim];
        for b in q_bits.iter_mut() { *b = round_f16(rng.f()).0; }
        let kv_len = n_kv * n_kv_head * head_dim;
        let mut k_bits = vec![0u16; kv_len]; let mut v_bits = vec![0u16; kv_len];
        let mut k_f = vec![0f32; kv_len]; let mut v_f = vec![0f32; kv_len];
        for i in 0..kv_len {
            let (kb, kv) = round_f16(rng.f()); k_bits[i] = kb; k_f[i] = kv;
            let (vb, vv) = round_f16(rng.f()); v_bits[i] = vb; v_f[i] = vv;
        }
        let mut d_q: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, q_bits.len())?; d_q.copy_from_host(&q_bits)?;
        let mut d_k16: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, kv_len)?; d_k16.copy_from_host(&k_bits)?;
        let mut d_v16: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, kv_len)?; d_v16.copy_from_host(&v_bits)?;
        let mut d_kf: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, kv_len)?; d_kf.copy_from_host(&k_f)?;
        let mut d_vf: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, kv_len)?; d_vf.copy_from_host(&v_f)?;
        let rows = (n_kv * n_kv_head) as u32;
        let mut d_k8: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, kv_len)?;
        let mut d_v8: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, kv_len)?;
        let mut d_ks: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, n_kv*n_kv_head)?;
        let mut d_vs: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, n_kv*n_kv_head)?;
        ops.quantize_fp8_kv(&st, &mut d_k8, &mut d_ks, &d_kf, rows, head_dim as u32)?;
        ops.quantize_fp8_kv(&st, &mut d_v8, &mut d_vs, &d_vf, rows, head_dim as u32)?;
        let n_splits = decode_kv_splits_hg(n_kv as u32);
        let mk = |n: usize| -> Result<DeviceBuffer<f32>> { Ok(DeviceBuffer::new(dev.id, n)?) };
        let mut out = mk(n_head*head_dim)?;
        let mut op = mk(n_head*n_splits as usize*head_dim)?;
        let mut mp = mk(n_head*n_splits as usize)?;
        let mut lp = mk(n_head*n_splits as usize)?;
        eprintln!("n_kv={n_kv} n_splits={n_splits}:");
        // f16 timing
        for _ in 0..WARM { gqa.single_query_splitkv_hg(&st, &mut out, &mut op, &mut mp, &mut lp,
            &d_q, &d_k16, &d_v16, n_head as u32, n_kv_head as u32, head_dim as u32, n_kv as u32, n_splits, scale, 0, n_kv as u32)?; }
        st.synchronize()?;
        let t = std::time::Instant::now();
        for _ in 0..ITERS { gqa.single_query_splitkv_hg(&st, &mut out, &mut op, &mut mp, &mut lp,
            &d_q, &d_k16, &d_v16, n_head as u32, n_kv_head as u32, head_dim as u32, n_kv as u32, n_splits, scale, 0, n_kv as u32)?; }
        st.synchronize()?;
        let f16us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        // fp8 timing
        for _ in 0..WARM { gqa.single_query_splitkv_hg_fp8(&st, &mut out, &mut op, &mut mp, &mut lp,
            &d_q, &d_k8, &d_v8, &d_ks, &d_vs, n_head as u32, n_kv_head as u32, head_dim as u32, n_kv as u32, n_splits, scale, 0, n_kv as u32)?; }
        st.synchronize()?;
        let t = std::time::Instant::now();
        for _ in 0..ITERS { gqa.single_query_splitkv_hg_fp8(&st, &mut out, &mut op, &mut mp, &mut lp,
            &d_q, &d_k8, &d_v8, &d_ks, &d_vs, n_head as u32, n_kv_head as u32, head_dim as u32, n_kv as u32, n_splits, scale, 0, n_kv as u32)?; }
        st.synchronize()?;
        let fp8us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        // Achieved read BW: each kernel streams the whole K+V cache once.
        // f16 = 2 B/elem; fp8 = 1 B/elem + the f32 per-(token,kv_head) scale sidecar.
        let kv_elems = (n_kv * n_kv_head * head_dim) as f64;
        let f16_bytes = kv_elems * 2.0 * 2.0;
        let fp8_bytes = kv_elems * 1.0 * 2.0 + (n_kv * n_kv_head) as f64 * 4.0 * 2.0;
        let gbs = |bytes: f64, us: f64| bytes / (us * 1e-6) / 1e9;
        let (f16gbs, fp8gbs) = (gbs(f16_bytes, f16us), gbs(fp8_bytes, fp8us));
        eprintln!("  f16: {f16us:.1} us   fp8: {fp8us:.1} us   => fp8 {:.2}x ({:+.1}%)", f16us/fp8us, 100.0*(f16us-fp8us)/f16us);
        eprintln!("  BW: f16 {f16gbs:.0} GB/s ({:.0}% of 600)   fp8 {fp8gbs:.0} GB/s ({:.0}% of 600)",
            100.0*f16gbs/600.0, 100.0*fp8gbs/600.0);
    }
    Ok(())
}
