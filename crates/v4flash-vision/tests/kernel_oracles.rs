//! Per-kernel GPU-vs-CPU oracles at a size where grid.y > 1 and the
//! attention K-loop runs many tiles (n = 1024). IGNORED by default; no
//! mmproj needed (synthetic weights). Device 1 (gfx1151) only.

use v4flash_hip::{Device, DeviceBuffer, Stream};
use v4flash_vision::kernels::VitKernels;
use v4flash_vision::reference;
use v4flash_vision::rope::{apply_rotary_host, vision_cos_sin};
use v4flash_vision::{VIT_DIM, VIT_HEAD_DIM, VIT_N_HEADS, VIT_RMS_EPS};

use v4flash_core::kquants::{f16_to_f32, f32_to_f16_bits};

fn device() -> Device {
    let id: i32 = std::env::var("DEEPSTRIX_VISION_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    assert_ne!(id, 0, "refusing to touch the dGPU");
    Device::new(id)
}

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * s).collect()
    }
}

/// f32 -> f16 bits, and the exactly-representable f32 the GPU will see.
fn to_f16(v: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = v.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let back: Vec<f32> = bits.iter().map(|&b| f16_to_f32(b)).collect();
    (bits, back)
}

fn stats(got: &[f32], want: &[f32]) -> (f32, f32, f32) {
    assert_eq!(got.len(), want.len());
    let n = want.len() as f64;
    let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n).sqrt() as f32;
    let mx = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let mean = (got.iter().zip(want).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / n) as f32;
    (mx / rms.max(1e-20), mean / rms.max(1e-20), rms)
}

fn env() -> (Device, VitKernels, Stream) {
    let d = device();
    d.set_current().unwrap();
    let arch = d.properties().unwrap().gcn_arch_name;
    let kk = VitKernels::for_arch(&arch).unwrap();
    let st = Stream::new(d.id).unwrap();
    (d, kk, st)
}

fn up16(d: &Device, v: &[u16]) -> DeviceBuffer<u16> {
    let mut b = DeviceBuffer::<u16>::new(d.id, v.len()).unwrap();
    b.copy_from_host(v).unwrap();
    b
}
fn up32(d: &Device, v: &[f32]) -> DeviceBuffer<f32> {
    let mut b = DeviceBuffer::<f32>::new(d.id, v.len()).unwrap();
    b.copy_from_host(v).unwrap();
    b
}

#[test]
#[ignore]
fn gemm_matches_cpu() {
    let (d, kk, st) = env();
    let mut rng = Rng(7);
    for &(n, k, m) in &[(24usize, 1024usize, 3072usize), (1024, 1024, 3072), (1024, 2816, 1024), (350, 9216, 4096)] {
        let (xb, xf) = to_f16(&rng.vec(n * k, 1.0));
        let (wb, _) = to_f16(&rng.vec(m * k, 0.05));
        let bias = rng.vec(m, 0.1);
        let want = reference::linear(&xf, n, k, &wb, m, Some(&bias));
        let xd = up16(&d, &xb);
        let wd = up16(&d, &wb);
        let bd = up32(&d, &bias);
        let mut od = DeviceBuffer::<f32>::new(d.id, n * m).unwrap();
        kk.gemm(&st, Some(&mut od), None, &xd, &wd, Some(&bd), n as u32, k as u32, m as u32, 0).unwrap();
        st.synchronize().unwrap();
        let mut got = vec![0f32; n * m];
        od.copy_to_host(&mut got).unwrap();
        let (mx, mean, rms) = stats(&got, &want);
        eprintln!("gemm n={n:<5} k={k:<5} m={m:<5} rms={rms:.4} max/rms={mx:.3e} mean/rms={mean:.3e}");
        assert!(mx < 5e-3, "gemm {n}x{k}x{m}: max/rms {mx:.3e}");
    }
    drop(st);
    d.synchronize().unwrap();
}

#[test]
#[ignore]
fn attention_matches_cpu() {
    let (d, kk, st) = env();
    let mut rng = Rng(11);
    for &n in &[24usize, 64, 65, 128, 1024, 1610] {
        let (qb, qf) = to_f16(&rng.vec(n * VIT_DIM, 1.0));
        let (kb, kf) = to_f16(&rng.vec(n * VIT_DIM, 1.0));
        let (vb, vf) = to_f16(&rng.vec(n * VIT_DIM, 1.0));
        let want = reference::attention(&qf, &kf, &vf, n);
        let qd = up16(&d, &qb);
        let kd = up16(&d, &kb);
        let vd = up16(&d, &vb);
        let mut od = DeviceBuffer::<u16>::new(d.id, n * VIT_DIM).unwrap();
        let scale = 1.0 / (VIT_HEAD_DIM as f32).sqrt();
        kk.attention(&st, &mut od, &qd, &kd, &vd, n as u32, scale).unwrap();
        st.synchronize().unwrap();
        let mut gb = vec![0u16; n * VIT_DIM];
        od.copy_to_host(&mut gb).unwrap();
        let got: Vec<f32> = gb.iter().map(|&b| f16_to_f32(b)).collect();
        let (mx, mean, rms) = stats(&got, &want);
        eprintln!("attn n={n:<5} rms={rms:.4} max/rms={mx:.3e} mean/rms={mean:.3e}");
        assert!(mx < 2e-2, "attention n={n}: max/rms {mx:.3e}");
    }
    drop(st);
    d.synchronize().unwrap();
}

#[test]
#[ignore]
fn rmsnorm_and_rope_match_cpu() {
    let (d, kk, st) = env();
    let mut rng = Rng(13);
    let n = 1024usize;
    // rmsnorm
    let x = rng.vec(n * VIT_DIM, 1.0);
    let w = rng.vec(VIT_DIM, 1.0);
    let want = reference::rms_norm(&x, n, VIT_DIM, &w);
    let xd = up32(&d, &x);
    let wd = up32(&d, &w);
    let mut od = DeviceBuffer::<u16>::new(d.id, n * VIT_DIM).unwrap();
    kk.rmsnorm_f16(&st, &mut od, &xd, &wd, n as u32, VIT_DIM as u32, VIT_RMS_EPS).unwrap();
    st.synchronize().unwrap();
    let mut gb = vec![0u16; n * VIT_DIM];
    od.copy_to_host(&mut gb).unwrap();
    let got: Vec<f32> = gb.iter().map(|&b| f16_to_f32(b)).collect();
    let (mx, mean, _) = stats(&got, &want);
    eprintln!("rmsnorm n={n} max/rms={mx:.3e} mean/rms={mean:.3e}");
    assert!(mx < 5e-3);

    // rope_split on a fused qkv row block
    let (n_h, n_w) = (32u32, 32u32);
    let n = (n_h * n_w) as usize;
    let qkv = rng.vec(n * 3 * VIT_DIM, 1.0);
    let (cos, sin) = vision_cos_sin(n_h, n_w);
    let mut wq: Vec<f32> = (0..n).flat_map(|t| qkv[t * 3072..t * 3072 + 1024].to_vec()).collect();
    let mut wk: Vec<f32> = (0..n).flat_map(|t| qkv[t * 3072 + 1024..t * 3072 + 2048].to_vec()).collect();
    let wv: Vec<f32> = (0..n).flat_map(|t| qkv[t * 3072 + 2048..t * 3072 + 3072].to_vec()).collect();
    apply_rotary_host(&mut wq, &cos, &sin);
    apply_rotary_host(&mut wk, &cos, &sin);
    let qkvd = up32(&d, &qkv);
    let cd = up32(&d, &cos);
    let sd = up32(&d, &sin);
    let mut qd = DeviceBuffer::<u16>::new(d.id, n * VIT_DIM).unwrap();
    let mut kd = DeviceBuffer::<u16>::new(d.id, n * VIT_DIM).unwrap();
    let mut vd = DeviceBuffer::<u16>::new(d.id, n * VIT_DIM).unwrap();
    kk.rope_split(&st, &mut qd, &mut kd, &mut vd, &qkvd, &cd, &sd, n as u32).unwrap();
    st.synchronize().unwrap();
    for (name, dev, want) in [("q", &qd, &wq), ("k", &kd, &wk), ("v", &vd, &wv)] {
        let mut b = vec![0u16; n * VIT_DIM];
        dev.copy_to_host(&mut b).unwrap();
        let got: Vec<f32> = b.iter().map(|&x| f16_to_f32(x)).collect();
        let (mx, mean, _) = stats(&got, want);
        eprintln!("rope {name} max/rms={mx:.3e} mean/rms={mean:.3e}");
        assert!(mx < 5e-3, "rope {name}");
    }
    let _ = VIT_N_HEADS;
    drop(st);
    d.synchronize().unwrap();
}
