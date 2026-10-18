//! `V41_ATTN_O_FP8`: the device side of the 1017 reference's attention-output FP8 rounding and the
//! FP8 x FP8 `wo_a`, against the CPU emulation (`v4flash_kernels::act_quant_fp8`, itself checked
//! against torch golden vectors of the reference kernel's arithmetic in its unit tests).
//!
//! Synthetic data at the real shapes (no model load; < 200 MB of device memory). GPU tests are
//! `#[ignore]`: run them only in an owner-approved gate window (production uses both GPUs).
//!
//!   act_quant_kernels_match_the_cpu_bit_for_bit (dGPU AND iGPU): `act_quant_fp8_wave` codes +
//!     scales, `act_quant_fp8_inplace_wave` values (signed zeros included) and the drafter's
//!     `q8_0_quantize_o8_wave` (rounding, then Q8_0 of the rounded values) BIT FOR BIT vs
//!     `act_quant_row` / `act_quant_inplace` / `q8_0_block_device`, on random rows over 18 decades
//!     plus every edge block (all zero, all -0, below the 1e-4 floor, the floor scale's subnormal
//!     grid and its ties, amax at / one ulp around 448 * 2^k, every e4m3 midpoint, mixed magnitudes,
//!     near f32 max), at b = 1, 5, 16, 512 rows of 32768.
//!   rope_inv_quant_fp8_matches_the_cpu: `RopeTail::launch_inverse_quant_fp8` leaves the SAME heads as
//!     `launch_inverse_quant_q8` (only the quantize differs) and its codes / scales equal
//!     `act_quant_row` of those heads bit for bit, b = 1..16, 64, 128, 512, plain + YaRN rope.
//!   fp8_wo_a_matches_the_exact_product: `Fp8WoA::matvec` (b = 1..9, 16, 17, 40) and `Fp8WoA::gemm`
//!     (b = 1, 16, 17, 128, 129, 300, 512) against `fp8_grouped_matvec_ref` (exact products, f64 sum)
//!     within the order-independent f32 summation bound (n - 1) u / (1 - (n - 1) u) * sum |term|,
//!     n = 4096 terms per output, u = 2^-24 (every product is exact in f32 and the block scales are
//!     powers of two, so only the summation rounds; the reference's own bf16 einsum rounds its f32 sum
//!     to bf16 on top). Also reports the observed max error / sum|term| (expected ~1e-7). Random
//!     e4m3 weights over every finite code (subnormals included) with e8m0 scales 2^-14..2^-5; with
//!     `V41_HF_DIR` set, layer `FP8_WO_A_LAYER` (default 0) of the real checkpoint's wo_a too.
//!   fp8_wo_a_subnormals_and_signed_zeros_are_exact: one nonzero 32-block per output whose exact sum
//!     is representable: the GEMV / GEMM must return it EXACTLY (catches an FP8 denormal flush in
//!     v_dot4_f32_fp8_fp8 / v_wmma_f32_16x16x16_fp8_fp8, which the bound above could hide).
//!   fp8_wo_a_refuses_a_q8_0_weight: the launchers assert the FP8 buffer size (a Q8_0 wo_a under the
//!     knob = the knob changed after load), and the iGPU module refuses the FP8 dot.
//!
//! Window command (hub DOWN or not -- these only need ~200 MB; box 2 not needed):
//!   HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release \
//!     --features v41 -p v4flash-kernels --test fp8_wo_a -- --ignored --nocapture --test-threads=1
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::act_quant_fp8::{
    act_quant_inplace, act_quant_row, e4m3fn_decode, f32_sum_bound, fp8_grouped_matvec_ref, fp8_weight_bytes,
    q8_0_block_device,
};
use v4flash_kernels::config::{GROUP_DIM, N_GROUPS, OUT_LOW, Q_FLAT, RANK};
use v4flash_kernels::fp8_wo_a::Fp8WoA;
use v4flash_kernels::{RopeParams, RopeTail};

const QF: usize = Q_FLAT as usize;
const MB: usize = 1 << 20;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 29) as f32 / (1u64 << 24) as f32
    }
    fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len())?;
    d.copy_from_host(host)?;
    Ok(d)
}
fn zeros<T: Copy + Default>(id: i32, n: usize) -> eyre::Result<DeviceBuffer<T>> {
    upload(id, &vec![T::default(); n])
}
fn download<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}

fn device(prefix: &str) -> eyre::Result<Option<(Device, String)>> {
    for d in Device::all()? {
        let arch = d.properties()?.gcn_arch_name;
        if arch.starts_with(prefix) {
            d.set_current()?;
            let (free, total) = Device::mem_info_current()?;
            eprintln!("{arch}: {} MB free of {} MB", free >> 20, total >> 20);
            if free < 400 * MB {
                return Err(eyre!("only {} MB free on {arch}: not starting beside production", free >> 20));
            }
            return Ok(Some((d, arch)));
        }
    }
    Ok(None)
}
fn dgpu() -> eyre::Result<(Device, String)> {
    device("gfx1201")?.ok_or_else(|| eyre!("no gfx1201 visible (HIP_VISIBLE_DEVICES must include the dGPU)"))
}

/// `rows` rows of 32768: random blocks over 18 decades, then every edge block, cycled.
fn test_rows(rng: &mut Lcg, rows: usize) -> Vec<f32> {
    let mut edge: Vec<[f32; 32]> = Vec::new();
    edge.push([0.0; 32]);
    edge.push([-0.0; 32]);
    edge.push([5e-5; 32]);
    let q = 2f32.powi(-31); // the floor scale's subnormal quantum
    edge.push(std::array::from_fn(|i| i as f32 * q));
    edge.push(std::array::from_fn(|i| (i as f32 + 0.5) * q * if i % 2 == 0 { 1.0 } else { -1.0 }));
    for k in [-20i32, -7, -1, 0, 1, 6, 30, 100] {
        let b = 448.0f32 * 2f32.powi(k);
        for d in [-1i64, 0, 1] {
            let mut blk: [f32; 32] = std::array::from_fn(|_| rng.sym(b * 0.3));
            blk[0] = f32::from_bits((b.to_bits() as i64 + d) as u32);
            blk[1] = -2f32.powi(k);
            edge.push(blk);
        }
    }
    // every e4m3 midpoint at scale 1 (pinned by a 448 element) and at 2^-12 / 2^20
    let vals: Vec<f32> = (0u8..0x7F).map(e4m3fn_decode).collect();
    let mids: Vec<f32> = vals.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
    for k in [0i32, -12, 20] {
        for ch in mids.chunks(31) {
            let mut blk = [0.0f32; 32];
            for (i, &m) in ch.iter().enumerate() {
                blk[i] = m * 2f32.powi(k) * if i % 3 == 0 { -1.0 } else { 1.0 };
            }
            blk[31] = 448.0 * 2f32.powi(k);
            edge.push(blk);
        }
    }
    for _ in 0..8 {
        edge.push(std::array::from_fn(|_| rng.sym(1.0) * 10f32.powi(rng.below(12) as i32 - 9)));
    }
    let mut big: [f32; 32] = std::array::from_fn(|_| rng.sym(1e37));
    big[0] = 3.0e38;
    edge.push(big);
    let mut x = vec![0f32; rows * QF];
    for (bi, blk) in x.chunks_exact_mut(32).enumerate() {
        if bi % 7 == 3 {
            blk.copy_from_slice(&edge[(bi / 7) % edge.len()]);
        } else {
            let mag = 10f32.powi((bi % 19) as i32 - 13);
            for v in blk.iter_mut() {
                *v = rng.sym(mag);
            }
        }
    }
    x
}

#[test]
fn knob_defaults_off() {
    // the knob is static and default OFF: with it unset the engine loads the Q8_0 wo_a and runs the
    // Q8_0 path (V41_ATTN_O_FP8 must not leak in from the environment of a test run)
    if std::env::var("V41_ATTN_O_FP8").is_err() {
        assert!(!v4flash_kernels::fp8_wo_a::attn_o_fp8(), "V41_ATTN_O_FP8 must default to off");
    }
}

#[test]
#[ignore]
fn act_quant_kernels_match_the_cpu_bit_for_bit() -> eyre::Result<()> {
    let _ = install_panic_handler();
    let mut bad = 0usize;
    let mut ran = 0;
    for prefix in ["gfx1201", "gfx1151"] {
        let Some((dev, arch)) = device(prefix)? else {
            eprintln!("{prefix}: not visible, skipped");
            continue;
        };
        ran += 1;
        let id = dev.id;
        let s = Stream::new(id)?;
        let k = Fp8WoA::for_arch(&arch)?;
        let mut rng = Lcg(0x1017_0f8);
        for rows in [1usize, 5, 16, 512] {
            let x = test_rows(&mut rng, rows);
            let n = x.len();
            let dx = upload(id, &x)?;
            let mut xq = upload(id, &vec![0x55i8; n])?;
            let mut xs = upload(id, &vec![f32::NAN; n / 32])?;
            // 1. codes + scales
            k.act_quant(&s, &mut xq, &mut xs, &dx, Q_FLAT, rows as u32)?;
            s.synchronize()?;
            let (mut codes, mut scales) = (vec![0u8; n], vec![0f32; n / 32]);
            act_quant_row(&x, &mut codes, &mut scales);
            let gq = download(&xq)?;
            let gs = download(&xs)?;
            let dq = codes.iter().zip(&gq).filter(|(a, b)| **a != **b as u8).count();
            let ds = scales.iter().zip(&gs).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            // 2. in place
            let mut dxi = upload(id, &x)?;
            k.act_quant_inplace(&s, &mut dxi, n as u32)?;
            s.synchronize()?;
            let mut y = x.clone();
            act_quant_inplace(&mut y);
            let gy = download(&dxi)?;
            let dy = y.iter().zip(&gy).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            // 3. the drafter's round-then-Q8_0
            k.quantize_q8_o8(&s, &mut xq, &mut xs, &dx, Q_FLAT, rows as u32)?;
            s.synchronize()?;
            let gq8 = download(&xq)?;
            let gs8 = download(&xs)?;
            let mut d8 = 0;
            for (bi, blk) in y.chunks_exact(32).enumerate() {
                let (q, d) = q8_0_block_device(blk);
                d8 += usize::from(d.to_bits() != gs8[bi].to_bits());
                d8 += q.iter().zip(&gq8[bi * 32..bi * 32 + 32]).filter(|(a, b)| a != b).count();
            }
            eprintln!("{arch} rows={rows}: codes {dq} scales {ds} inplace {dy} q8_o8 {d8} mismatches of {n}");
            if dq + ds + dy + d8 != 0 {
                bad += 1;
                for (i, (a, b)) in y.iter().zip(&gy).enumerate().filter(|(_, (a, b))| a.to_bits() != b.to_bits()).take(6) {
                    eprintln!("  [{i}] x {:e}: cpu {:e} gpu {:e}", x[i], a, b);
                }
            }
        }
    }
    if ran == 0 {
        return Err(eyre!("no GPU visible"));
    }
    if bad != 0 {
        return Err(eyre!("{bad} cases not bit-identical"));
    }
    Ok(())
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
fn rope_inv_quant_fp8_matches_the_cpu() -> eyre::Result<()> {
    let _ = install_panic_handler();
    let (dev, arch) = dgpu()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let rope = RopeTail::for_arch(&arch)?;
    let mut rng = Lcg(0x0f8_2026);
    let mut bad = 0usize;
    let mut sizes: Vec<usize> = (1..=16).collect();
    sizes.extend([64, 128, 512]);
    for (pname, params) in rope_params() {
        for &b in &sizes {
            let x = test_rows(&mut rng, b);
            let n = x.len();
            let pos: Vec<i32> = (0..b).map(|i| (rng.below(900_000) + i as u64) as i32).collect();
            let dpos = upload(id, &pos)?;
            let mut h_q8 = upload(id, &x)?;
            let mut h_f8 = upload(id, &x)?;
            let mut xq = zeros::<i8>(id, n)?;
            let mut xs = zeros::<f32>(id, n / 32)?;
            let mut xq8 = upload(id, &vec![0x55i8; n])?;
            let mut xs8 = upload(id, &vec![f32::NAN; n / 32])?;
            rope.launch_inverse_quant_q8(&s, &mut h_q8, &mut xq, &mut xs, &dpos, 64, 512, 64, b as u32, &params)?;
            rope.launch_inverse_quant_fp8(&s, &mut h_f8, &mut xq8, &mut xs8, &dpos, 64, 512, 64, b as u32, &params)?;
            s.synchronize()?;
            let hq = download(&h_q8)?;
            let hf = download(&h_f8)?;
            let dh = hq.iter().zip(&hf).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let (mut codes, mut scales) = (vec![0u8; n], vec![0f32; n / 32]);
            act_quant_row(&hf, &mut codes, &mut scales);
            let gq = download(&xq8)?;
            let gs = download(&xs8)?;
            let dq = codes.iter().zip(&gq).filter(|(a, b)| **a != **b as u8).count();
            let ds = scales.iter().zip(&gs).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            if dh + dq + ds != 0 {
                bad += 1;
                eprintln!("MISMATCH rope={pname} b={b}: heads vs q8 kernel {dh}, codes {dq}, scales {ds}");
            }
        }
        eprintln!("rope {pname}: {} sizes checked", sizes.len());
    }
    if bad != 0 {
        return Err(eyre!("{bad} cases failed"));
    }
    Ok(())
}

/// A random FP8 wo_a: every finite e4m3 code (subnormals and -0 included), e8m0 scales 2^-14..2^-5.
fn random_weight(rng: &mut Lcg) -> Vec<u8> {
    let (out, k) = (OUT_LOW as usize, GROUP_DIM as usize);
    let mut w = vec![0u8; fp8_weight_bytes(out, k)];
    for c in w[..out * k].iter_mut() {
        let v = rng.below(256) as u8;
        *c = if v & 0x7F == 0x7F { v & 0xF0 } else { v };
    }
    for s in w[out * k..].iter_mut() {
        *s = 113 + rng.below(10) as u8;
    }
    w
}

/// Activation codes + scales for `b` rows: act_quant of random heads with per-block magnitudes.
fn random_acts(rng: &mut Lcg, b: usize) -> (Vec<u8>, Vec<f32>) {
    let x = test_rows(rng, b);
    let (mut codes, mut scales) = (vec![0u8; x.len()], vec![0f32; x.len() / 32]);
    act_quant_row(&x, &mut codes, &mut scales);
    (codes, scales)
}

#[derive(Default)]
struct Err {
    checked: usize,
    over: usize,
    worst_ratio: f64,
}

/// One output of the exact product (`fp8_grouped_matvec_ref`'s arithmetic for a single (row of o,
/// output)): (sum, sum |term|) in f64.
fn exact_out(w: &[u8], dec: &[f64; 256], codes: &[u8], scales: &[f32], o: usize) -> (f64, f64) {
    let (kd, out_dim) = (GROUP_DIM as usize, OUT_LOW as usize);
    let kbn = kd / 32;
    let g = o / RANK as usize;
    let (mut s, mut m) = (0f64, 0f64);
    for kb in 0..kbn {
        let sw = v4flash_kernels::act_quant_fp8::e8m0_to_f32(w[out_dim * kd + (o / 32) * kbn + kb]) as f64;
        let sa = scales[g * kbn + kb] as f64;
        for kk in kb * 32..kb * 32 + 32 {
            let t = dec[w[o * kd + kk] as usize] * dec[codes[g * kd + kk] as usize] * sw * sa;
            s += t;
            m += t.abs();
        }
    }
    (s, m)
}

/// `got[b, r]` against the exact product for the outputs in `rows_of` (all when None).
fn check_against_ref(w: &[u8], codes: &[u8], scales: &[f32], b: usize, got: &[f32], rows_of: Option<&[usize]>, e: &mut Err) {
    let out_dim = OUT_LOW as usize;
    let dec: [f64; 256] = std::array::from_fn(|c| if c as u8 & 0x7F == 0x7F { f64::NAN } else { e4m3fn_decode(c as u8) as f64 });
    let all: Vec<usize> = (0..out_dim).collect();
    for bi in 0..b {
        let cb = &codes[bi * QF..(bi + 1) * QF];
        let sb = &scales[bi * QF / 32..(bi + 1) * QF / 32];
        for &o in rows_of.unwrap_or(&all) {
            let (r, mag) = exact_out(w, &dec, cb, sb, o);
            let g = got[bi * out_dim + o] as f64;
            let bound = f32_sum_bound(GROUP_DIM as usize, mag) + 1e-38;
            let err = (g - r).abs();
            e.checked += 1;
            if !(err <= bound) {
                e.over += 1;
                if e.over <= 5 {
                    eprintln!("  out [{bi}, {o}]: gpu {g:e} exact {r:e} |err| {err:e} > bound {bound:e}");
                }
            }
            if mag > 0.0 {
                e.worst_ratio = e.worst_ratio.max(err / mag);
            }
        }
    }
}

#[test]
#[ignore]
fn fp8_wo_a_matches_the_exact_product() -> eyre::Result<()> {
    let _ = install_panic_handler();
    let (dev, arch) = dgpu()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let k = Fp8WoA::for_arch(&arch)?;
    let mut rng = Lcg(0x0a17_e5);
    let mut weights: Vec<(String, Vec<u8>)> = vec![("random".into(), random_weight(&mut rng))];
    if let Ok(dir) = std::env::var("V41_HF_DIR") {
        let layer: usize = std::env::var("FP8_WO_A_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let hf = v4flash_core::hf_v41::V41HfWeights::open(&dir, Some(0))?;
        let (bytes, out, inn) = hf.read_fp8_block32_raw(layer, "attn.wo_a")?;
        if (out, inn) != (OUT_LOW as usize, GROUP_DIM as usize) {
            return Err(eyre!("{dir} layer {layer} wo_a is [{out}, {inn}]"));
        }
        weights.push((format!("hf layer {layer}"), bytes));
    }
    let out_dim = OUT_LOW as usize;
    let mut bad = 0usize;
    for (wname, w) in &weights {
        let dw = upload(id, w)?;
        let cases: Vec<(&str, usize)> = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 40].iter().map(|&b| ("gemv", b))
            .chain([1usize, 16, 17, 128, 129, 300, 512].iter().map(|&b| ("gemm", b))).collect();
        for (kind, b) in cases {
            let (codes, scales) = random_acts(&mut rng, b);
            let xq = upload(id, &codes.iter().map(|&c| c as i8).collect::<Vec<_>>())?;
            let xs = upload(id, &scales)?;
            let mut out = upload(id, &vec![f32::NAN; b * out_dim])?;
            match kind {
                "gemv" => k.matvec(&s, &mut out, &dw, &xq, &xs, GROUP_DIM, RANK, N_GROUPS, b as u32)?,
                _ => k.gemm(&s, &mut out, &dw, &xq, &xs, GROUP_DIM, RANK, N_GROUPS, b as u32)?,
            }
            s.synchronize()?;
            let got = download(&out)?;
            let mut e = Err::default();
            if b <= 17 {
                check_against_ref(w, &codes, &scales, b, &got, None, &mut e);
            } else {
                // 512 sampled outputs of every row, every output of 3 rows (the f64 reference is
                // 33.5 M products per full row)
                let sample: Vec<usize> = (0..512).map(|_| rng.below(out_dim as u64) as usize).collect();
                check_against_ref(w, &codes, &scales, b, &got, Some(&sample), &mut e);
                for bi in [0, b / 2, b - 1] {
                    check_against_ref(w, &codes[bi * QF..(bi + 1) * QF], &scales[bi * QF / 32..(bi + 1) * QF / 32], 1,
                                      &got[bi * out_dim..(bi + 1) * out_dim], None, &mut e);
                }
            }
            let nan = got.iter().filter(|v| !v.is_finite()).count();
            eprintln!("{wname} {kind} b={b}: {} outputs checked, {} over the bound, {nan} non-finite, max |err| / sum|term| = {:.3e}",
                      e.checked, e.over, e.worst_ratio);
            if e.over + nan != 0 {
                bad += 1;
            }
        }
    }
    if bad != 0 {
        return Err(eyre!("{bad} cases outside the f32 summation bound"));
    }
    Ok(())
}

#[test]
#[ignore]
fn fp8_wo_a_subnormals_and_signed_zeros_are_exact() -> eyre::Result<()> {
    let _ = install_panic_handler();
    let (dev, arch) = dgpu()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let k = Fp8WoA::for_arch(&arch)?;
    let (out_dim, kd) = (OUT_LOW as usize, GROUP_DIM as usize);
    // weights: zero (+0 / -0 codes alternating) except block kb(r) = r % 128 of each row, which
    // holds subnormal and small-normal codes; activations: subnormal codes in every block, scale 1.
    let mut w = vec![0u8; fp8_weight_bytes(out_dim, kd)];
    for r in 0..out_dim {
        for c in 0..kd {
            w[r * kd + c] = if (r + c) % 2 == 0 { 0x00 } else { 0x80 };
        }
        let kb = r % 128;
        for j in 0..32 {
            w[r * kd + kb * 32 + j] = [0x01u8, 0x07, 0x81, 0x08, 0x0F, 0x03, 0x85, 0x00][(r + j) % 8];
        }
    }
    for sc in w[out_dim * kd..].iter_mut() {
        *sc = 127; // 2^0
    }
    let b = 3usize;
    let mut codes = vec![0u8; b * QF];
    for (i, c) in codes.iter_mut().enumerate() {
        *c = [0x01u8, 0x02, 0x87, 0x05, 0x80, 0x06, 0x81, 0x04][i % 8];
    }
    let scales = vec![1.0f32; b * QF / 32];
    let (r, _) = fp8_grouped_matvec_ref(&w, &codes, &scales, kd, RANK as usize, N_GROUPS as usize, b);
    let dw = upload(id, &w)?;
    let xq = upload(id, &codes.iter().map(|&c| c as i8).collect::<Vec<_>>())?;
    let xs = upload(id, &scales)?;
    let mut bad = 0usize;
    for kind in ["gemv", "gemm"] {
        let mut out = upload(id, &vec![f32::NAN; b * out_dim])?;
        match kind {
            "gemv" => k.matvec(&s, &mut out, &dw, &xq, &xs, GROUP_DIM, RANK, N_GROUPS, b as u32)?,
            _ => k.gemm(&s, &mut out, &dw, &xq, &xs, GROUP_DIM, RANK, N_GROUPS, b as u32)?,
        }
        s.synchronize()?;
        let got = download(&out)?;
        // every term is a multiple of 2^-18 below 2^-6: the f32 sum is exact in any order
        let d = got.iter().zip(&r).filter(|(g, e)| (**g as f64) != **e).count();
        let zero_ref = r.iter().filter(|v| **v == 0.0).count();
        eprintln!("{kind}: {d} of {} outputs differ from the exact sum ({zero_ref} exact zeros)", got.len());
        for (i, (g, e)) in got.iter().zip(&r).enumerate().filter(|(_, (g, e))| (**g as f64) != **e).take(5) {
            eprintln!("  [{i}] gpu {g:e} exact {e:e}");
        }
        bad += d;
    }
    if bad != 0 {
        return Err(eyre!("FP8 subnormal products not exact ({bad} outputs): denormal flush in the FP8 dot / WMMA?"));
    }
    Ok(())
}

#[test]
#[ignore]
fn fp8_wo_a_refuses_a_q8_0_weight() -> eyre::Result<()> {
    let _ = install_panic_handler();
    let (dev, arch) = dgpu()?;
    let id = dev.id;
    let s = Stream::new(id)?;
    let k = Fp8WoA::for_arch(&arch)?;
    let q8_bytes = OUT_LOW as usize * (GROUP_DIM as usize / 32) * 34;
    let w = zeros::<u8>(id, q8_bytes)?;
    let xq = zeros::<i8>(id, QF)?;
    let xs = zeros::<f32>(id, QF / 32)?;
    let mut out = zeros::<f32>(id, OUT_LOW as usize)?;
    if k.matvec(&s, &mut out, &w, &xq, &xs, GROUP_DIM, RANK, N_GROUPS, 1).is_ok()
        || k.gemm(&s, &mut out, &w, &xq, &xs, GROUP_DIM, RANK, N_GROUPS, 1).is_ok()
    {
        return Err(eyre!("a Q8_0-sized wo_a was accepted by the FP8 kernels"));
    }
    if let Some((idev, iarch)) = device("gfx1151")? {
        let ik = Fp8WoA::for_arch(&iarch)?;
        let is = Stream::new(idev.id)?;
        let iw = zeros::<u8>(idev.id, fp8_weight_bytes(OUT_LOW as usize, GROUP_DIM as usize))?;
        let ixq = zeros::<i8>(idev.id, QF)?;
        let ixs = zeros::<f32>(idev.id, QF / 32)?;
        let mut iout = zeros::<f32>(idev.id, OUT_LOW as usize)?;
        if ik.matvec(&is, &mut iout, &iw, &ixq, &ixs, GROUP_DIM, RANK, N_GROUPS, 1).is_ok() {
            return Err(eyre!("the iGPU module accepted an FP8 GEMV (gfx1151 has no FP8 dot)"));
        }
    }
    Ok(())
}
