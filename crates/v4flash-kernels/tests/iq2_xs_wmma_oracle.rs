//! Oracle for the f16 WMMA prototype `iq2_xs_pair_matvec_fused_swiglu_wmma`
//! (and its `_cvt` twin) against a CPU f32 dequant-and-dot reference.
//!
//! The WMMA path takes f16 activations, so the reference is computed from
//! the SAME f16-rounded activations in f32. Its only extra roundings are the
//! f16 A-fragment weights (≤2 roundings, ~1e-3 rel per weight) and the f32
//! accumulation order. For scale, the Q8_K kwide kernel is run on a Q8_K
//! quantization of the same activations and compared to the same f32
//! reference: that difference is the 8-bit activation rounding the WMMA
//! path removes.
//!
//! Layout exercised: 3 experts with 23, 32 and 7 members (1-tile, 2-tile
//! full, 1-tile partial), chunk 32, B=40, n_rows=2048, K=4096.
//!
//! Run (iGPU): `cargo test --release -p v4flash-kernels --test iq2_xs_wmma_oracle -- --ignored --nocapture`

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EXPERT, N_EXPERT_USED, N_FF_EXP};
use v4flash_kernels::iq2_xs::{Iq2XsPairMatvec, BLOCK_IQ2_XS_BYTES};
use v4flash_kernels::iq2_xs_tables::IQ2XS_GRID;
use v4flash_kernels::iq2_xxs_tables::{f16_to_f32, KSIGNS_IQ2XS};
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;
use v4flash_kernels::weight_contract::f32_to_f16_bits;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 { (self.next() & 0xff) as u8 }
    fn unit(&mut self) -> f32 { (self.next() as f32) / (u32::MAX as f32) }
}

/// Dequantize one IQ2_XS row (n_blocks super-blocks) to f32.
fn dequant_row_iq2_xs(n_blocks: usize, w: &[u8], out: &mut [f32]) {
    for bi in 0..n_blocks {
        let blk = &w[bi * BLOCK_IQ2_XS_BYTES..(bi + 1) * BLOCK_IQ2_XS_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for ib32 in 0..8 {
            let sc = blk[66 + ib32];
            for l in 0..4 {
                let ls = if l < 2 { 2 * (sc & 0xf) as i32 + 1 } else { 2 * (sc >> 4) as i32 + 1 };
                let qi = 2 + (4 * ib32 + l) * 2;
                let q2 = u16::from_le_bytes([blk[qi], blk[qi + 1]]);
                let grid = IQ2XS_GRID[(q2 & 511) as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[(q2 >> 9) as usize];
                let base = bi * 256 + (4 * ib32 + l) * 8;
                for j in 0..8 {
                    let mag = grid[j] as f32 * ls as f32 * 0.125 * d;
                    out[base + j] = if (signs >> j) & 1 == 1 { -mag } else { mag };
                }
            }
        }
    }
}

/// Q8_K-quantize one f32 row of n_blocks*256 into the llama.cpp block layout
/// (f32 d | i8 qs[256] | i16 bsums[16]) — the same recipe as the on-device
/// q8_k_quantize kernel (amax/127 scale, round-to-nearest).
fn quantize_q8_k(x: &[f32], n_blocks: usize, out: &mut [u8]) {
    for bi in 0..n_blocks {
        let xs = &x[bi * 256..(bi + 1) * 256];
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let o = bi * BLOCK_Q8_K_BYTES;
        out[o..o + 4].copy_from_slice(&d.to_le_bytes());
        let mut bsums = [0i16; 16];
        for k in 0..256 {
            let q = (xs[k] * id).round().clamp(-127.0, 127.0) as i8;
            out[o + 4 + k] = q as u8;
            bsums[k / 16] += q as i16;
        }
        for (j, s) in bsums.iter().enumerate() {
            out[o + 260 + 2 * j..o + 262 + 2 * j].copy_from_slice(&s.to_le_bytes());
        }
    }
}

fn swiglu_ref(g: f32, u: f32, ew: f32, clamp: f32) -> f32 {
    let mut g = g;
    let mut u = u;
    if clamp > 1.0e-6 {
        if g > clamp { g = clamp; }
        if u > clamp { u = clamp; }
        if u < -clamp { u = -clamp; }
    }
    let sig = 1.0 / (1.0 + (-g).exp());
    g * sig * u * ew
}

fn report(name: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    let mut max_diff = 0f32;
    let mut max_ref = 0f32;
    let mut sum_sq_diff = 0f64;
    let mut sum_sq_ref = 0f64;
    let mut n_nan = 0usize;
    for (g, w) in got.iter().zip(want) {
        if !g.is_finite() { n_nan += 1; continue; }
        max_diff = max_diff.max((g - w).abs());
        max_ref = max_ref.max(w.abs());
        sum_sq_diff += ((g - w) as f64).powi(2);
        sum_sq_ref += (*w as f64).powi(2);
    }
    let rel = max_diff / max_ref.max(1e-30);
    let rms_rel = (sum_sq_diff / sum_sq_ref.max(1e-30)).sqrt();
    eprintln!(
        "{name}: n={} max|ref|={max_ref:.4} max_diff={max_diff:.6} rel={rel:.2e} rms_rel={rms_rel:.2e} non_finite={n_nan}",
        got.len()
    );
    if n_nan > 0 { return (f32::INFINITY, f32::INFINITY); }
    (rel, rms_rel as f32)
}

#[test]
#[ignore]
fn iq2_xs_wmma_matches_cpu_f32() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2xs = Iq2XsPairMatvec::for_arch(&arch)?;

    let n_used = N_EXPERT_USED as usize;
    let n_rows = N_FF_EXP as usize;           // 2048
    let nb = BLOCKS_Q8K_GATE_IN as usize;     // 16 super-blocks → K = 4096
    let k_dim = nb * 256;
    let bpe = n_rows * nb * BLOCK_IQ2_XS_BYTES;
    let clamp = 10.0f32;
    let mut rng = Lcg::new(0x2026_0908);

    // Weights: random bytes for the touched experts, realistic d.
    let experts: [(usize, usize); 3] = [(9, 23), (128, 32), (200, 7)];
    let f16_scales: [u16; 4] = [0x2400, 0x2800, 0x2c00, 0x3000]; // 2^-10 .. 2^-7
    let mut gate_h = vec![0u8; (N_EXPERT as usize) * bpe];
    let mut up_h = vec![0u8; (N_EXPERT as usize) * bpe];
    for w in [&mut gate_h, &mut up_h] {
        for &(e, _) in &experts {
            for r in 0..n_rows {
                for bi in 0..nb {
                    let o = e * bpe + (r * nb + bi) * BLOCK_IQ2_XS_BYTES;
                    let d = f16_scales[(rng.next() & 3) as usize].to_le_bytes();
                    w[o..o + 2].copy_from_slice(&d);
                    for i in 2..BLOCK_IQ2_XS_BYTES {
                        w[o + i] = rng.next_byte();
                    }
                }
            }
        }
    }

    // Activations: f32 in [-2, 2]/256 with 1% outliers at 10x, rounded to
    // f16 once; that f16 value is THE input for both the WMMA kernel and the
    // reference. The 1/256 keeps the raw gate/up dots at O(1-10) for these
    // random weights (unscaled they hit ~1700 and the SwiGLU clamp of 10
    // saturates every output, turning the comparison into a sign-flip
    // count of near-zero dots — the trap the first version of this oracle
    // fell into).
    let b: usize = 40;
    let mut x_f32 = vec![0f32; b * k_dim];
    for v in x_f32.iter_mut() {
        let r = rng.unit();
        *v = if rng.next() % 97 == 0 { (r - 0.5) * 40.0 } else { (r - 0.5) * 4.0 };
        *v /= 256.0;
    }
    let x16_h: Vec<u16> = x_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let x_ref: Vec<f32> = x16_h.iter().map(|&h| f16_to_f32(h)).collect();
    let mut xq_h = vec![0u8; b * nb * BLOCK_Q8_K_BYTES];
    for t in 0..b {
        quantize_q8_k(&x_ref[t * k_dim..(t + 1) * k_dim], nb,
                      &mut xq_h[t * nb * BLOCK_Q8_K_BYTES..(t + 1) * nb * BLOCK_Q8_K_BYTES]);
    }
    let ew_h: Vec<f32> = (0..b * n_used).map(|i| 0.05 + 0.01 * (i % 17) as f32).collect();

    // Work items: one expert per chunk of ≤32 members.
    let max_per_expert = b;
    let chunk: u32 = 32;
    let mut gc_h = vec![0i32; N_EXPERT as usize];
    let mut em_h = vec![0i32; (N_EXPERT as usize) * max_per_expert];
    let mut wi_h: Vec<i32> = Vec::new();
    let mut touched = Vec::new();
    let mut np = 0usize;
    for &(e, n) in &experts {
        gc_h[e] = n as i32;
        let mut start = 0i32;
        while (start as usize) < n {
            wi_h.push(((e as i32) << 16) | start);
            start += chunk as i32;
        }
        for i in 0..n {
            let bi = np % b;
            let sl = (np / b) % n_used;
            em_h[e * max_per_expert + i] = ((bi as i32) << 16) | (sl as i32);
            touched.push((e, bi, sl));
            np += 1;
        }
    }

    // Reference: f32 dequant · f16-rounded activations.
    let mut wrow = vec![0f32; k_dim];
    let mut want = Vec::with_capacity(touched.len() * n_rows);
    for &(e, bi, sl) in &touched {
        let x = &x_ref[bi * k_dim..(bi + 1) * k_dim];
        for row in 0..n_rows {
            let go = e * bpe + row * nb * BLOCK_IQ2_XS_BYTES;
            dequant_row_iq2_xs(nb, &gate_h[go..go + nb * BLOCK_IQ2_XS_BYTES], &mut wrow);
            let g: f32 = wrow.iter().zip(x).map(|(w, x)| w * x).sum();
            dequant_row_iq2_xs(nb, &up_h[go..go + nb * BLOCK_IQ2_XS_BYTES], &mut wrow);
            let u: f32 = wrow.iter().zip(x).map(|(w, x)| w * x).sum();
            want.push(swiglu_ref(g, u, ew_h[bi * n_used + sl], clamp));
        }
    }

    // Device buffers.
    let mut gate_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, gate_h.len())?;
    gate_d.copy_from_host(&gate_h)?;
    let mut up_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, up_h.len())?;
    up_d.copy_from_host(&up_h)?;
    let mut x16_d: DeviceBuffer<u16> = DeviceBuffer::new(igpu.id, x16_h.len())?;
    x16_d.copy_from_host(&x16_h)?;
    let mut xq_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq_h.len())?;
    xq_d.copy_from_host(&xq_h)?;
    let mut ew_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, ew_h.len())?;
    ew_d.copy_from_host(&ew_h)?;
    let mut gc_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, gc_h.len())?;
    gc_d.copy_from_host(&gc_h)?;
    let mut em_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, em_h.len())?;
    em_d.copy_from_host(&em_h)?;
    let mut wi_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, wi_h.len())?;
    wi_d.copy_from_host(&wi_h)?;
    let mut mid_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, b * n_used * n_rows)?;
    let mut mid_h = vec![0f32; b * n_used * n_rows];

    let gather = |mid_h: &[f32]| -> Vec<f32> {
        let mut got = Vec::with_capacity(touched.len() * n_rows);
        for &(_, bi, sl) in &touched {
            for row in 0..n_rows {
                got.push(mid_h[(bi * n_used + sl) * n_rows + row]);
            }
        }
        got
    };

    // 1. kwide on Q8_K — the production kernel, for scale.
    mid_d.fill_zero()?;
    iq2xs.launch_fused_swiglu_kwide(&stream, &mut mid_d, &gate_d, &up_d, &xq_d, &ew_d,
        &gc_d, &em_d, &wi_d, wi_h.len() as u32, bpe as u32, bpe as u32,
        n_used as u32, max_per_expert as u32, chunk, clamp, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    mid_d.copy_to_host(&mut mid_h)?;
    let got_kwide = gather(&mid_h);
    let (rel_kwide, rms_kwide) = report("kwide (Q8_K acts) vs f32 ref", &got_kwide, &want);
    if std::env::var_os("ORACLE_DEBUG").is_some() {
        // Per-member rms error + the worst 5 elements, to tell a layout bug
        // (structured) from quantization noise (broad, uniform).
        for (ti, &(e, bi, sl)) in touched.iter().enumerate() {
            let g = &got_kwide[ti * n_rows..(ti + 1) * n_rows];
            let w = &want[ti * n_rows..(ti + 1) * n_rows];
            let (mut sd, mut sr) = (0f64, 0f64);
            for (a, b) in g.iter().zip(w) { sd += ((a - b) as f64).powi(2); sr += (*b as f64).powi(2); }
            eprintln!("  member {ti:2} e={e:3} b={bi:2} slot={sl}: rms_rel={:.3e}", (sd / sr.max(1e-30)).sqrt());
        }
        // Is it the kernel or my Q8_K quantization? CPU Q8_K dot on the SAME
        // xq bytes for member 0, vs both the kernel and the f32 reference.
        {
            use v4flash_kernels::iq2_xs_tables::cpu_dot_iq2_xs_q8_k;
            let (e, bi, sl) = touched[0];
            let xq_s = &xq_h[bi * nb * BLOCK_Q8_K_BYTES..(bi + 1) * nb * BLOCK_Q8_K_BYTES];
            let mut cpu_q8 = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let go = e * bpe + row * nb * BLOCK_IQ2_XS_BYTES;
                let g = cpu_dot_iq2_xs_q8_k(nb, &gate_h[go..go + nb * BLOCK_IQ2_XS_BYTES], xq_s);
                let u = cpu_dot_iq2_xs_q8_k(nb, &up_h[go..go + nb * BLOCK_IQ2_XS_BYTES], xq_s);
                cpu_q8.push(swiglu_ref(g, u, ew_h[bi * n_used + sl], clamp));
            }
            // Elementwise Q8_K reconstruction error of token bi.
            let mut rec = Vec::with_capacity(k_dim);
            for blk in 0..nb {
                let o = blk * BLOCK_Q8_K_BYTES;
                let d = f32::from_le_bytes([xq_s[o], xq_s[o + 1], xq_s[o + 2], xq_s[o + 3]]);
                for k in 0..256 { rec.push(d * (xq_s[o + 4 + k] as i8) as f32); }
            }
            let _ = report("  token: Q8_K reconstruction vs f16 x", &rec, &x_ref[bi * k_dim..(bi + 1) * k_dim]);
            // My dequant · reconstructed Q8_K x  vs  the repo's CPU Q8_K dot
            // (gate only, raw dot, no swiglu). Must agree to f32 roundoff if
            // the two references implement the same format math.
            let mut mine = Vec::with_capacity(n_rows);
            let mut theirs = Vec::with_capacity(n_rows);
            let mut wr = vec![0f32; k_dim];
            for row in 0..n_rows {
                let go = e * bpe + row * nb * BLOCK_IQ2_XS_BYTES;
                dequant_row_iq2_xs(nb, &gate_h[go..go + nb * BLOCK_IQ2_XS_BYTES], &mut wr);
                mine.push(wr.iter().zip(&rec).map(|(w, x)| w * x).sum::<f32>());
                theirs.push(cpu_dot_iq2_xs_q8_k(nb, &gate_h[go..go + nb * BLOCK_IQ2_XS_BYTES], xq_s));
            }
            let _ = report("  gate dot: my dequant·rec vs cpu_dot_iq2_xs_q8_k", &mine, &theirs);
            let mut f32dot = Vec::with_capacity(n_rows);
            let x = &x_ref[bi * k_dim..(bi + 1) * k_dim];
            for row in 0..n_rows {
                let go = e * bpe + row * nb * BLOCK_IQ2_XS_BYTES;
                dequant_row_iq2_xs(nb, &gate_h[go..go + nb * BLOCK_IQ2_XS_BYTES], &mut wr);
                f32dot.push(wr.iter().zip(x).map(|(w, x)| w * x).sum::<f32>());
            }
            let _ = report("  gate dot: my dequant·rec vs my dequant·f16x", &mine, &f32dot);
            let _ = report("  member0: kernel kwide vs CPU Q8_K dot", &got_kwide[..n_rows], &cpu_q8);
            let _ = report("  member0: CPU Q8_K dot vs f32 ref", &cpu_q8, &want[..n_rows]);
        }
        let mut idx: Vec<usize> = (0..want.len()).collect();
        idx.sort_by(|&a, &b| (got_kwide[b] - want[b]).abs().partial_cmp(&(got_kwide[a] - want[a]).abs()).unwrap());
        for &i in idx.iter().take(5) {
            eprintln!("  worst: touched={} row={} got={:.4} want={:.4}", i / n_rows, i % n_rows, got_kwide[i], want[i]);
        }
    }

    // 2. WMMA (the kernel: magic-number dequant).
    mid_d.fill_zero()?;
    iq2xs.launch_fused_swiglu_wmma(&stream, &mut mid_d, &gate_d, &up_d, &x16_d, &ew_d,
        &gc_d, &em_d, &wi_d, wi_h.len() as u32, bpe as u32, bpe as u32,
        n_used as u32, max_per_expert as u32, chunk, clamp, n_rows as u32, nb as u32, 0)?;
    stream.synchronize()?;
    mid_d.copy_to_host(&mut mid_h)?;
    let (rel_wmma, rms_wmma) = report("wmma (magic, the kernel) vs f32 ref", &gather(&mid_h), &want);

    // 3. WMMA (cvt dequant).
    mid_d.fill_zero()?;
    iq2xs.launch_fused_swiglu_wmma(&stream, &mut mid_d, &gate_d, &up_d, &x16_d, &ew_d,
        &gc_d, &em_d, &wi_d, wi_h.len() as u32, bpe as u32, bpe as u32,
        n_used as u32, max_per_expert as u32, chunk, clamp, n_rows as u32, nb as u32, 1)?;
    stream.synchronize()?;
    mid_d.copy_to_host(&mut mid_h)?;
    let (rel_cvt, rms_cvt) = report("wmma (cvt twin) vs f32 ref", &gather(&mid_h), &want);

    // 4. WMMA (f16-LUT twin).
    mid_d.fill_zero()?;
    iq2xs.launch_fused_swiglu_wmma(&stream, &mut mid_d, &gate_d, &up_d, &x16_d, &ew_d,
        &gc_d, &em_d, &wi_d, wi_h.len() as u32, bpe as u32, bpe as u32,
        n_used as u32, max_per_expert as u32, chunk, clamp, n_rows as u32, nb as u32, 6)?;
    stream.synchronize()?;
    mid_d.copy_to_host(&mut mid_h)?;
    let (rel_magic, rms_magic) = report("wmma (f16 LUT twin) vs f32 ref", &gather(&mid_h), &want);

    // Untouched (b, slot) rows must stay zero: padded tile columns must not
    // be written.
    let touched_set: std::collections::HashSet<(usize, usize)> =
        touched.iter().map(|&(_, bi, sl)| (bi, sl)).collect();
    let mut leaks = 0usize;
    for bi in 0..b {
        for sl in 0..n_used {
            if touched_set.contains(&(bi, sl)) { continue; }
            let o = (bi * n_used + sl) * n_rows;
            if mid_h[o..o + n_rows].iter().any(|&v| v != 0.0) { leaks += 1; }
        }
    }
    eprintln!("padded-column leaks: {leaks}");

    // Tolerances: the WMMA path's own error budget is the f16 weight
    // rounding (≤2 roundings, ~1e-3 rel per weight, averaging down over
    // K=4096): max-rel < 1e-2 and rms-rel < 2e-3. The Q8_K path carries the
    // 8-bit activation rounding (~2% rms on these outlier-laden
    // activations) and is checked loosely, as the scale reference.
    let mut bad = Vec::new();
    if !(rel_wmma < 1e-2 && rms_wmma < 2e-3) { bad.push(format!("wmma rel={rel_wmma:.2e} rms={rms_wmma:.2e}")); }
    if !(rel_cvt < 1e-2 && rms_cvt < 2e-3) { bad.push(format!("wmma_cvt rel={rel_cvt:.2e} rms={rms_cvt:.2e}")); }
    if !(rel_magic < 1e-2 && rms_magic < 2e-3) { bad.push(format!("wmma_magic rel={rel_magic:.2e} rms={rms_magic:.2e}")); }
    if !(rms_kwide < 1e-1) { bad.push(format!("kwide rel={rel_kwide:.2e} rms={rms_kwide:.2e}")); }
    if leaks > 0 { bad.push(format!("{leaks} padded columns written")); }
    if !bad.is_empty() {
        return Err(eyre!("iq2_xs wmma oracle failed: {}", bad.join(", ")));
    }
    Ok(())
}
