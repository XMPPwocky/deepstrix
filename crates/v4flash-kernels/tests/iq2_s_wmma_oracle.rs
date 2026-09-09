//! Oracle for `iq2_s_pair_matvec_fused_swiglu_wmma` (+ the `_h` f16-out
//! production twin) against a CPU f32 dequant-and-dot reference — the IQ2_S
//! port of iq2_xs_wmma_oracle (see that header for the error budget and
//! the activation-scaling trap).
//!
//! Layout exercised: 3 experts with 23, 32 and 7 members (1-tile, 2-tile
//! full, 1-tile partial), chunk 32, B=40, n_rows=2048, K=4096. Only the 3
//! experts are allocated (~16 MB) so this runs beside a live server.
//!
//! Run (iGPU): `cargo test --release -p v4flash-kernels --test iq2_s_wmma_oracle -- --ignored --nocapture`
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EXPERT_USED, N_FF_EXP};
use v4flash_kernels::iq2_s::{Iq2SPairMatvec, BLOCK_IQ2_S_BYTES};
use v4flash_kernels::iq2_s_tables::IQ2S_GRID;
use v4flash_kernels::iq2_xxs_tables::f16_to_f32;
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

/// Dequantize one IQ2_S row (n_blocks super-blocks) to f32.
fn dequant_row_iq2_s(n_blocks: usize, w: &[u8], out: &mut [f32]) {
    for bi in 0..n_blocks {
        let blk = &w[bi * BLOCK_IQ2_S_BYTES..(bi + 1) * BLOCK_IQ2_S_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for ib32 in 0..8 {
            let sc = blk[74 + ib32];
            let qh = blk[66 + ib32] as u32;
            for l in 0..4usize {
                let ls = if l < 2 { 2 * (sc & 0xf) as i32 + 1 } else { 2 * (sc >> 4) as i32 + 1 };
                let gidx = (blk[2 + 4 * ib32 + l] as usize) | (((qh << (8 - 2 * l)) & 0x300) as usize);
                let grid = IQ2S_GRID[gidx].to_le_bytes();
                let signs = blk[34 + 4 * ib32 + l];
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
fn iq2_s_wmma_matches_cpu_f32() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2s = Iq2SPairMatvec::for_arch(&arch)?;

    let n_used = N_EXPERT_USED as usize;
    let n_rows = N_FF_EXP as usize;           // 2048
    let nb = BLOCKS_Q8K_GATE_IN as usize;     // 16 super-blocks → K = 4096
    let k_dim = nb * 256;
    let bpe = n_rows * nb * BLOCK_IQ2_S_BYTES;
    let clamp = 10.0f32;
    let mut rng = Lcg::new(0x2026_0908);

    // Weights: random bytes for the touched experts, realistic d.
    let experts: [(usize, usize); 3] = [(0, 23), (1, 32), (2, 7)];
    let n_e: usize = 3;
    let f16_scales: [u16; 4] = [0x2400, 0x2800, 0x2c00, 0x3000]; // 2^-10 .. 2^-7
    let mut gate_h = vec![0u8; n_e * bpe];
    let mut up_h = vec![0u8; n_e * bpe];
    for w in [&mut gate_h, &mut up_h] {
        for &(e, _) in &experts {
            for r in 0..n_rows {
                for bi in 0..nb {
                    let o = e * bpe + (r * nb + bi) * BLOCK_IQ2_S_BYTES;
                    let d = f16_scales[(rng.next() & 3) as usize].to_le_bytes();
                    w[o..o + 2].copy_from_slice(&d);
                    for i in 2..BLOCK_IQ2_S_BYTES {
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
    let mut gc_h = vec![0i32; n_e];
    let mut em_h = vec![0i32; n_e * max_per_expert];
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
            let go = e * bpe + row * nb * BLOCK_IQ2_S_BYTES;
            dequant_row_iq2_s(nb, &gate_h[go..go + nb * BLOCK_IQ2_S_BYTES], &mut wrow);
            let g: f32 = wrow.iter().zip(x).map(|(w, x)| w * x).sum();
            dequant_row_iq2_s(nb, &up_h[go..go + nb * BLOCK_IQ2_S_BYTES], &mut wrow);
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
    iq2s.launch_fused_swiglu_kwide(&stream, &mut mid_d, &gate_d, &up_d, &xq_d, &ew_d,
        &gc_d, &em_d, &wi_d, wi_h.len() as u32, bpe as u32, bpe as u32,
        n_used as u32, max_per_expert as u32, chunk, clamp, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    mid_d.copy_to_host(&mut mid_h)?;
    let got_kwide = gather(&mid_h);
    let (rel_kwide, rms_kwide) = report("kwide (Q8_K acts) vs f32 ref", &got_kwide, &want);
    // 2. WMMA (f32 out).
    mid_d.fill_zero()?;
    iq2s.launch_fused_swiglu_wmma(&stream, &mut mid_d, &gate_d, &up_d, &x16_d, &ew_d,
        &gc_d, &em_d, &wi_d, wi_h.len() as u32, bpe as u32, bpe as u32,
        n_used as u32, max_per_expert as u32, chunk, clamp, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    mid_d.copy_to_host(&mut mid_h)?;
    let (rel_wmma, rms_wmma) = report("wmma (f32 out) vs f32 ref", &gather(&mid_h), &want);

    // 3. WMMA production twin (f16 out) — the down kernel's operand.
    let mut mid16_d: DeviceBuffer<u16> = DeviceBuffer::new(igpu.id, b * n_used * n_rows)?;
    mid16_d.fill_zero()?;
    iq2s.launch_fused_swiglu_wmma_f16out(&stream, &mut mid16_d, &gate_d, &up_d, &x16_d, &ew_d,
        &gc_d, &em_d, &wi_d, wi_h.len() as u32, bpe as u32, bpe as u32,
        n_used as u32, max_per_expert as u32, chunk, clamp, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    let mut mid16_h = vec![0u16; b * n_used * n_rows];
    mid16_d.copy_to_host(&mut mid16_h)?;
    let mid16_f: Vec<f32> = mid16_h.iter().map(|&h| f16_to_f32(h)).collect();
    let (rel_h, rms_h) = report("wmma_h (f16 out) vs f32 ref", &gather(&mid16_f), &want);
    for (i, &v) in mid16_f.iter().enumerate() { mid_h[i] = if v != 0.0 { v } else { mid_h[i] }; }

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
    if !(rel_h < 1e-2 && rms_h < 2e-3) { bad.push(format!("wmma_h rel={rel_h:.2e} rms={rms_h:.2e}")); }
    if !(rms_kwide < 1e-1) { bad.push(format!("kwide rel={rel_kwide:.2e} rms={rms_kwide:.2e}")); }
    if leaks > 0 { bad.push(format!("{leaks} padded columns written")); }
    if !bad.is_empty() {
        return Err(eyre!("iq2_s wmma oracle failed: {}", bad.join(", ")));
    }
    Ok(())
}
