//! Oracles for the two blk.26/blk.42 kernel families vs their scalar CPU
//! references on RANDOM blocks:
//!   - MXFP4 down: batched + hetsplit identity + kwide2 partials
//!   - IQ2_S gate/up: fused-SwiGLU batch + hetsplit identity + chunked
//!
//! Run: nix develop -c cargo test --release -p v4flash-kernels \
//!     --test mxfp4_iq2s_oracle -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP};
use v4flash_kernels::iq2_s::{Iq2SPairMatvec, BLOCK_IQ2_S_BYTES};
use v4flash_kernels::iq2_s_tables::cpu_dot_iq2_s_q8_k;
use v4flash_kernels::mxfp4::{Mxfp4Matvec, SUPER_MXFP4_BYTES};
use v4flash_kernels::mxfp4_tables::cpu_dot_mxfp4_q8_k;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

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
}

const F16_SCALES: [u16; 4] = [0x2400, 0x2800, 0x2c00, 0x3000];

fn gen_xq(rng: &mut Lcg, xq: &mut [u8], n_tokens: usize, stride: usize, nb: usize) {
    for t in 0..n_tokens {
        for bi in 0..nb {
            let o = t * stride + bi * BLOCK_Q8_K_BYTES;
            let d_val = 0.02f32 + ((rng.next() & 0xff) as f32) * 0.0008f32;
            xq[o..o + 4].copy_from_slice(&d_val.to_le_bytes());
            let mut bsums = [0i16; 16];
            for k in 0..256 {
                let v = rng.next_byte() as i8;
                xq[o + 4 + k] = v as u8;
                bsums[k / 16] += v as i16;
            }
            for (j, s) in bsums.iter().enumerate() {
                xq[o + 260 + 2 * j..o + 262 + 2 * j].copy_from_slice(&s.to_le_bytes());
            }
        }
    }
}

fn check(name: &str, got: &[f32], want: &[f32], tol: f32) -> eyre::Result<()> {
    let mut max_diff = 0f32;
    let mut max_ref = 0f32;
    for (g, w) in got.iter().zip(want) {
        max_diff = max_diff.max((g - w).abs());
        max_ref = max_ref.max(w.abs());
    }
    let rel = max_diff / max_ref.max(1e-30);
    eprintln!("{name}: n={} max|ref|={max_ref:.4} max_diff={max_diff:.6} rel={rel:.2e}", got.len());
    if rel >= tol {
        return Err(eyre!("{name} diverges: rel={rel}"));
    }
    Ok(())
}

#[test]
#[ignore]
fn mxfp4_kernels_match_cpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let mx = Mxfp4Matvec::for_arch(&arch)?;

    let n_used = N_EXPERT_USED as usize;
    let n_rows = N_EMBD as usize;
    let nb = BLOCKS_Q8K_DOWN_IN as usize;
    let dbpe = n_rows * nb * SUPER_MXFP4_BYTES;
    let stride = nb * BLOCK_Q8_K_BYTES;

    let mut rng = Lcg::new(0x4f4d58); // "MXO"
    let sel: [i32; 6] = [5, 200, 11, 5, 77, 130];
    let mut w_host = vec![0u8; (N_EXPERT as usize) * dbpe];
    for e in [5usize, 200, 11, 77, 130] {
        // e8m0 exponents near 127 (scale ~1); qs fully random.
        for r in 0..n_rows {
            for bi in 0..nb {
                let o = e * dbpe + (r * nb + bi) * SUPER_MXFP4_BYTES;
                for b8 in 0..8 {
                    let bo = o + b8 * 17;
                    w_host[bo] = 120 + (rng.next() & 0x0f) as u8; // 2^(-8)..2^7
                    for j in 1..17 {
                        w_host[bo + j] = rng.next_byte();
                    }
                }
            }
        }
    }
    let mut xq_host = vec![0u8; n_used * stride];
    gen_xq(&mut rng, &mut xq_host, n_used, stride, nb);

    let mut w_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, w_host.len())?;
    w_d.copy_from_host(&w_host)?;
    let mut xq_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq_host.len())?;
    xq_d.copy_from_host(&xq_host)?;
    let mut sel_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_used)?;
    sel_d.copy_from_host(&sel)?;

    let mut want = vec![0f32; n_rows];
    for (s, &e) in sel.iter().enumerate() {
        let xq_s = &xq_host[s * stride..(s + 1) * stride];
        for row in 0..n_rows {
            let wo = (e as usize) * dbpe + row * nb * SUPER_MXFP4_BYTES;
            want[row] += cpu_dot_mxfp4_q8_k(nb, &w_host[wo..wo + nb * SUPER_MXFP4_BYTES], xq_s);
        }
    }

    let mut out_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_rows)?;
    mx.launch_batched(&stream, &mut out_d, &w_d, &xq_d, &sel_d,
        dbpe as u32, stride as u32, n_used as u32, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    let mut got = vec![0f32; n_rows];
    out_d.copy_to_host(&mut got)?;
    check("mxfp4 batched", &got, &want, 5e-3)?;

    // hetsplit identity
    let mut remap_h = vec![-1i32; 256];
    remap_h[200] = 0;
    remap_h[77] = 1;
    let mut remap_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, 256)?;
    remap_d.copy_from_host(&remap_h)?;
    let mut hot_host = vec![0u8; 2 * dbpe];
    hot_host[..dbpe].copy_from_slice(&w_host[200 * dbpe..201 * dbpe]);
    hot_host[dbpe..].copy_from_slice(&w_host[77 * dbpe..78 * dbpe]);
    let mut hot_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, hot_host.len())?;
    hot_d.copy_from_host(&hot_host)?;
    let mut o0: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_rows)?;
    let mut o1: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_rows)?;
    mx.launch_batched_hetsplit(&stream, &mut o0, &w_d, &xq_d, &sel_d, &remap_d, 0, 2,
        dbpe as u32, stride as u32, n_used as u32, n_rows as u32, nb as u32)?;
    mx.launch_batched_hetsplit(&stream, &mut o1, &hot_d, &xq_d, &sel_d, &remap_d, 1, 2,
        dbpe as u32, stride as u32, n_used as u32, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    let mut g0 = vec![0f32; n_rows];
    let mut g1 = vec![0f32; n_rows];
    o0.copy_to_host(&mut g0)?;
    o1.copy_to_host(&mut g1)?;
    let sum: Vec<f32> = g0.iter().zip(&g1).map(|(a, b)| a + b).collect();
    check("mxfp4 hetsplit m0+m1", &sum, &want, 5e-3)?;

    // kwide2 partials (full + partial chunk)
    let b: usize = 48;
    let max_per_expert = b;
    let chunk: u32 = 32;
    let experts: [(usize, i32); 2] = [(5, 32), (130, 19)];
    let n_tokens = b * n_used;
    let mut xq2 = vec![0u8; n_tokens * stride];
    gen_xq(&mut rng, &mut xq2, n_tokens, stride, nb);
    let mut gc_h = vec![0i32; N_EXPERT as usize];
    let mut em_h = vec![0i32; (N_EXPERT as usize) * max_per_expert];
    let mut wi_h: Vec<i32> = Vec::new();
    let mut touched = Vec::new();
    let mut np = 0usize;
    for (e, n) in experts {
        gc_h[e] = n;
        for i in 0..(n as usize) {
            let bi = np % b;
            let sl = (np / b) % n_used;
            em_h[e * max_per_expert + i] = ((bi as i32) << 16) | (sl as i32);
            touched.push((e, bi, sl));
            np += 1;
        }
        wi_h.push(((e as i32) << 16) | 0);
    }
    let mut xq2_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq2.len())?;
    xq2_d.copy_from_host(&xq2)?;
    let mut gc_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, gc_h.len())?;
    gc_d.copy_from_host(&gc_h)?;
    let mut em_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, em_h.len())?;
    em_d.copy_from_host(&em_h)?;
    let mut wi_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, wi_h.len())?;
    wi_d.copy_from_host(&wi_h)?;
    let mut part_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, b * n_used * n_rows)?;
    part_d.fill_zero()?;
    mx.launch_by_expert_kwide2(&stream, &mut part_d, &w_d, &xq2_d, &gc_d, &em_d, &wi_d,
        wi_h.len() as u32, dbpe as u32, stride as u32, n_used as u32,
        max_per_expert as u32, chunk, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    let mut part = vec![0f32; b * n_used * n_rows];
    part_d.copy_to_host(&mut part)?;
    let mut got_t = Vec::new();
    let mut want_t = Vec::new();
    for &(e, bi, sl) in &touched {
        let xo = (bi * n_used + sl) * stride;
        for row in 0..n_rows {
            let wo = e * dbpe + row * nb * SUPER_MXFP4_BYTES;
            want_t.push(cpu_dot_mxfp4_q8_k(nb, &w_host[wo..wo + nb * SUPER_MXFP4_BYTES],
                &xq2[xo..xo + stride]));
            got_t.push(part[(bi * n_used + sl) * n_rows + row]);
        }
    }
    check("mxfp4 kwide2", &got_t, &want_t, 5e-3)?;
    Ok(())
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

#[test]
#[ignore]
fn iq2_s_kernels_match_cpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2s = Iq2SPairMatvec::for_arch(&arch)?;

    let n_used = N_EXPERT_USED as usize;
    let n_rows = N_FF_EXP as usize; // gate/up out dim = 2048
    let nb = BLOCKS_Q8K_GATE_IN as usize; // K = 4096 → 16 superblocks
    let bpe = n_rows * nb * BLOCK_IQ2_S_BYTES;
    let stride = nb * BLOCK_Q8_K_BYTES;
    let clamp = 10.0f32;

    let mut rng = Lcg::new(0x125bad);
    let sel: [i32; 6] = [9, 240, 33, 9, 61, 128];
    let used: [usize; 5] = [9, 240, 33, 61, 128];
    let mut gate_h = vec![0u8; (N_EXPERT as usize) * bpe];
    let mut up_h = vec![0u8; (N_EXPERT as usize) * bpe];
    for w in [&mut gate_h, &mut up_h] {
        for &e in &used {
            for r in 0..n_rows {
                for bi in 0..nb {
                    let o = e * bpe + (r * nb + bi) * BLOCK_IQ2_S_BYTES;
                    let d = F16_SCALES[(rng.next() & 3) as usize].to_le_bytes();
                    w[o..o + 2].copy_from_slice(&d);
                    for i in 2..BLOCK_IQ2_S_BYTES {
                        w[o + i] = rng.next_byte();
                    }
                }
            }
        }
    }
    // Decode: ONE token's activation, shared by every selected expert
    // (the pair kernels read xq[0..n_blocks*292] only).
    let mut xq_host = vec![0u8; stride];
    gen_xq(&mut rng, &mut xq_host, 1, stride, nb);
    let ew_h: Vec<f32> = (0..n_used).map(|i| 0.1 + 0.15 * i as f32).collect();

    let mut gate_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, gate_h.len())?;
    gate_d.copy_from_host(&gate_h)?;
    let mut up_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, up_h.len())?;
    up_d.copy_from_host(&up_h)?;
    let mut xq_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq_host.len())?;
    xq_d.copy_from_host(&xq_host)?;
    let mut sel_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_used)?;
    sel_d.copy_from_host(&sel)?;
    let mut ew_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_used)?;
    ew_d.copy_from_host(&ew_h)?;

    // CPU reference mid[slot][row].
    let mut want = vec![0f32; n_used * n_rows];
    for (s, &e) in sel.iter().enumerate() {
        let xq_s = &xq_host[..];
        for row in 0..n_rows {
            let go = (e as usize) * bpe + row * nb * BLOCK_IQ2_S_BYTES;
            let g = cpu_dot_iq2_s_q8_k(nb, &gate_h[go..go + nb * BLOCK_IQ2_S_BYTES], xq_s);
            let u = cpu_dot_iq2_s_q8_k(nb, &up_h[go..go + nb * BLOCK_IQ2_S_BYTES], xq_s);
            want[s * n_rows + row] = swiglu_ref(g, u, ew_h[s], clamp);
        }
    }

    let mut mid_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_used * n_rows)?;
    iq2s.launch_fused_swiglu_batch(&stream, &mut mid_d, &gate_d, &up_d, &xq_d, &ew_d, &sel_d,
        bpe as u32, bpe as u32, n_used as u32, clamp, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    let mut got = vec![0f32; n_used * n_rows];
    mid_d.copy_to_host(&mut got)?;
    check("iq2_s fused batch", &got, &want, 1e-2)?;

    // hetsplit identity
    let mut remap_h = vec![-1i32; 256];
    remap_h[240] = 0;
    remap_h[61] = 1;
    let mut remap_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, 256)?;
    remap_d.copy_from_host(&remap_h)?;
    let mut hot_g = vec![0u8; 2 * bpe];
    hot_g[..bpe].copy_from_slice(&gate_h[240 * bpe..241 * bpe]);
    hot_g[bpe..].copy_from_slice(&gate_h[61 * bpe..62 * bpe]);
    let mut hot_u = vec![0u8; 2 * bpe];
    hot_u[..bpe].copy_from_slice(&up_h[240 * bpe..241 * bpe]);
    hot_u[bpe..].copy_from_slice(&up_h[61 * bpe..62 * bpe]);
    let mut hot_g_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, hot_g.len())?;
    hot_g_d.copy_from_host(&hot_g)?;
    let mut hot_u_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, hot_u.len())?;
    hot_u_d.copy_from_host(&hot_u)?;
    let mut m0: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_used * n_rows)?;
    let mut m1: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_used * n_rows)?;
    iq2s.launch_fused_swiglu_batch_hetsplit(&stream, &mut m0, &gate_d, &up_d, &xq_d, &ew_d,
        &sel_d, &remap_d, 0, 2, bpe as u32, bpe as u32, n_used as u32, clamp,
        n_rows as u32, nb as u32)?;
    iq2s.launch_fused_swiglu_batch_hetsplit(&stream, &mut m1, &hot_g_d, &hot_u_d, &xq_d, &ew_d,
        &sel_d, &remap_d, 1, 2, bpe as u32, bpe as u32, n_used as u32, clamp,
        n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    let mut g0 = vec![0f32; n_used * n_rows];
    let mut g1 = vec![0f32; n_used * n_rows];
    m0.copy_to_host(&mut g0)?;
    m1.copy_to_host(&mut g1)?;
    let sum: Vec<f32> = g0.iter().zip(&g1).map(|(a, b)| a + b).collect();
    check("iq2_s hetsplit m0+m1", &sum, &want, 1e-2)?;

    // chunked prefill (full + partial chunk)
    let b: usize = 40;
    let max_per_expert = b;
    let chunk: u32 = 16;
    let experts: [(usize, i32); 2] = [(9, 16), (128, 11)];
    // Prefill chunked: xq is per-TOKEN ([B, n_blocks*292]); slots of one
    // token share its activation.
    let mut xq2 = vec![0u8; b * stride];
    gen_xq(&mut rng, &mut xq2, b, stride, nb);
    let ew2: Vec<f32> = (0..b * n_used).map(|i| 0.05 + 0.01 * (i % 17) as f32).collect();
    let mut gc_h = vec![0i32; N_EXPERT as usize];
    let mut em_h = vec![0i32; (N_EXPERT as usize) * max_per_expert];
    let mut wi_h: Vec<i32> = Vec::new();
    let mut touched = Vec::new();
    let mut np = 0usize;
    for (e, n) in experts {
        gc_h[e] = n;
        let mut start = 0;
        while start < n {
            wi_h.push(((e as i32) << 16) | start);
            start += chunk as i32;
        }
        for i in 0..(n as usize) {
            let bi = np % b;
            let sl = (np / b) % n_used;
            em_h[e * max_per_expert + i] = ((bi as i32) << 16) | (sl as i32);
            touched.push((e, bi, sl));
            np += 1;
        }
    }
    let mut xq2_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq2.len())?;
    xq2_d.copy_from_host(&xq2)?;
    let mut ew2_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, ew2.len())?;
    ew2_d.copy_from_host(&ew2)?;
    let mut gc_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, gc_h.len())?;
    gc_d.copy_from_host(&gc_h)?;
    let mut em_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, em_h.len())?;
    em_d.copy_from_host(&em_h)?;
    let mut wi_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, wi_h.len())?;
    wi_d.copy_from_host(&wi_h)?;
    let mut mid2_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, b * n_used * n_rows)?;
    mid2_d.fill_zero()?;
    iq2s.launch_fused_swiglu_chunked(&stream, &mut mid2_d, &gate_d, &up_d, &xq2_d, &ew2_d,
        &gc_d, &em_d, &wi_d, wi_h.len() as u32, bpe as u32, bpe as u32,
        n_used as u32, max_per_expert as u32, chunk, clamp, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    let mut mid2 = vec![0f32; b * n_used * n_rows];
    mid2_d.copy_to_host(&mut mid2)?;
    let mut got_t = Vec::new();
    let mut want_t = Vec::new();
    for &(e, bi, sl) in &touched {
        let xq_s = &xq2[bi * stride..(bi + 1) * stride];
        for row in 0..n_rows {
            let go = e * bpe + row * nb * BLOCK_IQ2_S_BYTES;
            let g = cpu_dot_iq2_s_q8_k(nb, &gate_h[go..go + nb * BLOCK_IQ2_S_BYTES], xq_s);
            let u = cpu_dot_iq2_s_q8_k(nb, &up_h[go..go + nb * BLOCK_IQ2_S_BYTES], xq_s);
            want_t.push(swiglu_ref(g, u, ew2[bi * n_used + sl], clamp));
            got_t.push(mid2[(bi * n_used + sl) * n_rows + row]);
        }
    }
    check("iq2_s chunked", &got_t, &want_t, 1e-2)?;
    Ok(())
}

/// Focused single-block debug: one expert, one slot, tiny rows, no clamp
/// effect — prints g/u pairs so a systematic transform (sign, swap, scale)
/// is visible directly.
#[test]
#[ignore]
fn iq2_s_debug_single() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2s = Iq2SPairMatvec::for_arch(&arch)?;

    let n_used = 1usize;
    let n_rows = 8usize;
    let nb = 1usize; // ONE superblock
    let bpe = n_rows * nb * BLOCK_IQ2_S_BYTES;
    let stride = nb * BLOCK_Q8_K_BYTES;
    let clamp = 0.0f32; // disabled

    let mut rng = Lcg::new(0xd1a6);
    let mut gate_h = vec![0u8; bpe];
    let mut up_h = vec![0u8; bpe];
    for w in [&mut gate_h, &mut up_h] {
        for r in 0..n_rows {
            let o = r * BLOCK_IQ2_S_BYTES;
            w[o..o + 2].copy_from_slice(&F16_SCALES[(rng.next() & 3) as usize].to_le_bytes());
            for i in 2..BLOCK_IQ2_S_BYTES {
                w[o + i] = rng.next_byte();
            }
        }
    }
    let mut xq_host = vec![0u8; stride];
    gen_xq(&mut rng, &mut xq_host, 1, stride, nb);

    let mut gate_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, bpe)?;
    gate_d.copy_from_host(&gate_h)?;
    let mut up_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, bpe)?;
    up_d.copy_from_host(&up_h)?;
    let mut xq_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, stride)?;
    xq_d.copy_from_host(&xq_host)?;
    let mut sel_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, 1)?;
    sel_d.copy_from_host(&[0i32])?;
    let mut ew_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, 1)?;
    ew_d.copy_from_host(&[1.0f32])?;

    let mut mid_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_rows)?;
    iq2s.launch_fused_swiglu_batch(&stream, &mut mid_d, &gate_d, &up_d, &xq_d, &ew_d, &sel_d,
        bpe as u32, bpe as u32, 1, clamp, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    let mut got = vec![0f32; n_rows];
    mid_d.copy_to_host(&mut got)?;

    for row in 0..n_rows {
        let o = row * BLOCK_IQ2_S_BYTES;
        let g = cpu_dot_iq2_s_q8_k(nb, &gate_h[o..o + BLOCK_IQ2_S_BYTES], &xq_host);
        let u = cpu_dot_iq2_s_q8_k(nb, &up_h[o..o + BLOCK_IQ2_S_BYTES], &xq_host);
        let want = swiglu_ref(g, u, 1.0, clamp);
        eprintln!("row {row}: cpu g={g:+.4} u={u:+.4} want={want:+.4}  got={:+.4}", got[row]);
    }
    Ok(())
}
