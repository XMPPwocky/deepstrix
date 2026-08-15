//! GPU oracle for the IQ2_XS gate/up fused-SwiGLU kernels, versus the
//! scalar CPU reference.
//!
//! Covers all three load-bearing variants:
//!   - fused-SwiGLU decode batch
//!   - hetsplit identity: mode 0 + mode 1 must reconstruct the full result
//!   - chunked by-expert prefill
//!
//! The CPU side (`cpu_dot_iq2_xs_q8_k`) is itself pinned against
//! llama.cpp's own reference in tests/iq2_xs_cpu_ref.rs, so a pass here
//! chains back to upstream rather than to a self-consistent assumption.
//!
//! NOTE (same trap as the iq2_s oracle): the decode pair kernels read ONE
//! shared token's xq, and the chunked prefill kernel indexes xq per-TOKEN,
//! not per-(token, slot).
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP};
use v4flash_kernels::iq2_xs::{Iq2XsPairMatvec, BLOCK_IQ2_XS_BYTES};
use v4flash_kernels::iq2_xs_tables::cpu_dot_iq2_xs_q8_k;
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
fn iq2_xs_kernels_match_cpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2s = Iq2XsPairMatvec::for_arch(&arch)?;

    let n_used = N_EXPERT_USED as usize;
    let n_rows = N_FF_EXP as usize; // gate/up out dim = 2048
    let nb = BLOCKS_Q8K_GATE_IN as usize; // K = 4096 → 16 superblocks
    let bpe = n_rows * nb * BLOCK_IQ2_XS_BYTES;
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
                    let o = e * bpe + (r * nb + bi) * BLOCK_IQ2_XS_BYTES;
                    let d = F16_SCALES[(rng.next() & 3) as usize].to_le_bytes();
                    w[o..o + 2].copy_from_slice(&d);
                    for i in 2..BLOCK_IQ2_XS_BYTES {
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
            let go = (e as usize) * bpe + row * nb * BLOCK_IQ2_XS_BYTES;
            let g = cpu_dot_iq2_xs_q8_k(nb, &gate_h[go..go + nb * BLOCK_IQ2_XS_BYTES], xq_s);
            let u = cpu_dot_iq2_xs_q8_k(nb, &up_h[go..go + nb * BLOCK_IQ2_XS_BYTES], xq_s);
            want[s * n_rows + row] = swiglu_ref(g, u, ew_h[s], clamp);
        }
    }

    let mut mid_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_used * n_rows)?;
    iq2s.launch_fused_swiglu_batch(&stream, &mut mid_d, &gate_d, &up_d, &xq_d, &ew_d, &sel_d,
        bpe as u32, bpe as u32, n_used as u32, clamp, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    let mut got = vec![0f32; n_used * n_rows];
    mid_d.copy_to_host(&mut got)?;
    check("iq2_xs fused batch", &got, &want, 1e-2)?;

    // hetsplit identity
    // M63: the kernels read the miss branch as -(iGPU slot + 1), not a bare
    // -1. Go through the real encoder so this test exercises exactly what
    // HetModelWeights::load_all builds (packed=false => slot == expert id).
    let mut remap_h = vec![-1i32; 256];
    remap_h[240] = 0;
    remap_h[61] = 1;
    let mut remap_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, 256)?;
    v4flash_kernels::het::weights::encode_igpu_remap(&mut remap_h, false);
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
    check("iq2_xs hetsplit m0+m1", &sum, &want, 1e-2)?;

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
            let go = e * bpe + row * nb * BLOCK_IQ2_XS_BYTES;
            let g = cpu_dot_iq2_xs_q8_k(nb, &gate_h[go..go + nb * BLOCK_IQ2_XS_BYTES], xq_s);
            let u = cpu_dot_iq2_xs_q8_k(nb, &up_h[go..go + nb * BLOCK_IQ2_XS_BYTES], xq_s);
            want_t.push(swiglu_ref(g, u, ew2[bi * n_used + sl], clamp));
            got_t.push(mid2[(bi * n_used + sl) * n_rows + row]);
        }
    }
    check("iq2_xs chunked", &got_t, &want_t, 1e-2)?;

    // kwide prefill: same work-items contract, same inputs, same expected
    // values (f32 reduction order differs -> same tolerance as chunked).
    mid2_d.fill_zero()?;
    iq2s.launch_fused_swiglu_kwide(&stream, &mut mid2_d, &gate_d, &up_d, &xq2_d, &ew2_d,
        &gc_d, &em_d, &wi_d, wi_h.len() as u32, bpe as u32, bpe as u32,
        n_used as u32, max_per_expert as u32, chunk, clamp, n_rows as u32, nb as u32)?;
    stream.synchronize()?;
    mid2_d.copy_to_host(&mut mid2)?;
    let mut got_k = Vec::new();
    for &(_, bi, sl) in &touched {
        for row in 0..n_rows {
            got_k.push(mid2[(bi * n_used + sl) * n_rows + row]);
        }
    }
    check("iq2_xs kwide", &got_k, &want_t, 1e-2)?;
    Ok(())
}
