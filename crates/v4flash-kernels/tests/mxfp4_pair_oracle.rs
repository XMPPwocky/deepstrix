//! MXFP4 fused gate+up SwiGLU pair kernels vs a CPU reference
//! (`cpu_dot_mxfp4_q8_k`, pinned to ggml's block_mxfp4 semantics, then the
//! same clamp+SwiGLU the engine applies). Shapes come from `config.rs`, so
//! the same test runs at V4-Flash (2048 rows, 16 superblocks) and, with
//! `--features v41`, at V4.1 (2304 rows, 20 superblocks — the 25% lane
//! imbalance case). Small: only the selected experts are materialised.
//!
//!   cargo test -p v4flash-kernels --release --test mxfp4_pair_oracle -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EXPERT, N_EXPERT_USED, N_FF_EXP};
use v4flash_kernels::mxfp4_pair::Mxfp4PairMatvec;
use v4flash_kernels::mxfp4_tables::{cpu_dot_mxfp4_q8_k, SUPER_MXFP4_BYTES};
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

const QK_K: usize = 256;

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
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
    fn next_unit(&mut self) -> f32 {
        (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn quantize_q8_k(x: &[f32], out: &mut [u8]) {
    assert_eq!(x.len() % QK_K, 0);
    assert_eq!(out.len(), x.len() / QK_K * BLOCK_Q8_K_BYTES);
    for (bi, xs) in x.chunks_exact(QK_K).enumerate() {
        let o = bi * BLOCK_Q8_K_BYTES;
        let mut max = 0f32;
        let mut amax = 0f32;
        for &v in xs {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            out[o..o + BLOCK_Q8_K_BYTES].fill(0);
            continue;
        }
        let iscale = -127.0f32 / max;
        let mut bsums = [0i16; 16];
        for (k, &v) in xs.iter().enumerate() {
            let q = (iscale * v).round() as i32;
            let q = q.min(127) as i8;
            out[o + 4 + k] = q as u8;
            bsums[k / 16] += q as i16;
        }
        out[o..o + 4].copy_from_slice(&(1.0f32 / iscale).to_le_bytes());
        for (j, s) in bsums.iter().enumerate() {
            out[o + 260 + 2 * j..o + 262 + 2 * j].copy_from_slice(&s.to_le_bytes());
        }
    }
}

fn swiglu_ref(g: f32, u: f32, ew: f32, clamp: f32) -> f32 {
    let (mut g, mut u) = (g, u);
    if clamp > 1.0e-6 {
        g = g.min(clamp);
        u = u.min(clamp).max(-clamp);
    }
    let sig = 1.0 / (1.0 + (-g).exp());
    g * sig * u * ew
}

fn check(name: &str, got: &[f32], want: &[f32], tol: f32) -> eyre::Result<()> {
    assert_eq!(got.len(), want.len());
    let mut max_diff = 0f32;
    let mut max_ref = 0f32;
    for (a, b) in got.iter().zip(want) {
        max_diff = max_diff.max((a - b).abs());
        max_ref = max_ref.max(b.abs());
    }
    let rel = max_diff / max_ref.max(1e-30);
    eprintln!("{name}: n={} max|ref|={max_ref:.4} max_abs_diff={max_diff:.3e} rel={rel:.3e}", got.len());
    if !rel.is_finite() || rel >= tol {
        return Err(eyre!("{name} diverges: rel={rel}"));
    }
    Ok(())
}

#[test]
#[ignore]
fn mxfp4_pair_kernels_match_cpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let k = Mxfp4PairMatvec::for_arch(&arch)?;

    let n_used = N_EXPERT_USED;
    let n_rows = N_FF_EXP as usize;
    let nb = BLOCKS_Q8K_GATE_IN as usize; // superblocks per row (in = N_EMBD)
    let bpe = n_rows * nb * SUPER_MXFP4_BYTES;
    let stride = nb * BLOCK_Q8_K_BYTES;
    eprintln!("shape: n_rows={n_rows} superblocks={nb} bpe={bpe} n_expert={N_EXPERT}");

    // Only the ids we select are materialised: keep them < n_slots.
    let n_slots = 8usize;
    let sel: Vec<i32> = [5, 2, 7, 5, 0, 3, 1, 6][..n_used].to_vec();
    let mut rng = Lcg::new(0x4d5850_34); // "MXP4"
    let mut gate_h = vec![0u8; n_slots * bpe];
    let mut up_h = vec![0u8; n_slots * bpe];
    for w in [&mut gate_h, &mut up_h] {
        for e in 0..n_slots {
            for r in 0..n_rows {
                for bi in 0..nb {
                    let o = e * bpe + (r * nb + bi) * SUPER_MXFP4_BYTES;
                    // v2 super-block: nibbles at b8*16, scale at 128+b8.
                    for b8 in 0..8 {
                        w[o + 128 + b8] = 118 + (rng.next() & 0x0f) as u8; // 2^-10..2^5
                        for j in 0..16 {
                            w[o + b8 * 16 + j] = rng.next_byte();
                        }
                    }
                }
            }
        }
    }
    let x: Vec<f32> = (0..nb * QK_K).map(|_| rng.next_unit()).collect();
    let mut xq_h = vec![0u8; stride];
    quantize_q8_k(&x, &mut xq_h);
    let ew_h: Vec<f32> = (0..n_used).map(|i| 0.3 + 0.1 * i as f32).collect();
    let clamp = 10.0f32;

    let mut want = vec![0f32; n_used * n_rows];
    for (s, &e) in sel.iter().enumerate() {
        for row in 0..n_rows {
            let wo = (e as usize) * bpe + row * nb * SUPER_MXFP4_BYTES;
            let g = cpu_dot_mxfp4_q8_k(nb, &gate_h[wo..wo + nb * SUPER_MXFP4_BYTES], &xq_h);
            let u = cpu_dot_mxfp4_q8_k(nb, &up_h[wo..wo + nb * SUPER_MXFP4_BYTES], &xq_h);
            want[s * n_rows + row] = swiglu_ref(g, u, ew_h[s], clamp);
        }
    }

    let mut gate_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, gate_h.len())?;
    gate_d.copy_from_host(&gate_h)?;
    let mut up_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, up_h.len())?;
    up_d.copy_from_host(&up_h)?;
    let mut xq_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq_h.len())?;
    xq_d.copy_from_host(&xq_h)?;
    let mut ew_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_used)?;
    ew_d.copy_from_host(&ew_h)?;
    let mut sel_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_used)?;
    sel_d.copy_from_host(&sel)?;
    let mut mid_d: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_used * n_rows)?;

    k.launch_fused_swiglu_batch(
        &stream, &mut mid_d, &gate_d, &up_d, &xq_d, &ew_d, &sel_d,
        bpe as u32, bpe as u32, n_used as u32, clamp, n_rows as u32, nb as u32,
    )?;
    stream.synchronize()?;
    let mut got = vec![0f32; n_used * n_rows];
    mid_d.copy_to_host(&mut got)?;
    check("mxfp4 pair batch", &got, &want, 5e-3)?;
    // MXFP4_PAIR_BENCH=1: time the decode gate/up leg (n_used experts, all weight bytes
    // read once) → effective GB/s; the down kernel reads half as many bytes.
    if std::env::var("MXFP4_PAIR_BENCH").is_ok() {
        let iters = 200;
        for _ in 0..20 {
            k.launch_fused_swiglu_batch(&stream, &mut mid_d, &gate_d, &up_d, &xq_d, &ew_d, &sel_d,
                bpe as u32, bpe as u32, n_used as u32, clamp, n_rows as u32, nb as u32)?;
        }
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            k.launch_fused_swiglu_batch(&stream, &mut mid_d, &gate_d, &up_d, &xq_d, &ew_d, &sel_d,
                bpe as u32, bpe as u32, n_used as u32, clamp, n_rows as u32, nb as u32)?;
        }
        stream.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let bytes = 2.0 * n_used as f64 * bpe as f64;
        eprintln!("BENCH pair batch: {n_used} experts × 2 × {bpe} B = {:.1} MB in {ms:.3} ms/launch → {:.0} GB/s effective",
            bytes / 1e6, bytes / ms / 1e6);
    }

    // Het-split identity: hot set {2, 7} on a "dGPU" buffer (dense slots
    // 0/1), everything else cold at its raw id; mode-0 + mode-1 outputs sum
    // to the plain batch result.
    let mut remap_h = vec![-1i32; N_EXPERT as usize];
    remap_h[2] = 0;
    remap_h[7] = 1;
    v4flash_kernels::het::weights::encode_igpu_remap(&mut remap_h, false);
    let mut remap_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, remap_h.len())?;
    remap_d.copy_from_host(&remap_h)?;
    let mut hot_gate = vec![0u8; 2 * bpe];
    let mut hot_up = vec![0u8; 2 * bpe];
    for (dense, e) in [(0usize, 2usize), (1, 7)] {
        hot_gate[dense * bpe..(dense + 1) * bpe].copy_from_slice(&gate_h[e * bpe..(e + 1) * bpe]);
        hot_up[dense * bpe..(dense + 1) * bpe].copy_from_slice(&up_h[e * bpe..(e + 1) * bpe]);
    }
    let mut hg_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, hot_gate.len())?;
    hg_d.copy_from_host(&hot_gate)?;
    let mut hu_d: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, hot_up.len())?;
    hu_d.copy_from_host(&hot_up)?;
    let mut m0: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_used * n_rows)?;
    let mut m1: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_used * n_rows)?;
    let cap = 2u32;
    k.launch_fused_swiglu_batch_hetsplit(
        &stream, &mut m0, &gate_d, &up_d, &xq_d, &ew_d, &sel_d, &remap_d, 0, cap,
        bpe as u32, bpe as u32, n_used as u32, clamp, n_rows as u32, nb as u32,
    )?;
    k.launch_fused_swiglu_batch_hetsplit(
        &stream, &mut m1, &hg_d, &hu_d, &xq_d, &ew_d, &sel_d, &remap_d, 1, cap,
        bpe as u32, bpe as u32, n_used as u32, clamp, n_rows as u32, nb as u32,
    )?;
    stream.synchronize()?;
    let mut g0 = vec![0f32; n_used * n_rows];
    let mut g1 = vec![0f32; n_used * n_rows];
    m0.copy_to_host(&mut g0)?;
    m1.copy_to_host(&mut g1)?;
    let sum: Vec<f32> = g0.iter().zip(&g1).map(|(a, b)| a + b).collect();
    check("mxfp4 pair hetsplit (iGPU + dGPU halves)", &sum, &got, 1e-6)?;
    // And each half only touched its own slots.
    for (s, &e) in sel.iter().enumerate() {
        let hot = e == 2 || e == 7;
        let (own, other) = if hot { (&g1, &g0) } else { (&g0, &g1) };
        assert!(other[s * n_rows..(s + 1) * n_rows].iter().all(|&v| v == 0.0), "slot {s} zero-fill");
        assert!(own[s * n_rows..(s + 1) * n_rows].iter().any(|&v| v != 0.0), "slot {s} computed");
    }

    // Prefill twins (work-items contract): chunked (per-member re-dequant) and
    // kwide (dequant once per super-block pair) at three batch shapes — a
    // single partial chunk, full + partial chunks, and chunk 32 (=
    // MXFP4_KW_MAX_CHUNK) with 33 members → 2 work items. Members are spread
    // over (token, slot) pairs; untouched pairs must stay zero.
    for (b, chunk, experts) in [
        (7usize, 16u32, vec![(5usize, 7i32), (2, 5), (7, 3)]),
        (40, 16, vec![(5, 40), (2, 17), (0, 3)]),
        (64, 32, vec![(5, 33), (3, 64), (1, 1)]),
    ] {
        let max_per_expert = b.max(experts.iter().map(|&(_, n)| n as usize).max().unwrap_or(0));
        let mut xq2 = vec![0u8; b * stride];
        for t in 0..b {
            let x: Vec<f32> = (0..nb * QK_K).map(|_| rng.next_unit()).collect();
            quantize_q8_k(&x, &mut xq2[t * stride..(t + 1) * stride]);
        }
        let ew2: Vec<f32> = (0..b * n_used).map(|i| 0.05 + 0.01 * (i % 17) as f32).collect();
        let mut gc_h = vec![0i32; n_slots];
        let mut em_h = vec![0i32; n_slots * max_per_expert];
        let mut wi_h: Vec<i32> = Vec::new();
        let mut touched = Vec::new();
        let mut np = 0usize;
        for &(e, n) in &experts {
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
        assert!(np <= b * n_used, "distinct (token, slot) pairs exhausted");
        let mut want_t = Vec::with_capacity(touched.len() * n_rows);
        for &(e, bi, sl) in &touched {
            let xq_s = &xq2[bi * stride..(bi + 1) * stride];
            for row in 0..n_rows {
                let wo = e * bpe + row * nb * SUPER_MXFP4_BYTES;
                let g = cpu_dot_mxfp4_q8_k(nb, &gate_h[wo..wo + nb * SUPER_MXFP4_BYTES], xq_s);
                let u = cpu_dot_mxfp4_q8_k(nb, &up_h[wo..wo + nb * SUPER_MXFP4_BYTES], xq_s);
                want_t.push(swiglu_ref(g, u, ew2[bi * n_used + sl], clamp));
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
        let mut mid2 = vec![0f32; b * n_used * n_rows];
        let shape = format!("B={b} chunk={chunk} items={} members={}", wi_h.len(), touched.len());
        for kwide in [false, true] {
            mid2_d.fill_zero()?;
            if kwide {
                k.launch_fused_swiglu_kwide(
                    &stream, &mut mid2_d, &gate_d, &up_d, &xq2_d, &ew2_d, &gc_d, &em_d, &wi_d,
                    wi_h.len() as u32, bpe as u32, bpe as u32, n_used as u32, max_per_expert as u32,
                    chunk, clamp, n_rows as u32, nb as u32,
                )?;
            } else {
                k.launch_fused_swiglu_chunked(
                    &stream, &mut mid2_d, &gate_d, &up_d, &xq2_d, &ew2_d, &gc_d, &em_d, &wi_d,
                    wi_h.len() as u32, bpe as u32, bpe as u32, n_used as u32, max_per_expert as u32,
                    chunk, clamp, n_rows as u32, nb as u32,
                )?;
            }
            stream.synchronize()?;
            mid2_d.copy_to_host(&mut mid2)?;
            let mut got_t = Vec::with_capacity(want_t.len());
            for &(_, bi, sl) in &touched {
                got_t.extend_from_slice(&mid2[(bi * n_used + sl) * n_rows..(bi * n_used + sl + 1) * n_rows]);
            }
            let touched_set: std::collections::HashSet<(usize, usize)> =
                touched.iter().map(|&(_, bi, sl)| (bi, sl)).collect();
            for bi in 0..b {
                for sl in 0..n_used {
                    if !touched_set.contains(&(bi, sl)) {
                        let base = (bi * n_used + sl) * n_rows;
                        if mid2[base..base + n_rows].iter().any(|&v| v != 0.0) {
                            return Err(eyre!("prefill wrote an untouched (token {bi}, slot {sl})"));
                        }
                    }
                }
            }
            let name = if kwide { "kwide" } else { "chunked" };
            check(&format!("mxfp4 pair prefill {name} {shape}"), &got_t, &want_t, 5e-3)?;
        }
    }
    eprintln!("OK");
    Ok(())
}
