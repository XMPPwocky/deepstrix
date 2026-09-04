//! GPU oracle for the IQ3_S gate/up fused-SwiGLU PAIR kernels
//! (`kernels/iq3_s_pair_matvec.hip`) versus the scalar CPU reference.
//!
//! Covers all four variants on BOTH devices (the iGPU holds the cold copy
//! of blk.26; the dGPU runs the hot-expert decode hetsplit + the
//! hot-expert prefill kwide on gfx1201):
//!   - fused-SwiGLU decode batch (B=1, duplicate expert in `sel`)
//!   - hetsplit identity: mode 0 + mode 1 must reconstruct the full result
//!   - chunked by-expert prefill at B=7, B=40 (chunk 16: full + partial
//!     chunk) and B=64 (chunk 32 = IQ3S_KW_MAX_CHUNK, 33 members → 2 items)
//!   - kwide prefill at the same three batch shapes
//!
//! Weights come from two sources: LCG-random blocks (every one of the 512
//! grid rows, random qh/signs/scales, `d` VARIED per super-block — a uniform
//! `d` once let a wrong kernel pass) and the REAL blk.26 experts 0..4 of
//! unsloth's Vision-Exp UD-IQ3_XXS (`DEEPSTRIX_IQ3S_BLOB_DIR` holding
//! `blk26_gate_exps_e0-4.iq3s` / `blk26_up_exps_e0-4.iq3s`, 5 × 3,604,480 B
//! each; the real tests skip when unset). Activations are random f32 vectors
//! quantized to Q8_K on the host exactly as `quantize_row_q8_K` does, so the
//! GPU and CPU consume the same bytes.
//!
//! The CPU side (`cpu_dot_iq3_s_q8_k`) is pinned to llama.cpp's own
//! `ggml_vec_dot_iq3_s_q8_K_generic` by `tests/iq3_s_cpu_ref.rs`; the decode
//! batch is additionally checked against an f64 dequant-then-dot so the
//! integer path and the float path agree independently.
//!
//! Memory: 8 synthetic (or 5 real) experts × 3.6 MB × 2 matrices ≈ 58 MB
//! (36 MB) per device plus < 4 MB of activations/outputs — deliberately tiny
//! (a production server shares these GPUs). Run one test at a time:
//!
//!   DEEPSTRIX_IQ3S_BLOB_DIR=... nix develop -c cargo test --release \
//!     -p v4flash-kernels --test iq3_s_oracle -- --ignored --nocapture \
//!     --test-threads=1
//!
//! NOTE (same trap as the iq2_s oracle): the decode pair kernels read ONE
//! shared token's xq, and the prefill kernels index xq per-TOKEN, not
//! per-(token, slot).
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EXPERT_USED, N_FF_EXP};
use v4flash_kernels::iq3_s::{Iq3SPairMatvec, BLOCK_IQ3_S_BYTES};
use v4flash_kernels::iq3_s_tables::{cpu_dot_iq3_s_q8_k, dequant_row_iq3_s};
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

const QK_K: usize = 256;

fn pick_device(prefix: &str) -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(prefix) {
            return Ok(d);
        }
    }
    Err(eyre!("no {prefix} device"))
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
    /// Uniform in [-1, 1).
    fn next_unit(&mut self) -> f32 {
        (self.next() as f32) / 2147483648.0 - 1.0
    }
}

/// f16 `d` values varied per super-block (1.1·2^-14 .. 1.1·2^-11; the real
/// blk.26 experts have d ≤ 5.5e-4 ≈ 2^-11). Smaller than the siblings'
/// 2^-6..2^-3 because IQ3_S has no 0.125/0.25 prefactor and its `ls` reaches
/// 31: with the larger range nearly every gate/up dot exceeded the SwiGLU
/// clamp and the oracle degenerated to comparing clamped constants. The
/// clamped fraction is printed per check to keep this honest. Mantissa bits
/// are deliberately non-zero (0x..66, as in the sibling oracles): exact
/// powers of two would let an f16 mantissa/byte-order decode bug in `d`
/// pass the synthetic run.
const F16_SCALES: [u16; 4] = [0x0466, 0x0866, 0x0c66, 0x1066];

/// Pass tolerance on `max_abs_diff / max|ref|` (and on the unclamped
/// normaliser). Measured kernel error is 1e-7..4e-7 on both devices,
/// synthetic and real weights (f32 accumulation-order noise), so 1e-4 keeps
/// >100× headroom; a CPU probe with injected unpack bugs (one wrong sign
/// per block, one dropped qh bit, swapped nibbles, a 0.125 prefactor, …)
/// lands at rel 2.7e-1..1.5 on this distribution. The sibling iq2_s /
/// iq3_xxs oracles keep 1e-2; theirs measure 2.6e-5..1.2e-4.
const TOL: f32 = 1e-4;

/// Host port of ggml `quantize_row_q8_K_ref`: per 256-block, `iscale =
/// -127/max` (signed max-magnitude element), `q = round(iscale*x)` clamped
/// to 127, `bsums` over 16-element groups, `d = 1/iscale`.
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

/// `n_tokens` random f32 activations of `nb*256` elements, Q8_K-quantized
/// into `xq` (row stride `stride` bytes). Amplitude varied per token so the
/// per-block `y.d` differs across tokens and blocks.
fn gen_xq(rng: &mut Lcg, xq: &mut [u8], n_tokens: usize, stride: usize, nb: usize) {
    let mut x = vec![0f32; nb * QK_K];
    for t in 0..n_tokens {
        let amp = 0.5 + ((rng.next() & 0xff) as f32) / 64.0;
        for (bi, blk) in x.chunks_exact_mut(QK_K).enumerate() {
            let bamp = amp * (0.25 + ((rng.next() & 0xff) as f32) / 128.0);
            let _ = bi;
            for v in blk.iter_mut() {
                *v = bamp * rng.next_unit();
            }
        }
        quantize_q8_k(&x, &mut xq[t * stride..t * stride + nb * BLOCK_Q8_K_BYTES]);
    }
}

/// How often the SwiGLU clamp saturates a gate/up dot in the reference: a
/// clamped output hides kernel errors, so the oracle prints this fraction.
#[derive(Default)]
struct ClampStats {
    n: usize,
    g_clamped: usize,
    u_clamped: usize,
    /// max|out| over dots where neither gate nor up saturated — the
    /// normaliser for `check`'s second metric, so the clamp-saturated
    /// constant (10·σ(10)·10·ew_max ≈ 85) cannot dominate it.
    unclamped_max_ref: f32,
}

impl ClampStats {
    fn add(&mut self, g: f32, u: f32, clamp: f32, out: f32) {
        self.n += 1;
        let mut saturated = false;
        if clamp > 1.0e-6 {
            if g > clamp {
                self.g_clamped += 1;
                saturated = true;
            }
            if u.abs() > clamp {
                self.u_clamped += 1;
                saturated = true;
            }
        }
        if !saturated {
            self.unclamped_max_ref = self.unclamped_max_ref.max(out.abs());
        }
    }
    fn report(&self) -> String {
        format!(
            "clamped: gate {:.1}% up {:.1}% of {} dots",
            100.0 * self.g_clamped as f64 / self.n.max(1) as f64,
            100.0 * self.u_clamped as f64 / self.n.max(1) as f64,
            self.n
        )
    }
}

#[allow(dead_code)]
struct CheckStats {
    max_diff: f32,
    max_ref: f32,
    rel: f32,
}

/// `unclamped_ref` = max|ref| over outputs whose gate/up did NOT saturate
/// (`ClampStats::unclamped_max_ref`): the same `max_abs_diff` is also
/// required to pass against that normaliser, so a clamp-bounded max|ref|
/// cannot loosen the check for the unclamped rows.
fn check(
    name: &str,
    got: &[f32],
    want: &[f32],
    unclamped_ref: f32,
    tol: f32,
) -> eyre::Result<CheckStats> {
    assert_eq!(got.len(), want.len());
    let mut max_diff = 0f32;
    let mut max_ref = 0f32;
    let mut sum_abs = 0f64;
    let mut nonfinite = 0usize;
    for (g, w) in got.iter().zip(want) {
        if !g.is_finite() {
            nonfinite += 1;
        }
        max_diff = max_diff.max((g - w).abs());
        max_ref = max_ref.max(w.abs());
        sum_abs += (g - w).abs() as f64;
    }
    let rel = max_diff / max_ref.max(1e-30);
    let rel_unclamped = max_diff / unclamped_ref.max(1e-30);
    eprintln!(
        "{name}: n={} max|ref|={max_ref:.5} unclamped max|ref|={unclamped_ref:.5} max_abs_diff={max_diff:.3e} mean_abs_diff={:.3e} rel(max_diff/max|ref|)={rel:.2e} rel_unclamped={rel_unclamped:.2e}",
        got.len(),
        sum_abs / got.len().max(1) as f64
    );
    if nonfinite > 0 {
        return Err(eyre!("{name}: {nonfinite} non-finite outputs"));
    }
    if unclamped_ref <= 0.0 {
        return Err(eyre!("{name}: every reference output is clamp-saturated"));
    }
    if rel >= tol || rel_unclamped >= tol {
        return Err(eyre!(
            "{name} diverges: rel={rel} rel_unclamped={rel_unclamped}"
        ));
    }
    Ok(CheckStats {
        max_diff,
        max_ref,
        rel,
    })
}

fn swiglu_ref(g: f32, u: f32, ew: f32, clamp: f32) -> f32 {
    let mut g = g;
    let mut u = u;
    if clamp > 1.0e-6 {
        if g > clamp {
            g = clamp;
        }
        if u > clamp {
            u = clamp;
        }
        if u < -clamp {
            u = -clamp;
        }
    }
    let sig = 1.0 / (1.0 + (-g).exp());
    g * sig * u * ew
}

/// f64 dequant-then-dot: Σ w_f32[k] · (y.d · q8[k]) over `nb` blocks.
fn dot_dequant_f64(nb: usize, w: &[u8], xq: &[u8]) -> f64 {
    let mut wf = vec![0f32; nb * QK_K];
    dequant_row_iq3_s(&w[..nb * BLOCK_IQ3_S_BYTES], &mut wf);
    let mut acc = 0f64;
    for b in 0..nb {
        let o = b * BLOCK_Q8_K_BYTES;
        let yd = f32::from_le_bytes([xq[o], xq[o + 1], xq[o + 2], xq[o + 3]]) as f64;
        for k in 0..QK_K {
            let q = xq[o + 4 + k] as i8 as f64;
            acc += wf[b * QK_K + k] as f64 * yd * q;
        }
    }
    acc
}

/// Expert weights for the oracle: `n_e` experts, `bpe` bytes each, gate and
/// up laid out `[expert][row][block]` exactly as the GGUF tensor is.
struct Weights {
    label: String,
    n_e: usize,
    gate: Vec<u8>,
    up: Vec<u8>,
}

fn synthetic_weights(seed: u64, n_e: usize, n_rows: usize, nb: usize) -> Weights {
    let bpe = n_rows * nb * BLOCK_IQ3_S_BYTES;
    let mut rng = Lcg::new(seed);
    let mut gate = vec![0u8; n_e * bpe];
    let mut up = vec![0u8; n_e * bpe];
    for w in [&mut gate, &mut up] {
        for e in 0..n_e {
            for r in 0..n_rows {
                for bi in 0..nb {
                    let o = e * bpe + (r * nb + bi) * BLOCK_IQ3_S_BYTES;
                    let d = F16_SCALES[(rng.next() & 3) as usize].to_le_bytes();
                    w[o..o + 2].copy_from_slice(&d);
                    // qs (64) | qh (8) | signs (32) | scales (4): all bit
                    // patterns are valid IQ3_S — every grid row, every
                    // sign combination, every scale nibble is reachable.
                    for i in 2..BLOCK_IQ3_S_BYTES {
                        w[o + i] = rng.next_byte();
                    }
                }
            }
        }
    }
    Weights {
        label: format!("synthetic(seed={seed:#x},n_e={n_e})"),
        n_e,
        gate,
        up,
    }
}

/// Real blk.26 experts 0..4 (Vision-Exp UD-IQ3_XXS), or None when the
/// blob dir is not configured.
fn real_weights(n_rows: usize, nb: usize) -> eyre::Result<Option<Weights>> {
    let Ok(dir) = std::env::var("DEEPSTRIX_IQ3S_BLOB_DIR") else {
        eprintln!("DEEPSTRIX_IQ3S_BLOB_DIR unset — real-expert oracle skipped");
        return Ok(None);
    };
    let bpe = n_rows * nb * BLOCK_IQ3_S_BYTES;
    let gate = std::fs::read(format!("{dir}/blk26_gate_exps_e0-4.iq3s"))?;
    let up = std::fs::read(format!("{dir}/blk26_up_exps_e0-4.iq3s"))?;
    if gate.len() % bpe != 0 || up.len() != gate.len() || gate.is_empty() {
        return Err(eyre!(
            "blob sizes gate={} up={} not a multiple of {bpe} B/expert",
            gate.len(),
            up.len()
        ));
    }
    let n_e = gate.len() / bpe;
    // Sanity: every d finite, and |w| ≤ 15·31·d (grid max × ls max).
    let mut d_max = 0f32;
    for w in [&gate, &up] {
        for blk in w.chunks_exact(BLOCK_IQ3_S_BYTES) {
            let d = v4flash_core::kquants::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            if !d.is_finite() {
                return Err(eyre!("real blob: non-finite d"));
            }
            d_max = d_max.max(d.abs());
        }
    }
    eprintln!(
        "real blk.26 experts: n_e={n_e}, d_max={d_max:.3e}, |w|max ≤ {:.3e}",
        15.0 * 31.0 * d_max
    );
    Ok(Some(Weights {
        label: format!("real blk.26 e0..{}", n_e - 1),
        n_e,
        gate,
        up,
    }))
}

/// Everything the four variants need on one device. Buffers are dropped
/// with the struct — one device's working set is ~60 MB.
struct Ctx {
    dev: Device,
    stream: Stream,
    k: Iq3SPairMatvec,
    n_rows: usize,
    nb: usize,
    bpe: usize,
    stride: usize,
    clamp: f32,
    gate_d: DeviceBuffer<u8>,
    up_d: DeviceBuffer<u8>,
}

fn upload<T: Copy>(dev: &Device, h: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(dev.id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}

/// Decode batch (B=1) + hetsplit identity. `sel` carries a duplicated
/// expert; `hot` = the two experts resident on the "dGPU" side (slots of
/// `sel` that map to them go to mode 1).
fn run_decode(
    c: &Ctx,
    w: &Weights,
    rng: &mut Lcg,
    sel: [i32; 6],
    hot: [usize; 2],
) -> eyre::Result<()> {
    let n_used = N_EXPERT_USED as usize;
    let (n_rows, nb, bpe, stride, clamp) = (c.n_rows, c.nb, c.bpe, c.stride, c.clamp);
    let mut xq_h = vec![0u8; stride];
    gen_xq(rng, &mut xq_h, 1, stride, nb);
    let ew_h: Vec<f32> = (0..n_used).map(|i| 0.1 + 0.15 * i as f32).collect();

    let xq_d = upload(&c.dev, &xq_h)?;
    let sel_d = upload(&c.dev, &sel)?;
    let ew_d = upload(&c.dev, &ew_h)?;

    // CPU reference mid[slot][row], integer path + f64 dequant path.
    let mut want = vec![0f32; n_used * n_rows];
    let mut want64 = vec![0f32; n_used * n_rows];
    let mut max_int_vs_f64 = 0f64;
    let mut clamped = ClampStats::default();
    for (s, &e) in sel.iter().enumerate() {
        for row in 0..n_rows {
            let go = (e as usize) * bpe + row * nb * BLOCK_IQ3_S_BYTES;
            let gw = &w.gate[go..go + nb * BLOCK_IQ3_S_BYTES];
            let uw = &w.up[go..go + nb * BLOCK_IQ3_S_BYTES];
            let g = cpu_dot_iq3_s_q8_k(nb, gw, &xq_h);
            let u = cpu_dot_iq3_s_q8_k(nb, uw, &xq_h);
            let out = swiglu_ref(g, u, ew_h[s], clamp);
            clamped.add(g, u, clamp, out);
            want[s * n_rows + row] = out;
            let g64 = dot_dequant_f64(nb, gw, &xq_h);
            let u64_ = dot_dequant_f64(nb, uw, &xq_h);
            max_int_vs_f64 = max_int_vs_f64
                .max(((g as f64 - g64) / g64.abs().max(1e-3)).abs())
                .max(((u as f64 - u64_) / u64_.abs().max(1e-3)).abs());
            want64[s * n_rows + row] = swiglu_ref(g64 as f32, u64_ as f32, ew_h[s], clamp);
        }
    }
    eprintln!(
        "[{}] cpu int-dot vs f64 dequant-dot: max rel {max_int_vs_f64:.2e}; {}",
        w.label,
        clamped.report()
    );

    let mut mid_d: DeviceBuffer<f32> = DeviceBuffer::new(c.dev.id, n_used * n_rows)?;
    c.k.launch_fused_swiglu_batch(
        &c.stream,
        &mut mid_d,
        &c.gate_d,
        &c.up_d,
        &xq_d,
        &ew_d,
        &sel_d,
        bpe as u32,
        bpe as u32,
        n_used as u32,
        clamp,
        n_rows as u32,
        nb as u32,
    )?;
    c.stream.synchronize()?;
    let mut got = vec![0f32; n_used * n_rows];
    mid_d.copy_to_host(&mut got)?;
    check(
        &format!("[{}] iq3_s fused batch (B=1) vs int ref", w.label),
        &got,
        &want,
        clamped.unclamped_max_ref,
        TOL,
    )?;
    check(
        &format!("[{}] iq3_s fused batch (B=1) vs f64 ref", w.label),
        &got,
        &want64,
        clamped.unclamped_max_ref,
        TOL,
    )?;

    // hetsplit identity. M63: the kernels read the miss branch as
    // -(iGPU slot + 1), not a bare -1 — go through the real encoder
    // (packed=false => iGPU slot == expert id).
    let mut remap_h = vec![-1i32; 256];
    remap_h[hot[0]] = 0;
    remap_h[hot[1]] = 1;
    v4flash_kernels::het::weights::encode_igpu_remap(&mut remap_h, false);
    let remap_d = upload(&c.dev, &remap_h)?;
    let mut hot_g = vec![0u8; 2 * bpe];
    let mut hot_u = vec![0u8; 2 * bpe];
    for (i, &e) in hot.iter().enumerate() {
        hot_g[i * bpe..(i + 1) * bpe].copy_from_slice(&w.gate[e * bpe..(e + 1) * bpe]);
        hot_u[i * bpe..(i + 1) * bpe].copy_from_slice(&w.up[e * bpe..(e + 1) * bpe]);
    }
    let hot_g_d = upload(&c.dev, &hot_g)?;
    let hot_u_d = upload(&c.dev, &hot_u)?;
    let mut m0: DeviceBuffer<f32> = DeviceBuffer::new(c.dev.id, n_used * n_rows)?;
    let mut m1: DeviceBuffer<f32> = DeviceBuffer::new(c.dev.id, n_used * n_rows)?;
    c.k.launch_fused_swiglu_batch_hetsplit(
        &c.stream,
        &mut m0,
        &c.gate_d,
        &c.up_d,
        &xq_d,
        &ew_d,
        &sel_d,
        &remap_d,
        0,
        2,
        bpe as u32,
        bpe as u32,
        n_used as u32,
        clamp,
        n_rows as u32,
        nb as u32,
    )?;
    // Mode 1 (dGPU side) reads the packed 2-expert hot buffer via `dense`.
    c.k.launch_fused_swiglu_batch_hetsplit(
        &c.stream,
        &mut m1,
        &hot_g_d,
        &hot_u_d,
        &xq_d,
        &ew_d,
        &sel_d,
        &remap_d,
        1,
        2,
        bpe as u32,
        bpe as u32,
        n_used as u32,
        clamp,
        n_rows as u32,
        nb as u32,
    )?;
    c.stream.synchronize()?;
    let mut g0 = vec![0f32; n_used * n_rows];
    let mut g1 = vec![0f32; n_used * n_rows];
    m0.copy_to_host(&mut g0)?;
    m1.copy_to_host(&mut g1)?;
    // Each side must have produced exact zeros for the other's slots. The
    // dGPU side takes a slot iff its expert is resident AND fewer than
    // `dgpu_cap` resident slots precede it (kernel `res_rank`); an over-cap
    // resident slot stays on the iGPU side at its raw expert id (9828fa7).
    // With 5 real experts `sel` carries three resident slots against cap 2,
    // so this exercises the over-cap path too.
    let dgpu_cap = 2usize;
    let mut res_rank = 0usize;
    let mut takes_dgpu = [false; N_EXPERT_USED as usize];
    for (s, &e) in sel.iter().enumerate() {
        let resident = hot.contains(&(e as usize));
        takes_dgpu[s] = resident && res_rank < dgpu_cap;
        if resident {
            res_rank += 1;
        }
    }
    eprintln!(
        "[{}] hetsplit: sel={sel:?} hot={hot:?} cap={dgpu_cap} -> mode1 slots {:?}",
        w.label,
        takes_dgpu
            .iter()
            .enumerate()
            .filter(|(_, &t)| t)
            .map(|(s, _)| s)
            .collect::<Vec<_>>()
    );
    let mut n_hot_rows = 0usize;
    for s in 0..sel.len() {
        let is_hot = takes_dgpu[s];
        let (mine, theirs) = if is_hot { (&g1, &g0) } else { (&g0, &g1) };
        for row in 0..n_rows {
            if theirs[s * n_rows + row] != 0.0 {
                return Err(eyre!("hetsplit: slot {s} written by the wrong side"));
            }
            if is_hot && mine[s * n_rows + row] == 0.0 && want[s * n_rows + row] != 0.0 {
                return Err(eyre!("hetsplit: hot slot {s} row {row} not written"));
            }
        }
        if is_hot {
            n_hot_rows += n_rows;
        }
    }
    eprintln!(
        "[{}] hetsplit: {} hot (mode 1) outputs, {} cold (mode 0)",
        w.label,
        n_hot_rows,
        n_used * n_rows - n_hot_rows
    );
    let sum: Vec<f32> = g0.iter().zip(&g1).map(|(a, b)| a + b).collect();
    check(
        &format!("[{}] iq3_s hetsplit m0+m1", w.label),
        &sum,
        &want,
        clamped.unclamped_max_ref,
        TOL,
    )?;
    Ok(())
}

/// chunked + kwide prefill at one batch shape. `experts` = (expert id,
/// member count); members are spread over tokens 0..b and slots.
fn run_prefill(
    c: &Ctx,
    w: &Weights,
    rng: &mut Lcg,
    b: usize,
    chunk: u32,
    experts: &[(usize, i32)],
) -> eyre::Result<()> {
    let n_used = N_EXPERT_USED as usize;
    let (n_rows, nb, bpe, stride, clamp) = (c.n_rows, c.nb, c.bpe, c.stride, c.clamp);
    let max_per_expert = b.max(experts.iter().map(|&(_, n)| n as usize).max().unwrap_or(0));
    let mut xq2 = vec![0u8; b * stride];
    gen_xq(rng, &mut xq2, b, stride, nb);
    let ew2: Vec<f32> = (0..b * n_used)
        .map(|i| 0.05 + 0.01 * (i % 17) as f32)
        .collect();
    let mut gc_h = vec![0i32; w.n_e];
    let mut em_h = vec![0i32; w.n_e * max_per_expert];
    let mut wi_h: Vec<i32> = Vec::new();
    let mut touched = Vec::new();
    let mut np = 0usize;
    for &(e, n) in experts {
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
    let xq2_d = upload(&c.dev, &xq2)?;
    let ew2_d = upload(&c.dev, &ew2)?;
    let gc_d = upload(&c.dev, &gc_h)?;
    let em_d = upload(&c.dev, &em_h)?;
    let wi_d = upload(&c.dev, &wi_h)?;

    let mut want_t = Vec::with_capacity(touched.len() * n_rows);
    let mut clamped = ClampStats::default();
    for &(e, bi, sl) in &touched {
        let xq_s = &xq2[bi * stride..(bi + 1) * stride];
        for row in 0..n_rows {
            let go = e * bpe + row * nb * BLOCK_IQ3_S_BYTES;
            let g = cpu_dot_iq3_s_q8_k(nb, &w.gate[go..go + nb * BLOCK_IQ3_S_BYTES], xq_s);
            let u = cpu_dot_iq3_s_q8_k(nb, &w.up[go..go + nb * BLOCK_IQ3_S_BYTES], xq_s);
            let out = swiglu_ref(g, u, ew2[bi * n_used + sl], clamp);
            clamped.add(g, u, clamp, out);
            want_t.push(out);
        }
    }
    eprintln!("[{}] prefill B={b}: {}", w.label, clamped.report());

    let mut mid2_d: DeviceBuffer<f32> = DeviceBuffer::new(c.dev.id, b * n_used * n_rows)?;
    let mut mid2 = vec![0f32; b * n_used * n_rows];
    let shape = format!(
        "B={b} chunk={chunk} items={} members={}",
        wi_h.len(),
        touched.len()
    );
    for kwide in [false, true] {
        mid2_d.fill_zero()?;
        if kwide {
            c.k.launch_fused_swiglu_kwide(
                &c.stream,
                &mut mid2_d,
                &c.gate_d,
                &c.up_d,
                &xq2_d,
                &ew2_d,
                &gc_d,
                &em_d,
                &wi_d,
                wi_h.len() as u32,
                bpe as u32,
                bpe as u32,
                n_used as u32,
                max_per_expert as u32,
                chunk,
                clamp,
                n_rows as u32,
                nb as u32,
            )?;
        } else {
            c.k.launch_fused_swiglu_chunked(
                &c.stream,
                &mut mid2_d,
                &c.gate_d,
                &c.up_d,
                &xq2_d,
                &ew2_d,
                &gc_d,
                &em_d,
                &wi_d,
                wi_h.len() as u32,
                bpe as u32,
                bpe as u32,
                n_used as u32,
                max_per_expert as u32,
                chunk,
                clamp,
                n_rows as u32,
                nb as u32,
            )?;
        }
        c.stream.synchronize()?;
        mid2_d.copy_to_host(&mut mid2)?;
        let mut got_t = Vec::with_capacity(want_t.len());
        for &(_, bi, sl) in &touched {
            for row in 0..n_rows {
                got_t.push(mid2[(bi * n_used + sl) * n_rows + row]);
            }
        }
        // Untouched (token, slot) pairs must stay zero (caller pre-zeroes).
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
        check(
            &format!("[{}] iq3_s {name} {shape}", w.label),
            &got_t,
            &want_t,
            clamped.unclamped_max_ref,
            TOL,
        )?;
    }
    Ok(())
}

fn run_all(arch_prefix: &str, w: Weights, seed: u64) -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_device(arch_prefix)?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    eprintln!("=== {} on device {} ({arch}) ===", w.label, dev.id);
    let stream = Stream::new(dev.id)?;
    let k = Iq3SPairMatvec::for_arch(&arch)?;
    let n_rows = N_FF_EXP as usize; // gate/up out dim = 2048
    let nb = BLOCKS_Q8K_GATE_IN as usize; // K = 4096 → 16 super-blocks
    let bpe = n_rows * nb * BLOCK_IQ3_S_BYTES;
    assert_eq!(bpe, 3_604_480);
    let stride = nb * BLOCK_Q8_K_BYTES;
    let gate_d = upload(&dev, &w.gate)?;
    let up_d = upload(&dev, &w.up)?;
    eprintln!(
        "device working set: {:.1} MB weights",
        2.0 * w.gate.len() as f64 / 1e6
    );
    let c = Ctx {
        dev,
        stream,
        k,
        n_rows,
        nb,
        bpe,
        stride,
        clamp: 10.0,
        gate_d,
        up_d,
    };
    let mut rng = Lcg::new(seed);

    let n_e = w.n_e;
    // Decode: 6 slots over 5 distinct experts (one duplicate), two "hot".
    let pick = |i: usize| (i % n_e) as i32;
    let sel = [
        pick(1),
        pick(n_e - 2),
        pick(3),
        pick(1),
        pick(n_e - 1),
        pick(2),
    ];
    let hot = [sel[1] as usize, sel[4] as usize];
    run_decode(&c, &w, &mut rng, sel, hot)?;

    // Prefill shapes: B=7 (single partial chunk), B=40 chunk 16 (full +
    // partial), B=64 chunk 32 (= IQ3S_KW_MAX_CHUNK; 33 members → 2 items).
    run_prefill(&c, &w, &mut rng, 7, 16, &[(1 % n_e, 7), (3 % n_e, 4)])?;
    run_prefill(&c, &w, &mut rng, 40, 16, &[(0, 16), (n_e - 1, 11)])?;
    run_prefill(&c, &w, &mut rng, 64, 32, &[(2 % n_e, 33), (4 % n_e, 19)])?;
    c.stream.synchronize()?;
    drop(c);
    Ok(())
}

#[test]
#[ignore]
fn iq3_s_synthetic_igpu() -> eyre::Result<()> {
    let w = synthetic_weights(0x1a35, 8, N_FF_EXP as usize, BLOCKS_Q8K_GATE_IN as usize);
    run_all("gfx1151", w, 0x51)
}

#[test]
#[ignore]
fn iq3_s_synthetic_dgpu() -> eyre::Result<()> {
    let w = synthetic_weights(0x1a36, 8, N_FF_EXP as usize, BLOCKS_Q8K_GATE_IN as usize);
    run_all("gfx1201", w, 0x52)
}

#[test]
#[ignore]
fn iq3_s_real_igpu() -> eyre::Result<()> {
    match real_weights(N_FF_EXP as usize, BLOCKS_Q8K_GATE_IN as usize)? {
        Some(w) => run_all("gfx1151", w, 0x53),
        None => Ok(()),
    }
}

#[test]
#[ignore]
fn iq3_s_real_dgpu() -> eyre::Result<()> {
    match real_weights(N_FF_EXP as usize, BLOCKS_Q8K_GATE_IN as usize)? {
        Some(w) => run_all("gfx1201", w, 0x54),
        None => Ok(()),
    }
}
