//! Laguna-S-2.1 CORRECTNESS SPIKE — throwaway forward pass that composes the
//! deepstrix kernels and checks parity against the llama.cpp oracle
//! (`/home/claude-code/laguna-oracle/out/hidden.txt`) layer-by-layer.
//!
//! Hack freely: dims hardcoded, single stream, tons of host<->device copies,
//! host-side elementwise ops (norm/rope/softplus/router/swiglu) so the only
//! GPU kernels exercised are the LOAD-BEARING ones we're porting:
//!   - f16 matvec (attn q/k/v/o/gate)
//!   - Q4_K dense matvec (dense FFN gate/up, expert gate/up, shared gate/up)
//!   - Q6_K dense matvec (dense/expert/shared down, LM head)
//!   - GQA single-query attention
//!
//! Run (server must be stopped; GPU free):
//!   nix develop --command cargo test --release -p v4flash-kernels \
//!       --test laguna_spike -- --ignored --nocapture
//!
//! Prompt "The quick brown fox" -> token ids [2, 785, 3454, 21438, 42850].

use std::fs::File;
use std::os::unix::fs::FileExt;

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::{GgufType, GgufValue};
use v4flash_core::MappedGguf;
use v4flash_hip::{Device, DeviceBuffer, Stream};
use v4flash_kernels::iq2_xxs_tables::f16_to_f32;
use v4flash_kernels::{F16Matvec, GqaAttention, Q4_KDenseMatvec, Q6_KDenseMatvec};

const GGUF_PATH: &str = "/persist/lumi/models/laguna-s-2.1-int4/laguna-s-2.1-Q4_K_M.gguf";

const HIDDEN: usize = 3072;
const HEAD_DIM: usize = 128;
const N_KV_HEAD: usize = 8;
const N_LAYER: usize = 48;
const N_EXPERT: usize = 256;
const TOPK: usize = 10;
const FF_EXP: usize = 1024;
const FF_SHEXP: usize = 1024;
const FF_DENSE: usize = 12288;
const EPS: f32 = 1e-6;

// oracle per-tensor sum checksums (prompt "The quick brown fox", 5 tokens)
// layer 0
const O_EMBD: f32 = 24.273634;
const O_ATTN_NORM0: f32 = 5.083115;
const O_QCUR0: f32 = -36.165260;
const O_QNORMED0: f32 = -735.040527;
const O_QROPE0: f32 = -871.135620;
const O_KCUR0: f32 = -4.511584;
const O_KNORMED0: f32 = -97.719940;
const O_KROPE0: f32 = -120.663544;
const O_VCUR0: f32 = 5.482293;
const O_ATTNOUT0: f32 = 46.179028;
const O_GATEPROJ0: f32 = -338.479492;
const O_SOFTPLUS0: f32 = 58.180592;
const O_ATTNGATED0: f32 = 13.291483;
const O_OPROJ0: f32 = -3.890045;
const O_FFNINP0: f32 = 20.383478;
const O_FFNNORM0: f32 = 0.690120;
const O_FFNGATE0: f32 = -210.977798;
const O_FFNUP0: f32 = 1.129863;
const O_FFNSWIGLU0: f32 = -0.195828;
const O_FFNOUT0: f32 = 0.072774;
const O_LOUT0: f32 = 20.456112;
// layer 1 (first MoE layer)
const O_LOUT1_FFNOUT: f32 = -0.718842; // ffn_out-1 (moe_out + shexp)
const O_MOEOUT1: f32 = 0.479422;
const O_SHEXP1: f32 = -1.198263;
const O_WSCALED1: f32 = 12.499999;
// final
const O_RESULT_NORM: f32 = -14.071781;
const O_RESULT_OUTPUT: f32 = 267398.468750;

fn rel(a: f32, b: f32) -> f32 {
    let d = (a - b).abs();
    let m = a.abs().max(b.abs()).max(1e-6);
    d / m
}

fn check(name: &str, got: f32, want: f32) {
    let r = rel(got, want);
    let flag = if r < 2e-2 { "OK " } else { "!!!" };
    println!("  [{flag}] {name:<24} got={got:>16.5}  want={want:>16.5}  rel={r:.3e}");
}

// ---------------------------------------------------------------------------
// f16 encode (from tests/q4_k_dense_matvec.rs)
// ---------------------------------------------------------------------------
fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x007f_ffff;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    } else if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half_mant = (m >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let mut result = sign | half_mant;
        if (m & round_bit) != 0 && ((m & (round_bit - 1)) != 0 || (half_mant & 1) != 0) {
            result += 1;
        }
        return result;
    }
    let half_mant = (mant >> 13) as u16;
    let mut result = sign | ((e as u16) << 10) | half_mant;
    if (mant & 0x0000_1000) != 0 && ((mant & 0x0000_0fff) != 0 || (half_mant & 1) != 0) {
        result += 1;
    }
    result
}

// ---------------------------------------------------------------------------
// Q4_K CPU dequant (for token embedding rows) — from tests/q4_k_dense_matvec.rs
// ---------------------------------------------------------------------------
fn get_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q4k_superblock(blk: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
    let scales = &blk[4..16];
    let qs = &blk[16..144];
    for g in 0..4 {
        let (sc1, m1) = get_scale_min(2 * g, scales);
        let (sc2, m2) = get_scale_min(2 * g + 1, scales);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;
        for l in 0..32 {
            let byte = qs[32 * g + l];
            out[64 * g + l] = d1 * (byte & 0x0F) as f32 - min1;
            out[64 * g + 32 + l] = d2 * (byte >> 4) as f32 - min2;
        }
    }
}

// ---------------------------------------------------------------------------
// host elementwise
// ---------------------------------------------------------------------------
fn rmsnorm_weighted(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mss + eps).sqrt();
    x.iter().zip(w).map(|(v, wi)| v * inv * wi).collect()
}

fn rmsnorm_head(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    // x is [n_head*128]; normalize each 128-block, multiply by shared w[128].
    let mut out = vec![0f32; x.len()];
    for h in 0..(x.len() / HEAD_DIM) {
        let seg = &x[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let mss: f32 = seg.iter().map(|v| v * v).sum::<f32>() / HEAD_DIM as f32;
        let inv = 1.0 / (mss + eps).sqrt();
        for d in 0..HEAD_DIM {
            out[h * HEAD_DIM + d] = seg[d] * inv * w[d];
        }
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    // log(1+exp(x)) numerically stable
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// RoPE (NEOX, partial rotary, YaRN) — host, matching ggml rope_yarn
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct RopeCfg {
    n_rot: usize,
    freq_base: f32,
    freq_scale: f32, // 1/factor for full, 1.0 for swa
    ext_factor: f32, // 1.0 full, 0.0 swa
    mscale: f32,     // net yarn_attn_factor
    corr_low: f32,
    corr_high: f32,
}

fn corr_dim(n_dims: f32, n_ctx_orig: f32, n_rot_arg: f32, base: f32) -> f32 {
    n_dims * (n_ctx_orig / (n_rot_arg * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
}

fn make_corr_dims(n_rot: usize, n_ctx_orig: f32, base: f32, beta_fast: f32, beta_slow: f32) -> (f32, f32) {
    let start = corr_dim(n_rot as f32, n_ctx_orig, beta_fast, base).floor();
    let end = corr_dim(n_rot as f32, n_ctx_orig, beta_slow, base).ceil();
    (start.max(0.0), end.min((n_rot - 1) as f32))
}

fn rope_yarn_ramp(low: f32, high: f32, ic: usize) -> f32 {
    let y = (ic as f32 - low) / (high - low).max(0.001);
    1.0 - y.min(1.0).max(0.0)
}

/// Apply NEOX partial-rotary RoPE in place to a [n_head*128] vector at `pos`.
fn apply_rope(x: &mut [f32], n_head: usize, pos: usize, c: &RopeCfg) {
    let n_rot = c.n_rot;
    let half = n_rot / 2;
    let theta_scale = c.freq_base.powf(-2.0 / n_rot as f32);
    for h in 0..n_head {
        let base = h * HEAD_DIM;
        for ic in 0..half {
            let theta_extrap = pos as f32 * theta_scale.powi(ic as i32);
            let theta_interp = c.freq_scale * theta_extrap;
            let theta = if c.ext_factor != 0.0 {
                let ramp = rope_yarn_ramp(c.corr_low, c.corr_high, ic) * c.ext_factor;
                theta_interp * (1.0 - ramp) + theta_extrap * ramp
            } else {
                theta_interp
            };
            let cos = theta.cos() * c.mscale;
            let sin = theta.sin() * c.mscale;
            let x0 = x[base + ic];
            let x1 = x[base + ic + half];
            x[base + ic] = x0 * cos - x1 * sin;
            x[base + ic + half] = x0 * sin + x1 * cos;
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF metadata helpers
// ---------------------------------------------------------------------------
fn meta_f32(g: &v4flash_core::gguf::Gguf, key: &str) -> Option<f32> {
    match g.metadata(key)? {
        GgufValue::F32(v) => Some(*v),
        GgufValue::F64(v) => Some(*v as f32),
        GgufValue::U32(v) => Some(*v as f32),
        GgufValue::I32(v) => Some(*v as f32),
        _ => None,
    }
}

fn load_f32_tensor(gguf: &MappedGguf, name: &str) -> eyre::Result<Vec<f32>> {
    let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
    if t.dtype != GgufType::F32 {
        return Err(eyre!("{name} not F32 (is {:?})", t.dtype));
    }
    let bytes = gguf.read_tensor(t)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ---------------------------------------------------------------------------
// device matvec helpers (fresh buffers each call — correctness first)
// ---------------------------------------------------------------------------
struct Engine {
    dev: i32,
    stream: Stream,
    f16: F16Matvec,
    q4: Q4_KDenseMatvec,
    q6: Q6_KDenseMatvec,
    gqa: GqaAttention,
}

impl Engine {
    fn upload_u8(&self, bytes: &[u8]) -> eyre::Result<DeviceBuffer<u8>> {
        let mut b = DeviceBuffer::<u8>::new(self.dev, bytes.len())?;
        b.copy_from_host(bytes)?;
        Ok(b)
    }
    fn upload_f32(&self, x: &[f32]) -> eyre::Result<DeviceBuffer<f32>> {
        let mut b = DeviceBuffer::<f32>::new(self.dev, x.len())?;
        b.copy_from_host(x)?;
        Ok(b)
    }
    fn download(&self, b: &DeviceBuffer<f32>, n: usize) -> eyre::Result<Vec<f32>> {
        let mut v = vec![0f32; n];
        b.copy_to_host(&mut v)?;
        Ok(v)
    }

    /// f16 weight [n_rows, k] (raw f16 bytes on device) times host x[k].
    fn f16_matvec(&self, w: &DeviceBuffer<u8>, x: &[f32], n_rows: usize, k: usize) -> eyre::Result<Vec<f32>> {
        let xd = self.upload_f32(x)?;
        let mut out = DeviceBuffer::<f32>::new(self.dev, n_rows)?;
        self.f16.matvec(&self.stream, &mut out, w, &xd, n_rows as u32, k as u32)?;
        self.stream.synchronize()?;
        self.download(&out, n_rows)
    }
    fn q4_matvec(&self, wbytes: &[u8], x: &[f32], n_rows: usize, k: usize) -> eyre::Result<Vec<f32>> {
        let w = self.upload_u8(wbytes)?;
        let xd = self.upload_f32(x)?;
        let mut out = DeviceBuffer::<f32>::new(self.dev, n_rows)?;
        self.q4.matvec(&self.stream, &mut out, &w, &xd, n_rows as u32, k as u32)?;
        self.stream.synchronize()?;
        self.download(&out, n_rows)
    }
    fn q6_matvec(&self, wbytes: &[u8], x: &[f32], n_rows: usize, k: usize) -> eyre::Result<Vec<f32>> {
        let w = self.upload_u8(wbytes)?;
        let xd = self.upload_f32(x)?;
        let mut out = DeviceBuffer::<f32>::new(self.dev, n_rows)?;
        self.q6.matvec(&self.stream, &mut out, &w, &xd, n_rows as u32, k as u32)?;
        self.stream.synchronize()?;
        self.download(&out, n_rows)
    }
    /// dtype-dispatched dense matvec (the Q4_K_M quant map is NOT uniform).
    fn qmatvec(&self, dt: GgufType, wbytes: &[u8], x: &[f32], n_rows: usize, k: usize) -> eyre::Result<Vec<f32>> {
        match dt {
            GgufType::Q4_K => self.q4_matvec(wbytes, x, n_rows, k),
            GgufType::Q6_K => self.q6_matvec(wbytes, x, n_rows, k),
            other => Err(eyre!("qmatvec: unsupported dtype {other:?}")),
        }
    }
}

/// block bytes for the supported quant dtypes.
fn block_bytes(dt: GgufType) -> usize {
    match dt {
        GgufType::Q4_K => 144,
        GgufType::Q6_K => 210,
        _ => 0,
    }
}

fn sum(v: &[f32]) -> f32 {
    v.iter().copied().sum()
}
fn sum2(vs: &[Vec<f32>]) -> f32 {
    vs.iter().map(|v| sum(v)).sum()
}

#[test]
#[ignore = "drives the GPU + needs the 75GB Laguna GGUF; run explicitly"]
fn laguna_forward_spike() -> eyre::Result<()> {
    v4flash_hip::install_panic_handler();

    // ----- device -----
    let dev = Device::all()?
        .into_iter()
        .find(|d| {
            d.properties()
                .map(|p| p.gcn_arch_name.starts_with("gfx1201"))
                .unwrap_or(false)
        })
        .ok_or_else(|| eyre!("no gfx1201 device"))?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    println!("device id={} arch={arch}", dev.id);
    let stream = Stream::new(dev.id)?;
    let eng = Engine {
        dev: dev.id,
        stream,
        f16: F16Matvec::for_arch(&arch)?,
        q4: Q4_KDenseMatvec::for_arch(&arch)?,
        q6: Q6_KDenseMatvec::for_arch(&arch)?,
        gqa: GqaAttention::for_arch(&arch)?,
    };

    // ----- gguf + hparams -----
    let gguf = MappedGguf::open(GGUF_PATH)?;
    let raw_file = File::open(GGUF_PATH)?;
    let g = gguf.gguf();
    let factor = meta_f32(g, "laguna.rope.scaling.factor").unwrap_or(32.0);
    let orig_ctx = meta_f32(g, "laguna.rope.scaling.original_context_length").unwrap_or(8192.0);
    let yarn_attn = meta_f32(g, "laguna.rope.scaling.yarn_attn_factor").unwrap_or(1.0);
    let beta_fast = meta_f32(g, "laguna.rope.scaling.yarn_beta_fast").unwrap_or(32.0);
    let beta_slow = meta_f32(g, "laguna.rope.scaling.yarn_beta_slow").unwrap_or(1.0);
    let freq_base = meta_f32(g, "laguna.rope.freq_base").unwrap_or(500000.0);
    let freq_base_swa = meta_f32(g, "laguna.rope.freq_base_swa").unwrap_or(10000.0);
    let n_rot_full = meta_f32(g, "laguna.rope.dimension_count").unwrap_or(64.0) as usize;
    let n_rot_swa = meta_f32(g, "laguna.rope.dimension_count_swa").unwrap_or(128.0) as usize;
    let scale_scale = meta_f32(g, "laguna.expert_weights_scale").unwrap_or(2.5);
    println!(
        "hparams: factor={factor} orig_ctx={orig_ctx} yarn_attn={yarn_attn} beta_fast={beta_fast} beta_slow={beta_slow}\n         freq_base={freq_base} freq_base_swa={freq_base_swa} n_rot_full={n_rot_full} n_rot_swa={n_rot_swa} moe_scale={scale_scale}"
    );

    // full-layer YaRN rope cfg (mscale = net yarn_attn_factor; framework cancels ggml internal amp)
    let (corr_low, corr_high) = make_corr_dims(n_rot_full, orig_ctx, freq_base, beta_fast, beta_slow);
    // EMPIRICAL (from oracle pos-0 = pure scale 1.346): the YaRN mscale
    // amplification (1 + 0.1*ln(1/freq_scale)) IS applied — the laguna.cpp
    // "framework pre-divides yarn_attn_factor to cancel it" comment does NOT
    // hold for this GGUF. mscale = yarn_attn_factor * (1 + 0.1*ln(factor)).
    let mscale_full = yarn_attn * (1.0 + 0.1 * factor.ln());
    println!("rope full mscale = {mscale_full}  (corr_dims {corr_low}..{corr_high})");
    let rope_full = RopeCfg {
        n_rot: n_rot_full,
        freq_base,
        freq_scale: 1.0 / factor,
        ext_factor: 1.0,
        mscale: mscale_full,
        corr_low,
        corr_high,
    };
    let rope_swa = RopeCfg {
        n_rot: n_rot_swa,
        freq_base: freq_base_swa,
        freq_scale: 1.0,
        ext_factor: 0.0,
        mscale: 1.0,
        corr_low: 0.0,
        corr_high: 0.0,
    };

    // ----- tokens -----
    let tokens: [usize; 5] = [2, 785, 3454, 21438, 42850];
    let n_tok = tokens.len();

    // ----- embedding (host Q4_K dequant of token_embd rows) -----
    let tok_embd_t = g.tensor("token_embd.weight").ok_or_else(|| eyre!("no token_embd"))?;
    let row_bytes = (HIDDEN / 256) * 144; // 1728
    let mut hidden: Vec<Vec<f32>> = Vec::new();
    for &tid in &tokens {
        let mut rb = vec![0u8; row_bytes];
        raw_file.read_exact_at(&mut rb, tok_embd_t.abs_offset + (tid as u64) * row_bytes as u64)?;
        let mut row = vec![0f32; HIDDEN];
        for sb in 0..(HIDDEN / 256) {
            dequant_q4k_superblock(&rb[sb * 144..(sb + 1) * 144], &mut row[sb * 256..(sb + 1) * 256]);
        }
        hidden.push(row);
    }
    println!("\n=== MILESTONE 1: embedding ===");
    check("embd", sum2(&hidden), O_EMBD);

    // scratch for MoE expert weights read straight from file
    let ge_t = |i: usize, name: &str| -> String { format!("blk.{i}.{name}") };

    // ----- layers -----
    for il in 0..N_LAYER {
        let is_full = il % 4 == 0;
        let n_head = if is_full { 48 } else { 72 };
        let n_embd_q = n_head * HEAD_DIM;
        let rope = if is_full { &rope_full } else { &rope_swa };
        let n_rot = rope.n_rot;

        // load per-layer weights
        let attn_norm = load_f32_tensor(&gguf, &format!("blk.{il}.attn_norm.weight"))?;
        let ffn_norm = load_f32_tensor(&gguf, &format!("blk.{il}.ffn_norm.weight"))?;
        let q_norm = load_f32_tensor(&gguf, &format!("blk.{il}.attn_q_norm.weight"))?;
        let k_norm = load_f32_tensor(&gguf, &format!("blk.{il}.attn_k_norm.weight"))?;
        let wq = load_w(&gguf, &format!("blk.{il}.attn_q.weight"), &eng)?;
        let wk = load_w(&gguf, &format!("blk.{il}.attn_k.weight"), &eng)?;
        let wv = load_w(&gguf, &format!("blk.{il}.attn_v.weight"), &eng)?;
        let wo = load_w(&gguf, &format!("blk.{il}.attn_output.weight"), &eng)?;
        let wg = load_w(&gguf, &format!("blk.{il}.attn_gate.weight"), &eng)?;

        // per-position attention pre-compute
        let mut attn_input: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
        let mut q_rope: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
        let mut k_rope: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
        let mut v_all: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
        // diagnostics
        let (mut s_qcur, mut s_qn, mut s_qr, mut s_kcur, mut s_kn, mut s_kr, mut s_vcur) =
            (0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
        for t in 0..n_tok {
            let ain = rmsnorm_weighted(&hidden[t], &attn_norm, EPS);
            let q = eng.f16_matvec(&wq, &ain, n_embd_q, HIDDEN)?;
            let k = eng.f16_matvec(&wk, &ain, N_KV_HEAD * HEAD_DIM, HIDDEN)?;
            let v = eng.f16_matvec(&wv, &ain, N_KV_HEAD * HEAD_DIM, HIDDEN)?;
            s_qcur += sum(&q);
            s_kcur += sum(&k);
            s_vcur += sum(&v);
            let mut qn = rmsnorm_head(&q, &q_norm, EPS);
            let mut kn = rmsnorm_head(&k, &k_norm, EPS);
            s_qn += sum(&qn);
            s_kn += sum(&kn);
            apply_rope(&mut qn, n_head, t, rope);
            apply_rope(&mut kn, N_KV_HEAD, t, rope);
            s_qr += sum(&qn);
            s_kr += sum(&kn);
            attn_input.push(ain);
            q_rope.push(qn);
            k_rope.push(kn);
            v_all.push(v);
        }

        // GQA attention per position over causal history (window irrelevant, seq<512)
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn_out: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
        for t in 0..n_tok {
            let n_kv = t + 1;
            // build k/v cache [n_kv, 8, 128] f16
            let mut kc = vec![0u16; n_kv * N_KV_HEAD * HEAD_DIM];
            let mut vc = vec![0u16; n_kv * N_KV_HEAD * HEAD_DIM];
            for j in 0..n_kv {
                for e in 0..(N_KV_HEAD * HEAD_DIM) {
                    kc[j * N_KV_HEAD * HEAD_DIM + e] = f32_to_f16(k_rope[j][e]);
                    vc[j * N_KV_HEAD * HEAD_DIM + e] = f32_to_f16(v_all[j][e]);
                }
            }
            let qf: Vec<u16> = q_rope[t].iter().map(|&x| f32_to_f16(x)).collect();
            let mut qd = DeviceBuffer::<u16>::new(eng.dev, qf.len())?;
            qd.copy_from_host(&qf)?;
            let mut kd = DeviceBuffer::<u16>::new(eng.dev, kc.len())?;
            kd.copy_from_host(&kc)?;
            let mut vd = DeviceBuffer::<u16>::new(eng.dev, vc.len())?;
            vd.copy_from_host(&vc)?;
            let mut od = DeviceBuffer::<f32>::new(eng.dev, n_head * HEAD_DIM)?;
            eng.gqa.single_query(
                &eng.stream,
                &mut od,
                &qd,
                &kd,
                &vd,
                n_head as u32,
                N_KV_HEAD as u32,
                HEAD_DIM as u32,
                n_kv as u32,
                scale,
            )?;
            eng.stream.synchronize()?;
            attn_out.push(eng.download(&od, n_head * HEAD_DIM)?);
        }

        // gate + o_proj + residual
        let (mut s_gp, mut s_sp, mut s_ag, mut s_op) = (0f32, 0f32, 0f32, 0f32);
        let mut ffn_inp: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
        for t in 0..n_tok {
            let gp = eng.f16_matvec(&wg, &attn_input[t], n_head, HIDDEN)?;
            s_gp += sum(&gp);
            let g_sp: Vec<f32> = gp.iter().map(|&x| softplus(x)).collect();
            s_sp += sum(&g_sp);
            let mut gated = vec![0f32; n_embd_q];
            for h in 0..n_head {
                for d in 0..HEAD_DIM {
                    gated[h * HEAD_DIM + d] = attn_out[t][h * HEAD_DIM + d] * g_sp[h];
                }
            }
            s_ag += sum(&gated);
            let op = eng.f16_matvec(&wo, &gated, HIDDEN, n_embd_q)?;
            s_op += sum(&op);
            let fi: Vec<f32> = op.iter().zip(&hidden[t]).map(|(a, b)| a + b).collect();
            ffn_inp.push(fi);
        }

        // FFN
        let mut ffn_out: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
        let mut moe_out_sum = 0f32;
        let mut shexp_sum = 0f32;
        let mut wscaled_sum = 0f32;
        let (mut s_fg, mut s_fu, mut s_fs) = (0f32, 0f32, 0f32);
        if il == 0 {
            let (wgate, dg) = load_bytes(&gguf, &format!("blk.{il}.ffn_gate.weight"))?;
            let (wup, du) = load_bytes(&gguf, &format!("blk.{il}.ffn_up.weight"))?;
            let (wdown, dd) = load_bytes(&gguf, &format!("blk.{il}.ffn_down.weight"))?;
            println!("L0 dense FFN dtypes: gate={dg:?} up={du:?} down={dd:?}");
            for t in 0..n_tok {
                let fn_in = rmsnorm_weighted(&ffn_inp[t], &ffn_norm, EPS);
                let gate = eng.qmatvec(dg, &wgate, &fn_in, FF_DENSE, HIDDEN)?;
                let up = eng.qmatvec(du, &wup, &fn_in, FF_DENSE, HIDDEN)?;
                s_fg += sum(&gate);
                s_fu += sum(&up);
                let sw: Vec<f32> = gate.iter().zip(&up).map(|(gg, uu)| silu(*gg) * uu).collect();
                s_fs += sum(&sw);
                let down = eng.qmatvec(dd, &wdown, &sw, HIDDEN, FF_DENSE)?;
                ffn_out.push(down);
            }
        } else {
            // MoE
            let router = load_f32_tensor(&gguf, &format!("blk.{il}.ffn_gate_inp.weight"))?; // [256,3072] row-major (ne0=3072)
            let bias = load_f32_tensor(&gguf, &format!("blk.{il}.exp_probs_b.bias"))?; // [256]
            let gate_exps_t = g.tensor(&ge_t(il, "ffn_gate_exps.weight")).unwrap();
            let up_exps_t = g.tensor(&ge_t(il, "ffn_up_exps.weight")).unwrap();
            let down_exps_t = g.tensor(&ge_t(il, "ffn_down_exps.weight")).unwrap();
            let (dge, due, dde) = (gate_exps_t.dtype, up_exps_t.dtype, down_exps_t.dtype);
            let gate_stride = FF_EXP * (HIDDEN / 256) * block_bytes(dge); // per-expert [1024 rows,3072 k]
            let up_stride = FF_EXP * (HIDDEN / 256) * block_bytes(due);
            let down_stride = HIDDEN * (FF_EXP / 256) * block_bytes(dde); // per-expert [3072 rows,1024 k]
            let (wgs, dgs) = load_bytes(&gguf, &format!("blk.{il}.ffn_gate_shexp.weight"))?;
            let (wus, dus) = load_bytes(&gguf, &format!("blk.{il}.ffn_up_shexp.weight"))?;
            let (wds, dds) = load_bytes(&gguf, &format!("blk.{il}.ffn_down_shexp.weight"))?;
            if il == 1 {
                println!("L1 MoE dtypes: gate_exps={dge:?} up_exps={due:?} down_exps={dde:?} | shexp g={dgs:?} u={dus:?} d={dds:?}");
            }

            for t in 0..n_tok {
                let fn_in = rmsnorm_weighted(&ffn_inp[t], &ffn_norm, EPS);
                // router (host)
                let mut logits = vec![0f32; N_EXPERT];
                for e in 0..N_EXPERT {
                    let mut acc = 0f32;
                    let wrow = &router[e * HIDDEN..(e + 1) * HIDDEN];
                    for k in 0..HIDDEN {
                        acc += wrow[k] * fn_in[k];
                    }
                    logits[e] = acc;
                }
                let probs: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();
                let biased: Vec<f32> = probs.iter().zip(&bias).map(|(p, b)| p + b).collect();
                // top-10 by biased
                let mut idx: Vec<usize> = (0..N_EXPERT).collect();
                idx.sort_by(|&a, &b| biased[b].partial_cmp(&biased[a]).unwrap());
                let sel = &idx[..TOPK];
                let sel_w: Vec<f32> = sel.iter().map(|&e| probs[e]).collect();
                let wsum: f32 = sel_w.iter().sum::<f32>().max(1e-20);
                let weights: Vec<f32> = sel_w.iter().map(|w| (w / wsum) * scale_scale).collect();
                wscaled_sum += weights.iter().sum::<f32>();

                let mut acc = vec![0f32; HIDDEN];
                for (si, &e) in sel.iter().enumerate() {
                    let mut gb = vec![0u8; gate_stride];
                    raw_file.read_exact_at(&mut gb, gate_exps_t.abs_offset + (e as u64) * gate_stride as u64)?;
                    let mut ub = vec![0u8; up_stride];
                    raw_file.read_exact_at(&mut ub, up_exps_t.abs_offset + (e as u64) * up_stride as u64)?;
                    let mut db = vec![0u8; down_stride];
                    raw_file.read_exact_at(&mut db, down_exps_t.abs_offset + (e as u64) * down_stride as u64)?;
                    let gate = eng.qmatvec(dge, &gb, &fn_in, FF_EXP, HIDDEN)?;
                    let up = eng.qmatvec(due, &ub, &fn_in, FF_EXP, HIDDEN)?;
                    let sw: Vec<f32> = gate.iter().zip(&up).map(|(gg, uu)| silu(*gg) * uu).collect();
                    let down = eng.qmatvec(dde, &db, &sw, HIDDEN, FF_EXP)?;
                    let w = weights[si];
                    for k in 0..HIDDEN {
                        acc[k] += down[k] * w;
                    }
                }
                moe_out_sum += sum(&acc);

                // shared expert
                let gate_s = eng.qmatvec(dgs, &wgs, &fn_in, FF_SHEXP, HIDDEN)?;
                let up_s = eng.qmatvec(dus, &wus, &fn_in, FF_SHEXP, HIDDEN)?;
                let sw_s: Vec<f32> = gate_s.iter().zip(&up_s).map(|(gg, uu)| silu(*gg) * uu).collect();
                let down_s = eng.qmatvec(dds, &wds, &sw_s, HIDDEN, FF_SHEXP)?;
                shexp_sum += sum(&down_s);
                let out: Vec<f32> = acc.iter().zip(&down_s).map(|(a, b)| a + b).collect();
                ffn_out.push(out);
            }
        }

        // residual
        let mut new_hidden = Vec::with_capacity(n_tok);
        for t in 0..n_tok {
            let lo: Vec<f32> = ffn_out[t].iter().zip(&ffn_inp[t]).map(|(a, b)| a + b).collect();
            new_hidden.push(lo);
        }

        // ----- parity reporting -----
        if il == 0 {
            println!("\n=== MILESTONE 2/3: layer 0 (attn + dense FFN) ===");
            check("attn_norm-0", sum2(&attn_input), O_ATTN_NORM0);
            check("Qcur-0", s_qcur, O_QCUR0);
            check("Qcur_normed-0", s_qn, O_QNORMED0);
            check("Qcur_rope-0", s_qr, O_QROPE0);
            check("Kcur-0", s_kcur, O_KCUR0);
            check("Kcur_normed-0", s_kn, O_KNORMED0);
            check("Kcur_rope-0", s_kr, O_KROPE0);
            check("Vcur-0", s_vcur, O_VCUR0);
            check("attn_out-0", sum2(&attn_out), O_ATTNOUT0);
            check("attn_gate_proj-0", s_gp, O_GATEPROJ0);
            check("attn_gate_softplus-0", s_sp, O_SOFTPLUS0);
            check("attn_gated-0", s_ag, O_ATTNGATED0);
            check("attn_o_proj-0", s_op, O_OPROJ0);
            check("ffn_inp-0", sum2(&ffn_inp), O_FFNINP0);
            check("ffn_gate-0", s_fg, O_FFNGATE0);
            check("ffn_up-0", s_fu, O_FFNUP0);
            check("ffn_swiglu-0", s_fs, O_FFNSWIGLU0);
            check("ffn_out-0", sum2(&ffn_out), O_FFNOUT0);
            check("l_out-0", sum2(&new_hidden), O_LOUT0);
        } else if il == 1 {
            println!("\n=== MILESTONE 4: layer 1 (attn + MoE FFN) ===");
            check("attn_out-1", sum2(&attn_out), 70.182571);
            check("ffn_moe_weights_scaled-1", wscaled_sum, O_WSCALED1);
            check("ffn_moe_out-1", moe_out_sum, O_MOEOUT1);
            check("ffn_shexp-1", shexp_sum, O_SHEXP1);
            check("ffn_out-1", sum2(&ffn_out), O_LOUT1_FFNOUT);
        }
        hidden = new_hidden;
        println!("layer {il:>2} done  (l_out sum = {:>14.5})", sum2(&hidden));
    }

    // ----- final norm + LM head (last position only) -----
    println!("\n=== MILESTONE 5: output norm + LM head ===");
    let output_norm = load_f32_tensor(&gguf, "output_norm.weight")?;
    let last = n_tok - 1;
    let rn = rmsnorm_weighted(&hidden[last], &output_norm, EPS);
    check("result_norm", sum(&rn), O_RESULT_NORM);

    let (lm_bytes, dlm) = load_bytes(&gguf, "output.weight")?;
    println!("LM head dtype = {dlm:?}");
    let logits = eng.qmatvec(dlm, &lm_bytes, &rn, 100352, HIDDEN)?;
    check("result_output", sum(&logits), O_RESULT_OUTPUT);

    // argmax = greedy next token
    let (argmax, maxv) = logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
    println!("\n>>> greedy next token id = {argmax}  (logit {maxv:.4})");
    println!(">>> top-5 logits:");
    let mut ord: Vec<usize> = (0..logits.len()).collect();
    ord.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    for &i in ord.iter().take(5) {
        println!("      tok {i:>6}  logit {:.4}", logits[i]);
    }

    Ok(())
}

/// Load an f16 weight tensor to device as raw bytes (must be F16 in the GGUF).
fn load_w(gguf: &MappedGguf, name: &str, eng: &Engine) -> eyre::Result<DeviceBuffer<u8>> {
    let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
    if t.dtype != GgufType::F16 {
        return Err(eyre!("{name} expected F16, got {:?}", t.dtype));
    }
    let bytes = gguf.read_tensor(t)?;
    eng.upload_u8(&bytes)
}

/// Load a quantized weight tensor's raw bytes (host) + its dtype.
fn load_bytes(gguf: &MappedGguf, name: &str) -> eyre::Result<(Vec<u8>, GgufType)> {
    let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
    Ok((gguf.read_tensor(t)?, t.dtype))
}
