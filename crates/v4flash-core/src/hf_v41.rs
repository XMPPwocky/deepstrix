//! DeepSeek-V4.1-Flash weights straight from the HF safetensors checkpoint,
//! presented under the engine's GGUF (llama.cpp `deepseek4`) tensor names and
//! storage formats.
//!
//! This is `scripts/v41_convert/to_gguf.py` moved into the loader: the same
//! per-tensor transforms run at load time instead of producing a second
//! ~350 GB copy of the weights on disk. The Python converter stays as the
//! byte-level oracle (`tests/hf_v41_fixture.rs` compares a GGUF it wrote
//! against this reader tensor by tensor).
//!
//! Per-role formats:
//!   routed experts   MXFP4 stacked `[n_expert, out, in]` in ggml's 17-byte
//!                    blocks (`e8m0 | 16 B: elems 0..15 low nibbles, 16..31
//!                    high`), repacked from HF's packed-nibble `[out, in/2]`
//!                    + e8m0 scale `[out, in/32]`. Scale bytes copy verbatim:
//!                    ggml's doubled k-values and its 2^(e-128) scale cancel.
//!   token_embd       F16 (the engine's host-side embed path has no Q8_0 arm)
//!   fp8 projections  e4m3 with 32x32 e8m0 block scales → dequantised → Q8_0
//!                    (bit-exact with gguf-py / ggml `quantize_row_q8_0_ref`).
//!   bf16 tensors     F32 / F16 / raw BF16 per the contract's role.
//!   Engram tables    98 GiB of fp8 rows, NOT presented (they are gathered
//!                    per token, never loaded whole) — use [`V41HfWeights::raw`].
//!   DSpark (`mtp.*`), vision, aligner: not presented yet.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use color_eyre::eyre::{self, eyre, Context};

use crate::gguf::{GgufTensor, GgufType};
use crate::kquants::f32_to_f16_bits;
use crate::safetensors::{SafetensorsDir, StDtype, StTensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cast {
    F32,
    F16,
    Bf16Raw,
}

#[derive(Debug, Clone)]
enum Kind {
    Cast { src: String, to: Cast },
    Q8 { w: String, scale: Option<String> },
    /// Stacked MXFP4 experts under `prefix` (`layers.{l}.ffn.experts.` for the
    /// main model, `mtp.{s}.ffn.experts.` for a drafter stage). Carries the
    /// count because the drafter has 128 where the main model has 384.
    Experts { prefix: String, which: &'static str, n: usize },
}

/// A tensor as the engine sees it: GGUF name, ggml dtype, ggml dims
/// (innermost first), byte size — plus how to materialise it.
#[derive(Debug, Clone)]
pub struct VTensor {
    pub name: String,
    pub dtype: GgufType,
    pub dims: Vec<u64>,
    pub elements: u64,
    pub byte_size: u64,
    kind: Kind,
}

impl VTensor {
    pub fn is_stacked_experts(&self) -> bool {
        matches!(self.kind, Kind::Experts { .. })
    }
}

pub struct V41HfWeights {
    st: SafetensorsDir,
    config: serde_json::Value,
    n_layers: usize,
    n_expert: usize,
    table: Vec<VTensor>,
    /// GGUF-shaped descriptors (shard/offset fields 0) so the loaders' code
    /// that reads `dims / dtype / byte_size` works unchanged via `WeightSrc`.
    gguf_table: Vec<GgufTensor>,
    index: HashMap<String, usize>,
    threads: usize,
}

/// DSpark drafter: three stages, each a transformer layer with a 128-wide router.
pub const MTP_STAGES: usize = 3;
pub const MTP_N_EXPERT: usize = 128;

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// torch `float8_e8m0fnu`: 2^(e-127); 0x00 = 2^-127 (an f32 denormal), 0xFF = NaN.
#[inline]
pub fn e8m0_to_f32(e: u8) -> f32 {
    match e {
        0xFF => f32::NAN,
        0 => f32::from_bits(1 << 22),
        e => f32::from_bits((e as u32) << 23),
    }
}

/// torch `float8_e4m3fn`: S EEEE MMM, bias 7, exp 0 = denormal (m·2^-9),
/// no infinities, 0x7F/0xFF = NaN.
pub fn e4m3_to_f32(b: u8) -> f32 {
    if b & 0x7F == 0x7F {
        return f32::NAN;
    }
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    let v = if e == 0 { m * 2f32.powi(-9) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    sign * v
}

fn e4m3_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|i| e4m3_to_f32(i as u8)))
}

/// One Q8_0 block from 32 f32 values, bit-exact with ggml's reference
/// quantiser (and gguf-py's): d = amax/127 in f32, id = 1/d, q = round-half-away.
pub fn quantize_q8_0_block(x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(x.len(), 32);
    debug_assert_eq!(out.len(), 34);
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let d = amax / 127.0;
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };
    out[..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
    for (o, &v) in out[2..].iter_mut().zip(x) {
        *o = ((v * id).round() as i8) as u8;
    }
}

fn reversed(shape: &[u64]) -> Vec<u64> {
    shape.iter().rev().copied().collect()
}

/// Threads used to split ONE expert's weight pread. `V41_EXPERT_PREAD_THREADS`.
///
/// A decode miss is a single ~5.9 MB read and, at a ~90% hit rate, a layer
/// averages 0.39 misses — so there is almost never a second miss to issue
/// alongside it. One pread at a time measured 2.17 GB/s, ~80% of this drive's
/// SINGLE-THREAD figure (2.70) and half its depth figure (4.31 at 32 threads).
/// Splitting the miss's own read is the only way to give the queue depth.
///
/// Default 8, from the END-TO-END sweep (2026-09-13, 256-token generation,
/// 52 GB pool): 3.9 / 4.2 / 4.7 tok/s and 10.49 / 9.22 / 7.90 ms per miss at
/// 1 / 4 / 8 threads. That contradicts the synthetic drive sweep this default
/// used to cite (2.70/4.11/4.22/4.31 GB/s at 1/4/8/32, "most of the win is at
/// 4"), because the synthetic read a cold file with nothing else in flight —
/// in production the miss competes with the engine's own allocations and the
/// page cache, so the extra queue depth still pays at 8. Prefer the e2e number.
///
/// NOTE: `decode_pread_gbps` is meaningless above 1 thread — it divides bytes
/// by the SUM of per-thread time, so it FALLS as parallelism improves the wall
/// clock. Judge this knob by `ms_per_miss` and tok/s only.
/// `V41_EXPERT_ODIRECT=1`: read expert weights through an `O_DIRECT` handle.
///
/// The buffered path exists so the pager's LRU refills can hit the page cache.
/// On box 2 that is a losing trade: 101 GB of experts against ~5 GB of page
/// cache (<5% cacheable), and the copy through the cache costs more than the
/// hits save. Measured there, 18.80 MB at random offsets, daemon idle:
/// O_DIRECT 4.70 ms (4.00 GB/s) vs buffered 12.55 ms (1.50 GB/s).
///
/// Default OFF: a box whose page cache CAN hold a useful slice of the expert
/// set still wants the buffered path.
pub fn expert_odirect() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_EXPERT_ODIRECT").as_deref() == Ok("1")
    });
    *B
}

pub fn expert_pread_threads() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_EXPERT_PREAD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
    });
    (*N).max(1)
}

impl V41HfWeights {
    /// `dir` is the HF snapshot (with `inference/config.json`). `expert_limit`
    /// caps the routed experts presented per layer (fixture / dry runs).
    pub fn open(dir: impl AsRef<Path>, expert_limit: Option<usize>) -> eyre::Result<Self> {
        let dir = dir.as_ref();
        let st = SafetensorsDir::open(dir)?;
        let cfg_path = dir.join("inference").join("config.json");
        let config: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&cfg_path).wrap_err_with(|| format!("open {}", cfg_path.display()))?,
        )
        .wrap_err_with(|| format!("parse {}", cfg_path.display()))?;
        let n_layers = config["n_layers"]
            .as_u64()
            .ok_or_else(|| eyre!("config: n_layers missing"))? as usize;
        let n_routed = config["n_routed_experts"]
            .as_u64()
            .ok_or_else(|| eyre!("config: n_routed_experts missing"))? as usize;
        let n_expert = expert_limit.map_or(n_routed, |l| l.min(n_routed));
        let threads = std::env::var("DEEPSTRIX_HF_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(4, |n| n.get().min(8))
            })
            .max(1);
        let mut this = Self {
            st,
            config,
            n_layers,
            n_expert,
            table: Vec::new(),
            gguf_table: Vec::new(),
            index: HashMap::new(),
            threads,
        };
        this.build_table()?;
        Ok(this)
    }

    pub fn raw(&self) -> &SafetensorsDir {
        &self.st
    }

    pub fn config(&self) -> &serde_json::Value {
        &self.config
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    pub fn tensors(&self) -> &[VTensor] {
        &self.table
    }

    pub fn tensor(&self, name: &str) -> Option<&VTensor> {
        self.index.get(name).map(|&i| &self.table[i])
    }

    /// GGUF-shaped descriptor of a presented tensor (see `gguf_table`).
    pub fn gguf_tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.index.get(name).map(|&i| &self.gguf_table[i])
    }

    pub fn gguf_tensors(&self) -> &[GgufTensor] {
        &self.gguf_table
    }

    pub fn get(&self, name: &str) -> eyre::Result<&VTensor> {
        self.tensor(name).ok_or_else(|| eyre!("tensor {name:?} not presented by the V4.1 HF view"))
    }

    // ----- table construction (mirrors to_gguf.py::write_layer / write_globals) -----

    fn push(&mut self, name: String, dtype: GgufType, dims: Vec<u64>, kind: Kind) -> eyre::Result<()> {
        let elements: u64 = dims.iter().product();
        let byte_size = dtype
            .size_of(elements)
            .map_err(|e| eyre!("{name}: {e:?}"))?;
        if self.index.contains_key(&name) {
            return Err(eyre!("{name}: presented twice"));
        }
        self.index.insert(name.clone(), self.table.len());
        self.gguf_table.push(GgufTensor {
            name: name.clone(),
            dims: dims.clone(),
            dtype,
            rel_offset: 0,
            abs_offset: 0,
            elements,
            byte_size,
            shard: 0,
        });
        self.table.push(VTensor { name, dtype, dims, elements, byte_size, kind });
        Ok(())
    }

    fn push_cast(&mut self, name: &str, src: &str, to: Cast) -> eyre::Result<()> {
        let t = self.st.get(src)?;
        let dtype = match to {
            Cast::F32 => GgufType::F32,
            Cast::F16 => GgufType::F16,
            Cast::Bf16Raw => GgufType::BF16,
        };
        let dims = reversed(&t.shape);
        self.push(name.to_owned(), dtype, dims, Kind::Cast { src: src.to_owned(), to })
    }

    fn push_q8(&mut self, name: &str, w: &str) -> eyre::Result<()> {
        let t = self.st.get(w)?;
        if t.shape.len() != 2 {
            return Err(eyre!("{w}: Q8_0 role needs a 2-D tensor, got {:?}", t.shape));
        }
        if t.shape[1] % 32 != 0 {
            return Err(eyre!("{w}: inner dim {} not a multiple of 32", t.shape[1]));
        }
        let scale = w
            .strip_suffix("weight")
            .map(|p| format!("{p}scale"))
            .filter(|s| self.st.has(s));
        let dims = reversed(&t.shape);
        self.push(name.to_owned(), GgufType::Q8_0, dims, Kind::Q8 { w: w.to_owned(), scale })
    }

    fn push_experts(&mut self, name: &str, prefix: &str, which: &'static str, n: usize) -> eyre::Result<()> {
        let t = self.st.get(&format!("{prefix}0.{which}.weight"))?;
        let (out, half) = match t.shape[..] {
            [out, half] => (out, half),
            _ => return Err(eyre!("{}: expected [out, in/2], got {:?}", t.name, t.shape)),
        };
        let inn = half * 2;
        if inn % 32 != 0 {
            return Err(eyre!("{}: in={inn} not a multiple of 32", t.name));
        }
        let dims = vec![inn, out, n as u64];
        self.push(name.to_owned(), GgufType::MXFP4, dims,
                  Kind::Experts { prefix: prefix.to_owned(), which, n })
    }

    fn build_table(&mut self) -> eyre::Result<()> {
        // F16, not Q8_0: the embed path (host-side row gather) has F16/K-quant
        // arms only, and bf16->f16 is exact for every value in range.
        self.push_cast("token_embd.weight", "embed.weight", Cast::F16)?;
        self.push_q8("output.weight", "head.weight")?;
        self.push_cast("output_norm.weight", "norm.weight", Cast::F32)?;
        for l in 0..self.n_layers {
            self.build_layer(l)?;
        }
        // DSpark drafter stages, if the checkpoint carries them. Absent on
        // checkpoints without MTP, so this is best-effort by design.
        for sgi in 0..MTP_STAGES {
            if self.st.has(&format!("mtp.{sgi}.attn_norm.weight")) {
                self.build_mtp(sgi)?;
            }
        }
        Ok(())
    }

    /// One layer of the DSpark drafter, presented under `mtp.{s}.*`.
    ///
    /// The three `mtp.*` groups are NOT three independent drafters — they are a
    /// 3-LAYER draft model, run autoregressively to emit K draft tokens:
    ///
    ///   mtp.0   main_proj + main_norm -> layer   (entry: eats the main residuals)
    ///   mtp.1   layer                            (middle)
    ///   mtp.2   layer -> norm -> confidence_head + markov_head   (exit: logits)
    ///
    /// Each layer is otherwise identical to a main layer — MLA attention, mHC, a
    /// routed MoE, a shared expert — except the router is 128-wide instead of
    /// 384. See `docs/v41/DSPARK_DESIGN.md`.
    fn build_mtp(&mut self, sgi: usize) -> eyre::Result<()> {
        let p = format!("mtp.{sgi}.");
        // Presented as `blk.{n_layers + s}` so `DgpuLayerWeights::load` and
        // `IgpuLayerWeights::load` pick the drafter up UNCHANGED — a drafter layer
        // carries exactly a main layer's tensor set, and the 128-wide router is
        // fine because `router_topk` takes `n_expert` at runtime. Nothing iterates
        // past `n_layers`, so these are only ever fetched explicitly.
        let b = format!("blk.{}.", self.n_layers + sgi);
        // Entry/exit extras keep the `mtp.` prefix: they have no main-layer
        // counterpart and must not be mistaken for one.
        let m = format!("mtp.{sgi}.");
        // Entry stage only.
        if self.st.has(&format!("{p}main_proj.weight")) {
            self.push_q8(&format!("{m}main_proj.weight"), &format!("{p}main_proj.weight"))?;
            self.push_cast(&format!("{m}main_norm.weight"), &format!("{p}main_norm.weight"), Cast::F32)?;
        }
        // Exit stage only: final norm plus the confidence and markov heads.
        if self.st.has(&format!("{p}norm.weight")) {
            self.push_cast(&format!("{m}norm.weight"), &format!("{p}norm.weight"), Cast::F32)?;
        }
        if self.st.has(&format!("{p}confidence_head.proj.weight")) {
            self.push_cast(&format!("{m}confidence.weight"), &format!("{p}confidence_head.proj.weight"), Cast::F32)?;
        }
        if self.st.has(&format!("{p}markov_head.embed.weight")) {
            self.push_cast(&format!("{m}markov_embd.weight"), &format!("{p}markov_head.embed.weight"), Cast::F16)?;
            self.push_q8(&format!("{m}markov_head.weight"), &format!("{p}markov_head.head.weight"))?;
        }
        for (src, dst) in [
            ("attn_norm.weight", "attn_norm.weight"),
            ("ffn_norm.weight", "ffn_norm.weight"),
            ("attn.q_norm.weight", "attn_q_a_norm.weight"),
            ("attn.kv_norm.weight", "attn_kv_a_norm.weight"),
            ("attn.attn_sink", "attn_sinks.weight"),
            ("hc_attn_fn", "hc_attn_fn.weight"),
            ("hc_ffn_fn", "hc_ffn_fn.weight"),
            ("hc_attn_base", "hc_attn_base.weight"),
            ("hc_ffn_base", "hc_ffn_base.weight"),
            ("hc_attn_scale", "hc_attn_scale.weight"),
            ("hc_ffn_scale", "hc_ffn_scale.weight"),
            ("ffn.gate.bias", "exp_probs_b.bias"),
            ("ffn.gate.bias_vl", "exp_probs_b_vl.bias"),
        ] {
            self.push_cast(&format!("{b}{dst}"), &format!("{p}{src}"), Cast::F32)?;
        }
        for (src, dst) in [
            ("attn.wq_a", "attn_q_a"),
            ("attn.wq_b", "attn_q_b"),
            ("attn.wkv", "attn_kv"),
            ("attn.wo_a", "attn_output_a"),
            ("attn.wo_b", "attn_output_b"),
        ] {
            self.push_q8(&format!("{b}{dst}.weight"), &format!("{p}{src}.weight"))?;
        }
        self.push_cast(&format!("{b}ffn_gate_inp.weight"), &format!("{p}ffn.gate.weight"), Cast::Bf16Raw)?;
        for (src, dst) in [("w1", "ffn_gate_shexp"), ("w3", "ffn_up_shexp"), ("w2", "ffn_down_shexp")] {
            self.push_q8(&format!("{b}{dst}.weight"), &format!("{p}ffn.shared_experts.{src}.weight"))?;
        }
        for (src, dst) in [("w1", "ffn_gate_exps"), ("w3", "ffn_up_exps"), ("w2", "ffn_down_exps")] {
            self.push_experts(
                &format!("{b}{dst}.weight"),
                &format!("{p}ffn.experts."),
                src,
                MTP_N_EXPERT,
            )?;
        }
        Ok(())
    }

    fn build_layer(&mut self, l: usize) -> eyre::Result<()> {
        let p = format!("layers.{l}.");
        let b = format!("blk.{l}.");
        for (src, dst) in [
            ("attn_norm.weight", "attn_norm.weight"),
            ("ffn_norm.weight", "ffn_norm.weight"),
            ("attn.q_norm.weight", "attn_q_a_norm.weight"),
            ("attn.kv_norm.weight", "attn_kv_a_norm.weight"),
            ("attn.attn_sink", "attn_sinks.weight"),
            ("hc_attn_fn", "hc_attn_fn.weight"),
            ("hc_ffn_fn", "hc_ffn_fn.weight"),
            ("hc_attn_base", "hc_attn_base.weight"),
            ("hc_ffn_base", "hc_ffn_base.weight"),
            ("hc_attn_scale", "hc_attn_scale.weight"),
            ("hc_ffn_scale", "hc_ffn_scale.weight"),
            ("ffn.gate.bias", "exp_probs_b.bias"),
            ("ffn.gate.bias_vl", "exp_probs_b_vl.bias"),
        ] {
            self.push_cast(&format!("{b}{dst}"), &format!("{p}{src}"), Cast::F32)?;
        }
        for (src, dst) in [
            ("attn.wq_a", "attn_q_a"),
            ("attn.wq_b", "attn_q_b"),
            ("attn.wkv", "attn_kv"),
            ("attn.wo_a", "attn_output_a"),
            ("attn.wo_b", "attn_output_b"),
        ] {
            self.push_q8(&format!("{b}{dst}.weight"), &format!("{p}{src}.weight"))?;
        }
        self.push_cast(&format!("{b}ffn_gate_inp.weight"), &format!("{p}ffn.gate.weight"), Cast::Bf16Raw)?;
        for (src, dst) in [("w1", "ffn_gate_shexp"), ("w3", "ffn_up_shexp"), ("w2", "ffn_down_shexp")] {
            self.push_q8(
                &format!("{b}{dst}.weight"),
                &format!("{p}ffn.shared_experts.{src}.weight"),
            )?;
        }
        for (src, dst) in [("w1", "ffn_gate_exps"), ("w3", "ffn_up_exps"), ("w2", "ffn_down_exps")] {
            self.push_experts(&format!("{b}{dst}.weight"), &format!("{p}ffn.experts."), src, self.n_expert)?;
        }
        if self.st.has(&format!("{p}attn.compressor.wkv.weight")) {
            self.push_cast(
                &format!("{b}attn_compressor_kv.weight"),
                &format!("{p}attn.compressor.wkv.weight"),
                Cast::F16,
            )?;
            if self.st.has(&format!("{p}attn.compressor.wgate.weight")) {
                self.push_cast(
                    &format!("{b}attn_compressor_gate.weight"),
                    &format!("{p}attn.compressor.wgate.weight"),
                    Cast::F16,
                )?;
            }
            self.push_cast(
                &format!("{b}attn_compressor_norm.weight"),
                &format!("{p}attn.compressor.norm.weight"),
                Cast::F32,
            )?;
        }
        if self.st.has(&format!("{p}attn.indexer.wq_b.weight")) {
            self.push_q8(&format!("{b}indexer.attn_q_b.weight"), &format!("{p}attn.indexer.wq_b.weight"))?;
            self.push_cast(
                &format!("{b}indexer.proj.weight"),
                &format!("{p}attn.indexer.weights_proj.weight"),
                Cast::F32,
            )?;
            if self.st.has(&format!("{p}attn.indexer.wk.weight")) {
                self.push_cast(
                    &format!("{b}indexer.attn_k.weight"),
                    &format!("{p}attn.indexer.wk.weight"),
                    Cast::F16,
                )?;
                self.push_cast(
                    &format!("{b}indexer.k_norm.weight"),
                    &format!("{p}attn.indexer.k_norm.weight"),
                    Cast::F32,
                )?;
            }
        }
        if self.st.has(&format!("{p}engram.wkv.weight")) {
            self.push_q8(&format!("{b}engram_wkv.weight"), &format!("{p}engram.wkv.weight"))?;
            self.push_cast(&format!("{b}engram_q.weight"), &format!("{p}engram.q_weight"), Cast::F32)?;
            self.push_cast(&format!("{b}engram_k.weight"), &format!("{p}engram.k_weight"), Cast::F32)?;
        }
        Ok(())
    }

    // ----- materialisation -----

    /// Whole tensor, transformed, into `dst` (`dst.len() == byte_size`).
    /// Stacked experts are produced in parallel (`DEEPSTRIX_HF_THREADS`).
    pub fn read_into(&self, vt: &VTensor, dst: &mut [u8]) -> eyre::Result<()> {
        if dst.len() as u64 != vt.byte_size {
            return Err(eyre!("{}: dst len {} != byte_size {}", vt.name, dst.len(), vt.byte_size));
        }
        match &vt.kind {
            Kind::Cast { src, to } => self.read_cast(src, *to, dst),
            Kind::Q8 { w, scale } => self.read_q8(w, scale.as_deref(), dst),
            Kind::Experts { prefix, which, .. } => {
                let per = self.expert_bytes(vt);
                let mut jobs: Vec<(usize, &mut [u8])> = dst.chunks_mut(per).enumerate().collect();
                let group = jobs.len().div_ceil(self.threads).max(1);
                let err: Mutex<Option<eyre::Report>> = Mutex::new(None);
                std::thread::scope(|sc| {
                    for grp in jobs.chunks_mut(group) {
                        let err = &err;
                        sc.spawn(move || {
                            for (e, slice) in grp.iter_mut() {
                                if let Err(x) = self.read_expert_raw(prefix, which, *e, slice) {
                                    *err.lock().unwrap() = Some(x);
                                    return;
                                }
                            }
                        });
                    }
                });
                match err.into_inner().unwrap() {
                    Some(e) => Err(e.wrap_err(format!("{}: stacked expert read", vt.name))),
                    None => Ok(()),
                }
            }
        }
    }

    pub fn read(&self, vt: &VTensor) -> eyre::Result<Vec<u8>> {
        let mut v = vec![0u8; vt.byte_size as usize];
        self.read_into(vt, &mut v)?;
        Ok(v)
    }

    /// Bytes per expert of a stacked expert tensor.
    ///
    /// Divides by the TENSOR's own expert count, not `self.n_expert`. The DSpark
    /// drafter has 128 where the main model has 384; using the model-wide
    /// constant sliced a drafter tensor into 384 phantom experts and read past
    /// the end (`mtp.0.ffn.experts.336.w1.weight` does not exist).
    pub fn expert_bytes(&self, vt: &VTensor) -> usize {
        let n = match &vt.kind {
            Kind::Experts { n, .. } => *n as u64,
            _ => self.n_expert as u64,
        };
        (vt.byte_size / n) as usize
    }

    /// One expert of a stacked expert tensor (`dst.len() == expert_bytes`).
    pub fn read_expert_into(&self, vt: &VTensor, e: usize, dst: &mut [u8]) -> eyre::Result<()> {
        let Kind::Experts { prefix, which, n } = &vt.kind else {
            return Err(eyre!("{}: not a stacked expert tensor", vt.name));
        };
        if e >= *n {
            return Err(eyre!("{}: expert {e} >= {n}", vt.name));
        }
        if dst.len() != self.expert_bytes(vt) {
            return Err(eyre!("{}: dst len {} != expert bytes {}", vt.name, dst.len(), self.expert_bytes(vt)));
        }
        self.read_expert_raw(prefix, which, e, dst)
    }

    /// [`Self::read_expert_into`] but leaving the bytes in the HF layout; see
    /// [`Self::read_expert_hf_layout`].
    pub fn read_expert_hf_layout_into(&self, vt: &VTensor, e: usize, dst: &mut [u8]) -> eyre::Result<()> {
        let Kind::Experts { prefix, which, n } = &vt.kind else {
            return Err(eyre!("{}: not a stacked expert tensor", vt.name));
        };
        if e >= *n {
            return Err(eyre!("{}: expert {e} >= {n}", vt.name));
        }
        if dst.len() != self.expert_bytes(vt) {
            return Err(eyre!("{}: dst len {} != expert bytes {}", vt.name, dst.len(), self.expert_bytes(vt)));
        }
        self.read_expert_hf_layout(prefix, which, e, dst)
    }

    /// Arbitrary byte range of the transformed tensor. Expert-granular for
    /// stacked experts (only the experts overlapping the range are produced);
    /// other roles materialise the whole tensor and copy the slice.
    pub fn read_range_into(&self, vt: &VTensor, byte_off: u64, dst: &mut [u8]) -> eyre::Result<()> {
        let end = byte_off + dst.len() as u64;
        if end > vt.byte_size {
            return Err(eyre!("{}: range [{byte_off},{end}) exceeds {}", vt.name, vt.byte_size));
        }
        if dst.is_empty() {
            return Ok(());
        }
        if !vt.is_stacked_experts() {
            let whole = self.read(vt)?;
            dst.copy_from_slice(&whole[byte_off as usize..end as usize]);
            return Ok(());
        }
        let per = self.expert_bytes(vt) as u64;
        let e0 = byte_off / per;
        let e1 = (end - 1) / per;
        let mut tmp = Vec::new();
        for e in e0..=e1 {
            let es = e * per;
            let ee = es + per;
            let a = byte_off.max(es);
            let b = end.min(ee);
            let out = &mut dst[(a - byte_off) as usize..(b - byte_off) as usize];
            if a == es && b == ee {
                self.read_expert_into(vt, e as usize, out)?;
            } else {
                tmp.resize(per as usize, 0);
                self.read_expert_into(vt, e as usize, &mut tmp)?;
                out.copy_from_slice(&tmp[(a - es) as usize..(b - es) as usize]);
            }
        }
        Ok(())
    }

    fn read_cast(&self, src: &str, to: Cast, dst: &mut [u8]) -> eyre::Result<()> {
        let t = self.st.get(src)?;
        let raw = self.st.read(t)?;
        let widen = |raw: &[u8], dtype: StDtype, i: usize| -> f32 {
            match dtype {
                StDtype::BF16 => bf16_to_f32(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])),
                StDtype::F32 => f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()),
                _ => unreachable!(),
            }
        };
        match (t.dtype, to) {
            (StDtype::BF16, Cast::Bf16Raw) | (StDtype::F32, Cast::F32) => dst.copy_from_slice(&raw),
            (StDtype::BF16, Cast::F32) => {
                for (i, o) in dst.chunks_exact_mut(4).enumerate() {
                    o.copy_from_slice(&widen(&raw, t.dtype, i).to_le_bytes());
                }
            }
            (StDtype::BF16 | StDtype::F32, Cast::F16) => {
                for (i, o) in dst.chunks_exact_mut(2).enumerate() {
                    o.copy_from_slice(&f32_to_f16_bits(widen(&raw, t.dtype, i)).to_le_bytes());
                }
            }
            (d, c) => return Err(eyre!("{src}: no cast {d:?} -> {c:?}")),
        }
        Ok(())
    }

    fn read_q8(&self, w: &str, scale: Option<&str>, dst: &mut [u8]) -> eyre::Result<()> {
        let t = self.st.get(w)?;
        let (out, inn) = (t.shape[0] as usize, t.shape[1] as usize);
        let raw = self.st.read(t)?;
        let scale: Option<(Vec<u8>, &StTensor)> = match scale {
            Some(s) => {
                let st_ = self.st.get(s)?;
                if st_.dtype != StDtype::F8E8M0 || st_.shape.len() != 2 {
                    return Err(eyre!("{s}: expected 2-D F8_E8M0 block scales, got {:?} {:?}", st_.dtype, st_.shape));
                }
                if out % st_.shape[0] as usize != 0 || inn % st_.shape[1] as usize != 0 {
                    return Err(eyre!("{s}: scale grid {:?} does not tile [{out},{inn}]", st_.shape));
                }
                Some((self.st.read(st_)?, st_))
            }
            None => None,
        };
        if t.dtype == StDtype::F8E4M3 && scale.is_none() {
            return Err(eyre!("{w}: fp8 weight without a scale tensor"));
        }
        let row_out = inn / 32 * 34;
        let nthreads = self.threads.min(out).max(1);
        let rows_per = out.div_ceil(nthreads);
        let err: Mutex<Option<eyre::Report>> = Mutex::new(None);
        let (raw, scale, lut) = (&raw, &scale, e4m3_lut());
        std::thread::scope(|sc| {
            for (i, chunk) in dst.chunks_mut(rows_per * row_out).enumerate() {
                let err = &err;
                sc.spawn(move || {
                    let mut f = vec![0f32; inn];
                    let r0 = i * rows_per;
                    for (k, orow) in chunk.chunks_exact_mut(row_out).enumerate() {
                        let r = r0 + k;
                        match (t.dtype, scale) {
                            (StDtype::BF16, _) => {
                                for c in 0..inn {
                                    let i = 2 * (r * inn + c);
                                    f[c] = bf16_to_f32(u16::from_le_bytes([raw[i], raw[i + 1]]));
                                }
                            }
                            (StDtype::F32, _) => {
                                for c in 0..inn {
                                    let i = 4 * (r * inn + c);
                                    f[c] = f32::from_le_bytes(raw[i..i + 4].try_into().unwrap());
                                }
                            }
                            (StDtype::F8E4M3, Some((sbytes, st_))) => {
                                let (sr, sc_) = (st_.shape[0] as usize, st_.shape[1] as usize);
                                let (br, bc) = (out / sr, inn / sc_);
                                // Deriving the block size by division is only valid for EXACT
                                // tiling. Under ceil-tiling a non-dividing dim can still satisfy
                                // `out % sr == 0` by coincidence while `br` is wrong, which
                                // mis-scales every element and would not necessarily show up in
                                // single-prompt parity. Today every V4.1 fp8 tensor is exactly
                                // 32x32 tiled (verified across layers 0/2/20/39); fail loudly if a
                                // checkpoint ever deviates.
                                if br * sr != out || bc * sc_ != inn {
                                    *err.lock().unwrap() = Some(eyre!(
                                        "{w}: fp8 scale grid {sr}x{sc_} does not exactly tile {out}x{inn} \
                                         (derived block {br}x{bc}); refusing to guess the block size"
                                    ));
                                    return;
                                }
                                let srow = &sbytes[(r / br) * sc_..(r / br + 1) * sc_];
                                for c in 0..inn {
                                    f[c] = lut[raw[r * inn + c] as usize] * e8m0_to_f32(srow[c / bc]);
                                }
                            }
                            (d, _) => {
                                *err.lock().unwrap() = Some(eyre!("{w}: cannot Q8_0-quantise {d:?}"));
                                return;
                            }
                        }
                        for (blk, o) in f.chunks_exact(32).zip(orow.chunks_exact_mut(34)) {
                            quantize_q8_0_block(blk, o);
                        }
                    }
                });
            }
        });
        match err.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// One expert's HF bytes, left in the **HF** layout: `dst` gets the packed
    /// nibbles (`out * nb * 16`) followed by the e8m0 scales (`out * nb`).
    ///
    /// That is exactly `out * nb * 17` bytes — the same size as the ggml form —
    /// because HF's `half` is `nb * 16`. So the pager's staging buffers, its H2D
    /// volume and the device slot size are all unchanged; the only thing that
    /// moves is WHERE the nibble permutation runs. On the CPU it measured
    /// **1.50 ms of an 11.5 ms miss**; `mxfp4_repack.hip` does it on the iGPU at
    /// streaming bandwidth. The pread also lands straight in the caller's buffer,
    /// so the two `vec![0u8; ..]` staging allocations disappear with it.
    ///
    /// DUPLICATED, not a branch inside `read_expert_raw`, and that is deliberate.
    /// Folding the two together behind a `hf_layout: bool` — with this function's
    /// `split_at_mut` sitting ahead of the scalar loop — made the CPU repack
    /// **2.8x slower** (8.8 s -> 24.9 s over a 256-token generation, measured
    /// 2026-09-13). Same pathology as the reverted thread-local staging: the hot
    /// nibble loop only gets good codegen while the compiler can prove
    /// `packed`/`scale`/`dst` are distinct allocations, and anything that muddies
    /// that costs more than the branch saves. Keep the two loops apart.
    pub fn read_expert_hf_layout(&self, prefix: &str, which: &str, e: usize, dst: &mut [u8]) -> eyre::Result<()> {
        let p = format!("{prefix}{e}.{which}.");
        let wt = self.st.get(&format!("{p}weight"))?;
        let sc = self.st.get(&format!("{p}scale"))?;
        if !matches!(wt.dtype, StDtype::I8 | StDtype::U8) || sc.dtype != StDtype::F8E8M0 {
            return Err(eyre!("{p}: expected I8 weight + F8_E8M0 scale, got {:?} + {:?}", wt.dtype, sc.dtype));
        }
        let (out, half) = (wt.shape[0] as usize, wt.shape[1] as usize);
        let nb = half * 2 / 32;
        if sc.shape[..] != [out as u64, nb as u64] {
            return Err(eyre!("{p}: scale shape {:?} != [{out},{nb}]", sc.shape));
        }
        if dst.len() != out * nb * 17 || wt.len as usize != out * nb * 16 || sc.len as usize != out * nb {
            return Err(eyre!(
                "{p}: HF-layout sizes dst={} wt={} sc={} != {}/{}/{}",
                dst.len(), wt.len, sc.len, out * nb * 17, out * nb * 16, out * nb
            ));
        }
        let t_pread = std::time::Instant::now();
        let (dp, ds) = dst.split_at_mut(out * nb * 16);
        // Same CACHED reads as the ggml path: the LRU re-reads evicted experts,
        // and POSIX_FADV_DONTNEED would force every refill back to the SSD.
        //
        // TIMED SEPARATELY (2026-09-18): `pread_ns` used to bracket all of this,
        // so the O_DIRECT weight read, the BUFFERED scale read and the setup were
        // indistinguishable. Production reads ~480 MB/s per stream where fio does
        // ~3,400 MB/s on the same block size against the same device, and one
        // aggregate counter cannot say which of the three is responsible.
        let t_w = std::time::Instant::now();
        if !(expert_odirect() && self.st.read_range_into_direct(wt, 0, dp)?) {
            self.st.read_range_into_cached_par(wt, 0, dp, expert_pread_threads())?;
        }
        EXPERT_READ_PROF.weight_ns.fetch_add(t_w.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.weight_bytes.fetch_add(wt.len, Relaxed);
        let t_s = std::time::Instant::now();
        self.st.read_range_into_cached(sc, 0, ds)?;
        EXPERT_READ_PROF.scale_ns.fetch_add(t_s.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.scale_bytes.fetch_add(sc.len, Relaxed);
        EXPERT_READ_PROF.pread_ns.fetch_add(t_pread.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.pread_bytes.fetch_add(wt.len + sc.len, Relaxed);
        EXPERT_READ_PROF.calls.fetch_add(1, Relaxed);
        Ok(())
    }

    /// Staging bytes [`Self::read_expert_hf_layout_direct`] needs for one role.
    pub fn hf_layout_direct_capacity(&self, vt: &VTensor) -> eyre::Result<usize> {
        let (packed, scale) = self.hf_layout_parts(vt)?;
        Ok(crate::safetensors::SafetensorsDir::direct_capacity_for(packed)
            + crate::safetensors::SafetensorsDir::direct_capacity_for(scale))
    }

    fn hf_layout_parts(&self, vt: &VTensor) -> eyre::Result<(usize, usize)> {
        let Kind::Experts { prefix, which, .. } = &vt.kind else {
            return Err(eyre!("{}: not a stacked expert tensor", vt.name));
        };
        let p = format!("{prefix}0.{which}.");
        let wt = self.st.get(&format!("{p}weight"))?;
        let (out, half) = (wt.shape[0] as usize, wt.shape[1] as usize);
        let nb = half * 2 / 32;
        Ok((out * nb * 16, out * nb))
    }

    /// [`Self::read_expert_hf_layout`] straight into PADDED, 4096-aligned staging
    /// with **no bounce buffer and no copy** — the whole point of O_DIRECT here.
    ///
    /// Returns `(packed_off, scale_off, out_rows, nb)`: where in `dst` each
    /// region's bytes actually begin, since O_DIRECT places them at the file
    /// offset's own 4096-residue. `Ok(None)` = no O_DIRECT handle for this shard.
    ///
    /// `dst` must be 4096-aligned and at least `hf_layout_direct_capacity`.
    /// Read ALL THREE roles of expert `e` in TWO preads instead of six.
    ///
    /// The checkpoint stores experts EXPERT-MAJOR, role-minor: for every expert
    /// the w1/w2/w3 weight planes are byte-contiguous (3 x 5.625 = 16.875 MB)
    /// and so are its three scale planes (3 x 0.352 = 1.055 MB), with the two
    /// runs far apart. VERIFIED across encoder and decoder layers, low and high
    /// expert ids, and different shards — and CHECKED again here at run time,
    /// returning `Ok(None)` (caller falls back) rather than trusting it.
    ///
    /// Why it matters: the per-role path issues six preads of ~5.6 MB and 0.35 MB
    /// and measures 2.96 GB/s aggregate, while this drive does 4.47 GB/s on a
    /// single ~20 MB O_DIRECT read. NVMe strongly prefers one large read.
    ///
    /// `dst_w` takes the weight run, `dst_s` the scale run; both must be
    /// 4096-aligned and sized by `direct_capacity_for` of the RUN length.
    /// Returns per-role `(packed_off, scale_off, out, nb)` into `dst_w`/`dst_s`.
    pub fn read_expert_runs_direct(
        &self,
        vts: [&VTensor; 3],
        e: usize,
        dst_w: &mut [u8],
        dst_s: &mut [u8],
    ) -> eyre::Result<Option<[(usize, usize, u32, u32); 3]>> {
        // ROLE order is the engine's (gate, up, down). PHYSICAL order in the
        // checkpoint is NOT the same: the loader maps gate<-w1, up<-w3, down<-w2,
        // so a run laid out w1,w2,w3 is gate,down,up. Rather than encode that
        // mapping a second time (getting it wrong swapped up and down and made
        // generation non-deterministic), derive each role's position in the run
        // from its actual FILE OFFSET.
        let mut wts = Vec::with_capacity(3);
        let mut scs = Vec::with_capacity(3);
        for vt in vts {
            let Kind::Experts { prefix, which, n } = &vt.kind else {
                return Err(eyre!("{}: not a stacked expert tensor", vt.name));
            };
            if e >= *n {
                return Err(eyre!("{}: expert {e} >= {n}", vt.name));
            }
            let p = format!("{prefix}{e}.{which}.");
            let wt = self.st.get(&format!("{p}weight"))?;
            let sc = self.st.get(&format!("{p}scale"))?;
            if !matches!(wt.dtype, StDtype::I8 | StDtype::U8) || sc.dtype != StDtype::F8E8M0 {
                return Err(eyre!("{p}: expected I8 weight + F8_E8M0 scale"));
            }
            wts.push(wt);
            scs.push(sc);
        }
        // Geometry is PER ROLE: gate/up are [N_FF_EXP, N_EMBD/2] while down is
        // [N_EMBD, N_FF_EXP/2]. Only the BYTE lengths are uniform (both work out
        // to rows*nb*16), and `repack_in_place` asserts (rows, nb) against its
        // own per-role geometry — so these must be returned per role, not taken
        // from role 0. Requiring equal shapes here silently disabled coalescing.
        let mut geom = [(0usize, 0usize); 3];
        for (r, t) in wts.iter().enumerate() {
            let (out, half) = (t.shape[0] as usize, t.shape[1] as usize);
            geom[r] = (out, half * 2 / 32);
        }
        let (packed_len, scale_len) = {
            let (out0, nb0) = geom[0];
            (out0 * nb0 * 16, out0 * nb0)
        };
        // Uniform byte length across roles is what makes one run sliceable.
        if geom.iter().any(|&(o, n)| o * n * 16 != packed_len)
            || wts.iter().any(|t| t.len as usize != packed_len)
            || scs.iter().any(|t| t.len as usize != scale_len)
        {
            return Ok(None);
        }
        // Rank each role by file offset, then require the run to be contiguous
        // in that order and all in one shard.
        let rank = |v: &[&crate::safetensors::StTensor]| -> Option<[usize; 3]> {
            let mut idx = [0usize, 1, 2];
            idx.sort_by_key(|&i| v[i].offset);
            let (a, b, c) = (idx[0], idx[1], idx[2]);
            if v[a].shard != v[b].shard || v[b].shard != v[c].shard {
                return None;
            }
            if v[a].offset + v[a].len != v[b].offset || v[b].offset + v[b].len != v[c].offset {
                return None;
            }
            // position[role] = its slot within the run
            let mut pos = [0usize; 3];
            for (slot, &role) in idx.iter().enumerate() {
                pos[role] = slot;
            }
            Some(pos)
        };
        let (Some(pos_w), Some(pos_s)) = (rank(&wts), rank(&scs)) else {
            return Ok(None);
        };
        let (run_w, run_s) = (packed_len * 3, scale_len * 3);
        let (w0, s0) = (
            wts.iter().map(|t| t.offset).min().unwrap(),
            scs.iter().map(|t| t.offset).min().unwrap(),
        );
        let t_pread = std::time::Instant::now();
        let t_w = std::time::Instant::now();
        let Some(pad_w) = self.st.read_span_into_direct_padded(wts[0].shard, w0, run_w, dst_w)?
        else {
            return Ok(None);
        };
        EXPERT_READ_PROF.weight_ns.fetch_add(t_w.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.weight_bytes.fetch_add(run_w as u64, Relaxed);
        let t_s = std::time::Instant::now();
        let Some(pad_s) = self.st.read_span_into_direct_padded(scs[0].shard, s0, run_s, dst_s)?
        else {
            return Ok(None);
        };
        EXPERT_READ_PROF.scale_ns.fetch_add(t_s.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.scale_bytes.fetch_add(run_s as u64, Relaxed);
        EXPERT_READ_PROF.pread_ns.fetch_add(t_pread.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.pread_bytes.fetch_add((run_w + run_s) as u64, Relaxed);
        EXPERT_READ_PROF.calls.fetch_add(1, Relaxed);
        let mut offs = [(0usize, 0usize, 0u32, 0u32); 3];
        for (r, o) in offs.iter_mut().enumerate() {
            *o = (
                pad_w + pos_w[r] * packed_len,
                pad_s + pos_s[r] * scale_len,
                geom[r].0 as u32,
                geom[r].1 as u32,
            );
        }
        Ok(Some(offs))
    }

    /// Byte length of one expert's weight run and scale run (all three roles).
    pub fn expert_run_lens(&self, vt: &VTensor) -> eyre::Result<(usize, usize)> {
        let Kind::Experts { prefix, .. } = &vt.kind else {
            return Err(eyre!("{}: not a stacked expert tensor", vt.name));
        };
        let wt = self.st.get(&format!("{prefix}0.w1.weight"))?;
        let (out, half) = (wt.shape[0] as usize, wt.shape[1] as usize);
        let nb = half * 2 / 32;
        Ok((out * nb * 16 * 3, out * nb * 3))
    }

    pub fn read_expert_hf_layout_direct(
        &self,
        vt: &VTensor,
        e: usize,
        dst: &mut [u8],
    ) -> eyre::Result<Option<(usize, usize, u32, u32)>> {
        let Kind::Experts { prefix, which, n } = &vt.kind else {
            return Err(eyre!("{}: not a stacked expert tensor", vt.name));
        };
        if e >= *n {
            return Err(eyre!("{}: expert {e} >= {n}", vt.name));
        }
        let p = format!("{prefix}{e}.{which}.");
        let wt = self.st.get(&format!("{p}weight"))?;
        let sc = self.st.get(&format!("{p}scale"))?;
        if !matches!(wt.dtype, StDtype::I8 | StDtype::U8) || sc.dtype != StDtype::F8E8M0 {
            return Err(eyre!("{p}: expected I8 weight + F8_E8M0 scale, got {:?} + {:?}", wt.dtype, sc.dtype));
        }
        let (out, half) = (wt.shape[0] as usize, wt.shape[1] as usize);
        let nb = half * 2 / 32;
        let (packed_len, scale_len) = (out * nb * 16, out * nb);
        if wt.len as usize != packed_len || sc.len as usize != scale_len {
            return Err(eyre!("{p}: sizes wt={} sc={} != {packed_len}/{scale_len}", wt.len, sc.len));
        }
        let cap_w = crate::safetensors::SafetensorsDir::direct_capacity_for(packed_len);
        if dst.len() < cap_w + crate::safetensors::SafetensorsDir::direct_capacity_for(scale_len) {
            return Err(eyre!("{p}: direct staging {} too small", dst.len()));
        }
        let t_pread = std::time::Instant::now();
        let (region_w, region_s) = dst.split_at_mut(cap_w);
        let t_w = std::time::Instant::now();
        let Some(pad_w) = self.st.read_range_into_direct_padded(wt, 0, packed_len, region_w)? else {
            return Ok(None);
        };
        EXPERT_READ_PROF.weight_ns.fetch_add(t_w.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.weight_bytes.fetch_add(wt.len, Relaxed);
        let t_s = std::time::Instant::now();
        let Some(pad_s) = self.st.read_range_into_direct_padded(sc, 0, scale_len, region_s)? else {
            return Ok(None);
        };
        EXPERT_READ_PROF.scale_ns.fetch_add(t_s.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.scale_bytes.fetch_add(sc.len, Relaxed);
        EXPERT_READ_PROF.pread_ns.fetch_add(t_pread.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.pread_bytes.fetch_add(wt.len + sc.len, Relaxed);
        EXPERT_READ_PROF.calls.fetch_add(1, Relaxed);
        Ok(Some((pad_w, cap_w + pad_s, out as u32, nb as u32)))
    }

    /// HF packed nibbles + e8m0 scales of one expert → ggml MXFP4 blocks.
    fn read_expert_raw(&self, prefix: &str, which: &str, e: usize, dst: &mut [u8]) -> eyre::Result<()> {
        let p = format!("{prefix}{e}.{which}.");
        let wt = self.st.get(&format!("{p}weight"))?;
        let sc = self.st.get(&format!("{p}scale"))?;
        if !matches!(wt.dtype, StDtype::I8 | StDtype::U8) || sc.dtype != StDtype::F8E8M0 {
            return Err(eyre!("{p}: expected I8 weight + F8_E8M0 scale, got {:?} + {:?}", wt.dtype, sc.dtype));
        }
        let (out, half) = (wt.shape[0] as usize, wt.shape[1] as usize);
        let nb = half * 2 / 32;
        if sc.shape[..] != [out as u64, nb as u64] {
            return Err(eyre!("{p}: scale shape {:?} != [{out},{nb}]", sc.shape));
        }
        if dst.len() != out * nb * 17 {
            return Err(eyre!("{p}: dst len {} != {}", dst.len(), out * nb * 17));
        }
        // CACHED reads: the M7 expert pager re-reads the same experts as its LRU evicts and
        // refills, so `POSIX_FADV_DONTNEED` (what `st.read` issues) would force every refill back
        // to the SSD instead of the page cache. Engram already uses the cached path for the same
        // reason. See docs/v41/M7_EXPERT_TIER.md.
        // THIS LOOP IS CODEGEN-FRAGILE. Twice now, a change that touched nothing
        // inside it made it 2.7-2.8x slower, both times by costing the compiler
        // its proof that `packed`, `scale` and `dst` are distinct allocations:
        //
        //   * reusing a thread-local staging pair (borrowed through a RefCell):
        //     allocations 973 -> 1 ms, but repack 8390 -> 23118 ms, net -15% tok/s;
        //   * merging the HF-layout reader in as an `if hf_layout { .. }` branch,
        //     whose `split_at_mut(dst)` sat ahead of this loop: repack
        //     8758 -> 24934 ms over the same 256-token generation, 4.7 -> 4.1 tok/s.
        //
        // Both were reverted (2026-09-13). Fresh Vecs and a function that does
        // nothing else are cheaper than either "optimisation". If you change this
        // function AT ALL, measure `repack_ns` — not just `alloc_ns`, and not just
        // tok/s, which buries a 3x regression in one term under the SSD read.
        // The permutation itself now runs on the iGPU anyway (see
        // `read_expert_hf_layout`); this path is the rollback, so keep it simple.
        let t_alloc = std::time::Instant::now();
        let mut packed = vec![0u8; wt.len as usize];
        let mut scale = vec![0u8; sc.len as usize];
        let t_pread = std::time::Instant::now();
        EXPERT_READ_PROF.alloc_ns.fetch_add(
            t_pread.duration_since(t_alloc).as_nanos() as u64, Relaxed);
        let t_w = std::time::Instant::now();
        if !(expert_odirect() && self.st.read_range_into_direct(wt, 0, &mut packed)?) {
            self.st.read_range_into_cached_par(wt, 0, &mut packed, expert_pread_threads())?;
        }
        EXPERT_READ_PROF.weight_ns.fetch_add(t_w.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.weight_bytes.fetch_add(wt.len, Relaxed);
        let t_s = std::time::Instant::now();
        self.st.read_range_into_cached(sc, 0, &mut scale)?;
        EXPERT_READ_PROF.scale_ns.fetch_add(t_s.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.scale_bytes.fetch_add(sc.len, Relaxed);
        let t_repack = std::time::Instant::now();
        EXPERT_READ_PROF.pread_ns.fetch_add(
            t_repack.duration_since(t_pread).as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.pread_bytes.fetch_add(wt.len + sc.len, Relaxed);
        assert_eq!(nb % 8, 0, "MXFP4 super-block layout needs nb % 8 == 0, got {nb}");
        for r in 0..out {
            let prow = &packed[r * half..(r + 1) * half];
            let srow = &scale[r * nb..(r + 1) * nb];
            let drow = &mut dst[r * nb * 17..(r + 1) * nb * 17];
            // SUPER-BLOCK v2 (must match `mxfp4_repack.hip` exactly): each
            // 136-byte super-block is [8 x 16 B nibbles][8 B scales]. Same size
            // as the old 8 x [scale][16 nibbles]; the matvecs read nibbles 8
            // bytes at a time, which the interleaved form made impossible.
            for (sb, sup) in drow.chunks_exact_mut(136).enumerate() {
                for k in 0..8 {
                    let j = sb * 8 + k;
                    sup[128 + k] = srow[j];
                    let pb = &prow[j * 16..(j + 1) * 16];
                    for i in 0..8 {
                        // elements 2i, 2i+1 (low half) and 16+2i, 16+2i+1 (high half)
                        let lo = pb[i];
                        let hi = pb[8 + i];
                        sup[k * 16 + 2 * i] = (lo & 0x0F) | (hi << 4);
                        sup[k * 16 + 1 + 2 * i] = (lo >> 4) | (hi & 0xF0);
                    }
                }
            }
        }
        EXPERT_READ_PROF.repack_ns.fetch_add(t_repack.elapsed().as_nanos() as u64, Relaxed);
        EXPERT_READ_PROF.calls.fetch_add(1, Relaxed);
        Ok(())
    }
}

/// Per-role expert-read phase profile (M8 measurement E).
///
/// One `read_expert_raw` = two fresh host allocations, two cached preads of the
/// HF shard, and a scalar nibble repack (`out × nb × 8` iterations) from HF's
/// `[out, in/2]` byte order into ggml's 17-byte MXFP4 blocks. The pager's
/// "8 ms read" bucket is all three; `read_ns` alone cannot tell which the SSD
/// owns and which the CPU owns. These counters split it. Free when unread
/// (four relaxed adds per 6.3 MB role).
#[derive(Default)]
pub struct ExpertReadProfile {
    pub calls: AtomicU64,
    pub alloc_ns: AtomicU64,
    pub pread_ns: AtomicU64,
    pub repack_ns: AtomicU64,
    pub pread_bytes: AtomicU64,
    /// `pread_ns` split: the O_DIRECT weight read vs the BUFFERED scale read.
    /// `pread_ns - weight_ns - scale_ns` is the setup left over.
    pub weight_ns: AtomicU64,
    pub weight_bytes: AtomicU64,
    pub scale_ns: AtomicU64,
    pub scale_bytes: AtomicU64,
}

pub static EXPERT_READ_PROF: ExpertReadProfile = ExpertReadProfile {
    calls: AtomicU64::new(0),
    alloc_ns: AtomicU64::new(0),
    pread_ns: AtomicU64::new(0),
    repack_ns: AtomicU64::new(0),
    pread_bytes: AtomicU64::new(0),
    weight_ns: AtomicU64::new(0),
    weight_bytes: AtomicU64::new(0),
    scale_ns: AtomicU64::new(0),
    scale_bytes: AtomicU64::new(0),
};

/// `(calls, alloc_ns, pread_ns, repack_ns, pread_bytes)` — cumulative.
/// The `pread_ns` split: `(weight_ns, weight_bytes, scale_ns, scale_bytes)`.
/// Separate from `expert_read_profile` so the existing 5-tuple's arity — and its
/// four call sites — stay untouched.
pub fn expert_read_split() -> (u64, u64, u64, u64) {
    (
        EXPERT_READ_PROF.weight_ns.load(Relaxed),
        EXPERT_READ_PROF.weight_bytes.load(Relaxed),
        EXPERT_READ_PROF.scale_ns.load(Relaxed),
        EXPERT_READ_PROF.scale_bytes.load(Relaxed),
    )
}

pub fn expert_read_profile() -> (u64, u64, u64, u64, u64) {
    (
        EXPERT_READ_PROF.calls.load(Relaxed),
        EXPERT_READ_PROF.alloc_ns.load(Relaxed),
        EXPERT_READ_PROF.pread_ns.load(Relaxed),
        EXPERT_READ_PROF.repack_ns.load(Relaxed),
        EXPERT_READ_PROF.pread_bytes.load(Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kquants::f16_to_f32;

    #[test]
    fn e4m3_values() {
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0x08), 2f32.powi(-6));
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x80).to_bits(), (-0.0f32).to_bits());
        assert!(e4m3_to_f32(0x7F).is_nan() && e4m3_to_f32(0xFF).is_nan());
    }

    #[test]
    fn e8m0_values() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
        assert_eq!(e8m0_to_f32(254), 2f32.powi(127));
        assert!(e8m0_to_f32(255).is_nan());
    }

    #[test]
    fn q8_block_reference() {
        let mut x = [0f32; 32];
        x[0] = 254.0;
        x[1] = -1.0;
        x[2] = 0.75;
        x[3] = -0.75;
        let mut o = [0u8; 34];
        quantize_q8_0_block(&x, &mut o);
        assert_eq!(f16_to_f32(u16::from_le_bytes([o[0], o[1]])), 2.0);
        assert_eq!(o[2] as i8, 127);
        assert_eq!(o[3] as i8, -1); // -0.5 rounds away from zero
        assert_eq!(o[4] as i8, 0); // 0.375
        assert_eq!(o[5] as i8, 0);
        let z = [0f32; 32];
        quantize_q8_0_block(&z, &mut o);
        assert!(o.iter().all(|&b| b == 0));
    }

    #[test]
    fn mxfp4_repack_nibble_placement() {
        // Elements k = 0..31 with value k & 0xF, HF-packed: element 2i in the
        // low nibble of byte i.
        let packed: Vec<u8> = (0..16u8).map(|i| ((2 * i) & 0xF) | (((2 * i + 1) & 0xF) << 4)).collect();
        // v2 super-block: nibbles for block k at k*16, scale at 128+k. This
        // test covers block 0, so nibbles are at offset 0.
        let mut sup = [0u8; 136];
        sup[128] = 0x7F;
        for i in 0..8 {
            let lo = packed[i];
            let hi = packed[8 + i];
            sup[2 * i] = (lo & 0x0F) | (hi << 4);
            sup[1 + 2 * i] = (lo >> 4) | (hi & 0xF0);
        }
        // ggml nibble order is unchanged: byte i holds element i (low) and
        // element 16+i (high).
        for i in 0..16 {
            let lo = sup[i] & 0xF;
            let hi = sup[i] >> 4;
            assert_eq!(lo as usize, i & 0xF, "elem {i}");
            assert_eq!(hi as usize, (16 + i) & 0xF, "elem {}", 16 + i);
        }
    }
}
