//! Laguna-S-2.1 — real, on-device, runnable forward with a KV cache.
//!
//! This promotes the throwaway correctness spike (`tests/laguna_spike.rs`,
//! host-side norm/rope/router/gate/softmax) into a device forward: everything
//! perf-relevant runs on the GPU. The math is unchanged from the spike; only
//! the *placement* moves host -> device.
//!
//! ON DEVICE (this module):
//!   - RMSNorm (attn/ffn/output + per-head QK-norm) via [`crate::RmsNorm`]
//!   - q/k/v/o/gate projections via [`crate::F16Matvec`] (F16 weights)
//!   - RoPE (YaRN + mscale on full layers, plain NEOX on SWA) via [`crate::RopeTail`]
//!   - GQA single-query attention via [`crate::GqaAttention`]
//!   - softplus per-head attention gate  (NEW kernel `laguna_softplus_gate`)
//!   - sigmoid router + top-k + weights   (NEW kernel `laguna_router`, folds the
//!     F32 router matvec)
//!   - dense/expert/shared SwiGLU FFN via Q4_K/Q6_K matvecs + batched MoE
//!   - KV append as an f16 cast into a persistent per-layer cache
//!
//! EXPERTS RESIDENT: at load, ALL routed experts (256 × 47 MoE layers,
//! ~59 GB) are copied to device buffers once. Decode/prefill index them
//! by the on-device selected ids — no per-token mmap gather, no H2D. This
//! is the big decode lever (was streaming ~2.8 GB/token). Requires a
//! large-VRAM device (the iGPU with `no_system_mem_limit`); the dGPU's
//! 16 GB cannot hold the experts.
//!
//! STILL HOST (negligible / unavoidable for this quant map):
//!   - token-embedding row dequant (Q4_K, one row per token)
//!   - final argmax (400 KB D->H per generated token)
//!
//! Not a full ds4 `Model` abstraction — a clean-ish standalone module that
//! could become one. See `tests/laguna_decode.rs` for the runnable driver.

use std::fs::File;
use std::os::unix::fs::FileExt;

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::{GgufType, GgufValue};
use v4flash_core::MappedGguf;
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

use crate::het::graph_cache::GraphCache;
use crate::iq2_xxs_tables::f16_to_f32;
use crate::{
    F16Matvec, GqaAttention, Q4KMatvec, Q4_KDenseMatvec, Q6KMatvec, Q6_KDenseMatvec, RmsNorm,
    RopeParams, RopeTail, Swiglu, Q8KQuantize, VecAddInplace,
};

pub const HIDDEN: usize = 3072;
pub const HEAD_DIM: usize = 128;
pub const N_KV_HEAD: usize = 8;
pub const N_LAYER: usize = 48;
pub const N_EXPERT: usize = 256;
pub const TOPK: usize = 10;
pub const FF_EXP: usize = 1024;
pub const FF_SHEXP: usize = 1024;
pub const FF_DENSE: usize = 12288;
pub const VOCAB: usize = 100352;
pub const EPS: f32 = 1e-6;

// --------------------------------------------------------------------------
// New Laguna kernels (kernels/laguna_ops.hip).
// --------------------------------------------------------------------------
pub struct LagunaOps {
    module: Module,
}

impl LagunaOps {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            include_bytes!(env!("KERNEL_LAGUNA_OPS_GFX1201"))
        } else if arch.starts_with("gfx1151") {
            include_bytes!(env!("KERNEL_LAGUNA_OPS_GFX1151"))
        } else {
            return Err(eyre!("unsupported arch for laguna_ops: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// Fused sigmoid router: matvec(router_w · x) -> sigmoid -> +bias ->
    /// top-`n_used` -> sum-normalized weights × `scale`.
    #[allow(clippy::too_many_arguments)]
    pub fn router(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        weights: &mut DeviceBuffer<f32>,
        router_w: &DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        n_expert: u32,
        hidden: u32,
        n_used: u32,
        scale: f32,
        weight_eps: f32,
    ) -> eyre::Result<()> {
        let f = self.module.get_function("laguna_router")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [
            selected.raw(), weights.raw(), router_w.raw(), x.raw(), bias.raw(),
            n_expert, hidden, n_used, scale, weight_eps
        ])
    }

    /// Multi-WG router: scores kernel (grid = n_expert) computes
    /// probs/scores, then a single-WG top-k kernel selects + weights.
    /// Same numerics as [`router`] but the 256×hidden matvec is spread
    /// across the grid instead of serialized on one WG.
    #[allow(clippy::too_many_arguments)]
    pub fn router_split(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        weights: &mut DeviceBuffer<f32>,
        probs: &mut DeviceBuffer<f32>,
        scores: &mut DeviceBuffer<f32>,
        router_w: &DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        n_expert: u32,
        hidden: u32,
        n_used: u32,
        scale: f32,
        weight_eps: f32,
    ) -> eyre::Result<()> {
        let fs = self.module.get_function("laguna_router_scores")?;
        let cfg_s = LaunchConfig { grid: (n_expert, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(fs, cfg_s, stream, [
            probs.raw(), scores.raw(), router_w.raw(), x.raw(), bias.raw(), n_expert, hidden
        ])?;
        let ft = self.module.get_function("laguna_router_topk")?;
        let cfg_t = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(ft, cfg_t, stream, [
            selected.raw(), weights.raw(), probs.raw(), scores.raw(), n_expert, n_used, scale, weight_eps
        ])
    }

    /// BATCHED multi-WG router (grid.z = B). Same numerics as
    /// [`router_split`] for each of the B tokens, in two launches instead of
    /// 2*B. Tight layouts: `x[B,hidden]`, `probs`/`scores[B,n_expert]`,
    /// `selected`/`weights[B,n_used]`. Router weight + bias shared across B.
    #[allow(clippy::too_many_arguments)]
    pub fn router_split_batched(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        weights: &mut DeviceBuffer<f32>,
        probs: &mut DeviceBuffer<f32>,
        scores: &mut DeviceBuffer<f32>,
        router_w: &DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        n_expert: u32,
        hidden: u32,
        n_used: u32,
        scale: f32,
        weight_eps: f32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        // Router scores: default WEIGHT-READ-ONCE (grid over experts, weight in
        // registers, reused across B tokens — bit-identical, 1× vs B× DRAM
        // weight traffic). Set LAGUNA_ROUTER_WRO=0 to fall back to the old
        // grid.z=B kernel that re-reads the router weight per token.
        let variant = std::env::var("LAGUNA_ROUTER_WRO").unwrap_or_else(|_| "warp".to_string());
        match variant.as_str() {
            // read-once-in-registers (grid over experts, weight in regs). Kept
            // for A/B; MEASURED slower than baseline (latency-bound, not BW).
            "wro" | "1" => {
                let fs = self.module.get_function("laguna_router_scores_batched_wro")?;
                let cfg_s = LaunchConfig { grid: (n_expert, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
                launch_kernel!(fs, cfg_s, stream, [
                    probs.raw(), scores.raw(), router_w.raw(), x.raw(), bias.raw(), n_expert, hidden, batch
                ])?;
            }
            // old grid.z=B, 256-thread LDS tree reduction.
            "0" | "old" => {
                let fs = self.module.get_function("laguna_router_scores_batched")?;
                let cfg_s = LaunchConfig { grid: (n_expert, 1, batch), block: (256, 1, 1), shared_mem_bytes: 0 };
                launch_kernel!(fs, cfg_s, stream, [
                    probs.raw(), scores.raw(), router_w.raw(), x.raw(), bias.raw(), n_expert, hidden
                ])?;
            }
            // default: warp-shuffle (1 warp per (expert,token), no LDS/barrier).
            _ => {
                let fs = self.module.get_function("laguna_router_scores_batched_warp")?;
                let zdim = batch.div_ceil(8);
                let cfg_s = LaunchConfig { grid: (n_expert, 1, zdim), block: (256, 1, 1), shared_mem_bytes: 0 };
                launch_kernel!(fs, cfg_s, stream, [
                    probs.raw(), scores.raw(), router_w.raw(), x.raw(), bias.raw(), n_expert, hidden, batch
                ])?;
            }
        }
        let ft = self.module.get_function("laguna_router_topk_batched")?;
        let cfg_t = LaunchConfig { grid: (1, 1, batch), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(ft, cfg_t, stream, [
            selected.raw(), weights.raw(), probs.raw(), scores.raw(), n_expert, n_used, scale, weight_eps
        ])
    }

    /// Per-head softplus attention gate, in place: `attn_out[h,:] *=
    /// softplus(gate_logits[h])`.
    pub fn softplus_gate(
        &self,
        stream: &Stream,
        attn_out: &mut DeviceBuffer<f32>,
        gate_logits: &DeviceBuffer<f32>,
        n_head: u32,
        head_dim: u32,
    ) -> eyre::Result<()> {
        let f = self.module.get_function("laguna_softplus_gate")?;
        let cfg = LaunchConfig { grid: (n_head, 1, 1), block: (head_dim, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [attn_out.raw(), gate_logits.raw(), n_head, head_dim])
    }

    /// Fused per-head QK RMSNorm: normalize each of `n_head` `head_dim`-wide
    /// slices of `input` by its own RMS, scale by the shared `weight[head_dim]`,
    /// write to `out`. Replaces the `n_head` per-slice `RmsNorm::launch_weighted`
    /// launches with a single grid launch (one WG per head). `out`/`input` may
    /// be larger scratch buffers; only the first `n_head*head_dim` are touched.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_rmsnorm(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        n_head: u32,
        head_dim: u32,
        eps: f32,
    ) -> eyre::Result<()> {
        let n = (n_head * head_dim) as usize;
        if out.len() < n || input.len() < n || weight.len() != head_dim as usize {
            return Err(eyre!(
                "qk_rmsnorm len mismatch: n_head={n_head} head_dim={head_dim}, out={} in={} w={}",
                out.len(), input.len(), weight.len()
            ));
        }
        let f = self.module.get_function("laguna_qk_rmsnorm")?;
        let cfg = LaunchConfig { grid: (n_head, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [out.raw(), input.raw(), weight.raw(), n_head, head_dim, eps])
    }

    /// Contiguous f32 -> f16 cast.
    pub fn cast_f16(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u16>,
        input: &DeviceBuffer<f32>,
        n: u32,
    ) -> eyre::Result<()> {
        let f = self.module.get_function("laguna_cast_f32_f16")?;
        let block = 256u32;
        let cfg = LaunchConfig { grid: (n.div_ceil(block), 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [out.raw(), input.raw(), n])
    }

    /// FP8 (e4m3fn) KV-cache quantize (LAGUNA_FP8_KV). Quantizes `n_rows`
    /// contiguous f32 rows of `head_dim` (<=128) into 1-byte e4m3fn (`out`) plus
    /// one f32 per-row symmetric scale (`scale`, `amax/448`). One WG per row.
    pub fn quantize_fp8_kv(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u8>,
        scale: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        n_rows: u32,
        head_dim: u32,
    ) -> eyre::Result<()> {
        debug_assert!(head_dim <= 128, "quantize_fp8_kv head_dim={head_dim} > 128");
        let n = (n_rows * head_dim) as usize;
        if out.len() < n || scale.len() < n_rows as usize || input.len() < n {
            return Err(eyre!(
                "quantize_fp8_kv len mismatch: n_rows={n_rows} head_dim={head_dim}, out={} scale={} in={}",
                out.len(), scale.len(), input.len()
            ));
        }
        let f = self.module.get_function("laguna_quantize_fp8_kv")?;
        let cfg = LaunchConfig { grid: (n_rows, 1, 1), block: (128, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [out.raw(), scale.raw(), input.raw(), n_rows, head_dim])
    }

    /// FP8 (e4m3fn) FAKE-QUANT round-trip (LAGUNA_FP8_FAKE, diagnostic). Quantizes
    /// each `head_dim` row to e4m3fn + per-row amax/448 scale and dequantizes back
    /// to f16 in `out`, so downstream attention uses the normal f16 read kernels.
    /// Isolates the pure numerical error of fp8 (K/V independently). One WG/row.
    pub fn roundtrip_fp8_kv(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u16>,
        input: &DeviceBuffer<f32>,
        n_rows: u32,
        head_dim: u32,
        blk: u32,
        fmt: u32,
    ) -> eyre::Result<()> {
        debug_assert!(head_dim <= 128, "roundtrip_fp8_kv head_dim={head_dim} > 128");
        let n = (n_rows * head_dim) as usize;
        if out.len() < n || input.len() < n {
            return Err(eyre!(
                "roundtrip_fp8_kv len mismatch: n_rows={n_rows} head_dim={head_dim}, out={} in={}",
                out.len(), input.len()
            ));
        }
        let f = self.module.get_function("laguna_fp8_roundtrip_kv")?;
        let cfg = LaunchConfig { grid: (n_rows, 1, 1), block: (128, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [out.raw(), input.raw(), n_rows, head_dim, blk, fmt])
    }
}

// --------------------------------------------------------------------------
// Hyperparameters read from the GGUF header.
// --------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct LagunaHparams {
    pub freq_base: f32,
    pub freq_base_swa: f32,
    pub factor: f32,
    pub orig_ctx: u64,
    pub yarn_attn: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub n_rot_full: usize,
    pub n_rot_swa: usize,
    pub moe_scale: f32,
}

fn meta_f32(g: &v4flash_core::gguf::Gguf, key: &str) -> Option<f32> {
    match g.metadata(key)? {
        GgufValue::F32(v) => Some(*v),
        GgufValue::F64(v) => Some(*v as f32),
        GgufValue::U32(v) => Some(*v as f32),
        GgufValue::I32(v) => Some(*v as f32),
        _ => None,
    }
}

impl LagunaHparams {
    pub fn from_gguf(g: &v4flash_core::gguf::Gguf) -> Self {
        Self {
            freq_base: meta_f32(g, "laguna.rope.freq_base").unwrap_or(500000.0),
            freq_base_swa: meta_f32(g, "laguna.rope.freq_base_swa").unwrap_or(10000.0),
            factor: meta_f32(g, "laguna.rope.scaling.factor").unwrap_or(32.0),
            orig_ctx: meta_f32(g, "laguna.rope.scaling.original_context_length").unwrap_or(8192.0) as u64,
            yarn_attn: meta_f32(g, "laguna.rope.scaling.yarn_attn_factor").unwrap_or(1.0),
            beta_fast: meta_f32(g, "laguna.rope.scaling.yarn_beta_fast").unwrap_or(32.0),
            beta_slow: meta_f32(g, "laguna.rope.scaling.yarn_beta_slow").unwrap_or(1.0),
            n_rot_full: meta_f32(g, "laguna.rope.dimension_count").unwrap_or(64.0) as usize,
            n_rot_swa: meta_f32(g, "laguna.rope.dimension_count_swa").unwrap_or(128.0) as usize,
            moe_scale: meta_f32(g, "laguna.expert_weights_scale").unwrap_or(2.5),
        }
    }

    /// YaRN full-layer rope. `attn_factor`+`freq_scale` feed the kernel's
    /// `mscale_eff = attn_factor·(1 + 0.1·ln(1/freq_scale))` = 1.3466, matching
    /// the spike's host rope exactly.
    pub(crate) fn rope_full(&self) -> RopeParams {
        RopeParams {
            freq_base: self.freq_base,
            freq_scale: 1.0 / self.factor,
            ext_factor: 1.0,
            attn_factor: self.yarn_attn,
            beta_fast: self.beta_fast,
            beta_slow: self.beta_slow,
            n_ctx_orig: self.orig_ctx,
        }
    }
    pub(crate) fn rope_swa(&self) -> RopeParams {
        RopeParams {
            freq_base: self.freq_base_swa,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: self.beta_fast,
            beta_slow: self.beta_slow,
            n_ctx_orig: 0,
        }
    }
}

// --------------------------------------------------------------------------
// Per-layer resident (non-expert) weights on device.
// --------------------------------------------------------------------------
pub(crate) struct QWeight {
    pub(crate) bytes: DeviceBuffer<u8>,
    pub(crate) dtype: GgufType,
}

struct LayerWeights {
    is_full: bool,
    n_head: usize,
    // norms
    attn_norm: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    // attention (F16 raw bytes)
    wq: DeviceBuffer<u8>,
    wk: DeviceBuffer<u8>,
    wv: DeviceBuffer<u8>,
    wo: DeviceBuffer<u8>,
    wg: DeviceBuffer<u8>,
    // FFN
    dense: Option<(QWeight, QWeight, QWeight)>, // gate, up, down  (layer 0)
    moe: Option<MoeWeights>,                    // layers >= 1
}

#[allow(dead_code)] // gate_dt/up_dt kept for provenance; kernel assumes Q4_K
struct MoeWeights {
    router: DeviceBuffer<f32>, // [n_expert, hidden]
    bias: DeviceBuffer<f32>,   // [n_expert]
    // shared expert
    sh_gate: QWeight,
    sh_up: QWeight,
    sh_down: QWeight,
    // ALL routed experts, resident on device (no per-token streaming).
    // Each is the full GGUF tensor `[N_EXPERT * <stride>]` bytes; the MoE
    // kernels index row `e` at byte offset `e * <stride>`.
    gate_all: DeviceBuffer<u8>,
    up_all: DeviceBuffer<u8>,
    down_all: DeviceBuffer<u8>,
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
    gate_dt: GgufType,
    up_dt: GgufType,
    down_dt: GgufType,
}

pub(crate) fn block_bytes(dt: GgufType) -> usize {
    match dt {
        GgufType::Q4_K => 144,
        GgufType::Q6_K => 210,
        _ => 0,
    }
}

// --------------------------------------------------------------------------
// Reusable device scratch (sized for the widest layer, SWA n_head=72).
// --------------------------------------------------------------------------
struct Scratch {
    h: DeviceBuffer<f32>,       // running hidden [HIDDEN]
    ain: DeviceBuffer<f32>,     // attn_norm out
    q: DeviceBuffer<f32>,       // [n_embd_q_max]
    qn: DeviceBuffer<f32>,
    qf: DeviceBuffer<u16>,
    k: DeviceBuffer<f32>,       // [N_KV_HEAD*HEAD_DIM]
    kn: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    od: DeviceBuffer<f32>,      // attention output [n_embd_q_max]
    gate_logits: DeviceBuffer<f32>, // [n_head_max]
    op: DeviceBuffer<f32>,      // o_proj / ffn_inp (residual carrier) [HIDDEN]
    fn_in: DeviceBuffer<f32>,   // ffn_norm out [HIDDEN]
    ffn_out: DeviceBuffer<f32>, // [HIDDEN]
    // dense FFN
    gate_big: DeviceBuffer<f32>, // [FF_DENSE]
    up_big: DeviceBuffer<f32>,
    sw_big: DeviceBuffer<f32>,
    // MoE
    sel: DeviceBuffer<i32>,     // [TOPK] selected expert ids (device-resident)
    ew: DeviceBuffer<f32>,      // routing weights [TOPK]
    router_probs: DeviceBuffer<f32>, // [N_EXPERT] sigmoid probs (router_split)
    router_scores: DeviceBuffer<f32>, // [N_EXPERT] prob+bias (router_split)
    xq_hidden: DeviceBuffer<u8>,   // q8k(fn_in) [12*292]
    mid: DeviceBuffer<f32>,     // [TOPK*FF_EXP]
    xq_mid: DeviceBuffer<u8>,   // [TOPK*4*292]
    acc: DeviceBuffer<f32>,     // routed sum [HIDDEN]
    gate_s: DeviceBuffer<f32>,  // [FF_SHEXP]
    up_s: DeviceBuffer<f32>,
    sw_s: DeviceBuffer<f32>,
    down_s: DeviceBuffer<f32>,  // [HIDDEN]
    // output
    rn: DeviceBuffer<f32>,      // output_norm out
    logits: DeviceBuffer<f32>,  // [VOCAB]
}

// --------------------------------------------------------------------------
// The model: resident weights + kernels + scratch + KV cache.
// --------------------------------------------------------------------------
#[allow(dead_code)] // dev/gguf/max_kv retained to own resources / for clarity
pub struct LagunaModel {
    dev: i32,
    stream: Stream,
    gguf: MappedGguf,
    raw_file: File,
    hp: LagunaHparams,
    rope_full: RopeParams,
    rope_swa: RopeParams,

    // kernels
    f16: F16Matvec,
    q4d: Q4_KDenseMatvec,
    q6d: Q6_KDenseMatvec,
    gqa: GqaAttention,
    rms: RmsNorm,
    rope: RopeTail,
    q8k: Q8KQuantize,
    q4b: Q4KMatvec,
    q6b: Q6KMatvec,
    ops: LagunaOps,
    swiglu: Swiglu,
    vadd: VecAddInplace,

    // weights
    layers: Vec<LayerWeights>,
    output_norm: DeviceBuffer<f32>,
    output_w: QWeight,
    tok_embd_off: u64,
    tok_embd_row_bytes: usize,

    // KV cache: per layer [max_kv, N_KV_HEAD, HEAD_DIM] f16 (k and v)
    max_kv: usize,
    kc: Vec<DeviceBuffer<u16>>,
    vc: Vec<DeviceBuffer<u16>>,
    kv_len: usize, // current filled positions

    scratch: Scratch,

    // HIP-graph cache for the pos-independent per-layer sub-chains (Fix 2).
    // Keyed by (stage, layer); captured once (first token / prefill) and
    // replayed per token to remove per-launch host dispatch overhead. The
    // pos-dependent kernels (rope, KV-append at pos offset, attention over
    // n_kv) stay OUTSIDE the graphs. Disabled via LAGUNA_DECODE_GRAPH=0.
    graphs: GraphCache,
    use_graph: bool,
}

impl LagunaModel {
    pub fn load(gguf_path: &str, dev: i32, arch: &str, max_kv: usize) -> eyre::Result<Self> {
        let gguf = MappedGguf::open(gguf_path)?;
        let raw_file = File::open(gguf_path)?;
        let hp = LagunaHparams::from_gguf(gguf.gguf());
        let rope_full = hp.rope_full();
        let rope_swa = hp.rope_swa();

        let mk_qweight = |name: &str| -> eyre::Result<QWeight> {
            let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
            let bytes = gguf.read_tensor(t)?;
            let mut b = DeviceBuffer::<u8>::new(dev, bytes.len())?;
            b.copy_from_host(&bytes)?;
            Ok(QWeight { bytes: b, dtype: t.dtype })
        };
        let mk_u8 = |name: &str| -> eyre::Result<DeviceBuffer<u8>> {
            let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
            if t.dtype != GgufType::F16 {
                return Err(eyre!("{name} expected F16, got {:?}", t.dtype));
            }
            let bytes = gguf.read_tensor(t)?;
            let mut b = DeviceBuffer::<u8>::new(dev, bytes.len())?;
            b.copy_from_host(&bytes)?;
            Ok(b)
        };
        let mk_f32 = |name: &str| -> eyre::Result<DeviceBuffer<f32>> {
            let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
            if t.dtype != GgufType::F32 {
                return Err(eyre!("{name} expected F32, got {:?}", t.dtype));
            }
            let bytes = gguf.read_tensor(t)?;
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut b = DeviceBuffer::<f32>::new(dev, v.len())?;
            b.copy_from_host(&v)?;
            Ok(b)
        };

        // per-layer weights
        let mut layers = Vec::with_capacity(N_LAYER);
        for il in 0..N_LAYER {
            let is_full = il % 4 == 0;
            let n_head = if is_full { 48 } else { 72 };
            let p = |s: &str| format!("blk.{il}.{s}");

            let dense = if il == 0 {
                Some((
                    mk_qweight(&p("ffn_gate.weight"))?,
                    mk_qweight(&p("ffn_up.weight"))?,
                    mk_qweight(&p("ffn_down.weight"))?,
                ))
            } else {
                None
            };
            let moe = if il == 0 {
                None
            } else {
                let g = gguf.gguf();
                let gate_t = g.tensor(&p("ffn_gate_exps.weight")).unwrap();
                let up_t = g.tensor(&p("ffn_up_exps.weight")).unwrap();
                let down_t = g.tensor(&p("ffn_down_exps.weight")).unwrap();
                let gate_stride = FF_EXP * (HIDDEN / 256) * block_bytes(gate_t.dtype);
                let up_stride = FF_EXP * (HIDDEN / 256) * block_bytes(up_t.dtype);
                let down_stride = HIDDEN * (FF_EXP / 256) * block_bytes(down_t.dtype);
                let (gate_dt, up_dt, down_dt) = (gate_t.dtype, up_t.dtype, down_t.dtype);
                // Resident copy of the full per-layer expert tensors. Each is
                // `N_EXPERT * <stride>` bytes; the kernels index expert `e` at
                // `e * <stride>`. This replaces the per-token mmap gather +
                // H2D copy that dominated decode latency.
                let mk_resident = |name: &str, stride: usize| -> eyre::Result<DeviceBuffer<u8>> {
                    let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
                    let bytes = gguf.read_tensor(t)?;
                    let want = N_EXPERT * stride;
                    if bytes.len() != want {
                        return Err(eyre!(
                            "{name}: expected {want} bytes ({N_EXPERT}*{stride}), got {}",
                            bytes.len()
                        ));
                    }
                    let mut b = DeviceBuffer::<u8>::new(dev, bytes.len())?;
                    b.copy_from_host(&bytes)?;
                    Ok(b)
                };
                Some(MoeWeights {
                    router: mk_f32(&p("ffn_gate_inp.weight"))?,
                    bias: mk_f32(&p("exp_probs_b.bias"))?,
                    sh_gate: mk_qweight(&p("ffn_gate_shexp.weight"))?,
                    sh_up: mk_qweight(&p("ffn_up_shexp.weight"))?,
                    sh_down: mk_qweight(&p("ffn_down_shexp.weight"))?,
                    gate_all: mk_resident(&p("ffn_gate_exps.weight"), gate_stride)?,
                    up_all: mk_resident(&p("ffn_up_exps.weight"), up_stride)?,
                    down_all: mk_resident(&p("ffn_down_exps.weight"), down_stride)?,
                    gate_stride,
                    up_stride,
                    down_stride,
                    gate_dt,
                    up_dt,
                    down_dt,
                })
            };

            layers.push(LayerWeights {
                is_full,
                n_head,
                attn_norm: mk_f32(&p("attn_norm.weight"))?,
                ffn_norm: mk_f32(&p("ffn_norm.weight"))?,
                q_norm: mk_f32(&p("attn_q_norm.weight"))?,
                k_norm: mk_f32(&p("attn_k_norm.weight"))?,
                wq: mk_u8(&p("attn_q.weight"))?,
                wk: mk_u8(&p("attn_k.weight"))?,
                wv: mk_u8(&p("attn_v.weight"))?,
                wo: mk_u8(&p("attn_output.weight"))?,
                wg: mk_u8(&p("attn_gate.weight"))?,
                dense,
                moe,
            });
        }

        let output_norm = mk_f32("output_norm.weight")?;
        let output_w = mk_qweight("output.weight")?;
        let tok_embd_t = gguf.gguf().tensor("token_embd.weight").ok_or_else(|| eyre!("no token_embd"))?;
        let tok_embd_off = tok_embd_t.abs_offset;
        let tok_embd_row_bytes = (HIDDEN / 256) * 144;

        // KV cache
        let mut kc = Vec::with_capacity(N_LAYER);
        let mut vc = Vec::with_capacity(N_LAYER);
        for _ in 0..N_LAYER {
            kc.push(DeviceBuffer::<u16>::new(dev, max_kv * N_KV_HEAD * HEAD_DIM)?);
            vc.push(DeviceBuffer::<u16>::new(dev, max_kv * N_KV_HEAD * HEAD_DIM)?);
        }

        // scratch
        let n_embd_q_max = 72 * HEAD_DIM;
        let mk = |n: usize| DeviceBuffer::<f32>::new(dev, n);
        let scratch = Scratch {
            h: mk(HIDDEN)?,
            ain: mk(HIDDEN)?,
            q: mk(n_embd_q_max)?,
            qn: mk(n_embd_q_max)?,
            qf: DeviceBuffer::<u16>::new(dev, n_embd_q_max)?,
            k: mk(N_KV_HEAD * HEAD_DIM)?,
            kn: mk(N_KV_HEAD * HEAD_DIM)?,
            v: mk(N_KV_HEAD * HEAD_DIM)?,
            od: mk(n_embd_q_max)?,
            gate_logits: mk(72)?,
            op: mk(HIDDEN)?,
            fn_in: mk(HIDDEN)?,
            ffn_out: mk(HIDDEN)?,
            gate_big: mk(FF_DENSE)?,
            up_big: mk(FF_DENSE)?,
            sw_big: mk(FF_DENSE)?,
            sel: DeviceBuffer::<i32>::new(dev, TOPK)?,
            ew: mk(TOPK)?,
            router_probs: mk(N_EXPERT)?,
            router_scores: mk(N_EXPERT)?,
            xq_hidden: DeviceBuffer::<u8>::new(dev, (HIDDEN / 256) * 292)?,
            mid: mk(TOPK * FF_EXP)?,
            xq_mid: DeviceBuffer::<u8>::new(dev, TOPK * (FF_EXP / 256) * 292)?,
            acc: mk(HIDDEN)?,
            gate_s: mk(FF_SHEXP)?,
            up_s: mk(FF_SHEXP)?,
            sw_s: mk(FF_SHEXP)?,
            down_s: mk(HIDDEN)?,
            rn: mk(HIDDEN)?,
            logits: mk(VOCAB)?,
        };

        let stream = Stream::new(dev)?;
        Ok(Self {
            dev,
            stream,
            gguf,
            raw_file,
            hp,
            rope_full,
            rope_swa,
            f16: F16Matvec::for_arch(arch)?,
            q4d: Q4_KDenseMatvec::for_arch(arch)?,
            q6d: Q6_KDenseMatvec::for_arch(arch)?,
            gqa: GqaAttention::for_arch(arch)?,
            rms: RmsNorm::for_arch(arch)?,
            rope: RopeTail::for_arch(arch)?,
            q8k: Q8KQuantize::for_arch(arch)?,
            q4b: Q4KMatvec::for_arch(arch)?,
            q6b: Q6KMatvec::for_arch(arch)?,
            ops: LagunaOps::for_arch(arch)?,
            swiglu: Swiglu::for_arch(arch)?,
            vadd: VecAddInplace::for_arch(arch)?,
            layers,
            output_norm,
            output_w,
            tok_embd_off,
            tok_embd_row_bytes,
            max_kv,
            kc,
            vc,
            kv_len: 0,
            scratch,
            graphs: GraphCache::new(),
            use_graph: std::env::var("LAGUNA_DECODE_GRAPH").map(|v| v != "0").unwrap_or(true),
        })
    }

    pub fn hparams(&self) -> &LagunaHparams {
        &self.hp
    }

    pub fn reset(&mut self) {
        self.kv_len = 0;
    }

    /// Host Q4_K dequant of one token-embedding row -> device hidden.
    fn embed(&mut self, tok_id: usize) -> eyre::Result<()> {
        let mut rb = vec![0u8; self.tok_embd_row_bytes];
        self.raw_file
            .read_exact_at(&mut rb, self.tok_embd_off + (tok_id as u64) * self.tok_embd_row_bytes as u64)?;
        let mut row = vec![0f32; HIDDEN];
        for sb in 0..(HIDDEN / 256) {
            dequant_q4k_superblock(&rb[sb * 144..(sb + 1) * 144], &mut row[sb * 256..(sb + 1) * 256]);
        }
        self.scratch.h.copy_from_host(&row)?;
        Ok(())
    }

    /// One full transformer layer at position `pos`, updating `scratch.h` in
    /// place and appending K/V to layer `il`'s cache.
    fn layer(&mut self, il: usize, pos: usize) -> eyre::Result<()> {
        let s = &mut self.scratch;
        let lw = &self.layers[il];
        let n_head = lw.n_head;
        let n_embd_q = n_head * HEAD_DIM;
        let (rope, n_rot) = if lw.is_full {
            (&self.rope_full, self.hp.n_rot_full as u32)
        } else {
            (&self.rope_swa, self.hp.n_rot_swa as u32)
        };
        let n_kv = pos + 1;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        // Bind field references to locals so the graph-capture closures can
        // borrow disjoint fields (kernels + scratch) without capturing all
        // of `self` (which would collide with the `self.graphs` receiver).
        let f16 = &self.f16;
        let ops = &self.ops;
        let rms = &self.rms;
        let vadd = &self.vadd;
        let q8k = &self.q8k;
        let q4b = &self.q4b;
        let q6b = &self.q6b;
        let q4d = &self.q4d;
        let q6d = &self.q6d;
        let swiglu = &self.swiglu;
        let gqa = &self.gqa;
        let rope_kern = &self.rope;
        let stream = &self.stream;
        let graphs = &self.graphs;
        let use_graph = self.use_graph;
        let moe_scale = self.hp.moe_scale;

        // ============================================================
        // GRAPH A ("pre_attn"): attn-norm + q/k/v projections + fused
        // per-head QK-norm. All pos-independent (stable buffers/pointers),
        // so it captures once and replays every token.
        // ============================================================
        let mut pre_attn = |st: &Stream| -> eyre::Result<()> {
            rms.launch_weighted(st, &mut s.ain, &s.h, &lw.attn_norm, HIDDEN as u32, EPS)?;
            f16.matvec(st, &mut s.q, &lw.wq, &s.ain, n_embd_q as u32, HIDDEN as u32)?;
            f16.matvec(st, &mut s.k, &lw.wk, &s.ain, (N_KV_HEAD * HEAD_DIM) as u32, HIDDEN as u32)?;
            f16.matvec(st, &mut s.v, &lw.wv, &s.ain, (N_KV_HEAD * HEAD_DIM) as u32, HIDDEN as u32)?;
            ops.qk_rmsnorm(st, &mut s.qn, &s.q, &lw.q_norm, n_head as u32, HEAD_DIM as u32, EPS)?;
            ops.qk_rmsnorm(st, &mut s.kn, &s.k, &lw.k_norm, N_KV_HEAD as u32, HEAD_DIM as u32, EPS)?;
            Ok(())
        };
        if use_graph {
            graphs.run("pre_attn", il as u32, stream, pre_attn)?;
        } else {
            pre_attn(stream)?;
        }

        // ============================================================
        // POS-DEPENDENT (outside any graph): rope (uses `pos`), KV-append
        // (writes the `pos`-offset cache slot), Q->f16, GQA attention
        // (reads the `n_kv`-long causal history). Pointers/scalars change
        // every token, so these launch individually.
        // ============================================================
        rope_kern.launch_forward(stream, &mut s.qn, n_head as u32, HEAD_DIM as u32, n_rot, pos as u32, rope)?;
        rope_kern.launch_forward(stream, &mut s.kn, N_KV_HEAD as u32, HEAD_DIM as u32, n_rot, pos as u32, rope)?;
        {
            let mut kslot = self.kc[il].slice_view_mut(pos * N_KV_HEAD * HEAD_DIM, N_KV_HEAD * HEAD_DIM);
            ops.cast_f16(stream, &mut kslot, &s.kn, (N_KV_HEAD * HEAD_DIM) as u32)?;
            let mut vslot = self.vc[il].slice_view_mut(pos * N_KV_HEAD * HEAD_DIM, N_KV_HEAD * HEAD_DIM);
            ops.cast_f16(stream, &mut vslot, &s.v, (N_KV_HEAD * HEAD_DIM) as u32)?;
        }
        ops.cast_f16(stream, &mut s.qf, &s.qn, n_embd_q as u32)?;
        {
            let qf_v = s.qf.slice_view(0, n_embd_q);
            let k_v = self.kc[il].slice_view(0, n_kv * N_KV_HEAD * HEAD_DIM);
            let v_v = self.vc[il].slice_view(0, n_kv * N_KV_HEAD * HEAD_DIM);
            let mut od_v = s.od.slice_view_mut(0, n_embd_q);
            // Non-het base Laguna: full causal history in a contiguous [0, n_kv)
            // buffer (no SWA ring). k_base=0 and kv_capacity=n_kv make the kernel's
            // physical-slot modulo a no-op (j % n_kv == j for j < n_kv).
            if crate::gqa_attention::decode_attn_use_naive() {
                gqa.single_query(
                    stream, &mut od_v, &qf_v, &k_v, &v_v,
                    n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, n_kv as u32, scale,
                    0, n_kv as u32,
                )?;
            } else {
                gqa.single_query_flash(
                    stream, &mut od_v, &qf_v, &k_v, &v_v,
                    n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, n_kv as u32, scale,
                    0, n_kv as u32,
                )?;
            }
        }

        // ============================================================
        // Attention tail + FFN. For MoE layers this whole back-half is
        // pos-independent (expert ids live in the device buffer `s.sel`
        // that the batched kernels read — the launch params are static),
        // so GRAPH B ("attn_ffn") captures it. Layer 0 (dense) stays inline.
        // ============================================================
        if let Some((gw, uw, dw)) = &lw.dense {
            // --- softplus gate + o_proj + residual + ffn norm ---
            f16.matvec(stream, &mut s.gate_logits, &lw.wg, &s.ain, n_head as u32, HIDDEN as u32)?;
            ops.softplus_gate(stream, &mut s.od, &s.gate_logits, n_head as u32, HEAD_DIM as u32)?;
            f16.matvec(stream, &mut s.op, &lw.wo, &s.od, HIDDEN as u32, n_embd_q as u32)?;
            vadd.launch(stream, &mut s.op, &s.h, HIDDEN as u32)?;
            rms.launch_weighted(stream, &mut s.fn_in, &s.op, &lw.ffn_norm, HIDDEN as u32, EPS)?;
            // dense SwiGLU
            qmatvec(q4d, q6d, stream, &mut s.gate_big, gw, &s.fn_in, FF_DENSE as u32, HIDDEN as u32)?;
            qmatvec(q4d, q6d, stream, &mut s.up_big, uw, &s.fn_in, FF_DENSE as u32, HIDDEN as u32)?;
            swiglu.launch(stream, &mut s.sw_big, &s.gate_big, &s.up_big, FF_DENSE as u32)?;
            qmatvec(q4d, q6d, stream, &mut s.ffn_out, dw, &s.sw_big, HIDDEN as u32, FF_DENSE as u32)?;
            // residual: h = ffn_out + op(ffn_inp)
            s.h.copy_from_buffer(&s.ffn_out)?;
            vadd.launch(stream, &mut s.h, &s.op, HIDDEN as u32)?;
        } else {
            let mut attn_ffn = |st: &Stream| -> eyre::Result<()> {
                // softplus per-head gate then o_proj
                f16.matvec(st, &mut s.gate_logits, &lw.wg, &s.ain, n_head as u32, HIDDEN as u32)?;
                ops.softplus_gate(st, &mut s.od, &s.gate_logits, n_head as u32, HEAD_DIM as u32)?;
                f16.matvec(st, &mut s.op, &lw.wo, &s.od, HIDDEN as u32, n_embd_q as u32)?;
                // residual: op = o_proj + h  (op now holds ffn_inp)
                vadd.launch(st, &mut s.op, &s.h, HIDDEN as u32)?;
                // ffn norm
                rms.launch_weighted(st, &mut s.fn_in, &s.op, &lw.ffn_norm, HIDDEN as u32, EPS)?;

                let moe = lw.moe.as_ref().unwrap();
                // router (device): selected + weights (sum-norm × scale).
                // Multi-WG split: scores matvec across the grid, then top-k.
                ops.router_split(
                    st, &mut s.sel, &mut s.ew, &mut s.router_probs, &mut s.router_scores,
                    &moe.router, &s.fn_in, &moe.bias,
                    N_EXPERT as u32, HIDDEN as u32, TOPK as u32, moe_scale, 1e-20,
                )?;
                let gate_bpe = moe.gate_stride;
                let up_bpe = moe.up_stride;
                let down_bpe = moe.down_stride;
                let n_blk_hidden = (HIDDEN / 256) as u32;
                let n_blk_mid = (FF_EXP / 256) as u32;
                q8k.launch(st, &mut s.xq_hidden, &s.fn_in, n_blk_hidden)?;
                q4b.launch_pair_swiglu_batched(
                    st, &mut s.mid, &moe.gate_all, &moe.up_all, &s.xq_hidden, &s.ew, &s.sel,
                    gate_bpe as u32, up_bpe as u32, TOPK as u32, 0.0, FF_EXP as u32, n_blk_hidden,
                )?;
                q8k.launch(st, &mut s.xq_mid, &s.mid, (TOPK as u32) * n_blk_mid)?;
                let xq_slot_stride = n_blk_mid * 292;
                match moe.down_dt {
                    GgufType::Q6_K => q6b.launch_batched(
                        st, &mut s.acc, &moe.down_all, &s.xq_mid, &s.sel,
                        down_bpe as u32, xq_slot_stride, TOPK as u32, HIDDEN as u32, n_blk_mid,
                    )?,
                    GgufType::Q4_K => q4b.launch_batched(
                        st, &mut s.acc, &moe.down_all, &s.xq_mid, &s.sel,
                        down_bpe as u32, xq_slot_stride, TOPK as u32, HIDDEN as u32, n_blk_mid,
                    )?,
                    other => return Err(eyre!("moe down dtype {other:?}")),
                }
                // shared expert (dense SwiGLU) added to routed sum
                qmatvec(q4d, q6d, st, &mut s.gate_s, &moe.sh_gate, &s.fn_in, FF_SHEXP as u32, HIDDEN as u32)?;
                qmatvec(q4d, q6d, st, &mut s.up_s, &moe.sh_up, &s.fn_in, FF_SHEXP as u32, HIDDEN as u32)?;
                swiglu.launch(st, &mut s.sw_s, &s.gate_s, &s.up_s, FF_SHEXP as u32)?;
                qmatvec(q4d, q6d, st, &mut s.down_s, &moe.sh_down, &s.sw_s, HIDDEN as u32, FF_SHEXP as u32)?;
                // ffn_out = acc + down_s  (async DtoD so it captures cleanly)
                s.ffn_out.copy_from_buffer_async(&s.acc, st)?;
                vadd.launch(st, &mut s.ffn_out, &s.down_s, HIDDEN as u32)?;
                // residual: h = ffn_out + op(ffn_inp)
                s.h.copy_from_buffer_async(&s.ffn_out, st)?;
                vadd.launch(st, &mut s.h, &s.op, HIDDEN as u32)?;
                Ok(())
            };
            if use_graph {
                graphs.run("attn_ffn", il as u32, stream, attn_ffn)?;
            } else {
                attn_ffn(stream)?;
            }
        }
        Ok(())
    }

    /// Run all layers for one token at `pos`; advances `kv_len`. Does NOT
    /// compute logits.
    pub fn forward_no_logits(&mut self, tok_id: usize, pos: usize) -> eyre::Result<()> {
        self.embed(tok_id)?;
        for il in 0..N_LAYER {
            self.layer(il, pos)?;
        }
        self.stream.synchronize()?;
        self.kv_len = pos + 1;
        Ok(())
    }

    /// Run all layers + output norm + LM head; returns the greedy argmax token.
    pub fn forward_logits(&mut self, tok_id: usize, pos: usize) -> eyre::Result<(usize, f32)> {
        self.embed(tok_id)?;
        for il in 0..N_LAYER {
            self.layer(il, pos)?;
        }
        // output norm + LM head
        self.rms.launch_weighted(&self.stream, &mut self.scratch.rn, &self.scratch.h, &self.output_norm, HIDDEN as u32, EPS)?;
        let out_dt = self.output_w.dtype;
        match out_dt {
            GgufType::Q6_K => self.q6d.matvec(&self.stream, &mut self.scratch.logits, &self.output_w.bytes, &self.scratch.rn, VOCAB as u32, HIDDEN as u32)?,
            GgufType::Q4_K => self.q4d.matvec(&self.stream, &mut self.scratch.logits, &self.output_w.bytes, &self.scratch.rn, VOCAB as u32, HIDDEN as u32)?,
            other => return Err(eyre!("LM head dtype {other:?}")),
        }
        self.stream.synchronize()?;
        self.kv_len = pos + 1;

        let mut logits = vec![0f32; VOCAB];
        self.scratch.logits.copy_to_host(&mut logits)?;
        let (argmax, maxv) = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
        Ok((argmax, maxv))
    }

    /// Prefill `tokens` (build KV, positions 0..len), returning the greedy
    /// next-token after the last prompt token.
    pub fn prefill(&mut self, tokens: &[usize]) -> eyre::Result<(usize, f32)> {
        self.reset();
        let last = tokens.len() - 1;
        for (pos, &tok) in tokens.iter().enumerate() {
            if pos == last {
                return self.forward_logits(tok, pos);
            }
            self.forward_no_logits(tok, pos)?;
        }
        unreachable!()
    }

    /// One greedy decode step: feed `tok_id` at `pos`, return next token.
    pub fn decode_step(&mut self, tok_id: usize, pos: usize) -> eyre::Result<(usize, f32)> {
        self.forward_logits(tok_id, pos)
    }
}

/// Batched dtype-dispatched dense quantized matvec (grid.z = B). `x[B,k]`,
/// `out[B,n_rows]`, weight `[n_rows,k]` shared across the batch. Mirrors
/// [`qmatvec`] for the batched-prefill path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatvec_batched(
    q4d: &Q4_KDenseMatvec,
    q6d: &Q6_KDenseMatvec,
    stream: &Stream,
    out: &mut DeviceBuffer<f32>,
    w: &QWeight,
    x: &DeviceBuffer<f32>,
    n_rows: u32,
    k: u32,
    batch: u32,
) -> eyre::Result<()> {
    match w.dtype {
        GgufType::Q4_K => q4d.matvec_batched(stream, out, &w.bytes, x, n_rows, k, batch),
        GgufType::Q6_K => q6d.matvec_batched(stream, out, &w.bytes, x, n_rows, k, batch),
        other => Err(eyre!("qmatvec_batched: unsupported dtype {other:?}")),
    }
}

/// Dispatch a dense quantized matvec on the tensor's actual dtype. Free
/// function (not a `&self` method) so it can be called while `&mut
/// self.scratch` is held — it only needs the two kernel handles + stream.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatvec(
    q4d: &Q4_KDenseMatvec,
    q6d: &Q6_KDenseMatvec,
    stream: &Stream,
    out: &mut DeviceBuffer<f32>,
    w: &QWeight,
    x: &DeviceBuffer<f32>,
    n_rows: u32,
    k: u32,
) -> eyre::Result<()> {
    match w.dtype {
        GgufType::Q4_K => q4d.matvec(stream, out, &w.bytes, x, n_rows, k),
        GgufType::Q6_K => q6d.matvec(stream, out, &w.bytes, x, n_rows, k),
        other => Err(eyre!("qmatvec: unsupported dtype {other:?}")),
    }
}

// --------------------------------------------------------------------------
// Q4_K CPU dequant for token-embedding rows (matches the spike).
// --------------------------------------------------------------------------
fn get_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

pub(crate) fn dequant_q4k_superblock(blk: &[u8], out: &mut [f32]) {
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
