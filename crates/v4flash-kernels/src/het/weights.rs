//! Per-device weight splits for V4-Flash.
//!
//! * [`DgpuLayerWeights`] — attention LoRAs, mHC, compressor, shared
//!   expert, attention norms, router. ~200 MiB/layer × 43 layers = ~9 GiB.
//! * [`IgpuLayerWeights`] — routed MoE (gate/up/down for 256 experts).
//!   ~1.2 GiB/layer × 43 layers = ~52 GiB.
//! * [`HetGlobalWeights`] — output head + embedding, dGPU-resident.

use std::path::{Path, PathBuf};

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, WeightSrc};
use v4flash_hip::{Device, DeviceBuffer};

use crate::config::{
    COMPRESS_RATIOS, HC_MIX_DIM, INDEXER_COMP_WIDTH, N_EMBD, N_EXPERT, N_HASH_LAYERS, N_HC,
    N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LAYER, N_LORA_Q,
};
use crate::model_weights::{
    load_f32_weight, load_i32_tensor, CompressorWeights, RoutedExpertWeights, SharedExpertWeights,
};
use crate::rope::RopeParams;
use crate::weight_contract;
use crate::weights::{load_to_device, DeviceWeight};
use crate::config::{ENGRAM_IN, ENGRAM_OUT};

/// V4.1 Engram (layers 1, 14; ARCH_SPEC §1.7): `wkv` Q8_0 `[ENGRAM_OUT, ENGRAM_IN]`
/// (24 gathered rows → N_HC keys + one value) and the gate product
/// q_weight ⊙ k_weight `[N_HC, N_EMBD]`. Rows themselves are gathered on the
/// host (`v4flash_core::EngramTable`) and staged per token.
pub struct EngramWeights {
    pub wkv: DeviceWeight,
    pub qk: DeviceBuffer<f32>,
}

pub struct DgpuLayerWeights {
    pub layer_idx: i32,
    pub ratio: u32,


    // mHC
    pub hc_attn_fn: DeviceWeight,
    pub hc_attn_scale: DeviceBuffer<f32>,
    pub hc_attn_base: DeviceBuffer<f32>,
    pub hc_ffn_fn: DeviceWeight,
    pub hc_ffn_scale: DeviceBuffer<f32>,
    pub hc_ffn_base: DeviceBuffer<f32>,

    // Attention
    pub attn_norm: DeviceBuffer<f32>,
    pub attn_q_a: DeviceWeight,
    pub attn_q_b: DeviceWeight,
    pub q_a_norm: DeviceBuffer<f32>,
    pub attn_kv: DeviceWeight,
    pub kv_a_norm: DeviceBuffer<f32>,
    pub attn_sinks: DeviceBuffer<f32>,
    pub attn_output_a: DeviceWeight,
    pub attn_output_b: DeviceWeight,
    pub rope_params: RopeParams,
    /// V4.1 Engram module when this layer has one (`blk.N.engram_wkv.weight` present).
    pub engram: Option<EngramWeights>,

    // FFN norm + shared expert
    pub ffn_norm: DeviceBuffer<f32>,
    pub shared: SharedExpertWeights,

    // Compressor lives on dGPU because (a) 9070 XT has 2.6× the BW of
    // Strix iGPU → f16 matvec is faster locally, and (b) attn_input_norm
    // is computed on dGPU, so the compressor's input is local and no
    // peer push is needed.
    pub compressor: Option<CompressorWeights>,

    // CSA indexer — only present at ratio==4 layers. `indexer` holds the
    // two projection weights (attn_q_b, proj) that build the per-token
    // indexer query + head weights; `indexer_compressor` is a second
    // compressor instance at head_dim=128 (vs main's 512) that maintains
    // the index_comp_kv cache the scoring kernel reads. See ds4.c:6977-7106
    // (`indexer_allowed_decode_one`) and ds4.c:7791-7811 (the indexer
    // compressor call site).
    pub indexer: Option<IndexerWeights>,
    pub indexer_compressor: Option<CompressorWeights>,

    // Router lives on dGPU because (a) the f16 matvec is ~1.5 ms faster
    // on dGPU's BW, and (b) keeping it off iGPU lifts it from the iGPU
    // MoE critical path. After the router runs on dGPU, selected/d_ew
    // are peer-pushed to iGPU and the MoE pipeline starts immediately.
    pub is_hash_router: bool,
    pub ffn_gate_inp: DeviceWeight,
    pub tid2eid: Option<Vec<i32>>,
    pub router_bias_dev: Option<DeviceBuffer<f32>>,
    /// Vision-Exp: `layers.N.ffn.gate.bias_vl` `[N_EXPERT]` f32 — the
    /// selection bias for IMAGE rows (token id >= N_VOCAB) on EVERY layer,
    /// hash layers included (there it drives a top-k instead of tid2eid).
    /// Absent from the GGUF; loaded from the `bias_vl.bin` sidecar
    /// (`bias_vl_sidecar_path`) when present. `None` = text-only model.
    pub router_bias_vl_dev: Option<DeviceBuffer<f32>>,
}

/// CSA indexer projection weights (per ratio==4 layer).
/// - `attn_q_b`: F16 [N_LORA_Q × (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM)]
///   matvec input qr_normed → indexer_q[N_INDEXER_HEAD, N_INDEXER_HEAD_DIM]
/// - `proj`: F16 [N_EMBD × N_INDEXER_HEAD]
///   matvec input attn_input_norm → head_weights[N_INDEXER_HEAD]
pub struct IndexerWeights {
    pub attn_q_b: DeviceWeight,
    pub proj: DeviceWeight,
    /// V4.1 ONLY. V4-Flash derives index K from a second (ratio-4) compressor;
    /// V4.1 has NO indexer compressor — index K is `k_norm(wk(latent))` on the
    /// PRE-RoPE latent (`model.py:535-537`). Both are `None` on V4-Flash.
    pub attn_k: Option<DeviceWeight>,
    /// f32, `N_INDEXER_HEAD_DIM` — consumed directly by `RmsNorm::launch_weighted`,
    /// so it loads via `load_f32_weight` rather than as a quantized `DeviceWeight`.
    pub k_norm: Option<DeviceBuffer<f32>>,
}

/// Does `layer` own indexer weights under V4.1 (CSA2 `index_source_layer_ids`)?
/// V4-Flash keys indexer presence off `ratio == 4` instead, so this is false there.
fn is_index_source(layer: i32) -> bool {
    crate::config::INDEX_SOURCE_LAYERS.contains(&layer)
}

/// Does `layer` own the index-K projection? All 8 index-source layers SCORE, but
/// index K exists only on the 4 KV-source layers (2, 8, 14, 20) — the others reuse
/// the nearest source's keys, exactly as the main compressed store is reused.
/// Assuming otherwise fails loudly at load: `blk.24.indexer.attn_k.weight` is absent.
fn owns_index_k(layer: i32) -> bool {
    crate::config::KV_SOURCE_LAYERS.contains(&layer)
}

pub struct IgpuLayerWeights {
    pub layer_idx: i32,
    pub ratio: u32,
    pub is_hash_router: bool,

    /// Routed experts: a 1-slot placeholder (the ExpertPager holds the real ones). Historically all 256; with `IGPU_DEDUP_HOT` only the
    /// `256 - n_hot` experts that are NOT dGPU-resident, packed dense
    /// (`routed.n_slots`).
    pub routed: RoutedExpertWeights,

    /// Per-layer RoPE params. Mirrors the dGPU side for any future
    /// iGPU-resident RoPE call (currently unused — all RoPE runs on dGPU).
    pub rope_params: RopeParams,
}

pub struct HetGlobalWeights {
    /// Embedding lookup on dGPU (output head also lives here).
    // M57: token_embd is NOT device-resident — the server embeds host-side
    // from the gguf mmap (deepstrix-server/src/embed.rs); the old 1.06 GB
    // dGPU copy had zero kernel consumers. Dtype is still validated below.
    pub output: DeviceWeight,
    pub output_norm: DeviceBuffer<f32>,
    pub output_hc_fn: DeviceWeight,
    pub output_hc_scale: DeviceBuffer<f32>,
    pub output_hc_base: DeviceBuffer<f32>,
}

/// DSpark drafter weights: three layers, fully resident.
///
/// Deliberately NOT `DgpuLayerWeights`/`IgpuLayerWeights`. A drafter layer
/// carries a main layer's tensor set, but the main loaders assume main-layer
/// facts that are false here: `COMPRESS_RATIOS[layer]` is sized to `N_LAYER` so
/// `blk.40` would panic, the packed-expert path keys off `N_EXPERT`, and the
/// hash-router / indexer / engram / compressor branches are all main-model-only.
/// The drafter needs none of them.
///
/// 7.9 GB for all three layers, so this is resident and never paged — the
/// drafter must not contend with the expert pager on the critical path.
pub struct MtpLayerWeights {
    pub hc_attn_fn: DeviceWeight,
    pub hc_attn_scale: DeviceBuffer<f32>,
    pub hc_attn_base: DeviceBuffer<f32>,
    pub hc_ffn_fn: DeviceWeight,
    pub hc_ffn_scale: DeviceBuffer<f32>,
    pub hc_ffn_base: DeviceBuffer<f32>,

    pub attn_norm: DeviceBuffer<f32>,
    pub attn_q_a: DeviceWeight,
    pub attn_q_b: DeviceWeight,
    pub q_a_norm: DeviceBuffer<f32>,
    pub attn_kv: DeviceWeight,
    pub kv_a_norm: DeviceBuffer<f32>,
    pub attn_sinks: DeviceBuffer<f32>,
    pub attn_output_a: DeviceWeight,
    pub attn_output_b: DeviceWeight,

    pub ffn_norm: DeviceBuffer<f32>,
    pub ffn_gate_inp: DeviceWeight,
    pub exp_probs_b: DeviceBuffer<f32>,
    pub shared: SharedExpertWeights,
    pub routed: RoutedExpertWeights,
}

/// The whole drafter: an entry projection, three layers, and the exit heads.
pub struct MtpWeights {
    /// `mtp.0.main_proj` [5120, 15360] — eats the concatenated residuals
    /// entering layers 37/38/39, then `main_norm`.
    pub main_proj: DeviceWeight,
    pub main_norm: DeviceBuffer<f32>,
    pub layers: Vec<MtpLayerWeights>,
}

/// The drafter's EXIT weights, loaded separately because they live on a
/// different device than the layers.
///
/// The drafter's head is TIED to the main model's `output` weight, which is
/// dGPU-resident, and the drafter's 7.93 GB of layers only fit on the iGPU. So
/// the layer stack runs on the iGPU and the exit on the dGPU, with the residual
/// handed across. Loading these onto the layer device instead would mean either
/// a second 662 MB copy of the vocab projection or a cross-device matvec.
pub struct MtpExitWeights {
    /// `mtp.2.norm.weight` — final norm before the tied head.
    pub norm: DeviceBuffer<f32>,
    /// `mtp.2.markov_head.head.weight` [N_VOCAB, 256] — projects a markov
    /// embedding back to a full-vocab logit bias. Its EMBEDDING half stays
    /// host-side like `token_embd` (M57): the ids it looks up are produced one
    /// at a time by the exit's sequential loop, so a device gather would buy
    /// nothing.
    pub markov_head: DeviceWeight,
    /// `mtp.2.confidence_head.proj.weight` [1, N_EMBD + 256]. Loaded but not yet
    /// consumed — confidence gates HOW MANY drafts to submit, which is a policy
    /// knob on top of a correct draft, not part of producing one.
    pub confidence: DeviceBuffer<f32>,
}

impl MtpExitWeights {
    pub fn load<'a>(gguf: impl Into<WeightSrc<'a>>, device: Device) -> eyre::Result<Self> {
        let gguf: WeightSrc<'a> = gguf.into();
        device.set_current()?;
        let id = device.id;
        let last = v4flash_core::hf_v41::MTP_STAGES - 1;
        Ok(Self {
            norm: load_f32_weight(gguf, &format!("mtp.{last}.norm.weight"), id, N_EMBD as usize)?,
            markov_head: load_to_device(gguf, &format!("mtp.{last}.markov_head.weight"), id)?,
            confidence: load_f32_weight(
                gguf,
                &format!("mtp.{last}.confidence.weight"),
                id,
                N_EMBD as usize + crate::het::mtp::MTP_MARKOV_RANK,
            )?,
        })
    }
}

impl MtpWeights {
    /// Load all three drafter layers onto `device`. `n_layers` is the MAIN
    /// model's layer count — the drafter is presented as `blk.{n_layers + s}`.
    pub fn load<'a>(
        gguf: impl Into<WeightSrc<'a>>,
        device: Device,
        n_layers: usize,
    ) -> eyre::Result<Self> {
        let gguf: WeightSrc<'a> = gguf.into();
        device.set_current()?;
        let device_id = device.id;
        let n_stages = v4flash_core::hf_v41::MTP_STAGES;
        let n_exp = v4flash_core::hf_v41::MTP_N_EXPERT as u32;

        let main_proj = load_to_device(gguf, "mtp.0.main_proj.weight", device_id)?;
        let main_norm = load_f32_weight(gguf, "mtp.0.main_norm.weight", device_id, N_EMBD as usize)?;
        let mut layers = Vec::with_capacity(n_stages);
        for sgi in 0..n_stages {
            let l = n_layers + sgi;
            let routed = RoutedExpertWeights {
                gate: load_to_device(gguf, &format!("blk.{l}.ffn_gate_exps.weight"), device_id)?,
                up: load_to_device(gguf, &format!("blk.{l}.ffn_up_exps.weight"), device_id)?,
                down: load_to_device(gguf, &format!("blk.{l}.ffn_down_exps.weight"), device_id)?,
                gate_bytes_per_expert: 0,
                up_bytes_per_expert: 0,
                down_bytes_per_expert: 0,
                n_slots: n_exp,
            };
            let routed = RoutedExpertWeights {
                gate_bytes_per_expert: routed.gate.buffer.len() / n_exp as usize,
                up_bytes_per_expert: routed.up.buffer.len() / n_exp as usize,
                down_bytes_per_expert: routed.down.buffer.len() / n_exp as usize,
                ..routed
            };
            layers.push(MtpLayerWeights {
                hc_attn_fn: load_to_device(gguf, &format!("blk.{l}.hc_attn_fn.weight"), device_id)?,
                hc_attn_scale: load_f32_weight(gguf, &format!("blk.{l}.hc_attn_scale.weight"), device_id, 3)?,
                hc_attn_base: load_f32_weight(gguf, &format!("blk.{l}.hc_attn_base.weight"), device_id, HC_MIX_DIM as usize)?,
                hc_ffn_fn: load_to_device(gguf, &format!("blk.{l}.hc_ffn_fn.weight"), device_id)?,
                hc_ffn_scale: load_f32_weight(gguf, &format!("blk.{l}.hc_ffn_scale.weight"), device_id, 3)?,
                hc_ffn_base: load_f32_weight(gguf, &format!("blk.{l}.hc_ffn_base.weight"), device_id, HC_MIX_DIM as usize)?,
                attn_norm: load_f32_weight(gguf, &format!("blk.{l}.attn_norm.weight"), device_id, N_EMBD as usize)?,
                attn_q_a: load_to_device(gguf, &format!("blk.{l}.attn_q_a.weight"), device_id)?,
                attn_q_b: load_to_device(gguf, &format!("blk.{l}.attn_q_b.weight"), device_id)?,
                q_a_norm: load_f32_weight(gguf, &format!("blk.{l}.attn_q_a_norm.weight"), device_id, N_LORA_Q as usize)?,
                attn_kv: load_to_device(gguf, &format!("blk.{l}.attn_kv.weight"), device_id)?,
                kv_a_norm: load_f32_weight(gguf, &format!("blk.{l}.attn_kv_a_norm.weight"), device_id, N_HEAD_DIM as usize)?,
                attn_sinks: load_f32_weight(gguf, &format!("blk.{l}.attn_sinks.weight"), device_id, N_HEAD as usize)?,
                attn_output_a: load_to_device(gguf, &format!("blk.{l}.attn_output_a.weight"), device_id)?,
                attn_output_b: load_to_device(gguf, &format!("blk.{l}.attn_output_b.weight"), device_id)?,
                ffn_norm: load_f32_weight(gguf, &format!("blk.{l}.ffn_norm.weight"), device_id, N_EMBD as usize)?,
                ffn_gate_inp: load_to_device(gguf, &format!("blk.{l}.ffn_gate_inp.weight"), device_id)?,
                exp_probs_b: load_f32_weight(gguf, &format!("blk.{l}.exp_probs_b.bias"), device_id, n_exp as usize)?,
                shared: SharedExpertWeights {
                    gate: load_to_device(gguf, &format!("blk.{l}.ffn_gate_shexp.weight"), device_id)?,
                    up: load_to_device(gguf, &format!("blk.{l}.ffn_up_shexp.weight"), device_id)?,
                    down: load_to_device(gguf, &format!("blk.{l}.ffn_down_shexp.weight"), device_id)?,
                },
                routed,
            });
        }
        Ok(Self { main_proj, main_norm, layers })
    }
}

pub struct HetModelWeights {
    pub global: HetGlobalWeights,
    pub dgpu_layers: Vec<DgpuLayerWeights>,
    pub igpu_layers: Vec<IgpuLayerWeights>,
}

impl HetGlobalWeights {
    pub fn load<'a>(gguf: impl Into<WeightSrc<'a>>, dgpu_device: Device) -> eyre::Result<Self> {
        let gguf: WeightSrc<'a> = gguf.into();
        dgpu_device.set_current()?;
        let dgpu_id = dgpu_device.id;
        // Validate token_embd dtype without uploading (host-side embed path).
        {
            let te = gguf.tensor("token_embd.weight")
                .ok_or_else(|| eyre!("token_embd.weight not found"))?;
            if !weight_contract::TOKEN_EMBD_ALLOWED.contains(&te.dtype) {
                return Err(eyre!(
                    "token_embd dtype {:?} unsupported (allowed: {:?})",
                    te.dtype,
                    weight_contract::TOKEN_EMBD_ALLOWED
                ));
            }
        }
        // output.weight dtype is enforced by the contract inside
        // load_to_device (Quant role).
        let output = load_to_device(gguf, "output.weight", dgpu_id)?;
        let output_norm = load_f32_weight(gguf, "output_norm.weight", dgpu_id, N_EMBD as usize)?;
        // V4.1 has no head-level hyper-connection projection. Its final collapse
        // reuses the CARRIED pre_mix from the last block (single-pass mHC:
        // `blocks[-1].hc_pre(h, pre_mix)` in model.py), so `output_hc_*` simply do
        // not exist in the checkpoint. Stub them; `forward_head` takes the v41
        // branch and never reads these.
        let (output_hc_fn, output_hc_scale, output_hc_base) = {
            let _ = &gguf;
            (
                DeviceWeight {
                    buffer: DeviceBuffer::<u8>::new(dgpu_id, 32)?,
                    n_elements: 0,
                    dtype: GgufType::F16,
                    shape: vec![0],
                },
                DeviceBuffer::<f32>::new(dgpu_id, 1)?,
                DeviceBuffer::<f32>::new(dgpu_id, N_HC as usize)?,
            )
        };
        Ok(Self {
            output,
            output_norm,
            output_hc_fn,
            output_hc_scale,
            output_hc_base,
        })
    }
}

impl DgpuLayerWeights {
    pub fn load<'a>(
        gguf: impl Into<WeightSrc<'a>>,
        dgpu_device: Device,
        layer: i32,
        rope_params_for_layer: &dyn Fn(i32) -> eyre::Result<RopeParams>,
    ) -> eyre::Result<Self> {
        let gguf: WeightSrc<'a> = gguf.into();
        dgpu_device.set_current()?;
        let device_id = dgpu_device.id;
        let ratio = COMPRESS_RATIOS[layer as usize];

        let hc_attn_fn = load_to_device(gguf, &format!("blk.{layer}.hc_attn_fn.weight"), device_id)?;
        let hc_attn_scale =
            load_f32_weight(gguf, &format!("blk.{layer}.hc_attn_scale.weight"), device_id, 3)?;
        let hc_attn_base = load_f32_weight(
            gguf,
            &format!("blk.{layer}.hc_attn_base.weight"),
            device_id,
            HC_MIX_DIM as usize,
        )?;
        let hc_ffn_fn = load_to_device(gguf, &format!("blk.{layer}.hc_ffn_fn.weight"), device_id)?;
        let hc_ffn_scale =
            load_f32_weight(gguf, &format!("blk.{layer}.hc_ffn_scale.weight"), device_id, 3)?;
        let hc_ffn_base = load_f32_weight(
            gguf,
            &format!("blk.{layer}.hc_ffn_base.weight"),
            device_id,
            HC_MIX_DIM as usize,
        )?;

        let attn_norm = load_f32_weight(
            gguf,
            &format!("blk.{layer}.attn_norm.weight"),
            device_id,
            N_EMBD as usize,
        )?;
        let engram = {
            let wkv_name = format!("blk.{layer}.engram_wkv.weight");
            if gguf.tensor(&wkv_name).is_some() {
                let wkv = load_to_device(gguf, &wkv_name, device_id)?;
                if wkv.n_elements != (ENGRAM_IN as u64) * (ENGRAM_OUT as u64) {
                    return Err(eyre!("{wkv_name}: {} elements, want {}×{}", wkv.n_elements, ENGRAM_OUT, ENGRAM_IN));
                }
                let read_f32 = |name: &str| -> eyre::Result<Vec<f32>> {
                    let t = gguf.tensor(name).ok_or_else(|| eyre!("tensor `{name}` not found"))?;
                    Ok(gguf.read_tensor(t)?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
                };
                let q = read_f32(&format!("blk.{layer}.engram_q.weight"))?;
                let k = read_f32(&format!("blk.{layer}.engram_k.weight"))?;
                let n = (crate::config::N_HC * crate::config::N_EMBD) as usize;
                if q.len() != n || k.len() != n {
                    return Err(eyre!("blk.{layer}.engram_{{q,k}}.weight: {}/{} elements, want {n}", q.len(), k.len()));
                }
                let qk_h: Vec<f32> = q.iter().zip(&k).map(|(a, b)| a * b).collect();
                let mut qk = DeviceBuffer::<f32>::new(device_id, n)?;
                qk.copy_from_host(&qk_h)?;
                Some(EngramWeights { wkv, qk })
            } else {
                None
            }
        };
        let attn_q_a = load_to_device(gguf, &format!("blk.{layer}.attn_q_a.weight"), device_id)?;
        let attn_q_b = load_to_device(gguf, &format!("blk.{layer}.attn_q_b.weight"), device_id)?;
        let q_a_norm = load_f32_weight(
            gguf,
            &format!("blk.{layer}.attn_q_a_norm.weight"),
            device_id,
            N_LORA_Q as usize,
        )?;
        let attn_kv = load_to_device(gguf, &format!("blk.{layer}.attn_kv.weight"), device_id)?;
        let kv_a_norm = load_f32_weight(
            gguf,
            &format!("blk.{layer}.attn_kv_a_norm.weight"),
            device_id,
            N_HEAD_DIM as usize,
        )?;
        let attn_sinks = load_f32_weight(
            gguf,
            &format!("blk.{layer}.attn_sinks.weight"),
            device_id,
            N_HEAD as usize,
        )?;
        let attn_output_a =
            load_to_device(gguf, &format!("blk.{layer}.attn_output_a.weight"), device_id)?;
        let attn_output_b =
            load_to_device(gguf, &format!("blk.{layer}.attn_output_b.weight"), device_id)?;
        let rope_params = rope_params_for_layer(layer)?;

        let ffn_norm = load_f32_weight(
            gguf,
            &format!("blk.{layer}.ffn_norm.weight"),
            device_id,
            N_EMBD as usize,
        )?;
        let shared = SharedExpertWeights {
            gate: load_to_device(
                gguf,
                &format!("blk.{layer}.ffn_gate_shexp.weight"),
                device_id,
            )?,
            up: load_to_device(gguf, &format!("blk.{layer}.ffn_up_shexp.weight"), device_id)?,
            down: load_to_device(
                gguf,
                &format!("blk.{layer}.ffn_down_shexp.weight"),
                device_id,
            )?,
        };

        // Compressor weights live on dGPU.
        // V4.1 reuse layers (ratio > 0, not a KV source) carry no compressor
        // tensors: they attend over the source layer's store (config::kv_source_of).
        let has_compressor = gguf.tensor(&format!("blk.{layer}.attn_compressor_kv.weight")).is_some();
        let compressor = if ratio > 0 && has_compressor {
            let comp_width = if ratio == 4 { 1024 } else { 512 };
            // V4.1's compressor has no absolute position embedding (ARCH_SPEC
            // §1.3): a zero f16 table of the same shape keeps `state_write`
            // unchanged (it adds 0).
            let ape_name = format!("blk.{layer}.attn_compressor_ape.weight");
            let ape = if gguf.tensor(&ape_name).is_some() {
                load_to_device(gguf, &ape_name, device_id)?
            } else {
                let n = (ratio * comp_width) as usize;
                let mut buffer = DeviceBuffer::<u8>::new(device_id, n * 2)?;
                buffer.fill_zero()?;
                DeviceWeight { buffer, n_elements: n as u64, dtype: GgufType::F16, shape: vec![comp_width as u64, ratio as u64] }
            };
            // V4.1 ratio-1 compressor (layer 20): `norm(wkv(x))`, no gate — the
            // stage then runs a single matvec and skips state/pool (ARCH_SPEC §1.3).
            let gate_name = format!("blk.{layer}.attn_compressor_gate.weight");
            let wgate = if gguf.tensor(&gate_name).is_some() {
                load_to_device(gguf, &gate_name, device_id)?
            } else {
                if ratio != 1 {
                    return Err(eyre!("{gate_name} missing at ratio {ratio}"));
                }
                // 32-byte stub: never launched (ratio 1 takes the single-matvec path).
                let mut buffer = DeviceBuffer::<u8>::new(device_id, 32)?;
                buffer.fill_zero()?;
                DeviceWeight { buffer, n_elements: 0, dtype: GgufType::F16, shape: vec![] }
            };
            Some(CompressorWeights {
                wkv: load_to_device(
                    gguf,
                    &format!("blk.{layer}.attn_compressor_kv.weight"),
                    device_id,
                )?,
                wgate,
                ape,
                norm: load_f32_weight(
                    gguf,
                    &format!("blk.{layer}.attn_compressor_norm.weight"),
                    device_id,
                    N_HEAD_DIM as usize,
                )?,
                width: comp_width,
                head_dim: N_HEAD_DIM,
            })
        } else {
            None
        };

        // CSA indexer (ratio==4 only). Tensor names per ds4.c:2610-2615.
        let (indexer, indexer_compressor) = if ratio == 4 {
            let iw = IndexerWeights {
                attn_q_b: load_to_device(
                    gguf,
                    &format!("blk.{layer}.indexer.attn_q_b.weight"),
                    device_id,
                )?,
                proj: load_to_device(
                    gguf,
                    &format!("blk.{layer}.indexer.proj.weight"),
                    device_id,
                )?,
                attn_k: None,
                k_norm: None,
            };
            let ic = CompressorWeights {
                wkv: load_to_device(
                    gguf,
                    &format!("blk.{layer}.indexer_compressor_kv.weight"),
                    device_id,
                )?,
                wgate: load_to_device(
                    gguf,
                    &format!("blk.{layer}.indexer_compressor_gate.weight"),
                    device_id,
                )?,
                ape: load_to_device(
                    gguf,
                    &format!("blk.{layer}.indexer_compressor_ape.weight"),
                    device_id,
                )?,
                norm: load_f32_weight(
                    gguf,
                    &format!("blk.{layer}.indexer_compressor_norm.weight"),
                    device_id,
                    N_INDEXER_HEAD_DIM as usize,
                )?,
                width: INDEXER_COMP_WIDTH,
                head_dim: N_INDEXER_HEAD_DIM,
            };
            (Some(iw), Some(ic))
        } else if is_index_source(layer) {
            // V4.1 CSA2 (S0): the 8 `index_source_layer_ids` own indexer weights.
            // There is NO `indexer_compressor` here — V4.1 builds index K directly
            // from the pre-RoPE latent, so the second compressor that V4-Flash
            // allocates at ratio 4 has no counterpart and stays None.
            let iw = IndexerWeights {
                attn_q_b: load_to_device(
                    gguf,
                    &format!("blk.{layer}.indexer.attn_q_b.weight"),
                    device_id,
                )?,
                proj: load_to_device(
                    gguf,
                    &format!("blk.{layer}.indexer.proj.weight"),
                    device_id,
                )?,
                attn_k: if owns_index_k(layer) {
                    Some(load_to_device(
                        gguf,
                        &format!("blk.{layer}.indexer.attn_k.weight"),
                        device_id,
                    )?)
                } else {
                    None
                },
                k_norm: if owns_index_k(layer) {
                    Some(load_f32_weight(
                        gguf,
                        &format!("blk.{layer}.indexer.k_norm.weight"),
                        device_id,
                        N_INDEXER_HEAD_DIM as usize,
                    )?)
                } else {
                    None
                },
            };
            (Some(iw), None)
        } else {
            (None, None)
        };

        // Router weights live on dGPU.
        let is_hash_router = layer < N_HASH_LAYERS;
        let ffn_gate_inp =
            load_to_device(gguf, &format!("blk.{layer}.ffn_gate_inp.weight"), device_id)?;
        let tid2eid = if is_hash_router {
            Some(load_i32_tensor(
                gguf,
                &format!("blk.{layer}.ffn_gate_tid2eid.weight"),
            )?)
        } else {
            None
        };
        let router_bias_dev = if !is_hash_router {
            let bias_name = format!("blk.{layer}.exp_probs_b.bias");
            if let Some(t) = gguf.tensor(&bias_name) {
                if t.dtype != GgufType::F32 {
                    return Err(eyre!("{bias_name} dtype {:?} != F32", t.dtype));
                }
                let bytes = gguf.read_tensor(t)?;
                let host: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(device_id, host.len())?;
                buf.copy_from_host(&host)?;
                Some(buf)
            } else {
                None
            }
        } else {
            None
        };

        Ok(DgpuLayerWeights {
            layer_idx: layer,
            ratio,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            hc_ffn_fn,
            hc_ffn_scale,
            hc_ffn_base,
            attn_norm,
            attn_q_a,
            attn_q_b,
            q_a_norm,
            attn_kv,
            kv_a_norm,
            attn_sinks,
            attn_output_a,
            attn_output_b,
            rope_params,
            engram,
            ffn_norm,
            shared,
            compressor,
            indexer,
            indexer_compressor,
            is_hash_router,
            ffn_gate_inp,
            tid2eid,
            router_bias_dev,
            router_bias_vl_dev: None,
        })
    }
}

/// Sidecar file name for the Vision-Exp routing bias.
pub const BIAS_VL_FILE: &str = "bias_vl.bin";

/// `~/.cache/deepstrix/models/<gguf file stem>/bias_vl.bin` — the same
/// per-model sidecar directory the server uses for `expert_stats.json` /
/// `hot_experts.txt`. `DEEPSTRIX_BIAS_VL_FILE` overrides the full path.
///
/// Format: `N_LAYER * N_EXPERT` (43 × 256) f32 little-endian, layer-major
/// — layer `l` at `[l*256 .. (l+1)*256)`. Produced once by
/// `scripts/fetch_bias_vl.py` from the HF safetensors shards of
/// `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (tensor names
/// `layers.{0..42}.ffn.gate.bias_vl`, F32 `[256]`).
pub fn bias_vl_sidecar_path(gguf_path: &Path) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("DEEPSTRIX_BIAS_VL_FILE") {
        return Some(PathBuf::from(p));
    }
    let stem = gguf_path.file_stem()?.to_str()?;
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".cache/deepstrix/models")
            .join(stem)
            .join(BIAS_VL_FILE),
    )
}

/// Parse a `bias_vl.bin` sidecar into `N_LAYER * N_EXPERT` f32.
pub fn read_bias_vl_sidecar(path: &Path) -> eyre::Result<Vec<f32>> {
    let bytes = std::fs::read(path)
        .map_err(|e| eyre!("read bias_vl sidecar {}: {e}", path.display()))?;
    let want = (N_LAYER as usize) * (N_EXPERT as usize) * 4;
    if bytes.len() != want {
        return Err(eyre!(
            "{}: expected {want} bytes ({N_LAYER}x{N_EXPERT} f32 LE), got {}",
            path.display(),
            bytes.len()
        ));
    }
    let v: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if let Some(bad) = v.iter().position(|x| !x.is_finite()) {
        return Err(eyre!("{}: non-finite bias_vl value at index {bad}", path.display()));
    }
    Ok(v)
}

/// Write a sidecar in the format `read_bias_vl_sidecar` reads (tooling / tests).
pub fn write_bias_vl_sidecar(path: &Path, values: &[f32]) -> eyre::Result<()> {
    let want = (N_LAYER as usize) * (N_EXPERT as usize);
    if values.len() != want {
        return Err(eyre!("write_bias_vl_sidecar: expected {want} values, got {}", values.len()));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| eyre!("mkdir {}: {e}", dir.display()))?;
    }
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out).map_err(|e| eyre!("write {}: {e}", path.display()))
}

impl HetModelWeights {
    /// `true` once `load_bias_vl_sidecar` attached the Vision-Exp routing
    /// bias to every dGPU layer. Without it, the router falls over on the
    /// FIRST image row in prefill (`forward_prefill`: "image rows in
    /// prefill but no bias_vl loaded"), so a server started with
    /// `--mmproj` should check this at startup rather than discovering it
    /// mid-request. The sidecar path is derived from the GGUF stem while
    /// `--mmproj` is a separate flag, so the two can be inconsistent with
    /// no other diagnostic.
    pub fn has_bias_vl(&self) -> bool {
        !self.dgpu_layers.is_empty()
            && self
                .dgpu_layers
                .iter()
                .all(|l| l.router_bias_vl_dev.is_some())
    }

    /// Attach the Vision-Exp routing bias from the sidecar next to
    /// `gguf_path` (see `bias_vl_sidecar_path`). No sidecar → no-op
    /// (text-only); a present-but-corrupt sidecar is an error. Uploads
    /// one `[N_EXPERT]` f32 buffer per layer to the dGPU (43 KiB total).
    pub fn load_bias_vl_sidecar(
        &mut self,
        gguf_path: &Path,
        dgpu_device: Device,
    ) -> eyre::Result<Option<PathBuf>> {
        let Some(path) = bias_vl_sidecar_path(gguf_path) else {
            return Ok(None);
        };
        if !path.exists() {
            return Ok(None);
        }
        let all = read_bias_vl_sidecar(&path)?;
        dgpu_device.set_current()?;
        let n = N_EXPERT as usize;
        for (l, dlw) in self.dgpu_layers.iter_mut().enumerate() {
            let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(dgpu_device.id, n)?;
            buf.copy_from_host(&all[l * n..(l + 1) * n])?;
            dlw.router_bias_vl_dev = Some(buf);
        }
        Ok(Some(path))
    }
}

impl IgpuLayerWeights {
    /// Routed experts are never resident here: the ExpertPager pages the
    /// router's actual picks into its own pool (V4.1's ~289 GB of experts
    /// cannot fit), so `routed` is a 1-slot placeholder that keeps the dtype /
    /// stride bookkeeping uniform. Nothing reads its bytes.
    pub fn load<'a>(
        gguf: impl Into<WeightSrc<'a>>,
        igpu_device: Device,
        layer: i32,
        rope_params_for_layer: &dyn Fn(i32) -> eyre::Result<RopeParams>,
    ) -> eyre::Result<Self> {
        let gguf: WeightSrc<'a> = gguf.into();
        igpu_device.set_current()?;
        let device_id = igpu_device.id;
        let ratio = COMPRESS_RATIOS[layer as usize];
        let is_hash_router = layer < N_HASH_LAYERS;

        let placeholder = |which: &str, k: u64, rows: u64| -> eyre::Result<(DeviceWeight, usize)> {
            let name = format!("blk.{layer}.ffn_{which}_exps.weight");
            let t = gguf
                .tensor(&name)
                .ok_or_else(|| eyre!("tensor `{name}` not found"))?;
            let bpe = weight_contract::bytes_per_expert(t.dtype, k, rows)?;
            let buffer = DeviceBuffer::<u8>::new(device_id, bpe)?;
            Ok((
                DeviceWeight {
                    buffer,
                    n_elements: k * rows,
                    dtype: t.dtype,
                    shape: vec![1, rows, k],
                },
                bpe,
            ))
        };
        let n_ff = crate::config::N_FF_EXP as u64;
        let (gate, gate_bytes_per_expert) = placeholder("gate", N_EMBD as u64, n_ff)?;
        let (up, up_bytes_per_expert) = placeholder("up", N_EMBD as u64, n_ff)?;
        let (down, down_bytes_per_expert) = placeholder("down", n_ff, N_EMBD as u64)?;
        Ok(IgpuLayerWeights {
            layer_idx: layer,
            ratio,
            is_hash_router,
            routed: RoutedExpertWeights {
                gate,
                up,
                down,
                gate_bytes_per_expert,
                up_bytes_per_expert,
                down_bytes_per_expert,
                n_slots: 1,
            },
            rope_params: rope_params_for_layer(layer)?,
        })
    }
}

/// M63: upload one routed-expert tensor keeping only `cold_ids`, packed dense
/// in slot order.
///
/// Streams a single expert at a time (pread → small host staging → device
/// sub-range) rather than materialising the whole tensor host-side: the
/// full-tensor path costs an 0.8–1.1 GiB transient `Vec` per tensor, which is
/// most of the load-time VmHWM spike.
/// DEEPSTRIX_EXPERT_LOAD_PROFILE=1: accumulate read vs device-copy time inside
/// the expert loop. Added 2026-09-12 after THREE failed optimisation attempts
/// (parallel-direct-to-device 95.9 s, parallel-with-staging 193 s, vs the
/// original 81.8-84.7 s) — all guesses made from a microbenchmark of a single
/// 18.8 MB extent, which is not what this loop does. Measure first.
pub static EXPERT_READ_S: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static EXPERT_COPY_S: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn expert_load_profile() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("DEEPSTRIX_EXPERT_LOAD_PROFILE").map(|v| v != "0").unwrap_or(false)
    });
    *ON
}







/// M63: drop the iGPU copies of the experts the dGPU holds (`IGPU_DEDUP_HOT`).
///
/// The iGPU MoE kernels already SKIP every dGPU-resident expert
/// (`mode 0: process slot iff remap[e] < 0`), so those bytes were read in
/// exactly one situation: when more than `DGPU_HOT_CAP` of a token's
/// `N_EXPERT_USED` picks were resident, the surplus fell back to the iGPU and
/// indexed at the raw expert id. Pinning the cap at `N_EXPERT_USED` makes that
/// branch unreachable, and the copies become dead weight worth
/// `K × N_LAYER × (gate+up+down)` bytes of GTT — ~2.4 GiB at K=8 on the
/// unsloth UD-IQ2_XXS mix.
///
/// Off by default until the GPU A/B lands.
pub fn igpu_dedup_hot() -> bool {
    std::env::var("IGPU_DEDUP_HOT").map(|v| v != "0").unwrap_or(false)
}


/// M58.3 leg balance: max dGPU-resident slots the dGPU computes per token;
/// the rest overflow back to the otherwise-idle iGPU. Default 4 (misses×32.5 µs
/// vs hits×12.3 µs + shared ~46 µs cross near h*≈3.3-4).
///
/// Single source of truth — the decode dGPU leg, the decode iGPU leg and the
/// prefill group builder must all pass the SAME value or the two devices
/// disagree about who owns a slot (dropped or double-counted experts).
///
/// M63 pins it at `N_EXPERT_USED` under de-dup: with no iGPU copy left, an
/// overflow slot has nowhere to run.
pub fn dgpu_hot_cap() -> u32 {
    static CAP: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        if igpu_dedup_hot() {
            return crate::config::N_EXPERT_USED as u32;
        }
        std::env::var("DGPU_HOT_CAP").ok().and_then(|s| s.parse().ok()).unwrap_or(4)
    });
    *CAP
}


/// Rewrite the miss branch of a dGPU remap into the iGPU slot encoding.
///
/// In: `remap[e]` = dGPU dense slot, or a bare -1 for "not resident".
/// Out: misses carry the iGPU slot as `-(slot + 1)`, which the MoE kernels
/// decode with `-remap[e] - 1`. Hits are untouched.
///
/// `packed` must match how the iGPU buffer was actually built: packed ⇒ slots
/// are the cold experts numbered in ascending order; otherwise slot == id.
pub fn encode_igpu_remap(remap: &mut [i32], packed: bool) {
    if packed {
        let mut slot = 0i32;
        for v in remap.iter_mut() {
            if *v < 0 {
                *v = -slot - 1;
                slot += 1;
            }
        }
    } else {
        for (e, v) in remap.iter_mut().enumerate() {
            if *v < 0 {
                *v = -(e as i32) - 1;
            }
        }
    }
}





impl HetModelWeights {
    pub fn load_all<'a>(
        gguf: impl Into<WeightSrc<'a>>,
        dgpu_device: Device,
        igpu_device: Device,
        rope_params_for_layer: &dyn Fn(i32) -> eyre::Result<RopeParams>,
    ) -> eyre::Result<Self> {
        let gguf: WeightSrc<'a> = gguf.into();
        // Fail up front with the COMPLETE list of contract violations
        // (unsupported dtypes / wrong dims) instead of erroring on the
        // first tensor mid-load — or worse, slicing at a wrong stride.
        let prof = expert_load_profile();
        let mark = std::time::Instant::now();
        weight_contract::validate_model(gguf.tensors())?;
        if let Some(shape) = gguf.model_shape()? {
            if shape.n_layer != N_LAYER as u32 || shape.n_embd != N_EMBD || shape.n_expert != N_EXPERT {
                return Err(eyre!(
                    "weights at {} are for a {}-layer / {}-wide / {}-expert model, but this binary is {} \
                     ({} layers / {} wide / {} experts); rebuild with the matching model feature",
                    gguf.path().display(), shape.n_layer, shape.n_embd, shape.n_expert,
                    crate::config::MODEL_NAME, N_LAYER, N_EMBD, N_EXPERT
                ));
            }
        }
        let t_validate = mark.elapsed().as_secs_f64();
        let mark = std::time::Instant::now();
        let global = HetGlobalWeights::load(gguf, dgpu_device)?;
        let t_global = mark.elapsed().as_secs_f64();

        let mut dgpu_layers = Vec::with_capacity(N_LAYER as usize);
        let mut igpu_layers = Vec::with_capacity(N_LAYER as usize);
        let (mut t_dgpu, mut t_igpu) = (0f64, 0f64);
        for layer in 0..N_LAYER {
            let mk = std::time::Instant::now();
            dgpu_layers.push(DgpuLayerWeights::load(
                gguf,
                dgpu_device,
                layer,
                rope_params_for_layer,
            )?);
            t_dgpu += mk.elapsed().as_secs_f64();
            let mk = std::time::Instant::now();
            igpu_layers.push(IgpuLayerWeights::load(
                gguf,
                igpu_device,
                layer,
                rope_params_for_layer,
            )?);
            t_igpu += mk.elapsed().as_secs_f64();
        }

        let mut weights = Self {
            global,
            dgpu_layers,
            igpu_layers,
        };
        if prof {
            let r = EXPERT_READ_S.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
            let c = EXPERT_COPY_S.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
            eprintln!(
                "LOAD PROFILE: validate {t_validate:.1} | global {t_global:.1} | \
                 dgpu-layers {t_dgpu:.1} | igpu-layers {t_igpu:.1} (of which expert pread \
                 {r:.1} + copy {c:.1})   [seconds]"
            );
        }
        // Vision-Exp routing bias sidecar (text-only models simply have none).
        if let Some(path) = weights.load_bias_vl_sidecar(gguf.path(), dgpu_device)? {
            eprintln!("vision: loaded bias_vl sidecar {}", path.display());
        }
        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_vl_sidecar_roundtrip_and_path() {
        let dir = std::env::temp_dir().join(format!("v4flash-biasvl-{}", std::process::id()));
        let p = dir.join(BIAS_VL_FILE);
        let n = (N_LAYER as usize) * (N_EXPERT as usize);
        let vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.001 - 3.0).collect();
        write_bias_vl_sidecar(&p, &vals).unwrap();
        assert_eq!(read_bias_vl_sidecar(&p).unwrap(), vals);
        // wrong size / NaN are rejected
        std::fs::write(&p, &[0u8; 16]).unwrap();
        assert!(read_bias_vl_sidecar(&p).is_err());
        std::fs::remove_dir_all(&dir).ok();
        if std::env::var_os("DEEPSTRIX_BIAS_VL_FILE").is_none() {
            let sp = bias_vl_sidecar_path(Path::new(
                "/x/y/DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003.gguf",
            ))
            .unwrap();
            assert!(sp.ends_with(
                ".cache/deepstrix/models/DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003/bias_vl.bin"
            ));
        }
    }

    /// What the MoE kernels do with the miss branch: `-remap[e] - 1`.
    fn decode(remap: &[i32], e: u32) -> i32 {
        -remap[e as usize] - 1
    }

    /// Build a dGPU-side remap the way the removed hot-tier loader did.
    fn dgpu_remap(hot_ids: &[u32]) -> Vec<i32> {
        let mut r = vec![-1i32; N_EXPERT as usize];
        for (dense, &e) in hot_ids.iter().enumerate() {
            r[e as usize] = dense as i32;
        }
        r
    }




    /// Without de-dup the encoding must reproduce the raw expert id — this is
    /// what keeps the kernel change a no-op for the default configuration.
    #[test]
    fn encoding_without_packing_decodes_to_the_raw_expert_id() {
        let hot = [7u32, 99];
        let mut remap = dgpu_remap(&hot);
        encode_igpu_remap(&mut remap, false);
        for e in 0..N_EXPERT as u32 {
            if hot.contains(&e) {
                assert!(remap[e as usize] >= 0, "hot expert {e} must stay a dGPU slot");
            } else {
                assert_eq!(decode(&remap, e), e as i32);
            }
        }
    }


    /// The skip predicate the kernels use (`remap[e] < 0` ⇔ iGPU computes it)
    /// must be invariant under the encoding — that is the whole reason the
    /// negative branch could be overloaded in the first place.
    #[test]
    fn encoding_preserves_the_residency_predicate() {
        let hot = [1u32, 2, 255];
        for packed in [false, true] {
            let before = dgpu_remap(&hot);
            let mut after = before.clone();
            encode_igpu_remap(&mut after, packed);
            for e in 0..N_EXPERT as usize {
                assert_eq!(before[e] >= 0, after[e] >= 0, "expert {e}, packed={packed}");
            }
        }
    }
}
