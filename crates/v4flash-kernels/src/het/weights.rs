//! Per-device weight splits for V4-Flash.
//!
//! * [`DgpuLayerWeights`] — attention LoRAs, mHC, compressor, shared
//!   expert, attention norms, router. ~200 MiB/layer × 43 layers = ~9 GiB.
//! * [`IgpuLayerWeights`] — routed MoE (gate/up/down for 256 experts).
//!   ~1.2 GiB/layer × 43 layers = ~52 GiB.
//! * [`HetGlobalWeights`] — output head + embedding, dGPU-resident.

use std::path::{Path, PathBuf};

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
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

pub struct DgpuLayerWeights {
    pub layer_idx: i32,
    pub ratio: u32,

    /// M56 het-split: dGPU-resident hot routed experts (None = feature off).
    pub hot_experts: Option<HotExpertWeights>,

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
}

pub struct IgpuLayerWeights {
    pub layer_idx: i32,
    pub ratio: u32,
    pub is_hash_router: bool,

    /// Routed experts. All 256 by default; with `IGPU_DEDUP_HOT` only the
    /// `256 - n_hot` experts that are NOT dGPU-resident, packed dense
    /// (`routed.n_slots`).
    pub routed: RoutedExpertWeights,

    /// M56 het-split: iGPU-resident copy of the resident-expert remap
    /// (remap[e] >= 0 ⇔ expert e is ALSO dGPU-resident; the iGPU MoE
    /// kernels then skip those slots). None = feature off.
    ///
    /// M63: the negative branch is no longer a bare -1 — it carries this
    /// layer's iGPU slot for expert e as `-(slot + 1)`. Without de-dup
    /// `slot == e`, so the kernels' `-remap[e] - 1` is the old raw id.
    pub hot_remap: Option<DeviceBuffer<i32>>,

    /// M63: true when `routed` holds only the cold experts. Every iGPU MoE
    /// launch for this layer MUST then go through the hetsplit kernels —
    /// the plain ones index by raw expert id and would read the wrong
    /// expert. Enforced at load by `validate_dedup_preconditions`.
    pub igpu_packed: bool,

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

pub struct HetModelWeights {
    pub global: HetGlobalWeights,
    pub dgpu_layers: Vec<DgpuLayerWeights>,
    pub igpu_layers: Vec<IgpuLayerWeights>,
}

impl HetGlobalWeights {
    pub fn load(gguf: &MappedGguf, dgpu_device: Device) -> eyre::Result<Self> {
        dgpu_device.set_current()?;
        let dgpu_id = dgpu_device.id;
        // Validate token_embd dtype without uploading (host-side embed path).
        {
            let te = gguf
                .gguf()
                .tensor("token_embd.weight")
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
        let output_hc_fn = load_to_device(gguf, "output_hc_fn.weight", dgpu_id)?;
        let output_hc_scale = load_f32_weight(gguf, "output_hc_scale.weight", dgpu_id, 1)?;
        let output_hc_base =
            load_f32_weight(gguf, "output_hc_base.weight", dgpu_id, N_HC as usize)?;
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
    pub fn load(
        gguf: &MappedGguf,
        dgpu_device: Device,
        layer: i32,
        rope_params_for_layer: &dyn Fn(i32) -> eyre::Result<RopeParams>,
    ) -> eyre::Result<Self> {
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
        let compressor = if ratio > 0 {
            let comp_width = if ratio == 4 { 1024 } else { 512 };
            Some(CompressorWeights {
                wkv: load_to_device(
                    gguf,
                    &format!("blk.{layer}.attn_compressor_kv.weight"),
                    device_id,
                )?,
                wgate: load_to_device(
                    gguf,
                    &format!("blk.{layer}.attn_compressor_gate.weight"),
                    device_id,
                )?,
                ape: load_to_device(
                    gguf,
                    &format!("blk.{layer}.attn_compressor_ape.weight"),
                    device_id,
                )?,
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
            if let Some(t) = gguf.gguf().tensor(&bias_name) {
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
            hot_experts: None,
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
    /// `cold_ids` is this layer's iGPU slot space: the expert ids to keep,
    /// in slot order. Pass all of `0..N_EXPERT` for the classic full-residency
    /// layout; pass the complement of the dGPU-resident set for M63 de-dup.
    pub fn load(
        gguf: &MappedGguf,
        igpu_device: Device,
        layer: i32,
        cold_ids: &[u32],
        rope_params_for_layer: &dyn Fn(i32) -> eyre::Result<RopeParams>,
    ) -> eyre::Result<Self> {
        igpu_device.set_current()?;
        let device_id = igpu_device.id;
        let ratio = COMPRESS_RATIOS[layer as usize];
        let is_hash_router = layer < N_HASH_LAYERS;
        let igpu_packed = cold_ids.len() != N_EXPERT as usize;

        // Routed expert weights — iGPU-resident (~1.2 GiB/layer at full
        // residency). The full-residency path is kept byte-for-byte as it
        // was: a straight whole-tensor upload with no repacking.
        let (gate, up, down) = if igpu_packed {
            (
                load_experts_packed(gguf, layer, "gate", device_id, cold_ids)?,
                load_experts_packed(gguf, layer, "up", device_id, cold_ids)?,
                load_experts_packed(gguf, layer, "down", device_id, cold_ids)?,
            )
        } else {
            (
                load_to_device(gguf, &format!("blk.{layer}.ffn_gate_exps.weight"), device_id)?,
                load_to_device(gguf, &format!("blk.{layer}.ffn_up_exps.weight"), device_id)?,
                load_to_device(gguf, &format!("blk.{layer}.ffn_down_exps.weight"), device_id)?,
            )
        };
        // Strides from the actual dtype, not compile-time constants — the
        // unsloth UD mix varies expert dtypes per layer (blk.26 gate/up
        // IQ2_S, blk.26/42 down MXFP4). Cross-checked against the real
        // buffer size so a stride bug is a load error, not garbage output.
        let n_ff = crate::config::N_FF_EXP as u64;
        let gate_bytes_per_expert =
            weight_contract::bytes_per_expert(gate.dtype, N_EMBD as u64, n_ff)?;
        let up_bytes_per_expert =
            weight_contract::bytes_per_expert(up.dtype, N_EMBD as u64, n_ff)?;
        let down_bytes_per_expert =
            weight_contract::bytes_per_expert(down.dtype, n_ff, N_EMBD as u64)?;
        let n_slots = cold_ids.len();
        for (name, w, bpe) in [
            ("gate_exps", &gate, gate_bytes_per_expert),
            ("up_exps", &up, up_bytes_per_expert),
            ("down_exps", &down, down_bytes_per_expert),
        ] {
            let expect = n_slots * bpe;
            if w.buffer.len() != expect {
                return Err(eyre!(
                    "blk.{layer}.ffn_{name}: buffer {} B != {} experts × {} B/expert ({:?})",
                    w.buffer.len(),
                    n_slots,
                    bpe,
                    w.dtype
                ));
            }
        }
        let routed = RoutedExpertWeights {
            gate,
            up,
            down,
            gate_bytes_per_expert,
            up_bytes_per_expert,
            down_bytes_per_expert,
            n_slots: n_slots as u32,
        };

        let rope_params = rope_params_for_layer(layer)?;

        Ok(IgpuLayerWeights {
            layer_idx: layer,
            ratio,
            is_hash_router,
            routed,
            hot_remap: None,
            igpu_packed,
            rope_params,
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
fn load_experts_packed(
    gguf: &MappedGguf,
    layer: i32,
    which: &str,
    device_id: i32,
    cold_ids: &[u32],
) -> eyre::Result<DeviceWeight> {
    let name = format!("blk.{layer}.ffn_{which}_exps.weight");
    let tensor = gguf
        .gguf()
        .tensor(&name)
        .ok_or_else(|| eyre!("tensor `{name}` not found in GGUF"))?;
    let n_ff = crate::config::N_FF_EXP as u64;
    let (k, rows) = if which == "down" {
        (n_ff, N_EMBD as u64)
    } else {
        (N_EMBD as u64, n_ff)
    };
    let bpe = weight_contract::bytes_per_expert(tensor.dtype, k, rows)?;
    if tensor.byte_size as usize != (N_EXPERT as usize) * bpe {
        return Err(eyre!(
            "{name}: byte_size {} != {} experts × {bpe} B/expert ({:?})",
            tensor.byte_size,
            N_EXPERT,
            tensor.dtype
        ));
    }
    let mut buffer = DeviceBuffer::<u8>::new(device_id, cold_ids.len() * bpe)?;
    let mut staging = vec![0u8; bpe];
    for (slot, &e) in cold_ids.iter().enumerate() {
        gguf.read_range_into(
            tensor.shard,
            tensor.abs_offset + (e as u64) * (bpe as u64),
            &mut staging,
        )?;
        buffer
            .slice_view_mut(slot * bpe, bpe)
            .copy_from_host(&staging)?;
    }
    Ok(DeviceWeight {
        buffer,
        n_elements: (cold_ids.len() as u64) * k * rows,
        dtype: tensor.dtype,
        shape: vec![k, rows, cold_ids.len() as u64],
    })
}

/// M56: dGPU-resident copies of the K hottest routed experts of one layer
/// (packed dense), plus the id→dense remap. The dGPU computes these picks
/// during its former MoE wait; the iGPU skips them.
pub struct HotExpertWeights {
    pub gate: DeviceBuffer<u8>,
    pub up: DeviceBuffer<u8>,
    pub down: DeviceBuffer<u8>,
    /// remap[e] = dense slot in the packed buffers, or -1.
    pub remap: DeviceBuffer<i32>,
    pub n_hot: u32,
}

impl HotExpertWeights {
    pub fn load(
        gguf: &MappedGguf,
        dgpu_device: Device,
        layer: i32,
        expert_ids: &[u32],
    ) -> eyre::Result<(Self, Vec<i32>)> {
        dgpu_device.set_current()?;
        let device_id = dgpu_device.id;
        let k = expert_ids.len();

        let mut remap_host = vec![-1i32; 256];
        for (dense, &e) in expert_ids.iter().enumerate() {
            remap_host[e as usize] = dense as i32;
        }

        // Per-expert stride from each tensor's actual dtype (per-layer
        // variable in the unsloth UD mix), cross-checked against the
        // tensor's real byte size — the second copy of the old
        // hardcoded-stride bug lived here.
        let pack = |name: &str, kdim: u64, rows: u64| -> eyre::Result<DeviceBuffer<u8>> {
            let tensor = gguf
                .gguf()
                .tensor(name)
                .ok_or_else(|| eyre!("tensor `{name}` not found"))?;
            let bpe = weight_contract::bytes_per_expert(tensor.dtype, kdim, rows)?;
            if tensor.byte_size as usize != (N_EXPERT as usize) * bpe {
                return Err(eyre!(
                    "{name}: byte_size {} != {} experts × {} B/expert ({:?})",
                    tensor.byte_size,
                    N_EXPERT,
                    bpe,
                    tensor.dtype
                ));
            }
            let host = gguf.read_tensor(tensor)?;
            let mut packed = vec![0u8; k * bpe];
            for (dense, &e) in expert_ids.iter().enumerate() {
                let src = &host[(e as usize) * bpe..(e as usize + 1) * bpe];
                packed[dense * bpe..(dense + 1) * bpe].copy_from_slice(src);
            }
            let mut buf = DeviceBuffer::<u8>::new(device_id, packed.len())?;
            buf.copy_from_host(&packed)?;
            Ok(buf)
        };

        let n_ff = crate::config::N_FF_EXP as u64;
        let gate = pack(&format!("blk.{layer}.ffn_gate_exps.weight"), N_EMBD as u64, n_ff)?;
        let up = pack(&format!("blk.{layer}.ffn_up_exps.weight"), N_EMBD as u64, n_ff)?;
        let down = pack(&format!("blk.{layer}.ffn_down_exps.weight"), n_ff, N_EMBD as u64)?;
        let mut remap = DeviceBuffer::<i32>::new(device_id, 256)?;
        remap.copy_from_host(&remap_host)?;

        Ok((
            HotExpertWeights {
                gate,
                up,
                down,
                remap,
                n_hot: k as u32,
            },
            remap_host,
        ))
    }
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

/// This layer's iGPU slot space: the expert ids the iGPU keeps, in slot order.
///
/// The identity `0..N_EXPERT` unless de-dup is on AND the dGPU actually takes
/// experts here — a layer the global-greedy allocator gave zero slots stays
/// fully resident, and so keeps working through the plain kernels.
pub fn igpu_slot_space(hot_ids: &[u32], dedup: bool) -> Vec<u32> {
    if !dedup || hot_ids.is_empty() {
        return (0..N_EXPERT as u32).collect();
    }
    let mut is_hot = [false; N_EXPERT as usize];
    for &e in hot_ids {
        is_hot[e as usize] = true;
    }
    (0..N_EXPERT as u32).filter(|e| !is_hot[*e as usize]).collect()
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

/// Everything that would let a raw expert id reach a packed iGPU buffer.
///
/// All of these are load-time-decidable, so de-dup fails the load rather than
/// silently computing the wrong expert. The runtime predicates that gate the
/// het-split (`prefill_hot_active`, and the decode `hot_experts.is_some()`
/// check) are all derived from these same inputs.
fn validate_dedup_preconditions(placement: &[Vec<u32>]) -> eyre::Result<()> {
    let n_used = crate::config::N_EXPERT_USED as u32;
    let mut problems = Vec::new();
    for (var, val) in [
        ("DGPU_HOT_CAP", std::env::var("DGPU_HOT_CAP").ok()),
        ("DGPU_HOT_CAP_PREFILL", std::env::var("DGPU_HOT_CAP_PREFILL").ok()),
    ] {
        if let Some(v) = val.as_deref().and_then(|s| s.parse::<u32>().ok()) {
            if v < n_used {
                problems.push(format!(
                    "{var}={v} < N_EXPERT_USED={n_used}: the overflow slots would fall back to \
                     the iGPU, whose copies IGPU_DEDUP_HOT removes"
                ));
            }
        }
    }
    if std::env::var("DGPU_HOT_PREFILL").map(|v| v == "0").unwrap_or(false) {
        problems.push(
            "DGPU_HOT_PREFILL=0 routes prefill through the plain by-expert builder, which \
             indexes by raw expert id"
                .into(),
        );
    }
    if placement.iter().all(|l| l.is_empty()) {
        problems.push(
            "no hot experts placed (DGPU_HOT_EXPERTS=0 or unusable placement file) — nothing \
             to de-duplicate"
                .into(),
        );
    }
    if problems.is_empty() {
        return Ok(());
    }
    Err(eyre!(
        "IGPU_DEDUP_HOT is set but cannot be used safely:\n  - {}",
        problems.join("\n  - ")
    ))
}

/// Path to the hot-expert placement file (`DGPU_HOT_EXPERTS_FILE`, else the
/// in-repo default). Relative, so it resolves against the process CWD — the
/// server runs from the repo root.
pub fn hot_expert_file_path() -> String {
    std::env::var("DGPU_HOT_EXPERTS_FILE")
        .unwrap_or_else(|_| "reference/decode_hot_experts.txt".into())
}

/// Hot experts per layer to mirror onto the dGPU (M56 het-split).
///
/// `DGPU_HOT_EXPERTS` overrides. The DEFAULT is 6 **when a placement file is
/// actually present**, and 0 otherwise.
///
/// Why not a flat default: the het-split is worth ~+16% decode, and defaulting
/// it to 0 meant a bare `deepstrix-server` silently ran without it — no error,
/// just a slower server. Why gate on the file: residency cannot be built
/// without placement data, so with no file the answer is 0 regardless, and
/// this cannot spend dGPU VRAM on a feature that can't engage. K=6 matches the
/// known-good production launch (K<=6 fits alongside a 192K KV cache; K=8
/// overflows the dGPU budget past 128K).
///
/// NOTE: this is RESIDENCY. `DGPU_HOT_CAP` (default 4) separately caps how many
/// slots per token the dGPU actually computes; overflow returns to the iGPU.
pub fn dgpu_hot_experts() -> usize {
    if let Some(k) = std::env::var("DGPU_HOT_EXPERTS").ok().and_then(|s| s.parse().ok()) {
        return k;
    }
    if std::path::Path::new(&hot_expert_file_path()).exists() { 6 } else { 0 }
}

/// Parse the hot-expert placement file and allocate the slot budget.
///
/// File format (DEEPSTRIX_EXPERT_STATS probe): one line per layer,
/// descending-frequency `id` or `id:count` entries. With counts present,
/// the TOTAL budget of `k_avg × N_LAYER` slots is allocated by GLOBAL
/// GREEDY — all (layer, expert) pairs ranked by count, top budget taken —
/// so skewed layers get more slots than flat ones (optimal for expected
/// hits under a stationary distribution: every expert costs the same
/// 7.08 MB and a miss in any layer is equally serial). Legacy count-less
/// files fall back to uniform per-layer top-k.
pub fn parse_hot_expert_file(path: &str, k_avg: usize) -> eyre::Result<Vec<Vec<u32>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| eyre!("read hot-expert file {path}: {e}"))?;
    let mut per_layer: Vec<Vec<(u32, u64)>> = Vec::with_capacity(N_LAYER as usize);
    let mut have_counts = true;
    for line in text.lines().take(N_LAYER as usize) {
        let mut row = Vec::new();
        for tok in line.split(',') {
            let tok = tok.trim();
            if tok.is_empty() {
                continue;
            }
            if let Some((id, cnt)) = tok.split_once(':') {
                if let (Ok(id), Ok(cnt)) = (id.parse::<u32>(), cnt.parse::<u64>()) {
                    row.push((id, cnt));
                }
            } else if let Ok(id) = tok.parse::<u32>() {
                have_counts = false;
                row.push((id, 0));
            }
        }
        per_layer.push(row);
    }
    if per_layer.len() != N_LAYER as usize {
        return Err(eyre!(
            "hot-expert file {path}: {} lines, need {}",
            per_layer.len(),
            N_LAYER
        ));
    }
    let budget = k_avg * N_LAYER as usize;
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); N_LAYER as usize];
    if have_counts {
        let mut all: Vec<(u64, usize, u32)> = Vec::new();
        for (l, row) in per_layer.iter().enumerate() {
            for &(id, cnt) in row {
                all.push((cnt, l, id));
            }
        }
        all.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        for &(_, l, id) in all.iter().take(budget) {
            out[l].push(id);
        }
        let (min_k, max_k) = (
            out.iter().map(|v| v.len()).min().unwrap_or(0),
            out.iter().map(|v| v.len()).max().unwrap_or(0),
        );
        eprintln!(
            "hot-expert global-greedy: budget {budget} slots, per-layer K range {min_k}..{max_k}"
        );
    } else {
        for (l, row) in per_layer.iter().enumerate() {
            out[l] = row.iter().take(k_avg).map(|&(id, _)| id).collect();
        }
    }
    Ok(out)
}

impl HetModelWeights {
    pub fn load_all(
        gguf: &MappedGguf,
        dgpu_device: Device,
        igpu_device: Device,
        rope_params_for_layer: &dyn Fn(i32) -> eyre::Result<RopeParams>,
    ) -> eyre::Result<Self> {
        // Fail up front with the COMPLETE list of contract violations
        // (unsupported dtypes / wrong dims) instead of erroring on the
        // first tensor mid-load — or worse, slicing at a wrong stride.
        weight_contract::validate_model(gguf.gguf())?;
        let global = HetGlobalWeights::load(gguf, dgpu_device)?;

        // M56 het-split: place the K hottest experts per layer on the dGPU too.
        // K defaults to 6 when a placement file exists (see dgpu_hot_experts).
        //
        // M63: resolved BEFORE the layer loop, because the iGPU upload now
        // needs to know which experts the dGPU will take in order to skip
        // them. A placement file that won't parse disables the split (warn,
        // as before) rather than failing the load.
        let hot_k: usize = dgpu_hot_experts();
        let placement: Vec<Vec<u32>> = if hot_k > 0 {
            let path = hot_expert_file_path();
            match parse_hot_expert_file(&path, hot_k) {
                Ok(lists) => lists,
                Err(e) => {
                    eprintln!("WARN DGPU_HOT_EXPERTS set but placement file unusable: {e}");
                    vec![Vec::new(); N_LAYER as usize]
                }
            }
        } else {
            vec![Vec::new(); N_LAYER as usize]
        };
        let dedup = igpu_dedup_hot();
        if dedup {
            validate_dedup_preconditions(&placement)?;
        }

        let mut dgpu_layers = Vec::with_capacity(N_LAYER as usize);
        let mut igpu_layers = Vec::with_capacity(N_LAYER as usize);
        let mut freed_bytes: u64 = 0;
        for layer in 0..N_LAYER {
            let cold_ids = igpu_slot_space(&placement[layer as usize], dedup);
            dgpu_layers.push(DgpuLayerWeights::load(
                gguf,
                dgpu_device,
                layer,
                rope_params_for_layer,
            )?);
            igpu_layers.push(IgpuLayerWeights::load(
                gguf,
                igpu_device,
                layer,
                &cold_ids,
                rope_params_for_layer,
            )?);
            let r = &igpu_layers[layer as usize].routed;
            freed_bytes += (N_EXPERT as u64 - r.n_slots as u64)
                * (r.gate_bytes_per_expert + r.up_bytes_per_expert + r.down_bytes_per_expert)
                    as u64;
        }

        if hot_k > 0 {
            for layer in 0..N_LAYER as usize {
                if placement[layer].is_empty() {
                    // Global greedy gave this (flat-routing) layer zero
                    // slots — leave it fully iGPU-resident.
                    continue;
                }
                match HotExpertWeights::load(gguf, dgpu_device, layer as i32, &placement[layer]) {
                    Ok((hot, remap_host)) => {
                        // M63: rewrite the miss branch from a bare -1 to
                        // -(iGPU slot + 1). Without de-dup that is -(e + 1),
                        // which the kernels decode straight back to e.
                        let mut remap_host = remap_host;
                        encode_igpu_remap(&mut remap_host, igpu_layers[layer].igpu_packed);
                        igpu_device.set_current()?;
                        let mut r = DeviceBuffer::<i32>::new(igpu_device.id, 256)?;
                        r.copy_from_host(&remap_host)?;
                        igpu_layers[layer].hot_remap = Some(r);
                        dgpu_layers[layer].hot_experts = Some(hot);
                    }
                    Err(e) => {
                        if igpu_layers[layer].igpu_packed {
                            // The iGPU already dropped these experts — there
                            // is no fallback path that can compute them.
                            return Err(eyre!(
                                "IGPU_DEDUP_HOT: hot-expert load failed at L{layer} and the \
                                 iGPU copies are already gone: {e}"
                            ));
                        }
                        eprintln!("WARN hot-expert load failed at L{layer} (disabled): {e}");
                    }
                }
            }
            dgpu_device.set_current()?;
            eprintln!("M56 het-split: {hot_k} hot experts/layer resident on dGPU");
            if dedup {
                eprintln!(
                    "M63 iGPU de-dup: dropped {:.2} GiB of duplicate iGPU expert copies",
                    freed_bytes as f64 / (1u64 << 30) as f64
                );
            }
        }
        let mut weights = Self {
            global,
            dgpu_layers,
            igpu_layers,
        };
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

    /// Build the dGPU-side remap exactly as `HotExpertWeights::load` does.
    fn dgpu_remap(hot_ids: &[u32]) -> Vec<i32> {
        let mut r = vec![-1i32; N_EXPERT as usize];
        for (dense, &e) in hot_ids.iter().enumerate() {
            r[e as usize] = dense as i32;
        }
        r
    }

    #[test]
    fn slot_space_is_identity_without_dedup() {
        let hot = [3u32, 200, 41];
        let cold = igpu_slot_space(&hot, false);
        assert_eq!(cold.len(), N_EXPERT as usize);
        assert!(cold.iter().enumerate().all(|(i, &e)| i as u32 == e));
    }

    /// A layer the allocator gave no slots must stay fully resident even with
    /// de-dup on — it still runs the plain, raw-id kernels.
    #[test]
    fn slot_space_is_identity_for_a_layer_with_no_hot_experts() {
        assert_eq!(igpu_slot_space(&[], true).len(), N_EXPERT as usize);
    }

    #[test]
    fn slot_space_drops_hot_experts_and_stays_ascending() {
        let hot = [200u32, 3, 41]; // placement order is by frequency, not id
        let cold = igpu_slot_space(&hot, true);
        assert_eq!(cold.len(), N_EXPERT as usize - 3);
        assert!(!cold.contains(&3) && !cold.contains(&41) && !cold.contains(&200));
        assert!(cold.windows(2).all(|w| w[0] < w[1]));
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

    /// With packing, decoding a cold expert must land on the slot that
    /// `load_experts_packed` actually wrote it to — i.e. its index in
    /// `igpu_slot_space`.
    #[test]
    fn encoding_with_packing_decodes_to_the_packed_slot() {
        let hot = [200u32, 3, 41];
        let cold = igpu_slot_space(&hot, true);
        let mut remap = dgpu_remap(&hot);
        encode_igpu_remap(&mut remap, true);
        for (slot, &e) in cold.iter().enumerate() {
            assert_eq!(decode(&remap, e), slot as i32, "expert {e}");
        }
        // Hits keep their dGPU dense slot, so the skip predicate (>= 0) and
        // the dGPU weight index are both unchanged.
        for (dense, &e) in hot.iter().enumerate() {
            assert_eq!(remap[e as usize], dense as i32);
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
