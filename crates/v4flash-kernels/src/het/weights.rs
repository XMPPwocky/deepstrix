//! Per-device weight splits for V4-Flash.
//!
//! * [`DgpuLayerWeights`] — attention LoRAs, mHC, compressor, shared
//!   expert, attention norms, router. ~200 MiB/layer × 43 layers = ~9 GiB.
//! * [`IgpuLayerWeights`] — routed MoE (gate/up/down for 256 experts).
//!   ~1.2 GiB/layer × 43 layers = ~52 GiB.
//! * [`HetGlobalWeights`] — output head + embedding, dGPU-resident.

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{Device, DeviceBuffer};

use crate::config::{
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, COMPRESS_RATIOS, HC_MIX_DIM, INDEXER_COMP_WIDTH,
    N_EMBD, N_HASH_LAYERS, N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LAYER, N_LORA_Q,
};
use crate::model_weights::{
    load_f32_weight, load_i32_tensor, CompressorWeights, RoutedExpertWeights, SharedExpertWeights,
};
use crate::iq2_xxs::BLOCK_IQ2_XXS_BYTES;
use crate::q2_k::BLOCK_Q2_K_BYTES;
use crate::rope::RopeParams;
use crate::weights::{load_to_device, DeviceWeight};

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

    // Routed experts (all 256, iGPU-resident)
    pub routed: RoutedExpertWeights,

    /// Per-layer RoPE params. Mirrors the dGPU side for any future
    /// iGPU-resident RoPE call (currently unused — all RoPE runs on dGPU).
    pub rope_params: RopeParams,
}

pub struct HetGlobalWeights {
    /// Embedding lookup on dGPU (output head also lives here).
    pub token_embd: DeviceWeight,
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
        let token_embd = load_to_device(gguf, "token_embd.weight", dgpu_id)?;
        if token_embd.dtype != GgufType::F16 {
            return Err(eyre!("token_embd dtype {:?} != F16", token_embd.dtype));
        }
        let output = load_to_device(gguf, "output.weight", dgpu_id)?;
        if output.dtype != GgufType::Q8_0 {
            return Err(eyre!("output dtype {:?} != Q8_0", output.dtype));
        }
        let output_norm = load_f32_weight(gguf, "output_norm.weight", dgpu_id, N_EMBD as usize)?;
        let output_hc_fn = load_to_device(gguf, "output_hc_fn.weight", dgpu_id)?;
        let output_hc_scale = load_f32_weight(gguf, "output_hc_scale.weight", dgpu_id, 1)?;
        let output_hc_base =
            load_f32_weight(gguf, "output_hc_base.weight", dgpu_id, N_HC as usize)?;
        Ok(Self {
            token_embd,
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
        })
    }
}

impl IgpuLayerWeights {
    pub fn load(
        gguf: &MappedGguf,
        igpu_device: Device,
        layer: i32,
        rope_params_for_layer: &dyn Fn(i32) -> eyre::Result<RopeParams>,
    ) -> eyre::Result<Self> {
        igpu_device.set_current()?;
        let device_id = igpu_device.id;
        let ratio = COMPRESS_RATIOS[layer as usize];
        let is_hash_router = layer < N_HASH_LAYERS;

        // Routed expert weights — fully iGPU-resident (~1.2 GiB/layer).
        let gate = load_to_device(
            gguf,
            &format!("blk.{layer}.ffn_gate_exps.weight"),
            device_id,
        )?;
        let up = load_to_device(gguf, &format!("blk.{layer}.ffn_up_exps.weight"), device_id)?;
        let down = load_to_device(
            gguf,
            &format!("blk.{layer}.ffn_down_exps.weight"),
            device_id,
        )?;
        let gate_bytes_per_expert =
            (crate::config::N_FF_EXP as usize) * (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_IQ2_XXS_BYTES;
        let up_bytes_per_expert = gate_bytes_per_expert;
        let down_bytes_per_expert =
            (N_EMBD as usize) * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q2_K_BYTES;
        let routed = RoutedExpertWeights {
            gate,
            up,
            down,
            gate_bytes_per_expert,
            up_bytes_per_expert,
            down_bytes_per_expert,
        };

        let rope_params = rope_params_for_layer(layer)?;

        Ok(IgpuLayerWeights {
            layer_idx: layer,
            ratio,
            is_hash_router,
            routed,
            rope_params,
        })
    }
}

impl HetModelWeights {
    pub fn load_all(
        gguf: &MappedGguf,
        dgpu_device: Device,
        igpu_device: Device,
        rope_params_for_layer: &dyn Fn(i32) -> eyre::Result<RopeParams>,
    ) -> eyre::Result<Self> {
        let global = HetGlobalWeights::load(gguf, dgpu_device)?;
        let mut dgpu_layers = Vec::with_capacity(N_LAYER as usize);
        let mut igpu_layers = Vec::with_capacity(N_LAYER as usize);
        for layer in 0..N_LAYER {
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
                rope_params_for_layer,
            )?);
        }
        Ok(Self {
            global,
            dgpu_layers,
            igpu_layers,
        })
    }
}
