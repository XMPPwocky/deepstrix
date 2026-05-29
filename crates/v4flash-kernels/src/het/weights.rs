//! Per-device weight splits for V4-Flash.
//!
//! * [`DgpuLayerWeights`] — attention LoRAs, mHC, compressor (M13.1; moves
//!   to iGPU in M13.5), shared expert, attention norms. ~200 MiB/layer ×
//!   43 layers = ~9 GiB.
//! * [`IgpuLayerWeights`] — routed MoE (gate/up/down for 256 experts) +
//!   router (ffn_gate_inp + tid2eid/router_bias). ~1.2 GiB/layer ×
//!   43 layers = ~52 GiB.
//! * [`HetGlobalWeights`] — output head + embedding, dGPU-resident.

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{Device, DeviceBuffer};

use crate::config::{
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, COMPRESS_RATIOS, HC_MIX_DIM, N_EMBD, N_HASH_LAYERS,
    N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LAYER, N_LORA_Q,
};
use crate::model_weights::{
    load_f32_weight, load_i32_tensor, CompressorWeights, IndexerWeights, RoutedExpertWeights,
    SharedExpertWeights,
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

    // M14L: compressor migrated to dGPU. Loaded here instead of on iGPU
    // because (a) 9070 XT has 2.6× the BW of Strix iGPU → f16 matvec is
    // faster locally, and (b) attn_input_norm is computed on dGPU, so
    // running compressor on dGPU eliminates a 16-byte peer push and
    // the iGPU.compressor.wait it gates.
    pub compressor: Option<CompressorWeights>,

    // M16: router migrated to dGPU. The dGPU has 2.6× iGPU BW so the
    // router matvec is ~1.5 ms faster; more importantly running it on
    // dGPU lifts it off the iGPU's critical path (iGPU.router.wait was
    // gating MoE start). After the router runs on dGPU, the resulting
    // selected/d_ew are peer-pushed to iGPU and the MoE pipeline starts
    // there immediately upon their arrival.
    pub is_hash_router: bool,
    pub ffn_gate_inp: DeviceWeight,
    pub tid2eid: Option<Vec<i32>>,
    pub router_bias_dev: Option<DeviceBuffer<f32>>,
}

pub struct IgpuLayerWeights {
    pub layer_idx: i32,
    pub ratio: u32,
    pub is_hash_router: bool,

    // Router — M16: moved to dGPU. iGPU keeps these as None to avoid
    // duplicate ~86 MB of router weights (was OOM'ing the system).
    pub ffn_gate_inp: Option<DeviceWeight>,
    pub tid2eid: Option<Vec<i32>>,
    pub router_bias: Option<Vec<f32>>,
    /// Device-resident copy of `router_bias` for the M13.3 `router_topk`
    /// kernel. Only present for learned routers (L≥3).
    pub router_bias_dev: Option<DeviceBuffer<f32>>,

    // Routed experts (all 256, iGPU-resident)
    pub routed: RoutedExpertWeights,

    // CSA producers (M13.5: now iGPU-resident; iGPU runs the
    // compressor and peer-pushes `comp_row` to the dGPU's `comp_kv` on
    // boundaries).
    pub compressor: Option<CompressorWeights>,
    pub indexer_compressor: Option<CompressorWeights>,
    pub indexer: Option<IndexerWeights>,
    /// Per-layer RoPE params (compressor needs them for the boundary
    /// comp_row RoPE pass; copy mirrors dGPU's).
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

        // M14L: compressor weights now on dGPU.
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

        // M16: router weights on dGPU.
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

        // M40-P5.1: router moved BACK to iGPU for pair-forward, where
        // starting iGPU MoE doesn't have to wait for dGPU shared_expert.
        // ffn_gate_inp is small (~2 MB/layer F16) — the 86 MB OOM from M16
        // came from putting all the LoRA stuff on iGPU too. Just the router
        // matvec weight is cheap. router_bias for learned routers also lives
        // here (small, ~1 KB).
        let ffn_gate_inp: Option<DeviceWeight> = Some(load_to_device(
            gguf,
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            device_id,
        )?);
        let tid2eid: Option<Vec<i32>> = if is_hash_router {
            Some(load_i32_tensor(
                gguf,
                &format!("blk.{layer}.ffn_gate_tid2eid.weight"),
            )?)
        } else {
            None
        };
        let router_bias: Option<Vec<f32>> = if !is_hash_router {
            let bias_name = format!("blk.{layer}.exp_probs_b.bias");
            if let Some(t) = gguf.gguf().tensor(&bias_name) {
                if t.dtype != GgufType::F32 {
                    return Err(eyre!("{bias_name} dtype {:?} != F32", t.dtype));
                }
                let bytes = gguf.read_tensor(t)?;
                Some(
                    bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            } else {
                None
            }
        } else {
            None
        };

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

        // M40-P5.1: router_bias_dev mirrors router_bias on iGPU.
        let router_bias_dev: Option<DeviceBuffer<f32>> = if let Some(b) = router_bias.as_ref() {
            let mut buf = DeviceBuffer::<f32>::new(device_id, b.len())?;
            buf.copy_from_host(b)?;
            Some(buf)
        } else {
            None
        };

        // M14L: attn compressor moved to dGPU. Field kept (None) so the
        // IgpuLayerWeights shape is unchanged for non-compressor consumers
        // (indexer_compressor below remains iGPU-resident).
        let compressor: Option<CompressorWeights> = None;
        let indexer_compressor = if ratio == 4 {
            Some(CompressorWeights {
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
                width: 256,
                head_dim: N_INDEXER_HEAD_DIM,
            })
        } else {
            None
        };
        let indexer = if ratio == 4 {
            Some(IndexerWeights {
                q_b: load_to_device(
                    gguf,
                    &format!("blk.{layer}.indexer.attn_q_b.weight"),
                    device_id,
                )?,
                proj: load_to_device(
                    gguf,
                    &format!("blk.{layer}.indexer.proj.weight"),
                    device_id,
                )?,
            })
        } else {
            None
        };
        let rope_params = rope_params_for_layer(layer)?;

        Ok(IgpuLayerWeights {
            layer_idx: layer,
            ratio,
            is_hash_router,
            ffn_gate_inp,
            tid2eid,
            router_bias,
            router_bias_dev,
            routed,
            compressor,
            indexer_compressor,
            indexer,
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
