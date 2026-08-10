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

    /// M56 het-split: iGPU-resident copy of the resident-expert remap
    /// (remap[e] >= 0 ⇔ expert e is ALSO dGPU-resident; the iGPU MoE
    /// kernels then skip those slots). None = feature off.
    pub hot_remap: Option<DeviceBuffer<i32>>,

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
            if te.dtype != GgufType::F16 {
                return Err(eyre!("token_embd dtype {:?} != F16", te.dtype));
            }
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
            hot_remap: None,
            rope_params,
        })
    }
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
        let gate_bpe = (crate::config::N_FF_EXP as usize)
            * (BLOCKS_Q8K_GATE_IN as usize)
            * BLOCK_IQ2_XXS_BYTES;
        let down_bpe =
            (N_EMBD as usize) * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q2_K_BYTES;
        let k = expert_ids.len();

        let mut remap_host = vec![-1i32; 256];
        for (dense, &e) in expert_ids.iter().enumerate() {
            remap_host[e as usize] = dense as i32;
        }

        let pack = |name: &str, bpe: usize| -> eyre::Result<DeviceBuffer<u8>> {
            let tensor = gguf
                .gguf()
                .tensor(name)
                .ok_or_else(|| eyre!("tensor `{name}` not found"))?;
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

        let gate = pack(&format!("blk.{layer}.ffn_gate_exps.weight"), gate_bpe)?;
        let up = pack(&format!("blk.{layer}.ffn_up_exps.weight"), gate_bpe)?;
        let down = pack(&format!("blk.{layer}.ffn_down_exps.weight"), down_bpe)?;
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
        // M56 het-split: place the K hottest experts per layer on the dGPU too.
        // K defaults to 6 when a placement file exists (see dgpu_hot_experts).
        let hot_k: usize = dgpu_hot_experts();
        if hot_k > 0 {
            let path = hot_expert_file_path();
            match parse_hot_expert_file(&path, hot_k) {
                Ok(lists) => {
                    for layer in 0..N_LAYER as usize {
                        if lists[layer].is_empty() {
                            // Global greedy gave this (flat-routing) layer
                            // zero slots — leave it fully iGPU-resident.
                            continue;
                        }
                        match HotExpertWeights::load(
                            gguf,
                            dgpu_device,
                            layer as i32,
                            &lists[layer],
                        ) {
                            Ok((hot, remap_host)) => {
                                igpu_device.set_current()?;
                                let mut r =
                                    DeviceBuffer::<i32>::new(igpu_device.id, 256)?;
                                r.copy_from_host(&remap_host)?;
                                igpu_layers[layer].hot_remap = Some(r);
                                dgpu_layers[layer].hot_experts = Some(hot);
                            }
                            Err(e) => {
                                eprintln!(
                                    "WARN hot-expert load failed at L{layer} (disabled): {e}"
                                );
                            }
                        }
                    }
                    dgpu_device.set_current()?;
                    eprintln!("M56 het-split: {hot_k} hot experts/layer resident on dGPU");
                }
                Err(e) => {
                    eprintln!("WARN DGPU_HOT_EXPERTS set but placement file unusable: {e}");
                }
            }
        }
        Ok(Self {
            global,
            dgpu_layers,
            igpu_layers,
        })
    }
}
